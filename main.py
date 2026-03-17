import warnings
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision  # requested import
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
TICKERS = ["SPY", "LMT", "RTX", "NOC"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
SPLIT_RATIO = 0.8  # chronological split
WINDOW_SIZE = 60
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 20
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-4
USE_SPY_ONLY_FEATURES = True
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
TRIAL_SEEDS = [7, 21, 42]


def get_device() -> torch.device:
    """Prefer CUDA when available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device


def set_seed(seed: int):
    """Make runs more reproducible and enable fair trial comparison."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== DATA ACQUISITION ====================
def fetch_adjusted_close(tickers: list[str]) -> pd.DataFrame:
    """Fetch adjusted close (or close fallback) per ticker with robust handling."""
    print("Fetching data from yfinance...")
    # Avoid yfinance tz cache lock issues on Windows by using a local cache folder.
    try:
        yf.set_tz_cache_location("./yf_tz_cache")
    except Exception:
        pass

    series_list = []
    failed = []

    for ticker in tickers:
        try:
            # First attempt: direct download API
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            # Fallback: per-ticker history API
            if df.empty:
                df = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE, auto_adjust=False)

            if df.empty:
                failed.append(ticker)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = df.columns.get_level_values(0)
                if "Adj Close" in lvl0:
                    raw = df["Adj Close"]
                elif "Close" in lvl0:
                    raw = df["Close"]
                else:
                    failed.append(ticker)
                    continue

                # raw can be a 1-col DataFrame indexed by ticker.
                if isinstance(raw, pd.DataFrame):
                    s = raw.iloc[:, 0]
                else:
                    s = raw
                s = s.rename(ticker)
            else:
                price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                s = df[price_col].rename(ticker)

            series_list.append(s)
        except Exception:
            failed.append(ticker)

    if not series_list:
        raise RuntimeError("No ticker data was downloaded from yfinance.")

    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.ffill().bfill()

    print(f"Downloaded tickers: {list(prices.columns)}")
    if failed:
        print(f"Failed tickers (ignored): {failed}")

    return prices


# ==================== FEATURE ENGINEERING ====================
def calculate_sma(data: pd.Series, period: int = 50) -> pd.Series:
    return data.rolling(window=period, min_periods=1).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Create trailing-window indicators for each ticker."""
    print("Engineering features...")
    features = pd.DataFrame(index=prices.index)

    for ticker in prices.columns:
        close = prices[ticker]
        features[f"{ticker}_close"] = close
        features[f"{ticker}_sma_50"] = calculate_sma(close, 50)
        features[f"{ticker}_rsi_14"] = calculate_rsi(close, 14)
        macd, sig, hist = calculate_macd(close, 12, 26, 9)
        features[f"{ticker}_macd"] = macd
        features[f"{ticker}_macd_signal"] = sig
        features[f"{ticker}_macd_hist"] = hist

    # Required missing-value handling
    features = features.ffill().bfill()
    return features


def select_model_features(features: pd.DataFrame) -> pd.DataFrame:
    """Use a simpler feature set to reduce noise and overfitting risk."""
    if USE_SPY_ONLY_FEATURES:
        spy_cols = [c for c in features.columns if c.startswith("SPY_")]
        selected = features[spy_cols].copy()
        print(f"Using SPY-only features: {len(spy_cols)} columns")
        return selected

    print(f"Using all features: {features.shape[1]} columns")
    return features


# ==================== PREPROCESSING ====================
def chronological_split(df: pd.DataFrame, ratio: float = SPLIT_RATIO):
    """Strict temporal split (no shuffle)."""
    split_idx = int(len(df) * ratio)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    print(f"Train rows: {len(train_df)}, Validation rows: {len(val_df)}")
    return train_df, val_df


def normalize(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Fit scaler only on train, then transform both sets."""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df), index=train_df.index, columns=train_df.columns
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df), index=val_df.index, columns=val_df.columns
    )
    return train_scaled, val_scaled, scaler


def get_target_col(df: pd.DataFrame) -> str:
    target_col = "SPY_close"
    if target_col not in df.columns:
        # Fallback to first ticker close if SPY is unavailable.
        first_close = [c for c in df.columns if c.endswith("_close")][0]
        target_col = first_close
    return target_col


def create_sequences(df: pd.DataFrame, raw_target_close: pd.Series, window_size: int = WINDOW_SIZE):
    """Create scaled input windows with return targets from raw close prices."""
    target_col = get_target_col(df)

    x_list, y_return_list, prev_close_list, next_close_list = [], [], [], []
    values = df.values
    raw_close = raw_target_close.to_numpy(dtype=np.float32)

    for i in range(len(df) - window_size):
        x_list.append(values[i : i + window_size, :])
        prev_close = raw_close[i + window_size - 1]
        next_close = raw_close[i + window_size]
        next_return = (next_close - prev_close) / (abs(prev_close) + 1e-10)
        y_return_list.append(next_return)
        prev_close_list.append(prev_close)
        next_close_list.append(next_close)

    x = np.asarray(x_list, dtype=np.float32)
    y_return = np.asarray(y_return_list, dtype=np.float32)
    prev_close = np.asarray(prev_close_list, dtype=np.float32)
    next_close = np.asarray(next_close_list, dtype=np.float32)
    return x, y_return, prev_close, next_close, target_col


# ==================== MODEL ====================
class GRUNeuralNetwork(nn.Module):
    """2-layer GRU: 100 units then 50 units, with dropout=0.2."""

    def __init__(self, input_size: int, dropout: float = DROPOUT_RATE):
        super().__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=HIDDEN_SIZE_1, batch_first=True)
        self.gru2 = nn.GRU(input_size=HIDDEN_SIZE_1, hidden_size=HIDDEN_SIZE_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(HIDDEN_SIZE_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x[:, -1, :])
        return self.head(x)


class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.best_epoch = -1
        self.best_state = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module, epoch: int):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True


def train_model(model, train_loader, val_loader, device):
    """Adam + MSE + early stopping on validation loss."""
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    early = EarlyStopping(EARLY_STOPPING_PATIENCE)

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                pred = model(xb)
                val_running += criterion(pred, yb).item()

        val_loss = val_running / max(len(val_loader), 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.6e}"
            )

        early.step(val_loss, model, epoch)
        if early.should_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)
        print(f"Restored best checkpoint from epoch {early.best_epoch + 1}")

    return train_losses, val_losses


def predict_full(model, x_tensor: torch.Tensor, device):
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor.to(device)).squeeze(1).cpu().numpy()
    return preds


def print_simple_metric_guide(
    train_mse,
    val_mse,
    train_rmse,
    val_rmse,
    train_mae,
    val_mae,
    train_r2,
    val_r2,
    baseline_train_mse,
    baseline_val_mse,
):
    """Print an easy-to-read summary of what each metric means."""
    train_mse_gain = (baseline_train_mse - train_mse) / (baseline_train_mse + 1e-10) * 100.0
    val_mse_gain = (baseline_val_mse - val_mse) / (baseline_val_mse + 1e-10) * 100.0

    print("\n" + "=" * 70)
    print("SIMPLE SCORECARD")
    print("=" * 70)
    print("How to read: Lower MSE/RMSE/MAE is better. Higher R2 is better.")
    print(f"Validation MSE improvement vs baseline: {val_mse_gain:.2f}%")
    if val_mse_gain > 0:
        print("Status: Good - model beats baseline on validation MSE.")
    else:
        print("Status: Needs work - model does not beat baseline on validation MSE.")

    print("\nPrediction error (price units):")
    print(f"Train MAE: {train_mae:.4f} | Validation MAE: {val_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f} | Validation RMSE: {val_rmse:.4f}")

    print("\nModel fit quality (R2):")
    print(f"Train R2: {train_r2:.4f} | Validation R2: {val_r2:.4f}")
    if val_r2 > 0:
        print("R2 check: Positive validation R2 means the model explains some unseen variation.")
    else:
        print("R2 check: Non-positive validation R2 means poor generalization.")

    print("\nBaseline comparison:")
    print(f"Train MSE - model: {train_mse:.4f}, baseline: {baseline_train_mse:.4f}, improvement: {train_mse_gain:.2f}%")
    print(f"Val   MSE - model: {val_mse:.4f}, baseline: {baseline_val_mse:.4f}, improvement: {val_mse_gain:.2f}%")


def fit_delta_calibration(pred_delta: np.ndarray, true_delta: np.ndarray):
    """Fit y = a*x + b on training deltas to de-bias model outputs."""
    x = pred_delta.astype(np.float64)
    y = true_delta.astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 1.0, 0.0
    a = np.sum((x - x_mean) * (y - y_mean)) / denom
    b = y_mean - a * x_mean
    return float(a), float(b)


def denormalize_target(preds: np.ndarray, scaler: StandardScaler, n_features: int, target_idx: int):
    dummy = np.zeros((len(preds), n_features), dtype=np.float32)
    dummy[:, target_idx] = preds
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("GRU STOCK FORECAST PIPELINE")
    print("=" * 70)
    print(f"torch={torch.__version__} | torchvision={torchvision.__version__}")

    device = get_device()

    prices = fetch_adjusted_close(TICKERS)
    print(f"Price shape: {prices.shape}")

    features = engineer_features(prices)
    features = select_model_features(features)
    print(f"Feature shape: {features.shape}")

    train_df, val_df = chronological_split(features, SPLIT_RATIO)
    train_scaled, val_scaled, scaler = normalize(train_df, val_df)

    target_col = get_target_col(train_scaled)
    x_train, y_train, train_prev_close, train_actual_close, _ = create_sequences(
        train_scaled, train_df[target_col], WINDOW_SIZE
    )
    x_val, y_val, val_prev_close, val_actual_close, _ = create_sequences(
        val_scaled, val_df[target_col], WINDOW_SIZE
    )

    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_val:   {x_val.shape}, y_val:   {y_val.shape}")
    print(f"Target column: {target_col}")

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_tensor, y_val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin,
    )

    print("Running multi-seed ensemble training...")
    train_pred_return_trials = []
    val_pred_return_trials = []
    trial_scores = []
    trial_train_histories = []
    trial_val_histories = []
    trial_states = []

    for seed in TRIAL_SEEDS:
        print(f"\n--- Trial seed: {seed} ---")
        set_seed(seed)
        trial_model = GRUNeuralNetwork(input_size=x_train.shape[2], dropout=DROPOUT_RATE).to(device)
        print(trial_model)

        train_hist, val_hist = train_model(trial_model, train_loader, val_loader, device)
        trial_train_pred_returns = predict_full(trial_model, x_train_tensor, device)
        trial_val_pred_returns = predict_full(trial_model, x_val_tensor, device)
        trial_val_ret_mse = mean_squared_error(y_val, trial_val_pred_returns)
        print(f"Trial seed {seed} validation return MSE: {trial_val_ret_mse:.6f}")

        train_pred_return_trials.append(trial_train_pred_returns)
        val_pred_return_trials.append(trial_val_pred_returns)
        trial_scores.append(trial_val_ret_mse)
        trial_train_histories.append(train_hist)
        trial_val_histories.append(val_hist)
        trial_states.append({k: v.detach().cpu().clone() for k, v in trial_model.state_dict().items()})

    # Ensemble predicted returns by averaging across seeds.
    train_pred_returns_ens = np.mean(np.vstack(train_pred_return_trials), axis=0)
    val_pred_returns_ens = np.mean(np.vstack(val_pred_return_trials), axis=0)

    # Calibrate predicted price deltas on training set only.
    train_pred_delta_raw = train_prev_close * train_pred_returns_ens
    train_true_delta = train_actual_close - train_prev_close
    a_cal, b_cal = fit_delta_calibration(train_pred_delta_raw, train_true_delta)

    val_pred_delta_raw = val_prev_close * val_pred_returns_ens
    train_pred_delta_cal = a_cal * train_pred_delta_raw + b_cal
    val_pred_delta_cal = a_cal * val_pred_delta_raw + b_cal

    # Keep representative training curves from the best seed for display purposes.
    best_idx = int(np.argmin(np.array(trial_scores)))
    best_seed = TRIAL_SEEDS[best_idx]
    best_val_ret_mse = trial_scores[best_idx]
    train_losses = trial_train_histories[best_idx]
    val_losses = trial_val_histories[best_idx]
    best_state = trial_states[best_idx]

    # Candidate A: best single seed
    train_pred_ret_best = train_pred_return_trials[best_idx]
    val_pred_ret_best = val_pred_return_trials[best_idx]
    train_pred_close_best = train_prev_close * (1.0 + train_pred_ret_best)
    val_pred_close_best = val_prev_close * (1.0 + val_pred_ret_best)

    # Candidate B: raw ensemble
    train_pred_close_ens = train_prev_close * (1.0 + train_pred_returns_ens)
    val_pred_close_ens = val_prev_close * (1.0 + val_pred_returns_ens)

    # Candidate C: calibrated ensemble
    train_pred_close_cal = train_prev_close + train_pred_delta_cal
    val_pred_close_cal = val_prev_close + val_pred_delta_cal
    train_pred_ret_cal = train_pred_delta_cal / (np.abs(train_prev_close) + 1e-10)
    val_pred_ret_cal = val_pred_delta_cal / (np.abs(val_prev_close) + 1e-10)

    candidate_val_mse = {
        "best_seed": mean_squared_error(val_actual_close, val_pred_close_best),
        "ensemble": mean_squared_error(val_actual_close, val_pred_close_ens),
        "ensemble_calibrated": mean_squared_error(val_actual_close, val_pred_close_cal),
    }
    selected_method = min(candidate_val_mse, key=candidate_val_mse.get)

    if selected_method == "best_seed":
        train_pred_returns = train_pred_ret_best
        val_pred_returns = val_pred_ret_best
        train_pred_close = train_pred_close_best
        val_pred_close = val_pred_close_best
    elif selected_method == "ensemble":
        train_pred_returns = train_pred_returns_ens
        val_pred_returns = val_pred_returns_ens
        train_pred_close = train_pred_close_ens
        val_pred_close = val_pred_close_ens
    else:
        train_pred_returns = train_pred_ret_cal
        val_pred_returns = val_pred_ret_cal
        train_pred_close = train_pred_close_cal
        val_pred_close = val_pred_close_cal

    # Keep a model object from best seed for persistence compatibility.
    model = GRUNeuralNetwork(input_size=x_train.shape[2], dropout=DROPOUT_RATE).to(device)
    model.load_state_dict(best_state)

    print(
        f"\nEnsemble complete ({len(TRIAL_SEEDS)} seeds). "
        f"Best single seed={best_seed}, best val return MSE={best_val_ret_mse:.6f}"
    )
    print(
        f"Selected prediction method: {selected_method} "
        f"(val price MSE={candidate_val_mse[selected_method]:.6f})"
    )
    print(f"Calibration applied on training deltas: a={a_cal:.4f}, b={b_cal:.6f}")

    # Naive baseline: predict no change from previous close.
    baseline_train_close = train_prev_close.copy()
    baseline_val_close = val_prev_close.copy()

    print("\n" + "=" * 70)
    print("NEXT-DAY PREDICTIONS (SAMPLES)")
    print("=" * 70)
    print("Train actual return:    ", y_train[:10])
    print("Train predicted return: ", train_pred_returns[:10])
    print("Train actual close:     ", train_actual_close[:10])
    print("Train predicted close:  ", train_pred_close[:10])
    print("Validation actual return:   ", y_val[:10])
    print("Validation predicted return:", val_pred_returns[:10])
    print("Validation actual close:    ", val_actual_close[:10])
    print("Validation predicted close: ", val_pred_close[:10])

    # Return-space metrics
    train_ret_mse = mean_squared_error(y_train, train_pred_returns)
    val_ret_mse = mean_squared_error(y_val, val_pred_returns)

    # Price-space metrics (easier to interpret)
    train_mse = mean_squared_error(train_actual_close, train_pred_close)
    val_mse = mean_squared_error(val_actual_close, val_pred_close)
    train_rmse = float(np.sqrt(train_mse))
    val_rmse = float(np.sqrt(val_mse))
    train_mae = mean_absolute_error(train_actual_close, train_pred_close)
    val_mae = mean_absolute_error(val_actual_close, val_pred_close)
    train_r2 = r2_score(train_actual_close, train_pred_close)
    val_r2 = r2_score(val_actual_close, val_pred_close)

    baseline_train_mse = mean_squared_error(train_actual_close, baseline_train_close)
    baseline_val_mse = mean_squared_error(val_actual_close, baseline_val_close)
    baseline_train_r2 = r2_score(train_actual_close, baseline_train_close)
    baseline_val_r2 = r2_score(val_actual_close, baseline_val_close)

    print("\n" + "=" * 70)
    print("METRICS")
    print("=" * 70)
    print(f"Train Return MSE: {train_ret_mse:.6f}")
    print(f"Val   Return MSE: {val_ret_mse:.6f}")
    print(f"Train Price MSE:  {train_mse:.6f}")
    print(f"Val   Price MSE:  {val_mse:.6f}")
    print(f"Train Price RMSE: {train_rmse:.6f}")
    print(f"Val   Price RMSE: {val_rmse:.6f}")
    print(f"Train Price MAE:  {train_mae:.6f}")
    print(f"Val   Price MAE:  {val_mae:.6f}")
    print(f"Train Price R2:   {train_r2:.6f}")
    print(f"Val   Price R2:   {val_r2:.6f}")
    print("\nBaseline (no-change) on price:")
    print(f"Baseline Train MSE: {baseline_train_mse:.6f}")
    print(f"Baseline Val   MSE: {baseline_val_mse:.6f}")
    print(f"Baseline Train R2:  {baseline_train_r2:.6f}")
    print(f"Baseline Val   R2:  {baseline_val_r2:.6f}")

    print("\n" + "=" * 70)
    print("PLAIN-ENGLISH SUMMARY")
    print("=" * 70)
    if val_mse < baseline_val_mse:
        print("Good: The GRU beats a simple no-change baseline on validation price MSE.")
    else:
        print("Warning: The GRU does not beat the no-change baseline on validation MSE yet.")

    if val_r2 > 0:
        print("Validation R2 is positive: model explains some unseen variation.")
    else:
        print("Validation R2 is <= 0: generalization is still weak on unseen data.")

    print_simple_metric_guide(
        train_mse,
        val_mse,
        train_rmse,
        val_rmse,
        train_mae,
        val_mae,
        train_r2,
        val_r2,
        baseline_train_mse,
        baseline_val_mse,
    )

    # Persist best seed model plus ensemble settings.
    torch.save(
        {
            "best_seed_model_state_dict": model.state_dict(),
            "selected_method": selected_method,
            "seeds": TRIAL_SEEDS,
            "calibration_a": a_cal,
            "calibration_b": b_cal,
            "feature_mode": "SPY_ONLY" if USE_SPY_ONLY_FEATURES else "ALL",
            "hidden_sizes": [HIDDEN_SIZE_1, HIDDEN_SIZE_2],
        },
        "gru_model.pth",
    )
    print("\nSaved model settings -> gru_model.pth")

    # Save full predictions for downstream analysis.
    pd.DataFrame(
        {
            "actual_return": y_train,
            "pred_return": train_pred_returns,
            "prev_close": train_prev_close,
            "actual_close": train_actual_close,
            "pred_close": train_pred_close,
        }
    ).to_csv("train_predictions.csv", index=False)

    pd.DataFrame(
        {
            "actual_return": y_val,
            "pred_return": val_pred_returns,
            "prev_close": val_prev_close,
            "actual_close": val_actual_close,
            "pred_close": val_pred_close,
        }
    ).to_csv("validation_predictions.csv", index=False)

    print("Saved predictions -> train_predictions.csv, validation_predictions.csv")

    return model, scaler, train_pred_returns, val_pred_returns, train_losses, val_losses
"""
Main Orchestrator — Member 1
Runs the end-to-end pipeline:
  1. Data acquisition & feature engineering
  2. Random Forest training
  3. Prediction generation & export
  4. Evaluation & benchmarking
"""

from data_pipeline import build_pipeline
from random_forest_model import build_rf, train_rf, predict_rf, save_predictions
from evaluation import full_report


def main() -> None:
    ticker = "SPY"
    print("=" * 55)
    print("  MEMBER 1 — Random Forest Baseline Pipeline")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Step 1: Data pipeline
    # ------------------------------------------------------------------
    print("\n[1/4] Fetching data & engineering features …")
    data = build_pipeline(ticker)
    print(f"  Train samples: {data['X_train'].shape[0]}")
    print(f"  Test samples : {data['X_test'].shape[0]}")
    print(f"  Features     : {data['X_train'].shape[1]}")

    # ------------------------------------------------------------------
    # Step 2: Train Random Forest
    # ------------------------------------------------------------------
    print("\n[2/4] Training Random Forest (100 estimators) …")
    model = build_rf(n_estimators=100)
    train_rf(model, data["X_train"], data["y_train"])
    print("  Training complete ✓")

    # ------------------------------------------------------------------
    # Step 3: Generate & save predictions
    # ------------------------------------------------------------------
    print("\n[3/4] Generating predictions …")
    train_preds = predict_rf(model, data["X_train"])
    test_preds = predict_rf(model, data["X_test"])
    save_predictions(
        train_preds=train_preds,
        test_preds=test_preds,
        train_index=data["train_df"].index,
        test_index=data["test_df"].index,
        path="rf_predictions.csv",
    )

    # ------------------------------------------------------------------
    # Step 4: Evaluation & benchmarking
    # ------------------------------------------------------------------
    print("\n[4/4] Running evaluation …")
    full_report(
        y_test=data["y_test"],
        test_preds=test_preds,
        test_df=data["test_df"],
        label="Random Forest",
    )

    print("Pipeline complete ✓")
Member 3: XGBoost Meta-Learner & Risk Governance
Hackathon Trading Model — Defense Stocks vs SPY

This script implements:
  1. Shared data pipeline & feature engineering (Hour 1)
  2. Financial turbulence index with capital protection (Hour 2)
  3. XGBoost meta-learner ensembling RF + GRU predictions (Hour 3)
  4. Trading simulation, benchmarking, & evaluation (Hour 4)

Members 1 & 2: Replace the mock prediction functions with your real models.
See `integrate_member_models()` at the bottom for the integration point.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ============================================================================
# CONFIGURATION
# ============================================================================
TICKERS = ["SPY", "LMT", "RTX", "NOC"]
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
TRAIN_RATIO = 0.80
TURBULENCE_WINDOW = 252       # ~1 trading year for covariance estimation
TURBULENCE_PERCENTILE = 90    # Threshold at 90th percentile of historical turbulence
GRU_LOOKBACK = 60             # 60-day sliding window (used by Member 2's GRU)
RANDOM_STATE = 42


# ============================================================================
# SECTION 1 — DATA PIPELINE & FEATURE ENGINEERING
# ============================================================================

def fetch_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data for multiple tickers, forward-fill missing values."""
    print(f"[Pipeline] Downloading data for {tickers} from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns (field, ticker) for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ['_'.join(col).strip() for col in raw.columns]

    raw = raw.ffill().bfill()
    print(f"[Pipeline] Fetched {len(raw)} trading days, {raw.shape[1]} columns.")
    return raw


def compute_daily_returns(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Compute daily percentage returns for each ticker."""
    returns = pd.DataFrame(index=df.index)
    for t in tickers:
        close_col = f"Close_{t}"
        if close_col in df.columns:
            returns[t] = df[close_col].pct_change()
    return returns.dropna()


def add_technical_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add SMA(50), RSI(14), and MACD(12,26,9) for a single ticker."""
    close = df[f"Close_{ticker}"]

    # SMA 50
    df[f"SMA50_{ticker}"] = close.rolling(window=50).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"RSI_{ticker}"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df[f"MACD_{ticker}"] = macd_line
    df[f"MACD_signal_{ticker}"] = signal_line
    df[f"MACD_hist_{ticker}"] = macd_line - signal_line

    return df


def build_feature_matrix(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Build the full feature matrix with technical indicators for all tickers."""
    for t in tickers:
        df = add_technical_indicators(df, t)
    df = df.dropna()
    return df


def split_data(df: pd.DataFrame, ratio: float):
    """Chronological train/test split — no shuffling."""
    split_idx = int(len(df) * ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"[Pipeline] Train: {len(train)} days | Test: {len(test)} days "
          f"(split at {train.index[-1].strftime('%Y-%m-%d')})")
    return train, test


def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        feature_cols: list[str]):
    """Z-score normalize: fit on train ONLY, transform both (prevent leakage)."""
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    print(f"[Pipeline] Normalized {len(feature_cols)} features (scaler fit on train only).")
    return train_df, test_df, scaler


# ============================================================================
# SECTION 2 — FINANCIAL TURBULENCE INDEX
# ============================================================================

def compute_turbulence_index(returns_df: pd.DataFrame,
                              window: int = TURBULENCE_WINDOW) -> pd.Series:
    """
    Compute Mahalanobis-distance-based financial turbulence index.

    Turbulence_t = (y_t - μ) Σ^-1 (y_t - μ)'

    where y_t = today's return vector, μ = rolling mean, Σ = rolling covariance.
    """
    turbulence = pd.Series(index=returns_df.index, dtype=float)

    for i in range(window, len(returns_df)):
        hist = returns_df.iloc[i - window:i]
        y_t = returns_df.iloc[i].values

        mu = hist.mean().values
        cov = hist.cov().values

        diff = (y_t - mu).reshape(1, -1)

        try:
            cov_inv = np.linalg.inv(cov)
            turb = (diff @ cov_inv @ diff.T).item()
        except np.linalg.LinAlgError:
            # Singular matrix — fallback to pseudo-inverse
            cov_inv = np.linalg.pinv(cov)
            turb = (diff @ cov_inv @ diff.T).item()

        turbulence.iloc[i] = turb

    turbulence = turbulence.dropna()
    print(f"[Turbulence] Computed for {len(turbulence)} days. "
          f"Mean={turbulence.mean():.4f}, Max={turbulence.max():.4f}")
    return turbulence


def get_turbulence_threshold(turbulence: pd.Series,
                              percentile: float = TURBULENCE_PERCENTILE) -> float:
    """Set turbulence threshold at the given percentile of historical values."""
    threshold = np.percentile(turbulence.dropna(), percentile)
    print(f"[Turbulence] Threshold set at {percentile}th percentile: {threshold:.4f}")
    return threshold


# ============================================================================
# SECTION 3 — MOCK BASE MODEL PREDICTIONS (Replace with real models)
# ============================================================================

def mock_random_forest_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                     feature_cols: list[str],
                                     target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Placeholder Random Forest model for development.
    Member 1: Replace this with your trained RF model.
    """
    print("[Mock RF] Training placeholder Random Forest (100 trees)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train_df[feature_cols], train_df[target_col])

    train_preds = rf.predict(train_df[feature_cols])
    test_preds = rf.predict(test_df[feature_cols])
    print(f"[Mock RF] Generated {len(train_preds)} train + {len(test_preds)} test predictions.")
    return train_preds, test_preds


def mock_gru_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Placeholder GRU predictions using a simple rolling-mean proxy.
    Member 2: Replace this with your trained GRU model's output.

    The real GRU uses a 60-day sliding window of normalized features.
    This mock uses a 5-day rolling mean of the target as an approximation.
    """
    print("[Mock GRU] Generating placeholder GRU-style predictions (rolling mean proxy)...")
    train_preds = train_df[target_col].rolling(5).mean().bfill().values
    test_preds = test_df[target_col].rolling(5).mean().bfill().values
    print(f"[Mock GRU] Generated {len(train_preds)} train + {len(test_preds)} test predictions.")
    return train_preds, test_preds


# ============================================================================
# SECTION 4 — XGBOOST META-LEARNER
# ============================================================================

def build_meta_features(rf_preds: np.ndarray, gru_preds: np.ndarray,
                         df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Combine base model predictions with volatility/technical features
    to form the meta-learner's input dataset.
    """
    meta = pd.DataFrame(index=df.index)
    meta["rf_pred"] = rf_preds
    meta["gru_pred"] = gru_preds

    # Add recent volatility (20-day rolling std of SPY returns)
    if "Close_SPY" in df.columns:
        spy_ret = df["Close_SPY"].pct_change()
        meta["volatility_20d"] = spy_ret.rolling(20).std()

    # Add RSI and MACD for SPY as additional context
    for feat in ["RSI_SPY", "MACD_hist_SPY"]:
        if feat in df.columns:
            meta[feat] = df[feat].values

    # Prediction spread — how much the models disagree
    meta["pred_spread"] = np.abs(rf_preds - gru_preds)

    meta = meta.fillna(0)
    return meta


def train_xgboost_meta_learner(meta_train: pd.DataFrame,
                                 y_train: np.ndarray) -> XGBRegressor:
    """
    Train XGBoost meta-learner on base model predictions + volatility features.
    XGBoost learns which base model to trust under different market conditions.
    """
    print("[XGBoost] Training meta-learner...")
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    xgb.fit(meta_train.values, y_train)
    print(f"[XGBoost] Meta-learner trained on {len(meta_train)} samples, "
          f"{meta_train.shape[1]} features.")
    return xgb


# ============================================================================
# SECTION 5 — TRADING SIMULATION & BENCHMARKING
# ============================================================================

def simulate_trading(predictions: np.ndarray, actual_returns: np.ndarray,
                      turbulence: pd.Series, threshold: float,
                      test_index: pd.DatetimeIndex) -> pd.Series:
    """
    Trading logic:
      - If XGBoost predicts positive return → 100% long
      - If XGBoost predicts negative return → hold cash (0%)
      - OVERRIDE: If turbulence > threshold → force cash (liquidate all)
    """
    portfolio_returns = pd.Series(0.0, index=test_index)

    for i, date in enumerate(test_index):
        # Turbulence kill-switch check
        if date in turbulence.index and turbulence.loc[date] > threshold:
            portfolio_returns.iloc[i] = 0.0  # Force cash — crash mode
            continue

        # Normal trading — follow XGBoost signal
        if predictions[i] > 0:
            portfolio_returns.iloc[i] = actual_returns[i]  # Long position
        else:
            portfolio_returns.iloc[i] = 0.0  # Cash

    return portfolio_returns


def buy_and_hold_benchmark(actual_returns: np.ndarray,
                            test_index: pd.DatetimeIndex) -> pd.Series:
    """Simple buy-and-hold SPY strategy."""
    return pd.Series(actual_returns, index=test_index)


def sma_crossover_benchmark(close_prices: pd.Series,
                              test_index: pd.DatetimeIndex) -> pd.Series:
    """SMA(10)/SMA(50) crossover strategy on SPY."""
    sma10 = close_prices.rolling(10).mean()
    sma50 = close_prices.rolling(50).mean()
    returns = close_prices.pct_change()

    signal = (sma10 > sma50).astype(int)  # 1 = long, 0 = cash
    strategy_returns = signal.shift(1) * returns  # Trade next day
    return strategy_returns.reindex(test_index).fillna(0)


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown from a cumulative return series."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def print_metrics(name: str, predictions: np.ndarray, actuals: np.ndarray,
                   portfolio_returns: pd.Series):
    """Print prediction accuracy and portfolio performance metrics."""
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)

    cum_ret = (1 + portfolio_returns).cumprod()
    total_return = cum_ret.iloc[-1] - 1
    max_dd = calculate_max_drawdown(cum_ret)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  MSE:              {mse:.8f}")
    print(f"  RMSE:             {rmse:.8f}")
    print(f"  MAE:              {mae:.8f}")
    print(f"  Cumulative Return: {total_return*100:+.2f}%")
    print(f"  Max Drawdown:      {max_dd*100:.2f}%")
    print(f"{'='*60}")


def plot_equity_curves(strategies: dict[str, pd.Series], save_path: str = "equity_curves.png"):
    """Plot and save equity curves for all strategies."""
    plt.figure(figsize=(14, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for idx, (name, returns) in enumerate(strategies.items()):
        cum = (1 + returns).cumprod()
        color = colors[idx % len(colors)]
        plt.plot(cum.index, cum.values, label=name, linewidth=2, color=color)

    plt.title("Equity Curves — XGBoost Meta-Learner vs Benchmarks",
              fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (starting at $1)", fontsize=12)
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[Plot] Equity curves saved to {save_path}")


# ============================================================================
# SECTION 6 — INTEGRATION POINT FOR MEMBERS 1 & 2
# ============================================================================

def integrate_member_models(rf_model=None, gru_model=None):
    """
    Integration point for Members 1 & 2.

    Usage once real models are ready:
        # Member 1 provides trained RF model
        rf_train_preds = rf_model.predict(train_features)
        rf_test_preds = rf_model.predict(test_features)

        # Member 2 provides trained GRU model
        gru_train_preds = gru_model.predict(train_sequences)
        gru_test_preds = gru_model.predict(test_sequences)

    Then pass these predictions into build_meta_features() and
    train_xgboost_meta_learner() instead of the mock versions.
    """
    print("[Integration] Replace mock predictions with real RF/GRU outputs.")
    print("[Integration] See integrate_member_models() docstring for instructions.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("  HACKATHON TRADING MODEL — Member 3: XGBoost Meta-Learner")
    print("=" * 70)

    # ------------------------------------------------------------------
    # HOUR 1: Data Pipeline & Feature Engineering
    # ------------------------------------------------------------------
    print("\n>>> HOUR 1: Data Pipeline & Feature Engineering\n")

    # 1a. Fetch data
    raw_df = fetch_data(TICKERS, START_DATE, END_DATE)

    # 1b. Compute daily returns (for turbulence calculation)
    daily_returns = compute_daily_returns(raw_df, TICKERS)

    # 1c. Add technical indicators
    full_df = build_feature_matrix(raw_df, TICKERS)

    # 1d. Define feature columns and target
    target_col = "Close_SPY"

    feature_cols = []
    for t in TICKERS:
        for prefix in ["Close", "Open", "High", "Low", "Volume",
                        "SMA50", "RSI", "MACD", "MACD_signal", "MACD_hist"]:
            col = f"{prefix}_{t}"
            if col in full_df.columns:
                feature_cols.append(col)

    # Create next-day return as the prediction target
    full_df["target_return"] = full_df["Close_SPY"].pct_change().shift(-1)
    full_df = full_df.dropna(subset=["target_return"])

    # 1e. Chronological split
    train_df, test_df = split_data(full_df, TRAIN_RATIO)

    # 1f. Z-score normalization (fit on train only!)
    train_df, test_df, scaler = normalize_features(train_df, test_df, feature_cols)

    # ------------------------------------------------------------------
    # HOUR 2: Turbulence Index & Capital Protection
    # ------------------------------------------------------------------
    print("\n>>> HOUR 2: Turbulence Index & Capital Protection\n")

    turbulence = compute_turbulence_index(daily_returns, window=TURBULENCE_WINDOW)
    turb_threshold = get_turbulence_threshold(turbulence, TURBULENCE_PERCENTILE)

    # Show some turbulence stats
    print(f"[Turbulence] Sample high-turbulence dates (top 5):")
    top5 = turbulence.nlargest(5)
    for date, val in top5.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {val:.4f}")

    # ------------------------------------------------------------------
    # HOUR 3: Base Model Predictions + XGBoost Meta-Learner
    # ------------------------------------------------------------------
    print("\n>>> HOUR 3: XGBoost Meta-Learner & Ensembling\n")

    # 3a. Generate base model predictions (mock — replace with real models)
    rf_train_preds, rf_test_preds = mock_random_forest_predictions(
        train_df, test_df, feature_cols, "target_return"
    )
    gru_train_preds, gru_test_preds = mock_gru_predictions(
        train_df, test_df, "target_return"
    )

    # 3b. Build meta-feature datasets
    meta_train = build_meta_features(rf_train_preds, gru_train_preds, train_df, TICKERS)
    meta_test = build_meta_features(rf_test_preds, gru_test_preds, test_df, TICKERS)

    # 3c. Train XGBoost meta-learner
    y_train = train_df["target_return"].values
    y_test = test_df["target_return"].values

    xgb_model = train_xgboost_meta_learner(meta_train, y_train)

    # 3d. Generate final ensemble predictions
    ensemble_preds = xgb_model.predict(meta_test.values)
    print(f"[XGBoost] Generated {len(ensemble_preds)} test predictions.")

    # ------------------------------------------------------------------
    # HOUR 4: Trading Simulation, Evaluation, & Polish
    # ------------------------------------------------------------------
    print("\n>>> HOUR 4: Trading Simulation & Evaluation\n")

    test_index = test_df.index

    # Actual SPY returns during test period
    actual_spy_returns = y_test

    # 4a. XGBoost Meta-Learner strategy (with turbulence override)
    xgb_portfolio = simulate_trading(
        ensemble_preds, actual_spy_returns, turbulence, turb_threshold, test_index
    )

    # Count turbulence override days
    turb_days = sum(
        1 for d in test_index
        if d in turbulence.index and turbulence.loc[d] > turb_threshold
    )
    print(f"[Trading] Turbulence override triggered on {turb_days}/{len(test_index)} test days.")

    # 4b. Buy & Hold benchmark
    bh_portfolio = buy_and_hold_benchmark(actual_spy_returns, test_index)

    # 4c. SMA crossover benchmark
    # Need un-normalized close prices — use raw_df
    spy_close = raw_df["Close_SPY"].reindex(full_df.index)
    sma_portfolio = sma_crossover_benchmark(spy_close, test_index)

    # 4d. Print performance metrics
    print_metrics("XGBoost Meta-Learner (with Turbulence Shield)",
                   ensemble_preds, actual_spy_returns, xgb_portfolio)

    # For benchmarks, use actual returns as "predictions" for a trivial comparison
    print_metrics("Buy & Hold SPY",
                   actual_spy_returns, actual_spy_returns, bh_portfolio)

    print_metrics("SMA(10)/SMA(50) Crossover",
                   actual_spy_returns, actual_spy_returns, sma_portfolio)

    # 4e. Plot equity curves
    strategies = {
        "XGBoost Meta-Learner": xgb_portfolio,
        "Buy & Hold SPY": bh_portfolio,
        "SMA Crossover": sma_portfolio,
    }
    plot_equity_curves(strategies)

    # ------------------------------------------------------------------
    # Feature importance from XGBoost
    # ------------------------------------------------------------------
    print("\n[XGBoost] Feature Importance:")
    importance = xgb_model.feature_importances_
    feat_names = meta_train.columns
    for name, imp in sorted(zip(feat_names, importance), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {name:20s} {imp:.4f} {bar}")

    print("\n" + "=" * 70)
    print("  DONE — Member 3 pipeline complete.")
    print("  Replace mock RF/GRU with real models from Members 1 & 2.")
    print("=" * 70)


if __name__ == "__main__":
    main()
