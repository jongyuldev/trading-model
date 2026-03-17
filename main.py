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


if __name__ == "__main__":
    main()
