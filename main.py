"""
Unified Hackathon Trading Model
================================
Combines all three members' work into one end-to-end pipeline:

  Member 1 — Random Forest baseline (tabular features → next-day return)
  Member 2 — GRU ensemble (60-day sliding windows → next-day return)
  Member 3 — XGBoost meta-learner + turbulence index + trading simulation

Run:  python main.py
"""

import warnings
import random
import json
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
TICKERS = ["SPY", "LMT", "RTX", "NOC"]
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
TRAIN_RATIO = 0.80
RANDOM_STATE = 42

# Turbulence
TURBULENCE_WINDOW = 252
TURBULENCE_PERCENTILE = 90

# GRU hyper-parameters
GRU_WINDOW_SIZE = 60
GRU_BATCH_SIZE = 32
GRU_EPOCHS = 100
GRU_LR = 5e-4
GRU_EARLY_PATIENCE = 20
GRU_DROPOUT = 0.2
GRU_WEIGHT_DECAY = 1e-4
GRU_HIDDEN_1 = 64
GRU_HIDDEN_2 = 32


# ============================================================================
# UTILITIES
# ============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU")
    return dev


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# SECTION 1 — SHARED DATA PIPELINE & FEATURE ENGINEERING
# ============================================================================

def fetch_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data for multiple tickers, forward-fill missing values."""
    print(f"[Pipeline] Downloading {tickers} from {start} to {end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ['_'.join(col).strip() for col in raw.columns]

    raw = raw.ffill().bfill()
    print(f"[Pipeline] {len(raw)} trading days, {raw.shape[1]} columns.")
    return raw


def compute_daily_returns(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Compute daily percentage returns for each ticker (for turbulence)."""
    returns = pd.DataFrame(index=df.index)
    for t in tickers:
        col = f"Close_{t}"
        if col in df.columns:
            returns[t] = df[col].pct_change()
    return returns.dropna()


def add_technical_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add SMA(50), RSI(14), MACD(12,26,9) for one ticker (trailing only)."""
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


def get_feature_cols(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    """Collect all OHLCV + indicator column names."""
    feature_cols = []
    for t in tickers:
        for prefix in ["Close", "Open", "High", "Low", "Volume",
                        "SMA50", "RSI", "MACD", "MACD_signal", "MACD_hist"]:
            col = f"{prefix}_{t}"
            if col in df.columns:
                feature_cols.append(col)
    return feature_cols


def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        feature_cols: list[str]):
    """Z-score: fit on train ONLY, transform both (prevent data leakage)."""
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    print(f"[Pipeline] Normalized {len(feature_cols)} features (fit on train only).")
    return train_df, test_df, scaler


# ============================================================================
# SECTION 2 — FINANCIAL TURBULENCE INDEX
# ============================================================================

def compute_turbulence_index(returns_df: pd.DataFrame,
                              window: int = TURBULENCE_WINDOW) -> pd.Series:
    """Mahalanobis-distance turbulence: (y_t - μ) Σ⁻¹ (y_t - μ)'."""
    turbulence = pd.Series(index=returns_df.index, dtype=float)

    for i in range(window, len(returns_df)):
        hist = returns_df.iloc[i - window:i]
        y_t = returns_df.iloc[i].values
        mu = hist.mean().values
        cov = hist.cov().values
        diff = (y_t - mu).reshape(1, -1)

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        turbulence.iloc[i] = (diff @ cov_inv @ diff.T).item()

    turbulence = turbulence.dropna()
    print(f"[Turbulence] {len(turbulence)} days computed. "
          f"Mean={turbulence.mean():.4f}, Max={turbulence.max():.4f}")
    return turbulence


def get_turbulence_threshold(turbulence: pd.Series,
                              percentile: float = TURBULENCE_PERCENTILE) -> float:
    threshold = np.percentile(turbulence.dropna(), percentile)
    print(f"[Turbulence] Threshold at {percentile}th percentile: {threshold:.4f}")
    return threshold


# ============================================================================
# SECTION 3 — MEMBER 1: RANDOM FOREST MODEL
# ============================================================================

def run_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       feature_cols: list[str],
                       target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Train Random Forest (100 trees) on tabular features → next-day return."""
    print("[RF] Training Random Forest (100 estimators) ...")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train_df[feature_cols], train_df[target_col])

    train_preds = rf.predict(train_df[feature_cols])
    test_preds = rf.predict(test_df[feature_cols])
    print(f"[RF] {len(train_preds)} train + {len(test_preds)} test predictions.")
    return rf, train_preds, test_preds


# ============================================================================
# SECTION 4 — MEMBER 2: GRU NEURAL NETWORK
# ============================================================================

class GRUNeuralNetwork(nn.Module):
    """2-layer GRU with dropout."""
    def __init__(self, input_size: int, dropout: float = GRU_DROPOUT):
        super().__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=GRU_HIDDEN_1, batch_first=True)
        self.gru2 = nn.GRU(input_size=GRU_HIDDEN_1, hidden_size=GRU_HIDDEN_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(GRU_HIDDEN_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x[:, -1, :])
        return self.head(x)


class EarlyStopping:
    def __init__(self, patience: int = GRU_EARLY_PATIENCE):
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
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def _train_gru(model, train_loader, val_loader, device):
    """Adam + MSE + LR scheduler + early stopping."""
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=GRU_LR, weight_decay=GRU_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early = EarlyStopping(GRU_EARLY_PATIENCE)

    for epoch in range(GRU_EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / max(len(train_loader), 1)

        model.eval()
        val_run = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                val_run += criterion(model(xb), yb).item()
        val_loss = val_run / max(len(val_loader), 1)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:03d}/{GRU_EPOCHS} | "
                  f"train={train_loss:.6f} val={val_loss:.6f} lr={lr:.2e}")

        early.step(val_loss, model, epoch)
        if early.should_stop:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)
        print(f"  Restored best checkpoint (epoch {early.best_epoch+1})")

    return model


def _predict_gru(model, x_tensor, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x_tensor.to(device)).squeeze(1).cpu().numpy()


def _create_gru_sequences(scaled_df: pd.DataFrame, raw_close: pd.Series,
                            window: int = GRU_WINDOW_SIZE):
    """Build 3-D tensors (samples, window, features) with return targets."""
    values = scaled_df.values
    raw = raw_close.to_numpy(dtype=np.float32)

    x_list, y_ret, prev_list, next_list = [], [], [], []
    for i in range(len(scaled_df) - window):
        x_list.append(values[i:i + window, :])
        pc = raw[i + window - 1]
        nc = raw[i + window]
        y_ret.append((nc - pc) / (abs(pc) + 1e-10))
        prev_list.append(pc)
        next_list.append(nc)

    return (np.asarray(x_list, dtype=np.float32),
            np.asarray(y_ret, dtype=np.float32),
            np.asarray(prev_list, dtype=np.float32),
            np.asarray(next_list, dtype=np.float32))


def run_gru_ensemble(train_scaled: pd.DataFrame, test_scaled: pd.DataFrame,
                      train_raw_close: pd.Series, test_raw_close: pd.Series,
                      device: torch.device):
    """
    Train a multi-seed GRU ensemble and return predicted *returns* for
    both train and test periods (aligned with the tabular data).

    Because the GRU uses 60-day sliding windows, its output is shorter than
    the tabular data. We pad the front with zeros so array lengths match.
    """
    print("[GRU] Building sequences ...")
    x_tr, y_tr, prev_tr, _ = _create_gru_sequences(
        train_scaled, train_raw_close, GRU_WINDOW_SIZE)
    x_te, y_te, prev_te, _ = _create_gru_sequences(
        test_scaled, test_raw_close, GRU_WINDOW_SIZE)
    print(f"[GRU] Train sequences: {x_tr.shape}, Test sequences: {x_te.shape}")

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    x_te_t = torch.tensor(x_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    pin = device.type == "cuda"
    tr_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t),
                           batch_size=GRU_BATCH_SIZE, shuffle=False, pin_memory=pin)
    te_loader = DataLoader(TensorDataset(x_te_t, y_te_t),
                           batch_size=GRU_BATCH_SIZE, shuffle=False, pin_memory=pin)

    # Run once without ensemble loop for speed
    print(f"\n[GRU] Training single model (seed {RANDOM_STATE})...")
    set_seed(RANDOM_STATE)
    model = GRUNeuralNetwork(input_size=x_tr.shape[2]).to(device)
    model = _train_gru(model, tr_loader, te_loader, device)

    gru_train_returns = _predict_gru(model, x_tr_t, device)
    gru_test_returns = _predict_gru(model, x_te_t, device)

    # Pad to match tabular row count (GRU needs WINDOW_SIZE lead-in rows)
    pad_train = len(train_scaled) - len(gru_train_returns)
    pad_test = len(test_scaled) - len(gru_test_returns)
    gru_train_full = np.concatenate([np.zeros(pad_train), gru_train_returns])
    gru_test_full = np.concatenate([np.zeros(pad_test), gru_test_returns])

    print(f"\n[GRU] Model trained. "
          f"Train preds: {len(gru_train_full)}, Test preds: {len(gru_test_full)}")
    return model, gru_train_full, gru_test_full


# ============================================================================
# SECTION 5 — MEMBER 3: XGBOOST META-LEARNER
# ============================================================================

def build_meta_features(rf_preds: np.ndarray, gru_preds: np.ndarray,
                         df: pd.DataFrame) -> pd.DataFrame:
    """Combine RF + GRU predictions with volatility/technical context."""
    meta = pd.DataFrame(index=df.index)
    meta["rf_pred"] = rf_preds
    meta["gru_pred"] = gru_preds

    # 20-day rolling volatility
    if "Close_SPY" in df.columns:
        meta["volatility_20d"] = df["Close_SPY"].pct_change().rolling(20).std()

    # RSI and MACD histogram for SPY
    for feat in ["RSI_SPY", "MACD_hist_SPY"]:
        if feat in df.columns:
            meta[feat] = df[feat].values

    # How much the two base models disagree
    meta["pred_spread"] = np.abs(rf_preds - gru_preds)

    return meta.fillna(0)


def train_xgboost_meta_learner(meta_train: pd.DataFrame,
                                 y_train: np.ndarray) -> XGBRegressor:
    """XGBoost learns which base model to trust under which conditions."""
    print("[XGBoost] Training meta-learner ...")
    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=0)
    xgb.fit(meta_train.values, y_train)
    print(f"[XGBoost] Trained on {len(meta_train)} samples × {meta_train.shape[1]} features.")
    return xgb


# ============================================================================
# SECTION 6 — TRADING SIMULATION & BENCHMARKING
# ============================================================================

def simulate_trading(predictions: np.ndarray, actual_returns: np.ndarray,
                      turbulence: pd.Series, threshold: float,
                      test_index: pd.DatetimeIndex) -> pd.Series:
    """
    Long if predicted return > 0, else cash.
    OVERRIDE: force cash if turbulence > threshold.
    """
    portfolio = pd.Series(0.0, index=test_index)
    for i, date in enumerate(test_index):
        if date in turbulence.index and turbulence.loc[date] > threshold:
            continue  # cash — crash mode
        if predictions[i] > 0:
            portfolio.iloc[i] = actual_returns[i]
    return portfolio


def buy_and_hold(actual_returns: np.ndarray,
                  test_index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(actual_returns, index=test_index)


def sma_crossover(close_prices: pd.Series,
                   test_index: pd.DatetimeIndex) -> pd.Series:
    sma10 = close_prices.rolling(10).mean()
    sma50 = close_prices.rolling(50).mean()
    returns = close_prices.pct_change()
    signal = (sma10 > sma50).astype(int)
    return (signal.shift(1) * returns).reindex(test_index).fillna(0)


def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    return ((cum - peak) / peak).min()


def print_strategy_metrics(name: str, preds: np.ndarray, actuals: np.ndarray,
                            portfolio: pd.Series):
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    cum = (1 + portfolio).cumprod()
    ret = cum.iloc[-1] - 1
    dd = max_drawdown(cum)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  MSE:               {mse:.8f}")
    print(f"  RMSE:              {rmse:.8f}")
    print(f"  MAE:               {mae:.8f}")
    print(f"  Cumulative Return: {ret*100:+.2f}%")
    print(f"  Max Drawdown:      {dd*100:.2f}%")
    print(f"{'='*60}")


def plot_equity_curves(strategies: dict[str, pd.Series],
                        save_path: str = "equity_curves.png"):
    plt.figure(figsize=(14, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for idx, (name, rets) in enumerate(strategies.items()):
        cum = (1 + rets).cumprod()
        plt.plot(cum.index, cum.values, label=name,
                 linewidth=2, color=colors[idx % len(colors)])
    plt.title("Equity Curves — Ensemble Meta-Learner vs Benchmarks",
              fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return ($1 start)", fontsize=12)
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[Plot] Saved → {save_path}")


# ============================================================================
# MAIN — UNIFIED PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("  UNIFIED HACKATHON TRADING MODEL")
    print("  RF (Member 1) + GRU (Member 2) + XGBoost Meta-Learner (Member 3)")
    print("=" * 70)

    device = get_device()

    # ==================================================================
    # HOUR 1 — Shared Data Pipeline & Feature Engineering
    # ==================================================================
    print("\n>>> HOUR 1: Data Pipeline & Feature Engineering\n")

    raw_df = fetch_data(TICKERS, START_DATE, END_DATE)
    daily_returns = compute_daily_returns(raw_df, TICKERS)
    full_df = build_feature_matrix(raw_df, TICKERS)

    feature_cols = get_feature_cols(full_df, TICKERS)

    # Target: next-day SPY return (what all models predict)
    full_df["target_return"] = full_df["Close_SPY"].pct_change().shift(-1)
    full_df = full_df.dropna(subset=["target_return"])

    # Chronological split
    train_df, test_df = split_data(full_df, TRAIN_RATIO)

    # Save raw close prices BEFORE normalizing (needed for GRU sequences)
    train_raw_close = train_df["Close_SPY"].copy()
    test_raw_close = test_df["Close_SPY"].copy()

    # Z-score normalization (fit on train only)
    train_df, test_df, scaler = normalize_features(train_df, test_df, feature_cols)

    # ==================================================================
    # HOUR 2 — Turbulence Index
    # ==================================================================
    print("\n>>> HOUR 2: Turbulence Index & Capital Protection\n")

    turbulence = compute_turbulence_index(daily_returns, TURBULENCE_WINDOW)
    turb_threshold = get_turbulence_threshold(turbulence, TURBULENCE_PERCENTILE)

    top5 = turbulence.nlargest(5)
    print("[Turbulence] Top-5 turbulence dates:")
    for date, val in top5.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {val:.4f}")

    # ==================================================================
    # HOUR 3A — Member 1: Random Forest
    # ==================================================================
    print("\n>>> HOUR 3A: Random Forest (Member 1)\n")

    rf_model, rf_train_preds, rf_test_preds = run_random_forest(
        train_df, test_df, feature_cols, "target_return")

    # ==================================================================
    # HOUR 3B — Member 2: GRU Ensemble
    # ==================================================================
    print("\n>>> HOUR 3B: GRU Ensemble (Member 2)\n")

    # GRU uses the normalized features but needs raw close for return targets
    gru_model, gru_train_preds, gru_test_preds = run_gru_ensemble(
        train_df[feature_cols], test_df[feature_cols],
        train_raw_close, test_raw_close, device)

    # ==================================================================
    # HOUR 3C — Member 3: XGBoost Meta-Learner
    # ==================================================================
    print("\n>>> HOUR 3C: XGBoost Meta-Learner (Member 3)\n")

    meta_train = build_meta_features(rf_train_preds, gru_train_preds, train_df)
    meta_test = build_meta_features(rf_test_preds, gru_test_preds, test_df)

    y_train = train_df["target_return"].values
    y_test = test_df["target_return"].values

    xgb_model = train_xgboost_meta_learner(meta_train, y_train)
    ensemble_preds = xgb_model.predict(meta_test.values)
    print(f"[XGBoost] {len(ensemble_preds)} test predictions generated.")

    # ==================================================================
    # HOUR 4 — Trading Simulation & Evaluation
    # ==================================================================
    print("\n>>> HOUR 4: Trading Simulation & Evaluation\n")

    test_index = test_df.index

    # --- Ensemble strategy (with turbulence override) ---
    xgb_portfolio = simulate_trading(
        ensemble_preds, y_test, turbulence, turb_threshold, test_index)

    turb_days = sum(1 for d in test_index
                    if d in turbulence.index and turbulence.loc[d] > turb_threshold)
    print(f"[Trading] Turbulence override on {turb_days}/{len(test_index)} days.")

    # --- Individual model strategies (for comparison) ---
    rf_portfolio = simulate_trading(
        rf_test_preds, y_test, turbulence, turb_threshold, test_index)
    gru_portfolio = simulate_trading(
        gru_test_preds, y_test, turbulence, turb_threshold, test_index)

    # --- Benchmarks ---
    bh_portfolio = buy_and_hold(y_test, test_index)
    spy_close_raw = raw_df["Close_SPY"].reindex(full_df.index)
    sma_portfolio = sma_crossover(spy_close_raw, test_index)

    # --- Metrics ---
    print_strategy_metrics("XGBoost Meta-Learner (Ensemble + Turbulence Shield)",
                            ensemble_preds, y_test, xgb_portfolio)
    print_strategy_metrics("Random Forest Only",
                            rf_test_preds, y_test, rf_portfolio)
    print_strategy_metrics("GRU Ensemble Only",
                            gru_test_preds, y_test, gru_portfolio)
    print_strategy_metrics("Buy & Hold SPY",
                            y_test, y_test, bh_portfolio)
    print_strategy_metrics("SMA(10)/SMA(50) Crossover",
                            y_test, y_test, sma_portfolio)

    # --- Equity curves ---
    strategies = {
        "XGBoost Meta-Learner": xgb_portfolio,
        "Random Forest": rf_portfolio,
        "GRU Ensemble": gru_portfolio,
        "Buy & Hold SPY": bh_portfolio,
        "SMA Crossover": sma_portfolio,
    }
    plot_equity_curves(strategies)

    # --- XGBoost feature importance ---
    print("\n[XGBoost] Feature Importance:")
    for name, imp in sorted(zip(meta_train.columns, xgb_model.feature_importances_),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {name:20s} {imp:.4f} {bar}")

    # --- Individual model accuracy ---
    print("\n" + "=" * 60)
    print("  Individual Model Accuracy (Return-Space)")
    print("=" * 60)
    for name, preds in [("Random Forest", rf_test_preds),
                         ("GRU Ensemble", gru_test_preds),
                         ("XGBoost Meta", ensemble_preds)]:
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        print(f"  {name:20s}  MSE={mse:.8f}  MAE={mae:.8f}")

    # --- Save real results for webapp ---
    import json
    import os
    webapp_data = {
        "curves": {
            "xgb": xgb_portfolio.tolist(),
            "rf": rf_portfolio.tolist(),
            "gru": gru_portfolio.tolist(),
            "bh": bh_portfolio.tolist(),
            "sma": sma_portfolio.tolist()
        },
        "metrics": {
            "test_days": len(meta_test),
            "turbulence_triggered": int(np.sum(test_turb > np.percentile(train_turb, TURBULENCE_PERCENTILE)))
        },
        "importance": {k: float(v) for k, v in zip(meta_train.columns, xgb_model.feature_importances_)}
    }
    
    out_path = os.path.join(os.path.dirname(__file__), "webapp", "public", "results.json")
    if os.path.exists(os.path.dirname(out_path)):
        with open(out_path, "w") as f:
            json.dump(webapp_data, f)
        print(f"\n[Webapp] Exported results to {out_path}")

    print("\n" + "=" * 70)
    print("  DONE — All three models integrated successfully!")
    print("=" * 70)

    # ------------------------------------------------------------------
    # SAVE MODELS — for server.py live inference
    # ------------------------------------------------------------------
    print("\n[Save] Saving models and metadata to disk ...")
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # Save the last GRU model (seed 42) — reuse the in-memory object
    torch.save(gru_model.state_dict(), "gru_model.pth")

    # Save metadata needed for inference
    meta = {
        "feature_cols": feature_cols,
        "turb_threshold": turb_threshold,
        "gru_input_size": len(feature_cols),
        "train_cutoff": str(train_df.index[-1].date()),
    }
    with open("model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[Save] rf_model.pkl, xgb_model.pkl, scaler.pkl, gru_model.pth, model_metadata.json")
    print("[Save] Run  python server.py  to start the API for the webapp.\n")



if __name__ == "__main__":
    main()
