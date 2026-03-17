"""
Member 3: XGBoost Meta-Learner & Risk Governance
=================================================
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
