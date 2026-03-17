"""
Data Pipeline & Feature Engineering Module
==========================================
Shared data module for the hackathon trading model.
Handles OHLCV data acquisition, technical indicator computation,
chronological train/test splitting, and Z-Score normalization.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "LMT", "RTX", "NOC"]
TRAIN_RATIO = 0.80
SMA_WINDOW = 50
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# ---------------------------------------------------------------------------
# Data Acquisition
# ---------------------------------------------------------------------------
def fetch_ohlcv(tickers: list[str] = TICKERS,
                start: str = "2015-01-01",
                end: str = "2025-12-31") -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for each ticker via yfinance.

    Returns a dict mapping ticker -> DataFrame with columns:
    Open, High, Low, Close, Volume.
    Missing values are forward-filled.
    """
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        data[ticker] = df
    return data


# ---------------------------------------------------------------------------
# Feature Engineering (trailing indicators only – no look-ahead bias)
# ---------------------------------------------------------------------------
def compute_sma(series: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    """Compute trailing Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Compute trailing Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(series: pd.Series,
                 fast: int = MACD_FAST,
                 slow: int = MACD_SLOW,
                 signal: int = MACD_SIGNAL) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, Signal line, and MACD histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator features to a single-ticker DataFrame.

    Also creates the target column 'Next_Close' (next day's closing price).
    Drops rows with NaN introduced by rolling windows.
    """
    df = df.copy()

    # Technical indicators
    df["SMA_50"] = compute_sma(df["Close"], SMA_WINDOW)
    df["RSI_14"] = compute_rsi(df["Close"], RSI_WINDOW)
    macd, macd_signal, macd_hist = compute_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist

    # Daily return (for evaluation / trading logic later)
    df["Daily_Return"] = df["Close"].pct_change()

    # Target: next day's closing price
    df["Next_Close"] = df["Close"].shift(-1)

    # Drop rows with NaN from rolling windows or shift
    df.dropna(inplace=True)

    return df


# ---------------------------------------------------------------------------
# Train / Test Split (strictly chronological)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_50", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
]
TARGET_COL = "Next_Close"


def split_data(df: pd.DataFrame,
               feature_cols: list[str] = FEATURE_COLS,
               target_col: str = TARGET_COL,
               train_ratio: float = TRAIN_RATIO):
    """Chronological train/test split — absolutely no shuffling.

    Returns (X_train, X_test, y_train, y_test, train_df, test_df).
    train_df / test_df retain the full DataFrame rows for later analysis.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    return X_train, X_test, y_train, y_test, train_df, test_df


# ---------------------------------------------------------------------------
# Z-Score Normalization (fit on train only — no data leakage)
# ---------------------------------------------------------------------------
def normalize(X_train: np.ndarray,
              X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Apply StandardScaler. Fit on training data only, transform both."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Convenience: full pipeline for one ticker
# ---------------------------------------------------------------------------
def build_pipeline(ticker: str = "SPY",
                   start: str = "2015-01-01",
                   end: str = "2025-12-31"):
    """Run the entire data pipeline for a single ticker.

    Returns a dict with all artefacts needed downstream:
        X_train, X_test, y_train, y_test   – scaled numpy arrays / targets
        scaler          – fitted StandardScaler
        train_df, test_df  – full DataFrames (unscaled, with all columns)
        feature_cols    – list of feature column names
    """
    raw = fetch_ohlcv([ticker], start=start, end=end)
    df = add_features(raw[ticker])
    X_train, X_test, y_train, y_test, train_df, test_df = split_data(df)
    X_train_s, X_test_s, scaler = normalize(X_train, X_test)

    return {
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": FEATURE_COLS,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running data pipeline for SPY …")
    result = build_pipeline("SPY")
    print(f"  Training samples : {result['X_train'].shape[0]}")
    print(f"  Testing samples  : {result['X_test'].shape[0]}")
    print(f"  Features         : {result['X_train'].shape[1]}")
    print("Data pipeline OK ✓")
