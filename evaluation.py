"""
Evaluation & Benchmarking Module
=================================
Computes prediction accuracy metrics (MSE, RMSE, MAE),
simulates a simple long/cash trading strategy,
and benchmarks against Buy-and-Hold and SMA crossover.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------------------
# Prediction Accuracy Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray) -> dict[str, float]:
    """Return MSE, RMSE, and MAE for model predictions."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


def print_metrics(metrics: dict[str, float], label: str = "Model") -> None:
    """Pretty-print prediction accuracy metrics."""
    print(f"\n{'='*50}")
    print(f"  {label} — Prediction Accuracy")
    print(f"{'='*50}")
    for name, val in metrics.items():
        print(f"  {name:6s}: {val:12.4f}")
    print()


# ---------------------------------------------------------------------------
# Trading Simulation Helpers
# ---------------------------------------------------------------------------
def _daily_returns(prices: pd.Series) -> pd.Series:
    """Compute daily percentage returns from a price series."""
    return prices.pct_change().fillna(0.0)


def simulate_strategy(test_df: pd.DataFrame,
                      predictions: np.ndarray) -> pd.DataFrame:
    """Simulate a long/cash strategy based on predicted next-day return.

    Rule: if predicted next-day price > today's close → go long (1);
          otherwise hold cash (0).

    Returns a DataFrame with columns:
        Close, Daily_Return, Signal, Strategy_Return, Cumulative_Strategy
    """
    sim = test_df[["Close"]].copy()
    sim["Daily_Return"] = sim["Close"].pct_change().fillna(0.0)

    # Signal: 1 = long, 0 = cash  (decision made at close of day t,
    # return earned on day t+1)
    sim["Signal"] = (predictions > sim["Close"].values).astype(int)
    # Strategy return: signal shifted by 1 day (we act on *today's* signal
    # and earn *tomorrow's* return, but since our test_df rows are already
    # aligned so that row t has target = close(t+1), we can simply multiply)
    sim["Strategy_Return"] = sim["Signal"] * sim["Daily_Return"]
    sim["Cumulative_Strategy"] = (1 + sim["Strategy_Return"]).cumprod()
    return sim


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def buy_and_hold(test_df: pd.DataFrame) -> pd.Series:
    """Buy-and-Hold benchmark: cumulative return of holding SPY."""
    daily_ret = test_df["Close"].pct_change().fillna(0.0)
    return (1 + daily_ret).cumprod()


def sma_crossover(test_df: pd.DataFrame,
                  short_window: int = 10,
                  long_window: int = 50) -> pd.Series:
    """SMA(10)/SMA(50) crossover benchmark on the test set.

    Signal: long when SMA_short > SMA_long, else cash.
    Returns cumulative return series.
    """
    close = test_df["Close"]
    sma_short = close.rolling(short_window, min_periods=1).mean()
    sma_long = close.rolling(long_window, min_periods=1).mean()
    signal = (sma_short > sma_long).astype(int)
    daily_ret = close.pct_change().fillna(0.0)
    strat_ret = signal * daily_ret
    return (1 + strat_ret).cumprod()


# ---------------------------------------------------------------------------
# Portfolio Metrics
# ---------------------------------------------------------------------------
def max_drawdown(cum_returns: pd.Series) -> float:
    """Calculate maximum drawdown from a cumulative return series."""
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()


def total_return(cum_returns: pd.Series) -> float:
    """Calculate total return from cumulative return series."""
    return cum_returns.iloc[-1] - 1.0


# ---------------------------------------------------------------------------
# Full Evaluation Report
# ---------------------------------------------------------------------------
def full_report(y_test: np.ndarray,
                test_preds: np.ndarray,
                test_df: pd.DataFrame,
                label: str = "Random Forest") -> None:
    """Print a comprehensive evaluation report."""
    # 1. Prediction accuracy
    metrics = compute_metrics(y_test, test_preds)
    print_metrics(metrics, label=label)

    # 2. Trading simulation
    sim = simulate_strategy(test_df, test_preds)
    bh = buy_and_hold(test_df)
    sma = sma_crossover(test_df)

    strat_ret = total_return(sim["Cumulative_Strategy"])
    strat_dd = max_drawdown(sim["Cumulative_Strategy"])
    bh_ret = total_return(bh)
    bh_dd = max_drawdown(bh)
    sma_ret = total_return(sma)
    sma_dd = max_drawdown(sma)

    print(f"{'='*50}")
    print(f"  Strategy Comparison (Test Period)")
    print(f"{'='*50}")
    print(f"  {'Strategy':<25s} {'Return':>10s} {'Max DD':>10s}")
    print(f"  {'-'*45}")
    print(f"  {label+' Strategy':<25s} {strat_ret:>9.2%} {strat_dd:>9.2%}")
    print(f"  {'Buy & Hold SPY':<25s} {bh_ret:>9.2%} {bh_dd:>9.2%}")
    print(f"  {'SMA(10/50) Crossover':<25s} {sma_ret:>9.2%} {sma_dd:>9.2%}")
    print()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Evaluation module loaded OK ✓")
