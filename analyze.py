"""
analyze.py -- Post-Training Model Analysis & Improvement Recommendations

Reads the saved output from main.py (model_results.npz) and performs:
  1. Error decomposition (bias vs variance, by stock, by time window)
  2. Rolling performance analysis (detect when the model degrades)
  3. Market regime analysis (high vs low volatility performance)
  4. Residual diagnostics (autocorrelation, normality)
  5. Actionable improvement recommendations based on findings

Usage:
    python main.py          # Train the model first (saves model_results.npz)
    python analyze.py       # Run this script to analyze the results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# Load Results
# =============================================================================

RESULTS_FILE = "model_results.npz"

if not os.path.exists(RESULTS_FILE):
    print(f"ERROR: '{RESULTS_FILE}' not found.")
    print("Run 'python main.py' first to train the model and save results.")
    sys.exit(1)

data = np.load(RESULTS_FILE, allow_pickle=True)
predictions = data["predictions"]   # (num_samples, num_stocks) in dollar prices
actuals = data["actuals"]           # (num_samples, num_stocks) in dollar prices
train_losses = data["train_losses"]
val_losses = data["val_losses"]
tickers = list(data["tickers"])

num_samples, num_stocks = predictions.shape
errors = predictions - actuals
pct_errors = (errors / (actuals + 1e-10)) * 100

print("=" * 80)
print("              POST-TRAINING ANALYSIS & IMPROVEMENT REPORT")
print("=" * 80)
print(f"  Loaded {num_samples} test samples across {num_stocks} stocks")
print(f"  Training epochs completed: {len(train_losses)}")
print()


# =============================================================================
# 1. ERROR DECOMPOSITION -- Bias vs Variance
# =============================================================================

print("=" * 80)
print("  1. ERROR DECOMPOSITION (Bias vs Variance)")
print("=" * 80)

print(f"\n  {'Ticker':<8} {'Bias ($)':>10} {'Std ($)':>10} {'|Bias|/Std':>12} {'Diagnosis':>18}")
print("-" * 70)

bias_issues = []
variance_issues = []

for idx, ticker in enumerate(tickers):
    stock_errors = errors[:, idx]
    bias = np.mean(stock_errors)         # Systematic over/under-prediction
    std = np.std(stock_errors)            # Prediction noise
    ratio = abs(bias) / (std + 1e-10)

    if ratio > 0.5:
        diagnosis = "HIGH BIAS"
        bias_issues.append(ticker)
    elif std > np.mean(np.abs(actuals[:, idx])) * 0.15:
        diagnosis = "HIGH VARIANCE"
        variance_issues.append(ticker)
    else:
        diagnosis = "Balanced"

    print(f"  {ticker:<8} {bias:>+10.2f} {std:>10.2f} {ratio:>12.3f} {diagnosis:>18}")

print()
if bias_issues:
    print(f"  [!] Bias-dominated stocks: {', '.join(bias_issues)}")
    print("    -> The model consistently over/under-predicts these. Consider:")
    print("      - Adding stock-specific output bias correction")
    print("      - Including sector/market-cap features to differentiate stocks")
if variance_issues:
    print(f"  [!] Variance-dominated stocks: {', '.join(variance_issues)}")
    print("    -> Predictions are noisy/unstable. Consider:")
    print("      - Increasing dropout or adding more regularization")
    print("      - Using ensemble predictions (average multiple model runs)")
print()


# =============================================================================
# 2. ROLLING PERFORMANCE -- When Does the Model Degrade?
# =============================================================================

print("=" * 80)
print("  2. ROLLING PERFORMANCE (30-Day Windows)")
print("=" * 80)

window = 30
if num_samples >= window:
    rolling_rmse_per_stock = {}
    overall_rolling_rmse = []

    for start in range(0, num_samples - window + 1, window // 2):
        end = start + window
        window_errors = errors[start:end, :]
        window_rmse = np.sqrt(np.mean(window_errors ** 2))
        overall_rolling_rmse.append((start, end, window_rmse))

    # Find best and worst windows
    sorted_windows = sorted(overall_rolling_rmse, key=lambda x: x[2])
    best = sorted_windows[0]
    worst = sorted_windows[-1]

    print(f"\n  Best 30-day window:  days {best[0]:3d}-{best[1]:3d}  RMSE: ${best[2]:.2f}")
    print(f"  Worst 30-day window: days {worst[0]:3d}-{worst[1]:3d}  RMSE: ${worst[2]:.2f}")
    print(f"  Degradation factor:  {worst[2]/best[2]:.1f}x")

    # Check if performance degrades over time (trend in rolling RMSE)
    rmse_values = [w[2] for w in overall_rolling_rmse]
    midpoints = [(w[0] + w[1]) / 2 for w in overall_rolling_rmse]
    if len(midpoints) >= 3:
        trend_slope, _, r_value, p_value, _ = stats.linregress(midpoints, rmse_values)
        trend_dir = "DEGRADING (up)" if trend_slope > 0 else "IMPROVING (down)"
        significance = "significant" if p_value < 0.05 else "not statistically significant"
        print(f"\n  Performance trend: {trend_dir} (slope: {trend_slope:.4f}, p={p_value:.3f} -- {significance})")

        if trend_slope > 0 and p_value < 0.05:
            print("    -> Model performance worsens over time. This suggests:")
            print("      - The market regime shifted after the training period")
            print("      - Consider retraining on more recent data")
            print("      - Consider a shorter lookback window to adapt faster")
        elif trend_slope < 0 and p_value < 0.05:
            print("    -> Model performs better on later data -- the test set's early")
            print("      portion may include an unusual market transition period.")

    # Generate rolling performance plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(midpoints, rmse_values, marker="o", linewidth=2, color="#2196F3")
    ax.set_title("Rolling 30-Day RMSE Over Test Period", fontsize=14)
    ax.set_xlabel("Test Day", fontsize=12)
    ax.set_ylabel("RMSE ($)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rolling_performance.png", dpi=150)
    print("\n  Plot saved: rolling_performance.png")
else:
    print(f"\n  Not enough samples ({num_samples}) for 30-day rolling analysis.")
print()


# =============================================================================
# 3. VOLATILITY REGIME ANALYSIS
# =============================================================================

print("=" * 80)
print("  3. VOLATILITY REGIME ANALYSIS")
print("=" * 80)

# Measure each day's realized volatility as the absolute % change
daily_returns = np.diff(actuals, axis=0) / (actuals[:-1] + 1e-10)
daily_vol = np.abs(daily_returns).mean(axis=1)  # Average across stocks

if len(daily_vol) > 10:
    vol_median = np.median(daily_vol)

    # Split errors into high-vol and low-vol days
    high_vol_mask = daily_vol > vol_median
    low_vol_mask = ~high_vol_mask

    # Align errors (we lose one sample due to diff)
    aligned_errors = errors[1:]  # Drop first to match daily_returns length

    if high_vol_mask.sum() > 0 and low_vol_mask.sum() > 0:
        high_vol_rmse = np.sqrt(np.mean(aligned_errors[high_vol_mask] ** 2))
        low_vol_rmse = np.sqrt(np.mean(aligned_errors[low_vol_mask] ** 2))

        print(f"\n  Low-volatility days (n={low_vol_mask.sum()}):   RMSE = ${low_vol_rmse:.2f}")
        print(f"  High-volatility days (n={high_vol_mask.sum()}):  RMSE = ${high_vol_rmse:.2f}")
        print(f"  Ratio (high/low): {high_vol_rmse/low_vol_rmse:.2f}x")

        if high_vol_rmse / low_vol_rmse > 1.5:
            print("\n    -> The model struggles significantly during volatile periods.")
            print("      Recommendations:")
            print("      - Add a VIX or realized-vol feature to condition predictions")
            print("      - Train with a volatility-weighted loss (penalize high-vol days less)")
            print("      - Consider separate models for high/low volatility regimes")
        else:
            print("\n    -> Model handles both regimes reasonably well.")
print()


# =============================================================================
# 4. RESIDUAL DIAGNOSTICS
# =============================================================================

print("=" * 80)
print("  4. RESIDUAL DIAGNOSTICS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Residual Diagnostics", fontsize=16, fontweight="bold")

# Flatten all errors for overall diagnostics
flat_errors = errors.flatten()
flat_pct_errors = pct_errors.flatten()

# (a) Error distribution
ax = axes[0, 0]
ax.hist(flat_pct_errors, bins=50, edgecolor="black", alpha=0.7, color="#2196F3")
ax.axvline(0, color="red", linestyle="--", linewidth=2)
ax.set_title("Error Distribution (% Error)")
ax.set_xlabel("Percentage Error")
ax.set_ylabel("Frequency")

# Normality test
shapiro_stat, shapiro_p = stats.shapiro(flat_pct_errors[:5000])  # Shapiro max ~5000 samples
skewness = stats.skew(flat_pct_errors)
kurtosis = stats.kurtosis(flat_pct_errors)

print(f"\n  Error Distribution:")
print(f"    Skewness:  {skewness:+.3f}  ({'right-skewed (tends to overpredict)' if skewness > 0.5 else 'left-skewed (tends to underpredict)' if skewness < -0.5 else 'approximately symmetric'})")
print(f"    Kurtosis:  {kurtosis:+.3f}  ({'heavy tails -- occasional large errors' if kurtosis > 1 else 'light tails -- errors are well-contained' if kurtosis < -0.5 else 'near-normal tails'})")
print(f"    Normality: p={shapiro_p:.4f} ({'Normal' if shapiro_p > 0.05 else 'Non-normal -- consider robust loss like Huber'})")

# (b) Q-Q Plot
ax = axes[0, 1]
stats.probplot(flat_pct_errors, dist="norm", plot=ax)
ax.set_title("Q-Q Plot (% Errors vs Normal)")
ax.get_lines()[0].set_color("#2196F3")
ax.get_lines()[1].set_color("#FF5722")

# (c) Autocorrelation of errors (do errors persist over time?)
ax = axes[1, 0]
avg_errors_per_day = errors.mean(axis=1)  # Average error across stocks per day
max_lags = min(40, len(avg_errors_per_day) - 1)
autocorr = [np.corrcoef(avg_errors_per_day[:-lag], avg_errors_per_day[lag:])[0, 1]
            for lag in range(1, max_lags + 1)]
ax.bar(range(1, max_lags + 1), autocorr, color="#2196F3", alpha=0.7)
ax.axhline(y=1.96/np.sqrt(len(avg_errors_per_day)), color="red", linestyle="--", alpha=0.5)
ax.axhline(y=-1.96/np.sqrt(len(avg_errors_per_day)), color="red", linestyle="--", alpha=0.5)
ax.set_title("Error Autocorrelation (by lag)")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")

significant_lags = [i + 1 for i, ac in enumerate(autocorr)
                    if abs(ac) > 1.96 / np.sqrt(len(avg_errors_per_day))]

print(f"\n  Autocorrelation:")
if significant_lags:
    print(f"    Significant lags: {significant_lags}")
    print("    -> Errors are NOT independent -- the model repeats mistakes.")
    print("      Consider adding lagged-error features or an error-correction layer.")
else:
    print("    No significant autocorrelation -- errors appear random (good).")

# (d) Error vs Actual Price scatter
ax = axes[1, 1]
ax.scatter(actuals.flatten(), flat_pct_errors, alpha=0.15, s=5, color="#2196F3")
ax.axhline(0, color="red", linestyle="--")
ax.set_title("% Error vs Actual Price")
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("% Error")

# Check if error correlates with price level
price_error_corr, price_error_p = stats.pearsonr(actuals.flatten(), flat_pct_errors)
print(f"\n  Price-Error Correlation:")
print(f"    r={price_error_corr:+.3f}, p={price_error_p:.4f}")
if abs(price_error_corr) > 0.15:
    if price_error_corr > 0:
        print("    -> Model has LARGER % errors on expensive stocks.")
        print("      Consider log-transforming prices before scaling.")
    else:
        print("    -> Model has LARGER % errors on cheaper stocks.")
        print("      Consider per-stock normalization instead of global MinMaxScaler.")
else:
    print("    -> No significant bias by price level (good).")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("residual_diagnostics.png", dpi=150)
print(f"\n  Plot saved: residual_diagnostics.png")
print()


# =============================================================================
# 5. TRAINING DYNAMICS ANALYSIS
# =============================================================================

print("=" * 80)
print("  5. TRAINING DYNAMICS ANALYSIS")
print("=" * 80)

final_train = train_losses[-1]
final_val = val_losses[-1]
best_val = min(val_losses)
best_val_epoch = np.argmin(val_losses) + 1
gap = final_val / final_train if final_train > 0 else float("inf")

print(f"\n  Final train loss:     {final_train:.6f}")
print(f"  Final val loss:       {final_val:.6f}")
print(f"  Best val loss:        {best_val:.6f} (epoch {best_val_epoch})")
print(f"  Overfit ratio:        {gap:.2f}x  (val/train)")

if gap > 5:
    print("\n  [!!] SEVERE OVERFITTING detected.")
    print("    Recommendations:")
    print("    - Increase dropout from 0.3 -> 0.4 or 0.5")
    print("    - Reduce model complexity (fewer LSTM hidden units or dense layers)")
    print("    - Add data augmentation (jittering input features)")
    print("    - Try a shorter lookback window (e.g., 30 instead of 60)")
elif gap > 2:
    print("\n  [!] MODERATE OVERFITTING detected.")
    print("    Recommendations:")
    print("    - Slight increase in dropout or weight decay")
    print("    - Consider adding noise to training inputs")
elif gap < 1.3:
    print("\n  [OK] Good fit -- train and val losses are close. Model may benefit from")
    print("    more capacity (more hidden units, more layers) to learn subtler patterns.")
else:
    print("\n  [OK] Reasonable fit -- some overfitting but within acceptable range.")

# Check for training instability
val_diffs = np.diff(val_losses)
spikes = np.sum(val_diffs > np.std(val_diffs) * 2)
print(f"\n  Validation loss spikes: {spikes}")
if spikes > 3:
    print("    -> Training is unstable. Consider:")
    print("      - Lowering the initial learning rate (e.g., 0.0005)")
    print("      - Using gradient clipping with a smaller max_norm")
    print("      - Warming up the learning rate (linear warmup scheduler)")
print()


# =============================================================================
# 6. PER-STOCK IMPROVEMENT PRIORITIES
# =============================================================================

print("=" * 80)
print("  6. PER-STOCK IMPROVEMENT PRIORITIES")
print("=" * 80)

stock_metrics = []
for idx, ticker in enumerate(tickers):
    stock_err = errors[:, idx]
    stock_act = actuals[:, idx]
    rmse = np.sqrt(np.mean(stock_err ** 2))
    mape = np.mean(np.abs(stock_err / (stock_act + 1e-10))) * 100
    bias = np.mean(stock_err)

    # Directional accuracy
    if len(stock_act) > 1:
        actual_dir = np.diff(stock_act) > 0
        pred_dir = np.diff(predictions[:, idx]) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100
    else:
        dir_acc = 50.0

    # Calculate a composite "problem score" -- higher = needs more attention
    problem_score = (mape / 5) + (abs(bias) / rmse) + max(0, (55 - dir_acc) / 10)
    stock_metrics.append({
        "ticker": ticker, "rmse": rmse, "mape": mape,
        "bias": bias, "dir_acc": dir_acc, "score": problem_score
    })

# Sort by problem score (worst first)
stock_metrics.sort(key=lambda x: x["score"], reverse=True)

print(f"\n  {'Rank':<6} {'Ticker':<8} {'MAPE%':>8} {'Bias($)':>10} {'DirAcc%':>10} {'Priority':>10}")
print("-" * 60)
for rank, m in enumerate(stock_metrics, 1):
    priority = "[!!!] HIGH" if m["score"] > 3 else "[!!] MED" if m["score"] > 1.5 else "[OK] LOW"
    print(f"  {rank:<6} {m['ticker']:<8} {m['mape']:>8.2f} {m['bias']:>+10.2f} {m['dir_acc']:>10.1f} {priority:>10}")

print()


# =============================================================================
# 7. CONSOLIDATED RECOMMENDATIONS
# =============================================================================

print("=" * 80)
print("  7. CONSOLIDATED IMPROVEMENT RECOMMENDATIONS")
print("=" * 80)

recommendations = []

# Based on overfitting
if gap > 3:
    recommendations.append(
        "REGULARIZATION: Increase dropout to 0.4-0.5, or increase weight_decay to 1e-4."
    )

# Based on bias
if len(bias_issues) >= 3:
    recommendations.append(
        "BIAS CORRECTION: Add a learnable per-stock bias in the output layer, "
        "or include market-cap/sector as input features."
    )

# Based on volatility regime
if 'high_vol_rmse' in dir() and 'low_vol_rmse' in dir():
    if high_vol_rmse / low_vol_rmse > 1.5:
        recommendations.append(
            "VOLATILITY: Add VIX or a realized-volatility feature. "
            "Consider training with Huber loss to reduce sensitivity to outliers."
        )

# Based on autocorrelation
if significant_lags:
    recommendations.append(
        "ERROR PERSISTENCE: Add lagged prediction errors as input features "
        "(error-correction mechanism), since errors carry over between days."
    )

# Based on non-normality
if shapiro_p < 0.05 and kurtosis > 1:
    recommendations.append(
        "LOSS FUNCTION: Switch from MSE to Huber loss (nn.HuberLoss) -- "
        "the heavy-tailed error distribution makes MSE overly sensitive to outliers."
    )

# Based on training instability
if spikes > 3:
    recommendations.append(
        "STABILITY: Lower learning rate to 0.0005 and add a warmup scheduler."
    )

# Based on directional accuracy
worst_dir = min(stock_metrics, key=lambda x: x["dir_acc"])
if worst_dir["dir_acc"] < 50:
    recommendations.append(
        f"DIRECTION: {worst_dir['ticker']} has <50% directional accuracy. "
        "Consider adding a direction-classification auxiliary loss alongside MSE."
    )

# General recommendations
recommendations.append(
    "DATA: Expand training data by starting from 2015-01-01 instead of 2020-01-01 "
    "to give the model more market cycles to learn from."
)
recommendations.append(
    "ENSEMBLE: Train 3-5 models with different random seeds and average predictions "
    "to reduce variance and improve stability."
)

print()
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")
    print()

print("=" * 80)
print("  Analysis complete. Plots saved:")
print("    - rolling_performance.png    : RMSE over time")
print("    - residual_diagnostics.png   : Error distribution & diagnostics")
print("=" * 80)
