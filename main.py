import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =============================================================================
# Technical Indicator Functions
# =============================================================================

def compute_rsi(series, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, span1=12, span2=26):
    """Calculates the MACD line."""
    ema_fast = series.ewm(span=span1, adjust=False).mean()
    ema_slow = series.ewm(span=span2, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd


def compute_bollinger_band_width(series, window=20):
    """Calculates the Bollinger Band Width (normalized by the middle band)."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    bb_width = (2 * std) / sma  # Normalized width
    return bb_width


def compute_atr(high, low, close, window=14):
    """Calculates the Average True Range (ATR).
    When only Close is available, approximates High/Low from Close."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def compute_stochastic_oscillator(close, window=14):
    """Calculates the Stochastic Oscillator %K."""
    lowest_low = close.rolling(window=window).min()
    highest_high = close.rolling(window=window).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    return stoch_k


def compute_rate_of_change(series, window=10):
    """Calculates the Rate of Change (ROC)."""
    roc = (series - series.shift(window)) / (series.shift(window) + 1e-10) * 100
    return roc


# =============================================================================
# Data Pipeline
# =============================================================================

def fetch_and_prep_advanced_data(tickers, start_date="2015-01-01", lookback_window=60):
    """Expanded to 2015 for more market cycles (analyze.py rec #7)."""
    print(f"Downloading data for {len(tickers)} tickers from {start_date}...")

    # Download Close, High, Low, and Volume data
    data = yf.download(tickers, start=start_date)[["Close", "High", "Low", "Volume"]]

    # Download VIX as a market-wide volatility feature (analyze.py rec: VOLATILITY)
    vix_data = yf.download("^VIX", start=start_date)[["Close"]]
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = ["_".join(col).strip() for col in vix_data.columns.values]
        vix_col_name = [c for c in vix_data.columns if "Close" in c][0]
    else:
        vix_col_name = "Close"
    vix_series = vix_data[vix_col_name].reindex(data.index).ffill().bfill()

    # Flatten the multi-index columns if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() for col in data.columns.values]

    feature_list = []

    # Calculate indicators for each ticker
    for ticker in tickers:
        df_ticker = pd.DataFrame()
        close_col = f"Close_{ticker}" if f"Close_{ticker}" in data.columns else ticker
        high_col = f"High_{ticker}" if f"High_{ticker}" in data.columns else ticker
        low_col = f"Low_{ticker}" if f"Low_{ticker}" in data.columns else ticker
        vol_col = f"Volume_{ticker}" if f"Volume_{ticker}" in data.columns else ticker

        df_ticker["Close"] = data[close_col]
        df_ticker["Volume"] = data[vol_col]
        df_ticker["RSI"] = compute_rsi(df_ticker["Close"])
        df_ticker["MACD"] = compute_macd(df_ticker["Close"])
        df_ticker["BB_Width"] = compute_bollinger_band_width(df_ticker["Close"])
        df_ticker["ATR"] = compute_atr(data[high_col], data[low_col], data[close_col])
        df_ticker["Stoch_K"] = compute_stochastic_oscillator(df_ticker["Close"])
        df_ticker["ROC"] = compute_rate_of_change(df_ticker["Close"])
        df_ticker["VIX"] = vix_series.values[:len(df_ticker)]  # Market volatility

        feature_list.append(df_ticker)

    # Concatenate all stocks horizontally and drop NA values created by rolling windows
    combined_df = pd.concat(feature_list, axis=1, keys=tickers)
    combined_df.dropna(inplace=True)

    # 1. Build Graph Edge Index (Relational Data based on Close Price Correlation)
    close_prices = combined_df.xs("Close", axis=1, level=1)
    corr_matrix = close_prices.corr().values

    edge_index_list = []
    edge_weight_list = []
    threshold = 0.6

    num_nodes = len(tickers)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and abs(corr_matrix[i, j]) > threshold:
                edge_index_list.append([i, j])
                edge_weight_list.append(corr_matrix[i, j])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    # 2. Per-Stock Normalization (analyze.py rec: price-level bias fix)
    #    Each stock gets its own scaler instead of a single global scaler.
    num_features = 9  # Close, Volume, RSI, MACD, BB_Width, ATR, Stoch_K, ROC, VIX
    per_stock_scalers = []
    scaled_per_stock = []

    for i, ticker in enumerate(tickers):
        stock_data = combined_df[ticker].values  # (timesteps, num_features)
        scaler = MinMaxScaler()
        scaled_stock = scaler.fit_transform(stock_data)
        per_stock_scalers.append(scaler)
        scaled_per_stock.append(scaled_stock)

    # Stack into (Timesteps, Nodes, Features)
    scaled_values = np.stack(scaled_per_stock, axis=1)

    # 3. Generate Temporal Sequences (Sliding Window)
    X_temporal, y_target = [], []
    for i in range(len(scaled_values) - lookback_window):
        X_temporal.append(scaled_values[i : i + lookback_window])
        y_target.append(scaled_values[i + lookback_window, :, 0])  # Next day Close

    X_temporal = torch.tensor(np.array(X_temporal), dtype=torch.float32)
    y_target = torch.tensor(np.array(y_target), dtype=torch.float32)

    return X_temporal, y_target, edge_index, edge_weight, per_stock_scalers


# =============================================================================
# Model Architecture
# =============================================================================

class SimpleGCN(nn.Module):
    """Graph Convolutional Network with dropout."""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.3):
        super(SimpleGCN, self).__init__()
        self.W1 = nn.Linear(in_features, hidden_features)
        self.W2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_features)

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = edge_weight
        adj = adj + torch.eye(num_nodes, device=x.device)

        # Normalize adjacency
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj = adj / degree

        x = torch.relu(self.norm(self.W1(torch.matmul(adj, x))))
        x = self.dropout(x)
        x = self.W2(torch.matmul(adj, x))
        return x


class TemporalAttention(nn.Module):
    """Learns which timesteps in the LSTM sequence are most important."""
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        scores = self.attn(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)       # (batch, seq_len)
        # Weighted sum across timesteps
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_size)
        return context, weights


class LSTMGNNHybrid(nn.Module):
    """Improved LSTM-GNN hybrid with attention, dropout, and layer normalization."""
    def __init__(
        self,
        lstm_input,
        lstm_hidden,
        lstm_layers,
        gnn_input,
        gnn_hidden,
        gnn_output,
        dense_hidden,
        final_output,
        num_nodes=10,
        dropout=0.3,
    ):
        super(LSTMGNNHybrid, self).__init__()

        # LSTM with built-in dropout between layers
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Temporal attention mechanism
        self.attention = TemporalAttention(lstm_hidden)

        # Layer normalization after LSTM
        self.lstm_norm = nn.LayerNorm(lstm_hidden)

        # GCN with dropout
        self.gcn = SimpleGCN(gnn_input, gnn_hidden, gnn_output, dropout=dropout)

        # Dense layers with dropout
        combined_size = lstm_hidden + (num_nodes * gnn_output)
        self.fc1 = nn.Linear(combined_size, dense_hidden)
        self.fc2 = nn.Linear(dense_hidden, dense_hidden)
        self.fc3 = nn.Linear(dense_hidden, final_output)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, lstm_x, gnn_x, edge_index, edge_weight):
        # LSTM with attention
        lstm_out, _ = self.lstm(lstm_x)
        lstm_context, _ = self.attention(lstm_out)
        lstm_context = self.lstm_norm(lstm_context)
        lstm_context = self.dropout(lstm_context)

        # GNN branch
        gnn_out = self.gcn(gnn_x, edge_index, edge_weight)
        gnn_flat = gnn_out.view(-1)
        gnn_flat = gnn_flat.unsqueeze(0).repeat(lstm_context.size(0), 1)

        # Combine and predict
        combined = torch.cat((lstm_context, gnn_flat), dim=1)
        x = self.dropout(self.relu(self.fc1(combined)))
        x = self.dropout(self.relu(self.fc2(x)))
        out = self.fc3(x)

        return out


# =============================================================================
# Main Pipeline
# =============================================================================

# Target semiconductor stocks
target_tickers = [
    "AMD", "ASML", "AVGO", "INTC", "MRVL",
    "MU", "NVDA", "QCOM", "TSM", "TXN",
]

# Run the data pipeline
X_seq, y_target, edge_idx, edge_wt, stock_scalers = fetch_and_prep_advanced_data(
    target_tickers
)

print("\n--- Final PyTorch Tensor Shapes ---")
print(f"X_seq (Input): {X_seq.shape} -> (Samples, Sequence Length, Num Nodes, Num Features)")
print(f"y_target (Labels): {y_target.shape} -> (Samples, Num Nodes)")
print(f"edge_idx (Graph): {edge_idx.shape} -> (2, Num Edges)")

# --- 1. Chronological Train/Test Split ---
dataset_size = len(X_seq)
train_size = int(dataset_size * 0.8)

X_train, y_train = X_seq[:train_size], y_target[:train_size]
X_test, y_test = X_seq[train_size:], y_target[train_size:]

# --- 2. DataLoader Setup ---
batch_size = 32

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- 3. Model Initialization ---
num_nodes = 10
num_features = 9  # Close, Volume, RSI, MACD, BB_Width, ATR, Stoch_K, ROC, VIX

lstm_input_size = num_nodes * num_features
lstm_hidden_size = 128
lstm_layers = 2

gnn_input_size = num_features
gnn_hidden_size = 64
gnn_output_size = 32

dense_hidden_size = 64
final_output_size = num_nodes

model = LSTMGNNHybrid(
    lstm_input=lstm_input_size,
    lstm_hidden=lstm_hidden_size,
    lstm_layers=lstm_layers,
    gnn_input=gnn_input_size,
    gnn_hidden=gnn_hidden_size,
    gnn_output=gnn_output_size,
    dense_hidden=dense_hidden_size,
    final_output=final_output_size,
    num_nodes=num_nodes,
    dropout=0.4,  # Increased from 0.3 (analyze.py rec: overfitting)
)

# --- 4. Optimizer, Loss, and Scheduler ---
# Huber loss instead of MSE (analyze.py rec: heavy-tailed error distribution)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Higher weight decay
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# Gaussian noise scale for training input augmentation (analyze.py rec: regularization)
INPUT_NOISE_STD = 0.01

# --- 5. Training Loop with Early Stopping + Gradient Clipping ---
epochs = 200
patience = 15
best_val_loss = float("inf")
epochs_no_improve = 0

train_losses = []
val_losses = []

print("\nStarting training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # Add Gaussian noise to training inputs (analyze.py rec: regularization)
        batch_X_noisy = batch_X + torch.randn_like(batch_X) * INPUT_NOISE_STD

        # Reshape LSTM input: (Batch, Seq_Len, Nodes * Features)
        lstm_input = batch_X_noisy.view(batch_X_noisy.size(0), batch_X_noisy.size(1), -1)

        # GNN input: latest timestep features averaged across batch
        gnn_input = batch_X_noisy[:, -1, :, :].mean(dim=0)

        # Forward pass
        predictions = model(lstm_input, gnn_input, edge_idx, edge_wt)

        # Calculate loss and backpropagate
        loss = criterion(predictions, batch_y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            lstm_input = batch_X.view(batch_X.size(0), batch_X.size(1), -1)
            gnn_input = batch_X[:, -1, :, :].mean(dim=0)

            predictions = model(lstm_input, gnn_input, edge_idx, edge_wt)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Step the learning rate scheduler
    scheduler.step(avg_val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}"
        )

    # --- Early Stopping Logic ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

# Load the best model weights for evaluation
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
print(f"\nLoaded best model (val loss: {best_val_loss:.6f})")


# =============================================================================
# Loss Plot
# =============================================================================

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)
plt.title("Training and Validation Loss Over Epochs", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss (MSE)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=150)
print("Loss plot saved as 'loss_plot.png'")


# =============================================================================
# Evaluation & Analysis
# =============================================================================

# --- 1. Generate Predictions on the Test Set ---
model.eval()
all_predictions = []
all_actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        lstm_input = batch_X.view(batch_X.size(0), batch_X.size(1), -1)
        gnn_input = batch_X[:, -1, :, :].mean(dim=0)

        preds = model(lstm_input, gnn_input, edge_idx, edge_wt)
        all_predictions.append(preds.numpy())
        all_actuals.append(batch_y.numpy())

all_predictions = np.concatenate(all_predictions, axis=0)
all_actuals = np.concatenate(all_actuals, axis=0)

# --- 2. Inverse Transformation (Per-Stock Scalers) ---
num_samples = all_predictions.shape[0]

dollar_predictions = np.zeros((num_samples, num_nodes))
dollar_actuals = np.zeros((num_samples, num_nodes))

for i in range(num_nodes):
    # Build a dummy array matching what the per-stock scaler expects (num_features columns)
    dummy_pred = np.zeros((num_samples, num_features))
    dummy_actual = np.zeros((num_samples, num_features))

    # Close is feature index 0
    dummy_pred[:, 0] = all_predictions[:, i]
    dummy_actual[:, 0] = all_actuals[:, i]

    # Inverse transform using this stock's scaler
    inv_pred = stock_scalers[i].inverse_transform(dummy_pred)
    inv_actual = stock_scalers[i].inverse_transform(dummy_actual)

    dollar_predictions[:, i] = inv_pred[:, 0]
    dollar_actuals[:, i] = inv_actual[:, 0]


# --- 3. Comprehensive Performance Metrics ---
print("\n" + "=" * 80)
print("                     MODEL PERFORMANCE ANALYSIS")
print("=" * 80)

# Overall metrics
overall_rmse = np.sqrt(mean_squared_error(dollar_actuals, dollar_predictions))
overall_mae = mean_absolute_error(dollar_actuals, dollar_predictions)
overall_mape = np.mean(np.abs((dollar_actuals - dollar_predictions) / (dollar_actuals + 1e-10))) * 100

print(f"\n{'Overall Metrics':^80}")
print("-" * 80)
print(f"  RMSE:  ${overall_rmse:.2f}")
print(f"  MAE:   ${overall_mae:.2f}")
print(f"  MAPE:  {overall_mape:.2f}%")


# --- 4. Per-Stock Metrics Table ---
print(f"\n{'Per-Stock Performance':^80}")
print("-" * 80)
print(f"  {'Ticker':<8} {'RMSE ($)':>10} {'MAE ($)':>10} {'MAPE (%)':>10} {'Dir Acc (%)':>12} {'Avg Price ($)':>14}")
print("-" * 80)

directional_accuracies = []
per_stock_metrics = []

for idx, ticker in enumerate(target_tickers):
    stock_pred = dollar_predictions[:, idx]
    stock_actual = dollar_actuals[:, idx]

    # Basic metrics
    stock_rmse = np.sqrt(mean_squared_error(stock_actual, stock_pred))
    stock_mae = mean_absolute_error(stock_actual, stock_pred)
    stock_mape = np.mean(np.abs((stock_actual - stock_pred) / (stock_actual + 1e-10))) * 100
    avg_price = np.mean(stock_actual)

    # Directional accuracy: does the model predict the correct direction of price change?
    if len(stock_actual) > 1:
        actual_direction = np.diff(stock_actual) > 0
        pred_direction = np.diff(stock_pred) > 0
        dir_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        dir_accuracy = 0.0

    directional_accuracies.append(dir_accuracy)
    per_stock_metrics.append({
        "ticker": ticker,
        "rmse": stock_rmse,
        "mae": stock_mae,
        "mape": stock_mape,
        "dir_acc": dir_accuracy,
        "avg_price": avg_price,
    })

    print(f"  {ticker:<8} {stock_rmse:>10.2f} {stock_mae:>10.2f} {stock_mape:>10.2f} {dir_accuracy:>12.1f} {avg_price:>14.2f}")

print("-" * 80)
mean_dir_acc = np.mean(directional_accuracies)
print(f"  {'AVERAGE':<8} {'':>10} {'':>10} {'':>10} {mean_dir_acc:>12.1f}")
print()

# --- 5. Trading Signal Analysis ---
print(f"{'Trading Signal Analysis':^80}")
print("-" * 80)

# Sort stocks by directional accuracy (most predictable first)
sorted_metrics = sorted(per_stock_metrics, key=lambda x: x["dir_acc"], reverse=True)

print("  Stocks ranked by predictability (directional accuracy):")
for rank, m in enumerate(sorted_metrics, 1):
    signal = "STRONG" if m["dir_acc"] >= 60 else "MODERATE" if m["dir_acc"] >= 52 else "WEAK"
    print(f"    {rank}. {m['ticker']:<6} -- {m['dir_acc']:.1f}% directional accuracy [{signal}]")

print()

# Error relative to price (which stocks the model handles best)
print("  Stocks ranked by relative accuracy (MAPE -- lower is better):")
sorted_by_mape = sorted(per_stock_metrics, key=lambda x: x["mape"])
for rank, m in enumerate(sorted_by_mape, 1):
    quality = "EXCELLENT" if m["mape"] < 3 else "GOOD" if m["mape"] < 5 else "FAIR" if m["mape"] < 10 else "POOR"
    print(f"    {rank}. {m['ticker']:<6} -- {m['mape']:.2f}% MAPE [{quality}]")

print()


# --- 6. Sample Predictions ---
print(f"{'Sample Predictions (Last 5 Test Days)':^80}")
print("-" * 80)
for idx, ticker in enumerate(target_tickers):
    print(f"\n  {ticker}:")
    for i in range(-5, 0):
        pred_val = dollar_predictions[i, idx]
        actual_val = dollar_actuals[i, idx]
        error = pred_val - actual_val
        pct_error = (error / actual_val) * 100
        direction = "^" if error > 0 else "v"
        print(f"    Predicted: ${pred_val:>9.2f} | Actual: ${actual_val:>9.2f} | "
              f"Error: {direction} ${abs(error):.2f} ({pct_error:+.2f}%)")
print()


# =============================================================================
# Prediction vs Actual Plots
# =============================================================================

fig, axes = plt.subplots(5, 2, figsize=(18, 22))
fig.suptitle("Predicted vs Actual Close Prices (Test Period)", fontsize=16, fontweight="bold")

for idx, (ax, ticker) in enumerate(zip(axes.flat, target_tickers)):
    stock_pred = dollar_predictions[:, idx]
    stock_actual = dollar_actuals[:, idx]

    ax.plot(stock_actual, label="Actual", linewidth=1.5, color="#2196F3")
    ax.plot(stock_pred, label="Predicted", linewidth=1.5, color="#FF5722", alpha=0.8)
    ax.fill_between(
        range(len(stock_actual)),
        stock_actual,
        stock_pred,
        alpha=0.15,
        color="#FF5722",
    )
    ax.set_title(f"{ticker}  (MAPE: {per_stock_metrics[idx]['mape']:.2f}%)", fontsize=12)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("predictions_plot.png", dpi=150)
print("Predictions plot saved as 'predictions_plot.png'")


# =============================================================================
# Correlation of Errors Plot
# =============================================================================

errors = dollar_predictions - dollar_actuals
error_df = pd.DataFrame(errors, columns=target_tickers)
error_corr = error_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(error_corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(target_tickers)))
ax.set_yticks(range(len(target_tickers)))
ax.set_xticklabels(target_tickers, rotation=45, ha="right")
ax.set_yticklabels(target_tickers)

# Annotate cells
for i in range(len(target_tickers)):
    for j in range(len(target_tickers)):
        ax.text(j, i, f"{error_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

ax.set_title("Prediction Error Correlation Between Stocks", fontsize=14)
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig("error_correlation.png", dpi=150)
print("Error correlation plot saved as 'error_correlation.png'")

# =============================================================================
# Export Results for analyze.py
# =============================================================================

np.savez(
    "model_results.npz",
    predictions=dollar_predictions,
    actuals=dollar_actuals,
    train_losses=np.array(train_losses),
    val_losses=np.array(val_losses),
    tickers=np.array(target_tickers),
)
print("Results data saved as 'model_results.npz'")

print("\n" + "=" * 80)
print("  All analysis complete. Outputs saved:")
print("    - loss_plot.png          : Training/validation loss curves")
print("    - predictions_plot.png   : Predicted vs actual prices per stock")
print("    - error_correlation.png  : How prediction errors correlate across stocks")
print("    - best_model.pth         : Best model weights checkpoint")
print("    - model_results.npz      : Results data for analyze.py")
print("=" * 80)

# Automatically run post-training analysis
import subprocess
print("\n>>> Launching analyze.py automatically...\n")
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "analyze.py")], cwd=os.path.dirname(__file__))

