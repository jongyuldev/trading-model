#!/usr/bin/env python
"""
Ensemble Trading Model
======================
Combines a Random Forest baseline, a 2-layer GRU deep learning model (PyTorch),
and an XGBoost meta-learner with a turbulence-index kill-switch for capital
protection.

Usage:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TICKERS       = ["SPY", "LMT", "RTX", "NOC"]
START_DATE    = "2015-01-01"
END_DATE      = "2025-12-31"
TARGET_TICKER = "SPY"
WINDOW        = 60          # GRU look-back
TURB_WINDOW   = 252         # turbulence look-back

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# 1.  DATA ACQUISITION
# ──────────────────────────────────────────────
print("=" * 70)
print("ENSEMBLE TRADING MODEL")
print("=" * 70)
print(f"\n[1/7] Downloading OHLCV data for {TICKERS} …")

raw = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=False)

close_df  = raw["Close"].ffill().bfill()
volume_df = raw["Volume"].ffill().bfill()
high_df   = raw["High"].ffill().bfill()
low_df    = raw["Low"].ffill().bfill()
open_df   = raw["Open"].ffill().bfill()

# ──────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("[2/7] Engineering features (SMA50, RSI14, MACD, Vol20) …")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.rolling(period, min_periods=period).mean()
    avg_l = loss.rolling(period, min_periods=period).mean()
    rs = avg_g / avg_l
    return 100 - 100 / (1 + rs)


def compute_macd(series, fast=12, slow=26, sig=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line  = ema_f - ema_s
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal


features = pd.DataFrame(index=close_df.index)

for t in TICKERS:
    c = close_df[t]
    features[f"{t}_Close"]     = c
    features[f"{t}_Open"]      = open_df[t]
    features[f"{t}_High"]      = high_df[t]
    features[f"{t}_Low"]       = low_df[t]
    features[f"{t}_Volume"]    = volume_df[t]
    features[f"{t}_SMA50"]     = c.rolling(50).mean()
    features[f"{t}_RSI14"]     = compute_rsi(c)
    ml, ms, mh = compute_macd(c)
    features[f"{t}_MACD"]      = ml
    features[f"{t}_MACD_Sig"]  = ms
    features[f"{t}_MACD_Hist"] = mh
    features[f"{t}_Vol20"]     = c.pct_change().rolling(20).std()

# Daily returns for turbulence
returns_df = close_df.pct_change()

# Target = next-day SPY close
features["Target"] = close_df[TARGET_TICKER].shift(-1)
features.dropna(inplace=True)
returns_df = returns_df.loc[features.index]

# ──────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT  (chronological)
# ──────────────────────────────────────────────
print("[3/7] Splitting 80 / 20 (chronological, no shuffle) …")

split = int(len(features) * 0.8)
train_df = features.iloc[:split].copy()
test_df  = features.iloc[split:].copy()

y_train = train_df.pop("Target").values
y_test  = test_df.pop("Target").values
feat_cols = train_df.columns.tolist()

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(train_df)
X_test_sc  = scaler.transform(test_df)

X_train_tab = pd.DataFrame(X_train_sc, columns=feat_cols, index=train_df.index)
X_test_tab  = pd.DataFrame(X_test_sc,  columns=feat_cols, index=test_df.index)

# ──────────────────────────────────────────────
# 4.  MEMBER 1 — RANDOM FOREST  (100 trees)
# ──────────────────────────────────────────────
print("[4/7] Training Random Forest …")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_tab, y_train)

rf_train_preds = rf.predict(X_train_tab)
rf_test_preds  = rf.predict(X_test_tab)

# ──────────────────────────────────────────────
# 5.  MEMBER 2 — GRU (PyTorch)
# ──────────────────────────────────────────────
print("[5/7] Training GRU model (60-day window, PyTorch) …")


def build_sequences(X, y, window=60):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class GRUModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.gru1 = nn.GRU(n_features, 100, batch_first=True)
        self.drop1 = nn.Dropout(0.20)
        self.gru2 = nn.GRU(100, 50, batch_first=True)
        self.drop2 = nn.Dropout(0.20)
        self.fc1  = nn.Linear(50, 25)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.drop1(out)
        out, _ = self.gru2(out)
        out = self.drop2(out[:, -1, :])   # last timestep
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


X_tr_seq, y_tr_seq = build_sequences(X_train_sc, y_train, WINDOW)

# Validation split (last 10 % of training sequences)
val_n = max(1, int(len(X_tr_seq) * 0.1))
X_val_seq, y_val_seq = X_tr_seq[-val_n:], y_tr_seq[-val_n:]
X_tr_seq2, y_tr_seq2 = X_tr_seq[:-val_n], y_tr_seq[:-val_n]

train_ds = TensorDataset(torch.from_numpy(X_tr_seq2), torch.from_numpy(y_tr_seq2))
val_ds   = TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq))
train_dl = DataLoader(train_ds, batch_size=32, shuffle=False)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False)

n_feat = X_tr_seq2.shape[2]
gru = GRUModel(n_feat).to(DEVICE)
opt = torch.optim.Adam(gru.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

best_val, patience, wait = float("inf"), 10, 0
best_state = None

for epoch in range(100):
    gru.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss_fn(gru(xb), yb).backward()
        opt.step()

    gru.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += loss_fn(gru(xb), yb).item() * len(yb)
    val_loss /= len(val_ds)

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in gru.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break

if best_state:
    gru.load_state_dict(best_state)
gru.eval()


def gru_predict(X_seq):
    t = torch.from_numpy(X_seq).to(DEVICE)
    with torch.no_grad():
        return gru(t).cpu().numpy()


# GRU train preds (full training sequences)
gru_train_full, _ = build_sequences(X_train_sc, y_train, WINDOW)
gru_train_preds = gru_predict(gru_train_full)

# GRU test preds (need WINDOW lead-in from training tail)
X_all_sc = np.concatenate([X_train_sc, X_test_sc])
y_all    = np.concatenate([y_train, y_test])
X_test_seq, y_test_seq = build_sequences(
    X_all_sc[split - WINDOW:], y_all[split - WINDOW:], WINDOW
)
gru_test_preds = gru_predict(X_test_seq)

# Align arrays — GRU lost first WINDOW rows
rf_train_al = rf_train_preds[WINDOW:]
y_train_al  = y_train[WINDOW:]
idx_train_al = train_df.index[WINDOW:]

rf_test_al  = rf_test_preds
y_test_al   = y_test_seq
idx_test_al = test_df.index[:len(y_test_seq)]

# ──────────────────────────────────────────────
# 6.  MEMBER 3 — XGBOOST META-LEARNER
# ──────────────────────────────────────────────
print("[6/7] Training XGBoost meta-learner …")


def meta_features(rf_p, gru_p, idx, tab):
    m = pd.DataFrame(index=idx)
    m["RF_Pred"]  = rf_p
    m["GRU_Pred"] = gru_p
    for t in TICKERS:
        for s in ["Vol20", "RSI14", "MACD_Hist"]:
            col = f"{t}_{s}"
            m[col] = tab.loc[idx, col].values
    return m


meta_tr = meta_features(rf_train_al, gru_train_preds, idx_train_al, X_train_tab)
meta_te = meta_features(rf_test_al,  gru_test_preds,  idx_test_al,  X_test_tab)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                    random_state=42, verbosity=0)
xgb.fit(meta_tr, y_train_al)
xgb_preds = xgb.predict(meta_te)

# ──────────────────────────────────────────────
# 7.  TURBULENCE INDEX  &  TRADING SIMULATION
# ──────────────────────────────────────────────
print("[7/7] Running trading simulation …\n")

turb_ret = returns_df.loc[features.index]
turbulence = pd.Series(np.nan, index=turb_ret.index)

for i in range(TURB_WINDOW, len(turb_ret)):
    hist = turb_ret.iloc[i - TURB_WINDOW:i]
    mu   = hist.mean().values.reshape(1, -1)
    cov  = hist.cov().values
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    yt   = turb_ret.iloc[i].values.reshape(1, -1)
    diff = yt - mu
    turbulence.iloc[i] = (diff @ cov_inv @ diff.T).item()

turbulence.dropna(inplace=True)

# Threshold = 90th-pctl of training-set turbulence
train_turb = turbulence.loc[turbulence.index.isin(train_df.index)]
TURB_THRESH = float(train_turb.quantile(0.90))

# --- Trading signals ---
test_dates = idx_test_al
prev_close = close_df.loc[test_dates, TARGET_TICKER].values
pred_ret   = (xgb_preds - prev_close) / prev_close
actual_ret = returns_df.loc[test_dates, TARGET_TICKER].shift(-1).fillna(0).values

positions = np.where(pred_ret > 0, 1.0, 0.0)

# Turbulence override
turb_test = turbulence.reindex(test_dates).fillna(0)
override  = (turb_test.values > TURB_THRESH).astype(float)
positions *= (1 - override)

strat_ret = positions * actual_ret
bh_ret    = actual_ret

# SMA crossover benchmark
spy_c  = close_df.loc[test_dates, TARGET_TICKER]
sma10  = spy_c.rolling(10, min_periods=1).mean()
sma50  = spy_c.rolling(50, min_periods=1).mean()
sma_sig = np.where(sma10 > sma50, 1.0, 0.0)
sma_ret = sma_sig * actual_ret

cum_strat = (1 + strat_ret).cumprod()
cum_bh    = (1 + bh_ret).cumprod()
cum_sma   = (1 + sma_ret).cumprod()


def max_dd(c):
    peak = np.maximum.accumulate(c)
    return float(((c - peak) / peak).min())


# ──────────────────────────────────────────────
# 8.  METRICS  &  OUTPUT
# ──────────────────────────────────────────────
mse  = mean_squared_error(y_test_al, xgb_preds)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_al, xgb_preds)

print("=" * 70)
print("PREDICTION ACCURACY  (XGBoost Meta-Learner)")
print("-" * 70)
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print()
print("TRADING SIMULATION RESULTS")
print("-" * 70)
print(f"  Ensemble Strategy  —  Return: {(cum_strat[-1]-1)*100:+.2f}%  |  "
      f"Max Drawdown: {max_dd(cum_strat)*100:.2f}%")
print(f"  Buy & Hold SPY     —  Return: {(cum_bh[-1]-1)*100:+.2f}%  |  "
      f"Max Drawdown: {max_dd(cum_bh)*100:.2f}%")
print(f"  SMA(10/50) Cross.  —  Return: {(cum_sma[-1]-1)*100:+.2f}%  |  "
      f"Max Drawdown: {max_dd(cum_sma)*100:.2f}%")
print(f"\n  Turbulence Threshold : {TURB_THRESH:.2f}")
print(f"  Days in Cash (turb)  : {int(override.sum())} / {len(override)}")
print("=" * 70)

# --- Charts ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

axes[0].plot(test_dates, cum_strat, label="Ensemble Strategy", lw=2)
axes[0].plot(test_dates, cum_bh,    label="Buy & Hold SPY", lw=1.5, alpha=.7)
axes[0].plot(test_dates, cum_sma,   label="SMA(10/50) Crossover", lw=1.5, alpha=.7)
axes[0].set_title("Cumulative Returns", fontsize=14, fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(test_dates, y_test_al, label="Actual", lw=1.5)
axes[1].plot(test_dates, xgb_preds, label="XGBoost Predicted", lw=1.5, alpha=.8)
axes[1].set_title(f"{TARGET_TICKER} — Predicted vs Actual Close",
                  fontsize=14, fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=.3)

tp = turbulence.reindex(test_dates).fillna(0)
axes[2].plot(test_dates, tp.values, label="Turbulence Index", lw=1.2)
axes[2].axhline(TURB_THRESH, color="red", ls="--",
                label=f"Threshold ({TURB_THRESH:.1f})")
axes[2].set_title("Financial Turbulence Index", fontsize=14, fontweight="bold")
axes[2].legend(); axes[2].grid(alpha=.3)

plt.savefig("trading_model_results.png", dpi=150)
print("\nChart saved → trading_model_results.png")
