# UNIFIED HACKATHON TRADING MODEL

This repository contains a robust, unified trading model pipeline developed for a hackathon. The model focuses on predicting stock prices for a basket of defense stocks (LMT, RTX, NOC) alongside the S&P 500 ETF (SPY), utilizing an advanced machine learning ensemble and a capital protection "kill-switch."

The project relies on three key predictive models and a strict financial turbulence index to ensure effective risk management before wrapping it up into an interactive React dashboard.

## Overview

The trading engine relies on three distinct "Members" conceptually:

1. **Member 1 (The Classical Modeler): Random Forest**
   - **Role:** Captures stable, non-linear market trends.
   - **Implementation:** An ensemble of 100 decision trees utilizing tabular Open-High-Low-Close-Volume (OHLCV) features and engineering technical indicators (SMA, RSI, MACD).

2. **Member 2 (The Deep Learning Modeler): GRU Network**
   - **Role:** Recognizes volatile, sequential patterns. 
   - **Implementation:** A 2-layer Gated Recurrent Unit (GRU) with a 60-day sliding window, capturing long-term sequential dependencies in our time-series stock data. Less computationally intensive than an LSTM, fitting perfectly for hackathon constraints.

3. **Member 3 (The Meta-Learner): XGBoost Ensembler**
   - **Role:** Dynamically weighs and ensembles the predictions of Member 1 and Member 2 to forecast the final return.
   - **Implementation:** XGBoost interprets current market volatility and determines whether to trust the RF model (during stable trends) or the GRU model (during volatile conditions).

## Financial Turbulence Index (Capital Protection)

In order to survive large market crashes (e.g., COVID-19 in March 2020), this pipeline utilizes a strict Turbulence Index to measure how far today's market behavior deviates from normal historical behavior using the Mahalanobis distance.

If the calculated turbulence index crosses the 90th percentile threshold, the algorithm triggers a **Crash Mode**. The trading algorithm ignores the XGBoost prediction, immediately liquidates all positions into a cash hold, and only resumes trading once turbulence normalizes. 

## Project Structure

```
├── main.py                # Main data pipeline, ML model execution, and simulation
├── Information.md         # Origin thesis and implementation guidelines
├── gru_model.pth          # PyTorch GRU saved parameters
└── webapp/
    ├── package.json       # React frontend dependencies
    ├── src/
    │   └── App.jsx        # Dashboard visualizing data output and simulated charts
    └── public/
        └── results.json   # Exported results from main.py used by the front-end
```

## Setup & Installation

### 1. Python Trading Pipeline

Ensure you have your data-science environment set up:
```bash
python -m venv .venv
source .venv/Scripts/activate # Windows
pip install pandas numpy scikit-learn xgboost torch yfinance matplotlib
```

To run the pipeline (downloads data, trains models, evaluates metrics, and exports results):
```bash
python main.py
```

### 2. React Dashboard

The frontend visualizes the results of the pipeline through Vite + React. 

```bash
cd webapp
npm install
npm run dev
```

Navigate to `http://localhost:5173` to see the live dashboard!

## Evaluation Metrics

The ensemble pipeline evaluates its success over the test period (e.g., 2023-2025) using:
- **Prediction Accuracy:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
- **Quantitative Metrics:** Cumulative Return (%), Max Drawdown (%)

We specifically compare these metrics against static strategies like Buy & Hold (SPY) and simple Moving Average (SMA10/SMA50) crossovers to demonstrate the ensemble's risk-adjusted returns and drawdown protections.
