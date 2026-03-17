## Inspiration
In the fast-paced world of algorithmic trading, relying on a single predictive model can be incredibly risky. We were inspired by the mechanics of modern hedge funds, which use diverse teams of analysts and systems to evaluate different aspects of the market. Our goal was to create "Trading Ensemble"—a robust, unified trading bot that balances classical machine learning, deep learning, and a dedicated risk-management layer to avoid massive drawdowns during volatile market crashes.

## What it does
Trading Ensemble is an intelligent algorithmic trading pipeline that predicts stock prices for a basket of defense stocks (LMT, RTX, NOC) and the S&P 500 ETF (SPY). It features an interactive React dashboard for visualization and relies on a three-pronged AI approach:
1. **The Classical Modeler:** A Random Forest algorithm capturing stable, non-linear market trends from technical indicators (SMA, RSI, MACD).
2. **The Deep Learning Modeler:** A Gated Recurrent Unit (GRU) neural network utilizing a 60-day sliding window to recognize volatile, long-term sequential patterns.
3. **The Meta-Learner:** An XGBoost Ensembler that dynamically weighs the predictions of the Random Forest and GRU models based on current market volatility.

Crucially, it includes an automated **"Crash Mode"** powered by a Financial Turbulence Index (using Mahalanobis distance). When market behavior deviates too far from historical norms (crossing the 90th percentile threshold), the algorithm ignores all predictions and immediately liquidates positions into cash to protect capital.

## How we built it
We built the backend data pipeline and ML execution engine in Python.
- We used `yfinance` to fetch historical Open-High-Low-Close-Volume (OHLCV) ticker data.
- We engineered technical indicators and trained the Random Forest model using `scikit-learn`.
- We designed and trained the deep learning GRU model in `PyTorch`—choosing GRU over LSTM to fit within hackathon time and computational constraints.
- We trained the XGBoost meta-learner to interpret market volatility and combine the outputs.
- We developed the front-end interactive dashboard using `Vite` and `React`, which visualizes the exported JSON simulation data from the Python backend.

## Challenges we ran into
- **Computational Complexity:** Tuning the PyTorch GRU model and handling sliding-window time-series data required careful optimization to train efficiently within the hackathon timeframe.
- **Ensembling Diverse Models:** Effectively combining tabular data predictions (Random Forest) with time-series sequential outputs (GRU) without causing data leakage.
- **Financial Turbulence Calibration:** Designing the Mahalanobis distance metric to accurately trigger the capital protection "kill-switch" without resulting in too many false positive triggers.
- **Overfitting:** Preventing the models from simply memorizing historical financial data.

## Accomplishments that we're proud of
- Successfully combining three distinct machine learning architectures into a single, cohesive, and effective trading pipeline.
- Implementing the capital protection "kill-switch" that effectively minimizes max drawdowns during simulated market crashes (like the March 2020 COVID-19 crash).
- Building a complete full-stack project—from data ingestion, feature engineering, and model training, all the way to a clean, interactive React visualization dashboard.

## What we learned
- How to implement and train Recurrent Neural Networks (GRUs) in PyTorch specifically for financial time-series forecasting.
- The critical importance of risk management and capital preservation in algorithmic trading, rather than just purely optimizing for maximum predictive return.
- How to calculate and apply the Mahalanobis distance for multivariate financial anomaly detection.
- How to efficiently integrate Python-based model pipelines with modern JavaScript frameworks.

## What's next for Trading Ensemble
- **Live Paper Trading:** Integrating with real-time broker APIs such as Alpaca or Interactive Brokers to test the model's predictive power live.
- **Alternative Data Sources:** Incorporating Natural Language Processing (NLP) to analyze real-time financial news sentiment as an additional prediction feature.
- **Diverse Asset Classes:** Expanding the model to support and balance a wider variety of asset classes like cryptocurrency and commodities.
- **Hyperparameter Optimization:** Using Bayesian optimization tools like Optuna to automatically tune the models for higher accuracy.
