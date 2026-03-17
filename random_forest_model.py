"""
Random Forest Model — Member 1
===============================
Builds a Random Forest regressor (100 estimators) to predict next-day
closing price from tabular OHLCV + technical indicator features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_rf(n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    """Return an untrained Random Forest regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def train_rf(model: RandomForestRegressor,
             X_train: np.ndarray,
             y_train: np.ndarray) -> RandomForestRegressor:
    """Fit the Random Forest on training data."""
    model.fit(X_train, y_train)
    return model


def predict_rf(model: RandomForestRegressor,
               X: np.ndarray) -> np.ndarray:
    """Generate predictions from a trained Random Forest."""
    return model.predict(X)


# ---------------------------------------------------------------------------
# Prediction export (for Member 3's XGBoost meta-learner)
# ---------------------------------------------------------------------------
def save_predictions(train_preds: np.ndarray,
                     test_preds: np.ndarray,
                     train_index: pd.DatetimeIndex,
                     test_index: pd.DatetimeIndex,
                     path: str = "rf_predictions.csv") -> pd.DataFrame:
    """Save train + test predictions to a single CSV for the meta-learner.

    The CSV has columns: Date, RF_Pred, Split ('train' or 'test').
    """
    df_train = pd.DataFrame({
        "Date": train_index,
        "RF_Pred": train_preds,
        "Split": "train",
    })
    df_test = pd.DataFrame({
        "Date": test_index,
        "RF_Pred": test_preds,
        "Split": "test",
    })
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"  Random Forest predictions saved → {path}")
    return df


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_pipeline import build_pipeline

    print("Training Random Forest on SPY …")
    data = build_pipeline("SPY")
    model = build_rf()
    train_rf(model, data["X_train"], data["y_train"])

    train_preds = predict_rf(model, data["X_train"])
    test_preds = predict_rf(model, data["X_test"])

    print(f"  Train predictions: {len(train_preds)}")
    print(f"  Test predictions : {len(test_preds)}")
    print("Random Forest OK ✓")
