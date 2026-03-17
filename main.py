"""
Main Orchestrator — Member 1
=============================
Runs the end-to-end pipeline:
  1. Data acquisition & feature engineering
  2. Random Forest training
  3. Prediction generation & export
  4. Evaluation & benchmarking
"""

from data_pipeline import build_pipeline
from random_forest_model import build_rf, train_rf, predict_rf, save_predictions
from evaluation import full_report


def main() -> None:
    ticker = "SPY"
    print("=" * 55)
    print("  MEMBER 1 — Random Forest Baseline Pipeline")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Step 1: Data pipeline
    # ------------------------------------------------------------------
    print("\n[1/4] Fetching data & engineering features …")
    data = build_pipeline(ticker)
    print(f"  Train samples: {data['X_train'].shape[0]}")
    print(f"  Test samples : {data['X_test'].shape[0]}")
    print(f"  Features     : {data['X_train'].shape[1]}")

    # ------------------------------------------------------------------
    # Step 2: Train Random Forest
    # ------------------------------------------------------------------
    print("\n[2/4] Training Random Forest (100 estimators) …")
    model = build_rf(n_estimators=100)
    train_rf(model, data["X_train"], data["y_train"])
    print("  Training complete ✓")

    # ------------------------------------------------------------------
    # Step 3: Generate & save predictions
    # ------------------------------------------------------------------
    print("\n[3/4] Generating predictions …")
    train_preds = predict_rf(model, data["X_train"])
    test_preds = predict_rf(model, data["X_test"])
    save_predictions(
        train_preds=train_preds,
        test_preds=test_preds,
        train_index=data["train_df"].index,
        test_index=data["test_df"].index,
        path="rf_predictions.csv",
    )

    # ------------------------------------------------------------------
    # Step 4: Evaluation & benchmarking
    # ------------------------------------------------------------------
    print("\n[4/4] Running evaluation …")
    full_report(
        y_test=data["y_test"],
        test_preds=test_preds,
        test_df=data["test_df"],
        label="Random Forest",
    )

    print("Pipeline complete ✓")


if __name__ == "__main__":
    main()
