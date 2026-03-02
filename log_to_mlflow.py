"""
Run this script from the airflow_pipeline/ directory to train the credit card
default model and log everything to your local MLflow server.

Usage:
    cd airflow_pipeline
    python log_to_mlflow.py --port 5001   # match your MLflow port
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import pandas as pd

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port",     default="5001", help="MLflow server port")
parser.add_argument("--data-dir", default="data", help="Folder containing the CSVs")
args = parser.parse_args()

MLFLOW_URI = f"http://127.0.0.1:{args.port}"
DATA_DIR   = args.data_dir

# ── Point at your local ML pipeline modules ───────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dags"))

from ML_pipeline import (
    dataset,
    data_splitting,
    data_preprocessing,
    feature_engineering,
    upsampling_minorityClass,
    scaling_features,
    model_params,
    train_model  as tm,
    predict_model as pm,
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)

# ── Override data paths ───────────────────────────────────────────────────────
dataset.train_path      = os.path.join(DATA_DIR, "cs-training.csv")
dataset.validation_path = os.path.join(DATA_DIR, "cs-test.csv")

# ── Connect to MLflow ─────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("credit_card_default_prediction")
print(f"✓ Connected to MLflow at {MLFLOW_URI}")

# ═════════════════════════════════════════════════════════════════════════════
with mlflow.start_run(run_name="credit_risk_lgbm") as run:
    print(f"  Run ID: {run.info.run_id}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("→ Loading data...")
    train, val = dataset.read_data()
    mlflow.log_param("train_rows",      len(train))
    mlflow.log_param("validation_rows", len(val))

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("→ Splitting data...")
    df_test, df_train, y_test, y_train, train_df, test_df = \
        data_splitting.training_testing_dataset(train)
    mlflow.log_param("test_ratio", 0.2)
    mlflow.log_param("random_state", 42)

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    print("→ Preprocessing...")
    train_clean, test_clean, val_clean = data_preprocessing.data_preprocessing(
        train_df, test_df, val,
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "MonthlyIncome", "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines",
        "age", "NumberOfDependents", "SeriousDlqin2yrs",
    )
    mlflow.log_metric("missing_after_preprocess",
                      train_clean.isnull().sum().sum())

    # ── 4. Feature engineering ────────────────────────────────────────────────
    print("→ Engineering features...")
    train_fe, test_fe, val_fe = feature_engineering.feature_engineering(
        train_clean, test_clean, val_clean,
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "MonthlyIncome", "NumberOfDependents", "DebtRatio", "age",
    )
    mlflow.log_param("total_features", len(train_fe.columns))

    # ── 5. Upsample + scale ───────────────────────────────────────────────────
    print("→ Upsampling & scaling...")
    USE_SMOTE   = False
    USE_SCALING = False
    mlflow.log_param("use_smote",           USE_SMOTE)
    mlflow.log_param("use_feature_scaling", USE_SCALING)

    train_x, train_y, test_x, test_y, val_x = \
        upsampling_minorityClass.upsampling_class(train_fe, test_fe, val_fe, USE_SMOTE)
    train_x_s, test_x_s, val_x_s = \
        scaling_features.scaling_features(train_x, test_x, val_x, USE_SCALING)

    # ── 6. Model hyperparameters ──────────────────────────────────────────────
    clf = model_params.model_params()
    params = {
        "model":             "LGBMClassifier",
        "colsample_bytree":  0.65,
        "max_depth":         4,
        "min_data_in_leaf":  400,
        "min_split_gain":    0.25,
        "num_leaves":        70,
        "reg_lambda":        5,
        "subsample":         0.65,
        "scale_pos_weight":  10,
        "random_state":      42,
    }
    mlflow.log_params(params)

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print("→ Training LightGBM...")
    model = tm.train_model(clf, train_x_s, train_y, test_x_s, test_y)

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    preds  = model.predict(test_x_s)
    probas = model.predict_proba(test_x_s)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(test_y, preds),   4),
        "roc_auc":   round(roc_auc_score(test_y, probas),   4),
        "precision": round(precision_score(test_y, preds),  4),
        "recall":    round(recall_score(test_y, preds),     4),
        "f1_score":  round(f1_score(test_y, preds),         4),
    }
    mlflow.log_metrics(metrics)
    print(f"\n  📊 Metrics:")
    for k, v in metrics.items():
        print(f"     {k:12s}: {v}")

    # ── 9. Log model ──────────────────────────────────────────────────────────
    print("\n→ Logging model to MLflow...")
    mlflow.sklearn.log_model(
        model,
        artifact_path        = "lgbm_model",
        registered_model_name= "CreditCardDefaultLGBM",
    )

    # ── 10. Predict & save CSV ────────────────────────────────────────────────
    print("→ Predicting validation set...")
    val_preds = pm.predict_model(model, val_x_s.copy())
    os.makedirs("output", exist_ok=True)
    out_path = "output/predictions.csv"
    val_preds.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path, artifact_path="predictions")
    mlflow.log_metric("predicted_default_rate",
                      round(val_preds["predictions"].mean(), 4))

    print(f"\n✅ Done! View your run at {MLFLOW_URI}/#/experiments")
    print(f"   Predictions saved to {out_path}")
