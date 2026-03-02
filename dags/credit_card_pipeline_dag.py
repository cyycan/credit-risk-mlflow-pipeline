"""
Credit Card Default Prediction - Airflow + MLflow Pipeline DAG
Mirrors the modular ML pipeline structure from engine.py, broken into
individual Airflow tasks with MLflow experiment tracking.
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME     = "credit_card_default_prediction"
DATA_DIR            = os.getenv("DATA_DIR", "/opt/airflow/data")
OUTPUT_DIR          = os.getenv("OUTPUT_DIR", "/opt/airflow/output")

# Ensure output dir exists at import time
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# DAG Default Arguments
# ──────────────────────────────────────────────
default_args = {
    "owner":            "airflow",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

# ──────────────────────────────────────────────
# Helper: shared XCom keys
# ──────────────────────────────────────────────
XCOM_RUN_ID = "mlflow_run_id"


def _get_run_id(context):
    ti = context["ti"]
    return ti.xcom_pull(key=XCOM_RUN_ID, task_ids="start_mlflow_run")


# ══════════════════════════════════════════════
# Task functions
# ══════════════════════════════════════════════

def start_mlflow_run(**context):
    """Create / resume an MLflow experiment and push run_id to XCom."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run = mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_id = run.info.run_id
    logger.info(f"MLflow run started: {run_id}")

    mlflow.log_param("dag_run_id", context["run_id"])
    mlflow.end_run()

    context["ti"].xcom_push(key=XCOM_RUN_ID, value=run_id)
    return run_id


def load_data(**context):
    """Task 1: Load training & validation CSVs, push shape metadata."""
    sys.path.insert(0, os.path.dirname(__file__))
    from ML_pipeline import dataset

    # Temporarily override paths to configurable DATA_DIR
    dataset.train_path      = os.path.join(DATA_DIR, "cs-training.csv")
    dataset.validation_path = os.path.join(DATA_DIR, "cs-test.csv")

    train, val = dataset.read_data()

    run_id = _get_run_id(context)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("train_rows",      len(train))
        mlflow.log_param("validation_rows", len(val))
        mlflow.log_param("train_columns",   list(train.columns))

    # Persist to parquet for downstream tasks (avoids XCom size limits)
    train_path = os.path.join(OUTPUT_DIR, "raw_train.parquet")
    val_path   = os.path.join(OUTPUT_DIR, "raw_val.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)

    logger.info(f"Loaded train {train.shape}, val {val.shape}")
    context["ti"].xcom_push(key="train_path", value=train_path)
    context["ti"].xcom_push(key="val_path",   value=val_path)


def split_data(**context):
    """Task 2: Split training data into train / test splits."""
    sys.path.insert(0, os.path.dirname(__file__))
    from ML_pipeline import data_splitting

    ti         = context["ti"]
    train_path = ti.xcom_pull(key="train_path", task_ids="load_data")
    train      = pd.read_parquet(train_path)

    df_test, df_train, y_test, y_train, train_df, test_df = \
        data_splitting.training_testing_dataset(train)

    run_id = _get_run_id(context)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("train_split_size", len(train_df))
        mlflow.log_param("test_split_size",  len(test_df))
        mlflow.log_param("test_ratio",       0.2)

    # Persist splits
    paths = {
        "train_split": os.path.join(OUTPUT_DIR, "train_split.parquet"),
        "test_split":  os.path.join(OUTPUT_DIR, "test_split.parquet"),
    }
    train_df.to_parquet(paths["train_split"])
    test_df.to_parquet(paths["test_split"])

    for k, v in paths.items():
        ti.xcom_push(key=k, value=v)

    logger.info(f"Split: train {train_df.shape}, test {test_df.shape}")


def preprocess_data(**context):
    """Task 3: Clean outliers and impute missing values."""
    sys.path.insert(0, os.path.dirname(__file__))
    from ML_pipeline import data_preprocessing

    ti         = context["ti"]
    train_path = ti.xcom_pull(key="train_split", task_ids="split_data")
    test_path  = ti.xcom_pull(key="test_split",  task_ids="split_data")
    val_path   = ti.xcom_pull(key="val_path",    task_ids="load_data")

    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)
    val   = pd.read_parquet(val_path)

    train_clean, test_clean, val_clean = data_preprocessing.data_preprocessing(
        train, test, val,
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "MonthlyIncome",
        "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfDependents",
        "SeriousDlqin2yrs",
    )

    run_id = _get_run_id(context)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("train_rows_after_preprocessing", len(train_clean))
        mlflow.log_param("test_rows_after_preprocessing",  len(test_clean))
        missing_train = train_clean.isnull().sum().sum()
        mlflow.log_metric("remaining_missing_train", missing_train)

    paths = {
        "train_clean": os.path.join(OUTPUT_DIR, "train_clean.parquet"),
        "test_clean":  os.path.join(OUTPUT_DIR, "test_clean.parquet"),
        "val_clean":   os.path.join(OUTPUT_DIR, "val_clean.parquet"),
    }
    train_clean.to_parquet(paths["train_clean"])
    test_clean.to_parquet(paths["test_clean"])
    val_clean.to_parquet(paths["val_clean"])

    for k, v in paths.items():
        ti.xcom_push(key=k, value=v)

    logger.info(f"Preprocessing done: train {train_clean.shape}")


def engineer_features(**context):
    """Task 4: Create new derived features."""
    sys.path.insert(0, os.path.dirname(__file__))
    from ML_pipeline import feature_engineering

    ti          = context["ti"]
    train_clean = pd.read_parquet(ti.xcom_pull(key="train_clean", task_ids="preprocess_data"))
    test_clean  = pd.read_parquet(ti.xcom_pull(key="test_clean",  task_ids="preprocess_data"))
    val_clean   = pd.read_parquet(ti.xcom_pull(key="val_clean",   task_ids="preprocess_data"))

    train_df, test_df, val_df = feature_engineering.feature_engineering(
        train_clean, test_clean, val_clean,
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "MonthlyIncome",
        "NumberOfDependents",
        "DebtRatio",
        "age",
    )

    new_features = ["CombinedPastDue", "CombinedCreditLoans",
                    "MonthlyIncomePerPerson", "MonthlyDebt",
                    "isRetired", "hasRevolvingLines",
                    "hasMultipleRealEstates", "IsAlone"]

    run_id = _get_run_id(context)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("engineered_features", new_features)
        mlflow.log_param("total_features",      len(train_df.columns))

    paths = {
        "train_fe": os.path.join(OUTPUT_DIR, "train_fe.parquet"),
        "test_fe":  os.path.join(OUTPUT_DIR, "test_fe.parquet"),
        "val_fe":   os.path.join(OUTPUT_DIR, "val_fe.parquet"),
    }
    train_df.to_parquet(paths["train_fe"])
    test_df.to_parquet(paths["test_fe"])
    val_df.to_parquet(paths["val_fe"])

    for k, v in paths.items():
        context["ti"].xcom_push(key=k, value=v)

    logger.info(f"Feature engineering done: {len(train_df.columns)} columns")


def upsample_and_scale(**context):
    """Task 5: Optional SMOTE upsampling + feature scaling."""
    sys.path.insert(0, os.path.dirname(__file__))
    from ML_pipeline import upsampling_minorityClass, scaling_features

    ti       = context["ti"]
    train_fe = pd.read_parquet(ti.xcom_pull(key="train_fe", task_ids="engineer_features"))
    test_fe  = pd.read_parquet(ti.xcom_pull(key="test_fe",  task_ids="engineer_features"))
    val_fe   = pd.read_parquet(ti.xcom_pull(key="val_fe",   task_ids="engineer_features"))

    # Upsampling (set True to enable SMOTE)
    USE_SMOTE   = False
    USE_SCALING = False

    train_x, train_y, test_x, test_y, val_x = \
        upsampling_minorityClass.upsampling_class(train_fe, test_fe, val_fe, USE_SMOTE)

    train_x_scaled, test_x_scaled, val_x_scaled = \
        scaling_features.scaling_features(train_x, test_x, val_x, USE_SCALING)

    run_id = _get_run_id(context)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("use_smote",          USE_SMOTE)
        mlflow.log_param("use_feature_scaling", USE_SCALING)
        mlflow.log_param("train_x_shape",      str(train_x_scaled.shape))
        if USE_SMOTE:
            mlflow.log_metric("smote_train_samples", len(train_x_scaled))

    # Save arrays via parquet
    train_x_scaled["__target__"] = train_y.values
    paths = {
        "train_x_scaled": os.path.join(OUTPUT_DIR, "train_x_scaled.parquet"),
        "test_x_scaled":  os.path.join(OUTPUT_DIR, "test_x_scaled.parquet"),
        "val_x_scaled":   os.path.join(OUTPUT_DIR, "val_x_scaled.parquet"),
        "test_y":         os.path.join(OUTPUT_DIR, "test_y.parquet"),
    }
    train_x_scaled.to_parquet(paths["train_x_scaled"])
    test_x_scaled.to_parquet(paths["test_x_scaled"])
    val_x_scaled.to_parquet(paths["val_x_scaled"])
    pd.DataFrame({"target": test_y}).to_parquet(paths["test_y"])

    for k, v in paths.items():
        ti.xcom_push(key=k, value=v)

    logger.info(f"Upsample & scale done: train_x {train_x_scaled.shape}")


def train_model(**context):
    """Task 6: Fit the LightGBM model, log to MLflow."""
    import pickle
    sys.path.insert(0, os.path.dirname(__file__))
    from ML_pipeline import model_params, train_model as tm
    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                  precision_score, recall_score, f1_score)

    ti = context["ti"]
    train_df  = pd.read_parquet(ti.xcom_pull(key="train_x_scaled", task_ids="upsample_and_scale"))
    test_x    = pd.read_parquet(ti.xcom_pull(key="test_x_scaled",  task_ids="upsample_and_scale"))
    test_y_df = pd.read_parquet(ti.xcom_pull(key="test_y",         task_ids="upsample_and_scale"))

    train_y  = train_df.pop("__target__")
    train_x  = train_df
    test_y   = test_y_df["target"]

    classifier = model_params.model_params()
    model      = tm.train_model(classifier, train_x, train_y, test_x, test_y)

    # Evaluate on held-out test split
    preds  = model.predict(test_x)
    probas = model.predict_proba(test_x)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(test_y, preds),          4),
        "roc_auc":   round(roc_auc_score(test_y, probas),           4),
        "precision": round(precision_score(test_y, preds),          4),
        "recall":    round(recall_score(test_y, preds),             4),
        "f1_score":  round(f1_score(test_y, preds),                 4),
    }
    logger.info(f"Model metrics: {metrics}")

    run_id = _get_run_id(context)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        # Hyperparameters
        mlflow.log_param("model_type",         "LGBMClassifier")
        mlflow.log_param("colsample_bytree",   0.65)
        mlflow.log_param("max_depth",          4)
        mlflow.log_param("min_data_in_leaf",   400)
        mlflow.log_param("num_leaves",         70)
        mlflow.log_param("reg_lambda",         5)
        mlflow.log_param("scale_pos_weight",   10)
        mlflow.log_param("subsample",          0.65)
        # Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        # Log model artifact
        mlflow.sklearn.log_model(model, "lgbm_model",
                                  registered_model_name="CreditCardDefaultLGBM")

    # Persist model locally too
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="metrics",    value=metrics)


def predict_validation(**context):
    """Task 7: Score the validation set and save predictions CSV."""
    import pickle
    sys.path.insert(0, os.path.dirname(__file__))
    from ML_pipeline import predict_model as pm

    ti          = context["ti"]
    model_path  = ti.xcom_pull(key="model_path", task_ids="train_model")
    val_x       = pd.read_parquet(ti.xcom_pull(key="val_x_scaled", task_ids="upsample_and_scale"))

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    val_predictions = pm.predict_model(model, val_x.copy())

    output_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    val_predictions.to_csv(output_path, index=False)

    run_id = _get_run_id(context)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(output_path, artifact_path="predictions")
        mlflow.log_metric("val_predictions_count", len(val_predictions))
        positive_rate = round(val_predictions["predictions"].mean(), 4)
        mlflow.log_metric("predicted_default_rate", positive_rate)

    logger.info(f"Predictions saved to {output_path} — default rate: {positive_rate:.2%}")


def end_mlflow_run(**context):
    """Task 8: Finalise the MLflow run."""
    run_id = _get_run_id(context)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    client.set_tag(run_id, "status", "completed")
    client.set_tag(run_id, "dag_id", context["dag"].dag_id)
    logger.info(f"MLflow run {run_id} finalised.")


# ══════════════════════════════════════════════
# DAG Definition
# ══════════════════════════════════════════════
with DAG(
    dag_id            = "credit_card_default_pipeline",
    description       = "End-to-end credit card default prediction with MLflow tracking",
    default_args      = default_args,
    schedule_interval = "@weekly",
    start_date        = days_ago(1),
    catchup           = False,
    tags              = ["mlops", "credit", "lgbm", "mlflow"],
) as dag:

    # ── Milestones ─────────────────────────────
    start = DummyOperator(task_id="start")
    end   = DummyOperator(task_id="end")

    # ── Tasks ──────────────────────────────────
    t_start_run = PythonOperator(
        task_id         = "start_mlflow_run",
        python_callable = start_mlflow_run,
        provide_context = True,
    )

    t_load = PythonOperator(
        task_id         = "load_data",
        python_callable = load_data,
        provide_context = True,
    )

    t_split = PythonOperator(
        task_id         = "split_data",
        python_callable = split_data,
        provide_context = True,
    )

    t_preprocess = PythonOperator(
        task_id         = "preprocess_data",
        python_callable = preprocess_data,
        provide_context = True,
    )

    t_features = PythonOperator(
        task_id         = "engineer_features",
        python_callable = engineer_features,
        provide_context = True,
    )

    t_upsample = PythonOperator(
        task_id         = "upsample_and_scale",
        python_callable = upsample_and_scale,
        provide_context = True,
    )

    t_train = PythonOperator(
        task_id         = "train_model",
        python_callable = train_model,
        provide_context = True,
    )

    t_predict = PythonOperator(
        task_id         = "predict_validation",
        python_callable = predict_validation,
        provide_context = True,
    )

    t_end_run = PythonOperator(
        task_id         = "end_mlflow_run",
        python_callable = end_mlflow_run,
        provide_context = True,
    )

    # ── Pipeline Graph ─────────────────────────
    (
        start
        >> t_start_run
        >> t_load
        >> t_split
        >> t_preprocess
        >> t_features
        >> t_upsample
        >> t_train
        >> t_predict
        >> t_end_run
        >> end
    )
