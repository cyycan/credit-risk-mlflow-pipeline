# Credit Card Default Prediction — Airflow + MLflow Pipeline

End-to-end MLOps pipeline that orchestrates the credit card default prediction model
using **Apache Airflow** for workflow scheduling and **MLflow** for experiment tracking.

---

## Project Structure

```
airflow_pipeline/
├── dags/
│   ├── credit_card_pipeline_dag.py     ← Main DAG definition
│   └── ML_pipeline/                    ← Modular ML modules (unchanged from src)
│       ├── __init__.py
│       ├── dataset.py
│       ├── data_splitting.py
│       ├── data_preprocessing.py
│       ├── feature_engineering.py
│       ├── upsampling_minorityClass.py
│       ├── scaling_features.py
│       ├── model_params.py
│       ├── train_model.py
│       └── predict_model.py
├── data/                               ← Mount your CSVs here
│   ├── cs-training.csv
│   └── cs-test.csv
├── output/                             ← Intermediate parquet & final predictions
├── logs/                               ← Airflow task logs
├── plugins/                            ← Custom Airflow plugins (optional)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## DAG Pipeline Graph

```
start
  └─► start_mlflow_run     (Creates MLflow experiment + run)
        └─► load_data           (Reads CSVs, logs row counts)
              └─► split_data        (80/20 stratified split)
                    └─► preprocess_data   (Outlier removal + imputation)
                          └─► engineer_features  (8 new derived features)
                                └─► upsample_and_scale  (SMOTE optional + scaling)
                                      └─► train_model     (LightGBM, logs metrics)
                                            └─► predict_validation  (Scores val set)
                                                  └─► end_mlflow_run
                                                        └─► end
```

---

## Quickstart

### 1. Add your data files

```bash
cp /path/to/cs-training.csv data/
cp /path/to/cs-test.csv     data/
```

### 2. Set environment variable

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
```

### 3. Initialise and start the stack

```bash
docker-compose up airflow-init   # one-time setup
docker-compose up -d             # start all services
```

### 4. Access the UIs

| Service      | URL                        | Credentials     |
|--------------|----------------------------|-----------------|
| Airflow UI   | http://localhost:8080      | admin / admin   |
| MLflow UI    | http://localhost:5000      | —               |

### 5. Trigger the DAG

In the Airflow UI, enable and trigger the **`credit_card_default_pipeline`** DAG,
or via CLI:

```bash
docker-compose exec airflow-webserver \
  airflow dags trigger credit_card_default_pipeline
```

---

## MLflow Tracked Artefacts

Each DAG run logs the following to MLflow:

| Stage             | Logged Items                                                  |
|-------------------|---------------------------------------------------------------|
| load_data         | train/val row counts, column names                           |
| split_data        | split sizes, test ratio                                      |
| preprocess_data   | post-cleaning row counts, remaining missing values           |
| engineer_features | new feature names, total feature count                       |
| upsample_and_scale| SMOTE flag, scaling flag, final training shape               |
| train_model       | hyperparameters, accuracy, ROC-AUC, precision, recall, F1   |
| predict_validation| prediction count, predicted default rate, predictions CSV    |

The trained model is registered in the MLflow **Model Registry** as
`CreditCardDefaultLGBM`.

---

## Configuration

| Environment Variable   | Default                  | Description                   |
|------------------------|--------------------------|-------------------------------|
| `MLFLOW_TRACKING_URI`  | `http://mlflow:5000`     | MLflow server URI             |
| `DATA_DIR`             | `/opt/airflow/data`      | Directory with input CSVs     |
| `OUTPUT_DIR`           | `/opt/airflow/output`    | Directory for artifacts       |

To enable **SMOTE upsampling** or **feature scaling**, set `USE_SMOTE = True` /
`USE_SCALING = True` inside the `upsample_and_scale` task function.

---

## Outputs

After a successful run, the `output/` directory will contain:

- `raw_train.parquet`, `raw_val.parquet`  — loaded data
- `train_split.parquet`, `test_split.parquet` — train/test splits
- `train_clean.parquet`, `test_clean.parquet`, `val_clean.parquet` — preprocessed data
- `train_fe.parquet`, `test_fe.parquet`, `val_fe.parquet` — feature-engineered data
- `train_x_scaled.parquet`, `test_x_scaled.parquet`, `val_x_scaled.parquet` — final features
- `model.pkl` — serialised LightGBM model
- `predictions.csv` — validation set predictions with probability scores
