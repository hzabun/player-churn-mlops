from pathlib import Path

from prefect import flow, task

from src.preprocess import preprocess_all_players

RAW_DATA_PATH = Path("data/raw_parquet")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_FILE_NAME = "player_features.parquet"
LABEL_FILE_PATH = Path("data/label/train_labeld.csv")


@task(name="preprocess_data_task")
def preprocess_data_task():
    preprocess_all_players(
        raw_dir=RAW_DATA_PATH,
        output_dir=PROCESSED_DATA_PATH,
        output_filename=PROCESSED_FILE_NAME,
        label_file_path=LABEL_FILE_PATH,
    )


@task(name="materialize_features_task")
def materialize_features_task():
    """Materialize features to Feast online store for real-time serving."""
    from datetime import datetime, timedelta

    from feast import FeatureStore

    fs = FeatureStore(repo_path="feature_store")

    # Materialize last 7 days of features to online store
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"Materializing features from {start_date} to {end_date}")
    fs.materialize(start_date=start_date, end_date=end_date)
    print("✓ Features materialized to online store")


@task(name="train_model_task")
def train_model_task():
    # TODO: Implement model training using Feast features
    pass


@task(name="deploy_model_task")
def deploy_model_task():
    # TODO: Implement model deployment
    pass


@flow(name="ml_pipeline_flow")
def ml_pipeline_flow():
    """Complete ML pipeline: preprocess → materialize → train → deploy."""
    preprocess_data_task()
    materialize_features_task()
    # train_model_task()
    # deploy_model_task()
