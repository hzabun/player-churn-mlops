from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore
from prefect import flow, task

from src.preprocess import preprocess_all_players

RAW_DATA_PATH = Path("data/raw_parquet")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_FILE_NAME = "player_features.parquet"
LABEL_FILE_PATH = Path("data/label/train_labeld.csv")


@task(name="preprocess_data_task")
def preprocess_data_task(n_workers: int = 4):
    """Preprocess raw player data into features using Dask.

    Args:
        n_workers: Number of Dask workers for parallel processing (default: 4)
    """
    preprocess_all_players(
        raw_dir=RAW_DATA_PATH,
        output_dir=PROCESSED_DATA_PATH,
        output_filename=PROCESSED_FILE_NAME,
        label_file_path=LABEL_FILE_PATH,
        n_workers=n_workers,
    )


@task(name="materialize_features_task")
def materialize_features_task():
    """Materialize features to Feast online store for real-time serving.

    Automatically determines the date range from the processed data file.
    """

    fs = FeatureStore(repo_path="feature_store")

    # Read the processed data to determine date range
    data_path = PROCESSED_DATA_PATH / PROCESSED_FILE_NAME
    df = pd.read_parquet(data_path, columns=["last_session_timestamp"])

    # Get min and max timestamps from the data
    min_date = df["last_session_timestamp"].min()
    max_date = df["last_session_timestamp"].max()

    # Add a small buffer to ensure all data is included
    start_date = min_date - timedelta(days=1)
    end_date = max_date + timedelta(days=1)

    print(f"Materializing features from {start_date} to {end_date}")
    print(f"  Data range: {min_date} to {max_date}")
    print(f"  Total records: {len(df):,}")

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
    # preprocess_data_task()
    materialize_features_task()
    # train_model_task()
    # deploy_model_task()


if __name__ == "__main__":
    ml_pipeline_flow()
