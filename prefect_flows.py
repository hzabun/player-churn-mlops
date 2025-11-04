import logging
from datetime import timedelta

import pandas as pd
from feast import FeatureStore
from prefect import flow, task

from src.preprocess import preprocess_all_players
from src.train import train_model_pipeline

logger = logging.getLogger(__name__)

RAW_DATA_PATH = "s3://placeholder-bucket/raw_parquet/"
PROCESSED_DATA_FILE_PATH = "s3://placeholder-bucket/processed/player_features.parquet"
LABEL_FILE_PATH = "s3://placeholder-bucket/label/train_labeld.csv"


@task(name="preprocess_data_task")
def preprocess_data_task(worker_image: str, n_workers: int = 4):
    """Preprocess raw player data into features using Dask.

    Args:
        n_workers: Number of Dask workers for parallel processing (default: 4)
    """
    preprocess_all_players(
        raw_data_path=RAW_DATA_PATH,
        output_file_path=PROCESSED_DATA_FILE_PATH,
        n_workers=n_workers,
        worker_image=worker_image,
    )


@task(name="materialize_features_task")
def materialize_features_task():
    """Materialize features to Feast online store for real-time serving.

    Automatically determines the date range from the processed data file.
    """

    fs = FeatureStore(repo_path="feature_store")

    # Read the processed data to determine date range
    data_file_path = PROCESSED_DATA_FILE_PATH
    df = pd.read_parquet(data_file_path, columns=["last_session_timestamp"])

    # Get min and max timestamps from the data
    min_date = df["last_session_timestamp"].min()
    max_date = df["last_session_timestamp"].max()

    # Add a small buffer to ensure all data is included
    start_date = min_date - timedelta(days=1)
    end_date = max_date + timedelta(days=1)

    logger.info(f"Materializing features from {start_date} to {end_date}")
    logger.info(f"  Data range: {min_date} to {max_date}")
    logger.info(f"  Total records: {len(df):,}")

    fs.materialize(start_date=start_date, end_date=end_date)
    logger.info("Features materialized to online store")


@task(name="train_model_task")
def train_model_task(
    mlflow_tracking_uri: str,
    feature_store_path: str = "feature_store",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """Train LightGBM model for churn prediction.

    Args:
        feature_store_path: Path to Feast feature store
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with evaluation metrics
    """

    metrics = train_model_pipeline(
        mlflow_tracking_uri=mlflow_tracking_uri,
        feature_store_path=feature_store_path,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        label_file_path=LABEL_FILE_PATH,
    )

    return {"metrics": metrics}


@task(name="deploy_model_task")
def deploy_model_task():
    # TODO: Implement model deployment
    pass


@flow(name="ml_pipeline_flow")
def ml_pipeline_flow(run_preprocessing: bool = False, run_training: bool = True):
    """Complete ML pipeline: preprocess → materialize → train → deploy.

    Args:
        run_preprocessing: Whether to run preprocessing step (default: False)
        run_training: Whether to run training step (default: True)
    """
    if run_preprocessing:
        preprocess_data_task(worker_image="placerhold_dask_image")
        materialize_features_task()

    if run_training:
        train_results = train_model_task(mlflow_tracking_uri="placeholder_mlflow_uri")
        logger.info(f"Training completed: {train_results}")

    # deploy_model_task()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ml_pipeline_flow()
