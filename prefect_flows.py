from pathlib import Path

from prefect import flow, task

from src.preprocess import preprocess_all_players

RAW_DATA_PATH = Path("data/raw_parquet")
PROCESSED_DATA_PATH = Path("data/processed")


@task(name="preprocess_data_task")
def preprocess_data_task():
    preprocess_all_players(raw_dir=RAW_DATA_PATH, output_dir=PROCESSED_DATA_PATH)


@task(name="train_model_task")
def train_model_task():
    pass


@task(name="deploy_model_task")
def deploy_model_task():
    pass


@flow(name="ml_pipeline_flow")
def ml_pipeline_flow():
    preprocess_data_task()
