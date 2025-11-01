import logging

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from feast import FeatureStore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_training_data(feature_store_path: str = "feature_store") -> pd.DataFrame:
    """Fetch historical features from Feast for model training.

    Args:
        feature_store_path: Path to Feast feature store repository

    Returns:
        DataFrame with features ready for training
    """
    logger.info("Loading Feast feature store...")
    fs = FeatureStore(repo_path=feature_store_path)

    features_df = pd.read_parquet("data/processed/player_features.parquet")

    labeled_df = features_df[features_df["churn_yn"].notna()].copy()
    logger.info(f"Found {len(labeled_df):,} labeled players")

    # Prepare entity dataframe for Feast
    entity_df = labeled_df[["actor_account_id", "last_session_timestamp"]].copy()
    entity_df = entity_df.rename(columns={"last_session_timestamp": "event_timestamp"})

    feature_list = [
        "player_features:total_sessions",
        "player_features:account_lifespan_days",
        "player_features:average_sessions_per_day",
        "player_features:total_playtime_minutes",
        "player_features:avg_session_duration_minutes",
        "player_features:std_session_duration_minutes",
        "player_features:min_session_duration_minutes",
        "player_features:max_session_duration_minutes",
        "player_features:delete_pc",
        "player_features:level_ups_across_all_characters",
        "player_features:invite_party",
        "player_features:refuse_party",
        "player_features:join_party",
        "player_features:die",
        "player_features:duel_end_pc",
        "player_features:duel_end_team",
        "player_features:party_battle_end_team",
        "player_features:expand_warehouse",
        "player_features:change_item_look",
        "player_features:put_main_auction",
        "player_features:use_gathering_item",
        "player_features:complete_quest",
        "player_features:complete_challenge_today",
        "player_features:complete_challenge_week",
        "player_features:create_guild",
        "player_features:destroy_guild",
        "player_features:invite_guild",
        "player_features:join_guild",
        "player_features:dismiss_guild",
        "player_features:churn_yn",  # Include label for easier processing
    ]

    logger.info("Fetching historical features from Feast...")
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=feature_list,
    ).to_df()

    logger.info(
        f"Retrieved {len(training_df):,} records with {training_df.shape[1]} columns"
    )
    return training_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for model training.

    Args:
        df: DataFrame with features and labels from Feast

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Drop identifier and timestamp columns
    exclude_cols = ["actor_account_id", "event_timestamp", "churn_yn"]
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    y = df["churn_yn"]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    logger.info(f"Churn rate: {y.mean():.2%}")

    return X, y


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> lgb.Booster:
    """Train a LightGBM classifier for churn prediction.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Optional model parameters (uses defaults if None)

    Returns:
        Trained LightGBM Booster model
    """
    logger.info("Training LightGBM model...")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Set parameters for binary classification
    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "min_data_in_leaf": 20,
        }

    mlflow.log_params(params)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10),
        ],
    )

    mlflow.log_metric("best_iteration", model.best_iteration)
    mlflow.log_metric("num_trees", model.best_iteration)

    for dataset_name, scores in model.best_score.items():
        for metric_name, value in scores.items():
            mlflow.log_metric(f"{dataset_name}_{metric_name}", value)

    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score: {model.best_score}")

    return model


def evaluate_model(model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model performance on test set.

    Args:
        model: Trained LightGBM model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    for metric_name, value in metrics.items():
        mlflow.log_metric(f"test_{metric_name}", value)

    logger.info("Test Set Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name.capitalize()}: {value:.4f}")

    return metrics


def train_model_pipeline(
    mlflow_tracking_uri: str,
    experiment_name: str = "player_churn_prediction",
    run_name: str | None = None,
    feature_store_path: str = "feature_store",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Complete model training pipeline with MLflow tracking.

    Args:
        feature_store_path: Path to Feast feature store
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
        experiment_name: MLflow experiment name
        run_name: Optional MLflow run name
        mlflow_tracking_uri: MLflow tracking URI (local directory or remote server)

    Returns:
        Dictionary of evaluation metrics
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("=" * 70)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 70)
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"MLflow Experiment: {experiment_name}")
        logger.info("=" * 70)

        mlflow.log_param("feature_store_path", feature_store_path)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("random_state", random_state)

        # Step 1: Get training data from Feast
        training_df = get_training_data(feature_store_path)
        mlflow.log_param("total_samples", len(training_df))
        mlflow.log_param(
            "num_features", training_df.shape[1] - 3
        )  # Exclude ID, timestamp, label

        # Step 2: Prepare features and target
        X, y = prepare_features(training_df)

        mlflow.log_metric("churn_rate", y.mean())
        mlflow.log_metric("num_churned", y.sum())
        mlflow.log_metric("num_retained", (1 - y).sum())

        # Step 3: Split data into train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=y_temp,
        )

        logger.info("Data split:")
        logger.info(f"  Train: {len(X_train):,} samples")
        logger.info(f"  Validation: {len(X_val):,} samples")
        logger.info(f"  Test: {len(X_test):,} samples")

        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("val_samples", len(X_val))
        mlflow.log_metric("test_samples", len(X_test))

        # Step 4: Train model
        model = train_lightgbm_model(X_train, y_train, X_val, y_val)

        # Log feature importance (top 10 features)
        feature_importance = model.feature_importance(importance_type="gain")
        feature_names = X_train.columns
        for idx in feature_importance.argsort()[-10:][::-1]:
            mlflow.log_metric(
                f"feature_importance_{feature_names[idx]}",
                float(feature_importance[idx]),
            )

        # Step 5: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Step 6: Log model
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="player_churn_lightgbm",
        )
        logger.info("Model logged to MLflow")

        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("target", "churn_yn")

        logger.info("=" * 70)
        logger.info("Training Pipeline Complete")
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"MLflow UI: mlflow ui --backend-store-uri {mlflow_tracking_uri}")
        logger.info("=" * 70)

        return metrics


if __name__ == "__main__":
    metrics = train_model_pipeline(mlflow_tracking_uri="placeholder")
    logger.info(f"Final Metrics: {metrics}")
