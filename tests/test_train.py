from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_features_df():
    """Create a small sample DataFrame mimicking player features."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "actor_account_id": [f"player_{i}" for i in range(n_samples)],
            "last_session_timestamp": pd.date_range(
                "2024-01-01", periods=n_samples, freq="h"
            ),
            "total_sessions": np.random.randint(1, 50, n_samples),
            "account_lifespan_days": np.random.uniform(1, 100, n_samples),
            "average_sessions_per_day": np.random.uniform(0.1, 5, n_samples),
            "total_playtime_minutes": np.random.randint(10, 1000, n_samples),
            "avg_session_duration_minutes": np.random.uniform(10, 60, n_samples),
            "std_session_duration_minutes": np.random.uniform(5, 30, n_samples),
            "min_session_duration_minutes": np.random.randint(1, 10, n_samples),
            "max_session_duration_minutes": np.random.randint(60, 120, n_samples),
            "delete_pc": np.random.randint(0, 5, n_samples),
            "level_ups_across_all_characters": np.random.randint(0, 50, n_samples),
            "invite_party": np.random.randint(0, 20, n_samples),
            "refuse_party": np.random.randint(0, 10, n_samples),
            "join_party": np.random.randint(0, 15, n_samples),
            "die": np.random.randint(0, 100, n_samples),
            "duel_end_pc": np.random.randint(0, 20, n_samples),
            "duel_end_team": np.random.randint(0, 10, n_samples),
            "party_battle_end_team": np.random.randint(0, 15, n_samples),
            "expand_warehouse": np.random.randint(0, 5, n_samples),
            "change_item_look": np.random.randint(0, 10, n_samples),
            "put_main_auction": np.random.randint(0, 30, n_samples),
            "use_gathering_item": np.random.randint(0, 50, n_samples),
            "complete_quest": np.random.randint(0, 100, n_samples),
            "complete_challenge_today": np.random.randint(0, 10, n_samples),
            "complete_challenge_week": np.random.randint(0, 5, n_samples),
            "create_guild": np.random.randint(0, 2, n_samples),
            "destroy_guild": np.random.randint(0, 2, n_samples),
            "invite_guild": np.random.randint(0, 10, n_samples),
            "join_guild": np.random.randint(0, 3, n_samples),
            "dismiss_guild": np.random.randint(0, 2, n_samples),
            "churn_yn": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def sample_training_data(sample_features_df):
    """Training data with Feast-style columns."""
    df = sample_features_df.copy()
    df = df.rename(columns={"last_session_timestamp": "event_timestamp"})
    return df


@pytest.fixture
def sample_X_y(sample_features_df):
    """Prepared features (X) and target (y) for training."""
    exclude_cols = ["actor_account_id", "last_session_timestamp", "churn_yn"]
    X = sample_features_df.drop(columns=exclude_cols)
    y = sample_features_df["churn_yn"]
    return X, y


class TestGetTrainingData:
    """Tests for get_training_data function."""

    @patch("src.train.FeatureStore")
    @patch("src.train.pd.read_parquet")
    def test_get_training_data_returns_dataframe(
        self, mock_read_parquet, mock_feast_store, sample_features_df
    ):
        """Test that get_training_data returns a DataFrame."""
        from src.train import get_training_data

        # Mock the parquet file read
        mock_read_parquet.return_value = sample_features_df

        # Mock Feast FeatureStore
        mock_fs_instance = MagicMock()
        mock_feast_store.return_value = mock_fs_instance

        # Mock get_historical_features to return sample data
        mock_historical = MagicMock()
        mock_historical.to_df.return_value = sample_features_df
        mock_fs_instance.get_historical_features.return_value = mock_historical

        # Call the function
        training_data = get_training_data()

        # Assertions
        assert isinstance(training_data, pd.DataFrame)
        assert len(training_data) > 0
        mock_feast_store.assert_called_once()
        mock_fs_instance.get_historical_features.assert_called_once()

    @patch("src.train.FeatureStore")
    @patch("src.train.pd.read_parquet")
    def test_get_training_data_filters_labeled_players(
        self, mock_read_parquet, mock_feast_store, sample_features_df
    ):
        """Test that only labeled players are included."""
        from src.train import get_training_data

        # Add some NaN values to churn_yn
        df_with_nans = sample_features_df.copy()
        df_with_nans.loc[:10, "churn_yn"] = None

        mock_read_parquet.return_value = df_with_nans

        mock_fs_instance = MagicMock()
        mock_feast_store.return_value = mock_fs_instance

        # The function should only use labeled data for entity_df
        mock_historical = MagicMock()
        mock_historical.to_df.return_value = sample_features_df
        mock_fs_instance.get_historical_features.return_value = mock_historical

        get_training_data()

        # Verify get_historical_features was called with labeled data only
        call_args = mock_fs_instance.get_historical_features.call_args
        entity_df = call_args.kwargs["entity_df"]

        # Entity df should have fewer rows than original (NaNs filtered)
        assert len(entity_df) < len(df_with_nans)


class TestPrepareFeatures:
    """Tests for prepare_features function."""

    def test_prepare_features_returns_X_and_y(self, sample_training_data):
        """Test that prepare_features returns X and y."""
        from src.train import prepare_features

        X, y = prepare_features(sample_training_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_prepare_features_excludes_correct_columns(self, sample_training_data):
        """Test that identifier and label columns are excluded from X."""
        from src.train import prepare_features

        X, y = prepare_features(sample_training_data)

        # These columns should not be in X
        assert "actor_account_id" not in X.columns
        assert "event_timestamp" not in X.columns
        assert "churn_yn" not in X.columns

        # But feature columns should be present
        assert "total_sessions" in X.columns
        assert "avg_session_duration_minutes" in X.columns

    def test_prepare_features_y_is_churn_column(self, sample_training_data):
        """Test that y contains the churn_yn values."""
        from src.train import prepare_features

        X, y = prepare_features(sample_training_data)

        """Test that y contains the churn_yn values."""
        from src.train import prepare_features

        X, y = prepare_features(sample_training_data)

        assert y.name == "churn_yn"
        assert set(y.unique()).issubset({0, 1})


class TestTrainLightGBMModel:
    """Tests for train_lightgbm_model function."""

    @patch("src.train.mlflow")
    @patch("src.train.lgb.train")
    def test_train_lightgbm_model_returns_booster(
        self, mock_lgb_train, mock_mlflow, sample_X_y
    ):
        """Test that train_lightgbm_model returns a Booster object."""
        from src.train import train_lightgbm_model

        X, y = sample_X_y
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        # Mock the trained model
        mock_model = MagicMock(spec=lgb.Booster)
        mock_model.best_iteration = 50
        mock_model.best_score = {
            "train": {"binary_logloss": 0.5},
            "valid": {"binary_logloss": 0.6},
        }
        mock_lgb_train.return_value = mock_model

        result = train_lightgbm_model(X_train, y_train, X_val, y_val)

        assert result is mock_model
        mock_lgb_train.assert_called_once()
        mock_mlflow.log_params.assert_called()

    @patch("src.train.mlflow")
    @patch("src.train.lgb.train")
    def test_train_lightgbm_model_uses_custom_params(
        self, mock_lgb_train, mock_mlflow, sample_X_y
    ):
        """Test that custom parameters are passed to LightGBM."""
        from src.train import train_lightgbm_model

        X, y = sample_X_y
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        custom_params = {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 64,
        }

        mock_model = MagicMock(spec=lgb.Booster)
        mock_model.best_iteration = 50
        mock_model.best_score = {"train": {"binary_logloss": 0.5}}
        mock_lgb_train.return_value = mock_model

        train_lightgbm_model(X_train, y_train, X_val, y_val, params=custom_params)

        # Check that lgb.train was called with custom params
        call_args = mock_lgb_train.call_args
        params_used = call_args[0][0]
        assert params_used["learning_rate"] == 0.1
        assert params_used["num_leaves"] == 64


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @patch("src.train.mlflow")
    def test_evaluate_model_returns_metrics_dict(self, mock_mlflow, sample_X_y):
        """Test that evaluate_model returns a dictionary of metrics."""
        from src.train import evaluate_model

        X, y = sample_X_y
        X_test, y_test = X[:20], y[:20]

        # Create a mock model that returns predictions
        mock_model = MagicMock(spec=lgb.Booster)
        mock_model.best_iteration = 50
        mock_model.predict.return_value = np.random.random(len(X_test))

        metrics = evaluate_model(mock_model, X_test, y_test)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # All metrics should be between 0 and 1
        for metric_value in metrics.values():
            assert 0 <= metric_value <= 1

    @patch("src.train.mlflow")
    def test_evaluate_model_logs_to_mlflow(self, mock_mlflow, sample_X_y):
        """Test that metrics are logged to MLflow."""
        from src.train import evaluate_model

        X, y = sample_X_y
        X_test, y_test = X[:20], y[:20]

        mock_model = MagicMock(spec=lgb.Booster)
        mock_model.best_iteration = 50
        mock_model.predict.return_value = np.random.random(len(X_test))

        evaluate_model(mock_model, X_test, y_test)

        # Verify MLflow log_metric was called for each metric
        assert mock_mlflow.log_metric.call_count == 4  # accuracy, precision, recall, f1


class TestTrainModelPipeline:
    """Tests for the complete train_model_pipeline function."""

    @patch("src.train.mlflow")
    @patch("src.train.get_training_data")
    @patch("src.train.train_lightgbm_model")
    def test_train_model_pipeline_executes_successfully(
        self, mock_train_model, mock_get_data, mock_mlflow, sample_training_data
    ):
        """Test that the full pipeline executes without errors."""
        from src.train import train_model_pipeline

        # Mock data retrieval
        mock_get_data.return_value = sample_training_data

        # Mock model training
        mock_model = MagicMock(spec=lgb.Booster)
        mock_model.best_iteration = 50
        mock_model.best_score = {"train": {"binary_logloss": 0.5}}
        mock_model.predict.return_value = np.random.random(20)
        mock_model.feature_importance.return_value = np.random.random(28)
        mock_train_model.return_value = mock_model

        # Mock MLflow context
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run the pipeline
        metrics = train_model_pipeline(
            mlflow_tracking_uri="./test_mlruns",
            experiment_name="test_experiment",
            run_name="test_run",
        )

        # Assertions
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        mock_get_data.assert_called_once()
        mock_train_model.assert_called_once()
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()

    @patch("src.train.mlflow")
    @patch("src.train.get_training_data")
    @patch("src.train.train_lightgbm_model")
    def test_train_model_pipeline_logs_parameters(
        self, mock_train_model, mock_get_data, mock_mlflow, sample_training_data
    ):
        """Test that pipeline parameters are logged to MLflow."""
        from src.train import train_model_pipeline

        mock_get_data.return_value = sample_training_data

        # Create a mock model that adapts to the test set size
        mock_model = MagicMock(spec=lgb.Booster)
        mock_model.best_iteration = 50
        mock_model.best_score = {"train": {"binary_logloss": 0.5}}
        # Mock predict to return the right size based on input
        mock_model.predict.side_effect = lambda X, **kwargs: np.random.random(len(X))
        mock_model.feature_importance.return_value = np.random.random(28)
        mock_train_model.return_value = mock_model

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        train_model_pipeline(
            mlflow_tracking_uri="./test_mlruns",
            test_size=0.25,
            val_size=0.15,
            random_state=123,
        )

        # Check that log_param was called with expected parameters
        param_calls = [call[0][0] for call in mock_mlflow.log_param.call_args_list]
        assert "test_size" in param_calls
        assert "val_size" in param_calls
        assert "random_state" in param_calls

    @patch("src.train.mlflow")
    @patch("src.train.get_training_data")
    @patch("src.train.train_lightgbm_model")
    def test_train_model_pipeline_splits_data_correctly(
        self, mock_train_model, mock_get_data, mock_mlflow, sample_training_data
    ):
        """Test that data is split into train/val/test sets."""
        from src.train import train_model_pipeline

        mock_get_data.return_value = sample_training_data

        # Create a mock model that adapts to the test set size
        mock_model = MagicMock(spec=lgb.Booster)
        mock_model.best_iteration = 50
        mock_model.best_score = {"train": {"binary_logloss": 0.5}}
        # Mock predict to return the right size based on input
        mock_model.predict.side_effect = lambda X, **kwargs: np.random.random(len(X))
        mock_model.feature_importance.return_value = np.random.random(28)
        mock_train_model.return_value = mock_model

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        train_model_pipeline(
            mlflow_tracking_uri="./test_mlruns", test_size=0.2, val_size=0.1
        )

        # Check that metrics were logged for train/val/test splits
        metric_calls = [call[0][0] for call in mock_mlflow.log_metric.call_args_list]
        assert "train_samples" in metric_calls
        assert "val_samples" in metric_calls
        assert "test_samples" in metric_calls
