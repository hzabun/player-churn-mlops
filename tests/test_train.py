import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.train import evaluate_model, prepare_features, train_lightgbm_model


@pytest.fixture
def sample_training_data():
    """Create a small sample DataFrame mimicking player features with Feast-style columns."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "actor_account_id": [f"player_{i}" for i in range(n_samples)],
            "event_timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="h"),
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


class TestPrepareFeatures:
    """Tests for prepare_features function."""

    def test_prepare_features_returns_X_and_y(self, sample_training_data):
        """Test that prepare_features returns X and y."""
        X, y = prepare_features(sample_training_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_prepare_features_excludes_correct_columns(self, sample_training_data):
        """Test that identifier and label columns are excluded from X."""
        X, y = prepare_features(sample_training_data)

        # These columns should not be in X
        assert "actor_account_id" not in X.columns
        assert "event_timestamp" not in X.columns
        assert "churn_yn" not in X.columns

        # But feature columns should be present
        assert "total_sessions" in X.columns
        assert "avg_session_duration_minutes" in X.columns

    def test_prepare_features_includes_correct_columns(self, sample_training_data):
        X, y = prepare_features(sample_training_data)

        exclude_list = ["actor_account_id", "event_timestamp", "churn_yn"]

        include_list = sample_training_data.drop(
            columns=exclude_list, errors="ignore"
        ).columns.to_list()

        assert set(include_list).issubset(set(X.columns))

    def test_prepare_features_y_is_churn_column(self, sample_training_data):
        """Test that y contains the churn_yn values."""
        X, y = prepare_features(sample_training_data)

        assert y.name == "churn_yn"
        assert set(y.unique()).issubset({0, 1})

    def test_prepare_features_preserves_row_count(self, sample_training_data):
        """Test that no rows are lost during feature preparation."""
        X, y = prepare_features(sample_training_data)

        assert len(X) == len(sample_training_data)
        assert len(y) == len(sample_training_data)


class TestTrainLightGBMModel:
    """Tests for train_lightgbm_model function."""

    def test_train_lightgbm_model_returns_booster(self, sample_training_data):
        """Test that train_lightgbm_model returns a Booster object."""
        X, y = prepare_features(sample_training_data)
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        with tempfile.TemporaryDirectory() as tmpdir:
            import mlflow

            mlflow.set_tracking_uri(tmpdir)
            mlflow.set_experiment("test_experiment")

            with mlflow.start_run():
                model = train_lightgbm_model(X_train, y_train, X_val, y_val)

            assert isinstance(model, lgb.Booster)
            assert model.best_iteration > 0

    def test_train_lightgbm_model_uses_custom_params(self, sample_training_data):
        """Test that custom parameters are passed to LightGBM."""
        X, y = prepare_features(sample_training_data)
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        custom_params = {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 16,
            "verbose": -1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            import mlflow

            mlflow.set_tracking_uri(tmpdir)
            mlflow.set_experiment("test_experiment")

            with mlflow.start_run():
                model = train_lightgbm_model(
                    X_train, y_train, X_val, y_val, params=custom_params
                )

            assert isinstance(model, lgb.Booster)
            # Model should have trained successfully with custom params
            assert model.best_iteration > 0

    def test_train_lightgbm_model_can_predict(self, sample_training_data):
        """Test that the trained model can make predictions."""
        X, y = prepare_features(sample_training_data)
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        with tempfile.TemporaryDirectory() as tmpdir:
            import mlflow

            mlflow.set_tracking_uri(tmpdir)
            mlflow.set_experiment("test_experiment")

            with mlflow.start_run():
                model = train_lightgbm_model(X_train, y_train, X_val, y_val)

            predictions = model.predict(X_val)
            predictions_array = np.array(predictions)
            assert len(predictions_array) == len(X_val)
            assert all(0 <= p <= 1 for p in predictions_array)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_evaluate_model_returns_metrics_dict(self, sample_training_data):
        """Test that evaluate_model returns a dictionary of metrics."""
        X, y = prepare_features(sample_training_data)
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        X_test, y_test = X_val[:10], y_val[:10]

        with tempfile.TemporaryDirectory() as tmpdir:
            import mlflow

            mlflow.set_tracking_uri(tmpdir)
            mlflow.set_experiment("test_experiment")

            with mlflow.start_run():
                model = train_lightgbm_model(X_train, y_train, X_val, y_val)
                metrics = evaluate_model(model, X_test, y_test)

            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics

    def test_evaluate_model_metrics_in_valid_range(self, sample_training_data):
        """Test that all metrics are between 0 and 1."""
        X, y = prepare_features(sample_training_data)
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        X_test, y_test = X_val[:10], y_val[:10]

        with tempfile.TemporaryDirectory() as tmpdir:
            import mlflow

            mlflow.set_tracking_uri(tmpdir)
            mlflow.set_experiment("test_experiment")

            with mlflow.start_run():
                model = train_lightgbm_model(X_train, y_train, X_val, y_val)
                metrics = evaluate_model(model, X_test, y_test)

            for metric_name, metric_value in metrics.items():
                assert 0 <= metric_value <= 1, (
                    f"{metric_name} should be between 0 and 1, got {metric_value}"
                )
