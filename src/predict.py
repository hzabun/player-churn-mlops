import sys

import mlflow
import mlflow.lightgbm
import pandas as pd
from feast import FeatureStore


def load_model(
    mlflow_tracking_uri: str,
    model_uri: str = "models:/player_churn_lightgbm/latest",
):
    """Load trained LightGBM model from MLflow.

    Args:
        model_uri: MLflow model URI. Options:
            - "models:/player_churn_lightgbm/latest" (latest version)
            - "models:/player_churn_lightgbm/production" (production version)
            - "models:/player_churn_lightgbm/3" (specific version)
            - "runs:/<run_id>/model" (specific run)
        mlflow_tracking_uri: MLflow tracking URI

    Returns:
        Loaded LightGBM model
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model = mlflow.lightgbm.load_model(model_uri)
    print(f"✓ Model loaded from MLflow: {model_uri}")
    return model


def get_online_features(
    actor_account_ids: list[str], feature_store_path: str = "feature_store"
) -> pd.DataFrame:
    """Fetch features for players from Feast online store.

    Args:
        actor_account_ids: List of player account IDs
        feature_store_path: Path to Feast feature store repository

    Returns:
        DataFrame with features for the specified players
    """
    fs = FeatureStore(repo_path=feature_store_path)

    # Prepare entity rows
    entity_rows = [{"actor_account_id": aid} for aid in actor_account_ids]

    # Define all features to fetch (excluding churn_yn for prediction)
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
    ]

    # Get online features
    features_df = fs.get_online_features(
        features=feature_list,
        entity_rows=entity_rows,
    ).to_df()

    return features_df


def predict_churn(
    mlflow_tracking_uri: str,
    actor_account_ids: list[str],
    model_uri: str = "models:/player_churn_lightgbm/latest",
    feature_store_path: str = "feature_store",
) -> pd.DataFrame:
    """Predict churn probability for players.

    Args:
        actor_account_ids: List of player account IDs
        model_uri: MLflow model URI
        feature_store_path: Path to Feast feature store
        mlflow_tracking_uri: MLflow tracking URI

    Returns:
        DataFrame with account IDs, churn predictions, and probabilities
    """
    # Load model
    model = load_model(model_uri, mlflow_tracking_uri)

    # Get features from Feast online store
    print(f"Fetching features for {len(actor_account_ids)} players...")
    features_df = get_online_features(actor_account_ids, feature_store_path)

    # Remove actor_account_id for prediction
    X = features_df.drop(columns=["actor_account_id"])

    # Make predictions
    churn_probabilities = model.predict(X, num_iteration=model.best_iteration)
    churn_predictions = (churn_probabilities > 0.5).astype(int)

    # Prepare results
    results = pd.DataFrame(
        {
            "actor_account_id": features_df["actor_account_id"],
            "churn_prediction": churn_predictions,
            "churn_probability": churn_probabilities,
        }
    )

    return results


def predict_single_player(
    actor_account_id: str,
    model_uri: str = "models:/player_churn_lightgbm/latest",
    feature_store_path: str = "feature_store",
    mlflow_tracking_uri: str = "mlruns",
) -> dict:
    """Predict churn for a single player.

    Args:
        actor_account_id: Player account ID
        model_uri: MLflow model URI
        feature_store_path: Path to Feast feature store
        mlflow_tracking_uri: MLflow tracking URI

    Returns:
        Dictionary with prediction results
    """
    results = predict_churn(
        [actor_account_id], model_uri, feature_store_path, mlflow_tracking_uri
    )

    return {
        "actor_account_id": actor_account_id,
        "will_churn": bool(results["churn_prediction"].iloc[0]),
        "churn_probability": float(results["churn_probability"].iloc[0]),
    }


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Churn Prediction Example")
    print("=" * 70)

    # Load a sample player ID from the data
    try:
        if len(sys.argv) > 1:
            # Use account ID from command line argument
            account_id = sys.argv[1]
        else:
            # Load a sample from the processed data
            df = pd.read_parquet("data/processed/player_features.parquet", nrows=5)
            account_id = str(df["actor_account_id"].iloc[0])

        print(f"\nPredicting churn for player: {account_id}")
        result = predict_single_player(account_id)

        print("\nPrediction Result:")
        print(f"  Player ID: {result['actor_account_id']}")
        print(f"  Will Churn: {result['will_churn']}")
        print(f"  Churn Probability: {result['churn_probability']:.2%}")

        print("\n" + "=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: Data not found - {e}")
        print("Please ensure data/processed/player_features.parquet exists")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError: MLflow model not found - {e}")
        print("Please run training first: python src/train.py")
        print("Or check MLflow tracking URI and model name")
    except Exception as e:
        print(f"\nError: {e}")
