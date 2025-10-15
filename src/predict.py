import mlflow
import pandas as pd

from src.preprocess import preprocess_features


def predict_churn(data_path, model_uri=None):
    """
    Make predictions on new data using the trained model.

    Args:
        data_path: Path to CSV file with new data
        model_uri: MLflow model URI (if None, loads latest from tracking server)

    Returns:
        DataFrame with predictions
    """
    print("=" * 60)
    print("CHURN PREDICTION")
    print("=" * 60)

    # Load and preprocess data
    print(f"\nLoading and preprocessing {data_path}...")
    df = pd.read_csv(data_path)
    original_df = df.copy()

    X = preprocess_features(df)
    # Drop target if present in test data
    if "churn_yn" in X.columns:
        X = X.drop(columns=["churn_yn"])

    print(f"   Processed shape: {X.shape}")

    # Load model
    print(f"\nLoading model from MLflow...")
    if model_uri is None:
        # Load latest model from MLflow
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("player_churn_prediction")
        runs = client.search_runs(
            experiment.experiment_id, order_by=["start_time DESC"], max_results=1
        )
        if runs:
            run_id = runs[0].info.run_id
            model_uri = f"runs:/{run_id}/model"
            print(f"   Using model from run: {run_id}")

    model = mlflow.lightgbm.load_model(model_uri)

    # Make predictions
    print("\nMaking predictions...")
    predictions_proba = model.predict(X)
    predictions = (predictions_proba > 0.5).astype(int)

    # Prepare results
    results = (
        original_df[["actor_account_id"]].copy()
        if "actor_account_id" in original_df.columns
        else pd.DataFrame()
    )
    results["churn_probability"] = predictions_proba
    results["churn_prediction"] = predictions

    print(f"\nResults:")
    print(f"   Total predictions: {len(results)}")
    print(f"   Predicted churners: {predictions.sum()} ({predictions.mean():.2%})")

    print("\n" + "=" * 60)
    print("✓ PREDICTION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # Example: Make predictions on test data
    import sys

    test_file = sys.argv[1] if len(sys.argv) > 1 else "data/test1.csv"

    results = predict_churn(test_file)

    # Save predictions
    output_file = test_file.replace(".csv", "_predictions.csv")
    results.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")

    # Show sample predictions
    print("\nSample predictions:")
    print(results.head(10))
