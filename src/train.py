import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Import preprocessing module
from src.preprocess import load_and_preprocess_data


def train_with_cv(X, y, params, n_folds=5):
    """Train model with k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        # Evaluate
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        cv_scores.append(auc)
        models.append(model)

        print(f"Fold {fold + 1} - AUC: {auc:.4f}")

    avg_auc = np.mean(cv_scores)
    print(f"\nAverage CV AUC: {avg_auc:.4f} (+/- {np.std(cv_scores):.4f})")

    # Return best model (highest AUC)
    best_idx = np.argmax(cv_scores)
    return models[best_idx], avg_auc, cv_scores


def objective(trial, X, y):
    """Optuna objective function for hyperparameter tuning."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }

    # Quick 3-fold CV for hyperparameter search
    _, avg_auc, _ = train_with_cv(X, y, params, n_folds=3)

    return avg_auc


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Player Churn Prediction - LightGBM Training")
    print("=" * 60)

    # Load and preprocess data
    X, y = load_and_preprocess_data(train_path="data/train.csv", validate=True)
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")

    # Set MLflow experiment
    mlflow.set_experiment("player_churn_prediction")

    with mlflow.start_run(run_name="lightgbm_churn_model"):

        # Hyperparameter tuning
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        study = optuna.create_study(direction="maximize", study_name="lgbm_tuning")
        study.optimize(
            lambda trial: objective(trial, X, y), n_trials=20, show_progress_bar=True
        )

        print(f"\nBest AUC: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        # Final training with best params
        print("\n" + "=" * 60)
        print("FINAL MODEL TRAINING")
        print("=" * 60)
        best_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            **study.best_params,
        }

        final_model, avg_auc, cv_scores = train_with_cv(X, y, best_params, n_folds=5)

        # Calculate additional metrics on full training set
        y_pred_proba = final_model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            "cv_auc_mean": avg_auc,
            "cv_auc_std": np.std(cv_scores),
            "train_auc": roc_auc_score(y, y_pred_proba),
            "train_accuracy": accuracy_score(y, y_pred),
            "train_precision": precision_score(y, y_pred),
            "train_recall": recall_score(y, y_pred),
            "train_f1": f1_score(y, y_pred),
        }

        # Log to MLflow
        print("\n" + "=" * 60)
        print("MLFLOW LOGGING")
        print("=" * 60)
        print("  → Logging parameters...")
        mlflow.log_params(best_params)

        print("  → Logging metrics...")
        mlflow.log_metrics(metrics)

        print("  → Logging model...")
        mlflow.lightgbm.log_model(final_model, "model")

        # Log feature importance
        print("  → Logging feature importance...")
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": final_model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)

        importance_df.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"CV AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
        print(f"Train AUC: {metrics['train_auc']:.4f}")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Train F1: {metrics['train_f1']:.4f}")
        print(f"\n✓ Model logged to MLflow")
        print("=" * 60)


if __name__ == "__main__":
    main()
