import numpy as np
import pandas as pd

# Features identified from EDA on training data
ZERO_VAR_FEATURES = [
    "party_kick_cnt",
    "party_kick_cnt_last",
    "party_kick_cnt_slope",
    "party_kick_cnt_cv",
]

ID_COLUMNS = ["", "actor_account_id", "Unnamed: 0"]


def validate_data(df):
    """Validate data quality and print summary."""
    print("\n  Data Validation:")

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  ⚠ Missing values: {missing_count}")
    else:
        print("  ✓ No missing values")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        print(f"  ⚠ Infinite values: {inf_count}")
    else:
        print("  ✓ No infinite values")

    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"  ⚠ Duplicate rows: {dup_count}")
    else:
        print("  ✓ No duplicate rows")

    # Check target distribution (if exists)
    if "churn_yn" in df.columns:
        churn_rate = df["churn_yn"].mean()
        print(f"  ✓ Churn rate: {churn_rate:.2%}")


def preprocess_features(df):
    """
    Apply preprocessing transformations to features.

    This function is the SAME for both training and inference.
    Features to drop are HARDCODED based on training data analysis.

    Steps:
    1. Drop ID columns (not predictive)
    2. Handle missing values (defensive)
    3. Handle infinite values (defensive)
    4. Encode categorical feature (actor_job → integer)
    5. Drop zero-variance features (identified from training)

    Args:
        df: Input dataframe

    Returns:
        Preprocessed dataframe
    """
    df = df.copy()

    # 1. Drop ID columns
    cols_to_drop = [col for col in ID_COLUMNS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 2. Handle missing values (defensive - shouldn't have any)
    if df.isnull().sum().sum() > 0:
        df = df.fillna(0)

    # 3. Handle infinite values (defensive - shouldn't have any)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            max_val = df[col][~np.isinf(df[col])].max()
            df[col] = df[col].replace([np.inf, -np.inf], max_val)

    # 4. Encode categorical feature (actor_job)
    if "actor_job" in df.columns:
        df["actor_job"] = df["actor_job"].astype(int)

    # 5. Drop zero-variance features (hardcoded from training analysis)
    # errors='ignore' means won't fail if column doesn't exist in test data
    df = df.drop(columns=ZERO_VAR_FEATURES, errors="ignore")

    return df


def load_and_preprocess_data(train_path="data/train.csv", validate=True):
    """
    Load and preprocess training data.

    Args:
        train_path: Path to training data CSV
        validate: Whether to run data validation

    Returns:
        X: Preprocessed features
        y: Target variable
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(train_path)
    print(f"   Raw data shape: {df.shape}")

    # Validate raw data
    if validate:
        print("\n2. Validating raw data...")
        validate_data(df)

    # Preprocess
    print("\n3. Preprocessing features...")
    df_processed = preprocess_features(df)

    # Separate features and target
    y = df["churn_yn"]
    X = (
        df_processed.drop(columns=["churn_yn"])
        if "churn_yn" in df_processed.columns
        else df_processed
    )

    print(f"\n4. Final data shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")

    print("\n" + "=" * 60)

    return X, y


if __name__ == "__main__":
    # Test the preprocessor
    X, y = load_and_preprocess_data()
    print("\n✓ Preprocessing test completed successfully!")
    print(f"  Final shape: X={X.shape}, y={y.shape}")

    # Test on test data to ensure consistency
    print("\n" + "=" * 60)
    print("Testing on test1.csv...")
    print("=" * 60)
    test_df = pd.read_csv("data/test1.csv")
    print(f"Raw test shape: {test_df.shape}")
    test_processed = preprocess_features(test_df)
    print(f"Processed test shape: {test_processed.shape}")
    print(
        f"\n✓ Train and test have same features: {X.shape[1] == test_processed.shape[1]}"
    )
    if X.shape[1] != test_processed.shape[1]:
        print(f"  ⚠ Warning: Different number of features!")
        print(f"     Train: {X.shape[1]}, Test: {test_processed.shape[1]}")
