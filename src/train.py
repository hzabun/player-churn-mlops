import pandas as pd
from feast import FeatureStore

fs = FeatureStore(repo_path="feature_store")

features_df = pd.read_parquet("feature_store/data/player_features.parquet")

labeled_df = features_df[features_df["churn_yn"].notna()].copy()

entity_df = labeled_df[
    ["actor_account_id", "last_session_timestamp", "churn_yn"]
].copy()
entity_df = entity_df.rename(columns={"last_session_timestamp": "event_timestamp"})

# Get historical features (point-in-time correct)
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        "player_features:churn_yn",
        "player_features:total_sessions",
        "player_features:average_sessions_per_day",
        # ... more features as needed
    ],
).to_df()

X = training_df.drop(columns=["actor_account_id", "event_timestamp", "churn_yn"])
y = training_df["churn_yn"]

# Train your LightGBM model
# model.fit(X, y)
