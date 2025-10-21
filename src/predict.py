from feast import FeatureStore

fs = FeatureStore(repo_path="feature_store")

# Get features for a newly logged-in player
entity_rows = [{"actor_account_id": "ABC123"}]

features = fs.get_online_features(
    features=[
        "player_features:total_sessions",
        "player_features:average_sessions_per_day",
        # ... all features
    ],
    entity_rows=entity_rows,
).to_df()

# Use features for churn prediction
churn_probability = model.predict_proba(features)
