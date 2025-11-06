from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import Float64, Int64

# ============================================================================
# Entity Definition
# ============================================================================

player = Entity(
    name="player",
    join_keys=["actor_account_id"],
    value_type=ValueType.STRING,
    description="Player account entity identified by actor_account_id",
)


# ============================================================================
# Data Source
# ============================================================================

player_source = FileSource(
    file_format=ParquetFormat(),
    name="player_features_source",
    path="s3://player-churn-bns-mlops/data/processed/player-features.parquet",
    timestamp_field="last_session_timestamp",
    description="Parquet file containing aggregated player session features",
)


# ============================================================================
# Feature View
# ============================================================================

player_features = FeatureView(
    name="player_features",
    entities=[player],
    ttl=timedelta(days=0),  # Set to 0 for testing purposes
    schema=[
        # Temporal and engagement metrics
        Field(
            name="total_sessions",
            dtype=Int64,
            description="Total number of game sessions",
        ),
        Field(
            name="account_lifespan_days",
            dtype=Float64,
            description="Days between first and last session",
        ),
        Field(
            name="average_sessions_per_day",
            dtype=Float64,
            description="Average sessions per day",
        ),
        # Playtime metrics
        Field(
            name="total_playtime_minutes",
            dtype=Int64,
            description="Total gameplay time in minutes",
        ),
        Field(
            name="avg_session_duration_minutes",
            dtype=Float64,
            description="Average session duration in minutes",
        ),
        Field(
            name="std_session_duration_minutes",
            dtype=Float64,
            description="Standard deviation of session duration",
        ),
        Field(
            name="min_session_duration_minutes",
            dtype=Int64,
            description="Minimum session duration in minutes",
        ),
        Field(
            name="max_session_duration_minutes",
            dtype=Int64,
            description="Maximum session duration in minutes",
        ),
        # Character management events
        Field(
            name="delete_pc", dtype=Int64, description="Number of character deletions"
        ),
        Field(
            name="level_ups_across_all_characters",
            dtype=Int64,
            description="Total level-ups across all characters",
        ),
        # Social interaction events
        Field(
            name="invite_party",
            dtype=Int64,
            description="Number of party invitations sent",
        ),
        Field(
            name="refuse_party",
            dtype=Int64,
            description="Number of party invitations refused",
        ),
        Field(
            name="join_party", dtype=Int64, description="Number of times joined a party"
        ),
        # Combat events
        Field(name="die", dtype=Int64, description="Number of character deaths"),
        Field(
            name="duel_end_pc",
            dtype=Int64,
            description="Number of player duels completed",
        ),
        Field(
            name="duel_end_team",
            dtype=Int64,
            description="Number of team duels completed",
        ),
        Field(
            name="party_battle_end_team",
            dtype=Int64,
            description="Number of party battles completed",
        ),
        # Economy and customization events
        Field(
            name="expand_warehouse",
            dtype=Int64,
            description="Number of warehouse expansions",
        ),
        Field(
            name="change_item_look",
            dtype=Int64,
            description="Number of item appearance changes",
        ),
        Field(
            name="put_main_auction",
            dtype=Int64,
            description="Number of items put on auction",
        ),
        Field(
            name="use_gathering_item",
            dtype=Int64,
            description="Number of gathering items used",
        ),
        # Quest and achievement events
        Field(
            name="complete_quest", dtype=Int64, description="Number of quests completed"
        ),
        Field(
            name="complete_challenge_today",
            dtype=Int64,
            description="Number of daily challenges completed",
        ),
        Field(
            name="complete_challenge_week",
            dtype=Int64,
            description="Number of weekly challenges completed",
        ),
        # Guild events
        Field(name="create_guild", dtype=Int64, description="Number of guilds created"),
        Field(
            name="destroy_guild", dtype=Int64, description="Number of guilds destroyed"
        ),
        Field(
            name="invite_guild",
            dtype=Int64,
            description="Number of guild invitations sent",
        ),
        Field(
            name="join_guild", dtype=Int64, description="Number of times joined a guild"
        ),
        Field(
            name="dismiss_guild",
            dtype=Int64,
            description="Number of times dismissed from guild",
        ),
    ],
    source=player_source,
    online=True,
)
