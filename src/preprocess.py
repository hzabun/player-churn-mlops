import re
from pathlib import Path

import pandas as pd
from dask.base import compute
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster

feature_columns = [
    "time",
    "logid",
    "session",
    "log_detail_code",
    "actor_account_id",
    "actor_level",
]
simple_log_ids = {
    1013,
    1101,
    1103,
    1202,
    1404,
    1406,
    1422,
    2112,
    2141,
    2301,
    2405,
    5004,
    5011,
    5015,
    6001,
    6002,
    6004,
    6005,
    6009,
}
tuple_log_ids = [(1012, 1), (1102, 1)]
logid_label_mapping = [
    (1012, "delete_pc"),
    (1013, "pc_level_up"),
    (1101, "invite_party"),
    (1103, "refuse_party"),
    (1102, "join_party"),
    (1202, "die"),
    (1404, "duel_end_pc"),
    (1406, "duel_end_team"),
    (1422, "party_battle_end_team"),
    (2112, "expand_warehouse"),
    (2141, "change_item_look"),
    (2301, "put_main_auction"),
    (2405, "use_gathering_item"),
    (5004, "complete_quest"),
    (5011, "complete_challenge_today"),
    (5015, "complete_challenge_week"),
    (6001, "create_guild"),
    (6002, "destroy_guild"),
    (6004, "invite_guild"),
    (6005, "join_guild"),
    (6009, "dismiss_guild"),
]


def validate_raw_data(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate raw data before processing.

    Args:
        df: Raw DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df.empty:
        return False, "DataFrame is empty"

    # Check required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # Check for nulls in critical columns
    critical_cols = ["time", "session", "actor_account_id", "logid"]
    null_cols = [col for col in critical_cols if df[col].isnull().any()]
    if null_cols:
        return False, f"Null values found in critical columns: {null_cols}"

    # Check data types
    if not pd.api.types.is_numeric_dtype(df["session"]):
        return False, "Column 'session' must be numeric"
    if not pd.api.types.is_numeric_dtype(df["logid"]):
        return False, "Column 'logid' must be numeric"
    if not pd.api.types.is_numeric_dtype(df["actor_level"]):
        return False, "Column 'actor_level' must be numeric"

    # Check timestamp format (sample first row)
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$"
    if not re.match(pattern, str(df["time"].iloc[0])):
        return False, f"Invalid timestamp format: {df['time'].iloc[0]}"

    # Check value ranges
    if (df["session"] < 0).any():
        return False, "Column 'session' contains negative values"
    if (df["actor_level"] < 0).any():
        return False, "Column 'actor_level' contains negative values"
    if (df["actor_level"] > 55).any():
        return (
            False,
            "Column 'actor_level' contains values higher than the maximum in the game (>55)",
        )

    return True, ""


def filter_by_logids(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame by configured logids and log_detail_codes.

    Args:
        df: Input DataFrame with 'logid' and 'log_detail_code' columns

    Returns:
        Filtered DataFrame
    """
    mask = df["logid"].isin(simple_log_ids)
    for logid, detail in tuple_log_ids:
        mask |= (df["logid"] == logid) & (df["log_detail_code"] == detail)
    return df.loc[mask].copy()


def validate_filtered_data(df: pd.DataFrame) -> bool:
    """Validate that filtered data meets processing requirements.

    Args:
        df: Filtered DataFrame

    Returns:
        True if data is valid, False otherwise
    """
    if df.empty:
        return False
    if not (df["session"] > 0).any():
        return False

    return True


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp strings and add session_date column.

    Args:
        df: DataFrame with 'time' column

    Returns:
        DataFrame with 'timestamp' and 'session_date' columns added
    """
    df["timestamp"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S.%f")
    df["session_date"] = df["timestamp"].dt.date
    return df


def get_session_boundaries(df_unfiltered: pd.DataFrame) -> pd.DataFrame:
    """Extract session boundaries from unfiltered data.

    Args:
        df_unfiltered: Unfiltered DataFrame with all session data

    Returns:
        DataFrame with session boundaries indexed by (session, session_date)
    """
    # Parse timestamps
    df_unfiltered["timestamp"] = pd.to_datetime(
        df_unfiltered["time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    df_unfiltered["session_date"] = df_unfiltered["timestamp"].dt.date

    # Extract boundaries
    boundaries = (
        df_unfiltered[df_unfiltered["session"] > 0]
        .groupby(["session", "session_date"])
        .agg(
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
            actor_id=("actor_account_id", "first"),
        )
    )
    return boundaries


def handle_deletepc_events(df: pd.DataFrame) -> pd.DataFrame:
    """Handle DeletePC events that have session=0 by matching to next session.

    Args:
        df: DataFrame with session data

    Returns:
        DataFrame with DeletePC events assigned to appropriate sessions
    """
    deletepc_mask = (df["session"] == 0) & (df["logid"] == 1012)

    if not deletepc_mask.any():
        return df[df["session"] > 0].copy()

    df_valid = df[~deletepc_mask].copy()

    if df_valid.empty:
        return df_valid

    deletepc = df[deletepc_mask].copy().sort_values("timestamp")
    sessions = (
        df_valid[["timestamp", "session"]]
        .drop_duplicates("session")
        .sort_values("timestamp")
    )
    matched = pd.merge_asof(
        deletepc[["timestamp"]], sessions, on="timestamp", direction="forward"
    )
    deletepc["session"] = matched["session"]

    return pd.concat(
        [df_valid, deletepc[deletepc["session"].notna()]], ignore_index=True
    )


def aggregate_session_stats(
    df: pd.DataFrame, boundaries: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Aggregate data to session level with timestamps and actor info.

    Args:
        df: DataFrame with session data
        boundaries: Optional session boundaries from unfiltered data

    Returns:
        DataFrame with session-level statistics
    """
    stats = df.groupby(["session", "session_date"]).agg(
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
        actor_id=("actor_account_id", "first"),
        level=("actor_level", "last"),
    )

    # Merge with boundaries if available
    if boundaries is not None:
        stats = stats.join(boundaries, rsuffix="_b", how="left")
        stats["first_ts"] = stats["first_ts_b"].fillna(stats["first_ts"])
        stats["last_ts"] = stats["last_ts_b"].fillna(stats["last_ts"])
        stats["actor_id"] = stats["actor_id_b"].fillna(stats["actor_id"])
        stats.drop(columns=["first_ts_b", "last_ts_b", "actor_id_b"], inplace=True)

    stats["duration_min"] = (
        ((stats["last_ts"] - stats["first_ts"]).dt.total_seconds() / 60)
        .round()
        .astype(int)
    )

    return stats


def add_event_counts_and_finalize(
    stats: pd.DataFrame, df_valid: pd.DataFrame
) -> pd.DataFrame:
    """Count events per session, add to stats, and finalize columns.

    Args:
        stats: Session statistics DataFrame
        df_valid: DataFrame with valid session data for counting events

    Returns:
        Finalized DataFrame with event counts and proper column names
    """
    # Count events per session
    logid_map = dict(logid_label_mapping)
    df_events = df_valid[df_valid["logid"].isin(logid_map.keys())]

    if not df_events.empty:
        df_events["key"] = (
            df_events["session"].astype(str)
            + "_"
            + df_events["session_date"].astype(str)
        )
        event_counts = pd.crosstab(df_events["key"], df_events["logid"])
        event_counts.columns = [
            logid_map.get(int(c), f"logid_{c}") for c in event_counts.columns
        ]

        # Add event counts to stats
        stats = stats.reset_index()
        stats["key"] = (
            stats["session"].astype(str) + "_" + stats["session_date"].astype(str)
        )
        stats = (
            stats.set_index("key").join(event_counts, how="left").reset_index(drop=True)
        )

    # Fill missing event columns
    for _, label in logid_label_mapping:
        if label not in stats.columns:
            stats[label] = 0
        else:
            stats[label] = stats[label].fillna(0).astype(int)

    # Rename and reorder columns
    stats.drop(columns=["session", "session_date"], inplace=True, errors="ignore")
    stats.rename(
        columns={
            "first_ts": "first_timestamp",
            "last_ts": "last_timestamp",
            "actor_id": "actor_account_id",
            "level": "actor_level",
            "duration_min": "session_duration_minutes",
        },
        inplace=True,
    )

    event_cols = [log_label for _, log_label in logid_label_mapping]
    column_order = [
        "first_timestamp",
        "last_timestamp",
        "actor_account_id",
        "session_duration_minutes",
        "actor_level",
    ] + event_cols

    return stats[column_order].sort_values("last_timestamp").reset_index(drop=True)


def filter_and_process_session(
    df: pd.DataFrame, file_path: Path, df_unfiltered: pd.DataFrame | None = None
) -> pd.DataFrame | None:
    """Filter data and process into session-level features.

    Args:
        df: Input DataFrame with player logs (from Parquet file)
        df_unfiltered: Optional unfiltered DataFrame for accurate session boundaries

    Returns:
        Processed session-level DataFrame or None if validation fails
    """
    # Step 0: Validate raw data
    is_valid, error_msg = validate_raw_data(df)
    if not is_valid:
        print(f"Validation error for file: {file_path}\n Error message: {error_msg}")
        return None

    # Step 1: Filter by logids
    df_filtered = filter_by_logids(df)

    # Step 2: Validate filtered data
    if not validate_filtered_data(df_filtered):
        return None

    # Step 3: Parse timestamps
    df_filtered = parse_timestamps(df_filtered)

    # Step 4: Get session boundaries
    boundaries = None
    if df_unfiltered is not None:
        boundaries = get_session_boundaries(df_unfiltered)

    # Step 5: Handle DeletePC events
    df_valid = handle_deletepc_events(df_filtered)

    # Step 6: Aggregate session stats
    stats = aggregate_session_stats(df_valid, boundaries)

    # Step 7: Add event counts and finalize
    return add_event_counts_and_finalize(stats, df_valid)


def aggregate_and_transform_to_players(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sessions to player level and calculate derived features.

    Args:
        sessions: Session-level DataFrame

    Returns:
        Player-level DataFrame with aggregated and derived features
    """
    # Identify event columns
    event_cols = [
        c
        for c in sessions.columns
        if c
        not in [
            "first_timestamp",
            "last_timestamp",
            "actor_account_id",
            "session_duration_minutes",
            "actor_level",
        ]
    ]

    # Aggregate per player
    agg_dict = {
        "last_timestamp": ["min", "max", "count"],
        "session_duration_minutes": ["mean", "sum", "std", "min", "max"],
    }
    for col in event_cols:
        agg_dict[col] = ["sum"]

    players = sessions.groupby("actor_account_id", as_index=False).agg(agg_dict)  # type: ignore[call-overload]
    players.columns = ["_".join(c) if c[1] else c[0] for c in players.columns.values]

    # Rename columns
    players.rename(
        columns={
            "last_timestamp_min": "first_session_timestamp",
            "last_timestamp_max": "last_session_timestamp",
            "last_timestamp_count": "total_sessions",
            "session_duration_minutes_mean": "avg_session_duration_minutes",
            "session_duration_minutes_sum": "total_playtime_minutes",
            "session_duration_minutes_std": "std_session_duration_minutes",
            "session_duration_minutes_min": "min_session_duration_minutes",
            "session_duration_minutes_max": "max_session_duration_minutes",
        },
        inplace=True,
    )

    # Remove _sum suffix from events
    rename_map = {}
    for c in players.columns:
        if c.endswith("_sum"):
            new_name = c[:-4]
            if new_name == "pc_level_up":
                rename_map[c] = "level_ups_across_all_characters"
            else:
                rename_map[c] = new_name
    players.rename(columns=rename_map, inplace=True)

    # Calculate derived features
    players["account_lifespan_days"] = (
        (
            players["last_session_timestamp"] - players["first_session_timestamp"]
        ).dt.total_seconds()
        / 86400
    ).round(2)

    players["average_sessions_per_day"] = (
        players["total_sessions"] / players["account_lifespan_days"].replace(0, 1)
    ).round(2)

    # Round statistics
    players["avg_session_duration_minutes"] = players[
        "avg_session_duration_minutes"
    ].round(2)
    players["std_session_duration_minutes"] = (
        players["std_session_duration_minutes"].fillna(0).round(2)
    )

    # Reorder columns
    base = [
        "actor_account_id",
        "last_session_timestamp",
        "total_sessions",
        "account_lifespan_days",
        "average_sessions_per_day",
        "total_playtime_minutes",
        "avg_session_duration_minutes",
        "std_session_duration_minutes",
        "min_session_duration_minutes",
        "max_session_duration_minutes",
    ]
    events = [
        c for c in players.columns if c not in base and c != "first_session_timestamp"
    ]

    # Drop first_session_timestamp
    if "first_session_timestamp" in players.columns:
        players = players.drop(columns=["first_session_timestamp"])

    return (
        players[base + events]
        .sort_values("last_session_timestamp", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_to_players(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session data to player level.

    Args:
        sessions: Session-level DataFrame

    Returns:
        Player-level DataFrame with aggregated features
    """
    sessions["last_timestamp"] = pd.to_datetime(sessions["last_timestamp"])
    return aggregate_and_transform_to_players(sessions)


def process_file(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Process a single Parquet file.

    Args:
        path: Path to Parquet file

    Returns:
        Tuple of (processed_dataframe, error_message)
    """
    try:
        # Read unfiltered data for accurate session boundaries
        df_unfiltered = pd.read_parquet(path, columns=feature_columns)
        df_unfiltered["actor_account_id"] = df_unfiltered["actor_account_id"].astype(
            str
        )

        # Process with both filtered and unfiltered data
        result = filter_and_process_session(df_unfiltered.copy(), path, df_unfiltered)
        return result, None
    except Exception as e:
        return None, str(e)


def preprocess_all_players(
    raw_dir: Path,
    output_dir: Path,
    output_filename: str,
    label_file_path: Path,
    n_workers: int = 4,
) -> pd.DataFrame | None:
    """Run the complete preprocessing pipeline with Dask distributed processing.

    Args:
        raw_dir: Directory containing raw Parquet files
        output_dir: Directory to save processed output
        output_filename: Name of output file
        label_file_path: Path to the label CSV file
        n_workers: Number of Dask workers for parallel processing

    Returns:
        Processed player-level DataFrame or None if processing fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Error: {raw_dir} not found")
        return None

    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files in {raw_dir}")
        return None

    print("=" * 70)
    print(f"Using Dask with {n_workers} workers for parallel processing")
    print("=" * 70)
    print(f"Found {len(parquet_files)} Parquet files\n")

    print("Setting up Dask cluster...")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=2,
        memory_limit="2GB",
        silence_logs=True,
    )
    client = Client(cluster)
    print(f"✓ Dask dashboard available at: {client.dashboard_link}\n")

    try:
        print("Processing files in parallel...")
        delayed_results = [delayed(process_file)(f) for f in parquet_files]

        results = compute(*delayed_results)

        # Process results
        all_sessions = []
        success = skipped = failed = 0
        for result, error in results:
            if error:
                failed += 1
            elif result is None:
                skipped += 1
            else:
                all_sessions.append(result)
                success += 1

        print(f"\n✓ Processed: {success} | ⊘ Skipped: {skipped} | ✗ Failed: {failed}\n")

        if not all_sessions:
            print("No data processed")
            return None

        print("Aggregating to player level...")
        combined = pd.concat(all_sessions, ignore_index=True)
        players = aggregate_to_players(combined)

        # Read and join labels
        players = _join_labels(players, label_file_path)
        if players is None:
            return None

        # Save output
        output = output_dir / output_filename
        players.to_parquet(output, index=False, engine="pyarrow", compression="snappy")

        print("\n" + "=" * 70)
        print(f"✓ Saved: {output}")
        print(f"  Players: {len(players):,} | Features: {players.shape[1]}")
        print(f"\nSample:\n{players.head(3)}")
        print("=" * 70)

        return players

    finally:
        # Always close the client and cluster
        client.close()
        cluster.close()


def _join_labels(players: pd.DataFrame, label_file_path: Path) -> pd.DataFrame | None:
    """Join player features with labels.

    Args:
        players: Player-level features DataFrame
        label_file_path: Path to the label CSV file

    Returns:
        DataFrame with joined labels or None if joining fails
    """
    try:
        print(f"Reading labels from {label_file_path}...")
        labels = pd.read_csv(label_file_path)
        labels["actor_account_id"] = labels["actor_account_id"].astype(str)

        # Join labels with player features (inner join to keep only labeled players)
        players_before = len(players)
        players = players.merge(labels, on="actor_account_id", how="inner")
        players_after = len(players)

        # Validate no null labels
        null_labels = players["churn_yn"].isna().sum()
        if null_labels > 0:
            print(f"ERROR: Found {null_labels} players with null labels after merge!")
            return None

        print(
            f"✓ Joined labels: {players_after:,} players (dropped {players_before - players_after:,} unlabeled)"
        )
        return players
    except Exception:
        print(f"Error joining label file at {label_file_path}!")
        return None
