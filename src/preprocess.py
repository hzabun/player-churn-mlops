import logging
import os
import re
import sys

import pandas as pd
import s3fs
from dask.base import compute
from dask.delayed import delayed
from dask.distributed import Client
from dask_kubernetes.operator import KubeCluster, make_cluster_spec

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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


def _validate_raw_data(df: pd.DataFrame) -> tuple[bool, str]:
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


def _filter_by_logids(df: pd.DataFrame) -> pd.DataFrame:
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


def _validate_filtered_data(df: pd.DataFrame) -> bool:
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


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp strings and add session_date column.

    Args:
        df: DataFrame with 'time' column

    Returns:
        DataFrame with 'timestamp' and 'session_date' columns added
    """
    df["timestamp"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S.%f")
    df["session_date"] = df["timestamp"].dt.date
    return df


def _get_session_boundaries(df_unfiltered: pd.DataFrame) -> pd.DataFrame:
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


def _handle_deletepc_events(df: pd.DataFrame) -> pd.DataFrame:
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


def _aggregate_session_stats(
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


def _add_event_counts_and_finalize(
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


def _filter_and_process_session(
    df: pd.DataFrame, file_path: str, df_unfiltered: pd.DataFrame | None = None
) -> pd.DataFrame | None:
    """Filter data and process into session-level features.

    Args:
        df: Input DataFrame with player logs (from Parquet file)
        file_path: Path to the file (for logging purposes)
        df_unfiltered: Optional unfiltered DataFrame for accurate session boundaries

    Returns:
        Processed session-level DataFrame or None if validation fails
    """
    # Step 0: Validate raw data
    is_valid, error_msg = _validate_raw_data(df)
    if not is_valid:
        logger.error(f"Validation error for file: {file_path}")
        logger.error(f"Error message: {error_msg}")
        return None

    # Step 1: Filter by logids
    df_filtered = _filter_by_logids(df)

    # Step 2: Validate filtered data
    if not _validate_filtered_data(df_filtered):
        return None

    # Step 3: Parse timestamps
    df_filtered = _parse_timestamps(df_filtered)

    # Step 4: Get session boundaries
    boundaries = None
    if df_unfiltered is not None:
        boundaries = _get_session_boundaries(df_unfiltered)

    # Step 5: Handle DeletePC events
    df_valid = _handle_deletepc_events(df_filtered)

    # Step 6: Aggregate session stats
    stats = _aggregate_session_stats(df_valid, boundaries)

    # Step 7: Add event counts and finalize
    return _add_event_counts_and_finalize(stats, df_valid)


def _aggregate_and_transform_to_players(sessions: pd.DataFrame) -> pd.DataFrame:
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


def _aggregate_to_players(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session data to player level.

    Args:
        sessions: Session-level DataFrame

    Returns:
        Player-level DataFrame with aggregated features
    """
    sessions["last_timestamp"] = pd.to_datetime(sessions["last_timestamp"])
    return _aggregate_and_transform_to_players(sessions)


def _process_file(
    path: str, fs: s3fs.S3FileSystem
) -> tuple[pd.DataFrame | None, str | None]:
    """Process a single Parquet file from S3.

    Args:
        path: S3 path to Parquet file
        fs: S3FileSystem instance for reading files

    Returns:
        Tuple of (processed_dataframe, error_message)
    """
    try:
        logger.debug(f"Reading file: {path}")
        with fs.open(path, "rb") as f:
            df_unfiltered = pd.read_parquet(f, columns=feature_columns)
        logger.debug(f"Successfully read {len(df_unfiltered)} rows from {path}")

        df_unfiltered["actor_account_id"] = df_unfiltered["actor_account_id"].astype(
            str
        )

        result = _filter_and_process_session(df_unfiltered.copy(), path, df_unfiltered)
        if result is None:
            logger.debug(f"File {path} returned None after processing")
        else:
            logger.debug(f"File {path} produced {len(result)} session records")
        return result, None
    except Exception as e:
        logger.error(f"Exception processing {path}: {e}")
        import traceback

        return None, f"{str(e)}\n{traceback.format_exc()}"


def preprocess_all_players(
    raw_data_path: str,
    output_file_path: str,
    cluster_name: str = "preprocessing-cluster",
    namespace: str = "processing",
    n_workers: int = 2,
) -> pd.DataFrame | None:
    """Run the complete preprocessing pipeline with Dask distributed processing.

    Args:
        raw_data_path: S3 path to directory containing raw Parquet files (e.g., 's3://bucket/path/')
        output_file_path: S3 path to save processed output (e.g., 's3://bucket/output/features.parquet')
        cluster_name: Name for the dynamically created Dask cluster (default: 'preprocessing-cluster')
        namespace: Kubernetes namespace to deploy cluster in (default: 'processing')
        n_workers: Number of worker pods to create (default: 2)

    Returns:
        Processed player-level DataFrame with features or None if processing fails
    """

    logger.info("Initializing S3 filesystem...")
    fs = s3fs.S3FileSystem()
    logger.info("S3 filesystem initialized successfully")

    # Ensure raw_dir ends with /
    if not raw_data_path.endswith("/"):
        raw_data_path = raw_data_path + "/"

    logger.info(f"Checking if path exists: {raw_data_path}")
    try:
        if not fs.exists(raw_data_path):
            logger.error(f"Error: {raw_data_path} not found")
            return None
        logger.info(f"Path {raw_data_path} exists")
    except Exception as e:
        logger.error(f"Error accessing {raw_data_path}: {e}")
        logger.exception("Full traceback:")
        return None

    logger.info(f"Listing parquet files in {raw_data_path}")
    try:
        parquet_files = fs.glob(f"{raw_data_path}*.parquet")
        # Add s3:// prefix if not present
        parquet_files = [
            f"s3://{f}" if not f.startswith("s3://") else f for f in parquet_files
        ]
        logger.info(f"Found {len(parquet_files)} files")
    except Exception as e:
        logger.error(f"Error listing files in {raw_data_path}: {e}")
        logger.exception("Full traceback:")
        return None

    if not parquet_files:
        logger.error(f"No Parquet files in {raw_data_path}")
        return None

    logger.info("=" * 70)
    logger.info(f"Found {len(parquet_files)} Parquet files")
    logger.info("=" * 70)

    # Get AWS configuration from environment
    aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    aws_account_id = os.environ.get("AWS_ACCOUNT_ID")
    image_tag = os.environ.get("IMAGE_TAG", "latest")

    if not aws_account_id:
        logger.error("AWS_ACCOUNT_ID environment variable not set")
        return None

    # Get Dask cluster configuration from environment
    worker_memory_request = os.environ.get("DASK_WORKER_MEMORY_REQUEST", "2Gi")
    worker_memory_limit = os.environ.get("DASK_WORKER_MEMORY_LIMIT", "4Gi")
    worker_cpu_request = os.environ.get("DASK_WORKER_CPU_REQUEST", "1000m")
    worker_cpu_limit = os.environ.get("DASK_WORKER_CPU_LIMIT", "2000m")

    scheduler_memory_request = os.environ.get("DASK_SCHEDULER_MEMORY_REQUEST", "1Gi")
    scheduler_memory_limit = os.environ.get("DASK_SCHEDULER_MEMORY_LIMIT", "2Gi")
    scheduler_cpu_request = os.environ.get("DASK_SCHEDULER_CPU_REQUEST", "500m")
    scheduler_cpu_limit = os.environ.get("DASK_SCHEDULER_CPU_LIMIT", "1000m")

    worker_threads = os.environ.get("DASK_WORKER_THREADS", "1")
    worker_memory_limit_gb = os.environ.get("DASK_WORKER_MEMORY_LIMIT_GB", "2GiB")
    worker_death_timeout = os.environ.get("DASK_WORKER_DEATH_TIMEOUT", "60")
    service_account = os.environ.get("DASK_SERVICE_ACCOUNT", "preprocess-sa")

    image_repo = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/player-churn/preprocess:{image_tag}"

    logger.info(
        f"Creating dynamic Dask cluster: {cluster_name} in namespace {namespace}"
    )
    logger.info(f"Workers: {n_workers}, Image: {image_repo}")

    try:
        # Create base cluster spec using make_cluster_spec
        cluster_spec = make_cluster_spec(
            name=cluster_name,
            image=image_repo,
            n_workers=n_workers,
            resources={
                "requests": {
                    "memory": worker_memory_request,
                    "cpu": worker_cpu_request,
                },
                "limits": {"memory": worker_memory_limit, "cpu": worker_cpu_limit},
            },
            env={"AWS_DEFAULT_REGION": aws_region},
        )

        # Configure service account for IRSA (S3 access)
        cluster_spec["spec"]["worker"]["spec"]["serviceAccountName"] = service_account
        cluster_spec["spec"]["scheduler"]["spec"]["serviceAccountName"] = (
            service_account
        )

        # Configure worker command with custom arguments
        worker_container = cluster_spec["spec"]["worker"]["spec"]["containers"][0]
        worker_container["args"] = [
            "dask-worker",
            "--nthreads",
            worker_threads,
            "--memory-limit",
            worker_memory_limit_gb,
            "--death-timeout",
            worker_death_timeout,
        ]

        # Configure scheduler resources
        scheduler_container = cluster_spec["spec"]["scheduler"]["spec"]["containers"][0]
        scheduler_container["resources"] = {
            "requests": {
                "memory": scheduler_memory_request,
                "cpu": scheduler_cpu_request,
            },
            "limits": {
                "memory": scheduler_memory_limit,
                "cpu": scheduler_cpu_limit,
            },
        }

        # Ensure metadata includes namespace
        cluster_spec["metadata"]["namespace"] = namespace

        # Create a new Dask cluster dynamically using the operator
        cluster = KubeCluster(custom_cluster_spec=cluster_spec, namespace=namespace)

        client = Client(cluster)
        logger.info("Successfully created Dask cluster")
        logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    except Exception as e:
        logger.error(f"Failed to create Dask cluster '{cluster_name}': {e}")
        logger.exception("Full traceback:")
        return None

    try:
        logger.info("Processing files in parallel...")
        delayed_results = [delayed(_process_file)(f, fs) for f in parquet_files]

        results = compute(*delayed_results)

        # Process results
        all_sessions = []
        success = skipped = failed = 0
        for i, (result, error) in enumerate(results):
            if error:
                failed += 1
                logger.error(f"Failed to process {parquet_files[i]}: {error}")
            elif result is None:
                skipped += 1
                logger.warning(
                    f"Skipped {parquet_files[i]} (validation failed or no valid data)"
                )
            else:
                all_sessions.append(result)
                success += 1

        logger.info(f"Processed: {success} | Skipped: {skipped} | Failed: {failed}")

        if not all_sessions:
            logger.error("No data processed")
            return None

        logger.info("Aggregating to player level...")
        combined = pd.concat(all_sessions, ignore_index=True)
        players = _aggregate_to_players(combined)

        # Write features to S3
        with fs.open(output_file_path, "wb") as f:
            players.to_parquet(f, index=False, engine="pyarrow", compression="snappy")

        logger.info("=" * 70)
        logger.info(f"Saved: {output_file_path}")
        logger.info(f"  Players: {len(players):,} | Features: {players.shape[1]}")
        logger.info(f"Sample:\n{players.head(3)}")
        logger.info("=" * 70)

        return players

    finally:
        # Close cluster - this will delete all pods and clean up resources
        logger.info("Cleaning up Dask cluster...")
        client.close()
        cluster.close()


if __name__ == "__main__":
    raw_data_path = os.environ.get(
        "RAW_DATA_PATH", "s3://placeholder-bucket/raw-parquet/"
    )
    output_file_path = os.environ.get(
        "OUTPUT_FILE_PATH", "s3://placeholder-bucket/processed/player-features.parquet"
    )
    cluster_name = os.environ.get("DASK_CLUSTER_NAME", "preprocessing-cluster")
    namespace = os.environ.get("DASK_NAMESPACE", "processing")
    n_workers = int(os.environ.get("DASK_N_WORKERS", "2"))

    preprocess_all_players(
        raw_data_path=raw_data_path,
        output_file_path=output_file_path,
        cluster_name=cluster_name,
        namespace=namespace,
        n_workers=n_workers,
    )
