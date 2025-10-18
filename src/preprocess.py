import re
from pathlib import Path

import pandas as pd

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


def filter_and_process_session(df, df_unfiltered):
    """Filter CSV and process into session-level features."""
    # Filter rows
    mask = df["logid"].isin(simple_log_ids)
    for logid, detail in tuple_log_ids:
        mask |= (df["logid"] == logid) & (df["log_detail_code"] == detail)
    df = df.loc[mask].copy()

    if df.empty or not (df["session"] > 0).any():
        return None
    if not re.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$", str(df["time"].iloc[0])
    ):
        return None

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S.%f")
    df["session_date"] = df["timestamp"].dt.date

    # Get session boundaries from unfiltered data
    if df_unfiltered is not None:
        df_unfiltered["timestamp"] = pd.to_datetime(
            df_unfiltered["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
        )
        df_unfiltered["session_date"] = df_unfiltered["timestamp"].dt.date
        boundaries = (
            df_unfiltered[df_unfiltered["session"] > 0]
            .groupby(["session", "session_date"])
            .agg(
                first_ts=("timestamp", "min"),
                last_ts=("timestamp", "max"),
                actor_id=("actor_account_id", "first"),
            )
        )
    else:
        boundaries = None

    # Handle DeletePC events (session=0)
    deletepc_mask = (df["session"] == 0) & (df["logid"] == 1012)
    df_valid = df[~deletepc_mask].copy()
    if deletepc_mask.any() and not df_valid.empty:
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
        df_valid = pd.concat(
            [df_valid, deletepc[deletepc["session"].notna()]], ignore_index=True
        )

    # Aggregate session stats
    stats = df_valid.groupby(["session", "session_date"]).agg(
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
        actor_id=("actor_account_id", "first"),
        level=("actor_level", "last"),
    )

    # Merge with boundaries
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

    # Count events
    logid_map = dict(logid_label_mapping)
    df_events = df_valid[df_valid["logid"].isin(logid_map.keys())].copy()
    df_events["key"] = (
        df_events["session"].astype(str) + "_" + df_events["session_date"].astype(str)
    )
    if not df_events.empty:
        counts = pd.crosstab(df_events["key"], df_events["logid"])
        counts.columns = [logid_map.get(int(c), f"logid_{c}") for c in counts.columns]
        stats = stats.reset_index()
        stats["key"] = (
            stats["session"].astype(str) + "_" + stats["session_date"].astype(str)
        )
        stats = stats.set_index("key").join(counts, how="left").reset_index(drop=True)

    # Fill missing events
    for _, label in logid_label_mapping:
        if label not in stats.columns:
            stats[label] = 0
        else:
            stats[label] = stats[label].fillna(0).astype(int)

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

    event_cols = [l for _, l in logid_label_mapping]
    return (
        stats[
            [
                "first_timestamp",
                "last_timestamp",
                "actor_account_id",
                "session_duration_minutes",
                "actor_level",
            ]
            + event_cols
        ]
        .sort_values("last_timestamp")
        .reset_index(drop=True)
    )


def aggregate_to_players(sessions) -> pd.DataFrame:
    """Aggregate session data to player level."""
    sessions["last_timestamp"] = pd.to_datetime(sessions["last_timestamp"])
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

    agg_dict = {
        "last_timestamp": ["min", "max", "count"],
        "session_duration_minutes": ["mean", "sum", "std", "min", "max"],
    }
    for col in event_cols:
        agg_dict[col] = ["sum"]

    players = sessions.groupby("actor_account_id", as_index=False).agg(agg_dict)
    players.columns = ["_".join(c) if c[1] else c[0] for c in players.columns.values]
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
            new_name = c[:-4]  # Remove _sum
            # Special case: rename pc_level_up to level_ups_across_all_characters
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

    # Round session duration statistics to 2 decimal places
    players["avg_session_duration_minutes"] = players[
        "avg_session_duration_minutes"
    ].round(2)
    players["std_session_duration_minutes"] = (
        players["std_session_duration_minutes"].fillna(0).round(2)
    )

    # Drop unwanted columns
    players.drop(columns=["first_session_timestamp"], inplace=True, errors="ignore")

    # Reorder
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
    events = [c for c in players.columns if c not in base]
    return (
        players[base + events]
        .sort_values("last_session_timestamp", ascending=False)
        .reset_index(drop=True)
    )


def process_file(path):
    """Process a single Parquet file."""
    try:
        df = pd.read_parquet(path, columns=feature_columns)
        df["actor_account_id"] = df["actor_account_id"].astype(str)
        orig_rows = len(df)
        result = filter_and_process_session(df, df)
        return result, orig_rows, None
    except Exception as e:
        return None, 0, str(e)


if __name__ == "__main__":
    raw_dir = Path("data/raw_parquet")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Error: {raw_dir} not found")
        exit(1)

    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files in {raw_dir}")
        exit(1)

    print("=" * 70)
    print("Player Churn Preprocessing Pipeline")
    print("=" * 70)
    print(f"Found {len(parquet_files)} Parquet files\n")

    all_sessions = []
    success = skipped = failed = 0
    for i, f in enumerate(parquet_files, 1):
        if i == 1 or i == len(parquet_files) or i % 100 == 0:
            print(
                f"Processing {i}/{len(parquet_files)} ({i/len(parquet_files)*100:.1f}%)...",
                end="\r",
            )
        result, _, error = process_file(f)
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
        exit(1)

    print("Aggregating to player level...")
    combined = pd.concat(all_sessions, ignore_index=True)
    players = aggregate_to_players(combined)

    output = output_dir / "player_features.parquet"
    players.to_parquet(output, index=False, engine="pyarrow", compression="snappy")

    print("\n" + "=" * 70)
    print(f"✓ Saved: {output}")
    print(f"  Players: {len(players):,} | Features: {players.shape[1]}")
    print(f"\nSample:\n{players.head(3)}")
    print("=" * 70)
