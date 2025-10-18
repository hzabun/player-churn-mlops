import re
from pathlib import Path

import numpy as np
import pandas as pd

# Feature columns to extract from raw data
feature_columns = [
    "time",
    "logid",
    "session",
    "log_detail_code",
    "actor_account_id",
    "actor_level",
]

# Simple logids (match by logid only)
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

# Tuple conditions: (logid, log_detail_code) - match both columns
tuple_log_ids = [
    (1012, 1),
    (1102, 1),
]

# Log ID to label mapping for feature engineering
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


def validate_time_format(time_str):
    """
    Validate that time string matches the expected format: YYYY-MM-DD HH:MM:SS.mmm

    Args:
        time_str: Time string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$"
    return bool(re.match(pattern, str(time_str)))


def filter_csv(df_column_filtered):
    """
    Filter CSV file by specified column and logid/log_detail_code combinations.

    Args:
        df_column_filtered: Loaded CSV as DataFrame with selected columns

    Returns:
        Filtered DataFrame or None if validation fails

    Raises:
        ValueError: If any required columns are missing from the CSV file
    """
    missing_columns = set(feature_columns) - set(df_column_filtered.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in CSV file: {list(missing_columns)}\n"
            f"Required columns: {feature_columns}\n"
            f"Available columns: {list(df_column_filtered.columns)}"
        )

    mask = df_column_filtered["logid"].isin(simple_log_ids)

    if tuple_log_ids:
        tuple_masks = [
            (df_column_filtered["logid"] == logid)
            & (df_column_filtered["log_detail_code"] == detail_code)
            for logid, detail_code in tuple_log_ids
        ]
        if tuple_masks:
            # Combine all tuple masks with OR
            tuple_mask_combined = tuple_masks[0]
            for tm in tuple_masks[1:]:
                tuple_mask_combined |= tm
            mask |= tuple_mask_combined

    df_result = df_column_filtered.loc[mask].copy()

    # Check 1: Ensure we have non-empty rows
    if df_result.empty:
        return None

    # Check 2: Ensure at least one session value is > 0 (use any() for early exit)
    if not (df_result["session"] > 0).any():
        return None

    # Check 3: Validate and convert time format if needed
    # Just check first row - if it's valid, assume rest are too
    first_time = str(df_result["time"].iloc[0])
    if not validate_time_format(first_time):
        # Try conversion
        try:
            df_result["time"] = (
                pd.to_datetime(df_result["time"], errors="coerce")
                .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                .str[:-3]
            )
            # Check if conversion resulted in NaT
            if df_result["time"].isna().any():
                return None
        except Exception:
            return None

    return df_result


def preprocess_player_logs(df, df_unfiltered=None):
    """
    Preprocess player log data by grouping by session and creating features.

    Args:
        df: DataFrame containing filtered player logs
        df_unfiltered: DataFrame containing all unfiltered logs (for accurate session duration)

    Returns:
        DataFrame with sessions as rows and log counts as features
    """
    df["timestamp"] = pd.to_datetime(
        df["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    df.drop(columns=["time"], inplace=True)

    # Add date column to handle session ID reuse across days
    df["session_date"] = df["timestamp"].dt.date

    # If we have unfiltered data, calculate actual session boundaries
    session_boundaries = None
    if df_unfiltered is not None:
        df_unfiltered["timestamp"] = pd.to_datetime(
            df_unfiltered["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
        )

        # Add date column to handle session ID reuse across days
        df_unfiltered["session_date"] = df_unfiltered["timestamp"].dt.date

        # Filter to valid sessions (session > 0) and group efficiently
        # Group by BOTH session AND date to handle session ID reuse
        valid_sessions = df_unfiltered[df_unfiltered["session"] > 0]
        session_boundaries = valid_sessions.groupby(
            ["session", "session_date"], sort=False
        ).agg({"timestamp": ["min", "max"], "actor_account_id": "first"})

        # Flatten multi-level columns
        session_boundaries.columns = [
            "first_timestamp",
            "last_timestamp",
            "actor_account_id",
        ]
        session_boundaries = session_boundaries.to_dict("index")

    # Handle logid 1012 (DeletePC) which appears with session=0
    session_zero_mask = df["session"] == 0
    deletepc_mask = session_zero_mask & (df["logid"] == 1012)

    has_deletepc = deletepc_mask.any()

    if has_deletepc:
        deletepc_session_zero = df[deletepc_mask].copy()
        df_valid = df[~session_zero_mask].copy()

        if not deletepc_session_zero.empty and not df_valid.empty:
            deletepc_session_zero.sort_values("timestamp", inplace=True)
            df_valid.sort_values("timestamp", inplace=True)

            deletepc_session_zero_reset = deletepc_session_zero.reset_index(drop=True)
            df_valid_sessions = (
                df_valid[["timestamp", "session"]]
                .drop_duplicates("session")
                .sort_values("timestamp")
            )

            # Find next session for each DeletePC event
            matched = pd.merge_asof(
                deletepc_session_zero_reset[["timestamp"]],
                df_valid_sessions,
                on="timestamp",
                direction="forward",
            )
            deletepc_session_zero_reset["session"] = matched["session"]

            # Filter out any that didn't match (no future session)
            deletepc_with_session = deletepc_session_zero_reset[
                deletepc_session_zero_reset["session"].notna()
            ]

            if not deletepc_with_session.empty:
                df_valid = pd.concat(
                    [df_valid, deletepc_with_session], ignore_index=True
                )
    else:
        # No DeletePC to handle, just filter out session=0
        df_valid = df[~session_zero_mask]

    # First, get the latest actor_level per session+date combination
    df_valid_sorted = df_valid.sort_values(["session", "session_date", "timestamp"])
    latest_levels = df_valid_sorted.groupby(["session", "session_date"], sort=False)[
        "actor_level"
    ].last()

    # Filter to only the logids we care about
    logid_to_label = dict(logid_label_mapping)
    df_valid_filtered = df_valid[df_valid["logid"].isin(logid_to_label.keys())]

    # Count occurrences using crosstab via composite index session+date
    if not df_valid_filtered.empty:
        df_valid_filtered["session_composite"] = (
            df_valid_filtered["session"].astype(str)
            + "_"
            + df_valid_filtered["session_date"].astype(str)
        )
        log_counts = pd.crosstab(
            df_valid_filtered["session_composite"], df_valid_filtered["logid"]
        )
        # Rename columns to labels (convert col to int if needed)
        log_counts.columns = [
            logid_to_label.get(int(col), f"logid_{col}") for col in log_counts.columns
        ]
    else:
        log_counts = pd.DataFrame()

    # Get session timestamps from filtered data (group by session+date)
    filtered_session_stats = df_valid.groupby(
        ["session", "session_date"], sort=False
    ).agg({"timestamp": ["min", "max"], "actor_account_id": "first"})
    filtered_session_stats.columns = [
        "first_timestamp_filtered",
        "last_timestamp_filtered",
        "actor_account_id",
    ]

    if session_boundaries is not None:
        boundaries_df = pd.DataFrame.from_dict(session_boundaries, orient="index")
        boundaries_df.index.names = ["session", "session_date"]
        # Merge with filtered stats, preferring boundary data
        result_df = filtered_session_stats.join(
            boundaries_df, how="left", rsuffix="_boundary"
        )

        # Use boundary timestamps where available, else fall back to filtered
        result_df["first_timestamp"] = result_df["first_timestamp"].fillna(
            result_df["first_timestamp_filtered"]
        )
        result_df["last_timestamp"] = result_df["last_timestamp"].fillna(
            result_df["last_timestamp_filtered"]
        )
        result_df["actor_account_id"] = result_df["actor_account_id_boundary"].fillna(
            result_df["actor_account_id"]
        )

        # Clean up temporary columns
        result_df.drop(
            columns=[
                "first_timestamp_filtered",
                "last_timestamp_filtered",
                "actor_account_id_boundary",
            ],
            inplace=True,
        )
    else:
        result_df = filtered_session_stats.copy()
        result_df.columns = ["first_timestamp", "last_timestamp", "actor_account_id"]

    result_df["session_duration_minutes"] = (
        (
            (
                result_df["last_timestamp"] - result_df["first_timestamp"]
            ).dt.total_seconds()
            / 60
        )
        .round()
        .astype(int)
    )

    # Join with latest levels
    result_df = result_df.join(latest_levels, how="left")
    result_df.rename(columns={"actor_level": "actor_level"}, inplace=True)

    # Join with log counts
    if not log_counts.empty:
        # Create composite key in result_df to match log_counts index
        result_df_reset = result_df.reset_index()
        result_df_reset["session_composite"] = (
            result_df_reset["session"].astype(str)
            + "_"
            + result_df_reset["session_date"].astype(str)
        )
        result_df_reset.set_index("session_composite", inplace=True)
        result_df = result_df_reset.join(log_counts, how="left")
        # Drop the composite key column but keep session and session_date
        result_df.reset_index(drop=True, inplace=True)
    else:
        # No log counts, just reset the multi-level index
        result_df.reset_index(inplace=True)

    # Fill missing log counts with 0
    for logid, label in logid_label_mapping:
        if label not in result_df.columns:
            result_df[label] = 0
        else:
            result_df[label] = result_df[label].fillna(0).astype(int)

    result_df.sort_values("last_timestamp", inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    result_df = result_df.drop(columns=["session", "session_date"])

    log_columns = [label for _, label in logid_label_mapping]
    column_order = [
        "first_timestamp",
        "last_timestamp",
        "actor_account_id",
        "session_duration_minutes",
        "actor_level",
    ] + log_columns
    result_df = result_df[column_order]

    return result_df


if __name__ == "__main__":
    raw_data_dir = Path("data/raw")
    output_dir = Path("data/processed")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_dir.exists():
        print(f"Error: Directory not found: {raw_data_dir}")
        print("Please ensure the data/raw directory exists.")
        exit(1)

    csv_files = list(raw_data_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {raw_data_dir}")
        exit(1)

    print("=" * 70)
    print("Player Log Preprocessing Pipeline")
    print("=" * 70)
    print(f"\nFound {len(csv_files)} CSV files to process")
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # ========================================================================
    # SINGLE-PASS PROCESSING: Filter -> Feature Engineering -> Accumulate
    # ========================================================================
    print("=" * 70)
    print("Processing: Filter + Feature Engineering (Single Pass)")
    print("=" * 70)
    print()

    total_original_rows = 0
    total_filtered_rows = 0
    successful_files = 0
    skipped_files = 0
    failed_files = 0
    all_processed_data = []
    errors = []
    total_files = len(csv_files)

    for idx, csv_file in enumerate(csv_files, 1):
        try:
            # Show progress every 100 files or on first/last file
            if idx == 1 or idx == len(csv_files) or idx % 100 == 0:
                print(
                    f"Processing file {idx}/{total_files} ({idx/total_files*100:.1f}%) - "
                    f"Success: {successful_files}, Skipped: {skipped_files}, Failed: {failed_files}",
                    end="\r",
                )

            df_column_filtered = pd.read_csv(
                csv_file,
                usecols=feature_columns,
                dtype={"actor_account_id": str},
                low_memory=False,
            )
            original_rows = len(df_column_filtered)
            total_original_rows += original_rows

            # PHASE 1: Filter and validate
            filtered_df = filter_csv(df_column_filtered)

            if filtered_df is None:
                # File was skipped due to validation failures
                skipped_files += 1
                continue

            # PHASE 2: Feature engineering
            # Pass the unfiltered data to calculate accurate session durations
            processed_df = preprocess_player_logs(filtered_df, df_column_filtered)
            all_processed_data.append(processed_df)

            # Track filtered rows
            filtered_rows = len(processed_df)
            total_filtered_rows += filtered_rows

            successful_files += 1

        except Exception as e:
            errors.append((csv_file.name, str(e)))
            failed_files += 1

    # Clear the progress line
    print(" " * 120, end="\r")

    print()
    print("=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"✓ Successfully processed: {successful_files} files")
    print(f"⊘ Skipped (validation failed): {skipped_files} files")
    print(f"✗ Failed (errors): {failed_files} files")
    print(f"  Total original rows: {total_original_rows:,}")
    if total_original_rows > 0:
        print(
            f"  Total filtered rows: {total_filtered_rows:,} ({total_filtered_rows/total_original_rows*100:.1f}%)"
        )

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for filename, error in errors[:10]:  # Show first 10 errors
            print(f"  - {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # ========================================================================
    # FINAL: Combine and Save Results
    # ========================================================================
    if all_processed_data:
        print()
        print("=" * 70)
        print("Combining all processed data...")
        print("=" * 70)

        combined_df = pd.concat(all_processed_data, ignore_index=True)

        print("Calculating days_since_last_login per player...")

        # Sort by player and first_timestamp (when session started, not ended)
        # This ensures we calculate time since previous session started, not ended
        combined_df.sort_values(["actor_account_id", "first_timestamp"], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        # Group by player and calculate days since last login
        # Use first_timestamp for both previous and current session
        combined_df["prev_session_start"] = combined_df.groupby("actor_account_id")[
            "first_timestamp"
        ].shift(1)

        # Calculate time difference in days (from previous session start to current session start)
        time_diff = (
            combined_df["first_timestamp"] - combined_df["prev_session_start"]
        ).dt.total_seconds() / 86400

        # First session per player gets 0, others get the floored integer
        combined_df["days_since_last_login"] = (
            time_diff.fillna(0).astype(int).astype("Int64")
        )

        # Drop temporary columns
        combined_df.drop(
            columns=["prev_session_start", "first_timestamp"], inplace=True
        )

        # Reorder columns with days_since_last_login in the right position
        cols = combined_df.columns.tolist()
        # Move days_since_last_login after session_duration_minutes
        cols.remove("days_since_last_login")
        idx = cols.index("session_duration_minutes") + 1
        cols.insert(idx, "days_since_last_login")
        combined_df = combined_df[cols]

        # Save combined result
        output_file = output_dir / "all_sessions_processed.csv"
        combined_df.to_csv(output_file, index=False)

        print()
        print("=" * 70)
        print("Final Output")
        print("=" * 70)
        print(f"✓ Combined dataset saved to: {output_file}")
        print(f"  Total sessions: {len(combined_df):,}")
        print(f"  Shape: {combined_df.shape}")
        print()
        print("Column names:")
        for i, col in enumerate(combined_df.columns, 1):
            print(f"  {i:2d}. {col}")

        print()
        print("=" * 70)
        print("Dataset Statistics")
        print("=" * 70)
        print(f"Unique players: {combined_df['actor_account_id'].nunique():,}")
        print(
            f"Average session duration: {combined_df['session_duration_minutes'].mean():.1f} minutes"
        )
        print(
            f"Average days since last login: {combined_df['days_since_last_login'].mean():.2f} days"
        )
        print(f"Total events captured: {combined_df.iloc[:, 5:].sum().sum():,.0f}")

        print()
        print("Sample of combined data (first 5 rows):")
        print(combined_df.head())

        print()
        print("=" * 70)
        print("✓ Pipeline completed successfully!")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("⚠ No data was processed successfully.")
        print("=" * 70)
