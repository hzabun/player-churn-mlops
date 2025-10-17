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
simple_log_ids = [
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
]

# Tuple conditions: (logid, log_detail_code) - match both columns
tuple_log_ids = [
    (1012, 1),
    (1102, 1),
]

# Log ID to label mapping for feature engineering
logid_label_mapping = [
    (1012, "DeletePC"),
    (1013, "PcLevelUp"),
    (1101, "InviteParty"),
    (1103, "RefuseParty"),
    (1102, "JoinParty"),
    (1202, "Die"),
    (1404, "DuelEnd(PC)"),
    (1406, "DuelEnd(Team)"),
    (1422, "PartyBattleEnd(Team)"),
    (2112, "ExpandWarehouse"),
    (2141, "ChangeItemLook"),
    (2301, "PutMainAuction"),
    (2405, "UseGatheringItem"),
    (5004, "CompleteQuest"),
    (5011, "CompleteChallengeToday"),
    (5015, "CompleteChallengeWeek"),
    (6001, "CreateGuild"),
    (6002, "DestoryGuild"),
    (6004, "InviteGuild"),
    (6005, "JoinGuild"),
    (6009, "DissmissGuild"),
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


def convert_time_format(df):
    """
    Convert time column to the expected format: YYYY-MM-DD HH:MM:SS.mmm

    Args:
        df: DataFrame with 'time' column

    Returns:
        DataFrame with converted time format
    """
    try:
        # Try to parse the time column as datetime
        df["time"] = pd.to_datetime(df["time"])
        # Format to the expected format with milliseconds
        df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
        return df
    except Exception as e:
        print(f"Error converting time format: {e}")
        return df


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
    # Read only the required columns to avoid dtype warnings and improve performance
    # First, check if all columns exist by reading just the header
    missing_columns = [
        col for col in feature_columns if col not in df_column_filtered.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in CSV file: {missing_columns}\n"
            f"Required columns: {feature_columns}\n"
            f"Available columns: {list(df_column_filtered.columns)}"
        )

    # Create filter mask
    mask = df_column_filtered["logid"].isin(simple_log_ids)
    for logid, detail_code in tuple_log_ids:
        mask |= (df_column_filtered["logid"] == logid) & (
            df_column_filtered["log_detail_code"] == detail_code
        )

    # Apply the mask
    df_result = df_column_filtered[mask].copy()

    # Validation checks
    # Check 1: Ensure we have non-empty rows
    if len(df_result) == 0:
        return None

    # Check 2: Ensure at least one session value is > 0
    if (df_result["session"] <= 0).all():
        return None

    # Check 3: Validate and convert time format if needed
    # Sample a few rows to check time format (checking all can be slow)
    sample_size = min(100, len(df_result))
    time_sample = df_result["time"].sample(n=sample_size, random_state=42)

    invalid_times = [t for t in time_sample if not validate_time_format(t)]
    if invalid_times:
        df_result = convert_time_format(df_result)

        # Verify conversion was successful
        time_sample_after = df_result["time"].sample(
            n=min(10, len(df_result)), random_state=42
        )
        still_invalid = [t for t in time_sample_after if not validate_time_format(t)]
        if still_invalid:
            return None

    return df_result


def preprocess_player_logs(df):
    """
    Preprocess player log data by grouping by session and creating features.

    Args:
        df: DataFrame containing filtered player logs

    Returns:
        DataFrame with sessions as rows and log counts as features
    """
    # Parse timestamp column
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])  # Remove original 'time' column after renaming

    # Handle logid 1012 (DeletePC) which appears with session=0
    # These need to be assigned to the next available session based on timestamp
    session_zero_mask = df["session"] == 0
    deletepc_session_zero = df[session_zero_mask & (df["logid"] == 1012)].copy()

    # Get valid sessions (non-zero)
    df_valid = df[~session_zero_mask].copy()

    # For each DeletePC with session=0, find the next session based on timestamp
    if not deletepc_session_zero.empty and not df_valid.empty:
        # Sort both by timestamp
        deletepc_session_zero = deletepc_session_zero.sort_values("timestamp")
        df_valid = df_valid.sort_values("timestamp")

        # For each DeletePC event, find which session it should belong to
        for idx, deletepc_row in deletepc_session_zero.iterrows():
            deletepc_timestamp = deletepc_row["timestamp"]

            # Find the next session after this timestamp
            future_sessions = df_valid[df_valid["timestamp"] >= deletepc_timestamp]

            if not future_sessions.empty:
                # Assign to the first session that occurs at or after this timestamp
                next_session = future_sessions.iloc[0]["session"]

                # Create a new row for this DeletePC event with the proper session
                new_row = deletepc_row.copy()
                new_row["session"] = next_session
                df_valid = pd.concat(
                    [df_valid, pd.DataFrame([new_row])], ignore_index=True
                )

    # Group by session
    grouped = df_valid.groupby("session")

    # Initialize result dictionary
    result_data = []

    for session_id, session_df in grouped:
        # Sort session_df by timestamp to ensure we get the correct timestamps
        session_df_sorted = session_df.sort_values("timestamp")

        first_timestamp = session_df_sorted["timestamp"].iloc[0]
        last_timestamp = session_df_sorted["timestamp"].iloc[-1]

        # Calculate session duration in minutes (rounded to nearest integer for accuracy)
        session_duration_seconds = (last_timestamp - first_timestamp).total_seconds()
        session_duration_minutes = int(round(session_duration_seconds / 60))

        session_record = {
            "session": session_id,
            "actor_account_id": session_df_sorted["actor_account_id"].iloc[0],
            "last_timestamp": last_timestamp,
            "first_timestamp": first_timestamp,
            "session_duration_minutes": session_duration_minutes,
            "actor_level": session_df_sorted["actor_level"].iloc[
                -1
            ],  # Latest level in session
        }

        # Count occurrences of each logid
        logid_counts = session_df_sorted["logid"].value_counts()

        # Create columns for each log type
        for logid, label in logid_label_mapping:
            count = logid_counts.get(logid, 0)
            session_record[label] = count

        result_data.append(session_record)

    # Create result dataframe
    result_df = pd.DataFrame(result_data)

    # Sort by last_timestamp to calculate days_since_last_login
    result_df = result_df.sort_values("last_timestamp").reset_index(drop=True)

    # Calculate days_since_last_login (time between end of previous session and start of current session)
    days_since_last_login = []
    for i in range(len(result_df)):
        if i == 0:
            # First session has no previous session, use 0
            days_since_last_login.append(0)
        else:
            # Calculate time between previous session's last_timestamp and current session's first_timestamp
            prev_session_last_timestamp = result_df.iloc[i - 1]["last_timestamp"]
            curr_session_first_timestamp = result_df.iloc[i]["first_timestamp"]
            time_diff_days = (
                pd.to_datetime(curr_session_first_timestamp)
                - pd.to_datetime(prev_session_last_timestamp)
            ).total_seconds() / 86400  # Convert to days
            # Convert to integer days (floor)
            days_since_last_login.append(int(time_diff_days))

    result_df["days_since_last_login"] = days_since_last_login

    # Convert days_since_last_login to proper integer type (Int64 allows NaN for first session)
    result_df["days_since_last_login"] = result_df["days_since_last_login"].astype(
        "Int64"
    )

    # Drop the session column and first_timestamp
    result_df = result_df.drop(columns=["session", "first_timestamp"])

    # Reorder columns: last_timestamp first, then actor_account_id, session_duration_minutes, days_since_last_login, actor_level, then all log columns
    log_columns = [label for _, label in logid_label_mapping]
    column_order = [
        "last_timestamp",
        "actor_account_id",
        "session_duration_minutes",
        "days_since_last_login",
        "actor_level",
    ] + log_columns
    result_df = result_df[column_order]

    return result_df


if __name__ == "__main__":
    # Setup paths
    raw_data_dir = Path("data/raw")
    output_dir = Path("data/processed")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if raw data directory exists
    if not raw_data_dir.exists():
        print(f"Error: Directory not found: {raw_data_dir}")
        print("Please ensure the data/raw directory exists.")
        exit(1)

    # Get all CSV files in the raw data directory
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

    # Statistics tracking
    total_original_rows = 0
    total_filtered_rows = 0
    successful_files = 0
    skipped_files = 0
    failed_files = 0
    all_processed_data = []
    errors = []
    total_files = len(csv_files)

    # Process each file through both phases
    for idx, csv_file in enumerate(csv_files, 1):
        try:
            # Show progress every 100 files or on first/last file
            if idx == 1 or idx == len(csv_files) or idx % 100 == 0:
                print(
                    f"Processing file {idx}/{total_files} ({idx/total_files*100:.1f}%) - "
                    f"Success: {successful_files}, Skipped: {skipped_files}, Failed: {failed_files}",
                    end="\r",
                )

            # Get original row count
            df_column_filtered = pd.read_csv(csv_file, usecols=feature_columns)
            original_rows = len(df_column_filtered)
            total_original_rows += original_rows

            # PHASE 1: Filter and validate
            filtered_df = filter_csv(df_column_filtered)

            if filtered_df is None:
                # File was skipped due to validation failures
                skipped_files += 1
                continue

            # PHASE 2: Feature engineering
            processed_df = preprocess_player_logs(filtered_df)
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

    # Print processing summary
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

    # Show errors if any
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

        # Show some statistics
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
