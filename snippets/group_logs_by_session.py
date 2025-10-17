"""
Player Log Preprocessing Script

This script preprocesses player game log data by grouping events by session and creating features.

Key preprocessing steps:
1. Groups all log entries by session ID
2. Creates a column for each log type (based on simple_log_ids)
3. Counts occurrences of each log type per session
4. Keeps the latest actor_level value for each session
5. Handles special case: logid 1012 (DeletePC) with session=0
   - These events are added to the first valid session for the player

Output format:
- One row per session
- Columns: session, actor_account_id, actor_level, [log type counts]
"""

from pathlib import Path

import numpy as np
import pandas as pd

simple_log_ids = [
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


def preprocess_player_logs(csv_path):
    """
    Preprocess player log data by grouping by session and creating features.

    Args:
        csv_path: Path to the CSV file containing player logs

    Returns:
        DataFrame with sessions as rows and log counts as features
    """
    # Read the CSV file and parse timestamp column
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])  # Remove original 'time' column after renaming

    # Create a mapping from logid to label
    logid_to_label = dict(simple_log_ids)

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
        for logid, label in simple_log_ids:
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
    log_columns = [label for _, label in simple_log_ids]
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
    raw_data_dir = Path("../data/raw")
    output_dir = Path("../data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all CSV files in raw data directory
    csv_files = list(raw_data_dir.glob("*.csv"))

    print("=" * 70)
    print("Player Log Preprocessing - Batch Processing")
    print("=" * 70)
    print(f"\nFound {len(csv_files)} CSV files to process")
    print(f"Output directory: {output_dir.absolute()}")
    print()

    print("\nProcessing files...")
    print()

    # Track statistics
    successful = 0
    failed = 0
    all_processed_data = []
    failed_files = []
    total_files = len(csv_files)

    # Process each file
    for i, csv_file in enumerate(csv_files, 1):
        try:
            # Process the file
            processed_df = preprocess_player_logs(csv_file)
            all_processed_data.append(processed_df)
            successful += 1

            # Show progress every 100 files
            if i % 100 == 0 or i == total_files:
                print(
                    f"Progress: {i}/{total_files} files ({i/total_files*100:.1f}%) - Success: {successful}, Failed: {failed}"
                )

        except Exception as e:
            failed += 1
            failed_files.append((csv_file.name, str(e)))
            print(f"✗ Error processing {csv_file.name}: {e}")

    print()
    print("=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"✓ Successfully processed: {successful} files")
    print(f"✗ Failed: {failed} files")

    if failed_files:
        print("\nFailed files:")
        for filename, error in failed_files[:10]:  # Show first 10 errors
            print(f"  - {filename}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    # Combine all processed data
    if all_processed_data:
        print("\nCombining all processed data...")
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
        print(f"  Columns: {list(combined_df.columns)}")
        print()
        print("Sample of combined data:")
        print(combined_df.head(10))

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
    else:
        print("\n⚠ No data was processed successfully.")
