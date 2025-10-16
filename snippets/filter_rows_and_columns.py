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


def filter_csv(input_path, output_path=None):
    """
    Filter CSV file by specified column and logid/log_detail_code combinations.

    Args:
        input_path: Path to the input CSV file
        output_path: Path to save filtered CSV (optional)

    Returns:
        Filtered DataFrame

    Raises:
        ValueError: If any required columns are missing from the CSV file
    """
    # Read only the required columns to avoid dtype warnings and improve performance
    # First, check if all columns exist by reading just the header
    df_header = pd.read_csv(input_path, nrows=0)
    missing_columns = [col for col in feature_columns if col not in df_header.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in CSV file: {missing_columns}\n"
            f"Required columns: {feature_columns}\n"
            f"Available columns: {list(df_header.columns)}"
        )

    # Read only the columns we need (avoids mixed dtype warnings and is faster)
    df_filtered = pd.read_csv(input_path, usecols=feature_columns)

    # Create filter mask
    mask = df_filtered["logid"].isin(simple_log_ids)
    for logid, detail_code in tuple_log_ids:
        mask |= (df_filtered["logid"] == logid) & (
            df_filtered["log_detail_code"] == detail_code
        )

    # Apply the mask
    df_result = df_filtered[mask].copy()

    # Save to output file if specified
    if output_path:
        df_result.to_csv(output_path, index=False)
        print(f"Filtered data saved to: {output_path}")

    return df_result


# Example usage
if __name__ == "__main__":
    # Directories
    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")

    # Create output directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Found {len(csv_files)} CSV files to process")
    print(f"{'='*60}")

    # Process each CSV file
    total_original_rows = 0
    total_filtered_rows = 0
    successful_files = 0
    failed_files = 0
    errors = []

    for idx, csv_file in enumerate(csv_files, 1):
        try:
            # Show progress every 100 files or on first/last file
            if idx == 1 or idx == len(csv_files) or idx % 100 == 0:
                print(f"Processing file {idx}/{len(csv_files)}...", end="\r")

            # Filter the data
            filtered_df = filter_csv(csv_file)

            # Get statistics (more efficient - we already have filtered_df)
            original_rows = len(pd.read_csv(csv_file, usecols=feature_columns))
            filtered_rows = len(filtered_df)

            total_original_rows += original_rows
            total_filtered_rows += filtered_rows

            # Save to output file
            output_file = processed_data_dir / f"{csv_file.name}"
            filtered_df.to_csv(output_file, index=False)

            successful_files += 1

        except Exception as e:
            errors.append((csv_file.name, str(e)))
            failed_files += 1

    # Clear the progress line
    print(" " * 80, end="\r")

    # Print summary
    print(f"{'='*60}")
    print(f"SUMMARY:")
    print(f"  Successfully processed: {successful_files} files")
    print(f"  Failed: {failed_files} files")
    print(f"  Total original rows: {total_original_rows:,}")
    if total_original_rows > 0:
        print(
            f"  Total filtered rows: {total_filtered_rows:,} ({total_filtered_rows/total_original_rows*100:.1f}%)"
        )
    print(f"  Output directory: {processed_data_dir}")

    # Show errors if any
    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        for filename, error in errors[:10]:  # Show first 10 errors
            print(f"  - {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
