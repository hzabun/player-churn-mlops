from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(csv_path, parquet_path):
    """Convert a single CSV file to Parquet."""
    try:
        # Read CSV
        df = pd.read_csv(csv_path, dtype={"actor_account_id": str}, low_memory=False)

        # Save as Parquet
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

        # Get file sizes
        csv_size = csv_path.stat().st_size
        parquet_size = parquet_path.stat().st_size
        compression_ratio = (1 - parquet_size / csv_size) * 100

        return csv_size, parquet_size, compression_ratio, None

    except Exception as e:
        return 0, 0, 0, str(e)


if __name__ == "__main__":
    csv_dir = Path("data/raw")
    parquet_dir = Path("data/raw_parquet")

    # Create output directory
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Get all CSV files
    csv_files = list(csv_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        exit(1)

    print("=" * 70)
    print("CSV to Parquet Conversion")
    print("=" * 70)
    print(f"Input: {csv_dir}")
    print(f"Output: {parquet_dir}")
    print(f"Files to convert: {len(csv_files)}\n")

    # Convert files
    total_csv_size = 0
    total_parquet_size = 0
    success = 0
    failed = 0
    errors = []

    for i, csv_file in enumerate(csv_files, 1):
        if i == 1 or i == len(csv_files) or i % 50 == 0:
            print(
                f"Converting {i}/{len(csv_files)} ({i/len(csv_files)*100:.1f}%)...",
                end="\r",
            )

        parquet_file = parquet_dir / csv_file.with_suffix(".parquet").name

        csv_size, parquet_size, ratio, error = convert_csv_to_parquet(
            csv_file, parquet_file
        )

        if error:
            failed += 1
            errors.append((csv_file.name, error))
        else:
            success += 1
            total_csv_size += csv_size
            total_parquet_size += parquet_size

    print(" " * 80)  # Clear progress line

    # Print summary
    print("\n" + "=" * 70)
    print("Conversion Summary")
    print("=" * 70)
    print(f"✓ Successfully converted: {success} files")
    print(f"✗ Failed: {failed} files")

    if success > 0:
        overall_compression = (1 - total_parquet_size / total_csv_size) * 100
        print(f"\nStorage Statistics:")
        print(f"  Original CSV size: {total_csv_size / (1024**3):.2f} GB")
        print(f"  Parquet size: {total_parquet_size / (1024**3):.2f} GB")
        print(
            f"  Space saved: {(total_csv_size - total_parquet_size) / (1024**3):.2f} GB"
        )
        print(f"  Compression ratio: {overall_compression:.1f}%")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for filename, error in errors[:10]:
            print(f"  - {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    print("\n" + "=" * 70)
    print("✓ Conversion complete!")
    print("=" * 70)
    print(f"\nNext step: Update preprocess.py to read from {parquet_dir}")
