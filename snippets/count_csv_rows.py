#!/usr/bin/env python3
"""
Efficiently count rows in all CSV files in the data/raw folder.
This script uses efficient file reading without loading entire files into memory.
"""

import os
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple


def count_rows_in_csv(filepath: Path) -> Tuple[str, int]:
    """
    Count rows in a single CSV file efficiently.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (filename, row_count)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use sum with generator for memory efficiency
            # Subtract 1 to exclude header row
            row_count = sum(1 for _ in f) - 1
        return (filepath.name, row_count)
    except Exception as e:
        return (filepath.name, f"Error: {str(e)}")


def count_all_csv_rows(data_folder: str = "data/raw", use_parallel: bool = True) -> dict:
    """
    Count rows in all CSV files in the specified folder.
    
    Args:
        data_folder: Path to the folder containing CSV files
        use_parallel: Whether to use parallel processing (faster for many files)
        
    Returns:
        Dictionary mapping filenames to row counts
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Folder '{data_folder}' does not exist")
    
    # Get all CSV files
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{data_folder}'")
        return {}
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    results = {}
    
    if use_parallel and len(csv_files) > 10:
        # Use parallel processing for better performance with many files
        with ProcessPoolExecutor() as executor:
            # Submit all tasks
            future_to_file = {executor.submit(count_rows_in_csv, f): f for f in csv_files}
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_file), 1):
                filename, count = future.result()
                results[filename] = count
                
                # Progress indicator
                if i % 50 == 0 or i == len(csv_files):
                    print(f"Processed {i}/{len(csv_files)} files...")
    else:
        # Sequential processing for smaller numbers of files
        for i, filepath in enumerate(csv_files, 1):
            filename, count = count_rows_in_csv(filepath)
            results[filename] = count
            
            # Progress indicator
            if i % 50 == 0 or i == len(csv_files):
                print(f"Processed {i}/{len(csv_files)} files...")
    
    return results


def main():
    """Main execution function."""
    print("=" * 60)
    print("CSV Row Counter")
    print("=" * 60)
    
    # Count rows in all CSV files
    results = count_all_csv_rows("data/raw", use_parallel=True)
    
    # Calculate statistics
    total_rows = 0
    error_count = 0
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    # Sort results by filename for easier reading
    for filename in sorted(results.keys()):
        count = results[filename]
        if isinstance(count, int):
            print(f"{filename}: {count:,} rows")
            total_rows += count
        else:
            print(f"{filename}: {count}")
            error_count += 1
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Total files processed: {len(results)}")
    print(f"Total rows across all files: {total_rows:,}")
    if error_count > 0:
        print(f"Files with errors: {error_count}")
    print(f"Average rows per file: {total_rows // len(results) if results else 0:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
