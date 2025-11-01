import tempfile
from pathlib import Path

import pandas as pd

from src.preprocess import preprocess_all_players


def create_sample_data(temp_dir: Path):
    """Create minimal sample parquet files for testing."""
    sample_data = pd.DataFrame(
        {
            "time": ["2024-01-01 10:00:00.000"] * 5,
            "logid": [1013, 1101, 1202, 5004, 6001],
            "session": [1, 1, 1, 1, 1],
            "log_detail_code": [0, 0, 0, 0, 0],
            "actor_account_id": ["TEST001"] * 5,
            "actor_level": [10, 10, 10, 10, 10],
        }
    )

    test_file = temp_dir / "test_player.parquet"
    sample_data.to_parquet(test_file, index=False)

    labels_df = pd.DataFrame({"actor_account_id": ["TEST001"], "churn_yn": [0]})
    labels_file = temp_dir / "labels.csv"
    labels_df.to_csv(labels_file, index=False)

    return labels_file


def test_dask_preprocessing():
    """Test preprocessing with Dask."""

    print("\n" + "=" * 70)
    print("Testing Dask-based Preprocessing")
    print("=" * 70 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        raw_dir = temp_path / "raw"
        raw_dir.mkdir()
        output_dir = temp_path / "output"

        labels_file = create_sample_data(raw_dir)

        print("Testing Dask processing...")
        result = preprocess_all_players(
            raw_dir=raw_dir,
            output_dir=output_dir,
            output_filename="test_output.parquet",
            label_file_path=labels_file,
            n_workers=2,  # Use fewer workers for testing
        )

        assert result is not None, "Dask processing failed"
        assert len(result) > 0, "Dask processing produced no results"
        assert "actor_account_id" in result.columns, "Missing expected columns"
        assert "churn_yn" in result.columns, "Missing label column"

        print("\n" + "=" * 70)
        print("✅ Test passed! Dask integration is working correctly.")
        print(f"   Processed {len(result)} players with {result.shape[1]} features")
        print("=" * 70 + "\n")

        return result


if __name__ == "__main__":
    test_dask_preprocessing()
