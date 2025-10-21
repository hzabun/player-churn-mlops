import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import (
    add_event_counts_and_finalize,
    aggregate_and_transform_to_players,
    aggregate_session_stats,
    filter_by_logids,
    get_session_boundaries,
    handle_deletepc_events,
    logid_label_mapping,
    parse_timestamps,
    preprocess_all_players,
    simple_log_ids,
    tuple_log_ids,
    validate_filtered_data,
    validate_raw_data,
)


class TestValidateRawData:
    """Test raw data validation logic."""

    def test_valid_raw_data(self):
        """Test that valid raw data passes validation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45.123"] * 3,
                "logid": [1013, 1101, 1202],
                "session": [1, 2, 3],
                "log_detail_code": [0, 0, 0],
                "actor_account_id": ["player1", "player1", "player1"],
                "actor_level": [10, 11, 12],
            }
        )
        is_valid, error_msg = validate_raw_data(df)
        assert is_valid
        assert error_msg == ""

    def test_empty_dataframe(self):
        """Test that empty DataFrame fails validation."""
        df = pd.DataFrame()
        is_valid, error_msg = validate_raw_data(df)
        assert not is_valid
        assert "empty" in error_msg.lower()

    def test_missing_columns(self):
        """Test that missing required columns fails validation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45.123"],
                "logid": [1013],
                "session": [1],
                # Missing: log_detail_code, actor_account_id, actor_level
            }
        )
        is_valid, error_msg = validate_raw_data(df)
        assert not is_valid
        assert "missing" in error_msg.lower()
        assert "log_detail_code" in error_msg.lower()

    def test_null_in_critical_columns(self):
        """Test that null values in critical columns fail validation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45.123", None],
                "logid": [1013, 1101],
                "session": [1, 2],
                "log_detail_code": [0, 0],
                "actor_account_id": ["player1", "player1"],
                "actor_level": [10, 11],
            }
        )
        is_valid, error_msg = validate_raw_data(df)
        assert not is_valid
        assert "null" in error_msg.lower()
        assert "time" in error_msg.lower()

    def test_invalid_data_types(self):
        """Test that invalid data types fail validation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45.123"] * 3,
                "logid": ["not_a_number", "1101", "1202"],  # String instead of int
                "session": [1, 2, 3],
                "log_detail_code": [0, 0, 0],
                "actor_account_id": ["player1", "player1", "player1"],
                "actor_level": [10, 11, 12],
            }
        )
        is_valid, error_msg = validate_raw_data(df)
        assert not is_valid
        assert "logid" in error_msg.lower()
        assert "numeric" in error_msg.lower()

    def test_invalid_timestamp_format(self):
        """Test that invalid timestamp format fails validation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45"],  # Missing milliseconds
                "logid": [1013],
                "session": [1],
                "log_detail_code": [0],
                "actor_account_id": ["player1"],
                "actor_level": [10],
            }
        )
        is_valid, error_msg = validate_raw_data(df)
        assert not is_valid
        assert "timestamp format" in error_msg.lower()

    def test_negative_session_values(self):
        """Test that negative session values fail validation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45.123"] * 3,
                "logid": [1013, 1101, 1202],
                "session": [1, -1, 3],  # Negative session
                "log_detail_code": [0, 0, 0],
                "actor_account_id": ["player1", "player1", "player1"],
                "actor_level": [10, 11, 12],
            }
        )
        is_valid, error_msg = validate_raw_data(df)
        assert not is_valid
        assert "session" in error_msg.lower()
        assert "negative" in error_msg.lower()

    def test_negative_actor_level(self):
        """Test that negative actor level fails validation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45.123"] * 3,
                "logid": [1013, 1101, 1202],
                "session": [1, 2, 3],
                "log_detail_code": [0, 0, 0],
                "actor_account_id": ["player1", "player1", "player1"],
                "actor_level": [10, -5, 12],  # Negative level
            }
        )
        is_valid, error_msg = validate_raw_data(df)
        assert not is_valid
        assert "actor_level" in error_msg.lower()
        assert "negative" in error_msg.lower()

    def test_suspiciously_high_actor_level(self):
        """Test that suspiciously high actor level fails validation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45.123"] * 3,
                "logid": [1013, 1101, 1202],
                "session": [1, 2, 3],
                "log_detail_code": [0, 0, 0],
                "actor_account_id": ["player1", "player1", "player1"],
                "actor_level": [10, 150, 12],  # Level 150 is suspicious
            }
        )
        is_valid, error_msg = validate_raw_data(df)
        assert not is_valid
        assert "actor_level" in error_msg.lower()
        assert "55" in error_msg


class TestFilterByLogids:
    """Test logid filtering logic."""

    def test_filter_simple_logids(self):
        df = pd.DataFrame(
            {
                "logid": [1013, 9999, 1101, 8888, 1202],
                "log_detail_code": [0, 0, 0, 0, 0],
            }
        )
        result = filter_by_logids(df)
        assert len(result) == 3
        assert result["logid"].isin(simple_log_ids).all()

    def test_filter_tuple_logids(self):
        df = pd.DataFrame(
            {
                "logid": [1012, 1012, 1102, 1102, 9999],
                "log_detail_code": [1, 2, 1, 2, 1],
            }
        )
        result = filter_by_logids(df)
        # Should only keep (1012, 1) and (1102, 1)
        assert len(result) == 2
        assert all(
            (tuple(row) in tuple_log_ids for row in result.itertuples(index=False))
        )

    def test_filter_empty_dataframe(self):
        df = pd.DataFrame(
            {
                "logid": [],
                "log_detail_code": [],
            }
        )
        result = filter_by_logids(df)
        assert len(result) == 0

    def test_filter_no_matches(self):
        df = pd.DataFrame(
            {
                "logid": [9999, 8888, 7777],
                "log_detail_code": [0, 0, 0],
            }
        )
        result = filter_by_logids(df)
        assert len(result) == 0


class TestValidateFilteredData:
    """Test filtered data validation logic."""

    def test_valid_data(self):
        df = pd.DataFrame(
            {
                "session": [1, 2, 3],
                "time": ["2024-01-15 10:30:45.123"] * 3,
            }
        )
        assert validate_filtered_data(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"session": [], "time": []})
        assert not validate_filtered_data(df)

    def test_no_positive_sessions(self):
        df = pd.DataFrame(
            {
                "session": [0, 0, -1],
                "time": ["2024-01-15 10:30:45.123"] * 3,
            }
        )
        assert not validate_filtered_data(df)


class TestParseTimestamps:
    """Test timestamp parsing."""

    def test_parse_timestamps(self):
        df = pd.DataFrame(
            {
                "time": [
                    "2024-01-15 10:30:45.123",
                    "2024-01-15 11:45:30.456",
                    "2024-01-16 09:00:00.000",
                ]
            }
        )
        result = parse_timestamps(df)

        assert "timestamp" in result.columns
        assert "session_date" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        assert result["session_date"].iloc[0] == datetime(2024, 1, 15).date()
        assert result["session_date"].iloc[2] == datetime(2024, 1, 16).date()

    def test_preserves_original_data(self):
        df = pd.DataFrame(
            {
                "time": ["2024-01-15 10:30:45.123"],
                "other_col": [42],
            }
        )
        result = parse_timestamps(df)
        assert "other_col" in result.columns
        assert result["other_col"].iloc[0] == 42


class TestGetSessionBoundaries:
    """Test session boundary extraction (includes timestamp parsing)."""

    def test_extract_boundaries(self):
        df = pd.DataFrame(
            {
                "time": [
                    "2024-01-15 10:00:00.000",
                    "2024-01-15 10:30:00.000",
                    "2024-01-15 11:00:00.000",
                ],
                "session": [1, 1, 2],
                "actor_account_id": ["player1", "player1", "player1"],
            }
        )
        result = get_session_boundaries(df)

        assert len(result) == 2
        assert "first_ts" in result.columns
        assert "last_ts" in result.columns
        assert "actor_id" in result.columns

    def test_boundary_timestamps_are_correct(self):
        """Test that first_ts and last_ts contain the correct timestamp values."""
        df = pd.DataFrame(
            {
                "time": [
                    "2024-01-15 10:00:00.000",
                    "2024-01-15 10:30:00.000",
                    "2024-01-15 10:45:00.000",
                    "2024-01-15 11:00:00.000",
                    "2024-01-15 11:30:00.000",
                ],
                "session": [1, 1, 1, 2, 2],
                "actor_account_id": [
                    "player1",
                    "player1",
                    "player1",
                    "player1",
                    "player1",
                ],
            }
        )
        result = get_session_boundaries(df)

        # Session 1: should span from 10:00:00 to 10:45:00
        session_1 = result.loc[(1, datetime(2024, 1, 15).date())]
        assert session_1["first_ts"] == pd.Timestamp("2024-01-15 10:00:00")
        assert session_1["last_ts"] == pd.Timestamp("2024-01-15 10:45:00")
        assert session_1["actor_id"] == "player1"

        # Session 2: should span from 11:00:00 to 11:30:00
        session_2 = result.loc[(2, datetime(2024, 1, 15).date())]
        assert session_2["first_ts"] == pd.Timestamp("2024-01-15 11:00:00")
        assert session_2["last_ts"] == pd.Timestamp("2024-01-15 11:30:00")
        assert session_2["actor_id"] == "player1"


class TestHandleDeletePCEvents:
    """Test DeletePC event handling."""

    def test_no_deletepc_events(self):
        df = pd.DataFrame(
            {
                "session": [1, 2, 3],
                "logid": [1013, 1101, 1202],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-15 10:00:00",
                        "2024-01-15 11:00:00",
                        "2024-01-15 12:00:00",
                    ]
                ),
            }
        )
        result = handle_deletepc_events(df)
        assert len(result) == 3
        assert all(result["session"] > 0)

    def test_deletepc_matched_to_next_session(self):
        df = pd.DataFrame(
            {
                "session": [0, 1, 2],
                "logid": [1012, 1013, 1101],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-15 10:00:00",
                        "2024-01-15 10:30:00",
                        "2024-01-15 11:00:00",
                    ]
                ),
            }
        )
        result = handle_deletepc_events(df)

        # DeletePC should be assigned to session 1
        deletepc_rows = result[result["logid"] == 1012]
        assert len(deletepc_rows) == 1
        assert deletepc_rows.iloc[0]["session"] == 1
        # Verify the timestamp is preserved correctly
        assert deletepc_rows.iloc[0]["timestamp"] == pd.Timestamp("2024-01-15 10:00:00")

        # Verify other events maintain their timestamps
        session_1_events = result[result["session"] == 1]
        assert len(session_1_events) == 2  # DeletePC + one regular event
        regular_event = result[(result["logid"] == 1013) & (result["session"] == 1)]
        assert regular_event.iloc[0]["timestamp"] == pd.Timestamp("2024-01-15 10:30:00")

    def test_deletepc_with_no_future_session(self):
        df = pd.DataFrame(
            {
                "session": [1, 0],
                "logid": [1013, 1012],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-15 10:00:00",
                        "2024-01-15 11:00:00",  # DeletePC after last session
                    ]
                ),
            }
        )
        result = handle_deletepc_events(df)

        # DeletePC with no future session should be dropped
        assert len(result) == 1
        assert 1012 not in result["logid"].values


class TestAggregateSessionStats:
    """Test session aggregation."""

    def test_basic_aggregation(self):
        df = pd.DataFrame(
            {
                "session": [1, 1, 2, 2],
                "session_date": [datetime(2024, 1, 15).date()] * 4,
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-15 10:00:00",
                        "2024-01-15 10:30:00",
                        "2024-01-15 11:00:00",
                        "2024-01-15 11:45:00",
                    ]
                ),
                "actor_account_id": ["player1"] * 4,
                "actor_level": [10, 11, 11, 11],
            }
        )
        result = aggregate_session_stats(df)

        assert len(result) == 2
        assert "first_ts" in result.columns
        assert "last_ts" in result.columns
        assert "actor_id" in result.columns
        assert "level" in result.columns
        assert "duration_min" in result.columns

    def test_aggregation_with_boundaries(self):
        df = pd.DataFrame(
            {
                "session": [1, 1],
                "session_date": [datetime(2024, 1, 15).date()] * 2,
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-15 10:10:00",
                        "2024-01-15 10:20:00",
                    ]
                ),
                "actor_account_id": ["player1"] * 2,
                "actor_level": [10, 10],
            }
        )

        # Boundaries show session actually started earlier
        boundaries = pd.DataFrame(
            {
                "first_ts": pd.to_datetime(["2024-01-15 10:00:00"]),
                "last_ts": pd.to_datetime(["2024-01-15 10:25:00"]),
                "actor_id": ["player1"],
            },
            index=pd.MultiIndex.from_tuples(
                [(1, datetime(2024, 1, 15).date())], names=["session", "session_date"]
            ),
        )

        result = aggregate_session_stats(df, boundaries)
        assert len(result) == 1


class TestAddEventCountsAndFinalize:
    """Test event counting and finalization."""

    def test_count_and_finalize(self):
        """Test that events are counted and columns are properly finalized."""
        # Create session stats
        stats = pd.DataFrame(
            {
                "session": [1, 2],
                "session_date": [datetime(2024, 1, 15).date()] * 2,
                "first_ts": pd.to_datetime(
                    ["2024-01-15 10:00:00", "2024-01-15 11:00:00"]
                ),
                "last_ts": pd.to_datetime(
                    ["2024-01-15 10:30:00", "2024-01-15 11:45:00"]
                ),
                "actor_id": ["player1", "player1"],
                "level": [10, 11],
                "duration_min": [30, 45],
            }
        )
        stats = stats.set_index(["session", "session_date"])

        # Create valid data with events
        df_valid = pd.DataFrame(
            {
                "session": [1, 1, 1, 2, 2],
                "session_date": [datetime(2024, 1, 15).date()] * 5,
                "logid": [1013, 1013, 1101, 1013, 1202],
            }
        )

        result = add_event_counts_and_finalize(stats, df_valid)

        # Check structure
        assert "first_timestamp" in result.columns
        assert "last_timestamp" in result.columns
        assert "actor_account_id" in result.columns
        assert "session_duration_minutes" in result.columns
        assert "actor_level" in result.columns

        # Check event columns
        assert "pc_level_up" in result.columns
        assert "invite_party" in result.columns
        assert "die" in result.columns

        # Check event counts
        assert result["pc_level_up"].iloc[0] == 2  # Session 1 has 2 level ups
        assert result["invite_party"].iloc[0] == 1  # Session 1 has 1 invite
        assert result["die"].iloc[1] == 1  # Session 2 has 1 die

    def test_missing_event_columns_filled(self):
        """Test that missing event columns are filled with zeros."""
        stats = pd.DataFrame(
            {
                "session": [1],
                "session_date": [datetime(2024, 1, 15).date()],
                "first_ts": pd.to_datetime(["2024-01-15 10:00:00"]),
                "last_ts": pd.to_datetime(["2024-01-15 10:30:00"]),
                "actor_id": ["player1"],
                "level": [10],
                "duration_min": [30],
            }
        )
        stats = stats.set_index(["session", "session_date"])

        # Only one event type
        df_valid = pd.DataFrame(
            {
                "session": [1],
                "session_date": [datetime(2024, 1, 15).date()],
                "logid": [1013],
            }
        )

        result = add_event_counts_and_finalize(stats, df_valid)

        # All event columns should exist
        for _, event_name in logid_label_mapping:
            assert event_name in result.columns

        # Only pc_level_up should be non-zero
        assert result["pc_level_up"].iloc[0] == 1
        assert result["die"].iloc[0] == 0
        assert result["invite_party"].iloc[0] == 0


class TestAggregateAndTransformToPlayers:
    """Test complete player-level transformation."""

    def test_aggregate_and_transform_multiple_sessions(self):
        """Test aggregation, renaming, derived features, and reordering."""
        sessions = pd.DataFrame(
            {
                "actor_account_id": ["player1"] * 3,
                "last_timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00",
                        "2024-01-05 11:00:00",
                        "2024-01-11 12:00:00",
                    ]
                ),
                "session_duration_minutes": [30, 45, 60],
                "pc_level_up": [1, 2, 0],
                "die": [5, 3, 10],
            }
        )
        result = aggregate_and_transform_to_players(sessions)

        # Check basic structure
        assert len(result) == 1
        assert result["actor_account_id"].iloc[0] == "player1"

        # Check aggregations
        assert result["total_sessions"].iloc[0] == 3
        assert result["total_playtime_minutes"].iloc[0] == 135  # 30+45+60
        assert result["avg_session_duration_minutes"].iloc[0] == 45.0

        # Check event renaming
        assert "level_ups_across_all_characters" in result.columns
        assert result["level_ups_across_all_characters"].iloc[0] == 3  # 1+2+0
        assert "die" in result.columns
        assert result["die"].iloc[0] == 18  # 5+3+10

        # Check derived features
        assert "account_lifespan_days" in result.columns
        # Account for time precision (hours/minutes included in calculation)
        assert result["account_lifespan_days"].iloc[0] == pytest.approx(10.0, abs=0.1)
        assert "average_sessions_per_day" in result.columns
        assert result["average_sessions_per_day"].iloc[0] == pytest.approx(
            0.3, abs=0.01
        )

        # Check column ordering - should not have first_session_timestamp
        assert "first_session_timestamp" not in result.columns
        expected_start = [
            "actor_account_id",
            "last_session_timestamp",
            "total_sessions",
            "account_lifespan_days",
            "average_sessions_per_day",
        ]
        actual_start = result.columns[:5].tolist()
        assert actual_start == expected_start

    def test_aggregate_multiple_players(self):
        """Test that multiple players are aggregated separately."""
        sessions = pd.DataFrame(
            {
                "actor_account_id": ["player1", "player1", "player2"],
                "last_timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00",
                        "2024-01-02 11:00:00",
                        "2024-01-01 10:30:00",
                    ]
                ),
                "session_duration_minutes": [30, 45, 60],
                "pc_level_up": [1, 2, 5],
            }
        )
        result = aggregate_and_transform_to_players(sessions)

        assert len(result) == 2
        player1 = result[result["actor_account_id"] == "player1"].iloc[0]
        player2 = result[result["actor_account_id"] == "player2"].iloc[0]

        assert player1["total_sessions"] == 2
        assert player2["total_sessions"] == 1
        assert player1["level_ups_across_all_characters"] == 3
        assert player2["level_ups_across_all_characters"] == 5

    def test_handle_zero_lifespan(self):
        """Test that zero lifespan is handled correctly."""
        sessions = pd.DataFrame(
            {
                "actor_account_id": ["player1"],
                "last_timestamp": pd.to_datetime(["2024-01-01 10:00:00"]),
                "session_duration_minutes": [30],
                "pc_level_up": [1],
            }
        )
        result = aggregate_and_transform_to_players(sessions)

        # Should handle division by zero
        assert result["account_lifespan_days"].iloc[0] == 0.0
        assert result["average_sessions_per_day"].iloc[0] == 1.0  # Replaces 0 with 1

    def test_rounding_of_float_columns(self):
        """Test that float columns are rounded appropriately."""
        sessions = pd.DataFrame(
            {
                "actor_account_id": ["player1"] * 3,
                "last_timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00",
                        "2024-01-02 11:00:00",
                        "2024-01-03 12:00:00",
                    ]
                ),
                "session_duration_minutes": [30.123456, 45.789012, 60.111111],
            }
        )
        result = aggregate_and_transform_to_players(sessions)

        # Check that values are rounded to 2 decimal places
        assert result["avg_session_duration_minutes"].iloc[0] == round(
            (30.123456 + 45.789012 + 60.111111) / 3, 2
        )


class TestPreprocessAllPlayersWithLabels:
    """Test the complete preprocessing pipeline with label joining."""

    def test_preprocess_with_labels_inner_join(self, tmp_path):
        """Test that labels are correctly joined with player features.

        Uses inner join to keep only players with labels.
        """
        # Create temporary directories
        raw_dir = tmp_path / "raw_parquet"
        raw_dir.mkdir()
        output_dir = tmp_path / "processed"
        label_file = tmp_path / "labels.csv"

        # Create a sample parquet file with valid data
        sample_data = pd.DataFrame(
            {
                "time": ["2024-01-15 10:00:00.000"] * 5
                + ["2024-01-15 11:00:00.000"] * 5,
                "logid": [1013, 1101, 1202, 1013, 1101] * 2,
                "session": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "log_detail_code": [0] * 10,
                "actor_account_id": ["player1"] * 5 + ["player2"] * 5,
                "actor_level": [10] * 10,
            }
        )
        sample_data.to_parquet(raw_dir / "test_data.parquet")

        # Create labels CSV with only player1 (to test inner join)
        labels_data = pd.DataFrame(
            {"actor_account_id": ["player1", "player3"], "churn_yn": [1, 0]}
        )
        labels_data.to_csv(label_file, index=False)

        # Run preprocessing
        result = preprocess_all_players(
            raw_dir=raw_dir,
            output_dir=output_dir,
            output_filename="test_output.parquet",
            label_file_path=label_file,
        )

        # Assertions
        assert result is not None
        assert "churn_yn" in result.columns
        # Should only have player1 due to inner join (player2 not in labels, player3 not in features)
        assert len(result) == 1
        assert result["actor_account_id"].iloc[0] == "player1"
        assert result["churn_yn"].iloc[0] == 1

    def test_preprocess_without_labels_file(self, tmp_path):
        """Test preprocessing when label file doesn't exist - should return None."""
        # Create temporary directories
        raw_dir = tmp_path / "raw_parquet"
        raw_dir.mkdir()
        output_dir = tmp_path / "processed"
        label_file = tmp_path / "nonexistent_labels.csv"

        # Create a sample parquet file
        sample_data = pd.DataFrame(
            {
                "time": ["2024-01-15 10:00:00.000"] * 5,
                "logid": [1013, 1101, 1202, 1013, 1101],
                "session": [1, 1, 1, 1, 1],
                "log_detail_code": [0] * 5,
                "actor_account_id": ["player1"] * 5,
                "actor_level": [10] * 5,
            }
        )
        sample_data.to_parquet(raw_dir / "test_data.parquet")

        # Run preprocessing without labels - should fail and return None
        result = preprocess_all_players(
            raw_dir=raw_dir,
            output_dir=output_dir,
            output_filename="test_output.parquet",
            label_file_path=label_file,
        )

        # Should return None when label file doesn't exist
        assert result is None

    def test_label_datatype_consistency(self, tmp_path):
        """Test that actor_account_id is converted to string for proper joining."""
        # Create temporary directories
        raw_dir = tmp_path / "raw_parquet"
        raw_dir.mkdir()
        output_dir = tmp_path / "processed"
        label_file = tmp_path / "labels.csv"

        # Create sample data with string account IDs
        sample_data = pd.DataFrame(
            {
                "time": ["2024-01-15 10:00:00.000"] * 5,
                "logid": [1013, 1101, 1202, 1013, 1101],
                "session": [1, 1, 1, 1, 1],
                "log_detail_code": [0] * 5,
                "actor_account_id": ["ABC123"] * 5,
                "actor_level": [10] * 5,
            }
        )
        sample_data.to_parquet(raw_dir / "test_data.parquet")

        # Create labels CSV
        labels_data = pd.DataFrame({"actor_account_id": ["ABC123"], "churn_yn": [0]})
        labels_data.to_csv(label_file, index=False)

        # Run preprocessing
        result = preprocess_all_players(
            raw_dir=raw_dir,
            output_dir=output_dir,
            output_filename="test_output.parquet",
            label_file_path=label_file,
        )

        # Should successfully join
        assert result is not None
        assert len(result) == 1
        assert result["actor_account_id"].iloc[0] == "ABC123"
        assert result["churn_yn"].iloc[0] == 0

    def test_multiple_players_with_mixed_labels(self, tmp_path):
        """Test preprocessing with multiple players, some with labels and some without."""
        # Create temporary directories
        raw_dir = tmp_path / "raw_parquet"
        raw_dir.mkdir()
        output_dir = tmp_path / "processed"
        label_file = tmp_path / "labels.csv"

        # Create sample data with three players
        sample_data = pd.DataFrame(
            {
                "time": ["2024-01-15 10:00:00.000"] * 15,
                "logid": [1013, 1101, 1202, 1013, 1101] * 3,
                "session": [1] * 5 + [2] * 5 + [3] * 5,
                "log_detail_code": [0] * 15,
                "actor_account_id": ["player1"] * 5 + ["player2"] * 5 + ["player3"] * 5,
                "actor_level": [10] * 15,
            }
        )
        sample_data.to_parquet(raw_dir / "test_data.parquet")

        # Create labels for only player1 and player2
        labels_data = pd.DataFrame(
            {"actor_account_id": ["player1", "player2"], "churn_yn": [1, 0]}
        )
        labels_data.to_csv(label_file, index=False)

        # Run preprocessing
        result = preprocess_all_players(
            raw_dir=raw_dir,
            output_dir=output_dir,
            output_filename="test_output.parquet",
            label_file_path=label_file,
        )

        # With inner join, only player1 and player2 should be in result
        assert result is not None
        assert len(result) == 2
        assert set(result["actor_account_id"].values) == {"player1", "player2"}

        # Check churn labels
        player1_churn = result[result["actor_account_id"] == "player1"][
            "churn_yn"
        ].iloc[0]
        player2_churn = result[result["actor_account_id"] == "player2"][
            "churn_yn"
        ].iloc[0]
        assert player1_churn == 1
        assert player2_churn == 0

    def test_output_file_created_with_labels(self, tmp_path):
        """Test that the output parquet file is created and contains labels."""
        # Create temporary directories
        raw_dir = tmp_path / "raw_parquet"
        raw_dir.mkdir()
        output_dir = tmp_path / "processed"
        label_file = tmp_path / "labels.csv"
        output_file = output_dir / "test_output.parquet"

        # Create sample data
        sample_data = pd.DataFrame(
            {
                "time": ["2024-01-15 10:00:00.000"] * 5,
                "logid": [1013, 1101, 1202, 1013, 1101],
                "session": [1, 1, 1, 1, 1],
                "log_detail_code": [0] * 5,
                "actor_account_id": ["player1"] * 5,
                "actor_level": [10] * 5,
            }
        )
        sample_data.to_parquet(raw_dir / "test_data.parquet")

        # Create labels
        labels_data = pd.DataFrame({"actor_account_id": ["player1"], "churn_yn": [1]})
        labels_data.to_csv(label_file, index=False)

        # Run preprocessing
        preprocess_all_players(
            raw_dir=raw_dir,
            output_dir=output_dir,
            output_filename="test_output.parquet",
            label_file_path=label_file,
        )

        # Check that output file was created
        assert output_file.exists()

        # Read the output file and verify it contains labels
        saved_data = pd.read_parquet(output_file)
        assert "churn_yn" in saved_data.columns
        assert len(saved_data) == 1
        assert saved_data["churn_yn"].iloc[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
