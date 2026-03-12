from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from src.preprocessing.cleaner import DataCleaner

# ----------------------------------------------------------------------
# remove_duplicates tests
# ----------------------------------------------------------------------


def test_remove_duplicates_all_columns(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)

    result = cleaner.remove_duplicates()
    expected = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

    assert_frame_equal(result.data_frame.reset_index(drop=True), expected)


def test_remove_duplicates_subset(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)

    result = cleaner.remove_duplicates(subset=["A"])
    expected = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

    assert_frame_equal(result.data_frame.reset_index(drop=True), expected)


def test_remove_duplicates_keep_last(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)

    result = cleaner.remove_duplicates(keep="last")
    expected = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

    assert_frame_equal(result.data_frame.reset_index(drop=True), expected)


def test_remove_duplicates_keep_false(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)

    result = cleaner.remove_duplicates(keep=False)
    expected = pd.DataFrame({"A": [1], "B": ["x"]})

    assert_frame_equal(result.data_frame.reset_index(drop=True), expected)


def test_remove_duplicates_returns_cleaner(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)

    result = cleaner.remove_duplicates()

    assert isinstance(result, DataCleaner)


def test_remove_duplicates_does_not_modify_original(df_with_duplicates):
    original = df_with_duplicates.copy(deep=True)

    cleaner = DataCleaner(df_with_duplicates)
    cleaner.remove_duplicates()

    assert_frame_equal(df_with_duplicates, original)


def test_remove_duplicates_missing_subset_column(df_with_duplicates, caplog):
    cleaner = DataCleaner(df_with_duplicates)

    with caplog.at_level("WARNING"):
        cleaner.remove_duplicates(subset=["A", "X"])

    assert "X" in caplog.text


def test_remove_duplicates_empty_df(empty_df):
    cleaner = DataCleaner(empty_df)

    result = cleaner.remove_duplicates()

    assert result.data_frame.empty


def test_remove_duplicates_no_duplicates(sample_df):
    cleaner = DataCleaner(sample_df)

    result = cleaner.remove_duplicates()

    assert_frame_equal(result.data_frame, sample_df)


def test_remove_duplicates_subset_multiple_columns():
    df = pd.DataFrame(
        {
            "A": [1, 1, 2, 2],
            "B": ["x", "x", "y", "z"],
        }
    )

    cleaner = DataCleaner(df)
    result = cleaner.remove_duplicates(subset=["A", "B"])

    assert len(result.data_frame) == 3


def test_remove_duplicates_preserves_order(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)

    result = cleaner.remove_duplicates()

    assert result.data_frame.iloc[0]["A"] == 1


def test_remove_duplicates_nan_subset_behavior():
    df = pd.DataFrame(
        {
            "A": [1, 1, None, None],
            "B": ["x", "x", "y", "z"],
        }
    )

    cleaner = DataCleaner(df)

    result = cleaner.remove_duplicates(subset=["A"])

    assert len(result.data_frame) == 2


def test_remove_duplicates_is_stable(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)

    result = cleaner.remove_duplicates()

    assert result.data_frame.index.is_monotonic_increasing
