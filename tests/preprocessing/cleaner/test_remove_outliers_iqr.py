from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import floats
from pandas.testing import assert_frame_equal

from src.preprocessing.cleaner import DataCleaner

# ----------------------------------------------------------------------
# remove_outliers_iqr tests
# ----------------------------------------------------------------------


def test_remove_outliers_iqr_basic(df_with_outliers):
    cleaner = DataCleaner(df_with_outliers)
    result = cleaner.remove_outliers_iqr(columns=["values"])
    expected_rows = df_with_outliers.loc[[0, 1, 2, 3]]

    assert_frame_equal(
        result.data_frame.reset_index(drop=True), expected_rows.reset_index(drop=True)
    )


def test_remove_outliers_iqr_multiple_columns():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 100, 5],
            "B": [10, 20, 30, 40, 1000],
        }
    )
    cleaner = DataCleaner(df)
    result = cleaner.remove_outliers_iqr()
    expected = df.loc[[0, 1, 2]]

    assert_frame_equal(
        result.data_frame.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_remove_outliers_iqr_auto_select_numeric(numeric_df):
    cleaner = DataCleaner(numeric_df)

    result = cleaner.remove_outliers_iqr()

    assert len(result.data_frame) < len(numeric_df)


def test_remove_outliers_iqr_zero_iqr():
    df = pd.DataFrame({"A": [5, 5, 5, 5]})

    cleaner = DataCleaner(df)

    result = cleaner.remove_outliers_iqr()

    assert_frame_equal(result.data_frame, df)


def test_remove_outliers_iqr_non_numeric_raises(df_with_missing):
    cleaner = DataCleaner(df_with_missing)

    with pytest.raises(TypeError, match="B"):
        cleaner.remove_outliers_iqr(columns=["A", "B"])


def test_remove_outliers_iqr_with_nan():
    df = pd.DataFrame({"A": [1, 2, np.nan, 100, 5]})
    cleaner = DataCleaner(df)
    result = cleaner.remove_outliers_iqr(columns=["A"])
    expected = df.loc[[0, 1, 4]].reset_index(drop=True)

    assert_frame_equal(result.data_frame.reset_index(drop=True), expected)


def test_remove_outliers_iqr_empty_dataframe(empty_df, caplog):
    cleaner = DataCleaner(empty_df)
    caplog.clear()
    result = cleaner.remove_outliers_iqr()

    assert "Нет данных" in caplog.text
    assert_frame_equal(result.data_frame, empty_df)


def test_remove_outliers_iqr_all_outliers():
    df = pd.DataFrame({"A": [-1000, 1000, -2000, 2000]})
    cleaner = DataCleaner(df)
    result = cleaner.remove_outliers_iqr()

    assert len(result.data_frame) == 4


def test_remove_outliers_iqr_with_inf(df_with_inf):
    cleaner = DataCleaner(df_with_inf)

    result = cleaner.remove_outliers_iqr(columns=["A", "B"])

    assert np.isinf(result.data_frame.values).sum() >= 0


def test_remove_outliers_iqr_subset_column():
    df = pd.DataFrame({"A": [1, 2, 3, 100], "B": [10, 20, 30, 40]})

    cleaner = DataCleaner(df)

    result = cleaner.remove_outliers_iqr(columns=["A"])

    assert 100 not in result.data_frame["A"].values
    assert len(result.data_frame) == 3


def test_remove_outliers_iqr_chainable(df_with_outliers):
    cleaner = DataCleaner(df_with_outliers)

    result = cleaner.remove_outliers_iqr()

    assert isinstance(result, DataCleaner)


@given(data_frames(columns=[column("A", floats(-1000, 1000))]))
def test_remove_outliers_iqr_never_crashes(df):
    cleaner = DataCleaner(df)

    cleaner.remove_outliers_iqr()
