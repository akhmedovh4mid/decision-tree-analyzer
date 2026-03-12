from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.preprocessing.cleaner import DataCleaner

# ----------------------------------------------------------------------
# remove_outliers_zscore tests
# ----------------------------------------------------------------------


def test_remove_outliers_zscore_basic(df_with_outliers):
    cleaner = DataCleaner(df_with_outliers)
    result = cleaner.remove_outliers_zscore(columns=["values"], threshold=2)

    assert_frame_equal(
        result.data_frame.reset_index(drop=True),
        df_with_outliers.reset_index(drop=True),
    )


def test_remove_outliers_zscore_multiple_columns():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 100, 5],
            "B": [10, 20, 30, 40, 1000],
        }
    )
    cleaner = DataCleaner(df)
    result = cleaner.remove_outliers_zscore(threshold=2)

    assert len(result.data_frame) <= len(df)


def test_remove_outliers_zscore_non_numeric_raises(df_with_missing):
    cleaner = DataCleaner(df_with_missing)

    with pytest.raises(TypeError, match="B"):
        cleaner.remove_outliers_zscore(columns=["A", "B"])


def test_remove_outliers_zscore_with_nan():
    df = pd.DataFrame({"A": [1, 2, np.nan, 100, 5]})
    cleaner = DataCleaner(df)
    result = cleaner.remove_outliers_zscore(columns=["A"])
    expected = df.loc[[0, 1, 3, 4]].reset_index(drop=True)

    assert_frame_equal(result.data_frame.reset_index(drop=True), expected)


def test_remove_outliers_zscore_std_zero(caplog):
    df = pd.DataFrame({"A": [5, 5, 5, 5]})
    cleaner = DataCleaner(df)
    caplog.clear()
    result = cleaner.remove_outliers_zscore()

    assert "нулев" in caplog.text.lower()
    assert len(result.data_frame) == 4


def test_remove_outliers_zscore_empty_dataframe(empty_df, caplog):
    cleaner = DataCleaner(empty_df)
    caplog.clear()
    result = cleaner.remove_outliers_zscore()

    assert "нет данных" in caplog.text.lower()
    assert_frame_equal(result.data_frame, empty_df)


def test_remove_outliers_zscore_all_nan():
    df = pd.DataFrame({"A": [np.nan, np.nan, np.nan]})

    cleaner = DataCleaner(df)

    result = cleaner.remove_outliers_zscore()

    assert len(result.data_frame) == 0


def test_remove_outliers_zscore_subset_column():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 1000000],
            "B": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )

    cleaner = DataCleaner(df)

    result = cleaner.remove_outliers_zscore(columns=["A"], threshold=2)

    assert 1000000 not in result.data_frame["A"].values
    assert len(result.data_frame) == 7


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_remove_outliers_zscore_with_inf(df_with_inf):
    cleaner = DataCleaner(df_with_inf)

    result = cleaner.remove_outliers_zscore()

    assert isinstance(result.data_frame, pd.DataFrame)


def test_remove_outliers_zscore_chainable(df_with_outliers):
    cleaner = DataCleaner(df_with_outliers)

    result = cleaner.remove_outliers_zscore()

    assert isinstance(result, DataCleaner)
