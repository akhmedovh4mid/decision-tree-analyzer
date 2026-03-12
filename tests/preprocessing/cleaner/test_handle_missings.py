from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.preprocessing.cleaner import DataCleaner

# ----------------------------------------------------------------------
# handle_missings tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy, expected_na_count",
    [
        ("drop_rows", 0),
        ("drop_columns", 0),
        ("fill_constant", 0),
        ("fill_mean", 5),
        ("fill_median", 5),
        ("fill_mode", 4),
        ("fill_ffill", 4),
        ("fill_bfill", 5),
    ],
)
def test_handle_missings_basic(strategy, expected_na_count, df_with_missing):
    cleaner = DataCleaner(df_with_missing)

    kwargs = {}
    if strategy == "fill_constant":
        kwargs["fill_value"] = 0

    cleaner.handle_missings(strategy=strategy, **kwargs)

    remaining_na = cleaner.data_frame.isna().sum().sum()

    assert remaining_na == expected_na_count


def test_handle_missings_drop_rows(df_with_missing):
    cleaner = DataCleaner(df_with_missing)
    result = cleaner.handle_missings("drop_rows")

    assert result.data_frame.empty
    assert result.data_frame.isna().sum().sum() == 0


def test_handle_missings_drop_columns(df_with_missing):
    cleaner = DataCleaner(df_with_missing)
    result = cleaner.handle_missings("drop_columns")

    assert result.data_frame.columns.tolist() == []


def test_handle_missings_fill_constant(df_with_missing):
    cleaner = DataCleaner(df_with_missing)

    result = cleaner.handle_missings("fill_constant", fill_value=99)

    mask = df_with_missing.isna()

    assert result.data_frame.isna().sum().sum() == 0

    filled_values = result.data_frame[mask]

    assert (filled_values.dropna() == 99).all().all()


def test_handle_missings_fill_constant_no_value(df_with_missing):
    cleaner = DataCleaner(df_with_missing)

    with pytest.raises(ValueError):
        cleaner.handle_missings("fill_constant")


def test_handle_missings_fill_mean(df_with_missing):
    cleaner = DataCleaner(df_with_missing)
    result = cleaner.handle_missings("fill_mean")

    assert result.data_frame["A"].iloc[2] == pytest.approx((1 + 2 + 4) / 3)
    assert result.data_frame["C"].iloc[3] == pytest.approx((1.1 + 2.2 + 3.3) / 3)
    assert result.data_frame["B"].isna().sum() == df_with_missing["B"].isna().sum()
    assert result.data_frame["D"].isna().sum() == 4


def test_handle_missings_fill_median(df_with_missing):
    cleaner = DataCleaner(df_with_missing)
    result = cleaner.handle_missings("fill_median")

    assert result.data_frame["A"].iloc[2] == 2
    assert result.data_frame["C"].iloc[3] == 2.2


def test_handle_missings_fill_mode(df_with_missing):
    df = pd.DataFrame(
        {
            "A": [1, 1, 2, None],
            "B": ["x", "y", "x", None],
            "C": [None, None, None, None],
        }
    )
    cleaner = DataCleaner(df)
    result = cleaner.handle_missings("fill_mode")

    assert result.data_frame["A"].iloc[3] == 1
    assert result.data_frame["B"].iloc[3] == "x"
    assert pd.isna(result.data_frame["C"].iloc[3])


def test_handle_missings_fill_ffill():
    df = pd.DataFrame({"A": [None, 2, None, 4]})
    cleaner = DataCleaner(df)

    result = cleaner.handle_missings("fill_ffill")

    expected = pd.Series([np.nan, 2, 2, 4], name="A")

    pd.testing.assert_series_equal(result.data_frame["A"], expected)


def test_handle_missings_fill_bfill():
    df = pd.DataFrame({"A": [1, None, None, 4]})
    cleaner = DataCleaner(df)
    result = cleaner.handle_missings("fill_bfill")
    expected = [1, 4, 4, 4]

    assert result.data_frame["A"].tolist() == expected


def test_handle_missings_with_columns_subset(df_with_missing):
    cleaner = DataCleaner(df_with_missing)
    result = cleaner.handle_missings("drop_rows", columns=["A", "B"])

    assert len(result.data_frame) == 2
    assert result.data_frame.index.tolist() == [0, 3]


def test_handle_missings_warn_on_missing_columns(df_with_missing, caplog):
    cleaner = DataCleaner(df_with_missing)
    caplog.clear()

    with caplog.at_level("WARNING"):
        cleaner.handle_missings("drop_rows", columns=["A", "X", "Y"])

    assert "X" in caplog.text
    assert "Y" in caplog.text


def test_handle_missings_empty_columns_list(df_with_missing, caplog):
    cleaner = DataCleaner(df_with_missing)
    caplog.clear()
    result = cleaner.handle_missings("drop_rows", columns=[])

    assert_frame_equal(result.data_frame, df_with_missing)


def test_handle_missings_non_numeric_warning_fill_mean(df_with_missing, caplog):
    cleaner = DataCleaner(df_with_missing)
    caplog.clear()

    with caplog.at_level("WARNING"):
        cleaner.handle_missings("fill_mean")

    assert "B" in caplog.text


def test_handle_missings_mode_not_found_warning(df_with_missing, caplog):
    cleaner = DataCleaner(df_with_missing)
    caplog.clear()

    with caplog.at_level("WARNING"):
        cleaner.handle_missings("fill_mode")

    assert "D" in caplog.text


def test_methods_are_chainable(df_with_missing):
    cleaner = DataCleaner(df_with_missing)
    result = (
        cleaner.handle_missings("fill_constant", fill_value=0)
        .remove_duplicates()
        .reset_index()
    )

    assert isinstance(result, DataCleaner)
