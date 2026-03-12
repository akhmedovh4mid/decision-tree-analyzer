from __future__ import annotations

from pandas.testing import assert_frame_equal

from src.preprocessing.cleaner import DataCleaner

# ----------------------------------------------------------------------
# Инициализация и основные методы
# ----------------------------------------------------------------------


def test_init_creates_copy(sample_df):
    cleaner = DataCleaner(sample_df)

    assert cleaner.data_frame is not sample_df
    assert_frame_equal(cleaner.data_frame, sample_df)


def test_copy_method(sample_df):
    cleaner = DataCleaner(sample_df)

    copy_cleaner = cleaner.copy()

    assert copy_cleaner is not cleaner
    assert_frame_equal(copy_cleaner.data_frame, sample_df)

    cleaner.data_frame.iloc[0, 0] = 999

    assert cleaner.data_frame.iloc[0, 0] == 999
    assert copy_cleaner.data_frame.iloc[0, 0] == 1


def test_shape(sample_df):
    cleaner = DataCleaner(sample_df)

    assert cleaner.shape == sample_df.shape


def test_head(sample_df):
    cleaner = DataCleaner(sample_df)

    result = cleaner.head(2)
    expected = sample_df.head(2)

    assert_frame_equal(result, expected)


def test_get_data_returns_copy(sample_df):
    cleaner = DataCleaner(sample_df)

    data = cleaner.get_data()

    assert_frame_equal(data, sample_df)
    assert data is not cleaner.data_frame

    data.iloc[0, 0] = 999

    assert cleaner.data_frame.iloc[0, 0] != 999
