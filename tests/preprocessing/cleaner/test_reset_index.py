from pandas.testing import assert_frame_equal

from src.preprocessing.cleaner import DataCleaner

# ----------------------------------------------------------------------
# reset_index tests
# ----------------------------------------------------------------------


def test_reset_index_drop_true(sample_df):
    cleaner = DataCleaner(sample_df)

    cleaner.data_frame.index = [10, 20, 30, 40]

    result = cleaner.reset_index(drop=True)

    expected = sample_df.copy().reset_index(drop=True)

    assert_frame_equal(result.data_frame, expected)


def test_reset_index_drop_false(sample_df):
    cleaner = DataCleaner(sample_df)

    cleaner.data_frame.index = [10, 20, 30, 40]

    result = cleaner.reset_index(drop=False)

    expected = sample_df.copy()
    expected.index = [10, 20, 30, 40]
    expected = expected.reset_index(drop=False)

    assert_frame_equal(result.data_frame, expected)


def test_reset_index_chainable(sample_df):
    cleaner = DataCleaner(sample_df)

    result = cleaner.reset_index()

    assert isinstance(result, DataCleaner)


def test_reset_index_empty_df(empty_df):
    cleaner = DataCleaner(empty_df)

    result = cleaner.reset_index()

    assert result.data_frame.empty
