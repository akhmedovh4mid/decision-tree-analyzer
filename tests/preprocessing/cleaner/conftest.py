from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Простой DataFrame без пропущенных значений."""
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "d"],
            "C": [1.1, 2.2, 3.3, 4.4],
        }
    )


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """DataFrame с пропущенными значениями в различных столбцах."""
    return pd.DataFrame(
        {
            "A": [1, 2, None, 4],
            "B": ["a", None, "c", "d"],
            "C": [1.1, 2.2, 3.3, None],
            "D": [None, None, None, None],
        }
    )


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """DataFrame с повторяющимися строками."""
    return pd.DataFrame(
        {
            "A": [1, 2, 2, 3, 3, 3],
            "B": ["x", "y", "y", "z", "z", "z"],
        }
    )


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """Все данные в формате DataFrame числовые."""
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 100],
            "num2": [10, 20, 30, 40, 50],
            "num3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame с четко выраженными выбросами."""
    return pd.DataFrame(
        {
            "values": [1, 2, 3, 4, 100, -50],
            "normal": [10, 20, 30, 40, 50, 60],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Пустой DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def df_with_inf() -> pd.DataFrame:
    """DataFrame с значениями inf (обрабатываются как числовые, а не NaN)."""
    return pd.DataFrame(
        {
            "A": [1, 2, np.inf, 4],
            "B": [np.inf, 2, 3, 4],
            "C": [1, 2, 3, np.nan],
        }
    )


@pytest.fixture
def large_df() -> pd.DataFrame:
    """Большой DataFrame (10 000 строк) для тестирования производительности."""
    n = 10_000
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "A": rng.integers(0, 100, size=n),
            "B": rng.normal(size=n),
            "C": rng.choice(np.array([1, 2, 3, None], dtype=object), size=n),
        }
    )
