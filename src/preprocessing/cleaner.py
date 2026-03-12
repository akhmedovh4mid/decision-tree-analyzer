from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype

from src.logging.error_messages import DataCleanerError as DCE
from src.logging.log_messages import DataCleanerLog as DCL
from src.logging.logger import get_logger

logger = get_logger(name=__name__)


Strategy = Literal[
    "drop_rows",
    "drop_columns",
    "fill_constant",
    "fill_mean",
    "fill_median",
    "fill_mode",
    "fill_ffill",
    "fill_bfill",
]

DropKeep = Literal["first", "last", False]


class DataCleaner:
    """
    Класс для очистки данных:
    - обработка пропусков
    - удаление дубликатов
    - удаление выбросов
    - сброс индекса

    Методы chainable.
    """

    def __init__(self, data_frame: DataFrame) -> None:
        self.data_frame: DataFrame = data_frame.copy()
        logger.debug(DCL.INIT, self.data_frame.shape)

    def __repr__(self) -> str:
        return f"DataCleaner(shape={self.data_frame.shape})"

    def copy(self) -> DataCleaner:
        return DataCleaner(self.data_frame.copy())

    @property
    def shape(self) -> tuple[int, int]:
        return self.data_frame.shape

    def head(self, n: int = 5) -> DataFrame:
        return self.data_frame.head(n)

    def get_data(self) -> DataFrame:
        return self.data_frame.copy()

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    def handle_missings(
        self,
        strategy: Strategy = "drop_rows",
        fill_value: int | float | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> DataCleaner:
        """
        Обрабатывает пропущенные значения в указанных колонках.

        Parameters
            strategy : Strategy
                Стратегия обработки:
                "drop_rows"      : удалить строки с пропусками
                "drop_columns"   : удалить колонки, содержащие пропуски
                "fill_constant"  : заполнить константой (требуется fill_value)
                "fill_mean"      : заполнить средним (только для числовых колонок)
                "fill_median"    : заполнить медианой (только для числовых)
                "fill_mode"      : заполнить модой (работает для любых типов)
                "fill_ffill"     : forward fill
                "fill_bfill"     : backward fill
            fill_value : int, float, str, optional
                Значение для заполнения при strategy="fill_constant".
            columns : list of str, optional
                Список колонок для обработки. Если None, обрабатываются все колонки.

        Returns
            self
        """

        if columns is None:
            cols = list(self.data_frame.columns)
        else:
            df_cols = set(self.data_frame.columns)
            cols = [c for c in columns if c in df_cols]

            missing_cols = set(columns) - set(cols)
            for col in missing_cols:
                logger.warning(DCL.COLUMN_NOT_FOUND, col)

        if not cols:
            logger.warning(DCE.MISSING_COLUMNS_NOT_FOUND_ERROR)
            return self

        logger.info(DCL.START_HANDLE_MISSINGS, strategy, cols)

        missing_before = self.data_frame[cols].isna().sum().sum()
        logger.debug(DCL.MISSING_BEFORE, missing_before)

        match strategy:
            case "drop_rows":
                before = len(self.data_frame)
                self.data_frame = self.data_frame.dropna(subset=cols)
                logger.info(DCL.ROWS_DROPPED, before - len(self.data_frame))

            case "drop_columns":
                cols_with_na = (
                    self.data_frame[cols]
                    .columns[self.data_frame[cols].isna().any()]
                    .tolist()
                )
                self.data_frame = self.data_frame.drop(columns=cols_with_na)
                logger.info(DCL.COLUMNS_DROPPED, cols_with_na)

            case "fill_constant":
                if fill_value is None:
                    raise ValueError(DCE.FILL_CONSTANT_MISSING_VALUE_ERROR)

                self.data_frame[cols] = self.data_frame[cols].fillna(fill_value)
                logger.info(DCL.FILLED_CONSTANT, fill_value)

            case "fill_mean":
                for col in cols:
                    if is_numeric_dtype(self.data_frame[col]):
                        mean_val = self.data_frame[col].mean()
                        self.data_frame[col] = self.data_frame[col].fillna(mean_val)
                        logger.debug(DCL.FILLED_MEAN, col, mean_val)
                    else:
                        logger.warning(DCL.NON_NUMERIC_COLUMN, col, "fill_mean")

            case "fill_median":
                for col in cols:
                    if is_numeric_dtype(self.data_frame[col]):
                        median_val = self.data_frame[col].median()
                        self.data_frame[col] = self.data_frame[col].fillna(median_val)
                        logger.debug(DCL.FILLED_MEDIAN, col, median_val)
                    else:
                        logger.warning(DCL.NON_NUMERIC_COLUMN, col, "fill_median")

            case "fill_mode":
                for col in cols:
                    mode_vals = self.data_frame[col].mode()

                    if not mode_vals.empty:
                        val = mode_vals.iloc[0]
                        self.data_frame[col] = self.data_frame[col].fillna(val)
                        logger.debug(DCL.FILLED_MODE, col, val)
                    else:
                        logger.warning(DCL.MODE_NOT_FOUND, col)

            case "fill_ffill":
                self.data_frame[cols] = self.data_frame[cols].ffill()
                logger.info(DCL.APPLY_FFILL)

            case "fill_bfill":
                self.data_frame[cols] = self.data_frame[cols].bfill()
                logger.info(DCL.APPLY_BFILL)

        remaining_cols = [c for c in cols if c in self.data_frame.columns]

        if remaining_cols:
            missing_after = self.data_frame[remaining_cols].isna().sum().sum()
        else:
            missing_after = 0

        logger.debug(DCL.MISSING_AFTER, missing_after)

        return self

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    def remove_duplicates(
        self,
        subset: Sequence[str] | None = None,
        keep: DropKeep = "first",
    ) -> DataCleaner:
        """
        Удаляет дубликаты строк.

        Parameters
            subset : list of str, optional
                Рассматривать только указанные колонки для определения дубликатов.
                Если None, используются все колонки.
            keep : 'first', 'last', False
                first : оставить первое вхождение
                last  : оставить последнее вхождение
                False : удалить все дубликаты

        Returns
            self
        """
        self.data_frame = self.data_frame.drop_duplicates(subset=subset, keep=keep)
        return self

    # --------------------------------------------------------
    # Outliers IQR
    # --------------------------------------------------------

    def remove_outliers_iqr(
        self,
        columns: Sequence[str] | None = None,
        multiplier: float = 1.5,
    ) -> DataCleaner:
        """
        Удаляет выбросы на основе межквартильного размаха (IQR).

        Для каждой указанной колонки вычисляются Q1, Q3, IQR.
        Удаляются строки, в которых значение хотя бы в одной колонке
        выходит за пределы [Q1 - multiplier*IQR, Q3 + multiplier*IQR].

        Важно: строки, содержащие NaN в анализируемых колонках,
        будут удалены, так как они не могут быть проверены на попадание в интервал.
        Если требуется иная обработка пропусков, выполните её перед вызовом этого метода
        с помощью `handle_missings`.

        Parameters
            columns : list of str, optional
                Список числовых колонок для анализа. Если None, берутся все числовые.
            multiplier : float, default=1.5
                Множитель IQR для определения границ.

        Returns
            self
        """
        if columns is None:
            columns = self.data_frame.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        for col in columns:
            if not is_numeric_dtype(self.data_frame[col]):
                raise TypeError(DCE.NON_NUMERIC_COLUMN_IQR_ERROR.format(col=col))

        df_subset = self.data_frame[list(columns)]

        if df_subset.empty:
            logger.warning(DCL.EMPY_DATA)
            return self

        values = df_subset.to_numpy()

        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        # Маска: True для строк, которые НЕ являются выбросами
        # Строки с NaN дадут False в сравнении, поэтому будут удалены
        mask = np.all((values >= lower) & (values <= upper), axis=1)

        self.data_frame = self.data_frame.loc[mask].reset_index(drop=True)

        return self

    # --------------------------------------------------------
    # Outliers Z-score
    # --------------------------------------------------------

    def remove_outliers_zscore(
        self,
        columns: Sequence[str] | None = None,
        threshold: float = 3.0,
    ) -> DataCleaner:
        """
        Удаляет выбросы на основе Z-оценки (стандартизации).

        Для каждой указанной колонки вычисляются среднее и стандартное отклонение.
        Удаляются строки, где |Z| > threshold хотя бы в одной колонке.

        Важно: строки, содержащие NaN в анализируемых колонках,
        будут удалены, так как Z-оценка для них не определена.
        Если требуется иная обработка пропусков, выполните её перед вызовом этого метода
        с помощью `handle_missings`.

        Parameters
            columns : list of str, optional
                Список числовых колонок для анализа. Если None, берутся все числовые.
            threshold : float, default=3.0
                Пороговое значение Z-оценки.

        Returns
            self
        """
        if columns is None:
            columns = self.data_frame.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        for col in columns:
            if not is_numeric_dtype(self.data_frame[col]):
                raise TypeError(DCE.NON_NUMERIC_COLUMN_ZSCORE_ERROR.format(col=col))

        df_subset = self.data_frame[list(columns)]

        if df_subset.empty:
            logger.warning(DCL.EMPY_DATA)
            return self

        # Вычисляем Z-оценки (среднее и std по каждой колонке, игнорируя NaN)
        means = df_subset.mean()
        stds = df_subset.std()

        # Избегаем деления на ноль
        if (stds == 0).any():  # pyright: ignore[reportAttributeAccessIssue]
            logger.warning(DCL.DEVISION_ZERO)
            stds = stds.replace(0, np.inf)  # pyright: ignore[reportAttributeAccessIssue]

        z_scores = (df_subset - means) / stds

        # Маска: True для строк, которые НЕ являются выбросами
        # Строки с NaN дадут False в сравнении, поэтому будут удалены
        mask = (z_scores.abs() <= threshold).all(axis=1)

        self.data_frame = self.data_frame.loc[mask].reset_index(drop=True)

        return self

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    def reset_index(self, drop: bool = True) -> DataCleaner:
        """Сбрасывает индекс DataFrame."""
        self.data_frame = self.data_frame.reset_index(drop=drop)
        return self
