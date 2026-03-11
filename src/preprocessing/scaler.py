from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from src.logging.error_messages import ScalerError as SCE
from src.logging.log_messages import ScalerLog as SCL
from src.logging.logger import get_logger

logger = get_logger(name=__name__)


ScaleStrategy = Literal[
    "standard",  # StandardScaler (среднее=0, дисперсия=1)
    "minmax",  # MinMaxScaler ([0, 1])
    "robust",  # RobustScaler (медиана, квартили)
    "maxabs",  # MaxAbsScaler ([-1, 1], сохраняет разреженность)
    "quantile",  # QuantileTransformer (равномерное или нормальное распределение)
    "power",  # PowerTransformer (Yeo-Johnson / Box-Cox)
]


class Scaler:
    """
    Класс для масштабирования (нормализации/стандартизации) числовых признаков.

    Поддерживаемые стратегии:
      - standard  : StandardScaler (z = (x - μ) / σ)
      - minmax    : MinMaxScaler (в диапазон [0, 1])
      - robust    : RobustScaler (использует медиану и квартили, устойчив к выбросам)
      - maxabs    : MaxAbsScaler (деление на максимум, сохраняет знак)
      - quantile  : QuantileTransformer (приводит к равномерному или нормальному распределению)
      - power     : PowerTransformer (Yeo-Johnson или Box-Cox для приведения к нормальному)

    Все методы сохраняют обученные скейлеры во внутреннем словаре `self.scalers_`,
    что позволяет впоследствии применить те же преобразования к новым данным
    с помощью метода `transform`.

    Объект работает как chainable: методы, изменяющие данные, возвращают `self`.
    """

    def __init__(self, data_frame: DataFrame) -> None:
        """
        Инициализация с копией исходного DataFrame.

        Args:
            data_frame: pandas DataFrame для масштабирования.
        """
        self.data_frame = data_frame.copy()
        self.scalers_: dict[str, Any] = {}  # сохранённые скейлеры для каждой колонки
        logger.debug(SCL.INIT, self.data_frame.shape)

    def __repr__(self) -> str:
        return f"Scaler(shape={self.data_frame.shape})"

    def copy(self) -> Scaler:
        """Возвращает копию текущего экземпляра."""
        new_scaler = Scaler(self.data_frame.copy())
        new_scaler.scalers_ = self.scalers_.copy()
        return new_scaler

    @property
    def shape(self) -> tuple[int, int]:
        return self.data_frame.shape

    def get_data(self) -> DataFrame:
        """Возвращает текущий DataFrame (после всех преобразований)."""
        return self.data_frame

    # ------------------------------------------------------------------
    # Основной метод scale (выбор стратегии)
    # ------------------------------------------------------------------

    def scale(
        self,
        strategy: ScaleStrategy,
        columns: Sequence[str] | None = None,
        **kwargs,
    ) -> Scaler:
        """
        Применить выбранную стратегию масштабирования к указанным колонкам.

        Args:
            strategy: Стратегия масштабирования.
            columns:  Список имён колонок для обработки.
                      Если None – обрабатываются все числовые колонки.
            **kwargs: Дополнительные параметры, передаваемые в конструктор скейлера
                      (например, `with_mean=False` для StandardScaler,
                      `output_distribution='normal'` для QuantileTransformer).

        Returns:
            self
        """
        if columns is None:
            # По умолчанию выбираем все числовые колонки
            columns = self.data_frame.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        # Проверка наличия колонок
        existing_cols = []
        missing_cols = []
        for col in columns:
            if col in self.data_frame.columns:
                # Дополнительно проверяем, что колонка числовая
                if not is_numeric_dtype(self.data_frame[col]):
                    logger.warning(SCL.NON_NUMERIC_COLUMN, col)
                    continue
                existing_cols.append(col)
            else:
                missing_cols.append(col)

        if missing_cols:
            logger.warning(SCL.COLUMN_NOT_FOUND, missing_cols)

        if not existing_cols:
            logger.warning(SCE.NO_COLUMNS_TO_SCALE)
            return self

        logger.info(SCL.START_SCALE, strategy, existing_cols)

        # Выбор стратегии
        if strategy == "standard":
            self._fit_standard(existing_cols, **kwargs)
            self._transform_standard(existing_cols)
        elif strategy == "minmax":
            self._fit_minmax(existing_cols, **kwargs)
            self._transform_minmax(existing_cols)
        elif strategy == "robust":
            self._fit_robust(existing_cols, **kwargs)
            self._transform_robust(existing_cols)
        elif strategy == "maxabs":
            self._fit_maxabs(existing_cols, **kwargs)
            self._transform_maxabs(existing_cols)
        elif strategy == "quantile":
            self._fit_quantile(existing_cols, **kwargs)
            self._transform_quantile(existing_cols)
        elif strategy == "power":
            self._fit_power(existing_cols, **kwargs)
            self._transform_power(existing_cols)
        else:
            raise ValueError(SCE.UNKNOWN_STRATEGY.format(strategy=strategy))

        logger.info(SCL.SCALE_COMPLETE, strategy)
        return self

    # ------------------------------------------------------------------
    # Приватные методы fit для каждой стратегии
    # ------------------------------------------------------------------

    def _fit_standard(self, columns: list[str], **kwargs) -> None:
        """Обучает StandardScaler для каждой колонки."""
        for col in columns:
            data = self.data_frame[[col]].dropna().to_numpy()
            scaler = StandardScaler(**kwargs)
            scaler.fit(data)
            self.scalers_[col] = scaler
            logger.debug(SCL.FITTED_STANDARD, col)

    def _fit_minmax(self, columns: list[str], **kwargs) -> None:
        """Обучает MinMaxScaler для каждой колонки."""
        for col in columns:
            data = self.data_frame[[col]].dropna().to_numpy()
            scaler = MinMaxScaler(**kwargs)
            scaler.fit(data)
            self.scalers_[col] = scaler
            logger.debug(SCL.FITTED_MINMAX, col)

    def _fit_robust(self, columns: list[str], **kwargs) -> None:
        """Обучает RobustScaler для каждой колонки."""
        for col in columns:
            data = self.data_frame[[col]].dropna().to_numpy()
            scaler = RobustScaler(**kwargs)
            scaler.fit(data)
            self.scalers_[col] = scaler
            logger.debug(SCL.FITTED_ROBUST, col)

    def _fit_maxabs(self, columns: list[str], **kwargs) -> None:
        """Обучает MaxAbsScaler для каждой колонки."""
        for col in columns:
            data = self.data_frame[[col]].dropna().to_numpy()
            scaler = MaxAbsScaler(**kwargs)
            scaler.fit(data)
            self.scalers_[col] = scaler
            logger.debug(SCL.FITTED_MAXABS, col)

    def _fit_quantile(self, columns: list[str], **kwargs) -> None:
        """Обучает QuantileTransformer для каждой колонки."""
        for col in columns:
            data = self.data_frame[[col]].dropna().to_numpy()
            scaler = QuantileTransformer(**kwargs)
            scaler.fit(data)
            self.scalers_[col] = scaler
            logger.debug(SCL.FITTED_QUANTILE, col)

    def _fit_power(self, columns: list[str], **kwargs) -> None:
        """Обучает PowerTransformer для каждой колонки."""
        for col in columns:
            data = self.data_frame[[col]].dropna().to_numpy()
            scaler = PowerTransformer(**kwargs)
            scaler.fit(data)
            self.scalers_[col] = scaler
            logger.debug(SCL.FITTED_POWER, col)

    # ------------------------------------------------------------------
    # Приватные методы transform для каждой стратегии
    # ------------------------------------------------------------------

    def _transform_standard(self, columns: list[str]) -> None:
        """Применяет StandardScaler к указанным колонкам."""
        self._apply_scaler_transform(columns, "standard")

    def _transform_minmax(self, columns: list[str]) -> None:
        """Применяет MinMaxScaler к указанным колонкам."""
        self._apply_scaler_transform(columns, "minmax")

    def _transform_robust(self, columns: list[str]) -> None:
        """Применяет RobustScaler к указанным колонкам."""
        self._apply_scaler_transform(columns, "robust")

    def _transform_maxabs(self, columns: list[str]) -> None:
        """Применяет MaxAbsScaler к указанным колонкам."""
        self._apply_scaler_transform(columns, "maxabs")

    def _transform_quantile(self, columns: list[str]) -> None:
        """Применяет QuantileTransformer к указанным колонкам."""
        self._apply_scaler_transform(columns, "quantile")

    def _transform_power(self, columns: list[str]) -> None:
        """Применяет PowerTransformer к указанным колонкам."""
        self._apply_scaler_transform(columns, "power")

    def _apply_scaler_transform(self, columns: list[str], strategy: str) -> None:
        """
        Общий метод применения сохранённого скейлера к колонке.
        """
        for col in columns:
            scaler = self.scalers_.get(col)
            if scaler is None:
                raise RuntimeError(SCE.SCALER_NOT_FITTED.format(col=col))

            # Преобразуем данные, временно заполняя NaN (скейлеры не работают с NaN)
            col_data = self.data_frame[col].to_numpy().reshape(-1, 1)
            # Сохраняем маску NaN, чтобы потом их восстановить
            nan_mask = np.isnan(col_data).flatten()

            if nan_mask.any():
                # Заполняем NaN временно, например, средним по обученной выборке?
                # Но лучше не применять скейлер к NaN, а оставить NaN.
                # Временно заполним нулём, а потом вернём NaN.
                fill_value = 0.0
                col_data_filled = np.where(nan_mask, fill_value, col_data)
            else:
                col_data_filled = col_data

            transformed = scaler.transform(col_data_filled).flatten()

            # Возвращаем NaN на исходные места
            if nan_mask.any():
                transformed = np.where(nan_mask, np.nan, transformed)

            self.data_frame[col] = transformed
            logger.debug(SCL.TRANSFORM_APPLIED, col, strategy)

    # ------------------------------------------------------------------
    # Преобразование новых данных (transform)
    # ------------------------------------------------------------------

    def transform(self, new_data: DataFrame) -> DataFrame:
        """
        Применить сохранённые скейлеры к новому DataFrame (например, тестовой выборке).

        Args:
            new_data: Новый DataFrame с теми же колонками (до масштабирования).

        Returns:
            Преобразованный DataFrame.
        """
        result = new_data.copy()

        for col, scaler in self.scalers_.items():
            if col not in result.columns:
                logger.warning(SCL.COLUMN_MISSING_FOR_TRANSFORM, col)
                continue

            if not is_numeric_dtype(result[col]):
                logger.warning(SCL.NON_NUMERIC_COLUMN_TRANSFORM, col)
                continue

            # Аналогичная обработка NaN
            col_data = result[col].to_numpy().reshape(-1, 1)
            nan_mask = np.isnan(col_data).flatten()

            if nan_mask.any():
                fill_value = 0.0
                col_data_filled = np.where(nan_mask, fill_value, col_data)
            else:
                col_data_filled = col_data

            transformed = scaler.transform(col_data_filled).flatten()

            if nan_mask.any():
                transformed = np.where(nan_mask, np.nan, transformed)

            result[col] = transformed

        return result

    # ------------------------------------------------------------------
    # Обратное преобразование (inverse_transform)
    # ------------------------------------------------------------------

    def inverse_transform(self, scaled_data: DataFrame | None = None) -> DataFrame:
        """
        Восстанавливает исходные значения для всех масштабированных колонок.
        Если scaled_data не передан, используется self.data_frame.

        Args:
            scaled_data: DataFrame с масштабированными данными. Если None, берётся текущий.

        Returns:
            DataFrame с восстановленными значениями (только для колонок, имеющих скейлер).
        """
        if scaled_data is None:
            df = self.data_frame.copy()
        else:
            df = scaled_data.copy()

        for col, scaler in self.scalers_.items():
            if col not in df.columns:
                continue

            col_data = df[col].to_numpy().reshape(-1, 1)
            nan_mask = np.isnan(col_data).flatten()

            if nan_mask.any():
                fill_value = 0.0
                col_data_filled = np.where(nan_mask, fill_value, col_data)
            else:
                col_data_filled = col_data

            inverted = scaler.inverse_transform(col_data_filled).flatten()

            if nan_mask.any():
                inverted = np.where(nan_mask, np.nan, inverted)

            df[col] = inverted

        return df

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def reset_index(self, drop: bool = True) -> Scaler:
        """Сброс индекса DataFrame."""
        self.data_frame.reset_index(drop=drop, inplace=True)
        return self
