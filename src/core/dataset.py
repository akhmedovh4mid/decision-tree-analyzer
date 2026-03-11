from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype

from src.data_io import DataLoader
from src.logging.error_messages import DatasetError as DSE
from src.logging.log_messages import DatasetLog as DSL
from src.logging.logger import get_logger

logger = get_logger(__name__)


class Dataset:
    """
    Центральная модель данных приложения.
    """

    def __init__(self, data_frame: DataFrame | None = None) -> None:
        self._data_frame: DataFrame | None = (
            data_frame.copy() if data_frame is not None else None
        )
        self._target_column: str | None = None
        self._loader: DataLoader | None = None

        logger.debug(DSL.INIT)

    def __repr__(self) -> str:
        if self._data_frame is None:
            return "Dataset(empty)"

        return f"Dataset(shape={self._data_frame.shape}, target={self._target_column})"

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    def load_from_file(
        self,
        file_path: str | Path,
        **kwargs,
    ) -> Dataset:

        loader = DataLoader().load(file_path, **kwargs)

        self._loader = loader
        self._data_frame = loader.get_data().copy()

        logger.info(DSL.DATA_LOADED, file_path)

        return self

    # --------------------------------------------------------
    # Data access
    # --------------------------------------------------------

    def get_data(self) -> DataFrame:

        if self._data_frame is None:
            raise RuntimeError(DSE.DATASET_EMPTY)

        return self._data_frame.copy()

    @property
    def shape(self) -> tuple[int, int] | None:
        if self._data_frame is None:
            return None
        return self._data_frame.shape

    @property
    def columns(self) -> list[str]:

        if self._data_frame is None:
            raise RuntimeError(DSE.DATASET_EMPTY)

        return list(self._data_frame.columns)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    def set_target(self, column_name: str) -> Dataset:

        if self._data_frame is None:
            raise RuntimeError(DSE.DATASET_EMPTY)

        if column_name not in self._data_frame.columns:
            raise KeyError(DSE.COLUMN_NOT_FOUND.format(col=column_name))

        self._target_column = column_name

        logger.info(DSL.TARGET_SET, column_name)

        return self

    def get_target(self) -> str | None:
        return self._target_column

    # --------------------------------------------------------
    # Features / target
    # --------------------------------------------------------

    def get_X_y(self) -> tuple[DataFrame, Series]:

        if self._data_frame is None:
            raise RuntimeError(DSE.DATASET_EMPTY)

        if self._target_column is None:
            raise RuntimeError(DSE.TARGET_NOT_SET)

        X = self._data_frame.drop(columns=[self._target_column])
        y = cast(Series, self._data_frame[self._target_column])

        return X.copy(), y.copy()

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    def get_numeric_columns(self) -> list[str]:

        if self._data_frame is None:
            return []

        return [
            col
            for col in self._data_frame.columns
            if is_numeric_dtype(self._data_frame[col])
        ]

    def get_categorical_columns(self) -> list[str]:

        if self._data_frame is None:
            return []

        return [
            col
            for col in self._data_frame.columns
            if not is_numeric_dtype(self._data_frame[col])
        ]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:

        if self._data_frame is None:
            return {"loaded": False}

        df = self._data_frame

        return {
            "loaded": True,
            "shape": df.shape,
            "columns": list(df.columns),
            "target": self._target_column,
            "numeric_columns": self.get_numeric_columns(),
            "categorical_columns": self.get_categorical_columns(),
            "missing_values": df.isna().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    def head(self, n: int = 5) -> DataFrame:

        if self._data_frame is None:
            raise RuntimeError(DSE.DATASET_EMPTY)

        return self._data_frame.head(n).copy()

    def tail(self, n: int = 5) -> DataFrame:

        if self._data_frame is None:
            raise RuntimeError(DSE.DATASET_EMPTY)

        return self._data_frame.tail(n).copy()

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update_data(self, new_data: DataFrame) -> Dataset:

        self._data_frame = new_data.copy()

        logger.info(DSL.DATA_UPDATED, self._data_frame.shape)

        return self

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def clear(self) -> Dataset:

        self._data_frame = None
        self._target_column = None
        self._loader = None

        logger.info(DSL.DATASET_CLEARED)

        return self
