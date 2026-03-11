from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pandas import DataFrame

from src.logging.error_messages import LoaderError as LDE
from src.logging.log_messages import LoaderLog as LDL
from src.logging.logger import get_logger

logger = get_logger(name=__name__)


FileFormat = Literal["csv", "excel", "json", "parquet"]


class DataLoader:
    """
    Класс для загрузки данных из файлов различных форматов.
    Поддерживает CSV, Excel, JSON, Parquet.
    Сохраняет загруженный DataFrame и метаинформацию о файле.
    Методы chainable.
    """

    def __init__(self) -> None:
        self.data_frame: DataFrame | None = None
        self.file_path: Path | None = None
        self.file_format: FileFormat | None = None
        logger.debug(LDL.INIT)

    def __repr__(self) -> str:
        if self.data_frame is not None:
            return f"DataLoader(shape={self.data_frame.shape}, file={self.file_path})"
        return "DataLoader(no data loaded)"

    @property
    def shape(self) -> tuple[int, int] | None:
        """Возвращает размер загруженного DataFrame или None."""
        return self.data_frame.shape if self.data_frame is not None else None

    def get_data(self) -> DataFrame:
        """Возвращает загруженный DataFrame. Если данные не загружены, выбрасывает исключение."""
        if self.data_frame is None:
            raise RuntimeError(LDE.NO_DATA_LOADED)
        return self.data_frame

    # ------------------------------------------------------------------
    # Основные методы загрузки
    # ------------------------------------------------------------------

    def load(
        self,
        file_path: str | Path,
        format: FileFormat | None = None,
        **kwargs,
    ) -> DataLoader:
        """
        Загружает данные из файла. Формат определяется по расширению, если не указан явно.

        Args:
            file_path: Путь к файлу.
            format: Явное указание формата (csv, excel, json, parquet).
            **kwargs: Дополнительные параметры, передаваемые в функцию pandas (например, sep, encoding).

        Returns:
            self
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(LDE.FILE_NOT_FOUND.format(path=path))

        # Определяем формат
        if format is None:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                fmt = "csv"
            elif suffix in (".xls", ".xlsx", ".xlsm"):
                fmt = "excel"
            elif suffix == ".json":
                fmt = "json"
            elif suffix == ".parquet":
                fmt = "parquet"
            else:
                raise ValueError(LDE.UNSUPPORTED_FORMAT.format(suffix=suffix))
        else:
            fmt = format

        logger.info(LDL.START_LOAD, path, fmt)

        # Выбор метода загрузки
        try:
            if fmt == "csv":
                self.data_frame = pd.read_csv(path, **kwargs)
            elif fmt == "excel":
                self.data_frame = pd.read_excel(path, **kwargs)
            elif fmt == "json":
                self.data_frame = pd.read_json(path, **kwargs)
            elif fmt == "parquet":
                self.data_frame = pd.read_parquet(path, **kwargs)
            else:
                # Недостижимо, если fmt корректен
                raise ValueError(LDE.UNSUPPORTED_FORMAT.format(suffix=fmt))
        except Exception as e:
            raise RuntimeError(LDE.LOAD_FAILED.format(path=path)) from e

        self.file_path = path
        self.file_format = fmt

        assert self.data_frame is not None
        logger.info(LDL.LOAD_SUCCESS, self.data_frame.shape)
        return self

    # ------------------------------------------------------------------
    # Специализированные методы для удобства
    # ------------------------------------------------------------------

    def load_csv(self, file_path: str | Path, **kwargs) -> DataLoader:
        """Загружает CSV-файл."""
        return self.load(file_path, format="csv", **kwargs)

    def load_excel(self, file_path: str | Path, **kwargs) -> DataLoader:
        """Загружает Excel-файл."""
        return self.load(file_path, format="excel", **kwargs)

    def load_json(self, file_path: str | Path, **kwargs) -> DataLoader:
        """Загружает JSON-файл."""
        return self.load(file_path, format="json", **kwargs)

    def load_parquet(self, file_path: str | Path, **kwargs) -> DataLoader:
        """Загружает Parquet-файл."""
        return self.load(file_path, format="parquet", **kwargs)

    # ------------------------------------------------------------------
    # Информационные методы
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Возвращает словарь с информацией о загруженных данных."""
        if self.data_frame is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "file_path": str(self.file_path),
            "format": self.file_format,
            "shape": self.data_frame.shape,
            "columns": list(self.data_frame.columns),
            "dtypes": self.data_frame.dtypes.to_dict(),
        }

    def head(self, n: int = 5) -> DataFrame:
        """Возвращает первые n строк загруженного DataFrame."""
        if self.data_frame is None:
            raise RuntimeError(LDE.NO_DATA_LOADED)
        return self.data_frame.head(n)

    # ------------------------------------------------------------------
    # Сброс (очистка)
    # ------------------------------------------------------------------

    def clear(self) -> DataLoader:
        """Очищает загруженные данные."""
        self.data_frame = None
        self.file_path = None
        self.file_format = None
        logger.debug(LDL.CLEAR)
        return self
