from __future__ import annotations

import json
import pickle

# ------------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------------
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pandas import DataFrame

joblib: typing.Any | None = None

try:
    import joblib as _joblib

    joblib = _joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

try:
    import matplotlib  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

from src.logging.error_messages import SaverError as SVE  # noqa: E402
from src.logging.log_messages import SaverLog as SVL  # noqa: E402
from src.logging.logger import get_logger  # noqa: E402

logger = get_logger(name=__name__)

# ------------------------------------------------------------------
# Types
# ------------------------------------------------------------------

DataFormat = Literal["csv", "excel", "json", "parquet"]
ImageFormat = Literal["png", "jpg", "jpeg", "pdf"]

_DATA_SUFFIX_MAP: dict[str, DataFormat] = {
    ".csv": "csv",
    ".xls": "excel",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".json": "json",
    ".parquet": "parquet",
}

# ------------------------------------------------------------------
# DataSaver
# ------------------------------------------------------------------


class DataSaver:
    """
    Универсальный класс для сохранения:
    - DataFrame
    - ML моделей
    - отчётов
    - графиков
    """

    def __init__(self) -> None:
        self.last_saved_path: Path | None = None
        logger.debug(SVL.INIT)

    def __repr__(self) -> str:
        return (
            f"DataSaver(last_saved={self.last_saved_path})"
            if self.last_saved_path
            else "DataSaver(no saves yet)"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_path(self, path: Path, overwrite: bool) -> None:
        """Проверяет файл и создаёт директории."""
        if path.exists() and not overwrite:
            raise FileExistsError(SVE.FILE_EXISTS.format(path=path))

        path.parent.mkdir(parents=True, exist_ok=True)

    def _detect_data_format(self, path: Path) -> DataFormat:
        suffix = path.suffix.lower()

        if suffix not in _DATA_SUFFIX_MAP:
            raise ValueError(SVE.UNSUPPORTED_FORMAT.format(suffix=suffix))

        return _DATA_SUFFIX_MAP[suffix]

    # ------------------------------------------------------------------
    # Save dataframe
    # ------------------------------------------------------------------

    def save_data(
        self,
        data_frame: DataFrame,
        file_path: str | Path,
        format: DataFormat | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> DataSaver:

        path = Path(file_path)

        self._prepare_path(path, overwrite)

        fmt = format or self._detect_data_format(path)

        logger.info(SVL.START_SAVE_DATA, path, fmt)

        try:
            if fmt == "csv":
                data_frame.to_csv(path, **kwargs)

            elif fmt == "excel":
                data_frame.to_excel(path, **kwargs)

            elif fmt == "json":
                data_frame.to_json(path, **kwargs)

            elif fmt == "parquet":
                data_frame.to_parquet(path, **kwargs)

        except Exception as e:
            raise RuntimeError(SVE.SAVE_FAILED.format(path=path)) from e

        self.last_saved_path = path
        logger.info(SVL.SAVE_SUCCESS, path)

        return self

    # ------------------------------------------------------------------
    # Shorthand methods
    # ------------------------------------------------------------------

    def to_csv(self, df: DataFrame, path: str | Path, **kw) -> DataSaver:
        return self.save_data(df, path, "csv", **kw)

    def to_excel(self, df: DataFrame, path: str | Path, **kw) -> DataSaver:
        return self.save_data(df, path, "excel", **kw)

    def to_json(self, df: DataFrame, path: str | Path, **kw) -> DataSaver:
        return self.save_data(df, path, "json", **kw)

    def to_parquet(self, df: DataFrame, path: str | Path, **kw) -> DataSaver:
        return self.save_data(df, path, "parquet", **kw)

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: Any,
        file_path: str | Path,
        overwrite: bool = False,
        use_joblib: bool = True,
    ) -> DataSaver:

        path = Path(file_path)

        self._prepare_path(path, overwrite)

        logger.info(SVL.START_SAVE_MODEL, path)

        try:
            if use_joblib and JOBLIB_AVAILABLE and joblib is not None:
                joblib.dump(model, path)
                logger.debug(SVL.MODEL_SAVED_JOBLIB)

            else:
                with open(path, "wb") as f:
                    pickle.dump(model, f)

                logger.debug(SVL.MODEL_SAVED_PICKLE)

        except Exception as e:
            raise RuntimeError(SVE.SAVE_FAILED.format(path=path)) from e

        self.last_saved_path = path
        logger.info(SVL.SAVE_SUCCESS, path)

        return self

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------

    def save_report(
        self,
        report: dict[str, Any],
        file_path: str | Path,
        format: Literal["json", "txt"] = "json",
        overwrite: bool = False,
    ) -> DataSaver:

        path = Path(file_path)

        self._prepare_path(path, overwrite)

        logger.info(SVL.START_SAVE_REPORT, path, format)

        try:
            if format == "json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)

            elif format == "txt":
                with open(path, "w", encoding="utf-8") as f:
                    for k, v in report.items():
                        f.write(f"{k}: {v}\n")

            else:
                raise ValueError(SVE.UNSUPPORTED_REPORT_FORMAT.format(fmt=format))

        except Exception as e:
            raise RuntimeError(SVE.SAVE_FAILED.format(path=path)) from e

        self.last_saved_path = path
        logger.info(SVL.SAVE_SUCCESS, path)

        return self

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------

    def save_figure(
        self,
        figure: Figure,
        file_path: str | Path,
        format: ImageFormat | None = None,
        overwrite: bool = False,
        dpi: int = 300,
        bbox_inches: str = "tight",
        **kwargs,
    ) -> DataSaver:

        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(SVE.MATPLOTLIB_NOT_AVAILABLE)

        path = Path(file_path)

        self._prepare_path(path, overwrite)

        fmt = format or path.suffix.lower().lstrip(".")

        if fmt not in {"png", "jpg", "jpeg", "pdf"}:
            raise ValueError(SVE.UNSUPPORTED_IMAGE_FORMAT.format(suffix=fmt))

        logger.info(SVL.START_SAVE_FIGURE, path, fmt)

        try:
            figure.savefig(
                path,
                format=fmt,
                dpi=dpi,
                bbox_inches=bbox_inches,
                **kwargs,
            )

        except Exception as e:
            raise RuntimeError(SVE.SAVE_FAILED.format(path=path)) from e

        self.last_saved_path = path
        logger.info(SVL.SAVE_SUCCESS, path)

        return self

    # ------------------------------------------------------------------
    # Reset last saved
    # ------------------------------------------------------------------

    def clear_last(self) -> DataSaver:
        self.last_saved_path = None
        logger.debug(SVL.CLEAR_LAST)
        return self
