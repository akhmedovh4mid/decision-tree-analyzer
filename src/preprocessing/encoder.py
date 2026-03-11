from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.logging.error_messages import EncoderError as EE
from src.logging.log_messages import EncoderLog as EL
from src.logging.logger import get_logger

logger = get_logger(name=__name__)

EncodeStrategy = Literal[
    "onehot",
    "label",
    "ordinal",
    "frequency",
    "target",
]

EncoderType = OneHotEncoder | OrdinalEncoder | dict[Any, Any]


class Encoder:
    """
    Класс для кодирования категориальных признаков.

    Поддерживаемые стратегии:
    - onehot
    - label
    - ordinal
    - frequency
    - target

    Методы chainable.
    """

    def __init__(self, data_frame: pd.DataFrame) -> None:
        self.data_frame = data_frame.copy()

        self.encoders_: dict[str, EncoderType] = {}
        self.encoded_columns_: dict[str, list[str]] = {}

        logger.debug(EL.INIT, self.data_frame.shape)

    def __repr__(self) -> str:
        return f"Encoder(shape={self.data_frame.shape})"

    def copy(self) -> "Encoder":
        new = Encoder(self.data_frame.copy())
        new.encoders_ = self.encoders_.copy()
        new.encoded_columns_ = self.encoded_columns_.copy()
        return new

    @property
    def shape(self) -> tuple[int, int]:
        return self.data_frame.shape

    def get_data(self) -> pd.DataFrame:
        return self.data_frame

    # --------------------------------------------------------
    # Utils
    # --------------------------------------------------------

    @staticmethod
    def _to_numpy(series: pd.Series | pd.DataFrame) -> Any:
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        return series.fillna("__NaN__").astype(str).to_numpy().reshape(-1, 1)

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    def encode(
        self,
        strategy: EncodeStrategy,
        columns: Sequence[str] | None = None,
        *,
        target_column: str | None = None,
        categories: dict[str, list[Any]] | None = None,
        handle_unknown: Literal["error", "use_encoded_value"] = "error",
        unknown_value: int | float = -1,
        **kwargs,
    ) -> "Encoder":

        if columns is None:
            columns = self.data_frame.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        df_cols = set(self.data_frame.columns)
        cols = [c for c in columns if c in df_cols]

        missing = set(columns) - set(cols)
        for col in missing:
            logger.warning(EL.COLUMN_NOT_FOUND, col)

        if not cols:
            logger.warning(EE.NO_COLUMNS_TO_ENCODE)
            return self

        logger.info(EL.START_ENCODE, strategy, cols)

        if strategy == "onehot":
            self._fit_onehot(cols, **kwargs)
            self._transform_onehot(cols)

        elif strategy == "label":
            self._fit_label(cols, handle_unknown, unknown_value, **kwargs)
            self._transform_label(cols)

        elif strategy == "ordinal":
            if categories is None:
                raise ValueError(EE.ORDINAL_CATEGORIES_REQUIRED)

            self._fit_ordinal(cols, categories, handle_unknown, unknown_value, **kwargs)
            self._transform_ordinal(cols)

        elif strategy == "frequency":
            self._fit_frequency(cols)
            self._transform_frequency(cols)

        elif strategy == "target":
            if target_column is None:
                raise ValueError(EE.TARGET_COLUMN_REQUIRED)

            if target_column not in self.data_frame.columns:
                raise KeyError(EE.TARGET_COLUMN_NOT_FOUND.format(col=target_column))

            self._fit_target(cols, target_column)
            self._transform_target(cols)

        else:
            raise ValueError(EE.UNKNOWN_STRATEGY.format(strategy=strategy))

        logger.info(EL.ENCODE_COMPLETE, strategy)

        return self

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def _fit_onehot(self, columns: Sequence[str], **kwargs) -> None:

        for col in columns:
            data = self._to_numpy(self.data_frame[col])

            enc = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                **kwargs,
            )

            enc.fit(data)

            self.encoders_[col] = enc

            logger.debug(EL.FITTED_ONEHOT, col)

    def _fit_label(
        self,
        columns: Sequence[str],
        handle_unknown: str,
        unknown_value: int | float,
        **kwargs,
    ) -> None:

        for col in columns:
            data = self._to_numpy(self.data_frame[col])

            params: dict[str, Any] = dict(handle_unknown=handle_unknown, **kwargs)

            if handle_unknown == "use_encoded_value":
                params["unknown_value"] = unknown_value

            enc = OrdinalEncoder(**params)

            enc.fit(data)

            self.encoders_[col] = enc

            logger.debug(EL.FITTED_LABEL, col)

    def _fit_ordinal(
        self,
        columns: Sequence[str],
        categories: dict[str, list[Any]],
        handle_unknown: str,
        unknown_value: int | float,
        **kwargs,
    ) -> None:

        for col in columns:
            if col not in categories:
                raise KeyError(EE.CATEGORIES_NOT_FOUND.format(col=col))

            data = self._to_numpy(self.data_frame[col])

            cats: list[list[Any]] = [categories[col]]

            params: dict[str, Any] = dict(
                categories=cast(Any, cats),
                handle_unknown=handle_unknown,
                **kwargs,
            )

            if handle_unknown == "use_encoded_value":
                params["unknown_value"] = unknown_value

            enc = OrdinalEncoder(**params)

            enc.fit(data)

            self.encoders_[col] = enc

            logger.debug(EL.FITTED_ORDINAL, col)

    def _fit_frequency(self, columns: Sequence[str]) -> None:

        for col in columns:
            freq_series = self.data_frame[col].value_counts(normalize=True)

            freq = dict(freq_series)

            self.encoders_[col] = freq

            logger.debug(EL.FITTED_FREQUENCY, col)

    def _fit_target(self, columns: Sequence[str], target_column: str) -> None:

        for col in columns:
            series = self.data_frame.groupby(col)[target_column].mean()

            mapping = cast(pd.Series, series).to_dict()

            self.encoders_[col] = mapping

            logger.debug(EL.FITTED_TARGET, col)

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    def _transform_onehot(self, columns: Sequence[str]) -> None:

        for col in columns:
            enc = cast(OneHotEncoder | None, self.encoders_.get(col))

            if enc is None:
                raise RuntimeError(EE.ENCODER_NOT_FITTED.format(col=col))

            data = self._to_numpy(self.data_frame[col])

            encoded = enc.transform(data)

            categories = enc.categories_[0]

            new_cols = [f"{col}_{cat}" for cat in categories]

            encoded_df = pd.DataFrame(
                encoded,
                columns=new_cols,
                index=self.data_frame.index,
            ).astype(int)

            self.encoded_columns_[col] = new_cols

            self.data_frame = pd.concat(
                [self.data_frame.drop(columns=[col]), encoded_df],
                axis=1,
            )

            logger.debug(EL.TRANSFORM_ONEHOT, col, len(new_cols))

    def _transform_label(self, columns: Sequence[str]) -> None:

        for col in columns:
            enc = cast(OrdinalEncoder | None, self.encoders_.get(col))

            if enc is None:
                raise RuntimeError(EE.ENCODER_NOT_FITTED.format(col=col))

            data = self._to_numpy(self.data_frame[col])

            encoded = enc.transform(data).flatten()

            self.data_frame[col] = encoded

            logger.debug(EL.TRANSFORM_LABEL, col)

    def _transform_ordinal(self, columns: Sequence[str]) -> None:
        self._transform_label(columns)

    def _transform_frequency(self, columns: Sequence[str]) -> None:

        for col in columns:
            mapping = cast(dict[Any, Any] | None, self.encoders_.get(col))

            if mapping is None:
                raise RuntimeError(EE.ENCODER_NOT_FITTED.format(col=col))

            self.data_frame[col] = self.data_frame[col].replace(mapping)

            logger.debug(EL.TRANSFORM_FREQUENCY, col)

    def _transform_target(self, columns: Sequence[str]) -> None:

        for col in columns:
            mapping = cast(dict[Any, Any] | None, self.encoders_.get(col))

            if mapping is None:
                raise RuntimeError(EE.ENCODER_NOT_FITTED.format(col=col))

            self.data_frame[col] = self.data_frame[col].replace(mapping)

            logger.debug(EL.TRANSFORM_TARGET, col)

    # --------------------------------------------------------
    # Apply to new dataset
    # --------------------------------------------------------

    def transform(self, new_data: pd.DataFrame) -> pd.DataFrame:

        result = new_data.copy()

        for col, enc in self.encoders_.items():
            if col not in result.columns:
                logger.warning(EL.COLUMN_MISSING_FOR_TRANSFORM, col)
                continue

            if isinstance(enc, OneHotEncoder):
                data = self._to_numpy(result[col])

                encoded = enc.transform(data)

                categories = enc.categories_[0]

                new_cols = [f"{col}_{cat}" for cat in categories]

                encoded_df = pd.DataFrame(
                    encoded,
                    columns=new_cols,
                    index=result.index,
                ).astype(int)

                result = pd.concat(
                    [result.drop(columns=[col]), encoded_df],
                    axis=1,
                )

            elif isinstance(enc, OrdinalEncoder):
                data = self._to_numpy(result[col])

                result[col] = enc.transform(data).flatten()

            elif isinstance(enc, dict):
                result[col] = result[col].replace(enc)

            else:
                logger.warning(EL.UNKNOWN_ENCODER_TYPE, col, type(enc))

        return result

    # --------------------------------------------------------
    # Inverse transform
    # --------------------------------------------------------

    def inverse_transform(
        self,
        encoded_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:

        df = self.data_frame.copy() if encoded_data is None else encoded_data.copy()

        for col, enc in self.encoders_.items():
            if isinstance(enc, OneHotEncoder):
                bin_cols = self.encoded_columns_.get(col)

                if not bin_cols:
                    continue

                if not all(c in df.columns for c in bin_cols):
                    logger.warning(EL.MISSING_BIN_COLS, col)
                    continue

                X = df[bin_cols].to_numpy()

                original = enc.inverse_transform(X).flatten()

                df[col] = original

                df = df.drop(columns=bin_cols)

            elif isinstance(enc, OrdinalEncoder):
                if col not in df.columns:
                    continue

                data = df[col].to_numpy().reshape(-1, 1)

                original = enc.inverse_transform(data).flatten()

                df[col] = original

            elif isinstance(enc, dict):
                logger.warning(EL.INVERSE_NOT_POSSIBLE, col)

        return df

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    def reset_index(self, drop: bool = True) -> "Encoder":

        self.data_frame = self.data_frame.reset_index(drop=drop)

        return self
