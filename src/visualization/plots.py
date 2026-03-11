from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.logging.error_messages import VisualizationError as VSE
from src.logging.log_messages import VisualizationLog as VSL
from src.logging.logger import get_logger

logger = get_logger(__name__)


class Plotter:
    """
    Класс для построения графиков анализа данных и моделей.
    """

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    @staticmethod
    def _check_dataframe(df: DataFrame) -> None:

        if df is None or df.empty:
            raise ValueError(VSE.DATAFRAME_EMPTY)

    @staticmethod
    def _check_column(df: DataFrame, col: str) -> None:

        if col not in df.columns:
            raise KeyError(VSE.COLUMN_NOT_FOUND.format(col=col))

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    def histogram(
        self,
        df: DataFrame,
        column: str,
        bins: int = 30,
        figsize: tuple[int, int] = (6, 4),
    ):

        self._check_dataframe(df)
        self._check_column(df, column)

        try:
            fig, ax = plt.subplots(figsize=figsize)

            ax.hist(df[column].dropna(), bins=bins)

            ax.set_title(f"Гистограмма: {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Частота")

        except Exception as e:
            raise RuntimeError(VSE.PLOT_FAILED) from e

        logger.info(VSL.HISTOGRAM, column)

        return fig

    # --------------------------------------------------------
    # Boxplot
    # --------------------------------------------------------

    def boxplot(
        self,
        df: DataFrame,
        column: str,
        figsize: tuple[int, int] = (6, 4),
    ):

        self._check_dataframe(df)
        self._check_column(df, column)

        try:
            fig, ax = plt.subplots(figsize=figsize)

            ax.boxplot(df[column].dropna())

            ax.set_title(f"Boxplot: {column}")
            ax.set_ylabel(column)

        except Exception as e:
            raise RuntimeError(VSE.PLOT_FAILED) from e

        logger.info(VSL.BOXPLOT, column)

        return fig

    # --------------------------------------------------------
    # Scatter plot
    # --------------------------------------------------------

    def scatter(
        self,
        df: DataFrame,
        x: str,
        y: str,
        figsize: tuple[int, int] = (6, 4),
    ):

        self._check_dataframe(df)
        self._check_column(df, x)
        self._check_column(df, y)

        try:
            fig, ax = plt.subplots(figsize=figsize)

            ax.scatter(df[x], df[y])

            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{x} vs {y}")

        except Exception as e:
            raise RuntimeError(VSE.PLOT_FAILED) from e

        logger.info(VSL.SCATTER, x, y)

        return fig

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    def correlation_matrix(
        self,
        df: DataFrame,
        figsize: tuple[int, int] = (8, 6),
    ):

        self._check_dataframe(df)

        try:
            corr = df.corr(numeric_only=True)

            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(corr)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(
                        j,
                        i,
                        f"{corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

            fig.colorbar(im)

        except Exception as e:
            raise RuntimeError(VSE.PLOT_FAILED) from e

        logger.info(VSL.CORRELATION)

        return fig

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    def feature_importance(
        self,
        importance_df: DataFrame,
        top_n: int | None = None,
        figsize: tuple[int, int] = (8, 5),
    ):

        if importance_df.empty:
            raise ValueError(VSE.DATAFRAME_EMPTY)

        try:
            df = importance_df.copy()

            if top_n is not None:
                df = df.head(top_n)

            fig, ax = plt.subplots(figsize=figsize)

            ax.barh(df["feature"], df["importance"])

            ax.set_xlabel("Важность")
            ax.set_title("Важность признаков")

            ax.invert_yaxis()

        except Exception as e:
            raise RuntimeError(VSE.PLOT_FAILED) from e

        logger.info(VSL.FEATURE_IMPORTANCE)

        return fig

    # --------------------------------------------------------
    # Confusion matrix plot
    # --------------------------------------------------------

    def confusion_matrix(
        self,
        cm: DataFrame | np.ndarray,
        labels: list[str] | None = None,
        figsize: tuple[int, int] = (6, 5),
    ):

        try:
            if isinstance(cm, pd.DataFrame):
                matrix = cm.values
            else:
                matrix = cm

            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(matrix)

            if labels is not None:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))

                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, matrix[i, j], ha="center", va="center")

            ax.set_xlabel("Предсказанный класс")
            ax.set_ylabel("Истинный класс")

            fig.colorbar(im)

        except Exception as e:
            raise RuntimeError(VSE.PLOT_FAILED) from e

        logger.info(VSL.CONFUSION_MATRIX)

        return fig
