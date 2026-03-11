from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_text,
    plot_tree,
)

from src.logging.error_messages import TreePlotterError as TPE
from src.logging.log_messages import TreePlotterLog as TPL
from src.logging.logger import get_logger

logger = get_logger(__name__)


class TreePlotter:
    """
    Класс для визуализации дерева решений.

    Поддерживает:
    - построение графического дерева
    - экспорт текстового представления
    """

    def __init__(
        self,
        model: DecisionTreeClassifier | DecisionTreeRegressor,
        feature_names: List[str],
        class_names: List[str] | None = None,
    ) -> None:

        if model is None:
            raise ValueError(TPE.MODEL_NOT_TRAINED)

        if not feature_names:
            raise ValueError(TPE.FEATURE_NAMES_REQUIRED)

        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names

    # -------------------------------------------------------
    # Plot tree
    # -------------------------------------------------------

    def plot(
        self,
        max_depth: int | None = None,
        figsize: tuple[int, int] = (16, 8),
        filled: bool = True,
        rounded: bool = True,
        fontsize: int = 10,
    ):
        """
        Построение графического дерева решений.

        Returns
        -------
        matplotlib.figure.Figure
        """

        try:
            fig, ax = plt.subplots(figsize=figsize)

            plot_tree(
                self.model,
                feature_names=self.feature_names,
                class_names=self.class_names,
                filled=filled,
                rounded=rounded,
                max_depth=max_depth,
                fontsize=fontsize,
                ax=ax,
            )

            ax.set_title("Дерево решений")

        except Exception as e:
            raise RuntimeError(TPE.PLOT_FAILED) from e

        logger.info(TPL.TREE_PLOT_CREATED)

        return fig

    # -------------------------------------------------------
    # Text representation
    # -------------------------------------------------------

    def export_text(
        self,
        max_depth: int | None = None,
    ) -> str:
        """
        Возвращает текстовое представление дерева.

        Returns
        -------
        str
        """

        try:
            if max_depth is None:
                text = export_text(
                    self.model,
                    feature_names=self.feature_names,
                )
            else:
                text = export_text(
                    self.model,
                    feature_names=self.feature_names,
                    max_depth=max_depth,
                )

        except Exception as e:
            raise RuntimeError(TPE.PLOT_FAILED) from e

        logger.info(TPL.TREE_EXPORT_TEXT)

        return text
