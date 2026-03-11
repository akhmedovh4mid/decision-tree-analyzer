from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

from src.logging.error_messages import EvaluatorError as EVE
from src.logging.log_messages import EvaluatorLog as EVL
from src.logging.logger import get_logger

logger = get_logger(__name__)

TaskType = Literal["classification", "regression"]


class Evaluator:
    """
    Класс для оценки качества ML моделей.

    Поддерживает:
    - классификацию
    - регрессию
    - вычисление метрик
    - confusion matrix
    - cross-validation
    - визуализацию результатов
    """

    def __init__(self, task: TaskType = "classification") -> None:

        if task not in ("classification", "regression"):
            raise ValueError(EVE.INVALID_TASK.format(task=task))

        self.task = task

        logger.info(EVL.INIT, task)

    def __repr__(self) -> str:
        return f"Evaluator(task={self.task})"

    # --------------------------------------------------------
    # Основная оценка
    # --------------------------------------------------------

    def evaluate(
        self,
        y_true: Series | np.ndarray,
        y_pred: Series | np.ndarray,
    ) -> dict[str, Any]:

        logger.info(EVL.START_EVALUATION)

        try:
            if self.task == "classification":
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(
                        y_true, y_pred, average="weighted", zero_division="warn"
                    ),
                    "recall": recall_score(
                        y_true, y_pred, average="weighted", zero_division="warn"
                    ),
                    "f1_score": f1_score(
                        y_true, y_pred, average="weighted", zero_division="warn"
                    ),
                    "classification_report": classification_report(
                        y_true, y_pred, output_dict=True
                    ),
                }

            else:
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

                metrics = {
                    "r2": r2_score(y_true, y_pred),
                    "rmse": rmse,
                    "mae": mean_absolute_error(y_true, y_pred),
                    "mse": mean_squared_error(y_true, y_pred),
                }

        except Exception as e:
            raise RuntimeError(EVE.EVALUATION_FAILED) from e

        logger.info(EVL.EVALUATION_COMPLETE)

        return metrics

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    def confusion_matrix(
        self,
        y_true: Series | np.ndarray,
        y_pred: Series | np.ndarray,
    ) -> DataFrame:

        if self.task != "classification":
            raise ValueError(EVE.INVALID_TASK.format(task=self.task))

        cm = confusion_matrix(y_true, y_pred)

        df = pd.DataFrame(cm)

        logger.info(EVL.CONFUSION_MATRIX)

        return df

    # --------------------------------------------------------
    # Cross validation
    # --------------------------------------------------------

    def cross_validate(
        self,
        model: Any,
        X: DataFrame,
        y: Series,
        cv: int = 5,
        scoring: str | None = None,
    ) -> dict[str, float]:

        try:
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
            )

        except Exception as e:
            raise RuntimeError(EVE.EVALUATION_FAILED) from e

        logger.info(EVL.CROSS_VALIDATION)

        return {
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "scores": scores.tolist(),
        }

    # --------------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: Series | np.ndarray,
        y_pred: Series | np.ndarray,
        figsize: tuple[int, int] = (6, 5),
    ):

        if self.task != "classification":
            raise ValueError(EVE.INVALID_TASK.format(task=self.task))

        try:
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=figsize)

            ax.imshow(cm)

            ax.set_xlabel("Предсказанный класс")
            ax.set_ylabel("Истинный класс")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            logger.info(EVL.PLOT_CREATED)

            return fig

        except Exception as e:
            raise RuntimeError(EVE.PLOT_FAILED) from e

    # --------------------------------------------------------
    # Regression plot
    # --------------------------------------------------------

    def plot_regression(
        self,
        y_true: Series | np.ndarray,
        y_pred: Series | np.ndarray,
        figsize: tuple[int, int] = (6, 5),
    ):

        if self.task != "regression":
            raise ValueError(EVE.INVALID_TASK.format(task=self.task))

        try:
            fig, ax = plt.subplots(figsize=figsize)

            ax.scatter(y_true, y_pred)

            ax.set_xlabel("Истинные значения")
            ax.set_ylabel("Предсказанные значения")

            logger.info(EVL.PLOT_CREATED)

            return fig

        except Exception as e:
            raise RuntimeError(EVE.PLOT_FAILED) from e
