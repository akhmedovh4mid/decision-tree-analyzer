from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from src.logging.error_messages import ModelError as MDE
from src.logging.log_messages import ModelLog as MDL
from src.logging.logger import get_logger

logger = get_logger(__name__)

TaskType = Literal["classification", "regression"]


class DecisionTreeModel:
    """
    Универсальная модель дерева решений.

    Поддерживает:
    - классификацию
    - регрессию
    - обучение
    - предсказание
    - оценку качества
    - feature importance
    - визуализацию дерева
    """

    def __init__(
        self,
        task: TaskType = "classification",
        **tree_params,
    ) -> None:

        if task not in ("classification", "regression"):
            raise ValueError(MDE.INVALID_TASK.format(task=task))

        self.task = task
        self.tree_params = tree_params

        self.model: DecisionTreeClassifier | DecisionTreeRegressor | None = None
        self.feature_names: list[str] | None = None
        self.is_fitted: bool = False

        logger.info(MDL.INIT, task)

    def __repr__(self) -> str:
        return f"DecisionTreeModel(task={self.task}, fitted={self.is_fitted})"

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def fit(
        self,
        X: DataFrame,
        y: Series,
    ) -> DecisionTreeModel:

        logger.info(MDL.START_TRAIN)

        try:
            if self.task == "classification":
                self.model = DecisionTreeClassifier(**self.tree_params)

            else:
                self.model = DecisionTreeRegressor(**self.tree_params)

            self.model.fit(X, y)

            self.feature_names = list(X.columns)
            self.is_fitted = True

        except Exception as e:
            raise RuntimeError(MDE.TRAIN_FAILED) from e

        logger.info(MDL.TRAIN_COMPLETE)

        return self

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(
        self,
        X: DataFrame,
    ) -> np.ndarray:

        if not self.is_fitted or self.model is None:
            raise RuntimeError(MDE.MODEL_NOT_FITTED)

        logger.info(MDL.START_PREDICT)

        try:
            preds = self.model.predict(X)

        except Exception as e:
            raise RuntimeError(MDE.PREDICTION_FAILED) from e

        logger.info(MDL.PREDICT_COMPLETE)

        return preds

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    def evaluate(
        self,
        X: DataFrame,
        y_true: Series,
    ) -> dict[str, Any]:

        if not self.is_fitted:
            raise RuntimeError(MDE.MODEL_NOT_FITTED)

        logger.info(MDL.START_EVALUATE)

        try:
            y_pred = self.predict(X)

            if self.task == "classification":
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "f1_score": f1_score(y_true, y_pred, average="weighted"),
                    "classification_report": classification_report(
                        y_true, y_pred, output_dict=True
                    ),
                }

            else:
                metrics = {
                    "r2": r2_score(y_true, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "mae": mean_absolute_error(y_true, y_pred),
                }

        except Exception as e:
            raise RuntimeError(MDE.EVALUATION_FAILED) from e

        logger.info(MDL.EVALUATE_COMPLETE)

        return metrics

    # --------------------------------------------------------
    # Cross validation
    # --------------------------------------------------------

    def cross_validate(
        self,
        X: DataFrame,
        y: Series,
        cv: int = 5,
    ) -> dict[str, float]:

        if self.model is None:
            raise RuntimeError(MDE.MODEL_NOT_FITTED)

        scores = cross_val_score(self.model, X, y, cv=cv)

        return {
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
        }

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    def get_feature_importance(self) -> DataFrame:

        if not self.is_fitted or self.model is None:
            raise RuntimeError(MDE.MODEL_NOT_FITTED)

        if self.feature_names is None:
            raise RuntimeError(MDE.FEATURE_NAMES_REQUIRED)

        importance = self.model.feature_importances_

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        logger.info(MDL.FEATURE_IMPORTANCE)

        return df

    # --------------------------------------------------------
    # Plot tree
    # --------------------------------------------------------

    def plot(
        self,
        figsize: tuple[int, int] = (12, 8),
        max_depth: int | None = None,
    ):

        if not self.is_fitted or self.model is None:
            raise RuntimeError(MDE.MODEL_NOT_FITTED)

        plt.figure(figsize=figsize)

        plot_tree(
            self.model,
            feature_names=self.feature_names,
            filled=True,
            max_depth=max_depth,
        )

        logger.info(MDL.TREE_PLOTTED)

        return plt.gcf()

    # --------------------------------------------------------
    # Model info
    # --------------------------------------------------------

    def get_params(self) -> dict[str, Any]:

        if self.model is None:
            return {}

        return self.model.get_params()

    def depth(self) -> int | None:

        if self.model is None:
            return None

        return self.model.get_depth()

    def n_leaves(self) -> int | None:

        if self.model is None:
            return None

        return self.model.get_n_leaves()
