from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.figure as mplfig
import pandas as pd
from pandas import DataFrame, Series

from src.core.dataset import Dataset
from src.data_io.loader import DataLoader
from src.data_io.saver import DataSaver
from src.logging.error_messages import ControllerError as CE
from src.logging.log_messages import ControllerLog as CL
from src.logging.logger import get_logger
from src.modeling.metrics import Metrics
from src.modeling.tree_model import DecisionTreeModel
from src.modeling.tree_model import TaskType as TreeTaskType
from src.preprocessing.scaler import Scaler, ScaleStrategy
from src.preprocessing.splitter import Splitter
from src.visualization.plots import Plotter
from src.visualization.tree_plotter import TreePlotter

logger = get_logger(__name__)


class Controller:
    """
    Главный контроллер приложения.

    Отвечает за управление полным пайплайном анализа данных:

        загрузка данных
        ↓
        предобработка
        ↓
        разделение train/test
        ↓
        обучение дерева решений
        ↓
        вычисление метрик
        ↓
        визуализация
    """

    def __init__(self) -> None:

        self.dataset: Dataset | None = None
        self.scaler: Scaler | None = None
        self.splitter: Splitter | None = None
        self.model: DecisionTreeModel | None = None

        self.plotter = Plotter()

        logger.info(CL.INIT)

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    def load_data(self, path: str | Path) -> Controller:
        """
        Загрузка датасета.
        """

        df = DataLoader().load(Path(path))

        self.dataset = Dataset(data_frame=df.get_data())

        logger.info(CL.DATA_LOADED, df.shape)

        return self

    def save_data(self, path: str | Path) -> None:
        """
        Сохранение текущего DataFrame.
        """

        if self.dataset is None:
            raise RuntimeError(CE.NO_DATASET)

        DataSaver().save_data(self.dataset.get_data(), path)

        logger.info(CL.DATA_SAVED, path)

    def set_target(self, column: str) -> Controller:
        """
        Установка целевой переменной.
        """

        if self.dataset is None:
            raise RuntimeError(CE.NO_DATASET)

        self.dataset.set_target(column)

        logger.info(CL.TARGET_SET, column)

        return self

    def get_dataframe(self) -> DataFrame:

        if self.dataset is None:
            raise RuntimeError(CE.NO_DATASET)

        return self.dataset.get_data()

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------

    def scale(
        self,
        strategy: ScaleStrategy = "standard",
        columns: list[str] | None = None,
        **kwargs,
    ) -> Controller:
        """
        Масштабирование признаков.
        """

        if self.dataset is None:
            raise RuntimeError(CE.NO_DATASET)

        df = self.dataset.get_data()

        self.scaler = Scaler(df)
        self.scaler.scale(strategy=strategy, columns=columns, **kwargs)

        self.dataset.update_data(self.scaler.get_data())

        logger.info(CL.DATA_SCALED, strategy)

        return self

    # ---------------------------------------------------------
    # SPLIT
    # ---------------------------------------------------------

    def split(
        self,
        test_size: float = 0.2,
        random_state: int | None = None,
        stratify: bool = False,
    ) -> Controller:

        if self.dataset is None:
            raise RuntimeError(CE.NO_DATASET)

        target = self.dataset.get_target()
        if target is None:
            raise RuntimeError(CE.NO_TARGET)

        df = self.dataset.get_data()

        self.splitter = Splitter(df)

        self.splitter.split(
            target_column=target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        logger.info(CL.DATA_SPLIT)

        return self

    def get_train_test(
        self,
    ) -> tuple[tuple[DataFrame, Series], tuple[DataFrame, Series]]:

        if self.splitter is None:
            raise RuntimeError(CE.NOT_SPLIT)

        return self.splitter.get_train_data(), self.splitter.get_test_data()

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    def train_tree(
        self,
        task: TreeTaskType = "classification",
        **params,
    ) -> Controller:
        """
        Обучение дерева решений.
        """

        if self.splitter is None:
            raise RuntimeError(CE.NOT_SPLIT)

        (X_train, y_train) = self.splitter.get_train_data()

        self.model = DecisionTreeModel(task=task, **params)

        self.model.fit(X_train, y_train)

        logger.info(CL.MODEL_TRAINED)

        return self

    def predict(self) -> pd.Series:

        if self.model is None:
            raise RuntimeError(CE.MODEL_NOT_TRAINED)

        if self.splitter is None:
            raise RuntimeError(CE.NOT_SPLIT)

        X_test, _ = self.splitter.get_test_data()

        preds = self.model.predict(X_test)

        logger.info(CL.PREDICTION_DONE)

        return pd.Series(preds, index=X_test.index, name="prediction")

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        """
        Вычисление метрик.
        """

        if self.model is None:
            raise RuntimeError(CE.MODEL_NOT_TRAINED)

        if self.splitter is None:
            raise RuntimeError(CE.NOT_SPLIT)

        _, y_test = self.splitter.get_test_data()

        preds = self.predict()

        results = Metrics.evaluate(y_test, preds)

        logger.info(CL.METRICS_COMPUTED)

        return results

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    def feature_importance(self) -> DataFrame:

        if self.model is None:
            raise RuntimeError(CE.MODEL_NOT_TRAINED)

        assert self.model.model is not None

        importance = self.model.get_feature_importance()

        logger.info(CL.FEATURE_IMPORTANCE)

        return importance

    # ---------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------

    def plot_histogram(self, column: str) -> mplfig.Figure:

        df = self.get_dataframe()

        return self.plotter.histogram(df, column)

    def plot_boxplot(self, column: str) -> mplfig.Figure:

        df = self.get_dataframe()

        return self.plotter.boxplot(df, column)

    def plot_scatter(self, x: str, y: str) -> mplfig.Figure:

        df = self.get_dataframe()

        return self.plotter.scatter(df, x, y)

    def plot_correlation(self) -> mplfig.Figure:

        df = self.get_dataframe()

        return self.plotter.correlation_matrix(df)

    def plot_feature_importance(self) -> mplfig.Figure:

        importance = self.feature_importance()

        return self.plotter.feature_importance(importance)

    # ---------------------------------------------------------
    # TREE VISUALIZATION
    # ---------------------------------------------------------

    def plot_tree(self) -> mplfig.Figure:

        if self.model is None:
            raise RuntimeError(CE.MODEL_NOT_TRAINED)

        assert self.model.model is not None
        assert self.model.feature_names is not None

        plotter = TreePlotter(
            model=self.model.model,
            feature_names=self.model.feature_names,
            class_names=None,
        )

        return plotter.plot()

    def export_tree_text(self) -> str:

        if self.model is None:
            raise RuntimeError(CE.MODEL_NOT_TRAINED)

        assert self.model.model is not None
        assert self.model.feature_names is not None

        plotter = TreePlotter(
            model=self.model.model,
            feature_names=self.model.feature_names,
            class_names=None,
        )

        return plotter.export_text()
