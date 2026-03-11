from __future__ import annotations

from typing import cast

from pandas import DataFrame, Index, Series
from sklearn.model_selection import train_test_split

from src.logging.error_messages import SplitterError as SPE
from src.logging.log_messages import SplitterLog as SPL
from src.logging.logger import get_logger

logger = get_logger(name=__name__)


class Splitter:
    """
    Класс для разделения данных на обучающую и тестовую выборки.
    Сохраняет индексы и сами подвыборки.
    Методы chainable.
    """

    def __init__(self, data_frame: DataFrame) -> None:
        self.data_frame = data_frame.copy()
        self.X_train: DataFrame | None = None
        self.X_test: DataFrame | None = None
        self.y_train: Series | None = None
        self.y_test: Series | None = None
        self.train_indices: Index | None = None
        self.test_indices: Index | None = None
        logger.debug(SPL.INIT, self.data_frame.shape)

    def __repr__(self) -> str:
        return f"Splitter(shape={self.data_frame.shape})"

    def copy(self) -> Splitter:
        """Возвращает копию текущего экземпляра."""
        new = Splitter(self.data_frame.copy())
        new.X_train = self.X_train.copy() if self.X_train is not None else None
        new.X_test = self.X_test.copy() if self.X_test is not None else None
        new.y_train = self.y_train.copy() if self.y_train is not None else None
        new.y_test = self.y_test.copy() if self.y_test is not None else None
        new.train_indices = (
            self.train_indices.copy() if self.train_indices is not None else None
        )
        new.test_indices = (
            self.test_indices.copy() if self.test_indices is not None else None
        )
        return new

    @property
    def shape(self) -> tuple[int, int]:
        return self.data_frame.shape

    def get_data(self) -> DataFrame:
        """Возвращает исходный DataFrame."""
        return self.data_frame

    def get_splits(self) -> dict[str, DataFrame | Series | Index | None]:
        """Возвращает словарь со всеми разделёнными данными."""
        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
        }

    # ------------------------------------------------------------------
    # Основной метод разделения
    # ------------------------------------------------------------------

    def split(
        self,
        target_column: str,
        test_size: float = 0.2,
        train_size: float | None = None,
        random_state: int | None = None,
        shuffle: bool = True,
        stratify: bool = False,
    ) -> Splitter:
        """
        Разделяет данные на обучающую и тестовую выборки.

        Args:
            target_column: Имя целевой колонки.
            test_size: Доля тестовой выборки (если float) или абсолютное число (если int).
            train_size: Доля обучающей выборки. Если None, дополнение до test_size.
            random_state: Seed для воспроизводимости.
            shuffle: Перемешивать ли данные перед разделением.
            stratify: Если True, выполнить стратифицированное разделение по целевой колонке.
                      Для регрессии не применяется.

        Returns:
            self
        """
        if target_column not in self.data_frame.columns:
            raise KeyError(SPE.TARGET_COLUMN_NOT_FOUND.format(col=target_column))

        # Отделяем признаки от целевой переменной
        X = self.data_frame.drop(columns=[target_column])
        y = self.data_frame[target_column]

        # Определяем stratify параметр для sklearn
        stratify_param = y if stratify else None

        # Выполняем разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_param,
        )

        self.X_train = cast(DataFrame, X_train)
        self.X_test = cast(DataFrame, X_test)
        self.y_train = cast(Series, y_train)
        self.y_test = cast(Series, y_test)

        # Сохраняем индексы
        self.train_indices = self.X_train.index
        self.test_indices = self.X_test.index

        logger.info(
            SPL.SPLIT_COMPLETE,
            len(self.X_train),
            len(self.X_test),
            target_column,
        )

        return self

    # ------------------------------------------------------------------
    # Дополнительные методы для получения отдельных частей
    # ------------------------------------------------------------------

    def get_train_data(self) -> tuple[DataFrame, Series]:
        """Возвращает (X_train, y_train)."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError(SPE.NOT_SPLIT_YET)
        return self.X_train, self.y_train

    def get_test_data(self) -> tuple[DataFrame, Series]:
        """Возвращает (X_test, y_test)."""
        if self.X_test is None or self.y_test is None:
            raise RuntimeError(SPE.NOT_SPLIT_YET)
        return self.X_test, self.y_test

    def get_train_indices(self) -> Index:
        """Возвращает индексы обучающей выборки."""
        if self.train_indices is None:
            raise RuntimeError(SPE.NOT_SPLIT_YET)
        return self.train_indices

    def get_test_indices(self) -> Index:
        """Возвращает индексы тестовой выборки."""
        if self.test_indices is None:
            raise RuntimeError(SPE.NOT_SPLIT_YET)
        return self.test_indices

    # ------------------------------------------------------------------
    # Сброс индекса (для цепочек)
    # ------------------------------------------------------------------

    def reset_index(self, drop: bool = True) -> Splitter:
        """Сброс индекса исходного DataFrame."""
        self.data_frame.reset_index(drop=drop, inplace=True)
        return self
