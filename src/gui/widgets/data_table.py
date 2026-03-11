from __future__ import annotations

import pandas as pd
from pandas import DataFrame
from PyQt5 import QtCore
from PyQt5.QtCore import QAbstractTableModel, QModelIndex
from PyQt5.QtWidgets import QTableView


class PandasTableModel(QAbstractTableModel):
    """
    Модель Qt для отображения pandas DataFrame.
    """

    def __init__(self, data: DataFrame | None = None) -> None:
        super().__init__()
        self._data = data if data is not None else pd.DataFrame()

    # -------------------------------------------------------
    # Размеры таблицы
    # -------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._data.columns)

    # -------------------------------------------------------
    # Данные ячеек
    # -------------------------------------------------------

    def data(
        self,
        index: QModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):

        if not index.isValid():
            return None

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]

            if pd.isna(value):
                return ""

            return str(value)

        return None

    # -------------------------------------------------------
    # Заголовки
    # -------------------------------------------------------

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):

        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None

        try:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self._data.columns[section])

            if orientation == QtCore.Qt.Orientation.Vertical:
                return str(self._data.index[section])

        except IndexError, KeyError:
            return None

        return None

    # -------------------------------------------------------
    # Флаги ячеек
    # -------------------------------------------------------

    def flags_(self, index: QModelIndex):

        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags

        return QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled

    # -------------------------------------------------------
    # Сортировка
    # -------------------------------------------------------

    def sort(
        self,
        column: int,
        order: QtCore.Qt.SortOrder = QtCore.Qt.SortOrder.AscendingOrder,
    ):

        if self._data.empty:
            return

        col_name = str(self._data.columns[column])

        self.layoutAboutToBeChanged.emit()

        self._data = self._data.sort_values(
            by=col_name,
            ascending=order == QtCore.Qt.SortOrder.AscendingOrder,
            kind="mergesort",
        )

        self.layoutChanged.emit()

    # -------------------------------------------------------
    # Установка DataFrame
    # -------------------------------------------------------

    def set_dataframe(self, df: DataFrame) -> None:

        self.beginResetModel()
        self._data = df.copy()
        self.endResetModel()

    # -------------------------------------------------------
    # Получение DataFrame
    # -------------------------------------------------------

    def get_dataframe(self) -> DataFrame:
        return self._data.copy()


class DataTable(QTableView):
    """
    Виджет таблицы для отображения pandas DataFrame.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._model = PandasTableModel()
        self.setModel(self._model)

        # UI настройки
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)

        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)

        header = self.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)

    # -------------------------------------------------------
    # Публичные методы
    # -------------------------------------------------------

    def set_dataframe(self, df: DataFrame) -> None:
        self._model.set_dataframe(df)

    def get_dataframe(self) -> DataFrame:
        return self._model.get_dataframe()

    def clear(self) -> None:
        self._model.set_dataframe(pd.DataFrame())
