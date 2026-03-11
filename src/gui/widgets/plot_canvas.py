from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget


class PlotCanvas(QWidget):
    """
    Виджет для отображения matplotlib графиков внутри PyQt5.

    Позволяет:
    - отображать Figure
    - очищать график
    - обновлять отображение
    - использовать toolbar matplotlib
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        width: int = 6,
        height: int = 4,
        dpi: int = 100,
        show_toolbar: bool = True,
    ) -> None:
        super().__init__(parent)

        self._figure = Figure(figsize=(width, height), dpi=dpi)
        self._canvas = FigureCanvasQTAgg(self._figure)

        self._toolbar: Optional[NavigationToolbar2QT] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        if show_toolbar:
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            layout.addWidget(self._toolbar)

        layout.addWidget(self._canvas)

        self.setLayout(layout)

    # -------------------------------------------------------
    # Получение Figure
    # -------------------------------------------------------

    def get_figure(self) -> Figure:
        """
        Возвращает текущий matplotlib Figure.
        """
        return self._figure

    # -------------------------------------------------------
    # Установка Figure
    # -------------------------------------------------------

    def set_figure(self, figure: Figure) -> None:
        """
        Устанавливает новую Figure для отображения.
        """

        self._figure = figure
        self._canvas.figure = figure
        self._canvas.draw()

    # -------------------------------------------------------
    # Очистка графика
    # -------------------------------------------------------

    def clear(self) -> None:
        """
        Очищает текущий график.
        """

        self._figure.clear()
        self._canvas.draw()

    # -------------------------------------------------------
    # Перерисовка
    # -------------------------------------------------------

    def redraw(self) -> None:
        """
        Перерисовывает текущий график.
        """

        self._canvas.draw()

    # -------------------------------------------------------
    # Добавление subplot
    # -------------------------------------------------------

    def add_subplot(self, *args, **kwargs):
        """
        Создаёт subplot внутри Figure.
        """

        ax = self._figure.add_subplot(*args, **kwargs)
        self._canvas.draw()

        return ax
