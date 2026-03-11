from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class AboutDialog(QDialog):
    """
    Диалог «О программе».
    Отображает информацию о приложении.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setWindowTitle("О программе")
        self.setMinimumWidth(400)

        self._init_ui()

    # -------------------------------------------------------
    # UI
    # -------------------------------------------------------

    def _init_ui(self) -> None:

        layout = QVBoxLayout()

        title = QLabel("Информационная система анализа данных")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        description = QLabel(
            "Программное обеспечение предназначено для анализа данных\n"
            "с использованием методов построения деревьев решений.\n\n"
            "Функциональность системы включает:\n"
            "• загрузку и просмотр данных\n"
            "• предобработку данных\n"
            "• обучение модели дерева решений\n"
            "• оценку качества модели\n"
            "• визуализацию результатов\n"
        )
        description.setAlignment(Qt.AlignmentFlag.AlignLeft)
        description.setWordWrap(True)

        tech = QLabel(
            "Используемые технологии:\nPython, Pandas, Scikit-learn, Matplotlib, PyQt5"
        )
        tech.setWordWrap(True)

        author = QLabel(
            "Разработчик: студент\nУчебный проект по дисциплине анализа данных"
        )
        author.setWordWrap(True)

        # кнопка закрытия
        button_layout = QHBoxLayout()

        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)

        button_layout.addStretch()
        button_layout.addWidget(close_button)

        # сборка layout
        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(description)
        layout.addSpacing(10)
        layout.addWidget(tech)
        layout.addSpacing(10)
        layout.addWidget(author)
        layout.addSpacing(20)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    # -------------------------------------------------------
    # Статический helper
    # -------------------------------------------------------

    @staticmethod
    def show_dialog(parent=None) -> None:
        """
        Упрощённый вызов диалога.
        """

        dialog = AboutDialog(parent)
        dialog.exec_()
