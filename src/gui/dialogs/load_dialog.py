from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class LoadDialog(QDialog):
    """
    Диалог выбора файла данных (CSV / Excel).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._filepath: Optional[Path] = None

        self.setWindowTitle("Загрузка данных")
        self.setMinimumWidth(400)

        self._init_ui()

    # -------------------------------------------------------
    # UI
    # -------------------------------------------------------

    def _init_ui(self) -> None:

        layout = QVBoxLayout()

        # поле пути
        path_layout = QHBoxLayout()

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Выберите файл...")

        browse_button = QPushButton("Обзор")
        browse_button.clicked.connect(self._browse_file)

        path_layout.addWidget(QLabel("Файл:"))
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_button)

        layout.addLayout(path_layout)

        # кнопки
        buttons_layout = QHBoxLayout()

        self.load_button = QPushButton("Загрузить")
        self.cancel_button = QPushButton("Отмена")

        self.load_button.clicked.connect(self._accept)
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout.addStretch()
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.cancel_button)

        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    # -------------------------------------------------------
    # Выбор файла
    # -------------------------------------------------------

    def _browse_file(self) -> None:

        file_filter = (
            "Data files (*.csv *.xlsx *.xls);;"
            "CSV files (*.csv);;"
            "Excel files (*.xlsx *.xls);;"
            "All files (*)"
        )

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл данных",
            "",
            file_filter,
        )

        if filepath:
            self.path_edit.setText(filepath)

    # -------------------------------------------------------
    # Подтверждение
    # -------------------------------------------------------

    def _accept(self) -> None:

        text = self.path_edit.text().strip()

        if not text:
            return

        self._filepath = Path(text)

        self.accept()

    # -------------------------------------------------------
    # Получение пути
    # -------------------------------------------------------

    def get_filepath(self) -> Optional[Path]:
        """
        Возвращает выбранный путь к файлу.
        """

        return self._filepath

    # -------------------------------------------------------
    # Статический helper
    # -------------------------------------------------------

    @staticmethod
    def get_file(parent=None) -> Optional[Path]:
        """
        Упрощённый вызов диалога.

        Example:
            filepath = LoadDialog.get_file(self)
        """

        dialog = LoadDialog(parent)

        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_filepath()

        return None
