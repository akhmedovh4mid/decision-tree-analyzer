from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> int:
    """
    Точка входа приложения.
    """

    app = QApplication(sys.argv)

    app.setApplicationName("Decision Tree Analyzer")
    app.setOrganizationName("Data Analysis Project")

    window = MainWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
