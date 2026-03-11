from __future__ import annotations

from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core.controller import Controller
from src.gui.dialogs import AboutDialog, LoadDialog
from src.gui.widgets import DataTable, ParameterPanel, PlotCanvas


class MainWindow(QMainWindow):
    """
    Главное окно приложения.
    Управляет GUI и взаимодействует с Controller.
    """

    def __init__(self) -> None:
        super().__init__()

        self.controller = Controller()

        self.setWindowTitle("Decision Tree Analyzer")
        self.resize(1200, 800)

        self._init_ui()
        self._init_menu()

    # -------------------------------------------------------
    # UI
    # -------------------------------------------------------

    def _init_ui(self) -> None:

        central = QWidget()
        main_layout = QVBoxLayout()

        # Верхняя панель (target selection)
        top_layout = QHBoxLayout()

        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self._set_target)

        top_layout.addWidget(QLabel("Target column:"))
        top_layout.addWidget(self.target_combo)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)

        # Splitter
        splitter = QSplitter()

        # Таблица данных
        self.data_table = DataTable()

        # Правая панель
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        self.parameter_panel = ParameterPanel()
        self.parameter_panel.train_button.clicked.connect(self._train_model)

        self.plot_canvas = PlotCanvas()

        right_layout.addWidget(self.parameter_panel)
        right_layout.addWidget(self.plot_canvas)

        right_widget.setLayout(right_layout)

        splitter.addWidget(self.data_table)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())

    # -------------------------------------------------------
    # Menu
    # -------------------------------------------------------

    def _init_menu(self) -> None:

        menu_bar: QMenuBar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("Файл")

        load_action = QAction("Загрузить данные", self)
        load_action.triggered.connect(self._load_data)

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(load_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Model menu
        model_menu = menu_bar.addMenu("Модель")

        train_action = QAction("Обучить дерево", self)
        train_action.triggered.connect(self._train_model)

        importance_action = QAction("Feature importance", self)
        importance_action.triggered.connect(self._plot_importance)

        tree_action = QAction("Показать дерево", self)
        tree_action.triggered.connect(self._plot_tree)

        metrics_action = QAction("Метрики", self)
        metrics_action.triggered.connect(self._show_metrics)

        model_menu.addAction(train_action)
        model_menu.addAction(importance_action)
        model_menu.addAction(tree_action)
        model_menu.addAction(metrics_action)

        # Help menu
        help_menu = menu_bar.addMenu("Справка")

        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)

        help_menu.addAction(about_action)

    # -------------------------------------------------------
    # Data loading
    # -------------------------------------------------------

    def _load_data(self) -> None:

        filepath = LoadDialog.get_file(self)

        if filepath is None:
            return

        try:
            self.controller.load_data(filepath)

            df = self.controller.get_dataframe()

            self.data_table.set_dataframe(df)

            self._update_columns(df.columns)

            self.statusBar().showMessage(f"Данные загружены: {filepath}")

        except Exception as e:
            self._error(str(e))

    def _update_columns(self, columns) -> None:

        self.target_combo.clear()
        self.target_combo.addItems(list(columns))

    # -------------------------------------------------------
    # Target
    # -------------------------------------------------------

    def _set_target(self, column: str) -> None:

        if not column:
            return

        try:
            self.controller.set_target(column)

            self.statusBar().showMessage(f"Target установлен: {column}")

        except Exception as e:
            self._error(str(e))

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------

    def _train_model(self) -> None:

        try:
            params = self.parameter_panel.get_parameters()

            self.controller.split()

            self.controller.train_tree(**params)

            self.statusBar().showMessage("Модель обучена")

        except Exception as e:
            self._error(str(e))

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------

    def _show_metrics(self) -> None:

        try:
            metrics = self.controller.evaluate()

            text = "\n".join(f"{k}: {v}" for k, v in metrics.items())

            QMessageBox.information(
                self,
                "Метрики модели",
                text,
            )

        except Exception as e:
            self._error(str(e))

    # -------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------

    def _plot_importance(self) -> None:

        try:
            fig = self.controller.plot_feature_importance()

            self.plot_canvas.set_figure(fig)

        except Exception as e:
            self._error(str(e))

    # -------------------------------------------------------
    # Tree plot
    # -------------------------------------------------------

    def _plot_tree(self) -> None:

        try:
            fig = self.controller.plot_tree()

            self.plot_canvas.set_figure(fig)

        except Exception as e:
            self._error(str(e))

    # -------------------------------------------------------
    # About
    # -------------------------------------------------------

    def _show_about(self) -> None:

        AboutDialog.show_dialog(self)

    # -------------------------------------------------------
    # Error handling
    # -------------------------------------------------------

    def _error(self, message: str) -> None:

        QMessageBox.critical(
            self,
            "Ошибка",
            message,
        )
