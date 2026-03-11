from __future__ import annotations

from typing import Any, Dict

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ParameterPanel(QWidget):
    """
    Панель параметров модели дерева решений.
    Позволяет пользователю настроить основные гиперпараметры.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._init_ui()

    # -------------------------------------------------------
    # UI
    # -------------------------------------------------------

    def _init_ui(self) -> None:

        layout = QVBoxLayout()
        form = QFormLayout()

        # Criterion
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy"])
        form.addRow(QLabel("Criterion"), self.criterion_combo)

        # Max depth
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(0, 100)
        self.max_depth_spin.setValue(0)
        self.max_depth_spin.setToolTip("0 = без ограничения")
        form.addRow(QLabel("Max depth"), self.max_depth_spin)

        # Min samples split
        self.min_samples_split_spin = QSpinBox()
        self.min_samples_split_spin.setRange(2, 100)
        self.min_samples_split_spin.setValue(2)
        form.addRow(QLabel("Min samples split"), self.min_samples_split_spin)

        # Min samples leaf
        self.min_samples_leaf_spin = QSpinBox()
        self.min_samples_leaf_spin.setRange(1, 100)
        self.min_samples_leaf_spin.setValue(1)
        form.addRow(QLabel("Min samples leaf"), self.min_samples_leaf_spin)

        # Random state
        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 999999)
        self.random_state_spin.setValue(42)
        form.addRow(QLabel("Random state"), self.random_state_spin)

        # Использовать random_state
        self.use_random_state_checkbox = QCheckBox("Использовать random_state")
        self.use_random_state_checkbox.setChecked(True)
        form.addRow(self.use_random_state_checkbox)

        layout.addLayout(form)

        # Кнопка обучения
        self.train_button = QPushButton("Обучить модель")

        layout.addWidget(self.train_button)

        layout.addStretch()

        self.setLayout(layout)

    # -------------------------------------------------------
    # Получение параметров
    # -------------------------------------------------------

    def get_parameters(self) -> Dict[str, Any]:
        """
        Возвращает параметры модели в виде словаря.
        """

        max_depth = self.max_depth_spin.value()

        params = {
            "criterion": self.criterion_combo.currentText(),
            "max_depth": None if max_depth == 0 else max_depth,
            "min_samples_split": self.min_samples_split_spin.value(),
            "min_samples_leaf": self.min_samples_leaf_spin.value(),
        }

        if self.use_random_state_checkbox.isChecked():
            params["random_state"] = self.random_state_spin.value()

        return params

    # -------------------------------------------------------
    # Сброс параметров
    # -------------------------------------------------------

    def reset(self) -> None:
        """
        Сбрасывает параметры к значениям по умолчанию.
        """

        self.criterion_combo.setCurrentIndex(0)
        self.max_depth_spin.setValue(0)
        self.min_samples_split_spin.setValue(2)
        self.min_samples_leaf_spin.setValue(1)
        self.random_state_spin.setValue(42)
        self.use_random_state_checkbox.setChecked(True)
