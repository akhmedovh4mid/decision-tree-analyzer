from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class Metrics:
    @staticmethod
    def evaluate(y_true, y_pred) -> Dict[str, float]:

        results = {}

        try:
            results["accuracy"] = accuracy_score(y_true, y_pred)
            results["precision"] = precision_score(y_true, y_pred, average="weighted")
            results["recall"] = recall_score(y_true, y_pred, average="weighted")
            results["f1"] = f1_score(y_true, y_pred, average="weighted")

        except Exception:
            results["mse"] = mean_squared_error(y_true, y_pred)
            results["rmse"] = np.sqrt(results["mse"])
            results["r2"] = r2_score(y_true, y_pred)

        return results
