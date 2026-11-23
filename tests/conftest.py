import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.bundle import save_bundle
from src.features import FEATURE_COLUMNS


@pytest.fixture
def bundle_path(tmp_path):
    inputs = pd.DataFrame([[1] * len(FEATURE_COLUMNS), [2] * len(FEATURE_COLUMNS)],
                          columns=list(FEATURE_COLUMNS))
    model = make_pipeline(
        ColumnTransformer([("backlog", "passthrough", ["backlog_at_creation"])]),
        LogisticRegression(),
    ).fit(inputs, [0, 1])
    result = {
        "model_name": "logistic_regression",
        "threshold": 0.42,
        "metrics": {name: 0.5 for name in ("average_precision", "roc_auc", "precision", "recall", "f2")},
    }
    return save_bundle(tmp_path, model, "logistic_regression", 0.42,
                       {"validation": result, "test": result}, "dataset-hash")
