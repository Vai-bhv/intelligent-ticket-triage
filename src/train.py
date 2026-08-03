from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


TARGET_COLUMN = "sla_breached"

TEXT_COLUMN = "description"

NUMERICAL_FEATURES = [
    "description_length",
    "backlog_at_creation",
    "team_experience_years",
    "previous_reassignments",
    "weekend_created",
    "sla_target_hours",
    "day_of_week",
    "month",
    "quarter",
    "hour",
    "is_month_end",
]

CATEGORICAL_FEATURES = [
    "region",
    "category",
    "priority",
    "assigned_team",
    "channel",
    "ticket_age_bucket",
]

FEATURE_COLUMNS = (
    [TEXT_COLUMN]
    + NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


def get_project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def load_data() -> pd.DataFrame:
    """Load the engineered ticket dataset."""
    path = (
        get_project_root()
        / "data"
        / "processed"
        / "tickets_features.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run `python src/features.py` first."
        )

    data = pd.read_csv(
        path,
        parse_dates=["created_at"],
    )

    required_columns = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN, "ticket_id", "created_at"]
    )

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "The processed dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        data
        .sort_values("created_at")
        .reset_index(drop=True)
    )


def create_time_split(
    data: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split tickets chronologically.

    Older tickets are used for training and newer tickets are used
    as the untouched test set.
    """
    split_position = int(
        len(data) * (1 - test_fraction)
    )

    train_data = data.iloc[:split_position].copy()
    test_data = data.iloc[split_position:].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            "The time split produced an empty dataset."
        )

    return train_data, test_data


def build_preprocessor() -> ColumnTransformer:
    """Build text, numerical, and categorical preprocessing."""
    text_pipeline = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=3000,
        sublinear_tf=True,
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "text",
                text_pipeline,
                TEXT_COLUMN,
            ),
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_models(
    negative_count: int,
    positive_count: int,
) -> dict[str, Pipeline]:
    """Create candidate classification pipelines."""
    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    return {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost_classifier": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.04,
                        min_child_weight=3,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=2.0,
                        scale_pos_weight=scale_pos_weight,
                        tree_method="hist",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate classification metrics."""
    return {
        "accuracy": round(
            float(
                accuracy_score(
                    actual,
                    predicted,
                )
            ),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),
        "roc_auc": round(
            float(
                roc_auc_score(
                    actual,
                    probabilities,
                )
            ),
            4,
        ),
        "pr_auc": round(
            float(
                average_precision_score(
                    actual,
                    probabilities,
                )
            ),
            4,
        ),
    }


def save_model(
    model: Pipeline,
    model_name: str,
) -> Path:
    """Save a trained pipeline."""
    model_directory = (
        get_project_root() / "models"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        model_directory
        / f"{model_name}.pkl"
    )

    with path.open("wb") as file:
        pickle.dump(model, file)

    return path


def save_results(
    results: dict,
    predictions: pd.DataFrame,
    best_model_name: str,
) -> None:
    """Save metrics and test predictions."""
    report_directory = (
        get_project_root() / "reports"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "selection_metric": "pr_auc",
        "best_model": best_model_name,
        "models": results,
    }

    with (
        report_directory
        / "model_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    predictions.to_csv(
        report_directory
        / "test_predictions.csv",
        index=False,
    )


def train_models() -> None:
    """Train, compare, and save SLA-risk classifiers."""
    data = load_data()

    train_data, test_data = create_time_split(
        data
    )

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    print("Training configuration")
    print("----------------------")
    print(f"Training rows: {len(train_data):,}")
    print(f"Testing rows: {len(test_data):,}")
    print(
        f"Training period: "
        f"{train_data['created_at'].min()} to "
        f"{train_data['created_at'].max()}"
    )
    print(
        f"Testing period: "
        f"{test_data['created_at'].min()} to "
        f"{test_data['created_at'].max()}"
    )
    print(
        f"Training breach rate: "
        f"{y_train.mean() * 100:.2f}%"
    )
    print(
        f"Testing breach rate: "
        f"{y_test.mean() * 100:.2f}%"
    )
    print()

    models = build_models(
        negative_count=negative_count,
        positive_count=positive_count,
    )

    results = {}

    prediction_output = test_data[
        [
            "ticket_id",
            "created_at",
            "region",
            "category",
            "priority",
            "description",
            TARGET_COLUMN,
        ]
    ].copy()

    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        started_at = time.time()

        model.fit(
            X_train,
            y_train,
        )

        predicted = model.predict(X_test)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        metrics = calculate_metrics(
            actual=y_test,
            predicted=predicted,
            probabilities=probabilities,
        )

        elapsed_seconds = (
            time.time() - started_at
        )

        metrics["training_seconds"] = round(
            elapsed_seconds,
            2,
        )

        results[model_name] = metrics
        trained_models[model_name] = model

        prediction_output[
            f"{model_name}_prediction"
        ] = predicted

        prediction_output[
            f"{model_name}_probability"
        ] = probabilities.round(6)

        print(f"Metrics: {metrics}")
        print("Confusion matrix:")
        print(
            confusion_matrix(
                y_test,
                predicted,
            )
        )
        print()
        print(
            classification_report(
                y_test,
                predicted,
                target_names=[
                    "No Breach",
                    "SLA Breach",
                ],
                zero_division=0,
            )
        )
        print("-" * 60)

    best_model_name = max(
        results,
        key=lambda model_name:
            results[model_name]["pr_auc"],
    )

    best_model = trained_models[
        best_model_name
    ]

    best_model_path = save_model(
        best_model,
        "best_sla_risk_model",
    )

    save_results(
        results=results,
        predictions=prediction_output,
        best_model_name=best_model_name,
    )

    print("\nModel ranking by PR-AUC")
    print("-----------------------")

    ranking = sorted(
        results.items(),
        key=lambda item:
            item[1]["pr_auc"],
        reverse=True,
    )

    for position, (
        model_name,
        metrics,
    ) in enumerate(ranking, start=1):
        print(
            f"{position}. {model_name}: "
            f"PR-AUC={metrics['pr_auc']}, "
            f"ROC-AUC={metrics['roc_auc']}, "
            f"Recall={metrics['recall']}, "
            f"F1={metrics['f1']}"
        )

    print(
        f"\nSelected model: {best_model_name}"
    )
    print(
        f"Saved final pipeline to: "
        f"{best_model_path}"
    )


if __name__ == "__main__":
    train_models()