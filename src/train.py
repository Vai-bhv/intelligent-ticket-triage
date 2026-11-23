"""Train and release a model with a validation-selected policy."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

if __package__:
    from . import bundle, evaluate, features
else:
    import bundle
    import evaluate
    import features


TARGET_COLUMN = "sla_breached"
DEFAULT_OUTPUT_DIR = features.PROJECT_ROOT / "models" / "runs"


def create_time_split(
    data: pd.DataFrame,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split oldest to newest, assigning ties at a boundary to the later partition."""
    if not (
        validation_fraction > 0
        and test_fraction > 0
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")

    ordered = data.assign(created_at=features.parse_created_at(data["created_at"]))
    ordered = ordered.sort_values("created_at", kind="stable").reset_index(drop=True)
    size = len(ordered)
    train_end = int(size * (1 - (validation_fraction + test_fraction)))
    validation_end = int(size * (1 - test_fraction))
    if not 0 < train_end < validation_end < size:
        raise ValueError("Not enough tickets for three chronological partitions.")

    timestamps = ordered["created_at"]
    train_end = timestamps.searchsorted(timestamps.iloc[train_end], side="left")
    validation_end = timestamps.searchsorted(timestamps.iloc[validation_end], side="left")
    if not 0 < train_end < validation_end < size:
        raise ValueError("Timestamp ties leave an empty chronological partition.")

    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def build_preprocessor() -> ColumnTransformer:
    """Fit text, numeric, and categorical transforms within each training pipeline."""
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                max_features=3000,
                sublinear_tf=True,
            ), features.TEXT_COLUMN),
            ("numerical", make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler()
            ), list(features.NUMERICAL_FEATURES)),
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ), list(features.CATEGORICAL_FEATURES)),
        ]
    )


def build_models(negative_count: int, positive_count: int) -> dict[str, Pipeline]:
    """Build the same candidate estimators with independent preprocessing."""
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42
            )),
        ]),
        "xgboost_classifier": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", XGBClassifier(
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
                scale_pos_weight=negative_count / positive_count,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def prepare_partition(tickets: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in tickets:
        raise ValueError(f"{name} tickets are missing {TARGET_COLUMN}.")
    target = tickets[TARGET_COLUMN]
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError(f"{name} must contain both binary labels and no missing labels.")
    inputs = features.engineer_features(tickets).loc[:, list(features.FEATURE_COLUMNS)]
    return inputs, target


def label_available_at(tickets: pd.DataFrame) -> pd.Series:
    """Use the simulated resolution event or the deadline, whichever reveals the label."""
    resolution = pd.to_numeric(tickets["resolution_hours"])
    sla = pd.to_numeric(tickets["sla_target_hours"])
    breached = tickets[TARGET_COLUMN]
    if (
        not np.isfinite(resolution).all() or not np.isfinite(sla).all()
        or (resolution <= 0).any() or (sla <= 0).any()
        or not breached.isin([0, 1]).all()
        # The generator rounds resolution after labeling. At reported equality
        # the stored label disambiguates an exact tie from a rounded breach.
        or ((resolution < sla) & breached.eq(1)).any()
        or ((resolution > sla) & breached.eq(0)).any()
    ):
        raise ValueError("Ticket durations and labels disagree with the simulator's SLA rules.")
    hours = resolution.where(breached.eq(0), sla)
    return tickets["created_at"] + pd.to_timedelta(hours, unit="h").dt.round("ms")


def mature_partition(tickets: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    mature = tickets.loc[label_available_at(tickets) <= cutoff]
    return mature, {
        "label_cutoff": cutoff.isoformat(),
        "excluded_immature_rows": len(tickets) - len(mature),
    }


def describe_partition(tickets: pd.DataFrame) -> dict:
    return {
        "rows": len(tickets),
        "positives": int(tickets[TARGET_COLUMN].sum()),
        "first_created_at": tickets["created_at"].min().isoformat(),
        "last_created_at": tickets["created_at"].max().isoformat(),
    }


def train_models(
    data_path: Path = features.RAW_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    evaluation_cutoff: str | None = None,
) -> Path:
    """Select on validation, score the frozen winner once on test, then publish it."""
    source = data_path.read_bytes()
    train, validation, test = create_time_split(pd.read_csv(BytesIO(source)))
    validation_start = validation["created_at"].iat[0]
    test_start = test["created_at"].iat[0]
    train, train_maturity = mature_partition(train, validation_start)
    validation, validation_maturity = mature_partition(validation, test_start)
    train_inputs, train_labels = prepare_partition(train, "Train")
    validation_inputs, validation_labels = prepare_partition(validation, "Validation")

    positives = int(train_labels.sum())
    candidates = build_models(len(train_labels) - positives, positives)
    validation_scores = {}
    validation_ap = {}
    for name, model in candidates.items():
        model.fit(train_inputs, train_labels)
        scores = model.predict_proba(validation_inputs)[:, 1]
        validation_scores[name] = scores
        validation_ap[name] = float(average_precision_score(validation_labels, scores))

    winner = max(validation_ap, key=validation_ap.__getitem__)
    model = candidates[winner]
    threshold, floor_met = evaluate.select_operating_threshold(
        validation_labels, validation_scores[winner]
    )
    validation_metrics = evaluate.calculate_metrics(
        validation_labels, validation_scores[winner], threshold
    )

    # The simulator is fully observed through the final ticket's latest SLA
    # deadline. This uses creation times and SLA targets, never test outcomes.
    cutoff = (
        features.parse_created_at(pd.Series([evaluation_cutoff])).iat[0]
        if evaluation_cutoff is not None
        else (test["created_at"] + pd.to_timedelta(test["sla_target_hours"], unit="h")).max()
    )
    test, test_maturity = mature_partition(test, cutoff)
    test_inputs, test_labels = prepare_partition(test, "Test")
    test_scores = model.predict_proba(test_inputs)[:, 1]
    test_metrics = evaluate.calculate_metrics(test_labels, test_scores, threshold)
    evaluation = {
        "train": {**describe_partition(train), **train_maturity},
        "validation": {
            **describe_partition(validation),
            **validation_maturity,
            "model_name": winner,
            "candidate_average_precision": validation_ap,
            "threshold": threshold,
            "threshold_selection": "maximum_f2",
            "minimum_precision": evaluate.MIN_VALIDATION_PRECISION,
            "minimum_precision_met": floor_met,
            "metrics": validation_metrics,
        },
        "test": {
            **describe_partition(test),
            **test_maturity,
            "model_name": winner,
            "threshold": threshold,
            "metrics": test_metrics,
        },
    }
    path = bundle.save_bundle(output_dir, model, winner, threshold, evaluation, sha256(source).hexdigest())
    print(json.dumps(evaluation, indent=2))
    print(f"Saved model bundle to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=features.RAW_DATA_PATH, help="Synthetic ticket CSV"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation-cutoff", help="UTC observation cutoff for test labels")
    args = parser.parse_args()
    train_models(args.data, args.output_dir, args.evaluation_cutoff)


if __name__ == "__main__":
    main()
