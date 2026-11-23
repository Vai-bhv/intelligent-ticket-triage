"""Score a frozen decision policy or inspect a saved bundle's evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__:
    from .bundle import read_manifest
else:
    from bundle import read_manifest


MIN_VALIDATION_PRECISION = 0.30
THRESHOLDS = np.arange(10, 91) / 100


def select_operating_threshold(
    actual: pd.Series, probabilities: np.ndarray
) -> tuple[float, bool]:
    """Maximize validation F2, preferring thresholds that meet the precision floor."""
    scores = np.asarray(probabilities, dtype=float)
    precision, recall, cutoffs = precision_recall_curve(actual, scores)
    # A grid cutoff predicts the same positives as the next distinct score.
    positions = np.searchsorted(cutoffs, THRESHOLDS)
    precision, recall = precision[positions], recall[positions]
    denominator = 4 * precision + recall
    f2 = np.divide(
        5 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    eligible = (precision >= MIN_VALIDATION_PRECISION) & (THRESHOLDS <= cutoffs[-1])
    floor_met = bool(eligible.any())
    candidates = np.where(eligible, f2, -1) if floor_met else f2
    return float(THRESHOLDS[candidates.argmax()]), floor_met


def calculate_metrics(
    actual: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Report ranking metrics and operating metrics at the supplied threshold."""
    scores = np.asarray(probabilities, dtype=float)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "average_precision": float(average_precision_score(actual, scores)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "f2": float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(predicted.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Show evaluation saved in a model bundle.")
    parser.add_argument("bundle", type=Path, help="Run directory containing manifest.json")
    args = parser.parse_args()
    print(json.dumps(read_manifest(args.bundle)["evaluation"], indent=2))


if __name__ == "__main__":
    main()
