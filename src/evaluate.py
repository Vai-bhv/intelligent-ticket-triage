from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


TARGET_COLUMN = "sla_breached"
PROBABILITY_COLUMN = "xgboost_classifier_probability"


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def load_predictions() -> pd.DataFrame:
    """Load saved test predictions."""
    path = (
        get_project_root()
        / "reports"
        / "test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found at {path}. "
            "Run src/train.py first."
        )

    return pd.read_csv(
        path,
        parse_dates=["created_at"],
    )


def calculate_threshold_metrics(
    actual: pd.Series,
    probabilities: pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate classification metrics at one threshold."""
    predicted = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": round(float(threshold), 3),
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
        "f2": round(
            float(
                fbeta_score(
                    actual,
                    predicted,
                    beta=2,
                    zero_division=0,
                )
            ),
            4,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_risk_tickets": int(
            predicted.sum()
        ),
    }


def evaluate_thresholds(
    actual: pd.Series,
    probabilities: pd.Series,
) -> pd.DataFrame:
    """Evaluate thresholds from 0.10 through 0.90."""
    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    rows = [
        calculate_threshold_metrics(
            actual,
            probabilities,
            threshold,
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(rows)


def select_operating_threshold(
    threshold_results: pd.DataFrame,
) -> pd.Series:
    """
    Select the threshold with the highest F2 score.

    F2 gives recall twice as much importance as precision,
    which is suitable when missed SLA breaches are costly.
    """
    eligible_results = threshold_results[
        threshold_results["precision"] >= 0.30
    ]

    if eligible_results.empty:
        eligible_results = threshold_results

    best_index = eligible_results[
        "f2"
    ].idxmax()

    return threshold_results.loc[best_index]


def plot_threshold_tradeoff(
    threshold_results: pd.DataFrame,
    selected_threshold: float,
    output_directory: Path,
) -> None:
    """Plot precision, recall, F1 and F2 by threshold."""
    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for metric in [
        "precision",
        "recall",
        "f1",
        "f2",
    ]:
        axis.plot(
            threshold_results["threshold"],
            threshold_results[metric],
            label=metric.upper(),
        )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            f"Selected threshold "
            f"({selected_threshold:.2f})"
        ),
    )

    axis.set_title(
        "Classification Performance by Decision Threshold"
    )
    axis.set_xlabel("Decision Threshold")
    axis.set_ylabel("Metric Score")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_directory
        / "threshold_tradeoff.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_probability_distribution(
    predictions: pd.DataFrame,
    selected_threshold: float,
    output_directory: Path,
) -> None:
    """Plot predicted probability distributions by actual class."""
    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    no_breach = predictions[
        predictions[TARGET_COLUMN] == 0
    ][PROBABILITY_COLUMN]

    breach = predictions[
        predictions[TARGET_COLUMN] == 1
    ][PROBABILITY_COLUMN]

    axis.hist(
        no_breach,
        bins=30,
        alpha=0.6,
        label="No Breach",
    )

    axis.hist(
        breach,
        bins=30,
        alpha=0.6,
        label="SLA Breach",
    )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            f"Selected threshold "
            f"({selected_threshold:.2f})"
        ),
    )

    axis.set_title(
        "Predicted SLA-Risk Probability Distribution"
    )
    axis.set_xlabel("Predicted Breach Probability")
    axis.set_ylabel("Ticket Count")
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_directory
        / "probability_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Tune and save the SLA-risk decision threshold."""
    predictions = load_predictions()

    actual = predictions[TARGET_COLUMN]
    probabilities = predictions[
        PROBABILITY_COLUMN
    ]

    threshold_results = evaluate_thresholds(
        actual,
        probabilities,
    )

    selected = select_operating_threshold(
        threshold_results
    )

    default_metrics = calculate_threshold_metrics(
        actual,
        probabilities,
        threshold=0.50,
    )

    selected_threshold = float(
        selected["threshold"]
    )

    selected_metrics = (
        selected.to_dict()
    )

    print("Threshold evaluation")
    print("--------------------")

    print("\nDefault threshold: 0.50")
    for key, value in default_metrics.items():
        if key != "threshold":
            print(f"{key}: {value}")

    print(
        f"\nSelected threshold: "
        f"{selected_threshold:.2f}"
    )

    for key, value in selected_metrics.items():
        if key != "threshold":
            print(f"{key}: {value}")

    print("\nConfusion matrix comparison")
    print("---------------------------")

    print(
        f"Default:  TP={default_metrics['true_positives']}, "
        f"FN={default_metrics['false_negatives']}, "
        f"FP={default_metrics['false_positives']}, "
        f"TN={default_metrics['true_negatives']}"
    )

    print(
        f"Selected: TP={int(selected['true_positives'])}, "
        f"FN={int(selected['false_negatives'])}, "
        f"FP={int(selected['false_positives'])}, "
        f"TN={int(selected['true_negatives'])}"
    )

    report_directory = (
        get_project_root()
        / "reports"
    )

    figure_directory = (
        report_directory
        / "figures"
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_results.to_csv(
        report_directory
        / "threshold_results.csv",
        index=False,
    )

    threshold_config = {
        "model": "xgboost_classifier",
        "selection_method": (
            "Maximum F2 with minimum precision of 0.30"
        ),
        "selected_threshold": selected_threshold,
        "default_threshold_metrics": default_metrics,
        "selected_threshold_metrics": {
            key: (
                int(value)
                if key in {
                    "true_negatives",
                    "false_positives",
                    "false_negatives",
                    "true_positives",
                    "predicted_risk_tickets",
                }
                else float(value)
            )
            for key, value in selected_metrics.items()
        },
    }

    with (
        report_directory
        / "threshold_config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            threshold_config,
            file,
            indent=4,
        )

    plot_threshold_tradeoff(
        threshold_results,
        selected_threshold,
        figure_directory,
    )

    plot_probability_distribution(
        predictions,
        selected_threshold,
        figure_directory,
    )

    print(
        "\nSaved threshold configuration to: "
        "reports/threshold_config.json"
    )


if __name__ == "__main__":
    main()