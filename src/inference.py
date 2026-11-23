"""Apply the Phase 1 feature and decision policy to a validated ticket."""

import pandas as pd

from .api_schemas import (
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    RiskClassification,
)
from .bundle import ModelBundle
from .features import FEATURE_COLUMNS, engineer_features


def build_ticket_dataframe(**raw_ticket: object) -> pd.DataFrame:
    """Turn a raw ticket into the same model inputs used for training."""
    return engineer_features(pd.DataFrame([raw_ticket])).loc[:, list(FEATURE_COLUMNS)]


def calculate_risk_level(
    probability: float,
    threshold: float,
) -> RiskClassification:
    """Map probability to a business-facing risk level."""
    if probability >= threshold:
        return "Critical" if probability >= 0.75 else "High"

    if probability >= 0.25:
        return "Moderate"

    return "Low"


def create_recommendations(
    probability: float,
    threshold: float,
    backlog: int,
    previous_reassignments: int,
    weekend_created: int,
    priority: str,
) -> list[str]:
    """Create simple operational recommendations."""
    recommendations = []

    if probability >= threshold:
        recommendations.append(
            "Escalate the ticket for proactive review."
        )

    if backlog >= 60:
        recommendations.append(
            "Consider moving the ticket to a lower-load support team."
        )

    if previous_reassignments >= 2:
        recommendations.append(
            "Assign clear ownership to prevent further reassignment."
        )

    if weekend_created == 1:
        recommendations.append(
            "Confirm weekend or on-call coverage."
        )

    if priority in {"Low", "Medium"} and probability >= threshold:
        recommendations.append(
            "Review whether the current priority should be increased."
        )

    if not recommendations:
        recommendations.append(
            "Continue standard processing and monitor progress."
        )

    return recommendations


def predict(bundle: ModelBundle, ticket: PredictionRequest) -> PredictionResponse:
    features = build_ticket_dataframe(**ticket.model_dump())
    score = float(bundle.model.predict_proba(features)[0, 1])
    threshold = bundle.threshold
    risk = calculate_risk_level(score, threshold)
    priority = ticket.priority
    if score >= threshold:
        priority = "Critical" if "Critical" in (priority, risk) else "High"

    return PredictionResponse(
        model_version=bundle.metadata["bundle_version"],
        risk_score=score,
        decision_threshold=threshold,
        risk_classification=risk,
        suggested_priority=priority,
        recommendations=create_recommendations(
            probability=score,
            threshold=threshold,
            backlog=ticket.backlog_at_creation,
            previous_reassignments=ticket.previous_reassignments,
            weekend_created=int(features["weekend_created"].iat[0]),
            priority=ticket.priority,
        ),
    )


def describe_model(bundle: ModelBundle) -> ModelInfoResponse:
    metadata = bundle.metadata
    return ModelInfoResponse(
        model_version=metadata["bundle_version"],
        model_name=metadata["model_name"],
        feature_schema_version=metadata["feature_schema"]["version"],
        decision_threshold=bundle.threshold,
        evaluation_data="synthetic",
        holdout_metrics=metadata["evaluation"]["test"]["metrics"],
    )
