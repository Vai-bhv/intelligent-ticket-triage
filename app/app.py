from pathlib import Path
import json
import pickle
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_sla_risk_model.pkl"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "threshold_config.json"
)


st.set_page_config(
    page_title="Intelligent Ticket Triage",
    page_icon="🎫",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the trained SLA-risk model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python src/train.py` first."
        )

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_threshold() -> float:
    """Load the tuned decision threshold."""
    if not THRESHOLD_PATH.exists():
        return 0.50

    with THRESHOLD_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = json.load(file)

    return float(
        configuration.get(
            "selected_threshold",
            0.50,
        )
    )


def build_ticket_dataframe(
    description: str,
    region: str,
    category: str,
    priority: str,
    assigned_team: str,
    channel: str,
    backlog_at_creation: int,
    team_experience_years: float,
    previous_reassignments: int,
    weekend_created: int,
    sla_target_hours: int,
    created_hour: int,
    day_of_week: int,
    month: int,
) -> pd.DataFrame:
    """Create one model-ready ticket observation."""
    description_length = len(
        description.split()
    )

    quarter = (
        (month - 1) // 3
    ) + 1

    ticket_age_bucket = pd.cut(
        [backlog_at_creation],
        bins=[0, 20, 40, 60, 120],
        labels=[
            "Low",
            "Medium",
            "High",
            "Critical",
        ],
        include_lowest=True,
    )[0]

    is_month_end = 0

    return pd.DataFrame(
        [
            {
                "description": description,
                "description_length":
                    description_length,
                "backlog_at_creation":
                    backlog_at_creation,
                "team_experience_years":
                    team_experience_years,
                "previous_reassignments":
                    previous_reassignments,
                "weekend_created":
                    weekend_created,
                "sla_target_hours":
                    sla_target_hours,
                "day_of_week":
                    day_of_week,
                "month":
                    month,
                "quarter":
                    quarter,
                "hour":
                    created_hour,
                "is_month_end":
                    is_month_end,
                "region":
                    region,
                "category":
                    category,
                "priority":
                    priority,
                "assigned_team":
                    assigned_team,
                "channel":
                    channel,
                "ticket_age_bucket":
                    str(ticket_age_bucket),
            }
        ]
    )


def calculate_risk_level(
    probability: float,
    threshold: float,
) -> str:
    """Map probability to a business-facing risk level."""
    if probability >= 0.75:
        return "Critical"

    if probability >= threshold:
        return "High"

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


def main() -> None:
    """Render the ticket-triage interface."""
    st.title("Intelligent Ticket Triage & SLA Risk Prediction")

    st.caption(
        """
        Estimate whether a newly created operational ticket is likely
        to breach its SLA using text, workload, team and ticket
        characteristics.
        """
    )

    model = load_model()
    threshold = load_threshold()

    input_column, result_column = st.columns(
        [1.15, 0.85]
    )

    with input_column:
        st.subheader("Ticket Details")

        description = st.text_area(
            "Ticket description",
            value=(
                "Shared service is unavailable and blocking "
                "multiple users in production"
            ),
            height=120,
        )

        first_row_1, first_row_2 = st.columns(2)

        with first_row_1:
            region = st.selectbox(
                "Region",
                [
                    "Europe",
                    "North America",
                    "Asia Pacific",
                ],
            )

            category = st.selectbox(
                "Category",
                [
                    "Access Management",
                    "Application Support",
                    "Data Quality",
                    "Infrastructure",
                    "Workforce Operations",
                ],
                index=3,
            )

            priority = st.selectbox(
                "Current priority",
                [
                    "Low",
                    "Medium",
                    "High",
                    "Critical",
                ],
                index=1,
            )

        with first_row_2:
            assigned_team = st.selectbox(
                "Assigned team",
                [
                    "Global Operations",
                    "Regional Support",
                    "Application Services",
                    "Data Management",
                ],
            )

            channel = st.selectbox(
                "Channel",
                [
                    "Portal",
                    "Email",
                    "Phone",
                    "Monitoring",
                ],
            )

            sla_target_hours = st.number_input(
                "SLA target in hours",
                min_value=1,
                max_value=168,
                value=36,
                step=1,
            )

        second_row_1, second_row_2 = st.columns(2)

        with second_row_1:
            backlog_at_creation = st.slider(
                "Team backlog at ticket creation",
                min_value=0,
                max_value=120,
                value=55,
            )

            team_experience_years = st.slider(
                "Average team experience",
                min_value=0.5,
                max_value=10.0,
                value=3.0,
                step=0.1,
            )

            previous_reassignments = st.slider(
                "Previous reassignments",
                min_value=0,
                max_value=4,
                value=1,
            )

        with second_row_2:
            weekend_created = st.selectbox(
                "Created during weekend",
                options=[0, 1],
                format_func=lambda value:
                    "Yes" if value == 1 else "No",
            )

            created_hour = st.slider(
                "Created hour",
                min_value=0,
                max_value=23,
                value=10,
            )

            day_of_week = st.selectbox(
                "Day of week",
                options=list(range(7)),
                format_func=lambda value: [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ][value],
            )

            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                index=0,
            )

        predict_clicked = st.button(
            "Predict SLA Risk",
            type="primary",
            width="stretch",
        )

    with result_column:
        st.subheader("Risk Assessment")

        if predict_clicked:
            input_data = build_ticket_dataframe(
                description=description,
                region=region,
                category=category,
                priority=priority,
                assigned_team=assigned_team,
                channel=channel,
                backlog_at_creation=int(
                    backlog_at_creation
                ),
                team_experience_years=float(
                    team_experience_years
                ),
                previous_reassignments=int(
                    previous_reassignments
                ),
                weekend_created=int(
                    weekend_created
                ),
                sla_target_hours=int(
                    sla_target_hours
                ),
                created_hour=int(
                    created_hour
                ),
                day_of_week=int(
                    day_of_week
                ),
                month=int(
                    month
                ),
            )

            probability = float(
                model.predict_proba(
                    input_data
                )[0, 1]
            )

            predicted_breach = (
                probability >= threshold
            )

            risk_level = calculate_risk_level(
                probability,
                threshold,
            )

            metric_1, metric_2 = st.columns(2)

            metric_1.metric(
                "SLA Breach Probability",
                f"{probability * 100:.1f}%",
            )

            metric_2.metric(
                "Decision Threshold",
                f"{threshold * 100:.0f}%",
            )

            if risk_level == "Critical":
                st.error(
                    f"**CRITICAL RISK** — predicted breach "
                    f"probability is {probability * 100:.1f}%."
                )

            elif risk_level == "High":
                st.error(
                    f"**HIGH RISK** — this ticket is predicted "
                    f"to breach SLA."
                )

            elif risk_level == "Moderate":
                st.warning(
                    f"**MODERATE RISK** — monitor this ticket "
                    f"closely."
                )

            else:
                st.success(
                    f"**LOW RISK** — the ticket is currently "
                    f"unlikely to breach SLA."
                )

            st.progress(
                min(
                    max(probability, 0.0),
                    1.0,
                )
            )

            st.subheader("Suggested Action")

            if predicted_breach:
                suggested_priority = (
                    "Critical"
                    if probability >= 0.75
                    else "High"
                )
            else:
                suggested_priority = priority

            st.write(
                f"Suggested operational priority: "
                f"**{suggested_priority}**"
            )

            recommendations = create_recommendations(
                probability=probability,
                threshold=threshold,
                backlog=int(
                    backlog_at_creation
                ),
                previous_reassignments=int(
                    previous_reassignments
                ),
                weekend_created=int(
                    weekend_created
                ),
                priority=priority,
            )

            for recommendation in recommendations:
                st.write(
                    f"- {recommendation}"
                )

            with st.expander(
                "Model information"
            ):
                st.write(
                    f"""
                    **Selected model:** XGBoost classifier

                    **Decision threshold:** {threshold:.2f}

                    **Test performance:**

                    - PR-AUC: 0.507
                    - ROC-AUC: 0.869
                    - Recall at selected threshold: 0.800
                    - Precision at selected threshold: 0.344
                    - F2 score: 0.633

                    The threshold was selected to prioritize recall,
                    because missing a likely SLA breach is considered
                    more costly than reviewing an additional alert.
                    """
                )

        else:
            st.info(
                """
                Enter ticket details and click
                **Predict SLA Risk** to generate an assessment.
                """
            )


if __name__ == "__main__":
    main()