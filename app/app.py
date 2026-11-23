from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import get_args

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.api_schemas import (
    AssignedTeam,
    Category,
    Channel,
    MAX_DESCRIPTION_LENGTH,
    ModelInfoResponse,
    PredictionResponse,
    Priority,
    Region,
)


def main() -> None:
    """Render the ticket-triage interface."""
    st.set_page_config(
        page_title="Intelligent Ticket Triage",
        page_icon="🎫",
        layout="wide",
    )
    st.title("Intelligent Ticket Triage & SLA Risk Prediction")

    st.caption(
        """
        Estimate whether a newly created operational ticket is likely
        to breach its SLA using text, workload, team and ticket
        characteristics.
        """
    )

    st.caption("Trained and evaluated on synthetic tickets. Real-ticket performance is unvalidated.")

    api_url = os.environ.get("SLA_API_URL", "http://127.0.0.1:8000").rstrip("/")

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
            max_chars=MAX_DESCRIPTION_LENGTH,
        )

        first_row_1, first_row_2 = st.columns(2)

        with first_row_1:
            region = st.selectbox("Region", get_args(Region))
            category = st.selectbox("Category", get_args(Category), index=3)
            priority = st.selectbox("Current priority", get_args(Priority), index=1)

        with first_row_2:
            assigned_team = st.selectbox("Assigned team", get_args(AssignedTeam))
            channel = st.selectbox("Channel", get_args(Channel))

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
            created_date = st.date_input(
                "Creation date (UTC)",
                value=datetime.now(timezone.utc).date(),
            )
            created_hour = st.slider(
                "Creation hour (UTC)",
                min_value=0,
                max_value=23,
                value=10,
            )

        predict_clicked = st.button(
            "Predict SLA Risk",
            type="primary",
            width="stretch",
        )

    with result_column:
        st.subheader("Risk Assessment")

        if predict_clicked:
            created_at = datetime(
                created_date.year, created_date.month, created_date.day,
                created_hour, tzinfo=timezone.utc,
            )
            if not description.strip():
                st.error("Enter a ticket description before predicting.")
                return

            ticket = {
                "description": description,
                "region": region,
                "category": category,
                "priority": priority,
                "assigned_team": assigned_team,
                "channel": channel,
                "backlog_at_creation": backlog_at_creation,
                "team_experience_years": team_experience_years,
                "previous_reassignments": previous_reassignments,
                "sla_target_hours": sla_target_hours,
                "created_at": created_at.isoformat(),
            }
            try:
                response = requests.post(
                    f"{api_url}/v1/predict",
                    json=ticket,
                    timeout=(2, 10),
                    allow_redirects=False,
                )
                if response.status_code == 422:
                    st.error("Check your ticket inputs and try again.")
                    return
                response.raise_for_status()
                prediction = PredictionResponse.model_validate(response.json())
            except (requests.RequestException, ValueError):
                st.error("Prediction is currently unavailable. Check the inference API and try again.")
                return

            probability = prediction.risk_score
            threshold = prediction.decision_threshold
            risk_level = prediction.risk_classification

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

            st.write(f"Suggested operational priority: **{prediction.suggested_priority}**")
            for recommendation in prediction.recommendations:
                st.write(f"- {recommendation}")

            with st.expander("Model information"):
                st.write(f"**Bundle version:** {prediction.model_version}")
                try:
                    response = requests.get(f"{api_url}/v1/model/info", timeout=(2, 5))
                    response.raise_for_status()
                    info = ModelInfoResponse.model_validate(response.json())
                except (requests.RequestException, ValueError):
                    st.info("Model details are currently unavailable.")
                else:
                    if info.model_version == prediction.model_version:
                        metrics = info.holdout_metrics
                        st.write(f"**Selected model:** {info.model_name.replace('_', ' ').title()}")
                        st.write("**Synthetic holdout performance**")
                        st.write(
                            f"Average precision {metrics.average_precision:.3f} · "
                            f"ROC-AUC {metrics.roc_auc:.3f} · "
                            f"Precision {metrics.precision:.3f} · "
                            f"Recall {metrics.recall:.3f} · F2 {metrics.f2:.3f}"
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
