from datetime import datetime, timezone

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.inference import build_ticket_dataframe
from src.train import prepare_partition


@pytest.fixture
def raw_ticket():
    return {
        "description": "Production   log unavailable",
        "region": "Europe",
        "category": "Infrastructure",
        "priority": "Medium",
        "assigned_team": "Global Operations",
        "channel": "Portal",
        "backlog_at_creation": 0,
        "team_experience_years": 3.0,
        "previous_reassignments": 1,
        "sla_target_hours": 36,
        "created_at": datetime(2024, 8, 31, 23, tzinfo=timezone.utc),
    }


def test_training_and_inference_use_identical_features(raw_ticket):
    # Fields derived by older generators cannot override the creation timestamp or text.
    training_row = {
        **raw_ticket,
        "created_at": "2024-09-01T01:00:00+02:00",
        "description_length": 100,
        "weekend_created": 0,
        "is_month_end": 0,
    }
    tickets = pd.DataFrame([
        {**training_row, "sla_breached": 0},
        {**training_row, "sla_breached": 1},
    ])
    training_features, _ = prepare_partition(tickets, "Train")
    served_features = build_ticket_dataframe(**raw_ticket)

    assert_frame_equal(training_features.iloc[[0]], served_features)
    assert served_features.at[0, "description_length"] == 3
    assert served_features.at[0, "hour"] == 23


@pytest.mark.parametrize(
    ("backlog", "bucket"),
    [(0, "Low"), (20, "Low"), (21, "Medium"), (40, "Medium"), (41, "High"),
     (60, "High"), (61, "Critical"), (121, "Critical")],
)
def test_backlog_boundaries(raw_ticket, backlog, bucket):
    assert build_ticket_dataframe(**(raw_ticket | {"backlog_at_creation": backlog})).at[
        0, "backlog_bucket"
    ] == bucket


@pytest.mark.parametrize(
    ("created_at", "day_of_week", "weekend", "month_end"),
    [
        (datetime(2024, 8, 31), 5, 1, 1),
        (datetime(2024, 9, 1), 6, 1, 0),
        (datetime(2024, 2, 29), 3, 0, 1),
    ],
)
def test_weekend_and_month_end_come_from_the_same_date(
    raw_ticket, created_at, day_of_week, weekend, month_end
):
    row = build_ticket_dataframe(**(raw_ticket | {"created_at": created_at})).iloc[0]
    assert (row["day_of_week"], row["weekend_created"], row["is_month_end"]) == (
        day_of_week, weekend, month_end
    )
