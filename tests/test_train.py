from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src import train


def test_chronological_split_keeps_tied_timestamps_together():
    tickets = pd.DataFrame({
        "ticket_id": range(10),
        "created_at": [
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            "2025-01-05", "2025-01-06", "2025-01-06",
            "2025-01-08", "2025-01-08", "2025-01-09",
        ],
    }).sample(frac=1, random_state=42)
    training, validation, test = train.create_time_split(tickets)

    assert set(training["ticket_id"]) == set(range(5))
    assert set(validation["ticket_id"]) == {5, 6}
    assert set(test["ticket_id"]) == {7, 8, 9}
    assert training["created_at"].max() < validation["created_at"].min()
    assert validation["created_at"].max() < test["created_at"].min()


def test_split_rejects_ties_that_would_empty_a_partition():
    tickets = pd.DataFrame({"created_at": ["2025-01-01"] * 10})
    with pytest.raises(ValueError, match="ties leave an empty"):
        train.create_time_split(tickets)


def test_only_validation_selects_winner_and_threshold(tmp_path, monkeypatch):
    training_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0]
    validation_labels = [1, 0, 1]
    test_labels = [0, 1, 0]
    tickets = pd.DataFrame({
        "created_at": pd.date_range("2025-01-01", periods=15),
        "description": "ticket",
        "backlog_at_creation": 0,
        "team_experience_years": 3.0,
        "previous_reassignments": 0,
        "sla_target_hours": 36,
        "resolution_hours": [48 if label else 12 for label in
                             training_labels + validation_labels + test_labels],
        "region": "Europe",
        "category": "Infrastructure",
        "priority": "Medium",
        "assigned_team": "Global Operations",
        "channel": "Portal",
        "sla_breached": training_labels + validation_labels + test_labels,
    })
    tickets.loc[8, "resolution_hours"] = 30  # Train resolution after validation start.
    tickets.loc[14, "resolution_hours"] = 24  # Test resolution after evaluation cutoff.
    source = tmp_path / "tickets.csv"
    tickets.to_csv(source, index=False)

    def candidate(validation_scores, test_scores=None):
        model = Mock()

        def predict(inputs):
            if inputs.index.tolist() == [9, 10]:
                values = np.asarray(validation_scores)
            else:
                assert inputs.index.tolist() == [12, 13]
                assert test_scores is not None
                values = np.asarray(test_scores)
            return np.column_stack((1 - values, values))

        model.predict_proba.side_effect = predict
        return model

    winner = candidate([0.8, 0.7], [0.75, 0.71])
    loser = candidate([0.4, 0.9])
    save_bundle = Mock(return_value=tmp_path / "run")
    select_threshold = Mock(wraps=train.evaluate.select_operating_threshold)
    monkeypatch.setattr(train, "build_models", lambda *_: {"winner": winner, "loser": loser})
    monkeypatch.setattr(train.bundle, "save_bundle", save_bundle)
    monkeypatch.setattr(train.evaluate, "select_operating_threshold", select_threshold)

    assert train.train_models(source, tmp_path, "2025-01-15T13:00:00Z") == tmp_path / "run"

    for model in (winner, loser):
        model.fit.assert_called_once()
        assert model.fit.call_args.args[0].index.tolist() == list(range(8))
        assert model.fit.call_args.args[1].tolist() == training_labels[:-1]
    assert loser.predict_proba.call_count == 1
    assert winner.predict_proba.call_count == 2
    select_threshold.assert_called_once()
    assert select_threshold.call_args.args[0].tolist() == validation_labels[:-1]
    assert select_threshold.call_args.args[1].tolist() == [0.8, 0.7]
    _, selected, name, threshold, evaluation, _ = save_bundle.call_args.args
    assert selected is winner
    assert name == evaluation["test"]["model_name"] == "winner"
    assert threshold == evaluation["test"]["threshold"] == 0.71
    assert evaluation["test"]["metrics"]["true_positives"] == 1
    assert evaluation["test"]["metrics"]["false_positives"] == 1
    assert [evaluation[name]["excluded_immature_rows"]
            for name in ("train", "validation", "test")] == [1, 1, 1]


def test_label_observable_at_resolution_or_deadline_including_exact_cutoff():
    created = pd.Timestamp("2025-01-01T00:00:00Z")
    tickets = pd.DataFrame({
        "created_at": [created] * 3,
        "sla_target_hours": [24] * 3,
        "resolution_hours": [12, 24, 36],
        "sla_breached": [0, 0, 1],
    })
    assert train.label_available_at(tickets).tolist() == [
        created + pd.Timedelta(hours=hours) for hours in (12, 24, 24)
    ]
    mature, _ = train.mature_partition(tickets, created + pd.Timedelta(hours=24))
    assert len(mature) == 3
    with pytest.raises(ValueError, match="disagree"):
        train.label_available_at(tickets.assign(sla_breached=[1, 0, 1]))
