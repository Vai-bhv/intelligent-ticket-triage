import json
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from src import api
from src.bundle import MANIFEST_FILENAME


TICKET = {
    "description": "Synthetic example. Shared service unavailable.",
    "region": "Europe",
    "category": "Infrastructure",
    "priority": "Medium",
    "assigned_team": "Global Operations",
    "channel": "Portal",
    "backlog_at_creation": 60,
    "team_experience_years": 3.0,
    "previous_reassignments": 2,
    "sla_target_hours": 36,
    "created_at": "2024-09-01T01:00:00+02:00",
}


@pytest.fixture
def client(bundle_path, monkeypatch):
    monkeypatch.setenv("SLA_MODEL_BUNDLE", str(bundle_path))
    with TestClient(api.app) as client:
        yield client


def test_startup_loads_once_and_serves_deterministic_predictions(bundle_path, monkeypatch):
    load_bundle = Mock(wraps=api.load_bundle)
    monkeypatch.setattr(api, "load_bundle", load_bundle)
    monkeypatch.setenv("SLA_MODEL_BUNDLE", str(bundle_path))

    with TestClient(api.app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready"}

        model_info = client.get("/v1/model/info")
        assert model_info.status_code == 200
        assert model_info.json() == {
            "model_version": bundle_path.name,
            "model_name": "logistic_regression",
            "feature_schema_version": 1,
            "decision_threshold": 0.42,
            "evaluation_data": "synthetic",
            "holdout_metrics": dict.fromkeys(
                ("average_precision", "roc_auc", "precision", "recall", "f2"), 0.5
            ),
        }

        response = client.post("/v1/predict", json=TICKET)
        assert response.status_code == 200
        prediction = response.json()
        assert client.post("/v1/predict", json=TICKET).json() == prediction
        assert prediction["model_version"] == model_info.json()["model_version"]
        assert prediction["decision_threshold"] == 0.42
        assert 0.75 <= prediction["risk_score"] <= 1
        assert prediction["risk_classification"] == "Critical"
        assert prediction["suggested_priority"] == "Critical"
        assert "Escalate the ticket for proactive review." in prediction["recommendations"]
        assert "Confirm weekend or on-call coverage." in prediction["recommendations"]
        assert "Assign clear ownership to prevent further reassignment." in prediction["recommendations"]

    load_bundle.assert_called_once_with(bundle_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("description", ""),
        ("description", "x" * 4097),
        ("backlog_at_creation", -1),
        ("backlog_at_creation", "60"),
        ("previous_reassignments", 5),
        ("sla_target_hours", 0),
        ("team_experience_years", 10.1),
        ("priority", "P1"),
        ("created_at", "2024-08-31T10:00:00"),
        ("created_at", 1_724_000_000),
        ("created_at", "9999-08-31T10:00:00Z"),
        ("unexpected", "any value"),
    ],
)
def test_invalid_input_has_structured_redacted_errors(client, field, value):
    response = client.post("/v1/predict", json={**TICKET, field: value})
    assert response.status_code == 422
    expected_field = "body" if field == "unexpected" else field
    assert response.json()["detail"][0]["field"] == expected_field
    assert response.json()["detail"][0]["code"]
    assert "input" not in response.json()["detail"][0]
    assert TICKET["description"] not in response.text


def test_invalid_json_does_not_echo_description(client, caplog):
    private_text = "PRIVATE_TICKET_TEXT"
    body = '{"description":"' + private_text + '"'
    response = client.post("/v1/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert private_text not in response.text
    assert private_text not in caplog.text


def assert_unavailable(client):
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}

    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json() == {"status": "not_ready"}

    for response in (
        client.get("/v1/model/info"),
        client.post("/v1/predict", json=TICKET),
    ):
        assert response.status_code == 503
        assert response.json() == {"detail": "Model unavailable."}


def test_missing_bundle_keeps_api_live_but_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("SLA_MODEL_BUNDLE", str(tmp_path / "missing"))
    with TestClient(api.app) as client:
        assert_unavailable(client)


def test_unconfigured_bundle_keeps_api_live_but_not_ready(monkeypatch):
    monkeypatch.delenv("SLA_MODEL_BUNDLE", raising=False)
    with TestClient(api.app) as client:
        assert_unavailable(client)


@pytest.mark.parametrize(("field", "value"), [
    ("threshold", 2),
    ("library_versions", {}),
    ("model_sha256", "unexpected"),
])
def test_invalid_or_incompatible_bundle_is_rejected_before_deserialization(
    bundle_path, monkeypatch, field, value
):
    manifest = bundle_path / MANIFEST_FILENAME
    metadata = json.loads(manifest.read_text())
    metadata[field] = value
    manifest.write_text(json.dumps(metadata))
    monkeypatch.setenv("SLA_MODEL_BUNDLE", str(bundle_path))
    deserialize = Mock(side_effect=AssertionError("Unvalidated model was deserialized."))
    monkeypatch.setattr("src.bundle.pickle.loads", deserialize)

    with TestClient(api.app) as client:
        assert_unavailable(client)

    deserialize.assert_not_called()


def test_prediction_failure_does_not_expose_exception_details(client, monkeypatch, caplog):
    private_text = "PRIVATE_TICKET_TEXT"
    monkeypatch.setattr(api, "predict", Mock(side_effect=ValueError(private_text)))
    response = client.post("/v1/predict", json={**TICKET, "description": private_text})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed."}
    assert private_text not in caplog.text


def wait_for_readiness(client, status):
    from time import monotonic, sleep

    deadline = monotonic() + 3
    while monotonic() < deadline:
        if client.get("/health/ready").status_code == status:
            return
        sleep(0.01)
    pytest.fail(f"Readiness did not reach {status}.")


def test_failed_startup_probe_and_recovery(bundle_path, monkeypatch):
    from threading import Event

    recovered = Event()
    real_predict = api.predict

    def predict(bundle, ticket):
        if not recovered.is_set():
            raise RuntimeError("probe failure")
        return real_predict(bundle, ticket)

    monkeypatch.setenv("SLA_MODEL_BUNDLE", str(bundle_path))
    monkeypatch.setattr(api, "PROBE_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(api, "predict", predict)
    with TestClient(api.app) as client:
        assert_unavailable(client)
        recovered.set()
        wait_for_readiness(client, 200)
        assert client.post("/v1/predict", json=TICKET).status_code == 200


def test_persistent_prediction_failure_changes_cached_readiness(bundle_path, monkeypatch):
    monkeypatch.setenv("SLA_MODEL_BUNDLE", str(bundle_path))
    monkeypatch.setattr(api, "PROBE_INTERVAL_SECONDS", 0.02)
    with TestClient(api.app) as client:
        assert client.get("/health/ready").status_code == 200
        real_predict = api.predict
        monkeypatch.setattr(api, "predict", Mock(side_effect=RuntimeError("offline")))
        wait_for_readiness(client, 503)
        assert_unavailable(client)
        monkeypatch.setattr(api, "predict", real_predict)
        wait_for_readiness(client, 200)
        assert client.post("/v1/predict", json=TICKET).status_code == 200


def test_timed_out_probe_never_overlaps_and_can_recover(bundle_path, monkeypatch):
    from threading import Event

    release = Event()
    real_predict = api.predict
    predict = Mock(side_effect=lambda bundle, ticket: (
        release.wait(5) and real_predict(bundle, ticket)
    ))
    monkeypatch.setenv("SLA_MODEL_BUNDLE", str(bundle_path))
    monkeypatch.setattr(api, "PROBE_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(api, "PROBE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(api, "predict", predict)
    with TestClient(api.app) as client:
        try:
            assert_unavailable(client)
            assert_unavailable(client)
            assert predict.call_count == 1
        finally:
            release.set()
        wait_for_readiness(client, 200)


def test_body_limit_checks_declared_and_streamed_lengths(client):
    from src.request_limits import MAX_BODY_BYTES

    headers = {"Content-Type": "application/json"}
    assert client.post("/v1/predict", headers={
        **headers, "Content-Length": str(MAX_BODY_BYTES + 1)
    }, content=b"{}").status_code == 413
    result = client.post("/v1/predict", headers=headers, content=iter([
        b" " * (MAX_BODY_BYTES // 2), b" " * (MAX_BODY_BYTES // 2), b"x",
    ]))
    assert result.status_code == 413
    assert result.json() == {"detail": "Request body too large."}
    encoded = json.dumps(TICKET).encode()
    assert client.post("/v1/predict", headers=headers, content=iter([
        encoded[:10], encoded[10:],
    ])).status_code == 200


def test_validation_issues_are_capped_and_hide_unexpected_fields(client):
    payload = {**TICKET, **{f"private_{index}": "secret" for index in range(30)}}
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
    assert len(response.json()["detail"]) == api.MAX_VALIDATION_ISSUES
    assert "private_" not in response.text and "secret" not in response.text
