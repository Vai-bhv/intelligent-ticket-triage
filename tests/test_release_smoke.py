from fastapi.testclient import TestClient
from threadpoolctl import threadpool_limits

from src import api, train
from src.bundle import read_manifest


def test_synthetic_training_bundle_serves_real_http_prediction(tmp_path, monkeypatch):
    with threadpool_limits(limits=2):
        bundle = train.train_models(output_dir=tmp_path)
    manifest = read_manifest(bundle)
    monkeypatch.setenv("SLA_MODEL_BUNDLE", str(bundle))

    with TestClient(api.app) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post("/v1/predict", json=api.PROBE.model_dump(mode="json"))
        assert response.status_code == 200
        prediction = response.json()
        assert prediction["model_version"] == manifest["bundle_version"]
        assert prediction["decision_threshold"] == manifest["threshold"]
        assert 0 <= prediction["risk_score"] <= 1
        assert client.get("/v1/model/info").json()["model_name"] == manifest["model_name"]
