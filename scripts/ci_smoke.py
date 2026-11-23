"""Train a local synthetic bundle and check the container over HTTP."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TICKET = {
    "description": "Shared service is unavailable and blocking multiple users in production",
    "region": "Europe",
    "category": "Infrastructure",
    "priority": "Medium",
    "assigned_team": "Global Operations",
    "channel": "Portal",
    "backlog_at_creation": 55,
    "team_experience_years": 3.0,
    "previous_reassignments": 1,
    "sla_target_hours": 36,
    "created_at": "2024-08-31T10:00:00Z",
}


def docker_output(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True, timeout=30).strip()


def request(base: str, path: str, payload: dict | None = None) -> dict:
    query = Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(query, timeout=3) as response:
        return json.load(response)


def check_api(base: str, manifest: dict) -> None:
    deadline = monotonic() + 45
    while True:
        try:
            assert request(base, "/health/ready") == {"status": "ready"}
            break
        except OSError as exc:
            if monotonic() >= deadline:
                raise RuntimeError("Container did not become ready within 45 seconds.") from exc
            sleep(0.5)

    assert request(base, "/health/live") == {"status": "ok"}
    info = request(base, "/v1/model/info")
    prediction = request(base, "/v1/predict", TICKET)
    for result in (info, prediction):
        assert result["model_version"] == manifest["bundle_version"], result
        assert result["decision_threshold"] == manifest["threshold"], result
    assert info["model_name"] == manifest["model_name"], info
    assert info["evaluation_data"] == "synthetic", info
    assert 0.0 <= prediction["risk_score"] <= 1.0, prediction
    assert prediction["recommendations"], prediction
    assert prediction == request(base, "/v1/predict", TICKET), prediction
    print(f"Container smoke passed for {prediction['model_version']}: score={prediction['risk_score']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="intelligent-ticket-triage:ci")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="triage-ci-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        directory = Path(temporary).resolve()
        subprocess.run(
            [sys.executable, "src/train.py", "--output-dir", str(directory)],
            cwd=ROOT, env={**os.environ, "OMP_THREAD_LIMIT": "2", "OPENBLAS_NUM_THREADS": "2"},
            timeout=120, check=True,
        )
        bundle, = directory.iterdir()
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        container = docker_output(
            "run", "--detach", "--publish", "127.0.0.1::8000",
            "--mount", f"type=bind,source={bundle},target=/app/model_bundle,readonly",
            args.image,
        )
        try:
            address = docker_output("port", container, "8000/tcp")
            check_api(f"http://{address}", manifest)
        except Exception:
            subprocess.run(["docker", "logs", container], timeout=15, check=False)
            raise
        finally:
            subprocess.run(["docker", "rm", "-f", container], timeout=30, check=True)


if __name__ == "__main__":
    main()
