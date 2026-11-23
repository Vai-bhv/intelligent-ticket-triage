"""Persist and load a versioned model and its decision policy."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from uuid import uuid4

from sklearn.pipeline import Pipeline

if __package__:
    from .features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, RAW_FEATURE_COLUMNS
else:
    from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, RAW_FEATURE_COLUMNS


BUNDLE_FORMAT_VERSION = 1
MODEL_FILENAME = "model.pkl"
MANIFEST_FILENAME = "manifest.json"
MODEL_LIBRARIES = ("numpy", "pandas", "scikit-learn", "scipy", "xgboost")


@dataclass(frozen=True)
class ModelBundle:
    model: Pipeline
    metadata: dict

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


def feature_schema() -> dict:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "raw_columns": list(RAW_FEATURE_COLUMNS),
        "model_columns": list(FEATURE_COLUMNS),
        "timestamp_timezone": "UTC",
    }


def library_versions() -> dict[str, str]:
    return {name: version(name) for name in MODEL_LIBRARIES}


def estimator_type(model: Pipeline) -> str:
    estimator = type(model.steps[-1][1])
    return f"{estimator.__module__}.{estimator.__qualname__}"


def read_manifest(path: Path) -> dict:
    """Read and validate model, decision policy, and evaluation metadata."""
    metadata = json.loads((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsupported model bundle format.")
    if metadata.get("feature_schema") != feature_schema():
        raise ValueError("Model bundle feature schema does not match this application.")

    threshold = metadata.get("threshold")
    if type(threshold) not in (int, float) or not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Model bundle must contain a decision threshold between 0 and 1.")

    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Model bundle is missing evaluation metadata.")
    for partition in ("validation", "test"):
        result = evaluation.get(partition)
        if (
            not isinstance(result, dict)
            or result.get("threshold") != threshold
            or result.get("model_name") != metadata.get("model_name")
        ):
            raise ValueError(f"Model bundle {partition} policy does not match its model and threshold.")
        metrics = result.get("metrics")
        if not isinstance(metrics, dict) or any(
            type(metrics.get(name)) not in (int, float) or not isfinite(metrics[name])
            for name in ("average_precision", "roc_auc", "precision", "recall", "f2")
        ):
            raise ValueError(f"Model bundle is missing valid {partition} metrics.")

    for field in ("bundle_version", "model_name", "estimator_type", "model_sha256"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError(f"Model bundle is missing {field}.")
    return metadata


def save_bundle(
    output_dir: Path,
    model: Pipeline,
    model_name: str,
    threshold: float,
    evaluation: dict,
    dataset_sha256: str,
) -> Path:
    """Publish a complete new bundle without updating any existing release."""
    created_at = datetime.now(timezone.utc)
    bundle_version = f"{created_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:12]}"
    path = output_dir / bundle_version
    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "bundle_version": bundle_version,
        "created_at": created_at.isoformat(),
        "model_name": model_name,
        "estimator_type": estimator_type(model),
        "model_sha256": sha256(model_bytes).hexdigest(),
        "threshold": threshold,
        "feature_schema": feature_schema(),
        "dataset_sha256": dataset_sha256,
        "library_versions": library_versions(),
        "evaluation": evaluation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temp_dir:
        staging = Path(temp_dir) / "bundle"
        staging.mkdir()
        (staging / MODEL_FILENAME).write_bytes(model_bytes)
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        read_manifest(staging)
        if path.exists():
            raise FileExistsError(f"Model bundle already exists at {path}.")
        staging.rename(path)
    return path


def load_bundle(path: Path) -> ModelBundle:
    """Load a trusted local bundle after validating schema and model integrity."""
    metadata = read_manifest(path)
    if metadata.get("library_versions") != library_versions():
        raise ValueError("Model bundle dependency versions differ from this environment.")
    model_bytes = (path / MODEL_FILENAME).read_bytes()
    if sha256(model_bytes).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model file checksum does not match the bundle manifest.")

    # The checksum detects accidental mixing or corruption. Only load trusted bundles.
    model = pickle.loads(model_bytes)
    if (
        not isinstance(model, Pipeline)
        or estimator_type(model) != metadata["estimator_type"]
        or list(model.feature_names_in_) != metadata["feature_schema"]["model_columns"]
        or list(model.classes_) != [0, 1]
    ):
        raise ValueError("Fitted model does not match the bundle manifest.")
    return ModelBundle(model, metadata)
