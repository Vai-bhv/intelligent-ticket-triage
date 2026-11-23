import json

import pytest

from src.bundle import MANIFEST_FILENAME, MODEL_FILENAME, load_bundle, save_bundle
from src.features import FEATURE_COLUMNS


def test_bundle_round_trip_keeps_the_model_policy_and_schema_together(bundle_path):
    bundle = load_bundle(bundle_path)
    assert bundle.threshold == bundle.metadata["evaluation"]["test"]["threshold"] == 0.42
    assert bundle.metadata["model_name"] == "logistic_regression"
    assert bundle.metadata["bundle_version"] == bundle_path.name
    assert list(bundle.model.feature_names_in_) == list(FEATURE_COLUMNS)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("threshold", None, "decision threshold"),
        ("threshold", 0.5, "policy does not match"),
        ("model_name", "another_model", "policy does not match"),
        ("feature_schema", {}, "feature schema"),
    ],
)
def test_inconsistent_bundle_manifest_is_rejected(bundle_path, field, value, error):
    manifest = bundle_path / MANIFEST_FILENAME
    payload = json.loads(manifest.read_text())
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        load_bundle(bundle_path)


def test_model_bytes_must_match_manifest(bundle_path):
    model = bundle_path / MODEL_FILENAME
    model.write_bytes(model.read_bytes() + b"extra bytes")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(bundle_path)


def test_missing_manifest_has_no_threshold_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_bundle(tmp_path)


def test_saving_again_publishes_a_new_bundle(bundle_path):
    original = load_bundle(bundle_path)
    original_manifest = (bundle_path / MANIFEST_FILENAME).read_bytes()
    second_path = save_bundle(
        bundle_path.parent,
        original.model,
        original.metadata["model_name"],
        original.threshold,
        original.metadata["evaluation"],
        original.metadata["dataset_sha256"],
    )
    assert second_path != bundle_path
    assert (bundle_path / MANIFEST_FILENAME).read_bytes() == original_manifest
    assert load_bundle(second_path).threshold == original.threshold
