"""HTTP API.

Skipped when the ``serve`` extra is not installed, so the core suite still runs
on a training-only environment.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="install wasteclf[serve] to run the serving tests")

from fastapi.testclient import TestClient  # noqa: E402

from wasteclf.serving.app import MAX_UPLOAD_BYTES, create_app  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.needs_tf]


@pytest.fixture(scope="module")
def client(trained):
    result, _, _ = trained
    with TestClient(create_app(result.run.path, threshold=0.9)) as c:
        yield c


@pytest.fixture(scope="module")
def sample_bytes(synthetic_root):
    return sorted((synthetic_root / "paper").glob("*.jpg"))[0].read_bytes()


def test_health_reports_the_loaded_contract(client, trained):
    _, manifest, _ = trained
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["classes"] == manifest.class_names
    assert payload["image_size"] == [32, 32]


def test_classes_endpoint(client, trained):
    _, manifest, _ = trained
    payload = client.get("/classes").json()
    assert payload["count"] == manifest.num_classes


def test_predict_returns_a_full_distribution(client, sample_bytes, trained):
    _, manifest, _ = trained
    response = client.post("/predict", files={"file": ("p.jpg", sample_bytes, "image/jpeg")})
    assert response.status_code == 200

    payload = response.json()
    assert payload["label"] in manifest.class_names
    assert set(payload["scores"]) == set(manifest.class_names)
    assert sum(payload["scores"].values()) == pytest.approx(1.0, abs=1e-3)
    assert "latency_ms" in payload
    # threshold=0.9 on an under-trained model: the flag must track the number.
    assert payload["low_confidence"] == (payload["confidence"] < 0.9)


def test_batch_predicts_every_image(client, sample_bytes):
    files = [("files", (f"i{i}.jpg", sample_bytes, "image/jpeg")) for i in range(3)]
    payload = client.post("/predict/batch", files=files).json()
    assert len(payload["predictions"]) == 3
    assert payload["failed"] == []


def test_batch_survives_one_bad_file(client, sample_bytes):
    files = [
        ("files", ("good.jpg", sample_bytes, "image/jpeg")),
        ("files", ("bad.jpg", b"definitely not an image", "image/jpeg")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200

    payload = response.json()
    assert len(payload["predictions"]) == 1
    assert len(payload["failed"]) == 1
    assert payload["failed"][0]["path"] == "bad.jpg"


def test_empty_upload_is_a_400(client):
    response = client.post("/predict", files={"file": ("e.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_undecodable_upload_is_a_400_not_a_500(client):
    response = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"]


def test_oversized_upload_is_a_413(client):
    blob = b"0" * (MAX_UPLOAD_BYTES + 1)
    response = client.post("/predict", files={"file": ("big.jpg", blob, "image/jpeg")})
    assert response.status_code == 413
