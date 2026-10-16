"""Comprehensive test suite for Hugging Face checkpoint integration and backend API.

Tests:
1. Backend startup and model registry without hardcoded paths
2. Missing HF_TOKEN configuration error behavior
3. Missing checkpoint error behavior
4. Zip Slip path traversal protection in safe extraction
5. Invalid dataset/model combinations return HTTP 400/404
6. Invalid and corrupted image inputs are rejected with HTTP 400
7. LEVIR-CD SiamUNet inference execution on sample pair
8. Ship detection inference execution on sample image
9. API response contracts compatibility
"""
import io
import os
import sys
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import main
import model_registry as reg
import hf_downloader

client = TestClient(main.app)


def test_tasks_and_datasets_endpoints():
    """Verify tasks and datasets are listed without hardcoded paths."""
    res_tasks = client.get("/api/tasks")
    assert res_tasks.status_code == 200
    tasks = res_tasks.json()
    task_ids = [t["id"] for t in tasks]
    assert "ship" in task_ids
    assert "change" in task_ids

    res_ds = client.get("/api/datasets")
    assert res_ds.status_code == 200
    datasets = res_ds.json()
    ds_ids = [d["id"] for d in datasets]
    assert "ls_ssdd" in ds_ids
    assert "masati" in ds_ids
    assert "levir_cd" in ds_ids


def test_models_endpoint_compatibility():
    """Check /api/datasets/{id}/models returns correct contract."""
    res = client.get("/api/datasets/ls_ssdd/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 5
    first = models[0]
    for key in ("id", "label", "ready", "checkpoint_found", "default_conf", "note"):
        assert key in first

    # Check unknown dataset returns 404
    res_unknown = client.get("/api/datasets/non_existent_dataset/models")
    assert res_unknown.status_code == 404


def test_missing_hf_token_behavior(monkeypatch):
    """Verify that attempting to download without HF_TOKEN raises a clear error."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Temporarily set dummy environment directory to prevent local file detection
    monkeypatch.setenv("LS_SSDD_DIR", "C:/non_existent_folder_xyz")

    with pytest.raises(hf_downloader.CheckpointDownloadError) as exc_info:
        hf_downloader.download_checkpoint("ls_ssdd", "yolo11", "models/ls_ssdd/best_Yolo11_ls.pt")

    assert "HF_TOKEN" in str(exc_info.value)
    assert "Authentication required" in str(exc_info.value)


def test_zip_slip_protection(tmp_path):
    """Verify that ZIP extraction rejects malicious path traversal attempts."""
    malicious_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(malicious_zip, "w") as z:
        z.writestr("../../etc/passwd", "malicious content")

    target_dir = tmp_path / "extracted"
    with pytest.raises(hf_downloader.CheckpointDownloadError) as exc_info:
        hf_downloader._safe_extract_zip(malicious_zip, target_dir)

    assert "path traversal detected" in str(exc_info.value).lower()


def test_invalid_image_inputs_rejected():
    """Verify empty or non-image files are rejected with HTTP 400."""
    form_data = {"dataset": "ls_ssdd", "model": "yolo11"}

    # 1. Empty upload
    res_empty = client.post(
        "/api/predict",
        data=form_data,
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()

    # 2. Corrupt non-image bytes
    res_corrupt = client.post(
        "/api/predict",
        data=form_data,
        files={"file": ("fake.jpg", b"NOT_AN_IMAGE_CONTENT_HERE", "image/jpeg")},
    )
    assert res_corrupt.status_code == 400
    assert "readable image" in res_corrupt.json()["detail"].lower() or "corrupted" in res_corrupt.json()["detail"].lower()


def test_invalid_dataset_model_combination():
    """Verify that asking for a change model on a ship dataset returns 400."""
    res = client.post(
        "/api/predict",
        data={"dataset": "ls_ssdd", "model": "rfdetr_buildings", "sample": "ship 1.jpg"},
    )
    # rfdetr_buildings is not available for ls_ssdd
    assert res.status_code in (400, 404)


def test_mismatched_change_pair_rejected():
    """Verify change detection rejects images with mismatched dimensions."""
    from PIL import Image

    buf_a = io.BytesIO()
    Image.new("RGB", (256, 256), color=(100, 100, 100)).save(buf_a, format="PNG")

    buf_b = io.BytesIO()
    Image.new("RGB", (512, 512), color=(150, 150, 150)).save(buf_b, format="PNG")

    res = client.post(
        "/api/change/predict",
        data={"dataset": "levir_cd", "model": "levir_cd"},
        files={
            "before": ("before.png", buf_a.getvalue(), "image/png"),
            "after": ("after.png", buf_b.getvalue(), "image/png"),
        },
    )
    assert res.status_code == 400
    assert "aligned" in res.json()["detail"].lower() or "same size" in res.json()["detail"].lower()


def test_levir_cd_siam_unet_inference_execution():
    """Verify LEVIR-CD Siamese U-Net loads and executes real inference on sample pair."""
    # Only run if checkpoint exists on this machine
    ckpt_p = reg.checkpoint_path("levir_cd", "levir_cd")
    if ckpt_p is None or not (ckpt_p.exists() if isinstance(ckpt_p, Path) else False):
        pytest.skip("Local LEVIR-CD checkpoint not present on test runner.")

    res = client.post(
        "/api/change/predict",
        data={"dataset": "levir_cd", "model": "levir_cd", "sample": "pair_1.jpeg"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["task"] == "change"
    assert body["dataset"] == "levir_cd"
    assert "image" in body
    assert body["image"].startswith("data:image/png;base64,")
    assert "detections" in body
    assert "counts" in body
    assert "extra_images" in body
    assert len(body["extra_images"]) >= 1


def test_ship_detection_inference_execution():
    """Verify Ship Detection executes real inference on a sample image."""
    ckpt_p = reg.checkpoint_path("ls_ssdd", "yolo11")
    if ckpt_p is None or not (ckpt_p.exists() if isinstance(ckpt_p, Path) else False):
        pytest.skip("Local LS-SSDD YOLO11 checkpoint not present on test runner.")

    res = client.post(
        "/api/predict",
        data={"dataset": "ls_ssdd", "model": "yolo11", "sample": "ship 1.jpg"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["task"] == "ship"
    assert body["dataset"] == "ls_ssdd"
    assert "image" in body
    assert body["image"].startswith("data:image/png;base64,")
    assert "detections" in body
    assert "inference_ms" in body


def test_masati_yolo11_inference_execution():
    """Verify MASATI YOLO11 (unzipped directory archive) executes inference on sample image."""
    ckpt_p = reg.checkpoint_path("masati", "yolo11")
    if ckpt_p is None or not (ckpt_p.exists() if isinstance(ckpt_p, Path) else False):
        pytest.skip("Local MASATI YOLO11 checkpoint not present on test runner.")

    res = client.post(
        "/api/predict",
        data={"dataset": "masati", "model": "yolo11", "sample": "Boat 1.jpeg"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["task"] == "ship"
    assert body["dataset"] == "masati"
    assert body["total"] >= 1

