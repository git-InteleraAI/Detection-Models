import base64
import io
import logging
import time
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

import inference
import model_registry as reg
from inference import CheckpointMissing, NotConfigured

log = logging.getLogger("tester")
app = FastAPI(title="Model Tester API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])

# samples/<ship dataset id>/*.jpg            one image per file
# samples/<change dataset id>/A/*.png         "before" images   (same file names in B = "after")
# samples/<change dataset id>/B/*.png
SAMPLES_DIR = reg.ROOT / "samples"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder: Path) -> dict:
    if not folder.is_dir():
        return {}
    return {p.name: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXT}


def pair_names(dataset_id: str) -> list:
    a = image_files(SAMPLES_DIR / dataset_id / "A")
    b = image_files(SAMPLES_DIR / dataset_id / "B")
    return sorted(set(a) & set(b))


def spec_for(dataset: str, model: str, task: str):
    try:
        spec = reg.get_spec(dataset, model)
    except KeyError:
        raise HTTPException(404, f"Model '{model}' is not available for dataset '{dataset}'.")
    if reg.dataset_task(dataset) != task:
        raise HTTPException(400, f"Dataset '{dataset}' is not a {task}-detection dataset.")
    return spec


def threshold_for(spec, conf):
    value = spec.conf if conf is None else conf
    if not 0 < value < 1:
        raise HTTPException(400, "Confidence threshold must be between 0 and 1.")
    return value


def open_image(raw: bytes, what: str = "uploaded file") -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"The {what} is not a readable image (use PNG, JPG, or similar).")


def run(dataset: str, model: str, spec, action):
    try:
        runner = inference.get_runner(f"{dataset}/{model}", spec, reg.checkpoint_path(dataset, model))
        start = time.perf_counter()
        prediction = action(runner)
        return prediction, (time.perf_counter() - start) * 1000
    except CheckpointMissing as e:
        raise HTTPException(404, str(e))
    except NotConfigured as e:
        raise HTTPException(501, str(e))
    except Exception as e:
        log.exception("Inference failed")
        raise HTTPException(500, f"Inference failed: {type(e).__name__}: {e}")


def encode(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_response(task, dataset, spec, prediction, elapsed_ms, size, threshold):
    counts = prediction.counts if prediction.counts is not None else dict(Counter(d.label for d in prediction.detections))
    return {
        "task": task,
        "dataset": dataset,
        "model": spec.label,
        "image": encode(prediction.image),
        "image_size": list(size),
        "detections": [{"label": d.label, "confidence": round(d.confidence, 4),
                        "box": [round(v, 1) for v in d.box]} for d in prediction.detections],
        "counts": counts,
        "total": len(prediction.detections),
        "inference_ms": round(elapsed_ms, 1),
        "device": prediction.device,
        "confidence_threshold": threshold,
        "notes": prediction.notes,
        "max_score": None if prediction.max_score is None else round(prediction.max_score, 3),
        "extra_images": [{"title": t, "image": encode(i)} for t, i in prediction.extra_images],
    }


@app.get("/api/tasks")
def tasks():
    return reg.list_tasks()


@app.get("/api/datasets")
def datasets(task: str | None = None):
    return reg.list_datasets(task)


@app.get("/api/datasets/{dataset_id}/models")
def models(dataset_id: str):
    try:
        specs = reg.list_models(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset '{dataset_id}'.")
    out = []
    for s in specs:
        ckpt = reg.checkpoint_path(dataset_id, s.id)
        out.append({"id": s.id, "label": s.label, "ready": inference.is_ready(s),
                    "checkpoint_found": True if ckpt is None else ckpt.exists(),   # None = downloaded model
                    "default_conf": s.conf, "note": s.note or None})
    return out


@app.get("/api/datasets/{dataset_id}/samples")
def samples(dataset_id: str):
    if dataset_id not in {d["id"] for d in reg.list_datasets()}:
        raise HTTPException(404, f"Unknown dataset '{dataset_id}'.")
    if reg.dataset_task(dataset_id) == "change":
        return pair_names(dataset_id)
    return list(image_files(SAMPLES_DIR / dataset_id))


@app.get("/api/samples/{dataset_id}/{filename}")
def sample_image(dataset_id: str, filename: str):
    path = image_files(SAMPLES_DIR / dataset_id).get(filename)
    if path is None:
        raise HTTPException(404, "Sample image not found.")
    return FileResponse(path)


@app.get("/api/pair-samples/{dataset_id}/{side}/{filename}")
def pair_sample_image(dataset_id: str, side: str, filename: str):
    if side not in ("A", "B"):
        raise HTTPException(404, "Side must be A (before) or B (after).")
    path = image_files(SAMPLES_DIR / dataset_id / side).get(filename)
    if path is None:
        raise HTTPException(404, "Sample image not found.")
    return FileResponse(path)


@app.post("/api/predict")
def predict(dataset: str = Form(...), model: str = Form(...), conf: float | None = Form(None),
            sample: str | None = Form(None), file: UploadFile | None = File(None)):
    spec = spec_for(dataset, model, "ship")
    threshold = threshold_for(spec, conf)

    if file is not None:
        raw = file.file.read()
    elif sample:
        path = image_files(SAMPLES_DIR / dataset).get(sample)
        if path is None:
            raise HTTPException(404, f"Sample image '{sample}' was not found in samples/{dataset}.")
        raw = path.read_bytes()
    else:
        raise HTTPException(400, "Choose a sample image or upload one.")
    image = open_image(raw)

    prediction, ms = run(dataset, model, spec, lambda r: r.predict(image, threshold))
    return build_response("ship", dataset, spec, prediction, ms, image.size, threshold)


@app.post("/api/change/predict")
def predict_change(dataset: str = Form(...), model: str = Form(...), conf: float | None = Form(None),
                   sample: str | None = Form(None),
                   before: UploadFile | None = File(None), after: UploadFile | None = File(None)):
    spec = spec_for(dataset, model, "change")
    threshold = threshold_for(spec, conf)

    if sample:
        a = image_files(SAMPLES_DIR / dataset / "A").get(sample)
        b = image_files(SAMPLES_DIR / dataset / "B").get(sample)
        if a is None or b is None:
            raise HTTPException(404, f"Sample pair '{sample}' needs a file with that name in both samples/{dataset}/A and /B.")
        raw_before, raw_after = a.read_bytes(), b.read_bytes()
    elif before is not None and after is not None:
        raw_before, raw_after = before.file.read(), after.file.read()
    else:
        raise HTTPException(400, "Choose a sample pair, or upload both the before and the after image.")

    img_before, img_after = open_image(raw_before, "before image"), open_image(raw_after, "after image")
    if img_before.size != img_after.size:
        raise HTTPException(400, f"The before image is {img_before.width}x{img_before.height} but the after image is "
                                 f"{img_after.width}x{img_after.height}. They must be the same size and aligned.")

    prediction, ms = run(dataset, model, spec, lambda r: r.predict_pair(img_before, img_after, threshold))
    return build_response("change", dataset, spec, prediction, ms, img_after.size, threshold)
