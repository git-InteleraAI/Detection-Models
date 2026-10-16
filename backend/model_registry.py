"""Task -> dataset -> model mapping. Checkpoint locations come from models/checkpoints.json.

Settings come from your notebooks / checkpoint report. Where a value is a default rather than
something read from a notebook, the comment says so.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CFG = json.loads((ROOT / "models" / "checkpoints.json").read_text(encoding="utf-8"))

TASKS = [
    {"id": "ship", "label": "Ship detection", "inputs": 1},
    {"id": "change", "label": "Change detection", "inputs": 2},
]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    kind: str                    # runner name in backend/inference/__init__.py
    conf: float = 0.25           # default confidence threshold (adjustable in the UI)
    class_name: str = "object"   # name shown for single-class torchvision / EfficientDet / DETR models
    imgsz: int | None = None     # Ultralytics inference size
    nms_iou: float | None = None  # extra NMS applied after the confidence filter
    note: str = ""               # shown under the model list when it is selected
    config: dict = field(default_factory=dict)


SHIP, BOAT = "ship", "boat"      # LS-SSDD yaml: names ['ship'];  MASATI yaml: names ['boat']

MODELS = {
    "ls_ssdd": [
        ModelSpec("yolo11", "YOLO11", "ultralytics_yolo", conf=0.25, imgsz=800, config={"names": {0: SHIP}}),
        ModelSpec("rtdetr", "RT-DETR", "ultralytics_rtdetr", conf=0.25, config={"names": {0: SHIP}}),
        ModelSpec("detr", "DETR", "detr", conf=0.50, class_name=SHIP,
                  config={"detr_name": "facebook/detr-resnet-50"}),
        ModelSpec("efficientdet", "EfficientDet", "efficientdet", conf=0.25, class_name=SHIP,
                  config={"effdet_name": "tf_efficientdet_d0"}),
        ModelSpec("fasterrcnn", "Faster R-CNN", "fasterrcnn", conf=0.50, class_name=SHIP),
        ModelSpec("retinanet", "RetinaNet", "retinanet", conf=0.50, class_name=SHIP),
        ModelSpec("ssd", "SSD300", "ssd", conf=0.25, class_name=SHIP, nms_iou=0.40),
    ],
    "masati": [
        ModelSpec("yolo11", "YOLO11", "ultralytics_yolo", conf=0.25, imgsz=512),
        ModelSpec("rtdetr", "RT-DETR", "ultralytics_rtdetr", conf=0.25, imgsz=640),
        ModelSpec("detr", "DETR", "detr", conf=0.50, class_name=BOAT),
        ModelSpec("efficientdet", "EfficientDet", "efficientdet", conf=0.25, class_name=BOAT,
                  config={"effdet_name": "tf_efficientdet_d0"}),
        ModelSpec("fasterrcnn", "Faster R-CNN", "fasterrcnn", conf=0.50, class_name=BOAT),
        ModelSpec("retinanet", "RetinaNet", "retinanet", conf=0.50, class_name=BOAT),
        ModelSpec("ssd", "SSD", "ssd", conf=0.25, class_name=BOAT, nms_iou=0.40),
    ],
    "levir_cd": [
        # Your trained model. No runner yet: its architecture and preprocessing are not known.
        ModelSpec("levir_cd", "LEVIR-CD trained model", "levir_cd", conf=0.50,
                  note="Setup pending: needs the model's architecture and preprocessing (see the checkpoint report)."),
        # Pretrained model from Hugging Face, ported from your Kaggle notebook (default threshold 0.18).
        ModelSpec("rfdetr_buildings", "RF-DETR new-building detection", "rfdetr_change", conf=0.18,
                  note="Downloads the pretrained model from Hugging Face on first use (needs internet). "
                       "Each image is processed in tiles, so this is slow on CPU.",
                  config={"hub_id": "merve/rf-detr-seg-satellite-buildings"}),
    ],
}


def list_tasks():
    return TASKS


def list_datasets(task: str | None = None):
    return [{"id": k, "label": v["label"], "task": v.get("task", "ship")}
            for k, v in _CFG["datasets"].items() if task is None or v.get("task", "ship") == task]


def dataset_task(dataset_id: str) -> str:
    return _CFG["datasets"][dataset_id].get("task", "ship")


def list_models(dataset_id: str):
    return MODELS[dataset_id]          # KeyError for unknown dataset


def get_spec(dataset_id: str, model_id: str) -> ModelSpec:
    for spec in MODELS[dataset_id]:
        if spec.id == model_id:
            return spec
    raise KeyError(model_id)


def checkpoint_path(dataset_id: str, model_id: str):
    """Local checkpoint path, or None for models that are downloaded (Hugging Face)."""
    name = _CFG["files"].get(dataset_id, {}).get(model_id)
    if name is None:
        return None
    ds = _CFG["datasets"][dataset_id]
    return Path(os.environ.get(ds["env"], ds["dir"])) / name
