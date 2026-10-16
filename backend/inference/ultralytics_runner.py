"""YOLO11 and RT-DETR checkpoints saved by Ultralytics."""
import numpy as np
from PIL import Image

from .base import Detection, Prediction


class UltralyticsRunner:
    def __init__(self, ckpt_path, family: str, spec):
        import torch
        from ultralytics import RTDETR, YOLO   # imported here so the API starts without them

        self.model = (RTDETR if family == "rtdetr" else YOLO)(str(ckpt_path))
        self.spec = spec
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, image: Image.Image, conf: float) -> Prediction:
        bgr = np.ascontiguousarray(np.array(image.convert("RGB"))[..., ::-1])  # Ultralytics expects BGR arrays
        kwargs = {"imgsz": self.spec.imgsz} if self.spec.imgsz else {}
        result = self.model.predict(bgr, conf=conf, verbose=False, **kwargs)[0]
        names = self.spec.config.get("names") or result.names
        result.names = names
        detections = [
            Detection(names[int(c)], float(p), [float(v) for v in box])
            for box, c, p in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
        ]
        plotted = np.ascontiguousarray(result.plot()[..., ::-1])               # back to RGB
        max_score = max((d.confidence for d in detections), default=None)
        if not detections:   # look again at a very low threshold, only to tell the user what the model saw
            low = self.model.predict(bgr, conf=0.01, verbose=False, **kwargs)[0]
            max_score = max(low.boxes.conf.tolist(), default=None)
        return Prediction(Image.fromarray(plotted), detections, self.device, max_score=max_score)
