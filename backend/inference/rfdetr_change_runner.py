"""New-building detection with the pretrained RF-DETR instance-segmentation model.

Ported from your Kaggle notebook (rf-detr-seg-satellite-buildings): detect buildings in the BEFORE and
AFTER images using overlapping tiles, then keep the AFTER buildings that do not overlap a BEFORE building.
The model is downloaded from Hugging Face the first time (internet needed), then cached.
"""
import numpy as np
from PIL import Image, ImageDraw

from .base import Detection, NotConfigured, Prediction, get_device

# Values from the notebook
TILE_SIZE = 384
TILE_OVERLAP = 96
MIN_BUILDING_AREA = 18
DUPLICATE_IOU = 0.35
DUPLICATE_SMALL_OVERLAP = 0.70
MAX_PAST_OVERLAP = 0.20          # an after-building overlapping a before-building by >= 20% is "existing"
ALIGNMENT_TOLERANCE = 3          # pixels of dilation to forgive small misalignment


def tile_positions(length, tile_size, overlap):
    tile = min(tile_size, length)
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def mask_overlap(a, b):
    """IoU and overlap-over-smaller for two cropped masks placed by their global boxes."""
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if x2 < x1 or y2 < y1:
        return 0.0, 0.0
    a_region = a["mask"][y1 - ay1: y2 - ay1 + 1, x1 - ax1: x2 - ax1 + 1]
    b_region = b["mask"][y1 - by1: y2 - by1 + 1, x1 - bx1: x2 - bx1 + 1]
    if a_region.shape != b_region.shape or a_region.size == 0:
        return 0.0, 0.0
    inter = int(np.logical_and(a_region, b_region).sum())
    if inter == 0:
        return 0.0, 0.0
    union = a["area"] + b["area"] - inter
    return float(inter / max(1, union)), float(inter / max(1, min(a["area"], b["area"])))


def remove_duplicates(detections):
    kept = []
    for cand in sorted(detections, key=lambda d: (d["score"], d["area"]), reverse=True):
        for existing in kept:
            iou, small = mask_overlap(cand, existing)
            if iou >= DUPLICATE_IOU or small >= DUPLICATE_SMALL_OVERLAP:
                break
        else:
            kept.append(cand)
    return kept


def make_occupancy(buildings, width, height):
    occ = np.zeros((height, width), dtype=np.uint8)
    for b in buildings:
        x1, y1, x2, y2 = b["box"]
        occ[y1:y2 + 1, x1:x2 + 1][b["mask"]] = 1
    return occ


def find_new_buildings(past, present, width, height):
    import cv2
    occ = make_occupancy(past, width, height)
    if ALIGNMENT_TOLERANCE > 0:
        k = ALIGNMENT_TOLERANCE * 2 + 1
        occ = cv2.dilate(occ, np.ones((k, k), dtype=np.uint8), iterations=1)
    occ = occ.astype(bool)
    new = []
    for b in present:
        x1, y1, x2, y2 = b["box"]
        overlap = np.logical_and(b["mask"], occ[y1:y2 + 1, x1:x2 + 1]).sum() / max(1, b["area"])
        if overlap < MAX_PAST_OVERLAP:
            c = b.copy()
            c["past_overlap"] = float(overlap)
            new.append(c)
    new.sort(key=lambda d: d["area"], reverse=True)
    return new


def render_buildings(image, buildings, numbered=False):
    rgb = np.asarray(image.convert("RGB")).copy()
    for b in buildings:
        x1, y1, x2, y2 = b["box"]
        region = rgb[y1:y2 + 1, x1:x2 + 1]
        m = b["mask"]
        region[m] = (region[m].astype(np.float32) * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
    result = Image.fromarray(rgb)
    if numbered:
        draw = ImageDraw.Draw(result)
        width = max(2, image.width // 450)
        for i, b in enumerate(buildings, start=1):
            x1, y1, x2, y2 = b["box"]
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=width)
            draw.text((x1, max(0, y1 - 20)), str(i), fill=(0, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    return result


class RfDetrChangeRunner:
    def __init__(self, spec):
        import torch
        try:
            from transformers import AutoImageProcessor, RfDetrForInstanceSegmentation
        except ImportError as e:
            raise NotConfigured('RF-DETR needs a newer transformers. Run: pip install -U "transformers>=5.9,<6" '
                                '"huggingface-hub>=1.23,<2" and restart the backend.') from e
        hub_id = spec.config["hub_id"]
        self.processor = AutoImageProcessor.from_pretrained(hub_id, use_fast=True)
        model = RfDetrForInstanceSegmentation.from_pretrained(hub_id, use_safetensors=True)
        self.device = get_device()
        self.model = model.to(self.device).eval()
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.building_ids = {k for k, name in labels.items() if "build" in name}
        if not self.building_ids:
            raise NotConfigured(f"Building class not found in the model's labels: {labels}")
        self.torch = torch

    def _detect_tile(self, tile, conf):
        import cv2
        torch = self.torch
        tile = tile.convert("RGB")
        inputs = {k: v.to(self.device) for k, v in self.processor(images=tile, return_tensors="pt").items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        pred = self.processor.post_process_instance_segmentation(
            outputs, threshold=conf, target_sizes=[(tile.height, tile.width)])[0]
        seg = pred.get("segmentation")
        if seg is None:
            return []
        seg = seg.cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
        found = []
        for segment in pred.get("segments_info", []):
            label_id = segment.get("label_id", segment.get("category_id"))
            if label_id is None or int(label_id) not in self.building_ids:
                continue
            mask = (seg == int(segment["id"])).astype(np.uint8)
            if mask.shape != (tile.height, tile.width):
                mask = cv2.resize(mask, (tile.width, tile.height), interpolation=cv2.INTER_NEAREST)
            if int(mask.sum()) < MIN_BUILDING_AREA:
                continue
            found.append({"mask": mask.astype(bool), "score": float(segment.get("score", 0.0))})
        return found

    def _detect_buildings(self, image, conf):
        image = image.convert("RGB")
        width, height = image.size
        tw, th = min(TILE_SIZE, width), min(TILE_SIZE, height)
        detections = []
        for y in tile_positions(height, th, TILE_OVERLAP):
            for x in tile_positions(width, tw, TILE_OVERLAP):
                for item in self._detect_tile(image.crop((x, y, x + tw, y + th)), conf):
                    ys, xs = np.where(item["mask"])
                    if len(xs) == 0:
                        continue
                    lx1, ly1, lx2, ly2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    crop = item["mask"][ly1:ly2 + 1, lx1:lx2 + 1]
                    detections.append({"mask": crop.copy(), "area": int(crop.sum()),
                                       "box": (x + lx1, y + ly1, x + lx2, y + ly2), "score": item["score"]})
        return remove_duplicates(detections)

    def predict_pair(self, before: Image.Image, after: Image.Image, conf: float) -> Prediction:
        width, height = after.size
        past = self._detect_buildings(before, conf)
        present = self._detect_buildings(after, conf)
        new = find_new_buildings(past, present, width, height)
        detections = [Detection("new building", b["score"], [float(v) for v in b["box"]]) for b in new]
        counts = {"buildings before": len(past), "buildings after": len(present), "new buildings": len(new)}
        extras = [("Before: all detected buildings", render_buildings(before, past)),
                  ("After: all detected buildings", render_buildings(after, present))]
        return Prediction(render_buildings(after, new, numbered=True), detections, str(self.device),
                          counts=counts, extra_images=extras)
