"""Faster R-CNN, RetinaNet and SSD300 (torchvision). The class count is read from the checkpoint's
head shapes, so it cannot disagree with the weights."""
from PIL import Image

from .base import (Detection, Prediction, draw_detections, extract_state_dict, get_device,
                   load_checkpoint, load_state_dict_checked)


class TorchvisionRunner:
    def __init__(self, ckpt_path, family: str, spec):
        import torch
        from torchvision.models import detection as det

        sd = extract_state_dict(load_checkpoint(ckpt_path))
        if family == "fasterrcnn":
            n = sd["roi_heads.box_predictor.cls_score.weight"].shape[0]          # includes background
            model = det.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=n)
        elif family == "retinanet":
            n = sd["head.classification_head.cls_logits.weight"].shape[0] // 9   # 9 anchors per location
            model = det.retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=n)
        elif family == "ssd":
            n = sd["head.classification_head.module_list.2.weight"].shape[0] // 6  # 6 anchors on that layer
            model = det.ssd300_vgg16(weights=None, weights_backbone=None, num_classes=n)
        else:
            raise ValueError(family)
        load_state_dict_checked(model, sd)
        # Lower the model's own cut-off (default 0.05) so the UI threshold can go down to 0.01.
        if family == "fasterrcnn":
            model.roi_heads.score_thresh = 0.01
        elif family == "retinanet":
            model.score_thresh = 0.01
        self.device = get_device()
        self.model = model.to(self.device).eval()
        self.spec, self.num_classes = spec, n
        self.torch = torch

    def predict(self, image: Image.Image, conf: float) -> Prediction:
        import torchvision
        from torchvision.transforms.functional import to_tensor

        torch = self.torch
        tensor = to_tensor(image.convert("RGB")).to(self.device)
        with torch.inference_mode():
            out = self.model([tensor])[0]
        boxes, scores = out["boxes"], out["scores"]
        max_score = float(scores.max()) if len(scores) else None
        keep = scores >= conf
        boxes, scores = boxes[keep], scores[keep]
        if self.spec.nms_iou and len(boxes):
            idx = torchvision.ops.nms(boxes, scores, self.spec.nms_iou)
            boxes, scores = boxes[idx], scores[idx]
        # Both datasets have a single object class, so every kept detection gets that class name.
        detections = [Detection(self.spec.class_name, float(s), [float(v) for v in b])
                      for b, s in zip(boxes.cpu().tolist(), scores.cpu().tolist())]
        return Prediction(draw_detections(image, detections), detections, str(self.device), max_score=max_score)
