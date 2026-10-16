"""DETR saved from Hugging Face transformers (keys 'model.*', 'class_labels_classifier', 'bbox_predictor').
The checkpoint stores 'model_name' and 'num_labels'. The first run needs internet to fetch the small
config/preprocessor files for that model name (they are cached afterwards)."""
from PIL import Image

from .base import (Detection, NotConfigured, Prediction, draw_detections, extract_state_dict,
                   get_device, load_checkpoint, load_state_dict_checked)


class DetrRunner:
    def __init__(self, ckpt_path, spec):
        import torch
        from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

        ck = load_checkpoint(ckpt_path)
        sd = extract_state_dict(ck)
        name = (ck.get("model_name") if isinstance(ck, dict) else None) or spec.config.get("detr_name")
        if not name:
            raise NotConfigured("This DETR checkpoint does not store 'model_name'.")
        n_labels = sd["class_labels_classifier.weight"].shape[0] - 1      # last output is 'no object'

        config = DetrConfig.from_pretrained(name)
        config.num_labels = n_labels
        config.use_pretrained_backbone = False                           # weights come from the checkpoint
        model = DetrForObjectDetection(config)
        load_state_dict_checked(model, sd, "If keys look renamed, update transformers: pip install -U transformers")
        self.processor = DetrImageProcessor.from_pretrained(name)
        self.device = get_device()
        self.model = model.to(self.device).eval()
        self.spec, self.torch = spec, torch
        self.note = (f"DETR ({name}) uses that model's standard image processor (resize and normalisation); "
                     "your training notebook was not available to confirm it.")

    def predict(self, image: Image.Image, conf: float) -> Prediction:
        torch = self.torch
        rgb = image.convert("RGB")
        inputs = {k: v.to(self.device) for k, v in self.processor(images=rgb, return_tensors="pt").items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[(rgb.height, rgb.width)])[0]
        boxes, scores = res["boxes"].cpu().tolist(), res["scores"].cpu().tolist()
        max_score = max(scores, default=None)
        detections = [Detection(self.spec.class_name, float(s), [float(v) for v in b])
                      for b, s in zip(boxes, scores) if s >= conf]
        return Prediction(draw_detections(image, detections), detections, str(self.device), [self.note], max_score)
