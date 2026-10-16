"""EfficientDet saved from the `effdet` package (DetBench wrapper: keys 'model.*' and 'anchors.boxes').
Variant, class count and input size are read from the checkpoint where it stores them."""
from PIL import Image

from .base import (Detection, NotConfigured, Prediction, draw_detections, effdet_image_size,
                   extract_state_dict, get_device, load_checkpoint, load_state_dict_checked)

FPN_CHANNELS_TO_VARIANT = {64: "efficientdet_d0", 88: "efficientdet_d1", 112: "efficientdet_d2",
                           160: "efficientdet_d3", 224: "efficientdet_d4", 288: "efficientdet_d5"}


class EffDetRunner:
    def __init__(self, ckpt_path, spec):
        import torch
        from effdet import DetBenchPredict, EfficientDet, get_efficientdet_config
        from effdet.efficientdet import HeadNet

        ck = load_checkpoint(ckpt_path)
        sd = extract_state_dict(ck)
        fpn_ch = sd["model.box_net.predict.conv_pw.weight"].shape[1]
        n_cls = sd["model.class_net.predict.conv_pw.weight"].shape[0] // 9
        size = effdet_image_size(sd["anchors.boxes"].shape[0])
        if size is None:
            raise NotConfigured("Could not work out the EfficientDet input size from the stored anchors.")
        name = (ck.get("model_name") if isinstance(ck, dict) else None) \
            or spec.config.get("effdet_name") or FPN_CHANNELS_TO_VARIANT.get(fpn_ch)
        if not name:
            raise NotConfigured(f"Unknown EfficientDet variant (FPN channels = {fpn_ch}).")

        cfg = get_efficientdet_config(name)
        cfg.num_classes, cfg.image_size = n_cls, (size, size)
        # Some variants carry extra bias weights in the FPN; read that from the checkpoint instead of guessing.
        try:
            cfg.redundant_bias = "model.fpn.cell.0.fnode.0.after_combine.conv.conv_pw.bias" in sd
        except Exception:
            pass
        net = EfficientDet(cfg, pretrained_backbone=False)
        net.class_net = HeadNet(cfg, num_outputs=cfg.num_classes)
        bench = DetBenchPredict(net)
        load_state_dict_checked(bench, sd, f"Variant used: {name}, input {size}x{size}.")

        self.device = get_device()
        self.bench = bench.to(self.device).eval()
        self.size, self.spec, self.torch = size, spec, torch
        self.mean = spec.config.get("mean") or getattr(cfg, "mean", (0.485, 0.456, 0.406))
        self.std = spec.config.get("std") or getattr(cfg, "std", (0.229, 0.224, 0.225))
        self.note = (f"EfficientDet ({name}): input is stretched to {size}x{size} and normalised with "
                     f"mean {tuple(round(m, 3) for m in self.mean)}. These come from the package defaults "
                     "and the stored anchors, not from your training notebook.")

    def predict(self, image: Image.Image, conf: float) -> Prediction:
        import numpy as np
        torch = self.torch
        rgb = image.convert("RGB")
        w, h = rgb.size
        arr = np.asarray(rgb.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            rows = self.bench(x)[0].cpu().numpy()          # x1, y1, x2, y2, score, class (input pixels)
        sx, sy = w / self.size, h / self.size
        max_score = float(rows[:, 4].max()) if len(rows) else None
        detections = [Detection(self.spec.class_name, float(r[4]),
                                [float(r[0] * sx), float(r[1] * sy), float(r[2] * sx), float(r[3] * sy)])
                      for r in rows if r[4] >= conf]
        return Prediction(draw_detections(image, detections), detections, str(self.device), [self.note], max_score)
