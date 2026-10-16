"""Siamese U-Net (FC-Siam-conc) change detection runner for LEVIR-CD.

The model accepts registered 'before' and 'after' optical images, extracts dual-branch features,
concatenates them across stages, and predicts a binary change probability mask.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import (
    Detection,
    Prediction,
    extract_state_dict,
    get_device,
    load_checkpoint,
    load_state_dict_checked,
    resolve_checkpoint,
)


def build_siam_unet(torch, nn):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch):
            super().__init__()
            self.conv1 = ConvBlock(in_ch, mid_ch)
            self.conv2 = ConvBlock(mid_ch, out_ch)

        def forward(self, x):
            return self.conv2(self.conv1(x))

    class UpBlock(nn.Module):
        def __init__(self, up_in, up_out, cat_in, conv_out):
            super().__init__()
            self.upconv = nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2)
            self.conv1 = ConvBlock(cat_in, conv_out)
            self.conv2 = ConvBlock(conv_out, conv_out)

        def forward(self, x, skip_a, skip_b):
            x = self.upconv(x)
            x = torch.cat([x, skip_a, skip_b], dim=1)
            return self.conv2(self.conv1(x))

    class SiamUNetConc(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2, 2)
            self.encoder1a = DoubleConv(3, 64, 64)
            self.encoder1b = DoubleConv(3, 64, 64)
            self.encoder2a = DoubleConv(64, 128, 128)
            self.encoder2b = DoubleConv(64, 128, 128)
            self.encoder3a = DoubleConv(128, 256, 256)
            self.encoder3b = DoubleConv(128, 256, 256)
            self.encoder4a = DoubleConv(256, 512, 512)
            self.encoder4b = DoubleConv(256, 512, 512)

            self.bottleneck = ConvBlock(1024, 1024)

            self.decoder4 = UpBlock(1024, 512, 1536, 512)
            self.decoder3 = UpBlock(512, 256, 768, 256)
            self.decoder2 = UpBlock(256, 128, 384, 128)
            self.decoder1 = UpBlock(128, 64, 192, 64)
            self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        def forward(self, x1, x2):
            e1a = self.encoder1a(x1)
            e2a = self.encoder2a(self.pool(e1a))
            e3a = self.encoder3a(self.pool(e2a))
            e4a = self.encoder4a(self.pool(e3a))

            e1b = self.encoder1b(x2)
            e2b = self.encoder2b(self.pool(e1b))
            e3b = self.encoder3b(self.pool(e2b))
            e4b = self.encoder4b(self.pool(e3b))

            b = self.bottleneck(self.pool(torch.cat([e4a, e4b], dim=1)))

            d4 = self.decoder4(b, e4a, e4b)
            d3 = self.decoder3(d4, e3a, e3b)
            d2 = self.decoder2(d3, e2a, e2b)
            d1 = self.decoder1(d2, e1a, e1b)
            return self.final_conv(d1)

    return SiamUNetConc()


class SiamUnetRunner:
    def __init__(self, ckpt_path: Path, spec):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = get_device()
        self.spec = spec

        real_ckpt = resolve_checkpoint(ckpt_path)
        ck = load_checkpoint(real_ckpt)
        sd = extract_state_dict(ck)

        model = build_siam_unet(torch, nn)
        load_state_dict_checked(model, sd, "SiamUNet architecture mismatch.")
        self.model = model.to(self.device).eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict_pair(self, before: Image.Image, after: Image.Image, conf: float) -> Prediction:
        import cv2
        torch = self.torch

        w, h = after.size
        # The model downsamples 4 times (stride 16), so dimensions must be multiples of 16
        pad_w = (16 - (w % 16)) % 16
        pad_h = (16 - (h % 16)) % 16
        target_w, target_h = w + pad_w, h + pad_h

        img1 = before.convert("RGB")
        img2 = after.convert("RGB")
        if pad_w or pad_h:
            img1 = img1.resize((target_w, target_h), Image.BILINEAR)
            img2 = img2.resize((target_w, target_h), Image.BILINEAR)

        arr1 = (np.array(img1, dtype=np.float32) / 255.0 - self.mean) / self.std
        arr2 = (np.array(img2, dtype=np.float32) / 255.0 - self.mean) / self.std

        t1 = torch.from_numpy(arr1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(arr2).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(t1, t2)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        if pad_w or pad_h:
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

        max_score = float(prob.max()) if prob.size > 0 else None
        binary_mask = (prob >= conf).astype(np.uint8)

        # Find connected components / change contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        min_area = 15

        annotated = np.array(after.convert("RGB")).copy()
        # Create semi-transparent overlay on changed areas (bright red-orange)
        change_pixels = binary_mask.astype(bool)
        if np.any(change_pixels):
            overlay_color = np.array([255, 69, 0], dtype=np.uint8)
            annotated[change_pixels] = (
                annotated[change_pixels].astype(np.float32) * 0.45 + overlay_color * 0.55
            ).astype(np.uint8)

        output_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(output_image)
        line_w = max(2, w // 400)

        for i, c in enumerate(contours, start=1):
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            # Find mean confidence within this change region
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 1, -1)
            score = float(prob[c_mask == 1].mean()) if np.any(c_mask == 1) else float(conf)

            box = [float(x), float(y), float(x + cw), float(y + ch)]
            detections.append(Detection("changed area", score, box))

            draw.rectangle((x, y, x + cw, y + ch), outline=(255, 69, 0), width=line_w)
            draw.text((x, max(0, y - 16)), f"{score * 100:.0f}%", fill=(255, 69, 0))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        counts = {"change regions": len(detections), "changed pixels": int(binary_mask.sum())}

        # Diagnostics: binary mask image
        mask_vis = Image.fromarray((binary_mask * 255).astype(np.uint8))
        extras = [("Binary Change Mask", mask_vis)]

        return Prediction(
            image=output_image,
            detections=detections,
            device=str(self.device),
            notes=["Architecture: Siam-UNet (FC-Siam-conc) with dual feature encoders."],
            max_score=max_score,
            counts=counts,
            extra_images=extras,
        )
