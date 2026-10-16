import zipfile
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


class CheckpointMissing(Exception):
    """Checkpoint file/folder not found or not usable."""


class NotConfigured(Exception):
    """Model exists in the registry but cannot run yet (missing package or setup)."""


@dataclass
class Detection:
    label: str
    confidence: float
    box: list  # [x1, y1, x2, y2] in original-image pixels


@dataclass
class Prediction:
    image: object            # PIL.Image with boxes and labels drawn
    detections: list
    device: str = "cpu"
    notes: list = field(default_factory=list)
    max_score: float = None   # best candidate score seen, even below the threshold
    counts: dict = None       # optional custom counts (change detection)
    extra_images: list = field(default_factory=list)   # [(title, PIL image)] diagnostic views


# ---------- checkpoint files ----------

def _repack_torch_archive(folder: Path, out_name: str) -> Path:
    """A .pt file is a zip. If it was unzipped into a folder (data.pkl, data/, version...),
    zip it back (uncompressed, as PyTorch requires) into backend/.cache and return that file."""
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / out_name
    if out.is_file():
        return out
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, f"{folder.name}/{p.relative_to(folder).as_posix()}")
    tmp.replace(out)
    return out


def resolve_checkpoint(path: Path) -> Path:
    """Return a real weight file. Handles a '.pt' name that is actually a folder."""
    if path.is_file():
        return path
    if path.is_dir():
        if (path / "data.pkl").is_file():        # this folder itself is an unzipped torch.save archive
            return _repack_torch_archive(path, path.name if path.name.endswith(".pt") else path.name + ".pt")
        for candidate in (path / "weights" / "best.pt", path / "best.pt"):
            if candidate.is_file():
                return candidate
        for sub in sorted(p for p in path.iterdir() if p.is_dir()):
            if (sub / "data.pkl").is_file():          # unzipped torch.save archive
                return _repack_torch_archive(sub, path.name)
        found = sorted(path.rglob("*.pt"))
        if len(found) == 1:
            return found[0]
        listing = ", ".join(str(p.relative_to(path)) for p in sorted(path.rglob("*"))[:12]) or "empty"
        raise CheckpointMissing(
            f"'{path.name}' is a folder, not a weight file, and no usable weights were found inside "
            f"(contents: {listing})."
        )
    raise CheckpointMissing(f"Checkpoint not found: {path}")


# ---------- shared torch helpers (torch is imported lazily) ----------

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:                                   # your own files, so the full loader is acceptable
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(ck):
    if isinstance(ck, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ck and isinstance(ck[key], dict):
                return ck[key]
        if ck and all(hasattr(v, "shape") for v in list(ck.values())[:20]):
            return ck
    raise RuntimeError("Could not find a state dict inside this checkpoint.")


def load_state_dict_checked(model, sd, hint=""):
    """Strict in substance: any real key mismatch is an error. BatchNorm 'num_batches_tracked'
    buffers are ignored (frozen-BN checkpoints do not store them)."""
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the rebuilt architecture. "
            f"Missing keys: {missing[:4]}{'...' if len(missing) > 4 else ''}. "
            f"Unexpected keys: {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}. {hint}"
        )


def effdet_image_size(num_anchors: int, per_location: int = 9):
    """EfficientDet stores its anchor boxes in the checkpoint; their count reveals the input size."""
    for size in range(128, 2049, 128):
        if per_location * sum((size // 2 ** lvl) ** 2 for lvl in range(3, 8)) == num_anchors:
            return size
    return None


# ---------- drawing ----------

def draw_detections(image, detections):
    from PIL import ImageDraw, ImageFont
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(out.size) / 400))
    size = max(12, round(min(out.size) / 45))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    for d in detections:
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=width)
        text = f"{d.label} {d.confidence * 100:.0f}%"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=(0, 0, 0))
        draw.text((x1 + 3, ty), text, fill=(255, 255, 255), font=font)
    return out
