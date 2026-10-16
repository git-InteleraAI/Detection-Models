from .base import CheckpointMissing, NotConfigured, resolve_checkpoint

INSTALL_HINTS = {"effdet": "pip install effdet timm", "transformers": "pip install -U transformers timm",
                 "timm": "pip install timm", "ultralytics": "pip install ultralytics",
                 "torchvision": "pip install torchvision", "cv2": "pip install opencv-python"}


def _ultralytics(family):
    def make(ckpt, spec):
        from .ultralytics_runner import UltralyticsRunner
        return UltralyticsRunner(ckpt, family, spec)
    return make


def _torchvision(family):
    def make(ckpt, spec):
        from .torchvision_runner import TorchvisionRunner
        return TorchvisionRunner(ckpt, family, spec)
    return make


def _effdet(ckpt, spec):
    from .effdet_runner import EffDetRunner
    return EffDetRunner(ckpt, spec)


def _detr(ckpt, spec):
    from .detr_runner import DetrRunner
    return DetrRunner(ckpt, spec)


def _rfdetr_change(ckpt, spec):
    from .rfdetr_change_runner import RfDetrChangeRunner
    return RfDetrChangeRunner(spec)


def _siam_unet(ckpt, spec):
    from .siam_unet_runner import SiamUnetRunner
    return SiamUnetRunner(ckpt, spec)


RUNNERS = {
    "ultralytics_yolo": _ultralytics("yolo"),
    "ultralytics_rtdetr": _ultralytics("rtdetr"),
    "fasterrcnn": _torchvision("fasterrcnn"),
    "retinanet": _torchvision("retinanet"),
    "ssd": _torchvision("ssd"),
    "efficientdet": _effdet,
    "detr": _detr,
    "rfdetr_change": _rfdetr_change,      # change detection: runner.predict_pair(before, after, conf)
    "siam_unet": _siam_unet,              # LEVIR-CD trained model runner
}
_CACHE = {}


def is_ready(spec) -> bool:
    return spec.kind in RUNNERS


def get_runner(key, spec, ckpt_path):
    if not is_ready(spec):
        raise NotConfigured(f"{spec.label}: {spec.note or 'no inference runner yet.'}")
    if key not in _CACHE:
        checkpoint = resolve_checkpoint(ckpt_path) if ckpt_path is not None else None
        try:
            _CACHE[key] = RUNNERS[spec.kind](checkpoint, spec)
        except ImportError as e:
            hint = INSTALL_HINTS.get((e.name or "").split(".")[0], "pip install -r requirements.txt")
            raise NotConfigured(f"{spec.label} needs the Python package '{e.name}'. Install it with: {hint}")
    return _CACHE[key]
