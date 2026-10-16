"""Print what is inside each trained checkpoint, so inference code is written from facts.

  python tools/inspect_checkpoints.py                       every checkpoint listed in models/checkpoints.json
  python tools/inspect_checkpoints.py "C:\\some\\file_or_folder"   also inspect extra paths

Handles a normal .pt/.pth file AND a folder that is an unzipped PyTorch archive (data.pkl, data/, version).
Writes checkpoint_report.txt; send that file back. Only run it on checkpoints you trained yourself
(torch.load can execute pickled code).
"""
import collections
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "models" / "checkpoints.json").read_text(encoding="utf-8"))
REPORT = io.StringIO()
HEADS = ("cls_score", "bbox_pred", "classification_head", "class_embed", "class_net", "box_net",
         "box_predictor", "bbox_embed", "cls_logits", "score", "classifier", "seg_head", "head")
META = ("name", "class", "names", "num_class", "epoch", "args", "config", "cfg", "imgsz", "label", "version",
        "size", "mean", "std", "arch", "backbone", "variant", "encoder", "in_channels", "threshold")


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    REPORT.write(line + "\n")


def shape(v):
    return tuple(v.shape) if hasattr(v, "shape") else type(v).__name__


def describe_state_dict(sd, indent="    "):
    keys = list(sd.keys())
    log(f"{indent}tensors: {len(keys)}")
    prefixes = collections.Counter(".".join(str(k).split(".")[:2]) for k in keys)
    log(f"{indent}key prefixes:", dict(prefixes.most_common(14)))
    for k in keys[:3]:
        log(f"{indent}first key:", k, shape(sd[k]))        # the first conv's input channels show how images are fed
    for k in keys[-4:]:
        log(f"{indent}last key:", k, shape(sd[k]))         # output layers
    heads = [k for k in keys if any(h in str(k) for h in HEADS)]
    for k in heads[-6:]:
        log(f"{indent}head layer:", k, shape(sd[k]))


def is_state_dict(d):
    return isinstance(d, dict) and d and all(hasattr(v, "shape") for v in list(d.values())[:20])


def repack(folder: Path) -> Path:
    """Zip an unzipped torch archive back into a loadable file (uncompressed, as PyTorch requires)."""
    out = Path(tempfile.mkdtemp()) / (folder.name + ".pt")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, f"{folder.name}/{p.relative_to(folder).as_posix()}")
    return out


def inspect_file(path: Path, shown_as: Path = None):
    shown_as = shown_as or path
    log(f"\n=== {shown_as}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # report and move on
        log("  torch.load FAILED:", type(e).__name__, str(e)[:600])
        return
    log("  top-level type:", type(ck).__name__)
    if hasattr(ck, "state_dict") and not isinstance(ck, dict):
        log("  whole model object:", type(ck).__module__ + "." + type(ck).__name__)
        if getattr(ck, "names", None):
            log("  names:", ck.names)
        describe_state_dict(ck.state_dict())
        return
    if is_state_dict(ck):
        describe_state_dict(ck)
        return
    if not isinstance(ck, dict):
        return
    log("  dict keys:", list(ck.keys())[:30])
    for k, v in ck.items():
        if hasattr(v, "state_dict"):
            log(f"  [{k}] model object:", type(v).__module__ + "." + type(v).__name__)
            if getattr(v, "names", None):
                log("    names:", v.names)
            describe_state_dict(v.state_dict())
        elif is_state_dict(v):
            log(f"  [{k}] state dict")
            describe_state_dict(v)
        elif any(m in str(k).lower() for m in META):
            log(f"  {k} =", repr(v)[:500])


def inspect_path(p: Path):
    if not p.exists():
        log(f"\n=== {p}  NOT FOUND")
        return
    if p.is_file():
        inspect_file(p)
        return
    if (p / "data.pkl").is_file():                       # the folder itself is an unzipped archive
        log(f"\n(note: {p} is an unzipped PyTorch archive; repacking a temporary copy to read it)")
        inspect_file(repack(p), p)
        return
    for sub in sorted(x for x in p.iterdir() if x.is_dir()):
        if (sub / "data.pkl").is_file():
            log(f"\n(note: {sub} is an unzipped PyTorch archive; repacking a temporary copy to read it)")
            inspect_file(repack(sub), sub)
            return
    inner = sorted(p.rglob("*"))
    log(f"\n=== {p} is a FOLDER, contents (first 40):")
    for q in inner[:40]:
        log("   ", q.relative_to(p), "" if q.is_dir() else f"({q.stat().st_size} bytes)")
    for q in inner:
        if q.suffix in (".pt", ".pth") and q.is_file():
            inspect_file(q)


def main():
    for ds_id, files in CFG["files"].items():
        ds = CFG["datasets"][ds_id]
        folder = Path(os.environ.get(ds["env"], ds["dir"]))
        log(f"\n##### {ds['label']}  ({ds.get('task', 'ship')})")
        for model_id, name in files.items():
            inspect_path(folder / name)
    for extra in sys.argv[1:]:
        log(f"\n##### extra path: {extra}")
        inspect_path(Path(extra))
    (ROOT / "checkpoint_report.txt").write_text(REPORT.getvalue(), encoding="utf-8")
    print("\nSaved checkpoint_report.txt")


if __name__ == "__main__":
    main()
