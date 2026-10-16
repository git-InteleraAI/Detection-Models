"""Write every tensor name and shape of one checkpoint to a text file.

  python tools/dump_keys.py "C:\\LEVIR_CD_Trained_Model"

Works on a normal file or a folder that is an unzipped PyTorch archive. Creates <name>_keys.txt in the
project folder. Only use it on checkpoints you trained yourself.
"""
import sys
from pathlib import Path

import torch
from inspect_checkpoints import ROOT, repack        # same folder, reuses the archive repacker

src = Path(sys.argv[1])
path = repack(src) if src.is_dir() and (src / "data.pkl").is_file() else src
ck = torch.load(path, map_location="cpu", weights_only=False)
if hasattr(ck, "state_dict") and not isinstance(ck, dict):
    sd = ck.state_dict()
elif isinstance(ck, dict) and isinstance(ck.get("model_state_dict"), dict):
    sd = ck["model_state_dict"]
elif isinstance(ck, dict) and isinstance(ck.get("state_dict"), dict):
    sd = ck["state_dict"]
else:
    sd = ck
lines = [f"{k}  {tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}" for k, v in sd.items()]
out = ROOT / f"{src.name}_keys.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"{len(lines)} entries written to {out}")
