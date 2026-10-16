"""Secure, on-demand checkpoint downloader for private Hugging Face repositories.

Authenticates via the HF_TOKEN environment variable and downloads only the requested
model checkpoint into a persistent cache.
"""
import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("hf_downloader")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "IntelEra/model-checkpoints"
CACHE_DIR = Path(os.environ.get("CHECKPOINT_CACHE_DIR", ROOT / ".cache" / "checkpoints"))


class CheckpointDownloadError(Exception):
    """Raised when a checkpoint fails to download, authenticate, or extract."""
    pass


def get_hf_token() -> str:
    return (os.environ.get("HF_TOKEN") or "").strip()


def get_repo_id() -> str:
    return (os.environ.get("HF_REPO_ID") or DEFAULT_REPO_ID).strip()


def is_hf_configured() -> bool:
    return bool(get_hf_token())


def get_cached_path(rel_path: str) -> Path:
    """Return the expected local path in CACHE_DIR for a given HF relative path."""
    return CACHE_DIR / rel_path


def is_cached(dataset_id: str, model_id: str, rel_path: str) -> bool:
    """Check if the given checkpoint is already present and valid in CACHE_DIR."""
    if rel_path.endswith(".zip"):
        extract_dir = CACHE_DIR / "extracted" / f"{dataset_id}_{model_id}"
        if extract_dir.is_dir():
            if (extract_dir / "data.pkl").is_file():
                return True
            for sub in extract_dir.rglob("data.pkl"):
                if sub.is_file():
                    return True
            if any(extract_dir.rglob("*.pt")) or any(extract_dir.rglob("*.pth")):
                return True
        return False

    target = CACHE_DIR / rel_path
    if target.is_file():
        return target.stat().st_size > 0
    if target.is_dir():
        if (target / "data.pkl").is_file():
            return True
        for sub in target.iterdir():
            if sub.is_dir() and (sub / "data.pkl").is_file():
                return True
        if any(target.rglob("*.pt")) or any(target.rglob("*.pth")):
            return True
    return False


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extract a ZIP archive with path traversal protection (prevent Zip Slip)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_target = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            member_target = (target_dir / member.filename).resolve()
            if not member_target.is_relative_to(resolved_target):
                raise CheckpointDownloadError(
                    f"Security violation: path traversal detected in ZIP archive ({member.filename})"
                )
            z.extract(member, target_dir)
    return target_dir


def download_checkpoint(dataset_id: str, model_id: str, rel_path: str) -> Path:
    """Download a single model checkpoint or folder from the private Hugging Face repository.

    Token is read securely from HF_TOKEN and never logged.
    """
    token = get_hf_token()
    if not token:
        raise CheckpointDownloadError(
            "Authentication required: HF_TOKEN environment variable is not set. "
            "Please configure your Hugging Face access token in Render's environment settings."
        )

    repo_id = get_repo_id()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ZIP Archive (e.g. models/levir_cd/LEVIR_CD_Trained_Model.zip)
    if rel_path.endswith(".zip"):
        extract_dir = CACHE_DIR / "extracted" / f"{dataset_id}_{model_id}"
        if is_cached(dataset_id, model_id, rel_path):
            return extract_dir

        try:
            from huggingface_hub import hf_hub_download
            log.info("Downloading ZIP checkpoint from %s: %s", repo_id, rel_path)
            zip_file = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                token=token,
                local_dir=str(CACHE_DIR),
            )
            extracted = _safe_extract_zip(Path(zip_file), extract_dir)
            return extracted
        except Exception as e:
            _handle_hf_error(e, repo_id, rel_path)

    # 2. Directory checkpoint (e.g. models/masati/yolo11/best)
    target_path = CACHE_DIR / rel_path
    is_directory_target = not rel_path.endswith((".pt", ".pth", ".bin", ".onnx", ".safetensors"))

    if is_directory_target:
        if is_cached(dataset_id, model_id, rel_path):
            return target_path

        try:
            from huggingface_hub import snapshot_download
            log.info("Downloading directory checkpoint from %s: %s", repo_id, rel_path)
            patterns = [f"{rel_path.rstrip('/')}/*", f"{rel_path.rstrip('/')}/**/*"]
            snapshot_download(
                repo_id=repo_id,
                token=token,
                allow_patterns=patterns,
                local_dir=str(CACHE_DIR),
            )
            if not target_path.exists():
                raise CheckpointDownloadError(
                    f"Directory '{rel_path}' was not found in the downloaded snapshot from '{repo_id}'."
                )
            return target_path
        except Exception as e:
            _handle_hf_error(e, repo_id, rel_path)

    # 3. Regular file checkpoint (e.g. .pt, .pth)
    if is_cached(dataset_id, model_id, rel_path):
        return target_path

    try:
        from huggingface_hub import hf_hub_download
        log.info("Downloading checkpoint file from %s: %s", repo_id, rel_path)
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            token=token,
            local_dir=str(CACHE_DIR),
        )
        return Path(file_path)
    except Exception as e:
        _handle_hf_error(e, repo_id, rel_path)


def _handle_hf_error(e: Exception, repo_id: str, rel_path: str):
    """Format Hugging Face errors without exposing tokens or credentials."""
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError

    if isinstance(e, RepositoryNotFoundError):
        raise CheckpointDownloadError(
            f"Hugging Face repository '{repo_id}' was not found. "
            "Please verify the repository name and make sure your HF_TOKEN has read access."
        ) from None
    if isinstance(e, EntryNotFoundError):
        raise CheckpointDownloadError(
            f"Checkpoint '{rel_path}' does not exist in Hugging Face repository '{repo_id}'."
        ) from None
    if isinstance(e, HfHubHTTPError):
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise CheckpointDownloadError(
                f"Access denied (HTTP {status}) accessing '{repo_id}'. "
                "Verify that your HF_TOKEN is valid and has read permissions for this private repository."
            ) from None
        raise CheckpointDownloadError(
            f"Hugging Face API error (HTTP {status}) while downloading '{rel_path}' from '{repo_id}': {e}"
        ) from None
    raise CheckpointDownloadError(f"Failed to download '{rel_path}' from '{repo_id}': {type(e).__name__}: {e}") from None
