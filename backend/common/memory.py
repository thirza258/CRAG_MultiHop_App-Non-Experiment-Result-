import logging
import os
from pathlib import Path
import hashlib
from router.models import Document, GuestUser

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODEL_CACHE_DIR", Path.cwd() / "models"))

def _local_model_path(model_name: str) -> str:
    local = MODELS_DIR / model_name.replace("/", "--")
    if (local / ".download_complete").exists():
        logger.info(f"Using local model: {local}")
        return str(local)
    # No completion marker: only trust the dir if it has real model files
    # (an interrupted snapshot_download leaves a hidden .cache dir behind).
    if local.exists() and any(p for p in local.iterdir() if not p.name.startswith(".")):
        logger.info(f"Using local model (no marker, non-empty dir): {local}")
        return str(local)
    logger.info(f"Local not found, falling back to HF hub: {model_name}")
    return model_name

def compute_file_hash(file) -> str:
    """SHA-256 of the uploaded file. Resets pointer after reading."""
    hasher = hashlib.sha256()
    for chunk in file.chunks():
        hasher.update(chunk)
    file.seek(0)  # reset so downstream readers aren't broken
    return hasher.hexdigest()


def compute_text_hash(text: str) -> str:
    """
    SHA-256 of a string source (pasted text, or a URL).

    Text and URL ingestion need the same per-user dedup key that uploads get from
    compute_file_hash, but they have no file object to read chunks from. Hashing
    the URL means resubmitting the same link is recognised as the same document;
    hashing pasted text means resubmitting the same passage is too.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def check_existing_document(self, user: GuestUser, file_hash: str) -> Document | None:
    return Document.objects.filter(
        user=user,
        file_hash=file_hash,
    ).exclude(status="failed").first()