import logging
from pathlib import Path
import hashlib
from router.models import Document, GuestUser

logger = logging.getLogger(__name__)

MODELS_DIR = Path.cwd() / "models"

def _local_model_path(model_name: str) -> str:
    local = MODELS_DIR / model_name.replace("/", "--")
    if local.exists() and any(local.iterdir()):
        logger.info(f"Using local model: {local}")
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

def check_existing_document(self, user: GuestUser, file_hash: str) -> Document | None:
    return Document.objects.filter(
        user=user,
        file_hash=file_hash,
    ).exclude(status="failed").first()