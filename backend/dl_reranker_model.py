"""Download and cache the HuggingFace models used by the app.

Safe to run repeatedly:
- A model is only skipped when its `.download_complete` marker exists,
  so an interrupted download is retried (and resumed) on the next run
  instead of being mistaken for a finished one.
- Each model gets several attempts with exponential backoff, and
  `snapshot_download` resumes partially downloaded files.

Runs standalone (no Django needed): `pip install huggingface-hub PyYAML` then
`python dl_reranker_model.py`.
"""

from pathlib import Path
import os
import sys
import time

from huggingface_hub import snapshot_download

MODELS_DIR = Path(os.getenv("MODEL_CACHE_DIR", Path.cwd() / "models"))

from rag.config import load_pipeline_config

_pipeline_config = load_pipeline_config()
HYBRID_RERANKER_LIST = [
    _pipeline_config["hybrid_config"]["reranker_model"],
    _pipeline_config["crag_config"]["evaluator_model"],
]

MAX_ATTEMPTS = int(os.getenv("MODEL_DOWNLOAD_MAX_ATTEMPTS", "5"))
BACKOFF_BASE_SECONDS = 5

COMPLETE_MARKER = ".download_complete"


def _is_complete(local_dir: Path) -> bool:
    return (local_dir / COMPLETE_MARKER).exists()


def _mark_complete(local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / COMPLETE_MARKER).touch()


def download_model(model_id: str) -> bool:
    local_dir = MODELS_DIR / model_id.replace("/", "--")

    if _is_complete(local_dir):
        print(f"[skip] Already downloaded: {model_id}")
        return True

    if local_dir.exists():
        print(f"[resume] Found incomplete download for {model_id}, resuming...")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[download] {model_id} -> {local_dir} (attempt {attempt}/{MAX_ATTEMPTS})")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                # Inference uses PyTorch. Exported ONNX/OpenVINO/TensorFlow
                # copies can multiply download size without being used.
                ignore_patterns=["onnx/**", "openvino/**", "*.onnx", "*.onnx_data",
                                 "tf_model.h5", "flax_model.msgpack", "rust_model.ot"],
            )
            _mark_complete(local_dir)
            print(f"[done] {model_id}")
            return True
        except Exception as exc:
            print(f"[error] {model_id} attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                print(f"[retry] Waiting {wait}s before retrying...")
                time.sleep(wait)

    print(f"[failed] Could not download {model_id} after {MAX_ATTEMPTS} attempts.")
    return False


def download_models() -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ok = True
    for model_id in HYBRID_RERANKER_LIST:
        ok = download_model(model_id) and ok

    if ok:
        print(f"\nAll models ready in: {MODELS_DIR}")
    else:
        print(
            "\nSome models failed to download. Re-run this script to resume; "
            "already-finished files are not downloaded again."
        )
    return ok


if __name__ == "__main__":
    sys.exit(0 if download_models() else 1)
