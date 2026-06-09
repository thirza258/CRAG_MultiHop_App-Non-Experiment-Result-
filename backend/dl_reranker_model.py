from pathlib import Path
import os
from huggingface_hub import snapshot_download

MODELS_DIR = Path(os.getenv("MODEL_CACHE_DIR", Path.cwd() / "models"))

HYBRID_RERANKER_LIST = [
    "jinaai/jina-reranker-v3",
    "intfloat/multilingual-e5-small"
]


def download_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_id in HYBRID_RERANKER_LIST:
        local_dir = MODELS_DIR / model_id.replace("/", "--")

        # Skip if already downloaded
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"Already downloaded: {model_id}")
            continue

        print(f"Downloading: {model_id}")
        print(f"Saving to: {local_dir}")

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

        print(f"Done: {model_id}")

    print(f"\nAll models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    download_models()