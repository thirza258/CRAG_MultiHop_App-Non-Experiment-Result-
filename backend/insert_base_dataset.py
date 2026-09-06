"""Prepare the base MultiHop-RAG dataset for the app.

Run automatically by the container entrypoint (safe to re-run):

1. Waits until ChromaDB is reachable (with retries).
2. Ensures the base collection exists so the app never crashes on a
   missing collection.
3. Ensures the corpus file is cached locally (downloads it from
   HuggingFace with retries and an atomic write if it is missing).
4. Optionally embeds + indexes the corpus into ChromaDB when
   INSERT_BASE_DATASET=true and an OPENROUTER_API_KEY is set.
   Indexing is skipped when the collection is already populated.
"""

import os
import sys
import time
from pathlib import Path

import django

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "ragreader.settings"
)
from dotenv import load_dotenv

load_dotenv()

django.setup()

import chromadb
import pandas as pd

CHROMA_COLLECTION_NAME = "ragreader_collection"

CORPUS_DIR = Path("corpus")
CORPUS_FILE = CORPUS_DIR / "corpus.json"
CORPUS_URL = "hf://datasets/yixuantt/MultiHopRAG/corpus.json"
CORPUS_REQUIRED_COLUMNS = {"title", "author", "source", "url", "category", "published_at", "body"}

CHROMA_CONNECT_ATTEMPTS = int(os.getenv("CHROMA_CONNECT_ATTEMPTS", "30"))
CORPUS_DOWNLOAD_ATTEMPTS = int(os.getenv("CORPUS_DOWNLOAD_ATTEMPTS", "3"))


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def connect_chroma() -> chromadb.HttpClient:
    """Connect to ChromaDB, retrying while the service starts up."""
    host = os.getenv("CHROMA_HOST", "localhost")
    port = os.getenv("CHROMA_PORT", "8000")

    last_error = None
    for attempt in range(1, CHROMA_CONNECT_ATTEMPTS + 1):
        try:
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            print(f"Connected to ChromaDB at {host}:{port}")
            return client
        except Exception as exc:
            last_error = exc
            print(
                f"Waiting for ChromaDB at {host}:{port}... "
                f"(attempt {attempt}/{CHROMA_CONNECT_ATTEMPTS})"
            )
            time.sleep(2)

    raise RuntimeError(f"Could not reach ChromaDB at {host}:{port}: {last_error}")


def _validate_corpus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Corpus is empty")
    missing = CORPUS_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")
    return df


def ensure_corpus_exists() -> pd.DataFrame:
    """Return the corpus, downloading and caching it locally if needed.

    The download is written to a temp file and renamed into place so an
    interrupted download can never leave a truncated corpus.json behind.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if CORPUS_FILE.exists():
        try:
            df = _validate_corpus(pd.read_json(CORPUS_FILE))
            print(f"Using cached corpus: {CORPUS_FILE} ({len(df)} articles)")
            return df
        except Exception as exc:
            print(f"Cached corpus is invalid ({exc}) — re-downloading...")
            CORPUS_FILE.unlink(missing_ok=True)

    last_error = None
    for attempt in range(1, CORPUS_DOWNLOAD_ATTEMPTS + 1):
        try:
            print(f"Downloading corpus dataset... (attempt {attempt}/{CORPUS_DOWNLOAD_ATTEMPTS})")
            df = _validate_corpus(pd.read_json(CORPUS_URL))

            tmp_file = CORPUS_FILE.with_suffix(".json.tmp")
            df.to_json(tmp_file, orient="records")
            os.replace(tmp_file, CORPUS_FILE)

            print(f"Saved corpus to: {CORPUS_FILE} ({len(df)} articles)")
            return df
        except Exception as exc:
            last_error = exc
            print(f"Corpus download failed: {exc}")
            if attempt < CORPUS_DOWNLOAD_ATTEMPTS:
                time.sleep(5 * attempt)

    raise RuntimeError(f"Could not download corpus after {CORPUS_DOWNLOAD_ATTEMPTS} attempts: {last_error}")


#: The model the base corpus is embedded with. The app pins query-time dense
#: retrieval to whatever a collection was indexed with, so changing this means
#: re-indexing the corpus from scratch — a query embedded by a different model
#: is a comparison across vector spaces.
BASE_EMBEDDING_MODEL = "google/gemini-embedding-2-preview"


def _record_collection(embedding_model: str, chunk_count: int) -> None:
    """Write the bookkeeping row for the corpus we just built.

    Without this the pipeline has to guess which model produced the vectors it
    is searching, and guessing is how a query ends up embedded into the wrong
    vector space. Best-effort: the corpus itself is already indexed by the time
    this runs, so a bookkeeping failure must not undo that.
    """
    try:
        from router.models import ChromaCollection

        record, created = ChromaCollection.objects.update_or_create(
            collection_name=CHROMA_COLLECTION_NAME,
            defaults={
                "collection_type": "corpus",
                "embedding_model": embedding_model,
                "chunk_count": chunk_count,
            },
        )
        print(
            f"{'Registered' if created else 'Updated'} collection record for "
            f"'{CHROMA_COLLECTION_NAME}' (embedding_model={embedding_model})"
        )
    except Exception as exc:
        print(
            f"WARNING: could not record the collection's embedding model "
            f"({exc}). Dense retrieval will fall back to the pipeline's "
            f"configured default."
        )


def index_corpus(collection, df: pd.DataFrame) -> None:
    """Embed the corpus via OpenRouter and insert it into ChromaDB."""
    from chroma.chroma_settings import insert_chunk_to_chromadb
    from common.dataset_settings import prepare_corpus
    from dense_rag.dense_rag import DenseRAG

    config = {
        "embedding_model": BASE_EMBEDDING_MODEL,
        "llm_model": "google/gemini-3-flash-preview",
        "top_k": 5,
        "collection_name": CHROMA_COLLECTION_NAME,
    }

    dense_embeddings = DenseRAG(config)

    all_documents, all_metadatas = prepare_corpus(df)
    print(f"Prepared {len(all_documents)} chunks. Generating embeddings...")

    embeddings = dense_embeddings._get_embeddings(all_documents, 50)

    if len(embeddings) != len(all_documents):
        # Never insert misaligned embeddings — a failed batch would silently
        # pair every following chunk with the wrong vector.
        raise RuntimeError(
            f"Embedding count mismatch: {len(embeddings)} embeddings for "
            f"{len(all_documents)} chunks. Aborting insert — re-run to retry."
        )

    print("Inserting into ChromaDB...")
    success = insert_chunk_to_chromadb(
        CHROMA_COLLECTION_NAME,
        all_documents,
        all_metadatas,
        embeddings,
    )

    if not success:
        raise RuntimeError("Some batches failed to insert. Re-run to retry.")

    _record_collection(BASE_EMBEDDING_MODEL, len(all_documents))

    print("Base dataset indexing completed successfully.")


def main():
    chroma_client = connect_chroma()

    # Always make sure the base collection exists — the app queries it at
    # startup and crashes if it is missing.
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=None,
    )

    current_count = collection.count()
    print(f"Existing documents in '{CHROMA_COLLECTION_NAME}': {current_count}")

    # Cache the corpus file even when indexing is disabled, so a later
    # opt-in run doesn't depend on the network.
    df = ensure_corpus_exists()

    if not _truthy(os.getenv("INSERT_BASE_DATASET", "false")):
        print(
            "INSERT_BASE_DATASET is not enabled — skipping corpus indexing. "
            "Set INSERT_BASE_DATASET=true in backend/.env to index the base "
            "dataset (requires OPENROUTER_API_KEY, uses embedding API credits)."
        )
        return

    if current_count > 0:
        print("Collection already populated. Skipping indexing.")
        return

    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "WARNING: INSERT_BASE_DATASET=true but OPENROUTER_API_KEY is not "
            "set — cannot generate embeddings. Skipping corpus indexing."
        )
        return

    index_corpus(collection, df)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"insert_base_dataset failed: {exc}")
        sys.exit(1)
