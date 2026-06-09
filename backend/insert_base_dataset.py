import os
from pathlib import Path

import django

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "ragreader.settings"
)
from dotenv import load_dotenv

django.setup()

import chromadb
import pandas as pd

from chroma.chroma_settings import insert_chunk_to_chromadb
from common.dataset_settings import prepare_corpus
from dense_rag.dense_rag import DenseRAG

CHROMA_COLLECTION_NAME = "ragreader_collection"

CORPUS_DIR = Path("corpus")
CORPUS_FILE = CORPUS_DIR / "corpus.json"

load_dotenv()

def ensure_corpus_exists() -> pd.DataFrame:
    """
    Download corpus only once and cache locally.
    """

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if CORPUS_FILE.exists():
        print(f"Using cached corpus: {CORPUS_FILE}")
        return pd.read_json(CORPUS_FILE)

    print("Downloading corpus dataset...")

    df = pd.read_json("hf://datasets/yixuantt/MultiHopRAG/corpus.json")

    df.to_json(CORPUS_FILE, orient="records")

    print(f"saved to: {CORPUS_FILE}")

    return df


def main():
    embedding_model = "google/gemini-embedding-2-preview"

    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=os.getenv("CHROMA_PORT", "8002"),
    )

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=None,
    )

    current_count = collection.count()

    print(f"Existing Chroma documents: {current_count}")

    # # Skip indexing if already populated
    # if current_count > 0:
    #     print("Collection already populated. Skipping indexing.")
    #     return

    # df = ensure_corpus_exists()

    # config = {
    #     "embedding_model": embedding_model,
    #     "llm_model": "google/gemini-3-flash-preview",
    #     "top_k": 5,
    #     "collection_name": CHROMA_COLLECTION_NAME,
    # }

    # dense_embeddings = DenseRAG(config)

    # all_documents, all_metadatas = prepare_corpus(df)

    # print("Generating embeddings...")

    # embeddings = dense_embeddings._get_embeddings(
    #     all_documents,
    #     50,
    # )

    # print("Inserting into ChromaDB...")

    # success = insert_chunk_to_chromadb(
    #     CHROMA_COLLECTION_NAME,
    #     all_documents,
    #     all_metadatas,
    #     embeddings,
    # )

    # if success:
    #     print("Indexing completed successfully")

if __name__ == "__main__":
    main()