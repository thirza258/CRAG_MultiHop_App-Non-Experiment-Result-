"""Deployment defaults, overridden per request by the chat settings panel."""
from copy import deepcopy
import os
from pathlib import Path
import yaml

DEFAULT_CONFIG = {'llm_model': 'qwen/qwen3-30b-a3b-instruct-2507',
 'collection_name': 'ragreader_collection',
 'dense_config': {'collection_name': 'ragreader_collection',
                  'embedding_model': 'google/gemini-embedding-2-preview',
                  'top_k': 4},
 'sparse_config': {'collection_name': 'ragreader_collection',
                   'top_k': 4,
                   'remove_stop_words': True},
 'hybrid_config': {'rerank_only': True,
                   'reranker_model': 'jinaai/jina-reranker-v3',
                   'top_k': 4,
                   'retrieval_top_k': 4},
 'crag_config': {'llm_model': 'mistralai/mistral-nemo',
                 'embedding_model': 'google/gemini-embedding-2-preview',
                 'upper_threshold': 0.91,
                 'lower_threshold': 0.87,
                 'strip_threshold': 0.88,
                 'top_k': 4,
                 'external_chunk_size': 1000,
                 'external_chunk_overlap': 200,
                 'evaluator_model': 'intfloat/multilingual-e5-small'},
 'multi_hop_config': {'llm_model': 'mistralai/mistral-nemo',
                      'max_hops': 3,
                      'top_k': 4},
 'evaluation_llm_model': 'google/gemma-4-26b-a4b-it',
 'evaluation_embedding_model': 'openai/text-embedding-3-small',
 'chunking': {'strategy': 'recursive', 'chunk_size': 500, 'overlap': 50}}


def load_pipeline_config():
    path = Path(os.environ.get("RAG_CONFIG_FILE", Path(__file__).resolve().parents[1] / "config.yml"))
    if os.environ.get("RAG_CONFIG_FILE") and not path.is_file():
        raise ValueError(f"RAG_CONFIG_FILE does not exist: {path}")
    config = deepcopy(DEFAULT_CONFIG)
    overrides = yaml.safe_load(path.read_text()) if path.exists() else {}
    if not isinstance(overrides, dict):
        raise ValueError("RAG_CONFIG_FILE must contain a YAML mapping.")
    for key, value in overrides.items():
        if key not in config:
            raise ValueError(f"Unknown pipeline configuration key: {key}")
        if isinstance(config[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a YAML mapping.")
            config[key].update(value)
        else:
            config[key] = value
    return config
