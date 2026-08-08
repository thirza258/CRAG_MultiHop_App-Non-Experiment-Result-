"""Tests for HybridRAG merge/dedup helpers and reranker fallback."""

import unittest
from unittest import mock

try:
    from hybrid_rag.hybrid_rag import HybridRAG
    IMPORT_ERROR = ""
except Exception as exc:  # needs torch/transformers + Django
    HybridRAG = None
    IMPORT_ERROR = str(exc)


def _bare_instance():
    """HybridRAG without running __init__ (no model loading)."""
    return object.__new__(HybridRAG)


@unittest.skipIf(HybridRAG is None, f"hybrid_rag unavailable: {IMPORT_ERROR}")
class StaticHelperTests(unittest.TestCase):
    def test_extract_text_only(self):
        chunk = "Title: A\nAuthor: B\n\nText: the actual content"
        self.assertEqual(HybridRAG._extract_text_only(chunk), "the actual content")
        self.assertEqual(HybridRAG._extract_text_only("plain chunk"), "plain chunk")

    def test_normalize_meta_clears_nan_like_values(self):
        meta = {
            "a": None,
            "b": float("nan"),
            "c": "NaN",
            "d": "keep-me",
            "e": 3,
        }
        normalized = HybridRAG._normalize_meta(meta)
        self.assertEqual(normalized["a"], "")
        self.assertEqual(normalized["b"], "")
        self.assertEqual(normalized["c"], "")
        self.assertEqual(normalized["d"], "keep-me")
        self.assertEqual(normalized["e"], 3)

    def test_meta_to_key_keeps_url_chunks_separate(self):
        meta = {"url": "https://example.com/a", "chunk_index": 0}
        other = {"url": "https://example.com/a", "chunk_index": 1}
        self.assertNotEqual(
            HybridRAG._meta_to_key(meta), HybridRAG._meta_to_key(other)
        )
        # Same chunk (trailing slash / case only) must collapse
        same = {"url": "https://EXAMPLE.com/a/", "chunk_index": 0}
        self.assertEqual(
            HybridRAG._meta_to_key(meta), HybridRAG._meta_to_key(same)
        )

    def test_meta_to_key_keeps_document_chunks_separate(self):
        a = {"document_id": 7, "chunk_index": 0}
        b = {"document_id": 7, "chunk_index": 1}
        self.assertNotEqual(HybridRAG._meta_to_key(a), HybridRAG._meta_to_key(b))

    def test_meta_to_key_falls_back_to_content_hash(self):
        self.assertEqual(
            HybridRAG._meta_to_key({}, "same text"),
            HybridRAG._meta_to_key({}, "same text"),
        )
        self.assertNotEqual(
            HybridRAG._meta_to_key({}, "text one"),
            HybridRAG._meta_to_key({}, "text two"),
        )

    def test_deduplicate_new_chunks_drops_duplicates(self):
        hr = _bare_instance()
        chunks = ["Text: chunk A", "Text: chunk A duplicate", "Text: chunk B"]
        metas = [
            {"document_id": 1, "chunk_index": 0},
            {"document_id": 1, "chunk_index": 0},  # same identity as first
            {"document_id": 1, "chunk_index": 1},
        ]

        new_chunks, new_metas = hr._deduplicate_new_chunks(chunks, metas, set())

        self.assertEqual(len(new_chunks), 2)
        self.assertEqual(len(new_metas), 2)
        self.assertIn("chunk A", new_chunks[0])
        self.assertIn("chunk B", new_chunks[1])


@unittest.skipIf(HybridRAG is None, f"hybrid_rag unavailable: {IMPORT_ERROR}")
class RerankFallbackTests(unittest.TestCase):
    def test_rerank_success_returns_indices(self):
        hr = _bare_instance()
        hr._model = mock.Mock()
        hr._model.rerank.return_value = [{"index": 2}, {"index": 0}, {"index": 1}]

        indices, status = hr._rerank("query", ["a", "b", "c"])

        self.assertEqual(indices, [2, 0, 1])
        self.assertEqual(status, "ok")

    def test_rerank_failure_falls_back_to_original_order_indices(self):
        # A broken reranker must degrade to original order — returning
        # anything other than indices crashes the pipeline downstream.
        hr = _bare_instance()
        hr._model = mock.Mock()
        hr._model.rerank.side_effect = RuntimeError("model exploded")

        indices, status = hr._rerank("query", ["a", "b", "c"])

        self.assertEqual(indices, [0, 1, 2])
        self.assertIn("ERROR", status)

    def test_rerank_empty_candidates(self):
        hr = _bare_instance()

        indices, status = hr._rerank("query", [])

        self.assertEqual(indices, [])
        self.assertEqual(status, "ok")


if __name__ == "__main__":
    unittest.main()
