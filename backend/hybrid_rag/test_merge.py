"""Tests for HybridRAG.retrieve_from_precomputed — the merge/dedup/rerank entry
point the pipeline actually calls.

Covers: merging both retriever sides, dropping duplicates while keeping
chunk/meta alignment, truncation to final_top_k, the reranked order following the
reranker's indices, the use_reranker=False opt-out, degradation when the reranker
model never loaded, and the empty/one-sided candidate sets.

Complements hybrid_rag/tests.py, which covers the static keying helpers and
_rerank's own return contract. The reranker is always faked here, so no model,
network, vector store or API key is required.
"""

import unittest
from unittest import mock

try:
    from hybrid_rag.hybrid_rag import HybridRAG
    IMPORT_ERROR = ""
except Exception as exc:  # needs torch/transformers + Django (common.memory -> router.models)
    HybridRAG = None
    IMPORT_ERROR = str(exc)


# ── Fake reranker bodies ──────────────────────────────────────────────────────
# _rerank_jina calls self._model.rerank(query, candidates, top_n=len(candidates))
# and reads r["index"] off every result, so that is the shape the fakes return.

def _identity_rerank(query, candidates, top_n=None):
    return [{"index": i} for i in range(len(candidates))]


def _reversed_rerank(query, candidates, top_n=None):
    return [{"index": i} for i in reversed(range(len(candidates)))]


def _empty_rerank(query, candidates, top_n=None):
    return []


# ── Construction helpers ──────────────────────────────────────────────────────
# rerank_only=True skips the DenseRAG/SparseRAG construction (Chroma client +
# OPENROUTER_API_KEY), and _load_reranker is patched out so no weights load.

def _hybrid(retrieval_top_k=5, rerank=_identity_rerank, **config):
    with mock.patch.object(HybridRAG, "_load_reranker"):
        hr = HybridRAG({"rerank_only": True, "retrieval_top_k": retrieval_top_k, **config})
    hr._model = mock.Mock()
    hr._model.rerank.side_effect = rerank
    return hr


def _hybrid_without_model_attribute(retrieval_top_k=5):
    """Simulate a partially constructed instance with no model/load state."""
    hr = HybridRAG({"rerank_only": True, "retrieval_top_k": retrieval_top_k})
    del hr._model
    del hr._reranker_state
    return hr


def _hybrid_with_failed_load(retrieval_top_k=5):
    """Run the lazy load failure path without downloading a snapshot."""
    with mock.patch.object(
        HybridRAG, "_load_reranker", side_effect=RuntimeError("no local snapshot")
    ):
        hr = HybridRAG({"rerank_only": True, "retrieval_top_k": retrieval_top_k})
        hr._ensure_reranker()
        return hr


# ── Fixtures ──────────────────────────────────────────────────────────────────
# Chunk bodies encode their own identity ("doc<document_id>-chunk<chunk_index>")
# so a misalignment between the returned docs and metas is detectable.

def _dense_side():
    return (
        ["doc1-chunk0", "doc1-chunk1", "doc1-chunk2"],
        [{"document_id": 1, "chunk_index": 0},
         {"document_id": 1, "chunk_index": 1},
         {"document_id": 1, "chunk_index": 2}],
    )


def _sparse_side():
    """First entry is a duplicate identity of doc1-chunk1 with a different body,
    so first-wins dedup is observable in the returned text."""
    return (
        ["sparse-copy-of-doc1-chunk1", "doc2-chunk0", "doc2-chunk1"],
        [{"document_id": 1, "chunk_index": 1},
         {"document_id": 2, "chunk_index": 0},
         {"document_id": 2, "chunk_index": 1}],
    )


_MERGE_ORDER = [
    "doc1-chunk0", "doc1-chunk1", "doc1-chunk2", "doc2-chunk0", "doc2-chunk1",
]


def _texts(docs):
    """Payload text of each enriched chunk, for exact-order assertions."""
    return [HybridRAG._extract_text_only(doc) for doc in docs]


def _assert_aligned(case, docs, metas):
    """Every returned doc must still sit next to its own metadata."""
    case.assertEqual(len(docs), len(metas))
    for doc, meta in zip(docs, metas):
        case.assertEqual(
            HybridRAG._extract_text_only(doc),
            f"doc{meta['document_id']}-chunk{meta['chunk_index']}",
        )


@unittest.skipIf(HybridRAG is None, f"hybrid_rag unavailable: {IMPORT_ERROR}")
class PrecomputedMergeTests(unittest.TestCase):
    def test_merges_both_sides_dedups_and_keeps_meta_alignment(self):
        hr = _hybrid(retrieval_top_k=10)
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", dense_chunks, sparse_chunks, dense_metas, sparse_metas
        )

        self.assertEqual(status, "ok")
        self.assertEqual(_texts(docs), _MERGE_ORDER)
        self.assertEqual(metas, dense_metas + sparse_metas[1:])
        _assert_aligned(self, docs, metas)
        # The dense copy of doc1-chunk1 wins; the sparse variant is gone entirely.
        for doc in docs:
            self.assertNotIn("sparse-copy", doc)

    def test_reranker_scores_the_deduplicated_candidates(self):
        hr = _hybrid(retrieval_top_k=10)
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        hr.retrieve_from_precomputed(
            "find me", dense_chunks, sparse_chunks, dense_metas, sparse_metas
        )

        self.assertEqual(hr._model.rerank.call_count, 1)
        args, kwargs = hr._model.rerank.call_args
        self.assertEqual(args[0], "find me")
        self.assertEqual(len(args[1]), 5)          # 6 in, 1 duplicate removed
        self.assertEqual(kwargs["top_n"], 5)

    def test_reranked_order_follows_returned_indices_then_truncates(self):
        # Truncation happens on the reranked indices, not on the merge order —
        # a regression that slices before reranking returns [a, b] instead.
        hr = _hybrid(retrieval_top_k=2, rerank=None)
        hr._model.rerank.return_value = [{"index": 2}, {"index": 0}, {"index": 1}]
        self.assertEqual(hr.final_top_k, 2)

        docs, metas, status = hr.retrieve_from_precomputed(
            "q",
            ["doc1-chunk0", "doc1-chunk1"],
            ["doc2-chunk0"],
            [{"document_id": 1, "chunk_index": 0}, {"document_id": 1, "chunk_index": 1}],
            [{"document_id": 2, "chunk_index": 0}],
        )

        self.assertEqual(status, "ok")
        self.assertEqual(_texts(docs), ["doc2-chunk0", "doc1-chunk0"])
        self.assertEqual(
            metas,
            [{"document_id": 2, "chunk_index": 0}, {"document_id": 1, "chunk_index": 0}],
        )

    def test_truncation_uses_retrieval_top_k_and_ignores_top_k(self):
        hr = _hybrid(retrieval_top_k=2, top_k=5)
        self.assertEqual(hr.final_top_k, 2)
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", dense_chunks, sparse_chunks, dense_metas, sparse_metas
        )

        self.assertEqual(status, "ok")
        self.assertEqual(_texts(docs), _MERGE_ORDER[:2])
        _assert_aligned(self, docs, metas)

    def test_fewer_candidates_than_final_top_k_returns_everything(self):
        hr = _hybrid(retrieval_top_k=5)

        docs, metas, status = hr.retrieve_from_precomputed(
            "q",
            ["doc1-chunk0"],
            ["doc2-chunk0"],
            [{"document_id": 1, "chunk_index": 0}],
            [{"document_id": 2, "chunk_index": 0}],
        )

        self.assertEqual(status, "ok")
        self.assertEqual(_texts(docs), ["doc1-chunk0", "doc2-chunk0"])
        _assert_aligned(self, docs, metas)

    def test_empty_candidate_set_returns_empty_lists_without_raising(self):
        hr = _hybrid()

        docs, metas, status = hr.retrieve_from_precomputed("q", [], [], None, None)

        self.assertEqual(docs, [])
        self.assertEqual(metas, [])
        self.assertEqual(status, "ERROR: no retrieved chunks in both dense and sparse")
        hr._model.rerank.assert_not_called()

    def test_empty_candidate_set_still_reports_error_with_reranker_disabled(self):
        hr = _hybrid()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", [], [], None, None, use_reranker=False
        )

        self.assertEqual(docs, [])
        self.assertEqual(metas, [])
        self.assertEqual(status, "ERROR: no retrieved chunks in both dense and sparse")

    def test_chunks_with_no_metas_at_all_report_the_meta_error(self):
        hr = _hybrid()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", ["doc1-chunk0"], ["doc2-chunk0"], [], []
        )

        self.assertEqual(docs, [])
        self.assertEqual(metas, [])
        self.assertEqual(status, "ERROR: no retrieved metas in both dense and sparse")

    def test_side_with_no_metas_at_all_is_dropped_whole(self):
        # Chunks and metas are zipped positionally, so a side that supplies no
        # metadata is discarded entirely rather than borrowing the other side's.
        hr = _hybrid(retrieval_top_k=5)

        docs, metas, status = hr.retrieve_from_precomputed(
            "q",
            ["doc1-chunk0", "doc1-chunk1"],
            ["doc2-chunk0"],
            [],
            [{"document_id": 2, "chunk_index": 0}],
        )

        self.assertEqual(status, "ok")
        self.assertEqual(_texts(docs), ["doc2-chunk0"])
        _assert_aligned(self, docs, metas)

    def test_chunks_whose_metas_belong_to_the_other_side_yield_no_context(self):
        hr = _hybrid()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q",
            ["orphan chunk"],
            [],
            [],
            [{"document_id": 9, "chunk_index": 0}],
        )

        self.assertEqual(docs, [])
        self.assertEqual(metas, [])
        self.assertEqual(status, "ERROR: no chunks after dedup and fallback")


@unittest.skipIf(HybridRAG is None, f"hybrid_rag unavailable: {IMPORT_ERROR}")
class RerankerToggleAndDegradationTests(unittest.TestCase):
    def test_disabled_reranking_never_loads_a_model(self):
        with mock.patch.object(HybridRAG, "_load_reranker") as load:
            hr = HybridRAG({"rerank_only": True})
            hr.retrieve_from_precomputed("q", ["text"], [], [{}], [], use_reranker=False)
        load.assert_not_called()

    def test_request_copies_load_one_shared_model(self):
        from copy import copy
        from concurrent.futures import ThreadPoolExecutor

        model = mock.Mock()
        model.rerank.side_effect = _identity_rerank
        hr = HybridRAG({"rerank_only": True})
        copies = [copy(hr), copy(hr)]

        def load(instance):
            instance._model = model

        with mock.patch.object(HybridRAG, "_load_reranker", autospec=True, side_effect=load) as loader:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda instance: instance._rerank("q", ["a", "b"]), copies))
        self.assertEqual(results, [([0, 1], "ok"), ([0, 1], "ok")])
        loader.assert_called_once()
        self.assertIs(copies[0]._model, copies[1]._model)

    def test_use_reranker_false_merges_and_truncates_without_calling_the_model(self):
        # The fake would reverse the order, so merge order in the result proves
        # the reranker was bypassed rather than merely returning identity.
        hr = _hybrid(retrieval_top_k=2, rerank=_reversed_rerank)
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", dense_chunks, sparse_chunks, dense_metas, sparse_metas,
            use_reranker=False,
        )

        self.assertEqual(status, "ok (rerank disabled)")
        hr._model.rerank.assert_not_called()
        self.assertEqual(_texts(docs), _MERGE_ORDER[:2])
        _assert_aligned(self, docs, metas)

    def test_use_reranker_false_still_dedups(self):
        hr = _hybrid(retrieval_top_k=10, rerank=_reversed_rerank)
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", dense_chunks, sparse_chunks, dense_metas, sparse_metas,
            use_reranker=False,
        )

        self.assertEqual(status, "ok (rerank disabled)")
        self.assertEqual(_texts(docs), _MERGE_ORDER)
        _assert_aligned(self, docs, metas)

    def test_use_reranker_cannot_be_passed_positionally(self):
        # Keyword-only is a contract with pipeline/app_pipeline.py, which relies
        # on the positional slots staying (query, dense, sparse, dm, sm).
        hr = _hybrid()

        with self.assertRaises(TypeError):
            hr.retrieve_from_precomputed(
                "q",
                ["doc1-chunk0"],
                ["doc2-chunk0"],
                [{"document_id": 1, "chunk_index": 0}],
                [{"document_id": 2, "chunk_index": 0}],
                False,
            )

    def test_reranker_that_failed_to_load_degrades_to_merge_order(self):
        hr = _hybrid_with_failed_load(retrieval_top_k=10)
        self.assertIsNone(hr._model)
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", dense_chunks, sparse_chunks, dense_metas, sparse_metas
        )

        self.assertEqual(status, "ok (reranking skipped: reranker unavailable)")
        self.assertEqual(_texts(docs), _MERGE_ORDER)
        _assert_aligned(self, docs, metas)

    def test_missing_model_attribute_degrades_instead_of_raising(self):
        # A partially constructed instance never gets _model assigned at all;
        # the getattr guard in _rerank is what keeps retrieval alive.
        hr = _hybrid_without_model_attribute(retrieval_top_k=10)
        self.assertFalse(hasattr(hr, "_model"))
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", dense_chunks, sparse_chunks, dense_metas, sparse_metas
        )

        self.assertEqual(status, "ok (reranking skipped: reranker unavailable)")
        self.assertEqual(_texts(docs), _MERGE_ORDER)

    def test_reranker_returning_no_indices_falls_back_to_merge_order(self):
        hr = _hybrid(retrieval_top_k=2, rerank=_empty_rerank)
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", dense_chunks, sparse_chunks, dense_metas, sparse_metas
        )

        self.assertEqual(status, "ERROR: reranker returned empty indices")
        self.assertEqual(_texts(docs), _MERGE_ORDER[:2])
        _assert_aligned(self, docs, metas)

    def test_reranker_raising_keeps_merge_order_and_reports_error(self):
        hr = _hybrid(retrieval_top_k=10, rerank=RuntimeError("model exploded"))
        dense_chunks, dense_metas = _dense_side()
        sparse_chunks, sparse_metas = _sparse_side()

        docs, metas, status = hr.retrieve_from_precomputed(
            "q", dense_chunks, sparse_chunks, dense_metas, sparse_metas
        )

        self.assertEqual(status, "ERROR: reranking failed, returning original order")
        self.assertEqual(_texts(docs), _MERGE_ORDER)
        _assert_aligned(self, docs, metas)


if __name__ == "__main__":
    unittest.main()
