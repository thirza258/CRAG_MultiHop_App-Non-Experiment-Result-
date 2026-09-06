"""Tests for MultiHopRetriever: dedup keys, the hop loop and its degradations.

Everything runs against hand-written stub retrievers / LLMs, so no network,
no API key and no Redis/Postgres/ChromaDB server are needed. MagicMock is
deliberately avoided for the wrapped retriever: a bare MagicMock's retrieve()
resolves to (*args, **kwargs), which makes _accepts_seen_urls() report True and
would hide signature regressions.
"""

import unittest
from unittest import mock

try:
    from common.runtime import context as runtime_context
    from common.runtime.config import normalize_pipeline_config
    from multi_hop import multi_hop_rag
    from multi_hop.multi_hop_rag import MultiHopRetriever
    IMPORT_ERROR = ""
except Exception as exc:  # needs langchain_openai/openai via ai_handler.llm
    multi_hop_rag = None
    MultiHopRetriever = None
    IMPORT_ERROR = str(exc)


# ──────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────

class NarrowRetriever:
    """Inner retriever whose retrieve() does NOT accept seen_urls (Dense/SparseRAG)."""

    def __init__(self, hops):
        # hops: list of (chunks, metas) tuples, or an Exception instance to raise.
        self._hops = list(hops)
        self.calls = []

    def _next(self, recorded):
        self.calls.append(recorded)
        if not self._hops:
            # Exhausted: behave like an empty index. Every test asserts the call
            # count, so an unexpected extra hop still shows up as a failure.
            return [], []
        result = self._hops.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def retrieve(self, query, keyword=None, where_filter=None):
        return self._next(
            {"query": query, "keyword": keyword, "where_filter": where_filter}
        )

    def set_emitter(self, emitter):
        self.emitter = emitter


class SeenUrlsRetriever(NarrowRetriever):
    """Inner retriever whose retrieve() takes seen_urls (CorrectiveRAG)."""

    def retrieve(self, query, keyword=None, where_filter=None, seen_urls=None):
        return self._next(
            {
                "query": query,
                "keyword": keyword,
                "where_filter": where_filter,
                "seen_urls": seen_urls,
            }
        )


class KwargsRetriever(NarrowRetriever):
    """Inner retriever that swallows anything via **kwargs."""

    def retrieve(self, query, keyword=None, **kwargs):
        recorded = {"query": query, "keyword": keyword}
        recorded.update(kwargs)
        return self._next(recorded)


class FakeLLM:
    """Stand-in for OpenRouterLLM: canned bridge replies, last one repeats."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = self._responses[0] if len(self._responses) == 1 else self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeEvaluator:
    """Stand-in for CRAGEvaluator's score_docs()."""

    def __init__(self, scores=None, error=None):
        self._scores = scores or []
        self._error = error
        self.calls = []

    def score_docs(self, query, chunks):
        self.calls.append((query, list(chunks)))
        if self._error:
            raise self._error
        return self._scores


def _bare_instance():
    """MultiHopRetriever without running __init__ (no LLM client construction)."""
    return object.__new__(MultiHopRetriever)


def _build(inner, llm=None, **config):
    """Construct a real MultiHopRetriever with the LLM client patched out."""
    cfg = {"max_hops": 3, "top_k": 5}
    cfg.update(config)
    llm = llm if llm is not None else FakeLLM(["SUFFICIENT"])
    with mock.patch.object(multi_hop_rag, "OpenRouterLLM", return_value=llm):
        return MultiHopRetriever(inner, cfg)


def _chunk(text):
    return f"Text: {text}"


def _meta(doc_id, chunk_index=0, **extra):
    meta = {"document_id": doc_id, "chunk_index": chunk_index}
    meta.update(extra)
    return meta


# ──────────────────────────────────────────────────────────────────────
# Static helpers
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class NormalizeUrlTests(unittest.TestCase):
    def test_trailing_slash_and_case_collapse(self):
        base = MultiHopRetriever._normalize_url("https://example.com/a")
        self.assertEqual(MultiHopRetriever._normalize_url("https://example.com/a/"), base)
        self.assertEqual(MultiHopRetriever._normalize_url("https://EXAMPLE.com/A"), base)
        self.assertEqual(MultiHopRetriever._normalize_url("  https://EXAMPLE.com/A/  "), base)

    def test_query_string_and_fragment_are_dropped(self):
        base = MultiHopRetriever._normalize_url("https://example.com/a")
        self.assertEqual(MultiHopRetriever._normalize_url("https://example.com/a?x=1"), base)
        self.assertEqual(MultiHopRetriever._normalize_url("https://example.com/a#frag"), base)
        self.assertEqual(MultiHopRetriever._normalize_url("https://example.com/a?x=1#frag"), base)

    def test_different_paths_stay_different(self):
        self.assertNotEqual(
            MultiHopRetriever._normalize_url("https://example.com/a"),
            MultiHopRetriever._normalize_url("https://example.com/b"),
        )


@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class ExtractTextOnlyTests(unittest.TestCase):
    def test_returns_trimmed_part_after_marker(self):
        chunk = "Title: A\nAuthor: B\nSource: S\nCategory: C\n\nText:   the body  "
        self.assertEqual(MultiHopRetriever._extract_text_only(chunk), "the body")

    def test_chunk_without_marker_passes_through_trimmed(self):
        self.assertEqual(MultiHopRetriever._extract_text_only("  plain chunk  "), "plain chunk")

    def test_only_the_first_marker_splits(self):
        # maxsplit=1: a body that mentions "Text:" itself must not be truncated.
        self.assertEqual(
            MultiHopRetriever._extract_text_only("Text: a Text: b"), "a Text: b"
        )


@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class MetaToKeyTests(unittest.TestCase):
    def test_url_key_normalizes_but_keeps_chunk_index(self):
        first = MultiHopRetriever._meta_to_key({"url": "https://example.com/a", "chunk_index": 0})
        same = MultiHopRetriever._meta_to_key({"url": "https://EXAMPLE.com/a/", "chunk_index": 0})
        other_chunk = MultiHopRetriever._meta_to_key({"url": "https://example.com/a", "chunk_index": 1})

        self.assertEqual(first, same)
        self.assertNotEqual(first, other_chunk)

    def test_url_key_defaults_missing_chunk_index_to_zero(self):
        self.assertEqual(
            MultiHopRetriever._meta_to_key({"url": "https://example.com/a"}),
            MultiHopRetriever._meta_to_key({"url": "https://example.com/a", "chunk_index": 0}),
        )

    def test_document_id_key_keeps_chunk_index_separate(self):
        a = MultiHopRetriever._meta_to_key({"document_id": 7, "chunk_index": 0})
        b = MultiHopRetriever._meta_to_key({"document_id": 7, "chunk_index": 1})
        c = MultiHopRetriever._meta_to_key({"document_id": 8, "chunk_index": 0})

        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_document_id_zero_is_still_an_identity(self):
        # `is not None` rather than truthiness: document_id 0 must not fall
        # through to the content hash.
        self.assertEqual(
            MultiHopRetriever._meta_to_key({"document_id": 0}, "some text"),
            MultiHopRetriever._meta_to_key({"document_id": 0}, "different text"),
        )

    def test_falls_back_to_text_hash_without_url_or_document_id(self):
        self.assertEqual(
            MultiHopRetriever._meta_to_key({}, "same text"),
            MultiHopRetriever._meta_to_key({}, "same text"),
        )
        self.assertNotEqual(
            MultiHopRetriever._meta_to_key({}, "text one"),
            MultiHopRetriever._meta_to_key({}, "text two"),
        )


# ──────────────────────────────────────────────────────────────────────
# _deduplicate_new_chunks
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class DeduplicateNewChunksTests(unittest.TestCase):
    def setUp(self):
        self.retriever = _bare_instance()

    def test_drops_already_seen_keys(self):
        chunks = [_chunk("a"), _chunk("b")]
        metas = [_meta(1, 0), _meta(1, 1)]
        seen = {MultiHopRetriever._meta_to_key(metas[0], chunks[0])}

        new_chunks, new_metas = self.retriever._deduplicate_new_chunks(chunks, metas, seen)

        self.assertEqual(len(new_chunks), 1)
        self.assertIn("Text: b", new_chunks[0])
        self.assertEqual(new_metas, [metas[1]])

    def test_groups_chunks_sharing_a_key_and_joins_their_text(self):
        chunks = [_chunk("first"), _chunk("second")]
        metas = [_meta(9, 3, title="old"), _meta(9, 3, title="new")]

        new_chunks, new_metas = self.retriever._deduplicate_new_chunks(chunks, metas, set())

        self.assertEqual(len(new_chunks), 1)
        self.assertIn("Text: first\n\nsecond", new_chunks[0])
        # meta_lookup is last-wins, so the group carries the later metadata.
        self.assertIn("Title: new", new_chunks[0])
        self.assertEqual(new_metas, [metas[1]])

    def test_same_url_with_different_chunk_index_does_not_collide(self):
        chunks = [_chunk("a"), _chunk("b")]
        metas = [
            {"url": "https://example.com/a", "chunk_index": 0},
            {"url": "https://example.com/a", "chunk_index": 1},
        ]

        new_chunks, new_metas = self.retriever._deduplicate_new_chunks(chunks, metas, set())

        self.assertEqual(len(new_chunks), 2)
        self.assertEqual(len(new_metas), 2)

    def test_enriches_chunk_with_metadata_header(self):
        meta = _meta(1, 0, title="T", author="A", source="S", category="C")

        new_chunks, _ = self.retriever._deduplicate_new_chunks([_chunk("body")], [meta], set())

        self.assertEqual(
            new_chunks[0],
            "Title: T\nAuthor: A\nSource: S\nCategory: C\n\nText: body",
        )

    def test_source_falls_back_to_username_and_category_defaults(self):
        meta = _meta(1, 0, username="bob")

        new_chunks, _ = self.retriever._deduplicate_new_chunks([_chunk("body")], [meta], set())

        self.assertIn("Source: bob", new_chunks[0])
        self.assertIn("Category: user_document", new_chunks[0])
        self.assertIn("Title: \n", new_chunks[0])

    def test_short_metadata_list_still_yields_equal_length_lists(self):
        # CorrectiveRAG's "incorrect_local_fallback" paths return (docs, []), so a
        # metas list shorter than the chunk list really does reach here. zip()
        # truncates: the equal-length contract holds, but the extra chunks are
        # dropped — pinning the current behaviour, not endorsing it.
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        metas = [_meta(1, 0), _meta(1, 1)]

        new_chunks, new_metas = self.retriever._deduplicate_new_chunks(chunks, metas, set())

        self.assertEqual(len(new_chunks), len(new_metas))
        self.assertEqual(len(new_chunks), 2)
        self.assertNotIn("Text: c", "".join(new_chunks))

    def test_mutates_seen_keys_so_a_repeat_call_yields_nothing(self):
        chunks = [_chunk("a"), _chunk("b")]
        metas = [_meta(1, 0), _meta(1, 1)]
        seen = set()

        first_chunks, first_metas = self.retriever._deduplicate_new_chunks(chunks, metas, seen)

        self.assertEqual(len(first_chunks), 2)
        self.assertEqual(len(first_chunks), len(first_metas))
        self.assertEqual(len(seen), 2)

        repeat_chunks, repeat_metas = self.retriever._deduplicate_new_chunks(chunks, metas, seen)

        self.assertEqual(repeat_chunks, [])
        self.assertEqual(repeat_metas, [])


# ──────────────────────────────────────────────────────────────────────
# Construction defaults
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class ConstructionDefaultsTests(unittest.TestCase):
    def test_emitter_defaults_to_null_emitter(self):
        # The pipeline injects the real emitter later; until then emits must be no-ops.
        retriever = _build(NarrowRetriever([]))
        self.assertIs(retriever.emitter, multi_hop_rag.NULL_EMITTER)

    def test_missing_evaluator_on_inner_retriever_is_tolerated(self):
        inner = NarrowRetriever([])
        self.assertFalse(hasattr(inner, "evaluator"))

        self.assertIsNone(_build(inner).evaluator)

    def test_evaluator_is_taken_from_the_inner_retriever(self):
        inner = NarrowRetriever([])
        evaluator = FakeEvaluator()
        inner.evaluator = evaluator

        self.assertIs(_build(inner).evaluator, evaluator)

    def test_top_k_prefers_config_then_inner_retriever_then_five(self):
        inner = NarrowRetriever([])
        inner.top_k = 7

        self.assertEqual(_build(inner, top_k=2).top_k, 2)

        with mock.patch.object(multi_hop_rag, "OpenRouterLLM", return_value=FakeLLM(["SUFFICIENT"])):
            self.assertEqual(MultiHopRetriever(inner, {}).top_k, 7)
            self.assertEqual(MultiHopRetriever(NarrowRetriever([]), {}).top_k, 5)

    def test_max_hops_defaults_to_three(self):
        with mock.patch.object(multi_hop_rag, "OpenRouterLLM", return_value=FakeLLM(["SUFFICIENT"])):
            self.assertEqual(MultiHopRetriever(NarrowRetriever([]), {}).max_hops, 3)

    def test_set_emitter_propagates_to_the_inner_retriever(self):
        # Dense/Sparse status events reach the browser only through this hand-off.
        inner = NarrowRetriever([])
        retriever = _build(inner)
        emitter = object()

        retriever.set_emitter(emitter)

        self.assertIs(retriever.emitter, emitter)
        self.assertIs(inner.emitter, emitter)

    def test_llm_is_built_with_the_configured_model(self):
        with mock.patch.object(multi_hop_rag, "OpenRouterLLM") as llm_cls:
            MultiHopRetriever(NarrowRetriever([]), {"llm_model": "vendor/model-x"})

        llm_cls.assert_called_once_with(model="vendor/model-x")


# ──────────────────────────────────────────────────────────────────────
# _accepts_seen_urls / seen_urls propagation
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class AcceptsSeenUrlsTests(unittest.TestCase):
    def test_true_for_explicit_seen_urls_parameter(self):
        self.assertTrue(MultiHopRetriever._accepts_seen_urls(SeenUrlsRetriever([])))

    def test_true_for_var_keyword_retriever(self):
        self.assertTrue(MultiHopRetriever._accepts_seen_urls(KwargsRetriever([])))

    def test_false_for_narrow_signature(self):
        self.assertFalse(MultiHopRetriever._accepts_seen_urls(NarrowRetriever([])))

    def test_false_when_there_is_no_retrieve_attribute(self):
        self.assertFalse(MultiHopRetriever._accepts_seen_urls(object()))

    def test_false_when_retrieve_cannot_be_introspected(self):
        # inspect.signature() raises TypeError for a non-callable retrieve;
        # assuming the narrow signature keeps the hop from TypeError-ing.
        class Hostile:
            retrieve = object()

        self.assertFalse(MultiHopRetriever._accepts_seen_urls(Hostile()))


@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class SeenUrlsPropagationTests(unittest.TestCase):
    def test_seen_urls_is_passed_when_the_signature_accepts_it(self):
        inner = SeenUrlsRetriever([([_chunk("a")], [_meta(1)])])
        retriever = _build(inner)

        retriever.multi_hop_retrieve("q")

        self.assertTrue(retriever._inner_accepts_seen_urls)
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(inner.calls[0]["seen_urls"], set())

    def test_seen_urls_is_withheld_from_a_narrow_signature(self):
        inner = NarrowRetriever([([_chunk("a")], [_meta(1)])])
        retriever = _build(inner)

        chunks, metas = retriever.multi_hop_retrieve("q")

        self.assertFalse(retriever._inner_accepts_seen_urls)
        # A wrongly-True probe would TypeError inside the hop and be swallowed,
        # leaving no chunks at all.
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(metas), 1)
        self.assertNotIn("seen_urls", inner.calls[0])


# ──────────────────────────────────────────────────────────────────────
# Bridge decision
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class BridgeDecisionTests(unittest.TestCase):
    def _decide(self, response):
        retriever = _bare_instance()
        retriever.llm = FakeLLM([response])
        return retriever._extract_bridge_or_done("question", ["passage"])

    def test_need_returns_the_bridge_entity(self):
        self.assertEqual(self._decide("NEED: Marie Curie"), "Marie Curie")

    def test_sufficient_returns_none(self):
        self.assertIsNone(self._decide("SUFFICIENT"))

    def test_unparseable_reply_returns_none(self):
        self.assertIsNone(self._decide("I think you should look at radium"))

    def test_empty_or_non_string_reply_returns_none(self):
        self.assertIsNone(self._decide("   "))
        self.assertIsNone(self._decide(None))

    def test_provider_error_body_is_not_treated_as_a_bridge_entity(self):
        # ai_handler.llm returns exhausted retries as a plain string; searching
        # for that string on the next hop would be nonsense.
        for prefix in multi_hop_rag._LLM_ERROR_PREFIXES:
            self.assertIsNone(self._decide(f"{prefix} (model-x): connection refused"))

    def test_llm_exception_returns_none_instead_of_propagating(self):
        self.assertIsNone(self._decide(RuntimeError("no api key")))


# ──────────────────────────────────────────────────────────────────────
# The hop loop
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class HopLoopTests(unittest.TestCase):
    def test_stops_at_max_hops_even_when_the_llm_keeps_asking(self):
        inner = NarrowRetriever([
            ([_chunk("a")], [_meta(1)]),
            ([_chunk("b")], [_meta(2)]),
            ([_chunk("c")], [_meta(3)]),
        ])
        llm = FakeLLM(["NEED: second", "NEED: third"])
        retriever = _build(inner, llm=llm, max_hops=2)

        chunks, metas = retriever.multi_hop_retrieve("first")

        self.assertEqual(len(inner.calls), 2)
        self.assertEqual([call["query"] for call in inner.calls], ["first", "second"])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(metas), 2)

    def test_stops_when_a_hop_returns_no_chunks(self):
        inner = NarrowRetriever([([], [])])
        llm = FakeLLM(["NEED: never asked"])
        retriever = _build(inner, llm=llm, max_hops=3)

        chunks, metas = retriever.multi_hop_retrieve("q")

        self.assertEqual((chunks, metas), ([], []))
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(llm.prompts, [])

    def test_stops_when_a_hop_returns_only_duplicates(self):
        duplicate_meta = _meta(1, 0)
        inner = NarrowRetriever([
            ([_chunk("a")], [dict(duplicate_meta)]),
            ([_chunk("a again")], [dict(duplicate_meta)]),
            ([_chunk("never reached")], [_meta(2)]),
        ])
        llm = FakeLLM(["NEED: second"])
        retriever = _build(inner, llm=llm, max_hops=3)

        chunks, _ = retriever.multi_hop_retrieve("q")

        self.assertEqual(len(inner.calls), 2)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Text: a", chunks[0])
        # The bridge is only consulted after a hop contributed something new.
        self.assertEqual(len(llm.prompts), 1)

    def test_follows_the_bridge_entity_as_the_next_query(self):
        inner = NarrowRetriever([
            ([_chunk("a")], [_meta(1)]),
            ([_chunk("b")], [_meta(2)]),
        ])
        llm = FakeLLM(["NEED: bridge entity", "SUFFICIENT"])
        retriever = _build(inner, llm=llm, max_hops=3)

        retriever.multi_hop_retrieve("original", keyword="kw", where_filter={"user_id": 1})

        self.assertEqual([call["query"] for call in inner.calls], ["original", "bridge entity"])
        # keyword / where_filter are hop-invariant.
        for call in inner.calls:
            self.assertEqual(call["keyword"], "kw")
            self.assertEqual(call["where_filter"], {"user_id": 1})

    def test_bridge_prompt_carries_every_chunk_gathered_so_far(self):
        inner = NarrowRetriever([
            ([_chunk("hop one body")], [_meta(1)]),
            ([_chunk("hop two body")], [_meta(2)]),
        ])
        llm = FakeLLM(["NEED: second", "SUFFICIENT"])
        retriever = _build(inner, llm=llm, max_hops=3)

        retriever.multi_hop_retrieve("original")

        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("hop one body", llm.prompts[0])
        self.assertIn("hop one body", llm.prompts[1])
        self.assertIn("hop two body", llm.prompts[1])
        # The bridge always reasons about the user's original question.
        self.assertIn("original", llm.prompts[1])

    def test_stops_on_sufficient(self):
        inner = NarrowRetriever([
            ([_chunk("a")], [_meta(1)]),
            ([_chunk("b")], [_meta(2)]),
        ])
        llm = FakeLLM(["SUFFICIENT"])
        retriever = _build(inner, llm=llm, max_hops=3)

        chunks, _ = retriever.multi_hop_retrieve("q")

        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(len(chunks), 1)

    def test_stops_on_an_unparseable_bridge_reply(self):
        inner = NarrowRetriever([
            ([_chunk("a")], [_meta(1)]),
            ([_chunk("b")], [_meta(2)]),
        ])
        llm = FakeLLM(["Sure! Here is what I think about the passages."])
        retriever = _build(inner, llm=llm, max_hops=3)

        chunks, _ = retriever.multi_hop_retrieve("q")

        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(len(chunks), 1)

    def test_stops_and_keeps_chunks_when_the_bridge_llm_raises(self):
        inner = NarrowRetriever([
            ([_chunk("a")], [_meta(1)]),
            ([_chunk("b")], [_meta(2)]),
        ])
        llm = FakeLLM([RuntimeError("openrouter unreachable")])
        retriever = _build(inner, llm=llm, max_hops=3)

        chunks, metas = retriever.multi_hop_retrieve("q")

        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(metas), 1)

    def test_stops_when_the_bridge_llm_returns_a_provider_error_string(self):
        inner = NarrowRetriever([
            ([_chunk("a")], [_meta(1)]),
            ([_chunk("b")], [_meta(2)]),
        ])
        llm = FakeLLM(["OpenRouter Error (model-x): 401 no credentials"])
        retriever = _build(inner, llm=llm, max_hops=3)

        chunks, _ = retriever.multi_hop_retrieve("q")

        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(len(chunks), 1)

    def test_inner_retriever_failure_keeps_earlier_hops(self):
        inner = NarrowRetriever([
            ([_chunk("kept")], [_meta(1)]),
            RuntimeError("chroma down"),
        ])
        llm = FakeLLM(["NEED: second"])
        retriever = _build(inner, llm=llm, max_hops=3)

        chunks, metas = retriever.multi_hop_retrieve("q")

        self.assertEqual(len(inner.calls), 2)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Text: kept", chunks[0])
        self.assertEqual(metas, [_meta(1)])

    def test_first_hop_failure_returns_two_empty_lists(self):
        inner = NarrowRetriever([RuntimeError("chroma down")])
        retriever = _build(inner, max_hops=3)

        self.assertEqual(retriever.multi_hop_retrieve("q"), ([], []))

    def test_each_hop_is_truncated_to_top_k(self):
        wide = [_chunk(f"c{i}") for i in range(5)]
        metas = [_meta(i) for i in range(5)]
        inner = NarrowRetriever([(wide, metas)])
        retriever = _build(inner, llm=FakeLLM(["SUFFICIENT"]), top_k=2)

        chunks, out_metas = retriever.multi_hop_retrieve("q")

        # No evaluator, so nothing truncates the final pool: a pool of 2 proves
        # the per-hop slice happened.
        self.assertIsNone(retriever.evaluator)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(out_metas, metas[:2])

    def test_retrieve_delegates_positionally_to_multi_hop_retrieve(self):
        inner = NarrowRetriever([([_chunk("a")], [_meta(1)])])
        retriever = _build(inner, llm=FakeLLM(["SUFFICIENT"]))

        retriever.retrieve("q", "kw", {"user_id": 3})

        self.assertEqual(
            inner.calls[0],
            {"query": "q", "keyword": "kw", "where_filter": {"user_id": 3}},
        )


# ──────────────────────────────────────────────────────────────────────
# Final re-rank
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class FinalRerankTests(unittest.TestCase):
    def _two_hop_retriever(self, evaluator=None, top_k=2):
        inner = NarrowRetriever([
            ([_chunk("c0"), _chunk("c1")], [_meta(0), _meta(1)]),
            ([_chunk("c2"), _chunk("c3")], [_meta(2), _meta(3)]),
        ])
        if evaluator is not None:
            inner.evaluator = evaluator
        llm = FakeLLM(["NEED: second", "SUFFICIENT"])
        return inner, _build(inner, llm=llm, max_hops=2, top_k=top_k)

    def test_returns_the_top_k_highest_scored_chunks(self):
        evaluator = FakeEvaluator(scores=[0.1, 0.9, 0.2, 0.8])
        inner, retriever = self._two_hop_retriever(evaluator=evaluator)

        chunks, metas = retriever.multi_hop_retrieve("original")

        self.assertEqual(len(inner.calls), 2)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Text: c1", chunks[0])
        self.assertIn("Text: c3", chunks[1])
        self.assertEqual(metas, [_meta(1), _meta(3)])
        # Scoring uses the user's original question, not the bridge query.
        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(evaluator.calls[0][0], "original")
        self.assertEqual(len(evaluator.calls[0][1]), 4)

    def test_pool_at_or_below_top_k_is_returned_unscored(self):
        evaluator = FakeEvaluator(scores=[0.1, 0.9, 0.2, 0.8])
        _, retriever = self._two_hop_retriever(evaluator=evaluator, top_k=4)

        chunks, _ = retriever.multi_hop_retrieve("original")

        self.assertEqual(len(chunks), 4)
        self.assertEqual(evaluator.calls, [])

    def test_without_an_evaluator_the_whole_pool_is_returned(self):
        # Nothing to rank by, so the pool is handed on untruncated — HybridRAG
        # re-ranks the merged dense+sparse result downstream.
        inner, retriever = self._two_hop_retriever(evaluator=None)

        chunks, metas = retriever.multi_hop_retrieve("original")

        self.assertIsNone(retriever.evaluator)
        self.assertEqual(retriever.top_k, 2)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(metas, [_meta(0), _meta(1), _meta(2), _meta(3)])

    def test_score_docs_failure_falls_back_to_the_first_top_k(self):
        evaluator = FakeEvaluator(error=RuntimeError("embedding model missing"))
        _, retriever = self._two_hop_retriever(evaluator=evaluator)

        chunks, metas = retriever.multi_hop_retrieve("original")

        self.assertEqual(len(chunks), 2)
        self.assertIn("Text: c0", chunks[0])
        self.assertIn("Text: c1", chunks[1])
        # The equal-length chunk/meta contract must survive the fallback.
        self.assertEqual(metas, [_meta(0), _meta(1)])


# ──────────────────────────────────────────────────────────────────────
# max_hops — per request, not per instance
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class MaxHopsPerRequestTests(unittest.TestCase):
    """One retriever instance serves every concurrent query.

    The pipeline used to assign the ceiling onto that instance before each
    query, so two requests asking for different hop counts raced and whichever
    wrote last set the ceiling for both. It is read from the request context now.
    """

    def _runtime(self, **params):
        return runtime_context.use_runtime(runtime_context.RuntimeSettings(
            params=normalize_pipeline_config(params)
        ))

    def test_the_request_ceiling_wins_over_the_configured_one(self):
        retriever = _build(NarrowRetriever([]), max_hops=3)

        with self._runtime(max_hops=1):
            self.assertEqual(retriever.max_hops, 1)

    def test_an_unset_ceiling_leaves_the_configured_one(self):
        # A deployment that tuned multi_hop_config down must not be pushed back
        # up by a request that never mentioned hops.
        retriever = _build(NarrowRetriever([]), max_hops=2)

        with self._runtime():
            self.assertEqual(retriever.max_hops, 2)

        self.assertEqual(retriever.max_hops, 2)

    def test_the_ceiling_does_not_outlive_the_request(self):
        retriever = _build(NarrowRetriever([]), max_hops=3)

        with self._runtime(max_hops=1):
            pass

        self.assertEqual(retriever.max_hops, 3)

    def test_two_concurrent_requests_do_not_share_a_ceiling(self):
        import threading

        retriever = _build(NarrowRetriever([]), max_hops=3)
        seen = {}
        start = threading.Barrier(2)

        def query(name, hops):
            with self._runtime(max_hops=hops):
                start.wait(timeout=5)   # both inside their runtime at once
                seen[name] = retriever.max_hops

        threads = [
            threading.Thread(target=query, args=("a", 1)),
            threading.Thread(target=query, args=("b", 3)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(seen, {"a": 1, "b": 3})

    def test_the_hop_loop_honours_the_request_ceiling(self):
        # Not just the attribute: the loop that reads it.
        inner = NarrowRetriever([
            ([_chunk("a")], [_meta(1)]),
            ([_chunk("b")], [_meta(2)]),
            ([_chunk("c")], [_meta(3)]),
        ])
        retriever = _build(
            inner, llm=FakeLLM(["INSUFFICIENT: more", "INSUFFICIENT: more", "SUFFICIENT"]),
            max_hops=3,
        )

        with self._runtime(max_hops=1):
            retriever.multi_hop_retrieve("q")

        self.assertEqual(len(inner.calls), 1)


# ──────────────────────────────────────────────────────────────────────
# set_max_hops
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(MultiHopRetriever is None, f"multi_hop unavailable: {IMPORT_ERROR}")
class SetMaxHopsTests(unittest.TestCase):
    def test_clamps_zero_and_negative_to_one(self):
        retriever = _build(NarrowRetriever([]), max_hops=3)

        retriever.set_max_hops(0)
        self.assertEqual(retriever.max_hops, 1)

        retriever.set_max_hops(-5)
        self.assertEqual(retriever.max_hops, 1)

    def test_coerces_numeric_strings_and_floats(self):
        retriever = _build(NarrowRetriever([]), max_hops=3)

        retriever.set_max_hops("4")
        self.assertEqual(retriever.max_hops, 4)

        retriever.set_max_hops(2.7)
        self.assertEqual(retriever.max_hops, 2)

    def test_invalid_values_keep_the_current_setting(self):
        retriever = _build(NarrowRetriever([]), max_hops=3)

        retriever.set_max_hops(None)
        self.assertEqual(retriever.max_hops, 3)

        retriever.set_max_hops("plenty")
        self.assertEqual(retriever.max_hops, 3)

    def test_clamped_value_makes_the_hop_loop_run_once(self):
        inner = NarrowRetriever([([_chunk("a")], [_meta(1)])])
        retriever = _build(inner, llm=FakeLLM(["SUFFICIENT"]), max_hops=0)

        # max_hops straight from config is not clamped, so range(0) skips the loop.
        self.assertEqual(retriever.multi_hop_retrieve("q"), ([], []))
        self.assertEqual(inner.calls, [])

        retriever.set_max_hops(0)
        chunks, _ = retriever.multi_hop_retrieve("q")

        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
