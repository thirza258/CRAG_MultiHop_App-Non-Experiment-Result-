"""Tests for CorrectiveRAG: filter/URL bookkeeping, the decision branches of
retrieve_with_decision, ambiguous resolution, external-search isolation and the
local-vs-external fallbacks of _fetch_fully_external.

CRAGEvaluator, ExternalSearcher and QueryExpander are patched at the point
corrective_rag imports them, so nothing here loads a model, needs an API key or
touches the network.
"""

import unittest
from unittest import mock

try:
    from common.runtime import context as runtime_context
    from corrective import corrective_rag
    from corrective.corrective_rag import CorrectiveRAG
    from corrective.external_search import ExternalSearcher
    from emitter.status import NULL_EMITTER
    IMPORT_ERROR = ""
except Exception as exc:  # corrective_evaluator needs torch/transformers + NLTK data + Django
    corrective_rag = None
    CorrectiveRAG = None
    ExternalSearcher = None
    NULL_EMITTER = None
    runtime_context = None
    IMPORT_ERROR = str(exc)


def _with_params(**params):
    """Install a request runtime carrying just these pipeline-config values."""
    return runtime_context.use_runtime(runtime_context.RuntimeSettings(params=params))


# ──────────────────────────────────────────────────────────────────────
# fakes
# ──────────────────────────────────────────────────────────────────────

class _RecordingRetriever:
    """Stand-in for the wrapped Dense/Sparse retriever.

    Mirrors their real signature (no seen_urls, no **kwargs), records every
    call so tests can assert on the keyword and where_filter that CorrectiveRAG
    built, and replays queued results in order (the last one repeats).
    """

    top_k = 4

    def __init__(self, results=None):
        self.results = list(results) if results else [([], [])]
        self.calls = []
        self.emitter = None

    def retrieve(self, query=None, keyword=None, where_filter=None):
        self.calls.append(
            {"query": query, "keyword": keyword, "where_filter": where_filter}
        )
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]

    def set_emitter(self, emitter):
        self.emitter = emitter


class _RecordingEmitter:
    """Captures emitted events instead of publishing them to Redis."""

    def __init__(self):
        self.events = []

    def emit(self, stage, message, meta=None):
        self.events.append((stage, message, meta))

    def messages(self):
        return [message for _, message, _ in self.events]


def _is_error(value):
    return isinstance(value, BaseException) or (
        isinstance(value, type) and issubclass(value, BaseException)
    )


def _configure(method, value):
    """Exception (class or instance) -> the call raises; else it is returned."""
    if _is_error(value):
        method.side_effect = value
    else:
        method.return_value = value


def _evaluator():
    """Fake CRAGEvaluator with an explicit default on every method the source calls.

    Several call sites (score_docs, knowledge_refinement) sit inside try/except
    blocks, so an unconfigured Mock return would raise there, get swallowed and
    silently fake the very fallback branch under test. Defaults must be real
    values, and tests that care about a branch override them.
    """
    evaluator = mock.Mock()
    evaluator.evaluate.return_value = ("incorrect", [], [], float("-inf"))
    evaluator.score_docs.return_value = []
    evaluator.knowledge_refinement.return_value = ""
    return evaluator


def _external(wiki=([], []), news=([], [])):
    """Fake ExternalSearcher; pass an Exception to make that source fail."""
    searcher = mock.Mock()
    _configure(searcher.search_wikipedia, wiki)
    _configure(searcher.search_news, news)
    return searcher


def _expander(keyword="kw", reformulated="reformulated query", alternatives=()):
    """Fake QueryExpander; pass an Exception for the call that should fail."""
    expander = mock.Mock()
    _configure(expander.to_keyword, keyword)
    _configure(expander.reformulate, reformulated)
    _configure(
        expander.expand_multiple,
        alternatives if _is_error(alternatives) else list(alternatives),
    )
    return expander


def _make_crag(retriever=None, evaluator=None, external=None, expander=None, config=None):
    """CorrectiveRAG with its three heavy collaborators swapped out at import point."""
    retriever = _RecordingRetriever() if retriever is None else retriever
    evaluator = _evaluator() if evaluator is None else evaluator
    external = _external() if external is None else external
    expander = _expander() if expander is None else expander

    with mock.patch.object(
        corrective_rag, "CRAGEvaluator", return_value=evaluator
    ), mock.patch.object(
        corrective_rag, "ExternalSearcher", return_value=external
    ), mock.patch.object(
        corrective_rag, "QueryExpander", return_value=expander
    ):
        return CorrectiveRAG(retriever, {} if config is None else config)


# ──────────────────────────────────────────────────────────────────────
# construction / emitter plumbing
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class ConstructionTests(unittest.TestCase):
    def test_top_k_is_taken_from_the_wrapped_retriever(self):
        self.assertEqual(_make_crag(retriever=_RecordingRetriever()).top_k, 4)

    def test_top_k_falls_back_to_five_when_the_retriever_has_none(self):
        self.assertEqual(_make_crag(retriever=mock.Mock(spec=[])).top_k, 5)

    def test_new_instance_defaults_to_the_null_emitter(self):
        self.assertIs(_make_crag().emitter, NULL_EMITTER)

    def test_instance_built_without_init_can_still_emit(self):
        # object.__new__ skips __init__, so only the class-level default keeps
        # the first self.emitter.emit(...) from raising AttributeError.
        bare = object.__new__(CorrectiveRAG)
        self.assertIs(bare.emitter, NULL_EMITTER)
        self.assertIsNone(bare.emitter.emit("corrective_pipeline", "no-op"))

    def test_set_emitter_propagates_to_the_inner_retriever(self):
        retriever = _RecordingRetriever()
        crag = _make_crag(retriever=retriever)
        emitter = _RecordingEmitter()

        crag.set_emitter(emitter)

        self.assertIs(crag.emitter, emitter)
        self.assertIs(retriever.emitter, emitter)


# ──────────────────────────────────────────────────────────────────────
# _build_filter / _track_urls
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class BuildFilterTests(unittest.TestCase):
    def setUp(self):
        self.crag = _make_crag()

    def test_no_filter_when_nothing_has_been_seen(self):
        # ChromaDB rejects an empty $nin, so an empty set must mean "no filter".
        self.assertIsNone(self.crag._build_filter(set()))
        self.assertIsNone(self.crag._build_filter(None))

    def test_filter_excludes_every_seen_url(self):
        built = self.crag._build_filter({"https://a", "https://b"})

        self.assertEqual(set(built), {"url"})
        self.assertEqual(set(built["url"]), {"$nin"})
        self.assertEqual(sorted(built["url"]["$nin"]), ["https://a", "https://b"])


@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class TrackUrlsTests(unittest.TestCase):
    def setUp(self):
        self.crag = _make_crag()

    def test_adds_urls_and_keeps_the_set_unique(self):
        seen = {"https://a"}

        self.crag._track_urls([{"url": "https://a"}, {"url": "https://b"}], seen)

        self.assertEqual(seen, {"https://a", "https://b"})

    def test_tolerates_none_and_empty_meta_lists(self):
        seen = {"https://a"}

        self.crag._track_urls(None, seen)
        self.crag._track_urls([], seen)

        self.assertEqual(seen, {"https://a"})

    def test_skips_non_dict_entries_and_metas_without_a_usable_url(self):
        # ChromaDB can hand back None rows, and local documents carry no url.
        seen = set()

        self.crag._track_urls(
            [None, "not-a-dict", 7, {}, {"url": None}, {"url": ""}, {"url": "https://a"}],
            seen,
        )

        self.assertEqual(seen, {"https://a"})


# ──────────────────────────────────────────────────────────────────────
# query-expander guards
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class SafeExpanderHelperTests(unittest.TestCase):
    def test_safe_keyword_returns_the_expansion(self):
        expander = _expander(keyword="bm25 sparse")
        crag = _make_crag(expander=expander)

        self.assertEqual(crag._safe_keyword("what is bm25?"), "bm25 sparse")
        expander.to_keyword.assert_called_once_with("what is bm25?")

    def test_safe_keyword_falls_back_to_the_raw_query(self):
        crag = _make_crag(expander=_expander(keyword=RuntimeError("no api key")))

        self.assertEqual(crag._safe_keyword("what is bm25?"), "what is bm25?")

    def test_safe_reformulate_returns_the_reformulation(self):
        crag = _make_crag(expander=_expander(reformulated="how does bm25 rank?"))

        self.assertEqual(crag._safe_reformulate("what is bm25?"), "how does bm25 rank?")

    def test_safe_reformulate_falls_back_to_the_raw_query(self):
        crag = _make_crag(expander=_expander(reformulated=RuntimeError("no api key")))

        self.assertEqual(crag._safe_reformulate("what is bm25?"), "what is bm25?")

    def test_safe_expand_multiple_returns_the_alternatives(self):
        expander = _expander(alternatives=["alt one", "alt two"])
        crag = _make_crag(expander=expander)

        self.assertEqual(crag._safe_expand_multiple("q"), ["alt one", "alt two"])
        expander.expand_multiple.assert_called_once_with("q", n=3)

    def test_safe_expand_multiple_returns_empty_on_failure(self):
        crag = _make_crag(expander=_expander(alternatives=RuntimeError("no api key")))

        self.assertEqual(crag._safe_expand_multiple("q"), [])


# ──────────────────────────────────────────────────────────────────────
# retrieve_with_decision
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class RetrieveWithDecisionTests(unittest.TestCase):
    def test_empty_local_result_goes_fully_external(self):
        retriever = _RecordingRetriever([([], [])])
        evaluator = _evaluator()
        evaluator.knowledge_refinement.return_value = "refined E1"
        evaluator.score_docs.return_value = [0.42]
        external = _external(wiki=(["E1"], [{"url": "w1"}]))
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)

        docs, metas, decision = crag.retrieve_with_decision("who?", keyword="kw")

        self.assertEqual(docs, ["refined E1"])
        self.assertEqual(metas, [{"url": "w1"}])
        self.assertEqual(decision, "incorrect")
        evaluator.evaluate.assert_not_called()  # nothing local to grade
        evaluator.score_docs.assert_called_once_with("who?", ["refined E1"])

    def test_correct_decision_returns_the_filtered_docs(self):
        retriever = _RecordingRetriever(
            [(["d1", "d2"], [{"url": "u1"}, {"url": "u2"}])]
        )
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("correct", ["d1"], [{"url": "u1"}], 0.91)
        external = _external()
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)

        docs, metas, decision = crag.retrieve_with_decision("q", keyword="kw")

        self.assertEqual((docs, metas, decision), (["d1"], [{"url": "u1"}], "correct"))
        external.search_wikipedia.assert_not_called()
        external.search_news.assert_not_called()

    def test_seen_urls_become_a_nin_filter_and_a_given_keyword_is_not_expanded(self):
        retriever = _RecordingRetriever([(["d1"], [{"url": "u2"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("correct", ["d1"], [{"url": "u2"}], 0.80)
        expander = _expander()
        crag = _make_crag(
            retriever=retriever, evaluator=evaluator, expander=expander
        )

        crag.retrieve_with_decision("q", keyword="given-kw", seen_urls={"u1"})

        self.assertEqual(retriever.calls[0]["keyword"], "given-kw")
        self.assertEqual(
            retriever.calls[0]["where_filter"], {"url": {"$nin": ["u1"]}}
        )
        expander.to_keyword.assert_not_called()

    def test_keyword_expansion_failure_falls_back_to_the_raw_query(self):
        retriever = _RecordingRetriever([(["d1"], [{}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("correct", ["d1"], [{}], 0.80)
        crag = _make_crag(
            retriever=retriever,
            evaluator=evaluator,
            expander=_expander(keyword=RuntimeError("no api key")),
        )

        docs, _, decision = crag.retrieve_with_decision("what is bm25?")

        self.assertEqual(retriever.calls[0]["keyword"], "what is bm25?")
        self.assertEqual((docs, decision), (["d1"], "correct"))

    def test_evaluator_failure_degrades_to_plain_retrieval(self):
        retriever = _RecordingRetriever([(["d1", "d2"], None)])
        evaluator = _evaluator()
        evaluator.evaluate.side_effect = RuntimeError("e5 model unavailable")
        external = _external()
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)
        emitter = _RecordingEmitter()
        crag.set_emitter(emitter)

        docs, metas, decision = crag.retrieve_with_decision("q", keyword="kw")

        self.assertEqual(docs, ["d1", "d2"])
        self.assertEqual(metas, [])  # None metas normalised so downstream zip() is safe
        self.assertEqual(decision, "correct_degraded")
        external.search_wikipedia.assert_not_called()
        self.assertTrue(
            any("Evaluator unavailable" in m for m in emitter.messages()),
            emitter.messages(),
        )

    def test_incorrect_decision_goes_external_when_external_scores_higher(self):
        retriever = _RecordingRetriever([(["local"], [{"url": "u1"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("incorrect", [], [], 0.10)
        evaluator.knowledge_refinement.return_value = ""  # refinement adds nothing
        evaluator.score_docs.return_value = [0.77]
        external = _external(news=(["N1"], [{"url": "n1"}]))
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)

        docs, metas, decision = crag.retrieve_with_decision("q", keyword="kw")

        self.assertEqual(docs, ["N1"])
        self.assertEqual(metas, [{"url": "n1"}])
        self.assertEqual(decision, "incorrect")
        evaluator.score_docs.assert_called_once_with("q", ["N1"])

    def test_incorrect_decision_keeps_local_when_it_ties_the_external_score(self):
        retriever = _RecordingRetriever([(["local"], [{"url": "u1"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("incorrect", [], [], 0.50)
        evaluator.knowledge_refinement.return_value = "R1"
        evaluator.score_docs.return_value = [0.50]
        external = _external(wiki=(["E1"], [{"url": "w1"}]))
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)

        docs, _, decision = crag.retrieve_with_decision("q", keyword="kw")

        self.assertEqual(docs, ["local"])
        self.assertEqual(decision, "incorrect_local_fallback")
        # Proves the local score from evaluate() reaches the comparison.
        evaluator.score_docs.assert_called_once_with("q", ["R1"])

    def test_ambiguous_resolved_by_handle_ambiguous(self):
        retriever = _RecordingRetriever([(["d1"], [{"url": "u1"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("ambiguous", ["d1"], [{"url": "u1"}], 0.45)
        external = _external()
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)
        crag.handle_ambiguous = mock.Mock(
            return_value=(["r1"], [{"url": "u9"}], "correct")
        )

        docs, metas, decision = crag.retrieve_with_decision("q", keyword="kw")

        self.assertEqual(
            (docs, metas, decision), (["r1"], [{"url": "u9"}], "ambiguous_resolved")
        )
        external.search_wikipedia.assert_not_called()
        # The already-seen url must be handed to the resolver so it can exclude it.
        self.assertEqual(crag.handle_ambiguous.call_args[0][2], {"u1"})

    def test_unresolved_ambiguous_combines_local_and_external_when_external_wins(self):
        retriever = _RecordingRetriever([(["d1"], [{"url": "u1"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("ambiguous", ["L1"], [{"url": "m1"}], 0.30)
        evaluator.knowledge_refinement.return_value = "R1"
        evaluator.score_docs.return_value = [0.90]
        external = _external(wiki=(["E1"], [{"url": "m3"}]))
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)
        crag.handle_ambiguous = mock.Mock(
            return_value=(["A1"], [{"url": "m2"}], "ambiguous")
        )

        docs, metas, decision = crag.retrieve_with_decision("q", keyword="kw")

        self.assertEqual(docs, ["L1", "A1", "R1"])
        self.assertEqual(metas, [{"url": "m1"}, {"url": "m2"}, {"url": "m3"}])
        self.assertEqual(decision, "ambiguous_external")
        evaluator.score_docs.assert_called_once_with("q", ["R1"])

    def test_unresolved_ambiguous_keeps_local_only_when_local_wins(self):
        retriever = _RecordingRetriever([(["d1"], [{"url": "u1"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("ambiguous", ["L1"], [{"url": "m1"}], 0.30)
        evaluator.knowledge_refinement.return_value = ""  # unrefinable chunk kept as-is
        evaluator.score_docs.return_value = [0.30]  # ties local -> external is excluded
        external = _external(wiki=(["E1"], [{"url": "m3"}]))
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)
        crag.handle_ambiguous = mock.Mock(
            return_value=(["A1"], [{"url": "m2"}], "ambiguous")
        )

        docs, metas, decision = crag.retrieve_with_decision("q", keyword="kw")

        self.assertEqual(docs, ["L1", "A1"])
        self.assertEqual(metas, [{"url": "m1"}, {"url": "m2"}])
        self.assertEqual(decision, "ambiguous_external")
        evaluator.score_docs.assert_called_once_with("q", ["E1"])

    def test_unresolved_ambiguous_tolerates_metas_none_from_the_resolver(self):
        retriever = _RecordingRetriever([(["d1"], [{"url": "u1"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("ambiguous", ["L1"], [{"url": "m1"}], 0.30)
        evaluator.knowledge_refinement.return_value = "R1"
        evaluator.score_docs.return_value = [0.10]
        external = _external(wiki=(["E1"], [{"url": "m3"}]))
        crag = _make_crag(retriever=retriever, evaluator=evaluator, external=external)
        crag.handle_ambiguous = mock.Mock(return_value=(["A1"], None, "incorrect"))

        docs, metas, decision = crag.retrieve_with_decision("q", keyword="kw")

        self.assertEqual(docs, ["L1", "A1"])
        self.assertEqual(metas, [{"url": "m1"}])
        self.assertEqual(decision, "ambiguous_external")


# ──────────────────────────────────────────────────────────────────────
# handle_ambiguous
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class HandleAmbiguousTests(unittest.TestCase):
    def test_seen_urls_none_still_builds_the_exclusion_filter(self):
        # Live bug: a None seen_urls used to reach _track_urls().add() and raise.
        # Swallowing the error would also "not raise", so the filter contents
        # are the real assertion here.
        retriever = _RecordingRetriever([(["d1"], [{"url": "u2"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("correct", ["d1"], [{"url": "u2"}], 0.88)
        crag = _make_crag(
            retriever=retriever,
            evaluator=evaluator,
            expander=_expander(keyword="kw", reformulated="rq"),
        )

        docs, metas, decision = crag.handle_ambiguous("q", [{"url": "u1"}], None)

        self.assertEqual(
            (docs, metas, decision), (["d1"], [{"url": "u2"}], "correct")
        )
        self.assertEqual(
            retriever.calls[0]["where_filter"], {"url": {"$nin": ["u1"]}}
        )
        self.assertEqual(retriever.calls[0]["query"], "rq kw")
        self.assertEqual(retriever.calls[0]["keyword"], "kw")

    def test_no_docs_after_filtering_escalates_to_external(self):
        retriever = _RecordingRetriever([([], [])])
        evaluator = _evaluator()
        crag = _make_crag(retriever=retriever, evaluator=evaluator)

        docs, metas, decision = crag.handle_ambiguous("q", [{"url": "u1"}], set())

        self.assertEqual((docs, metas, decision), ([], [], "incorrect"))
        evaluator.evaluate.assert_not_called()

    def test_reformulation_failure_falls_back_to_the_original_query(self):
        retriever = _RecordingRetriever([(["d1"], [{}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("correct", ["d1"], [{}], 0.70)
        crag = _make_crag(
            retriever=retriever,
            evaluator=evaluator,
            expander=_expander(keyword="kw", reformulated=RuntimeError("no api key")),
        )

        crag.handle_ambiguous("q", [], set())

        self.assertEqual(retriever.calls[0]["query"], "q kw")

    def test_alternative_query_can_resolve_the_ambiguity(self):
        retriever = _RecordingRetriever(
            [(["d1"], [{"url": "u2"}]), (["alt1"], [{"url": "u3"}])]
        )
        evaluator = _evaluator()
        evaluator.evaluate.side_effect = [
            ("ambiguous", ["d1"], [{"url": "u2"}], 0.45),
            ("correct", ["alt1"], [{"url": "u3"}], 0.81),
        ]
        expander = _expander(
            keyword="kw", reformulated="rq", alternatives=["alt query"]
        )
        crag = _make_crag(
            retriever=retriever, evaluator=evaluator, expander=expander
        )

        docs, metas, decision = crag.handle_ambiguous("q", [], set())

        self.assertEqual(
            (docs, metas, decision), (["alt1"], [{"url": "u3"}], "correct")
        )
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(retriever.calls[1]["query"], "alt query")
        expander.expand_multiple.assert_called_once_with("q", n=3)

    def test_alternative_query_returning_nothing_is_skipped_not_graded(self):
        retriever = _RecordingRetriever(
            [
                (["d1"], [{"url": "u2"}]),
                ([], []),
                (["alt2"], [{"url": "u4"}]),
            ]
        )
        evaluator = _evaluator()
        evaluator.evaluate.side_effect = [
            ("ambiguous", ["d1"], [{"url": "u2"}], 0.40),
            ("correct", ["alt2"], [{"url": "u4"}], 0.90),
        ]
        crag = _make_crag(
            retriever=retriever,
            evaluator=evaluator,
            expander=_expander(
                keyword="kw", reformulated="rq", alternatives=["a1", "a2"]
            ),
        )

        docs, metas, decision = crag.handle_ambiguous("q", [], set())

        self.assertEqual(
            (docs, metas, decision), (["alt2"], [{"url": "u4"}], "correct")
        )
        self.assertEqual(len(retriever.calls), 3)
        self.assertEqual(evaluator.evaluate.call_count, 2)

    def test_expansion_failure_skips_the_retries_and_keeps_the_first_decision(self):
        retriever = _RecordingRetriever([(["d1"], [{"url": "u2"}])])
        evaluator = _evaluator()
        evaluator.evaluate.return_value = ("ambiguous", ["d1"], [{"url": "u2"}], 0.45)
        crag = _make_crag(
            retriever=retriever,
            evaluator=evaluator,
            expander=_expander(
                keyword="kw",
                reformulated="rq",
                alternatives=RuntimeError("no api key"),
            ),
        )

        docs, metas, decision = crag.handle_ambiguous("q", [], set())

        self.assertEqual(
            (docs, metas, decision), (["d1"], [{"url": "u2"}], "ambiguous")
        )
        self.assertEqual(len(retriever.calls), 1)


# ──────────────────────────────────────────────────────────────────────
# _safe_external_search
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class SafeExternalSearchTests(unittest.TestCase):
    def test_wikipedia_failure_still_returns_the_news_chunks(self):
        external = _external(
            wiki=RuntimeError("wikipedia unreachable"),
            news=(["N1"], [{"url": "n1"}]),
        )
        crag = _make_crag(external=external)

        docs, metas = crag._safe_external_search("q")

        self.assertEqual(docs, ["N1"])
        self.assertEqual(metas, [{"url": "n1"}])
        external.search_wikipedia.assert_called_once_with("q")

    def test_news_failure_still_returns_the_wikipedia_chunks(self):
        external = _external(
            wiki=(["W1"], [{"url": "w1"}]),
            news=RuntimeError("newsapi 429"),
        )
        crag = _make_crag(external=external)

        docs, metas = crag._safe_external_search("q")

        self.assertEqual(docs, ["W1"])
        self.assertEqual(metas, [{"url": "w1"}])
        external.search_news.assert_called_once_with("q")

    def test_both_sources_failing_returns_empty_lists(self):
        external = _external(
            wiki=RuntimeError("wikipedia unreachable"),
            news=RuntimeError("newsapi down"),
        )
        crag = _make_crag(external=external)

        docs, metas = crag._safe_external_search("q")

        self.assertEqual((docs, metas), ([], []))
        external.search_wikipedia.assert_called_once_with("q")
        external.search_news.assert_called_once_with("q")

    def test_successful_sources_are_concatenated_wikipedia_first(self):
        external = _external(
            wiki=(["W1"], [{"url": "w1"}]), news=(["N1"], [{"url": "n1"}])
        )
        crag = _make_crag(external=external)

        docs, metas = crag._safe_external_search("q")

        self.assertEqual(docs, ["W1", "N1"])
        self.assertEqual(metas, [{"url": "w1"}, {"url": "n1"}])


# ──────────────────────────────────────────────────────────────────────
# _fetch_fully_external
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class FetchFullyExternalTests(unittest.TestCase):
    def test_prefers_local_when_the_scores_tie(self):
        evaluator = _evaluator()
        evaluator.knowledge_refinement.return_value = "R1"
        evaluator.score_docs.return_value = [0.50]
        external = _external(wiki=(["E1"], [{"url": "w1"}]))
        crag = _make_crag(evaluator=evaluator, external=external)

        docs, metas, decision = crag._fetch_fully_external(
            "q", local_docs=["L1"], local_score=0.50
        )

        self.assertEqual(docs, ["L1"])
        self.assertEqual(metas, [])
        self.assertEqual(decision, "incorrect_local_fallback")
        # Guards against a swallowed scoring error faking this branch.
        evaluator.score_docs.assert_called_once_with("q", ["R1"])

    def test_returns_external_when_it_outscores_local(self):
        evaluator = _evaluator()
        evaluator.knowledge_refinement.return_value = "R1"
        evaluator.score_docs.return_value = [0.90]
        external = _external(wiki=(["E1"], [{"url": "w1"}]))
        crag = _make_crag(evaluator=evaluator, external=external)

        docs, metas, decision = crag._fetch_fully_external(
            "q", local_docs=["L1"], local_score=0.20
        )

        self.assertEqual(docs, ["R1"])
        self.assertEqual(metas, [{"url": "w1"}])
        self.assertEqual(decision, "incorrect")
        evaluator.score_docs.assert_called_once_with("q", ["R1"])

    def test_empty_external_falls_back_to_the_local_docs(self):
        evaluator = _evaluator()
        external = _external()  # both sources return nothing
        crag = _make_crag(evaluator=evaluator, external=external)

        docs, metas, decision = crag._fetch_fully_external(
            "q", local_docs=["L1"], local_score=0.10
        )

        self.assertEqual((docs, metas, decision), (["L1"], [], "incorrect_local_fallback"))
        evaluator.score_docs.assert_not_called()
        evaluator.knowledge_refinement.assert_not_called()

    def test_nothing_local_and_nothing_external_is_reported_as_empty(self):
        crag = _make_crag(external=_external())

        self.assertEqual(
            crag._fetch_fully_external("q", local_docs=[]), ([], [], "incorrect_empty")
        )

    def test_scoring_failure_does_not_lose_the_external_chunks(self):
        evaluator = _evaluator()
        evaluator.knowledge_refinement.return_value = "R1"
        evaluator.score_docs.side_effect = RuntimeError("e5 model unavailable")
        external = _external(wiki=(["E1"], [{"url": "w1"}]))
        crag = _make_crag(evaluator=evaluator, external=external)

        docs, metas, decision = crag._fetch_fully_external("q")

        self.assertEqual(docs, ["R1"])
        self.assertEqual(metas, [{"url": "w1"}])
        self.assertEqual(decision, "incorrect")

    def test_refinement_failure_keeps_the_original_chunk(self):
        evaluator = _evaluator()
        evaluator.knowledge_refinement.side_effect = [RuntimeError("boom"), "R2"]
        evaluator.score_docs.return_value = [0.30, 0.40]
        external = _external(
            wiki=(["E1", "E2"], [{"url": "w1"}, {"url": "w2"}])
        )
        crag = _make_crag(evaluator=evaluator, external=external)

        docs, metas, decision = crag._fetch_fully_external("q")

        self.assertEqual(docs, ["E1", "R2"])
        self.assertEqual(metas, [{"url": "w1"}, {"url": "w2"}])
        self.assertEqual(decision, "incorrect")
        evaluator.score_docs.assert_called_once_with("q", ["E1", "R2"])

    def test_unrefinable_external_chunks_fall_back_to_the_raw_chunks(self):
        evaluator = _evaluator()
        evaluator.knowledge_refinement.return_value = ""
        evaluator.score_docs.return_value = [0.30]
        external = _external(news=(["N1"], [{"url": "n1"}]))
        crag = _make_crag(evaluator=evaluator, external=external)

        docs, metas, decision = crag._fetch_fully_external("q")

        self.assertEqual(docs, ["N1"])
        self.assertEqual(metas, [{"url": "n1"}])
        self.assertEqual(decision, "incorrect")
        evaluator.score_docs.assert_called_once_with("q", ["N1"])


# ──────────────────────────────────────────────────────────────────────
# retrieve()
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class RetrieveDelegationTests(unittest.TestCase):
    def test_retrieve_drops_the_decision_and_returns_a_two_tuple(self):
        crag = _make_crag()
        crag.retrieve_with_decision = mock.Mock(
            return_value=(["d1"], [{"url": "u1"}], "correct")
        )
        emitter = _RecordingEmitter()
        crag.set_emitter(emitter)

        result = crag.retrieve("q", keyword="kw", where_filter={"a": 1}, seen_urls={"u1"})

        self.assertEqual(result, (["d1"], [{"url": "u1"}]))
        crag.retrieve_with_decision.assert_called_once_with(
            "q", "kw", {"a": 1}, {"u1"}
        )
        self.assertTrue(
            any("Final results retrieved: 1" in m for m in emitter.messages()),
            emitter.messages(),
        )

    def test_retrieve_forwards_omitted_arguments_as_none(self):
        crag = _make_crag()
        crag.retrieve_with_decision = mock.Mock(return_value=([], [], "incorrect_empty"))

        self.assertEqual(crag.retrieve("q"), ([], []))
        crag.retrieve_with_decision.assert_called_once_with("q", None, None, None)


if __name__ == "__main__":
    unittest.main()


# ──────────────────────────────────────────────────────────────────────
# per-request switches
# ──────────────────────────────────────────────────────────────────────

@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class ExternalSearchToggleTests(unittest.TestCase):
    """Each external source, and the escalation as a whole, is switchable.

    Returning nothing is already a supported outcome — the callers fall back to
    the local chunks — so switching these off narrows the answer to the indexed
    corpus rather than breaking anything.
    """

    def _crag(self):
        external = _external(wiki=(["W1"], [{"url": "w1"}]), news=(["N1"], [{"url": "n1"}]))
        return _make_crag(external=external), external

    def test_both_sources_run_by_default(self):
        crag, external = self._crag()

        docs, metas = crag._safe_external_search("q")

        external.search_wikipedia.assert_called_once_with("q")
        external.search_news.assert_called_once_with("q")
        self.assertEqual(docs, ["W1", "N1"])
        self.assertEqual(metas, [{"url": "w1"}, {"url": "n1"}])

    def test_switching_external_search_off_calls_neither_source(self):
        crag, external = self._crag()

        with _with_params(use_external_search=False):
            self.assertEqual(crag._safe_external_search("q"), ([], []))

        external.search_wikipedia.assert_not_called()
        external.search_news.assert_not_called()

    def test_switching_external_search_off_says_so_on_the_status_channel(self):
        crag, _ = self._crag()
        emitter = _RecordingEmitter()
        crag.set_emitter(emitter)

        with _with_params(use_external_search=False):
            crag._safe_external_search("q")

        self.assertTrue(
            any("switched off" in message for message in emitter.messages()),
            emitter.messages(),
        )

    def test_wikipedia_can_be_switched_off_on_its_own(self):
        crag, external = self._crag()

        with _with_params(use_wikipedia=False):
            docs, metas = crag._safe_external_search("q")

        external.search_wikipedia.assert_not_called()
        external.search_news.assert_called_once_with("q")
        self.assertEqual(docs, ["N1"])
        self.assertEqual(metas, [{"url": "n1"}])

    def test_news_can_be_switched_off_on_its_own(self):
        crag, external = self._crag()

        with _with_params(use_news=False):
            docs, metas = crag._safe_external_search("q")

        external.search_news.assert_not_called()
        external.search_wikipedia.assert_called_once_with("q")
        self.assertEqual(docs, ["W1"])

    def test_switching_both_sources_off_matches_switching_the_escalation_off(self):
        crag, external = self._crag()

        with _with_params(use_wikipedia=False, use_news=False):
            self.assertEqual(crag._safe_external_search("q"), ([], []))

        external.search_wikipedia.assert_not_called()
        external.search_news.assert_not_called()


@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class QueryExpansionToggleTests(unittest.TestCase):
    """Query rewriting is three LLM calls, and the ambiguous-resolution retries
    are built on them — switching it off removes both."""

    def test_expansion_runs_by_default(self):
        expander = _expander(keyword="kw", reformulated="reformulated", alternatives=["a", "b"])
        crag = _make_crag(expander=expander)

        self.assertEqual(crag._safe_keyword("q"), "kw")
        self.assertEqual(crag._safe_reformulate("q"), "reformulated")
        self.assertEqual(crag._safe_expand_multiple("q"), ["a", "b"])

    def test_switched_off_the_raw_query_is_used_as_its_own_keyword(self):
        expander = _expander()
        crag = _make_crag(expander=expander)

        with _with_params(use_query_expansion=False):
            self.assertEqual(crag._safe_keyword("who wrote it?"), "who wrote it?")
            self.assertEqual(crag._safe_reformulate("who wrote it?"), "who wrote it?")

        expander.to_keyword.assert_not_called()
        expander.reformulate.assert_not_called()

    def test_switched_off_there_are_no_alternative_queries_to_retry_with(self):
        expander = _expander(alternatives=["a", "b", "c"])
        crag = _make_crag(expander=expander)

        with _with_params(use_query_expansion=False):
            self.assertEqual(crag._safe_expand_multiple("q"), [])

        expander.expand_multiple.assert_not_called()

    def test_the_number_of_alternatives_is_configurable(self):
        expander = _expander(alternatives=["a"])
        crag = _make_crag(expander=expander)

        with _with_params(expansion_queries=5):
            crag._safe_expand_multiple("q")

        expander.expand_multiple.assert_called_once_with("q", n=5)

    def test_the_default_alternative_count_is_unchanged(self):
        expander = _expander(alternatives=["a"])
        crag = _make_crag(expander=expander)

        crag._safe_expand_multiple("q")

        expander.expand_multiple.assert_called_once_with("q", n=3)


@unittest.skipIf(CorrectiveRAG is None, f"corrective_rag unavailable: {IMPORT_ERROR}")
class TopKDelegationTests(unittest.TestCase):
    """top_k has to be read live off the wrapped retriever.

    MultiHopRetriever truncates to whatever this reports, so a copy taken at
    construction would leave the layers working to different budgets — the outer
    one discarding chunks the inner one was asked to return.
    """

    def test_it_follows_the_retriever_rather_than_a_construction_time_copy(self):
        retriever = _RecordingRetriever()
        crag = _make_crag(retriever=retriever)
        self.assertEqual(crag.top_k, 4)

        retriever.top_k = 9
        self.assertEqual(crag.top_k, 9)


@unittest.skipIf(ExternalSearcher is None, f"external_search unavailable: {IMPORT_ERROR}")
class ExternalTopKTests(unittest.TestCase):
    """How many web passages a request keeps.

    Constructed bare: a real ExternalSearcher builds a QueryExpander and an LLM
    client, and none of that is what this is about. One searcher serves every
    concurrent query, so this has to resolve per call rather than be held.
    """

    def _searcher(self, configured=5):
        searcher = object.__new__(ExternalSearcher)
        searcher._configured_top_k = configured
        return searcher

    def test_the_request_value_wins(self):
        with _with_params(external_top_k=8):
            self.assertEqual(self._searcher().top_k, 8)

    def test_an_unset_value_leaves_the_configured_one(self):
        with _with_params():
            self.assertEqual(self._searcher(configured=3).top_k, 3)

    def test_it_does_not_outlive_the_request(self):
        searcher = self._searcher(configured=3)

        with _with_params(external_top_k=8):
            pass

        self.assertEqual(searcher.top_k, 3)

    def test_a_searcher_built_before_the_field_existed_still_answers(self):
        # An instance from an older pickle/partial construction has no
        # _configured_top_k; the property must not raise on it.
        searcher = object.__new__(ExternalSearcher)

        with _with_params():
            self.assertEqual(searcher.top_k, 5)
