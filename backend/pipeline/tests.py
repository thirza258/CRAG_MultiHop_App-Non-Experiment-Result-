"""Tests for AppRAGPipeline, the stage orchestrator in pipeline/app_pipeline.py.

Covered here:

* ``__init__`` composition — MultiHopRetriever(CorrectiveRAG(DenseRAG)) and the
  matching sparse chain, with every heavy collaborator patched out.
* ``_generate_answer`` — chunk normalisation (plain strings, dicts keyed
  text/content/page_content, unknown types skipped) and the honest no-context
  answer when there is no usable text to prompt with.
* ``_resolve_collection`` / ``_resolve_collection_for`` — user collection when it
  holds chunks, dataset collection when it is empty or the record is missing,
  the per-query corpus choice, and what happens when ChromaDB itself fails.
* ``evaluate`` — scores out of the RAGAs frame, and (None, None) instead of an
  exception when the judge is unavailable.
* ``_select_chain`` through ``_run_core`` — which retriever chain each stage
  combination actually queries, and that a switched-off retriever is not a
  degradation.
* ``_run_core`` degradation matrix — a healthy run, one retriever down, both
  retrievers down, reranker down, generation down, evaluation down, status
  channel down. The point of every case is that the returned dict still carries
  answer / source / context / evaluation / degraded and that a broken stage
  costs only that stage.
* ``_build_emitter`` / ``_safe_emit`` / ``_emit_degradations`` — the status
  channel contract, including the Redis channel name the websocket consumer
  subscribes to.

Nothing here touches the network: no ChromaDB, no Redis, no PostgreSQL, no
OPENROUTER_API_KEY and no model loads.
"""

import json
import unittest
from unittest import mock

try:
    from pipeline import app_pipeline
    from pipeline.app_pipeline import AppRAGPipeline
    IMPORT_ERROR = ""
except Exception as exc:  # needs Django settings + chromadb/torch/redis
    app_pipeline = None
    AppRAGPipeline = None
    IMPORT_ERROR = str(exc)


class MissingUserCollection(Exception):
    """Stand-in for UserCollection.DoesNotExist.

    The pipeline catches whatever ``UserCollection.DoesNotExist`` resolves to, so
    patching the model with a stub carrying a real exception class keeps these
    tests off the database entirely.
    """


# Mirrors the shape rag/rag_service.py builds, trimmed to the keys the pipeline
# reads outside __init__.
_CONFIG = {
    "collection_name": "dataset_collection",
    "hybrid_config": {"retrieval_top_k": 4, "top_k": 4},
    "multi_hop_config": {"max_hops": 3, "top_k": 4},
    "evaluation_llm_model": "judge-model",
    "evaluation_embedding_model": "judge-embedding-model",
}


def _bare_pipeline(config=None):
    """An AppRAGPipeline whose __init__ never ran.

    Constructing for real means DenseRAG (ValueError with no OPENROUTER_API_KEY),
    a chromadb.HttpClient and the jina reranker snapshot. Every collaborator
    _run_core touches is injected as a mock instead, pre-set to its healthy
    behaviour so each test only has to break the one stage it is about.
    """
    pipeline = object.__new__(AppRAGPipeline)
    pipeline.config = dict(_CONFIG if config is None else config)

    pipeline.dense_rag = mock.Mock()
    pipeline.sparse_rag = mock.Mock()
    pipeline.dense_corrective_rag = mock.Mock()
    pipeline.sparse_corrective_rag = mock.Mock()
    pipeline.dense_multi_hop = mock.Mock()
    pipeline.sparse_multi_hop = mock.Mock()
    pipeline.hybrid_rag = mock.Mock()
    pipeline.llm_client = mock.Mock()
    pipeline.dataset_collection = mock.Mock()

    # An unconfigured Mock return value would fail the tuple unpack / the str
    # answer assertions and surface as a degradation, so spell out the happy path.
    # Every object _select_chain can pick has to be ready to be the chain.
    for chain in (
        pipeline.dense_rag,
        pipeline.sparse_rag,
        pipeline.dense_corrective_rag,
        pipeline.sparse_corrective_rag,
        pipeline.dense_multi_hop,
        pipeline.sparse_multi_hop,
    ):
        chain.retrieve.return_value = ([], [])
    pipeline.hybrid_rag.retrieve_from_precomputed.return_value = ([], [], "ok")
    pipeline.llm_client._call_api.return_value = "generated answer"

    pipeline._build_emitter = mock.Mock(return_value=app_pipeline.NULL_EMITTER)
    pipeline.evaluate = mock.Mock(return_value=(0.91, 0.88))

    _install_resolver(pipeline, "bob_collection", "user_collection")
    return pipeline


def _install_resolver(pipeline, collection_name="bob_collection", source="user_collection", error=None):
    """Stub collection resolution for the _run_core tests.

    Both resolver names are stubbed with an arity-agnostic callable so these
    tests stay about the stages that run *after* resolution; the resolvers
    themselves are covered by ResolveCollectionTests.
    """
    def resolver(*_args, **_kwargs):
        if error is not None:
            raise error
        return collection_name, source

    pipeline._resolve_collection_for = resolver
    pipeline._resolve_collection = resolver
    return pipeline


def _fake_user_collection(record=None, missing=False):
    fake = mock.Mock()
    fake.DoesNotExist = MissingUserCollection
    if missing:
        fake.objects.get.side_effect = MissingUserCollection()
    else:
        fake.objects.get.return_value = record
    return fake


class _FakeScores:
    """Stands in for the RAGAs result DataFrame so pandas is not needed.

    ``result["answer_relevancy"].iloc[0]`` is the only access pattern evaluate()
    uses, and a dict indexed by 0 satisfies the ``.iloc[0]`` half of it.
    """

    def __init__(self, values=None, empty=False):
        self._values = values or {}
        self.empty = empty

    def __getitem__(self, key):
        return mock.Mock(iloc={0: self._values[key]})


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class ConstructionTests(unittest.TestCase):
    """The retriever composition is the pipeline's identity: MultiHop wraps
    Corrective wraps the base retriever, once per side."""

    def test_wraps_each_base_retriever_in_corrective_then_multi_hop(self):
        config = {
            "llm_model": "answer-model",
            "collection_name": "ragreader_collection",
            "dense_config": {"top_k": 4},
            "sparse_config": {"top_k": 4},
            "hybrid_config": {"rerank_only": True},
            "crag_config": {"top_k": 4},
            "multi_hop_config": {"max_hops": 3},
        }

        with mock.patch.object(app_pipeline, "DenseRAG") as dense_cls, \
                mock.patch.object(app_pipeline, "SparseRAG") as sparse_cls, \
                mock.patch.object(app_pipeline, "CorrectiveRAG") as crag_cls, \
                mock.patch.object(app_pipeline, "MultiHopRetriever") as hop_cls, \
                mock.patch.object(app_pipeline, "HybridRAG") as hybrid_cls, \
                mock.patch.object(app_pipeline, "OpenRouterLLM") as llm_cls, \
                mock.patch.object(app_pipeline, "get_chroma_client") as chroma, \
                mock.patch.object(app_pipeline, "DocumentChunker"), \
                mock.patch.object(app_pipeline, "DataLoader"):
            # Record what each wrapper was handed so the nesting can be asserted.
            crag_cls.side_effect = lambda retriever, cfg: mock.Mock(inner=retriever)
            hop_cls.side_effect = lambda retriever, cfg: mock.Mock(inner=retriever)

            pipeline = AppRAGPipeline(config)

        dense_cls.assert_called_once_with(config["dense_config"])
        sparse_cls.assert_called_once_with(config["sparse_config"])
        hybrid_cls.assert_called_once_with(config["hybrid_config"])
        llm_cls.assert_called_once_with(model="answer-model")
        chroma.assert_called_once_with(collection_name="ragreader_collection")

        self.assertIs(pipeline.dense_multi_hop.inner, pipeline.dense_corrective_rag)
        self.assertIs(pipeline.dense_corrective_rag.inner, pipeline.dense_rag)
        self.assertIs(pipeline.sparse_multi_hop.inner, pipeline.sparse_corrective_rag)
        self.assertIs(pipeline.sparse_corrective_rag.inner, pipeline.sparse_rag)

        self.assertEqual(
            [call.args[1] for call in crag_cls.call_args_list],
            [config["crag_config"], config["crag_config"]],
        )
        self.assertEqual(
            [call.args[1] for call in hop_cls.call_args_list],
            [config["multi_hop_config"], config["multi_hop_config"]],
        )

    def test_defaults_the_dataset_collection_name_when_config_omits_it(self):
        config = {
            "llm_model": "answer-model",
            "dense_config": {},
            "sparse_config": {},
            "hybrid_config": {},
            "crag_config": {},
            "multi_hop_config": {},
        }

        with mock.patch.object(app_pipeline, "DenseRAG"), \
                mock.patch.object(app_pipeline, "SparseRAG"), \
                mock.patch.object(app_pipeline, "CorrectiveRAG"), \
                mock.patch.object(app_pipeline, "MultiHopRetriever"), \
                mock.patch.object(app_pipeline, "HybridRAG"), \
                mock.patch.object(app_pipeline, "OpenRouterLLM"), \
                mock.patch.object(app_pipeline, "get_chroma_client") as chroma, \
                mock.patch.object(app_pipeline, "DocumentChunker"), \
                mock.patch.object(app_pipeline, "DataLoader"):
            AppRAGPipeline(config)

        chroma.assert_called_once_with(collection_name="dataset_collection")


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class GenerateAnswerTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _bare_pipeline()
        self.call_api = self.pipeline.llm_client._call_api

    def _prompt(self):
        return self.call_api.call_args.args[0]

    def test_joins_plain_string_chunks_into_the_prompt(self):
        answer = self.pipeline._generate_answer("who wrote it?", ["alpha", "beta"])

        self.assertEqual(answer, "generated answer")
        prompt = self._prompt()
        self.assertIn("alpha\n\nbeta", prompt)
        self.assertIn("Question: who wrote it?", prompt)

    def test_reads_text_content_and_page_content_keys(self):
        self.pipeline._generate_answer(
            "q",
            [
                {"text": "from-text"},
                {"content": "from-content"},
                {"page_content": "from-page-content"},
            ],
        )

        prompt = self._prompt()
        self.assertIn("from-text", prompt)
        self.assertIn("from-content", prompt)
        self.assertIn("from-page-content", prompt)

    def test_prefers_the_text_key_over_the_other_spellings(self):
        self.pipeline._generate_answer("q", [{"text": "winner", "content": "loser"}])

        self.assertIn("winner", self._prompt())
        self.assertNotIn("loser", self._prompt())

    def test_skips_chunks_of_an_unexpected_type(self):
        self.pipeline._generate_answer("q", ["keep-me", 42, None, ["nested"]])

        prompt = self._prompt()
        self.assertIn("keep-me", prompt)
        self.assertNotIn("42", prompt)
        self.assertNotIn("nested", prompt)
        self.call_api.assert_called_once()

    def test_empty_chunk_list_returns_the_no_context_answer_without_calling_the_llm(self):
        answer = self.pipeline._generate_answer("q", [])

        self.assertEqual(answer, app_pipeline._NO_CONTEXT_ANSWER)
        self.call_api.assert_not_called()

    def test_blank_chunk_text_returns_the_no_context_answer_without_calling_the_llm(self):
        answer = self.pipeline._generate_answer("q", [{"summary": "no text key"}, "   "])

        self.assertEqual(answer, app_pipeline._NO_CONTEXT_ANSWER)
        self.call_api.assert_not_called()


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class ResolveCollectionTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _bare_pipeline()
        # The resolvers are what is under test here, so drop the stubs the
        # _run_core fixture installs.
        del self.pipeline._resolve_collection
        del self.pipeline._resolve_collection_for

    def test_uses_the_user_collection_when_it_holds_chunks(self):
        record = mock.Mock(collection_name="user_bob_collection")
        chroma_collection = mock.Mock()
        chroma_collection.count.return_value = 3

        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(record)), \
                mock.patch.object(app_pipeline, "get_chroma_client", return_value=chroma_collection) as chroma:
            self.assertEqual(
                self.pipeline._resolve_collection("bob"),
                ("user_bob_collection", "user_collection"),
            )

        chroma.assert_called_once_with(collection_name="user_bob_collection")

    def test_falls_back_to_the_dataset_collection_when_the_user_collection_is_empty(self):
        record = mock.Mock(collection_name="user_bob_collection")
        chroma_collection = mock.Mock()
        chroma_collection.count.return_value = 0

        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(record)), \
                mock.patch.object(app_pipeline, "get_chroma_client", return_value=chroma_collection):
            self.assertEqual(
                self.pipeline._resolve_collection("bob"),
                ("dataset_collection", "dataset_collection"),
            )

    def test_falls_back_when_the_user_has_no_collection_record(self):
        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(missing=True)), \
                mock.patch.object(app_pipeline, "get_chroma_client") as chroma:
            self.assertEqual(
                self.pipeline._resolve_collection("bob"),
                ("dataset_collection", "dataset_collection"),
            )

        chroma.assert_not_called()

    def test_dataset_collection_name_comes_from_the_pipeline_config(self):
        self.pipeline.config = {"collection_name": "ragreader_collection"}

        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(missing=True)), \
                mock.patch.object(app_pipeline, "get_chroma_client"):
            self.assertEqual(
                self.pipeline._resolve_collection("bob"),
                ("ragreader_collection", "dataset_collection"),
            )

    def test_chromadb_failure_propagates_to_the_caller(self):
        record = mock.Mock(collection_name="user_bob_collection")

        # Only UserCollection.DoesNotExist is handled in here; a dead ChromaDB is
        # caught one level up, by _run_core's collection_resolve guard.
        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(record)), \
                mock.patch.object(app_pipeline, "get_chroma_client", side_effect=RuntimeError("chroma down")):
            with self.assertRaises(RuntimeError):
                self.pipeline._resolve_collection("bob")

    def test_corpus_base_forces_the_shared_collection_without_a_db_lookup(self):
        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(missing=True)) as user_col, \
                mock.patch.object(app_pipeline, "get_chroma_client") as chroma:
            self.assertEqual(
                self.pipeline._resolve_collection_for("bob", "base"),
                ("dataset_collection", "dataset_collection"),
            )

        user_col.objects.get.assert_not_called()
        chroma.assert_not_called()

    def test_corpus_user_uses_the_user_collection_without_counting_it(self):
        record = mock.Mock(collection_name="user_bob_collection")

        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(record)), \
                mock.patch.object(app_pipeline, "get_chroma_client") as chroma:
            self.assertEqual(
                self.pipeline._resolve_collection_for("bob", "user"),
                ("user_bob_collection", "user_collection"),
            )

        chroma.assert_not_called()

    def test_corpus_user_without_a_collection_resolves_to_no_collection(self):
        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(missing=True)):
            self.assertEqual(
                self.pipeline._resolve_collection_for("bob", "user"),
                (None, "user_collection_missing"),
            )

    def test_corpus_auto_delegates_to_the_historic_resolution(self):
        self.pipeline._resolve_collection = mock.Mock(return_value=("x", "user_collection"))

        self.assertEqual(
            self.pipeline._resolve_collection_for("bob", "auto"), ("x", "user_collection")
        )
        self.pipeline._resolve_collection.assert_called_once_with("bob")


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _bare_pipeline()
        del self.pipeline.evaluate  # exercise the real method, not the fixture stub

    def _patched_judge(self, judge_result=None, judge_error=None):
        return (
            mock.patch.object(app_pipeline, "convert_data_response_and_dataset_to_dataset", return_value="dataset"),
            mock.patch.object(
                app_pipeline,
                "ragas_llm_as_a_judge_generation_evaluation",
                return_value=judge_result,
                side_effect=judge_error,
            ),
            mock.patch.object(app_pipeline, "llm_langchain_wrapper", return_value="judge"),
            mock.patch.object(app_pipeline, "embeddings_langchain_wrapper", return_value="judge-embeddings"),
        )

    def test_returns_relevancy_and_faithfulness_from_the_judge_result(self):
        scores = _FakeScores({"answer_relevancy": 0.75, "faithfulness": 0.5})
        convert, judge, llm_wrapper, emb_wrapper = self._patched_judge(judge_result=scores)

        with convert as convert_mock, judge as judge_mock, llm_wrapper as llm_mock, emb_wrapper as emb_mock:
            self.assertEqual(self.pipeline.evaluate("q", ["chunk"], "answer"), (0.75, 0.5))

        convert_mock.assert_called_once_with(query="q", retrieved_chunks=["chunk"], generated_response="answer")
        judge_mock.assert_called_once_with(dataset="dataset", llm_judge="judge", judge_embeddings="judge-embeddings")
        llm_mock.assert_called_once_with("judge-model")
        emb_mock.assert_called_once_with("judge-embedding-model")

    def test_empty_judge_result_yields_no_scores(self):
        convert, judge, llm_wrapper, emb_wrapper = self._patched_judge(judge_result=_FakeScores(empty=True))

        with convert, judge, llm_wrapper, emb_wrapper:
            self.assertEqual(self.pipeline.evaluate("q", ["chunk"], "answer"), (None, None))

    def test_judge_failure_returns_no_scores_instead_of_raising(self):
        convert, judge, llm_wrapper, emb_wrapper = self._patched_judge(judge_error=RuntimeError("no api key"))

        with convert, judge, llm_wrapper, emb_wrapper:
            self.assertEqual(self.pipeline.evaluate("q", ["chunk"], "answer"), (None, None))

    def test_dataset_conversion_failure_returns_no_scores(self):
        with mock.patch.object(
            app_pipeline,
            "convert_data_response_and_dataset_to_dataset",
            side_effect=ValueError("bad chunks"),
        ):
            self.assertEqual(self.pipeline.evaluate("q", ["chunk"], "answer"), (None, None))


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class HelperTests(unittest.TestCase):
    def test_pipeline_top_k_prefers_the_pipeline_config_value(self):
        pipeline = _bare_pipeline({"top_k": 7, "hybrid_config": {"retrieval_top_k": 4}})

        self.assertEqual(pipeline._pipeline_top_k(), 7)

    def test_pipeline_top_k_falls_back_to_the_hybrid_retrieval_budget(self):
        pipeline = _bare_pipeline({"hybrid_config": {"retrieval_top_k": 6, "top_k": 4}})

        self.assertEqual(pipeline._pipeline_top_k(), 6)

    def test_pipeline_top_k_falls_back_to_the_hybrid_top_k(self):
        pipeline = _bare_pipeline({"hybrid_config": {"top_k": 3}})

        self.assertEqual(pipeline._pipeline_top_k(), 3)

    def test_pipeline_top_k_defaults_when_nothing_usable_is_configured(self):
        pipeline = _bare_pipeline(
            {"top_k": "not-a-number", "hybrid_config": {"retrieval_top_k": 0, "top_k": None}}
        )

        self.assertEqual(pipeline._pipeline_top_k(), 5)

    def test_pipeline_top_k_survives_a_non_dict_config(self):
        pipeline = _bare_pipeline()
        pipeline.config = "nonsense"

        self.assertEqual(pipeline._pipeline_top_k(), 5)

    def test_dataset_collection_name_survives_a_broken_config(self):
        pipeline = _bare_pipeline()

        pipeline.config = None
        self.assertEqual(pipeline._dataset_collection_name(), "dataset_collection")

        pipeline.config = {"collection_name": 42}
        self.assertEqual(pipeline._dataset_collection_name(), "dataset_collection")

        pipeline.config = {"collection_name": "ragreader_collection"}
        self.assertEqual(pipeline._dataset_collection_name(), "ragreader_collection")

    def test_merge_without_rerank_dedups_by_text_and_pairs_metas_per_side(self):
        pipeline = _bare_pipeline()

        chunks, metas = pipeline._merge_without_rerank(
            ["a", "b"],
            [{"side": "dense"}],          # short on purpose: "b" must not steal a sparse meta
            ["c", "a"],                   # "a" duplicates the dense hit
            [{"side": "sparse"}, {"side": "sparse-dupe"}],
        )

        self.assertEqual(chunks, ["a", "b", "c"])
        self.assertEqual(metas, [{"side": "dense"}, {}, {"side": "sparse"}])

    def test_merge_without_rerank_truncates_to_the_pipeline_top_k(self):
        pipeline = _bare_pipeline({"top_k": 2, "hybrid_config": {}})

        chunks, metas = pipeline._merge_without_rerank(
            ["a", "b", "c"], [{"i": 0}, {"i": 1}, {"i": 2}], ["d"], [{"i": 3}]
        )

        self.assertEqual(chunks, ["a", "b"])
        self.assertEqual(metas, [{"i": 0}, {"i": 1}])

    def test_no_context_result_has_the_full_shape_and_copies_the_degraded_list(self):
        pipeline = _bare_pipeline()
        degraded = ["dense_retrieval"]

        result = pipeline._no_context_result("dataset_collection", degraded)
        degraded.append("mutated-after-the-fact")

        self.assertEqual(result["answer"], app_pipeline._NO_CONTEXT_ANSWER)
        self.assertEqual(result["source"], "dataset_collection")
        self.assertEqual(result["context"], [])
        self.assertEqual(result["evaluation"], {"answer_relevancy": None, "faithfulness": None})
        self.assertEqual(result["degraded"], ["dense_retrieval"])


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class StatusChannelTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _bare_pipeline()

    def test_safe_emit_forwards_the_event_and_reports_success(self):
        emitter = mock.Mock()

        self.assertTrue(self.pipeline._safe_emit(emitter, "dense_retrieval", "msg", {"a": 1}))
        emitter.emit.assert_called_once_with("dense_retrieval", "msg", {"a": 1})

    def test_safe_emit_reports_failure_instead_of_raising(self):
        emitter = mock.Mock()
        emitter.emit.side_effect = RuntimeError("redis down")

        self.assertFalse(self.pipeline._safe_emit(emitter, "dense_retrieval", "msg"))

    def test_emit_degradations_stays_quiet_on_a_healthy_run(self):
        emitter = mock.Mock()

        self.pipeline._emit_degradations(emitter, [])

        emitter.emit.assert_not_called()

    def test_emit_degradations_publishes_the_stage_list_on_a_non_terminal_stage(self):
        emitter = mock.Mock()

        self.pipeline._emit_degradations(emitter, ["dense_retrieval", "evaluation"])

        stage, message, meta = emitter.emit.call_args.args
        # "result"/"error" would close the websocket stream early.
        self.assertEqual(stage, "degraded")
        self.assertEqual(meta, {"stages": ["dense_retrieval", "evaluation"]})
        self.assertIn("dense_retrieval", message)

    def test_build_emitter_returns_the_null_emitter_without_a_conversation_id(self):
        pipeline = object.__new__(AppRAGPipeline)

        for conversation_id in (None, 0, ""):
            with self.subTest(conversation_id=conversation_id):
                self.assertIs(pipeline._build_emitter(conversation_id), app_pipeline.NULL_EMITTER)

    def test_build_emitter_publishes_to_the_conversation_channel(self):
        pipeline = object.__new__(AppRAGPipeline)
        redis_client = mock.Mock()

        with mock.patch.object(app_pipeline.redis, "Redis", return_value=redis_client) as redis_cls:
            emitter = pipeline._build_emitter(7)

        self.assertIsNot(emitter, app_pipeline.NULL_EMITTER)
        self.assertIsInstance(emitter, app_pipeline.StatusEmitter)
        self.assertEqual(redis_cls.call_args.kwargs["db"], 0)

        emitter.emit("dense_retrieval", "searching", {"hop": 1})

        channel, payload = redis_client.publish.call_args.args
        # router/consumers.py subscribes to exactly this channel name.
        self.assertEqual(channel, "rag:status:7")
        self.assertEqual(
            json.loads(payload),
            {"stage": "dense_retrieval", "message": "searching", "meta": {"hop": 1}},
        )


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class RunCoreHealthyTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _bare_pipeline()
        self.pipeline.dense_multi_hop.retrieve.return_value = (
            ["dense-1", "dense-2"],
            [{"src": "dense-1"}, {"src": "dense-2"}],
        )
        self.pipeline.sparse_multi_hop.retrieve.return_value = (["sparse-1"], [{"src": "sparse-1"}])
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["dense-1", "sparse-1"],
            [{"src": "dense-1"}, {"src": "sparse-1"}],
            "ok",
        )

    def test_returns_answer_context_and_scores_with_no_degradations(self):
        result = self.pipeline._run_core("who wrote it?", "bob", 7)

        self.assertEqual(sorted(result), ["answer", "context", "degraded", "evaluation", "source"])
        self.assertEqual(result["degraded"], [])
        self.assertEqual(result["source"], "user_collection")
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual([entry["text"] for entry in result["context"]], ["dense-1", "sparse-1"])
        self.assertEqual(
            [entry["metadata"] for entry in result["context"]],
            [{"src": "dense-1"}, {"src": "sparse-1"}],
        )
        self.assertEqual(result["evaluation"], {"answer_relevancy": 0.91, "faithfulness": 0.88})

    def test_points_both_retrievers_at_the_resolved_collection(self):
        self.pipeline._run_core("q", "bob", 7)

        self.pipeline.dense_rag.set_collection.assert_called_once_with("bob_collection")
        self.pipeline.sparse_rag.set_collection.assert_called_once_with("bob_collection")
        self.pipeline.dense_multi_hop.set_emitter.assert_called_once()
        self.pipeline.sparse_multi_hop.set_emitter.assert_called_once()

    def test_hands_both_sides_to_the_reranker_unmerged(self):
        self.pipeline._run_core("q", "bob", 7)

        kwargs = self.pipeline.hybrid_rag.retrieve_from_precomputed.call_args.kwargs
        self.assertEqual(kwargs["query"], "q")
        self.assertEqual(kwargs["dense_chunks"], ["dense-1", "dense-2"])
        self.assertEqual(kwargs["sparse_chunks"], ["sparse-1"])
        self.assertEqual(kwargs["dense_metas"], [{"src": "dense-1"}, {"src": "dense-2"}])
        self.assertEqual(kwargs["sparse_metas"], [{"src": "sparse-1"}])

    def test_context_metadata_is_padded_when_the_reranker_returns_fewer_metas(self):
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["a", "b", "c"],
            [{"i": 0}, {"i": 1}],
            "ok",
        )

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual([entry["metadata"] for entry in result["context"]], [{"i": 0}, {"i": 1}, {}])

    def test_rerank_disabled_status_is_not_treated_as_a_degradation(self):
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["dense-1"], [{"src": "dense-1"}], "ok (rerank disabled)",
        )

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], [])
        self.assertEqual(result["answer"], "generated answer")

    def test_evaluate_returning_no_scores_is_not_treated_as_a_degradation(self):
        # (None, None) is the normal result with no OPENROUTER_API_KEY.
        self.pipeline.evaluate.return_value = (None, None)

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], [])
        self.assertEqual(result["evaluation"], {"answer_relevancy": None, "faithfulness": None})

    def test_run_forwards_its_arguments_to_run_core_in_the_expected_order(self):
        self.pipeline._run_core = mock.Mock(return_value={"answer": "x"})

        self.pipeline.run("bob", "who wrote it?", 7, {"corpus": "user"})

        self.pipeline._run_core.assert_called_once_with("who wrote it?", "bob", 7, {"corpus": "user"})

    def test_run_works_without_a_conversation_id_or_config(self):
        # The REST endpoint has no conversation to stream to at call time.
        self.pipeline._run_core = mock.Mock(return_value={"answer": "x"})

        self.pipeline.run("bob", "who wrote it?")

        self.pipeline._run_core.assert_called_once_with("who wrote it?", "bob", None, None)


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class RunCoreChainSelectionTests(unittest.TestCase):
    """Which chain each stage combination queries. Picking the wrong object here
    silently runs a different pipeline than the user asked for."""

    def setUp(self):
        self.pipeline = _bare_pipeline()
        for chain in (
            self.pipeline.dense_rag,
            self.pipeline.dense_corrective_rag,
            self.pipeline.dense_multi_hop,
        ):
            chain.retrieve.return_value = (["dense-1"], [{"src": "dense-1"}])
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["dense-1"], [{"src": "dense-1"}], "ok",
        )

    def _run(self, config):
        return self.pipeline._run_core("q", "bob", 7, config)

    def test_the_default_config_queries_the_multi_hop_chain(self):
        result = self._run(None)

        self.pipeline.dense_multi_hop.retrieve.assert_called_once_with("q")
        self.pipeline.dense_corrective_rag.retrieve.assert_not_called()
        self.pipeline.dense_rag.retrieve.assert_not_called()
        self.pipeline.dense_multi_hop.set_max_hops.assert_called_once_with(3)
        self.assertEqual(result["degraded"], [])

    def test_max_hops_is_applied_per_query(self):
        self._run({"max_hops": 2})

        self.pipeline.dense_multi_hop.set_max_hops.assert_called_once_with(2)
        self.pipeline.sparse_multi_hop.set_max_hops.assert_called_once_with(2)

    def test_a_chain_that_cannot_take_the_hop_ceiling_still_retrieves(self):
        self.pipeline.dense_multi_hop.set_max_hops.side_effect = RuntimeError("old retriever")

        result = self._run(None)

        self.pipeline.dense_multi_hop.retrieve.assert_called_once_with("q")
        self.assertEqual(result["degraded"], [])

    def test_multi_hop_off_queries_the_corrective_chain(self):
        result = self._run({"use_multi_hop": False})

        self.pipeline.dense_corrective_rag.retrieve.assert_called_once_with("q")
        self.pipeline.dense_multi_hop.retrieve.assert_not_called()
        self.pipeline.dense_corrective_rag.set_max_hops.assert_not_called()
        self.assertEqual(result["degraded"], [])

    def test_both_stages_off_queries_the_base_retriever(self):
        result = self._run({"use_multi_hop": False, "use_corrective": False})

        self.pipeline.dense_rag.retrieve.assert_called_once_with("q")
        self.pipeline.dense_corrective_rag.retrieve.assert_not_called()
        self.pipeline.dense_multi_hop.retrieve.assert_not_called()
        # DenseRAG has no set_max_hops; the hop ceiling must stay behind the
        # use_multi_hop guard rather than being aimed at a base retriever.
        self.pipeline.dense_rag.set_max_hops.assert_not_called()
        self.assertEqual(result["degraded"], [])

    def test_corrective_off_builds_a_multi_hop_chain_over_the_base_retriever(self):
        chain = mock.Mock()
        chain.retrieve.return_value = (["dense-1"], [{"src": "dense-1"}])

        with mock.patch.object(app_pipeline, "MultiHopRetriever", return_value=chain) as hop_cls:
            result = self._run({"use_corrective": False})
            # Cached per side, so a second query must not rebuild it.
            self._run({"use_corrective": False})

        self.assertEqual(
            [call.args for call in hop_cls.call_args_list],
            [
                (self.pipeline.dense_rag, _CONFIG["multi_hop_config"]),
                (self.pipeline.sparse_rag, _CONFIG["multi_hop_config"]),
            ],
        )
        self.assertTrue(chain.retrieve.called)
        self.pipeline.dense_corrective_rag.retrieve.assert_not_called()
        self.assertEqual(result["degraded"], [])

    def test_a_failed_bare_multi_hop_build_falls_back_to_the_corrective_chain(self):
        with mock.patch.object(app_pipeline, "MultiHopRetriever", side_effect=RuntimeError("no llm")):
            result = self._run({"use_corrective": False})

        self.pipeline.dense_corrective_rag.retrieve.assert_called_once_with("q")
        self.assertEqual(result["degraded"], [])
        self.assertEqual(result["answer"], "generated answer")

    def test_a_switched_off_sparse_retriever_is_not_a_degradation(self):
        result = self._run({"retrievers": "dense"})

        self.pipeline.sparse_rag.set_collection.assert_not_called()
        self.pipeline.sparse_multi_hop.retrieve.assert_not_called()
        self.assertEqual(result["degraded"], [])
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual(
            self.pipeline.hybrid_rag.retrieve_from_precomputed.call_args.kwargs["sparse_chunks"], []
        )

    def test_a_switched_off_dense_retriever_is_not_a_degradation(self):
        self.pipeline.sparse_multi_hop.retrieve.return_value = (["sparse-1"], [{"src": "sparse-1"}])
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["sparse-1"], [{"src": "sparse-1"}], "ok",
        )

        result = self._run({"retrievers": "sparse"})

        self.pipeline.dense_rag.set_collection.assert_not_called()
        self.pipeline.dense_multi_hop.retrieve.assert_not_called()
        self.assertEqual(result["degraded"], [])
        self.assertEqual([entry["text"] for entry in result["context"]], ["sparse-1"])

    def test_the_reranker_switch_is_passed_through_to_the_hybrid_stage(self):
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["dense-1"], [{"src": "dense-1"}], "ok (rerank disabled)",
        )

        result = self._run({"use_reranker": False})

        self.assertIs(
            self.pipeline.hybrid_rag.retrieve_from_precomputed.call_args.kwargs["use_reranker"], False
        )
        self.assertEqual(result["degraded"], [])


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class RunCoreDegradationTests(unittest.TestCase):
    """One broken dependency must cost exactly one stage."""

    def setUp(self):
        self.pipeline = _bare_pipeline()

    def _dense_returns(self, chunks, metas):
        self.pipeline.dense_multi_hop.retrieve.return_value = (chunks, metas)

    def _sparse_returns(self, chunks, metas):
        self.pipeline.sparse_multi_hop.retrieve.return_value = (chunks, metas)

    def _reranker_returns(self, chunks, metas, status="ok"):
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (chunks, metas, status)

    def test_dense_failure_still_answers_from_sparse(self):
        self.pipeline.dense_multi_hop.retrieve.side_effect = RuntimeError("embeddings down")
        self._sparse_returns(["sparse-1"], [{"src": "sparse-1"}])
        self._reranker_returns(["sparse-1"], [{"src": "sparse-1"}])

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["dense_retrieval"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual([entry["text"] for entry in result["context"]], ["sparse-1"])
        self.pipeline.sparse_multi_hop.retrieve.assert_called_once()
        self.assertEqual(
            self.pipeline.hybrid_rag.retrieve_from_precomputed.call_args.kwargs["dense_chunks"], []
        )

    def test_sparse_failure_still_answers_from_dense(self):
        self.pipeline.sparse_rag.set_collection.side_effect = RuntimeError("bm25 index gone")
        self._dense_returns(["dense-1"], [{"src": "dense-1"}])
        self._reranker_returns(["dense-1"], [{"src": "dense-1"}])

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["sparse_retrieval"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual([entry["text"] for entry in result["context"]], ["dense-1"])
        # set_collection raised, so the sparse hops never ran.
        self.pipeline.sparse_multi_hop.retrieve.assert_not_called()

    def test_both_retrievers_empty_returns_the_honest_no_context_answer(self):
        result = self.pipeline._run_core("q", "bob", 7)

        # An empty index is a healthy run over nothing, not a degraded pipeline.
        self.assertEqual(result["degraded"], [])
        self.assertEqual(result["answer"], app_pipeline._NO_CONTEXT_ANSWER)
        self.assertEqual(result["context"], [])
        self.assertEqual(result["source"], "user_collection")
        self.assertEqual(result["evaluation"], {"answer_relevancy": None, "faithfulness": None})
        self.pipeline.hybrid_rag.retrieve_from_precomputed.assert_not_called()
        self.pipeline.llm_client._call_api.assert_not_called()
        self.pipeline.evaluate.assert_not_called()

    def test_both_retrievers_raising_names_both_stages_and_never_raises(self):
        self.pipeline.dense_multi_hop.retrieve.side_effect = RuntimeError("chroma down")
        self.pipeline.sparse_multi_hop.retrieve.side_effect = RuntimeError("chroma down")

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["dense_retrieval", "sparse_retrieval"])
        self.assertEqual(result["answer"], app_pipeline._NO_CONTEXT_ANSWER)
        self.assertEqual(result["context"], [])

    def test_collection_resolve_failure_falls_back_to_the_dataset_collection(self):
        _install_resolver(self.pipeline, error=RuntimeError("postgres down"))
        self._dense_returns(["dense-1"], [{"src": "dense-1"}])
        self._reranker_returns(["dense-1"], [{"src": "dense-1"}])

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["collection_resolve"])
        self.assertEqual(result["source"], "dataset_collection_fallback")
        self.assertEqual(result["answer"], "generated answer")
        self.pipeline.dense_rag.set_collection.assert_called_once_with("dataset_collection")

    def test_corpus_user_without_documents_stops_before_retrieval(self):
        # Searching the shared corpus instead would quietly ignore what the user
        # asked for, so this path answers rather than retrieving.
        del self.pipeline._resolve_collection_for

        with mock.patch.object(app_pipeline, "UserCollection", _fake_user_collection(missing=True)):
            result = self.pipeline._run_core("q", "bob", 7, {"corpus": "user"})

        self.assertEqual(result["answer"], app_pipeline._NO_CONTEXT_ANSWER)
        self.assertEqual(result["context"], [])
        self.assertEqual(result["source"], "user_collection_missing")
        self.assertEqual(result["degraded"], [])
        self.pipeline.dense_multi_hop.retrieve.assert_not_called()
        self.pipeline.sparse_multi_hop.retrieve.assert_not_called()

    def test_rerank_failure_falls_back_to_merged_deduped_chunks(self):
        self.pipeline.hybrid_rag.retrieve_from_precomputed.side_effect = RuntimeError("reranker gone")
        self._dense_returns(["dense-1", "dense-2"], [{"src": "dense-1"}, {"src": "dense-2"}])
        self._sparse_returns(["sparse-1", "dense-1"], [{"src": "sparse-1"}, {"src": "sparse-dupe"}])

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["hybrid_rerank"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual(
            [entry["text"] for entry in result["context"]], ["dense-1", "dense-2", "sparse-1"]
        )
        self.assertEqual(
            [entry["metadata"] for entry in result["context"]],
            [{"src": "dense-1"}, {"src": "dense-2"}, {"src": "sparse-1"}],
        )

    def test_rerank_error_status_marks_a_degradation_but_keeps_its_chunks(self):
        self._dense_returns(["dense-1"], [{"src": "dense-1"}])
        self._reranker_returns(
            ["dense-1"], [{"src": "dense-1"}], "ERROR: reranking failed, returning original order"
        )

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["hybrid_rerank"])
        self.assertEqual([entry["text"] for entry in result["context"]], ["dense-1"])
        self.assertEqual(result["answer"], "generated answer")

    def test_rerank_returning_nothing_recovers_the_merged_chunks_once(self):
        self._dense_returns(["dense-1"], [{"src": "dense-1"}])
        self._reranker_returns([], [], "ERROR: no retrieved metas in both dense and sparse")

        result = self.pipeline._run_core("q", "bob", 7)

        # Flagged by the ERROR status and by the rescue path, but recorded once.
        self.assertEqual(result["degraded"], ["hybrid_rerank"])
        self.assertEqual([entry["text"] for entry in result["context"]], ["dense-1"])
        self.assertEqual(result["answer"], "generated answer")

    def test_answer_generation_failure_still_returns_the_retrieved_context(self):
        self._dense_returns(["dense-1"], [{"src": "dense-1"}])
        self._reranker_returns(["dense-1", "dense-2"], [{"src": "dense-1"}, {"src": "dense-2"}])
        self.pipeline.llm_client._call_api.side_effect = RuntimeError("llm down")

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["answer_generation"])
        self.assertEqual(result["answer"], app_pipeline._GENERATION_FAILED_ANSWER)
        self.assertEqual([entry["text"] for entry in result["context"]], ["dense-1", "dense-2"])
        self.assertEqual(result["evaluation"], {"answer_relevancy": None, "faithfulness": None})
        self.assertEqual(result["source"], "user_collection")
        # Scoring a placeholder answer would be a wasted judge call.
        self.pipeline.evaluate.assert_not_called()

    def test_evaluation_failure_preserves_the_answer_and_the_context(self):
        self._dense_returns(["dense-1"], [{"src": "dense-1"}])
        self._reranker_returns(["dense-1"], [{"src": "dense-1"}])
        self.pipeline.evaluate.side_effect = RuntimeError("judge exploded")

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["evaluation"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual([entry["text"] for entry in result["context"]], ["dense-1"])
        self.assertEqual(result["evaluation"], {"answer_relevancy": None, "faithfulness": None})

    def test_unpackable_evaluation_result_degrades_only_the_scores(self):
        self._dense_returns(["dense-1"], [{"src": "dense-1"}])
        self._reranker_returns(["dense-1"], [{"src": "dense-1"}])
        self.pipeline.evaluate.return_value = 0.5  # not a two-value tuple

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["evaluation"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual(result["evaluation"], {"answer_relevancy": None, "faithfulness": None})

    def test_several_stages_can_degrade_in_one_run(self):
        self.pipeline.dense_multi_hop.retrieve.side_effect = RuntimeError("chroma down")
        self._sparse_returns(["sparse-1"], [{"src": "sparse-1"}])
        self.pipeline.hybrid_rag.retrieve_from_precomputed.side_effect = RuntimeError("reranker gone")
        self.pipeline.evaluate.side_effect = RuntimeError("judge exploded")

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["dense_retrieval", "hybrid_rerank", "evaluation"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual([entry["text"] for entry in result["context"]], ["sparse-1"])


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class RunCoreStatusChannelTests(unittest.TestCase):
    """The status feed is cosmetic; losing it must never cost the answer."""

    def setUp(self):
        self.pipeline = _bare_pipeline()
        self.pipeline.dense_multi_hop.retrieve.return_value = (["dense-1"], [{"src": "dense-1"}])
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["dense-1"], [{"src": "dense-1"}], "ok",
        )

    def test_a_dead_emitter_degrades_only_the_status_stage(self):
        # StatusEmitter.emit swallows callback errors itself, so the way to reach
        # this path is an emitter object whose emit() raises.
        self.pipeline._build_emitter.return_value = mock.Mock(
            emit=mock.Mock(side_effect=RuntimeError("redis down"))
        )

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["status_emitter"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual([entry["text"] for entry in result["context"]], ["dense-1"])
        # The dead channel is found by the probe before it can be injected into
        # the retrievers, which emit unguarded.
        self.assertIs(
            self.pipeline.dense_multi_hop.set_emitter.call_args.args[0], app_pipeline.NULL_EMITTER
        )

    def test_emitter_construction_failure_degrades_only_the_status_stage(self):
        self.pipeline._build_emitter.side_effect = RuntimeError("redis unreachable")

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], ["status_emitter"])
        self.assertEqual(result["answer"], "generated answer")
        self.assertIs(
            self.pipeline.dense_multi_hop.set_emitter.call_args.args[0], app_pipeline.NULL_EMITTER
        )

    def test_a_failing_redis_publish_does_not_degrade_the_run(self):
        publish = mock.Mock(side_effect=RuntimeError("redis down"))
        self.pipeline._build_emitter.return_value = app_pipeline.StatusEmitter(callback=publish)

        result = self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(result["degraded"], [])
        self.assertEqual(result["answer"], "generated answer")
        self.assertTrue(publish.called)

    def test_a_working_emitter_is_injected_into_every_retriever(self):
        emitter = mock.Mock()
        self.pipeline._build_emitter.return_value = emitter

        self.pipeline._run_core("q", "bob", 7)

        self.assertIs(self.pipeline.dense_multi_hop.set_emitter.call_args.args[0], emitter)
        self.assertIs(self.pipeline.sparse_multi_hop.set_emitter.call_args.args[0], emitter)
        self.assertIs(self.pipeline.hybrid_rag.set_emitter.call_args.args[0], emitter)
        stages = [call.args[0] for call in emitter.emit.call_args_list]
        self.assertIn("pipeline_start", stages)
        self.assertIn("collection_resolved", stages)
        # "result"/"error" are terminal for the websocket consumer.
        self.assertNotIn("result", stages)
        self.assertNotIn("error", stages)


if __name__ == "__main__":
    unittest.main()
