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
    from ai_handler.openrouter import MissingAPIKeyError
    from common.runtime import context as runtime_context
    from common.runtime.errors import UnsupportedConfiguration
    from common.runtime.config import normalize_pipeline_config
    from pipeline import app_pipeline
    from pipeline.app_pipeline import AppRAGPipeline
    IMPORT_ERROR = ""
except Exception as exc:  # needs Django settings + chromadb/torch/redis
    app_pipeline = None
    AppRAGPipeline = None
    runtime_context = None
    MissingAPIKeyError = None
    UnsupportedConfiguration = None
    normalize_pipeline_config = None
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
    "dense_config": {"embedding_model": "configured/embed"},
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
    # Queries the database for real; EmbeddingModelPinningTests covers it.
    # "" means "nothing to pin", which is the pre-existing behaviour.
    pipeline._collection_embedding_model = mock.Mock(return_value="")

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
        judge_mock.assert_called_once_with(
            dataset="dataset",
            llm_judge="judge",
            judge_embeddings="judge-embeddings",
            metrics=["answer_relevancy", "faithfulness"],
        )
        # The judge stays on the deployment's configured models, not on the one
        # answering the question — a model scoring its own output is the bias
        # the separate judge configuration exists to avoid.
        llm_mock.assert_called_once_with("judge-model")
        emb_mock.assert_called_once_with("judge-embedding-model")

    def test_a_request_can_override_the_judge_models(self):
        scores = _FakeScores({"answer_relevancy": 0.75, "faithfulness": 0.5})
        convert, judge, llm_wrapper, emb_wrapper = self._patched_judge(judge_result=scores)

        settings = runtime_context.RuntimeSettings(
            llm_model="answering/model",
            params={
                "evaluation_llm_model": "chosen/judge",
                "evaluation_embedding_model": "chosen/judge-embedder",
            },
        )
        with convert, judge, llm_wrapper as llm_mock, emb_wrapper as emb_mock:
            with runtime_context.use_runtime(settings):
                self.pipeline.evaluate("q", ["chunk"], "answer")

        llm_mock.assert_called_once_with("chosen/judge")
        emb_mock.assert_called_once_with("chosen/judge-embedder")

    def test_a_switched_off_metric_is_not_requested_from_the_judge(self):
        scores = _FakeScores({"answer_relevancy": 0.75, "faithfulness": 0.5})
        convert, judge, llm_wrapper, emb_wrapper = self._patched_judge(judge_result=scores)

        settings = runtime_context.RuntimeSettings(
            params={"eval_answer_relevancy": True, "eval_faithfulness": False}
        )
        with convert, judge as judge_mock, llm_wrapper, emb_wrapper:
            with runtime_context.use_runtime(settings):
                self.pipeline.evaluate("q", ["chunk"], "answer")

        # Each metric is its own judge pass, so this is a real saving rather
        # than a hidden number.
        self.assertEqual(
            judge_mock.call_args.kwargs["metrics"], ["answer_relevancy"]
        )

    def test_every_metric_off_asks_the_judge_for_nothing(self):
        scores = _FakeScores({"answer_relevancy": 0.75, "faithfulness": 0.5})
        convert, judge, llm_wrapper, emb_wrapper = self._patched_judge(judge_result=scores)

        settings = runtime_context.RuntimeSettings(
            params={"eval_answer_relevancy": False, "eval_faithfulness": False}
        )
        with convert, judge as judge_mock, llm_wrapper, emb_wrapper:
            with runtime_context.use_runtime(settings):
                self.pipeline.evaluate("q", ["chunk"], "answer")

        self.assertEqual(judge_mock.call_args.kwargs["metrics"], [])

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
        notices = ["searched with a different embedding model"]

        result = pipeline._no_context_result("dataset_collection", degraded, notices)
        degraded.append("mutated-after-the-fact")
        notices.append("mutated-after-the-fact")

        self.assertEqual(result["answer"], app_pipeline._NO_CONTEXT_ANSWER)
        self.assertEqual(result["source"], "dataset_collection")
        self.assertEqual(result["context"], [])
        self.assertEqual(result["evaluation"], {"answer_relevancy": None, "faithfulness": None})
        self.assertEqual(result["degraded"], ["dense_retrieval"])
        self.assertEqual(result["notices"], ["searched with a different embedding model"])

    def test_no_context_result_defaults_notices_to_empty(self):
        # Callers that predate the field must keep working.
        result = _bare_pipeline()._no_context_result("dataset_collection", [])
        self.assertEqual(result["notices"], [])


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

        self.assertEqual(
            sorted(result),
            ["answer", "context", "degraded", "evaluation", "notices", "source"],
        )
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

        self.pipeline.run(
            "bob", "who wrote it?", 7, {"corpus": "user"}, {"openrouter": "k"}
        )

        self.pipeline._run_core.assert_called_once_with(
            "who wrote it?", "bob", 7, {"corpus": "user"}, {"openrouter": "k"}
        )

    def test_run_works_without_a_conversation_id_config_or_keys(self):
        # The REST endpoint has no conversation to stream to at call time, and a
        # caller that brought no key of its own falls back to the server's.
        self.pipeline._run_core = mock.Mock(return_value={"answer": "x"})

        self.pipeline.run("bob", "who wrote it?")

        self.pipeline._run_core.assert_called_once_with(
            "who wrote it?", "bob", None, None, None
        )


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
        self.assertEqual(result["degraded"], [])

    def test_the_hop_ceiling_is_never_written_onto_the_shared_chain(self):
        """The pipeline is one instance shared by every concurrent query.

        Assigning the ceiling onto the chain per query meant two requests
        asking for different hop counts raced, and whichever wrote last set the
        ceiling for both. MultiHopRetriever reads it from the request context
        instead, so the pipeline must not touch the setter at all.
        """
        self._run({"max_hops": 2})

        self.pipeline.dense_multi_hop.set_max_hops.assert_not_called()
        self.pipeline.sparse_multi_hop.set_max_hops.assert_not_called()

    def test_max_hops_reaches_the_chain_through_the_request_context(self):
        seen = {}
        self.pipeline.dense_multi_hop.retrieve.side_effect = (
            lambda *a, **k: (seen.setdefault(
                "max_hops", runtime_context.resolve_param("max_hops", 3)
            ), ([], []))[1]
        )

        self._run({"max_hops": 2})

        self.assertEqual(seen["max_hops"], 2)

    def test_multi_hop_off_queries_the_corrective_chain(self):
        result = self._run({"use_multi_hop": False})

        self.pipeline.dense_corrective_rag.retrieve.assert_called_once_with("q")
        self.pipeline.dense_multi_hop.retrieve.assert_not_called()
        self.assertEqual(result["degraded"], [])

    def test_both_stages_off_queries_the_base_retriever(self):
        result = self._run({"use_multi_hop": False, "use_corrective": False})

        self.pipeline.dense_rag.retrieve.assert_called_once_with("q")
        self.pipeline.dense_corrective_rag.retrieve.assert_not_called()
        self.pipeline.dense_multi_hop.retrieve.assert_not_called()
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


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class EvaluationToggleTests(unittest.TestCase):
    """Answer scoring is two LLM-judge passes plus an embedding call.

    Switching it off has to actually skip them — and must not be reported as a
    degradation, because nothing failed.
    """

    def setUp(self):
        self.pipeline = _bare_pipeline()
        self.pipeline.dense_multi_hop.retrieve.return_value = (["d1"], [{"src": "d1"}])
        self.pipeline.sparse_multi_hop.retrieve.return_value = (["s1"], [{"src": "s1"}])
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["d1"], [{"src": "d1"}], "ok",
        )

    def test_evaluation_runs_by_default(self):
        result = self.pipeline._run_core("q", "bob", 7)

        self.pipeline.evaluate.assert_called_once()
        self.assertEqual(
            result["evaluation"], {"answer_relevancy": 0.91, "faithfulness": 0.88}
        )

    def test_switching_it_off_skips_the_judge_entirely(self):
        result = self.pipeline._run_core("q", "bob", 7, config={"use_evaluation": False})

        self.pipeline.evaluate.assert_not_called()
        self.assertEqual(
            result["evaluation"], {"answer_relevancy": None, "faithfulness": None}
        )

    def test_switching_it_off_is_not_a_degradation(self):
        # Nothing failed — the user gave the scores up on purpose.
        result = self.pipeline._run_core("q", "bob", 7, config={"use_evaluation": False})
        self.assertEqual(result["degraded"], [])

    def test_the_answer_is_unaffected(self):
        result = self.pipeline._run_core("q", "bob", 7, config={"use_evaluation": False})
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual([entry["text"] for entry in result["context"]], ["d1"])


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class CollectionEmbeddingModelTests(unittest.TestCase):
    """Which embedding model a collection's vectors actually came from.

    This is the lookup that stops a user's embedding-model choice from being
    applied to a collection built by a different model — a comparison across
    vector spaces, which ChromaDB either rejects outright or answers with
    confident nonsense.
    """

    def setUp(self):
        self.pipeline = _bare_pipeline()
        del self.pipeline._collection_embedding_model  # exercise the real method

    def test_a_user_collection_with_chunks_reports_its_recorded_model(self):
        record = mock.Mock(chunk_count=12, embedding_model="recorded/model")
        user_col = mock.Mock()
        user_col.objects.filter.return_value.first.return_value = record

        with mock.patch.object(app_pipeline, "UserCollection", user_col):
            self.assertEqual(
                self.pipeline._collection_embedding_model("user_bob_collection", "user_collection"),
                "recorded/model",
            )

    def test_an_empty_user_collection_is_not_pinned(self):
        # Nothing has been embedded into it yet, so the user's choice still
        # governs — the first document they index decides.
        record = mock.Mock(chunk_count=0, embedding_model="recorded/model")
        user_col = mock.Mock()
        user_col.objects.filter.return_value.first.return_value = record

        with mock.patch.object(app_pipeline, "UserCollection", user_col):
            self.assertEqual(
                self.pipeline._collection_embedding_model("user_bob_collection", "user_collection"),
                "",
            )

    def test_a_missing_user_collection_record_is_not_pinned(self):
        user_col = mock.Mock()
        user_col.objects.filter.return_value.first.return_value = None

        with mock.patch.object(app_pipeline, "UserCollection", user_col):
            self.assertEqual(
                self.pipeline._collection_embedding_model("user_bob_collection", "user_collection"),
                "",
            )

    def test_the_shared_corpus_uses_its_bookkeeping_row(self):
        chroma_col = mock.Mock()
        chroma_col.objects.filter.return_value.first.return_value = mock.Mock(
            embedding_model="corpus/model"
        )

        with mock.patch.object(app_pipeline, "ChromaCollection", chroma_col):
            self.assertEqual(
                self.pipeline._collection_embedding_model("dataset_collection", "dataset_collection"),
                "corpus/model",
            )

    def test_the_shared_corpus_falls_back_to_the_configured_model(self):
        # A corpus indexed before per-collection recording existed: it was built
        # by this deployment's own dense_config, so that is what is in it.
        chroma_col = mock.Mock()
        chroma_col.objects.filter.return_value.first.return_value = None

        with mock.patch.object(app_pipeline, "ChromaCollection", chroma_col):
            self.assertEqual(
                self.pipeline._collection_embedding_model("dataset_collection", "dataset_collection"),
                "configured/embed",
            )

    def test_a_database_failure_is_not_pinned_rather_than_raising(self):
        # This runs on the hot path of every query; a bookkeeping lookup must
        # never be the reason a question goes unanswered.
        user_col = mock.Mock()
        user_col.objects.filter.side_effect = RuntimeError("database is down")

        with mock.patch.object(app_pipeline, "UserCollection", user_col):
            self.assertEqual(
                self.pipeline._collection_embedding_model("user_bob_collection", "user_collection"),
                "",
            )

    def test_an_empty_collection_name_is_not_pinned(self):
        self.assertEqual(self.pipeline._collection_embedding_model("", "user_collection"), "")

    def test_configured_embedding_model_survives_a_broken_config(self):
        for config in ({}, {"dense_config": None}, {"dense_config": "nonsense"}):
            with self.subTest(config=config):
                pipeline = _bare_pipeline(config)
                self.assertEqual(pipeline._configured_embedding_model(), "")


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class RunCoreRuntimeTests(unittest.TestCase):
    """The per-query model choice and credentials, as seen from inside a stage.

    The pipeline is a process-wide singleton, so these cannot be written onto
    it. They are installed as a request-scoped runtime for the duration of the
    query and read at call time by whichever stage needs them.
    """

    def setUp(self):
        self.pipeline = _bare_pipeline()
        self.pipeline.dense_multi_hop.retrieve.return_value = (["d1"], [{"src": "d1"}])
        self.pipeline.sparse_multi_hop.retrieve.return_value = (["s1"], [{"src": "s1"}])
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["d1"], [{"src": "d1"}], "ok",
        )

        # Answer generation is the last stage to run, so what it observes is
        # what the whole query saw.
        self.observed = {}

        # _generate_answer passes the request's temperature as a keyword.
        def record(_prompt, temperature=None):
            self.observed["temperature"] = temperature
            self.observed["llm_model"] = runtime_context.resolve_llm_model("stage-default")
            self.observed["embedding_model"] = runtime_context.resolve_embedding_model("stage-default")
            self.observed["openrouter_key"] = runtime_context.openrouter_api_key()
            self.observed["news_key"] = runtime_context.news_api_key()
            # What every other component resolves its own knobs through.
            self.observed["top_k"] = runtime_context.resolve_param("top_k", 4)
            self.observed["use_wikipedia"] = runtime_context.resolve_param("use_wikipedia", True)
            self.observed["strictness"] = runtime_context.resolve_param("corrective_strictness", "")
            return "generated answer"

        self.pipeline.llm_client._call_api.side_effect = record

    def test_the_chosen_model_reaches_the_stages(self):
        self.pipeline._run_core(
            "q", "bob", 7, config={"llm_model": "vendor/chosen"},
        )
        self.assertEqual(self.observed["llm_model"], "vendor/chosen")

    def test_no_choice_leaves_each_stage_on_its_own_configured_model(self):
        self.pipeline._run_core("q", "bob", 7)
        self.assertEqual(self.observed["llm_model"], "stage-default")

    def test_the_callers_keys_reach_the_stages(self):
        self.pipeline._run_core(
            "q", "bob", 7, keys={"openrouter": "caller-or-key", "news": "caller-news-key"},
        )
        self.assertEqual(self.observed["openrouter_key"], "caller-or-key")
        self.assertEqual(self.observed["news_key"], "caller-news-key")

    def test_a_malformed_keys_block_falls_back_instead_of_failing_the_query(self):
        for junk in ("string", 42, [], {"openrouter": "has a space"}):
            with self.subTest(junk=junk):
                result = self.pipeline._run_core("q", "bob", 7, keys=junk)
                self.assertEqual(result["answer"], "generated answer")

    def test_pipeline_config_values_reach_the_stages(self):
        # The whole chain end to end: CONFIG -> normalise -> RuntimeSettings.params
        # -> resolve_param inside a stage. Every knob other than the models and
        # the keys travels this way, so one case covers the mechanism.
        self.pipeline._run_core(
            "q", "bob", 7,
            config={
                "top_k": 9,
                "use_wikipedia": False,
                "corrective_strictness": "strict",
            },
        )

        self.assertEqual(self.observed["top_k"], 9)
        self.assertFalse(self.observed["use_wikipedia"])
        self.assertEqual(self.observed["strictness"], "strict")

    def test_unset_knobs_fall_through_to_each_components_own_default(self):
        # The sentinels: null for a number, "" for a name. Neither may be
        # mistaken for a real value, or every deployment's tuning is overridden.
        self.pipeline._run_core("q", "bob", 7)

        self.assertEqual(self.observed["top_k"], 4)
        self.assertTrue(self.observed["use_wikipedia"])
        self.assertEqual(self.observed["strictness"], "")

    def test_the_requests_temperature_reaches_answer_generation(self):
        self.pipeline._run_core("q", "bob", 7, config={"temperature": 0.8})
        self.assertEqual(self.observed["temperature"], 0.8)

    def test_an_unset_temperature_leaves_the_client_on_its_own(self):
        self.pipeline._run_core("q", "bob", 7)
        self.assertIsNone(self.observed["temperature"])

    def test_the_params_do_not_outlive_the_query(self):
        self.pipeline._run_core("q", "bob", 7, config={"top_k": 9})
        self.assertEqual(runtime_context.resolve_param("top_k", 4), 4)

    def test_the_runtime_does_not_outlive_the_query(self):
        # The REST path runs on a pooled thread: a leaked runtime would hand the
        # next request served by that thread this caller's key.
        self.pipeline._run_core(
            "q", "bob", 7,
            config={"llm_model": "vendor/chosen"},
            keys={"openrouter": "caller-or-key"},
        )
        self.assertEqual(runtime_context.current_runtime(), runtime_context.EMPTY_RUNTIME)
        self.assertEqual(runtime_context.resolve_llm_model("stage-default"), "stage-default")

    def test_the_runtime_is_reset_even_when_a_stage_raises(self):
        self.pipeline.llm_client._call_api.side_effect = RuntimeError("llm down")

        self.pipeline._run_core("q", "bob", 7, keys={"openrouter": "caller-or-key"})

        self.assertEqual(runtime_context.current_runtime(), runtime_context.EMPTY_RUNTIME)


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class EmbeddingModelPinningTests(unittest.TestCase):
    """The collection's embedding model overrides the user's pick, out loud.

    Silently ignoring a setting is what makes a settings panel untrustworthy, so
    the override is reported in the result's ``notices``.
    """

    def setUp(self):
        self.pipeline = _bare_pipeline()
        self.pipeline.dense_multi_hop.retrieve.return_value = (["d1"], [{"src": "d1"}])
        self.pipeline.sparse_multi_hop.retrieve.return_value = (["s1"], [{"src": "s1"}])
        self.pipeline.hybrid_rag.retrieve_from_precomputed.return_value = (
            ["d1"], [{"src": "d1"}], "ok",
        )
        self.pipeline._collection_embedding_model = mock.Mock(return_value="collection/actual")

        self.observed = {}

        def record(_prompt, temperature=None):
            self.observed["embedding_model"] = runtime_context.resolve_embedding_model("stage-default")
            return "generated answer"

        self.pipeline.llm_client._call_api.side_effect = record

    def test_the_collections_model_is_what_the_stages_use(self):
        self.pipeline._run_core("q", "bob", 7, config={"embedding_model": "user/choice"})
        self.assertEqual(self.observed["embedding_model"], "collection/actual")

    def test_the_override_is_reported_to_the_caller(self):
        result = self.pipeline._run_core("q", "bob", 7, config={"embedding_model": "user/choice"})

        self.assertEqual(len(result["notices"]), 1)
        notice = result["notices"][0]
        self.assertIn("collection/actual", notice)
        self.assertIn("user/choice", notice)

    def test_an_override_is_not_a_degradation(self):
        # Nothing failed — the pipeline ran exactly as designed.
        result = self.pipeline._run_core("q", "bob", 7, config={"embedding_model": "user/choice"})
        self.assertEqual(result["degraded"], [])

    def test_no_notice_when_the_choice_already_matches(self):
        result = self.pipeline._run_core(
            "q", "bob", 7, config={"embedding_model": "collection/actual"}
        )
        self.assertEqual(result["notices"], [])

    def test_no_notice_when_the_user_chose_nothing(self):
        # The pin still happens; there is just nothing to tell them about.
        result = self.pipeline._run_core("q", "bob", 7)
        self.assertEqual(result["notices"], [])
        self.assertEqual(self.observed["embedding_model"], "collection/actual")

    def test_an_unpinnable_collection_leaves_the_choice_in_place(self):
        self.pipeline._collection_embedding_model = mock.Mock(return_value="")

        result = self.pipeline._run_core("q", "bob", 7, config={"embedding_model": "user/choice"})

        self.assertEqual(self.observed["embedding_model"], "user/choice")
        self.assertEqual(result["notices"], [])

    def test_the_notice_survives_an_empty_retrieval(self):
        # _no_context_result is a separate exit path and used to drop it.
        self.pipeline.dense_multi_hop.retrieve.return_value = ([], [])
        self.pipeline.sparse_multi_hop.retrieve.return_value = ([], [])

        result = self.pipeline._run_core("q", "bob", 7, config={"embedding_model": "user/choice"})

        self.assertEqual(result["context"], [])
        self.assertEqual(len(result["notices"]), 1)


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class IndexEmbeddingModelTests(unittest.TestCase):
    """Which model a document gets indexed with.

    One ChromaDB collection is one vector space, so the first document a user
    indexes fixes the model for every later one. A newer pick can only be
    honoured while the collection is still empty.
    """

    def setUp(self):
        self.pipeline = _bare_pipeline()

    def test_an_empty_collection_adopts_the_users_choice(self):
        collection = mock.Mock(chunk_count=0, embedding_model="", collection_name="user_bob_collection")

        with runtime_context.use_runtime(
            runtime_context.RuntimeSettings(embedding_model="user/choice")
        ):
            self.assertEqual(self.pipeline._index_embedding_model(collection), "user/choice")

    def test_a_populated_collection_stays_locked_to_its_recorded_model(self):
        collection = mock.Mock(chunk_count=42, embedding_model="locked/model", collection_name="user_bob_collection")

        with runtime_context.use_runtime(
            runtime_context.RuntimeSettings(embedding_model="user/choice")
        ):
            self.assertEqual(self.pipeline._index_embedding_model(collection), "locked/model")

    def test_no_choice_falls_back_to_the_configured_model(self):
        collection = mock.Mock(chunk_count=0, embedding_model="", collection_name="user_bob_collection")
        self.assertEqual(self.pipeline._index_embedding_model(collection), "configured/embed")

    def test_a_collection_with_chunks_but_no_recorded_model_is_not_locked(self):
        # Nothing to lock to; the choice (or the configured default) applies.
        collection = mock.Mock(chunk_count=5, embedding_model="", collection_name="user_bob_collection")

        with runtime_context.use_runtime(
            runtime_context.RuntimeSettings(embedding_model="user/choice")
        ):
            self.assertEqual(self.pipeline._index_embedding_model(collection), "user/choice")


@unittest.skipIf(AppRAGPipeline is None, f"app_pipeline unavailable: {IMPORT_ERROR}")
class ChunkerForRequestTests(unittest.TestCase):
    """Which chunker an upload gets, and what happens when it cannot have it.

    The asymmetry under test: a setting the request left alone falls back
    silently, because the user never asked for the thing that is unavailable.
    A setting the request named cannot fall back silently, because the document
    would be stored under a split nobody chose and nothing afterwards can tell.
    """

    def setUp(self):
        from common.chunker import DocumentChunker

        self.DocumentChunker = DocumentChunker
        self.pipeline = object.__new__(AppRAGPipeline)
        self.pipeline.config = dict(_CONFIG)
        self.pipeline.chunker = DocumentChunker(
            strategy="recursive", chunk_size=500, overlap=50
        )
        self.pipeline.dense_rag = mock.Mock()
        self.pipeline.dense_rag.client = mock.Mock()

    def _chunker(self, **params):
        with runtime_context.use_runtime(
            runtime_context.RuntimeSettings(params=normalize_pipeline_config(params))
        ):
            return self.pipeline._chunker_for_request("openai/text-embedding-3-small")

    def _keyless(self):
        """A dense_rag whose client property raises, as it does with no API key."""
        broken = mock.Mock()
        type(broken).client = mock.PropertyMock(
            side_effect=MissingAPIKeyError("no OpenRouter key")
        )
        self.pipeline.dense_rag = broken

    def test_an_untouched_request_reuses_the_shared_chunker(self):
        # Identity, not equality: the common path must not allocate.
        self.assertIs(self._chunker(), self.pipeline.chunker)

    def test_a_chosen_size_and_overlap_are_applied(self):
        chunker = self._chunker(chunk_size=1200, chunk_overlap=120)

        self.assertEqual((chunker.chunk_size, chunker.overlap), (1200, 120))
        self.assertIsNot(chunker, self.pipeline.chunker)

    def test_a_chosen_strategy_is_applied(self):
        self.assertEqual(self._chunker(chunk_strategy="paragraph").strategy, "paragraph")

    def test_semantic_is_given_the_embedding_client(self):
        chunker = self._chunker(chunk_strategy="semantic")

        self.assertEqual(chunker.strategy, "semantic")
        self.assertIs(chunker.client, self.pipeline.dense_rag.client)

    def test_the_index_embedding_model_is_passed_through(self):
        # Whatever model _build_index pinned for this collection is the one the
        # semantic splitter embeds sentences with.
        with runtime_context.use_runtime(runtime_context.RuntimeSettings(
            params=normalize_pipeline_config({"chunk_size": 800})
        )):
            chunker = self.pipeline._chunker_for_request("baai/bge-m3")

        self.assertEqual(chunker.embedding_model, "baai/bge-m3")

    def test_no_index_model_leaves_the_chunkers_own_default(self):
        with runtime_context.use_runtime(runtime_context.RuntimeSettings(
            params=normalize_pipeline_config({"chunk_size": 800})
        )):
            chunker = self.pipeline._chunker_for_request("")

        self.assertEqual(
            chunker.embedding_model, self.pipeline.chunker.embedding_model
        )

    def test_semantic_without_a_key_is_refused_rather_than_downgraded(self):
        """The regression this class exists for.

        Falling back to `recursive` here indexed the document a different way
        and reported success. The upload path has no notices channel, so the
        only honest signal left is failing.
        """
        self._keyless()

        with self.assertRaises(UnsupportedConfiguration) as caught:
            self._chunker(chunk_strategy="semantic")

        self.assertIn("Semantic chunking", str(caught.exception))

    def test_a_deployment_defaulting_to_semantic_still_falls_back(self):
        # Nothing was *asked* for, so there is nobody to tell and no choice to
        # betray — the upload should succeed with the recursive splitter.
        self._keyless()
        self.pipeline.chunker = self.DocumentChunker(strategy="semantic")

        chunker = self._chunker()

        self.assertIs(chunker, self.pipeline.chunker)

    def test_an_unparseable_size_falls_back_instead_of_failing(self):
        # normalize_pipeline_config never produces this; a hand-built params
        # dict from an older stash could.
        with runtime_context.use_runtime(
            runtime_context.RuntimeSettings(params={"chunk_size": "wide"})
        ):
            chunker = self.pipeline._chunker_for_request("")

        self.assertIs(chunker, self.pipeline.chunker)


if __name__ == "__main__":
    unittest.main()
