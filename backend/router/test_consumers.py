"""Tests for QueryStreamConsumer — the websocket the chat UI actually talks to.

The focus is the contract between the browser and the pipeline: the CONFIG object
the sidebar sends must arrive at ``engine.run`` as a complete, validated config,
and a client that sends no CONFIG at all must still get the full default
pipeline. Everything external (Redis, the RAG engine, the database, the worker
thread) is stubbed, so these tests need no services.
"""

import json
import unittest
from unittest import mock

try:
    from router import consumers
    from common.pipeline_config import DEFAULT_PIPELINE_CONFIG
    IMPORT_ERROR = ""
except Exception as exc:  # needs Django settings + channels + redis
    consumers = None
    DEFAULT_PIPELINE_CONFIG = None
    IMPORT_ERROR = str(exc)


class _ImmediateThread:
    """threading.Thread stand-in that runs the target inline.

    The consumer hands the blocking pipeline call to a worker thread; running it
    synchronously makes the engine interaction observable without any waiting.
    """

    def __init__(self, target=None, **_kwargs):
        self._target = target
        self.started = False

    def start(self):
        self.started = True
        if self._target is not None:
            self._target()


def _result_message(answer="the answer"):
    """One Redis pub/sub frame carrying a finished pipeline result."""
    return {
        "type": "message",
        "data": json.dumps(
            {
                "stage": "result",
                "answer": answer,
                "context": [],
                "conversation_id": 7,
                "evaluation": {},
            }
        ),
    }


@unittest.skipIf(consumers is None, f"router.consumers unavailable: {IMPORT_ERROR}")
class QueryStreamConfigTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.consumer = consumers.QueryStreamConsumer()

        # Captures everything the consumer would have pushed to the browser.
        self.sent = []

        async def fake_send(text_data=None, **_kwargs):
            self.sent.append(json.loads(text_data))

        self.consumer.send = fake_send

        # A saved Conversation, without touching the database.
        conversation = mock.Mock()
        conversation.pk = 7
        self.consumer.save_conversation = mock.Mock(return_value=conversation)

        self.engine = mock.Mock()
        self.engine.run.return_value = {
            "answer": "the answer",
            "context": [],
            "evaluation": {},
        }

        registry = mock.Mock()
        registry.get_engine.return_value = self.engine

        self.pubsub = mock.AsyncMock()
        self.pubsub.get_message.return_value = _result_message()

        async_redis_client = mock.AsyncMock()
        async_redis_client.pubsub = mock.Mock(return_value=self.pubsub)

        patches = [
            mock.patch.object(consumers, "get_registry", return_value=registry),
            mock.patch.object(consumers, "threading", mock.Mock(Thread=_ImmediateThread)),
            # Publishing the result back to the websocket loop.
            mock.patch.object(consumers.redis, "Redis", return_value=mock.Mock()),
            # Imported inside receive() as `import redis.asyncio as aioredis`.
            mock.patch("redis.asyncio.Redis", return_value=async_redis_client),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def _receive(self, payload):
        await self.consumer.receive(json.dumps(payload))

    def _config_passed_to_engine(self):
        self.assertTrue(
            self.engine.run.called, "the pipeline was never invoked by the consumer"
        )
        return self.engine.run.call_args.kwargs["config"]

    async def test_missing_config_runs_the_full_default_pipeline(self):
        await self._receive({"USER": "bob", "QUERY": "who?"})
        self.assertEqual(self._config_passed_to_engine(), DEFAULT_PIPELINE_CONFIG)

    async def test_sidebar_config_reaches_the_engine_intact(self):
        # Exactly the shape frontend/src/context/PipelineConfigContext.tsx sends.
        await self._receive(
            {
                "USER": "bob",
                "QUERY": "who?",
                "CONFIG": {
                    "use_multi_hop": False,
                    "max_hops": 3,
                    "use_corrective": False,
                    "use_reranker": True,
                    "retrievers": "sparse",
                    "corpus": "base",
                },
            }
        )
        config = self._config_passed_to_engine()
        self.assertFalse(config["use_multi_hop"])
        self.assertFalse(config["use_corrective"])
        self.assertTrue(config["use_reranker"])
        self.assertEqual(config["retrievers"], "sparse")
        self.assertEqual(config["corpus"], "base")

    async def test_partial_config_is_completed_with_defaults(self):
        await self._receive(
            {"USER": "bob", "QUERY": "who?", "CONFIG": {"use_reranker": False}}
        )
        config = self._config_passed_to_engine()
        self.assertFalse(config["use_reranker"])
        self.assertEqual(set(config), set(DEFAULT_PIPELINE_CONFIG))
        self.assertTrue(config["use_multi_hop"])

    async def test_out_of_range_hop_count_is_clamped_not_rejected(self):
        await self._receive(
            {"USER": "bob", "QUERY": "who?", "CONFIG": {"max_hops": 99}}
        )
        self.assertEqual(self._config_passed_to_engine()["max_hops"], 3)

    async def test_malformed_config_falls_back_to_defaults(self):
        for junk in ("not-a-dict", 5, ["dense"], None):
            with self.subTest(junk=junk):
                self.engine.run.reset_mock()
                await self._receive({"USER": "bob", "QUERY": "who?", "CONFIG": junk})
                self.assertEqual(
                    self._config_passed_to_engine(), DEFAULT_PIPELINE_CONFIG
                )

    async def test_conversation_id_is_passed_so_status_events_can_stream(self):
        await self._receive({"USER": "bob", "QUERY": "who?"})
        self.assertEqual(self.engine.run.call_args.kwargs["conversation_id"], 7)

    async def test_the_result_frame_is_forwarded_to_the_browser(self):
        await self._receive({"USER": "bob", "QUERY": "who?"})
        stages = [payload.get("stage") for payload in self.sent]
        self.assertIn("result", stages)

    async def test_a_missing_username_is_rejected_before_running_anything(self):
        await self._receive({"QUERY": "who?"})
        self.assertFalse(self.engine.run.called)
        self.assertEqual(self.sent[-1]["stage"], "error")

    async def test_a_missing_query_is_rejected_before_running_anything(self):
        await self._receive({"USER": "bob"})
        self.assertFalse(self.engine.run.called)
        self.assertEqual(self.sent[-1]["stage"], "error")


if __name__ == "__main__":
    unittest.main()
