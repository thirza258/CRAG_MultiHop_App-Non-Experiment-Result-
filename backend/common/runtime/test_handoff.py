"""Tests for handing request settings to the indexing worker.

Two properties matter here and neither is visible from the call site:

* **Nothing is lost.** The worker is a separate process, so anything the upload
  chose that is not in the stash is silently ignored. The guard that decides
  "there is nothing to hand over" is therefore the single point where a whole
  group of controls can go dead without any error anywhere.
* **Nothing raises.** Indexing must not fail because Redis blinked. Both
  directions degrade to the server's own configuration instead.
"""

import json
import unittest
from unittest import mock

from common.runtime import handoff as runtime_handoff
from common.runtime.config import DEFAULT_PIPELINE_CONFIG, normalize_pipeline_config
from common.runtime.context import RuntimeSettings


class FakeRedis:
    """Enough of the client for these tests, with the calls recorded."""

    def __init__(self, store=None, fail=False):
        self.store = dict(store or {})
        self.fail = fail
        self.sets = []
        self.closed = False

    def set(self, key, value, ex=None):
        if self.fail:
            raise ConnectionError("redis is down")
        self.sets.append((key, value, ex))
        self.store[key] = value

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis is down")
        return self.store.get(key)

    def close(self):
        self.closed = True


def _params(**overrides):
    return normalize_pipeline_config(dict(overrides))


class StashGuardTests(unittest.TestCase):
    """Which settings are worth a round-trip to Redis."""

    def _stash(self, settings, fake=None):
        fake = fake or FakeRedis()
        with mock.patch.object(runtime_handoff, "_client", return_value=fake):
            return runtime_handoff.stash_runtime(settings), fake

    def test_an_empty_request_needs_no_handoff(self):
        token, fake = self._stash(RuntimeSettings())
        self.assertIsNone(token)
        self.assertEqual(fake.sets, [])

    def test_default_params_need_no_handoff(self):
        # Every control left alone: the worker's own configuration already
        # produces this, so the stash would be a copy of what it has.
        token, fake = self._stash(RuntimeSettings(params=dict(DEFAULT_PIPELINE_CONFIG)))
        self.assertIsNone(token)
        self.assertEqual(fake.sets, [])

    def test_a_chosen_model_is_handed_over(self):
        token, _ = self._stash(RuntimeSettings(embedding_model="openai/text-embedding-3-large"))
        self.assertTrue(token)

    def test_a_brought_key_is_handed_over(self):
        token, _ = self._stash(RuntimeSettings(keys={"openrouter": "sk-or-v1-abc"}))
        self.assertTrue(token)

    def test_an_empty_key_string_is_not_a_key(self):
        token, _ = self._stash(RuntimeSettings(keys={"openrouter": "", "news": ""}))
        self.assertIsNone(token)

    def test_a_chunk_size_on_its_own_is_handed_over(self):
        """The regression this file exists for.

        Chunk size, overlap and strategy live only in ``params``. A guard that
        looked at the models and the keys alone returned no token for an upload
        that changed nothing else, and the worker chunked with its own defaults
        while the panel showed the user's choice.
        """
        token, fake = self._stash(RuntimeSettings(params=_params(chunk_size=1200)))
        self.assertTrue(token, "a chunk size change must reach the worker")

        stored = json.loads(fake.sets[0][1])
        self.assertEqual(stored["params"]["chunk_size"], 1200)

    def test_a_chunk_strategy_on_its_own_is_handed_over(self):
        token, _ = self._stash(RuntimeSettings(params=_params(chunk_strategy="semantic")))
        self.assertTrue(token)

    def test_a_chunk_overlap_on_its_own_is_handed_over(self):
        token, _ = self._stash(RuntimeSettings(params=_params(chunk_overlap=120)))
        self.assertTrue(token)

    def test_a_non_settings_argument_is_refused(self):
        token, fake = self._stash({"embedding_model": "x"})
        self.assertIsNone(token)
        self.assertEqual(fake.sets, [])


class StashFailureTests(unittest.TestCase):
    def test_redis_being_down_does_not_fail_the_upload(self):
        fake = FakeRedis(fail=True)
        with mock.patch.object(runtime_handoff, "_client", return_value=fake):
            token = runtime_handoff.stash_runtime(
                RuntimeSettings(keys={"openrouter": "sk-or-v1-abc"})
            )
        self.assertIsNone(token)
        self.assertTrue(fake.closed, "the connection is released even on failure")

    def test_the_stored_payload_carries_the_key_but_the_log_line_does_not(self):
        fake = FakeRedis()
        with mock.patch.object(runtime_handoff, "_client", return_value=fake):
            with self.assertLogs(runtime_handoff.logger, level="INFO") as logs:
                runtime_handoff.stash_runtime(
                    RuntimeSettings(keys={"openrouter": "sk-or-v1-supersecret"})
                )

        self.assertIn("sk-or-v1-supersecret", fake.sets[0][1])
        self.assertNotIn("sk-or-v1-supersecret", "\n".join(logs.output))


class LoadTests(unittest.TestCase):
    def _load(self, token, fake):
        with mock.patch.object(runtime_handoff, "_client", return_value=fake):
            return runtime_handoff.load_runtime(token)

    def test_a_round_trip_preserves_the_choices(self):
        fake = FakeRedis()
        with mock.patch.object(runtime_handoff, "_client", return_value=fake):
            token = runtime_handoff.stash_runtime(RuntimeSettings(
                embedding_model="openai/text-embedding-3-large",
                keys={"openrouter": "sk-or-v1-abc"},
                params=_params(chunk_size=900, chunk_strategy="recursive"),
            ))

        loaded = self._load(token, fake)
        self.assertEqual(loaded.embedding_model, "openai/text-embedding-3-large")
        self.assertEqual(loaded.keys["openrouter"], "sk-or-v1-abc")
        self.assertEqual(loaded.params["chunk_size"], 900)

    def test_a_missing_token_means_the_server_configuration(self):
        self.assertEqual(runtime_handoff.load_runtime(None), RuntimeSettings())
        self.assertEqual(runtime_handoff.load_runtime(""), RuntimeSettings())

    def test_an_expired_entry_means_the_server_configuration(self):
        loaded = self._load("gone", FakeRedis())
        self.assertEqual(loaded, RuntimeSettings())

    def test_redis_being_down_means_the_server_configuration(self):
        loaded = self._load("tok", FakeRedis(fail=True))
        self.assertEqual(loaded, RuntimeSettings())

    def test_a_tampered_payload_is_re_validated_not_trusted(self):
        # The worker is a separate process reading a store it did not write.
        fake = FakeRedis(store={
            f"{runtime_handoff._KEY_PREFIX}tok": json.dumps({
                "embedding_model": "openai/text-embedding-3-small",
                "params": {"chunk_size": 10 ** 9, "retrievers": "not-a-retriever"},
            })
        })
        loaded = self._load("tok", fake)

        self.assertLess(loaded.params["chunk_size"], 10 ** 9)
        self.assertEqual(
            loaded.params["retrievers"], DEFAULT_PIPELINE_CONFIG["retrievers"]
        )


if __name__ == "__main__":
    unittest.main()
