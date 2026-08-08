"""Tests for the LLM client retry wrapper."""

import unittest
from unittest import mock

try:
    from ai_handler import llm
    IMPORT_ERROR = ""
except Exception as exc:
    llm = None
    IMPORT_ERROR = str(exc)


def _response(content):
    resp = mock.Mock()
    resp.choices = [mock.Mock(message=mock.Mock(content=content))]
    return resp


@unittest.skipIf(llm is None, f"ai_handler.llm unavailable: {IMPORT_ERROR}")
class ChatWithRetryTests(unittest.TestCase):
    def setUp(self):
        sleep_patcher = mock.patch.object(llm.time, "sleep")
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def _call(self, client):
        return llm._chat_with_retry(client, "OpenRouter", "test-model", "hi", 0.0)

    def test_returns_stripped_content_on_success(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = _response("  hello  ")

        self.assertEqual(self._call(client), "hello")
        client.chat.completions.create.assert_called_once()

    def test_none_content_becomes_empty_string(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = _response(None)

        self.assertEqual(self._call(client), "")

    def test_retries_transient_failure_then_succeeds(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = [
            ConnectionError("blip"),
            _response("answer"),
        ]

        self.assertEqual(self._call(client), "answer")
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_returns_error_string_after_all_attempts_fail(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = ConnectionError("down")

        result = self._call(client)

        self.assertEqual(
            client.chat.completions.create.call_count, llm._MAX_ATTEMPTS
        )
        self.assertIn("OpenRouter Error", result)
        self.assertIn("down", result)


if __name__ == "__main__":
    unittest.main()
