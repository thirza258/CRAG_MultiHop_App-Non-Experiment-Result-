"""Tests for the log filter that keeps request credentials out of the log file.

The call sites scrub the strings they compose, but ``exc_info=True`` is used in
roughly twenty places across the backend and renders the *original* exception,
not the scrubbed message beside it. This filter is what covers that, and any log
line added later by someone who does not know a user key can be in scope.
"""

import io
import logging
import unittest

try:
    from common.runtime import context as rc
    from common.runtime.api_keys import PLACEHOLDER
    from common.runtime.log_filters import SecretScrubbingFilter
    IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard
    rc = None
    IMPORT_ERROR = str(exc)

SECRET = "sk-or-v1-0123456789abcdef0123456789abcdef"


@unittest.skipIf(rc is None, f"common.runtime.log_filters unavailable: {IMPORT_ERROR}")
class SecretScrubbingFilterTests(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        handler = logging.StreamHandler(self.stream)
        handler.addFilter(SecretScrubbingFilter())

        self.logger = logging.getLogger(f"test.scrub.{id(self)}")
        self.logger.handlers = [handler]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def output(self) -> str:
        for handler in self.logger.handlers:
            handler.flush()
        return self.stream.getvalue()

    def test_a_key_in_a_plain_message_is_replaced(self):
        with rc.use_runtime(rc.RuntimeSettings(keys={"openrouter": SECRET})):
            self.logger.info("calling with Bearer %s", SECRET)

        self.assertNotIn(SECRET, self.output())
        self.assertIn(PLACEHOLDER, self.output())

    def test_a_key_inside_a_traceback_is_replaced(self):
        # The gap this filter exists for: the message beside exc_info may be
        # clean while the rendered exception is not.
        with rc.use_runtime(rc.RuntimeSettings(keys={"openrouter": SECRET})):
            try:
                raise ValueError(f"401 from provider, headers={{'Authorization': 'Bearer {SECRET}'}}")
            except ValueError:
                self.logger.error("Pipeline stage failed", exc_info=True)

        output = self.output()
        self.assertNotIn(SECRET, output)
        self.assertIn(PLACEHOLDER, output)

    def test_the_traceback_itself_is_still_reported(self):
        with rc.use_runtime(rc.RuntimeSettings(keys={"openrouter": SECRET})):
            try:
                raise ValueError(f"boom {SECRET}")
            except ValueError:
                self.logger.error("Pipeline stage failed", exc_info=True)

        output = self.output()
        # Scrubbing must not cost the diagnostic value of the traceback.
        self.assertIn("Pipeline stage failed", output)
        self.assertIn("ValueError", output)
        self.assertIn("Traceback", output)
        self.assertIn("test_the_traceback_itself_is_still_reported", output)

    def test_the_news_key_is_scrubbed_too(self):
        news = "9f8e7d6c5b4a39281706abcdef123456"
        with rc.use_runtime(rc.RuntimeSettings(keys={"news": news})):
            self.logger.warning("GET https://newsapi.org/v2/everything?apiKey=%s", news)

        self.assertNotIn(news, self.output())

    def test_records_are_untouched_when_no_key_is_in_scope(self):
        # The overwhelmingly common case; it must stay cheap and lossless.
        self.logger.info("plain message with %s and %d", "args", 42)
        self.assertIn("plain message with args and 42", self.output())

    def test_a_message_without_the_key_keeps_its_lazy_args(self):
        with rc.use_runtime(rc.RuntimeSettings(keys={"openrouter": SECRET})):
            self.logger.info("nothing secret here: %s", "value")
        self.assertIn("nothing secret here: value", self.output())

    def test_the_filter_never_drops_a_record(self):
        broken = logging.LogRecord(
            name="x", level=logging.ERROR, pathname="p", lineno=1,
            # %d against a string raises inside getMessage().
            msg="%d", args=("not-a-number",), exc_info=None,
        )
        with rc.use_runtime(rc.RuntimeSettings(keys={"openrouter": SECRET})):
            self.assertTrue(SecretScrubbingFilter().filter(broken))

    def test_the_filter_is_inert_outside_a_request(self):
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname="p", lineno=1,
            msg="value is %s", args=(SECRET,), exc_info=None,
        )
        self.assertTrue(SecretScrubbingFilter().filter(record))
        # No runtime installed means nothing was supplied by a user, so there is
        # nothing this filter is entitled to consider secret.
        self.assertEqual(record.args, (SECRET,))


if __name__ == "__main__":
    unittest.main()
