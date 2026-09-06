"""A last line of defence keeping request credentials out of the log file.

Individual call sites scrub the strings they build, but that only covers the
messages we compose ourselves. Two things slip past it:

* ``logger.error(..., exc_info=True)`` — used in 20-odd places — renders the
  original exception and its traceback, not the scrubbed message beside it.
* Any log line added later by someone who does not know a key can be in scope.

This filter closes both. It runs on the handler, so every record from every
logger passes through it, and it reads the request's own keys from the same
``ContextVar`` the pipeline uses — a ``logging`` filter executes inline on the
calling thread, so the right request's secrets are in scope.

It only ever *removes* text. A record with nothing to scrub is returned
untouched, and any failure inside the filter lets the record through rather than
silently dropping a log line.
"""

import logging
import traceback

from common.runtime.api_keys import PLACEHOLDER, scrub, secret_values
from common.runtime.context import request_secrets

logger = logging.getLogger(__name__)


class SecretScrubbingFilter(logging.Filter):
    """Replace any credential the current request supplied with a placeholder."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            secrets = request_secrets()
            if not secrets or not secret_values(secrets):
                # The overwhelmingly common case: no user-supplied key in scope.
                return True

            self._scrub_message(record, secrets)
            self._scrub_exception(record, secrets)
        except Exception:
            # A broken filter must not cost the deployment its logs.
            pass
        return True

    @staticmethod
    def _scrub_message(record: logging.LogRecord, secrets: dict) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return

        cleaned = scrub(message, secrets)
        if cleaned != message:
            # Collapse args into the already-formatted message: the substitution
            # has happened, so re-formatting would undo it.
            record.msg = cleaned
            record.args = ()

    @staticmethod
    def _scrub_exception(record: logging.LogRecord, secrets: dict) -> None:
        if not record.exc_info:
            return

        try:
            rendered = "".join(traceback.format_exception(*record.exc_info))
        except Exception:
            return

        cleaned = scrub(rendered, secrets)
        if cleaned == rendered:
            return

        # The handler would re-render exc_info itself, so the traceback has to
        # be carried as text instead. Nothing is lost — the scrubbed traceback
        # is appended in full.
        record.exc_info = None
        record.exc_text = None
        try:
            record.msg = f"{record.getMessage()}\n{cleaned}"
        except Exception:
            record.msg = f"[message unavailable]\n{cleaned}"
        record.args = ()


__all__ = ["SecretScrubbingFilter", "PLACEHOLDER"]
