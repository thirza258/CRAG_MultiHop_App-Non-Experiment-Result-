import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class StatusEmitter:
    def __init__(self, callback: Optional[Callable] = None):
        self._callback = callback

    def emit(self, stage: str, message: str, meta: dict = None):
        if self._callback:
            event = {
                "stage": stage,
                "message": message,
                "meta": meta or {}
            }
            # The callback publishes to Redis; a dead broker must degrade the UI
            # progress feed only, never abort the RAG query that is reporting.
            try:
                self._callback(event)
            except Exception as e:
                logger.warning(f"[StatusEmitter] emit failed for stage '{stage}', continuing: {e}")

NULL_EMITTER = StatusEmitter()
