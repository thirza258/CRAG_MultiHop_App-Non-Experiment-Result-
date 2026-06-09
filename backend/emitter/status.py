from typing import Callable, Optional

class StatusEmitter:
    def __init__(self, callback: Optional[Callable] = None):
        self._callback = callback

    def emit(self, stage: str, message: str, meta: dict = None):
        if self._callback:
            self._callback({
                "stage": stage,
                "message": message,
                "meta": meta or {}
            })

NULL_EMITTER = StatusEmitter()