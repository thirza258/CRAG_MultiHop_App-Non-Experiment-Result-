"""Everything a single request chose, and the credentials it brought with it.

The pipeline is one instance shared by every concurrent query, so none of this
can be stored on it. These modules carry it alongside the request instead:

* :mod:`~common.runtime.config` — the CONFIG the browser sends, validated and
  clamped into a known shape.
* :mod:`~common.runtime.context` — the ContextVar that shared pipeline stages
  read their per-request settings from.
* :mod:`~common.runtime.api_keys` — normalising, redacting and scrubbing keys
  the user brought.
* :mod:`~common.runtime.handoff` — getting both across a process boundary to a
  Celery worker without putting credentials in the broker.
* :mod:`~common.runtime.log_filters` — the last line of defence keeping those
  credentials out of the log file.
* :mod:`~common.runtime.errors` — telling a failed request apart from a
  failed run.

Deliberately no re-exports: one symbol, one import path.
"""
