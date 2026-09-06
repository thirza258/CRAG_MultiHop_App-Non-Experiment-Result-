"""Errors that describe the request rather than the run.

A retry is worth attempting when the failure was circumstantial: the embedding
API timed out, Redis blinked, the worker was killed mid-document. It is worth
nothing when the request asked for something this deployment cannot do — the
fourth attempt fails exactly like the first, three minutes later.

:class:`UnsupportedConfiguration` marks that second kind so the indexing task can
fail the document immediately instead of pretending the queue might fix it.
"""


class UnsupportedConfiguration(RuntimeError):
    """The request asked for a setting this deployment cannot honour.

    Raised only for an *explicit* choice. A setting the request left unset falls
    back to the deployment's own default and never raises: the user did not ask
    for the thing that is unavailable, so there is nothing to tell them about.
    """
