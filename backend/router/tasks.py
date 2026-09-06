from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from .models import DocumentChunk, Document
from common.runtime.errors import UnsupportedConfiguration
from common.runtime.context import use_runtime
from common.runtime.handoff import load_runtime
from rag.rag_service import get_registry
from django.db import transaction

import logging

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    acks_late=True,
    reject_on_worker_lost=True,
)
def build_index_task(self, document_id: int, username: str, runtime_token: str = None):
    """Chunk, embed and index one document.

    ``runtime_token`` points at the embedding model and API keys the uploading
    request chose (see :mod:`common.runtime.handoff`). It is a token rather than
    the values themselves because task arguments are persisted in the broker and
    logged on failure. A missing or expired token means the worker's own
    configuration applies, which is what indexing always did before users could
    bring their own keys.

    The token is deliberately not discarded on success: Celery can redeliver an
    acks_late task, and the entry expires on its own.
    """
    try:

        with transaction.atomic():

            document = (
                Document.objects
                .select_for_update()
                .get(pk=document_id)
            )

            # Prevent duplicate execution
            if document.status == "ready":
                logger.info(
                    f"[build_index_task] Document {document.pk} already ready"
                )
                return

            if document.status == "indexing":
                logger.info(
                    f"[build_index_task] Document {document.pk} already indexing"
                )
                return

            if DocumentChunk.objects.filter(document=document).exists():
                logger.warning(
                    f"[build_index_task] Chunks already exist "
                    f"for document {document.pk}"
                )

                document.status = "ready"
                document.save(update_fields=["status"])
                return

            document.status = "indexing"
            document.save(update_fields=["status"])

        # OUTSIDE transaction
        # do heavy processing here
        pipeline = get_registry().get_engine()
        # Scoped to this task: a worker thread indexes one document after
        # another, and leaving the credentials installed would spend the wrong
        # user's key on the next one.
        with use_runtime(load_runtime(runtime_token)):
            pipeline._build_index(
                username=username,
                document=document
            )

        document.status = "ready"
        document.save(update_fields=["status"])

        return {
            "status": "success",
            "document_id": document_id
        }

    except Document.DoesNotExist:
        logger.error(f"[Task] Document {document_id} not found")
        raise

    except SoftTimeLimitExceeded:
        logger.error(
            f"[Task] Document {document_id} indexing timed out"
        )
        document = Document.objects.filter(pk=document_id).first()
        if document:
            document.status = "failed"
            document.save(update_fields=["status"])
        # Don't retry on timeout — the task is inherently too slow
        raise

    except UnsupportedConfiguration as e:
        # The upload asked for something this deployment cannot do. Retrying
        # produces the same failure four minutes later, so fail it now and let
        # the user change the setting and re-upload.
        logger.error(
            f"[Task] Document {document_id} cannot be indexed as configured: {e}"
        )
        document = Document.objects.filter(pk=document_id).first()
        if document:
            document.status = "failed"
            document.save(update_fields=["status"])
        raise

    except Exception as e:
        logger.error(
            f"[Task] Document {document_id} indexing failed: {e}",
            exc_info=True,
        )

        document = Document.objects.filter(pk=document_id).first()

        if document:
            document.status = "failed"
            document.save(update_fields=["status"])

        raise self.retry(
            exc=e,
            countdown=60 * (2 ** self.request.retries)
        )