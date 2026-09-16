from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from openai import AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError, UnprocessableEntityError
from ai_handler.openrouter import MissingAPIKeyError
from .models import Document
from common.runtime.errors import UnsupportedConfiguration
from common.runtime.context import use_runtime
from common.runtime.handoff import load_runtime
from rag.rag_service import get_registry
from django.db import transaction
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from common.runtime.api_keys import scrub

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
    logged on failure. An absent token uses the server configuration; an
    expired issued token fails explicitly so the user can retry the upload.

    The token is deliberately not discarded on success: Celery can redeliver an
    acks_late task, and the entry expires on its own.
    """
    runtime = None
    try:

        with transaction.atomic():

            document = (
                Document.objects
                .select_for_update()
                .get(pk=document_id, user__username=username)
            )

            # Prevent duplicate execution
            if document.status == "ready":
                logger.info(
                    f"[build_index_task] Document {document.pk} already ready"
                )
                return

            if document.status == "indexing" and document.updated_at > timezone.now() - timedelta(seconds=settings.CELERY_TASK_TIME_LIMIT):
                logger.info(
                    f"[build_index_task] Document {document.pk} already indexing"
                )
                return

            document.error_message = ""
            document.status = "indexing"
            document.save(update_fields=["status", "error_message", "updated_at"])

        # OUTSIDE transaction
        # do heavy processing here
        # Scoped to this task: a worker thread indexes one document after
        # another, and leaving the credentials installed would spend the wrong
        # user's key on the next one.
        runtime = load_runtime(runtime_token)
        with use_runtime(runtime):
            pipeline = get_registry().get_engine()
            pipeline._build_index(
                username=username,
                document=document
            )

        document.error_message = ""
        document.status = "ready"
        document.save(update_fields=["status", "error_message", "updated_at"])

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
            document.error_message = "Indexing timed out. Try a smaller file or different chunk settings."
            document.save(update_fields=["status", "error_message", "updated_at"])
        # Don't retry on timeout — the task is inherently too slow
        raise

    except (UnsupportedConfiguration, MissingAPIKeyError, AuthenticationError,
            BadRequestError, NotFoundError, PermissionDeniedError, UnprocessableEntityError) as e:
        # The upload asked for something this deployment cannot do. Retrying
        # produces the same failure four minutes later, so fail it now and let
        # the user change the setting and re-upload.
        logger.error(
            f"[Task] Document {document_id} cannot be indexed as configured: {e}"
        )
        document = Document.objects.filter(pk=document_id).first()
        if document:
            document.status = "failed"
            document.error_message = scrub(str(e), runtime.keys if runtime else {})
            document.save(update_fields=["status", "error_message", "updated_at"])
        raise

    except Exception as e:
        logger.error(
            f"[Task] Document {document_id} indexing failed: {e}",
            exc_info=True,
        )

        document = Document.objects.filter(pk=document_id).first()

        if document:
            retrying = self.request.retries < self.max_retries
            document.status = "pending" if retrying else "failed"
            document.error_message = ("Indexing will retry. " if retrying else "Indexing failed. ") + scrub(str(e), runtime.keys if runtime else {})
            document.save(update_fields=["status", "error_message", "updated_at"])

        raise self.retry(
            exc=e,
            countdown=60 * (2 ** self.request.retries)
        )
