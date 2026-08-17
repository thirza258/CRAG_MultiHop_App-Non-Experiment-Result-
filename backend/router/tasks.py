from celery import shared_task
from .models import DocumentChunk, Document
from rag.rag_service import get_registry
from django.db import transaction

import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def build_index_task(self, document_id: int, username: str):
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

    except Exception as e:

        document = Document.objects.filter(pk=document_id).first()

        if document:
            document.status = "failed"
            document.save(update_fields=["status"])

        raise self.retry(
            exc=e,
            countdown=60 * (2 ** self.request.retries)
        )