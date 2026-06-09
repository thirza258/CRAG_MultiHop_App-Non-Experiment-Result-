from celery import shared_task
from .models import DocumentChunk, Job, AnalysisBatch, AnalysisResult, Document
from rag.rag_service import rag_registry, get_registry
from django.db import transaction

import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def initialize_rag_task(self, job_id, username, method, model_config):
    try:
        job = Job.objects.get(id=job_id)
        job.status = Job.Status.PROCESSING
        job.save()

        engine = rag_registry.get_engine()
      
        engine.init(username, job=job)

        job.status = Job.Status.READY
        job.progress = 100
        job.save()

        return True

    except Exception as e:
        if 'job' in locals():
            job.mark_failed(str(e))
            return False
        return False

@shared_task(bind=True)
def run_single_analysis(self, batch_id, username, query, variant_config):
    try:
        engine = rag_registry.get_engine(variant_config["method"], variant_config["model"])
        response = engine.run(username, query)

        context = response.get("context", [])  
        retrieved_chunks = [
            {
                "id": doc["chunk_id"],
                "text": doc["text"],
                "score": doc.get("score")
            }
            for doc in context
        ]

        metrics = [
            {
                "name": "retrieval_score",
                "value": [
                    {"chunk_id": doc["chunk_id"], "score": doc.get("score")}
                    for doc in context
                ]
            }
        ]

        AnalysisResult.objects.create(
            query=query,
            batch_id=batch_id,
            method=variant_config["method"],
            ai_model=variant_config["model"],
            answer=response.get("answer", ""),
            retrieved_chunks=retrieved_chunks,
            evaluation_metrics=metrics
        )
        return True
    except Exception as e:
        return False
    

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