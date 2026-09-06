import json
import threading
import uuid
import logging

from django.db import transaction, IntegrityError
from django.core.cache import cache
import redis
from rest_framework.views import APIView
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from ai_handler import model_catalog
from common.memory import compute_file_hash, compute_text_hash
from common.runtime import api_keys as api_keys_module
from common.runtime.api_keys import normalize_api_keys
from common.runtime.config import normalize_pipeline_config
from common.runtime.context import RuntimeSettings
from common.runtime.handoff import stash_runtime
from utils.insert_file import get_loader
from router.models import (
    Document,
    GuestUser,
    Conversation, ConversationHistory, UserCollection, DocumentChunk
)
from router.tasks import build_index_task

from rag.rag_service import get_registry
from router.serializers import (
    InsertDataSerializer, 
    InsertTextSerializer, 
    InsertURLSerializer, 
    QuerySerializer 
)
from common.schema import get_responses
from chroma.chroma_settings import get_chroma_client


logger = logging.getLogger(__name__)


def _runtime_token(serializer):
    """Stash this request's model choice and API keys for the indexing worker.

    Returns a token to hand to ``build_index_task``, or None when there is
    nothing to hand over (or Redis is unavailable) — the worker then uses the
    server's own configuration, exactly as indexing always did.

    Only the embedding model really matters here: it decides which vector space
    the document lands in, and a collection can only hold one.
    """
    config = normalize_pipeline_config(serializer.validated_data.get("CONFIG"))
    keys = normalize_api_keys(serializer.validated_data.get("KEYS"))
    return stash_runtime(RuntimeSettings.from_wire(config, keys))


class InsertDataView(GenericAPIView):
    serializer_class = InsertDataSerializer
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            username = serializer.validated_data["USER"]
            file = serializer.validated_data["FILE"]

            user = GuestUser.objects.get(username=username)

            file_hash = compute_file_hash(file)

            data = get_loader().process_input(file, username)

            try:
                with transaction.atomic():
                    document, created = Document.objects.get_or_create(
                        user=user,
                        file_hash=file_hash,
                        defaults={
                            "name": data["filename"],
                            "source_type": "pdf",
                            "source_path": data["source_path"],
                            "extracted_text_path": data["text_path"],
                            "status": "pending",
                        }
                    )

            except IntegrityError:
                document = Document.objects.get(
                    user=user,
                    file_hash=file_hash
                )
                created = False

            if not created:
                return get_responses().response_200(
                    f"Document already indexed "
                    f"(id={document.pk}, status={document.status})"
                )

            try:
                task = build_index_task.delay(
                    document_id=document.pk,
                    username=username,
                    runtime_token=_runtime_token(serializer),
                )
            except Exception as exc:
                logger.error(
                    "[%s] Failed to enqueue indexing task for document %s: %s",
                    self.__class__.__name__, document.pk, exc,
                    exc_info=True,
                )
                # Mark the document so the user can see something went wrong
                # and the sidebar doesn't show it stuck on "pending" forever.
                document.status = "failed"
                document.save(update_fields=["status"])
                return get_responses().response_500(
                    error="Background indexing could not be started. "
                          "Please try again in a moment."
                )

            return get_responses().response_200(
                "Data inserted successfully!"
            )

        except Exception as e:
            return get_responses().response_500(error=str(e))

class InsertURLView(GenericAPIView):
    serializer_class = InsertURLSerializer

    def post(self, request):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            username = serializer.validated_data["USER"]
            # Reads URL, not FILE: this view's serializer has no FILE field, so the
            # old lookup raised KeyError and every URL submission became a 500.
            url = serializer.validated_data["URL"]

            user = GuestUser.objects.get(username=username)

            file_hash = compute_text_hash(url)

            data = get_loader().process_input(url, username)

            try:
                with transaction.atomic():
                    document, created = Document.objects.get_or_create(
                        user=user,
                        file_hash=file_hash,
                        defaults={
                            "name": data["name"],
                            "source_type": data["source_type"],
                            "source_path": data["source_path"],
                            "extracted_text_path": data["text_path"],
                            "status": "pending",
                        }
                    )

            except IntegrityError:
                document = Document.objects.get(
                    user=user,
                    file_hash=file_hash
                )
                created = False

            if not created:
                return get_responses().response_200(
                    f"Document already indexed "
                    f"(id={document.pk}, status={document.status})"
                )

            try:
                task = build_index_task.delay(
                    document_id=document.pk,
                    username=username,
                    runtime_token=_runtime_token(serializer),
                )
            except Exception as exc:
                logger.error(
                    "[%s] Failed to enqueue indexing task for document %s: %s",
                    self.__class__.__name__, document.pk, exc,
                    exc_info=True,
                )
                # Mark the document so the user can see something went wrong
                # and the sidebar doesn't show it stuck on "pending" forever.
                document.status = "failed"
                document.save(update_fields=["status"])
                return get_responses().response_500(
                    error="Background indexing could not be started. "
                          "Please try again in a moment."
                )

            return get_responses().response_200(
                "Data inserted successfully!"
            )

        except Exception as e:
            return get_responses().response_500(error=str(e))

class InsertTextView(GenericAPIView):
    serializer_class = InsertTextSerializer

    def post(self, request):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            username = serializer.validated_data["USER"]
            # Reads TEXT, not FILE — same defect as InsertURLView: the serializer
            # defines TEXT, so the old lookup made every paste a 500.
            text = serializer.validated_data["TEXT"]

            user = GuestUser.objects.get(username=username)

            file_hash = compute_text_hash(text)

            data = get_loader().process_input(text, username)

            try:
                with transaction.atomic():
                    document, created = Document.objects.get_or_create(
                        user=user,
                        file_hash=file_hash,
                        defaults={
                            "name": data["name"],
                            "source_type": "text",
                            "source_path": data["source_path"],
                            "extracted_text_path": data["text_path"],
                            "status": "pending",
                        }
                    )

            except IntegrityError:
                document = Document.objects.get(
                    user=user,
                    file_hash=file_hash
                )
                created = False

            if not created:
                return get_responses().response_200(
                    f"Document already indexed "
                    f"(id={document.pk}, status={document.status})"
                )

            try:
                task = build_index_task.delay(
                    document_id=document.pk,
                    username=username,
                    runtime_token=_runtime_token(serializer),
                )
            except Exception as exc:
                logger.error(
                    "[%s] Failed to enqueue indexing task for document %s: %s",
                    self.__class__.__name__, document.pk, exc,
                    exc_info=True,
                )
                # Mark the document so the user can see something went wrong
                # and the sidebar doesn't show it stuck on "pending" forever.
                document.status = "failed"
                document.save(update_fields=["status"])
                return get_responses().response_500(
                    error="Background indexing could not be started. "
                          "Please try again in a moment."
                )

            return get_responses().response_200(
                "Data inserted successfully!"
            )

        except Exception as e:
            return get_responses().response_500(error=str(e))
        
class DocumentView(APIView):
    def get(self, request, username):
        try:
            user = GuestUser.objects.filter(username=username).first()
            if not user:
                return get_responses().response_404(error="User not found")
            
            documents = Document.objects.filter(user=user)
            if not documents.exists():
                return get_responses().response_404(error="Document not found for user")
            
            data = [{
                "id": document.pk,
                "name": document.name,
                "source_type": document.source_type,
                "source_path": document.source_path,
                "extracted_text_path": document.extracted_text_path[:100],
                "created_at": document.created_at
            } for document in documents]
            return get_responses().response_200(response=data)
        except Exception as e:
            return get_responses().response_500(error=str(e))

class ConversationView(GenericAPIView):
    def get(self, request, conversation_id):
        try:
            conversation = Conversation.objects.get(id=conversation_id)
            data = {
                "id": conversation.pk,
                "query": conversation.query,
                "response": conversation.response,
                "context": conversation.context,
                "created_at": conversation.created_at
            }
            return get_responses().response_200(response=data)
        except Conversation.DoesNotExist:
            return get_responses().response_404(error="Conversation not found")
        except Exception as e:
            return get_responses().response_500(error=str(e))
  
class ConversationHistoryView(GenericAPIView):
    def get(self, request, username):
        try:
            user = GuestUser.objects.get(username=username)
            conversation_histories = ConversationHistory.objects.filter(user=user).select_related('conversation').order_by('-created_at')
            data = [{
                "query": history.conversation.query,
                "response": history.conversation.response,
                "created_at": history.created_at
            } for history in conversation_histories]
            return get_responses().response_200(response=data)
        except GuestUser.DoesNotExist:
            return get_responses().response_404(error="User not found")
        except Exception as e:
            return get_responses().response_500(error=str(e))

class CorpusInfoView(APIView):
    """GET /api/v1/corpus/<username>/ — the embedding-model state of each corpus.

    The embedding model is the one setting the user cannot freely change per
    query: a collection's vectors all came from one model, and embedding a query
    with a different one compares across vector spaces. So the settings panel has
    to be able to say *why* the picker is locked, before the user runs a query
    and gets told after the fact.

    Reports, for the user's own collection and for the shared corpus, which model
    built it and whether the choice is still open. Never 404s on an empty
    collection — "you have no documents yet, so your choice applies" is exactly
    the state the panel needs to render.
    """

    #: Used when the pipeline cannot be reached to ask it. Matches the
    #: collection_name every deployment configures in rag/rag_service.py.
    FALLBACK_BASE_COLLECTION = "ragreader_collection"

    def get(self, request, username):
        configured = ""
        base_name = self.FALLBACK_BASE_COLLECTION
        try:
            engine = get_registry().get_engine()
            configured = engine._configured_embedding_model()
            base_name = engine._dataset_collection_name()
        except Exception as e:
            # The panel is still useful without this; a pipeline that will not
            # start is reported by the health endpoint, not here.
            logger.warning("[CorpusInfoView] could not read the pipeline config: %s", e)

        user_collection = UserCollection.objects.filter(
            user__username=username
        ).first()

        chunk_count = getattr(user_collection, "chunk_count", 0) or 0
        user_model = getattr(user_collection, "embedding_model", "") or ""
        # Locked once there are vectors to be inconsistent with. Until then the
        # first document indexed decides, so the choice is still the user's.
        user_locked = bool(chunk_count > 0 and user_model)

        base_record = ChromaCollection.objects.filter(
            collection_name=base_name
        ).first()
        base_model = (getattr(base_record, "embedding_model", "") or "") or configured

        return Response(
            {
                "user_corpus": {
                    "collection_name": getattr(user_collection, "collection_name", None),
                    "embedding_model": user_model,
                    "chunk_count": chunk_count,
                    "locked": user_locked,
                },
                "base_corpus": {
                    "collection_name": base_name,
                    "embedding_model": base_model,
                    # Always locked: nobody can re-index the shared corpus from
                    # the chat UI, so this choice is never theirs to make.
                    "locked": bool(base_model),
                },
                "configured_default": configured,
            },
            status=status.HTTP_200_OK,
        )


class ModelCatalogView(APIView):
    """GET /api/v1/models/ — what the model pickers offer.

    Proxied rather than fetched from the browser so the response can be cached
    once for every user, and so the picker does not depend on OpenRouter's CORS
    policy or rate limits. Neither upstream endpoint needs a key, so this works
    before the user has entered one.

    ``?refresh=1`` bypasses the cache, for when a model was just released.
    """

    def get(self, request):
        refresh = str(request.query_params.get("refresh", "")).lower() in (
            "1", "true", "yes",
        )
        try:
            catalog = model_catalog.get_catalog(force_refresh=refresh)
        except Exception as e:
            # get_catalog already falls back internally; this is the belt to its
            # braces, because a settings panel that cannot list models is a
            # settings panel the user cannot use.
            logger.error("[ModelCatalogView] catalog lookup failed: %s", e, exc_info=True)
            catalog = {
                "chat": list(model_catalog.FALLBACK_CATALOG["chat"]),
                "embedding": list(model_catalog.FALLBACK_CATALOG["embedding"]),
                "source": "fallback",
            }

        return Response(catalog, status=status.HTTP_200_OK)


class QueryView(GenericAPIView):
    serializer_class = QuerySerializer
    
    def save_conversation(self, username: str, query: str, answer: str, context: str ) -> Conversation:
        user = GuestUser.objects.get(username=username)
        
        conversation = Conversation.objects.create(
            user=user,
            query=query,
            response=answer,
            context=context
        )
        ConversationHistory.objects.create(
            user=user,
            conversation=conversation
        )
        return conversation

    def post(self, request):
        # Bound up front so the error path can always scrub against it, even when
        # validation fails before the keys are read.
        api_keys = {}
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            username = serializer.validated_data["USER"]
            query = serializer.validated_data["QUERY"]
            pipeline_config = normalize_pipeline_config(
                serializer.validated_data.get("CONFIG")
            )
            api_keys.update(normalize_api_keys(serializer.validated_data.get("KEYS")))

            conversation = self.save_conversation(username, query, "", "")

            # conversation_id was previously omitted here, which raised TypeError
            # because run() required it — this endpoint always 500'd.
            answer = get_registry().get_engine().run(
                username,
                query,
                conversation_id=conversation.pk,
                config=pipeline_config,
                keys=api_keys,
            )
            
            retrieved_chunks = answer.get("context", [])
            llm_answer = answer.get("answer", "")
            
            context_str = "\n\n".join(
                doc["text"] if isinstance(doc, dict) else doc
                for doc in retrieved_chunks
            )
            answer_record = self.save_conversation(username, query, llm_answer, context_str)
            
            answer["conversation_id"] = answer_record.pk
            
            return get_responses().response_200(response=answer)
        except Exception as e:
                # Provider SDKs quote the failing request, Authorization header
                # included, into their exception messages — and this one is
                # rendered straight into the HTTP response body.
                return get_responses().response_500(
                    error=api_keys_module.scrub(str(e), api_keys)
                )
        
class DeleteDocumentView(GenericAPIView):
    def delete(self, request, document_id, username):
        try:
            document = Document.objects.get(pk=document_id)
            user_collection = UserCollection.objects.get(user__username=username)

            chunk_records = DocumentChunk.objects.filter(
                document=document,
                user_collection=user_collection
            )
            chroma_ids = list(chunk_records.values_list("chroma_id", flat=True))

            if chroma_ids:
                collection = get_chroma_client(collection_name=user_collection.collection_name)
                collection.delete(ids=chroma_ids)

                deleted_count = len(chroma_ids)
                user_collection.chunk_count = max(0, user_collection.chunk_count - deleted_count)

                if user_collection.chunk_count == 0:
                    # The collection is empty, so it is no longer pinned to a
                    # vector space. Clearing the record is what lets the user
                    # pick a different embedding model — the next document they
                    # index adopts their current choice. Without this, the model
                    # would stay locked forever after the first upload.
                    user_collection.embedding_model = ""

                user_collection.save()
                chunk_records.delete()
            document.delete()
            return get_responses().response_200(response="Document and associated chunks deleted successfully")
        except Document.DoesNotExist:
            return get_responses().response_404(error="Document not found")
        except UserCollection.DoesNotExist:
            return get_responses().response_404(error="User collection not found")
        except Exception as e:
            return get_responses().response_500(error=str(e))