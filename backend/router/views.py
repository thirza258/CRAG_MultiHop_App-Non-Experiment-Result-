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
from common.memory import compute_file_hash, compute_text_hash
from common.pipeline_config import normalize_pipeline_config
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
                    username=username
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
                    username=username
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
                    username=username
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
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            username = serializer.validated_data["USER"]
            query = serializer.validated_data["QUERY"]
            pipeline_config = normalize_pipeline_config(
                serializer.validated_data.get("CONFIG")
            )

            conversation = self.save_conversation(username, query, "", "")

            # conversation_id was previously omitted here, which raised TypeError
            # because run() required it — this endpoint always 500'd.
            answer = get_registry().get_engine().run(
                username,
                query,
                conversation_id=conversation.pk,
                config=pipeline_config,
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
                return get_responses().response_500(error=str(e))
        
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