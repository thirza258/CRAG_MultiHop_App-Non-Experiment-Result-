from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from evaluation.models import Chunk, GroundTruthChunk, GroundTruthResponse
from router.models import Conversation, GuestUser, Document, AnalysisBatch, AnalysisResult
from common.chunker import DocumentChunker
from common.schema import get_responses

from utils.insert_file import DataLoader

import logging

logger = logging.getLogger(__name__)

class ChunkView(APIView):
    def get_document(self, username: str) -> Document | None:
        try:
            user = GuestUser.objects.filter(username=username).first()
            if not user:
                return None
            return Document.objects.filter(user=user).last()
        except Exception as e:
            logger.error(f"Error getting document for {username}: {e}")
            return None
        
    def create_chunk(self, document: Document, chunks, metadata: dict) -> Chunk:
        try:
            for chunk_text in chunks:
                chunk = Chunk.objects.create(document=document, text=chunk_text, metadata=metadata)
            return chunk
        except Exception as e:
            logger.error(f"Error creating chunk for document {document.id}: {e}")
            return None
        
    def post(self, request):
        try:
            username = request.data.get("USER")
            document = self.get_document(username)
            if not document:
                return Response({"error": "Document not found for user"}, status=status.HTTP_404_NOT_FOUND)
            
            config = {
                "chunk_strategy": "fixed",
                "chunk_size": 500,
                "overlap": 50,
                "embedding_client": None
            }
            
            document = self.get_document(username)
            chunker = DocumentChunker(
                strategy=config["chunk_strategy"],
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
                embedding_client=config["embedding_client"]
            )
            loader = DataLoader()
            extracted_text = loader.load(document.extracted_text_path)
            chunks = chunker.chunk(extracted_text)
            if not chunks:
                return Response({"error": "Failed to chunk document"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            if not document:
                return Response({"error": "Document not found for user"}, status=status.HTTP_404_NOT_FOUND)
            self.create_chunk(document, chunks, metadata=config)
            
            return get_responses().response_200("Chunk Created")
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

    def get(self, request, document_id):
        try:
            chunks = Chunk.objects.filter(document_id=document_id)
            chunk_data = [{"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]
            return Response({"chunks": chunk_data}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {document_id}: {e}")
            return Response({"error": "Failed to retrieve chunks"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

