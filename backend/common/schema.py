from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List
import time
from django.http import JsonResponse

class VoteDecision(BaseModel):
    decision: Literal["yes", "no"] = Field(
        ..., 
        description="Select 'yes' if the response is accurate based on the chunk, 'no' otherwise."
    )
    justification: str = Field(
        ..., 
        description="A concise one-sentence explanation of why you voted this way."
    )

class DeepAnalysisResponse(BaseModel):
    response: str = Field(
        ...,
        description="The response to the query."
    )
    retrieved_chunks: List[str] = Field(
        ...,
        description="The chunks retrieved from the vector store."
    )
    query: str = Field(
        ...,
        description="The query that was used to retrieve the chunks."
    )
    method: str = Field(
        ...,
        description="The method that was used to retrieve the chunks."
    )
    ai_model: str = Field(
        ...,
        description="The AI model that was used to retrieve the chunks."
    )
    evaluation_metrics: List[Dict] = Field(
        ...,
        description="The evaluation metrics."
    )

class RAGResponse(BaseModel):
    @staticmethod
    def response_200(response: str | dict):
        return JsonResponse({
            "status": 200,
            "message": "Response generated successfully",
            "timestamp": time.time(),
            "data": response
        }
        , status=200
        , content_type="application/json"
        )
    
    @staticmethod
    def response_202(message: dict | str): 
        return JsonResponse({
            "status": 202,
            "message": "Chat initialization started.",
            "timestamp": time.time(),
            "data": message
        }
        , status=202
        , content_type="application/json"
        )

    @staticmethod
    def response_400(error: str):
        return JsonResponse({
            "status": 400,
            "message": error,
            "timestamp": time.time(),
            "data": None
        }
        , status=400
        , content_type="application/json"
        )

    @staticmethod
    def response_500(error: str):
        return JsonResponse({
            "status": 500,
            "message": error,
            "timestamp": time.time(),
            "data": None
        }
        , status=500
        , content_type="application/json"
        )

    
    def response_404(error: str):
        return JsonResponse({
            "status": 404,
            "message": error,
            "timestamp": time.time(),
            "data": None
        }
        , status=404
        , content_type="application/json"
        )

    @staticmethod
    def response_201(response: str | dict):
        return JsonResponse({
            "status": 201,
            "message": "Object created successfully",
            "timestamp": time.time(),
            "data": response
        }
        , status=201
        , content_type="application/json"
        )

responses = RAGResponse()

_responses = None

def get_responses():
    global _responses
    if _responses is None:
        _responses = RAGResponse()
    return _responses