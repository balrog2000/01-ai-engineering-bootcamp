from fastapi import APIRouter, Request
import logging
from api.api.models import RAGRequest, RAGResponse
from api.rag.retrieval import rag_pipeline_wrapper
logger = logging.getLogger(__name__)
rag_router = APIRouter()

@rag_router.post("/rag")
async def rag(
    request: Request, 
    payload: RAGRequest
) -> RAGResponse:

    result = rag_pipeline_wrapper(payload.query)

    return RAGResponse(
        request_id=request.state.request_id,
        answer=result
    )


api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])