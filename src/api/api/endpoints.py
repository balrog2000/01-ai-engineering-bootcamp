from fastapi import APIRouter, Request
import logging
from api.api.models import RAGRequest, RAGResponse, RAGItem
from api.rag.retrieval import rag_pipeline_wrapper
logger = logging.getLogger(__name__)
rag_router = APIRouter()

@rag_router.post("/rag")
async def rag(
    request: Request, 
    payload: RAGRequest
) -> RAGResponse:

    result = rag_pipeline_wrapper(payload.query, embedding_type=payload.embedding_type, fusion=payload.fusion)
    items = [RAGItem(
        image_url=item['image_url'],
        price=item['price'],
        description=item['description']
    ) for item in result['items']]

    return RAGResponse(
        request_id=request.state.request_id,
        answer=result['answer'],
        items=items
    )


api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])