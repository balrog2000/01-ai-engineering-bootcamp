from fastapi import APIRouter, Request
import logging
from api.api.models import RAGRequest, RAGResponse, RAGItem
from api.rag.graph import run_agent_wrapper
from api.rag.tools import get_reviews, Review
from typing import List
logger = logging.getLogger(__name__)
api_router = APIRouter()
rag_router = APIRouter()

@rag_router.post("/rag")
async def rag(
    request: Request, 
    payload: RAGRequest
) -> RAGResponse:

    # result = run_agent_wrapper(payload.query, embedding_type=payload.embedding_type, fusion=payload.fusion)
    result = run_agent_wrapper(payload.query, thread_id=payload.thread_id)
    items = [RAGItem(
        image_url=item['image_url'],
        price=item['price'],
        description=item['description'],
        parent_asin=item['parent_asin']
    ) for item in result['items']]

    return RAGResponse( 
        request_id=request.state.request_id,
        answer=result['answer'],
        items=items,
        # used_context_count=result['used_context_count'],
        # not_used_context_count=result['not_used_context_count']
    )
api_router.include_router(rag_router, tags=["rag"])


dev_router = APIRouter()
@dev_router.get("/dev")
async def dev() -> List[Review]:
    reviews = get_reviews([1,5,7])
    return reviews


api_router.include_router(dev_router, tags=["dev"])