from fastapi import APIRouter, Request
import logging
from api.api.models import RAGRequest, RAGResponse, RAGItem, FeedbackRequest, FeedbackResponse
from api.processors.submit_feedback import submit_feedback
from api.rag.graph import run_agent_wrapper
logger = logging.getLogger(__name__)
rag_router = APIRouter()
feedback_router = APIRouter()

@rag_router.post("/rag")
async def rag(
    request: Request, 
    payload: RAGRequest
) -> RAGResponse:

    # result = run_agent_wrapper(payload.query, embedding_type=payload.embedding_type, fusion=payload.fusion)
    result = await run_agent_wrapper(payload.query, thread_id=payload.thread_id)
    items = [RAGItem(
        image_url=item['image_url'],
        price=item['price'],
        description=item['description']
    ) for item in result['items']]

    logger.info(f"RAG response: {result}")

    return RAGResponse(
        request_id=request.state.request_id,
        answer=result['answer'],
        items=items,
        trace_id=result['trace_id'], 
        # used_context_count=result['used_context_count'],
        # not_used_context_count=result['not_used_context_count']
    )

@feedback_router.post("/submit_feedback")
async def send_feedback(
    request: Request, 
    payload: FeedbackRequest
) -> FeedbackResponse:

    submit_feedback(
        trace_id=payload.trace_id,
        feedback_score=payload.feedback_score,
        feedback_text=payload.feedback_text,
        feedback_source_type=payload.feedback_source_type,
    )

    return FeedbackResponse(
        request_id=request.state.request_id,
        status="success"
    )


api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])
api_router.include_router(feedback_router, tags=["feedback"])