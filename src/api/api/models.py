from pydantic import BaseModel, Field
from typing import List, Any, Optional, Union
from enum import Enum

class EmbeddingType(str, Enum):
    TEXT = "text"
    IMAGE = "image"

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    embedding_type: Optional[EmbeddingType] = Field('text', description="The embedding mode to be used in the RAG pipeline")
    fusion: Optional[bool] = Field(True, description="Whether to use fusion of payload index and embeddings")
    thread_id: str = Field(..., description="The thread ID to be used in the RAG pipeline")

class RAGItem(BaseModel):
    image_url: str = Field(..., description="The URL of the image of the item")
    price: Optional[float] = Field(..., description="The price of the item")
    # description: str = Field(..., description="The description of the item", max_length=120)
    description: str = Field(..., description="The description of the item")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The content of the RAG response")
    items: List[RAGItem] = Field(..., description="The items that were used to answer the question")
    trace_id: str = Field(..., description="The trace ID")
    # used_context_count: int = Field(..., description="The count of the items that were used to answer the question")
    # not_used_context_count: int = Field(..., description="The count of the items that were not used to answer the question")


class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None] = Field(..., description="1 if the feedback is positive, 0 if it is negative")
    feedback_text: str = Field(..., description="The feedback text")
    feedback_source_type: str = Field(..., description="The type of the feedback. Human or API")
    trace_id: str = Field(..., description="The trace ID")
    thread_id: str = Field(..., description="The thread ID")


class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    status: str = Field(..., description="The status of the feedback submission")