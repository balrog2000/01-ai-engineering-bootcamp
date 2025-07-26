from pydantic import BaseModel, Field
from typing import List, Any, Optional
from enum import Enum

class EmbeddingType(str, Enum):
    TEXT = "text"
    IMAGE = "image"

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    embedding_type: EmbeddingType = Field(..., description="The embedding mode to be used in the RAG pipeline")
    fusion: bool = Field(..., description="Whether to use fusion of payload index and embeddings")
    thread_id: str = Field(..., description="The thread ID to be used in the RAG pipeline")

class RAGItem(BaseModel):
    image_url: str = Field(..., description="The URL of the image of the item")
    price: Optional[float] = Field(..., description="The price of the item")
    # description: str = Field(..., description="The description of the item", max_length=120)
    description: str = Field(..., description="The description of the item")
    parent_asin: str = Field(..., description="The parent ASIN of the item") 

class RAGResponse(BaseModel): 
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The content of the RAG response")
    items: List[RAGItem] = Field(..., description="The items that were used to answer the question")
    # used_context_count: int = Field(..., description="The count of the items that were used to answer the question")
    # not_used_context_count: int = Field(..., description="The count of the items that were not used to answer the question")
