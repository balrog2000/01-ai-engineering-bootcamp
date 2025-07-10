from pydantic import BaseModel, Field
from typing import List, Any, Optional


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The content of the RAG response")
