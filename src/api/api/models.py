from pydantic import BaseModel, Field
from typing import List, Any, Optional


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")

class RAGItem(BaseModel):
    image_url: str = Field(..., description="The URL of the image of the item")
    price: Optional[float] = Field(..., description="The price of the item")
    # description: str = Field(..., description="The description of the item", max_length=120)
    description: str = Field(..., description="The description of the item")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The content of the RAG response")
    items: List[RAGItem] = Field(..., description="The items that were used to answer the question")
