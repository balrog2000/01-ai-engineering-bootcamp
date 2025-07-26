from operator import add
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from api.rag.agent import RAGUsedContext, ToolCall
from api.rag.tools import get_formatted_context
from api.rag.utils.utils import get_tool_descriptions_from_node
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from api.rag.agent import agent_node
from api.core.config import config
from qdrant_client import QdrantClient

class State(BaseModel):
    messages: Annotated[List[Any], add] = [] # reducer (it will add messages to the list)
    iteration: int = Field(default=0)
    answer: str = ""
    final_answer: bool = Field(default=False)
    available_tools: List[Dict[str, Any]] = []
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    retrieved_context: Annotated[List[RAGUsedContext], add] = []


def tool_router(state: State) -> str:
    """Decide whether to continue or end"""

    if state.final_answer:
        return 'end'
    elif state.iteration > 3:
        return 'end'
    elif len(state.tool_calls) > 0:
        return 'tools'
    else:
        return 'end'

workflow = StateGraph(State)

tools = [get_formatted_context]

tool_node = ToolNode(tools)

tool_descriptions = get_tool_descriptions_from_node(tool_node)

workflow.add_node('agent_node', agent_node)
workflow.add_node('tool_node', tool_node)

workflow.add_edge(START, "agent_node")
workflow.add_conditional_edges(
    'agent_node',
    tool_router,
    {
        'tools': 'tool_node',
        'end': END
    }
)
workflow.add_edge('tool_node', 'agent_node')

graph = workflow.compile()

def run_agent(question: str):
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "available_tools": tool_descriptions
    }

    result = graph.invoke(initial_state)

    return result

def run_agent_wrapper(question: str):
    qdrant_client = QdrantClient(
        url=f'http://{config.QDRANT_HOST}:6333'
    )

    result = run_agent(question)

    items = []
    for context in result['retrieved_context']:
        payload = qdrant_client.retrieve(
            # collection_name=config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS if embedding_type == EmbeddingType.TEXT else config.QDRANT_COLLECTION_NAME_IMAGE_EMBEDDINGS,
            collection_name=config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS,
            ids=[context.id],
        )[0].payload
        
        image_url = payload.get('first_large_image')
        price = payload.get('price')
        if image_url:
            items.append(
                {
                    'image_url': image_url,
                    'price': price,
                    'description': context.description,
                }
            )


    return {
        'answer': result['answer'],
        'items': items,
    }