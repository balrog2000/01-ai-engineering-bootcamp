from operator import add
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue

from pydantic import BaseModel, Field

from api.rag.agent import RAGUsedContext, ToolCall
from api.rag.utils.utils import get_tool_descriptions_from_mcp_servers, mcp_tool_node
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from api.rag.agent import agent_node
from api.core.config import config
from qdrant_client import QdrantClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.pretty import pretty_repr, pprint
import logging

# Create a file for logging output
log_file_path = "logs/graph.log"

# Set up a Console that writes to the log file
log_file = open(log_file_path, "a", encoding="utf-8")
file_console = Console(file=log_file, force_terminal=True, color_system="auto", theme=Theme({}), width=190, soft_wrap=True)

graph_logger = logging.getLogger("graph")
graph_logger.propagate = False  # Prevent logging to stdout/stderr
if not graph_logger.handlers:
    rich_handler = RichHandler(
        console=file_console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
    )
    graph_logger.addHandler(rich_handler)
    graph_logger.setLevel(logging.INFO)

class State(BaseModel):
    messages: Annotated[List[Any], add] = [] # reducer (it will add messages to the list)
    iteration: int = Field(default=0)
    answer: str = ""
    final_answer: bool = Field(default=False)
    available_tools: List[Dict[str, Any]] = []
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    retrieved_context: List[RAGUsedContext] = []
    trace_id: str = ""


def tool_router(state: State) -> str:
    """Decide whether to continue or end"""

    if state.final_answer:
        return 'end'
    elif state.iteration > 5:
        return 'end'
    elif len(state.tool_calls) > 0:
        return 'tools'
    else:
        return 'end'

workflow = StateGraph(State)


workflow.add_node('agent_node', agent_node)
workflow.add_node('mcp_tool_node', mcp_tool_node)

workflow.add_edge(START, "agent_node")
workflow.add_conditional_edges(
    'agent_node',
    tool_router,
    {
        'tools': 'mcp_tool_node',
        'end': END
    }
)
workflow.add_edge('mcp_tool_node', 'agent_node')

async def run_agent(question: str, thread_id: str):

    mcp_servers = [
        'http://items_mcp_server:8000/mcp/',
        'http://reviews_mcp_server:8000/mcp/',
    ]

    tool_descriptions = await get_tool_descriptions_from_mcp_servers(mcp_servers)
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        'iteration': 0,
        "available_tools": tool_descriptions
    }

    checkpointer_config = {'configurable': {'thread_id': thread_id}}

    async with AsyncPostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        result = None 
        async for mode, chunk in graph.astream(initial_state, stream_mode=["values", "updates"],config=checkpointer_config):   
            if mode == "updates":
                graph_logger.info(pretty_repr(chunk))
            if mode == "values":
                result = chunk

    return result

async def run_agent_wrapper(question: str, thread_id: str):
    qdrant_client = QdrantClient(
        url=f'http://{config.QDRANT_HOST}:6333'
    )

    result = await run_agent(question, thread_id)

    items = []
    dummy_vector = np.zeros(1536).tolist()
    for context in result['retrieved_context']:
        payload = qdrant_client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS,
            query=dummy_vector,
            query_filter=Filter(
                must=FieldCondition(
                    key="parent_asin",
                    match=MatchValue(
                        value=context.id
                    )
                )
            ),
            with_payload=True,
            limit=1
        )
        payload = payload.points[0].payload
        
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
        'trace_id': result['trace_id'],
    }