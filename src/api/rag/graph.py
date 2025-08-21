from math import prod
from operator import add
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue

from pydantic import BaseModel, Field

from api.rag.agents import RAGUsedContext, ToolCall, MCPToolCall, Delegation
from api.rag.utils.utils import get_tool_descriptions_from_mcp_servers, mcp_tool_node, get_tool_descriptions_from_node
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from api.rag.agents import product_qa_agent_node, coordinator_agent_node, shopping_cart_agent_node
from api.core.config import config
from qdrant_client import QdrantClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.pretty import pretty_repr, pprint
import logging
from api.rag.tools import add_to_shopping_cart, remove_from_cart, get_shopping_cart
import os

# Ensure the logs directory exists to prevent errors
os.makedirs("logs", exist_ok=True)

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
    messages: Annotated[List[Any], add] = []
    answer: str = ""

    coordinator_iteration: int = Field(default=0)
    product_qa_iteration: int = Field(default=0)
    shopping_cart_iteration: int = Field(default=0)

    coordinator_final_answer: bool = Field(default=False)
    product_qa_final_answer: bool = Field(default=False)
    shopping_cart_final_answer: bool = Field(default=False)

    product_qa_available_tools: List[Dict[str, Any]] = []
    shopping_cart_available_tools: List[Dict[str, Any]] = []

    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    mcp_tool_calls: Optional[List[MCPToolCall]] = Field(default_factory=list)
    retrieved_context: List[RAGUsedContext] = Field(default_factory=list)
    
    user_id: str = ""
    cart_id: str = ""

    next_agent: str = ""
    plan: list[Delegation] = Field(default_factory=list)

    trace_id: str = ""


### Routers
def coordinator_router(state: State) -> str:
    """Decide whether to continue or end"""

    if state.coordinator_final_answer:
        return "end"
    elif state.coordinator_iteration > 4:
        return "end"
    elif state.next_agent == "product_qa_agent":
        return "product_qa_agent"
    elif state.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    else:
        return "end"

def product_qa_tool_router(state: State) -> str:
    
    if state.product_qa_final_answer:
        return "end"
    elif state.product_qa_iteration > 3:
        return "end"
    elif len(state.mcp_tool_calls) > 0:
        return "tools"
    else:
        return "end"

def shopping_cart_tool_router(state: State) -> str:
    if state.shopping_cart_final_answer:
        return "end"
    elif state.shopping_cart_iteration > 3:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"

shopping_cart_agent_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
shopping_cart_tool_node = ToolNode(shopping_cart_agent_tools)
shopping_cart_tool_descriptions = get_tool_descriptions_from_node(shopping_cart_tool_node)


workflow = StateGraph(State)
workflow.add_edge(START, "coordinator_agent_node")
workflow.add_node("coordinator_agent_node", coordinator_agent_node)
workflow.add_node("product_qa_agent_node", product_qa_agent_node)
workflow.add_node("shopping_cart_agent_node", shopping_cart_agent_node)
workflow.add_node("product_qa_tool_node", mcp_tool_node)
workflow.add_node("shopping_cart_tool_node", shopping_cart_tool_node)

workflow.add_conditional_edges(
    "coordinator_agent_node",
    coordinator_router,
    {
        "product_qa_agent": "product_qa_agent_node",
        "shopping_cart_agent": "shopping_cart_agent_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "product_qa_agent_node",
    product_qa_tool_router,
    {
        "tools": "product_qa_tool_node",
        "end": 'coordinator_agent_node'
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent_node",
    shopping_cart_tool_router,
    {
        "tools": "shopping_cart_tool_node",
        "end": 'coordinator_agent_node'
    }
)

workflow.add_edge("product_qa_tool_node", "product_qa_agent_node")
workflow.add_edge("shopping_cart_tool_node", "shopping_cart_agent_node")

async def run_agent(question: str, thread_id: str):

    mcp_servers = [
        'http://items_mcp_server:8000/mcp/',
        'http://reviews_mcp_server:8000/mcp/',
    ]

    product_qa_tool_descriptions = await get_tool_descriptions_from_mcp_servers(mcp_servers)
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        'user_id': thread_id,
        'cart_id': thread_id,

        'coordinator_iteration': 0,
        'product_qa_iteration': 0,
        'shopping_cart_iteration': 0,

        'coordinator_final_answer': False,
        'product_qa_final_answer': False,
        'shopping_cart_final_answer': False,

        "product_qa_available_tools": product_qa_tool_descriptions,
        "shopping_cart_available_tools": shopping_cart_tool_descriptions,
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
    for context in result.get('retrieved_context', []):
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
    shopping_cart = get_shopping_cart(thread_id, thread_id)
    shopping_cart_items = [{
        'price': x.get('price'),
        'quantity': x.get('quantity'),
        'currency': x.get('currency'),
        'product_image_url': x.get('product_image_url'),
        'total_price': x.get('total_price'),
    } for x in shopping_cart]


    return {
        'answer': result['answer'],
        'items': items,
        'shopping_cart': shopping_cart_items,
        'trace_id': result['trace_id'],
    }