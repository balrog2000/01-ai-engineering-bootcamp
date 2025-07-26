from operator import add
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from api.rag.agent import RAGUsedContext, ToolCall
from api.rag.tools import get_formatted_context, get_reviews
from api.rag.utils.utils import get_tool_descriptions_from_node
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from api.rag.agent import agent_node
from api.core.config import config
from qdrant_client import QdrantClient
from langgraph.checkpoint.postgres import PostgresSaver
from rich.pretty import pprint, pretty_repr
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

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

tools = [get_formatted_context, get_reviews]

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

def run_agent(question: str, thread_id: str):
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        'iteration': 0,
        "available_tools": tool_descriptions
    }

    checkpointer_config = {'configurable': {'thread_id': thread_id}}

    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        result = None
        for mode, chunk in graph.stream(initial_state, stream_mode=["values", "updates"],config=checkpointer_config):   
            if mode == "updates":
                graph_logger.info(pretty_repr(chunk))
            if mode == "values":
                result = chunk

    return result

def run_agent_wrapper(question: str, thread_id: str):
    qdrant_client = QdrantClient(
        url=f'http://{config.QDRANT_HOST}:6333'
    )

    result = run_agent(question, thread_id)

    items = []
    for context in result['retrieved_context']:
        payload = qdrant_client.retrieve(
            # collection_name=config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS if embedding_type == EmbeddingType.TEXT else config.QDRANT_COLLECTION_NAME_IMAGE_EMBEDDINGS,
            collection_name=config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS,
            ids=[context.id],
        )[0].payload
        
        image_url = payload.get('first_large_image')
        price = payload.get('price')
        parent_asin = payload.get('parent_asin')
        if image_url:
            items.append(
                {
                    'image_url': image_url,
                    'price': price,
                    'description': context.description,
                    'parent_asin': parent_asin,
                }
            )


    return {
        'answer': result['answer'],
        'items': items,
    }