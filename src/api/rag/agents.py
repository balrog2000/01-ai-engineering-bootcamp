from typing import List

import instructor
from langchain_core.messages import AIMessage
from langsmith import traceable
from langsmith import get_current_run_tree
from openai import OpenAI
from pydantic import BaseModel, Field
import os

from api.core.config import config

from api.rag.utils.utils import lc_messages_to_regular_messages, format_ai_message
from api.rag.utils.utils import prompt_template_config

class MCPToolCall(BaseModel):
    name: str
    arguments: dict
    server: str

class ToolCall(BaseModel):
    name: str
    arguments: dict

class RAGUsedContext(BaseModel):
    id: str
    description: str

class ProductQAAgentResponse(BaseModel): # structured output for pydantic
    answer: str 
    tool_calls: List[MCPToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context: List[RAGUsedContext]

class IntentRouterAgentResponse(BaseModel):
    user_intent: str
    answer: str

class ShoppingCartAgentResponse(BaseModel):
    answer: str
    final_answer: bool = Field(default=False)
    tool_calls: list[ToolCall] = Field(default_factory=list)

# Product QA Agent
@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def product_qa_agent_node(state) -> dict:
    from api.rag.graph import State
    state: State
    prompt_template = prompt_template_config(config.PROMPT_TEMPLATE_PATH, 'product_qa_agent')
    prompt = prompt_template.render(
        available_tools=state.product_qa_available_tools,
    )

    messages = state.messages
    conversation = []
    for message in messages:
        conversation.append(lc_messages_to_regular_messages(message))

    client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ProductQAAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )
    
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': raw_response.usage.prompt_tokens,
            'total_tokens': raw_response.usage.total_tokens,
            'output_tokens': raw_response.usage.completion_tokens,
        }
        # id returns the run_id (not the main trace_id)
        # sometimes the trace_id is not available, so we use the run_id
        trace_id = str(getattr(current_run, 'trace_id', current_run.id))

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "mcp_tool_calls": response.tool_calls,
        "product_qa_iteration": state.product_qa_iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "retrieved_context": response.retrieved_context,
        "trace_id": trace_id,
    }

# Intent Router Agent

@traceable(
    name="intent_router_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def intent_router_agent_node(state) -> dict:
    template = prompt_template_config(config.PROMPT_TEMPLATE_PATH, 'intent_router_agent')
    
    prompt = template.render()

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))

    response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1",
            response_model=IntentRouterAgentResponse,
            messages=[{"role": "system", "content": prompt}, *conversation],
            temperature=0,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': raw_response.usage.prompt_tokens,
            'total_tokens': raw_response.usage.total_tokens,
            'output_tokens': raw_response.usage.completion_tokens,
        }
        trace_id = str(getattr(current_run, 'trace_id', current_run.id))

    if response.user_intent == "product_qa":
        ai_message = []
    else:
        ai_message = [AIMessage(
            content=response.answer,
        )]

    return {
        "messages": ai_message,
        "user_intent": response.user_intent,
        "answer": response.answer,
        "trace_id": trace_id,
    }


# Shopping Cart Agent
@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def shopping_cart_agent_node(state) -> dict:
    from api.rag.graph import State
    state: State
    template = prompt_template_config(config.PROMPT_TEMPLATE_PATH, 'shopping_cart_agent')
    
    prompt = template.render(
        available_tools=state.shopping_cart_available_tools,
        user_id=state.user_id,
        cart_id=state.cart_id
    )

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


    response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1",
            response_model=ShoppingCartAgentResponse,
            messages=[{"role": "system", "content": prompt}, *conversation],
            temperature=0,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': raw_response.usage.prompt_tokens,
            'total_tokens': raw_response.usage.total_tokens,
            'output_tokens': raw_response.usage.completion_tokens,
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "shopping_cart_iteration": state.shopping_cart_iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
    }