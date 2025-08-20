from typing import List

import instructor
from langchain_core.messages import AIMessage
from langsmith import traceable
from langsmith import get_current_run_tree
from openai import OpenAI
from pydantic import BaseModel, Field
import os
from litellm import completion

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

class Delegation(BaseModel):
    agent: str
    task: str = Field(default="")

class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: list[Delegation]
    final_answer: bool = Field(default=False)
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
def product_qa_agent_node(state, models = ['gpt-4.1', 'groq/llama-3.3-70b-versatile']) -> dict:
    from api.rag.graph import State
    state: State

    messages = state.messages
    conversation = []
    for message in messages:
        conversation.append(lc_messages_to_regular_messages(message))

    client = instructor.from_litellm(completion)
    for model in models:
        template = prompt_template_config(config.PROMPT_TEMPLATE_PATH_PRODUCT_QA, model)
        prompt = template.render(
            available_tools=state.product_qa_available_tools,
        )
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ProductQAAgentResponse,
                messages=[{"role": "system", "content": prompt}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
    
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
        "mcp_tool_calls": response.tool_calls,
        "product_qa_iteration": state.product_qa_iteration + 1,
        "answer": response.answer,
        "product_qa_final_answer": response.final_answer,
        "retrieved_context": response.retrieved_context,
    }

# Coordinator Agent

@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def coordinator_agent_node(state, models = ['gpt-4.1', 'groq/llama-3.3-70b-versatile']) -> dict:
    

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            template = prompt_template_config(config.PROMPT_TEMPLATE_PATH_COORDINATOR, model)
            prompt = template.render()
            response, raw_response = client.chat.completions.create_with_completion(
                    model=model,
                    response_model=CoordinatorAgentResponse,
                    messages=[{"role": "system", "content": prompt}, *conversation],
                    temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': raw_response.usage.prompt_tokens,
            'total_tokens': raw_response.usage.total_tokens,
            'output_tokens': raw_response.usage.completion_tokens,
        }
        trace_id = str(getattr(current_run, 'trace_id', current_run.id))


    return {
        # "messages": ai_message,
        "next_agent": response.next_agent,
        "plan": response.plan,
        "coordinator_final_answer": response.final_answer,
        "coordinator_iteration": state.coordinator_iteration + 1,
        "answer": response.answer,
        "trace_id": trace_id,
    }


# Shopping Cart Agent
@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def shopping_cart_agent_node(state, models = ['gpt-4.1', 'groq/llama-3.3-70b-versatile']) -> dict:
    from api.rag.graph import State
    state: State

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    client = instructor.from_litellm(completion)

    for model in models:
        template = prompt_template_config(config.PROMPT_TEMPLATE_PATH_SHOPPING_CART, model)
        prompt = template.render(
            available_tools=state.shopping_cart_available_tools,
            user_id=state.user_id,
            cart_id=state.cart_id
        )
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ShoppingCartAgentResponse,
                messages=[{"role": "system", "content": prompt}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")

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
        "shopping_cart_final_answer": response.final_answer,
        "answer": response.answer,
    }