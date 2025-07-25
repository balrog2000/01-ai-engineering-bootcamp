from typing import List

import instructor
from langchain_core.messages import AIMessage
from langsmith import traceable
from langsmith import get_current_run_tree
from openai import OpenAI
from pydantic import BaseModel, Field

from api.core.config import config

from api.rag.utils.utils import lc_messages_to_regular_messages
from api.rag.utils.utils import prompt_template_config

class ToolCall(BaseModel):
    name: str
    arguments: dict

class RAGUsedContext(BaseModel):
    id: int
    description: str

class AgentResponse(BaseModel): # structured output for pydantic
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context: List[RAGUsedContext]

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def agent_node(state) -> dict:
    from api.rag.graph import State
    state: State
    prompt_template = prompt_template_config(config.PROMPT_TEMPLATE_PATH, 'rag_generation')
    prompt = prompt_template.render(
        available_tools=state.available_tools,
    )

    messages = state.messages
    conversation = []
    for message in messages:
        conversation.append(lc_messages_to_regular_messages(message))

    client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )
    
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': raw_response.usage.prompt_tokens,
            'total_tokens': raw_response.usage.total_tokens,
            'output_tokens': raw_response.usage.completion_tokens,
        }

    if response.tool_calls:
        tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            tool_calls.append({
                "id": f"call_{i}",
                "name": tc.name,
                "args": tc.arguments
            })

        ai_message = AIMessage(
            content=response.answer,
            tool_calls=tool_calls
        )
    else:
        ai_message = AIMessage(
            content=response.answer,
        )

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "retrieved_context": response.retrieved_context,
    }