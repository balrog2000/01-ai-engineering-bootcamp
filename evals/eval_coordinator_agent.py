from langsmith import Client
from src.api.rag.agents import coordinator_agent_node
from src.api.rag.graph import State
from src.api.core.config import config


ls_client = Client(api_key=config.LANGSMITH_API_KEY)

def next_agent_core_evaluator(run, example):
    next_agent_match = run.outputs['next_agent'] == example.outputs['next_agent']
    final_answer_match = run.outputs['coordinator_final_answer'] == example.outputs['coordinator_final_answer']
    return all([next_agent_match, final_answer_match])

def next_agent_evaluator_gpt_4_1(run, example):
    return next_agent_core_evaluator(run, example)

def next_agent_evaluator_gpt_4_1_mini(run, example):
    return next_agent_core_evaluator(run, example)

def next_agent_evaluator_groq_llama_3_3_70b_versatile(run, example):
    return next_agent_core_evaluator(run, example)

models_to_test = {
    'gpt-4.1': next_agent_evaluator_gpt_4_1,
    'gpt-4.1-mini': next_agent_evaluator_gpt_4_1_mini,
    'groq/llama-3.3-70b-versatile': next_agent_evaluator_groq_llama_3_3_70b_versatile,
}


for model, evaluator in models_to_test.items():
    results = ls_client.evaluate(
        lambda x: coordinator_agent_node(State(messages=x['messages']), models=[model]),
        data="coordinator-evaluation-dataset",
        num_repetitions=1,
        evaluators = [evaluator],
        experiment_prefix=model
    )