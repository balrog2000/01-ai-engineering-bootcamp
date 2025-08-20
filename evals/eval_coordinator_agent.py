from langsmith import Client
from src.api.rag.agents import coordinator_agent_node
from src.api.rag.graph import State
from src.api.core.config import config


ls_client = Client(api_key=config.LANGSMITH_API_KEY)


def next_agent_evaluator(run, example):
    next_agent_match = run.outputs['next_agent'] == example.outputs['next_agent']
    final_answer_match = run.outputs['coordinator_final_answer'] == example.outputs['coordinator_final_answer']

    return all([next_agent_match, final_answer_match])

results = ls_client.evaluate(
    lambda x: coordinator_agent_node(State(messages=x['messages'])),
    data="coordinator-evaluation-dataset",
    num_repetitions=4,
    evaluators = [
        next_agent_evaluator,
    ],
    experiment_prefix='coordinator-evaluation-dataset'
)