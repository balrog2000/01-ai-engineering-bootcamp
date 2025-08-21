from langsmith import Client
from src.api.rag.agents import coordinator_agent_node
from src.api.rag.graph import State
from src.api.core.config import config
import time
ACC_THRESHOLD = 0.5
SLEEP_TIME = 5
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

results = {}
output_message = "\n"
avg_metrics = []

for model, evaluator in models_to_test.items():
    results[model] = ls_client.evaluate(
        lambda x: coordinator_agent_node(State(messages=x['messages']), models=[model]),
        data="coordinator-evaluation-dataset",
        num_repetitions=1,
        evaluators = [evaluator],
        experiment_prefix=model
    )

time.sleep(SLEEP_TIME)

for model, evaluator in models_to_test.items():
    results_resp = ls_client.read_project(
        project_name=results[model].experiment_name,
        include_stats=True,
    )

    avg_metric = results_resp.feedback_stats[evaluator.__name__]['avg']
    avg_metrics.append(avg_metric)
    if avg_metric >= ACC_THRESHOLD:
        output_message += f"✅ {model} - Success: {avg_metric}\n"
    else:
        output_message += f"❌ {model} - Failure: {avg_metric}\n"

if all([metric >= ACC_THRESHOLD for metric in avg_metrics]):
    print(output_message, flush=True)
else:
    raise AssertionError(output_message)

