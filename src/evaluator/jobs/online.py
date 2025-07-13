from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import asyncio
from langsmith import Client
import logging

evaluator_logger = logging.getLogger(__name__)
evaluator_logger.setLevel(logging.INFO)
# Ensure the logger prints to screen
if not evaluator_logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('\033[95m%(asctime)s %(levelname)s %(name)s %(message)s\033[0m')
    stream_handler.setFormatter(formatter)
    evaluator_logger.addHandler(stream_handler)
evaluator_logger.propagate = False
evaluator_logger.info("Evaluator logger initialized")

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1", temperature=1.0))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

faithfulness = Faithfulness(llm=ragas_llm)
relevancy = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

def evaluate_ragas_metrics(question: str, answer: str, retrieved_contexts: list[str]) -> dict:
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts
    )

    evaluator_logger.info(f"Input to evaluate_ragas_metrics: question={question!r}, answer={answer!r}, retrieved_contexts={retrieved_contexts!r}")

    async def gather_scores():
        return await asyncio.gather(
            faithfulness.single_turn_ascore(sample),
            relevancy.single_turn_ascore(sample)
        )

    results = asyncio.run(gather_scores())

    output = {
        "faithfulness": results[0],
        "relevancy": results[1]
    }

    # fake low for a while
    # output = {
    #     "faithfulness": 0.3,
    #     "relevancy": 0.22,
    # }

    evaluator_logger.info(f"Output from evaluate_ragas_metrics: {output!r}")

    return output

def evaluate_current_run(run_id: str, question: str, answer: str, retrieved_contexts: list[str]) -> None:
    ls_client = Client()
    metrics = evaluate_ragas_metrics(question, answer, retrieved_contexts)
    ls_client.create_feedback(
        run_id=run_id,
        key="faithfulness",
        score=metrics['faithfulness']
    )

    ls_client.create_feedback(
        run_id=run_id,
        key="relevancy",
        score=metrics['relevancy']
    )
    evaluator_logger.info(f"Feedback created for run {run_id} -> faithfulness: {metrics['faithfulness']}, relevancy: {metrics['relevancy']}")