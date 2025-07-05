from qdrant_client import QdrantClient
from chatbot_ui.core.config import config
from chatbot_ui.retrieval import rag_pipeline

from langsmith import Client

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import os

from ragas.metrics import Faithfulness, NonLLMContextRecall, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall
from ragas.dataset_schema import SingleTurnSample

ls_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1", temperature=1.0))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

qdrant_client = QdrantClient(
    url=f"http://localhost:6333",
)

async def ragas_faithfulness(run, example):
    #example not used cause we are not checking the ground truth here
    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = Faithfulness(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)

async def ragas_response_relevancy(run, example):
    #example not used cause we are not checking the ground truth here
    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    return await scorer.single_turn_ascore(sample)

async def ragas_context_precision(run, example):
    sample = SingleTurnSample(
                user_input=run.outputs["question"],
                response=run.outputs["answer"],
                retrieved_contexts=run.outputs["retrieved_context"]
            )
    scorer = LLMContextPrecisionWithoutReference(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)


async def ragas_context_recall_llm(run, example):
    sample = SingleTurnSample(
                user_input=run.outputs["question"],
                response=run.outputs["answer"],
                reference=example.outputs["ground_truth"],
                retrieved_contexts=run.outputs["retrieved_context"]
            )
    scorer = LLMContextRecall(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)

async def ragas_context_recall(run, example):
    sample = SingleTurnSample(
                retrieved_contexts=run.outputs["retrieved_context"],
                reference_contexts=example.outputs["contexts"]
            )
    scorer = NonLLMContextRecall()
    return await scorer.single_turn_ascore(sample)

results = ls_client.evaluate(
    lambda x: rag_pipeline(x['question'], qdrant_client, top_k=5),
    data="rag-evaluation-dataset",
    evaluators = [
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision,
        ragas_context_recall_llm,
        ragas_context_recall,
    ],
    experiment_prefix='rag-evaluation-dataset'
)