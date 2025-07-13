from qdrant_client import QdrantClient
from api.core.config import config
from langsmith import traceable, get_current_run_tree
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery

import pandas as pd
import openai
import instructor
from openai import OpenAI
from pydantic import BaseModel
from pprint import pprint
from typing import List
import logging
import json
from google import genai
from api.api.models import EmbeddingType
from api.rag.utils.utils import prompt_template_config, prompt_template_registry    
import os
import open_clip
import torch
from api.rag.kafka_publisher import evaluation_publisher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(f"\033[94m%(asctime)s %(levelname)s %(name)s %(message)s\033[0m")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.propagate = False
logger.info("Retrieval logger initialized")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

def red(text: str) -> str:
    return f"\033[91m{text}\033[0m"

def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"

def blue(text: str) -> str:
    return f"\033[94m{text}\033[0m"

@traceable(
    name="embed_query_image",
    run_type="embedding",
    metadata={"ls_provider": 'openclip', "ls_model_name": 'ViT-B-32'}
)
def get_embedding_openclip(text):
    logger.info(f"Embedding with OpenCLIP")
    tokenized = clip_tokenizer(text)
    with torch.no_grad(), torch.autocast("cuda"):
        embedding = clip_model.encode_text(tokenized)[0]
        return embedding.cpu().numpy().tolist()

@traceable(
    name="embed_query_text",
    run_type="embedding",
    metadata={"ls_provider": config.EMBEDDING_MODEL_PROVIDER, "ls_model_name": config.EMBEDDING_MODEL}
)
def get_embedding_openai(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
        }
    return response.data[0].embedding

@traceable(
    name="retrieve_top_n",
    run_type="retriever",
)
def retrieve_context(query, qdrant_client, embedding_type, fusion, top_k=5):
    embedding_function = get_embedding_openai 
    if embedding_type == EmbeddingType.IMAGE:
        embedding_function = get_embedding_openclip
    query_embedding = embedding_function(query)
    collection_name = config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS if embedding_type == EmbeddingType.TEXT \
        else config.QDRANT_COLLECTION_NAME_IMAGE_EMBEDDINGS
    if fusion:
        logger.info("Fusion enabled. Retrieving top {} results from {} collection".format(top_k, collection_name))
        results = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    limit=20,
                ),
                Prefetch(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="text",
                                match=MatchText(
                                    text=query,
                                )
                            )
                        ]
                    ),
                    limit=20
                )
            ],
            query=FusionQuery(fusion="rrf"),
            limit=top_k,
        )
    else:
        logger.info("Fusion disabled. Retrieving top {} results from {} collection".format(top_k, collection_name))
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
        )
    logger.info(f"Retrieved {len(results.points)} results")
    current_run = get_current_run_tree()
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    for result in results.points:
        retrieved_context_ids.append(result.id)
        retrieved_context.append(result.payload['text'])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt",
)
def process_context(context):
    formatted_context = ""
    for index, chunk in zip(context['retrieved_context_ids'], context['retrieved_context']):
        formatted_context += f"- {index}: {chunk} \n"

    return formatted_context

@traceable(
    name="render_prompt",
    run_type="prompt",
)
def build_prompt(question, context):

    processed_context = process_context(context)
    prompt_template = prompt_template_config(config.PROMPT_TEMPLATE_PATH, 'rag_generation')
    # prompt_template = prompt_template_registry('rag-prompt')   
    prompt = prompt_template.render(
        processed_context=processed_context,
        question=question
    )
    return prompt

class RAGUsedContext(BaseModel):
    id: int
    description: str

class RAGGenerationResponse(BaseModel):
    answer: str
    retrieved_context: List[RAGUsedContext]
    used_context_count: int
    not_used_context_count: int


# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up a dedicated file logger for completion kwargs, disable stdout logging
instructor_logger = logging.getLogger("instructor")
instructor_logger.propagate = False  # Prevent logging to stdout/stderr
if not instructor_logger.handlers:
    file_handler = logging.FileHandler("logs/instructor.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s %(asctime)s %(levelname)s ')
    file_handler.setFormatter(formatter)
    instructor_logger.addHandler(file_handler)
    instructor_logger.setLevel(logging.INFO)
    
@traceable(
    name="instructor_wrapping",
    run_type="prompt"
)
def log_completion_kwargs(*args, **kwargs) -> None:
    instructor_logger.info(json.dumps(args, indent=2))
    instructor_logger.info(json.dumps(kwargs, indent=2))


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def generate_answer(prompt):
    client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))
    client.on("completion:kwargs", log_completion_kwargs)
    response, raw_response = client.chat.completions.create_with_completion(
        model='gpt-4.1',
        response_model=RAGGenerationResponse,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': raw_response.usage.prompt_tokens,
            'total_tokens': raw_response.usage.total_tokens,
            'output_tokens': raw_response.usage.completion_tokens,
        }
    return response, raw_response

@traceable(
    name="rag_pipeline",
)
def rag_pipeline(question, qdrant_client, embedding_type, fusion, top_k=5):
    retrieved_context = retrieve_context(question, qdrant_client, embedding_type, fusion, top_k)
    prompt = build_prompt(question, retrieved_context)
    answer, raw_response = generate_answer(prompt)

    current_run = get_current_run_tree()
    logger.info(blue(f"Run id: {current_run.id}"))
    final_result = {
        "answer": answer,
        "raw_response": raw_response,
        "question": question,
        "retrieved_context_ids": retrieved_context['retrieved_context_ids'],
        "retrieved_context": retrieved_context['retrieved_context'],
        "similarity_scores": retrieved_context['similarity_scores'],
    }
    current_run = get_current_run_tree()
    if current_run:
        # Publish evaluation request to Kafka instead of processing synchronously
        evaluation_publisher.publish_evaluation_request(
            run_id=str(current_run.id),
            question=question,
            answer=answer.answer,
            retrieved_contexts=retrieved_context['retrieved_context']
        )

    return final_result


def rag_pipeline_wrapper(question, embedding_type=EmbeddingType.TEXT, fusion=True, top_k=5):
    qdrant_client = QdrantClient(
        url=f'http://{config.QDRANT_HOST}:6333'
    )

    result = rag_pipeline(question, qdrant_client, embedding_type, fusion, top_k)

    items = []
    for context in result['answer'].retrieved_context:
        payload = qdrant_client.retrieve(
            collection_name=config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS if embedding_type == EmbeddingType.TEXT else config.QDRANT_COLLECTION_NAME_IMAGE_EMBEDDINGS,
            ids=[context.id],
        )[0].payload
        
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
        
    return {
        'answer': result['answer'].answer,
        'items': items,
        'used_context_count': result['answer'].used_context_count,
        'not_used_context_count': result['answer'].not_used_context_count,
    }