import openai
from src.reviews_mcp_server.core.config import config
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchAny, FusionQuery
import logging

logger = logging.getLogger(__name__)

def get_embedding_openai(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding


def retrieve_review_context(query, item_list, top_k=20):
    embedding_function = get_embedding_openai 
    qdrant_client = QdrantClient(
        url=f'http://{config.QDRANT_HOST}:6333'
    )
    query_embedding = embedding_function(query)
    results = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME_REVIEWS,
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=item_list,
                            )
                        )
                    ]
                ),
                limit=top_k
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
    )
    logger.info(f"Retrieved {len(results.points)} review results")
    retrieved_context_ids = []
    retrieved_context = []
    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context.append(result.payload['text'])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
    }


def process_review_context(context):
    formatted_context = ""
    for index, chunk in zip(context['retrieved_context_ids'], context['retrieved_context']):
        formatted_context += f"- {index}: {chunk} \n"

    return formatted_context