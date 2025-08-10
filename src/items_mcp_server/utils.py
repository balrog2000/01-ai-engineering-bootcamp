import openai
from src.items_mcp_server.core.config import config
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery
import logging

logger = logging.getLogger(__name__)

def get_embedding_openai(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding

def retrieve_item_context(query, embedding_type = 'text', fusion = True, top_k=5):
    qdrant_client = QdrantClient(
        url=f'http://{config.QDRANT_HOST}:6333'
    )
    embedding_function = get_embedding_openai 
    if embedding_type == 'image':
        embedding_function = get_embedding_openclip
    query_embedding = embedding_function(query)
    collection_name = config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS
    if fusion:
        logger.info("Fusion enabled. Retrieving for query: {} top {} results from {} collection".format(query, top_k, collection_name))
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
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context.append(result.payload['text'])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }

def process_item_context(context):
    formatted_context = ""
    for index, chunk in zip(context['retrieved_context_ids'], context['retrieved_context']):
        formatted_context += f"- {index}: {chunk} \n"

    return formatted_context