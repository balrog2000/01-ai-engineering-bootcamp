from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery, MatchAny
from langsmith import traceable, get_current_run_tree
import openai
import logging
from api.core.config import config
from fastmcp import Client

logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(
    url=f'http://{config.QDRANT_HOST}:6333'
)

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

### Items tool


@traceable(
    name="retrieve_top_n",
    run_type="retriever",
)
def retrieve_item_context(query, embedding_type = 'text', fusion = True, top_k=5):
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
    current_run = get_current_run_tree()
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

@traceable(
    name="format_retrieved_item_context",
    run_type="prompt",
)
def process_item_context(context):
    formatted_context = ""
    for index, chunk in zip(context['retrieved_context_ids'], context['retrieved_context']):
        formatted_context += f"- {index}: {chunk} \n"

    return formatted_context

def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_item_context(query, top_k=top_k)
    formatted_context = process_item_context(context)
    return formatted_context


### Reviews tool


def retrieve_review_context(query, item_list, top_k=20):
    embedding_function = get_embedding_openai 
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
    current_run = get_current_run_tree()
    retrieved_context_ids = []
    retrieved_context = []
    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context.append(result.payload['text'])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
    }


@traceable(
    name="format_retrieved_review_context",
    run_type="prompt",
)
def process_review_context(context):
    formatted_context = ""
    for index, chunk in zip(context['retrieved_context_ids'], context['retrieved_context']):
        formatted_context += f"- {index}: {chunk} \n"

    return formatted_context


def get_formatted_review_context(query: str, item_list: list[str], top_k: int = 20) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.
    
    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_review_context(query, item_list, top_k=top_k)
    formatted_context = process_review_context(context)
    return formatted_context

