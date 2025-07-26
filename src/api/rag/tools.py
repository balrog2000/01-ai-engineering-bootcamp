from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery
from langsmith import traceable, get_current_run_tree
import openai
import logging
from api.core.config import config
from typing import List
from pydantic import BaseModel
import sqlite3
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

@traceable(
    name="retrieve_top_n",
    run_type="retriever",
)
def retrieve_context(query, embedding_type = 'text', fusion = True, top_k=5):
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

def get_formatted_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_context(query, top_k=top_k)
    formatted_context = process_context(context)
    return formatted_context



class Review(BaseModel):
    chunk_id: int
    parent_asin: str
    score: int  # 1-5
    text: str

def get_reviews(chunk_ids: List[int]) -> List[Review]:
    """ Get the reviews for a given list of chunk IDs.

    Args:
        chunk_ids: The list of indexes of the chunks to get the reviews for
        
    Returns:
        A list of Review objects for the given chunk IDs. 
        Every Review object consists of chunk_id, parent_asin, score, text.
    """
    objects = qdrant_client.retrieve(
        collection_name=config.QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS,
        ids=chunk_ids,
    )
    asins_ids = {
        object.payload['parent_asin']: object.id for object in objects
    }
    review_cutoff = 50
    review_count_by_parent_asin = {}
    placeholders = ",".join("?" for _ in asins_ids.keys())
    
    db = sqlite3.connect('data/reviews_filtered.db')
    cursor = db.cursor()
    query_format = f"SELECT parent_asin, rating, title, text FROM reviews WHERE asin IN ({placeholders}) OR parent_asin IN ({placeholders})"
    cursor.execute(query_format, list(asins_ids.keys()) * 2)
    reviews = cursor.fetchall()
    result = []
    for review in reviews:
        review_count_by_parent_asin[review[0]] = review_count_by_parent_asin.get(review[0], 0) + 1
        if review_count_by_parent_asin[review[0]] <= review_cutoff:
            result.append(Review(chunk_id=asins_ids[review[0]], parent_asin=review[0], score=review[1], text=f'{review[2]} - {review[3]}'))
        else:
            print(f'Skipping review for {review[0]} because it has already been added {review_count_by_parent_asin[review[0]]} times')
    return result