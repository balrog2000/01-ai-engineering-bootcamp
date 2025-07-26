from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
import pandas as pd

collection_name = "Amazon-items-collection-08-hybrid"

df_items = pd.read_json("../data/meta_Electronics_1000.jsonl", lines=True)
df_items.head(3)


def preprocess_data(row):
    return f"{row['title']} {' '.join(row['features'])}"

def extract_first_large_image(row):
    return row["images"][0].get("large", '')

df_items["preprocessed_data"] = df_items.apply(preprocess_data, axis=1)
df_items["first_large_image"] = df_items.apply(extract_first_large_image, axis=1)

qdrant_client = QdrantClient(
    url="http://localhost:6333",
)

qdrant_client.delete_collection(collection_name=collection_name)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

qdrant_client.create_payload_index(
    collection_name=collection_name,
    field_name="text",
    field_schema=PayloadSchemaType.TEXT
)

df_sample = df_items.sample(n=50, random_state=25)

import openai

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding


data_to_embed = df_sample[["preprocessed_data", "first_large_image", "rating_number", "price", "average_rating", 'parent_asin']].to_dict(orient="records")

pointstructs = []
for i, data in enumerate(data_to_embed):
    embedding = get_embedding(data['preprocessed_data'])
    pointstructs.append(PointStruct(
        id=i,
        vector=embedding,
        payload={
            "text": data['preprocessed_data'],
            "first_large_image": data['first_large_image'],
            "average_rating": data['average_rating'],
            "rating_number": data['rating_number'],
            "price": data['price'],
            "parent_asin": data['parent_asin'],
        }
    ))


qdrant_client.upsert(
    collection_name=collection_name,
    points=pointstructs,
    wait=True
)

df_reviews = pd.read_json("../data/Electronics_1000.jsonl", lines=True)

parent_asins = set([item['parent_asin'] for item in data_to_embed])

df_reviews_filtered = df_reviews[
    df_reviews['asin'].isin(parent_asins) | df_reviews['parent_asin'].isin(parent_asins)
]

df_reviews_filtered.head(10)


import sqlite3
from pandas import Timestamp

conn = sqlite3.connect('../data/reviews_filtered.db')
cursor = conn.cursor()

cursor.execute('''
        DROP TABLE IF EXISTS reviews
''')

# Create table for reviews if it doesn't exist
cursor.execute('''
        CREATE TABLE [reviews] (
            [rating] FLOAT,
            [title] TEXT,
            [text] TEXT,
            [images] TEXT,
            [asin] TEXT,
            [parent_asin] TEXT,
            [user_id] TEXT,
            [helpful_vote] INTEGER,
            [verified_purchase] INTEGER
        )
''')


# Use the actual columns from the CREATE TABLE statement above
expected_columns = [
    'rating', 'title', 'text', 'images', 'asin', 'parent_asin',
    'user_id', 'helpful_vote', 'verified_purchase'
]
for col in expected_columns:
    if col not in df_reviews_filtered.columns:
        df_reviews_filtered[col] = None

import json

# Insert data into the table
for _, row in df_reviews_filtered.iterrows():
    values = []
    for col in [
        'rating', 'title', 'text', 'images', 'asin', 'parent_asin',
        'user_id',  'helpful_vote', 'verified_purchase'
    ]:
        val = row[col]
        # If not a scalar (str, int, float, bool, None), dump as JSON
        if not isinstance(val, (str, int, float, bool, type(None))):
            val = json.dumps(val)
        values.append(val)
    cursor.execute('''
        INSERT OR REPLACE INTO reviews (
            rating, title, text, images, asin, parent_asin,
            user_id, helpful_vote, verified_purchase
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', tuple(values))

conn.commit()
conn.close()