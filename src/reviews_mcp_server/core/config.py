from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    QDRANT_HOST: str = 'qdrant'
    QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS: str = "Amazon-items-collection-12-items"
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_PROVIDER: str
    QDRANT_COLLECTION_NAME_REVIEWS: str = "Amazon-items-collection-12-reviews"

    model_config = SettingsConfigDict(env_file=".env_api")

config = Config()