from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    QDRANT_HOST: str
    QDRANT_COLLECTION_NAME_TEXT_EMBEDDINGS: str
    QDRANT_COLLECTION_NAME_IMAGE_EMBEDDINGS: str
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_PROVIDER: str
    LANGSMITH_PROJECT: str
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    GENERATION_MODEL: str
    GENERATION_MODEL_PROVIDER: str
    PROMPT_TEMPLATE_PATH: str = "src/api/rag/prompts/rag_generation.yaml"
    

    # changing because we now use docker compose env

    model_config = SettingsConfigDict(env_file=".env")

class Settings(BaseSettings):
    DEFAULT_TIMEOUT: float = 30.0
    VERSION: str = "0.1.0"
config = Config()
settings = Settings()