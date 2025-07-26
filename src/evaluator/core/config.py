from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    LANGSMITH_PROJECT: str
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    KAFKA_HOST: str
    KAFKA_PORT: int
    KAFKA_TOPIC: str

    model_config = SettingsConfigDict(env_file=".env_evaluator")

config = Config()