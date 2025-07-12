from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    API_URL: str
    model_config = SettingsConfigDict()

config = Config()

