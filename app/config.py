from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./mentioniq.db"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 30.0


settings = Settings()
