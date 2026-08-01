import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DB: default sqlite for local dev; set DATABASE_URL to a Postgres URL in prod
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

    # JWT
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-prod")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Cookie
    cookie_name: str = "session_token"
    cookie_secure: bool = os.getenv("COOKIE_SECURE", "true").lower() == "true"

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # CORS
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "*")

    # Artifacts path
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")

    class Config:
        env_file = ".env"


settings = Settings()
