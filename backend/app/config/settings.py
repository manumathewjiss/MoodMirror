from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    
    APP_NAME: str = "MoodMirror"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "moodmirror_db"
    
    EMOTION_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
    MODEL_CACHE_DIR: str = "./model_cache"
    
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    LOG_LEVEL: str = "INFO"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
