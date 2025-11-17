from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config.settings import settings
from app.config.database import Database, create_indexes
from app.services.emotion_analyzer import get_emotion_analyzer

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MoodMirror application...")
    
    await Database.connect_db()
    await create_indexes()
    
    logger.info("Loading emotion analysis model...")
    get_emotion_analyzer()
    logger.info("Emotion model loaded successfully!")
    
    logger.info("MoodMirror is ready! 🎭")
    
    yield
    
    logger.info("Shutting down MoodMirror...")
    await Database.close_db()
    logger.info("Goodbye! 👋")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered emotion tracking platform for university students",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to MoodMirror API! 🎭",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "phase": "Phase 1 - Text-based Emotion Analysis"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


from app.routes.mood import router as mood_router
app.include_router(mood_router, prefix="/api/mood", tags=["Mood Tracking"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
