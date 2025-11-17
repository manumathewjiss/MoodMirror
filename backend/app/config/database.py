from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from .settings import settings
import logging

logger = logging.getLogger(__name__)


class Database:
    
    client: AsyncIOMotorClient = None
    
    @classmethod
    async def connect_db(cls):
        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}")
            cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
            
            await cls.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB!")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    @classmethod
    async def close_db(cls):
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")
    
    @classmethod
    def get_database(cls):
        if cls.client is None:
            raise RuntimeError("Database not connected. Call connect_db() first.")
        return cls.client[settings.DATABASE_NAME]


async def get_db():
    return Database.get_database()


async def get_users_collection():
    db = Database.get_database()
    return db.users


async def get_moods_collection():
    db = Database.get_database()
    return db.mood_entries


async def create_indexes():
    try:
        db = Database.get_database()
        
        await db.users.create_index("email", unique=True)
        await db.users.create_index("created_at")
        
        await db.mood_entries.create_index("user_id")
        await db.mood_entries.create_index("created_at")
        await db.mood_entries.create_index([("user_id", 1), ("created_at", -1)])
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
