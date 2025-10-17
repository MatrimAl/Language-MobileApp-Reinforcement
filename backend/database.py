from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from config import settings
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

mongodb = MongoDB()

async def connect_to_mongo():
    """MongoDB bağlantısını başlat"""
    logger.info("MongoDB'ye baglaniliyor...")
    
    try:
        mongodb.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000  # 5 saniye timeout
        )
        mongodb.db = mongodb.client[settings.MONGODB_DB_NAME]
        
        # Test connection
        await mongodb.client.admin.command('ping')
        logger.info("MongoDB baglantisi basarili!")
        
        # Collections
        users_collection = mongodb.db.users
        words_collection = mongodb.db.words
        user_progress_collection = mongodb.db.user_progress
        learning_history_collection = mongodb.db.learning_history
        
        # Indexes oluştur
        await users_collection.create_index([("email", ASCENDING)], unique=True)
        await words_collection.create_index([("difficulty", ASCENDING)])
        await user_progress_collection.create_index([("user_id", ASCENDING), ("word_id", ASCENDING)])
        await learning_history_collection.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
        
        logger.info("MongoDB indexes olusturuldu!")
        
    except Exception as e:
        logger.error(f"MongoDB baglanti hatasi: {e}")
        logger.warning("MongoDB olmadan devam ediliyor (mock mode)")
        # MongoDB olmadan da çalışabilir (development için)
        mongodb.client = None
        mongodb.db = None

async def close_mongo_connection():
    """MongoDB bağlantısını kapat"""
    logger.info("MongoDB bağlantısı kapatılıyor...")
    if mongodb.client is not None:
        mongodb.client.close()
        logger.info("MongoDB bağlantısı kapatıldı!")
    else:
        logger.info("MongoDB bağlantısı yoktu (mock mode)")

def get_database():
    """Database instance'ını döndür"""
    return mongodb.db
