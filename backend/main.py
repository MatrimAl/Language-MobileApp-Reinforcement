from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from database import connect_to_mongo, close_mongo_connection, get_database
from models import *
from typing import List
import sys

# Configure logging with UTF-8 encoding for Windows
import io

# Create UTF-8 compatible stream handler for console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setStream(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'))

# Create file handler
file_handler = logging.FileHandler('app.log', encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting...")
    await connect_to_mongo()
    logger.info("Application ready!")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    await close_mongo_connection()

# FastAPI app
app = FastAPI(
    title="Language Learning RL API",
    description="Reinforcement Learning powered language learning backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da değiştir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
async def root():
    return {
        "message": "Language Learning RL API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Import routers
from api.users import router as users_router
from api.words import router as words_router
from api.learning import router as learning_router
from api.rl import router as rl_router

app.include_router(users_router, prefix="/api/users", tags=["Users"])
app.include_router(words_router, prefix="/api/words", tags=["Words"])
app.include_router(learning_router, prefix="/api/learning", tags=["Learning"])
app.include_router(rl_router, prefix="/api/rl", tags=["Reinforcement Learning"])

if __name__ == "__main__":
    import uvicorn
    from config import settings
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
