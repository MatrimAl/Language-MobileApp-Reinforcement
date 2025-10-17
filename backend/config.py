import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # MongoDB
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "language_learning_rl")
    
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    
    # RL Model
    MODEL_SAVE_PATH: str = os.getenv("MODEL_SAVE_PATH", "./models/dqn_model")
    TRAINING_EPISODES: int = int(os.getenv("TRAINING_EPISODES", "1000"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    GAMMA: float = float(os.getenv("GAMMA", "0.95"))
    EPSILON_START: float = float(os.getenv("EPSILON_START", "1.0"))
    EPSILON_MIN: float = float(os.getenv("EPSILON_MIN", "0.01"))
    EPSILON_DECAY: float = float(os.getenv("EPSILON_DECAY", "0.995"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    TENSORBOARD_LOG_DIR: str = os.getenv("TENSORBOARD_LOG_DIR", "./logs/tensorboard")

settings = Settings()
