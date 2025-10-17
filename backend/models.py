from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

# User Models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    native_language: str = "tr"
    target_language: str = "en"

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    native_language: str
    target_language: str
    level: int = 1
    total_xp: int = 0
    created_at: datetime

# Word Models
class Word(BaseModel):
    word: str
    translation: str
    language: str
    difficulty: int = Field(..., ge=1, le=5)
    category: Optional[str] = None
    example_sentence: Optional[str] = None
    pronunciation: Optional[str] = None

class WordResponse(Word):
    id: str

# User Progress Models
class UserProgress(BaseModel):
    user_id: str
    word_id: str
    correct_count: int = 0
    incorrect_count: int = 0
    last_seen: Optional[datetime] = None
    next_review: Optional[datetime] = None
    mastery_level: float = 0.0  # 0.0 to 1.0
    retention_rate: float = 0.0

# Learning History Models
class LearningSession(BaseModel):
    user_id: str
    word_id: str
    is_correct: bool
    response_time: float  # saniye
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
class LearningHistoryResponse(LearningSession):
    id: str

# RL State Models
class RLState(BaseModel):
    """DQN için state representation"""
    user_level: int
    total_words_learned: int
    avg_accuracy: float
    recent_accuracy: float  # Son 10 sorunun accuracy'si
    current_streak: int
    time_since_last_session: float  # saat cinsinden
    mastery_distribution: List[float]  # [beginner, intermediate, advanced, expert, master]
    
class RLAction(BaseModel):
    """DQN action - hangi kelime gösterilecek"""
    word_id: str
    difficulty: int
    reason: str  # Debugging için

class RLReward(BaseModel):
    """Reward calculation"""
    base_reward: float
    accuracy_bonus: float
    speed_bonus: float
    retention_bonus: float
    total_reward: float

# API Request/Response Models
class QuizRequest(BaseModel):
    user_id: str

class QuizResponse(BaseModel):
    word: WordResponse
    options: List[str]
    session_id: str

class AnswerRequest(BaseModel):
    user_id: str
    word_id: str
    session_id: str
    answer: str
    response_time: float

class AnswerResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    reward: float
    xp_gained: int
    new_level: int
    mastery_level: float
    next_word: Optional[WordResponse] = None
