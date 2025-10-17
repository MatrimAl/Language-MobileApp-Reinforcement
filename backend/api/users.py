from fastapi import APIRouter, HTTPException, Depends
from database import get_database
from models import UserCreate, UserResponse
from datetime import datetime
from bson import ObjectId
import bcrypt

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db=Depends(get_database)):
    """Yeni kullanıcı kaydı"""
    users_collection = db.users
    
    # Check if user exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    
    # Create user document
    user_doc = {
        "email": user.email,
        "username": user.username,
        "password": hashed_password,
        "native_language": user.native_language,
        "target_language": user.target_language,
        "level": 1,
        "total_xp": 0,
        "created_at": datetime.utcnow()
    }
    
    result = await users_collection.insert_one(user_doc)
    
    user_doc["id"] = str(result.inserted_id)
    del user_doc["password"]
    del user_doc["_id"]
    
    return UserResponse(**user_doc)

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db=Depends(get_database)):
    """Kullanıcı bilgilerini getir"""
    users_collection = db.users
    
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user["id"] = str(user["_id"])
    del user["_id"]
    del user["password"]
    
    return UserResponse(**user)

@router.get("/{user_id}/stats")
async def get_user_stats(user_id: str, db=Depends(get_database)):
    """Kullanıcı istatistiklerini getir"""
    progress_collection = db.user_progress
    history_collection = db.learning_history
    
    # Total words learned
    total_progress = await progress_collection.count_documents({"user_id": user_id})
    
    # Mastered words (>70% accuracy)
    mastered = await progress_collection.count_documents({
        "user_id": user_id,
        "mastery_level": {"$gte": 0.7}
    })
    
    # Recent history
    recent_history = await history_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(20).to_list(20)
    
    # Calculate accuracy
    if recent_history:
        correct_count = sum(1 for h in recent_history if h["is_correct"])
        accuracy = correct_count / len(recent_history)
    else:
        accuracy = 0.0
    
    return {
        "total_words": total_progress,
        "mastered_words": mastered,
        "recent_accuracy": accuracy,
        "total_sessions": len(recent_history)
    }
