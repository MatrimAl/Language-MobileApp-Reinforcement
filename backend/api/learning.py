from fastapi import APIRouter, HTTPException, Depends
from database import get_database
from models import QuizRequest, QuizResponse, AnswerRequest, AnswerResponse, WordResponse
from bson import ObjectId
from datetime import datetime
import random
import uuid

router = APIRouter()

# Temporary session storage (production'da Redis kullan)
active_sessions = {}

@router.post("/quiz", response_model=QuizResponse)
async def get_quiz(request: QuizRequest, db=Depends(get_database)):
    """
    Kullanıcı için quiz sorusu oluştur
    TODO: Burada RL agent'i kullanarak kelime seçimi yapılacak
    """
    words_collection = db.words
    progress_collection = db.user_progress
    
    # Get user progress
    user_progress = await progress_collection.find({"user_id": request.user_id}).to_list(None)
    learned_word_ids = {p["word_id"] for p in user_progress}
    
    # Get all words
    all_words = await words_collection.find({"language": "en"}).to_list(None)
    
    if not all_words:
        raise HTTPException(status_code=404, detail="No words available")
    
    # Select word (şimdilik random, sonra RL agent kullanılacak)
    # Priority: words that need review > new words > random
    words_need_review = []
    new_words = []
    
    for word in all_words:
        word_id = str(word["_id"])
        if word_id in learned_word_ids:
            progress = next((p for p in user_progress if p["word_id"] == word_id), None)
            if progress and progress.get("mastery_level", 0) < 0.7:
                words_need_review.append(word)
        else:
            new_words.append(word)
    
    # Select word
    if words_need_review:
        selected_word = random.choice(words_need_review)
    elif new_words:
        selected_word = random.choice(new_words[:10])  # İlk 10 yeni kelimeden
    else:
        selected_word = random.choice(all_words)
    
    # Generate options (3 wrong + 1 correct)
    correct_translation = selected_word["translation"]
    wrong_options = [w["translation"] for w in all_words 
                     if w["translation"] != correct_translation]
    random.shuffle(wrong_options)
    wrong_options = wrong_options[:3]
    
    options = wrong_options + [correct_translation]
    random.shuffle(options)
    
    # Create session
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "word_id": str(selected_word["_id"]),
        "correct_answer": correct_translation,
        "start_time": datetime.utcnow()
    }
    
    # Prepare response
    selected_word["id"] = str(selected_word["_id"])
    del selected_word["_id"]
    del selected_word["translation"]  # Don't send correct answer
    
    return QuizResponse(
        word=WordResponse(**selected_word),
        options=options,
        session_id=session_id
    )

@router.post("/answer", response_model=AnswerResponse)
async def submit_answer(request: AnswerRequest, db=Depends(get_database)):
    """Kullanıcı cevabını değerlendir"""
    # Verify session
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[request.session_id]
    correct_answer = session["correct_answer"]
    is_correct = request.answer == correct_answer
    
    # Calculate reward
    base_reward = 1.0 if is_correct else -0.5
    speed_bonus = 0.2 if (is_correct and request.response_time < 5) else 0
    total_reward = base_reward + speed_bonus
    
    # Update user progress
    progress_collection = db.user_progress
    history_collection = db.learning_history
    users_collection = db.users
    
    # Update/create progress
    progress = await progress_collection.find_one({
        "user_id": request.user_id,
        "word_id": request.word_id
    })
    
    if progress:
        # Update existing
        if is_correct:
            progress["correct_count"] += 1
        else:
            progress["incorrect_count"] += 1
        
        total_attempts = progress["correct_count"] + progress["incorrect_count"]
        progress["mastery_level"] = progress["correct_count"] / total_attempts
        progress["last_seen"] = datetime.utcnow()
        
        await progress_collection.update_one(
            {"_id": progress["_id"]},
            {"$set": progress}
        )
    else:
        # Create new
        progress = {
            "user_id": request.user_id,
            "word_id": request.word_id,
            "correct_count": 1 if is_correct else 0,
            "incorrect_count": 0 if is_correct else 1,
            "mastery_level": 1.0 if is_correct else 0.0,
            "last_seen": datetime.utcnow()
        }
        await progress_collection.insert_one(progress)
    
    # Add to history
    history_doc = {
        "user_id": request.user_id,
        "word_id": request.word_id,
        "is_correct": is_correct,
        "response_time": request.response_time,
        "timestamp": datetime.utcnow()
    }
    await history_collection.insert_one(history_doc)
    
    # Update user XP and level
    xp_gained = 10 if is_correct else 2
    user = await users_collection.find_one({"_id": ObjectId(request.user_id)})
    new_xp = user["total_xp"] + xp_gained
    new_level = 1 + (new_xp // 100)
    
    await users_collection.update_one(
        {"_id": ObjectId(request.user_id)},
        {"$set": {"total_xp": new_xp, "level": new_level}}
    )
    
    # Clean up session
    del active_sessions[request.session_id]
    
    return AnswerResponse(
        is_correct=is_correct,
        correct_answer=correct_answer,
        reward=total_reward,
        xp_gained=xp_gained,
        new_level=new_level,
        mastery_level=progress["mastery_level"],
        next_word=None  # TODO: Get next word from RL agent
    )

@router.get("/history/{user_id}")
async def get_learning_history(user_id: str, limit: int = 50, db=Depends(get_database)):
    """Kullanıcının öğrenme geçmişini getir"""
    history_collection = db.learning_history
    
    history = await history_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    
    for item in history:
        item["id"] = str(item["_id"])
        del item["_id"]
    
    return history
