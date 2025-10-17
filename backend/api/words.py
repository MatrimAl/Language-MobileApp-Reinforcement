from fastapi import APIRouter, HTTPException, Depends
from database import get_database
from models import Word, WordResponse
from typing import List
from bson import ObjectId

router = APIRouter()

@router.post("/", response_model=WordResponse)
async def create_word(word: Word, db=Depends(get_database)):
    """Yeni kelime ekle"""
    words_collection = db.words
    
    word_doc = word.dict()
    result = await words_collection.insert_one(word_doc)
    
    word_doc["id"] = str(result.inserted_id)
    del word_doc["_id"]
    
    return WordResponse(**word_doc)

@router.post("/batch", response_model=List[WordResponse])
async def create_words_batch(words: List[Word], db=Depends(get_database)):
    """Toplu kelime ekleme"""
    words_collection = db.words
    
    word_docs = [word.dict() for word in words]
    result = await words_collection.insert_many(word_docs)
    
    response_words = []
    for i, inserted_id in enumerate(result.inserted_ids):
        word_docs[i]["id"] = str(inserted_id)
        response_words.append(WordResponse(**word_docs[i]))
    
    return response_words

@router.get("/", response_model=List[WordResponse])
async def get_words(
    language: str = "en",
    difficulty: int = None,
    limit: int = 100,
    db=Depends(get_database)
):
    """Kelimeleri getir (filter ile)"""
    words_collection = db.words
    
    query = {"language": language}
    if difficulty:
        query["difficulty"] = difficulty
    
    words = await words_collection.find(query).limit(limit).to_list(limit)
    
    for word in words:
        word["id"] = str(word["_id"])
        del word["_id"]
    
    return [WordResponse(**word) for word in words]

@router.get("/{word_id}", response_model=WordResponse)
async def get_word(word_id: str, db=Depends(get_database)):
    """Tek kelime getir"""
    words_collection = db.words
    
    word = await words_collection.find_one({"_id": ObjectId(word_id)})
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")
    
    word["id"] = str(word["_id"])
    del word["_id"]
    
    return WordResponse(**word)
