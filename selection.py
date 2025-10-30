import random
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select
from model import Word, UserWordStat, LEVELS

def pick_from_bucket(db: Session, user_id: int, bucket_level: str) -> Word:
    q = select(Word).where(Word.level == bucket_level).limit(400)
    words = db.execute(q).scalars().all()
    if not words:
        # yakın seviye fallback
        idx = LEVELS.index(bucket_level)
        for j in [1, -1, 2, -2]:
            k = idx + j
            if 0 <= k < len(LEVELS):
                w2 = db.execute(select(Word).where(Word.level == LEVELS[k]).limit(400)).scalars().all()
                if w2: words = w2; break
    now = datetime.utcnow()
    scored = []
    for w in words:
        st = db.get(UserWordStat, {"user_id": user_id, "word_id": w.id})
        if st is None:
            m = 0.5; reps=0; due=True
        else:
            m = st.alpha/(st.alpha+st.beta)
            reps = st.reps
            due = (st.due_at is None) or (st.due_at <= now)
        score = (1.0 - m) * (1.3 if due else 0.8) * (1.1 if reps < 3 else 1.0)
        scored.append((score, w))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    if not scored:
        # Hiç kelime yoksa hata
        raise ValueError(f"No words found for level {bucket_level} or nearby levels!")
    
    return scored[0][1]

def pick_distractors(db: Session, target: Word, k=2):
    pool = db.execute(select(Word).where((Word.level == target.level) & (Word.id != target.id))).scalars().all()
    if len(pool) < k:
        pool = db.execute(select(Word).where(Word.id != target.id)).scalars().all()
    random.shuffle(pool)
    return pool[:k]
