from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import get_db, engine
from model import Base, User, Word, UserWordStat, Attempt, UserLevelStat, LEVELS
from state import build_state, compute_reward, level_idx
from selection import pick_from_bucket, pick_distractors
from rl import AgentRegistry
import uuid, random
from datetime import datetime, timedelta

app = FastAPI(title="RL Language MVP")
Base.metadata.create_all(bind=engine)

STATE_DIM = 13
agent_registry = AgentRegistry(state_dim=STATE_DIM, n_actions=5)  # A1..C1

class StartIn(BaseModel):  user_id: int
class StartOut(BaseModel): session_id: str; state: list[float]; eps: float

@app.post("/session/start", response_model=StartOut)
def start_session(inp: StartIn, db: Session = Depends(get_db)):
    user = db.get(User, inp.user_id) or (_ for _ in ()).throw(HTTPException(404, "user not found"))
    s = build_state(db, user); eps = 0.1
    return {"session_id": str(uuid.uuid4()), "state": s, "eps": eps}

class NextOut(BaseModel):
    question_id: str; word_id: int; prompt: str; options: list[dict]
    bucket_level: str; action: int; state: list[float]

@app.get("/rl/next", response_model=NextOut)
def rl_next(user_id: int, session_id: str, db: Session = Depends(get_db)):
    user = db.get(User, user_id) or (_ for _ in ()).throw(HTTPException(404, "user not found"))
    s = build_state(db, user)
    agent = agent_registry.get(user.id)
    a = agent.act(s, eps=0.1)               # 0..4
    bucket_level = LEVELS[a]
    w = pick_from_bucket(db, user.id, bucket_level)
    distractors = pick_distractors(db, w, k=2)
    opts = [
        {"id": f"opt-{w.id}-1", "text": w.l1_text, "is_correct": True},
        {"id": f"opt-{distractors[0].id}-2", "text": distractors[0].l1_text, "is_correct": False},
        {"id": f"opt-{distractors[1].id}-3", "text": distractors[1].l1_text, "is_correct": False},
    ]
    random.shuffle(opts)
    qid = f"q-{w.id}-{uuid.uuid4().hex[:6]}"
    # kelime için stat yoksa oluştur
    st = db.get(UserWordStat, {"user_id": user.id, "word_id": w.id})
    if not st:
        db.add(UserWordStat(user_id=user.id, word_id=w.id)); db.commit()
    return {"question_id": qid, "word_id": w.id, "prompt": w.l2_text,
            "options": opts, "bucket_level": bucket_level, "action": a, "state": s}

class AnswerIn(BaseModel):
    user_id: int; session_id: str; question_id: str; word_id: int
    selected_text: str; response_ms: int; bucket_level: str; action: int

class StepOut(BaseModel):
    correct: bool; reward: float; new_state: list[float]; loss: float | None

@app.post("/rl/answer", response_model=StepOut)
def rl_answer(inp: AnswerIn, db: Session = Depends(get_db)):
    user = db.get(User, inp.user_id) or (_ for _ in ()).throw(HTTPException(404, "user not found"))
    word = db.get(Word, inp.word_id) or (_ for _ in ()).throw(HTTPException(404, "word not found"))
    correct = (inp.selected_text.strip().lower() == word.l1_text.strip().lower())
    st = db.get(UserWordStat, {"user_id": user.id, "word_id": word.id})
    if not st: 
        st = UserWordStat(user_id=user.id, word_id=word.id); db.add(st)
    if correct:
        st.alpha += 1.0; st.reps += 1
        st.interval_days = 1 if st.interval_days == 0 else int(st.interval_days * 1.6 + 1)
    else:
        st.beta  += 1.0; st.reps = max(0, st.reps - 1); st.interval_days = 0
    st.last_seen = datetime.utcnow()
    st.due_at = st.last_seen + timedelta(days=st.interval_days or 1)
    db.add(Attempt(user_id=user.id, word_id=word.id, is_correct=bool(correct),
                   response_ms=inp.response_ms, level=word.level, session_id=inp.session_id))
    st_lv = db.get(UserLevelStat, {"user_id": user.id, "level": word.level})
    if not st_lv:
        st_lv = UserLevelStat(user_id=user.id, level=word.level); db.add(st_lv)
    if correct:
        st_lv.correct += 1
    else:
        st_lv.wrong += 1
    db.commit()

    # ödül & yeni state
    due_flag = (st.due_at is None) or (st.last_seen >= st.due_at)
    r = compute_reward(correct, word.level, user.target_level, due_flag, inp.response_ms)
    s2 = build_state(db, user)

    # RL güncelleme
    agent = agent_registry.get(user.id)
    # Not: pratikte s'i /rl/next’ten taşırsın; basitlik için burada tekrar hesaplamak yeterli olur.
    s = build_state(db, user)
    agent.push(s, int(inp.action), float(r), s2, False)
    loss = agent.train_step()
    if agent.steps % 2000 == 0: agent.hard_update()

    return {"correct": bool(correct), "reward": float(r), "new_state": s2, "loss": loss}
