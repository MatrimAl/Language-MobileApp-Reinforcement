from __future__ import annotations

import uuid
import random
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db, engine
from model import Base, User, Word, UserWordStat, Attempt, UserLevelStat, LEVELS
from state import build_state, compute_reward, level_idx, adjust_target_level
from selection import pick_from_bucket, pick_distractors
from rl import AgentRegistry


app = FastAPI(title="RL Language MVP")
Base.metadata.create_all(bind=engine)

# -----------------------------
# Sabitler
# -----------------------------
STATE_DIM = 13
N_ACTIONS = len(LEVELS)

# Tek bir AgentRegistry (paylaşılan backbone + kullanıcı head’leri)
agent_registry = AgentRegistry(state_dim=STATE_DIM, n_actions=N_ACTIONS)

# Oturum bazlı geçici bellek: /rl/next -> /rl/answer
session_states: dict[str, dict] = {}

# -----------------------------
# FastAPI lifecycle
# -----------------------------
@app.on_event("startup")
async def startup_event():
    """Hot reload vb. durumlarda registry’yi yeniden kur ve session state’i sıfırla."""
    global agent_registry, session_states
    agent_registry = AgentRegistry(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    session_states = {}
    print(f"✅ Agent Registry yenilendi (state_dim={STATE_DIM}, actions={N_ACTIONS})")

# -----------------------------
# Pydantic modelleri
# -----------------------------
class StartIn(BaseModel):
    user_id: int

class StartOut(BaseModel):
    session_id: str
    state: list[float]
    eps: float

class NextOut(BaseModel):
    question_id: str
    word_id: int
    prompt: str
    options: list[dict]
    bucket_level: str
    action: int
    state: list[float]
    epsilon: float
    policy: str

class AnswerIn(BaseModel):
    user_id: int
    session_id: str
    question_id: str
    word_id: int
    selected_text: str
    response_ms: int
    bucket_level: str
    action: int

class StepOut(BaseModel):
    correct: bool
    reward: float
    new_state: list[float]
    loss: float | None

# -----------------------------
# Endpoint’ler
# -----------------------------
@app.post("/session/start", response_model=StartOut)
def start_session(inp: StartIn, db: Session = Depends(get_db)):
    user = db.get(User, inp.user_id) or (_ for _ in ()).throw(HTTPException(404, "user not found"))
    s = build_state(db, user)
    # State-boyutu güvenlik kontrolü
    assert len(s) == STATE_DIM, f"State len {len(s)} != expected {STATE_DIM}"
    agent = agent_registry.get(user.id)
    eps = getattr(agent, "eps", 0.2)
    return {"session_id": str(uuid.uuid4()), "state": s, "eps": eps}

@app.get("/rl/next", response_model=NextOut)
def rl_next(user_id: int, session_id: str, db: Session = Depends(get_db)):
    user = db.get(User, user_id) or (_ for _ in ()).throw(HTTPException(404, "user not found"))
    s = build_state(db, user)
    assert len(s) == STATE_DIM, f"State len {len(s)} != expected {STATE_DIM}"

    agent = agent_registry.get(user.id)
    target_idx = level_idx(user.target_level)
    a = agent.act_biased(s, target_idx=target_idx)
    eps = getattr(agent, "eps", 0.0)
    policy = "biased_dqn"

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

    # Kelime için stat yoksa oluştur
    st = db.get(UserWordStat, {"user_id": user.id, "word_id": w.id})
    if not st:
        db.add(UserWordStat(user_id=user.id, word_id=w.id)); db.commit()

    # Session state'ini sakla (feedback’te kullanılacak)
    session_states[session_id] = {
        "state": s,
        "action": a,
        "word_id": w.id,
        "epsilon": eps,
        "policy": policy,
    }
    # app.py -> /rl/next içinde return’dan önce
    print(f"[next] user={user.id} target={user.target_level} eps={eps:.3f} action={bucket_level}")


    return {
        "question_id": qid,
        "word_id": w.id,
        "prompt": w.l2_text,
        "options": opts,
        "bucket_level": bucket_level,
        "action": a,
        "state": s,
        "epsilon": eps,
        "policy": policy,
    }

@app.post("/rl/answer", response_model=StepOut)
def rl_answer(inp: AnswerIn, db: Session = Depends(get_db)):
    user = db.get(User, inp.user_id) or (_ for _ in ()).throw(HTTPException(404, "user not found"))
    word = db.get(Word, inp.word_id) or (_ for _ in ()).throw(HTTPException(404, "word not found"))

    correct = (inp.selected_text.strip().lower() == word.l1_text.strip().lower())

    # UserWordStat güncelle
    st = db.get(UserWordStat, {"user_id": user.id, "word_id": word.id})
    if not st:
        st = UserWordStat(user_id=user.id, word_id=word.id); db.add(st)
    if correct:
        st.alpha += 1.0
        st.reps  += 1
        st.interval_days = 1 if st.interval_days == 0 else int(st.interval_days * 1.6 + 1)
    else:
        st.beta  += 1.0
        st.reps   = max(0, st.reps - 1)
        st.interval_days = 0
    st.last_seen = datetime.utcnow()
    st.due_at    = st.last_seen + timedelta(days=st.interval_days or 1)

    # Attempt kaydı
    db.add(Attempt(
        user_id=user.id, word_id=word.id, is_correct=bool(correct),
        response_ms=inp.response_ms, level=word.level, session_id=inp.session_id
    ))

    # Seviye istatistikleri
    st_lv = db.get(UserLevelStat, {"user_id": user.id, "level": word.level})
    if not st_lv:
        st_lv = UserLevelStat(user_id=user.id, level=word.level); db.add(st_lv)
    if st_lv.correct is None:
        st_lv.correct = 0
    if st_lv.wrong is None:
        st_lv.wrong = 0
    if correct: st_lv.correct += 1
    else:       st_lv.wrong   += 1

    db.commit()

    # Ödül & yeni state
    now = datetime.utcnow()
    due_flag = (st.due_at is None) or (st.due_at <= now)
    r = compute_reward(correct, word.level, user.target_level, due_flag, inp.response_ms)

    # Otomatik seviye ayarı (yükselt/düşür)
    level_changed = adjust_target_level(db, user)

    s2 = build_state(db, user)

    # RL güncelleme
    agent = agent_registry.get(user.id)

    # /rl/next’te hesaplanan state-action’ı kullan
    session_data = session_states.pop(inp.session_id, None)
    if session_data:
        s = session_data["state"]
        a = session_data["action"]
    else:
        # Fallback (ideal değil ama çökmesin):
        s = build_state(db, user)
        a = inp.action

    agent.push(s, int(a), float(r), s2, False)
    loss = agent.train_step()

    if agent.steps % 2000 == 0:
        agent.hard_update()

    if loss is not None and len(agent.buf) >= 256 and agent.steps % 500 == 0:
        agent_registry.update_global_from_user(user.id, tau=0.1)

    return {"correct": bool(correct), "reward": float(r), "new_state": s2, "loss": loss}
