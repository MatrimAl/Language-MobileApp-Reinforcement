# state.py
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from model import LEVELS, Attempt, User, UserLevelStat, UserWordStat
from datetime import datetime

def level_idx(level: str) -> int: return LEVELS.index(level)

def laplace_acc(c: int, w: int) -> float: return (c + 1.0) / (c + w + 2.0)

def compute_level_accs(db: Session, user_id: int):
    accs = []
    for lv in LEVELS:
        st = db.get(UserLevelStat, {"user_id": user_id, "level": lv})
        accs.append(0.5 if st is None else laplace_acc(st.correct, st.wrong))
    return accs

def moving_accuracy(db: Session, user_id: int, k: int = 50) -> float:
    q = (select(Attempt.is_correct)
         .where(Attempt.user_id == user_id)
         .order_by(Attempt.created_at.desc())
         .limit(k))
    rows = db.execute(q).scalars().all()
    if not rows: return 0.5
    return sum(1 for x in rows if x) / len(rows)

def avg_response_ms(db: Session, user_id: int, k: int = 50) -> float:
    q = (select(Attempt.response_ms)
         .where(Attempt.user_id == user_id)
         .order_by(Attempt.created_at.desc())
         .limit(k))
    rows = db.execute(q).scalars().all()
    if not rows: return 6000.0
    return sum(rows)/len(rows)

def due_ratio(db: Session, user_id: int) -> float:
    now = datetime.utcnow()
    total = db.query(UserWordStat).filter(UserWordStat.user_id == user_id).count()
    due   = (db.query(UserWordStat)
               .filter(UserWordStat.user_id == user_id)
               .filter((UserWordStat.due_at == None) | (UserWordStat.due_at <= now))
               .count())
    return 0.0 if total == 0 else due/total

def build_state(db: Session, user: User) -> list[float]:
    accs = compute_level_accs(db, user.id)     # 5
    mov = moving_accuracy(db, user.id)         # 1
    avg_ms = avg_response_ms(db, user.id)      # 1
    avg_ms_norm = max(0.0, min(1.0, (avg_ms / 12000.0)))
    due = due_ratio(db, user.id)               # 1
    t_one = [0.0]*len(LEVELS)
    t_one[level_idx(user.target_level)] = 1.0  # 5
    return accs + [mov, avg_ms_norm, due] + t_one  # toplam 13 özellik

def compute_reward(correct: bool, word_level: str, target_level: str, due: bool, resp_ms: int) -> float:
    r = 1.0 if correct else 0.0
    gap = level_idx(word_level) - level_idx(target_level)  # +1: bir üst seviye
    diff_bonus = { -2:-0.1, -1:-0.05, 0:0.0, 1:0.2, 2:0.1 }.get(gap, -0.15)
    due_bonus  = 0.1 if (due and correct) else 0.0
    time_pen   = 0.0 if resp_ms <= 6000 else min(0.2, (resp_ms-6000)/20000)
    return r + 0.2*diff_bonus + 0.1*due_bonus - 0.05*time_pen
