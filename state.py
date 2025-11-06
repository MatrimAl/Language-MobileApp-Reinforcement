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

# state.py -> compute_reward

def compute_reward(correct: bool, word_level: str, target_level: str, due: bool, resp_ms: int) -> float:
    base = 1.0 if correct else -0.15

    t = level_idx(target_level)
    a = level_idx(word_level)
    gap = abs(a - t)

    # Hedef yakınlığı: hedefte büyük, komşuda küçük
    if gap == 0:
        diff = +0.80     # hedef
    elif gap == 1:
        diff = +0.25     # komşu
    elif gap == 2:
        diff = +0.10
    else:
        diff = 0.00

    # ALT seçim (hedefin altına inmek) için ekstra caydırma
    below = max(0, t - a)         # a < t ise 1..4
    diff -= 0.20 * below          # alt seviye seçime ceza

    due_bonus = 0.10 if (due and correct) else 0.0
    time_pen  = 0.0 if resp_ms <= 6000 else min(0.2, (resp_ms-6000)/20000)

    return float(base + diff + due_bonus - 0.05*time_pen)


def adjust_target_level(db: Session, user: User, min_attempts: int = 20) -> bool:
    """
    Otomatik seviye ayarlama mekanizması.
    
    Kurallar:
    - Hedef seviyede %75+ başarı + son 20'de %70+ → Seviye yükselt
    - Hedef seviyede %40 altı başarı → Seviye düşür
    - Minimum 20 deneme gerekli
    
    Returns:
        True eğer seviye değiştiyse, False değilse
    """
    # Toplam deneme sayısını kontrol et
    total_attempts = db.query(Attempt).filter(Attempt.user_id == user.id).count()
    if total_attempts < min_attempts:
        return False
    
    # Hedef seviyedeki performansı al
    current_target = user.target_level
    target_stat = db.get(UserLevelStat, {"user_id": user.id, "level": current_target})
    
    if target_stat is None:
        return False
    
    # Hedef seviyedeki başarı oranı
    total_at_target = target_stat.correct + target_stat.wrong
    if total_at_target < 10:  # En az 10 deneme hedef seviyede olmalı
        return False
    
    target_acc = target_stat.correct / total_at_target
    
    # Son 20 denemede genel başarı oranı
    recent_acc = moving_accuracy(db, user.id, k=20)
    
    current_idx = level_idx(current_target)
    new_level = None
    
    # YÜKSELTME KOŞULU: Hedef seviyede çok başarılı + genel de iyi
    if target_acc >= 0.75 and recent_acc >= 0.70:
        if current_idx < len(LEVELS) - 1:  # C1'den yukarı çıkamaz
            new_level = LEVELS[current_idx + 1]
            print(f"📈 Seviye yükseltme: {current_target} → {new_level} (Hedef acc: {target_acc:.1%}, Son 20: {recent_acc:.1%})")
    
    # DÜŞÜRME KOŞULU: Hedef seviyede zorlanıyor
    elif target_acc < 0.40:
        if current_idx > 0:  # A1'den aşağı inemez
            new_level = LEVELS[current_idx - 1]
            print(f"📉 Seviye düşürme: {current_target} → {new_level} (Hedef acc: {target_acc:.1%})")
    
    # Seviye değişikliği uygula
    if new_level:
        user.target_level = new_level
        db.commit()
        return True
    
    return False
