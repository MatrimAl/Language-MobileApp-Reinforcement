"""
User 2 için training scripti - Terminal'de çalışır
"""
import requests
import random
import time

BASE_URL = "http://127.0.0.1:8000"
USER_ID = 2
NUM_EPISODES = 200

print("=" * 60)
print(f"🎯 User {USER_ID} için Training Başlıyor")
print("=" * 60)

# Session başlat
response = requests.post(f"{BASE_URL}/session/start", json={"user_id": USER_ID})
session_data = response.json()
session_id = session_data['session_id']
print(f"✅ Session ID: {session_id}")

# Target level'i database'den al
import sys
sys.path.append('.')
from db import SessionLocal
from model import User
db = SessionLocal()
user = db.query(User).filter_by(id=USER_ID).first()
target_level = user.target_level
db.close()
print(f"🎯 Hedef Seviye: {target_level}\n")

level_counts = {'A1': 0, 'A2': 0, 'B1': 0, 'B2': 0, 'C1': 0}
correct_count = 0
total_count = 0

for episode in range(1, NUM_EPISODES + 1):
    # Soru al
    response = requests.get(f"{BASE_URL}/rl/next", params={"session_id": session_id, "user_id": USER_ID})
    question = response.json()
    
    word_id = question['word_id']
    level = question['bucket_level']
    action = question['action']
    question_id = question['question_id']
    
    # Doğru cevabı bul
    correct_option = [opt for opt in question['options'] if opt['is_correct']][0]
    l1_text = correct_option['text']
    
    level_counts[level] += 1
    
    # Simüle edilmiş cevap (70-90% doğruluk)
    # Hedef seviyeye göre doğruluk oranını ayarla
    if level == target_level:
        correct_prob = 0.75  # Hedef seviye %75
    elif abs(['A1','A2','B1','B2','C1'].index(level) - ['A1','A2','B1','B2','C1'].index(target_level)) == 1:
        correct_prob = 0.70  # Komşu seviyeler %70
    else:
        correct_prob = 0.80  # Uzak seviyeler daha kolay
    
    is_correct = random.random() < correct_prob
    response_time = random.randint(1500, 4000)
    
    if is_correct:
        correct_count += 1
        selected_text = l1_text  # Doğru cevap
    else:
        # Yanlış bir seçenek seç
        wrong_options = [opt for opt in question['options'] if not opt['is_correct']]
        selected_text = random.choice(wrong_options)['text']
    
    total_count += 1
    
    # Cevabı gönder
    response = requests.post(
        f"{BASE_URL}/rl/answer",
        json={
            "user_id": USER_ID,
            "session_id": session_id,
            "question_id": question_id,
            "word_id": word_id,
            "selected_text": selected_text,
            "response_ms": response_time,
            "bucket_level": level,
            "action": action
        }
    )
    
    acc = 100 * correct_count / total_count
    status = "✓" if is_correct else "✗"
    
    if episode % 10 == 0:
        print(f"Episode {episode:3d}: {status} Lvl={level}, Acc={acc:.1f}% | Dağılım: A1={level_counts['A1']} A2={level_counts['A2']} B1={level_counts['B1']} B2={level_counts['B2']} C1={level_counts['C1']}")

print("\n" + "=" * 60)
print(f"✅ Training Tamamlandı!")
print(f"📊 Toplam Accuracy: {100 * correct_count / total_count:.1f}%")
print("\n📈 Seviye Dağılımı:")
for level in ['A1', 'A2', 'B1', 'B2', 'C1']:
    pct = 100 * level_counts[level] / NUM_EPISODES
    marker = " ⭐" if level == target_level else ""
    print(f"  {level}: {level_counts[level]:3d} ({pct:5.1f}%){marker}")
print("=" * 60)
