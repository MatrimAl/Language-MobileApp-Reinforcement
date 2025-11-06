"""
Test otomatik seviye ayarlama mekanizmasını
"""
import requests
import random

BASE_URL = "http://127.0.0.1:8000"

print("🧪 Otomatik Seviye Ayarlama Testi\n")
print("=" * 60)

# Session başlat
response = requests.post(f"{BASE_URL}/session/start", json={"user_id": 1})
session_data = response.json()
session_id = session_data['session_id']
print(f"✅ Session başlatıldı: {session_id}\n")

# İlk hedef seviyeyi kontrol et (seed.py'de B1 olarak ayarlanmış)
print("📊 İlk hedef seviye: B1 (seed.py'den)")
print("\n🎯 Stratejisi: B1'de sürekli doğru cevap vererek seviyeyi yükseltelim\n")

for episode in range(1, 51):
    # Kelime al
    response = requests.get(f"{BASE_URL}/rl/next", 
                           params={"user_id": 1, "session_id": session_id})
    word_data = response.json()
    
    # Doğru cevabı bul
    correct_option = [opt for opt in word_data['options'] if opt['is_correct']][0]
    
    # Cevabı gönder
    payload = {
        "user_id": 1,
        "session_id": session_id,
        "question_id": word_data['question_id'],
        "word_id": word_data['word_id'],
        "selected_text": correct_option['text'],
        "response_ms": random.randint(2000, 4000),
        "bucket_level": word_data['bucket_level'],
        "action": word_data['action']
    }
    
    response = requests.post(f"{BASE_URL}/rl/answer", json=payload)
    result = response.json()
    
    status = "✓" if result['correct'] else "✗"
    print(f"Episode {episode:2d}: {status} Seviye={word_data['bucket_level']}, Ödül={result['reward']:+.2f}")
    
    # Her 10 episode'da bir durum raporu
    if episode % 10 == 0:
        print(f"\n📈 {episode} episode tamamlandı, seviye değişimi kontrol ediliyor...\n")

print("\n" + "=" * 60)
print("✅ Test tamamlandı!")
print("\nKonsola yazılan '📈 Seviye yükseltme' veya '📉 Seviye düşürme' mesajlarını kontrol edin.")
print("Beklenen: ~20-30 episode sonra B1 → B2 seviye yükseltmesi")
