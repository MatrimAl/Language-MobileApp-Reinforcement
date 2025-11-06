"""
B2'ye bir doğru cevap ekle ve seviye ayarlamasını test et
"""
from db import SessionLocal
from model import User, UserLevelStat
from state import adjust_target_level

db = SessionLocal()

# B2 istatistiğini güncelle (1 doğru cevap ekle)
user = db.query(User).filter(User.id == 1).first()
b2_stat = db.get(UserLevelStat, {"user_id": 1, "level": "B2"})

print("=" * 70)
print("🔬 B2 PERFORMANSINI ARTIRMA TESTİ")
print("=" * 70)

print(f"\n📊 MEVCUT DURUM:")
print(f"B2: {b2_stat.correct}/{b2_stat.correct + b2_stat.wrong} = {100*b2_stat.correct/(b2_stat.correct + b2_stat.wrong):.1f}%")
print(f"Hedef Seviye: {user.target_level}")

# 5 doğru cevap ekle (kesin %75'i geçmek için)
print(f"\n🔧 B2'ye 5 doğru cevap ekleniyor...")
b2_stat.correct += 5
db.commit()

print(f"\n📊 YENİ DURUM:")
total = b2_stat.correct + b2_stat.wrong
acc = 100 * b2_stat.correct / total
print(f"B2: {b2_stat.correct}/{total} = {acc:.1f}%")

# Seviye ayarlamasını test et
print(f"\n🔄 adjust_target_level() çağrılıyor...")
result = adjust_target_level(db, user)

if result:
    print(f"✅ BAŞARILI! Seviye değişti: B2 → {user.target_level} 🎉")
else:
    print(f"❌ Seviye değişmedi. Hala: {user.target_level}")
    print(f"   (Şartları kontrol edin - belki son 20 başarı eksik?)")

db.close()
