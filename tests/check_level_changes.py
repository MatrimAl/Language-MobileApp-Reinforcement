"""
Seviye değişikliklerini ve istatistikleri göster
"""
import argparse
from db import SessionLocal
from model import User, Attempt, UserLevelStat
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--user', type=int, default=1, help='User ID (default: 1)')
args = parser.parse_args()
USER_ID = args.user

db = SessionLocal()

# Kullanıcı bilgisi
user = db.query(User).filter(User.id == USER_ID).first()
total_attempts = db.query(Attempt).filter(Attempt.user_id == USER_ID).count()

print("=" * 60)
print("📊 KULLANICI PERFORMANS RAPORU")
print("=" * 60)
print(f"\n👤 User ID: {user.id}")
print(f"🎯 Mevcut Hedef Seviye: {user.target_level}")
print(f"📈 Toplam Deneme: {total_attempts}")
print(f"🕐 Oluşturulma: {user.created_at}")

# Seviye başarı istatistikleri
print("\n" + "=" * 60)
print("📚 SEVİYE BAŞARI İSTATİSTİKLERİ")
print("=" * 60)
stats = db.query(UserLevelStat).filter(UserLevelStat.user_id == USER_ID).order_by(UserLevelStat.level).all()
for s in stats:
    total = s.correct + s.wrong
    if total > 0:
        acc = 100 * s.correct / total
        bar = "█" * int(acc / 5)  # Her %5 için bir blok
        print(f"{s.level}: {s.correct:3d}/{total:3d} = {acc:5.1f}% {bar}")
    else:
        print(f"{s.level}:   0/  0 =   N/A")

# Son 20 deneme
print("\n" + "=" * 60)
print("🔍 SON 20 DENEME")
print("=" * 60)
recent = db.query(Attempt).filter(Attempt.user_id == USER_ID).order_by(Attempt.created_at.desc()).limit(20).all()
for i, att in enumerate(reversed(recent), 1):
    status = "✓" if att.is_correct else "✗"
    print(f"{i:2d}. {status} {att.level} | Cevap: {att.response_ms:4d}ms | {att.created_at.strftime('%H:%M:%S')}")

# Seviye dağılımı
print("\n" + "=" * 60)
print("📊 SEÇİLEN SEVİYE DAĞILIMI")
print("=" * 60)
all_attempts = db.query(Attempt).filter(Attempt.user_id == USER_ID).all()
level_counts = Counter([a.level for a in all_attempts])
for level in ['A1', 'A2', 'B1', 'B2', 'C1']:
    count = level_counts.get(level, 0)
    pct = 100 * count / total_attempts if total_attempts > 0 else 0
    bar = "█" * int(pct / 2)  # Her %2 için bir blok
    print(f"{level}: {count:3d} ({pct:5.1f}%) {bar}")

db.close()

print("\n" + "=" * 60)
if total_attempts > 0:
    print(f"✅ User {USER_ID} raporu tamamlandı!")
else:
    print(f"ℹ️  User {USER_ID} henüz deneme yapmamış.")
print("=" * 60)
