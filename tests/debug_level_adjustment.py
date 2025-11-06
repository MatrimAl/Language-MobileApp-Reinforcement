"""
Seviye ayarlama fonksiyonunu debug et
"""
from db import SessionLocal
from model import User, UserLevelStat, Attempt
from state import adjust_target_level, moving_accuracy, LEVELS

db = SessionLocal()
user = db.query(User).filter(User.id == 1).first()

print("=" * 70)
print("🔍 SEVİYE AYARLAMA DEBUG")
print("=" * 70)

print(f"\n📌 Mevcut Hedef Seviye: {user.target_level}")
print(f"📊 Toplam Deneme: {db.query(Attempt).filter(Attempt.user_id == 1).count()}")

print("\n" + "=" * 70)
print("📈 TÜM SEVİYELERİN DETAYLI DURUMU")
print("=" * 70)

for level in LEVELS:
    stat = db.get(UserLevelStat, {"user_id": user.id, "level": level})
    if stat:
        total = stat.correct + stat.wrong
        acc = (stat.correct / total * 100) if total > 0 else 0
        marker = " ⭐ HEDEF" if level == user.target_level else ""
        print(f"{level}: {stat.correct:3d}/{total:3d} = {acc:5.1f}%{marker}")
        
        # Hedef seviye için detaylı kontrol
        if level == user.target_level:
            print(f"     ├─ Minimum deneme (10): {'✅' if total >= 10 else '❌'} ({total})")
            print(f"     ├─ Hedef başarı ≥75%: {'✅' if acc >= 75 else '❌'} ({acc:.1f}%)")
            recent_acc = moving_accuracy(db, user.id, k=20)
            print(f"     └─ Son 20 başarı ≥70%: {'✅' if recent_acc >= 0.70 else '❌'} ({recent_acc*100:.1f}%)")
            
            # Yükseltme şartları kontrol
            if total >= 10 and acc >= 75 and recent_acc >= 0.70:
                current_idx = LEVELS.index(level)
                if current_idx < len(LEVELS) - 1:
                    next_level = LEVELS[current_idx + 1]
                    print(f"\n     🚀 UYARI: {level} → {next_level} yükseltme şartları SAĞLANDI!")
                else:
                    print(f"\n     ⚠️  Maksimum seviye (C1) - yükseltilemez")
    else:
        print(f"{level}:   0/  0 =   N/A")

print("\n" + "=" * 70)
print("🔄 MANUEL AYARLAMA FONKSİYONU TEST")
print("=" * 70)

print("\nAdjust_target_level() çağrılıyor...")
result = adjust_target_level(db, user)

if result:
    print(f"✅ Seviye değişti! Yeni hedef: {user.target_level}")
else:
    print(f"❌ Seviye değişmedi. Mevcut hedef: {user.target_level}")

# Son durumu göster
print("\n" + "=" * 70)
print("📊 SON DURUM")
print("=" * 70)
db.refresh(user)
print(f"Hedef Seviye: {user.target_level}")

db.close()
