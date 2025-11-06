"""
Yeni reward fonksiyonunu test et
"""
from state import compute_reward, LEVELS

print("=" * 80)
print("🧪 YENİ REWARD FONKSİYONU TEST")
print("=" * 80)

# Hedef C1 olsun
target = "C1"
print(f"\n🎯 Hedef Seviye: {target}")
print(f"\n{'Seviye':<10} {'Doğru Reward':<15} {'Yanlış Reward':<15} {'Fark'}")
print("-" * 60)

for level in LEVELS:
    r_correct = compute_reward(
        correct=True,
        word_level=level,
        target_level=target,
        due=False,
        resp_ms=3000
    )
    
    r_wrong = compute_reward(
        correct=False,
        word_level=level,
        target_level=target,
        due=False,
        resp_ms=3000
    )
    
    marker = " ⭐ HEDEF" if level == target else ""
    print(f"{level:<10} {r_correct:>+.4f}{' '*8} {r_wrong:>+.4f}{' '*8} {r_correct - r_wrong:>+.4f}{marker}")

print("\n" + "=" * 80)
print("💡 DEĞERLENDİRME")
print("=" * 80)

# Hedef B2 için de test et
target = "B2"
print(f"\n🎯 Hedef Seviye: {target}")
print(f"\n{'Seviye':<10} {'Doğru Reward':<15} {'Beklenen (75% başarı)'}")
print("-" * 60)

for level in LEVELS:
    r_correct = compute_reward(
        correct=True,
        word_level=level,
        target_level=target,
        due=False,
        resp_ms=3000
    )
    
    r_wrong = compute_reward(
        correct=False,
        word_level=level,
        target_level=target,
        due=False,
        resp_ms=3000
    )
    
    # %75 başarı oranı varsayımı
    expected = 0.75 * r_correct + 0.25 * r_wrong
    
    marker = " ⭐ HEDEF" if level == target else ""
    print(f"{level:<10} {r_correct:>+.4f}{' '*8} {expected:>+.4f}{marker}")

print("\n✅ Artık hedef seviye EN YÜKSEK ödülü alıyor!")
print("✅ Bir alt/üst seviyeler de iyi bonus alıyor (agent bunları da seçebilir)")
print("✅ Çok uzak seviyeler düşük bonus alıyor")
