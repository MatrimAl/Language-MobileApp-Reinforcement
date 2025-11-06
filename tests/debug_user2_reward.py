"""
User 2 (hedef A2) için farklı seviyelerin reward'larını göster
"""
from state import compute_reward, LEVELS

USER_TARGET = "A2"

print("=" * 70)
print(f"🎯 User Hedef Seviyesi: {USER_TARGET}")
print("=" * 70)
print("\nFarklı seviyelerden doğru cevap verildiğinde reward'lar:\n")

print("Seviye | Doğru Reward | Yanlış Reward | Fark")
print("-" * 60)

for level in LEVELS:
    r_correct = compute_reward(
        correct=True,
        word_level=level,
        target_level=USER_TARGET,
        due=False,
        resp_ms=2000
    )
    
    r_wrong = compute_reward(
        correct=False,
        word_level=level,
        target_level=USER_TARGET,
        due=False,
        resp_ms=2000
    )
    
    diff = r_correct - r_wrong
    marker = " ⭐ HEDEF" if level == USER_TARGET else ""
    
    print(f"{level:6s} | {r_correct:12.4f} | {r_wrong:13.4f} | {diff:6.4f}{marker}")

print("\n" + "=" * 70)
print("💡 Hedef seviye EN YÜKSEK reward'ı almalı!")
print("=" * 70)
