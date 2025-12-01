"""
Q-value ve Reward analizi - Heatmap görselleştirme
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from db import SessionLocal
from model import User, UserLevelStat, Attempt, LEVELS
from state import build_state, compute_reward
from rl import AgentRegistry

# Setup
db = SessionLocal()
user = db.query(User).filter(User.id == 1).first()

# YENİ: Agent oluşturmadan sadece reward analizi yap
# (Eski agent yüklü olabilir - server'ı yeniden başlatmadıysan)

print("=" * 80)
print("🔍 Q-VALUE VE REWARD ANALİZİ")
print("=" * 80)

# Mevcut state'i al
current_state = build_state(db, user)
print(f"\n📊 Mevcut Durum:")
print(f"Hedef Seviye: {user.target_level}")
print(f"State: {[f'{x:.2f}' for x in current_state]}")

# Q-value'ları göstermek yerine sadece reward analizi yap
print(f"\n⚠️  NOT: Q-value analizi için server'ı yeniden başlat")
print(f"   (Eski agent modeli yüklü olabilir)")
print(f"\n📊 YENİ REWARD ANALİZİ:")

# Farklı senaryolar için reward hesapla
print(f"\n💰 Simüle Edilmiş Reward'lar (Hedef: {user.target_level}):")

# Her seviye için doğru ve yanlış cevap reward'larını hesapla
reward_matrix = np.zeros((5, 2))  # 5 seviye x 2 (doğru/yanlış)

for i, level in enumerate(LEVELS):
    # Doğru cevap
    reward_correct = compute_reward(
        correct=True,
        word_level=level,
        target_level=user.target_level,
        due=False,
        resp_ms=3000
    )
    
    # Yanlış cevap
    reward_wrong = compute_reward(
        correct=False,
        word_level=level,
        target_level=user.target_level,
        due=False,
        resp_ms=3000
    )
    
    reward_matrix[i, 0] = reward_correct
    reward_matrix[i, 1] = reward_wrong
    
    marker = " 🎯" if level == user.target_level else ""
    print(f"{level}: Doğru={reward_correct:+.4f}, Yanlış={reward_wrong:+.4f}{marker}")

# Seviye başarı oranlarını al
stats_matrix = np.zeros(5)
for i, level in enumerate(LEVELS):
    stat = db.get(UserLevelStat, {"user_id": user.id, "level": level})
    if stat:
        total = stat.correct + stat.wrong
        stats_matrix[i] = (stat.correct / total * 100) if total > 0 else 0

# Beklenen reward'ları hesapla (başarı oranı * reward_doğru + (1-başarı) * reward_yanlış)
expected_rewards = np.zeros(5)
for i in range(5):
    acc = stats_matrix[i] / 100
    expected_rewards[i] = acc * reward_matrix[i, 0] + (1 - acc) * reward_matrix[i, 1]

print(f"\n📈 Beklenen Reward'lar (Başarı Oranı Dikkate Alınarak):")
for i, level in enumerate(LEVELS):
    marker = " ⭐ MAX" if i == expected_rewards.argmax() else ""
    marker += " 🎯 TARGET" if level == user.target_level else ""
    print(f"{level}: {expected_rewards[i]:+.4f} (Acc: {stats_matrix[i]:.1f}%){marker}")

# VİZÜALİZASYON
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Q-Values Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
colors = ['red' if level == user.target_level else 'skyblue' for level in LEVELS]
bars = ax1.bar(LEVELS, q_values, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax1.set_title('Q-Values (Hedef: Kırmızı)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Q-Value')
ax1.set_xlabel('Seviye')
ax1.grid(axis='y', alpha=0.3)
# Max değeri işaretle
max_idx = q_values.argmax()
ax1.text(max_idx, q_values[max_idx], f'{q_values[max_idx]:.4f}', 
         ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Reward Heatmap (Doğru/Yanlış)
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(reward_matrix.T, annot=True, fmt='.4f', cmap='RdYlGn', 
            xticklabels=LEVELS, yticklabels=['Doğru', 'Yanlış'],
            cbar_kws={'label': 'Reward'}, ax=ax2, vmin=-0.5, vmax=1.2)
ax2.set_title('Reward Matrix (Cevap Durumuna Göre)', fontsize=12, fontweight='bold')
# Hedef seviyeyi vurgula
target_idx = LEVELS.index(user.target_level)
ax2.add_patch(plt.Rectangle((target_idx, 0), 1, 2, fill=False, edgecolor='red', lw=3))

# 3. Başarı Oranları
ax3 = fig.add_subplot(gs[0, 2])
colors = ['red' if level == user.target_level else 'lightgreen' for level in LEVELS]
ax3.bar(LEVELS, stats_matrix, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(y=75, color='orange', linestyle='--', linewidth=2, label='Seviye Yükseltme Eşiği (75%)')
ax3.set_title('Seviye Başarı Oranları', fontsize=12, fontweight='bold')
ax3.set_ylabel('Başarı (%)')
ax3.set_xlabel('Seviye')
ax3.set_ylim([0, 100])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Beklenen Reward'lar
ax4 = fig.add_subplot(gs[1, 0])
colors = ['red' if level == user.target_level else 'orange' for level in LEVELS]
bars = ax4.bar(LEVELS, expected_rewards, color=colors, alpha=0.7, edgecolor='black')
ax4.set_title('Beklenen Reward (Acc × R_doğru + (1-Acc) × R_yanlış)', 
              fontsize=12, fontweight='bold')
ax4.set_ylabel('Beklenen Reward')
ax4.set_xlabel('Seviye')
ax4.grid(axis='y', alpha=0.3)
# Max değeri işaretle
max_idx = expected_rewards.argmax()
ax4.text(max_idx, expected_rewards[max_idx], f'{expected_rewards[max_idx]:.4f}', 
         ha='center', va='bottom', fontweight='bold', fontsize=10)

# 5. Q-Value vs Expected Reward Karşılaştırma
ax5 = fig.add_subplot(gs[1, 1])
x = np.arange(len(LEVELS))
width = 0.35
bars1 = ax5.bar(x - width/2, q_values, width, label='Q-Value', alpha=0.7, color='skyblue')
bars2 = ax5.bar(x + width/2, expected_rewards, width, label='Expected Reward', alpha=0.7, color='orange')
ax5.set_title('Q-Value vs Beklenen Reward', fontsize=12, fontweight='bold')
ax5.set_ylabel('Değer')
ax5.set_xlabel('Seviye')
ax5.set_xticks(x)
ax5.set_xticklabels(LEVELS)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
# Hedef seviyeyi vurgula
target_idx = LEVELS.index(user.target_level)
ax5.axvline(x=target_idx, color='red', linestyle='--', linewidth=2, alpha=0.5)

# 6. Reward Farkları (Her seviye - Hedef seviye)
ax6 = fig.add_subplot(gs[1, 2])
target_idx = LEVELS.index(user.target_level)
target_reward = expected_rewards[target_idx]
reward_diffs = expected_rewards - target_reward
colors = ['green' if x > 0 else 'red' for x in reward_diffs]
bars = ax6.bar(LEVELS, reward_diffs, color=colors, alpha=0.7, edgecolor='black')
ax6.axhline(y=0, color='black', linewidth=1)
ax6.set_title(f'Reward Farkı (Seviye - {user.target_level})', fontsize=12, fontweight='bold')
ax6.set_ylabel('Fark')
ax6.set_xlabel('Seviye')
ax6.grid(axis='y', alpha=0.3)

plt.suptitle(f'🔍 Q-Value & Reward Analizi (Hedef: {user.target_level})', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('q_value_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Grafik kaydedildi: q_value_analysis.png")
plt.show()

# ÖNEMLİ BULGULAR
print("\n" + "=" * 80)
print("🔍 ÖNEMLİ BULGULAR")
print("=" * 80)

q_max_level = LEVELS[q_values.argmax()]
exp_max_level = LEVELS[expected_rewards.argmax()]

print(f"\n1️⃣  Q-Value'ya göre en iyi seviye: {q_max_level} (Q={q_values.max():.4f})")
print(f"2️⃣  Beklenen reward'a göre en iyi: {exp_max_level} (R={expected_rewards.max():.4f})")
print(f"3️⃣  Hedef seviye: {user.target_level} (Q={q_values[LEVELS.index(user.target_level)]:.4f})")

if q_max_level != user.target_level:
    print(f"\n⚠️  PROBLEM: Agent {q_max_level} seçiyor ama hedef {user.target_level}!")
    print(f"   Q-value farkı: {q_values[LEVELS.index(q_max_level)] - q_values[LEVELS.index(user.target_level)]:.4f}")

if exp_max_level != user.target_level:
    print(f"\n⚠️  DİKKAT: Beklenen reward {exp_max_level}'de en yüksek!")
    print(f"   Sebep: {exp_max_level} seviyesi kolay → Yüksek başarı oranı → Yüksek beklenen reward")
    print(f"   Reward farkı: {expected_rewards[LEVELS.index(exp_max_level)] - expected_rewards[LEVELS.index(user.target_level)]:.4f}")

# Reward tasarım önerisi
print(f"\n💡 ÖNERİLER:")
target_idx = LEVELS.index(user.target_level)
for i, level in enumerate(LEVELS):
    if i != target_idx and expected_rewards[i] > expected_rewards[target_idx]:
        diff = expected_rewards[i] - expected_rewards[target_idx]
        print(f"   • {level} beklenen reward'ı {user.target_level}'den {diff:.4f} daha yüksek")
        print(f"     → Diff bonus katsayısını artır (şu an 0.2)")

db.close()
