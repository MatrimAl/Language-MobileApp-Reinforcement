import argparse
import requests
import json
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import random

BASE_URL = "http://127.0.0.1:8000"

# CLI: --auto to run automatic simulated responses, otherwise manual mode
parser = argparse.ArgumentParser()
parser.add_argument('--auto', action='store_true', help='Run in automatic mode (simulate answers)')
parser.add_argument('--user', type=int, default=1, help='User ID (default: 1)')
args = parser.parse_args()
AUTO_MODE = args.auto
USER_ID = args.user

# Veri depolama
max_points = 100
data = {
    'episode': deque(maxlen=max_points),
    'reward': deque(maxlen=max_points),
    'loss': deque(maxlen=max_points),
    'accuracy': deque(maxlen=max_points),
    'selected_level': deque(maxlen=max_points),
    'avg_reward': deque(maxlen=max_points),
    'response_time': deque(maxlen=max_points),
}

episode_count = 0
total_correct = 0
total_attempts = 0

# Session başlat
print("🚀 Session başlatılıyor...")
response = requests.post(f"{BASE_URL}/session/start", json={"user_id": USER_ID})
session_data = response.json()
session_id = session_data['session_id']
print(f"✅ Session ID: {session_id} (User ID: {USER_ID})\n")

# Grafik kurulumu
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
axes = [
    fig.add_subplot(gs[0, 0]),  # Ödül
    fig.add_subplot(gs[0, 1]),  # Loss
    fig.add_subplot(gs[0, 2]),  # Accuracy
    fig.add_subplot(gs[0, 3]),  # State tablo
    fig.add_subplot(gs[1, 0]),  # Seviye dağılım
    fig.add_subplot(gs[1, 1]),  # Seviye trend
    fig.add_subplot(gs[1, 2]),  # Cevap süresi
    fig.add_subplot(gs[1, 3]),  # State tablo (devam)
]
fig.suptitle('RL Agent Öğrenme İzleme Paneli', fontsize=14, fontweight='bold')

# Interaction state for manual mode
pending_word = None
user_choice = None
last_result = None

def get_next_word():
    """Agent'tan bir sonraki kelimeyi al"""
    response = requests.get(
        f"{BASE_URL}/rl/next",
        params={"user_id": USER_ID, "session_id": session_id}
    )
    return response.json()

def submit_answer_by_index(word_data, selected_idx):
    """Cevabı gönder (kullanıcının seçtiği seçenek indeksi)"""
    opt = word_data['options'][selected_idx]
    response_ms = random.randint(1000, 6000)
    payload = {
        "user_id": USER_ID,
        "session_id": session_id,
        "question_id": word_data['question_id'],
        "word_id": word_data['word_id'],
        "selected_text": opt['text'],
        "response_ms": response_ms,
        "bucket_level": word_data['bucket_level'],
        "action": word_data['action']
    }
    response = requests.post(f"{BASE_URL}/rl/answer", json=payload)
    return response.json(), response_ms

def on_key(event):
    global user_choice
    if pending_word is None:
        return
    if event.key in ('1','2','3'):
        idx = int(event.key) - 1
        # clamp
        if 0 <= idx < len(pending_word['options']):
            user_choice = idx
            print(f"Seçim yapıldı: tuş {event.key} -> '{pending_word['options'][idx]['text']}'")

fig.canvas.mpl_connect('key_press_event', on_key)

def update_plot(frame):
    global episode_count, total_correct, total_attempts, pending_word, user_choice, last_result

    # Eğer bekleyen bir soru yoksa yeni bir tane al
    if pending_word is None:
        pending_word = get_next_word()
        user_choice = None
        # gösterge metni
        qtxt = f"Soru: {pending_word['prompt']}"
        opts = [o['text'] for o in pending_word['options']]
        opt_lines = '\n'.join([f"{i+1}. {t}" for i,t in enumerate(opts)])
        fig.texts = [t for t in fig.texts if False]  # clear existing texts
        fig.text(0.02, 0.92, qtxt, fontsize=10, weight='bold')
        fig.text(0.02, 0.86, opt_lines, fontsize=10)
        fig.text(0.02, 0.80, "Seçmek için 1/2/3 tuşlarına basın (veya otomatik mod için --auto).", fontsize=9, color='gray')
        return

    # Eğer manuel moddaysak, bekle kullanıcı seçim yapana kadar
    if not AUTO_MODE:
        if user_choice is None:
            return
        else:
            # kullanıcı seçti, gönder ve ilerle
            result, response_ms = submit_answer_by_index(pending_word, user_choice)
    else:
        # otomatik simülasyon: %50..%80 arasında başarı
        success_rate = 0.5 + min(0.3, episode_count * 0.003)
        is_correct = random.random() < success_rate
        # seçilecek index: doğruysa doğru olan, yanlışsa rastgele yanlış olan
        if is_correct:
            idx = next(i for i,o in enumerate(pending_word['options']) if o['is_correct'])
        else:
            wrongs = [i for i,o in enumerate(pending_word['options']) if not o['is_correct']]
            idx = random.choice(wrongs)
        result, response_ms = submit_answer_by_index(pending_word, idx)

    # İstatistikleri güncelle
    episode_count += 1
    total_attempts += 1
    if result['correct']:
        total_correct += 1

    # Veri kaydet
    data['episode'].append(episode_count)
    data['reward'].append(result['reward'])
    data['loss'].append(result['loss'] if result['loss'] is not None else 0)
    data['accuracy'].append(total_correct / total_attempts)
    data['selected_level'].append(pending_word['action'])
    data['response_time'].append(response_ms / 1000)

    if len(data['reward']) >= 20:
        avg_rew = np.mean(list(data['reward'])[-20:])
    else:
        avg_rew = np.mean(list(data['reward']))
    data['avg_reward'].append(avg_rew)
    
    # Son state'i sakla
    current_state = result.get('new_state', [])

    # Grafikleri temizle
    for ax in axes:
        ax.clear()

    # 1. Ödül (axes[0])
    axes[0].plot(data['episode'], data['reward'], 'b-', alpha=0.3, label='Anlık Ödül')
    axes[0].plot(data['episode'], data['avg_reward'], 'r-', linewidth=2, label='Ortalama (20)')
    axes[0].set_title('Ödül İlerlemesi')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Ödül')
    axes[0].legend()

    # 2. Loss (axes[1])
    if any(l > 0 for l in data['loss']):
        axes[1].plot(data['episode'], data['loss'], 'g-')
    else:
        axes[1].text(0.5, 0.5, 'Henüz yeterli veri yok (64+ deneyim gerekli)', ha='center', va='center')
    axes[1].set_title('Training Loss')

    # 3. Accuracy (axes[2])
    axes[2].plot(data['episode'], data['accuracy'], 'purple')
    axes[2].axhline(y=0.5, color='gray', linestyle='--')
    axes[2].set_title('Başarı Oranı')
    axes[2].set_ylim([0,1])

    # 4. State Parametreleri (Üst Tablo) (axes[3])
    if len(current_state) == 13:
        # State'i decode et
        state_labels = [
            f"A1 Acc: {current_state[0]:.2f}",
            f"A2 Acc: {current_state[1]:.2f}",
            f"B1 Acc: {current_state[2]:.2f}",
            f"B2 Acc: {current_state[3]:.2f}",
            f"C1 Acc: {current_state[4]:.2f}",
            f"Moving Acc: {current_state[5]:.2f}",
            f"Resp Time: {current_state[6]:.2f}",
            f"Due Ratio: {current_state[7]:.2f}",
            f"Target: {['A1','A2','B1','B2','C1'][current_state[8:].index(1.0)] if 1.0 in current_state[8:] else 'N/A'}"
        ]
        
        # Tablo verisi hazırla (3 sütun x 3 satır)
        table_data = []
        for i in range(0, 9, 3):
            table_data.append(state_labels[i:i+3])
        
        axes[3].axis('tight')
        axes[3].axis('off')
        table = axes[3].table(cellText=table_data, cellLoc='left', loc='center',
                                  colWidths=[0.33, 0.33, 0.33])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[3].set_title('State Vektörü')
    else:
        axes[3].text(0.5, 0.5, 'State bilgisi bekleniyor...', ha='center', va='center')
        axes[3].axis('off')

    # 5. Seviye dağılım (axes[4])
    level_names = ['A1','A2','B1','B2','C1']
    level_counts = [list(data['selected_level']).count(i) for i in range(5)]
    axes[4].bar(level_names, level_counts)
    axes[4].set_title('Seviye Seçim Dağılımı')

    # 6. Son 20 seviye trend (axes[5])
    if len(data['selected_level'])>0:
        recent_levels = list(data['selected_level'])[-20:]
        recent_episodes = list(data['episode'])[-20:]
        axes[5].scatter(recent_episodes, recent_levels, c=recent_levels, cmap='viridis')
        axes[5].set_yticks([0,1,2,3,4]); axes[5].set_yticklabels(level_names)
    axes[5].set_title('Son 20 Episode - Seviye Trendi')

    # 7. Cevap süresi (axes[6])
    axes[6].plot(data['episode'], data['response_time'], 'orange')
    axes[6].axhline(y=6, color='red', linestyle='--')
    axes[6].set_title('Cevap Süresi (s)')
    
    # 8. State Parametreleri (Alt - Detaylı Bilgi) (axes[7])
    if len(current_state) == 13:
        info_lines = [
            f"Episode: {episode_count}",
            f"Doğruluk: {total_correct}/{total_attempts} ({100*total_correct/total_attempts if total_attempts > 0 else 0:.1f}%)",
            f"Son Ödül: {result.get('reward', 0):.2f}",
            f"Seçilen Seviye: {result.get('selected_level_name', 'N/A')}",
            "",
            "=== State Detayları ===",
            f"A1-C1 Başarı: {current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}, {current_state[3]:.2f}, {current_state[4]:.2f}",
            f"Hareketli Ort: {current_state[5]:.2f}",
            f"Norm. Süre: {current_state[6]:.2f}",
            f"Gecikmiş Oran: {current_state[7]:.2f}",
            f"Hedef Seviye: {['A1','A2','B1','B2','C1'][current_state[8:].index(1.0)] if 1.0 in current_state[8:] else 'N/A'}"
        ]
        axes[7].text(0.05, 0.95, '\n'.join(info_lines), transform=axes[7].transAxes,
                       verticalalignment='top', fontsize=9, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[7].axis('off')
        axes[7].set_title('Detaylı Bilgi')
    else:
        axes[7].text(0.5, 0.5, 'Detaylı bilgi bekleniyor...', ha='center', va='center')
        axes[7].axis('off')

    # tight_layout yerine constrained_layout kullan (tablo uyumlu)
    try:
        plt.tight_layout()
    except:
        pass  # Tablo varsa tight_layout bazen uyarı verir, görmezden gel

    # Konsola bilgi
    status = '✓' if result['correct'] else '✗'
    level_name = level_names[pending_word['action']]
    print(f"Episode {episode_count:3d}: {status} Seviye={level_name}, Ödül={result['reward']:+.2f}, Acc={100*total_correct/total_attempts:.1f}%")

    # reset pending word
    pending_word = None

print("\n🎬 Görselleştirme başlıyor... (Plot penceresindeyken 1/2/3 tuşlarıyla seçim yapabilirsiniz)\n")
ani = FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)
plt.show()

print("\n🎉 Görselleştirme kapandı")
