# tests/test_policy_bias.py
import os
import sys
import random
from collections import Counter
import numpy as np

# Proje kökünü Python path'ine ekle (tests/ klasörü içinden çalışırken gerekli)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rl import AgentRegistry       # hibrit rl.py dosyandaki sınıf
# Eğer LEVELS sabitin model.py'de ise:
LEVELS = ["A1","A2","B1","B2","C1"]

random.seed(42)
np.random.seed(42)

STATE_DIM = 13           # build_state çıktınla aynı
N_ACTIONS = len(LEVELS)

def fake_state(target_idx: int):
    """
    13 boyutlu sahte state üretelim.
    Yapı (seninkine paralel): 5 seviye acc + moving_acc + resp_time + due + 5 one-hot
    - Burada basit bir senaryo: hedef seviye acc=0.6, komşu=0.55, diğerleri 0.5
    """
    accs = [0.5]*5
    accs[target_idx] = 0.6
    if target_idx-1 >= 0: accs[target_idx-1] = 0.55
    if target_idx+1 < 5: accs[target_idx+1] = 0.55

    moving_acc = 0.55
    resp_norm  = 0.3
    due_ratio  = 0.2

    one_hot = [1.0 if i==target_idx else 0.0 for i in range(5)]

    vec = accs + [moving_acc, resp_norm, due_ratio] + one_hot
    assert len(vec) == STATE_DIM
    return vec

def sample_correct_prob(action_idx: int, target_idx: int):
    """
    Basit başarı olasılığı modeli:
      hedef = 0.75
      hedefin komşuları = 0.65
      uzak = 0.50
    """
    gap = abs(action_idx - target_idx)
    if gap == 0: return 0.75
    if gap == 1: return 0.65
    return 0.50

def synthetic_reward(correct: bool, action_idx: int, target_idx: int, resp_ms=3000):
    """
    compute_reward ile uyumlu basit bir yaklaşım:
      - doğru: +1.0, yanlış: -0.15
      - hedef yakınlık bonusu: 0:+1.0, 1:+0.6, 2:+0.2, 3:-0.2, 4:-0.4 (x0.5)
    """
    gap = abs(action_idx - target_idx)
    gap_map = {0:+1.0, 1:+0.6, 2:+0.2, 3:-0.2, 4:-0.4}
    base = 1.0 if correct else -0.15
    r = base + 0.5*gap_map.get(gap, -0.4)
    # süre ve due'yu sadeleştirdik
    return float(r)

def run_sim(target_level="B2", steps=400, batch_size=64, hard_update_every=200):
    target_idx = LEVELS.index(target_level)
    # Registry: state_dim'i doğru ayarla
    agents = AgentRegistry(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    ag = agents.get(user_id=1)  # tek kullanıcıyı test ediyoruz

    hist = Counter()
    rewards = []
    eps_track = []

    for t in range(1, steps+1):
        s = fake_state(target_idx)
        # Güvenlik: boyut uyuşması
        assert len(s) == agents.backbone.body[0].in_features

        a = ag.act_biased(s, target_idx=target_idx)
        hist[a] += 1

        # Doğruluk simülasyonu
        p = sample_correct_prob(a, target_idx)
        correct = (random.random() < p)

        r = synthetic_reward(correct, a, target_idx)
        rewards.append(r)

        s2 = fake_state(target_idx)   # bu örnekte sabit tutuyoruz
        done = False

        ag.push(s, a, r, s2, done)
        loss = ag.train_step(batch_size=batch_size)

        if t % hard_update_every == 0:
            ag.hard_update()

        eps_track.append(ag.eps)

    return hist, np.mean(rewards), eps_track

if __name__ == "__main__":
    hist, avg_r, eps = run_sim(target_level="B2", steps=400)
    total = sum(hist.values())
    print("Toplam adım:", total)
    for i, lvl in enumerate(LEVELS):
        print(f"{lvl}: {hist[i]}  ({hist[i]/total:.1%})")
    print("Ortalama reward:", round(avg_r, 3))
    print("ε ilk/son:", round(eps[0],3), "→", round(eps[-1],3))
