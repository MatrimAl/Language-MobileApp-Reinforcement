# A2 Seviye Seçim Problemi - Detaylı Analiz

## 🎯 Problem Özeti

**Durum:** Kullanıcının hedef seviyesi B2 olmasına rağmen, RL agent 20-30 episode boyunca sürekli A2 seviyesi seçti.

**Sistem:** DQN (Deep Q-Network) tabanlı Türkçe-İngilizce kelime öğrenme uygulaması

---

## 📊 Sistem Detayları

### Agent Yapısı
- **Model:** Deep Q-Network (DQN)
- **Mimari:** 13 input → 128 hidden → 128 hidden → 5 output
- **State Boyutu:** 13 özellik
- **Action Sayısı:** 5 (A1, A2, B1, B2, C1)
- **Epsilon:** 0.1 (10% exploration, 90% exploitation)
- **Replay Buffer:** 50,000 transition
- **Batch Size:** 64
- **Learning Rate:** 0.001
- **Gamma:** 0.99
- **Target Network Update:** Her 2000 step'te hard update

### State Vektörü (13 boyutlu)
```python
[
    A1_accuracy,      # 0: A1 seviyesi başarı oranı (Laplace smoothing)
    A2_accuracy,      # 1: A2 seviyesi başarı oranı
    B1_accuracy,      # 2: B1 seviyesi başarı oranı
    B2_accuracy,      # 3: B2 seviyesi başarı oranı
    C1_accuracy,      # 4: C1 seviyesi başarı oranı
    moving_accuracy,  # 5: Son 50 denemenin ortalaması
    response_time,    # 6: Normalize edilmiş cevap süresi (0-1)
    due_ratio,        # 7: Gecikmiş kelime oranı
    target_A1,        # 8: Hedef seviye one-hot (0 for B2)
    target_A2,        # 9: (0 for B2)
    target_B1,        # 10: (0 for B2)
    target_B2,        # 11: (1 for B2) ← HEDEF
    target_C1         # 12: (0 for B2)
]
```

### Reward Fonksiyonu
```python
r = base_reward + 0.2 * diff_bonus + 0.1 * due_bonus - 0.05 * time_penalty

# base_reward: 1.0 (doğru), 0.0 (yanlış)
# diff_bonus: Kelime seviyesi ile hedef seviye arasındaki uyum bonusu
#   - Optimal (0 fark): 1.0
#   - 1 seviye fark: 0.8
#   - 2 seviye fark: 0.6
# due_bonus: Gecikmiş kelime bonusu
# time_penalty: Cevap süresi cezası
```

**Örnek Ödüller (Hedef: B2):**
- B2 kelimesi doğru: `1.0 + 0.2*1.0 + 0.1*x - 0.05*y ≈ 1.00`
- A2 kelimesi doğru: `1.0 + 0.2*0.8 + 0.1*x - 0.05*y ≈ 0.98`
- **Fark sadece 0.02!** ⚠️

---

## 🔍 Gözlemlenen Davranış

### Episode Dağılımı
```
Episode  1-19: Çoğunlukla B2 seçildi ✅
Episode 20   : A2 seçildi (rastgele exploration)
Episode 21-40: Sürekli A2 seçildi! ❌
```

### Performans Metrikleri
```
Seviye | Başarı | Deneme Sayısı
-------|--------|---------------
A1     | 73.1%  |  93
A2     | 79.3%  | 198  ← En yüksek başarı!
B1     | 74.6%  | 551
B2     | 75.1%  | 668
C1     | 63.2%  | 155

Son 20 Episode Başarı: 95.0%
```

### A2 Seçim Paterni
```
Episode 20-40:
[A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, 
 A2, A2, A2, A2, A2, A2, B2]

A2 Ödülleri: 0.98 (istikrarlı ve yüksek)
B2 Ödülleri: 1.00 (ama bazen 0.0 - değişken)
```

---

## ❓ Sorular

### Ana Soru
**Hedef seviye B2 olmasına rağmen agent neden sürekli A2 seçiyor?**

### Alt Sorular

1. **Reward Tasarımı:**
   - A2 ile B2 arasındaki ödül farkı (0.02) çok küçük mü?
   - Hedef seviyeye uygunluk için daha büyük bir bonus gerekli mi?
   - Diff bonus katsayısı 0.2 yerine 0.5 veya 1.0 olmalı mı?

2. **State Temsili:**
   - Hedef seviyeyi one-hot encoding olarak ekledik ama agent bunu yeterince kullanıyor mu?
   - State'e başka özellikler eklemeli miyiz?

3. **Exploration/Exploitation:**
   - Epsilon 0.1 çok mu yüksek?
   - Agent yeterince exploitation yapamıyor mu?
   - Epsilon decay kullanmalı mıyız? (0.1 → 0.01 gibi)

4. **Q-Değerleri:**
   - DQN'nin Q-değerleri doğru yakınsıyor mu?
   - A2'nin Q-değeri yanlışlıkla B2'den yüksek mi öğrenildi?
   - Q-değerlerini nasıl inceleyebiliriz?

5. **Davranış Analizi:**
   - Bu geçici bir exploration phase mi?
   - Yoksa kalıcı bir öğrenme hatası mı?
   - A2'nin yüksek başarı oranı (%79.3) ve istikrarlı ödülleri agent'ı "kandırıyor" mu?

6. **Çözüm Stratejileri:**
   - Reward shaping yapmalı mıyız?
   - Hedef seviyeden uzaklaşma için penalty eklemeli miyiz?
   - Experience replay'deki önceliklendirme değiştirilmeli mi?

---

## 🎯 Beklenen Davranış

**İdeal:**
- Agent çoğunlukla B2 (hedef seviye) seçmeli
- Bazen B1 veya C1 seçebilir (yakın seviyeler)
- %10 exploration ile ara sıra A2 veya A1 seçmeli

**Gerçekleşen:**
- 20-30 episode boyunca sürekli A2 seçildi
- Hedef seviye B2 olmasına rağmen agent "takıldı"

---

## 💡 Hipotezler

### Hipotez 1: Reward Farkı Çok Küçük
A2 kelimeleri çok kolay olduğu için sürekli doğru cevaplanıyor → 0.98 ödül her seferinde garantili. B2 kelimeleri daha zor, bazen yanlış → 0.00 ödül riski var. Agent "güvenli" olanı (A2) tercih ediyor.

**Çözüm:** Diff bonus katsayısını artır (0.2 → 0.5)

### Hipotez 2: State Yeterince İyi Değil
One-hot encoding hedef seviyeyi gösteriyor ama agent DQN ağırlıklarında bunu yeterince kullanmıyor.

**Çözüm:** Hedef seviye ile action arasındaki farkı explicit olarak state'e ekle

### Hipotez 3: Exploration Fazla
%10 exploration ile random A2 seçildi, sonra Q-değerleri bu yönde güncellendi ve agent A2'de "takıldı".

**Çözüm:** Epsilon'u 0.05'e düşür veya epsilon decay kullan

---

## 🛠️ İstenen Yardım

1. Bu davranışın **kök nedenini** bulmak
2. **Çözüm önerileri** almak:
   - Reward fonksiyonu nasıl değiştirilmeli?
   - State tasarımı iyileştirilmeli mi?
   - Hyperparameter'lar (epsilon, learning rate, vb.) ayarlanmalı mı?
3. Benzer problemlerle karşılaşanların **deneyimleri**
4. DQN için **best practices** (RL context'inde hedef-driven selection)

---

## 📎 Kod Örnekleri

### Agent Act Fonksiyonu
```python
def act(self, s, eps=0.1):
    self.steps += 1
    import random
    if random.random() < eps:
        return random.randrange(self.n_actions)
    with torch.no_grad():
        q = self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())
```

### Reward Hesaplama
```python
def compute_reward(correct, word, user, response_ms):
    base = 1.0 if correct else 0.0
    
    # Kelime seviyesi ile hedef seviye arasındaki fark
    word_level_idx = LEVELS.index(word.level)
    target_level_idx = LEVELS.index(user.target_level)
    diff = abs(word_level_idx - target_level_idx)
    diff_bonus = max(0, 1 - diff * 0.2)
    
    # Due bonus
    due_bonus = calculate_due_bonus(...)
    
    # Time penalty
    time_penalty = min(response_ms / 12000.0, 1.0)
    
    r = base + 0.2 * diff_bonus + 0.1 * due_bonus - 0.05 * time_penalty
    return r
```

### State Oluşturma
```python
def build_state(db, user):
    # Seviye başarı oranları (Laplace smoothing)
    level_accs = []
    for level in LEVELS:
        stat = get_level_stat(db, user.id, level)
        acc = (stat.correct + 1) / (stat.correct + stat.wrong + 2)
        level_accs.append(acc)
    
    # Hareketli ortalama
    moving_acc = moving_accuracy(db, user.id, k=50)
    
    # Cevap süresi
    response_time = normalize_response_time(...)
    
    # Due ratio
    due_ratio = calculate_due_ratio(...)
    
    # Hedef seviye one-hot
    target_idx = LEVELS.index(user.target_level)
    target_one_hot = [1.0 if i == target_idx else 0.0 for i in range(5)]
    
    return level_accs + [moving_acc, response_time, due_ratio] + target_one_hot
```

---

**Not:** Bu bir eğitim projesi olduğu için teorik açıklamalar ve pratik çözümler bekliyorum. Teşekkürler! 🙏
