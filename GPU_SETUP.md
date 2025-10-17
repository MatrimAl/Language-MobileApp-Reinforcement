# 🚀 GPU Eğitim Kurulum Rehberi

## 📋 Durum Kontrolü

### Mevcut Durum
Şu anda sistem **CPU modunda** çalışıyor. GPU kullanımı için CUDA destekli PyTorch gerekli.

### GPU Kontrolü
```powershell
cd backend
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 🎯 GPU Kurulum Adımları

### 1. NVIDIA GPU Kontrolü
Bilgisayarınızda NVIDIA GPU olup olmadığını kontrol edin:
```powershell
nvidia-smi
```

**Eğer GPU varsa:** ✅ Devam edin  
**Eğer GPU yoksa:** ⚠️ CPU modunda çalışmaya devam edebilirsiniz

**CUDA Versiyonunuzu Kontrol Edin:**
```powershell
nvcc --version
# veya
nvidia-smi
```
Bu sistemde **CUDA 11.8** kullanılıyor.

### 2. CUDA Toolkit (Zaten Kurulu)
✅ CUDA 11.8 sisteminizde mevcut - ek kurulum gerekmez!

### 3. PyTorch GPU Kurulumu

#### Otomatik Kurulum (Önerilen)
```powershell
cd backend
.\install_pytorch_gpu.ps1
```

#### Manuel Kurulum
```powershell
cd backend

# Mevcut PyTorch'u kaldır
pip uninstall -y torch torchvision torchaudio

# GPU versiyonunu kur (CUDA 11.8 - SİZİN VERSİYONUNUZ)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Doğrulama
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Not:** CUDA 11.8 kullanıyorsunuz, bu yüzden `cu118` index'ini kullanın.

### 4. Backend'i Yeniden Başlat
```powershell
cd backend
python main.py
```

GPU algılandığında şu mesajı göreceksiniz:
```
🚀 GPU Training Enabled!
   ├─ Device: NVIDIA GeForce RTX 3060 (veya sizin GPU'nuz)
   ├─ Memory: 12.00 GB
   └─ CUDA Version: 11.8
```

## 📊 GPU Performans Farkı

### Eğitim Hızı Karşılaştırması

| Özellik | CPU | GPU (RTX 3060) | GPU (RTX 4090) |
|---------|-----|----------------|----------------|
| 50 Episode | ~30 saniye | ~5-8 saniye | ~2-3 saniye |
| 500 Episode | ~5 dakika | ~50-80 saniye | ~20-30 saniye |
| Batch Size | 32 | 64-128 | 256-512 |

### Önerilen Ayarlar

**CPU için:**
```python
batch_size = 32
memory_size = 10000
```

**GPU için:**
```python
batch_size = 128  # veya 256
memory_size = 50000
```

## 🔍 GPU İzleme

### API Endpoint
```bash
GET http://localhost:8000/api/rl/device/info
```

**Yanıt (GPU ile):**
```json
{
  "agent_loaded": true,
  "device_type": "cuda",
  "is_cuda": true,
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "gpu_count": 1,
  "cuda_version": "11.8",
  "memory_allocated_mb": 45.2,
  "memory_reserved_mb": 128.0,
  "memory_total_gb": 12.0
}
```

### Python Kodu
```python
from dqn_agent import DQNAgent

agent = DQNAgent()
device_info = agent.get_device_info()
print(device_info)
```

## ⚙️ Gelişmiş Ayarlar

### Mixed Precision Training (Opsiyonel)
Daha hızlı eğitim için mixed precision kullanabilirsiniz:

```python
# dqn_agent.py dosyasına eklenebilir
from torch.cuda.amp import autocast, GradScaler

# Training loop içinde
scaler = GradScaler()

with autocast():
    q_values = model(states)
    loss = criterion(q_values, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Multi-GPU (Gelecek Özellik)
Birden fazla GPU için DataParallel kullanılabilir:

```python
if torch.cuda.device_count() > 1:
    self.model = nn.DataParallel(self.model)
```

## 🐛 Sorun Giderme

### "CUDA out of memory" Hatası
```python
# Batch size'ı azalt
agent = DQNAgent(batch_size=32)  # veya 16

# Veya memory'yi temizle
torch.cuda.empty_cache()
```

### CUDA Sürüm Uyumsuzluğu
```powershell
# PyTorch'un desteklediği CUDA versiyonunu kontrol et
python -c "import torch; print(torch.version.cuda)"

# CUDA 11.8 için (SİZİN VERSİYONUNUZ):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### GPU Algılanmıyor
1. NVIDIA sürücüsünü güncelleyin
2. CUDA Toolkit'i yeniden kurun
3. PyTorch'u doğru CUDA versiyonu ile kurun
4. Bilgisayarı yeniden başlatın

## 📈 Benchmark

### Test Scripti
```python
import time
import torch
from dqn_agent import DQNAgent
from rl_environment import LanguageLearningEnv

agent = DQNAgent()
env = LanguageLearningEnv()

start = time.time()

# 100 episode eğitim
for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        if len(agent.memory) > agent.batch_size:
            agent.replay()

end = time.time()
print(f"Süre: {end - start:.2f} saniye")
print(f"Episode başına: {(end - start) / 100:.2f} saniye")
```

## 🎓 Tez Sunumu İçin

GPU kullanımını göstermek için:

1. **Öncesi-Sonrası Karşılaştırması**
   - CPU ile eğitim süresi
   - GPU ile eğitim süresi
   - Hız kazancı (örn: 5x daha hızlı)

2. **GPU Kullanım Grafikleri**
   - Dashboard'a `GET /api/rl/device/info` endpoint'inden veri çek
   - Memory kullanımını göster
   - Batch processing hızını vurgula

3. **Teknik Detaylar**
   - PyTorch + CUDA
   - Automatic device detection
   - Batch parallelization
   - Tensor operations on GPU

## 📚 Ek Kaynaklar

- [PyTorch CUDA Docs](https://pytorch.org/docs/stable/cuda.html)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [GPU Optimization Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**Not:** GPU olmadan da sistem tamamen çalışır. GPU sadece eğitim hızını artırır, sonuçları etkilemez.
