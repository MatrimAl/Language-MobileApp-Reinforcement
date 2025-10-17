# ⚡ HIZLI GPU KURULUM (CUDA 11.8)

## 🎯 Sisteminiz CUDA 11.8 Kullanıyor

### ✨ Tek Komutla Kurulum

```powershell
cd backend
.\install_pytorch_gpu.ps1
```

### 📝 Manuel Kurulum (3 Adım)

**1. Backend dizinine git:**
```powershell
cd backend
```

**2. Mevcut PyTorch'u kaldır ve GPU versiyonunu kur:**
```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Doğrula:**
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### ✅ Başarılı Kurulum Çıktısı

```
CUDA: True
GPU: [GPU Adınız - örn: NVIDIA GeForce RTX 3060]
```

### 🚀 Backend'i Başlat

```powershell
python main.py
```

Şu mesajı göreceksiniz:
```
🚀 GPU Training Enabled!
   ├─ Device: [GPU Adınız]
   ├─ Memory: [XX.XX GB]
   └─ CUDA Version: 11.8
```

### 📊 Performans Farkı

- **CPU:** 50 episode ~30 saniye
- **GPU:** 50 episode ~5-8 saniye
- **Hız:** 🚀 **5-6x daha hızlı!**

### ❓ Sorun mu Yaşıyorsunuz?

**CUDA algılanmıyor:**
```powershell
# CUDA kontrolü
nvidia-smi

# Driver güncellemesi gerekebilir
```

**Detaylı rehber:** `GPU_SETUP.md` dosyasına bakın.

---

**ÖNEMLİ:** GPU olmadan da sistem çalışır, sadece CPU modunda daha yavaş eğitir.
