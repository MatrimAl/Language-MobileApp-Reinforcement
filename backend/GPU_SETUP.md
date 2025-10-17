# GPU Kurulum Rehberi

## Mevcut Durum
- ✅ CUDA 11.8 kurulu (nvcc doğrulandı)
- ✅ PyTorch CPU versiyonu çalışıyor
- ❌ Disk alanı yetersiz (2.6 GB boş, PyTorch CUDA 2.8 GB)

## GPU için Gerekli Adımlar

### 1. Disk Alanı Temizliği (Gerekli: ~3 GB)

Aşağıdaki yerlerden boş alan oluştur:

```powershell
# Temp dosyalarını temizle
Remove-Item -Recurse -Force $env:TEMP\* -ErrorAction SilentlyContinue

# Windows Update cache temizle
dism.exe /online /Cleanup-Image /StartComponentCleanup

# Disk Cleanup çalıştır
cleanmgr.exe
```

### 2. PyTorch CUDA Kurulumu

Disk temizliği sonrası:

```powershell
cd c:\Users\matri\OneDrive\Masaüstü\reinFORCING_the_people\backend

# CPU versiyonunu kaldır
pip uninstall torch torchvision -y

# CUDA 11.8 versiyonunu kur
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. GPU Doğrulama

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Kod Değişikliği Gerekmez!

DQN agent zaten GPU'yu otomatik algılıyor:

```python
# dqn_agent.py satır 90-97
self.device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"🔧 Using device: {self.device}")
```

CUDA kurulumu sonrası kod otomatik olarak GPU'yu kullanacak!

## Alternatif: OneDrive Dışında Çalış

OneDrive senkronizasyonu disk alanı tüketebilir. Projeyi yerel diske taşı:

```powershell
# C:\Dev\ klasörüne taşı
xcopy "C:\Users\matri\OneDrive\Masaüstü\reinFORCING_the_people" "C:\Dev\reinFORCING_the_people" /E /I /H

cd C:\Dev\reinFORCING_the_people\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## GPU Eğitim Performansı Beklentisi

| Metrik | CPU | GPU (CUDA) |
|--------|-----|------------|
| 50 episode | ~30 sn | ~10 sn |
| 500 episode | ~5 dk | ~1.5 dk |
| Batch processing | 1x | 5-10x |

GPU ile eğitim **3-5x daha hızlı** olacak!
