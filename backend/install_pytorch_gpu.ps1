# PyTorch GPU (CUDA 11.8) Kurulum Scripti

Write-Host "================================" -ForegroundColor Cyan
Write-Host "PyTorch GPU Kurulum Başlıyor..." -ForegroundColor Cyan
Write-Host "CUDA 11.8 için optimize edildi" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Mevcut PyTorch'u kaldır
Write-Host "1. Mevcut PyTorch kaldırılıyor..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio

Write-Host ""
Write-Host "2. PyTorch GPU (CUDA 11.8) kuruluyor..." -ForegroundColor Yellow
Write-Host "   (Bu işlem 2-3 GB indirme gerektirebilir)" -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Write-Host ""
Write-Host "3. Kurulum doğrulanıyor..." -ForegroundColor Yellow
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "Kurulum tamamlandı!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
