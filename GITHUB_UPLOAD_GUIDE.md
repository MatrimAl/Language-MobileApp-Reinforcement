# 🚀 GitHub'a Yükleme Rehberi

## 📋 Hazırlık Adımları

### 1. Güvenlik Kontrolü
```powershell
# Güvenlik kontrolü scriptini çalıştır
.\check-security.ps1
```

Bu script şunları kontrol eder:
- ✅ `.env` dosyası git'te yok mu?
- ✅ API key'ler hardcode edilmemiş mi?
- ✅ Büyük model dosyaları var mı?
- ✅ `.gitignore` düzgün ayarlanmış mı?

### 2. Git Repository Başlatma

```powershell
# Git'i başlat (eğer henüz başlatmadıysan)
git init

# Tüm dosyaları ekle (.gitignore otomatik filtreler)
git add .

# Nelerin ekleneceğini kontrol et
git status

# İlk commit
git commit -m "feat: Complete RL language learning platform with PyTorch DQN"
```

### 3. GitHub Repository Oluştur

1. [GitHub](https://github.com) üzerinde yeni repository oluştur
2. Repository adı: `reinforcement-learning-language-app` (veya istediğin isim)
3. **Description'ı kopyala:** `GITHUB_DESCRIPTION.md` dosyasından
4. **Public** veya **Private** seç
5. **README, .gitignore, license ekleme** - bunlar zaten var

### 4. Remote Ekle ve Push Et

```powershell
# GitHub remote'u ekle (URL'i kendi repo'nunla değiştir)
git remote add origin https://github.com/KULLANICI_ADIN/REPO_ADIN.git

# Ana branch'i main olarak ayarla
git branch -M main

# Push et
git push -u origin main
```

### 5. GitHub Repository Ayarları

#### About Section
`GITHUB_DESCRIPTION.md` dosyasındaki "About Section" kısmını kopyala:

```
🎓 Reinforcement Learning Language Learning Platform

An intelligent language learning system that uses PyTorch-based DQN to dynamically adjust difficulty levels based on real-time user performance.

Features: DQN with experience replay, Adaptive difficulty, Real-time analytics, FastAPI REST API, React Native mobile app

Status: Fully functional backend & dashboard | Mobile app in development

Technologies: Python, PyTorch, FastAPI, React Native, Streamlit, MongoDB
```

#### Topics (Tags)
Settings → Topics kısmına şunları ekle:
```
reinforcement-learning
deep-q-network
language-learning
pytorch
fastapi
react-native
adaptive-learning
artificial-intelligence
educational-technology
machine-learning
dqn
streamlit
python
mobile-app
thesis-project
```

#### Website (Opsiyonel)
Eğer canlıya aldıysan demo URL'i ekle

## 🔒 Güvenlik Kontrol Listesi

Push etmeden önce:

- [ ] `.env` dosyası `.gitignore` içinde
- [ ] API key'ler environment variable'larda
- [ ] Şifreler hardcode edilmemiş
- [ ] `.env.example` güncel (dummy değerlerle)
- [ ] `check-security.ps1` çalıştırıldı ve geçti
- [ ] Model dosyaları (eğer çok büyükse) ignore edilmiş
- [ ] Personal data içeren database dump'ları yok

## 📦 .gitignore'da Olan Dosyalar

Bu dosyalar **GitHub'a yüklenmeyecek**:

### 🔐 Hassas
- `.env` - API key'ler ve şifreler
- `*.pem`, `*.key` - SSL sertifikaları
- `secrets.py`, `config.local.py` - Local ayarlar

### 🗄️ Büyük Dosyalar
- `venv/` - Python virtual environment
- `node_modules/` - Node packages
- `models/*.pth` - Eğitilmiş modeller (opsiyonel)

### 💾 Database
- `*.db`, `*.sqlite` - Local database
- Database dump'ları

### 🔧 IDE/OS
- `.vscode/`, `.idea/` - IDE ayarları
- `__pycache__/` - Python cache
- `.DS_Store` - macOS dosyaları

## 🎯 İlk Push Sonrası

### README Güncelleme
1. GitHub'da repo'nu aç
2. README.md otomatik görünecek
3. Görseller eklemek istersen:
   ```markdown
   ![Demo](docs/images/demo.gif)
   ![Architecture](docs/images/architecture.png)
   ```

### GitHub Actions (Opsiyonel)
CI/CD için `.github/workflows/` klasörü ekleyebilirsin:
- Otomatik test
- Lint check
- Security scan

### Releases
İlk stable versiyonu tag'le:
```powershell
git tag -a v1.0.0 -m "Initial release: RL Language Learning Platform"
git push origin v1.0.0
```

## 🌟 GitHub Features

### Issues
- Bug tracking
- Feature requests
- Roadmap

### Projects
- Kanban board
- Sprint planning

### Wiki
- Detaylı dokümantasyon
- Tutorials

### GitHub Pages (Opsiyonel)
Dashboard'u canlı demo olarak yayınla

## 📱 Sonraki Adımlar

1. **README'i düzenle** - Screenshot'lar ekle
2. **CONTRIBUTING.md** oluştur
3. **CODE_OF_CONDUCT.md** ekle
4. **License seç** - MIT önerilir
5. **Star ve Watch** ayarla

## ⚠️ Sorun Yaşarsan

### "Permission denied" hatası:
```powershell
# SSH key oluştur
ssh-keygen -t ed25519 -C "your_email@example.com"

# Public key'i GitHub'a ekle
# Settings → SSH and GPG keys → New SSH key
```

### ".env pushed accidentally":
```powershell
# Hemen SECURITY.md'deki "Accidentally Committed Secrets" bölümünü takip et!
```

### "Large files" hatası (>100MB):
```powershell
# Git LFS kullan
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

## 🎉 Tamamlandı!

Repo'n artık GitHub'da! 🚀

**Paylaş:**
- LinkedIn'de paylaş
- Twitter'da tweet at
- Özgeçmişe ekle

**README Badge Ekle:**
```markdown
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
```
