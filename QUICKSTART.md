# 🚀 Quick Start Guide

## Hızlı Kurulum (5 Dakika)

### 1️⃣ Backend Setup

```powershell
# Klasöre git
cd backend

# Virtual environment oluştur ve aktifleştir
python -m venv venv
.\venv\Scripts\Activate.ps1

# Dependencies kur
pip install -r requirements.txt

# .env dosyası oluştur
Copy-Item .env.example .env
```

### 2️⃣ MongoDB Kurulumu

**Option A: Docker (Önerilen)**
```powershell
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

**Option B: MongoDB Compass**
- [MongoDB Community Edition](https://www.mongodb.com/try/download/community) indir ve kur
- Varsayılan port: 27017

### 3️⃣ Backend Başlatma

```powershell
# Backend klasöründe
python main.py
```

✅ Backend çalışıyor: http://localhost:8000

📚 API Docs: http://localhost:8000/docs

### 4️⃣ Sample Data Yükleme

**Yeni terminal açın:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1

# Kelime veritabanını doldur
python seed_data.py
```

### 5️⃣ RL Model Initialization

**Browser'da aç:** http://localhost:8000/docs

1. `POST /api/rl/initialize` endpoint'ini bul
2. "Try it out" → "Execute"
3. 50 kelime ile hızlı model eğitimi (~30 saniye)

### 6️⃣ Dashboard Başlatma

**Yeni terminal:**
```powershell
cd dashboard

# Dependencies kur (ilk sefer)
pip install -r requirements.txt

# Dashboard başlat
streamlit run app.py
```

✅ Dashboard: http://localhost:8501

### 7️⃣ Mobile App (Opsiyonel)

```powershell
cd mobile

# Dependencies kur (ilk sefer)
npm install

# Expo başlat
npx expo start
```

Web'de test için: **w** tuşuna bas

---

## 🧪 Test Senaryosu

### 1. API Health Check
```powershell
curl http://localhost:8000/health
```

### 2. Kullanıcı Oluştur
```powershell
curl -X POST http://localhost:8000/api/users/register `
  -H "Content-Type: application/json" `
  -d '{\"email\":\"test@example.com\",\"username\":\"testuser\",\"password\":\"test123\"}'
```

Response'dan `user_id`'yi kaydet.

### 3. Quiz Al
```powershell
curl -X POST http://localhost:8000/api/learning/quiz `
  -H "Content-Type: application/json" `
  -d '{\"user_id\":\"<USER_ID>\"}'
```

### 4. Dashboard'da Visualize Et
- Browser'da http://localhost:8501
- "RL Visualization" tab'ına git
- State değerlerini ayarla
- "Predict Best Action" tıkla

---

## 📊 Proje Yapısı

```
reinFORCING_the_people/
│
├── backend/                 # Python FastAPI + DQN
│   ├── api/                # REST endpoints
│   ├── dqn_agent.py        # DQN implementasyonu
│   ├── rl_environment.py   # Gym environment
│   ├── main.py             # FastAPI app
│   └── seed_data.py        # Sample data loader
│
├── mobile/                 # React Native app
│   ├── App.js              # Main component
│   └── package.json
│
├── dashboard/              # Streamlit dashboard
│   ├── app.py              # Dashboard UI
│   └── requirements.txt
│
├── notebooks/              # Jupyter notebooks
│   └── 01_dqn_training.ipynb
│
├── docs/                   # Documentation
│   └── presentation_guide.md
│
└── README.md               # Ana dokümantasyon
```

---

## 🎯 Özellikler

### Backend (✅ Tamamlandı)
- [x] FastAPI REST API
- [x] MongoDB integration
- [x] DQN agent (TensorFlow)
- [x] Custom Gym environment
- [x] User management
- [x] Learning history tracking
- [x] Real-time RL predictions

### Dashboard (✅ Tamamlandı)
- [x] Training metrics visualization
- [x] Episode rewards chart
- [x] Epsilon decay tracking
- [x] RL decision visualization
- [x] Q-values bar chart
- [x] Real-time model status

### Mobile (🚧 Basic Prototype)
- [x] Backend connection
- [x] Model status display
- [ ] Quiz UI (Next step)
- [ ] Progress tracking
- [ ] User authentication

---

## 🔧 Troubleshooting

### MongoDB bağlantı hatası
```
pymongo.errors.ServerSelectionTimeoutError
```
**Çözüm:** MongoDB'nin çalıştığından emin ol
```powershell
docker ps  # MongoDB container'ı görmeli
```

### Port already in use (8000)
**Çözüm:** Farklı port kullan
```powershell
# config.py'da API_PORT değiştir
# veya
uvicorn main:app --port 8001
```

### PyTorch import hatası
**Çözüm:** Uyumlu versiyonu kur
```powershell
pip install torch torchvision --upgrade
```

### CUDA hatası (GPU kullanımı isterseniz)
**Not:** PyTorch CPU versiyonu otomatik kurulur. GPU için:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Streamlit beyaz ekran
**Çözüm:** Backend'in çalıştığından emin ol
```powershell
curl http://localhost:8000/health
```

---

## 📚 Detaylı Dokümantasyon

- 📖 [Ana README](../README.md) - Detaylı proje dokümantasyonu
- 🎓 [Sunum Rehberi](../docs/presentation_guide.md) - Bitirme projesi sunumu
- 📓 [Training Notebook](../notebooks/01_dqn_training.ipynb) - Model eğitimi

---

## 💡 Sonraki Adımlar

1. **Mobile UI Geliştirme**
   - Quiz ekranı
   - Progress tracker
   - Gamification

2. **Model İyileştirme**
   - Daha fazla episode ile eğitim
   - Hyperparameter tuning
   - A/B testing

3. **Deployment**
   - Docker containers
   - AWS/GCP deployment
   - CI/CD pipeline

---

## 🤝 Katkıda Bulunma

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📞 Destek

Sorularınız için:
- GitHub Issues
- Email: your-email@example.com

---

**🎉 Başarılı bir bitirme projesi için bol şans!**
