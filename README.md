RL Language MVP — Kısa README

Bu küçük proje, kullanıcılara seviye bazlı kelime soruları sunan ve PyTorch ile yazılmış bir DQN (Deep Q-Network) agent ile "kişiselleştirilmiş" zorluk seçen bir prototiptir.

Özet
- Backend: FastAPI
- RL: PyTorch DQN (kişiselleştirilmiş Agent per user)
- Veritabanı: SQLite (mvp.db)
- Görselleştirme: matplotlib (canlı izleme)

Önemli dosyalar
- `app.py`        : FastAPI uygulaması (API endpoint'leri)
- `model.py`      : SQLAlchemy modelleri (User, Word, Attempt, vb.)
- `db.py`         : SQLite motoru ve session helper
- `seed.py`       : CSV'den (turkish_english_vocab_from_xlsx.csv) kelimeleri yükler ve başlangıç verisi yaratır
- `state.py`      : State/Reward hesaplama mantığı
- `rl.py`         : DQN agent implementasyonu (AgentRegistry, replay buffer, vs.)
- `visualize_learning.py`: Canlı grafik; manuel (default) veya otomatik (`--auto`) mod
- `turkish_english_vocab_from_xlsx.csv`: Kelime havuzu (CSV)

Hızlı Başlangıç (Windows PowerShell)
1) Gerekli paketleri yükleyin (önerilen):

```powershell
C:/Users/matri/AppData/Local/Programs/Python/Python312/python.exe -m pip install fastapi uvicorn sqlalchemy torch pydantic requests matplotlib
```

(Not: sisteminizde `python` çalışıyorsa `python -m pip install ...` kullanabilirsiniz.)

2) Seed (veritabanı ve kelimeler):

```powershell
C:/Users/matri/AppData/Local/Programs/Python/Python312/python.exe seed.py
```

Bu işlem `mvp.db` dosyasını oluşturup CSV'deki kelimeleri ve bir demo kullanıcıyı yüklüyor.

3) Sunucuyu başlatın (FastAPI):

```powershell
C:/Users/matri/AppData/Local/Programs/Python/Python312/python.exe -m uvicorn app:app --reload --port 8000
```

API dokümantasyonu: http://127.0.0.1:8000/docs

4) Canlı görselleştirme (manuel seçim):
- Manuel mod (pencere açıkken 1/2/3 tuşlarıyla seçim yapabilirsiniz):

```powershell
C:/Users/matri/AppData/Local/Programs/Python/Python312/python.exe visualize_learning.py
```

- Otomatik simülasyon modu (eski davranış, cevaplar otomatik):

```powershell
C:/Users/matri/AppData/Local/Programs/Python/Python312/python.exe visualize_learning.py --auto
```

Kullanım notları
- `visualize_learning.py` çalışırken plot penceresinin önde ve odakta olduğundan emin olun; manuel modda tuş girdileri sadece pencere aktifken alınır.
- Eğer `requests.exceptions.JSONDecodeError` veya bağlantı hatası alırsanız (Expecting value / ConnectionRefused), muhtemelen FastAPI sunucusu çalışmıyordur. Önce sunucuyu başlatın.
- `target_level` (User.target_level): kullanıcının hedef dil seviyesi (A1..C1). Agent bu bilgiyi state içinde one-hot olarak alır ve reward hesaplamasında zorluk bonusu olarak kullanır. (Detaylar `state.py` içinde.)

Hızlı test
- `test_api.py` dosyası otomatik olarak birkaç istek atıp API'yi test eder. Sunucu çalışırken aşağıyı çalıştırabilirsiniz:

```powershell
C:/Users/matri/AppData/Local/Programs/Python/Python312/python.exe test_api.py
```

Sorun giderme
- `mvp.db` dosyası "file in use" hatası verirse öncelikle çalışan Python/uvicorn süreçlerini durdurun.
- Paket versiyon problemleri olursa `pip install --upgrade <package>` deneyin.

İleri adımlar (Öneriler)
- `rl.py` içindeki hiperparametreleri (gamma, lr, target update sıklığı) deneyin
- Reward fonksiyonunu değiştirmek için `state.py` içindeki `compute_reward` fonksiyonuna bakın
- Agent başına model eğitimini devam ettirmek/kalıcı hale getirmek için model ağırlıklarını kaydetme/geri yükleme ekleyin

İsterseniz README'yi genişletip kullanım örnekleri, test çıktıları veya bir `requirements.txt` dosyası ekleyebilirim.
