"""
Sistemi tamamen sıfırla - Database ve agent modellerini temizle
"""
import os
import shutil

print("=" * 80)
print("🔄 SİSTEM SIFIRLANACAK!")
print("=" * 80)

# 1. Database dosyasını sil
db_path = "mvp.db"
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"✅ Database silindi: {db_path}")
else:
    print(f"ℹ️  Database zaten yok: {db_path}")

# 2. __pycache__ temizle
if os.path.exists("__pycache__"):
    shutil.rmtree("__pycache__")
    print("✅ __pycache__ temizlendi")

print("\n" + "=" * 80)
print("✅ Sistem sıfırlandı!")
print("=" * 80)
print("\n📝 Şimdi yapman gerekenler:")
print("1. python seed.py                    # Yeni database oluştur")
print("2. uvicorn app:app --reload          # Server'ı başlat (yeni terminalde)")
print("3. python visualize_learning.py      # Görselleştirme başlat")
print("\nYeni reward fonksiyonu aktif olacak! 🎯")
