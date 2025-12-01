from db import SessionLocal
from model import Word, User

db = SessionLocal()

print("📊 Database Durumu:")
print(f"Kelime sayısı: {db.query(Word).count()}")
print(f"Kullanıcı sayısı: {db.query(User).count()}")

user = db.query(User).first()
if user:
    print(f"\n👤 User ID: {user.id}")
    print(f"   Hedef: {user.target_level}")
else:
    print("\n❌ Kullanıcı yok!")

db.close()
