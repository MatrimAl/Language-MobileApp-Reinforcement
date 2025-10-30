from db import engine, SessionLocal
from model import Base, User, Word, UserLevelStat, LEVELS
import csv
import os

def run():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    # Kullanıcılar oluştur
    if not db.query(User).filter_by(id=1).first():
        u1 = User(target_level="B1")
        db.add(u1)
        db.commit()
        print("✅ User 1 oluşturuldu (hedef: B1)")
    
    if not db.query(User).filter_by(id=2).first():
        u2 = User(target_level="A2")
        db.add(u2)
        db.commit()
        print("✅ User 2 oluşturuldu (hedef: A2)")
    
    # CSV'den kelimeleri yükle
    if not db.query(Word).first():
        csv_path = os.path.join(os.path.dirname(__file__), "turkish_english_vocab_from_xlsx.csv")
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV dosyası bulunamadı: {csv_path}")
            return
        
        print(f"📁 CSV dosyası okunuyor: {csv_path}")
        
        word_count = 0
        level_counts = {level: 0 for level in LEVELS}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            words_to_add = []
            
            for row in reader:
                l1 = row['l1_text'].strip()
                l2 = row['l2_text'].strip()
                level = row['level'].strip()
                pos = row['pos'].strip()
                
                # İlk satırı (başlık tekrarı) atla
                if l1 == 'Turkish' or l2 == 'English meaning(s)':
                    continue
                
                # Geçerli seviye kontrolü
                if level not in LEVELS:
                    print(f"⚠️ Geçersiz seviye atlandı: {l1} ({level})")
                    continue
                
                words_to_add.append(Word(
                    l1_text=l1,
                    l2_text=l2,
                    level=level,
                    pos=pos
                ))
                
                word_count += 1
                level_counts[level] += 1
                
                # Her 100 kelimede bir batch ekle (performans için)
                if len(words_to_add) >= 100:
                    db.add_all(words_to_add)
                    db.commit()
                    words_to_add = []
            
            # Kalan kelimeleri ekle
            if words_to_add:
                db.add_all(words_to_add)
                db.commit()
        
        print(f"\n✅ {word_count} kelime yüklendi!")
        print("\n📊 Seviye Dağılımı:")
        for level in LEVELS:
            print(f"  {level}: {level_counts[level]:4d} kelime")
    else:
        word_count = db.query(Word).count()
        print(f"ℹ️ Kelimeler zaten yüklü ({word_count} kelime)")
    
    # Seviye istatistikleri oluştur
    users = db.query(User).all()
    stats_created = 0
    for user in users:
        for lv in LEVELS:
            if not db.get(UserLevelStat, {"user_id": user.id, "level": lv}):
                db.add(UserLevelStat(user_id=user.id, level=lv))
                stats_created += 1
    
    if stats_created > 0:
        db.commit()
        print(f"\n✅ {stats_created} seviye istatistiği oluşturuldu")
    
    db.close()
    print("\n🎉 Seed işlemi tamamlandı!")

if __name__ == "__main__":
    run()
