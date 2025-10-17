"""
Sample word data seeder for development
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

MONGODB_URL = "mongodb://localhost:27017"
DB_NAME = "language_learning_rl"

# İngilizce - Türkçe kelime listesi
SAMPLE_WORDS = [
    # Beginner Level (1)
    {"word": "hello", "translation": "merhaba", "difficulty": 1, "language": "en", "category": "greetings"},
    {"word": "goodbye", "translation": "hoşçakal", "difficulty": 1, "language": "en", "category": "greetings"},
    {"word": "please", "translation": "lütfen", "difficulty": 1, "language": "en", "category": "politeness"},
    {"word": "thank you", "translation": "teşekkür ederim", "difficulty": 1, "language": "en", "category": "politeness"},
    {"word": "yes", "translation": "evet", "difficulty": 1, "language": "en", "category": "basics"},
    {"word": "no", "translation": "hayır", "difficulty": 1, "language": "en", "category": "basics"},
    {"word": "water", "translation": "su", "difficulty": 1, "language": "en", "category": "food"},
    {"word": "food", "translation": "yemek", "difficulty": 1, "language": "en", "category": "food"},
    {"word": "home", "translation": "ev", "difficulty": 1, "language": "en", "category": "places"},
    {"word": "family", "translation": "aile", "difficulty": 1, "language": "en", "category": "people"},
    
    # Elementary Level (2)
    {"word": "book", "translation": "kitap", "difficulty": 2, "language": "en", "category": "objects"},
    {"word": "table", "translation": "masa", "difficulty": 2, "language": "en", "category": "furniture"},
    {"word": "chair", "translation": "sandalye", "difficulty": 2, "language": "en", "category": "furniture"},
    {"word": "computer", "translation": "bilgisayar", "difficulty": 2, "language": "en", "category": "technology"},
    {"word": "phone", "translation": "telefon", "difficulty": 2, "language": "en", "category": "technology"},
    {"word": "school", "translation": "okul", "difficulty": 2, "language": "en", "category": "places"},
    {"word": "teacher", "translation": "öğretmen", "difficulty": 2, "language": "en", "category": "professions"},
    {"word": "student", "translation": "öğrenci", "difficulty": 2, "language": "en", "category": "professions"},
    {"word": "friend", "translation": "arkadaş", "difficulty": 2, "language": "en", "category": "people"},
    {"word": "city", "translation": "şehir", "difficulty": 2, "language": "en", "category": "places"},
    
    # Intermediate Level (3)
    {"word": "environment", "translation": "çevre", "difficulty": 3, "language": "en", "category": "nature"},
    {"word": "development", "translation": "gelişme", "difficulty": 3, "language": "en", "category": "abstract"},
    {"word": "opportunity", "translation": "fırsat", "difficulty": 3, "language": "en", "category": "abstract"},
    {"word": "experience", "translation": "deneyim", "difficulty": 3, "language": "en", "category": "abstract"},
    {"word": "knowledge", "translation": "bilgi", "difficulty": 3, "language": "en", "category": "education"},
    {"word": "understand", "translation": "anlamak", "difficulty": 3, "language": "en", "category": "verbs"},
    {"word": "explain", "translation": "açıklamak", "difficulty": 3, "language": "en", "category": "verbs"},
    {"word": "important", "translation": "önemli", "difficulty": 3, "language": "en", "category": "adjectives"},
    {"word": "different", "translation": "farklı", "difficulty": 3, "language": "en", "category": "adjectives"},
    {"word": "government", "translation": "hükümet", "difficulty": 3, "language": "en", "category": "politics"},
    
    # Advanced Level (4)
    {"word": "sophisticated", "translation": "karmaşık/gelişmiş", "difficulty": 4, "language": "en", "category": "adjectives"},
    {"word": "contemporary", "translation": "çağdaş", "difficulty": 4, "language": "en", "category": "adjectives"},
    {"word": "fundamental", "translation": "temel", "difficulty": 4, "language": "en", "category": "adjectives"},
    {"word": "substantial", "translation": "önemli/büyük", "difficulty": 4, "language": "en", "category": "adjectives"},
    {"word": "comprehensive", "translation": "kapsamlı", "difficulty": 4, "language": "en", "category": "adjectives"},
    {"word": "demonstrate", "translation": "göstermek", "difficulty": 4, "language": "en", "category": "verbs"},
    {"word": "establish", "translation": "kurmak", "difficulty": 4, "language": "en", "category": "verbs"},
    {"word": "indicate", "translation": "belirtmek", "difficulty": 4, "language": "en", "category": "verbs"},
    {"word": "philosophy", "translation": "felsefe", "difficulty": 4, "language": "en", "category": "education"},
    {"word": "infrastructure", "translation": "altyapı", "difficulty": 4, "language": "en", "category": "abstract"},
    
    # Expert Level (5)
    {"word": "juxtaposition", "translation": "yan yana koyma", "difficulty": 5, "language": "en", "category": "abstract"},
    {"word": "paradigm", "translation": "paradigma", "difficulty": 5, "language": "en", "category": "abstract"},
    {"word": "ubiquitous", "translation": "her yerde bulunan", "difficulty": 5, "language": "en", "category": "adjectives"},
    {"word": "ephemeral", "translation": "geçici", "difficulty": 5, "language": "en", "category": "adjectives"},
    {"word": "quintessential", "translation": "özünde/tipik", "difficulty": 5, "language": "en", "category": "adjectives"},
    {"word": "serendipity", "translation": "tesadüf eseri bulgu", "difficulty": 5, "language": "en", "category": "abstract"},
    {"word": "eloquent", "translation": "beliğ/etkili konuşan", "difficulty": 5, "language": "en", "category": "adjectives"},
    {"word": "ambiguous", "translation": "belirsiz/muğlak", "difficulty": 5, "language": "en", "category": "adjectives"},
    {"word": "procrastinate", "translation": "ertelemek", "difficulty": 5, "language": "en", "category": "verbs"},
    {"word": "dichotomy", "translation": "ikiye ayrılma", "difficulty": 5, "language": "en", "category": "abstract"},
]

async def seed_words():
    """Sample kelimeleri MongoDB'ye yükle"""
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    words_collection = db.words
    
    # Clear existing words
    await words_collection.delete_many({})
    
    # Insert sample words
    result = await words_collection.insert_many(SAMPLE_WORDS)
    
    print(f"✅ {len(result.inserted_ids)} kelime başarıyla eklendi!")
    print(f"📊 Zorluk dağılımı:")
    for level in range(1, 6):
        count = len([w for w in SAMPLE_WORDS if w["difficulty"] == level])
        level_name = ["Beginner", "Elementary", "Intermediate", "Advanced", "Expert"][level-1]
        print(f"  Level {level} ({level_name}): {count} kelime")
    
    client.close()

if __name__ == "__main__":
    print("🚀 Sample kelime verisi yükleniyor...")
    asyncio.run(seed_words())
