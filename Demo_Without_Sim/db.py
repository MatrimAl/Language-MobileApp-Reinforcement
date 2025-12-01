from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DB_URL = "sqlite:///./mvp.db"

engine = create_engine(
    DB_URL, 
    connect_args={
        "check_same_thread": False,
        "timeout": 30  # 30 saniye timeout - database lock sorununu azaltır
    },
    pool_pre_ping=True  # Bağlantı kontrolü
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()