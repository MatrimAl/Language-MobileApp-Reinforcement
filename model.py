from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, Boolean, ForeignKey, Float, Index
from datetime import datetime
from db import Base

LEVELS = ["A1","A2","B1","B2","C1"]

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    target_level: Mapped[str] = mapped_column(String(3), default="A2")
    stats_levels = relationship("UserLevelStat", back_populates="user", cascade="all, delete-orphan")
    word_stats  = relationship("UserWordStat",  back_populates="user", cascade="all, delete-orphan")

class Word(Base):
    __tablename__ = "words"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    l1_text: Mapped[str] = mapped_column(String(128))  # ana dil
    l2_text: Mapped[str] = mapped_column(String(128))  # hedef dil
    level:   Mapped[str] = mapped_column(String(3))    # A1..C1
    pos:     Mapped[str] = mapped_column(String(16), default="noun")
    tags:    Mapped[str] = mapped_column(String(128), default="")
    __table_args__ = (Index("ix_words_level", "level"),)

class UserWordStat(Base):
    __tablename__ = "user_word_stats"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"), primary_key=True)
    alpha:   Mapped[float] = mapped_column(Float, default=1.0)
    beta:    Mapped[float]  = mapped_column(Float, default=1.0)
    reps:    Mapped[int]    = mapped_column(Integer, default=0)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    interval_days: Mapped[int] = mapped_column(Integer, default=0)
    due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    user = relationship("User", back_populates="word_stats")
    word = relationship("Word")

class Attempt(Base):
    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"))
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    response_ms: Mapped[int] = mapped_column(Integer, default=0)
    level: Mapped[str] = mapped_column(String(3))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    session_id: Mapped[str] = mapped_column(String(64), default="")
    __table_args__ = (Index("ix_attempts_user_time", "user_id", "created_at"),)

class UserLevelStat(Base):
    __tablename__ = "user_level_stats"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    level:   Mapped[str] = mapped_column(String(3), primary_key=True)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    wrong:   Mapped[int] = mapped_column(Integer, default=0)
    user = relationship("User", back_populates="stats_levels")