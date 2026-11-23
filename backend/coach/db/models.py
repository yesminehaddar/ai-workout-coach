"""Schéma de la base SQLite (SQLAlchemy). Chaque flèche est un un-à-plusieurs :

users -> sessions -> sets -> reps -> form_events
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    duration_s: Mapped[float | None] = mapped_column(Float)
    avg_score: Mapped[float | None] = mapped_column(Float)
    user: Mapped[User] = relationship(back_populates="sessions")
    sets: Mapped[list["ExerciseSet"]] = relationship(back_populates="session")


class ExerciseSet(Base):
    __tablename__ = "sets"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    exercise: Mapped[str] = mapped_column(String(20))
    rep_count: Mapped[int] = mapped_column(Integer, default=0)
    partial_count: Mapped[int] = mapped_column(Integer, default=0)
    session: Mapped[Session] = relationship(back_populates="sets")
    reps: Mapped[list["Rep"]] = relationship(back_populates="exercise_set")


class Rep(Base):
    __tablename__ = "reps"
    id: Mapped[int] = mapped_column(primary_key=True)
    set_id: Mapped[int] = mapped_column(ForeignKey("sets.id"))
    rep_index: Mapped[int] = mapped_column(Integer)
    min_angle: Mapped[float] = mapped_column(Float)
    max_angle: Mapped[float] = mapped_column(Float)
    tempo_s: Mapped[float] = mapped_column(Float)
    form_score: Mapped[float | None] = mapped_column(Float)
    counted: Mapped[bool] = mapped_column(Boolean, default=True)
    exercise_set: Mapped[ExerciseSet] = relationship(back_populates="reps")
    events: Mapped[list["FormEvent"]] = relationship(back_populates="rep")


class FormEvent(Base):
    __tablename__ = "form_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    rep_id: Mapped[int] = mapped_column(ForeignKey("reps.id"))
    fault_code: Mapped[str] = mapped_column(String(40))
    tip_text: Mapped[str] = mapped_column(String(200))
    shown_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    rep: Mapped[Rep] = relationship(back_populates="events")
