"""Connexion : engine + sessionmaker à partir de DATABASE_URL (SQLite par défaut,
PostgreSQL plus tard sans changer le reste du code)."""
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from coach.db.models import Base

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URL = f"sqlite:///{(ROOT / 'data' / 'db' / 'coach.db').as_posix()}"


def get_engine(url: str | None = None) -> Engine:
    url = url or os.environ.get("DATABASE_URL") or DEFAULT_URL
    if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def init_db(url: str | None = None) -> sessionmaker:
    """Crée les tables si elles n'existent pas et renvoie une fabrique de sessions."""
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
