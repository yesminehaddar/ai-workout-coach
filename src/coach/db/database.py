"""Connexion : engine + sessionmaker à partir de DATABASE_URL (SQLite par défaut,
PostgreSQL plus tard sans changer le reste du code)."""


def get_engine(url: str = "sqlite:///data/db/coach.db"):
    raise NotImplementedError


def init_db(url: str) -> None:
    """Crée les tables si elles n'existent pas."""
    raise NotImplementedError
