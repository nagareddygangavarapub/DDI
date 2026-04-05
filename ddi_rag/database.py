"""
database.py — SQLAlchemy engine and session factory.

Usage:
    from database import get_db, init_db

    # In Flask routes:
    with get_db() as db:
        db.add(...)
        db.commit()

    # At startup:
    init_db()   ← creates all tables if they don't exist
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

log = logging.getLogger("ddi.db")

Base = declarative_base()

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set in your .env file.")
        is_sqlite = DATABASE_URL.startswith("sqlite")
        kwargs = {"pool_pre_ping": not is_sqlite}
        if not is_sqlite:
            kwargs.update({
                "pool_size": 5,
                "max_overflow": 10,
                "connect_args": {"connect_timeout": 10},
            })
        _engine = create_engine(DATABASE_URL, **kwargs)
        log.info("Database engine created (%s).", "SQLite" if is_sqlite else "PostgreSQL")
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(), autocommit=False, autoflush=False
        )
    return _SessionLocal


@contextmanager
def get_db():
    """Context manager that yields a DB session and handles commit/rollback."""
    Session = _get_session_factory()
    db = Session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Create all tables in the database (safe to call multiple times)."""
    import models  # noqa: F401 — registers models with Base.metadata
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    log.info("Database tables initialised.")


def ping_db() -> bool:
    """Return True if the database is reachable."""
    try:
        with _get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.error("Database ping failed: %s", exc)
        return False
