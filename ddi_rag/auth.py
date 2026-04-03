"""
auth.py — JWT authentication helpers and Flask-JWT-Extended setup.

Provides:
    init_jwt(app)         — attach JWT manager to Flask app
    hash_password(pw)     — bcrypt hash
    verify_password(pw, h)— bcrypt verify
    get_current_user()    — returns User from DB for current JWT identity
"""

import logging

import bcrypt
from flask_jwt_extended import JWTManager, get_jwt_identity

from config import JWT_ACCESS_TOKEN_MINS, JWT_SECRET_KEY
from database import get_db
from models import User

log = logging.getLogger("ddi.auth")

_jwt = JWTManager()


def init_jwt(app):
    """Attach JWTManager to the Flask app."""
    app.config["JWT_SECRET_KEY"]              = JWT_SECRET_KEY
    app.config["JWT_ACCESS_TOKEN_EXPIRES"]    = __import__("datetime").timedelta(
        minutes=JWT_ACCESS_TOKEN_MINS
    )
    _jwt.init_app(app)
    log.info("JWT initialised (token TTL: %d min).", JWT_ACCESS_TOKEN_MINS)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def get_current_user() -> User | None:
    """
    Returns the User object for the current JWT identity.
    Returns None if the token is missing or the user no longer exists.
    """
    try:
        user_id = get_jwt_identity()
        if not user_id:
            return None
        with get_db() as db:
            return db.query(User).filter(User.id == user_id).first()
    except Exception:
        return None
