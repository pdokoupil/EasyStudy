"""Central configuration for EasyStudy, driven by environment variables.

All settings fall back to the previous hard-coded defaults so existing local
deployments keep working with zero configuration, while production deployments can
override everything (secret key, database, sessions, redis, port) via the environment
(see docker-compose.yml / .env).
"""
import os
import secrets


def _bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class Config:
    # SECURITY: in production ALWAYS set SECRET_KEY in the environment. The random
    # fallback keeps dev working but invalidates sessions on every restart.
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(24)

    # Database. Default keeps the zero-config SQLite dev experience; set DATABASE_URL
    # (e.g. postgresql+psycopg://user:pass@host/db) for a scalable deployment.
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///db.sqlite")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Sessions. "sqlalchemy" (default) reuses the app DB; set SESSION_TYPE=redis +
    # REDIS_URL for a shared, faster session store under concurrency.
    SESSION_TYPE = os.environ.get("SESSION_TYPE", "sqlalchemy")
    SESSION_COOKIE_NAME = os.environ.get("SESSION_COOKIE_NAME", "easystudy_session")
    SESSION_PERMANENT = False

    # Optional redis (sessions / future task queue). Unset => redis not used at all.
    REDIS_URL = os.environ.get("REDIS_URL")

    PORT = int(os.environ.get("PORT", "5000"))
    DEBUG = _bool(os.environ.get("FLASK_DEBUG"))
