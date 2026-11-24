from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return f"sqlite:///{BASE_DIR / 'instance' / 'corazones.db'}"
    # Render entrega URLs postgres://; SQLAlchemy necesita postgresql+psycopg2://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-change-me")
    SQLALCHEMY_DATABASE_URI = _database_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    WTF_CSRF_TIME_LIMIT = None
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(8 * 1024 * 1024)))

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SECURE = SESSION_COOKIE_SECURE

    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@corazones.local")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
    ADMIN_FULL_NAME = os.getenv("ADMIN_FULL_NAME", "Administrador Corazones de Alegría")

    DATASET_REFERENCIA = os.getenv(
        "DATASET_REFERENCIA", str(BASE_DIR / "data" / "cognitive_impairment_dataset_increased_5000.csv")
    )
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "modelos" / "modelo_riesgo_cognitivo.joblib"))
    CSV_UPLOAD_FOLDER = os.getenv("CSV_UPLOAD_FOLDER", str(BASE_DIR / "tmp" / "uploads"))
    # Imágenes de los resultados MMSE: fuera de /static para que solo se vean con sesión iniciada
    EVALUACIONES_FOLDER = os.getenv("EVALUACIONES_FOLDER", str(BASE_DIR / "instance" / "evaluaciones"))
