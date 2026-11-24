import hashlib
import hmac
import os
import time

from dotenv import load_dotenv
from fastapi import Header, HTTPException
import jwt

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").lower()
_raw_secret = os.getenv("JWT_SECRET")
_raw_pepper = os.getenv("PEPPER")

if not _raw_secret or not _raw_pepper:
    if APP_ENV == "production":
        raise RuntimeError("JWT_SECRET and PEPPER must be set in production environment")
    _raw_secret = _raw_secret or "dev_only_secret_key_change_me"
    _raw_pepper = _raw_pepper or "dev_only_pepper"

SECRET = _raw_secret
PEPPER = _raw_pepper.encode("utf-8")
KEY_HASH = os.getenv("API_KEY_HMAC", "").strip()


def _hash(k: str) -> str:
    return hmac.new(PEPPER, k.encode("utf-8"), hashlib.sha256).hexdigest()


async def require_key(x_api_key: str = Header(..., description="Apni API Key yahan dalein")):
    if not KEY_HASH or not hmac.compare_digest(_hash(x_api_key), KEY_HASH):
        raise HTTPException(status_code=401, detail="Invalid or bad API key")
    return True


def issue_token(sub: str, expires_in_seconds: int = 900) -> str:
    now = int(time.time())
    payload = {
        "sub": sub,
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")


def require_token(authorization: str = Header(..., description="Bearer <token>")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must start with Bearer")

    tok = authorization.removeprefix("Bearer ").strip()
    if not tok:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        return jwt.decode(tok, SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired JWT token")