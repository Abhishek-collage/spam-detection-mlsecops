import hashlib
import io
import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import joblib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from pymongo import MongoClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config.config_path import MODEL_OUTPUT_PATH
from src.auth import issue_token, require_key, require_token
from src.guardrails import MessageIn

from src.logger import get_logger 

logger = get_logger(__name__)
load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET", "fallback_secret_key_change_in_prod_32bytes")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "user_login"
COLLECTION_NAME = "users"

AUDIT_FILE = os.getenv("AUDIT_FILE", "audit/predictions.jsonl")
PUBLIC_KEY_PATH = os.getenv("MODEL_PUBLIC_KEY_PATH", "artifacts/public.key")
PRIVATE_KEY_PATH = os.getenv("MODEL_PRIVATE_KEY_PATH", "artifacts/private.key")
AUTO_SIGN_MODEL = os.getenv("MODEL_AUTO_SIGN", "false").lower() == "true"

# --- Database Initialization ---
users_collection = None
if MONGO_URI:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    users_collection = db[COLLECTION_NAME]
    users_collection.create_index("email", unique=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

model = None

def audit(rec: dict) -> None:
    audit_dir = os.path.dirname(AUDIT_FILE)
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
    rec["ts"] = time.time()
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def sign_model_file(model_path: str, private_key_path: str = PRIVATE_KEY_PATH) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(private_key_path):
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

    with open(private_key_path, "rb") as f:
        priv_key = serialization.load_pem_private_key(f.read(), password=None)
    with open(model_path, "rb") as f:
        model_data = f.read()

    signature = priv_key.sign(model_data, ec.ECDSA(hashes.SHA256()))
    sig_path = f"{model_path}.sig"
    with open(sig_path, "wb") as f:
        f.write(signature)
    return sig_path

def load_verified_model(model_path: str):
    sig_path = f"{model_path}.sig"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}")
    if not os.path.exists(PUBLIC_KEY_PATH):
        raise FileNotFoundError(f"Public key missing: {PUBLIC_KEY_PATH}")

    if AUTO_SIGN_MODEL and os.path.exists(PRIVATE_KEY_PATH):
        sign_model_file(model_path, PRIVATE_KEY_PATH)

    if not os.path.exists(sig_path):
        raise FileNotFoundError(f"Signature file missing: {sig_path}")

    with open(PUBLIC_KEY_PATH, "rb") as f:
        pub_key = serialization.load_pem_public_key(f.read())
    with open(model_path, "rb") as f:
        model_data = f.read()
    with open(sig_path, "rb") as f:
        signature = f.read()

    try:
        pub_key.verify(signature, model_data, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise ValueError(
            "ECDSA signature mismatch: model was re-trained/modified or signed with a different key."
        ) from exc

    try:
        return pickle.loads(model_data)
    except Exception:
        return joblib.load(io.BytesIO(model_data))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        if os.path.exists(MODEL_OUTPUT_PATH):
            model = load_verified_model(MODEL_OUTPUT_PATH)
            print("Verified OK! Model signature matched and loaded safely.")
        else:
            model = None
            print(f"Warning: Model file not found at {MODEL_OUTPUT_PATH}")
    except Exception as e:
        model = None
        print(f"SECURITY ALERT: Model verification failed! Refusing to load model. Error: {e}")
    yield


app = FastAPI(title="SpamGuard MLSecOps API", version="1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., gt=0, lt=120)
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=50)
    age: Optional[int] = Field(None, gt=0, lt=120)
    password: Optional[str] = Field(None, min_length=6)


class Token(BaseModel):
    access_token: str
    token_type: str


# --- Helper Functions ---
def _digest_password(password: str) -> bytes:
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("utf-8")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_digest_password(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(_digest_password(plain_password), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def format_user(user_doc) -> dict:
    return {
        "id": str(user_doc["_id"]),
        "name": user_doc["name"],
        "age": user_doc["age"],
        "email": user_doc["email"],
    }


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database is not configured")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = users_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user


@app.get("/", response_class=HTMLResponse)
async def serve_auth_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="auth.html",
        context={"request": request},
    )


@app.get("/spam", response_class=HTMLResponse)
async def serve_spam_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="spam.html",
        context={"request": request},
    )


# ==========================================
# 2. USER CRUD OPERATIONS (Create, Read, Update, Delete)
# ==========================================
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserRegister):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database is not configured")

    if users_collection.find_one({"email": user_in.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    doc = {
        "name": user_in.name,
        "age": user_in.age,
        "email": user_in.email,
        "password": hash_password(user_in.password),
    }
    inserted = users_collection.insert_one(doc)
    doc["_id"] = inserted.inserted_id
    return format_user(doc)


@app.post("/login", response_model=Token)
def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database is not configured")

    user = users_collection.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me")
def read_user_me(current_user: dict = Depends(get_current_user)):
    return format_user(current_user)


@app.put("/users/me")
def update_user_me(
    user_update: UserUpdate, current_user: dict = Depends(get_current_user)
):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database is not configured")

    update_fields = {}
    if user_update.name is not None:
        update_fields["name"] = user_update.name
    if user_update.age is not None:
        update_fields["age"] = user_update.age
    if user_update.password is not None:
        update_fields["password"] = hash_password(user_update.password)

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    users_collection.update_one(
        {"_id": current_user["_id"]}, {"$set": update_fields}
    )
    updated_user = users_collection.find_one({"_id": current_user["_id"]})
    return format_user(updated_user)


@app.delete("/users/me")
def delete_user_me(current_user: dict = Depends(get_current_user)):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database is not configured")

    result = users_collection.delete_one({"_id": current_user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"Account ({current_user['email']}) deleted successfully."}



@app.get("/token")
async def get_token(_=Depends(require_key)):
    return {"access_token": issue_token("spamgard_client"), "token_type": "bearer"}


@app.post("/predict")
@limiter.limit("20/minute")
async def predict_service(request: Request, body: MessageIn, claims=Depends(require_token)):
    return _run_secure_inference(request, body.message, claims.get("sub", "service_client"))


import re


def _is_message_drifted(message: str) -> bool:
    """Single-message drift check for unseen links, gibberish, or unusual length."""
    text = message.strip()
    if not text:
        return False

    words = re.findall(r"[A-Za-z0-9_.-]+", text)
    has_url_or_handle = bool(
        re.search(r"(https?://|www\.|bit\.ly/|tinyurl\.com/|t\.me/|@\w+)", text.lower())
    )
    has_long_gibberish_token = any(
        len(w) >= 18 and len(set(w.lower())) <= 6 for w in words
    )
    is_unusually_long = len(text) > 220

    return bool(has_url_or_handle or has_long_gibberish_token or is_unusually_long)


@app.post("/predict/web")
@limiter.limit("20/minute")
async def predict_web(
    request: Request,
    body: MessageIn,
    current_user: dict = Depends(get_current_user),
):
    return _run_secure_inference(request, body.message, current_user["email"])


def _run_secure_inference(request: Request, message: str, subject: str) -> dict:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model load nahi hua hai (or signature verification failed).",
        )

    in_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
    drift_detected = _is_message_drifted(message)

    if drift_detected:
        rec = {
            "user": subject,
            "ip": request.client.host if request.client else "unknown",
            "in_hash": in_hash,
            "prediction": "DRIFT",
            "drift_detected": True,
        }
        audit(rec)
        if "logger" in globals() and logger is not None:
            logger.warning(f"Drift detected | user={subject} | hash={in_hash}")

        return {
            "drift_detected": True,
            "prediction": "DRIFT",
            "is_spam": False,
            "audit_hash": in_hash,
        }

    pred = int(model.predict([message])[0])
    label = "Spam" if pred == 1 else "Ham"

    rec = {
        "user": subject,
        "ip": request.client.host if request.client else "unknown",
        "in_hash": in_hash,
        "prediction": label,
        "drift_detected": False,
    }
    audit(rec)
    if "logger" in globals() and logger is not None:
        logger.info(f"Prediction={label} | user={subject} | hash={in_hash}")

    return {
        "drift_detected": False,
        "prediction": label,
        "is_spam": bool(pred),
        "audit_hash": in_hash,
    }




if __name__ == "__main__":
    import uvicorn

    uvicorn.run("application:app", host="127.0.0.1", port=8000, reload=True)