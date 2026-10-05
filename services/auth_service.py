import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator


DATABASE_PATH = Path(
    os.getenv(
        "RAG_AUTH_DB_PATH",
        Path(__file__).resolve().parents[1] / "storage" / "auth.sqlite3"
    )
)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
PASSWORD_HASH_ROUNDS = 600_000
ACCESS_TOKEN_TTL_SECONDS = 12 * 60 * 60
bearer_scheme = HTTPBearer(auto_error=False)
auth_router = APIRouter(prefix="/auth", tags=["authentication"])


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value):
        normalized = value.strip().lower()
        if len(normalized) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized):
            raise ValueError("Enter a valid email address.")
        return normalized


class LoginRequest(RegisterRequest):
    password: str = Field(min_length=1, max_length=128)


class DuplicateEmailError(Exception):
    pass


@contextmanager
def _connection():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize_database():
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS access_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS access_tokens_expiry_idx
                ON access_tokens(expires_at);

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS documents_user_id_idx
                ON documents(user_id);
            """
        )


def _hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ROUNDS
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256${PASSWORD_HASH_ROUNDS}${salt_text}${digest_text}"


def _verify_password(password, encoded_hash):
    try:
        algorithm, rounds_text, salt_text, expected_text = encoded_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        rounds = int(rounds_text)
        if rounds < 1 or rounds > PASSWORD_HASH_ROUNDS:
            return False
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(expected_text + "=" * (-len(expected_text) % 4))
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(actual, expected)


def register_user(email, password):
    user_id = str(uuid.uuid4())
    normalized_email = email.strip().lower()
    try:
        with _connection() as connection:
            connection.execute(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, normalized_email, _hash_password(password), int(time.time()))
            )
    except sqlite3.IntegrityError as error:
        raise DuplicateEmailError from error
    return {"id": user_id, "email": normalized_email}


def authenticate_user(email, password):
    with _connection() as connection:
        user = connection.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),)
        ).fetchone()

    if user is None or not _verify_password(password, user["password_hash"]):
        return None
    return {"id": user["id"], "email": user["email"]}


def issue_access_token(user_id):
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    expires_at = int(time.time()) + ACCESS_TOKEN_TTL_SECONDS
    with _connection() as connection:
        connection.execute(
            "INSERT INTO access_tokens (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at)
        )
        connection.execute(
            "DELETE FROM access_tokens WHERE expires_at <= ?",
            (int(time.time()),)
        )
    return token


def resolve_access_token(token):
    try:
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    except UnicodeEncodeError:
        return None
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.email
            FROM access_tokens
            JOIN users ON users.id = access_tokens.user_id
            WHERE access_tokens.token_hash = ? AND access_tokens.expires_at > ?
            """,
            (token_hash, int(time.time()))
        ).fetchone()
    return dict(row) if row else None


def revoke_access_token(token):
    try:
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    except UnicodeEncodeError:
        return
    with _connection() as connection:
        connection.execute("DELETE FROM access_tokens WHERE token_hash = ?", (token_hash,))


def record_document_owner(document_id, user_id):
    with _connection() as connection:
        connection.execute(
            "INSERT INTO documents (id, user_id, created_at) VALUES (?, ?, ?)",
            (document_id, user_id, int(time.time()))
        )


def user_owns_document(document_id, user_id):
    with _connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user_id)
        ).fetchone()
    return row is not None


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Sign in to continue.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = resolve_access_token(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Your session is invalid or expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


def _token_response(user):
    return {
        "access_token": issue_access_token(user["id"]),
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_TTL_SECONDS,
        "user": user,
    }


@auth_router.post("/register", status_code=201)
def register(request: RegisterRequest):
    try:
        user = register_user(request.email, request.password)
    except DuplicateEmailError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    return _token_response(user)


@auth_router.post("/login")
def login(request: LoginRequest):
    user = authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    return _token_response(user)


@auth_router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {"user": current_user}


@auth_router.post("/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    current_user: dict = Depends(get_current_user),
):
    if credentials is not None:
        revoke_access_token(credentials.credentials)
    return {"message": "You have been signed out."}


_initialize_database()