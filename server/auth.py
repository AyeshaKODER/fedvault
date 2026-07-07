"""JWT authentication and role-based access control for FedVault."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    CLIENT_CREDENTIALS,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    JWT_TOKEN_EXPIRE_MINUTES,
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

RoleType = Literal["admin", "client"]


class Token(BaseModel):
    access_token: str
    token_type: str
    role: RoleType
    node_id: str | None = None


class TokenData(BaseModel):
    username: str
    role: RoleType
    node_id: str | None = None


class User(BaseModel):
    username: str
    role: RoleType
    node_id: str | None = None


def _build_user_registry() -> dict[str, dict[str, str]]:
    registry: dict[str, dict[str, str]] = {
        ADMIN_USERNAME: {
            "username": ADMIN_USERNAME,
            "password": ADMIN_PASSWORD,
            "role": "admin",
            "node_id": "",
        }
    }
    for node_id, creds in CLIENT_CREDENTIALS.items():
        registry[creds["username"]] = {
            "username": creds["username"],
            "password": creds["password"],
            "role": "client",
            "node_id": node_id,
        }
    return registry


USER_REGISTRY = _build_user_registry()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password verification failed: {exc}",
        ) from exc


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> User | None:
    """Authenticate a user against the in-memory registry."""
    record = USER_REGISTRY.get(username)
    if record is None:
        return None

    stored_password = record["password"]
    if stored_password.startswith("$2b$") or stored_password.startswith("$2a$"):
        password_valid = verify_password(password, stored_password)
    else:
        password_valid = password == stored_password

    if not password_valid:
        return None

    return User(
        username=record["username"],
        role=record["role"],  # type: ignore[arg-type]
        node_id=record["node_id"] or None,
    )


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta is not None else timedelta(minutes=JWT_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    try:
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token creation failed: {exc}",
        ) from exc


def decode_access_token(token: str) -> TokenData:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        node_id = payload.get("node_id")
        if username is None or role is None:
            raise JWTError("Missing token claims")
        return TokenData(username=username, role=role, node_id=node_id)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    """FastAPI dependency that resolves the authenticated user."""
    token_data = decode_access_token(token)
    return User(username=token_data.username, role=token_data.role, node_id=token_data.node_id)


async def require_admin(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Require an authenticated admin user."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


async def require_client(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Require an authenticated client (hospital node) user."""
    if current_user.role != "client":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client node privileges required",
        )
    if not current_user.node_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client token missing node identifier",
        )
    return current_user


def login_for_access_token(form_data: OAuth2PasswordRequestForm) -> Token:
    """Authenticate credentials and return a JWT token response."""
    user = authenticate_user(form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
            "node_id": user.node_id or "",
        }
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user.role,
        node_id=user.node_id,
    )
