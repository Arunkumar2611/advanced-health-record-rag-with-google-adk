from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash

from health_rag.config import JWT_ISSUER, JWT_SECRET
from health_rag.persistence.database import Database

password_hash = PasswordHash.recommended()
oauth_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return password_hash.verify(password, hashed)


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": user_id, "iss": JWT_ISSUER, "iat": now, "exp": now + timedelta(hours=8), "jti": str(uuid4())}, JWT_SECRET, algorithm="HS256")


def current_user(token: str = Depends(oauth_scheme)) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], issuer=JWT_ISSUER)
        user = Database().user_by_id(payload["sub"])
    except (jwt.InvalidTokenError, KeyError):
        user = None
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token", headers={"WWW-Authenticate": "Bearer"})
    return user
