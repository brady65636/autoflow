from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class CurrentUser:
    user_id: str
    role: ActorRole


class ActorRole(StrEnum):
    CUSTOMER = "CUSTOMER"
    CUSTOMER_AGENT = "CUSTOMER_AGENT"
    SERVICE_ADVISOR = "SERVICE_ADVISOR"
    SYSTEM = "SYSTEM"


JWT_SECRET = os.getenv("AUTOFLOW_JWT_SECRET", "autoflow-local-dev-secret-change-me")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"pbkdf2_sha256$120000${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_text, digest_text = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), _unb64(salt_text), int(rounds)
        )
        return hmac.compare_digest(digest, _unb64(digest_text))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, role: ActorRole, expires_in: int = 3600) -> str:
    header = _encode_json({"alg": "HS256", "typ": "JWT"})
    payload = _encode_json(
        {"sub": subject, "role": role.value, "exp": int(time.time()) + expires_in}
    )
    signature = _sign(f"{header}.{payload}")
    return f"{header}.{payload}.{signature}"


def decode_access_token(token: str) -> dict[str, object]:
    try:
        header, payload, signature = token.split(".")
        if not hmac.compare_digest(signature, _sign(f"{header}.{payload}")):
            raise ValueError("bad signature")
        data = json.loads(_unb64(payload))
        if int(data["exp"]) <= int(time.time()):
            raise ValueError("expired token")
        ActorRole(str(data["role"]))
        return data
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token"
        ) from error


def require_roles(*allowed_roles: ActorRole):
    allowed = set(allowed_roles)

    def dependency(authorization: str = Header(...)) -> ActorRole:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="bearer token required")
        data = decode_access_token(token)
        role = ActorRole(str(data["role"]))
        if role not in allowed:
            raise HTTPException(status_code=403, detail="actor is not allowed")
        return CurrentUser(user_id=str(data["sub"]), role=role)

    return dependency


def _sign(value: str) -> str:
    return _b64(hmac.new(JWT_SECRET.encode(), value.encode(), hashlib.sha256).digest())


def _encode_json(value: dict[str, object]) -> str:
    return _b64(json.dumps(value, separators=(",", ":")).encode())


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
