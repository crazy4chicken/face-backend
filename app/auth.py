"""JWT 鉴权：HS256 对称签名。

- 签发：POST /auth/token 校验账号密码后签发；
- 校验：除 /healthz 与文档页外，所有业务路由依赖 require_token。
密钥、账号、有效期均走环境变量（见 config.py），生产环境必须改默认值。
"""
import hmac
import time

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config

_bearer = HTTPBearer(auto_error=False, description="JWT Bearer Token")


def create_token(subject: str) -> str:
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + config.JWT_EXPIRE_MINUTES * 60,
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def verify_credentials(username: str, password: str) -> bool:
    """校验登录账号。hmac.compare_digest 防时序侧信道。"""
    return hmac.compare_digest(username, config.ADMIN_USERNAME) and hmac.compare_digest(
        password, config.ADMIN_PASSWORD
    )


def require_token(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    """FastAPI 依赖：校验 Bearer Token，合法返回 sub，否则 401。"""
    if creds is None:
        raise HTTPException(
            status_code=401,
            detail="缺少 Authorization Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(creds.credentials, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期", headers={"WWW-Authenticate": "Bearer"})
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token 无效", headers={"WWW-Authenticate": "Bearer"})
    return payload["sub"]
