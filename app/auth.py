"""teamusers 鉴权：校验访问者的 EdDSA 访问令牌，并按权限点放行。

- 验签走 teamusers 的 JWKS（EdDSA），本服务不自己签发/校验令牌；
- 权限判定：face:check:any 放行识别与查询，face:modify:any 放行注册与删除；
- 未认证 → 401，已认证但无权限 → 403。
"""
from fastapi import HTTPException, Request
from teamusers_sdk import (
    Authenticate,
    Client,
    Claims,
    PermissionsClient,
    UnauthorizedError,
    Verifier,
)

from . import config

verifier = Verifier(config.TEAMUSERS_URL, audience=config.TEAMUSERS_AUDIENCE)
permissions = PermissionsClient(config.TEAMUSERS_URL, service_token=config.TEAMUSERS_SERVICE_TOKEN)
_client = Client(verifier=verifier, permissions=permissions)


def _authenticate(request: Request) -> Claims:
    try:
        return Authenticate({"headers": dict(request.headers)}, verifier)
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=401,
            detail=f"认证失败：{e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def _require(permission: str):
    def dependency(request: Request) -> Claims:
        claims = _authenticate(request)
        allowed, reason = _client.allow(claims, permission, None)
        if not allowed:
            raise HTTPException(status_code=403, detail=f"无权限 {permission}：{reason}")
        return claims

    return dependency


# FastAPI 依赖：检查权限 / 修改权限
require_check = _require(config.PERM_CHECK)
require_modify = _require(config.PERM_MODIFY)
