from fastapi_azure_auth import SingleTenantAzureAuthorizationCodeBearer
from app_config import CLIENT_ID, TENANT_ID, APP_SCOPE

import jwt
import httpx
import logging
from typing import Optional

# Create a logger for this module
logger = logging.getLogger(__name__)

# Azure AD public keys endpoint
AZURE_JWKS_URI = "https://login.microsoftonline.com/" + TENANT_ID + "/discovery/v2.0/keys"
EXPECTED_AUDIENCE = CLIENT_ID  # From Azure portal

# https://intility.github.io/fastapi-azure-auth/single-tenant/fastapi_configuration/
azure_scheme = SingleTenantAzureAuthorizationCodeBearer(
    app_client_id=CLIENT_ID,
    tenant_id=TENANT_ID,
    scopes={'api://' + APP_SCOPE + '/access_as_user':'access_as_user'},
    allow_guest_users=True
)


async def get_azure_public_keys():
    async with httpx.AsyncClient() as client:
        response = await client.get(AZURE_JWKS_URI)
        return response.json()["keys"]

async def get_kid(token):
    unverified_header = jwt.get_unverified_header(token)
    return unverified_header["kid"]

async def get_key(kid, keys):
    for key in keys:
        if key["kid"] == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(key)
    return None

async def validate_token(token: str) -> Optional[dict]:
    try:
        keys = await get_azure_public_keys()
        kid = await get_kid(token)
        key = await get_key(kid, keys)
        logger.info(f"Validating token with kid: {kid}")
        if not key:
            return None
        decoded = jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            audience=EXPECTED_AUDIENCE,
            options={"verify_exp": True}
        )
        return decoded
    except Exception as e:
        logger.error("JWT validation error:" + str(e), exc_info=True)
        return None