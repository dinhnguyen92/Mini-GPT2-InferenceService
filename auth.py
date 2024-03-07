from dotenv import dotenv_values
from fastapi.security.api_key import APIKeyHeader
from fastapi import Request, Security, HTTPException
from starlette.status import HTTP_403_FORBIDDEN


# Load environment variables
env_vars = dotenv_values('.env')

AUTHORIZATION_HEADER_NAME = "Authorization"
AUTHORIZATION_TOKEN_PREFIX = "Bearer "

public_endpoints = ['/health-check']

api_key_header = APIKeyHeader(name=AUTHORIZATION_HEADER_NAME, auto_error=False)

async def validate_api_key(request: Request, api_key: str = Security(api_key_header)):
    if api_key != f"{AUTHORIZATION_TOKEN_PREFIX}{env_vars['API_KEY']}" and request.url.path not in public_endpoints:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail='Unauthenticated')