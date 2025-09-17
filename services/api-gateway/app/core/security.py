"""
Security utilities for API Gateway
Authentication, authorization, and security dependencies
"""

from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/token",
    auto_error=False  # Don't automatically raise 401, let us handle it
)


async def authenticate_request(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Dependency to authenticate requests
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        User information dictionary
        
    Raises:
        HTTPException: If authentication fails
    """
    if not settings.ENABLE_AUTH:
        # If auth is disabled, return a default user
        return {
            "user_id": "anonymous",
            "username": "anonymous",
            "roles": ["user"],
            "is_authenticated": False
        }
    
    if not token:
        logger.warning("Missing authentication token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        # TODO: Implement actual JWT token validation
        # For now, we'll use a simple placeholder
        user_info = await validate_jwt_token(token)
        
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        logger.debug(
            "Request authenticated",
            user_id=user_info.get("user_id"),
            username=user_info.get("username")
        )
        
        return user_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def validate_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validate JWT token and return user information
    
    Args:
        token: JWT token string
        
    Returns:
        User information dictionary or None if invalid
    """
    try:
        # TODO: Implement actual JWT validation using python-jose
        # For MVP, we'll use a simple hardcoded validation
        
        if token == "fake-jwt-token-for-development":
            return {
                "user_id": "user-123",
                "username": "admin",
                "email": "admin@example.com",
                "roles": ["admin", "user"],
                "is_authenticated": True,
                "expires_at": datetime.utcnow().timestamp() + 3600  # 1 hour from now
            }
        
        # Invalid token
        return None
        
    except Exception as e:
        logger.error("Token validation error", error=str(e))
        return None


async def require_roles(required_roles: list[str]):
    """
    Dependency factory for role-based access control
    
    Args:
        required_roles: List of required roles
        
    Returns:
        Dependency function that checks user roles
    """
    async def check_roles(user_info: Dict[str, Any] = Depends(authenticate_request)):
        user_roles = user_info.get("roles", [])
        
        if not any(role in user_roles for role in required_roles):
            logger.warning(
                "Insufficient permissions",
                user_id=user_info.get("user_id"),
                required_roles=required_roles,
                user_roles=user_roles
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return user_info
    
    return check_roles


# Convenience dependencies for common role checks
require_admin = require_roles(["admin"])
require_user = require_roles(["user", "admin"])


async def get_current_user(user_info: Dict[str, Any] = Depends(authenticate_request)) -> Dict[str, Any]:
    """
    Get current authenticated user information
    
    Args:
        user_info: User info from authentication
        
    Returns:
        User information dictionary
    """
    return user_info


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[Dict[str, Any]]:
    """
    Get current user information if authenticated, None otherwise
    Useful for endpoints that work differently for authenticated vs anonymous users
    
    Args:
        token: Optional JWT token
        
    Returns:
        User information dictionary or None
    """
    if not token:
        return None
    
    try:
        return await validate_jwt_token(token)
    except Exception:
        return None


def generate_api_key() -> str:
    """
    Generate a new API key for service-to-service authentication
    
    Returns:
        API key string
    """
    import secrets
    import base64
    
    # Generate 32 bytes of random data
    key_bytes = secrets.token_bytes(32)
    
    # Encode as base64 and add prefix
    api_key = base64.b64encode(key_bytes).decode('utf-8')
    return f"aop_{api_key}"


async def validate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Validate API key for service-to-service authentication
    
    Args:
        api_key: API key string
        
    Returns:
        Service information dictionary or None if invalid
    """
    # TODO: Implement API key validation against database
    # For now, return None (invalid)
    return None