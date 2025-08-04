"""Authentication and authorization utilities."""

import os
import time
import hmac
import hashlib
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..exceptions import AuthenticationError, AuthorizationError
from ..monitoring.logging import get_structured_logger

logger = get_structured_logger(__name__)

# Security configuration
MASTER_API_KEY = os.getenv("VID_BENCH_MASTER_API_KEY")
JWT_SECRET = os.getenv("VID_BENCH_JWT_SECRET", secrets.token_urlsafe(32))
API_KEY_LENGTH = 32


class APIKey:
    """API key management."""
    
    def __init__(self, key: str, name: str, permissions: List[str], expires_at: Optional[datetime] = None):
        self.key = key
        self.name = name
        self.permissions = permissions
        self.expires_at = expires_at
        self.created_at = datetime.utcnow()
        self.last_used = None
        self.usage_count = 0
        
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
        
    def has_permission(self, permission: str) -> bool:
        """Check if API key has specific permission."""
        return "admin" in self.permissions or permission in self.permissions
        
    def use(self):
        """Mark API key as used."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "name": self.name,
            "permissions": self.permissions,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count
        }


class APIKeyManager:
    """Manages API keys and authentication."""
    
    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        self._load_master_key()
        
    def _load_master_key(self):
        """Load master API key from environment."""
        if MASTER_API_KEY:
            master_key = APIKey(
                key=MASTER_API_KEY,
                name="master",
                permissions=["admin"],
                expires_at=None
            )
            self._keys[MASTER_API_KEY] = master_key
            logger.info("Master API key loaded")
        else:
            logger.warning("No master API key configured")
            
    def generate_key(
        self, 
        name: str, 
        permissions: List[str], 
        expires_in_days: Optional[int] = None
    ) -> APIKey:
        """Generate a new API key.
        
        Args:
            name: Human-readable name for the key
            permissions: List of permissions for the key
            expires_in_days: Number of days until expiration (None for no expiration)
            
        Returns:
            APIKey instance
        """
        key = secrets.token_urlsafe(API_KEY_LENGTH)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
        api_key = APIKey(
            key=key,
            name=name,
            permissions=permissions,
            expires_at=expires_at
        )
        
        self._keys[key] = api_key
        
        logger.info(
            "Generated new API key",
            key_name=name,
            permissions=permissions,
            expires_at=expires_at.isoformat() if expires_at else None
        )
        
        return api_key
        
    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key.
        
        Args:
            key: API key to validate
            
        Returns:
            APIKey instance if valid, None otherwise
        """
        if not key or key not in self._keys:
            return None
            
        api_key = self._keys[key]
        
        if api_key.is_expired():
            logger.warning(f"Expired API key used: {api_key.name}")
            return None
            
        api_key.use()
        return api_key
        
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key.
        
        Args:
            key: API key to revoke
            
        Returns:
            True if key was revoked, False if not found
        """
        if key in self._keys:
            api_key = self._keys[key]
            del self._keys[key]
            logger.info(f"Revoked API key: {api_key.name}")
            return True
        return False
        
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key values).
        
        Returns:
            List of API key information
        """
        return [
            {
                **api_key.to_dict(),
                "key_preview": f"{api_key.key[:8]}..."
            }
            for api_key in self._keys.values()
        ]


# Global API key manager
api_key_manager = APIKeyManager()

# FastAPI security scheme
security = HTTPBearer(auto_error=False)


class APIKeyAuth:
    """FastAPI dependency for API key authentication."""
    
    def __init__(self, required_permission: Optional[str] = None):
        self.required_permission = required_permission
        
    def __call__(self, credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> APIKey:
        """Authenticate API key and check permissions."""
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        api_key = api_key_manager.validate_key(credentials.credentials)
        if not api_key:
            logger.warning("Invalid API key attempted", key_preview=credentials.credentials[:8])
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        # Check permissions
        if self.required_permission and not api_key.has_permission(self.required_permission):
            logger.warning(
                "Insufficient permissions",
                key_name=api_key.name,
                required_permission=self.required_permission,
                user_permissions=api_key.permissions
            )
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{self.required_permission}' required"
            )
            
        logger.debug(
            "API key authenticated",
            key_name=api_key.name,
            permissions=api_key.permissions
        )
        
        return api_key


# Common authentication dependencies
def get_current_user(api_key: APIKey = Depends(APIKeyAuth())) -> APIKey:
    """Get current authenticated user (API key)."""
    return api_key


def require_admin(api_key: APIKey = Depends(APIKeyAuth("admin"))) -> APIKey:
    """Require admin permissions."""
    return api_key


def require_benchmark_access(api_key: APIKey = Depends(APIKeyAuth("benchmark"))) -> APIKey:
    """Require benchmark access permissions."""
    return api_key


class SimpleTokenAuth:
    """Simple token-based authentication for development."""
    
    def __init__(self):
        self.valid_tokens = set()
        
        # Add development token if configured
        dev_token = os.getenv("VID_BENCH_DEV_TOKEN")
        if dev_token:
            self.valid_tokens.add(dev_token)
            logger.info("Development token configured")
            
    def generate_token(self) -> str:
        """Generate a simple token."""
        token = secrets.token_urlsafe(16)
        self.valid_tokens.add(token)
        return token
        
    def validate_token(self, token: str) -> bool:
        """Validate a token."""
        return token in self.valid_tokens
        
    def revoke_token(self, token: str):
        """Revoke a token."""
        self.valid_tokens.discard(token)


# Global token auth for development
simple_auth = SimpleTokenAuth()


def create_api_key_hash(key: str, salt: str) -> str:
    """Create a secure hash of an API key.
    
    Args:
        key: API key to hash
        salt: Salt for hashing
        
    Returns:
        Hexadecimal hash string
    """
    return hmac.new(
        salt.encode('utf-8'),
        key.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def verify_api_key_hash(key: str, salt: str, expected_hash: str) -> bool:
    """Verify an API key against its hash.
    
    Args:
        key: API key to verify
        salt: Salt used for hashing
        expected_hash: Expected hash value
        
    Returns:
        True if key matches hash
    """
    actual_hash = create_api_key_hash(key, salt)
    return hmac.compare_digest(actual_hash, expected_hash)


def create_request_signature(
    method: str, 
    url: str, 
    body: str, 
    timestamp: str, 
    secret: str
) -> str:
    """Create request signature for webhook verification.
    
    Args:
        method: HTTP method
        url: Request URL
        body: Request body
        timestamp: Request timestamp
        secret: Signing secret
        
    Returns:
        Request signature
    """
    message = f"{method}|{url}|{body}|{timestamp}"
    return hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def verify_request_signature(
    method: str,
    url: str,
    body: str,
    timestamp: str,
    signature: str,
    secret: str,
    max_age_seconds: int = 300
) -> bool:
    """Verify request signature.
    
    Args:
        method: HTTP method
        url: Request URL
        body: Request body
        timestamp: Request timestamp
        signature: Provided signature
        secret: Signing secret
        max_age_seconds: Maximum age of request in seconds
        
    Returns:
        True if signature is valid
    """
    # Check timestamp freshness
    try:
        request_time = datetime.fromisoformat(timestamp)
        age = (datetime.utcnow() - request_time).total_seconds()
        if age > max_age_seconds:
            return False
    except ValueError:
        return False
        
    # Verify signature
    expected_signature = create_request_signature(method, url, body, timestamp, secret)
    return hmac.compare_digest(signature, expected_signature)