"""Rate limiting implementation."""

import time
import asyncio
from typing import Dict, Tuple, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from ..exceptions import RateLimitError
from ..monitoring.logging import get_structured_logger

logger = get_structured_logger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    burst_requests: Optional[int] = None  # Burst allowance


class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second refill rate
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        now = time.time()
        
        # Refill tokens based on elapsed time
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
        
        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
        
    def tokens_available(self) -> float:
        """Get number of tokens currently available."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        return min(self.capacity, self.tokens + new_tokens)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self):
        self.windows: Dict[str, deque] = defaultdict(deque)
        
    def is_allowed(self, key: str, limit: RateLimit) -> Tuple[bool, float]:
        """Check if request is allowed.
        
        Args:
            key: Unique identifier for the client
            limit: Rate limit configuration
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window = self.windows[key]
        
        # Remove old entries outside the window
        cutoff = now - limit.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()
            
        # Check if limit is exceeded
        if len(window) >= limit.requests:
            # Calculate retry after time
            oldest_request = window[0] if window else now
            retry_after = limit.window_seconds - (now - oldest_request)
            return False, max(0, retry_after)
            
        # Allow request and record it
        window.append(now)
        return True, 0.0
        
    def cleanup_expired(self, max_age_seconds: int = 3600):
        """Clean up expired windows."""
        now = time.time()
        expired_keys = []
        
        for key, window in self.windows.items():
            if not window or (now - window[-1]) > max_age_seconds:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.windows[key]


class RateLimiter:
    """Main rate limiter class."""
    
    def __init__(self):
        self.sliding_window = SlidingWindowRateLimiter()
        self.token_buckets: Dict[str, TokenBucket] = {}
        
        # Default rate limits
        self.limits = {
            "default": RateLimit(requests=100, window_seconds=3600),  # 100/hour
            "benchmark": RateLimit(requests=10, window_seconds=3600),  # 10/hour
            "metrics": RateLimit(requests=1000, window_seconds=3600),  # 1000/hour
            "admin": RateLimit(requests=1000, window_seconds=60),  # 1000/minute
        }
        
    def check_rate_limit(
        self, 
        identifier: str, 
        limit_type: str = "default"
    ) -> Tuple[bool, float]:
        """Check if request should be rate limited.
        
        Args:
            identifier: Unique identifier (IP, API key, user ID)
            limit_type: Type of limit to apply
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        if limit_type not in self.limits:
            limit_type = "default"
            
        limit = self.limits[limit_type]
        key = f"{identifier}:{limit_type}"
        
        return self.sliding_window.is_allowed(key, limit)
        
    def check_burst_limit(
        self, 
        identifier: str, 
        capacity: int = 10, 
        refill_rate: float = 1.0
    ) -> bool:
        """Check burst rate limit using token bucket.
        
        Args:
            identifier: Unique identifier
            capacity: Token bucket capacity
            refill_rate: Tokens per second refill rate
            
        Returns:
            True if request is allowed
        """
        if identifier not in self.token_buckets:
            self.token_buckets[identifier] = TokenBucket(capacity, refill_rate)
            
        return self.token_buckets[identifier].consume()
        
    def get_limit_info(self, identifier: str, limit_type: str = "default") -> Dict[str, int]:
        """Get rate limit information for identifier.
        
        Args:
            identifier: Unique identifier
            limit_type: Type of limit
            
        Returns:
            Dictionary with limit information
        """
        if limit_type not in self.limits:
            limit_type = "default"
            
        limit = self.limits[limit_type]
        key = f"{identifier}:{limit_type}"
        
        window = self.sliding_window.windows[key]
        now = time.time()
        cutoff = now - limit.window_seconds
        
        # Count recent requests
        recent_requests = sum(1 for req_time in window if req_time > cutoff)
        
        return {
            "limit": limit.requests,
            "remaining": max(0, limit.requests - recent_requests),
            "reset_time": int(now + limit.window_seconds),
            "window_seconds": limit.window_seconds
        }
        
    def cleanup(self):
        """Clean up expired data."""
        self.sliding_window.cleanup_expired()
        
        # Clean up old token buckets
        now = time.time()
        expired_buckets = []
        
        for identifier, bucket in self.token_buckets.items():
            if (now - bucket.last_refill) > 3600:  # 1 hour idle
                expired_buckets.append(identifier)
                
        for identifier in expired_buckets:
            del self.token_buckets[identifier]


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_identifier(request: Request) -> str:
    """Get unique identifier for client.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Unique client identifier
    """
    # Try to get API key from authorization header
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
        return f"api_key:{api_key[:8]}"  # Use first 8 chars for privacy
        
    # Fall back to IP address
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
        
    return f"ip:{client_ip}"


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware for FastAPI.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/handler
        
    Returns:
        Response or rate limit error
    """
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/health/readiness", "/health/liveness"]:
        return await call_next(request)
        
    client_id = get_client_identifier(request)
    
    # Determine limit type based on path
    limit_type = "default"
    if request.url.path.startswith("/api/v1/benchmarks"):
        limit_type = "benchmark"
    elif request.url.path.startswith("/api/v1/metrics"):
        limit_type = "metrics"
    elif "admin" in request.url.path:
        limit_type = "admin"
        
    # Check rate limit
    allowed, retry_after = rate_limiter.check_rate_limit(client_id, limit_type)
    
    if not allowed:
        logger.warning(
            "Rate limit exceeded",
            client_id=client_id,
            limit_type=limit_type,
            path=request.url.path,
            retry_after=retry_after
        )
        
        limit_info = rate_limiter.get_limit_info(client_id, limit_type)
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": int(retry_after),
                "limit": limit_info["limit"],
                "window_seconds": limit_info["window_seconds"]
            },
            headers={
                "Retry-After": str(int(retry_after)),
                "X-RateLimit-Limit": str(limit_info["limit"]),
                "X-RateLimit-Remaining": str(limit_info["remaining"]),
                "X-RateLimit-Reset": str(limit_info["reset_time"])
            }
        )
        
    # Add rate limit headers to response
    response = await call_next(request)
    
    limit_info = rate_limiter.get_limit_info(client_id, limit_type)
    response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(limit_info["reset_time"])
    
    return response


def rate_limit(limit_type: str = "default"):
    """Decorator for rate limiting specific endpoints.
    
    Args:
        limit_type: Type of rate limit to apply
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
                    
            if not request:
                # If no request found, proceed without rate limiting
                return await func(*args, **kwargs)
                
            client_id = get_client_identifier(request)
            allowed, retry_after = rate_limiter.check_rate_limit(client_id, limit_type)
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry after {retry_after:.0f} seconds.",
                    headers={"Retry-After": str(int(retry_after))}
                )
                
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load."""
    
    def __init__(self, base_limiter: RateLimiter):
        self.base_limiter = base_limiter
        self.load_factor = 1.0
        self.last_adjustment = time.time()
        
    def update_load_factor(self, cpu_percent: float, memory_percent: float):
        """Update load factor based on system metrics.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
        """
        now = time.time()
        
        # Only adjust every 60 seconds
        if now - self.last_adjustment < 60:
            return
            
        # Calculate load factor
        max_usage = max(cpu_percent, memory_percent)
        
        if max_usage > 90:
            self.load_factor = 0.1  # Very restrictive
        elif max_usage > 80:
            self.load_factor = 0.25  # Restrictive
        elif max_usage > 70:
            self.load_factor = 0.5  # Moderate
        elif max_usage > 60:
            self.load_factor = 0.75  # Slightly restrictive
        else:
            self.load_factor = 1.0  # Normal
            
        self.last_adjustment = now
        
        logger.info(
            "Adjusted rate limit load factor",
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            load_factor=self.load_factor
        )
        
    def check_rate_limit(self, identifier: str, limit_type: str = "default") -> Tuple[bool, float]:
        """Check rate limit with adaptive adjustment."""
        # Adjust the effective limit based on load
        original_limit = self.base_limiter.limits[limit_type]
        adjusted_requests = max(1, int(original_limit.requests * self.load_factor))
        
        # Temporarily modify the limit
        adjusted_limit = RateLimit(
            requests=adjusted_requests,
            window_seconds=original_limit.window_seconds
        )
        self.base_limiter.limits[f"{limit_type}_adaptive"] = adjusted_limit
        
        # Check with adjusted limit
        return self.base_limiter.check_rate_limit(identifier, f"{limit_type}_adaptive")


# Global adaptive rate limiter
adaptive_rate_limiter = AdaptiveRateLimiter(rate_limiter)