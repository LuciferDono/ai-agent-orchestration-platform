"""
Logging middleware for API Gateway
Structured request/response logging with correlation IDs
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging
    """
    
    def __init__(self, app, exclude_paths: list[str] = None):
        """
        Initialize logging middleware
        
        Args:
            app: FastAPI application
            exclude_paths: List of paths to exclude from logging
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and response with structured logging
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response from the application
        """
        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())
        
        # Add correlation ID to request state
        request.state.correlation_id = correlation_id
        
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        
        # Log request start
        start_time = time.time()
        
        # Extract client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        logger.info(
            "Request started",
            correlation_id=correlation_id,
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
            headers=self._sanitize_headers(dict(request.headers))
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log successful response
            logger.info(
                "Request completed",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time=process_time,
                response_size=response.headers.get("content-length", "unknown")
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Calculate processing time for error case
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                process_time=process_time
            )
            
            # Re-raise the exception
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers (when behind proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _sanitize_headers(self, headers: dict) -> dict:
        """
        Sanitize headers for logging (remove sensitive information)
        
        Args:
            headers: Request headers dictionary
            
        Returns:
            Sanitized headers dictionary
        """
        sensitive_headers = {
            "authorization",
            "cookie",
            "x-api-key",
            "x-auth-token",
            "x-access-token"
        }
        
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Simplified request logging middleware for high-traffic scenarios
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log basic request information
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response from the application
        """
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log request summary
        logger.info(
            "Request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration=f"{process_time:.3f}s",
            client=request.client.host if request.client else "unknown"
        )
        
        return response