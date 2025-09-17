"""
AI Agent Orchestration Platform - API Gateway
Main FastAPI application with authentication, rate limiting, and request routing
"""

from contextlib import asynccontextmanager
from typing import List

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from app.core.config import settings
from app.core.security import authenticate_request
from app.core.rate_limiter import RateLimiter
from app.middleware.logging import LoggingMiddleware
from app.middleware.tracing import TracingMiddleware
from app.routers import auth, agents, workflows, health
from shared.models.exceptions import APIException

# Configure structured logging
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_gateway_requests_total',
    'Total API gateway requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'api_gateway_request_duration_seconds',
    'API gateway request duration',
    ['method', 'endpoint']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Starting API Gateway", version=settings.VERSION)
    
    # Initialize rate limiter
    app.state.rate_limiter = RateLimiter(
        redis_url=settings.REDIS_URL,
        default_limit=settings.RATE_LIMIT_PER_MINUTE
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway")
    if hasattr(app.state, 'rate_limiter'):
        await app.state.rate_limiter.close()


# Create FastAPI application
app = FastAPI(
    title="AI Agent Orchestration Platform - API Gateway",
    description="Enterprise-grade API Gateway for AI Agent Orchestration Platform",
    version=settings.VERSION,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(TracingMiddleware)
app.add_middleware(LoggingMiddleware)


# Middleware for metrics and rate limiting
@app.middleware("http")
async def metrics_and_rate_limit_middleware(request: Request, call_next):
    """Middleware to handle metrics collection and rate limiting"""
    
    # Extract client identifier for rate limiting
    client_id = request.client.host if request.client else "unknown"
    
    # Check rate limits
    rate_limiter = app.state.rate_limiter
    if not await rate_limiter.is_allowed(client_id):
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code="429"
        ).inc()
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Record request start time
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code)
    ).inc()
    
    return response


# Exception handlers
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions"""
    logger.error(
        "API Exception",
        error=exc.detail,
        status_code=exc.status_code,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "code": exc.error_code,
            "timestamp": exc.timestamp.isoformat()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(
        "HTTP Exception",
        error=exc.detail,
        status_code=exc.status_code,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "code": "HTTP_ERROR",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Include routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["authentication"]
)

app.include_router(
    agents.router,
    prefix=f"{settings.API_V1_PREFIX}/agents",
    tags=["agents"],
    dependencies=[Depends(authenticate_request)]
)

app.include_router(
    workflows.router,
    prefix=f"{settings.API_V1_PREFIX}/workflows",
    tags=["workflows"],
    dependencies=[Depends(authenticate_request)]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic platform information"""
    return {
        "name": "AI Agent Orchestration Platform",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "running",
        "docs": f"{settings.API_BASE_URL}/docs" if settings.ENVIRONMENT == "development" else None
    }


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


# Service proxy endpoints
@app.api_route(
    "/api/v1/registry/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_agent_registry(request: Request):
    """Proxy requests to Agent Registry service"""
    return await proxy_request(
        request,
        target_url=settings.AGENT_REGISTRY_URL,
        service_name="agent-registry"
    )


@app.api_route(
    "/api/v1/orchestration/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_orchestration_engine(request: Request):
    """Proxy requests to Orchestration Engine service"""
    return await proxy_request(
        request,
        target_url=settings.ORCHESTRATION_ENGINE_URL,
        service_name="orchestration-engine"
    )


@app.api_route(
    "/api/v1/memory/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_memory_management(request: Request):
    """Proxy requests to Memory Management service"""
    return await proxy_request(
        request,
        target_url=settings.MEMORY_MANAGEMENT_URL,
        service_name="memory-management"
    )


@app.api_route(
    "/api/v1/hitl/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_hitl_service(request: Request):
    """Proxy requests to HITL service"""
    return await proxy_request(
        request,
        target_url=settings.HITL_SERVICE_URL,
        service_name="hitl-service"
    )


@app.api_route(
    "/api/v1/observability/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_observability(request: Request):
    """Proxy requests to Observability service"""
    return await proxy_request(
        request,
        target_url=settings.OBSERVABILITY_URL,
        service_name="observability"
    )


async def proxy_request(request: Request, target_url: str, service_name: str):
    """Generic request proxy function"""
    import httpx
    from datetime import datetime
    import time
    
    path = request.path_params.get("path", "")
    target_full_url = f"{target_url.rstrip('/')}/{path.lstrip('/')}"
    
    # Add query parameters
    if request.url.query:
        target_full_url += f"?{request.url.query}"
    
    logger.info(
        "Proxying request",
        service=service_name,
        method=request.method,
        target_url=target_full_url
    )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Forward headers (excluding hop-by-hop headers)
            headers = {
                key: value for key, value in request.headers.items()
                if key.lower() not in [
                    'host', 'content-length', 'connection',
                    'upgrade', 'proxy-connection', 'te'
                ]
            }
            
            # Get request body if present
            body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
            
            # Make proxied request
            response = await client.request(
                method=request.method,
                url=target_full_url,
                headers=headers,
                content=body
            )
            
            # Forward response headers (excluding hop-by-hop headers)
            response_headers = {
                key: value for key, value in response.headers.items()
                if key.lower() not in [
                    'content-length', 'connection', 'upgrade',
                    'proxy-connection', 'te', 'transfer-encoding'
                ]
            }
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers
            )
            
    except httpx.TimeoutException:
        logger.error("Request timeout", service=service_name, target_url=target_full_url)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Service {service_name} timeout"
        )
    except httpx.RequestError as e:
        logger.error("Request error", service=service_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Service {service_name} unavailable"
        )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=1 if settings.API_RELOAD else settings.API_WORKERS,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": settings.LOG_LEVEL,
                "handlers": ["default"],
            },
        }
    )