"""
Health check endpoints for API Gateway
"""

import asyncio
import time
from typing import Dict, Any

import redis.asyncio as redis
import sqlalchemy
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import structlog

from app.core.config import settings

logger = structlog.get_logger()
router = APIRouter()


class HealthStatus(BaseModel):
    """Health status model"""
    status: str
    timestamp: float
    version: str
    environment: str
    uptime: float
    checks: Dict[str, Any]


class ServiceHealth(BaseModel):
    """Individual service health model"""
    status: str
    response_time: float
    error: str = None


# Store startup time for uptime calculation
startup_time = time.time()


@router.get("/", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint
    Returns overall system health status
    """
    start_time = time.time()
    
    # Perform health checks
    checks = {}
    overall_status = "healthy"
    
    # Check Redis connection
    try:
        redis_start = time.time()
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await client.ping()
        await client.close()
        
        checks["redis"] = {
            "status": "healthy",
            "response_time": time.time() - redis_start,
            "url": settings.REDIS_URL
        }
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        checks["redis"] = {
            "status": "unhealthy",
            "response_time": time.time() - redis_start if 'redis_start' in locals() else 0,
            "error": str(e),
            "url": settings.REDIS_URL
        }
        overall_status = "degraded"
    
    # Check database connection (if configured)
    if settings.DATABASE_URL:
        try:
            db_start = time.time()
            engine = sqlalchemy.create_engine(settings.DATABASE_URL)
            
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            
            engine.dispose()
            
            checks["database"] = {
                "status": "healthy",
                "response_time": time.time() - db_start,
                "url": settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else settings.DATABASE_URL
            }
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            checks["database"] = {
                "status": "unhealthy",
                "response_time": time.time() - db_start if 'db_start' in locals() else 0,
                "error": str(e)
            }
            overall_status = "degraded"
    
    # Calculate uptime
    uptime = time.time() - startup_time
    
    return HealthStatus(
        status=overall_status,
        timestamp=time.time(),
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        uptime=uptime,
        checks=checks
    )


@router.get("/live")
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint
    Returns 200 if the service is running
    """
    return {"status": "alive", "timestamp": time.time()}


@router.get("/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint
    Returns 200 if the service is ready to handle requests
    """
    try:
        # Check critical dependencies
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await client.ping()
        await client.close()
        
        return {"status": "ready", "timestamp": time.time()}
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/services")
async def service_health():
    """
    Check health of downstream services
    """
    import httpx
    
    services = {
        "agent-registry": settings.AGENT_REGISTRY_URL,
        "orchestration-engine": settings.ORCHESTRATION_ENGINE_URL,
        "memory-management": settings.MEMORY_MANAGEMENT_URL,
        "hitl-service": settings.HITL_SERVICE_URL,
        "observability": settings.OBSERVABILITY_URL,
    }
    
    service_status = {}
    
    async def check_service(name: str, url: str) -> ServiceHealth:
        """Check individual service health"""
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                response.raise_for_status()
                
                return ServiceHealth(
                    status="healthy",
                    response_time=time.time() - start_time
                )
        except Exception as e:
            return ServiceHealth(
                status="unhealthy",
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    # Check all services concurrently
    tasks = [check_service(name, url) for name, url in services.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, (service_name, service_url) in enumerate(services.items()):
        result = results[i]
        if isinstance(result, Exception):
            service_status[service_name] = ServiceHealth(
                status="error",
                response_time=0,
                error=str(result)
            )
        else:
            service_status[service_name] = result
    
    # Determine overall status
    overall_status = "healthy"
    unhealthy_count = sum(1 for s in service_status.values() if s.status != "healthy")
    
    if unhealthy_count > 0:
        if unhealthy_count == len(services):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "services": service_status,
        "summary": {
            "total": len(services),
            "healthy": len(services) - unhealthy_count,
            "unhealthy": unhealthy_count
        }
    }


@router.get("/metrics/basic")
async def basic_metrics():
    """
    Basic system metrics (non-Prometheus format)
    """
    import psutil
    import sys
    
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "timestamp": time.time(),
            "uptime": time.time() - startup_time,
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "process": {
                "pid": process.pid,
                "memory": {
                    "rss": process_memory.rss,
                    "vms": process_memory.vms,
                    "percent": process.memory_percent()
                },
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads()
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable
            },
            "settings": {
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG,
                "version": settings.VERSION
            }
        }
        
    except Exception as e:
        logger.error("Error collecting basic metrics", error=str(e))
        return {
            "error": str(e),
            "timestamp": time.time(),
            "uptime": time.time() - startup_time
        }


@router.get("/version")
async def version_info():
    """
    Get version and build information
    """
    return {
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "python_version": __import__('sys').version,
        "build_timestamp": startup_time,
        "uptime": time.time() - startup_time
    }