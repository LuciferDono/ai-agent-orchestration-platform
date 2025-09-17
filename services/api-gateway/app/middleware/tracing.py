"""
Tracing middleware for API Gateway
OpenTelemetry-based distributed tracing
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

# OpenTelemetry imports (will be properly implemented later)
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

logger = structlog.get_logger()


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for distributed tracing with OpenTelemetry
    """
    
    def __init__(self, app, service_name: str = "api-gateway"):
        """
        Initialize tracing middleware
        
        Args:
            app: FastAPI application
            service_name: Name of the service for tracing
        """
        super().__init__(app)
        self.service_name = service_name
        self.tracer = None
        
        if TRACING_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            logger.info("Tracing middleware initialized", service=service_name)
        else:
            logger.warning("OpenTelemetry not available, tracing disabled")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with tracing
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response from the application
        """
        if not TRACING_AVAILABLE or not self.tracer:
            # If tracing is not available, just pass through
            return await call_next(request)
        
        # Create span for the request
        span_name = f"{request.method} {request.url.path}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            # Set span attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("http.scheme", request.url.scheme)
            span.set_attribute("service.name", self.service_name)
            
            # Add client information
            if request.client:
                span.set_attribute("http.client_ip", request.client.host)
            
            # Add user agent
            user_agent = request.headers.get("user-agent")
            if user_agent:
                span.set_attribute("http.user_agent", user_agent)
            
            # Add correlation ID if available
            correlation_id = getattr(request.state, 'correlation_id', None)
            if correlation_id:
                span.set_attribute("correlation.id", correlation_id)
            
            start_time = time.time()
            
            try:
                # Process request
                response = await call_next(request)
                
                # Record successful response
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.response_size", 
                                 response.headers.get("content-length", "0"))
                
                # Set span status based on response code
                if response.status_code >= 400:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                else:
                    span.set_status(trace.Status(trace.StatusCode.OK))
                
                return response
                
            except Exception as e:
                # Record error information
                span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    description=str(e)
                ))
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                
                # Re-raise the exception
                raise
                
            finally:
                # Record processing time
                process_time = time.time() - start_time
                span.set_attribute("http.duration", process_time)


class SimpleTracingMiddleware(BaseHTTPMiddleware):
    """
    Simplified tracing middleware for when OpenTelemetry is not available
    Logs trace-like information using structured logging
    """
    
    def __init__(self, app, service_name: str = "api-gateway"):
        super().__init__(app)
        self.service_name = service_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with simple trace logging
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response from the application
        """
        # Generate simple trace ID
        import uuid
        trace_id = str(uuid.uuid4())[:8]
        
        # Get correlation ID if available
        correlation_id = getattr(request.state, 'correlation_id', trace_id)
        
        start_time = time.time()
        
        try:
            # Log span start
            logger.debug(
                "Span started",
                trace_id=trace_id,
                correlation_id=correlation_id,
                service=self.service_name,
                operation=f"{request.method} {request.url.path}",
                span_kind="server"
            )
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful span completion
            logger.debug(
                "Span completed",
                trace_id=trace_id,
                correlation_id=correlation_id,
                service=self.service_name,
                operation=f"{request.method} {request.url.path}",
                status_code=response.status_code,
                duration=duration,
                span_kind="server"
            )
            
            # Add trace headers to response
            response.headers["X-Trace-ID"] = trace_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error span
            logger.error(
                "Span error",
                trace_id=trace_id,
                correlation_id=correlation_id,
                service=self.service_name,
                operation=f"{request.method} {request.url.path}",
                error=str(e),
                error_type=type(e).__name__,
                duration=duration,
                span_kind="server"
            )
            
            # Re-raise the exception
            raise