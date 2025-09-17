# Multi-stage Docker build for API Gateway service
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set metadata
LABEL maintainer="AI Agent Orchestration Platform Team"
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="ai-agent-orchestration-platform-api-gateway"
LABEL org.label-schema.description="API Gateway for AI Agent Orchestration Platform"
LABEL org.label-schema.url="https://github.com/your-org/ai-agent-orchestration-platform"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vcs-url="https://github.com/your-org/ai-agent-orchestration-platform"
LABEL org.label-schema.vendor="Your Organization"
LABEL org.label-schema.version=$VERSION
LABEL org.label-schema.schema-version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy shared modules
COPY shared/ /app/shared/

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default environment
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Development stage
FROM builder as development

# Switch back to root for development setup
USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install debugging tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER appuser

# Set development environment
ENV ENVIRONMENT=development
ENV DEBUG=true
ENV LOG_LEVEL=DEBUG

# Override CMD for development with auto-reload
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Production stage (default)
FROM builder as production

# Ensure we're using the app user
USER appuser

# Set production environment variables
ENV ENVIRONMENT=production
ENV DEBUG=false
ENV LOG_LEVEL=INFO

# Use gunicorn for production
RUN pip install --no-cache-dir gunicorn[gthread]

# Production command with gunicorn
CMD ["gunicorn", "app.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--timeout", "30", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]