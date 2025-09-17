"""
Configuration settings for API Gateway service
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API Gateway configuration settings"""
    
    # Application
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # API Configuration
    API_V1_PREFIX: str = Field(default="/api/v1", description="API version 1 prefix")
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_WORKERS: int = Field(default=4, description="Number of API workers")
    API_RELOAD: bool = Field(default=True, description="Auto-reload in development")
    API_BASE_URL: str = Field(default="http://localhost:8000", description="Base URL for API")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:password@localhost:5432/ai_orchestration",
        description="Database connection URL"
    )
    
    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    
    # Authentication & Security
    JWT_SECRET: str = Field(
        default="your-super-secret-jwt-key-change-in-production",
        description="JWT secret key"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, description="JWT access token expiration in minutes"
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, description="JWT refresh token expiration in days"
    )
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, description="Rate limit per minute")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, description="Rate limit per hour")
    RATE_LIMIT_PER_DAY: int = Field(default=10000, description="Rate limit per day")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins"
    )
    
    # Security
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Allowed hosts"
    )
    SECURE_COOKIES: bool = Field(default=False, description="Use secure cookies")
    SESSION_TIMEOUT: int = Field(default=86400, description="Session timeout in seconds")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    ENABLE_TRACING: bool = Field(default=True, description="Enable distributed tracing")
    
    # Service URLs (for proxying)
    AGENT_REGISTRY_URL: str = Field(
        default="http://localhost:8001",
        description="Agent Registry service URL"
    )
    ORCHESTRATION_ENGINE_URL: str = Field(
        default="http://localhost:8002",
        description="Orchestration Engine service URL"
    )
    MEMORY_MANAGEMENT_URL: str = Field(
        default="http://localhost:8003",
        description="Memory Management service URL"
    )
    HITL_SERVICE_URL: str = Field(
        default="http://localhost:8004",
        description="HITL service URL"
    )
    OBSERVABILITY_URL: str = Field(
        default="http://localhost:8005",
        description="Observability service URL"
    )
    SECURITY_SERVICE_URL: str = Field(
        default="http://localhost:8006",
        description="Security service URL"
    )
    
    # External Service Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # Monitoring
    PROMETHEUS_URL: str = Field(
        default="http://localhost:9090",
        description="Prometheus server URL"
    )
    GRAFANA_URL: str = Field(
        default="http://localhost:3001",
        description="Grafana server URL"
    )
    JAEGER_URL: str = Field(
        default="http://localhost:16686",
        description="Jaeger server URL"
    )
    
    # Feature Flags
    ENABLE_AUTH: bool = Field(default=True, description="Enable authentication")
    ENABLE_RATE_LIMITING: bool = Field(default=True, description="Enable rate limiting")
    ENABLE_REQUEST_LOGGING: bool = Field(default=True, description="Enable request logging")
    ENABLE_PROXY: bool = Field(default=True, description="Enable service proxying")
    
    # Email Configuration (for notifications)
    SMTP_HOST: Optional[str] = Field(default=None, description="SMTP host")
    SMTP_PORT: Optional[int] = Field(default=587, description="SMTP port")
    SMTP_USERNAME: Optional[str] = Field(default=None, description="SMTP username")
    SMTP_PASSWORD: Optional[str] = Field(default=None, description="SMTP password")
    SMTP_FROM_EMAIL: Optional[str] = Field(default=None, description="SMTP from email")
    
    # File Upload
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    UPLOAD_PATH: str = Field(default="/tmp/uploads", description="Upload directory path")
    
    # Timeouts
    REQUEST_TIMEOUT: int = Field(default=30, description="HTTP request timeout in seconds")
    DATABASE_TIMEOUT: int = Field(default=10, description="Database operation timeout in seconds")
    REDIS_TIMEOUT: int = Field(default=5, description="Redis operation timeout in seconds")
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = Field(default=20, description="Default pagination page size")
    MAX_PAGE_SIZE: int = Field(default=100, description="Maximum pagination page size")
    
    # Development Settings
    AUTO_RELOAD: bool = Field(default=True, description="Auto-reload on file changes")
    ENABLE_DEBUG_TOOLBAR: bool = Field(default=False, description="Enable debug toolbar")
    ENABLE_PROFILER: bool = Field(default=False, description="Enable request profiler")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate environment-specific settings
        if self.ENVIRONMENT == "production":
            self._validate_production_settings()
        elif self.ENVIRONMENT == "development":
            self._set_development_defaults()
    
    def _validate_production_settings(self):
        """Validate production-specific settings"""
        if self.JWT_SECRET == "your-super-secret-jwt-key-change-in-production":
            raise ValueError("JWT_SECRET must be changed in production")
        
        if not self.SECURE_COOKIES and self.ENVIRONMENT == "production":
            import warnings
            warnings.warn("SECURE_COOKIES should be True in production")
        
        if self.DEBUG:
            import warnings
            warnings.warn("DEBUG should be False in production")
    
    def _set_development_defaults(self):
        """Set development-specific defaults"""
        if not hasattr(self, '_dev_defaults_set'):
            # Enable additional development features
            self.AUTO_RELOAD = True
            self.ENABLE_DEBUG_TOOLBAR = True
            self._dev_defaults_set = True
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.ENVIRONMENT.lower() == "testing"


# Global settings instance
settings = Settings()

# Export commonly used settings for convenience
__all__ = ["settings", "Settings"]