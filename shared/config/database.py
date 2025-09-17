# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Database Configuration

import os
from typing import Optional
from pydantic import BaseSettings, Field

class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    
    # Database connection
    DATABASE_URL: str = Field(
        default="postgresql://postgres:password@localhost:5432/ai_orchestration",
        description="Primary database connection URL"
    )
    
    ASYNC_DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/ai_orchestration",
        description="Async database connection URL"
    )
    
    # Redis connection
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching and sessions"
    )
    
    # Vector database
    CHROMA_HOST: str = Field(default="localhost", description="ChromaDB host")
    CHROMA_PORT: int = Field(default=8000, description="ChromaDB port")
    CHROMA_COLLECTION: str = Field(default="agent_memory", description="Default collection name")
    
    # Connection pool settings
    DATABASE_POOL_SIZE: int = Field(default=20, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, description="Max connections beyond pool size")
    DATABASE_POOL_TIMEOUT: int = Field(default=30, description="Pool connection timeout seconds")
    DATABASE_POOL_RECYCLE: int = Field(default=3600, description="Connection recycle time seconds")
    
    # Query settings
    DATABASE_ECHO: bool = Field(default=False, description="Echo SQL queries to console")
    DATABASE_ECHO_POOL: bool = Field(default=False, description="Echo pool events to console")
    
    # Migration settings
    ALEMBIC_CONFIG_PATH: str = Field(default="alembic.ini", description="Alembic configuration file")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class RedisConfig(BaseSettings):
    """Redis configuration settings"""
    
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # Connection settings
    REDIS_MAX_CONNECTIONS: int = Field(default=20, description="Max Redis connections in pool")
    REDIS_RETRY_ON_TIMEOUT: bool = Field(default=True, description="Retry on timeout")
    REDIS_SOCKET_TIMEOUT: int = Field(default=30, description="Socket timeout seconds")
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(default=30, description="Connect timeout seconds")
    
    # Cache settings
    REDIS_DEFAULT_EXPIRE: int = Field(default=3600, description="Default key expiration seconds")
    REDIS_SESSION_EXPIRE: int = Field(default=86400, description="Session expiration seconds")
    
    # Rate limiting
    REDIS_RATE_LIMIT_PREFIX: str = Field(default="rl:", description="Rate limit key prefix")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class VectorDBConfig(BaseSettings):
    """Vector database configuration"""
    
    # ChromaDB settings
    CHROMA_HOST: str = Field(default="localhost", description="ChromaDB host")
    CHROMA_PORT: int = Field(default=8000, description="ChromaDB port")
    CHROMA_HTTP_PORT: int = Field(default=8000, description="ChromaDB HTTP port")
    
    # Collections
    CHROMA_COLLECTION_MEMORY: str = Field(default="agent_memory", description="Agent memory collection")
    CHROMA_COLLECTION_DOCUMENTS: str = Field(default="documents", description="Document collection")
    CHROMA_COLLECTION_EMBEDDINGS: str = Field(default="embeddings", description="General embeddings collection")
    
    # Embedding settings
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Default embedding model")
    EMBEDDING_DIMENSION: int = Field(default=384, description="Embedding vector dimension")
    
    # Search settings
    DEFAULT_SEARCH_LIMIT: int = Field(default=10, description="Default search result limit")
    SIMILARITY_THRESHOLD: float = Field(default=0.7, description="Similarity threshold for results")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global configuration instances
db_config = DatabaseConfig()
redis_config = RedisConfig()
vector_config = VectorDBConfig()

# Database connection functions
def get_database_url(async_mode: bool = False) -> str:
    """Get database URL for sync or async connections"""
    if async_mode:
        return db_config.ASYNC_DATABASE_URL
    return db_config.DATABASE_URL

def get_redis_url() -> str:
    """Get Redis connection URL"""
    return redis_config.REDIS_URL

def get_chroma_url() -> str:
    """Get ChromaDB connection URL"""
    return f"http://{vector_config.CHROMA_HOST}:{vector_config.CHROMA_PORT}"