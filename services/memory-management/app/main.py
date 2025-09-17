# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Memory Management Service

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, or_, desc
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.models.database import Agent, AgentMemory
from shared.config.database import get_database_url, get_chroma_url, vector_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== Memory Management ========================

class MemoryManager:
    """Memory management service with vector database integration"""
    
    def __init__(self):
        self.db_engine = None
        self.db_session = None
        self.chroma_client = None
        self.embedding_model = None
        self.collections = {}
        
    async def initialize(self):
        """Initialize the memory management service"""
        try:
            # Setup database connection
            self.db_engine = create_async_engine(
                get_database_url(async_mode=True),
                echo=False
            )
            self.db_session = sessionmaker(
                self.db_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.HttpClient(
                host=vector_config.CHROMA_HOST,
                port=vector_config.CHROMA_PORT,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(vector_config.EMBEDDING_MODEL)
            
            # Initialize collections
            await self.initialize_collections()
            
            logger.info("Memory management service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory management service: {e}")
            raise
    
    async def initialize_collections(self):
        """Initialize ChromaDB collections"""
        try:
            # Agent memory collection
            self.collections["agent_memory"] = self.chroma_client.get_or_create_collection(
                name=vector_config.CHROMA_COLLECTION_MEMORY,
                metadata={"description": "Agent memory storage"}
            )
            
            # Documents collection
            self.collections["documents"] = self.chroma_client.get_or_create_collection(
                name=vector_config.CHROMA_COLLECTION_DOCUMENTS,
                metadata={"description": "Document storage"}
            )
            
            # General embeddings collection
            self.collections["embeddings"] = self.chroma_client.get_or_create_collection(
                name=vector_config.CHROMA_COLLECTION_EMBEDDINGS,
                metadata={"description": "General embeddings storage"}
            )
            
            logger.info("ChromaDB collections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
    
    async def store_memory(
        self,
        agent_id: str,
        memory_type: str,
        key: str,
        content: Any,
        context: Optional[Dict[str, Any]] = None,
        importance_score: float = 0.0,
        expires_at: Optional[datetime] = None
    ) -> str:
        """Store memory for an agent"""
        try:
            memory_id = str(uuid4())
            
            # Convert content to JSON if needed
            if isinstance(content, (dict, list)):
                content_json = content
                content_text = json.dumps(content, ensure_ascii=False)
            else:
                content_json = {"text": str(content)}
                content_text = str(content)
            
            # Generate embedding
            embedding = self.generate_embedding(content_text)
            
            # Store in database
            async with self.db_session() as session:
                memory = AgentMemory(
                    id=memory_id,
                    agent_id=agent_id,
                    memory_type=memory_type,
                    key=key,
                    content=content_json,
                    embedding=embedding,
                    context=context or {},
                    importance_score=importance_score,
                    expires_at=expires_at
                )
                
                session.add(memory)
                await session.commit()
            
            # Store in vector database
            if embedding:
                collection = self.collections["agent_memory"]
                collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content_text],
                    metadatas=[{
                        "agent_id": agent_id,
                        "memory_type": memory_type,
                        "key": key,
                        "importance_score": importance_score,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        **((context or {}))
                    }]
                )
            
            logger.info(f"Memory stored for agent {agent_id}: {key}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def retrieve_memory(
        self,
        agent_id: str,
        key: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memory for an agent"""
        try:
            async with self.db_session() as session:
                query = select(AgentMemory).where(AgentMemory.agent_id == agent_id)
                
                # Apply filters
                filters = []
                if key:
                    filters.append(AgentMemory.key == key)
                if memory_type:
                    filters.append(AgentMemory.memory_type == memory_type)
                
                # Filter out expired memories
                filters.append(
                    or_(
                        AgentMemory.expires_at.is_(None),
                        AgentMemory.expires_at > datetime.now(timezone.utc)
                    )
                )
                
                if filters:
                    query = query.where(and_(*filters))
                
                query = query.order_by(desc(AgentMemory.importance_score), desc(AgentMemory.created_at)).limit(limit)
                
                result = await session.execute(query)
                memories = result.scalars().all()
                
                # Convert to dict format
                memory_list = []
                for memory in memories:
                    memory_dict = {
                        "id": str(memory.id),
                        "agent_id": memory.agent_id,
                        "memory_type": memory.memory_type,
                        "key": memory.key,
                        "content": memory.content,
                        "context": memory.context,
                        "importance_score": memory.importance_score,
                        "created_at": memory.created_at.isoformat(),
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                        "access_count": memory.access_count
                    }
                    memory_list.append(memory_dict)
                
                # Update access count
                for memory in memories:
                    memory.access_count = (memory.access_count or 0) + 1
                    memory.last_accessed = datetime.now(timezone.utc)
                
                await session.commit()
                
                return memory_list
                
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return []
    
    async def search_memory(
        self,
        agent_id: str,
        query_text: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search memory using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query_text)
            if not query_embedding:
                return []
            
            # Search in vector database
            collection = self.collections["agent_memory"]
            
            # Build filter conditions
            where_conditions = {"agent_id": agent_id}
            if memory_type:
                where_conditions["memory_type"] = memory_type
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get more results to filter
                where=where_conditions,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            memory_results = []
            if results["ids"] and results["ids"][0]:
                for i, memory_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= similarity_threshold:
                        # Get full memory from database
                        async with self.db_session() as session:
                            memory = await session.get(AgentMemory, memory_id)
                            if memory and (not memory.expires_at or memory.expires_at > datetime.now(timezone.utc)):
                                memory_dict = {
                                    "id": str(memory.id),
                                    "agent_id": memory.agent_id,
                                    "memory_type": memory.memory_type,
                                    "key": memory.key,
                                    "content": memory.content,
                                    "context": memory.context,
                                    "importance_score": memory.importance_score,
                                    "similarity_score": similarity,
                                    "created_at": memory.created_at.isoformat(),
                                    "access_count": memory.access_count
                                }
                                memory_results.append(memory_dict)
                                
                                # Update access count
                                memory.access_count = (memory.access_count or 0) + 1
                                memory.last_accessed = datetime.now(timezone.utc)
                                await session.commit()
                                
                if len(memory_results) >= limit:
                    break
            
            # Sort by similarity score
            memory_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return memory_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        importance_score: Optional[float] = None
    ) -> bool:
        """Update existing memory"""
        try:
            async with self.db_session() as session:
                memory = await session.get(AgentMemory, memory_id)
                if not memory:
                    return False
                
                # Update fields
                if content is not None:
                    if isinstance(content, (dict, list)):
                        memory.content = content
                        content_text = json.dumps(content, ensure_ascii=False)
                    else:
                        memory.content = {"text": str(content)}
                        content_text = str(content)
                    
                    # Update embedding
                    embedding = self.generate_embedding(content_text)
                    memory.embedding = embedding
                    
                    # Update in vector database
                    if embedding:
                        collection = self.collections["agent_memory"]
                        collection.update(
                            ids=[memory_id],
                            embeddings=[embedding],
                            documents=[content_text]
                        )
                
                if context is not None:
                    memory.context = context
                
                if importance_score is not None:
                    memory.importance_score = importance_score
                
                memory.updated_at = datetime.now(timezone.utc)
                
                await session.commit()
                
                logger.info(f"Memory {memory_id} updated successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory"""
        try:
            async with self.db_session() as session:
                memory = await session.get(AgentMemory, memory_id)
                if not memory:
                    return False
                
                # Delete from database
                await session.delete(memory)
                await session.commit()
                
                # Delete from vector database
                try:
                    collection = self.collections["agent_memory"]
                    collection.delete(ids=[memory_id])
                except:
                    # Vector deletion might fail if not found, continue anyway
                    pass
                
                logger.info(f"Memory {memory_id} deleted successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False
    
    async def cleanup_expired_memories(self) -> int:
        """Clean up expired memories"""
        try:
            async with self.db_session() as session:
                # Find expired memories
                query = select(AgentMemory).where(
                    and_(
                        AgentMemory.expires_at.is_not(None),
                        AgentMemory.expires_at <= datetime.now(timezone.utc)
                    )
                )
                
                result = await session.execute(query)
                expired_memories = result.scalars().all()
                
                # Delete expired memories
                count = 0
                for memory in expired_memories:
                    await session.delete(memory)
                    
                    # Delete from vector database
                    try:
                        collection = self.collections["agent_memory"]
                        collection.delete(ids=[str(memory.id)])
                    except:
                        pass
                    
                    count += 1
                
                await session.commit()
                
                if count > 0:
                    logger.info(f"Cleaned up {count} expired memories")
                
                return count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired memories: {e}")
            return 0
    
    async def get_memory_statistics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            async with self.db_session() as session:
                # Build base query
                query = select(AgentMemory)
                if agent_id:
                    query = query.where(AgentMemory.agent_id == agent_id)
                
                result = await session.execute(query)
                memories = result.scalars().all()
                
                # Calculate statistics
                stats = {
                    "total_memories": len(memories),
                    "by_type": {},
                    "by_agent": {},
                    "average_importance": 0.0,
                    "total_access_count": 0,
                    "expired_count": 0
                }
                
                total_importance = 0
                current_time = datetime.now(timezone.utc)
                
                for memory in memories:
                    # By type
                    memory_type = memory.memory_type
                    stats["by_type"][memory_type] = stats["by_type"].get(memory_type, 0) + 1
                    
                    # By agent (if not filtering by specific agent)
                    if not agent_id:
                        agent_key = memory.agent_id
                        stats["by_agent"][agent_key] = stats["by_agent"].get(agent_key, 0) + 1
                    
                    # Importance
                    total_importance += memory.importance_score or 0
                    
                    # Access count
                    stats["total_access_count"] += memory.access_count or 0
                    
                    # Expired count
                    if memory.expires_at and memory.expires_at <= current_time:
                        stats["expired_count"] += 1
                
                if len(memories) > 0:
                    stats["average_importance"] = round(total_importance / len(memories), 2)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {}

# Global memory manager
memory_manager = MemoryManager()

# ======================== FastAPI Application ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await memory_manager.initialize()
    yield
    # Shutdown
    if memory_manager.db_engine:
        await memory_manager.db_engine.dispose()

app = FastAPI(
    title="AI Agent Memory Management Service",
    description="Vector-based memory management for AI agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== API Models ========================

class MemoryCreateRequest(BaseModel):
    agent_id: str
    memory_type: str = Field(..., description="Type of memory (short_term, long_term, episodic)")
    key: str = Field(..., description="Memory key/identifier")
    content: Any = Field(..., description="Memory content")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    importance_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Importance score 0-10")
    expires_hours: Optional[int] = Field(default=None, description="Expiration time in hours")

class MemoryUpdateRequest(BaseModel):
    content: Optional[Any] = None
    context: Optional[Dict[str, Any]] = None
    importance_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)

class MemorySearchRequest(BaseModel):
    agent_id: str
    query: str
    memory_type: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

# ======================== API Endpoints ========================

@app.post("/memory")
async def store_memory(request: MemoryCreateRequest):
    """Store memory for an agent"""
    try:
        # Calculate expiration
        expires_at = None
        if request.expires_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=request.expires_hours)
        
        memory_id = await memory_manager.store_memory(
            agent_id=request.agent_id,
            memory_type=request.memory_type,
            key=request.key,
            content=request.content,
            context=request.context,
            importance_score=request.importance_score,
            expires_at=expires_at
        )
        
        return {
            "memory_id": memory_id,
            "message": "Memory stored successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{agent_id}")
async def retrieve_memory(
    agent_id: str,
    key: Optional[str] = Query(default=None),
    memory_type: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=100)
):
    """Retrieve memory for an agent"""
    try:
        memories = await memory_manager.retrieve_memory(
            agent_id=agent_id,
            key=key,
            memory_type=memory_type,
            limit=limit
        )
        
        return {
            "agent_id": agent_id,
            "memories": memories,
            "count": len(memories)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search memory using semantic similarity"""
    try:
        results = await memory_manager.search_memory(
            agent_id=request.agent_id,
            query_text=request.query,
            memory_type=request.memory_type,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        return {
            "query": request.query,
            "agent_id": request.agent_id,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memory/{memory_id}")
async def update_memory(memory_id: str, request: MemoryUpdateRequest):
    """Update existing memory"""
    try:
        success = await memory_manager.update_memory(
            memory_id=memory_id,
            content=request.content,
            context=request.context,
            importance_score=request.importance_score
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"message": "Memory updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete memory"""
    try:
        success = await memory_manager.delete_memory(memory_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"message": "Memory deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/cleanup")
async def cleanup_expired_memories():
    """Clean up expired memories"""
    try:
        count = await memory_manager.cleanup_expired_memories()
        
        return {
            "message": f"Cleaned up {count} expired memories",
            "count": count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/statistics")
async def get_memory_statistics(agent_id: Optional[str] = Query(default=None)):
    """Get memory usage statistics"""
    try:
        stats = await memory_manager.get_memory_statistics(agent_id)
        
        return {
            "agent_id": agent_id,
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "memory-management",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )