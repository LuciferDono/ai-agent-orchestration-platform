# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Agent Registry Service

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, or_
from pydantic import BaseModel, Field

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.models.database import (
    Agent, AgentStatus, AgentCreate, AgentResponse,
    AgentExecution, ExecutionStatus
)
from shared.config.database import get_database_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== Agent Registry ========================

class AgentRegistry:
    """Agent registry for managing agent lifecycle"""
    
    def __init__(self):
        self.db_engine = None
        self.db_session = None
        
    async def initialize(self):
        """Initialize the agent registry"""
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
            
            logger.info("Agent registry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent registry: {e}")
            raise
    
    async def register_agent(self, agent_data: AgentCreate) -> Agent:
        """Register a new agent"""
        try:
            async with self.db_session() as session:
                # Check if agent with same name and version exists
                existing = await session.execute(
                    select(Agent).where(
                        and_(Agent.name == agent_data.name, Agent.version == agent_data.version)
                    )
                )
                if existing.scalar_one_or_none():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Agent {agent_data.name} version {agent_data.version} already exists"
                    )
                
                # Create new agent
                agent = Agent(
                    name=agent_data.name,
                    display_name=agent_data.display_name or agent_data.name,
                    description=agent_data.description,
                    version=agent_data.version,
                    capabilities=agent_data.capabilities,
                    requirements=agent_data.requirements,
                    config_schema=agent_data.config_schema,
                    default_config=agent_data.default_config,
                    author=agent_data.author,
                    tags=agent_data.tags,
                    status=AgentStatus.ACTIVE
                )
                
                session.add(agent)
                await session.commit()
                await session.refresh(agent)
                
                logger.info(f"Agent {agent.name} v{agent.version} registered successfully")
                return agent
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        try:
            async with self.db_session() as session:
                agent = await session.get(Agent, agent_id)
                return agent
                
        except Exception as e:
            logger.error(f"Failed to get agent: {e}")
            return None
    
    async def list_agents(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[AgentStatus] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None
    ) -> List[Agent]:
        """List agents with filtering"""
        try:
            async with self.db_session() as session:
                query = select(Agent)
                
                # Apply filters
                filters = []
                
                if status:
                    filters.append(Agent.status == status)
                
                if tags:
                    # Filter by tags (agent must have all specified tags)
                    for tag in tags:
                        filters.append(Agent.tags.op('@>')([tag]))
                
                if search:
                    # Search in name, display_name, and description
                    search_filter = or_(
                        Agent.name.ilike(f"%{search}%"),
                        Agent.display_name.ilike(f"%{search}%"),
                        Agent.description.ilike(f"%{search}%")
                    )
                    filters.append(search_filter)
                
                if filters:
                    query = query.where(and_(*filters))
                
                query = query.offset(skip).limit(limit).order_by(Agent.created_at.desc())
                
                result = await session.execute(query)
                agents = result.scalars().all()
                
                return list(agents)
                
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []
    
    async def update_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> Optional[Agent]:
        """Update agent information"""
        try:
            async with self.db_session() as session:
                agent = await session.get(Agent, agent_id)
                if not agent:
                    return None
                
                # Update fields
                for field, value in agent_data.items():
                    if hasattr(agent, field) and value is not None:
                        setattr(agent, field, value)
                
                agent.updated_at = datetime.now(timezone.utc)
                
                await session.commit()
                await session.refresh(agent)
                
                logger.info(f"Agent {agent.name} updated successfully")
                return agent
                
        except Exception as e:
            logger.error(f"Failed to update agent: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent"""
        try:
            async with self.db_session() as session:
                agent = await session.get(Agent, agent_id)
                if not agent:
                    return False
                
                agent.status = AgentStatus.INACTIVE
                agent.updated_at = datetime.now(timezone.utc)
                
                await session.commit()
                
                logger.info(f"Agent {agent.name} deactivated")
                return True
                
        except Exception as e:
            logger.error(f"Failed to deactivate agent: {e}")
            return False
    
    async def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get agent execution statistics"""
        try:
            async with self.db_session() as session:
                # Get total executions
                total_result = await session.execute(
                    select(AgentExecution).where(AgentExecution.agent_id == agent_id)
                )
                total_executions = len(total_result.scalars().all())
                
                # Get successful executions
                success_result = await session.execute(
                    select(AgentExecution).where(
                        and_(
                            AgentExecution.agent_id == agent_id,
                            AgentExecution.status == ExecutionStatus.COMPLETED
                        )
                    )
                )
                successful_executions = len(success_result.scalars().all())
                
                # Get failed executions
                failed_result = await session.execute(
                    select(AgentExecution).where(
                        and_(
                            AgentExecution.agent_id == agent_id,
                            AgentExecution.status == ExecutionStatus.FAILED
                        )
                    )
                )
                failed_executions = len(failed_result.scalars().all())
                
                # Calculate success rate
                success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
                
                return {
                    "total_executions": total_executions,
                    "successful_executions": successful_executions,
                    "failed_executions": failed_executions,
                    "success_rate": round(success_rate, 2)
                }
                
        except Exception as e:
            logger.error(f"Failed to get agent statistics: {e}")
            return {}
    
    async def discover_agents_by_capability(self, capability: str) -> List[Agent]:
        """Discover agents by capability"""
        try:
            async with self.db_session() as session:
                query = select(Agent).where(
                    and_(
                        Agent.status == AgentStatus.ACTIVE,
                        Agent.capabilities.op('@>')([capability])
                    )
                )
                
                result = await session.execute(query)
                agents = result.scalars().all()
                
                return list(agents)
                
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
            return []

# Global agent registry
agent_registry = AgentRegistry()

# ======================== FastAPI Application ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await agent_registry.initialize()
    yield
    # Shutdown
    if agent_registry.db_engine:
        await agent_registry.db_engine.dispose()

app = FastAPI(
    title="AI Agent Registry Service",
    description="Agent lifecycle management and discovery service",
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

class AgentUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    requirements: Optional[Dict[str, Any]] = None
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class AgentListResponse(BaseModel):
    agents: List[AgentResponse]
    total: int
    skip: int
    limit: int

class AgentStatistics(BaseModel):
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float

# ======================== API Endpoints ========================

@app.post("/agents", response_model=AgentResponse)
async def register_agent(agent_data: AgentCreate):
    """Register a new agent"""
    try:
        agent = await agent_registry.register_agent(agent_data)
        return AgentResponse.from_orm(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents", response_model=AgentListResponse)
async def list_agents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    status: Optional[AgentStatus] = Query(default=None),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags"),
    search: Optional[str] = Query(default=None)
):
    """List agents with filtering"""
    try:
        # Parse tags
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
        
        agents = await agent_registry.list_agents(
            skip=skip,
            limit=limit,
            status=status,
            tags=tag_list,
            search=search
        )
        
        return AgentListResponse(
            agents=[AgentResponse.from_orm(agent) for agent in agents],
            total=len(agents),
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get agent by ID"""
    try:
        agent = await agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse.from_orm(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, agent_update: AgentUpdate):
    """Update agent information"""
    try:
        # Filter out None values
        update_data = {k: v for k, v in agent_update.dict().items() if v is not None}
        
        agent = await agent_registry.update_agent(agent_id, update_data)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse.from_orm(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/deactivate")
async def deactivate_agent(agent_id: str):
    """Deactivate an agent"""
    try:
        success = await agent_registry.deactivate_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {"message": "Agent deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/activate")
async def activate_agent(agent_id: str):
    """Activate an agent"""
    try:
        agent = await agent_registry.update_agent(agent_id, {"status": AgentStatus.ACTIVE})
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {"message": "Agent activated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}/statistics", response_model=AgentStatistics)
async def get_agent_statistics(agent_id: str):
    """Get agent execution statistics"""
    try:
        # Check if agent exists
        agent = await agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        stats = await agent_registry.get_agent_statistics(agent_id)
        return AgentStatistics(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/discover/capability/{capability}")
async def discover_agents_by_capability(capability: str):
    """Discover agents by capability"""
    try:
        agents = await agent_registry.discover_agents_by_capability(capability)
        return {
            "capability": capability,
            "agents": [AgentResponse.from_orm(agent) for agent in agents]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capabilities")
async def get_all_capabilities():
    """Get all available capabilities"""
    try:
        async with agent_registry.db_session() as session:
            agents = await session.execute(
                select(Agent).where(Agent.status == AgentStatus.ACTIVE)
            )
            agents_list = agents.scalars().all()
            
            # Collect all unique capabilities
            capabilities = set()
            for agent in agents_list:
                if agent.capabilities:
                    capabilities.update(agent.capabilities)
            
            return {
                "capabilities": sorted(list(capabilities))
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tags")
async def get_all_tags():
    """Get all available tags"""
    try:
        async with agent_registry.db_session() as session:
            agents = await session.execute(
                select(Agent).where(Agent.status == AgentStatus.ACTIVE)
            )
            agents_list = agents.scalars().all()
            
            # Collect all unique tags
            tags = set()
            for agent in agents_list:
                if agent.tags:
                    tags.update(agent.tags)
            
            return {
                "tags": sorted(list(tags))
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent-registry",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )