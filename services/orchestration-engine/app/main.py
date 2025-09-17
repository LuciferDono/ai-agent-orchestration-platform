# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Orchestration Engine Service

import asyncio
import json
import logging
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union
from uuid import uuid4, UUID

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres_aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.models.database import (
    WorkflowExecution, Agent, Workflow, ExecutionStatus,
    AgentExecution, ApprovalRequest, ApprovalStatus
)
from shared.config.database import get_database_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== State Definitions ========================

class WorkflowState(BaseModel):
    """Base state for workflow execution"""
    execution_id: str
    workflow_id: str
    current_node: Optional[str] = None
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    messages: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    status: str = "running"
    retry_count: int = 0
    approval_pending: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class AgentExecutionState(BaseModel):
    """State for individual agent execution"""
    agent_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = {}
    error: Optional[str] = None
    duration: float = 0.0
    memory_used: int = 0
    
    class Config:
        arbitrary_types_allowed = True

# ======================== Orchestration Engine ========================

class OrchestrationEngine:
    """Core orchestration engine using LangGraph"""
    
    def __init__(self):
        self.active_executions: Dict[str, Any] = {}
        self.checkpointer = None
        self.db_engine = None
        self.db_session = None
        
    async def initialize(self):
        """Initialize the orchestration engine"""
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
            
            # Setup LangGraph checkpointer
            self.checkpointer = AsyncPostgresSaver.from_conn_string(
                get_database_url(async_mode=True)
            )
            await self.checkpointer.setup()
            
            logger.info("Orchestration engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestration engine: {e}")
            raise
    
    async def create_workflow_graph(self, workflow: Workflow) -> StateGraph:
        """Create LangGraph workflow from workflow definition"""
        try:
            # Create state graph
            graph = StateGraph(WorkflowState)
            
            # Add nodes from workflow definition
            definition = workflow.definition
            
            for node in definition.get("nodes", []):
                node_id = node["id"]
                node_type = node["type"]
                
                if node_type == "agent":
                    # Create agent node
                    agent_func = await self.create_agent_node(node)
                    graph.add_node(node_id, agent_func)
                elif node_type == "condition":
                    # Create condition node
                    condition_func = await self.create_condition_node(node)
                    graph.add_node(node_id, condition_func)
                elif node_type == "human_approval":
                    # Create human approval node
                    approval_func = await self.create_approval_node(node)
                    graph.add_node(node_id, approval_func)
                elif node_type == "start":
                    # Start node
                    graph.add_node(node_id, self.start_node)
                elif node_type == "end":
                    # End node
                    graph.add_node(node_id, self.end_node)
            
            # Add edges
            for edge in definition.get("edges", []):
                source = edge["source"]
                target = edge["target"]
                
                if edge.get("condition"):
                    # Conditional edge
                    condition_func = await self.create_edge_condition(edge["condition"])
                    graph.add_conditional_edges(
                        source,
                        condition_func,
                        edge.get("mapping", {target: target})
                    )
                else:
                    # Simple edge
                    graph.add_edge(source, target)
            
            # Set entry point
            entry_point = definition.get("entry_point", "start")
            graph.set_entry_point(entry_point)
            
            return graph
            
        except Exception as e:
            logger.error(f"Failed to create workflow graph: {e}")
            raise
    
    async def create_agent_node(self, node_config: Dict[str, Any]) -> Callable:
        """Create an agent execution node"""
        agent_id = node_config.get("agent_id")
        config = node_config.get("config", {})
        
        async def agent_node(state: WorkflowState) -> WorkflowState:
            """Execute agent node"""
            try:
                start_time = datetime.now(timezone.utc)
                
                # Load agent
                async with self.db_session() as session:
                    agent = await session.get(Agent, agent_id)
                    if not agent:
                        raise HTTPException(status_code=404, f"Agent {agent_id} not found")
                
                # Execute agent
                from agents.core.base_agent import BaseAgent
                
                # Create agent instance (simplified for demo)
                agent_instance = BaseAgent(
                    name=agent.name,
                    version=agent.version,
                    description=agent.description
                )
                
                # Execute with input data
                input_data = state.input_data.copy()
                input_data.update(state.context)
                
                result = await agent_instance.process(input_data)
                
                # Calculate duration
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Update state
                state.output_data.update(result)
                state.context[f"{node_config['id']}_output"] = result
                
                # Log execution
                async with self.db_session() as session:
                    execution = AgentExecution(
                        agent_id=agent_id,
                        workflow_execution_id=state.execution_id,
                        status=ExecutionStatus.COMPLETED,
                        input_data=input_data,
                        output_data=result,
                        started_at=start_time,
                        completed_at=datetime.now(timezone.utc),
                        duration_seconds=duration
                    )
                    session.add(execution)
                    await session.commit()
                
                logger.info(f"Agent {agent.name} executed successfully in {duration:.2f}s")
                return state
                
            except Exception as e:
                state.error_message = str(e)
                state.status = "failed"
                logger.error(f"Agent execution failed: {e}")
                return state
        
        return agent_node
    
    async def create_condition_node(self, node_config: Dict[str, Any]) -> Callable:
        """Create a condition evaluation node"""
        condition = node_config.get("condition", "")
        
        async def condition_node(state: WorkflowState) -> WorkflowState:
            """Evaluate condition node"""
            try:
                # Simple condition evaluation (can be enhanced)
                context = {
                    "input": state.input_data,
                    "output": state.output_data,
                    "context": state.context
                }
                
                # Evaluate condition (simplified)
                result = eval(condition, {"__builtins__": {}}, context)
                state.context[f"{node_config['id']}_result"] = result
                
                return state
                
            except Exception as e:
                state.error_message = f"Condition evaluation failed: {e}"
                logger.error(f"Condition evaluation failed: {e}")
                return state
        
        return condition_node
    
    async def create_approval_node(self, node_config: Dict[str, Any]) -> Callable:
        """Create a human approval node"""
        approval_config = node_config.get("approval", {})
        
        async def approval_node(state: WorkflowState) -> WorkflowState:
            """Handle human approval"""
            try:
                # Create approval request
                async with self.db_session() as session:
                    approval_request = ApprovalRequest(
                        workflow_execution_id=state.execution_id,
                        title=approval_config.get("title", "Approval Required"),
                        description=approval_config.get("description", ""),
                        data=state.output_data,
                        timeout_seconds=approval_config.get("timeout", 3600)
                    )
                    session.add(approval_request)
                    await session.commit()
                
                # Set approval pending
                state.approval_pending = True
                state.context["approval_request_id"] = str(approval_request.id)
                
                logger.info(f"Approval request created: {approval_request.id}")
                return state
                
            except Exception as e:
                state.error_message = f"Approval request failed: {e}"
                logger.error(f"Approval request failed: {e}")
                return state
        
        return approval_node
    
    async def create_edge_condition(self, condition_config: Dict[str, Any]) -> Callable:
        """Create conditional edge function"""
        condition = condition_config.get("expression", "")
        
        def edge_condition(state: WorkflowState) -> str:
            """Evaluate edge condition"""
            try:
                context = {
                    "input": state.input_data,
                    "output": state.output_data,
                    "context": state.context
                }
                
                # Evaluate condition
                result = eval(condition, {"__builtins__": {}}, context)
                return "true" if result else "false"
                
            except Exception as e:
                logger.error(f"Edge condition evaluation failed: {e}")
                return "false"
        
        return edge_condition
    
    async def start_node(self, state: WorkflowState) -> WorkflowState:
        """Workflow start node"""
        state.status = "running"
        state.current_node = "start"
        logger.info(f"Workflow {state.workflow_id} started")
        return state
    
    async def end_node(self, state: WorkflowState) -> WorkflowState:
        """Workflow end node"""
        state.status = "completed"
        state.current_node = "end"
        logger.info(f"Workflow {state.workflow_id} completed")
        return state
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a workflow"""
        try:
            if not execution_id:
                execution_id = str(uuid4())
            
            # Load workflow
            async with self.db_session() as session:
                workflow = await session.get(Workflow, workflow_id)
                if not workflow:
                    raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
            
            # Create initial state
            initial_state = WorkflowState(
                execution_id=execution_id,
                workflow_id=workflow_id,
                input_data=input_data
            )
            
            # Create workflow graph
            graph = await self.create_workflow_graph(workflow)
            app = graph.compile(checkpointer=self.checkpointer)
            
            # Execute workflow
            config = {"configurable": {"thread_id": execution_id}}
            
            result = await app.ainvoke(
                initial_state.dict(),
                config=config
            )
            
            # Update database
            async with self.db_session() as session:
                execution = await session.get(WorkflowExecution, execution_id)
                if execution:
                    execution.status = ExecutionStatus.COMPLETED if result.get("status") == "completed" else ExecutionStatus.FAILED
                    execution.output_data = result.get("output_data", {})
                    execution.completed_at = datetime.now(timezone.utc)
                    execution.error_message = result.get("error_message")
                    await session.commit()
            
            return {
                "execution_id": execution_id,
                "status": result.get("status"),
                "output_data": result.get("output_data", {}),
                "error_message": result.get("error_message")
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Update execution status to failed
            try:
                async with self.db_session() as session:
                    execution = await session.get(WorkflowExecution, execution_id)
                    if execution:
                        execution.status = ExecutionStatus.FAILED
                        execution.error_message = str(e)
                        execution.completed_at = datetime.now(timezone.utc)
                        await session.commit()
            except:
                pass
            
            raise HTTPException(status_code=500, detail=str(e))

# Global orchestration engine
orchestration_engine = OrchestrationEngine()

# ======================== FastAPI Application ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await orchestration_engine.initialize()
    yield
    # Shutdown
    if orchestration_engine.db_engine:
        await orchestration_engine.db_engine.dispose()

app = FastAPI(
    title="AI Agent Orchestration Engine",
    description="LangGraph-based workflow orchestration service",
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

class WorkflowExecutionRequest(BaseModel):
    workflow_id: str
    input_data: Dict[str, Any]
    configuration: Dict[str, Any] = {}

class WorkflowExecutionResponse(BaseModel):
    execution_id: str
    status: str
    output_data: Dict[str, Any]
    error_message: Optional[str] = None

# ======================== API Endpoints ========================

@app.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(request: WorkflowExecutionRequest):
    """Execute a workflow"""
    try:
        result = await orchestration_engine.execute_workflow(
            workflow_id=request.workflow_id,
            input_data=request.input_data
        )
        
        return WorkflowExecutionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute/async")
async def execute_workflow_async(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks
):
    """Execute a workflow asynchronously"""
    try:
        execution_id = str(uuid4())
        
        # Add to background tasks
        background_tasks.add_task(
            orchestration_engine.execute_workflow,
            request.workflow_id,
            request.input_data,
            execution_id
        )
        
        return {"execution_id": execution_id, "status": "started"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/execution/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get workflow execution status"""
    try:
        async with orchestration_engine.db_session() as session:
            execution = await session.get(WorkflowExecution, execution_id)
            if not execution:
                raise HTTPException(status_code=404, detail="Execution not found")
            
            return {
                "execution_id": str(execution.id),
                "workflow_id": str(execution.workflow_id),
                "status": execution.status.value,
                "current_node": execution.current_node,
                "output_data": execution.output_data,
                "error_message": execution.error_message,
                "started_at": execution.started_at,
                "completed_at": execution.completed_at
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execution/{execution_id}/resume")
async def resume_execution(execution_id: str):
    """Resume a paused workflow execution"""
    try:
        # Implementation for resuming workflow
        # This would load the checkpoint and continue execution
        raise HTTPException(status_code=501, detail="Resume functionality not yet implemented")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execution/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel a running workflow execution"""
    try:
        async with orchestration_engine.db_session() as session:
            execution = await session.get(WorkflowExecution, execution_id)
            if not execution:
                raise HTTPException(status_code=404, detail="Execution not found")
            
            execution.status = ExecutionStatus.CANCELLED
            execution.completed_at = datetime.now(timezone.utc)
            await session.commit()
            
            return {"message": "Execution cancelled"}
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "orchestration-engine",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )