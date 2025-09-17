"""
Base Agent class for AI Agent Orchestration Platform
Defines the core agent architecture with modular components
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import structlog

from agents.core.modules import (
    PerceptionModule,
    PlanningModule,
    MemoryModule,
    ReasoningModule,
    ActionModule,
    MonitoringModule,
    CommunicationModule
)

logger = structlog.get_logger()


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETE = "complete"
    TERMINATED = "terminated"


class AgentMetadata:
    """Agent metadata container"""
    
    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        capabilities: List[str] = None,
        dependencies: Dict[str, str] = None,
        author: str = None,
        created_at: datetime = None
    ):
        self.name = name
        self.version = version
        self.description = description
        self.capabilities = capabilities or []
        self.dependencies = dependencies or {}
        self.author = author
        self.created_at = created_at or datetime.utcnow()
        self.agent_id = f"{name}_{version}_{uuid.uuid4().hex[:8]}"


class AgentContext:
    """Agent execution context"""
    
    def __init__(
        self,
        session_id: str = None,
        user_id: str = None,
        workflow_id: str = None,
        parent_agent_id: str = None,
        environment: Dict[str, Any] = None
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.workflow_id = workflow_id
        self.parent_agent_id = parent_agent_id
        self.environment = environment or {}
        self.created_at = datetime.utcnow()
        self.correlation_id = str(uuid.uuid4())


class AgentResult:
    """Agent execution result"""
    
    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: str = None,
        metadata: Dict[str, Any] = None,
        confidence: float = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.confidence = confidence
        self.timestamp = datetime.utcnow()
        self.execution_time = None


class BaseAgent(ABC):
    """
    Base Agent class defining the core agent architecture
    
    All agents inherit from this class and implement the required methods.
    The agent architecture follows a modular design with separate modules
    for perception, planning, memory, reasoning, action, monitoring, and communication.
    """
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        capabilities: List[str] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name/identifier
            version: Agent version
            description: Human-readable description
            capabilities: List of agent capabilities
            config: Agent configuration dictionary
        """
        self.metadata = AgentMetadata(
            name=name,
            version=version,
            description=description,
            capabilities=capabilities or []
        )
        
        self.config = config or {}
        self.state = AgentState.IDLE
        self.logger = logger.bind(
            agent_name=name,
            agent_id=self.metadata.agent_id,
            agent_version=version
        )
        
        # Initialize agent modules
        self.perception = self._initialize_perception()
        self.planning = self._initialize_planning()
        self.memory = self._initialize_memory()
        self.reasoning = self._initialize_reasoning()
        self.action = self._initialize_action()
        self.monitoring = self._initialize_monitoring()
        self.communication = self._initialize_communication()
        
        # Execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.current_context: Optional[AgentContext] = None
        
        self.logger.info("Agent initialized", capabilities=self.metadata.capabilities)
    
    def _initialize_perception(self) -> PerceptionModule:
        """Initialize perception module - can be overridden by subclasses"""
        return PerceptionModule(agent=self)
    
    def _initialize_planning(self) -> PlanningModule:
        """Initialize planning module - can be overridden by subclasses"""
        return PlanningModule(agent=self)
    
    def _initialize_memory(self) -> MemoryModule:
        """Initialize memory module - can be overridden by subclasses"""
        return MemoryModule(agent=self)
    
    def _initialize_reasoning(self) -> ReasoningModule:
        """Initialize reasoning module - can be overridden by subclasses"""
        return ReasoningModule(agent=self)
    
    def _initialize_action(self) -> ActionModule:
        """Initialize action module - can be overridden by subclasses"""
        return ActionModule(agent=self)
    
    def _initialize_monitoring(self) -> MonitoringModule:
        """Initialize monitoring module - can be overridden by subclasses"""
        return MonitoringModule(agent=self)
    
    def _initialize_communication(self) -> CommunicationModule:
        """Initialize communication module - can be overridden by subclasses"""
        return CommunicationModule(agent=self)
    
    async def execute(
        self,
        input_data: Any,
        context: AgentContext = None,
        timeout: Optional[int] = None
    ) -> AgentResult:
        """
        Main agent execution method
        
        Args:
            input_data: Input data for agent processing
            context: Execution context
            timeout: Optional timeout in seconds
            
        Returns:
            AgentResult with execution outcome
        """
        start_time = datetime.utcnow()
        self.current_context = context or AgentContext()
        
        # Set up correlation logging
        execution_logger = self.logger.bind(
            correlation_id=self.current_context.correlation_id,
            session_id=self.current_context.session_id
        )
        
        try:
            execution_logger.info(
                "Agent execution started",
                input_type=type(input_data).__name__,
                timeout=timeout
            )
            
            # Change state to processing
            self.state = AgentState.PROCESSING
            
            # Execute with timeout if specified
            if timeout:
                result = await asyncio.wait_for(
                    self._execute_internal(input_data, execution_logger),
                    timeout=timeout
                )
            else:
                result = await self._execute_internal(input_data, execution_logger)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Update state based on result
            self.state = AgentState.COMPLETE if result.success else AgentState.ERROR
            
            # Record execution in history
            self._record_execution(input_data, result, execution_time)
            
            execution_logger.info(
                "Agent execution completed",
                success=result.success,
                execution_time=execution_time,
                confidence=result.confidence
            )
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.state = AgentState.ERROR
            
            result = AgentResult(
                success=False,
                error=f"Agent execution timed out after {timeout} seconds",
                metadata={"timeout": timeout, "execution_time": execution_time}
            )
            result.execution_time = execution_time
            
            execution_logger.error(
                "Agent execution timeout",
                timeout=timeout,
                execution_time=execution_time
            )
            
            self._record_execution(input_data, result, execution_time)
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.state = AgentState.ERROR
            
            result = AgentResult(
                success=False,
                error=str(e),
                metadata={"exception_type": type(e).__name__, "execution_time": execution_time}
            )
            result.execution_time = execution_time
            
            execution_logger.error(
                "Agent execution error",
                error=str(e),
                exception_type=type(e).__name__,
                execution_time=execution_time
            )
            
            self._record_execution(input_data, result, execution_time)
            return result
    
    async def _execute_internal(
        self,
        input_data: Any,
        execution_logger: structlog.stdlib.BoundLogger
    ) -> AgentResult:
        """
        Internal execution method that implements the agent processing pipeline
        
        Args:
            input_data: Input data to process
            execution_logger: Logger with execution context
            
        Returns:
            AgentResult with processing outcome
        """
        try:
            # Step 1: Perception - Process and understand input
            execution_logger.debug("Starting perception phase")
            perceived_data = await self.perception.process(input_data)
            
            # Step 2: Memory Retrieval - Get relevant context and history
            execution_logger.debug("Starting memory retrieval")
            memory_context = await self.memory.retrieve_context(perceived_data)
            
            # Step 3: Reasoning - Analyze and understand the situation
            execution_logger.debug("Starting reasoning phase")
            reasoning_result = await self.reasoning.analyze(
                perceived_data,
                memory_context
            )
            
            # Step 4: Planning - Create execution plan
            execution_logger.debug("Starting planning phase")
            plan = await self.planning.create_plan(
                perceived_data,
                reasoning_result,
                memory_context
            )
            
            # Step 5: Action Execution - Execute the plan
            execution_logger.debug("Starting action execution")
            action_result = await self.action.execute(plan)
            
            # Step 6: Memory Storage - Store results and learning
            execution_logger.debug("Storing results in memory")
            await self.memory.store_experience(
                input_data=input_data,
                perceived_data=perceived_data,
                reasoning_result=reasoning_result,
                plan=plan,
                action_result=action_result
            )
            
            # Step 7: Monitoring - Self-assessment and quality check
            execution_logger.debug("Starting self-monitoring")
            monitoring_result = await self.monitoring.assess_performance(
                input_data=input_data,
                output_data=action_result,
                reasoning_result=reasoning_result
            )
            
            # Prepare final result
            result = AgentResult(
                success=action_result.get("success", True),
                data=action_result.get("data"),
                error=action_result.get("error"),
                confidence=monitoring_result.get("confidence"),
                metadata={
                    "perception": perceived_data.get("metadata", {}),
                    "reasoning": reasoning_result.get("metadata", {}),
                    "planning": plan.get("metadata", {}),
                    "action": action_result.get("metadata", {}),
                    "monitoring": monitoring_result
                }
            )
            
            return result
            
        except Exception as e:
            execution_logger.error(
                "Error in agent execution pipeline",
                error=str(e),
                exception_type=type(e).__name__
            )
            raise
    
    def _record_execution(
        self,
        input_data: Any,
        result: AgentResult,
        execution_time: float
    ):
        """Record execution in agent history"""
        execution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_type": type(input_data).__name__,
            "success": result.success,
            "execution_time": execution_time,
            "confidence": result.confidence,
            "error": result.error,
            "correlation_id": self.current_context.correlation_id if self.current_context else None
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 100 executions to prevent memory bloat
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Abstract method that must be implemented by subclasses
        This is the main processing logic specific to each agent type
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed result
        """
        pass
    
    async def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data
        Can be overridden by subclasses for specific validation logic
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return input_data is not None
    
    async def cleanup(self):
        """Cleanup agent resources"""
        self.logger.info("Agent cleanup initiated")
        
        # Cleanup all modules
        await self.perception.cleanup()
        await self.planning.cleanup()
        await self.memory.cleanup()
        await self.reasoning.cleanup()
        await self.action.cleanup()
        await self.monitoring.cleanup()
        await self.communication.cleanup()
        
        self.state = AgentState.TERMINATED
        self.logger.info("Agent cleanup completed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.metadata.agent_id,
            "name": self.metadata.name,
            "version": self.metadata.version,
            "state": self.state.value,
            "capabilities": self.metadata.capabilities,
            "execution_count": len(self.execution_history),
            "last_execution": self.execution_history[-1] if self.execution_history else None,
            "uptime": (datetime.utcnow() - self.metadata.created_at).total_seconds()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "average_confidence": 0.0
            }
        
        successful_executions = [e for e in self.execution_history if e["success"]]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "success_rate": len(successful_executions) / len(self.execution_history),
            "average_execution_time": sum(e["execution_time"] for e in self.execution_history) / len(self.execution_history),
            "average_confidence": sum(e.get("confidence", 0) or 0 for e in successful_executions) / max(len(successful_executions), 1),
            "last_24h_executions": len([
                e for e in self.execution_history 
                if (datetime.utcnow() - datetime.fromisoformat(e["timestamp"])).total_seconds() < 86400
            ])
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.metadata.name}, version={self.metadata.version}, state={self.state.value})"