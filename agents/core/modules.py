"""
Agent Modules for AI Agent Orchestration Platform
Implements the core modular components of the agent architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger()


class BaseModule(ABC):
    """Base class for all agent modules"""
    
    def __init__(self, agent: 'BaseAgent', config: Dict[str, Any] = None):
        """
        Initialize base module
        
        Args:
            agent: Reference to the parent agent
            config: Module-specific configuration
        """
        self.agent = agent
        self.config = config or {}
        self.logger = logger.bind(
            agent_name=agent.metadata.name,
            agent_id=agent.metadata.agent_id,
            module=self.__class__.__name__
        )
        self.initialized_at = datetime.utcnow()
    
    @abstractmethod
    async def initialize(self):
        """Initialize the module"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup module resources"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status"""
        return {
            "module": self.__class__.__name__,
            "initialized_at": self.initialized_at.isoformat(),
            "config": self.config
        }


class PerceptionModule(BaseModule):
    """
    Perception Module: Processes and understands input data
    Handles data preprocessing, feature extraction, and input parsing
    """
    
    async def initialize(self):
        """Initialize perception module"""
        self.logger.debug("Perception module initialized")
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data and extract meaningful information
        
        Args:
            input_data: Raw input data
            
        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            self.logger.debug("Processing input data", input_type=type(input_data).__name__)
            
            # Basic input processing - can be extended by subclasses
            processed_data = await self._process_input(input_data)
            
            # Extract features and metadata
            features = await self._extract_features(processed_data)
            metadata = await self._generate_metadata(input_data, processed_data, features)
            
            result = {
                "raw_input": input_data,
                "processed_data": processed_data,
                "features": features,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.debug("Input processing completed", features_count=len(features))
            return result
            
        except Exception as e:
            self.logger.error("Error in perception processing", error=str(e))
            return {
                "raw_input": input_data,
                "processed_data": None,
                "features": {},
                "metadata": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _process_input(self, input_data: Any) -> Any:
        """Process raw input data - override in subclasses for specific processing"""
        # Default implementation - just return the input
        return input_data
    
    async def _extract_features(self, processed_data: Any) -> Dict[str, Any]:
        """Extract features from processed data"""
        features = {}
        
        if isinstance(processed_data, str):
            features.update({
                "length": len(processed_data),
                "word_count": len(processed_data.split()) if processed_data else 0,
                "is_empty": not bool(processed_data),
                "data_type": "text"
            })
        elif isinstance(processed_data, (list, tuple)):
            features.update({
                "length": len(processed_data),
                "is_empty": len(processed_data) == 0,
                "data_type": "sequence"
            })
        elif isinstance(processed_data, dict):
            features.update({
                "key_count": len(processed_data.keys()),
                "is_empty": len(processed_data) == 0,
                "data_type": "mapping"
            })
        else:
            features.update({
                "data_type": type(processed_data).__name__,
                "is_empty": processed_data is None
            })
        
        return features
    
    async def _generate_metadata(self, input_data: Any, processed_data: Any, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata about the processing"""
        return {
            "input_type": type(input_data).__name__,
            "processed_type": type(processed_data).__name__,
            "processing_time": datetime.utcnow().isoformat(),
            "features_extracted": len(features)
        }
    
    async def cleanup(self):
        """Cleanup perception module resources"""
        self.logger.debug("Perception module cleanup completed")


class PlanningModule(BaseModule):
    """
    Planning Module: Creates execution plans based on input and reasoning
    Handles goal decomposition, strategy formation, and action sequencing
    """
    
    async def initialize(self):
        """Initialize planning module"""
        self.logger.debug("Planning module initialized")
    
    async def create_plan(
        self,
        perceived_data: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        memory_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create an execution plan
        
        Args:
            perceived_data: Output from perception module
            reasoning_result: Output from reasoning module
            memory_context: Relevant context from memory
            
        Returns:
            Dictionary containing execution plan
        """
        try:
            self.logger.debug("Creating execution plan")
            
            # Analyze the situation
            situation = await self._analyze_situation(perceived_data, reasoning_result, memory_context)
            
            # Set goals
            goals = await self._set_goals(situation)
            
            # Generate action sequence
            actions = await self._generate_actions(goals, situation)
            
            # Create plan with priorities and dependencies
            plan = {
                "plan_id": f"plan_{datetime.utcnow().timestamp()}",
                "situation": situation,
                "goals": goals,
                "actions": actions,
                "priorities": await self._assign_priorities(actions),
                "dependencies": await self._identify_dependencies(actions),
                "estimated_duration": await self._estimate_duration(actions),
                "confidence": await self._calculate_plan_confidence(goals, actions),
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "action_count": len(actions),
                    "goal_count": len(goals)
                }
            }
            
            self.logger.debug("Execution plan created", action_count=len(actions), goal_count=len(goals))
            return plan
            
        except Exception as e:
            self.logger.error("Error creating execution plan", error=str(e))
            return {
                "plan_id": f"error_plan_{datetime.utcnow().timestamp()}",
                "situation": {},
                "goals": [],
                "actions": [],
                "priorities": {},
                "dependencies": {},
                "estimated_duration": 0,
                "confidence": 0.0,
                "metadata": {"error": str(e), "created_at": datetime.utcnow().isoformat()}
            }
    
    async def _analyze_situation(self, perceived_data: Dict, reasoning_result: Dict, memory_context: Dict) -> Dict[str, Any]:
        """Analyze the current situation"""
        return {
            "input_complexity": perceived_data.get("features", {}).get("length", 0),
            "context_available": len(memory_context.get("relevant_memories", [])),
            "reasoning_confidence": reasoning_result.get("confidence", 0.5),
            "situation_type": "standard"  # Can be enhanced with more sophisticated analysis
        }
    
    async def _set_goals(self, situation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Set goals based on situation analysis"""
        # Default goal - process the input and provide a response
        return [
            {
                "id": "process_input",
                "description": "Process the input and generate appropriate response",
                "priority": "high",
                "type": "primary"
            }
        ]
    
    async def _generate_actions(self, goals: List[Dict], situation: Dict) -> List[Dict[str, Any]]:
        """Generate action sequence to achieve goals"""
        actions = []
        
        for goal in goals:
            if goal["type"] == "primary":
                actions.extend([
                    {
                        "id": f"action_validate_input_{goal['id']}",
                        "type": "validation",
                        "description": "Validate input data",
                        "goal_id": goal["id"],
                        "required": True
                    },
                    {
                        "id": f"action_process_{goal['id']}",
                        "type": "processing",
                        "description": "Process input according to agent logic",
                        "goal_id": goal["id"],
                        "required": True
                    },
                    {
                        "id": f"action_validate_output_{goal['id']}",
                        "type": "validation",
                        "description": "Validate output quality",
                        "goal_id": goal["id"],
                        "required": False
                    }
                ])
        
        return actions
    
    async def _assign_priorities(self, actions: List[Dict]) -> Dict[str, int]:
        """Assign priorities to actions"""
        priorities = {}
        for i, action in enumerate(actions):
            if action.get("required", True):
                priorities[action["id"]] = 1  # High priority
            else:
                priorities[action["id"]] = 2  # Lower priority
        return priorities
    
    async def _identify_dependencies(self, actions: List[Dict]) -> Dict[str, List[str]]:
        """Identify dependencies between actions"""
        dependencies = {}
        
        # Simple dependency logic - validation before processing
        for action in actions:
            if action["type"] == "processing":
                # Find validation action for the same goal
                validation_actions = [
                    a["id"] for a in actions 
                    if a["type"] == "validation" 
                    and a.get("goal_id") == action.get("goal_id")
                    and "validate_input" in a["id"]
                ]
                if validation_actions:
                    dependencies[action["id"]] = validation_actions
        
        return dependencies
    
    async def _estimate_duration(self, actions: List[Dict]) -> float:
        """Estimate plan execution duration in seconds"""
        # Simple estimation - can be enhanced with historical data
        base_time_per_action = 1.0  # seconds
        return len(actions) * base_time_per_action
    
    async def _calculate_plan_confidence(self, goals: List[Dict], actions: List[Dict]) -> float:
        """Calculate confidence in the execution plan"""
        # Simple confidence calculation based on plan complexity
        if not goals or not actions:
            return 0.0
        
        # Higher confidence for simpler plans
        complexity_penalty = min(0.1 * len(actions), 0.5)
        base_confidence = 0.8
        
        return max(base_confidence - complexity_penalty, 0.1)
    
    async def cleanup(self):
        """Cleanup planning module resources"""
        self.logger.debug("Planning module cleanup completed")


class MemoryModule(BaseModule):
    """
    Memory Module: Manages agent memory and context
    Handles short-term, long-term, and episodic memory storage and retrieval
    """
    
    def __init__(self, agent: 'BaseAgent', config: Dict[str, Any] = None):
        super().__init__(agent, config)
        self.short_term_memory: List[Dict] = []
        self.long_term_memory: List[Dict] = []
        self.episodic_memory: List[Dict] = []
        self.max_short_term = config.get("max_short_term", 10)
        self.max_long_term = config.get("max_long_term", 100)
        self.max_episodic = config.get("max_episodic", 50)
    
    async def initialize(self):
        """Initialize memory module"""
        self.logger.debug("Memory module initialized")
    
    async def retrieve_context(self, perceived_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant context from memory
        
        Args:
            perceived_data: Current input data
            
        Returns:
            Dictionary containing relevant memory context
        """
        try:
            self.logger.debug("Retrieving memory context")
            
            # Get relevant memories from different memory types
            relevant_short_term = await self._search_short_term_memory(perceived_data)
            relevant_long_term = await self._search_long_term_memory(perceived_data)
            relevant_episodic = await self._search_episodic_memory(perceived_data)
            
            context = {
                "short_term_memories": relevant_short_term,
                "long_term_memories": relevant_long_term,
                "episodic_memories": relevant_episodic,
                "memory_stats": {
                    "short_term_count": len(self.short_term_memory),
                    "long_term_count": len(self.long_term_memory),
                    "episodic_count": len(self.episodic_memory)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.debug("Memory context retrieved", 
                            short_term=len(relevant_short_term),
                            long_term=len(relevant_long_term),
                            episodic=len(relevant_episodic))
            
            return context
            
        except Exception as e:
            self.logger.error("Error retrieving memory context", error=str(e))
            return {
                "short_term_memories": [],
                "long_term_memories": [],
                "episodic_memories": [],
                "memory_stats": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def store_experience(
        self,
        input_data: Any,
        perceived_data: Dict,
        reasoning_result: Dict,
        plan: Dict,
        action_result: Dict
    ):
        """
        Store experience in memory
        
        Args:
            input_data: Original input
            perceived_data: Processed input
            reasoning_result: Reasoning output
            plan: Execution plan
            action_result: Action execution result
        """
        try:
            experience = {
                "timestamp": datetime.utcnow().isoformat(),
                "input_data": input_data,
                "perceived_data": perceived_data,
                "reasoning_result": reasoning_result,
                "plan": plan,
                "action_result": action_result,
                "success": action_result.get("success", False),
                "confidence": reasoning_result.get("confidence", 0.5)
            }
            
            # Store in short-term memory
            await self._store_short_term(experience)
            
            # Decide if it should go to long-term memory
            if await self._should_store_long_term(experience):
                await self._store_long_term(experience)
            
            # Store as episodic memory if significant
            if await self._is_significant_episode(experience):
                await self._store_episodic(experience)
            
            self.logger.debug("Experience stored in memory", success=experience["success"])
            
        except Exception as e:
            self.logger.error("Error storing experience", error=str(e))
    
    async def _search_short_term_memory(self, perceived_data: Dict) -> List[Dict]:
        """Search short-term memory for relevant entries"""
        # Simple similarity search - can be enhanced with vector similarity
        relevant = []
        input_features = perceived_data.get("features", {})
        
        for memory in self.short_term_memory[-5:]:  # Last 5 entries
            if self._is_memory_relevant(memory, input_features):
                relevant.append(memory)
        
        return relevant
    
    async def _search_long_term_memory(self, perceived_data: Dict) -> List[Dict]:
        """Search long-term memory for relevant entries"""
        # Similar logic to short-term but with different scoring
        relevant = []
        input_features = perceived_data.get("features", {})
        
        for memory in self.long_term_memory:
            if (self._is_memory_relevant(memory, input_features) and 
                memory.get("success", False)):  # Only successful experiences
                relevant.append(memory)
        
        return relevant[:3]  # Limit to top 3
    
    async def _search_episodic_memory(self, perceived_data: Dict) -> List[Dict]:
        """Search episodic memory for relevant entries"""
        # Look for similar episodes
        return []  # Placeholder - implement episode similarity matching
    
    def _is_memory_relevant(self, memory: Dict, input_features: Dict) -> bool:
        """Check if a memory is relevant to current input"""
        memory_features = memory.get("perceived_data", {}).get("features", {})
        
        # Simple relevance check - can be enhanced
        if not memory_features or not input_features:
            return False
        
        # Check data type similarity
        if (memory_features.get("data_type") == input_features.get("data_type")):
            return True
        
        return False
    
    async def _store_short_term(self, experience: Dict):
        """Store experience in short-term memory"""
        self.short_term_memory.append(experience)
        
        # Keep only recent memories
        if len(self.short_term_memory) > self.max_short_term:
            self.short_term_memory = self.short_term_memory[-self.max_short_term:]
    
    async def _store_long_term(self, experience: Dict):
        """Store experience in long-term memory"""
        self.long_term_memory.append(experience)
        
        # Keep only important memories
        if len(self.long_term_memory) > self.max_long_term:
            # Remove least successful memories
            self.long_term_memory.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            self.long_term_memory = self.long_term_memory[:self.max_long_term]
    
    async def _store_episodic(self, experience: Dict):
        """Store experience as episodic memory"""
        episode = {
            "episode_id": f"ep_{datetime.utcnow().timestamp()}",
            "timestamp": experience["timestamp"],
            "context": experience.get("perceived_data", {}),
            "outcome": experience.get("action_result", {}),
            "significance": await self._calculate_significance(experience)
        }
        
        self.episodic_memory.append(episode)
        
        if len(self.episodic_memory) > self.max_episodic:
            # Remove least significant episodes
            self.episodic_memory.sort(key=lambda x: x.get("significance", 0), reverse=True)
            self.episodic_memory = self.episodic_memory[:self.max_episodic]
    
    async def _should_store_long_term(self, experience: Dict) -> bool:
        """Determine if experience should be stored in long-term memory"""
        # Store successful experiences with high confidence
        return (experience.get("success", False) and 
                experience.get("confidence", 0) > 0.7)
    
    async def _is_significant_episode(self, experience: Dict) -> bool:
        """Determine if experience is a significant episode"""
        # Store failures and highly successful experiences
        return (not experience.get("success", True) or 
                experience.get("confidence", 0) > 0.9)
    
    async def _calculate_significance(self, experience: Dict) -> float:
        """Calculate significance score for episodic memory"""
        base_score = 0.5
        
        # Increase significance for failures (learning opportunities)
        if not experience.get("success", True):
            base_score += 0.3
        
        # Increase significance for high confidence successes
        confidence = experience.get("confidence", 0)
        if confidence > 0.8:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    async def cleanup(self):
        """Cleanup memory module resources"""
        self.short_term_memory.clear()
        # Keep long-term and episodic for persistence
        self.logger.debug("Memory module cleanup completed")


class ReasoningModule(BaseModule):
    """
    Reasoning Module: Analyzes information and makes decisions
    Handles inference, decision-making logic, and situation assessment
    """
    
    async def initialize(self):
        """Initialize reasoning module"""
        self.logger.debug("Reasoning module initialized")
    
    async def analyze(self, perceived_data: Dict[str, Any], memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the situation and make reasoning decisions
        
        Args:
            perceived_data: Processed input data
            memory_context: Relevant memory context
            
        Returns:
            Dictionary containing reasoning results
        """
        try:
            self.logger.debug("Starting reasoning analysis")
            
            # Assess the situation
            situation_assessment = await self._assess_situation(perceived_data, memory_context)
            
            # Make decisions based on assessment
            decisions = await self._make_decisions(situation_assessment, memory_context)
            
            # Calculate confidence in reasoning
            confidence = await self._calculate_reasoning_confidence(situation_assessment, decisions)
            
            reasoning_result = {
                "situation_assessment": situation_assessment,
                "decisions": decisions,
                "confidence": confidence,
                "reasoning_path": await self._trace_reasoning_path(situation_assessment, decisions),
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "decision_count": len(decisions),
                    "memory_context_used": len(memory_context.get("short_term_memories", []))
                }
            }
            
            self.logger.debug("Reasoning analysis completed", confidence=confidence, decision_count=len(decisions))
            return reasoning_result
            
        except Exception as e:
            self.logger.error("Error in reasoning analysis", error=str(e))
            return {
                "situation_assessment": {},
                "decisions": [],
                "confidence": 0.0,
                "reasoning_path": [],
                "metadata": {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
            }
    
    async def _assess_situation(self, perceived_data: Dict, memory_context: Dict) -> Dict[str, Any]:
        """Assess the current situation"""
        features = perceived_data.get("features", {})
        
        assessment = {
            "complexity": self._calculate_complexity(features),
            "novelty": self._calculate_novelty(perceived_data, memory_context),
            "confidence_factors": self._identify_confidence_factors(perceived_data, memory_context),
            "risk_factors": self._identify_risk_factors(perceived_data),
            "opportunity_factors": self._identify_opportunities(perceived_data, memory_context)
        }
        
        return assessment
    
    async def _make_decisions(self, situation_assessment: Dict, memory_context: Dict) -> List[Dict[str, Any]]:
        """Make reasoning decisions based on situation assessment"""
        decisions = []
        
        # Primary decision: How to process this input
        processing_decision = {
            "id": "processing_approach",
            "type": "processing",
            "decision": await self._decide_processing_approach(situation_assessment),
            "rationale": "Based on complexity and novelty assessment",
            "confidence": situation_assessment.get("confidence_factors", {}).get("overall", 0.5)
        }
        decisions.append(processing_decision)
        
        # Risk mitigation decisions
        if situation_assessment.get("risk_factors", {}).get("high_risk", False):
            risk_decision = {
                "id": "risk_mitigation",
                "type": "safety",
                "decision": "apply_conservative_approach",
                "rationale": "High risk factors detected",
                "confidence": 0.8
            }
            decisions.append(risk_decision)
        
        return decisions
    
    def _calculate_complexity(self, features: Dict) -> str:
        """Calculate complexity level of the input"""
        if not features:
            return "unknown"
        
        # Simple complexity assessment based on input size
        length = features.get("length", 0)
        if length < 10:
            return "low"
        elif length < 100:
            return "medium"
        else:
            return "high"
    
    def _calculate_novelty(self, perceived_data: Dict, memory_context: Dict) -> str:
        """Calculate how novel/new the input is"""
        similar_memories = memory_context.get("short_term_memories", [])
        
        if not similar_memories:
            return "high"  # No similar memories = novel
        elif len(similar_memories) < 2:
            return "medium"
        else:
            return "low"
    
    def _identify_confidence_factors(self, perceived_data: Dict, memory_context: Dict) -> Dict[str, Any]:
        """Identify factors that affect confidence"""
        factors = {
            "clear_input": not perceived_data.get("features", {}).get("is_empty", True),
            "relevant_memory": len(memory_context.get("short_term_memories", [])) > 0,
            "data_quality": perceived_data.get("features", {}).get("data_type") != "unknown"
        }
        
        # Calculate overall confidence
        confidence_score = sum(factors.values()) / len(factors)
        factors["overall"] = confidence_score
        
        return factors
    
    def _identify_risk_factors(self, perceived_data: Dict) -> Dict[str, Any]:
        """Identify potential risk factors"""
        features = perceived_data.get("features", {})
        
        return {
            "empty_input": features.get("is_empty", False),
            "unknown_format": features.get("data_type") == "unknown",
            "high_risk": False  # Placeholder for more sophisticated risk assessment
        }
    
    def _identify_opportunities(self, perceived_data: Dict, memory_context: Dict) -> Dict[str, Any]:
        """Identify opportunities for optimization or learning"""
        return {
            "learning_opportunity": len(memory_context.get("short_term_memories", [])) == 0,
            "optimization_potential": perceived_data.get("features", {}).get("length", 0) > 50
        }
    
    async def _decide_processing_approach(self, situation_assessment: Dict) -> str:
        """Decide on the processing approach"""
        complexity = situation_assessment.get("complexity", "medium")
        novelty = situation_assessment.get("novelty", "medium")
        
        if complexity == "high" or novelty == "high":
            return "careful_analysis"
        elif complexity == "low" and novelty == "low":
            return "fast_processing"
        else:
            return "standard_processing"
    
    async def _calculate_reasoning_confidence(self, situation_assessment: Dict, decisions: List[Dict]) -> float:
        """Calculate overall confidence in reasoning"""
        if not decisions:
            return 0.0
        
        # Base confidence on situation assessment confidence factors
        base_confidence = situation_assessment.get("confidence_factors", {}).get("overall", 0.5)
        
        # Adjust based on decision confidence
        decision_confidences = [d.get("confidence", 0.5) for d in decisions]
        avg_decision_confidence = sum(decision_confidences) / len(decision_confidences)
        
        # Weighted average
        overall_confidence = (base_confidence * 0.6) + (avg_decision_confidence * 0.4)
        
        return min(max(overall_confidence, 0.0), 1.0)
    
    async def _trace_reasoning_path(self, situation_assessment: Dict, decisions: List[Dict]) -> List[str]:
        """Trace the reasoning path for explainability"""
        path = [
            f"Assessed situation complexity: {situation_assessment.get('complexity', 'unknown')}",
            f"Assessed novelty: {situation_assessment.get('novelty', 'unknown')}",
            f"Made {len(decisions)} decision(s)"
        ]
        
        for decision in decisions:
            path.append(f"Decision '{decision['id']}': {decision['decision']} - {decision['rationale']}")
        
        return path
    
    async def cleanup(self):
        """Cleanup reasoning module resources"""
        self.logger.debug("Reasoning module cleanup completed")


class ActionModule(BaseModule):
    """
    Action Module: Executes plans and interacts with external systems
    Handles tool execution, external API calls, and action coordination
    """
    
    async def initialize(self):
        """Initialize action module"""
        self.logger.debug("Action module initialized")
    
    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the given plan
        
        Args:
            plan: Execution plan from planning module
            
        Returns:
            Dictionary containing execution results
        """
        try:
            self.logger.debug("Executing action plan", plan_id=plan.get("plan_id"))
            
            actions = plan.get("actions", [])
            results = []
            
            # Execute actions in dependency order
            execution_order = await self._determine_execution_order(actions, plan.get("dependencies", {}))
            
            for action_id in execution_order:
                action = next((a for a in actions if a["id"] == action_id), None)
                if action:
                    result = await self._execute_single_action(action)
                    results.append(result)
                    
                    # Stop execution if a required action fails
                    if not result.get("success", False) and action.get("required", True):
                        self.logger.warning("Required action failed, stopping execution", action_id=action_id)
                        break
            
            # Aggregate results
            execution_result = await self._aggregate_results(results, plan)
            
            self.logger.debug("Action plan execution completed", 
                            success=execution_result.get("success", False),
                            actions_executed=len(results))
            
            return execution_result
            
        except Exception as e:
            self.logger.error("Error executing action plan", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "execution_error": True
                }
            }
    
    async def _determine_execution_order(self, actions: List[Dict], dependencies: Dict[str, List[str]]) -> List[str]:
        """Determine the order to execute actions based on dependencies"""
        # Simple topological sort for action dependencies
        ordered = []
        remaining = {action["id"]: action for action in actions}
        
        while remaining:
            # Find actions with no unmet dependencies
            ready = []
            for action_id, action in remaining.items():
                deps = dependencies.get(action_id, [])
                if all(dep_id in ordered for dep_id in deps):
                    ready.append(action_id)
            
            if not ready:
                # Circular dependency or error - just add remaining actions
                ready = list(remaining.keys())
            
            # Add ready actions to order
            for action_id in ready:
                ordered.append(action_id)
                del remaining[action_id]
        
        return ordered
    
    async def _execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action"""
        action_type = action.get("type", "unknown")
        action_id = action.get("id", "unknown")
        
        try:
            self.logger.debug("Executing single action", action_id=action_id, action_type=action_type)
            
            # Route to appropriate execution method based on action type
            if action_type == "validation":
                result = await self._execute_validation_action(action)
            elif action_type == "processing":
                result = await self._execute_processing_action(action)
            elif action_type == "communication":
                result = await self._execute_communication_action(action)
            else:
                result = await self._execute_generic_action(action)
            
            result["action_id"] = action_id
            result["action_type"] = action_type
            
            return result
            
        except Exception as e:
            self.logger.error("Error executing single action", action_id=action_id, error=str(e))
            return {
                "action_id": action_id,
                "action_type": action_type,
                "success": False,
                "error": str(e),
                "data": None,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_validation_action(self, action: Dict) -> Dict[str, Any]:
        """Execute validation action"""
        # Simple validation - always succeeds for now
        return {
            "success": True,
            "data": {"validation_passed": True},
            "message": f"Validation completed for {action.get('description', 'action')}",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_processing_action(self, action: Dict) -> Dict[str, Any]:
        """Execute processing action"""
        # Call the agent's process method - this is where the main logic goes
        try:
            # Get the input data from the agent's current context
            if hasattr(self.agent, 'current_context') and self.agent.current_context:
                # This would be the processed input from perception
                input_data = getattr(self.agent.current_context, 'input_data', None)
            else:
                input_data = None
            
            # Call the agent's specific processing logic
            result_data = await self.agent.process(input_data)
            
            return {
                "success": True,
                "data": result_data,
                "message": f"Processing completed: {action.get('description', 'action')}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "message": f"Processing failed: {action.get('description', 'action')}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_communication_action(self, action: Dict) -> Dict[str, Any]:
        """Execute communication action"""
        # Placeholder for communication actions
        return {
            "success": True,
            "data": {"message_sent": True},
            "message": f"Communication completed: {action.get('description', 'action')}",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_generic_action(self, action: Dict) -> Dict[str, Any]:
        """Execute generic action"""
        # Fallback for unknown action types
        return {
            "success": True,
            "data": {"generic_action_completed": True},
            "message": f"Generic action completed: {action.get('description', 'action')}",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _aggregate_results(self, results: List[Dict], plan: Dict) -> Dict[str, Any]:
        """Aggregate individual action results into overall result"""
        successful_actions = [r for r in results if r.get("success", False)]
        failed_actions = [r for r in results if not r.get("success", False)]
        
        overall_success = len(failed_actions) == 0
        
        # Get the main processing result if available
        processing_results = [r for r in results if r.get("action_type") == "processing"]
        main_data = processing_results[0].get("data") if processing_results else None
        
        return {
            "success": overall_success,
            "data": main_data,
            "error": failed_actions[0].get("error") if failed_actions else None,
            "metadata": {
                "plan_id": plan.get("plan_id"),
                "total_actions": len(results),
                "successful_actions": len(successful_actions),
                "failed_actions": len(failed_actions),
                "action_results": results,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    async def cleanup(self):
        """Cleanup action module resources"""
        self.logger.debug("Action module cleanup completed")


class MonitoringModule(BaseModule):
    """
    Monitoring Module: Self-assessment and quality assurance
    Handles performance monitoring, error detection, and quality metrics
    """
    
    async def initialize(self):
        """Initialize monitoring module"""
        self.logger.debug("Monitoring module initialized")
    
    async def assess_performance(
        self,
        input_data: Any,
        output_data: Any,
        reasoning_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess agent performance and output quality
        
        Args:
            input_data: Original input data
            output_data: Agent output
            reasoning_result: Reasoning module output
            
        Returns:
            Dictionary containing performance assessment
        """
        try:
            self.logger.debug("Assessing agent performance")
            
            # Quality assessment
            quality_score = await self._assess_output_quality(input_data, output_data)
            
            # Performance metrics
            performance_metrics = await self._calculate_performance_metrics(reasoning_result)
            
            # Confidence assessment
            confidence = await self._assess_confidence(reasoning_result, quality_score)
            
            # Error detection
            errors = await self._detect_errors(input_data, output_data, reasoning_result)
            
            assessment = {
                "quality_score": quality_score,
                "performance_metrics": performance_metrics,
                "confidence": confidence,
                "errors": errors,
                "recommendations": await self._generate_recommendations(quality_score, errors),
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "assessment_version": "1.0"
                }
            }
            
            self.logger.debug("Performance assessment completed", 
                            quality_score=quality_score, 
                            confidence=confidence,
                            error_count=len(errors))
            
            return assessment
            
        except Exception as e:
            self.logger.error("Error in performance assessment", error=str(e))
            return {
                "quality_score": 0.0,
                "performance_metrics": {},
                "confidence": 0.0,
                "errors": [{"type": "assessment_error", "message": str(e)}],
                "recommendations": [],
                "metadata": {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
            }
    
    async def _assess_output_quality(self, input_data: Any, output_data: Any) -> float:
        """Assess the quality of the output"""
        if output_data is None:
            return 0.0
        
        # Simple quality assessment based on output characteristics
        quality_score = 0.5  # Base score
        
        # Check if output is not empty/None
        if output_data:
            quality_score += 0.2
        
        # Check if output type matches input type (for simple cases)
        if type(output_data) == type(input_data):
            quality_score += 0.1
        
        # Check if output is structured (dict/list)
        if isinstance(output_data, (dict, list)):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    async def _calculate_performance_metrics(self, reasoning_result: Dict) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {
            "reasoning_confidence": reasoning_result.get("confidence", 0.0),
            "decision_count": len(reasoning_result.get("decisions", [])),
            "reasoning_complexity": len(reasoning_result.get("reasoning_path", []))
        }
    
    async def _assess_confidence(self, reasoning_result: Dict, quality_score: float) -> float:
        """Assess overall confidence in the result"""
        reasoning_confidence = reasoning_result.get("confidence", 0.0)
        
        # Weighted combination of reasoning confidence and quality score
        overall_confidence = (reasoning_confidence * 0.7) + (quality_score * 0.3)
        
        return min(max(overall_confidence, 0.0), 1.0)
    
    async def _detect_errors(self, input_data: Any, output_data: Any, reasoning_result: Dict) -> List[Dict[str, Any]]:
        """Detect potential errors or issues"""
        errors = []
        
        # Check for empty output
        if output_data is None:
            errors.append({
                "type": "empty_output",
                "message": "Agent produced no output",
                "severity": "high"
            })
        
        # Check for low reasoning confidence
        if reasoning_result.get("confidence", 0) < 0.3:
            errors.append({
                "type": "low_confidence",
                "message": "Reasoning confidence is very low",
                "severity": "medium"
            })
        
        # Check for reasoning errors
        if "error" in reasoning_result.get("metadata", {}):
            errors.append({
                "type": "reasoning_error",
                "message": "Error occurred during reasoning",
                "severity": "high"
            })
        
        return errors
    
    async def _generate_recommendations(self, quality_score: float, errors: List[Dict]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.append("Consider improving output quality assessment")
        
        if any(error["severity"] == "high" for error in errors):
            recommendations.append("Address high-severity errors before proceeding")
        
        if len(errors) > 3:
            recommendations.append("Multiple errors detected - review agent logic")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup monitoring module resources"""
        self.logger.debug("Monitoring module cleanup completed")


class CommunicationModule(BaseModule):
    """
    Communication Module: Inter-agent communication and messaging
    Handles messaging, coordination, and information sharing between agents
    """
    
    def __init__(self, agent: 'BaseAgent', config: Dict[str, Any] = None):
        super().__init__(agent, config)
        self.message_queue: List[Dict] = []
        self.subscriptions: List[str] = []
    
    async def initialize(self):
        """Initialize communication module"""
        self.logger.debug("Communication module initialized")
    
    async def send_message(self, recipient: str, message: Dict[str, Any]) -> bool:
        """
        Send message to another agent or service
        
        Args:
            recipient: Target agent/service identifier
            message: Message content
            
        Returns:
            True if message was sent successfully
        """
        try:
            message_packet = {
                "id": f"msg_{datetime.utcnow().timestamp()}",
                "sender": self.agent.metadata.agent_id,
                "recipient": recipient,
                "timestamp": datetime.utcnow().isoformat(),
                "content": message
            }
            
            # In a real implementation, this would send via message broker
            # For now, we'll just log it
            self.logger.info("Message sent", recipient=recipient, message_id=message_packet["id"])
            
            return True
            
        except Exception as e:
            self.logger.error("Error sending message", recipient=recipient, error=str(e))
            return False
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """
        Receive next message from queue
        
        Returns:
            Message dictionary or None if no messages
        """
        if self.message_queue:
            return self.message_queue.pop(0)
        return None
    
    async def subscribe(self, topic: str) -> bool:
        """
        Subscribe to a communication topic
        
        Args:
            topic: Topic name to subscribe to
            
        Returns:
            True if subscription successful
        """
        if topic not in self.subscriptions:
            self.subscriptions.append(topic)
            self.logger.debug("Subscribed to topic", topic=topic)
            return True
        return False
    
    async def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from a communication topic
        
        Args:
            topic: Topic name to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        if topic in self.subscriptions:
            self.subscriptions.remove(topic)
            self.logger.debug("Unsubscribed from topic", topic=topic)
            return True
        return False
    
    async def broadcast(self, message: Dict[str, Any], topic: str = None) -> int:
        """
        Broadcast message to all agents or topic subscribers
        
        Args:
            message: Message to broadcast
            topic: Optional topic for targeted broadcast
            
        Returns:
            Number of recipients message was sent to
        """
        # Placeholder implementation
        self.logger.info("Message broadcast", topic=topic, message_id=f"broadcast_{datetime.utcnow().timestamp()}")
        return 0  # Would return actual recipient count in real implementation
    
    async def cleanup(self):
        """Cleanup communication module resources"""
        self.message_queue.clear()
        self.subscriptions.clear()
        self.logger.debug("Communication module cleanup completed")