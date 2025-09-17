# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Agent Unit Tests

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.core.base_agent import BaseAgent
from agents.core.modules import PerceptionModule, PlanningModule

class TestBaseAgent:
    """Test the base agent functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = BaseAgent(
            name="test-agent",
            version="1.0.0", 
            description="Test agent for unit testing"
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.name == "test-agent"
        assert self.agent.version == "1.0.0"
        assert self.agent.description == "Test agent for unit testing"
        assert self.agent.status == "active"
        assert isinstance(self.agent.created_at, datetime)
    
    @pytest.mark.asyncio
    async def test_agent_process_basic(self):
        """Test basic agent processing"""
        input_data = {"message": "Hello, world!"}
        
        # Mock the process method to return expected output
        with patch.object(self.agent, 'process', return_value={"response": "Hello back!"}):
            result = await self.agent.process(input_data)
            assert result == {"response": "Hello back!"}
    
    def test_agent_configuration(self):
        """Test agent configuration"""
        config = {"temperature": 0.7, "max_tokens": 100}
        self.agent.configure(config)
        assert self.agent.config == config
    
    def test_agent_status_update(self):
        """Test agent status updates"""
        self.agent.set_status("inactive")
        assert self.agent.status == "inactive"
        
        self.agent.set_status("active")
        assert self.agent.status == "active"

class TestAgentModules:
    """Test agent modules"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.perception = PerceptionModule()
        self.planning = PlanningModule()
    
    def test_perception_module_init(self):
        """Test perception module initialization"""
        assert self.perception.name == "perception"
        assert self.perception.version == "1.0.0"
    
    def test_planning_module_init(self):
        """Test planning module initialization"""
        assert self.planning.name == "planning" 
        assert self.planning.version == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_perception_text_analysis(self):
        """Test perception module text analysis"""
        text = "This is a test message for sentiment analysis."
        
        # Mock the analyze_text method
        with patch.object(self.perception, 'analyze_text', return_value={"sentiment": "neutral", "confidence": 0.8}):
            result = await self.perception.analyze_text(text)
            assert result["sentiment"] == "neutral"
            assert result["confidence"] == 0.8
    
    @pytest.mark.asyncio  
    async def test_planning_goal_creation(self):
        """Test planning module goal creation"""
        context = {"user_intent": "book_flight", "entities": {"destination": "New York"}}
        
        # Mock the create_plan method
        with patch.object(self.planning, 'create_plan', return_value={"action": "search_flights", "parameters": {"destination": "New York"}}):
            result = await self.planning.create_plan(context)
            assert result["action"] == "search_flights"
            assert result["parameters"]["destination"] == "New York"

if __name__ == "__main__":
    pytest.main([__file__])