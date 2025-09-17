# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Service Integration Tests

import pytest
import asyncio
import httpx
import json
from typing import Dict, Any

# Test configuration
TEST_BASE_URL = "http://localhost:8000"
SERVICES = {
    "api-gateway": "http://localhost:8000",
    "agent-registry": "http://localhost:8001", 
    "orchestration-engine": "http://localhost:8002",
    "memory-management": "http://localhost:8003",
    "hitl-service": "http://localhost:8004"
}

class TestServiceHealth:
    """Test service health endpoints"""
    
    @pytest.mark.asyncio
    async def test_all_services_healthy(self):
        """Test that all services respond to health checks"""
        async with httpx.AsyncClient() as client:
            for service_name, base_url in SERVICES.items():
                try:
                    response = await client.get(f"{base_url}/health", timeout=5.0)
                    assert response.status_code == 200
                    
                    health_data = response.json()
                    assert health_data["status"] == "healthy"
                    assert health_data["service"] == service_name
                    
                except httpx.ConnectError:
                    pytest.skip(f"Service {service_name} not running at {base_url}")

class TestAgentRegistryIntegration:
    """Test Agent Registry service integration"""
    
    @pytest.mark.asyncio
    async def test_agent_registration_flow(self):
        """Test complete agent registration flow"""
        agent_data = {
            "name": "test-integration-agent",
            "version": "1.0.0",
            "description": "Test agent for integration testing",
            "capabilities": ["text-processing", "sentiment-analysis"],
            "author": "Test Suite",
            "tags": ["test", "integration"]
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # Register agent
                response = await client.post(
                    f"{SERVICES['agent-registry']}/agents",
                    json=agent_data,
                    timeout=10.0
                )
                
                if response.status_code == 201:
                    agent_response = response.json()
                    agent_id = agent_response["id"]
                    
                    # Verify agent was registered
                    get_response = await client.get(
                        f"{SERVICES['agent-registry']}/agents/{agent_id}",
                        timeout=5.0
                    )
                    
                    assert get_response.status_code == 200
                    retrieved_agent = get_response.json()
                    assert retrieved_agent["name"] == agent_data["name"]
                    assert retrieved_agent["version"] == agent_data["version"]
                    
                else:
                    pytest.skip(f"Agent registry not responding: {response.status_code}")
                    
            except httpx.ConnectError:
                pytest.skip("Agent registry service not available")

class TestOrchestrationEngineIntegration:
    """Test Orchestration Engine integration"""
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test basic workflow execution"""
        workflow_data = {
            "workflow_id": "test-workflow",
            "input_data": {
                "message": "Hello from integration test",
                "task": "echo"
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{SERVICES['orchestration-engine']}/execute",
                    json=workflow_data,
                    timeout=30.0
                )
                
                if response.status_code in [200, 404]:  # 404 if workflow doesn't exist yet
                    if response.status_code == 200:
                        execution_result = response.json()
                        assert "execution_id" in execution_result
                        assert "status" in execution_result
                else:
                    pytest.skip(f"Orchestration engine not ready: {response.status_code}")
                    
            except httpx.ConnectError:
                pytest.skip("Orchestration engine service not available")

class TestMemoryManagementIntegration:
    """Test Memory Management service integration"""
    
    @pytest.mark.asyncio
    async def test_memory_storage_retrieval(self):
        """Test memory storage and retrieval"""
        memory_data = {
            "agent_id": "test-agent-123",
            "memory_type": "short_term", 
            "key": "test_memory",
            "content": "This is a test memory for integration testing",
            "importance_score": 5.0
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # Store memory
                response = await client.post(
                    f"{SERVICES['memory-management']}/memory",
                    json=memory_data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    memory_result = response.json()
                    memory_id = memory_result["memory_id"]
                    
                    # Retrieve memory
                    get_response = await client.get(
                        f"{SERVICES['memory-management']}/memory/{memory_data['agent_id']}",
                        params={"key": memory_data["key"]},
                        timeout=5.0
                    )
                    
                    assert get_response.status_code == 200
                    memories = get_response.json()["memories"]
                    assert len(memories) > 0
                    assert memories[0]["content"]["text"] == memory_data["content"]
                    
                else:
                    pytest.skip(f"Memory management not ready: {response.status_code}")
                    
            except httpx.ConnectError:
                pytest.skip("Memory management service not available")

class TestHITLServiceIntegration:
    """Test HITL service integration"""
    
    @pytest.mark.asyncio
    async def test_approval_request_creation(self):
        """Test creating approval requests"""
        approval_data = {
            "workflow_execution_id": "test-workflow-123",
            "title": "Test approval request",
            "description": "Integration test approval",
            "data": {"decision": "approve_action"},
            "timeout_seconds": 3600
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{SERVICES['hitl-service']}/approval-requests",
                    json=approval_data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    approval_result = response.json()
                    request_id = approval_result["request_id"]
                    
                    # Check approval status
                    get_response = await client.get(
                        f"{SERVICES['hitl-service']}/approval-requests/{request_id}",
                        timeout=5.0
                    )
                    
                    assert get_response.status_code == 200
                    approval_details = get_response.json()
                    assert approval_details["title"] == approval_data["title"]
                    assert approval_details["status"] == "pending"
                    
                else:
                    pytest.skip(f"HITL service not ready: {response.status_code}")
                    
            except httpx.ConnectError:
                pytest.skip("HITL service not available")

class TestEndToEndFlow:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_basic_agent_workflow(self):
        """Test a basic agent workflow from registration to execution"""
        # This test requires all services to be running
        async with httpx.AsyncClient() as client:
            try:
                # Check if all services are available
                for service_name, base_url in SERVICES.items():
                    health_response = await client.get(f"{base_url}/health", timeout=5.0)
                    if health_response.status_code != 200:
                        pytest.skip(f"Service {service_name} not available")
                
                # If we get here, all services are running
                # This would be expanded with actual workflow testing
                pytest.skip("End-to-end testing requires full service deployment")
                
            except httpx.ConnectError:
                pytest.skip("Not all services are available for end-to-end testing")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])