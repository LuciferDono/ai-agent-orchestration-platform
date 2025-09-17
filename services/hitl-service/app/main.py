# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Human-in-the-Loop Service

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, or_, desc, func
from pydantic import BaseModel, Field, EmailStr

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.models.database import (
    ApprovalRequest, ApprovalStatus, User, WorkflowExecution
)
from shared.config.database import get_database_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== HITL Service ========================

class HITLService:
    """Human-in-the-Loop service for approval workflows"""
    
    def __init__(self):
        self.db_engine = None
        self.db_session = None
        self.notification_tasks = {}
        
    async def initialize(self):
        """Initialize the HITL service"""
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
            
            logger.info("HITL service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HITL service: {e}")
            raise
    
    async def create_approval_request(
        self,
        workflow_execution_id: str,
        title: str,
        description: str,
        data: Dict[str, Any],
        approver_email: Optional[str] = None,
        timeout_seconds: int = 3600,
        escalation_rules: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Create a new approval request"""
        try:
            request_id = str(uuid4())
            
            # Find approver by email if provided
            approver_id = None
            if approver_email:
                async with self.db_session() as session:
                    user_result = await session.execute(
                        select(User).where(User.email == approver_email)
                    )
                    user = user_result.scalar_one_or_none()
                    if user:
                        approver_id = str(user.id)
            
            # Calculate expiration time
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
            
            # Create approval request
            async with self.db_session() as session:
                approval_request = ApprovalRequest(
                    id=request_id,
                    workflow_execution_id=workflow_execution_id,
                    approver_id=approver_id,
                    title=title,
                    description=description,
                    data=data,
                    status=ApprovalStatus.PENDING,
                    created_at=datetime.now(timezone.utc),
                    expires_at=expires_at,
                    timeout_seconds=timeout_seconds,
                    escalation_rules=escalation_rules or []
                )
                
                session.add(approval_request)
                await session.commit()
            
            # Send notification
            if approver_email:
                await self.send_approval_notification(request_id, approver_email)
            
            # Schedule timeout and escalation checks
            await self.schedule_timeout_check(request_id, timeout_seconds)
            
            logger.info(f"Approval request {request_id} created for workflow {workflow_execution_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to create approval request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_approval(
        self,
        request_id: str,
        approver_id: str,
        approved: bool,
        reason: Optional[str] = None
    ) -> bool:
        """Process an approval decision"""
        try:
            async with self.db_session() as session:
                # Get approval request
                approval_request = await session.get(ApprovalRequest, request_id)
                if not approval_request:
                    raise HTTPException(status_code=404, detail="Approval request not found")
                
                # Check if already processed
                if approval_request.status != ApprovalStatus.PENDING:
                    raise HTTPException(status_code=400, detail="Approval request already processed")
                
                # Check if expired
                if approval_request.expires_at and approval_request.expires_at <= datetime.now(timezone.utc):
                    approval_request.status = ApprovalStatus.TIMEOUT
                    await session.commit()
                    raise HTTPException(status_code=400, detail="Approval request has expired")
                
                # Update approval request
                approval_request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
                approval_request.approver_id = approver_id
                approval_request.decision_reason = reason
                approval_request.decided_at = datetime.now(timezone.utc)
                
                await session.commit()
                
                # Cancel timeout task
                if request_id in self.notification_tasks:
                    self.notification_tasks[request_id].cancel()
                    del self.notification_tasks[request_id]
                
                # Notify workflow execution service
                await self.notify_workflow_service(request_id, approved)
                
                logger.info(f"Approval request {request_id} {'approved' if approved else 'rejected'}")
                return True
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to process approval: {e}")
            return False
    
    async def get_approval_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get approval request details"""
        try:
            async with self.db_session() as session:
                approval_request = await session.get(ApprovalRequest, request_id)
                if not approval_request:
                    return None
                
                # Get approver details
                approver_info = None
                if approval_request.approver_id:
                    approver = await session.get(User, approval_request.approver_id)
                    if approver:
                        approver_info = {
                            "id": str(approver.id),
                            "email": approver.email,
                            "full_name": approver.full_name
                        }
                
                # Get workflow details
                workflow_info = None
                if approval_request.workflow_execution_id:
                    workflow = await session.get(WorkflowExecution, approval_request.workflow_execution_id)
                    if workflow:
                        workflow_info = {
                            "id": str(workflow.id),
                            "workflow_id": str(workflow.workflow_id),
                            "status": workflow.status.value
                        }
                
                return {
                    "id": str(approval_request.id),
                    "workflow_execution_id": str(approval_request.workflow_execution_id),
                    "title": approval_request.title,
                    "description": approval_request.description,
                    "data": approval_request.data,
                    "status": approval_request.status.value,
                    "decision_reason": approval_request.decision_reason,
                    "created_at": approval_request.created_at.isoformat(),
                    "expires_at": approval_request.expires_at.isoformat() if approval_request.expires_at else None,
                    "decided_at": approval_request.decided_at.isoformat() if approval_request.decided_at else None,
                    "timeout_seconds": approval_request.timeout_seconds,
                    "escalation_rules": approval_request.escalation_rules,
                    "approver": approver_info,
                    "workflow": workflow_info
                }
                
        except Exception as e:
            logger.error(f"Failed to get approval request: {e}")
            return None
    
    async def list_approval_requests(
        self,
        status: Optional[ApprovalStatus] = None,
        approver_id: Optional[str] = None,
        workflow_execution_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List approval requests with filtering"""
        try:
            async with self.db_session() as session:
                query = select(ApprovalRequest)
                
                # Apply filters
                filters = []
                if status:
                    filters.append(ApprovalRequest.status == status)
                if approver_id:
                    filters.append(ApprovalRequest.approver_id == approver_id)
                if workflow_execution_id:
                    filters.append(ApprovalRequest.workflow_execution_id == workflow_execution_id)
                
                if filters:
                    query = query.where(and_(*filters))
                
                query = query.order_by(desc(ApprovalRequest.created_at)).offset(skip).limit(limit)
                
                result = await session.execute(query)
                approval_requests = result.scalars().all()
                
                # Convert to dict format
                requests_list = []
                for request in approval_requests:
                    request_dict = await self.get_approval_request(str(request.id))
                    if request_dict:
                        requests_list.append(request_dict)
                
                return requests_list
                
        except Exception as e:
            logger.error(f"Failed to list approval requests: {e}")
            return []
    
    async def check_timeouts(self):
        """Check for timed out approval requests"""
        try:
            async with self.db_session() as session:
                # Find pending requests that have expired
                current_time = datetime.now(timezone.utc)
                query = select(ApprovalRequest).where(
                    and_(
                        ApprovalRequest.status == ApprovalStatus.PENDING,
                        ApprovalRequest.expires_at <= current_time
                    )
                )
                
                result = await session.execute(query)
                expired_requests = result.scalars().all()
                
                # Update status to timeout
                for request in expired_requests:
                    request.status = ApprovalStatus.TIMEOUT
                    request.decided_at = current_time
                    
                    # Notify workflow service
                    await self.notify_workflow_service(str(request.id), False, "timeout")
                    
                    logger.info(f"Approval request {request.id} timed out")
                
                if expired_requests:
                    await session.commit()
                
                return len(expired_requests)
                
        except Exception as e:
            logger.error(f"Failed to check timeouts: {e}")
            return 0
    
    async def send_approval_notification(self, request_id: str, approver_email: str):
        """Send approval notification email"""
        try:
            # Get approval request details
            request_details = await self.get_approval_request(request_id)
            if not request_details:
                return
            
            # Create email content
            subject = f"Approval Required: {request_details['title']}"
            
            body = f"""
            Dear Approver,
            
            You have a new approval request that requires your attention:
            
            Title: {request_details['title']}
            Description: {request_details['description']}
            
            Workflow Execution ID: {request_details['workflow_execution_id']}
            Created: {request_details['created_at']}
            Expires: {request_details['expires_at']}
            
            Please review and approve/reject this request through the platform interface.
            
            Request ID: {request_id}
            
            Best regards,
            AI Agent Orchestration Platform
            """
            
            # For now, just log the notification (email sending would need SMTP configuration)
            logger.info(f"Notification sent to {approver_email} for approval request {request_id}")
            logger.debug(f"Email subject: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send approval notification: {e}")
    
    async def schedule_timeout_check(self, request_id: str, timeout_seconds: int):
        """Schedule a timeout check for an approval request"""
        try:
            async def timeout_task():
                await asyncio.sleep(timeout_seconds)
                async with self.db_session() as session:
                    request = await session.get(ApprovalRequest, request_id)
                    if request and request.status == ApprovalStatus.PENDING:
                        request.status = ApprovalStatus.TIMEOUT
                        request.decided_at = datetime.now(timezone.utc)
                        await session.commit()
                        
                        await self.notify_workflow_service(request_id, False, "timeout")
                        logger.info(f"Approval request {request_id} timed out after {timeout_seconds} seconds")
            
            # Create and store the task
            task = asyncio.create_task(timeout_task())
            self.notification_tasks[request_id] = task
            
        except Exception as e:
            logger.error(f"Failed to schedule timeout check: {e}")
    
    async def notify_workflow_service(self, request_id: str, approved: bool, reason: str = ""):
        """Notify the workflow orchestration service about approval decision"""
        try:
            # This would typically make an HTTP call to the orchestration service
            # For now, we'll just log the notification
            logger.info(f"Notifying workflow service: request {request_id}, approved: {approved}, reason: {reason}")
            
            # In a real implementation, you would call the orchestration service API
            # to resume the workflow based on the approval decision
            
        except Exception as e:
            logger.error(f"Failed to notify workflow service: {e}")
    
    async def get_approval_statistics(self) -> Dict[str, Any]:
        """Get approval request statistics"""
        try:
            async with self.db_session() as session:
                # Total requests
                total_result = await session.execute(select(func.count(ApprovalRequest.id)))
                total_requests = total_result.scalar()
                
                # Requests by status
                status_result = await session.execute(
                    select(ApprovalRequest.status, func.count(ApprovalRequest.id))
                    .group_by(ApprovalRequest.status)
                )
                status_counts = {status.value: count for status, count in status_result}
                
                # Average response time (for completed requests)
                avg_response_result = await session.execute(
                    select(func.avg(
                        func.extract('epoch', ApprovalRequest.decided_at - ApprovalRequest.created_at)
                    )).where(ApprovalRequest.decided_at.is_not(None))
                )
                avg_response_time = avg_response_result.scalar()
                
                return {
                    "total_requests": total_requests,
                    "by_status": status_counts,
                    "average_response_time_seconds": float(avg_response_time) if avg_response_time else 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get approval statistics: {e}")
            return {}

# Global HITL service
hitl_service = HITLService()

# ======================== FastAPI Application ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await hitl_service.initialize()
    
    # Start background task for timeout checking
    async def periodic_timeout_check():
        while True:
            try:
                await hitl_service.check_timeouts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in periodic timeout check: {e}")
                await asyncio.sleep(60)
    
    timeout_task = asyncio.create_task(periodic_timeout_check())
    
    yield
    
    # Shutdown
    timeout_task.cancel()
    if hitl_service.db_engine:
        await hitl_service.db_engine.dispose()

app = FastAPI(
    title="AI Agent HITL Service",
    description="Human-in-the-loop approval workflow service",
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

class ApprovalRequestCreate(BaseModel):
    workflow_execution_id: str
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    data: Dict[str, Any] = Field(default_factory=dict)
    approver_email: Optional[EmailStr] = None
    timeout_seconds: int = Field(default=3600, ge=60, le=86400)
    escalation_rules: Optional[List[Dict[str, Any]]] = None

class ApprovalDecision(BaseModel):
    approved: bool
    reason: Optional[str] = Field(default=None, max_length=1000)

class ApprovalRequestResponse(BaseModel):
    id: str
    workflow_execution_id: str
    title: str
    description: str
    data: Dict[str, Any]
    status: str
    decision_reason: Optional[str]
    created_at: str
    expires_at: Optional[str]
    decided_at: Optional[str]
    timeout_seconds: int
    escalation_rules: List[Dict[str, Any]]
    approver: Optional[Dict[str, Any]]
    workflow: Optional[Dict[str, Any]]

class ApprovalListResponse(BaseModel):
    requests: List[ApprovalRequestResponse]
    total: int
    skip: int
    limit: int

# ======================== API Endpoints ========================

@app.post("/approval-requests", response_model=Dict[str, str])
async def create_approval_request(request: ApprovalRequestCreate):
    """Create a new approval request"""
    try:
        request_id = await hitl_service.create_approval_request(
            workflow_execution_id=request.workflow_execution_id,
            title=request.title,
            description=request.description,
            data=request.data,
            approver_email=request.approver_email,
            timeout_seconds=request.timeout_seconds,
            escalation_rules=request.escalation_rules
        )
        
        return {
            "request_id": request_id,
            "message": "Approval request created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/approval-requests", response_model=ApprovalListResponse)
async def list_approval_requests(
    status: Optional[ApprovalStatus] = Query(default=None),
    approver_id: Optional[str] = Query(default=None),
    workflow_execution_id: Optional[str] = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000)
):
    """List approval requests with filtering"""
    try:
        requests = await hitl_service.list_approval_requests(
            status=status,
            approver_id=approver_id,
            workflow_execution_id=workflow_execution_id,
            skip=skip,
            limit=limit
        )
        
        return ApprovalListResponse(
            requests=requests,
            total=len(requests),
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/approval-requests/{request_id}", response_model=ApprovalRequestResponse)
async def get_approval_request(request_id: str):
    """Get approval request details"""
    try:
        request_details = await hitl_service.get_approval_request(request_id)
        if not request_details:
            raise HTTPException(status_code=404, detail="Approval request not found")
        
        return ApprovalRequestResponse(**request_details)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/approval-requests/{request_id}/decide")
async def process_approval(
    request_id: str,
    decision: ApprovalDecision,
    approver_id: str = Query(..., description="ID of the user making the decision")
):
    """Process an approval decision"""
    try:
        success = await hitl_service.process_approval(
            request_id=request_id,
            approver_id=approver_id,
            approved=decision.approved,
            reason=decision.reason
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to process approval")
        
        return {
            "message": f"Approval request {'approved' if decision.approved else 'rejected'} successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/approval-requests/pending/mine")
async def get_my_pending_approvals(approver_id: str = Query(..., description="Approver user ID")):
    """Get pending approval requests for a specific approver"""
    try:
        requests = await hitl_service.list_approval_requests(
            status=ApprovalStatus.PENDING,
            approver_id=approver_id
        )
        
        return {
            "approver_id": approver_id,
            "pending_requests": requests,
            "count": len(requests)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/approval-requests/check-timeouts")
async def check_timeouts():
    """Manually trigger timeout check"""
    try:
        count = await hitl_service.check_timeouts()
        
        return {
            "message": f"Checked timeouts, {count} requests timed out",
            "timed_out_count": count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/approval-requests/statistics")
async def get_approval_statistics():
    """Get approval request statistics"""
    try:
        stats = await hitl_service.get_approval_statistics()
        
        return {
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "hitl-service",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )