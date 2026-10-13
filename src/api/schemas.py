"""
Pydantic Schemas for FastAPI Endpoints
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    active_model_version: str
    uptime_seconds: float


class PredictionRequest(BaseModel):
    complaint_text: str = Field(..., min_length=5, description="Customer complaint text")


class PredictionResponse(BaseModel):
    predicted_category: str
    category_confidence: float
    predicted_sentiment: str
    sentiment_confidence: float
    predicted_urgency: str
    urgency_confidence: float
    predicted_action: str
    action_confidence: float
    category_explanation: Dict[str, Any]
    model_version: str


class FullAnalysisRequest(BaseModel):
    complaint_text: str = Field(..., min_length=5, description="Customer complaint narrative")
    complaint_id: Optional[str] = None
    customer_id: Optional[str] = "CUST-ANON"
    customer_name: Optional[str] = "Customer"
    customer_tier: Optional[str] = "Standard"
    channel: Optional[str] = "Web Portal"
    disputed_amount: Optional[float] = 0.0


class FullAnalysisResponse(BaseModel):
    trace_id: str
    complaint_id: str
    validation_status: str
    is_valid: bool
    redacted_text: Optional[str] = None
    pii_detected: bool = False
    predicted_category: Optional[str] = None
    category_confidence: Optional[float] = None
    predicted_sentiment: Optional[str] = None
    predicted_urgency: Optional[str] = None
    priority_score: Optional[int] = None
    priority_tier: Optional[str] = None
    target_sla_hours: Optional[int] = None
    target_department: Optional[str] = None
    legal_risk_flag: bool = False
    predicted_action: Optional[str] = None
    recommended_sop: Optional[List[str]] = None
    draft_customer_response: Optional[str] = None
    compensation_recommendation: Optional[str] = None
    compliance_notice: Optional[str] = None
    total_pipeline_latency_ms: Optional[float] = None
    execution_traces: List[Dict[str, Any]] = []


class BatchAnalysisRequest(BaseModel):
    complaints: List[FullAnalysisRequest]


class BatchAnalysisResponse(BaseModel):
    total_processed: int
    results: List[FullAnalysisResponse]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or question")


class ChatResponse(BaseModel):
    response: str
    history: List[Dict[str, str]]


class HumanFeedbackRequest(BaseModel):
    complaint_id: str
    complaint_text: str
    predicted_category: str
    corrected_category: str
    predicted_urgency: str
    corrected_urgency: str
    reviewer_notes: Optional[str] = ""


class RetrainRequest(BaseModel):
    new_version: Optional[str] = None
    incorporate_feedback: bool = True
