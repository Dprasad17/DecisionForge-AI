"""
DecisionForge AI — Backend Pydantic Schemas
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    """Incoming user inquiry for decision analysis."""
    text: str = Field(..., min_length=1, max_length=2000, description="Customer inquiry or business ticket text")
    bypass_cache: bool = Field(default=False, description="Whether to bypass in-memory caching")


class EvidenceItemSchema(BaseModel):
    """Retrieved knowledge base document evidence."""
    doc_id: str
    title: str
    category: str
    excerpt: str
    relevance_score: float
    source_metadata: Dict = Field(default_factory=dict)


class DecisionTraceSchema(BaseModel):
    """Observability trace detailing multi-stage decision rationale."""
    request_id: str
    timestamp: str
    query: str
    predicted_category: str
    classification_confidence: float
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    retrieved_sources: List[Dict] = Field(default_factory=list)
    policy_checks_passed: List[str] = Field(default_factory=list)
    policy_checks_blocked: List[str] = Field(default_factory=list)
    decision_status: str
    recommended_action: str
    escalation_reason: Optional[str] = None
    processing_route: str
    route_rationale: str
    cache_hit: bool
    latency_ms: float


class DecisionResponse(BaseModel):
    """Complete decision report returned to client."""
    request_id: str
    query: str
    category: str
    confidence: float
    status: str
    action: str
    response_text: str
    top_evidence: Optional[Dict] = None
    all_evidence: List[Dict] = Field(default_factory=list)
    escalation_reason: Optional[str] = None
    route_selected: str
    latency_ms: float
    cache_hit: bool
    trace: DecisionTraceSchema


class HealthResponse(BaseModel):
    """Health check status response."""
    status: str
    service: str
    version: str
    timestamp: str
    models_loaded: bool
    cache_enabled: bool


class SampleQuery(BaseModel):
    """Preset sample inquiry for reviewer demonstration."""
    id: str
    title: str
    text: str
    category: str
    scenario_type: str
