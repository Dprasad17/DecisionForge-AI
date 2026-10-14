"""
DecisionForge AI — Multi-Stage Decision Orchestrator

Connects all decision components:
1. Input validation & sanitization
2. In-memory caching optimization (R5)
3. Supervised Intent Classification (Approach A)
4. Hybrid / Semantic Knowledge Retrieval (Approach B)
5. Deterministic Policy Guardrails (R4)
6. Grounded Response Generation (R3)
7. Structured Decision Audit Trace & Observability (R8)
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine.classifier import IntentClassifier
from decision_engine.retrieval import HybridRetriever, RetrievedDocument
from decision_engine.policy_engine import PolicyEngine, PolicyEvaluationResult
from decision_engine.response_generator import ResponseGenerator
from decision_engine.config import (
    MAX_CACHE_SIZE,
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
    STATUS_AUTO_RESPONSE,
    STATUS_RECOMMENDATION_ONLY,
    STATUS_HUMAN_REVIEW,
    STATUS_BLOCKED_BY_POLICY,
)


@dataclass
class EvidenceItem:
    """Document evidence record formatted for decision trace."""
    doc_id: str
    title: str
    category: str
    excerpt: str
    relevance_score: float
    source_metadata: dict


@dataclass
class DecisionTrace:
    """Detailed audit trace for observability and review."""
    request_id: str
    timestamp: str
    query: str
    predicted_category: str
    classification_confidence: float
    class_probabilities: Dict[str, float]
    retrieved_sources: List[dict]
    policy_checks_passed: List[str]
    policy_checks_blocked: List[str]
    decision_status: str
    recommended_action: str
    escalation_reason: Optional[str]
    processing_route: str
    route_rationale: str
    cache_hit: bool
    latency_ms: float


@dataclass
class DecisionResult:
    """Complete user-facing decision report."""
    request_id: str
    query: str
    category: str
    confidence: float
    status: str
    action: str
    response_text: str
    top_evidence: Optional[dict]
    all_evidence: List[dict]
    escalation_reason: Optional[str]
    route_selected: str
    latency_ms: float
    cache_hit: bool
    trace: DecisionTrace

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


class DecisionEngineOrchestrator:
    """Orchestrates the multi-stage decision pipeline."""

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[HybridRetriever] = None,
        policy_engine: Optional[PolicyEngine] = None,
        response_generator: Optional[ResponseGenerator] = None,
        enable_cache: bool = True,
    ) -> None:
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or HybridRetriever()
        self.policy_engine = policy_engine or PolicyEngine()
        self.response_generator = response_generator or ResponseGenerator()
        self.enable_cache = enable_cache

        # In-Memory Cache (LRU-like dict)
        self._cache: Dict[str, DecisionResult] = {}

    def _get_cache_key(self, text: str) -> str:
        """Generates MD5 hash for normalized query string."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def analyze_request(self, query: str, bypass_cache: bool = False) -> DecisionResult:
        """
        Executes end-to-end multi-stage decision pipeline with latency tracking.
        """
        start_time = time.perf_counter()
        req_id = f"REQ-AUDIT-{int(time.time() * 1000)}"
        timestamp = datetime.now(timezone.utc).isoformat()

        clean_query = query.strip()
        if not clean_query:
            clean_query = "Empty query provided."

        cache_key = self._get_cache_key(clean_query)

        # -------------------------------------------------------------
        # Optimization: In-Memory Cache Check
        # -------------------------------------------------------------
        if self.enable_cache and not bypass_cache and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            end_time = time.perf_counter()
            cache_latency = round((end_time - start_time) * 1000.0, 3)

            # Return copy with updated request ID and cache hit marker
            cached_trace = DecisionTrace(
                request_id=req_id,
                timestamp=timestamp,
                query=clean_query,
                predicted_category=cached_result.category,
                classification_confidence=cached_result.confidence,
                class_probabilities=cached_result.trace.class_probabilities,
                retrieved_sources=cached_result.trace.retrieved_sources,
                policy_checks_passed=cached_result.trace.policy_checks_passed,
                policy_checks_blocked=cached_result.trace.policy_checks_blocked,
                decision_status=cached_result.status,
                recommended_action=cached_result.action,
                escalation_reason=cached_result.escalation_reason,
                processing_route="OPTIMIZED_IN_MEMORY_CACHE",
                route_rationale="Normalized query matched recent safe decision in cache.",
                cache_hit=True,
                latency_ms=cache_latency,
            )

            return DecisionResult(
                request_id=req_id,
                query=clean_query,
                category=cached_result.category,
                confidence=cached_result.confidence,
                status=cached_result.status,
                action=cached_result.action,
                response_text=cached_result.response_text,
                top_evidence=cached_result.top_evidence,
                all_evidence=cached_result.all_evidence,
                escalation_reason=cached_result.escalation_reason,
                route_selected="OPTIMIZED_IN_MEMORY_CACHE",
                latency_ms=cache_latency,
                cache_hit=True,
                trace=cached_trace,
            )

        # -------------------------------------------------------------
        # Stage 0: Unified Query Representation (Shared across stages)
        # -------------------------------------------------------------
        query_emb = self.retriever.encode_query(clean_query)

        # -------------------------------------------------------------
        # Stage 1: Request Classification (Approach A)
        # -------------------------------------------------------------
        category, confidence, class_probs = self.classifier.predict(
            clean_query, embedding=query_emb
        )

        # -------------------------------------------------------------
        # Stage 2: Knowledge Retrieval (Approach B)
        # -------------------------------------------------------------
        retrieved_docs: List[RetrievedDocument] = self.retriever.retrieve_top_k(
            query=clean_query, k=3, category=category, query_embedding=query_emb, apply_threshold=True
        )

        top_evidence_doc = retrieved_docs[0] if retrieved_docs else None
        top_evidence_dict = top_evidence_doc.to_dict() if top_evidence_doc else None
        all_evidence_dicts = [d.to_dict() for d in retrieved_docs]
        retrieval_score = top_evidence_doc.score if top_evidence_doc else 0.0

        # -------------------------------------------------------------
        # Stage 3: Deterministic Policy Evaluation (R4)
        # -------------------------------------------------------------
        policy_result: PolicyEvaluationResult = self.policy_engine.evaluate(
            query=clean_query,
            predicted_category=category,
            classification_confidence=confidence,
            top_evidence=top_evidence_dict,
            retrieval_score=retrieval_score,
        )

        # -------------------------------------------------------------
        # Stage 4: Grounded Response Generation (R3)
        # -------------------------------------------------------------
        response_text = self.response_generator.generate_response(
            query=clean_query,
            predicted_category=category,
            decision_status=policy_result.decision_status,
            recommended_action=policy_result.recommended_action,
            top_evidence=top_evidence_dict,
            escalation_reason=policy_result.escalation_reason,
        )

        # -------------------------------------------------------------
        # Stage 5: Routing & Latency Optimization Trace (R5 & R8)
        # -------------------------------------------------------------
        if policy_result.is_blocked:
            route_selected = "POLICY_GUARDRAIL_OVERRIDE"
            route_rationale = "Deterministic security policy blocked action regardless of model confidence."
        elif policy_result.decision_status == STATUS_HUMAN_REVIEW:
            route_selected = "HUMAN_ESCALATION_PATH"
            route_rationale = (
                f"Escalated due to: {policy_result.escalation_reason or 'Risk/ambiguity policy'}."
            )
        elif policy_result.decision_status == STATUS_AUTO_RESPONSE:
            route_selected = "OPTIMIZED_GROUNDED_LOCAL_ROUTE"
            route_rationale = "High confidence & sufficient evidence allowed zero-latency grounded execution."
        else:
            route_selected = "RECOMMENDATION_ONLY_ROUTE"
            route_rationale = "Moderate confidence provided informational guidance without execution."

        end_time = time.perf_counter()
        total_latency_ms = round((end_time - start_time) * 1000.0, 3)

        passed_rules = [r for r in policy_result.applied_rules if not r.startswith("POL-001") and not r.startswith("POL-002")]
        blocked_rules = [r for r in policy_result.applied_rules if r.startswith("POL-001") or r.startswith("POL-002")]

        trace = DecisionTrace(
            request_id=req_id,
            timestamp=timestamp,
            query=clean_query,
            predicted_category=category,
            classification_confidence=confidence,
            class_probabilities=class_probs,
            retrieved_sources=all_evidence_dicts,
            policy_checks_passed=passed_rules,
            policy_checks_blocked=blocked_rules,
            decision_status=policy_result.decision_status,
            recommended_action=policy_result.recommended_action,
            escalation_reason=policy_result.escalation_reason,
            processing_route=route_selected,
            route_rationale=route_rationale,
            cache_hit=False,
            latency_ms=total_latency_ms,
        )

        final_result = DecisionResult(
            request_id=req_id,
            query=clean_query,
            category=category,
            confidence=confidence,
            status=policy_result.decision_status,
            action=policy_result.recommended_action,
            response_text=response_text,
            top_evidence=top_evidence_dict,
            all_evidence=all_evidence_dicts,
            escalation_reason=policy_result.escalation_reason,
            route_selected=route_selected,
            latency_ms=total_latency_ms,
            cache_hit=False,
            trace=trace,
        )

        # Store in in-memory cache if safe (do not cache blocked or transient ambiguous inputs)
        if self.enable_cache and not policy_result.is_blocked:
            if len(self._cache) >= MAX_CACHE_SIZE:
                # Evict oldest entry
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = final_result

        return final_result


if __name__ == "__main__":
    orchestrator = DecisionEngineOrchestrator()
    sample = "I forgot my password and cannot sign in to my account."
    res = orchestrator.analyze_request(sample)
    print(f"Result for: '{sample}'")
    print(f"Status:   {res.status}")
    print(f"Category: {res.category} (Conf: {res.confidence})")
    print(f"Route:    {res.route_selected}")
    print(f"Latency:  {res.latency_ms} ms")
    print(f"Response: {res.response_text}")
