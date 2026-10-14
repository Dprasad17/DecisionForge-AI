"""
DecisionForge AI — Deterministic Policy & Guardrails Engine

Implements strict, deterministic safety checks and routing rules that cannot
be bypassed by model confidence or untrusted user text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from decision_engine.config import (
    STATUS_AUTO_RESPONSE,
    STATUS_RECOMMENDATION_ONLY,
    STATUS_HUMAN_REVIEW,
    STATUS_BLOCKED_BY_POLICY,
    ACTION_PROVIDE_INFORMATION,
    ACTION_REQUEST_CLARIFICATION,
    ACTION_RECOMMEND_TROUBLESHOOTING,
    ACTION_RECOMMEND_HUMAN_REVIEW,
    ACTION_RESTRICTED_FINANCIAL,
    ACTION_UNSUPPORTED,
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
)

# Adversarial & Prompt Injection Triggers
PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous\s+)?(rules|instructions|safety|guidelines)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(policy|checks|supervision|supervisor|verification|2fa)", re.IGNORECASE),
    re.compile(r"skip\s+(all\s+)?(policy|checks|supervision|verification|rules)", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+(injection|override)", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?(security|rules|lockdown|policy)", re.IGNORECASE),
    re.compile(r"disburse\s+\$?\d+", re.IGNORECASE),
    re.compile(r"transfer\s+\$?\d+", re.IGNORECASE),
    re.compile(r"wire\s+transfer\s+to", re.IGNORECASE),
    re.compile(r"root\s+shell", re.IGNORECASE),
]

# Prohibited / Restricted Financial Action Keywords
RESTRICTED_ACTION_KEYWORDS: List[str] = [
    "disburse",
    "wire transfer",
    "reverse transaction",
    "bypass 2fa",
    "instant payout",
    "force approval",
    "delete all",
    "root shell",
    "admin privileges",
    "chargeback",
    "skip all",
    "company reserves",
]

# Vague / Ambiguous terms
VAGUE_QUERY_TERMS: Set[str] = {
    "help", "help.", "broken", "issue", "problem", "stuff", "assist", "need help", "urgently broken", "need assistance"
}


@dataclass
class PolicyEvaluationResult:
    """Outcome of deterministic policy evaluation."""
    decision_status: str
    recommended_action: str
    applied_rules: List[str]
    is_blocked: bool
    escalation_reason: Optional[str] = None


class PolicyEngine:
    """Deterministic policy engine enforcing hard operational constraints."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        evidence_threshold: float = DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.evidence_threshold = evidence_threshold

    def evaluate(
        self,
        query: str,
        predicted_category: str,
        classification_confidence: float,
        top_evidence: Optional[dict] = None,
        retrieval_score: float = 0.0,
    ) -> PolicyEvaluationResult:
        """
        Evaluates input query, classification results, and retrieval context
        against deterministic policy rules.
        """
        clean_query = query.strip()
        query_lower = clean_query.lower()
        applied_rules: List[str] = []

        # ---------------------------------------------------------
        # Rule 1: Prompt Injection & Adversarial Bypass Check
        # ---------------------------------------------------------
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(query_lower):
                applied_rules.append(f"POL-001_INJECTION_DETECTED: Matched pattern '{pattern.pattern}'")
                return PolicyEvaluationResult(
                    decision_status=STATUS_BLOCKED_BY_POLICY,
                    recommended_action=ACTION_RESTRICTED_FINANCIAL,
                    applied_rules=applied_rules,
                    is_blocked=True,
                    escalation_reason="Adversarial prompt injection or security policy override attempt detected.",
                )

        # ---------------------------------------------------------
        # Rule 2: Restricted Financial & Elevated Security Actions
        # ---------------------------------------------------------
        for kw in RESTRICTED_ACTION_KEYWORDS:
            if kw in query_lower:
                applied_rules.append(f"POL-002_RESTRICTED_ACTION: Matched sensitive operation '{kw}'")
                return PolicyEvaluationResult(
                    decision_status=STATUS_BLOCKED_BY_POLICY,
                    recommended_action=ACTION_RESTRICTED_FINANCIAL,
                    applied_rules=applied_rules,
                    is_blocked=True,
                    escalation_reason="Request demands an unauthorized transaction or security bypass prohibited in demo.",
                )

        # ---------------------------------------------------------
        # Rule 3: Ambiguous or Underspecified Input Check
        # ---------------------------------------------------------
        if len(clean_query) < 10 or query_lower in VAGUE_QUERY_TERMS or predicted_category == "other_or_unclear":
            if retrieval_score < self.evidence_threshold:
                applied_rules.append("POL-003_AMBIGUOUS_INPUT: Query is underspecified or lacking domain context.")
                return PolicyEvaluationResult(
                    decision_status=STATUS_HUMAN_REVIEW,
                    recommended_action=ACTION_REQUEST_CLARIFICATION,
                    applied_rules=applied_rules,
                    is_blocked=False,
                    escalation_reason="Query is ambiguous or underspecified; requesting clarification.",
                )

        # ---------------------------------------------------------
        # Rule 4: Complaint & Escalation Policy
        # ---------------------------------------------------------
        if predicted_category == "complaint":
            applied_rules.append("POL-004_CUSTOMER_COMPLAINT: High-urgency customer sentiment requires human manager.")
            return PolicyEvaluationResult(
                decision_status=STATUS_HUMAN_REVIEW,
                recommended_action=ACTION_RECOMMEND_HUMAN_REVIEW,
                applied_rules=applied_rules,
                is_blocked=False,
                escalation_reason="Customer complaint escalated to support management.",
            )

        # ---------------------------------------------------------
        # Rule 5: Evidence Sufficiency Check
        # ---------------------------------------------------------
        has_sufficient_evidence = (
            top_evidence is not None and retrieval_score >= self.evidence_threshold
        )

        if not has_sufficient_evidence:
            applied_rules.append(
                f"POL-005_INSUFFICIENT_EVIDENCE: Top score {retrieval_score:.3f} < threshold {self.evidence_threshold}"
            )
            action = (
                ACTION_UNSUPPORTED
                if predicted_category == "other_or_unclear"
                else ACTION_RECOMMEND_HUMAN_REVIEW
            )
            return PolicyEvaluationResult(
                decision_status=STATUS_HUMAN_REVIEW,
                recommended_action=action,
                applied_rules=applied_rules,
                is_blocked=False,
                escalation_reason="No verified knowledge base document satisfies minimum relevance threshold.",
            )

        # ---------------------------------------------------------
        # Rule 6: Confidence Threshold Check
        # ---------------------------------------------------------
        if classification_confidence < self.confidence_threshold:
            applied_rules.append(
                f"POL-006_LOW_CONFIDENCE: Category confidence {classification_confidence:.2f} < threshold {self.confidence_threshold}"
            )
            return PolicyEvaluationResult(
                decision_status=STATUS_RECOMMENDATION_ONLY,
                recommended_action=ACTION_PROVIDE_INFORMATION,
                applied_rules=applied_rules,
                is_blocked=False,
                escalation_reason="Moderate classification confidence suggests recommendation-only routing.",
            )

        # ---------------------------------------------------------
        # Rule 7: Safe Standard Grounded Path
        # ---------------------------------------------------------
        action = (
            ACTION_RECOMMEND_TROUBLESHOOTING
            if predicted_category == "technical_problem"
            else ACTION_PROVIDE_INFORMATION
        )
        applied_rules.append(
            f"POL-007_SAFE_AUTOMATION: High confidence ({classification_confidence:.2f}) and verified evidence ({retrieval_score:.2f})."
        )
        return PolicyEvaluationResult(
            decision_status=STATUS_AUTO_RESPONSE,
            recommended_action=action,
            applied_rules=applied_rules,
            is_blocked=False,
            escalation_reason=None,
        )
