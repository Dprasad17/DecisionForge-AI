# DecisionForge AI — Architecture Specification

## 1. System Overview
DecisionForge AI is an intelligent, multi-stage business request decision and routing engine. It acts as a controlled decision-support system for customer support tickets and operational business inquiries.

## 2. Component Pipeline

```text
Input Request (Text)
       │
       ▼
[Stage 1] Input Validation & Sanitization
       │
       ▼
[Stage 2] Supervised Classifier (TF-IDF + Logistic Regression)
       ├── Category (e.g. payment_issue, refund_request)
       └── Classification Confidence Score [0.0, 1.0]
       │
       ▼
[Stage 3] Knowledge Retrieval Engine
       ├── Semantic / Hybrid Matching against knowledge_base.json
       └── Evidence Extraction (Doc Title, Excerpt, Similarity Score)
       │
       ▼
[Stage 4] Deterministic Policy & Safety Engine
       ├── Confidence Threshold Check (default: 0.65)
       ├── Evidence Sufficiency Check (default: 0.40)
       ├── Prohibited / Restricted Action Check (e.g. forced refunds, 2FA bypass)
       └── Adversarial Prompt Injection Guardrails
       │
       ▼
[Stage 5] Routing & Response Strategy
       ├── AUTO_RESPONSE: Safe grounded response using retrieved evidence
       ├── RECOMMENDATION_ONLY: Suggested operational path without mutation
       ├── HUMAN_REVIEW: Low-confidence, complex, or complaint escalation
       └── BLOCKED_BY_POLICY: Prohibited financial action or prompt injection
       │
       ▼
[Stage 6] Structured Audit & Observability Trace
       └── Formatted JSON record with latency, route chosen, reasons
```

## 3. Autonomous Action Guardrails
- **Simulated Only:** The engine never interacts with payment gateways, banking APIs, or database mutation endpoints.
- **Deterministic Override:** Even with 100% classification confidence, requests that attempt restricted operations or policy bypasses are immediately routed to `BLOCKED_BY_POLICY` or `HUMAN_REVIEW`.
