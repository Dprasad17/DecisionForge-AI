# DecisionForge AI — Baseline System Report (Phase 2)

## 1. Overview
The Phase 2 Baseline System implements a deliberately simple, fast, and deterministic decision engine to establish performance benchmarks for DecisionForge AI. It provides the foundation against which optimized supervised models (Phase 3) and advanced routing (Phases 4–6) are measured.

## 2. Baseline Architecture

### 2.1 Intent Classification
- **Method:** Heuristic keyword and token dictionary matching against the 9 predefined taxonomy categories.
- **Confidence Scoring:**
  - $\ge 2$ keyword hits: `0.85`
  - 1 keyword hit: `0.65`
  - 0 hits (fallback to `other_or_unclear`): `0.20`

### 2.2 Knowledge Retrieval
- **Corpus:** 12 synthetic policy and operational documents in `data/knowledge_base.json`.
- **Method:** Top-1 canonical Jaccard token overlap between query tokens and combined document title + content tokens.
- **Formula:**
  $$J(Q, D) = \frac{|Q \cap D|}{|Q \cup D|}$$
  where $Q$ and $D$ are alphanumeric token sets filtered for English stopwords.
- **Multi-Token Guard:** Requires at least 2 distinct intersecting content tokens ($|Q \cap D| \ge 2$) to eliminate single-word false positive matches.
- **Evidence Sufficiency Threshold:** A minimum overlap threshold of `J(Q, D) >= 0.03` is enforced. Queries failing to achieve this threshold trigger explicit no-evidence handling.

### 2.3 Policy Rules & Safety Guardrails
- **Restricted Action Overrides:** Hardcoded substring matching for sensitive or adversarial terms (`bypass`, `skip all`, `disburse`, `wire transfer`, `prompt injection`, `override`).
- **Deterministic Action Routing:**
  - Blocked keywords $\rightarrow$ `BLOCKED_BY_POLICY` / `restricted_financial_action`
  - Ambiguous / short input $\rightarrow$ `HUMAN_REVIEW` / `request_clarification`
  - Insufficient evidence $\rightarrow$ `HUMAN_REVIEW` / `recommend_human_review`
  - Grounded evidence found $\rightarrow$ `AUTO_RESPONSE` / `provide_information`

### 2.4 Response Strategy
- Deterministic response templates citing the document title, document ID, and an extracted relevant excerpt.
- Zero hallucination risk: never invents policies not present in the knowledge base.

---

## 3. Latency Measurement Methodology
- **Timer:** High-resolution monotonic clock (`time.perf_counter()`).
- **Scope:** Complete end-to-end processing pipeline, including string sanitization, intent classification, knowledge base scanning, policy evaluation, response formatting, and result object construction.
- **Unit:** Milliseconds (ms).

---

## 4. Known Baseline Limitations

| Limitation | Impact | Planned Resolution in Phase 3+ |
|---|---|---|
| **Lack of Semantic Generalization** | Fails to recognize synonyms or paraphrasing (e.g., "reimbursement" if not in the keyword list). | Supervised TF-IDF vectorizer + Logistic Regression classifier. |
| **Brittle Keyword Retrieval** | Jaccard overlap favors documents with common vocabulary rather than conceptual relevance. | Dense semantic embeddings (`sentence-transformers`) with cosine similarity. |
| **Coarse Confidence Heuristic** | Confidence is stepped (`0.85`, `0.65`, `0.20`) rather than a continuous probability distribution. | Class probability distributions from logistic regression (`predict_proba`). |
| **Rigid Template Responses** | Responses follow fixed phrasing and cannot adapt tone to nuanced customer inquiries. | Controlled grounded generation with explicit evidence constraints. |
