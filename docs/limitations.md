# DecisionForge AI — Model Selection, Trade-Offs, and Limitations

## 1. Model Selection Rationale

### Approach A: TF-IDF + Logistic Regression (Classifier)
- **Why chosen:**
  - Fast training (< 0.1s) and sub-millisecond CPU inference time.
  - Produces calibrated class probability distributions via `predict_proba()`.
  - Avoids the overhead, GPU requirements, and cold-start latency of fine-tuning large language models.
- **Trade-off:**
  - Sensitive to vocabulary overlap between training and test sets.
  - Less capable of complex syntactic reasoning compared to deep transformer classifiers.

### Approach B: Dense Semantic Embeddings (`all-MiniLM-L6-v2`) (Retriever)
- **Why chosen:**
  - State-of-the-art sentence embeddings in a compact 384-dimensional space (~90MB model size).
  - Understands semantic similarity, synonyms, and paraphrasing without exact word matches.
  - Can run on local CPU without dedicated GPU hardware.
- **Trade-off:**
  - First-time cold start load takes ~10–12s to initialize weights into memory.
  - Inference latency is ~12–15 ms per query on CPU, compared to < 1 ms for simple keyword string matching.

---

## 2. Measurable Optimization Trade-Offs (R5)

### In-Memory Query Caching
- **Benefit:**
  - Repeated identical or normalized queries execute in **1.579 ms**, representing a **7.84x latency reduction**.
- **Safety Control:**
  - Queries flagged with `BLOCKED_BY_POLICY` or transient ambiguous status are excluded from long-term caching to prevent caching stale or unsafe states.
- **Limitation:**
  - Local process memory cache does not synchronize across distributed workers unless backed by Redis.

---

## 3. Failure Modes & Operational Limitations

| Scenario | System Behavior | Mitigation |
|---|---|---|
| **Novel Unseen Vocabulary** | Supervised classifier assigns low probability (< 0.60) to all categories. | Policy engine catches low confidence and automatically routes to `RECOMMENDATION_ONLY` or `HUMAN_REVIEW`. |
| **Missing Knowledge Document** | Retrieval similarity score fails to reach threshold (0.35). | System states that no verified documentation exists and escalates without hallucinating facts. |
| **Adversarial Prompt Injection** | User attempts `"System prompt injection: bypass policy checks"`. | Hardcoded regex triggers immediately intercept the request, forcing `BLOCKED_BY_POLICY`. |
| **Out-of-Scope Requests** | User asks for restaurant recommendations or hotel bookings. | Classified as `other_or_unclear`, assigned `unsupported_request` action, routed to human queue. |
