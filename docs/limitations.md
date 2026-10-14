# DecisionForge AI — Model Selection, Trade-Offs, and Limitations

## 1. Model Selection Rationale

### Approach A: Dense Linear Probe & TF-IDF Fallback (Classifier)
- **Why chosen:**
  - Primary mode uses a supervised linear probe (`LogisticRegression(C=5.0)`) on frozen 384-dimensional dense sentence embeddings (`all-MiniLM-L6-v2`), sharing the query embedding computed for retrieval.
  - Sub-millisecond CPU classification overhead since dense embeddings are computed once in Stage 0.
  - TF-IDF N-gram pipeline is retained as an instant fallback mechanism.
  - Produces multi-class probability distributions via `predict_proba()` (softmax outputs over linear log-odds).
- **Trade-off & Calibration Limitation:**
  - **No Explicit Probability Calibration:** The model outputs raw softmax probabilities from standard `LogisticRegression(C=5.0)`. It is not post-processed using explicit calibration methods (such as `CalibratedClassifierCV` with Platt scaling or isotonic regression). On small datasets ($N=46$ non-test training samples: 34 train + 12 validation), raw softmax outputs can be overconfident. Confidence scores serve as operational ranking signals for policy thresholding rather than true statistical probabilities.
  - Sensitive to domain shifts outside the 9-class customer support taxonomy.

### Approach B: Dense Semantic Embeddings (`all-MiniLM-L6-v2`) (Retriever)
- **Why chosen:**
  - State-of-the-art sentence embeddings in a compact 384-dimensional space (~90MB model size).
  - Understands semantic similarity, synonyms, and paraphrasing without exact word matches.
  - Can run on local CPU without dedicated GPU hardware.
- **Trade-off:**
  - First-time cold start load takes ~10–12s to initialize weights into memory.
  - Inference latency is ~15–20 ms per query on CPU, compared to < 1 ms for simple keyword string matching.

---

## 2. Measurable Optimization Trade-Offs (R5)

### In-Memory Query Caching
- **Benefit:**
  - Repeated identical or normalized queries execute in **4.419 ms mean latency** (with median responses under **0.05 ms**), representing a **4.77x mean latency reduction**.
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
