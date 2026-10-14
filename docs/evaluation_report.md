# DecisionForge AI — Evaluation & Benchmark Report

## 1. Executive Summary
- **Test Set Size:** 32 held-out synthetic customer requests.
- **Timestamp:** `2026-10-14T14:18:03.649042+00:00`
- **Dataset:** `data/synthetic_requests.csv (test split)`
- **Classification Performance:** Intent accuracy changed from `0.6875` (baseline) to `0.6250` (improved) (-0.0625 decrease), with Macro-F1 changing from `0.6860` to `0.5805` (-0.1055 decrease). Hand-crafted keyword heuristics scored higher on this small synthetic dataset due to keyword alignment in test queries and training sample sparsity (34 training instances across 9 classes).
- **Retrieval Performance:** Retrieval Recall@1 changed from `0.9062` (baseline canonical Jaccard) to `0.7500` (improved hybrid) (-0.1562 decrease), while improved Recall@3 reached `0.7500`.
- **Safety & Guardrails:** Safety enforcement improved from `66.7%` (baseline: 4/6) to `83.3%` (improved: 5/6) (+16.7% improvement), demonstrating stronger deterministic policy protection against prompt injection and restricted transactions.
- **Latency & Caching Optimization:** The in-memory cache provides an operational **8.36x latency speedup** on repeated requests (1.977 ms vs 16.519 ms). Note: This speedup represents an operational latency reduction via caching, distinct from model classification accuracy.

---

## 2. Comparative Performance Table

| Metric | Baseline System | Improved Engine | Delta / Result |
|---|---|---|---|
| **Intent Accuracy** | 0.6875 | 0.6250 | **-0.0625** (decrease) |
| **Macro-F1 Score** | 0.6860 | 0.5805 | **-0.1055** (decrease) |
| **Macro-Precision** | 0.8089 | 0.6970 | -0.1119 |
| **Macro-Recall** | 0.6944 | 0.5741 | -0.1203 |
| **Retrieval Recall@1** | 0.9062 | 0.7500 | **-0.1562** (decrease) |
| **Retrieval Recall@3** | N/A (top-1) | 0.7500 | Top-3 candidate coverage |
| **Safety Enforcement** | 66.7% | 83.3% | **+16.7%** (improvement) |
| **Mean Latency (Uncached)** | 0.093 ms | 16.519 ms | Dense embedding compute trade-off |
| **Median Latency (Uncached)** | 0.084 ms | 15.707 ms | Local CPU neural representation |
| **Mean Latency (Cached Route)** | N/A | 1.977 ms | **8.36x speedup** (operational optimization) |

---

## 3. Analysis & Key Trade-offs

1. **Classification Trade-off (Supervised vs. Keyword Rules):**
   The baseline keyword classifier achieved higher accuracy (0.6875 vs 0.6250) on this synthetic test set because synthetic test prompts contain distinct lexical keywords directly targeted by baseline dictionary heuristics. The supervised TF-IDF + Logistic Regression model was trained on only 34 synthetic training examples across 9 classes (~3–4 samples per class), creating sample sparsity. However, the supervised classifier outputs continuous probability distributions across all classes, enabling principled confidence-based escalation thresholds rather than hardcoded step functions.

2. **Retrieval Trade-off (Lexical Jaccard vs. Dense Hybrid):**
   Baseline canonical Jaccard scored higher Recall@1 (0.9062 vs 0.7500) on short synthetic queries because queries shared exact tokens with knowledge document titles. For concise queries, dense embeddings (`all-MiniLM-L6-v2`) map to semantically related but adjacent documents (e.g., retrieving general account access instead of password reset), though Recall@3 reaches 0.7500. The hybrid model provides tolerance to natural paraphrasing and synonyms that strict token overlap misses.

3. **Deterministic Safety Enforcement (+16.7% Improvement):**
   The improved decision engine increased restricted action and prompt-injection enforcement from 66.7% (4/6) to 83.3% (5/6). Dedicated policy rules intercept adversarial instructions and unauthorized financial requests before response generation, independent of classifier confidence.

4. **Operational Optimization vs. Model Quality:**
   The in-memory SHA-256 query cache delivers a **8.36x latency speedup** on repeated requests, reducing response time from ~15 ms to ~1.9 ms. This is an operational efficiency optimization that reduces compute and latency without altering model quality or safety guarantees.
