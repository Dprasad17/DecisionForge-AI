"""
DecisionForge AI — Evaluation Metrics Calculation

Calculates classification, retrieval, safety, and latency metrics:
- Accuracy & Macro-F1
- Retrieval Recall@1 and Recall@3
- Mean and Median Latency
- Safety Enforcement Rate
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Computes multiclass accuracy, macro-F1, macro-precision, and macro-recall."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def compute_retrieval_recall_at_k(
    expected_doc_ids: List[str], retrieved_doc_ids_per_query: List[List[str]], k: int = 1
) -> float:
    """
    Computes Recall@k over queries with a non-empty expected document ID.
    Queries where no evidence is expected are evaluated based on empty retrieval.
    """
    hits = 0
    total = 0

    for expected, retrieved in zip(expected_doc_ids, retrieved_doc_ids_per_query):
        exp_clean = (expected or "").strip()
        top_k = retrieved[:k]

        if exp_clean:
            total += 1
            if exp_clean in top_k:
                hits += 1
        else:
            # When no doc expected, correct if top result is empty or None
            total += 1
            if not top_k or top_k[0] in {"", "None", None}:
                hits += 1

    return round(float(hits / total), 4) if total > 0 else 0.0


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes mean, median, min, max, and 95th percentile latency in milliseconds."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }
