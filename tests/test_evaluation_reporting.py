"""
Unit tests for DecisionForge AI Evaluation Report Generation Logic.
"""

from pathlib import Path
import tempfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import _generate_markdown_report


def test_markdown_report_generation_negative_and_positive_deltas():
    """Verify that negative deltas are labeled as decreases, not gains/improvements."""
    mock_summary = {
        "metadata": {
            "evaluation_timestamp": "2026-10-14T12:00:00+00:00",
            "test_set_size": 32,
            "synthetic_dataset": "data/synthetic_requests.csv (test split)",
        },
        "baseline_system": {
            "classification": {
                "accuracy": 0.6875,
                "macro_f1": 0.6860,
                "macro_precision": 0.8089,
                "macro_recall": 0.6944,
            },
            "retrieval": {
                "recall_at_1": 0.9062,
            },
            "latency": {
                "mean_ms": 0.077,
                "median_ms": 0.067,
            },
            "safety_enforcement_rate": 0.6667,
        },
        "improved_system": {
            "classification": {
                "accuracy": 0.6250,
                "macro_f1": 0.5805,
                "macro_precision": 0.6970,
                "macro_recall": 0.5741,
            },
            "retrieval": {
                "recall_at_1": 0.7500,
                "recall_at_3": 0.7500,
            },
            "latency_uncached": {
                "mean_ms": 15.558,
                "median_ms": 15.527,
            },
            "latency_cached": {
                "mean_ms": 1.954,
            },
            "safety_enforcement_rate": 0.8333,
        },
        "comparison_delta": {
            "accuracy_delta": -0.0625,
            "macro_f1_delta": -0.1055,
            "recall_1_delta": -0.1562,
            "safety_delta": 0.1666,
            "cache_speedup_factor": 7.96,
        },
    }

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False, suffix=".md") as tmp:
        tmp_path = Path(tmp.name)

    try:
        _generate_markdown_report(tmp_path, mock_summary)
        content = tmp_path.read_text(encoding="utf-8")

        # 1. Assert negative deltas are described as decreases, never gains/improved
        assert "Classification Gain" not in content
        assert "Retrieval Gain" not in content
        assert "-0.0625 decrease" in content
        assert "-0.1055 decrease" in content
        assert "-0.1562 decrease" in content
        assert "**-0.0625** (decrease)" in content

        # 2. Assert safety delta is positive improvement
        assert "+16.7% improvement" in content
        assert "**+16.7%** (improvement)" in content
        assert "66.7%" in content
        assert "83.3%" in content

        # 3. Assert caching speedup is separated from model quality
        assert "**7.96x speedup** (operational optimization)" in content
        assert "operational latency reduction via caching, distinct from model classification accuracy" in content

    finally:
        if tmp_path.exists():
            tmp_path.unlink()
