"""
Unit tests for DecisionForge AI Phase 1 synthetic data validation.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.validate_data import (
    validate_knowledge_base,
    validate_synthetic_requests,
    run_validation,
    MINIMUM_REQUEST_COUNT,
)


def test_knowledge_base_integrity():
    """Verify that knowledge_base.json exists, parses, and has valid document structure."""
    kb_path = PROJECT_ROOT / "data" / "knowledge_base.json"
    errors, valid_doc_ids = validate_knowledge_base(kb_path)
    assert not errors, f"Knowledge base validation errors: {errors}"
    assert len(valid_doc_ids) >= 10, f"Expected at least 10 documents, got {len(valid_doc_ids)}"


def test_synthetic_requests_integrity_and_leakage():
    """Verify synthetic requests schema, constraints, and absence of cross-split leakage."""
    kb_path = PROJECT_ROOT / "data" / "knowledge_base.json"
    csv_path = PROJECT_ROOT / "data" / "synthetic_requests.csv"

    _, valid_doc_ids = validate_knowledge_base(kb_path)
    errors, cat_counts, split_counts = validate_synthetic_requests(csv_path, valid_doc_ids)

    assert not errors, f"Synthetic requests errors: {errors}"
    total_count = sum(split_counts.values())
    assert total_count >= MINIMUM_REQUEST_COUNT, (
        f"Expected at least {MINIMUM_REQUEST_COUNT} requests, got {total_count}"
    )

    # All splits must be present
    for s in ["train", "val", "test"]:
        assert split_counts[s] > 0, f"Split '{s}' must have at least 1 example"

    # All categories must have at least 1 example
    for cat, count in cat_counts.items():
        assert count > 0, f"Category '{cat}' must have at least 1 example"


def test_full_pipeline_validation():
    """Verify that the end-to-end data validation runner succeeds."""
    assert run_validation(PROJECT_ROOT) is True
