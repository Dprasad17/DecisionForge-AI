"""
DecisionForge AI — Data Validation and Split Leakage Detection

Validates data/knowledge_base.json and data/synthetic_requests.csv against:
- Schema and required fields
- JSON and CSV formatting
- Primary key uniqueness
- Intent category and action enum validity
- Document reference integrity
- Train/Validation/Test split presence
- Text duplicates and cross-split data leakage
- Minimum dataset size (>= 35 requests)

Exits with code 0 on success, or code 1 with error details on failure.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Allowed Taxonomies
ALLOWED_INTENT_CATEGORIES: Set[str] = {
    "account_access",
    "payment_issue",
    "refund_request",
    "subscription_change",
    "technical_problem",
    "order_or_service_status",
    "policy_question",
    "complaint",
    "other_or_unclear",
}

ALLOWED_ACTION_TYPES: Set[str] = {
    "provide_information",
    "request_clarification",
    "recommend_troubleshooting",
    "recommend_human_review",
    "restricted_financial_action",
    "unsupported_request",
}

ALLOWED_SPLITS: Set[str] = {"train", "val", "test"}

MINIMUM_REQUEST_COUNT: int = 35

REQUIRED_KB_FIELDS: Set[str] = {
    "id",
    "title",
    "category",
    "content",
    "source_metadata",
}

REQUIRED_CSV_FIELDS: Set[str] = {
    "request_id",
    "text",
    "intent_category",
    "expected_action",
    "expected_doc_id",
    "is_restricted",
    "is_ambiguous",
    "split",
}


def parse_boolean(val: str) -> bool:
    """Parses a string boolean value, raising ValueError if invalid."""
    normalized = val.strip().lower()
    if normalized in {"true", "1", "t", "yes"}:
        return True
    if normalized in {"false", "0", "f", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: '{val}'")


def validate_knowledge_base(kb_path: Path) -> Tuple[List[str], Set[str]]:
    """Validates data/knowledge_base.json and returns (errors, valid_doc_ids)."""
    errors: List[str] = []
    doc_ids: Set[str] = set()

    if not kb_path.exists():
        errors.append(f"Knowledge base file not found at: {kb_path}")
        return errors, doc_ids

    try:
        with kb_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        errors.append(f"Failed to parse {kb_path} as JSON: {exc}")
        return errors, doc_ids

    if not isinstance(data, list):
        errors.append(f"Knowledge base root must be a JSON array, got {type(data).__name__}")
        return errors, doc_ids

    if len(data) == 0:
        errors.append("Knowledge base is empty; at least 1 document is required.")
        return errors, doc_ids

    for idx, doc in enumerate(data):
        if not isinstance(doc, dict):
            errors.append(f"Document at index {idx} is not a JSON object.")
            continue

        missing = REQUIRED_KB_FIELDS - set(doc.keys())
        if missing:
            errors.append(f"Document at index {idx} missing required fields: {sorted(missing)}")
            continue

        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            errors.append(f"Document at index {idx} has an empty 'id'.")
        elif doc_id in doc_ids:
            errors.append(f"Duplicate document id found: '{doc_id}'")
        else:
            doc_ids.add(doc_id)

        title = str(doc.get("title", "")).strip()
        if not title:
            errors.append(f"Document '{doc_id}' has an empty 'title'.")

        cat = str(doc.get("category", "")).strip()
        if cat not in ALLOWED_INTENT_CATEGORIES:
            errors.append(
                f"Document '{doc_id}' has invalid category '{cat}'. Allowed: {sorted(ALLOWED_INTENT_CATEGORIES)}"
            )

        content = str(doc.get("content", "")).strip()
        if len(content) < 20:
            errors.append(f"Document '{doc_id}' content is too short (< 20 chars).")

        meta = doc.get("source_metadata")
        if not isinstance(meta, dict):
            errors.append(f"Document '{doc_id}' source_metadata must be a dictionary.")

    return errors, doc_ids


def validate_synthetic_requests(
    csv_path: Path, valid_doc_ids: Set[str]
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """Validates data/synthetic_requests.csv and detects leakage."""
    errors: List[str] = []
    category_counts: Dict[str, int] = {cat: 0 for cat in ALLOWED_INTENT_CATEGORIES}
    split_counts: Dict[str, int] = {s: 0 for s in ALLOWED_SPLITS}

    if not csv_path.exists():
        errors.append(f"Synthetic requests file not found at: {csv_path}")
        return errors, category_counts, split_counts

    seen_request_ids: Set[str] = set()
    split_texts: Dict[str, Set[str]] = {s: set() for s in ALLOWED_SPLITS}
    total_rows = 0

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])

            missing_headers = REQUIRED_CSV_FIELDS - headers
            if missing_headers:
                errors.append(f"CSV missing required headers: {sorted(missing_headers)}")
                return errors, category_counts, split_counts

            for row_idx, row in enumerate(reader, start=2):
                total_rows += 1
                req_id = (row.get("request_id") or "").strip()
                text = (row.get("text") or "").strip()
                cat = (row.get("intent_category") or "").strip()
                action = (row.get("expected_action") or "").strip()
                expected_doc = (row.get("expected_doc_id") or "").strip()
                is_restr_str = (row.get("is_restricted") or "").strip()
                is_ambig_str = (row.get("is_ambiguous") or "").strip()
                split = (row.get("split") or "").strip().lower()

                # Check request_id
                if not req_id:
                    errors.append(f"Row {row_idx}: 'request_id' is empty.")
                elif req_id in seen_request_ids:
                    errors.append(f"Row {row_idx}: duplicate request_id '{req_id}'.")
                else:
                    seen_request_ids.add(req_id)

                # Check text
                if not text:
                    errors.append(f"Row {row_idx} ({req_id}): 'text' is empty.")

                # Check category
                if cat not in ALLOWED_INTENT_CATEGORIES:
                    errors.append(f"Row {row_idx} ({req_id}): invalid intent_category '{cat}'.")
                else:
                    category_counts[cat] += 1

                # Check expected action
                if action not in ALLOWED_ACTION_TYPES:
                    errors.append(f"Row {row_idx} ({req_id}): invalid expected_action '{action}'.")

                # Check expected doc ID
                if expected_doc and expected_doc not in valid_doc_ids:
                    errors.append(
                        f"Row {row_idx} ({req_id}): expected_doc_id '{expected_doc}' does not exist in knowledge base."
                    )

                # Check booleans
                try:
                    parse_boolean(is_restr_str)
                except ValueError as exc:
                    errors.append(f"Row {row_idx} ({req_id}): {exc}")

                try:
                    parse_boolean(is_ambig_str)
                except ValueError as exc:
                    errors.append(f"Row {row_idx} ({req_id}): {exc}")

                # Check split
                if split not in ALLOWED_SPLITS:
                    errors.append(f"Row {row_idx} ({req_id}): invalid split '{split}'.")
                else:
                    split_counts[split] += 1
                    norm_text = " ".join(text.lower().split())
                    if norm_text:
                        split_texts[split].add(norm_text)

    except Exception as exc:
        errors.append(f"Failed to read CSV at {csv_path}: {exc}")
        return errors, category_counts, split_counts

    # Minimum dataset size check
    if total_rows < MINIMUM_REQUEST_COUNT:
        errors.append(
            f"Dataset size ({total_rows}) is below the required minimum of {MINIMUM_REQUEST_COUNT} requests."
        )

    # Check presence of all splits
    for s in ALLOWED_SPLITS:
        if split_counts[s] == 0:
            errors.append(f"Split '{s}' has 0 examples; all splits must have examples.")

    # Cross-split leakage check (normalized exact duplicate matching across splits)
    train_texts = split_texts["train"]
    val_texts = split_texts["val"]
    test_texts = split_texts["test"]

    train_val_leak = train_texts & val_texts
    if train_val_leak:
        errors.append(f"Cross-split leakage between train and val: {len(train_val_leak)} shared texts.")

    train_test_leak = train_texts & test_texts
    if train_test_leak:
        errors.append(f"Cross-split leakage between train and test: {len(train_test_leak)} shared texts.")

    val_test_leak = val_texts & test_texts
    if val_test_leak:
        errors.append(f"Cross-split leakage between val and test: {len(val_test_leak)} shared texts.")

    return errors, category_counts, split_counts


def run_validation(base_dir: Path) -> bool:
    """Runs end-to-end data validation and prints formatted results."""
    data_dir = base_dir / "data"
    kb_path = data_dir / "knowledge_base.json"
    csv_path = data_dir / "synthetic_requests.csv"

    print("=================================================================")
    print("  DecisionForge AI — Data Validation & Leakage Check")
    print("=================================================================")
    print(f"Knowledge Base:  {kb_path}")
    print(f"Synthetic Data:  {csv_path}\n")

    all_errors: List[str] = []

    kb_errors, valid_doc_ids = validate_knowledge_base(kb_path)
    all_errors.extend(kb_errors)

    csv_errors, cat_counts, split_counts = validate_synthetic_requests(csv_path, valid_doc_ids)
    all_errors.extend(csv_errors)

    if all_errors:
        print(f"[FAILED] VALIDATION FAILED with {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"  - {err}")
        print("=================================================================")
        return False

    total_requests = sum(split_counts.values())
    print("[PASSED] ALL DATA VALIDATION CHECKS PASSED!\n")
    print(f"Knowledge Base Documents: {len(valid_doc_ids)}")
    print(f"Total Synthetic Requests: {total_requests} (Requirement: >= {MINIMUM_REQUEST_COUNT})\n")

    print("Intent Category Distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  - {cat:25s}: {count:2d}")

    print("\nDataset Split Distribution:")
    for s, count in sorted(split_counts.items()):
        pct = (count / total_requests) * 100 if total_requests else 0
        print(f"  - {s:8s}: {count:2d} ({pct:5.1f}%)")

    print("\nLeakage & Integrity Status:")
    print("  - Train/Val Leakage:   0 duplicates")
    print("  - Train/Test Leakage:  0 duplicates")
    print("  - Val/Test Leakage:    0 duplicates")
    print("  - Reference Integrity: All expected_doc_ids map to valid KB docs or empty string.")
    print("=================================================================")
    return True


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    success = run_validation(project_root)
    sys.exit(0 if success else 1)
