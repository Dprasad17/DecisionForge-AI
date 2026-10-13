# Synthetic Data & Knowledge Base

This directory contains the synthetic knowledge base and labeled request dataset for **DecisionForge AI**.

## 1. Safety & Data Privacy
- **100% Synthetic Data:** All customer queries, ticket IDs, and company policies are purely synthetic.
- **No Real Customer Data:** No real customer records, credentials, or actual financial account details are used.
- **Controlled Simulation:** Action types representing financial transactions (e.g., refunds, wire reversals) are restricted simulations and will never execute against a real banking or external API.

## 2. Dataset Files

### `knowledge_base.json`
A curated corpus of 12 policy and operational FAQ documents.
- `id`: Unique stable identifier (e.g. `DOC-ACC-001`, `DOC-REF-001`).
- `title`: Human-readable document title.
- `category`: Intent category alignment.
- `content`: Plaintext policy details and troubleshooting instructions.
- `source_metadata`: Dictionary with version, effective date, department, allowed actions, and restricted actions.

### `synthetic_requests.csv`
A labeled benchmark dataset containing 48 realistic user requests.
- `request_id`: Primary key (e.g. `REQ-001`).
- `text`: Raw input user request text.
- `intent_category`: Ground truth category (9 categories).
- `expected_action`: Prescribed next step (6 action types).
- `expected_doc_id`: ID of the relevant knowledge base document, or empty (`""`) if unsupported or no evidence is expected.
- `is_restricted`: Boolean indicating whether this request triggers restricted financial or elevated security guardrails.
- `is_ambiguous`: Boolean indicating whether the request is underspecified and requires user clarification.
- `split`: Dataset partition (`train`, `val`, `test`).

## 3. Taxonomy

### Intent Categories
- `account_access`
- `payment_issue`
- `refund_request`
- `subscription_change`
- `technical_problem`
- `order_or_service_status`
- `policy_question`
- `complaint`
- `other_or_unclear`

### Action Types
- `provide_information`
- `request_clarification`
- `recommend_troubleshooting`
- `recommend_human_review`
- `restricted_financial_action`
- `unsupported_request`

## 4. Split Distribution
- **Total Requests:** 78
- **Train Split:** 34 requests (~43.6%)
- **Validation Split:** 12 requests (~15.4%)
- **Test Split:** 32 requests (~41.0%) [Fulfills the $\ge 30$ evaluation test set requirement]

## 5. Validation
Run the standalone data validator:
```bash
python data/validate_data.py
```
This verifies:
1. File existence and JSON/CSV parse integrity.
2. Unique request IDs.
3. Strict adherence to taxonomy enums.
4. Valid booleans (`True`/`False`).
5. Document reference validity against `knowledge_base.json`.
6. Presence of all 3 splits.
7. Zero cross-split leakage or exact text duplicates.
8. Minimum dataset size requirement ($\ge 35$).
