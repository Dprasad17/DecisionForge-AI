"""
DecisionForge AI — Engine Configuration & Settings
"""

from pathlib import Path
import os
from typing import List, Set

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "decision_engine" / "models"

KB_PATH = DATA_DIR / "knowledge_base.json"
DATASET_PATH = DATA_DIR / "synthetic_requests.csv"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "tfidf_logreg_classifier.joblib"

# Decision Statuses
STATUS_AUTO_RESPONSE = "AUTO_RESPONSE"
STATUS_RECOMMENDATION_ONLY = "RECOMMENDATION_ONLY"
STATUS_HUMAN_REVIEW = "HUMAN_REVIEW"
STATUS_BLOCKED_BY_POLICY = "BLOCKED_BY_POLICY"

VALID_STATUSES: Set[str] = {
    STATUS_AUTO_RESPONSE,
    STATUS_RECOMMENDATION_ONLY,
    STATUS_HUMAN_REVIEW,
    STATUS_BLOCKED_BY_POLICY,
}

# Action Types
ACTION_PROVIDE_INFORMATION = "provide_information"
ACTION_REQUEST_CLARIFICATION = "request_clarification"
ACTION_RECOMMEND_TROUBLESHOOTING = "recommend_troubleshooting"
ACTION_RECOMMEND_HUMAN_REVIEW = "recommend_human_review"
ACTION_RESTRICTED_FINANCIAL = "restricted_financial_action"
ACTION_UNSUPPORTED = "unsupported_request"

# Intent Categories
INTENT_CATEGORIES: List[str] = [
    "account_access",
    "payment_issue",
    "refund_request",
    "subscription_change",
    "technical_problem",
    "order_or_service_status",
    "policy_question",
    "complaint",
    "other_or_unclear",
]

# Operational Thresholds
DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD = 0.35
MAX_CACHE_SIZE = 128

# Semantic Embedding Model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
