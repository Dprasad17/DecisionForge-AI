"""
Unit tests for DecisionForge AI Supervised Intent Classifier (Approach A).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine.classifier import IntentClassifier
from decision_engine.config import INTENT_CATEGORIES


def test_classifier_training_and_classes():
    """Verify classifier trains and recognizes expected classes."""
    classifier = IntentClassifier(auto_train=True)
    metrics = classifier.train()

    assert metrics["train_samples"] >= 30
    assert metrics["train_accuracy"] >= 0.90
    assert metrics["num_classes"] == len(INTENT_CATEGORIES)


def test_classifier_prediction_output():
    """Verify prediction output types, confidence range, and probability calibration."""
    classifier = IntentClassifier(auto_train=True)
    query = "I forgot my master password and cannot sign in."
    category, confidence, class_probs = classifier.predict(query)

    assert isinstance(category, str)
    assert category in INTENT_CATEGORIES
    assert 0.0 <= confidence <= 1.0
    assert len(class_probs) == len(INTENT_CATEGORIES)

    # Probabilities must sum to approximately 1.0
    prob_sum = sum(class_probs.values())
    assert abs(prob_sum - 1.0) < 0.05


def test_classifier_empty_input_fallback():
    """Verify graceful handling of empty or whitespace input."""
    classifier = IntentClassifier(auto_train=True)
    category, confidence, class_probs = classifier.predict("   ")

    assert category == "other_or_unclear"
    assert confidence == 0.0
