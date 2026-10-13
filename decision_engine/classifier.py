"""
DecisionForge AI — Lightweight Supervised Request Classifier (Approach A)

Uses scikit-learn Pipeline with TfidfVectorizer and LogisticRegression.
Trained strictly on the training partition of data/synthetic_requests.csv.
Provides calibrated probability confidence scores via predict_proba.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from decision_engine.config import (
    DATASET_PATH,
    CLASSIFIER_MODEL_PATH,
    MODELS_DIR,
    INTENT_CATEGORIES,
)


class IntentClassifier:
    """Supervised TF-IDF + Logistic Regression Classifier for request intent."""

    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        model_path: Path = CLASSIFIER_MODEL_PATH,
        auto_train: bool = True,
    ) -> None:
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None

        if self.model_path.exists():
            self._load_model()
        elif auto_train:
            self.train()

    def _load_model(self) -> None:
        """Loads serialized pipeline from disk."""
        self.pipeline = joblib.load(self.model_path)

    def train(self) -> Dict[str, float]:
        """
        Trains TF-IDF + Logistic Regression strictly on split == 'train'.
        Saves trained pipeline to model_path and returns training metrics.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        train_texts: List[str] = []
        train_labels: List[str] = []

        with self.dataset_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("split") or "").strip().lower() == "train":
                    text = (row.get("text") or "").strip()
                    category = (row.get("intent_category") or "").strip()
                    if text and category:
                        train_texts.append(text)
                        train_labels.append(category)

        if not train_texts:
            raise ValueError("No training examples found in dataset.")

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                    max_features=2500,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=5.0,
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ])

        pipeline.fit(train_texts, train_labels)
        train_acc = pipeline.score(train_texts, train_labels)

        joblib.dump(pipeline, self.model_path)
        self.pipeline = pipeline

        return {
            "train_samples": len(train_texts),
            "train_accuracy": float(train_acc),
            "num_classes": len(pipeline.classes_),
        }

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts intent category and returns:
        (best_category, confidence_score, all_class_probabilities)
        """
        if self.pipeline is None:
            raise RuntimeError("Classifier model is not trained or loaded.")

        clean_text = text.strip()
        if not clean_text:
            return "other_or_unclear", 0.0, {}

        probs = self.pipeline.predict_proba([clean_text])[0]
        classes = self.pipeline.classes_

        class_prob_map: Dict[str, float] = {
            cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)
        }

        best_idx = probs.argmax()
        best_category = classes[best_idx]
        best_confidence = round(float(probs[best_idx]), 4)

        return best_category, best_confidence, class_prob_map


if __name__ == "__main__":
    classifier = IntentClassifier(auto_train=True)
    metrics = classifier.train()
    print("Classifier trained successfully:", metrics)

    sample = "My payment failed but money was deducted from my account."
    cat, conf, probs = classifier.predict(sample)
    print(f"Sample: '{sample}' -> Category: {cat}, Conf: {conf}")
