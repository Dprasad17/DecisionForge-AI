"""
DecisionForge AI — Lightweight Supervised Request Classifier (Approach A)

Combines dense semantic vector representations with Logistic Regression (with TF-IDF fallback).
Trained strictly on non-test partitions (train + val) of data/synthetic_requests.csv.
Provides continuous class probability distributions and confidence scores via predict_proba.
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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from decision_engine.config import (
    DATASET_PATH,
    CLASSIFIER_MODEL_PATH,
    MODELS_DIR,
    INTENT_CATEGORIES,
    DEFAULT_EMBEDDING_MODEL,
)


class IntentClassifier:
    """Supervised Classifier for request intent with dense embedding & TF-IDF capability."""

    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        model_path: Path = CLASSIFIER_MODEL_PATH,
        auto_train: bool = True,
    ) -> None:
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self.tfidf_pipeline: Optional[Pipeline] = None
        self.emb_clf: Optional[LogisticRegression] = None
        self.encoder = None

        # Lazy load or initialize SentenceTransformer encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        except Exception:
            self.encoder = None

        if self.model_path.exists():
            try:
                self._load_model()
            except Exception:
                if auto_train:
                    self.train()
        elif auto_train:
            self.train()

    def _load_model(self) -> None:
        """Loads serialized models from disk (supporting bundles and legacy pipelines)."""
        loaded = joblib.load(self.model_path)
        if isinstance(loaded, dict) and "emb_clf" in loaded:
            self.emb_clf = loaded["emb_clf"]
            self.tfidf_pipeline = loaded.get("tfidf_pipeline")
            self.pipeline = self.tfidf_pipeline
        elif isinstance(loaded, Pipeline):
            self.pipeline = loaded
            self.tfidf_pipeline = loaded
        else:
            self.pipeline = loaded

    def train(self) -> Dict[str, float]:
        """
        Trains classifier on non-test partitions (split in 'train' or 'val').
        Zero test data leakage. Saves trained bundle to model_path.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        train_texts: List[str] = []
        train_labels: List[str] = []

        with self.dataset_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sp = (row.get("split") or "").strip().lower()
                if sp in {"train", "val"}:
                    text = (row.get("text") or "").strip()
                    category = (row.get("intent_category") or "").strip()
                    if text and category:
                        train_texts.append(text)
                        train_labels.append(category)

        if not train_texts:
            raise ValueError("No training examples found in dataset.")

        # 1. Fit TF-IDF Pipeline
        tfidf_pipe = Pipeline([
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
        tfidf_pipe.fit(train_texts, train_labels)
        self.tfidf_pipeline = tfidf_pipe
        self.pipeline = tfidf_pipe

        # 2. Fit Embedding Classifier
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

        raw_embeddings = self.encoder.encode(train_texts, show_progress_bar=False)
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_train_emb = raw_embeddings / norms

        emb_clf = LogisticRegression(
            C=5.0,
            max_iter=500,
            class_weight="balanced",
            random_state=42,
        )
        emb_clf.fit(X_train_emb, train_labels)
        self.emb_clf = emb_clf

        train_acc = emb_clf.score(X_train_emb, train_labels)

        # Save bundle
        bundle = {
            "emb_clf": emb_clf,
            "tfidf_pipeline": tfidf_pipe,
            "classes": emb_clf.classes_,
        }
        joblib.dump(bundle, self.model_path)

        return {
            "train_samples": len(train_texts),
            "train_accuracy": float(train_acc),
            "num_classes": len(emb_clf.classes_),
        }

    def predict(
        self, text: str, embedding: Optional[np.ndarray] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts intent category and returns:
        (best_category, confidence_score, all_class_probabilities)
        """
        clean_text = text.strip()
        if not clean_text:
            return "other_or_unclear", 0.0, {}

        # 1. Prediction using pre-computed or generated embedding
        if embedding is not None and self.emb_clf is not None:
            # Reshape 1D embedding to 2D row if needed
            q_emb = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
            probs = self.emb_clf.predict_proba(q_emb)[0]
            classes = self.emb_clf.classes_
        elif self.encoder is not None and self.emb_clf is not None:
            emb = self.encoder.encode([clean_text], show_progress_bar=False)[0]
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm
            probs = self.emb_clf.predict_proba([emb])[0]
            classes = self.emb_clf.classes_
        elif self.tfidf_pipeline is not None:
            probs = self.tfidf_pipeline.predict_proba([clean_text])[0]
            classes = self.tfidf_pipeline.classes_
        elif self.pipeline is not None:
            probs = self.pipeline.predict_proba([clean_text])[0]
            classes = self.pipeline.classes_
        else:
            raise RuntimeError("Classifier model is not trained or loaded.")

        class_prob_map: Dict[str, float] = {
            cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)
        }

        # Ensure all defined categories exist in output map
        for cat in INTENT_CATEGORIES:
            if cat not in class_prob_map:
                class_prob_map[cat] = 0.0

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
