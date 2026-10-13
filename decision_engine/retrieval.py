"""
DecisionForge AI — Hybrid & Semantic Knowledge Retrieval (Approach B)

Combines dense vector embeddings (sentence-transformers / all-MiniLM-L6-v2)
with sparse keyword token overlap over data/knowledge_base.json.
Provides top-k evidence extraction, cosine similarity scoring, and fallback.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from decision_engine.config import (
    KB_PATH,
    DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
)

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "will",
    "would", "should", "i", "me", "my", "you", "your", "it", "its", "we",
    "our", "they", "their", "this", "that", "these", "those", "how", "what",
    "where", "when", "why", "which", "who", "all", "any", "some", "not", "no"
}


def tokenize(text: str) -> List[str]:
    """Extracts alphanumeric tokens, filtering stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


@dataclass
class RetrievedDocument:
    """Structured representation of retrieved knowledge evidence."""
    doc_id: str
    title: str
    category: str
    excerpt: str
    score: float
    source_metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


class HybridRetriever:
    """Hybrid Knowledge Retrieval combining dense embeddings and sparse token matching."""

    def __init__(
        self,
        kb_path: Path = KB_PATH,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        threshold: float = DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
    ) -> None:
        self.kb_path = kb_path
        self.model_name = model_name
        self.threshold = threshold

        self.documents: List[dict] = self._load_documents()
        self.encoder = None
        self.doc_embeddings: Optional[np.ndarray] = None

        self._initialize_embeddings()

    def _load_documents(self) -> List[dict]:
        """Loads knowledge base documents from disk."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.kb_path}")
        with self.kb_path.open("r", encoding="utf-8") as f:
            docs = json.load(f)

        for doc in docs:
            doc["_tokens"] = set(tokenize(doc["title"] + " " + doc["content"]))
        return docs

    def _initialize_embeddings(self) -> None:
        """Pre-computes and normalizes document embeddings using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name)
            doc_texts = [f"{d['title']}. {d['content']}" for d in self.documents]
            raw_embeddings = self.encoder.encode(doc_texts, show_progress_bar=False)

            # L2 normalize embeddings for fast cosine similarity dot products
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.doc_embeddings = raw_embeddings / norms
        except Exception as exc:
            # Fallback to sparse keyword search only if embeddings fail
            self.encoder = None
            self.doc_embeddings = None

    def _calculate_keyword_score(self, query_tokens: Set[str], doc: dict) -> float:
        """Computes sparse keyword coverage and title overlap."""
        if not query_tokens:
            return 0.0

        content_tokens = doc.get("_tokens", set())
        title_tokens = set(tokenize(doc.get("title", "")))

        c_overlap = query_tokens & content_tokens
        t_overlap = query_tokens & title_tokens

        coverage = len(c_overlap) / len(query_tokens)
        title_bonus = len(t_overlap) * 0.15
        return min(1.0, coverage + title_bonus)

    def retrieve_top_k(
        self, query: str, k: int = 3, category: Optional[str] = None
    ) -> List[RetrievedDocument]:
        """
        Retrieves top-k documents using hybrid scoring:
        hybrid_score = 0.7 * semantic_similarity + 0.3 * keyword_score
        """
        clean_query = query.strip()
        query_tokens = set(tokenize(clean_query))

        if not clean_query:
            return []

        # 1. Semantic similarities
        semantic_scores = np.zeros(len(self.documents))
        if self.encoder is not None and self.doc_embeddings is not None:
            try:
                q_emb = self.encoder.encode([clean_query], show_progress_bar=False)[0]
                q_norm = np.linalg.norm(q_emb)
                if q_norm > 0:
                    q_emb = q_emb / q_norm
                    # Cosine similarities
                    semantic_scores = np.dot(self.doc_embeddings, q_emb)
                    # Clip to [0, 1]
                    semantic_scores = np.clip(semantic_scores, 0.0, 1.0)
            except Exception:
                semantic_scores = np.zeros(len(self.documents))

        # 2. Sparse keyword scores
        keyword_scores = np.array([
            self._calculate_keyword_score(query_tokens, doc) for doc in self.documents
        ])

        # 3. Hybrid fusion
        if self.encoder is not None and self.doc_embeddings is not None:
            combined_scores = 0.70 * semantic_scores + 0.30 * keyword_scores
        else:
            combined_scores = keyword_scores

        # Apply category alignment bonus
        if category:
            for idx, doc in enumerate(self.documents):
                if doc.get("category") == category:
                    combined_scores[idx] += 0.05

        # Rank indices
        top_indices = np.argsort(combined_scores)[::-1][:k]

        results: List[RetrievedDocument] = []
        for idx in top_indices:
            score = float(combined_scores[idx])
            doc = self.documents[idx]

            # Extract relevant sentence excerpt
            content = doc.get("content", "")
            sentences = re.split(r"(?<=[.!?])\s+", content)
            excerpt = sentences[0]
            for s in sentences:
                if any(t in s.lower() for t in query_tokens):
                    excerpt = s
                    break
            if len(excerpt) > 220:
                excerpt = excerpt[:217] + "..."

            results.append(
                RetrievedDocument(
                    doc_id=doc["id"],
                    title=doc["title"],
                    category=doc["category"],
                    excerpt=excerpt,
                    score=round(score, 4),
                    source_metadata=doc.get("source_metadata", {}),
                )
            )

        return results

    def retrieve_top1(
        self, query: str, category: Optional[str] = None
    ) -> Tuple[Optional[RetrievedDocument], float]:
        """
        Retrieves top-1 document, returning (doc, score) if score >= threshold,
        else (None, top_score).
        """
        candidates = self.retrieve_top_k(query=query, k=1, category=category)
        if not candidates:
            return None, 0.0

        top = candidates[0]
        if top.score < self.threshold:
            return None, top.score
        return top, top.score


if __name__ == "__main__":
    retriever = HybridRetriever()
    q = "My payment failed but money was deducted from my account."
    docs = retriever.retrieve_top_k(q, k=2, category="payment_issue")
    for d in docs:
        print(f"[{d.doc_id}] {d.title} (Score: {d.score})")
        print(f"   Excerpt: {d.excerpt}")
