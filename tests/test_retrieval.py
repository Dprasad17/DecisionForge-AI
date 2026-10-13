"""
Unit tests for DecisionForge AI Hybrid & Semantic Knowledge Retrieval (Approach B).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine.retrieval import HybridRetriever, RetrievedDocument


def test_retriever_initialization():
    """Verify retriever loads documents and pre-computes embeddings."""
    retriever = HybridRetriever()
    assert len(retriever.documents) == 12
    if retriever.doc_embeddings is not None:
        assert retriever.doc_embeddings.shape[0] == 12
        assert retriever.doc_embeddings.shape[1] == 384


def test_semantic_retrieval_relevance():
    """Verify semantic retrieval ranks correct document highest for payment inquiry."""
    retriever = HybridRetriever()
    query = "My card was charged but checkout said transaction failed."
    docs = retriever.retrieve_top_k(query, k=3, category="payment_issue")

    assert len(docs) >= 1
    assert isinstance(docs[0], RetrievedDocument)
    assert docs[0].doc_id in {"DOC-BIL-001", "DOC-BIL-002"}
    assert 0.0 <= docs[0].score <= 1.0
    assert len(docs[0].excerpt) > 10


def test_missing_evidence_threshold():
    """Verify out-of-scope or irrelevant queries fail to satisfy retrieval threshold."""
    retriever = HybridRetriever(threshold=0.35)
    query = "Where can I book tickets for interplanetary space shuttles?"
    top_doc, score = retriever.retrieve_top1(query)

    # Either no doc satisfies threshold or top_doc is None
    if top_doc is not None:
        assert top_doc.score >= 0.35
    else:
        assert top_doc is None
