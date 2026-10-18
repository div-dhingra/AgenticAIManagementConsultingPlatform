
from threading import Lock

import numpy as np

from consulting_platform.contracts import (
    Domain,
    RetrievalFilters,
)
from consulting_platform.knowledge.retrieval import HybridRetriever


class FakeModel:
    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        return np.array([[1.0, 0.0]])


class FakeVectorizer:
    def transform(self, texts):
        return np.array([[1.0, 0.0]])


class FakeTfidfMatrix:
    def __getitem__(self, indices):
        return FakeTfidfRows()


class FakeTfidfRows:
    def __matmul__(self, other):
        return FakeScores()


class FakeScores:
    def toarray(self):
        return np.array([[0.0], [0.0]])


def make_chunk(chunk_id, section, company="Cisco"):
    return {
        "chunk_id": chunk_id,
        "document_id": "cisco_2025",
        "text": f"Example text for {chunk_id}",
        "title": "Cisco Annual Report",
        "domain": "sec",
        "company": company,
        "report_year": 2025,
        "source_uri": "data/raw/cisco.pdf",
        "page": 1,
        "printed_page": "1",
        "section": section,
    }


def make_retriever():
    # Create a retriever without loading real index files.
    retriever = HybridRetriever.__new__(HybridRetriever)

    retriever.chunks = [
        make_chunk("general", None),
        make_chunk("risk", "item_1a"),
    ]

    # General chunk has a slightly higher semantic score.
    retriever.embeddings = np.array([
        [0.50, 0.0],
        [0.40, 0.0],
    ])

    retriever.vectorizer = FakeVectorizer()
    retriever.tfidf_matrix = FakeTfidfMatrix()
    retriever.model = FakeModel()
    retriever._encode_lock = Lock()

    return retriever


def test_risk_query_boost_changes_ranking():
    retriever = make_retriever()

    results = retriever.search(
        query="What are Cisco's major risk factors?",
        domains=[Domain.SEC],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "risk"
    assert results[1].chunk_id == "general"


def test_non_risk_query_keeps_original_ranking():
    retriever = make_retriever()

    results = retriever.search(
        query="What are Cisco's revenue drivers?",
        domains=[Domain.SEC],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "general"
    assert results[1].chunk_id == "risk"


def test_company_filter_is_preserved():
    retriever = make_retriever()

    results = retriever.search(
        query="What are the major risk factors?",
        domains=[Domain.SEC],
        filters=RetrievalFilters(
            companies=["Alphabet"],
        ),
        top_k=2,
    )

    assert results == []


def test_domain_filter_is_preserved():
    retriever = make_retriever()

    results = retriever.search(
        query="What are the major risk factors?",
        domains=[Domain.CORPORATE_RESPONSIBILITY],
        top_k=2,
    )

    assert results == []


def test_retrieval_scores_are_bounded():
    retriever = make_retriever()

    results = retriever.search(
        query="What are Cisco's major risk factors?",
        domains=[Domain.SEC],
        top_k=2,
    )

    for result in results:
        assert 0.0 <= result.retrieval_score <= 1.0
