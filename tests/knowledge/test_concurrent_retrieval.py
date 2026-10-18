"""Keep request-time embedding inference safe across domains and cancellations."""

import asyncio
import json
import threading
import time

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from consulting_platform.contracts import Domain
from consulting_platform.knowledge.service import LocalKnowledgeService


@pytest.fixture
def index_dir(tmp_path):
    chunk = {
        "chunk_id": "router-1", "document_id": "router-doc", "text": "Cisco router ports",
        "title": "Router data sheet", "domain": "technology", "company": "Cisco",
        "report_year": None, "source_uri": "data/raw/router.pdf", "page": 1,
        "printed_page": None,
    }
    (tmp_path / "chunks.json").write_text(json.dumps([chunk]))
    np.save(tmp_path / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk["text"]])
    joblib.dump(vectorizer, tmp_path / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, tmp_path / "tfidf_matrix.joblib")
    return tmp_path


class TrackingModel:
    def __init__(self):
        self.guard = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0
        self.started = threading.Event()
        self.release = None

    def encode(self, texts, **kwargs):
        with self.guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls += 1
        self.started.set()
        try:
            if self.release is not None:
                assert self.release.wait(3), "Test did not release native inference"
            else:
                time.sleep(0.01)
            return np.array([[1.0, 0.0]])
        finally:
            with self.guard:
                self.active -= 1


def service_with_model(monkeypatch, index_dir, model):
    def factory(name, **kwargs):
        # Prevent automatic Apple MPS selection even on a GPU-equipped Mac.
        assert kwargs["device"] == "cpu"
        assert kwargs["local_files_only"] is True
        return model

    monkeypatch.setattr("consulting_platform.knowledge.retrieval.SentenceTransformer", factory)
    service = LocalKnowledgeService(index_dir)
    assert service.retriever is not None
    return service


def test_simultaneous_requests_do_not_overlap_shared_model(monkeypatch, index_dir):
    model = TrackingModel()
    service = service_with_model(monkeypatch, index_dir, model)

    async def run():
        return await asyncio.gather(*(
            service.retrieve("Cisco router", [Domain.TECHNOLOGY]) for _ in range(12)
        ))

    results = asyncio.run(run())
    assert all(result.status == "ok" and result.chunks for result in results)
    assert model.calls == 12
    assert model.peak == 1


def test_cancelled_request_keeps_native_encoding_locked(monkeypatch, index_dir):
    model = TrackingModel()
    model.release = threading.Event()
    service = service_with_model(monkeypatch, index_dir, model)

    async def run():
        first = asyncio.create_task(service.retrieve("Cisco router", [Domain.TECHNOLOGY]))
        try:
            assert await asyncio.to_thread(model.started.wait, 2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(service.retrieve("Cisco ports", [Domain.TECHNOLOGY]))
            await asyncio.sleep(0.05)
            assert model.calls == 1
        finally:
            model.release.set()
        assert (await second).status == "ok"

    asyncio.run(run())
    assert model.calls == 2
    assert model.peak == 1


def test_real_index_concurrent_domains_use_cpu():
    service = LocalKnowledgeService()
    if service.retriever is None:
        pytest.skip("Real knowledge index is not available")
    assert str(service.retriever.model.device) == "cpu"
    queries = {
        Domain.TECHNOLOGY: "Cisco router Ethernet port specifications",
        Domain.CLINICAL: "Agilent Seahorse cell metabolic analysis",
        Domain.SEC: "Cisco operational risk factors",
        Domain.CORPORATE_RESPONSIBILITY: "SOFI food security nutrition financing",
    }

    async def run():
        return await asyncio.gather(*(
            service.retrieve(query, [domain], top_k=2)
            for _ in range(3) for domain, query in queries.items()
        ))

    results = asyncio.run(run())
    assert all(result.status == "ok" and result.chunks for result in results)
