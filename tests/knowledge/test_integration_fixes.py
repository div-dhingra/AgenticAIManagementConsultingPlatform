"""Agent-facing filtering, source fidelity and nonblocking retrieval regressions."""

import asyncio
import threading

import pytest
from transformers import AutoTokenizer

from consulting_platform.contracts import Domain, RetrievalFilters
from consulting_platform.knowledge.chunking import MODEL_NAME, split_text_into_chunks
from consulting_platform.knowledge.metadata import identify_report_year
from consulting_platform.knowledge.retrieval import HybridRetriever
from consulting_platform.knowledge.service import LocalKnowledgeService


@pytest.mark.parametrize("company", ["cisco", "Cisco", "Cisco Systems, Inc."])
def test_agent_company_filters_accept_canonical_aliases(company):
    retriever = HybridRetriever.__new__(HybridRetriever)
    chunk = {"domain": "sec", "company": "Cisco", "report_year": 2025, "document_id": "c"}
    assert retriever.matches_filters(
        chunk, [Domain.SEC], RetrievalFilters(companies=[company], report_years=[2025])
    )
    assert not retriever.matches_filters(
        chunk, [Domain.SEC], RetrievalFilters(companies=["Agilent"])
    )


def test_chunks_preserve_original_text_and_table_layout():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    text = "Cisco C8300-2N2S-6T | 6 1G ports\nAgilent’s Results: $12.50; SOFI 2024. " * 20
    chunks = split_text_into_chunks(text, tokenizer, max_tokens=70, overlap_tokens=8)
    assert len(chunks) > 1
    assert "Cisco C8300-2N2S-6T | 6 1G ports\nAgilent’s" in chunks[0]
    assert all(chunk in text for chunk in chunks)
    assert all(len(tokenizer.encode(chunk)) <= 70 for chunk in chunks)


def test_sofi_year_is_read_from_report_title(tmp_path):
    import pymupdf

    path = tmp_path / "unknown-name.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 40), "THE STATE OF FOOD SECURITY AND NUTRITION IN THE WORLD 2024")
        pdf.save(path)
    assert identify_report_year(path) == 2024


def test_embedding_search_runs_outside_event_loop(tmp_path):
    service = LocalKnowledgeService(tmp_path / "missing")
    loop_thread = threading.get_ident()

    class Retriever:
        chunks = [{"domain": "sec"}]

        def search(self, **kwargs):
            assert threading.get_ident() != loop_thread
            return []

    service.retriever = Retriever()
    assert asyncio.run(service.retrieve("risk", [Domain.SEC])).status == "ok"


def test_two_column_sofi_extraction_keeps_paragraphs_contiguous(tmp_path):
    import pymupdf

    from consulting_platform.knowledge.extraction import extract_pdf

    path = tmp_path / "two_columns.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 100), "Agree a common definition.\nTrack financing flows consistently.")
        page.insert_text((330, 100), "Improve coordination.\nMeasure the financing gap.")
        pdf.save(path)
    doc = {
        "local_path": str(path), "document_id": "columns", "title": "SOFI",
        "domain": "corporate_responsibility", "company": None, "report_year": 2024,
        "filing_type": None, "original_url": None,
    }
    text = " ".join(extract_pdf(doc)["pages"][0]["text"].split())
    assert "Agree a common definition. Track financing flows consistently." in text
    assert "Improve coordination. Measure the financing gap." in text
