
# Knowledge Module Integration Guide

## Overview

The Knowledge Module provides local hybrid document retrieval for the Agentic AI Management Consulting Platform.

It combines:
- Semantic search using SentenceTransformers
- TF-IDF lexical search
- Query-aware ranking for SEC risk-factor questions

The module implements the shared `KnowledgeService` contract.

## 1. Installation

From the project root, install the Knowledge dependencies:

```powershell
python -m pip install -e ".[knowledge]"
```

For development and testing:

```powershell
python -m pip install -e ".[dev,knowledge]"
```

## 2. Build the Local Index

The repository includes the processed chunks:

`data/chunks.jsonl`

Build the retrieval index by running:

```powershell
python scripts/build_index.py
```

The script creates:

```text
data/index/
    chunks.json
    embeddings.npy
    tfidf_matrix.joblib
    tfidf_vectorizer.joblib
```

The generated index is excluded from Git. Each developer should build it locally.

Pulling the code and chunks alone is not enough: install the `knowledge` extra,
build the index, and restart the backend. Model loading at service startup uses
the local cache populated during the build, so a running service does not need
to download the embedding model.

On this project's macOS virtual environment:

```bash
.venv/bin/python -m pip install -e '.[knowledge]'
.venv/bin/python scripts/build_index.py
# Stop the previous backend with Ctrl+C first.
APP_MODE=live CONSULT_EVIDENCE_FILE= .venv/bin/uvicorn consulting_platform.api.app:create_app --factory --port 8000
```

The explicit empty `CONSULT_EVIDENCE_FILE` disables controlled passages even if
the shell had previously exported a value. Refresh the Streamlit status after
restarting. Environment changes do not update an already running backend.

To regenerate chunks after importing source PDFs into the paths in the manifest:

```bash
.venv/bin/python scripts/extract_documents.py
.venv/bin/python scripts/chunk_documents.py
.venv/bin/python scripts/build_index.py
```

Chunks preserve the original extracted page text, including capitalization and
punctuation. SOFI report years are read from their report titles; company filters
accept the same capitalization and aliases as the agents.

The first build may download the `sentence-transformers/all-MiniLM-L6-v2` model and therefore requires an internet connection.

### macOS request-time embeddings

Query embeddings explicitly use CPU. PyTorch's automatic Apple GPU (MPS)
selection caused native `SIGSEGV` crashes during concurrent specialist searches
on the development Mac. Access to the shared embedding model is protected by a
thread lock, including when an awaiting request is cancelled. Retrieval stays
off the API event loop; specialist model API calls still run concurrently.

This uses the same embedding checkpoint and dimension as the index builder.
An existing index remains usable; changing the query device does not require a
rebuild. The bulk index builder can still select an available accelerator.

The corpus contains extracted text as well as citation URLs. The service searches
that local text; the agents do not automatically fetch the referenced web pages.

Multi-company searches reserve a top matching passage for each requested company
within `top_k`; otherwise Cisco results can crowd out an indexed Agilent filing.
SOFI extraction preserves native text-block order rather than sorting lines across
both page columns. The agents receive short exact source spans and can cite their
IDs, avoiding retyped or spliced quotations. Span IDs are private to the agent
workflow; public request/response contracts are unchanged.

For Command A+ and Command A Reasoning, structured calls use an explicit small
thinking budget (256 tokens). Summary tasks omit irrelevant candidate/action
fields from the provider schema. This reduces output truncation while keeping
the existing total-call, retry and workflow-time limits.

## 3. Use the Knowledge Service

Import the factory and shared contracts:

```python
from consulting_platform.contracts import Domain, RetrievalFilters
from consulting_platform.knowledge.service import create_knowledge_service
```

Create the service:

```python
service = create_knowledge_service()
```

Check domain availability:

```python
status = service.get_status()

for domain, info in status.domains.items():
    print(domain.value, info.status, info.chunk_count)
```

Retrieve evidence asynchronously:

```python
async def retrieve_evidence():
    result = await service.retrieve(
        query="What are Cisco's major risk factors?",
        domains=[Domain.SEC],
        filters=RetrievalFilters(
            companies=["Cisco"],
            report_years=[2025],
        ),
        top_k=5,
    )

    print("Status:", result.status)

    for warning in result.warnings:
        print("Warning:", warning)

    for chunk in result.chunks:
        print("Source:", chunk.source_uri)
        print("Page:", chunk.page)
        print("Score:", chunk.retrieval_score)
        print("Evidence:", chunk.text[:300])

    return result
```

Inside an existing asynchronous Agent or API function, call:

```python
result = await retrieve_evidence()
```

For a standalone synchronous script, use:

```python
import asyncio

result = asyncio.run(retrieve_evidence())
```

## 4. Retrieval Status

Possible retrieval statuses:

- `ok`: Retrieval completed for the requested indexed domains.
- `partial`: Some requested domains have no indexed documents, but evidence was retrieved from others.
- `unavailable`: The index is unavailable, or no evidence was found when requested domains are missing.

Always inspect `result.status` and `result.warnings`.

The service reports `mode="live"` because it uses local retrieval rather than demo data. This does not guarantee that an index is available.

## 5. Current Dataset Coverage

Current indexed document coverage:

| Domain | Status |
|---|---|
| SEC | Available |
| Corporate Responsibility | Available |
| Technology | Available: 7 documents |
| Clinical | Available: 2 documents |

Current SEC coverage includes Cisco, Alphabet, and Agilent reports.

Current Corporate Responsibility coverage includes SOFI reports.

Technology includes Cisco router datasheets and Google Cloud RAG/MCP documents.
Clinical includes Agilent Seahorse and digital-pathology material. These domains
have indexed evidence; that does not guarantee coverage of every specification
or question. The new source PDFs are not included in Git; their delivered chunks
and source URLs are enough to rebuild the index. Preserve these chunks when
regenerating the corpus unless all corresponding originals have been imported.

After rebuilding the index and restarting, all four domains should report ready.
With valid model configuration and a loaded retail artifact, `/health` should
report overall `ok`. Readiness does not test a live model call or answer quality.

The dataset is limited to the documents currently included in `data/chunks.jsonl`. Missing documents should not be treated as evidence that information does not exist.

## 6. Run Tests

Run Knowledge tests:

```powershell
python -m pytest tests/knowledge -q --import-mode=importlib
```

Run all project tests:

```powershell
python -m pytest -q --import-mode=importlib
```

Some integration tests are skipped automatically when the local index is unavailable.

## 7. Integration Notes

- Use `create_knowledge_service()` as the service entry point.
- Do not import `HybridRetriever` directly from Agent or API code.
- `retrieve()` is asynchronous.
- `get_status()` is synchronous.
- `top_k` must be between 1 and 20.
- Check `status` and `warnings` before presenting evidence.
- Preserve source information and page references when generating consulting reports.
- Do not present unavailable domains as fully supported.
