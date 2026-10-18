
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

The first build may download the `sentence-transformers/all-MiniLM-L6-v2` model and therefore requires an internet connection.

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
| Technology | No indexed documents |
| Clinical | No indexed documents |

Current SEC coverage includes Cisco, Alphabet, and Agilent reports.

Current Corporate Responsibility coverage includes SOFI reports.

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
