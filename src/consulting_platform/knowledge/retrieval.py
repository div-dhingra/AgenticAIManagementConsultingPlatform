
import json
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from consulting_platform.agents.grounding import normalize_company
from consulting_platform.contracts import (
    Domain,
    EvidenceChunk,
    RetrievalFilters,
)
from consulting_platform.knowledge.index import MODEL_NAME
from consulting_platform.knowledge.query_intent import get_risk_section_boost
from consulting_platform.knowledge.sections import assign_sections


class HybridRetriever:
    def __init__(self, index_dir):
        index_dir = Path(index_dir)

        # Load chunks from the saved index.
        with open(index_dir / "chunks.json", encoding="utf-8") as file:
            self.chunks = json.load(file)

        # Add SEC section metadata in memory.
        # This does not modify the original index files.
        self.chunks = assign_sections(self.chunks)

        # Load semantic embeddings.
        self.embeddings = np.load(
            index_dir / "embeddings.npy"
        )

        # Load TF-IDF vectorizer and matrix.
        self.vectorizer = joblib.load(
            index_dir / "tfidf_vectorizer.joblib"
        )

        self.tfidf_matrix = joblib.load(
            index_dir / "tfidf_matrix.joblib"
        )

        # Use CPU for request-time inference. On Apple Silicon the automatic
        # MPS device can segfault inside Metal when requests encode concurrently.
        # The checkpoint is unchanged, so existing index embeddings remain valid.
        self.model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
        self._encode_lock = Lock()

        # Check that the index files have matching sizes.
        chunk_count = len(self.chunks)

        if self.embeddings.shape[0] != chunk_count:
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        if self.tfidf_matrix.shape[0] != chunk_count:
            raise ValueError(
                "TF-IDF row count does not match chunk count."
            )

    def matches_filters(
        self,
        chunk: dict,
        domains: list[Domain],
        filters: RetrievalFilters | None,
    ) -> bool:
        # Domain filtering is always required.
        if chunk["domain"] not in domains:
            return False

        # No additional filters means all matching
        # chunks in the requested domains are allowed.
        if filters is None:
            return True

        # Filter by company.
        if filters.companies:
            company = normalize_company(chunk.get("company") or "", Domain(chunk["domain"]))
            if not company or company not in {
                normalize_company(value, Domain(chunk["domain"])) for value in filters.companies
            }:
                return False

        # Filter by report year.
        if filters.report_years:
            if chunk["report_year"] not in filters.report_years:
                return False

        # Filter by document ID.
        if filters.document_ids:
            if chunk["document_id"] not in filters.document_ids:
                return False

        return True

    def search(
        self,
        query: str,
        domains: list[Domain],
        filters: RetrievalFilters | None = None,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:

        # Step 1: Find chunks matching the requested filters.
        matching_indices = []

        for index, chunk in enumerate(self.chunks):
            if self.matches_filters(chunk, domains, filters):
                matching_indices.append(index)

        if not matching_indices:
            return []

        # Step 2: Generate a semantic embedding for the query.
        # This lock lives in the worker thread, so cancellation of an async
        # request cannot release it while native inference is still running.
        # Only local encoding is serialized; the specialist LLM calls stay parallel.
        with self._encode_lock:
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

        # Step 3: Calculate semantic similarity.
        semantic_scores = (
            self.embeddings[matching_indices] @ query_embedding
        )

        # Step 4: Calculate TF-IDF similarity.
        query_tfidf = self.vectorizer.transform([query])

        lexical_scores = (
            self.tfidf_matrix[matching_indices] @ query_tfidf.T
        ).toarray().flatten()

        # Step 5: Normalize scores to the range [0, 1].
        semantic_scores = np.clip(
            (semantic_scores + 1.0) / 2.0,
            0.0,
            1.0,
        )

        lexical_scores = np.clip(
            lexical_scores,
            0.0,
            1.0,
        )

        # Step 6: Combine semantic and lexical scores.
        hybrid_scores = (
            0.65 * semantic_scores
            + 0.35 * lexical_scores
        )

        # Step 7: Apply Query-Aware Section Boost.
        boost = get_risk_section_boost(query)

        adjusted_scores = hybrid_scores.copy()

        if boost > 0:
            for position, chunk_index in enumerate(matching_indices):
                chunk = self.chunks[chunk_index]

                # Only boost SEC Item 1A Risk Factors.
                if (
                    chunk["domain"] == Domain.SEC
                    and chunk["section"] == "item_1a"
                ):
                    adjusted_scores[position] += boost

        # Keep the final ranking scores between 0 and 1.
        adjusted_scores = np.clip(
            adjusted_scores,
            0.0,
            1.0,
        )

        # Step 8: Sort chunks by adjusted score.
        sorted_positions = np.argsort(
            -adjusted_scores,
            kind="stable",
        )

        # A multi-company question must not spend the entire evidence budget on
        # the company with the highest lexical match. Reserve its best matching
        # passage for each requested company, then fill the remaining slots.
        if filters and len(filters.companies) > 1:
            representatives = {}
            for position in sorted_positions:
                if adjusted_scores[position] < 0.20:
                    continue
                chunk = self.chunks[matching_indices[position]]
                company = normalize_company(chunk.get("company") or "", Domain(chunk["domain"]))
                representatives.setdefault(company, position)
            reserved = list(representatives.values())[:top_k]
            selected = set(reserved)
            for position in sorted_positions:
                if len(selected) >= top_k:
                    break
                if adjusted_scores[position] >= 0.20:
                    selected.add(position)
            sorted_positions = [position for position in sorted_positions if position in selected]

        # Step 9: Convert matching chunks to EvidenceChunk.
        results = []

        for position in sorted_positions:
            score = float(adjusted_scores[position])

            # Preserve the existing minimum score threshold.
            if score < 0.20:
                continue

            chunk_index = matching_indices[position]
            chunk = self.chunks[chunk_index]

            evidence = EvidenceChunk(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                text=chunk["text"],
                title=chunk["title"],
                domain=chunk["domain"],
                company=chunk["company"],
                report_year=chunk["report_year"],
                source_uri=chunk["source_uri"],
                page=chunk["page"],
                printed_page=chunk["printed_page"],
                retrieval_score=score,
            )

            results.append(evidence)

            if len(results) >= top_k:
                break

        return results
