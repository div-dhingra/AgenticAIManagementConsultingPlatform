"""Synthetic fixtures and deterministic test doubles. Never implicit live fallbacks."""

from __future__ import annotations

import copy
import re
from collections import deque

from consulting_platform.agents.grounding import filter_evidence
from consulting_platform.agents.service import keyword_route
from consulting_platform.contracts import (
    ConsultRequest,
    ConsultResponse,
    Domain,
    DomainKnowledgeStatus,
    EvidenceChunk,
    KnowledgeStatus,
    RetrievalFilters,
    RetrievalResult,
)


def fixture_chunks() -> list[EvidenceChunk]:
    rows = [
        (
            Domain.TECHNOLOGY,
            "Cisco",
            None,
            "Fixture Router A has 2 ports rated at 10 Gbps. "
            "The synthetic diagnostic guide requires the exact model and diagnostic logs before replacement.",
        ),
        (
            Domain.CLINICAL,
            "Agilent",
            None,
            "Fixture Analyzer A supports 16 samples per run. "
            "Fixture Analyzer B supports 8 samples per run.",
        ),
        (
            Domain.SEC,
            "Cisco",
            2025,
            "The synthetic Cisco 2025 filing identifies supply-chain disruption as an operational risk.",
        ),
        (
            Domain.SEC,
            "Agilent",
            2025,
            "The synthetic Agilent 2025 filing identifies supplier concentration as an operational risk.",
        ),
        (
            Domain.CORPORATE_RESPONSIBILITY,
            None,
            2023,
            "The synthetic SOFI 2023 report identifies conflict and rising food prices as food-security risks.",
        ),
        (
            Domain.CORPORATE_RESPONSIBILITY,
            None,
            2024,
            "The synthetic SOFI 2024 report identifies financing gaps as a barrier to food-security action.",
        ),
    ]
    return [
        EvidenceChunk(
            chunk_id=f"fixture-{domain.value}-{i}",
            document_id=f"fixture-doc-{i}",
            text=text,
            title=f"SYNTHETIC {domain.value} fixture {i}",
            domain=domain,
            company=company,
            report_year=year,
            source_uri="fixtures/consulting/evidence.json",
            page=None,
            printed_page=None,
            retrieval_score=0.8,
        )
        for i, (domain, company, year, text) in enumerate(rows, 1)
    ]


class FakeKnowledgeService:
    def __init__(
        self,
        chunks: list[EvidenceChunk] | None = None,
        *,
        mode: str = "demo",
        unavailable_domains: set[Domain] | None = None,
    ):
        self.chunks = list(chunks) if chunks is not None else fixture_chunks()
        self.mode = mode
        self.unavailable_domains = unavailable_domains or set()
        self.calls: list[dict] = []

    async def retrieve(
        self,
        query: str,
        domains: list[Domain],
        filters: RetrievalFilters | None = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        if not query.strip() or not domains or not 1 <= top_k <= 20:
            raise ValueError("Invalid retrieval input")
        filters = filters or RetrievalFilters()
        self.calls.append({"query": query, "domains": domains, "filters": filters, "top_k": top_k})
        working = [d for d in domains if d not in self.unavailable_domains]
        chunks = []
        for domain in working:
            matching, _ = filter_evidence(self.chunks, domain, filters)
            chunks.extend(matching)
        status = (
            "unavailable" if not working else "partial" if len(working) < len(domains) else "ok"
        )
        return RetrievalResult(mode=self.mode, status=status, chunks=chunks[:top_k], warnings=[])

    def get_status(self) -> KnowledgeStatus:
        return KnowledgeStatus(
            mode=self.mode,
            domains={
                d: DomainKnowledgeStatus(
                    status="unavailable" if d in self.unavailable_domains else "ready",
                    document_count=len({c.document_id for c in self.chunks if c.domain == d}),
                    chunk_count=sum(c.domain == d for c in self.chunks),
                    warnings=[],
                )
                for d in Domain
            },
        )


class FakeLLMClient:
    """Known fixture behavior, not an alternative general-purpose language model."""

    def __init__(self):
        self.calls: list[dict] = []

    async def generate_json(self, *, role: str, system: str, payload: dict, schema: dict) -> dict:
        self.calls.append({"role": role, "payload": copy.deepcopy(payload)})
        if role == "router":
            question = payload["question"]
            tasks = {}
            for domain in keyword_route(question):
                constraints = []
                requirements = payload["context"].get("requirements", [])
                raw = " ".join(requirements) or question
                if domain == Domain.CLINICAL and (match := re.search(r"(\d+)\s+samples", raw)):
                    constraints.append(
                        {
                            "id": "capacity",
                            "attribute": "samples_per_run",
                            "operator": "gte",
                            "expected_value": float(match[1]),
                            "unit": "samples/run",
                            "required": True,
                        }
                    )
                if domain == Domain.TECHNOLOGY and (match := re.search(r"(\d+)\s+ports", raw)):
                    constraints.append(
                        {
                            "id": "port_count",
                            "attribute": "port_count",
                            "operator": "gte",
                            "expected_value": float(match[1]),
                            "unit": "ports",
                            "required": True,
                        }
                    )
                tasks[domain.value] = {
                    "query": question,
                    "compare_candidates": bool(constraints),
                    "constraints": constraints,
                }
            return {"tasks": tasks, "clarification_questions": []}
        if role == "review":
            return {
                "accepted_finding_ids": [f["id"] for f in payload["findings"]],
                "accepted_recommendation_ids": [r["id"] for r in payload["recommendations"]],
                "limitations": [],
            }
        if role == "synthesis":
            recs = payload["recommendations"]
            first, rest, domains = [], [], set()
            for rec in recs:
                domain = rec["id"].split(":")[0]
                (first if domain not in domains else rest).append(rec["id"])
                domains.add(domain)
            return {
                "selected_finding_ids": [f["id"] for f in payload["findings"]],
                "selected_recommendation_ids": first + rest,
                "dependencies": [{"before_id": a, "after_id": b} for a, b in zip(first, first[1:])],
                "conflicting_finding_ids": [],
            }
        chunks = payload.get("chunks", [])
        findings, candidates, actions = [], [], []
        for chunk in chunks:
            # Fixtures deliberately have short self-contained statements.
            quote = chunk["text"].split(". ")[0].rstrip(".") + "."
            supports = [{"evidence_id": chunk["chunk_id"], "quote": quote}]
            findings.append({"statement": quote, "assessment": "supported", "supports": supports})
            if role == "clinical" and payload["task"].get("compare_candidates"):
                for label, capacity in re.findall(
                    r"(Fixture Analyzer [AB]) supports (\d+) samples per run", chunk["text"]
                ):
                    sentence = f"{label} supports {capacity} samples per run."
                    support = [{"evidence_id": chunk["chunk_id"], "quote": sentence}]
                    candidates.append(
                        {
                            "id": label.lower().replace(" ", "-"),
                            "label": label,
                            "identity_supports": support,
                            "attributes": {
                                "samples_per_run": {
                                    "value": float(capacity),
                                    "unit": "samples/run",
                                    "supports": support,
                                }
                            },
                        }
                    )
            if role == "technology" and payload["task"].get("compare_candidates"):
                candidates.append(
                    {
                        "id": "fixture-router-a",
                        "label": "Fixture Router A",
                        "identity_supports": supports,
                        "attributes": {
                            "port_count": {"value": 2, "unit": "ports", "supports": supports}
                        },
                    }
                )
            if role in {"sec", "corporate_responsibility"}:
                actions.append(
                    {
                        "kind": "business_action",
                        "title": f"Review the cited {role} priorities",
                        "rationale": "Use the cited disclosure as an input to the proposed planning review.",
                        "supports": supports,
                        "prerequisites": ["Validate priorities against local requirements."],
                        "next_steps": ["Assign an owner to review the cited risk or barrier."],
                        "success_metrics": [
                            "Number of cited priorities with an assigned owner and next action."
                        ],
                    }
                )
        return {
            "findings": findings,
            "candidates": candidates,
            "actions": actions,
            "clarification_questions": [],
            "follow_up_query": None,
            "limitations": [],
        }


class ScriptedLLMClient:
    """Queue results per role for failure/edge-case tests without paid model calls."""

    def __init__(self, scripts: dict[str, list], fallback=None):
        self.scripts = {role: deque(values) for role, values in scripts.items()}
        self.fallback = fallback or FakeLLMClient()
        self.calls = []

    async def generate_json(self, *, role, system, payload, schema):
        self.calls.append(role)
        queue = self.scripts.get(role)
        if queue:
            value = queue.popleft()
            if isinstance(value, Exception):
                raise value
            return copy.deepcopy(value)
        return await self.fallback.generate_json(
            role=role, system=system, payload=payload, schema=schema
        )


class FakeConsultingService:
    """C may inject this while integrating routes; always emits demo mode."""

    async def run_consultation(
        self, request: ConsultRequest, *, request_id: str
    ) -> ConsultResponse:
        from consulting_platform.agents.service import create_consulting_service
        from consulting_platform.config import Settings

        return await create_consulting_service(settings=Settings(mode="demo")).run_consultation(
            request,
            request_id=request_id,
        )
