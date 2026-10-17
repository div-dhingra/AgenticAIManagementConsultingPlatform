"""Four separately compiled domain graphs sharing bounded, testable node code."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from consulting_platform.agents.grounding import filter_evidence, ground_analysis
from consulting_platform.agents.llm import LLMClient, RequestBudget, generate
from consulting_platform.agents.models import DomainTask, RankedCandidate, SpecialistAnalysis
from consulting_platform.agents.prompts import specialist_prompt
from consulting_platform.agents.ranking import rank_candidates
from consulting_platform.contracts import (
    ConsultRequest,
    Domain,
    EvidenceChunk,
    KnowledgeService,
    Mode,
    RetrievalFilters,
    RetrievalResult,
    ServiceUnavailableError,
    TraceEvent,
    UpstreamModelError,
    UpstreamTimeoutError,
)


class SpecialistState(TypedDict, total=False):
    request: ConsultRequest
    task: DomainTask
    query: str
    filters: RetrievalFilters
    budget: RequestBudget
    rounds: int
    chunks: list[EvidenceChunk]
    analysis: SpecialistAnalysis
    rankings: list[RankedCandidate]
    trace: list[TraceEvent]
    warnings: list[str]
    unavailable: bool
    failure: Exception | None


class SpecialistAgent:
    def __init__(self, domain: Domain, knowledge: KnowledgeService, llm: LLMClient, mode: Mode):
        self.domain, self.knowledge, self.llm, self.mode = domain, knowledge, llm, mode
        builder = StateGraph(SpecialistState)
        for name, node in [
            ("retrieve", self._retrieve),
            ("analyze_extract", self._analyze),
            ("validate_evidence", self._validate),
            ("rank_candidates", self._rank),
        ]:
            builder.add_node(name, node)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "analyze_extract")
        builder.add_edge("analyze_extract", "validate_evidence")
        builder.add_edge("validate_evidence", "rank_candidates")
        builder.add_edge("rank_candidates", END)
        self.graph = builder.compile(name=f"{domain.value}_specialist")

    async def _retrieve(self, state: SpecialistState) -> dict:
        if state.get("rounds", 0) >= 2:
            return {"warnings": [*state.get("warnings", []), "Retrieval round limit reached."]}
        try:
            result = RetrievalResult.model_validate(
                await self.knowledge.retrieve(
                    state["query"],
                    [self.domain],
                    filters=state["filters"],
                    top_k=5,
                )
            )
        except (ServiceUnavailableError, UpstreamTimeoutError) as exc:
            return {
                "rounds": state.get("rounds", 0) + 1,
                "unavailable": True,
                "failure": exc,
                "warnings": [
                    *state.get("warnings", []),
                    f"{self.domain.value} retrieval unavailable.",
                ],
                "trace": [
                    *state.get("trace", []),
                    TraceEvent(
                        step="retrieve",
                        domain=self.domain,
                        status="failed",
                        detail="Knowledge service unavailable.",
                    ),
                ],
            }
        if result.mode != self.mode:
            raise ServiceUnavailableError("Knowledge and consulting modes do not match.")
        chunks, rejected = filter_evidence(result.chunks, self.domain, state["filters"])
        if len(chunks) > 5:
            chunks = chunks[:5]
            rejected.append("Knowledge service exceeded top_k; excess chunks were discarded.")
        registry = {c.chunk_id: c for c in state.get("chunks", [])}
        for chunk in chunks:
            if chunk.chunk_id in registry and registry[chunk.chunk_id].model_dump(
                exclude={"retrieval_score"}
            ) != chunk.model_dump(exclude={"retrieval_score"}):
                raise ServiceUnavailableError(
                    "Knowledge service reused an evidence ID for different text."
                )
            registry[chunk.chunk_id] = chunk
        unavailable = result.status == "unavailable" and not registry
        return {
            "rounds": state.get("rounds", 0) + 1,
            "chunks": list(registry.values()),
            "unavailable": unavailable,
            "warnings": [*state.get("warnings", []), *result.warnings, *rejected],
            "trace": [
                *state.get("trace", []),
                TraceEvent(
                    step="retrieve",
                    domain=self.domain,
                    status="failed" if unavailable else "completed",
                    detail=f"Retrieved {len(chunks)} allowed chunks in round {state.get('rounds', 0) + 1}.",
                ),
            ],
        }

    async def _analyze(self, state: SpecialistState) -> dict:
        if state.get("unavailable") or not state.get("chunks"):
            return {"analysis": SpecialistAnalysis(), "rankings": []}
        try:
            analysis = await generate(
                self.llm,
                state["budget"],
                SpecialistAnalysis,
                role=self.domain.value,
                system=specialist_prompt(self.domain),
                payload={
                    "domain": self.domain.value,
                    "question": state["request"].question,
                    "context": state["request"].context.model_dump(mode="json"),
                    "task": state["task"].model_dump(mode="json"),
                    "chunks": [c.model_dump(mode="json") for c in state["chunks"]],
                    "retrieval_round": state["rounds"],
                },
            )
            return {"analysis": analysis}
        except (UpstreamModelError, ServiceUnavailableError, UpstreamTimeoutError) as exc:
            return {
                "analysis": SpecialistAnalysis(),
                "failure": exc,
                "warnings": [
                    *state.get("warnings", []),
                    f"{self.domain.value} model analysis failed.",
                ],
            }

    async def _validate(self, state: SpecialistState) -> dict:
        analysis, warnings = ground_analysis(state["analysis"], state.get("chunks", []))
        return {
            "analysis": analysis,
            "warnings": [*state.get("warnings", []), *warnings, *analysis.limitations],
            "trace": [
                *state.get("trace", []),
                TraceEvent(
                    step="validate",
                    domain=self.domain,
                    status="failed" if state.get("failure") else "completed",
                    detail="Checked evidence identities, verbatim quotes, and extracted attribute values.",
                ),
            ],
        }

    async def _rank(self, state: SpecialistState) -> dict:
        compare = state["task"].compare_candidates
        rankings = (
            rank_candidates(
                state["analysis"].candidates,
                state["task"].constraints,
                top_k=20,
            )
            if compare and state["analysis"].candidates
            else []
        )
        return {
            "rankings": rankings,
            "trace": [
                *state.get("trace", []),
                TraceEvent(
                    step="rank",
                    domain=self.domain,
                    status="completed" if rankings else "skipped",
                    detail=f"Compared {len(rankings)} candidates."
                    if rankings
                    else "No supported candidate comparison was available or requested.",
                ),
            ],
        }
