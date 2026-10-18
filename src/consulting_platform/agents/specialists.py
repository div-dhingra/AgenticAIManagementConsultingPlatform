"""Four separately compiled domain graphs sharing bounded, testable node code."""

from __future__ import annotations

import asyncio
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from consulting_platform.agents.csr_actions import qualify_csr_actions
from consulting_platform.agents.evidence_spans import evidence_spans
from consulting_platform.agents.grounding import (
    filter_evidence,
    ground_analysis,
    grounded_candidates,
    normalize_company,
)
from consulting_platform.agents.llm import LLMClient, RequestBudget, generate
from consulting_platform.agents.models import DomainTask, RankedCandidate, SpecialistAnalysis
from consulting_platform.agents.port_tables import canonical_port_task, table_candidates
from consulting_platform.agents.prompts import specialist_prompt
from consulting_platform.agents.questions import subject_patterns, unrequested_model_detail
from consulting_platform.agents.ranking import rank_candidates
from consulting_platform.agents.reliability import LABELS, source_request
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
    retrieval_warnings: list[str]
    unavailable: bool
    failure: Exception | None
    table_proven_candidates: list[str]
    answer_chunks: dict[str, list[str]]


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
            searches = [(state["query"], state["filters"])]
            for requirement in state["task"].answer_requirements:
                scoped = state["filters"].model_copy(deep=True)
                if requirement.company:
                    company = normalize_company(requirement.company, self.domain)
                    if not scoped.companies or company in {
                        normalize_company(c, self.domain) for c in scoped.companies
                    }:
                        scoped.companies = [company]
                if requirement.report_year and (
                    not scoped.report_years or requirement.report_year in scoped.report_years
                ):
                    scoped.report_years = [requirement.report_year]
                query = " ".join(
                    str(part)
                    for part in (requirement.company, requirement.report_year, requirement.question)
                    if part
                )
                searches.append((query, scoped))
            if self.domain == Domain.SEC:
                searches = [
                    (
                        query + " Item 1A Risk Factors"
                        if re.search(r"\brisks?\b", query, re.I)
                        and not re.search(r"governance|manage cybersecurity", query, re.I)
                        else query,
                        filters,
                    )
                    for query, filters in searches
                ]
            # Bounded per-question searches share the same local index. No web
            # browsing, new model calls, or relaxed customer filters are involved.
            responses = [
                RetrievalResult.model_validate(r)
                for r in await asyncio.gather(
                    *(
                        self.knowledge.retrieve(query, [self.domain], filters=filters, top_k=5)
                        for query, filters in searches[:9]
                    )
                )
            ]
            if any(r.mode != self.mode for r in responses):
                raise ServiceUnavailableError("Knowledge and consulting modes do not match.")
            combined = {}
            # Reserve passages for each subquestion before taking second hits.
            for position in range(5):
                for response in responses:
                    if position < len(response.chunks):
                        chunk = response.chunks[position]
                        previous = combined.get(chunk.chunk_id)
                        if previous and previous.model_dump(
                            exclude={"retrieval_score"}
                        ) != chunk.model_dump(exclude={"retrieval_score"}):
                            raise ServiceUnavailableError(
                                "Knowledge service reused an evidence ID for different text."
                            )
                        combined.setdefault(chunk.chunk_id, chunk)
            limit = 10 if state["task"].answer_requirements else 5
            selected = list(combined.values())[:limit]
            expand = getattr(self.knowledge, "expand_context", None)
            if expand and state["task"].answer_requirements:
                selected.extend(await expand(selected, max_chunks=4))
            result = RetrievalResult(
                mode=self.mode,
                status="unavailable" if all(r.status == "unavailable" for r in responses) else "ok",
                chunks=selected,
                warnings=list(dict.fromkeys(w for r in responses for w in r.warnings)),
            )
        except (ServiceUnavailableError, UpstreamTimeoutError) as exc:
            return {
                "rounds": state.get("rounds", 0) + 1,
                "unavailable": True,
                "failure": exc,
                "warnings": [
                    *state.get("retrieval_warnings", []),
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
        limit = 14 if state["task"].answer_requirements else 5
        if len(chunks) > limit:
            chunks = chunks[:limit]
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
        retrieval_warnings = list(
            dict.fromkeys([*state.get("retrieval_warnings", []), *result.warnings, *rejected])
        )
        return {
            "rounds": state.get("rounds", 0) + 1,
            "answer_chunks": {
                requirement.id: [c.chunk_id for c in response.chunks if c.chunk_id in registry]
                for requirement, response in zip(state["task"].answer_requirements, responses[1:])
            },
            "chunks": list(registry.values()),
            "unavailable": unavailable,
            "warnings": retrieval_warnings,
            "retrieval_warnings": retrieval_warnings,
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
        context = state["request"].context.model_dump(mode="json")
        context["company"] = (
            state["filters"].companies[0] if len(state["filters"].companies) == 1 else None
        )
        context["report_years"] = state["filters"].report_years
        if not state["task"].compare_candidates:
            context["requirements"] = []
        if self.domain not in {Domain.TECHNOLOGY, Domain.CLINICAL}:
            context.update(product_model=None, symptoms=[])
        companies = state["filters"].companies if self.domain == Domain.SEC else []
        batches = [(company, None) for company in companies] if len(companies) > 1 else [(None, None)]
        if self.domain == Domain.CLINICAL and not state["task"].compare_candidates and len(state["task"].answer_requirements) > 1:
            requirements = state["task"].answer_requirements
            batches = [(None, requirements[::2]), (None, requirements[1::2])]

        async def analyze_batch(company, focused):
            task = state["task"]
            chunks = state["chunks"]
            batch_context = dict(context)
            system = specialist_prompt(self.domain)
            if company:
                chunks = [
                    c
                    for c in chunks
                    if normalize_company(c.company or "", self.domain)
                    == normalize_company(company, self.domain)
                ]
                if not chunks:
                    return SpecialistAnalysis(missing_evidence=[f"No passages for {company}."])
                requirements = [
                    r
                    for r in task.answer_requirements
                    if not r.company
                    or normalize_company(r.company, self.domain)
                    == normalize_company(company, self.domain)
                ]
                query = " ".join(r.question for r in requirements) or task.query
                task = task.model_copy(
                    update={
                        "query": f"Assess ONLY {company}. {query}",
                        "answer_requirements": requirements,
                    }
                )
                batch_context["company"] = company
                system += f"\nThis batch covers ONLY {company}. Return at most four atomic findings about that company; do not assess other companies."
            if focused:
                requested_ids = {c for r in focused for c in state.get("answer_chunks", {}).get(r.id, [])}
                if requested_ids:
                    chunks = [c for c in chunks if c.chunk_id in requested_ids]
                patterns = [p for r in focused for p in subject_patterns(r.question)]
                if patterns:
                    title_matches = [c for c in chunks if any(re.search(p, c.title, re.I) for p in patterns)]
                    chunks = title_matches or [c for c in chunks if any(re.search(p, c.text, re.I) for p in patterns)]
                task = task.model_copy(update={"query": " ".join(r.question for r in focused), "answer_requirements": focused})
                if not chunks:
                    return SpecialistAnalysis(missing_evidence=[task.query])
                system += "\nAnswer ONLY the questions in this batch. For a 'what do these instruments measure/do' question, provide their core function, not unrequested submodel features, brand comparisons or marketing claims. Prefer one or two concise findings, combining a directly documented list of measurements when appropriate. Do not replace this subject with another product family's description."
            intent = " ".join([task.query, *[r.question for r in task.answer_requirements]])
            allow_actions = not task.compare_candidates and bool(
                re.search(
                    r"\b(actions?|steps?|recommend|propose|proposed|plan|strategy|interventions?|hypothetical|pilot|could)\b",
                    intent,
                    re.I,
                )
            )
            try:
                analysis = await generate(
                    self.llm,
                    state["budget"],
                    SpecialistAnalysis,
                    role=self.domain.value,
                    system=system,
                    payload={
                        "domain": self.domain.value,
                        "question": task.query,
                        "context": batch_context,
                        "requested_companies": [company] if company else state["filters"].companies,
                        "task": task.model_dump(mode="json"),
                        "allow_actions": allow_actions,
                        "chunks": [
                            {**c.model_dump(mode="json"), "quote_spans": evidence_spans(c.text)}
                            for c in chunks
                        ],
                        "retrieval_round": state["rounds"],
                    },
                )
                if focused:
                    analysis = analysis.model_copy(update={"findings": [f for f in analysis.findings if not unrequested_model_detail(f.statement, [r.question for r in focused], chunks)]})
                return (
                    analysis.model_copy(update={"findings": analysis.findings[:4]})
                    if company or focused
                    else analysis
                )
            except (UpstreamModelError, ServiceUnavailableError, UpstreamTimeoutError) as exc:
                return exc

        outputs = await asyncio.gather(*(analyze_batch(company, focused) for company, focused in batches))
        analyses = [item for item in outputs if isinstance(item, SpecialistAnalysis)]
        failures = [item for item in outputs if isinstance(item, Exception)]
        merged = SpecialistAnalysis(
            findings=[f for item in analyses for f in item.findings],
            candidates=[c for item in analyses for c in item.candidates],
            actions=[a for item in analyses for a in item.actions],
            clarification_questions=[q for item in analyses for q in item.clarification_questions],
            missing_evidence=[q for item in analyses for q in item.missing_evidence],
            limitations=[q for item in analyses for q in item.limitations],
            follow_up_query=next(
                (item.follow_up_query for item in analyses if item.follow_up_query), None
            ),
        )
        return {
            "analysis": merged,
            "failure": failures[0] if failures and not analyses else None,
            "warnings": [
                *state.get("warnings", []),
                *(
                    [
                        f"{self.domain.value} model analysis failed for {len(failures)} scoped batch(es)."
                    ]
                    if failures
                    else []
                ),
            ],
            "trace": [
                *state.get("trace", []),
                *[
                    TraceEvent(
                        step="validate",
                        domain=self.domain,
                        status="failed",
                        detail=f"Specialist diagnostic: {type(exc).__name__}: {exc}",
                    )
                    for exc in failures
                ],
            ],
        }

    async def _validate(self, state: SpecialistState) -> dict:
        analysis, warnings = ground_analysis(state["analysis"], state.get("chunks", []))
        task = state["task"]
        extraction_notes = []
        table_proven = set()
        if self.domain == Domain.TECHNOLOGY and task.compare_candidates:
            original_task = task
            task = canonical_port_task(task, state["request"].context.requirements)
            explicit_port_criteria = task is not original_task
            candidates, task, extraction_notes = table_candidates(
                state.get("chunks", []),
                task,
                analysis.candidates,
                state["request"].question,
                proven=table_proven if explicit_port_criteria else None,
            )
            candidates, rejected = grounded_candidates(
                candidates, {c.chunk_id: c for c in state.get("chunks", [])}
            )
            warnings.extend(rejected)
            analysis = analysis.model_copy(update={"candidates": candidates})
        if self.domain == Domain.CORPORATE_RESPONSIBILITY:
            analysis, csr_notes = qualify_csr_actions(analysis, state["request"])
            extraction_notes.extend(csr_notes)
        source_questions = [q for q in analysis.clarification_questions if source_request(q)]
        analysis = analysis.model_copy(
            update={
                "clarification_questions": [
                    q for q in analysis.clarification_questions if not source_request(q)
                ],
                "missing_evidence": [*analysis.missing_evidence, *source_questions],
            }
        )
        notes = [*warnings, *analysis.limitations, *analysis.missing_evidence, *extraction_notes]
        if state["task"].compare_candidates and analysis.actions:
            notes.append(
                "Comparison actions were excluded; only the validated ranker may propose product options."
            )
            analysis = analysis.model_copy(update={"actions": []})
        return {
            "analysis": analysis,
            "task": task,
            "table_proven_candidates": sorted(table_proven),
            # Generated notes are scoped diagnostics, not global coverage facts.
            "warnings": [
                *state.get("warnings", []),
                *(
                    [
                        f"{LABELS[self.domain]}: Some generated content failed evidence validation and was excluded; see evidence check details."
                    ]
                    if warnings
                    else []
                ),
                *(
                    [
                        f"{LABELS[self.domain]}: The specialist reported analysis limitations; see evidence check details."
                    ]
                    if analysis.limitations
                    else []
                ),
            ],
            "trace": [
                *state.get("trace", []),
                *[
                    TraceEvent(
                        step="validate",
                        domain=self.domain,
                        status="skipped",
                        detail=f"Specialist diagnostic: {note}",
                    )
                    for note in notes
                ],
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
        warnings = list(state.get("warnings", []))
        questions = list(state["analysis"].clarification_questions)
        if compare and not state["task"].constraints:
            if state["request"].context.requirements or re.search(
                r"\b(?:at least|minimum)\b|\b\d+\s+(?:\d+[gGmM]\s+)?(?:Ethernet\s+)?(?:ports?|samples?)\b",
                state["request"].question,
                re.I,
            ):
                warnings.append(
                    f"{LABELS[self.domain]}: Stated requirements could not be converted into a validated comparison; no product is recommended."
                )
            else:
                questions.append(
                    "What specific requirements should the equipment comparison evaluate?"
                )
        elif compare and (not rankings or any(r.eligibility == "conditional" for r in rankings)):
            warnings.append(
                f"{LABELS[self.domain]}: Required candidate attributes could not all be validated; unresolved candidates are not recommended."
            )
        return {
            "rankings": rankings,
            "warnings": warnings,
            "analysis": state["analysis"].model_copy(update={"clarification_questions": questions}),
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
