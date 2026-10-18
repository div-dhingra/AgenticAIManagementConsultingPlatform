"""Private validated model outputs; these are not additional HTTP payloads."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from consulting_platform.contracts import ContractModel, Domain, Text, unique


class Support(ContractModel):
    evidence_id: Text
    quote: str = Field(default="", max_length=400)
    span_id: str | None = None

    @model_validator(mode="after")
    def require_source_reference(self):
        if not self.quote.strip() and not self.span_id:
            raise ValueError("Supply a quote or a source span_id")
        return self


class AttributeFact(ContractModel):
    value: str | float | bool
    unit: str | None = None
    supports: list[Support] = Field(min_length=1)


class Constraint(ContractModel):
    id: Text
    attribute: Text
    operator: Literal["eq", "gte", "lte", "contains"]
    expected_value: str | float | bool
    unit: str | None = None
    required: bool = True
    weight: float = Field(default=1, gt=0, allow_inf_nan=False)


class Candidate(ContractModel):
    id: Text
    label: Text
    attributes: dict[str, AttributeFact]
    identity_supports: list[Support] = Field(min_length=1)


class RankedCandidate(ContractModel):
    id: Text
    label: Text
    rank: int
    score: float
    eligibility: Literal["eligible", "conditional", "excluded"]
    matched_constraint_ids: list[str]
    unknown_constraint_ids: list[str]
    violated_constraint_ids: list[str]
    evidence_ids: list[str]
    explanation: str


class AnswerRequirement(ContractModel):
    """One independently answerable part of the customer's question."""

    id: Text
    question: Text
    company: str | None = None
    report_year: int | None = None


class CoverageCheck(ContractModel):
    requirement_id: Text
    finding_ids: list[str] = Field(default_factory=list)
    recommendation_ids: list[str] = Field(default_factory=list)
    status: Literal["answered", "unresolved"] = "unresolved"
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def legacy_answered_flag(cls, value):
        if isinstance(value, dict) and "answered" in value:
            value = dict(value)
            answered = value.pop("answered")
            value.setdefault("status", "answered" if answered is True else "unresolved")
        return value

    @property
    def answered(self):
        return self.status == "answered"


class DomainTask(ContractModel):
    query: Text
    compare_candidates: bool = False
    constraints: list[Constraint] = Field(default_factory=list)
    answer_requirements: list[AnswerRequirement] = Field(default_factory=list, max_length=8)

    @field_validator("constraints")
    @classmethod
    def unique_constraint_ids(cls, value):
        unique([constraint.id for constraint in value])
        return value


class RoutePlan(ContractModel):
    tasks: dict[Domain, DomainTask]
    clarification_questions: list[Text] = Field(default_factory=list)


class FindingDraft(ContractModel):
    statement: Text
    assessment: Literal["supported", "inference"] = "supported"
    supports: list[Support] = Field(min_length=1)


class ActionDraft(ContractModel):
    kind: Literal["business_action", "diagnostic_action"]
    title: Text
    rationale: Text
    priority: Literal["high", "medium", "low"] = "medium"
    supports: list[Support] = Field(min_length=1)
    prerequisites: list[Text] = Field(default_factory=list)
    next_steps: list[Text] = Field(min_length=1)
    success_metrics: list[Text] = Field(min_length=1)


class SpecialistAnalysis(ContractModel):
    findings: list[FindingDraft] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    actions: list[ActionDraft] = Field(default_factory=list)
    clarification_questions: list[Text] = Field(default_factory=list)
    follow_up_query: str | None = None
    missing_evidence: list[Text] = Field(default_factory=list)
    limitations: list[Text] = Field(default_factory=list)


class RecommendationDependency(ContractModel):
    before_id: Text
    after_id: Text


class SynthesisPlan(ContractModel):
    # Select/reorder validated content. Do not let final prose introduce new facts.
    selected_finding_ids: list[str]
    selected_recommendation_ids: list[str]
    dependencies: list[RecommendationDependency] = Field(default_factory=list)
    conflicting_finding_ids: list[str] = Field(default_factory=list)


class EvidenceReview(ContractModel):
    accepted_finding_ids: list[str]
    accepted_recommendation_ids: list[str]
    accepted_clarification_ids: list[str] = Field(default_factory=list)
    limitations: list[Text] = Field(default_factory=list)
    coverage: list[CoverageCheck] = Field(default_factory=list)
    unanswered_questions: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def covered_answers_reference_accepted_content(self):
        for check in self.coverage:
            if check.answered and (
                not (check.finding_ids or check.recommendation_ids)
                or not set(check.finding_ids).issubset(self.accepted_finding_ids)
                or not set(check.recommendation_ids).issubset(self.accepted_recommendation_ids)
            ):
                raise ValueError("Answered coverage must reference accepted content only")
        return self
