"""Private validated model outputs; these are not additional HTTP payloads."""

from typing import Literal

from pydantic import Field, field_validator

from consulting_platform.contracts import ContractModel, Domain, Text, unique


class Support(ContractModel):
    evidence_id: Text
    quote: str = Field(min_length=1, max_length=400)


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


class DomainTask(ContractModel):
    query: Text
    compare_candidates: bool = False
    constraints: list[Constraint] = Field(default_factory=list)

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
    limitations: list[Text] = Field(default_factory=list)
