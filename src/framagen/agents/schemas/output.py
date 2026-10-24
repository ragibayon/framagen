from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ACSLClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "requires",
        "ensures",
        "assigns",
        "loop_invariant",
        "loop_assigns",
        "loop_variant",
        "assert",
        "other",
    ]
    text: str = Field(min_length=1)
    function: str = Field(min_length=1)
    evidence: str = Field(
        description="Relevant source/property evidence, or 'assumption'."
    )
    justification: str = Field(
        description="Why this clause is meaningful for the task."
    )


class CandidateSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description="Concise description of program and specification."
    )
    annotated_c: str = Field(
        description="Full original C source with ACSL comments only; no Markdown fences.",
        min_length=1,
    )
    clauses: list[ACSLClause] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class GoalOutcome(BaseModel):
    """One WP proof obligation, as reported by ``-wp-report-json``."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    property: str
    function: str | None = None
    line: int | None = None
    smoke: bool = False
    # For proof goals: the goal was discharged. For smoke tests: the
    # inconsistency probe was NOT provable, i.e. the context is not vacuous.
    passed: bool
    verdict: str


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_file: str
    status: Literal[
        "proved",
        "partially_proved",
        "unresolved",
        "vacuous",
        "parse_error",
        "rejected",
        "verification_error",
        "not_checked",
    ]
    # Normalized outcome, aligned with SpecPylot/AutoReSpec reporting:
    # VERIFIED = every goal proved and no smoke test failed;
    # FAILED = parse error, rejected edit, vacuous spec, or a refuted goal;
    # INCONCLUSIVE = some goal unknown/timeout; ERROR = tooling failure.
    verdict: Literal["VERIFIED", "FAILED", "INCONCLUSIVE", "ERROR", "NOT_CHECKED"]
    parse_exit_code: int | None = None
    wp_exit_code: int | None = None
    proved_goals: int | None = None
    total_goals: int | None = None
    timed_out_goals: int | None = None
    remaining_goals: int | None = None
    smoke_tests_total: int | None = None
    smoke_tests_failed: list[str] = Field(default_factory=list)
    unproved_goals: list[str] = Field(default_factory=list)
    goals: list[GoalOutcome] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    frama_c_image: str
    frama_c_backend: str = "docker"
    prover: str
    prover_timeout_seconds: int
    wall_timeout_seconds: int
    rte: bool = True
    machdep: str | None = None
    check_termination: bool = True
    smoke_tests: bool = True
    smoke_dead_code: bool = True
