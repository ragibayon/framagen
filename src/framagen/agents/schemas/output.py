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


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_file: str
    status: Literal[
        "proved",
        "partially_proved",
        "unresolved",
        "parse_error",
        "verification_error",
        "not_checked",
    ]
    parse_exit_code: int | None = None
    wp_exit_code: int | None = None
    proved_goals: int | None = None
    total_goals: int | None = None
    timed_out_goals: int | None = None
    remaining_goals: int | None = None
    diagnostics: list[str] = Field(default_factory=list)
    frama_c_image: str
    prover: str
    prover_timeout_seconds: int
    wall_timeout_seconds: int
