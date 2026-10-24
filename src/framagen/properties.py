"""Target properties: what a task must prove, and whether WP proved exactly that.

A WP run that reports "all goals proved" is not yet an answer to the
benchmark question. For SV-COMP ``unreach-call`` the question is whether
every ``reach_error()`` call is unreachable, which the sv2acsl harness turns
into the call-site preconditions of ``reach_error`` and ``__VERIFIER_assert``.
:func:`assess_property` maps each property site in the source to its WP
goal(s), then decides:

- ``verified``: every site's goal is proved, every annotation those proofs
  rely on (loop invariants, asserts, assigns) is proved, smoke tests pass;
- ``conditional``: every site's goal is proved, but only under some
  unproved annotation;
- ``unproved``: some site's goal is not proved;
- ``vacuous``: a smoke test shows the annotations are inconsistent;
- ``missing-goals``: a site has no WP goal (the encoding did not apply);
- ``not-checked``: no WP result (parse error, rejected candidate, tool error).

It also compares the outcome with the benchmark's expected verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from framagen.agents.schemas.output import VerificationResult
from framagen.benchmarks.sv2acsl import HARNESS_END
from framagen.utils.c_tokens import iter_tokens

PropertyStatus = Literal["verified", "conditional", "unproved", "vacuous", "missing-goals", "not-checked"]


@dataclass(frozen=True)
class PropertySpec:
    name: str
    title: str
    # Shown to the agent: what must be proved, in plain words.
    description: str
    # Formal statement, e.g. the SV-COMP .prp text.
    formal: str | None
    # Calls whose preconditions encode the property (empty: every goal counts).
    site_functions: tuple[str, ...]


PROPERTIES: dict[str, PropertySpec] = {
    "unreach-call": PropertySpec(
        name="unreach-call",
        title="SV-COMP unreach-call: reach_error() is never called",
        description=(
            "No execution starting at main() calls reach_error(). Through the sv2acsl harness this "
            "means: every __VERIFIER_assert(c) call has c != 0 whenever it is reached, and every "
            "direct reach_error() call is unreachable. These call-site preconditions are the target "
            "goals; the loop annotations you write are supporting lemmas that must also be proved."
        ),
        formal="CHECK( init(main()), LTL(G ! call(reach_error())) )",
        site_functions=("__VERIFIER_assert", "reach_error"),
    ),
    "contract": PropertySpec(
        name="contract",
        title="Functional contract",
        description=(
            "Each function satisfies a contract you write (requires/assigns/ensures) that describes "
            "what it computes; with runtime-error checking on, the code is also free of undefined "
            "behavior (overflow, invalid memory access, division by zero). Every WP goal is a target."
        ),
        formal=None,
        site_functions=(),
    ),
}


class PropertySite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: str  # callee: __VERIFIER_assert / reach_error
    line: int
    goals: list[str] = Field(default_factory=list)
    proved: bool = False


class PropertyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property: str
    status: PropertyStatus
    sites: list[PropertySite] = Field(default_factory=list)
    target_goals_proved: int = 0
    target_goals_total: int = 0
    unproved_support_goals: list[str] = Field(default_factory=list)
    expected_verdict: bool | None = None
    # True: our outcome agrees with the benchmark's expected verdict.
    # None: inconclusive (an unproved goal is not a counterexample).
    matches_expected: bool | None = None
    summary: str = ""


def property_sites(source: str, spec: PropertySpec) -> list[tuple[str, int]]:
    """Return (callee, line) for each call to a site function after the harness."""
    if not spec.site_functions:
        return []
    start = source.find(HARNESS_END)
    offset = start + len(HARNESS_END) if start >= 0 else 0
    tokens = [t for t in iter_tokens(source) if t.start >= offset]
    sites = []
    for i, tok in enumerate(tokens[:-1]):
        if tok.kind == "word" and tok.text in spec.site_functions and tokens[i + 1].text == "(":
            # Skip declarations/definitions: a call is not preceded by a type name.
            prev = tokens[i - 1].text if i else ""
            if prev in {"void", "int", "extern"}:
                continue
            sites.append((tok.text, source.count("\n", 0, tok.start) + 1))
    return sites


_CALL_GOAL = re.compile(r"_call_(?P<callee>__VERIFIER_assert|reach_error)(?:_\d+)?_.*requires")


def assess_property(
    spec: PropertySpec,
    candidate: str,
    result: VerificationResult | None,
    *,
    expected_verdict: bool | None = None,
) -> PropertyAssessment:
    """Assess ``result`` for ``candidate`` (the annotated file WP checked; its line numbers)."""
    sites = [PropertySite(function=f, line=line) for f, line in property_sites(candidate, spec)]
    base = dict(property=spec.name, sites=sites, expected_verdict=expected_verdict)
    if result is None or not result.goals or result.status in {"parse_error", "rejected", "verification_error", "not_checked"}:
        return PropertyAssessment(status="not-checked", summary="no WP result for this candidate", **base)

    proof_goals = [g for g in result.goals if not g.smoke]
    if spec.site_functions:
        targets = [g for g in proof_goals if _CALL_GOAL.search(g.goal)]
        support = [g for g in proof_goals if not _CALL_GOAL.search(g.goal)]
        for site in sites:
            matched = [
                g for g in targets
                if g.line == site.line and _CALL_GOAL.search(g.goal).group("callee") == site.function
            ]
            site.goals = [g.goal for g in matched]
            site.proved = bool(matched) and all(g.passed for g in matched)
    else:
        targets, support = proof_goals, []

    proved_targets = sum(g.passed for g in targets)
    unproved_support = [g.goal for g in support if not g.passed]
    missing = [s for s in sites if not s.goals]
    if result.smoke_tests_failed:
        status: PropertyStatus = "vacuous"
    elif missing:
        status = "missing-goals"
    elif not targets or proved_targets < len(targets):
        status = "unproved"
    elif unproved_support:
        status = "conditional"
    else:
        status = "verified"

    matches: bool | None
    if expected_verdict is None:
        matches = None
    elif status == "verified":
        matches = expected_verdict is True
    elif status == "vacuous":
        matches = False
    else:
        matches = None  # no proof, but no counterexample either

    if spec.site_functions:
        done = sum(s.proved for s in sites)
        summary = f"{done}/{len(sites)} property sites proved"
    else:
        summary = f"{proved_targets}/{len(targets)} goals proved"
    if unproved_support:
        summary += f"; {len(unproved_support)} supporting annotation goal(s) unproved"
    if missing:
        summary += f"; no WP goal for sites at lines {[s.line for s in missing]}"
    if result.smoke_tests_failed:
        summary += "; smoke tests failed (vacuous annotations)"
    return PropertyAssessment(
        status=status,
        target_goals_proved=proved_targets,
        target_goals_total=len(targets),
        unproved_support_goals=unproved_support,
        matches_expected=matches,
        summary=summary,
        **base,
    )


def prompt_context(spec: PropertySpec, source: str) -> dict[str, object]:
    """Variables for the user prompt describing the target property."""
    return {
        "property_name": spec.name,
        "property_title": spec.title,
        "property_description": spec.description,
        "property_formal": spec.formal,
        "property_sites": [f"line {line}: {fn}(...)" for fn, line in property_sites(source, spec)],
    }
