"""Checks that a generated candidate only adds legitimate ACSL to the task.

A WP proof is only meaningful if the candidate (1) is the same C program,
(2) keeps the property encoding supplied by the benchmark, and (3) does not
smuggle in assumptions. Vacuous-but-legal specifications (e.g. ``requires
\\false``) are caught later by WP smoke tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from framagen.benchmarks.sv2acsl import harness_block
from framagen.utils.c_tokens import acsl_annotations, same_executable_tokens

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*\n(?P<body>.*?)\n\s*```\s*$", re.DOTALL)
# `admit` turns a clause into an unchecked assumption; ghost code can change
# control flow seen by WP. Neither is allowed in generated annotations.
_ALWAYS_FORBIDDEN = {"admit": r"\badmit\b", "ghost": r"\bghost\b"}
_AXIOM = r"\baxiom\b"


@dataclass
class GuardReport:
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def normalize_candidate(text: str) -> tuple[str, list[str]]:
    """Remove a Markdown code fence that models often add despite instructions."""
    match = _FENCE.match(text)
    if match:
        return match.group("body") + "\n", ["removed Markdown code fence"]
    return text, []


def check_candidate(original: str, candidate: str, *, allow_axioms: bool) -> GuardReport:
    report = GuardReport()
    if not same_executable_tokens(original, candidate):
        report.violations.append(
            "Executable C tokens differ from the original; only ACSL comments may be added "
            "(do not reformat braces, merge statements, or edit declarations)."
        )

    original_annotations = acsl_annotations(original)
    candidate_annotations = acsl_annotations(candidate)
    if not _is_subsequence(original_annotations, candidate_annotations):
        report.violations.append(
            "An ACSL annotation present in the task was modified or removed; keep existing annotations verbatim."
        )

    fence = harness_block(original)
    if fence is not None and harness_block(candidate) != fence:
        report.violations.append(
            "The sv2acsl harness block (between the 'sv2acsl harness' markers) was changed; it encodes "
            "the benchmark property and must be kept verbatim."
        )

    added = "\n".join(a for a in candidate_annotations if a not in set(original_annotations))
    for name, pattern in _ALWAYS_FORBIDDEN.items():
        if re.search(pattern, added):
            report.violations.append(f"Forbidden ACSL construct '{name}' in generated annotations.")
    if re.search(_AXIOM, added):
        if allow_axioms:
            report.notes.append("candidate introduces axioms (trusted, not proved)")
        else:
            report.violations.append("Axioms are not allowed for this task; use lemmas or inline definitions.")
    return report


def _is_subsequence(needles: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(item == candidate for candidate in it) for item in needles)
