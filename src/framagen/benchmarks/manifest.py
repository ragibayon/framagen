from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from framagen.utils.c_tokens import loop_shape, strip_acsl
from framagen.utils.frama_c import VerifyOptions

ControlFlow = Literal[
    "sequential",
    "branched",
    "pointer",
    "single-path-loop",
    "multi-path-loop",
    "nested-loop",
]
CONTROL_FLOWS: tuple[str, ...] = ControlFlow.__args__  # type: ignore[attr-defined]

MANIFEST_NAME = "manifest.jsonl"


class VerifySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rte: bool = True
    machdep: str | None = None
    check_termination: bool = True
    smoke_dead_code: bool = True

    def to_options(self, **overrides: object) -> VerifyOptions:
        return VerifyOptions(
            rte=self.rte,
            machdep=self.machdep,
            check_termination=self.check_termination,
            smoke_dead_code=self.smoke_dead_code,
            **overrides,  # type: ignore[arg-type]
        )


class BenchmarkTask(BaseModel):
    """One verification task, with enough provenance to reproduce it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    suite: Literal["svcomp", "frama-c-problems", "custom"]
    category: str
    # What the LLM must produce: only loop annotations for a fixed property
    # (SV-COMP), or complete function contracts (frama-c-problems).
    goal: Literal["loop-annotations", "full-contract"]
    upstream_repo: str
    upstream_ref: str
    upstream_path: str
    upstream_sha256: str
    task_file: str  # relative to the benchmark root; the agent's input
    reference_file: str | None = None  # ground-truth ACSL, never shown to the agent
    property_file: str | None = None  # SV-COMP .prp, relative to benchmark root
    # The property the task targets (see framagen.properties.PROPERTIES), and
    # the benchmark's expected verdict for it (SV-COMP; None if unknown).
    target_property: str = "contract"
    expected_verdict: bool | None = None
    verify: VerifySettings = Field(default_factory=VerifySettings)
    control_flow: ControlFlow
    loc: int
    loops: int
    max_loop_depth: int
    harness_baseline: str | None = None  # WP verdict with no generated annotations
    reference_verdict: str | None = None  # WP verdict of reference_file
    excluded: str | None = None  # reason, if not usable
    notes: list[str] = Field(default_factory=list)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_loc(text: str) -> int:
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", strip_acsl(text), flags=re.DOTALL)
    return sum(1 for line in code.splitlines() if line.strip())


def classify(text: str, *, skip_functions: set[str] | None = None) -> tuple[ControlFlow, int, int]:
    shape = loop_shape(text, skip_functions=skip_functions)
    if shape.loops == 0:
        if shape.has_pointers:
            kind: ControlFlow = "pointer"
        elif shape.has_branches:
            kind = "branched"
        else:
            kind = "sequential"
    elif shape.max_depth >= 2:
        kind = "nested-loop"
    elif shape.branching_loops:
        kind = "multi-path-loop"
    else:
        kind = "single-path-loop"
    return kind, shape.loops, shape.max_depth


def write_manifest(root: Path, tasks: Iterable[BenchmarkTask]) -> Path:
    path = root / MANIFEST_NAME
    with path.open("w", encoding="utf-8") as out:
        for task in sorted(tasks, key=lambda t: t.id):
            out.write(task.model_dump_json() + "\n")
    return path


def load_manifest(root: str | Path) -> list[BenchmarkTask]:
    path = Path(root) / MANIFEST_NAME
    with path.open(encoding="utf-8") as source:
        return [BenchmarkTask.model_validate_json(line) for line in source if line.strip()]


def load_split(root: str | Path, split: str) -> list[BenchmarkTask]:
    """Resolve a split name: 'all', 'usable', a suite name, or splits/<name>.txt."""
    tasks = load_manifest(root)
    if split == "all":
        return tasks
    usable = [t for t in tasks if t.excluded is None]
    if split == "usable":
        return usable
    if split in {"svcomp", "frama-c-problems", "custom"}:
        return [t for t in usable if t.suite == split]
    split_file = Path(root) / "splits" / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Unknown split '{split}' (no {split_file})")
    wanted = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    by_id = {t.id: t for t in tasks}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise KeyError(f"Split '{split}' lists unknown tasks: {missing}")
    return [by_id[i] for i in wanted]
