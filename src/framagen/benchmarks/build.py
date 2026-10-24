"""Fetch upstream datasets, build the task manifest, calibrate, and pick splits."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from framagen.benchmarks.frama_c_problems import FCP_REPO, build_frama_c_problems
from framagen.benchmarks.manifest import (
    CONTROL_FLOWS,
    BenchmarkTask,
    load_manifest,
    write_manifest,
)
from framagen.benchmarks.svcomp import LOOP_CATEGORIES, SVCOMP_REPO, SVCOMP_TAG, build_svcomp
from framagen.utils.frama_c import FramaCRunner, verify_candidate

FCP_COMMIT = "d7d6daae2708e696888eacd56f25061ff4fcf574"


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch(raw_dir: Path, *, svcomp_tag: str = SVCOMP_TAG, fcp_commit: str = FCP_COMMIT) -> dict[str, str]:
    """Clone pinned upstream revisions. SV-COMP is sparse (loop categories only)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    sv = raw_dir / "sv-benchmarks"
    if not sv.exists():
        _git("clone", "--quiet", "--depth", "1", "--branch", svcomp_tag,
             "--filter=blob:none", "--sparse", SVCOMP_REPO, str(sv))
        _git("sparse-checkout", "set", *(f"c/{c}" for c in LOOP_CATEGORIES), "c/properties", cwd=sv)
    fcp = raw_dir / "frama-c-problems"
    if not fcp.exists():
        _git("clone", "--quiet", FCP_REPO, str(fcp))
        _git("checkout", "--quiet", fcp_commit, cwd=fcp)
    return {
        "svcomp": f"{svcomp_tag}@{_git('rev-parse', 'HEAD', cwd=sv)}",
        "frama-c-problems": _git("rev-parse", "HEAD", cwd=fcp),
    }


def build(raw_dir: Path, out_root: Path) -> list[BenchmarkTask]:
    refs = {
        "svcomp": _describe(raw_dir / "sv-benchmarks"),
        "frama-c-problems": _git("rev-parse", "HEAD", cwd=raw_dir / "frama-c-problems"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    previous = {t.id: t for t in load_manifest(out_root)} if (out_root / "manifest.jsonl").is_file() else {}
    tasks = build_svcomp(raw_dir / "sv-benchmarks", out_root, ref=refs["svcomp"])
    tasks += build_frama_c_problems(raw_dir / "frama-c-problems", out_root, ref=refs["frama-c-problems"])
    tasks = [_keep_calibration(t, previous.get(t.id)) for t in tasks]
    write_manifest(out_root, tasks)
    return tasks


_CALIBRATION_NOTES = ("reference does not parse",)


def _keep_calibration(task: BenchmarkTask, old: BenchmarkTask | None) -> BenchmarkTask:
    """Reuse WP calibration from an existing manifest when the upstream file is unchanged."""
    if old is None or old.upstream_sha256 != task.upstream_sha256 or old.harness_baseline is None:
        return task
    return task.model_copy(update={
        "harness_baseline": old.harness_baseline,
        "reference_verdict": old.reference_verdict,
        "excluded": task.excluded or old.excluded,
        "notes": task.notes + [n for n in old.notes if n in _CALIBRATION_NOTES and n not in task.notes],
    })


def _describe(repo: Path) -> str:
    head = _git("rev-parse", "HEAD", cwd=repo)
    try:
        tag = _git("describe", "--tags", "--exact-match", cwd=repo)
    except subprocess.CalledProcessError:
        return head
    return f"{tag}@{head}"


def calibrate(
    out_root: Path,
    *,
    runner: FramaCRunner,
    jobs: int = 8,
    prover_timeout_seconds: int = 10,
    progress: Callable[[str], None] | None = None,
) -> list[BenchmarkTask]:
    """Run WP on each task as given (no generated annotations) and on references.

    - A task WP cannot process at all is excluded.
    - An SV-COMP task already proved by the harness alone is excluded as
      trivial: it does not exercise invariant generation.
    - The reference verdict documents whether a hand-written spec exists that
      this Frama-C version actually proves.
    """
    tasks = load_manifest(out_root)
    log_root = out_root / "_calibration"

    def check(task: BenchmarkTask) -> BenchmarkTask:
        if task.excluded and "reducer" in task.excluded:
            return task
        options = task.verify.to_options(
            prover_timeout_seconds=prover_timeout_seconds, wall_timeout_seconds=600, parallel_provers=1
        )
        logs = log_root / task.id
        logs.mkdir(parents=True, exist_ok=True)
        baseline = verify_candidate(out_root / task.task_file, runner=runner, options=options, log_dir=logs / "task")
        update: dict[str, object] = {"harness_baseline": baseline.verdict}
        if baseline.verdict == "ERROR":
            update["excluded"] = task.excluded or (
                "WP cannot process task (non-natural loop)" if _non_natural(logs / "task")
                else "WP error on task: " + "; ".join(baseline.diagnostics[-2:])
            )
        elif baseline.status == "parse_error":
            update["excluded"] = task.excluded or "Frama-C cannot parse task"
        elif task.suite == "svcomp" and baseline.verdict == "VERIFIED":
            update["excluded"] = task.excluded or "trivial: proved without generated annotations"
        if task.reference_file:
            reference = verify_candidate(
                out_root / task.reference_file, runner=runner, options=options, log_dir=logs / "reference"
            )
            update["reference_verdict"] = reference.verdict
            if reference.status == "parse_error" and "reference does not parse" not in task.notes:
                update["notes"] = [*task.notes, "reference does not parse"]
        if progress:
            progress(f"{task.id}: harness={update['harness_baseline']} reference={update.get('reference_verdict')}")
        return task.model_copy(update=update)

    for task in tasks:
        (log_root / task.id / "task").mkdir(parents=True, exist_ok=True)
        (log_root / task.id / "reference").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        calibrated = list(pool.map(check, tasks))
    write_manifest(out_root, calibrated)
    return calibrated


def _non_natural(log_dir: Path) -> bool:
    log = log_dir / "verification_latest.log"
    return log.is_file() and "Non-natural loop" in log.read_text(encoding="utf-8")


# The demo mirrors SpecPylot's control-flow split: straight-line, branching and
# pointer programs come from frama-c-problems (full contracts); loop programs
# come from SV-COMP (invariants for a fixed property).
_DEMO_SOURCES = {
    "sequential": "frama-c-problems",
    "branched": "frama-c-problems",
    "pointer": "frama-c-problems",
    "single-path-loop": "svcomp",
    "multi-path-loop": "svcomp",
    "nested-loop": "svcomp",
}


def select_demo(tasks: list[BenchmarkTask], *, per_class: int = 4, max_loc: int = 60) -> list[BenchmarkTask]:
    """Deterministically pick small tasks per control-flow class.

    Within a class, tasks are sorted by size and drawn round-robin across
    upstream categories so one directory does not dominate. Near-duplicate
    SV-COMP variants (``_abstracted``, ``-u``/``-ll`` siblings) are skipped
    when their base task is present. frama-c-problems tasks whose reference
    spec verifies are preferred (known feasible); tasks whose reference does
    not even parse are skipped.
    """
    chosen: list[BenchmarkTask] = []
    for kind in CONTROL_FLOWS:
        suite = _DEMO_SOURCES[kind]
        pool = [
            t for t in tasks
            if t.excluded is None and t.suite == suite and t.control_flow == kind and t.loc <= max_loc
            and "reference does not parse" not in t.notes
        ]
        pool = _drop_variants(pool)
        by_category: dict[str, list[BenchmarkTask]] = defaultdict(list)
        for task in sorted(pool, key=lambda t: (t.reference_verdict not in {None, "VERIFIED"}, t.loc, t.id)):
            by_category[task.category].append(task)
        picked: list[BenchmarkTask] = []
        queues = [by_category[c] for c in sorted(by_category)]
        while len(picked) < per_class and any(queues):
            for queue in queues:
                if queue and len(picked) < per_class:
                    picked.append(queue.pop(0))
        chosen += picked
    return chosen


def _drop_variants(pool: list[BenchmarkTask]) -> list[BenchmarkTask]:
    names = {t.id for t in pool}
    keep = []
    for task in pool:
        base = task.id
        for marker in ("_abstracted", "_variant", "-ll", "-u", "b"):
            if base.endswith(marker) and base[: -len(marker)] in names:
                break
        else:
            keep.append(task)
    return keep


def write_split(out_root: Path, name: str, tasks: list[BenchmarkTask], header: str = "") -> Path:
    split_dir = out_root / "splits"
    split_dir.mkdir(exist_ok=True)
    path = split_dir / f"{name}.txt"
    lines = [f"# {line}" for line in header.splitlines()] + [t.id for t in tasks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize(tasks: list[BenchmarkTask]) -> str:
    rows = defaultdict(lambda: defaultdict(int))
    excluded = defaultdict(int)
    for task in tasks:
        if task.excluded:
            excluded[(task.suite, task.excluded.split(":")[0].split(" (")[0])] += 1
            continue
        rows[task.suite][task.control_flow] += 1
    out = ["| suite | " + " | ".join(CONTROL_FLOWS) + " | total |", "|---" * (len(CONTROL_FLOWS) + 2) + "|"]
    for suite in sorted(rows):
        counts = [rows[suite][k] for k in CONTROL_FLOWS]
        out.append(f"| {suite} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    if excluded:
        out.append("")
        out.append("Excluded:")
        for (suite, reason), count in sorted(excluded.items()):
            out.append(f"- {suite}: {reason} — {count}")
    return "\n".join(out)
