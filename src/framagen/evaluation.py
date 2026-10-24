"""Run the agent over a benchmark split and aggregate results."""

from __future__ import annotations

import json
import queue
import re
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from framagen.agents.agent import run_job
from framagen.benchmarks.manifest import CONTROL_FLOWS, BenchmarkTask
from framagen.utils.frama_c import FramaCRunner
from framagen.utils.logger import logger

VERDICTS = ("VERIFIED", "FAILED", "INCONCLUSIVE", "ERROR")


def job_slug(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", task_id).strip("_.") or "task"


def run_task(
    task: BenchmarkTask,
    *,
    bench_root: Path,
    run_dir: Path,
    model: str | Any,
    model_name: str,
    runner: FramaCRunner,
    max_refinements: int,
    prover: str,
    prover_timeout_seconds: int,
    wall_timeout_seconds: int,
    self_check: bool,
    recursion_limit: int,
) -> dict[str, Any]:
    options = task.verify.to_options(
        prover=prover,
        prover_timeout_seconds=prover_timeout_seconds,
        wall_timeout_seconds=wall_timeout_seconds,
    )
    row: dict[str, Any] = {
        "task_id": task.id,
        "suite": task.suite,
        "category": task.category,
        "control_flow": task.control_flow,
        "goal": task.goal,
        "target_property": task.target_property,
        "expected_verdict": task.expected_verdict,
        "loc": task.loc,
        "model": model_name,
    }
    try:
        result = run_job(
            source_file=bench_root / task.task_file,
            property_file=bench_root / task.property_file if task.property_file else None,
            output_root=run_dir / "jobs",
            model=model,
            job_id=job_slug(task.id),
            task_id=task.id,
            target_property=task.target_property,
            expected_verdict=task.expected_verdict,
            goal=task.goal,
            max_refinements=max_refinements,
            runner=runner,
            verify_options=options,
            self_check=self_check,
            recursion_limit=recursion_limit,
        )
        verification = result["verification"] or {}
        prop = result["property"] or {}
        attempts = result["attempts"]
        row.update(
            verdict=verification.get("verdict", "ERROR"),
            status=verification.get("status"),
            proved_goals=verification.get("proved_goals"),
            total_goals=verification.get("total_goals"),
            attempts=len(attempts),
            best_attempt=result.get("best_attempt"),
            first_verdict=attempts[0]["verdict"] if attempts else None,
            property_status=prop.get("status", "not-checked"),
            property_summary=prop.get("summary"),
            property_sites_proved=sum(s["proved"] for s in prop.get("sites", [])),
            property_sites=len(prop.get("sites", [])),
            matches_expected=prop.get("matches_expected"),
            seconds=result["seconds"],
            **result["usage"],
            output_dir=result["output_dir"],
        )
    except Exception as exc:  # one task must not stop the run
        row.update(verdict="ERROR", status="agent_error", error=f"{type(exc).__name__}: {exc}")
        (run_dir / "errors").mkdir(exist_ok=True)
        (run_dir / "errors" / f"{job_slug(task.id)}.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return row


def run_evaluation(
    tasks: list[BenchmarkTask],
    *,
    bench_root: Path,
    run_dir: Path,
    jobs: int = 1,
    resume: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
    model: Any = None,
    models: list[Any] | None = None,
    **task_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run tasks, ``jobs`` at a time.

    With ``models`` (e.g. one Ollama server per GPU), each running task holds
    one model exclusively, so every server sees one request at a time; that
    keeps seeded sampling reproducible.
    """
    pool_models = list(models or [model])
    jobs = min(jobs, len(pool_models)) if models else jobs
    available: queue.Queue[Any] = queue.Queue()
    for m in pool_models:
        available.put(m)

    def run_one(task: BenchmarkTask) -> dict[str, Any]:
        held = available.get() if models else pool_models[0]
        try:
            return run_task(task, bench_root=bench_root, run_dir=run_dir, model=held, **task_kwargs)
        finally:
            if models:
                available.put(held)

    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["task_id"]] = row
    pending = [t for t in tasks if t.id not in done]
    if resume:
        # A job directory from an interrupted run blocks a fresh attempt.
        for task in pending:
            stale = run_dir / "jobs" / job_slug(task.id)
            if stale.exists():
                stale.rename(stale.with_name(stale.name + ".interrupted"))

    with (
        logger.to_file(run_dir / "run.log"),
        results_path.open("a", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=max(1, jobs)) as pool,
    ):
        for row in pool.map(run_one, pending):
            out.write(json.dumps(row) + "\n")
            out.flush()
            done[row["task_id"]] = row
            if progress:
                progress(row)
    rows = [done[t.id] for t in tasks if t.id in done]
    write_summary(run_dir, rows)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def bucket(selected: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {v: sum(r.get("verdict") == v for r in selected) for v in VERDICTS}
        n = len(selected)
        return {
            "n": n,
            **counts,
            "verified_rate": round(counts["VERIFIED"] / n, 3) if n else None,
            "first_attempt_verified": sum(r.get("first_verdict") == "VERIFIED" for r in selected),
            "property_verified": sum(r.get("property_status") == "verified" for r in selected),
            "property_conditional": sum(r.get("property_status") == "conditional" for r in selected),
            "matches_expected": sum(r.get("matches_expected") is True for r in selected),
            "contradicts_expected": sum(r.get("matches_expected") is False for r in selected),
            "mean_attempts": _mean(r.get("attempts") for r in selected),
            "mean_seconds": _mean(r.get("seconds") for r in selected),
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in selected),
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in selected),
            "llm_seconds": round(sum(float(r.get("llm_seconds") or 0) for r in selected), 1),
            "verify_seconds": round(sum(float(r.get("verify_seconds") or 0) for r in selected), 1),
        }

    by_flow: dict[str, list] = defaultdict(list)
    by_suite: dict[str, list] = defaultdict(list)
    for row in rows:
        by_flow[row["control_flow"]].append(row)
        by_suite[row["suite"]].append(row)
    return {
        "overall": bucket(rows),
        "by_control_flow": {k: bucket(by_flow[k]) for k in CONTROL_FLOWS if by_flow.get(k)},
        "by_suite": {k: bucket(v) for k, v in sorted(by_suite.items())},
    }


def _mean(values: Any) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return round(sum(numbers) / len(numbers), 2) if numbers else None


def render_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    header = (
        "| group | n | VERIFIED | FAILED | INCONCLUSIVE | ERROR | property verified | conditional "
        "| agrees w/ expected | verified@1 | mean attempts | mean s | tokens in/out |"
    )
    lines = [header, "|---" * 13 + "|"]

    def line(name: str, b: dict[str, Any]) -> str:
        return (
            f"| {name} | {b['n']} | {b['VERIFIED']} | {b['FAILED']} | {b['INCONCLUSIVE']} | {b['ERROR']} "
            f"| {b['property_verified']} | {b['property_conditional']} | {b['matches_expected']} "
            f"| {b['first_attempt_verified']} | {b['mean_attempts']} | {b['mean_seconds']} "
            f"| {b['input_tokens']}/{b['output_tokens']} |"
        )

    for name, b in summary["by_control_flow"].items():
        lines.append(line(name, b))
    for name, b in summary["by_suite"].items():
        lines.append(line(f"suite: {name}", b))
    lines.append(line("**overall**", summary["overall"]))
    lines += [
        "",
        "| task | verdict | property | sites proved | goals | attempts | seconds | tokens in/out |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        goals = f"{row.get('proved_goals')}/{row.get('total_goals')}" if row.get("total_goals") is not None else "-"
        sites = f"{row.get('property_sites_proved')}/{row.get('property_sites')}" if row.get("property_sites") else "-"
        lines.append(
            f"| {row['task_id']} | {row.get('verdict')} | {row.get('property_status', '-')} | {sites} | {goals} "
            f"| {row.get('attempts', '-')} | {row.get('seconds', '-')} "
            f"| {row.get('input_tokens', 0)}/{row.get('output_tokens', 0)} |"
        )
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_rows(rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(render_summary(summary, rows), encoding="utf-8")
    return summary


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "results.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["task_id"]] = row
    return list(rows.values())
