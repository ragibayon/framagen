from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from framagen.agents.schemas.output import GoalOutcome, VerificationResult

Backend = Literal["docker", "apptainer", "native"]

DEFAULT_IMAGE = "framac/frama-c:32.1"
# Inside the official image, opam installs Frama-C and Why3 here. Apptainer's
# --cleanenv drops the Docker ENTRYPOINT that would normally set this up.
_IMAGE_PATH = "/home/opam/.opam/4.14/bin:/usr/local/bin:/usr/bin:/bin"
_IMAGE_WHY3CONFIG = "/home/opam/.why3.conf"

_GOAL_SUMMARY = re.compile(r"Proved goals:\s*(\d+)\s*/\s*(\d+)")
_TIMEOUT_SUMMARY = re.compile(r"\bTimeout:\s*(\d+)\b")
_DIAGNOSTIC = re.compile(
    r"\[(?:wp|kernel|rte)[^\]]*\].*(?:Warning:|Error:|Timeout|Unknown|Failed)"
)
# Goal categories excluded when only partial correctness is checked.
_TERMINATION_CATEGORIES = "-@terminates,-@decreases,-@variant"


@dataclass(frozen=True)
class FramaCRunner:
    """How to invoke Frama-C: through Docker, Apptainer, or a native install.

    The defaults come from the environment so that the same code runs on a
    laptop with Docker and on an HPC node with Apptainer:

    - ``FRAMAGEN_FRAMAC_BACKEND``: docker | apptainer | native
    - ``FRAMAGEN_FRAMAC_IMAGE``: Docker image, or SIF/sandbox path for Apptainer
    """

    backend: Backend = "docker"
    image: str = DEFAULT_IMAGE

    @classmethod
    def from_env(cls, image: str | None = None, backend: str | None = None) -> "FramaCRunner":
        chosen = backend or os.environ.get("FRAMAGEN_FRAMAC_BACKEND", "docker")
        if chosen not in {"docker", "apptainer", "native"}:
            raise ValueError(f"Unknown Frama-C backend: {chosen}")
        return cls(
            backend=chosen,  # type: ignore[arg-type]
            image=image or os.environ.get("FRAMAGEN_FRAMAC_IMAGE", DEFAULT_IMAGE),
        )

    def describe(self) -> str:
        return self.image if self.backend != "native" else "native"

    def command(self, workdir: Path, args: list[str]) -> tuple[list[str], Path | None]:
        if self.backend == "native":
            return ["frama-c", *args], workdir
        if self.backend == "docker":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", self.image):
                raise ValueError("Unexpected container image name")
            return [
                "docker", "run", "--rm", "--platform", "linux/amd64",
                "--network", "none", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--pids-limit", "256", "--memory", "3g", "--cpus", "2",
                "--mount", f"type=bind,src={workdir},dst=/work",
                "-w", "/work", self.image, "frama-c", *args,
            ], None
        image = Path(self.image).expanduser()
        if not image.exists():
            raise ValueError(f"Apptainer image not found: {image}")
        return [
            "apptainer", "exec", "--containall", "--cleanenv", "--no-home",
            "--env", f"PATH={_IMAGE_PATH}",
            "--env", f"WHY3CONFIG={_IMAGE_WHY3CONFIG}",
            "--bind", f"{workdir}:/work", "--pwd", "/work",
            str(image), "frama-c", *args,
        ], None


@dataclass(frozen=True)
class VerifyOptions:
    prover: str = "alt-ergo"
    prover_timeout_seconds: int = 10
    wall_timeout_seconds: int = 180
    parallel_provers: int = 2
    rte: bool = True
    machdep: str | None = None
    check_termination: bool = True
    smoke_tests: bool = True
    # Dead-code/dead-call probes. Disable them for reachability tasks: there,
    # proving the error branch dead is the goal, and the harness functions
    # (reach_error, abort) never return by construction. Precondition and
    # loop-invariant consistency probes stay on.
    smoke_dead_code: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.prover_timeout_seconds < 1 or self.wall_timeout_seconds < 1 or self.parallel_provers < 1:
            raise ValueError("Timeouts and parallel_provers must be positive")
        if not re.fullmatch(r"[a-zA-Z0-9_,.:-]+", self.prover):
            raise ValueError("Unexpected prover name")
        if self.machdep is not None and not re.fullmatch(r"[a-zA-Z0-9_]+", self.machdep):
            raise ValueError("Unexpected machdep")

    def kernel_args(self) -> list[str]:
        return ["-machdep", self.machdep] if self.machdep else []

    def wp_args(self, report: str) -> list[str]:
        args = [
            "-wp",
            "-wp-prover", self.prover,
            "-wp-timeout", str(self.prover_timeout_seconds),
            "-wp-par", str(self.parallel_provers),
            "-wp-report-json", report,
        ]
        if self.rte:
            args.append("-wp-rte")
        if self.smoke_tests:
            args.append("-wp-smoke-tests")
            if not self.smoke_dead_code:
                args += ["-wp-no-smoke-dead-code", "-wp-no-smoke-dead-call"]
        if not self.check_termination:
            args.append(f"-wp-prop={_TERMINATION_CATEGORIES}")
        return [*self.kernel_args(), *args, *self.extra_args]


def _run(
    runner: FramaCRunner,
    workdir: Path,
    args: list[str],
    wall_timeout_seconds: int,
) -> tuple[int | None, str, str | None]:
    try:
        command, cwd = runner.command(workdir, args)
    except ValueError as exc:
        return None, "", str(exc)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=wall_timeout_seconds,
            cwd=cwd,
        )
        log = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
        return completed.returncode, log, None
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return None, f"{stdout}\n{stderr}", "Frama-C exceeded the wall-clock timeout"
    except (FileNotFoundError, OSError) as exc:
        return None, "", f"Could not invoke {runner.backend}: {exc}"


def _goals_from_report(report: Any) -> list[GoalOutcome]:
    goals: list[GoalOutcome] = []
    for entry in report if isinstance(report, list) else []:
        if not isinstance(entry, dict):
            continue
        goals.append(
            GoalOutcome(
                goal=str(entry.get("goal", "")),
                property=str(entry.get("property", "")),
                function=entry.get("function"),
                line=entry.get("line"),
                smoke=bool(entry.get("smoke", False)),
                passed=bool(entry.get("passed", False)),
                verdict=str(entry.get("verdict", "unknown")),
            )
        )
    return goals


def verify_candidate(
    candidate_path: str | Path,
    *,
    runner: FramaCRunner | None = None,
    options: VerifyOptions | None = None,
    log_dir: str | Path | None = None,
) -> VerificationResult:
    """Parse a candidate with Frama-C, then run WP and classify the outcome.

    The candidate is copied to a fresh scratch directory that is the only
    thing mounted in the container, so verification cannot modify outputs.
    Logs are written next to the candidate (or to ``log_dir``).
    """
    candidate = Path(candidate_path).expanduser().resolve(strict=True)
    if candidate.suffix != ".c" or not candidate.is_file():
        raise ValueError("candidate_path must refer to a C file")
    runner = runner or FramaCRunner.from_env()
    options = options or VerifyOptions()
    options.validate()

    version = re.search(r"_(\d+)\.c$", candidate.name)
    suffix = version.group(1) if version else "latest"
    out_dir = Path(log_dir).resolve() if log_dir else candidate.parent
    common = dict(
        candidate_file=candidate.name,
        frama_c_image=runner.describe(),
        frama_c_backend=runner.backend,
        prover=options.prover,
        prover_timeout_seconds=options.prover_timeout_seconds,
        wall_timeout_seconds=options.wall_timeout_seconds,
        rte=options.rte,
        machdep=options.machdep,
        check_termination=options.check_termination,
        smoke_tests=options.smoke_tests,
        smoke_dead_code=options.smoke_dead_code,
    )

    with tempfile.TemporaryDirectory(prefix="framagen-wp-") as scratch:
        workdir = Path(scratch)
        os.chmod(workdir, 0o777)  # container user may differ from host user
        shutil.copy2(candidate, workdir / candidate.name)

        parse_exit, parse_log, parse_error = _run(
            runner, workdir, [*options.kernel_args(), candidate.name], options.wall_timeout_seconds
        )
        (out_dir / f"parse_{suffix}.log").write_text(parse_log, encoding="utf-8")
        if parse_error or parse_exit != 0:
            return VerificationResult(
                status="verification_error" if parse_error else "parse_error",
                verdict="ERROR" if parse_error else "FAILED",
                parse_exit_code=parse_exit,
                diagnostics=(
                    [parse_error]
                    if parse_error
                    else [line for line in parse_log.splitlines() if "rror" in line][:40]
                ),
                **common,
            )

        report_name = "wp-report.json"
        wp_exit, wp_log, wp_error = _run(
            runner,
            workdir,
            [*options.wp_args(report_name), candidate.name],
            options.wall_timeout_seconds,
        )
        report_path = workdir / report_name
        report: Any = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = None
            shutil.copy2(report_path, out_dir / f"wp_report_{suffix}.json")

    (out_dir / f"verification_{suffix}.log").write_text(wp_log, encoding="utf-8")

    goals = _goals_from_report(report)
    goal_match = _GOAL_SUMMARY.search(wp_log)
    timeout_match = _TIMEOUT_SUMMARY.search(wp_log)
    proof_goals = [g for g in goals if not g.smoke]
    smoke_goals = [g for g in goals if g.smoke]
    if goals:
        proved = sum(g.passed for g in proof_goals)
        total = len(proof_goals)
        timed_out = sum(g.verdict == "timeout" for g in proof_goals)
    else:
        proved = int(goal_match.group(1)) if goal_match else None
        total = int(goal_match.group(2)) if goal_match else None
        timed_out = int(timeout_match.group(1)) if timeout_match else None
    remaining = total - proved if total is not None and proved is not None else None
    doomed = [g.goal for g in smoke_goals if not g.passed]
    unproved = [g.goal for g in proof_goals if not g.passed]
    diagnostics = [line.strip() for line in wp_log.splitlines() if _DIAGNOSTIC.search(line)]
    if wp_error:
        diagnostics.append(wp_error)

    verdict: Literal["VERIFIED", "FAILED", "INCONCLUSIVE", "ERROR"]
    if wp_error and "timeout" in wp_error:
        status, verdict = "verification_error", "INCONCLUSIVE"
    elif wp_error or wp_exit != 0:
        status, verdict = "verification_error", "ERROR"
    elif doomed:
        status, verdict = "vacuous", "FAILED"
    elif total is None or total == 0:
        status, verdict = "unresolved", "INCONCLUSIVE"
    elif proved == total:
        status, verdict = "proved", "VERIFIED"
    elif any(g.verdict == "invalid" for g in proof_goals):
        # WP's "failed" means a prover error, not a refutation; only an
        # explicit "invalid" verdict is evidence against the specification.
        status, verdict = "partially_proved", "FAILED"
    else:
        status, verdict = "partially_proved", "INCONCLUSIVE"

    return VerificationResult(
        status=status,
        verdict=verdict,
        parse_exit_code=parse_exit,
        wp_exit_code=wp_exit,
        proved_goals=proved,
        total_goals=total,
        timed_out_goals=timed_out,
        remaining_goals=remaining,
        smoke_tests_total=len(smoke_goals),
        smoke_tests_failed=doomed,
        unproved_goals=unproved,
        goals=goals,
        diagnostics=diagnostics[:80],
        **common,
    )
