from __future__ import annotations

import re
import subprocess
from pathlib import Path

from framagen.agents.schemas.output import VerificationResult

_GOAL_SUMMARY = re.compile(r"Proved goals:\s*(\d+)\s*/\s*(\d+)")
_TIMEOUT_SUMMARY = re.compile(r"\bTimeout:\s*(\d+)\b")
_DIAGNOSTIC = re.compile(
    r"\[(?:wp|kernel|rte)[^\]]*\].*(?:Warning:|Error:|Timeout|Unknown|Failed)"
)


def _run_container(
    *,
    directory: Path,
    filename: str,
    image: str,
    args: list[str],
    wall_timeout_seconds: int,
) -> tuple[int | None, str, str | None]:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--memory",
        "3g",
        "--cpus",
        "2",
        "--mount",
        f"type=bind,src={directory},dst=/work,readonly",
        "-w",
        "/work",
        image,
        "frama-c",
        *args,
        filename,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=wall_timeout_seconds,
        )
        log = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
        return completed.returncode, log, None
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return None, f"{stdout}\n{stderr}", "Docker command exceeded wall-clock timeout"
    except (FileNotFoundError, OSError) as exc:
        return None, "", f"Could not invoke Docker: {exc}"


def verify_candidate(
    candidate_path: str | Path,
    *,
    image: str = "framac/frama-c:32.1",
    prover: str = "alt-ergo",
    prover_timeout_seconds: int = 10,
    wall_timeout_seconds: int = 180,
    parallel_provers: int = 2,
) -> VerificationResult:
    candidate = Path(candidate_path).expanduser().resolve(strict=True)
    if candidate.suffix != ".c" or not candidate.is_file():
        raise ValueError("candidate_path must refer to a C file")
    if prover_timeout_seconds < 1 or wall_timeout_seconds < 1 or parallel_provers < 1:
        raise ValueError("Timeouts and parallel_provers must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", image):
        raise ValueError("Unexpected container image name")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", prover):
        raise ValueError("Unexpected prover name")

    version = re.search(r"_(\d+)\.c$", candidate.name)
    suffix = version.group(1) if version else "latest"
    output_dir = candidate.parent
    parse_exit, parse_log, parse_error = _run_container(
        directory=output_dir,
        filename=candidate.name,
        image=image,
        args=["-print"],
        wall_timeout_seconds=wall_timeout_seconds,
    )
    (output_dir / f"parse_{suffix}.log").write_text(parse_log, encoding="utf-8")
    if parse_error or parse_exit != 0:
        return VerificationResult(
            candidate_file=candidate.name,
            status="verification_error" if parse_error else "parse_error",
            parse_exit_code=parse_exit,
            diagnostics=(
                [parse_error]
                if parse_error
                else [
                    line
                    for line in parse_log.splitlines()
                    if "Error" in line or "error" in line
                ]
            ),
            frama_c_image=image,
            prover=prover,
            prover_timeout_seconds=prover_timeout_seconds,
            wall_timeout_seconds=wall_timeout_seconds,
        )

    wp_exit, wp_log, wp_error = _run_container(
        directory=output_dir,
        filename=candidate.name,
        image=image,
        args=[
            "-wp",
            "-wp-rte",
            "-wp-prover",
            prover,
            "-wp-timeout",
            str(prover_timeout_seconds),
            "-wp-par",
            str(parallel_provers),
        ],
        wall_timeout_seconds=wall_timeout_seconds,
    )
    (output_dir / f"verification_{suffix}.log").write_text(wp_log, encoding="utf-8")
    (output_dir / "verification.log").write_text(wp_log, encoding="utf-8")

    goal_match = _GOAL_SUMMARY.search(wp_log)
    timeout_match = _TIMEOUT_SUMMARY.search(wp_log)
    proved = int(goal_match.group(1)) if goal_match else None
    total = int(goal_match.group(2)) if goal_match else None
    timed_out = int(timeout_match.group(1)) if timeout_match else None
    remaining = total - proved if total is not None and proved is not None else None
    diagnostics = [
        line.strip() for line in wp_log.splitlines() if _DIAGNOSTIC.search(line)
    ]
    if wp_error:
        diagnostics.append(wp_error)

    if wp_error or wp_exit != 0:
        status = "verification_error"
    elif total is None or total == 0:
        status = "unresolved"
    elif proved == total:
        status = "proved"
    else:
        status = "partially_proved"

    return VerificationResult(
        candidate_file=candidate.name,
        status=status,
        parse_exit_code=parse_exit,
        wp_exit_code=wp_exit,
        proved_goals=proved,
        total_goals=total,
        timed_out_goals=timed_out,
        remaining_goals=remaining,
        diagnostics=diagnostics,
        frama_c_image=image,
        prover=prover,
        prover_timeout_seconds=prover_timeout_seconds,
        wall_timeout_seconds=wall_timeout_seconds,
    )
