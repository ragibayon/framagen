from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, Sequence

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from pydantic import ValidationError

from framagen.agents.guards import check_candidate, normalize_candidate
from framagen.agents.schemas.output import CandidateSpecification, VerificationResult
from framagen.llm import JSONLUsageRecorder, LLMCallRecord, LLMRunMetrics
from framagen.properties import PROPERTIES, PropertyAssessment, PropertySpec, assess_property, prompt_context
from framagen.utils.frama_c import FramaCRunner, VerifyOptions, verify_candidate
from framagen.utils.logger import logger

AGENT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = AGENT_DIR / "prompts"
SKILLS_DIR = AGENT_DIR / "skills"

Goal = Literal["loop-annotations", "full-contract"]


def _render_prompt(name: str, **context: Any) -> str:
    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context).strip()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_reader(source: Path, prop: Path | None) -> Any:
    @tool("read_job_input")
    def read_job_input(kind: Literal["source", "property"] = "source") -> str:
        """Read the exact C source or verification property assigned to this job."""
        if kind == "source":
            path = source
        elif kind == "property" and prop is not None:
            path = prop
        else:
            return "No verification-property file was provided."
        return f"Path: {path.name}\nSHA256: {_hash_file(path)}\n\n{path.read_text(encoding='utf-8', errors='replace')}"

    return read_job_input


def _self_check_tool(
    original: str, *, allow_axioms: bool, runner: FramaCRunner, options: VerifyOptions, log: list[dict]
) -> Any:
    @tool("check_candidate")
    def check_candidate_tool(annotated_c: str) -> str:
        """Check a complete annotated C file: annotation guards, ACSL parsing, and Frama-C/WP proof.

        Returns the verdict, unproved goal names, failed smoke tests, and a log
        excerpt. Use it to test a draft before returning the final answer.
        """
        text, _ = normalize_candidate(annotated_c)
        guard = check_candidate(original, text, allow_axioms=allow_axioms)
        if not guard.ok:
            log.append({"verdict": "REJECTED", "violations": guard.violations})
            return json.dumps({"verdict": "REJECTED", "violations": guard.violations}, indent=2)
        with tempfile.TemporaryDirectory(prefix="framagen-check-") as scratch:
            path = Path(scratch) / "draft.c"
            path.write_text(text, encoding="utf-8")
            result = verify_candidate(path, runner=runner, options=options)
            log_tail = ""
            for name in ("verification_latest.log", "parse_latest.log"):
                if (Path(scratch) / name).is_file():
                    log_tail = (Path(scratch) / name).read_text(encoding="utf-8")[-4000:]
                    break
        log.append({"verdict": result.verdict, "proved": result.proved_goals, "total": result.total_goals})
        return json.dumps(
            {
                "verdict": result.verdict,
                "status": result.status,
                "proved_goals": result.proved_goals,
                "total_goals": result.total_goals,
                "unproved_goals": result.unproved_goals[:30],
                "smoke_tests_failed": result.smoke_tests_failed,
                "log_tail": log_tail,
            },
            indent=2,
        )

    return check_candidate_tool


def _disable_subagents(model: str | Any) -> None:
    """Drop the `task` (general-purpose subagent) and `execute` tools for this provider.

    A job is one small C file; with an 8B model the subagent mostly re-reads
    the task in a fresh context and loops until the recursion limit.
    """
    from deepagents import HarnessProfile, register_harness_profile
    from deepagents._models import get_model_provider
    from deepagents.profiles.harness.harness_profiles import GeneralPurposeSubagentProfile

    provider = model.partition(":")[0] if isinstance(model, str) else get_model_provider(model)
    if provider:
        register_harness_profile(
            provider,
            HarnessProfile(
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                excluded_tools=frozenset({"execute"}),
            ),
        )


def build_agent(
    *,
    model: str | Any,
    workspace: str | Path,
    tools: Sequence[Any] | None = None,
    allow_subagents: bool = False,
) -> Any:
    root = Path(workspace).expanduser().resolve(strict=True)
    if not SKILLS_DIR.is_dir():
        raise FileNotFoundError(f"Missing shared skills: {SKILLS_DIR}")
    if not (root / "output").is_dir():
        raise FileNotFoundError(f"Missing job output directory: {root / 'output'}")

    if not allow_subagents:
        _disable_subagents(model)
    backend = CompositeBackend(
        default=FilesystemBackend(root_dir=root, virtual_mode=True),
        routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
    )
    return create_deep_agent(
        model=model,
        name="framgen",
        system_prompt=_render_prompt("system.j2"),
        backend=backend,
        skills=["/skills/"],
        tools=list(tools or []),
        response_format=ToolStrategy(schema=CandidateSpecification),
        permissions=[
            FilesystemPermission(operations=["write"], paths=["/output/**"], mode="allow"),
            FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
        ],
    )


_NUDGE = (
    "You stopped without returning the result. Call the CandidateSpecification tool now. Its "
    "`annotated_c` argument must be the COMPLETE C file with your ACSL annotations added and the "
    "executable code unchanged. Do not answer in plain text."
)
_C_BLOCK = re.compile(r"```(?:c|C|acsl)?\s*\n(.*?)\n\s*```", re.DOTALL)


def _structured(result: dict[str, Any]) -> CandidateSpecification | None:
    structured = result.get("structured_response")
    if structured is None:
        return None
    try:
        return CandidateSpecification.model_validate(structured)
    except ValidationError:
        return None


def _recover_from_text(result: dict[str, Any], original: str, *, allow_axioms: bool) -> CandidateSpecification | None:
    """Accept a complete annotated program the model printed instead of returning."""
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", None) != "ai":
            continue
        content = message.content if isinstance(message.content, str) else ""
        for block in sorted(_C_BLOCK.findall(content), key=len, reverse=True):
            if check_candidate(original, block + "\n", allow_axioms=allow_axioms).ok:
                return CandidateSpecification(
                    summary="Recovered from a plain-text answer (the model did not call CandidateSpecification).",
                    annotated_c=block + "\n",
                )
        break
    return None


def _transcript(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for message in result.get("messages", []):
        row: dict[str, Any] = {"type": getattr(message, "type", type(message).__name__)}
        content = getattr(message, "content", "")
        row["content"] = content if isinstance(content, str) else json.loads(json.dumps(content, default=str))
        for attr in ("tool_calls", "name", "tool_call_id", "usage_metadata"):
            value = getattr(message, attr, None)
            if value:
                row[attr] = json.loads(json.dumps(value, default=str))
        reasoning = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content")
        if reasoning:
            row["reasoning"] = reasoning
        rows.append(row)
    return rows


def _model_identity(model: str | Any) -> tuple[str, str]:
    if isinstance(model, str):
        provider, _, name = model.partition(":")
        return provider, name or model
    name = getattr(model, "model", None) or getattr(model, "model_name", None) or type(model).__name__
    provider = type(model).__name__.removeprefix("Chat").lower()
    return provider, str(name)


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rejected_result(candidate_file: str, violations: list[str], runner: FramaCRunner, options: VerifyOptions) -> VerificationResult:
    return VerificationResult(
        candidate_file=candidate_file,
        status="rejected",
        verdict="FAILED",
        diagnostics=violations,
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


def _feedback(previous: VerificationResult, assessment: PropertyAssessment | None, log_path: Path) -> str:
    """Verifier feedback for the next attempt: property first, then raw details."""
    parts = ["LATEST VERIFIER FEEDBACK"]
    if assessment is not None:
        parts.append(f"Target property ({assessment.property}): {assessment.status} — {assessment.summary}")
        unproved_sites = [f"line {s.line}: {s.function}" for s in assessment.sites if not s.proved]
        if unproved_sites:
            parts.append("Property sites not yet proved: " + "; ".join(unproved_sites))
        if assessment.unproved_support_goals:
            parts.append("Unproved supporting annotations: " + ", ".join(assessment.unproved_support_goals[:20]))
    summary = previous.model_dump(exclude={"goals"})
    parts.append(json.dumps(summary, indent=2))
    if log_path.exists():
        parts.append("Verifier log tail:\n" + log_path.read_text(encoding="utf-8")[-8000:])
    return "\n\n".join(parts)


def _rank(result: VerificationResult, assessment: PropertyAssessment | None) -> int:
    """Order attempts so the job reports its best one, not merely its last."""
    if result.verdict == "VERIFIED":
        return 5
    status = assessment.status if assessment else "not-checked"
    if status == "conditional":
        return 4
    if result.verdict == "INCONCLUSIVE":
        return 3
    if result.status == "vacuous":
        return 2
    return 1 if result.verdict == "FAILED" else 0


def run_job(
    *,
    source_file: str | Path,
    output_root: str | Path,
    model: str | Any,
    job_id: str | None = None,
    task_id: str | None = None,
    property_file: str | Path | None = None,
    target_property: str = "contract",
    expected_verdict: bool | None = None,
    goal: Goal = "full-contract",
    max_refinements: int = 0,
    runner: FramaCRunner | None = None,
    verify_options: VerifyOptions | None = None,
    verify: bool = True,
    self_check: bool = False,
    tools: Sequence[Any] | None = None,
    recursion_limit: int = 100,
    max_nudges: int = 2,
) -> dict[str, Any]:
    """Generate ACSL for one C file, verify it, and refine on verifier feedback.

    Every attempt is saved (``candidate_NN.c`` or ``rejected_candidate_NN.c``)
    with its verification record, property assessment, transcript, and LLM
    usage; nothing is overwritten.
    """
    source = Path(source_file).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".c":
        raise ValueError("source_file must be an existing .c file")
    prop = Path(property_file).expanduser().resolve(strict=True) if property_file else None
    if prop is not None and not prop.is_file():
        raise ValueError("property_file must be an existing file")
    if max_refinements < 0 or recursion_limit <= 0 or max_nudges < 0:
        raise ValueError("Refinement count, nudge count, or recursion limit is invalid")
    if target_property not in PROPERTIES:
        raise ValueError(f"Unknown target property {target_property!r}; choose from {sorted(PROPERTIES)}")
    spec = PROPERTIES[target_property]
    runner = runner or FramaCRunner.from_env()
    options = verify_options or VerifyOptions()
    options.validate()
    allow_axioms = goal == "full-contract"

    identifier = job_id or f"{source.stem}-{hashlib.sha256(str(source).encode()).hexdigest()[:8]}"
    if identifier in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
        raise ValueError("Invalid job_id")

    workspace = Path(output_root).expanduser().resolve() / identifier
    workspace.mkdir(parents=True, exist_ok=False)
    output_dir = workspace / "output"
    output_dir.mkdir()
    input_dir = workspace / "input"
    input_dir.mkdir()
    shutil.copy2(source, input_dir / source.name)

    with logger.job_context(identifier), logger.to_file(output_dir / "job.log", job=identifier):
        return _run_job(
            identifier=identifier, source=source, prop=prop, spec=spec, expected_verdict=expected_verdict,
            task_id=task_id, goal=goal, model=model, max_refinements=max_refinements, runner=runner,
            options=options, verify=verify, self_check=self_check, tools=tools,
            recursion_limit=recursion_limit, max_nudges=max_nudges, allow_axioms=allow_axioms,
            workspace=workspace, output_dir=output_dir,
        )


def _run_job(
    *,
    identifier: str,
    source: Path,
    prop: Path | None,
    spec: PropertySpec,
    expected_verdict: bool | None,
    task_id: str | None,
    goal: Goal,
    model: str | Any,
    max_refinements: int,
    runner: FramaCRunner,
    options: VerifyOptions,
    verify: bool,
    self_check: bool,
    tools: Sequence[Any] | None,
    recursion_limit: int,
    max_nudges: int,
    allow_axioms: bool,
    workspace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_hash = _hash_file(source)
    property_hash = _hash_file(prop) if prop else None
    original_text = source.read_text(encoding="utf-8", errors="replace")
    provider, model_name = _model_identity(model)
    usage_log = JSONLUsageRecorder(output_dir / "llm_usage.jsonl")
    metadata: dict[str, Any] = {
        "job_id": identifier,
        "task_id": task_id,
        "goal": goal,
        "target_property": spec.name,
        "expected_verdict": expected_verdict,
        "source_file": str(source),
        "source_sha256": source_hash,
        "property_file": str(prop) if prop else None,
        "property_sha256": property_hash,
        "model": model if isinstance(model, str) else repr(model),
        "provider": provider,
        "model_name": model_name,
        "frama_c_backend": runner.backend,
        "frama_c_image": runner.describe(),
        "verify_options": {
            "prover": options.prover,
            "prover_timeout_seconds": options.prover_timeout_seconds,
            "wall_timeout_seconds": options.wall_timeout_seconds,
            "rte": options.rte,
            "machdep": options.machdep,
            "check_termination": options.check_termination,
            "smoke_tests": options.smoke_tests,
            "smoke_dead_code": options.smoke_dead_code,
        },
        "max_refinements": max_refinements,
        "self_check": self_check,
        "status": "running",
    }
    _save_json(output_dir / "job.json", metadata)
    logger.info(f"job started: {source.name}, property {spec.name}, model {model_name}")

    self_check_log: list[dict] = []
    agent_tools = [_input_reader(source, prop), *(tools or [])]
    if self_check and verify:
        agent_tools.append(
            _self_check_tool(original_text, allow_axioms=allow_axioms, runner=runner, options=options, log=self_check_log)
        )
    agent = build_agent(model=model, workspace=workspace, tools=agent_tools)

    previous: VerificationResult | None = None
    assessment: PropertyAssessment | None = None
    previous_name: str | None = None
    candidates: list[str] = []
    attempts: list[dict[str, Any]] = []
    best: tuple[int, int, VerificationResult, PropertyAssessment | None] | None = None
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0,
              "llm_seconds": 0.0, "verify_seconds": 0.0}
    started = time.monotonic()
    try:
        for attempt in range(max_refinements + 1):
            candidate_name = f"candidate_{attempt:02d}.c"
            prompt = _render_prompt(
                "user.j2",
                job_id=identifier,
                source_file=source.name,
                property_file=prop.name if prop else None,
                property_text=prop.read_text(encoding="utf-8").strip() if prop else None,
                goal=goal,
                candidate_name=candidate_name,
                previous_candidate=f"/output/{previous_name}" if previous_name else None,
                output_dir="/output",
                max_refinements=max_refinements,
                attempt=attempt,
                frama_c=runner.describe(),
                prover=options.prover,
                timeout_seconds=options.prover_timeout_seconds,
                rte=options.rte,
                check_termination=options.check_termination,
                machdep=options.machdep,
                self_check=self_check and verify,
                **prompt_context(spec, original_text),
            )
            if previous is not None:
                log_path = output_dir / (
                    f"parse_{attempt - 1:02d}.log" if previous.status == "parse_error" else f"verification_{attempt - 1:02d}.log"
                )
                prompt += "\n\n" + _feedback(previous, assessment, log_path)

            # --- generation (with nudges when the model forgets to return) ---
            metrics = LLMRunMetrics()
            messages: list[Any] = [{"role": "user", "content": prompt}]
            candidate: CandidateSpecification | None = None
            recovered = False
            nudges = 0
            transcript: list[dict[str, Any]] = []
            error: BaseException | None = None
            with logger.timed(f"attempt {attempt}: generation") as gen_timing:
                try:
                    for nudges in range(max_nudges + 1):
                        result = agent.invoke(
                            {"messages": messages},
                            config={"recursion_limit": recursion_limit, "callbacks": [metrics]},
                        )
                        transcript = _transcript(result)
                        candidate = _structured(result)
                        if candidate is not None:
                            break
                        candidate = _recover_from_text(result, original_text, allow_axioms=allow_axioms)
                        if candidate is not None:
                            recovered = True
                            logger.warning("model answered in plain text; recovered the annotated program from it")
                            break
                        if nudges < max_nudges:
                            logger.warning("model returned no structured answer; re-prompting")
                            messages = [*result["messages"], {"role": "user", "content": _NUDGE}]
                except Exception as exc:  # recorded below, then re-raised
                    error = exc
            _save_json(output_dir / f"transcript_{attempt:02d}.json", transcript)
            stats = metrics.as_dict()
            token_usage = stats["token_usage"] or {}
            gen_status = (
                "error" if error is not None else "no_structured_response" if candidate is None
                else "recovered" if recovered else "accepted"
            )
            usage_log(LLMCallRecord(
                stage="specification_generation", provider=provider, model=model_name,
                latency_seconds=stats["latency_seconds"], token_usage=token_usage or None,
                status=gen_status, error=type(error).__name__ if error else None,
                call_count=stats["call_count"], attempt=attempt, job_id=identifier,
            ))
            usage = {
                "input_tokens": int(token_usage.get("input_tokens") or 0),
                "output_tokens": int(token_usage.get("output_tokens") or 0),
                "total_tokens": int(token_usage.get("total_tokens") or 0),
                "llm_calls": int(stats["call_count"]),
                "llm_seconds": round(gen_timing.get("seconds", 0.0), 2),
            }
            for key, value in usage.items():
                totals[key] += value
            if error is not None:
                raise error
            if candidate is None:
                raise ValueError(
                    f"Agent returned no structured_response after {max_nudges} re-prompt(s); "
                    f"see transcript_{attempt:02d}.json"
                )

            annotated, normalizations = normalize_candidate(candidate.annotated_c)
            guard = check_candidate(original_text, annotated, allow_axioms=allow_axioms)
            _save_json(output_dir / f"generation_{attempt:02d}.json", candidate.model_dump())

            verify_seconds = 0.0
            if not guard.ok:
                saved_name = f"rejected_{candidate_name}"
                (output_dir / saved_name).write_text(annotated, encoding="utf-8")
                previous = _rejected_result(saved_name, guard.violations, runner, options)
                assessment = None
                logger.warning(f"attempt {attempt}: candidate rejected — {guard.violations[0]}")
            else:
                saved_name = candidate_name
                candidate_path = output_dir / candidate_name
                candidate_path.write_text(annotated, encoding="utf-8")
                _save_json(
                    output_dir / "analysis.json",
                    {"summary": candidate.summary, "assumptions": candidate.assumptions, "risks": candidate.risks},
                )
                _save_json(output_dir / "contracts.json", [c.model_dump() for c in candidate.clauses])
                candidates.append(candidate_name)
                if not verify:
                    previous = _rejected_result(candidate_name, [], runner, options).model_copy(
                        update={"status": "not_checked", "verdict": "NOT_CHECKED"}
                    )
                else:
                    with logger.timed(f"attempt {attempt}: Frama-C/WP") as wp_timing:
                        previous = verify_candidate(candidate_path, runner=runner, options=options)
                    verify_seconds = round(wp_timing["seconds"], 2)
                assessment = assess_property(spec, annotated, previous, expected_verdict=expected_verdict)
                _save_json(output_dir / f"property_{attempt:02d}.json", assessment.model_dump())
                _save_json(output_dir / "property.json", assessment.model_dump())
                log = logger.success if assessment.status == "verified" else logger.warning
                log(f"attempt {attempt}: {previous.verdict}, property {assessment.status} ({assessment.summary})")
            totals["verify_seconds"] += verify_seconds
            previous_name = saved_name
            _save_json(output_dir / f"verification_{attempt:02d}.json", previous.model_dump())
            _save_json(output_dir / "verification.json", previous.model_dump())
            attempts.append({
                "attempt": attempt,
                "file": saved_name,
                "status": previous.status,
                "verdict": previous.verdict,
                "proved_goals": previous.proved_goals,
                "total_goals": previous.total_goals,
                "property_status": assessment.status if assessment else "not-checked",
                "property_sites_proved": sum(s.proved for s in assessment.sites) if assessment else None,
                "property_sites": len(assessment.sites) if assessment else None,
                "generation": gen_status,
                "nudges": nudges,
                "guard_notes": guard.notes + normalizations,
                **usage,
                "verify_seconds": verify_seconds,
            })
            if best is None or _rank(previous, assessment) > best[0]:
                best = (_rank(previous, assessment), attempt, previous, assessment)
            if previous.verdict in {"VERIFIED", "NOT_CHECKED", "ERROR"}:
                break

        last_verdict = previous.verdict if previous else "NOT_CHECKED"
        if best is not None:
            _, best_attempt, previous, assessment = best
            _save_json(output_dir / "verification.json", previous.model_dump())
            if assessment is not None:
                _save_json(output_dir / "property.json", assessment.model_dump())
        else:
            best_attempt = None

        if _hash_file(source) != source_hash or (prop and _hash_file(prop) != property_hash):
            raise RuntimeError("An original input changed during the job")

        totals = {k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()}
        metadata.update(
            status="completed",
            candidates=candidates,
            attempts=attempts,
            verification_status=previous.status if previous else "not_checked",
            verdict=previous.verdict if previous else "NOT_CHECKED",
            best_attempt=best_attempt,
            best_file=attempts[best_attempt]["file"] if best_attempt is not None else None,
            last_attempt_verdict=last_verdict,
            property=assessment.model_dump() if assessment else None,
            seconds=round(time.monotonic() - started, 2),
            usage=totals,
            self_check_calls=self_check_log,
        )
        _save_json(output_dir / "job.json", metadata)
        logger.info(
            f"job finished: {metadata['verdict']}, property "
            f"{assessment.status if assessment else 'not-checked'}, {metadata['seconds']}s, "
            f"{totals['input_tokens']}+{totals['output_tokens']} tokens"
        )
        return {
            "job_id": identifier,
            "task_id": task_id,
            "output_dir": str(output_dir),
            "candidate_files": candidates,
            "attempts": attempts,
            "best_attempt": best_attempt,
            "verification": previous.model_dump(exclude={"goals"}) if previous else None,
            "property": assessment.model_dump() if assessment else None,
            "seconds": metadata["seconds"],
            "usage": totals,
        }
    except Exception as exc:
        metadata.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            attempts=attempts,
            seconds=round(time.monotonic() - started, 2),
            usage={k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()},
        )
        _save_json(output_dir / "job.json", metadata)
        logger.error(f"job failed: {metadata['error']}")
        raise
