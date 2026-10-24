from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Sequence

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from pydantic import ValidationError

from framagen.agents.schemas.output import CandidateSpecification, VerificationResult
from framagen.utils.frama_c import verify_candidate

AGENT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = AGENT_DIR / "prompts"
SKILLS_DIR = AGENT_DIR / "skills"


_TOKEN = re.compile(
    r"""(?P<literal>(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|(?:u|U|L)?'(?:\\.|[^'\\])*')
       |(?P<comment>/\*.*?\*/|//[^\n]*)
       |(?P<space>\s+)
       |(?P<word>[A-Za-z_]\w*|\d[\w.]*)
       |(?P<symbol>\#\#|<<=|>>=|\.\.\.|\+\+|--|->|&&|\|\||<=|>=|==|!=|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\S)""",
    re.VERBOSE | re.DOTALL,
)
_ACSL_BLOCK = re.compile(r"/\*@.*?\*/", re.DOTALL)
_ACSL_LINE = re.compile(r"(?m)^[ \t]*//@[^\n]*(?:\n|$)")


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


def _c_tokens(text: str) -> list[str]:
    text = _ACSL_LINE.sub("\n", _ACSL_BLOCK.sub("", text))
    return [
        match.group()
        for match in _TOKEN.finditer(text)
        if match.lastgroup not in {"space", "comment"}
    ]


def _same_executable_tokens(original: str, candidate: str) -> bool:
    return _c_tokens(original) == _c_tokens(candidate)


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
        return f"Path: {path}\nSHA256: {_hash_file(path)}\n\n{path.read_text(encoding='utf-8', errors='replace')}"

    return read_job_input


def build_agent(
    *,
    model: str,
    workspace: str | Path,
    tools: Sequence[Any] | None = None,
) -> Any:
    root = Path(workspace).expanduser().resolve(strict=True)
    if not SKILLS_DIR.is_dir():
        raise FileNotFoundError(f"Missing shared skills: {SKILLS_DIR}")
    if not (root / "output").is_dir():
        raise FileNotFoundError(f"Missing job output directory: {root / 'output'}")

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
            FilesystemPermission(
                operations=["write"], paths=["/output/**"], mode="allow"
            ),
            FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
        ],
    )


def _candidate_from_result(result: dict[str, Any]) -> CandidateSpecification:
    structured = result.get("structured_response")
    if structured is None:
        raise ValueError(
            "Agent returned no structured_response. Check model tool-calling support."
        )
    try:
        return CandidateSpecification.model_validate(structured)
    except ValidationError as exc:
        raise ValueError(
            "Agent structured output did not match CandidateSpecification"
        ) from exc


def _save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run_job(
    *,
    source_file: str | Path,
    output_root: str | Path,
    model: str,
    job_id: str | None = None,
    property_file: str | Path | None = None,
    max_refinements: int = 0,
    frama_c_image: str = "framac/frama-c:32.1",
    prover: str = "alt-ergo",
    timeout_seconds: int = 10,
    wall_timeout_seconds: int = 180,
    verify: bool = True,
    tools: Sequence[Any] | None = None,
    recursion_limit: int = 100,
) -> dict[str, Any]:
    source = Path(source_file).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".c":
        raise ValueError("source_file must be an existing .c file")
    prop = (
        Path(property_file).expanduser().resolve(strict=True) if property_file else None
    )
    if prop is not None and not prop.is_file():
        raise ValueError("property_file must be an existing file")
    if (
        max_refinements < 0
        or timeout_seconds <= 0
        or wall_timeout_seconds <= 0
        or recursion_limit <= 0
    ):
        raise ValueError("Refinement count, timeout, or recursion limit is invalid")

    identifier = (
        job_id
        or f"{source.stem}-{hashlib.sha256(str(source).encode()).hexdigest()[:8]}"
    )
    if identifier in {".", ".."} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier
    ):
        raise ValueError("Invalid job_id")

    workspace = Path(output_root).expanduser().resolve() / identifier
    workspace.mkdir(parents=True, exist_ok=False)
    output_dir = workspace / "output"
    output_dir.mkdir()

    source_hash = _hash_file(source)
    property_hash = _hash_file(prop) if prop else None
    original_text = source.read_text(encoding="utf-8", errors="replace")
    metadata = {
        "job_id": identifier,
        "source_file": str(source),
        "source_sha256": source_hash,
        "property_file": str(prop) if prop else None,
        "property_sha256": property_hash,
        "model": model,
        "frama_c_image": frama_c_image,
        "prover": prover,
        "timeout_seconds": timeout_seconds,
        "max_refinements": max_refinements,
        "status": "running",
    }
    _save_json(output_dir / "job.json", metadata)

    agent = build_agent(
        model=model,
        workspace=workspace,
        tools=[_input_reader(source, prop), *(tools or [])],
    )
    previous: VerificationResult | None = None
    candidates: list[str] = []
    try:
        for attempt in range(max_refinements + 1):
            candidate_name = f"candidate_{attempt:02d}.c"
            prompt = _render_prompt(
                "user.j2",
                job_id=identifier,
                source_file=str(source),
                property_file=str(prop) if prop else None,
                candidate_name=candidate_name,
                previous_candidate=(
                    f"/output/candidate_{attempt - 1:02d}.c" if attempt else None
                ),
                previous_verification=(
                    (
                        f"/output/parse_{attempt - 1:02d}.log"
                        if previous and previous.status == "parse_error"
                        else f"/output/verification_{attempt - 1:02d}.log"
                    )
                    if attempt
                    else None
                ),
                output_dir="/output",
                max_refinements=max_refinements,
                frama_c_image=frama_c_image,
                prover=prover,
                timeout_seconds=timeout_seconds,
            )
            prompt += (
                "\n\nCall read_job_input(kind='source') to read the assigned source. "
                "Read relevant skills under /skills/. Do not read sibling data files. "
                "Return a structured CandidateSpecification; annotated_c must contain "
                "the COMPLETE original C file with ACSL comments added, no Markdown. "
                "Python will save your structured response and perform verification. "
                "Do not execute Docker commands."
            )
            if previous is not None:
                prompt += "\n\nLATEST VERIFIER FEEDBACK\n" + previous.model_dump_json(
                    indent=2
                )
                log_path = output_dir / (
                    f"parse_{attempt - 1:02d}.log"
                    if previous.status == "parse_error"
                    else f"verification_{attempt - 1:02d}.log"
                )
                if log_path.exists():
                    prompt += (
                        "\n\nVerifier log tail:\n"
                        + log_path.read_text(encoding="utf-8")[-12000:]
                    )

            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"recursion_limit": recursion_limit},
            )
            candidate = _candidate_from_result(result)
            candidate_path = output_dir / candidate_name
            if not _same_executable_tokens(original_text, candidate.annotated_c):
                (output_dir / f"rejected_{candidate_name}").write_text(
                    candidate.annotated_c, encoding="utf-8"
                )
                raise ValueError(
                    f"{candidate_name} changes original C tokens; only ACSL annotations are permitted. "
                    "See rejected_candidate file."
                )

            candidate_path.write_text(candidate.annotated_c, encoding="utf-8")
            _save_json(
                output_dir / f"generation_{attempt:02d}.json", candidate.model_dump()
            )
            _save_json(
                output_dir / "analysis.json",
                {
                    "summary": candidate.summary,
                    "assumptions": candidate.assumptions,
                    "risks": candidate.risks,
                },
            )
            _save_json(
                output_dir / "contracts.json",
                [c.model_dump() for c in candidate.clauses],
            )
            candidates.append(candidate_name)

            if not verify:
                previous = VerificationResult(
                    candidate_file=candidate_name,
                    status="not_checked",
                    frama_c_image=frama_c_image,
                    prover=prover,
                    prover_timeout_seconds=timeout_seconds,
                    wall_timeout_seconds=wall_timeout_seconds,
                )
            else:
                previous = verify_candidate(
                    candidate_path,
                    image=frama_c_image,
                    prover=prover,
                    prover_timeout_seconds=timeout_seconds,
                    wall_timeout_seconds=wall_timeout_seconds,
                )
            _save_json(
                output_dir / f"verification_{attempt:02d}.json", previous.model_dump()
            )
            _save_json(output_dir / "verification.json", previous.model_dump())
            if previous.status in {"proved", "not_checked", "verification_error"}:
                break

        if _hash_file(source) != source_hash or (
            prop and _hash_file(prop) != property_hash
        ):
            raise RuntimeError("An original input changed during the job")

        metadata["status"] = "completed"
        metadata["candidates"] = candidates
        metadata["verification_status"] = previous.status if previous else "not_checked"
        _save_json(output_dir / "job.json", metadata)
        return {
            "job_id": identifier,
            "output_dir": str(output_dir),
            "candidate_files": candidates,
            "verification": previous.model_dump() if previous else None,
        }
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        _save_json(output_dir / "job.json", metadata)
        raise
