"""Turn frama-c-problems (annotated C programs) into contract-generation tasks.

Each upstream file already carries hand-written ACSL. The task file is the
same program with every ACSL annotation removed; the upstream file is kept as
``reference.c`` for comparison and is never shown to the agent.
"""

from __future__ import annotations

import re
from pathlib import Path

from framagen.benchmarks.manifest import BenchmarkTask, VerifySettings, classify, count_loc, sha256_text

FCP_REPO = "https://github.com/manavpatnaik/frama-c-problems.git"
# Partial correctness plus runtime errors: the upstream specs predate WP's
# default `terminates` obligations, and almost none of them carry loop variants.
FCP_VERIFY = VerifySettings(rte=True, machdep=None, check_termination=False, smoke_dead_code=True)

_ACSL = re.compile(r"/\*@.*?\*/|//@[^\n]*", re.DOTALL)


def strip_all_acsl(text: str) -> str:
    stripped = _ACSL.sub("", text)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    return re.sub(r"\n{3,}", "\n\n", stripped).strip() + "\n"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def build_frama_c_problems(fcp_root: Path, out_root: Path, *, ref: str) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    for source_path in sorted(fcp_root.rglob("*.c")):
        rel = source_path.relative_to(fcp_root)
        category = rel.parts[0] if len(rel.parts) > 1 else "root"
        name = _slug(source_path.stem)
        task_id = f"frama-c-problems/{category}/{name}"
        rel_dir = Path("frama-c-problems") / category / name
        source = source_path.read_text(encoding="utf-8", errors="replace")
        task_text = strip_all_acsl(source)
        (out_root / rel_dir).mkdir(parents=True, exist_ok=True)
        (out_root / rel_dir / "task.c").write_text(task_text, encoding="utf-8")
        (out_root / rel_dir / "reference.c").write_text(source, encoding="utf-8")
        kind, loops, depth = classify(task_text)
        notes = []
        if re.search(r"\baxiom\b", source):
            notes.append("reference uses axioms")
        tasks.append(
            BenchmarkTask(
                id=task_id,
                suite="frama-c-problems",
                category=category,
                goal="full-contract",
                upstream_repo=FCP_REPO,
                upstream_ref=ref,
                upstream_path=str(rel),
                upstream_sha256=sha256_text(source),
                task_file=str(rel_dir / "task.c"),
                reference_file=str(rel_dir / "reference.c"),
                target_property="contract",
                verify=FCP_VERIFY,
                control_flow=kind,
                loc=count_loc(task_text),
                loops=loops,
                max_loop_depth=depth,
                notes=notes,
            )
        )
    return tasks
