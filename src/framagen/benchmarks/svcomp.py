"""Select and convert SV-COMP loop tasks into Frama-C/WP tasks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from framagen.benchmarks.manifest import BenchmarkTask, VerifySettings, classify, count_loc, sha256_text
from framagen.benchmarks.sv2acsl import ConversionError, convert

SVCOMP_REPO = "https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks.git"
SVCOMP_TAG = "svcomp26"
LOOP_CATEGORIES = (
    "loop-invariants",
    "loop-simple",
    "loop-lit",
    "loop-zilu",
    "nla-digbench",
    "loop-invgen",
)
# SV-COMP tasks use ILP32; GCC extensions appear in the preprocessed .i files.
SVCOMP_VERIFY = VerifySettings(
    rte=False,  # the property is reachability; many tasks are expected to overflow
    machdep="gcc_x86_32",
    check_termination=False,  # unreach-call is a safety property (partial correctness)
    smoke_dead_code=False,
)


def _unreach_call_verdict(task: dict) -> bool | None:
    for prop in task.get("properties", []):
        if os.path.basename(str(prop.get("property_file", ""))) == "unreach-call.prp":
            return prop.get("expected_verdict")
    return None


def build_svcomp(
    sv_root: Path,
    out_root: Path,
    *,
    ref: str,
    categories: tuple[str, ...] = LOOP_CATEGORIES,
) -> list[BenchmarkTask]:
    """Convert every ``unreach-call: true`` task in ``categories``.

    Tasks that cannot be used keep a manifest row with ``excluded`` set, so
    dataset statistics stay auditable.
    """
    c_root = sv_root / "c"
    prop_src = c_root / "properties" / "unreach-call.prp"
    prop_dst = out_root / "svcomp" / "properties" / "unreach-call.prp"
    prop_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prop_src, prop_dst)

    tasks: list[BenchmarkTask] = []
    for category in categories:
        for yml in sorted((c_root / category).glob("*.yml")):
            meta = yaml.safe_load(yml.read_text(encoding="utf-8"))
            if _unreach_call_verdict(meta) is not True:
                continue
            inputs = meta["input_files"]
            input_name = inputs[0] if isinstance(inputs, list) else inputs
            source_path = yml.parent / input_name
            source = source_path.read_text(encoding="utf-8", errors="replace")
            task_id = f"svcomp/{category}/{yml.stem}"
            rel_dir = Path("svcomp") / category / yml.stem
            notes: list[str] = []
            excluded: str | None = None
            if meta.get("options", {}).get("data_model") not in {None, "ILP32"}:
                notes.append(f"data model {meta['options']['data_model']}")
            if "-reducer" in input_name:
                excluded = "machine-generated reducer variant (goto-based; WP rejects non-natural loops)"
            try:
                conversion = convert(source)
                text = conversion.text
                notes += conversion.warnings
            except ConversionError as exc:
                text = source
                excluded = excluded or f"conversion failed: {exc}"
            (out_root / rel_dir).mkdir(parents=True, exist_ok=True)
            (out_root / rel_dir / "task.c").write_text(text, encoding="utf-8")
            kind, loops, depth = classify(text)
            if loops == 0 and excluded is None:
                excluded = "no loops"
            tasks.append(
                BenchmarkTask(
                    id=task_id,
                    suite="svcomp",
                    category=category,
                    goal="loop-annotations",
                    upstream_repo=SVCOMP_REPO,
                    upstream_ref=ref,
                    upstream_path=f"c/{category}/{input_name}",
                    upstream_sha256=sha256_text(source),
                    task_file=str(rel_dir / "task.c"),
                    property_file=str(prop_dst.relative_to(out_root)),
                    target_property="unreach-call",
                    expected_verdict=True,
                    verify=SVCOMP_VERIFY,
                    control_flow=kind,
                    loc=count_loc(text),
                    loops=loops,
                    max_loop_depth=depth,
                    excluded=excluded,
                    notes=notes,
                )
            )
    return tasks
