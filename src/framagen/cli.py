"""framagen — generate ACSL specifications for C programs with an LLM and check them with Frama-C/WP.

Quick start
  Verify one C file (the agent writes the ACSL, Frama-C checks it):
      framagen run prog.c --model ollama:qwen3:8b

  Choose what must be proved:
      framagen run prog.c --property contract        # full function contracts (default for plain C)
      framagen run sv_task.c --property unreach-call # SV-COMP: reach_error() is never called

  Check a file you annotated yourself (no LLM):
      framagen verify annotated.c

  Benchmarks:
      framagen fetch && framagen build --calibrate   # SV-COMP + frama-c-problems
      framagen list --split demo
      framagen run --task svcomp/loop-zilu/benchmark32_linear
      framagen eval --split svcomp --run-name svcomp-qwen3 --timeout 30

Run `framagen <command> --help` for every option.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from framagen.benchmarks.manifest import BenchmarkTask, VerifySettings, load_manifest, load_split
from framagen.utils.frama_c import FramaCRunner, verify_candidate

DEFAULT_RAW = Path("datasets/raw")
DEFAULT_BENCH = Path("benchmarks")
DEFAULT_RUNS = Path("runs")
DEFAULT_MODEL = "ollama:qwen3:8b"
DEFAULT_SEED = 42
PROPERTY_CHOICES = ("auto", "unreach-call", "contract")


class _Formatter(argparse.RawDescriptionHelpFormatter):
    """Show real defaults only (not None/False) and keep example epilogs verbatim."""

    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or ""
        if action.default not in (None, False, argparse.SUPPRESS) and "%(default)" not in text:
            text += " (default: %(default)s)"
        return text


# ---------------------------------------------------------------- arguments
def _add_framac_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Frama-C / WP")
    group.add_argument("--backend", choices=["docker", "apptainer", "native"],
                       help="how to run Frama-C (default: $FRAMAGEN_FRAMAC_BACKEND, else docker)")
    group.add_argument("--image", help="Docker image, or Apptainer SIF/sandbox path (default: $FRAMAGEN_FRAMAC_IMAGE)")
    group.add_argument("--prover", default="alt-ergo", help="SMT prover used by WP")
    group.add_argument("--timeout", type=int, default=10, metavar="SEC", help="prover timeout per proof goal")
    group.add_argument("--wall-timeout", type=int, default=900, metavar="SEC", help="limit for one Frama-C run")


def _add_settings_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("proof obligations (override the property/task defaults)")
    _bool_flag(group, "rte", "rte", "prove absence of runtime errors (-wp-rte)")
    _bool_flag(group, "termination", "termination", "prove termination (loop variants)")
    group.add_argument("--machdep", help="Frama-C machine model, e.g. gcc_x86_32 for SV-COMP (ILP32)")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LLM")
    group.add_argument("--model", default=DEFAULT_MODEL, help="LangChain provider:model")
    group.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed (use -1 for none)")
    group.add_argument("--temperature", type=float, help="sampling temperature (default: model's own)")
    group.add_argument("--num-ctx", type=int, help="Ollama context window in tokens (default 32768)")
    group.add_argument("--max-output-tokens", type=int, default=8192,
                       help="cap on tokens per model response (Ollama num_predict)")
    group.add_argument("--ollama-hosts", metavar="URL[,URL]",
                       help="Ollama servers; eval runs one task per server at a time (one server per GPU)")
    group.add_argument("--max-refinements", type=int, default=3, metavar="N",
                       help="extra attempts that feed WP results back to the model")
    group.add_argument("--self-check", action="store_true", help="let the agent call Frama-C itself before answering")
    group.add_argument("--recursion-limit", type=int, default=100, help="max agent steps per attempt")


def _bool_flag(parser, name: str, dest: str, help: str) -> None:
    parser.add_argument(f"--{name}", dest=dest, action="store_true", default=None, help=help)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=argparse.SUPPRESS)


def _runner(args: argparse.Namespace) -> FramaCRunner:
    return FramaCRunner.from_env(image=args.image, backend=args.backend)


def _model(args: argparse.Namespace, base_url: str | None = None):
    from framagen.models import make_model

    seed = None if args.seed is None or args.seed < 0 else args.seed
    return make_model(
        args.model, temperature=args.temperature, num_ctx=args.num_ctx, seed=seed,
        base_url=base_url, max_output_tokens=args.max_output_tokens,
    )


def _task(bench: Path, task_id: str) -> BenchmarkTask:
    for task in load_manifest(bench):
        if task.id == task_id:
            return task
    raise SystemExit(f"Unknown task: {task_id} (see `framagen list --split all`)")


def _settings_from_args(args: argparse.Namespace, base: VerifySettings) -> VerifySettings:
    update = {}
    if args.rte is not None:
        update["rte"] = args.rte
    if args.termination is not None:
        update["check_termination"] = args.termination
    if args.machdep:
        update["machdep"] = args.machdep
    return base.model_copy(update=update)


# ---------------------------------------------------- plain C file handling
def _detect_property(text: str) -> str:
    return "unreach-call" if ("__VERIFIER_assert" in text or "reach_error" in text) else "contract"


def _prepare_source(path: Path, prop: str, workdir: Path) -> tuple[Path, str, VerifySettings, list[str]]:
    """Turn a user's C file into a job input for ``prop``.

    unreach-call: an SV-COMP-style file (``__VERIFIER_*``, ``reach_error``) is
    converted with sv2acsl, unless it already carries the harness.
    """
    from framagen.benchmarks.svcomp import SVCOMP_VERIFY
    from framagen.benchmarks.sv2acsl import HARNESS_BEGIN, ConversionError, convert

    text = path.read_text(encoding="utf-8", errors="replace")
    if prop == "auto":
        prop = _detect_property(text)
    notes = [f"target property: {prop}"]
    if prop == "contract":
        return path, prop, VerifySettings(rte=True, check_termination=False), notes
    if HARNESS_BEGIN in text:
        return path, prop, SVCOMP_VERIFY, notes
    try:
        conversion = convert(text)
    except ConversionError as exc:
        raise SystemExit(f"{path}: cannot use --property unreach-call: {exc}")
    converted = workdir / (path.stem.replace(".", "_") + ".c")
    converted.write_text(conversion.text, encoding="utf-8")
    notes.append(f"converted SV-COMP harness with sv2acsl -> {converted}")
    return converted, prop, SVCOMP_VERIFY, notes


# ----------------------------------------------------------------- commands
def cmd_fetch(args: argparse.Namespace) -> int:
    from framagen.benchmarks.build import fetch

    print(json.dumps(fetch(args.raw, svcomp_tag=args.svcomp_tag), indent=2))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from framagen.benchmarks.build import build, calibrate, select_demo, summarize, write_split

    tasks = build(args.raw, args.bench)
    print(f"Built {len(tasks)} tasks into {args.bench}")
    if args.calibrate:
        tasks = calibrate(
            args.bench, runner=_runner(args), jobs=args.jobs, prover_timeout_seconds=args.timeout,
            progress=(lambda m: print("  " + m, flush=True)) if args.verbose else None,
        )
    demo = select_demo(tasks, per_class=args.per_class, max_loc=args.max_loc)
    write_split(
        args.bench, "demo", demo,
        header=f"Demo split: {args.per_class} tasks per control-flow class, loc <= {args.max_loc}.\n"
        "Generated by `framagen build`; edit freely.",
    )
    print(summarize(tasks))
    print(f"\nDemo split ({len(demo)} tasks) -> {args.bench / 'splits' / 'demo.txt'}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from framagen.benchmarks.build import calibrate, summarize

    print(summarize(calibrate(args.bench, runner=_runner(args), jobs=args.jobs, prover_timeout_seconds=args.timeout)))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    tasks = load_split(args.bench, args.split)
    for task in tasks:
        flag = f"  [excluded: {task.excluded}]" if task.excluded else ""
        print(f"{task.id:58s} {task.target_property:13s} {task.control_flow:17s} loc={task.loc:<4d}{flag}")
    print(f"{len(tasks)} tasks")
    return 0


def cmd_properties(args: argparse.Namespace) -> int:
    from framagen.properties import PROPERTIES

    for spec in PROPERTIES.values():
        print(f"{spec.name}\n  {spec.title}\n  {spec.description}")
        if spec.formal:
            print(f"  formal: {spec.formal}")
        print()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from framagen.properties import PROPERTIES, assess_property

    if args.task:
        task = _task(args.bench, args.task)
        prop, base, expected = task.target_property, task.verify, task.expected_verdict
    else:
        text = args.file.read_text(encoding="utf-8", errors="replace")
        prop = _detect_property(text) if args.property == "auto" else args.property
        if prop == "unreach-call":
            from framagen.benchmarks.svcomp import SVCOMP_VERIFY
            from framagen.benchmarks.sv2acsl import HARNESS_BEGIN

            if HARNESS_BEGIN not in text:
                raise SystemExit("unreach-call needs the sv2acsl harness; annotate a converted task.c instead")
            base = SVCOMP_VERIFY
        else:
            base = VerifySettings(rte=True, check_termination=False)
        expected = None
    options = _settings_from_args(args, base).to_options(
        prover=args.prover, prover_timeout_seconds=args.timeout, wall_timeout_seconds=args.wall_timeout
    )
    with tempfile.TemporaryDirectory(prefix="framagen-verify-") as logs:
        result = verify_candidate(args.file, runner=_runner(args), options=options, log_dir=logs)
    assessment = assess_property(PROPERTIES[prop], args.file.read_text(encoding="utf-8"), result, expected_verdict=expected)
    if args.json:
        print(json.dumps({"verification": result.model_dump(), "property": assessment.model_dump()}, indent=2))
    else:
        print(f"verdict:  {result.verdict} ({result.status}), goals {result.proved_goals}/{result.total_goals}")
        print(f"property: {prop}: {assessment.status} — {assessment.summary}")
        for goal in result.unproved_goals[:20]:
            print(f"  unproved: {goal}")
        for goal in result.smoke_tests_failed:
            print(f"  smoke test failed: {goal}")
    return 0 if result.verdict == "VERIFIED" else 1


def cmd_run(args: argparse.Namespace) -> int:
    from framagen.agents.agent import run_job
    from framagen.evaluation import job_slug

    if bool(args.task) == bool(args.source):
        raise SystemExit("Give a C file (framagen run prog.c) or a benchmark task (--task ID), not both")
    workdir = Path(tempfile.mkdtemp(prefix="framagen-input-"))
    if args.task:
        task = _task(args.bench, args.task)
        source = args.bench / task.task_file
        prop_file = args.bench / task.property_file if task.property_file else None
        target, base, task_id, expected = task.target_property, task.verify, task.id, task.expected_verdict
        notes: list[str] = []
    else:
        source, target, base, notes = _prepare_source(Path(args.source), args.property, workdir)
        prop_file, task_id, expected = args.property_file, None, None
        default_prp = args.bench / "svcomp/properties/unreach-call.prp"
        if target == "unreach-call" and prop_file is None and default_prp.is_file():
            prop_file = default_prp
    for note in notes:
        print(note)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_id = args.job_id or f"{job_slug(task_id or Path(args.source).stem)}-{stamp}"
    result = run_job(
        source_file=source,
        property_file=prop_file,
        output_root=args.output_root,
        model=_model(args),
        job_id=job_id,
        task_id=task_id,
        target_property=target,
        expected_verdict=expected,
        goal="loop-annotations" if target == "unreach-call" else "full-contract",
        max_refinements=args.max_refinements,
        runner=_runner(args),
        verify_options=_settings_from_args(args, base).to_options(
            prover=args.prover, prover_timeout_seconds=args.timeout, wall_timeout_seconds=args.wall_timeout
        ),
        verify=not args.no_verify,
        self_check=args.self_check,
        recursion_limit=args.recursion_limit,
    )
    verification = result["verification"] or {}
    prop = result["property"] or {}
    print(json.dumps({k: result[k] for k in ("job_id", "output_dir", "attempts", "seconds", "usage")}, indent=2))
    print(f"verdict:  {verification.get('verdict')}  goals {verification.get('proved_goals')}/{verification.get('total_goals')}")
    print(f"property: {target}: {prop.get('status', 'not-checked')} — {prop.get('summary', '')}")
    return 0 if verification.get("verdict") in {"VERIFIED", "NOT_CHECKED"} else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from framagen.evaluation import run_evaluation

    tasks = load_split(args.bench, args.split)
    if args.limit:
        tasks = tasks[: args.limit]
    run_name = args.run_name or f"{args.split}-{args.model.replace(':', '_').replace('/', '_')}"
    run_dir = args.runs / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}
    config["started"] = datetime.now(timezone.utc).isoformat()
    config["tasks"] = [t.id for t in tasks]
    (run_dir / f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    hosts = [h.strip() for h in (args.ollama_hosts or "").split(",") if h.strip()]

    def progress(row: dict) -> None:
        print(
            f"[{row['verdict']:12s}] {row['task_id']}  property={row.get('property_status', '-')} "
            f"attempts={row.get('attempts', '-')} {row.get('seconds', '-')}s",
            flush=True,
        )

    rows = run_evaluation(
        tasks,
        bench_root=args.bench,
        run_dir=run_dir,
        jobs=args.jobs,
        resume=not args.no_resume,
        progress=progress,
        model=_model(args),
        models=[_model(args, base_url=h) for h in hosts] or None,
        model_name=args.model,
        runner=_runner(args),
        max_refinements=args.max_refinements,
        prover=args.prover,
        prover_timeout_seconds=args.timeout,
        wall_timeout_seconds=args.wall_timeout,
        self_check=args.self_check,
        recursion_limit=args.recursion_limit,
    )
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    return 0 if rows else 1


def cmd_summarize(args: argparse.Namespace) -> int:
    from framagen.evaluation import load_rows, write_summary

    write_summary(args.run_dir, load_rows(args.run_dir))
    print((args.run_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="framagen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help="benchmark root (contains manifest.jsonl)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("run", formatter_class=_Formatter, help="generate + verify ACSL for one C file or task",
                       description="Generate ACSL for one C file with the LLM agent and verify it with Frama-C/WP.",
                       epilog="examples:\n"
                              "  framagen run prog.c\n"
                              "  framagen run prog.c --property contract --rte --timeout 30\n"
                              "  framagen run sv_task.c --property unreach-call\n"
                              "  framagen run --task svcomp/loop-zilu/benchmark32_linear --max-refinements 5\n"
                              "outputs: outputs/<job>/output/ (candidates, WP logs, property.json, llm_usage.jsonl, job.log)")
    target = p.add_argument_group("what to verify")
    target.add_argument("source", nargs="?", help="C file to annotate")
    target.add_argument("--task", help="benchmark task id instead of a file (see `framagen list`)")
    target.add_argument("--property", choices=PROPERTY_CHOICES, default="auto",
                        help="target property; auto = unreach-call if the file uses __VERIFIER_assert/reach_error, else contract")
    target.add_argument("--property-file", type=Path, help="optional property file shown to the agent (e.g. an SV-COMP .prp)")
    out = p.add_argument_group("output")
    out.add_argument("--output-root", type=Path, default=Path("outputs"), help="where job folders are created")
    out.add_argument("--job-id", help="job folder name (default: <file>-<timestamp>)")
    out.add_argument("--no-verify", action="store_true", help="only generate; skip Frama-C")
    _add_model_args(p)
    _add_settings_args(p)
    _add_framac_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify", formatter_class=_Formatter, help="check an annotated C file with Frama-C/WP (no LLM)",
                       description="Run Frama-C/WP on an annotated C file and report the verdict and the target property.",
                       epilog="examples:\n"
                              "  framagen verify my_annotated.c\n"
                              "  framagen verify cand.c --task svcomp/loop-simple/nested_2 --timeout 30\n"
                              "  framagen verify cand.c --rte --termination --json")
    p.add_argument("file", type=Path, help="annotated C file")
    p.add_argument("--task", help="use this benchmark task's property and WP settings")
    p.add_argument("--property", choices=PROPERTY_CHOICES, default="auto", help="target property to assess")
    p.add_argument("--json", action="store_true", help="print the full JSON result")
    _add_settings_args(p)
    _add_framac_args(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("eval", formatter_class=_Formatter, help="run the agent over a benchmark split",
                       description="Run the agent on every task of a split; resumable (results.jsonl).",
                       epilog="examples:\n"
                              "  framagen eval --split demo --run-name demo-qwen3\n"
                              "  framagen eval --split svcomp --timeout 30 --ollama-hosts http://127.0.0.1:11434,http://127.0.0.1:11435 --jobs 2\n"
                              "  sbatch scripts/slurm_eval.sh --split svcomp --run-name svcomp-qwen3 --timeout 30\n"
                              "outputs: runs/<name>/ (results.jsonl, summary.md, summary.json, run.log, jobs/)")
    p.add_argument("--split", default="demo", help="all | usable | svcomp | frama-c-problems | name of splits/<name>.txt")
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="parent folder for runs")
    p.add_argument("--run-name", help="run folder name (default: <split>-<model>)")
    p.add_argument("--jobs", type=int, default=1, help="tasks in parallel")
    p.add_argument("--limit", type=int, help="only the first N tasks")
    p.add_argument("--no-resume", action="store_true", help="re-run tasks already in results.jsonl")
    _add_model_args(p)
    _add_framac_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("summarize", formatter_class=_Formatter, help="rebuild summary.md for a run")
    p.add_argument("run_dir", type=Path)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("list", formatter_class=_Formatter, help="list benchmark tasks")
    p.add_argument("--split", default="usable", help="all | usable | svcomp | frama-c-problems | <splits/NAME.txt>")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("properties", help="describe the target properties")
    p.set_defaults(func=cmd_properties)

    p = sub.add_parser("fetch", formatter_class=_Formatter, help="download pinned SV-COMP and frama-c-problems")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--svcomp-tag", default="svcomp26")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", formatter_class=_Formatter, help="convert downloaded datasets into tasks")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--calibrate", action="store_true", help="run WP on unannotated tasks and references")
    p.add_argument("--jobs", type=int, default=8, help="parallel Frama-C runs")
    p.add_argument("--per-class", type=int, default=4, help="demo tasks per control-flow class")
    p.add_argument("--max-loc", type=int, default=60, help="largest demo task (lines of code)")
    p.add_argument("-v", "--verbose", action="store_true")
    _add_framac_args(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("calibrate", formatter_class=_Formatter, help="re-run WP baselines for the manifest")
    p.add_argument("--jobs", type=int, default=8)
    _add_framac_args(p)
    p.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
