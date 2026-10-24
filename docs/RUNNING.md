# Running FramGen

This guide walks through installing FramGen, checking a C file, running the SV-COMP benchmark,
and reading the results. Commands are run from the repository root unless noted.

- [1. Install](#1-install)
- [2. Frama-C](#2-frama-c)
- [3. The LLM (Ollama + qwen3:8b)](#3-the-llm-ollama--qwen38b)
- [4. Datasets](#4-datasets)
- [5. Run on a single C file](#5-run-on-a-single-c-file)
- [6. Check an annotated file without the LLM](#6-check-an-annotated-file-without-the-llm)
- [7. Run a benchmark task or a whole split](#7-run-a-benchmark-task-or-a-whole-split)
- [8. SV-COMP on the minsky GPU node (SLURM)](#8-sv-comp-on-the-minsky-gpu-node-slurm)
- [9. Reading the results](#9-reading-the-results)
- [10. Options reference](#10-options-reference)
- [11. Troubleshooting](#11-troubleshooting)

## 1. Install

You need git, Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is missing (installs to ~/.local/bin)
git clone https://github.com/ragibayon/framagen.git
cd framagen
uv sync --group dev                                  # creates .venv with framagen + pytest
source .venv/bin/activate                            # or prefix commands with `uv run`
framagen --help
```

Optional providers: `uv sync --extra anthropic` (`anthropic:*` models), `uv sync --extra openai`.

## 2. Frama-C

FramGen runs Frama-C 32.1 (WP + Alt-Ergo 2.5.4) from the official image `framac/frama-c:32.1`.
Choose one backend:

**Docker (laptop):** nothing to configure; `docker` is the default backend.

**Apptainer (HPC, e.g. minsky):**

```bash
mkdir -p ~/containers && cd ~/containers
apptainer pull frama-c_32.1.sif docker://framac/frama-c:32.1
# If running the SIF fails with "failed to open /dev/fuse", unpack it once:
APPTAINER_CACHEDIR=~/.apptainer-cache apptainer build --sandbox frama-c_32.1 frama-c_32.1.sif
cd -
export FRAMAGEN_FRAMAC_BACKEND=apptainer
export FRAMAGEN_FRAMAC_IMAGE=$HOME/containers/frama-c_32.1     # the sandbox dir (or the .sif)
```

Put the two `export` lines in `~/.bashrc` so every shell, and every SLURM job, uses them.

**Native opam install:** `export FRAMAGEN_FRAMAC_BACKEND=native` (needs `frama-c` on `PATH`).

Check the setup; this runs the Frama-C integration tests:

```bash
pytest -q          # 12 passed
```

## 3. The LLM (Ollama + qwen3:8b)

The default model is `ollama:qwen3:8b`, with seed 42, a 32k context window, and at most 8192
tokens per response.

**Install Ollama without root** (minsky). We use v0.40.2; any recent release works.

```bash
mkdir -p ~/.local/ollama-v0.40.2 && cd ~/.local/ollama-v0.40.2
curl -fsSL -o ollama.tar.zst https://github.com/ollama/ollama/releases/download/v0.40.2/ollama-linux-amd64.tar.zst
tar --zstd -xf ollama.tar.zst && rm ollama.tar.zst
ln -sfn ~/.local/ollama-v0.40.2 ~/.local/ollama
cd -
```

**Download the model once** (no GPU needed for the download):

```bash
OLLAMA_HOST=127.0.0.1:11534 ~/.local/ollama/bin/ollama serve &      # temporary server
OLLAMA_HOST=127.0.0.1:11534 ~/.local/ollama/bin/ollama pull qwen3:8b
kill %1
```

On minsky the GPUs are only usable inside a SLURM job (section 8). For interactive work, start a
server inside a GPU allocation:

```bash
srun --gres=gpu:nvidia_l40s:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=32768 \
    ~/.local/ollama/bin/ollama serve > ollama.log 2>&1 &
export OLLAMA_HOST=http://127.0.0.1:11434
```

On a laptop, the normal Ollama app works: `ollama pull qwen3:8b`.

## 4. Datasets

Skip this section if you only want to check your own C files.

```bash
framagen fetch                         # SV-COMP @ svcomp26 (loop categories, sparse) + frama-c-problems
framagen build                         # writes benchmarks/svcomp/** and benchmarks/frama-c-problems/**
framagen list --split demo             # 24-task demo split
framagen list --split svcomp           # 106 usable SV-COMP tasks
```

`benchmarks/manifest.jsonl` (versioned) already contains the WP calibration. `build` keeps it for
every upstream file whose hash is unchanged. To recompute it (about 5 minutes with 48 jobs):
`framagen build --calibrate --jobs 48`.

## 5. Run on a single C file

```bash
framagen run path/to/prog.c
```

What happens:

1. The target property is chosen with `--property` (default `auto`):
   - `contract`: plain C. The agent writes `requires`/`assigns`/`ensures` and loop annotations;
     WP checks them with runtime-error goals (`-wp-rte`).
   - `unreach-call`: an SV-COMP-style file (uses `__VERIFIER_assert` / `reach_error`). The file
     is converted automatically with sv2acsl; the agent writes loop annotations so that WP proves
     `reach_error()` is never called.
2. The agent reads the file, writes `candidate_00.c`, and FramGen checks it:
   - only ACSL may be added, and executable C tokens must be unchanged;
   - no `admit`, no ghost code, no axioms for unreach-call;
   - the sv2acsl harness must be untouched.
3. Frama-C/WP runs with smoke tests (vacuity checks). Unproved goals are sent back to the model,
   for up to `--max-refinements` more attempts (default 3).

Examples:

```bash
framagen run data/cstrcat_unsafe.c --property contract --timeout 30
framagen run my_sv_task.c --property unreach-call --timeout 30 --max-refinements 5
framagen run prog.c --model anthropic:claude-sonnet-5-5 --seed -1     # hosted model, no seed
framagen run prog.c --no-verify                                        # generate only
python scripts/submit_job.py prog.c                                    # same as `framagen run prog.c`
```

The last lines of output give the job folder and the outcome:

```
verdict:  VERIFIED  goals 6/6
property: unreach-call: verified — 1/1 property sites proved
```

## 6. Check an annotated file without the LLM

```bash
framagen verify annotated.c                         # property auto-detected
framagen verify candidate_02.c --task svcomp/loop-simple/nested_2 --timeout 30
framagen verify annotated.c --rte --termination --json
```

The exit code is 0 only for `VERIFIED`.

## 7. Run a benchmark task or a whole split

```bash
framagen run --task svcomp/loop-zilu/benchmark32_linear --timeout 30
framagen eval --split demo --run-name demo-qwen3 --timeout 30
framagen eval --split svcomp --limit 5 --run-name try5
framagen summarize runs/demo-qwen3                  # rebuild summary.md
```

Splits: `all`, `usable`, `svcomp`, `frama-c-problems`, or any `benchmarks/splits/<name>.txt`
(one task id per line). Runs resume: re-running the same `--run-name` skips tasks already in
`results.jsonl`; `--no-resume` starts over.

## 8. SV-COMP on the minsky GPU node (SLURM)

`scripts/slurm_eval.sh` requests one L40S (48 GB; the `gpu-limited` QOS allows **1 GPU,
32 CPUs and 240 GB of RAM per user**, so a second GPU is refused with `QOSMaxGRESPerUser`).
It starts 4 Ollama servers on that GPU (qwen3:8b with a 32k context uses about 10 GB each),
each with `OLLAMA_NUM_PARALLEL=1`, so every server handles one request at a time and seed 42
stays reproducible. It warms up the model and runs `framagen eval` with one task per server.
Measured with this setup: one SV-COMP task takes about 6 minutes (4 attempts), so the 106
tasks take roughly 3–4 hours.

```bash
# smoke test: 1 task, ~6 minutes
FRAMAGEN_SERVERS_PER_GPU=1 sbatch --time=01:00:00 scripts/slurm_eval.sh \
    --split svcomp --limit 1 --run-name smoke --model ollama:qwen3:8b --seed 42 --timeout 30

# full SV-COMP run (106 tasks), 30 s per proof goal
sbatch scripts/slurm_eval.sh --split svcomp --run-name svcomp-qwen3-8b-s42-t30 \
    --model ollama:qwen3:8b --seed 42 --timeout 30
```

Monitor and resume:

```bash
squeue -u $USER
tail -f runs/slurm-<jobid>.out                      # one line per finished task
tail -f runs/svcomp-qwen3-8b-s42-t30/run.log         # every job's log lines
# after the 8 h time limit, submit the same command again; finished tasks are skipped
```

Script settings, overridable through environment variables:

| variable | default |
|---|---|
| `FRAMAGEN_FRAMAC_BACKEND` / `FRAMAGEN_FRAMAC_IMAGE` | `apptainer` / `~/containers/frama-c_32.1` |
| `OLLAMA_BIN` | `~/.local/ollama/bin/ollama` |
| `FRAMAGEN_OLLAMA_MODEL` | `qwen3:8b` (the tag to warm up) |
| `FRAMAGEN_NUM_CTX` | `32768` |
| `FRAMAGEN_SERVERS_PER_GPU` | `4` (parallel tasks) |

Ollama server logs go to `runs/ollama-<jobid>-gpu<G>-s<N>.log`.

Faster runs, when time is short:

```bash
# the 24-task demo split (12 SV-COMP + 12 frama-c-problems; about 40 minutes)
sbatch scripts/slurm_eval.sh --split demo --run-name demo-qwen3 --timeout 30
# fewer refinement rounds (attempts take ~85 s each)
sbatch scripts/slurm_eval.sh --split svcomp --run-name svcomp-r1 --timeout 30 --max-refinements 1
```

Unit tests: `pytest tests/test_offline.py` (about 1 s, no Frama-C); `pytest` runs all of them
(about 25 s, needs a Frama-C backend).

## 9. Reading the results

### One job: `outputs/<job>/output/` (or `runs/<run>/jobs/<task>/output/`)

| file | content |
|---|---|
| `job.json` | settings, per-attempt results, totals: `seconds`, `usage.input_tokens`, `usage.output_tokens`, `usage.llm_calls`, `usage.llm_seconds`, `usage.verify_seconds` |
| `job.log` | timestamped log of the job |
| `candidate_NN.c` | each accepted attempt (never overwritten) |
| `rejected_candidate_NN.c` | an attempt that changed code or used a forbidden construct |
| `verification_NN.json` | WP verdict, per-goal outcomes, smoke tests, settings |
| `verification_NN.log`, `parse_NN.log`, `wp_report_NN.json` | raw Frama-C output |
| `property_NN.json` | target-property cross-check (see below) |
| `transcript_NN.json` | the agent conversation: messages, tool calls, reasoning |
| `llm_usage.jsonl` | one record per generation: latency, token usage, LLM calls, status (`accepted`, `recovered`, `no_structured_response`, `error`) |
| `generation_NN.json`, `analysis.json`, `contracts.json` | the model's structured answer and clause inventory |

### Verdict (whole file)

| verdict | meaning |
|---|---|
| `VERIFIED` | every WP goal proved, no smoke test failed |
| `FAILED` | parse error, rejected candidate, vacuous spec, or a refuted goal |
| `INCONCLUSIVE` | some goal unknown or timed out |
| `ERROR` | tool failure |

### Target property (`property.json`)

For `unreach-call`, each `__VERIFIER_assert(...)` / `reach_error()` call in the candidate is
matched, by line, to its WP goal.

| status | meaning |
|---|---|
| `verified` | every property site proved, every annotation the proof uses proved, smoke tests pass |
| `conditional` | every site proved, but some supporting loop invariant or assert is unproved |
| `unproved` | some site not proved |
| `vacuous` | the annotations are inconsistent (smoke test) |
| `missing-goals` | a site has no WP goal |
| `not-checked` | no WP result |

`matches_expected` compares this with the SV-COMP expected verdict (`true` for every selected
task): `true` if verified, `false` if vacuous, `null` otherwise. An unproved goal is not a
counterexample.

### A run: `runs/<name>/`

- `results.jsonl`: one row per task (verdict, property status, sites proved, attempts, seconds, tokens)
- `summary.md` / `summary.json`: counts per control-flow class and suite, property verified /
  conditional, agreement with the expected verdict, verified on the first attempt, mean attempts
  and time, total tokens
- `run.log`: all job logs interleaved; `errors/<task>.txt`: tracebacks of agent errors
- `run-<timestamp>.json`: the exact command-line configuration of each submission

## 10. Options reference

`framagen run --help` and `framagen eval --help` list every option. The most used:

| option | default | meaning |
|---|---|---|
| `--property {auto,unreach-call,contract}` | `auto` | what must be proved |
| `--model` | `ollama:qwen3:8b` | LangChain `provider:model` |
| `--seed` | `42` | sampling seed (`-1` = none) |
| `--max-output-tokens` | `8192` | cap per model response (Ollama `num_predict`) |
| `--num-ctx` | `32768` | Ollama context window |
| `--max-refinements` | `3` | extra attempts with verifier feedback |
| `--timeout` | `10` | prover seconds per proof goal (use `30` for SV-COMP runs) |
| `--wall-timeout` | `900` | seconds per Frama-C run |
| `--rte` / `--no-rte` | per property | runtime-error goals (`-wp-rte`) |
| `--termination` / `--no-termination` | off | prove termination |
| `--machdep` | per property | e.g. `gcc_x86_32` (SV-COMP is ILP32) |
| `--self-check` | off | agent may run Frama-C itself before answering |
| `--ollama-hosts` | — | comma-separated servers for `eval` (one task per server) |
| `--backend`, `--image` | env | Frama-C backend and image |

Default WP settings per property:

| property | `-wp-rte` | termination | machdep | smoke tests |
|---|---|---|---|---|
| `unreach-call` | off (WP then assumes no signed overflow) | off | `gcc_x86_32` | preconditions + loop invariants |
| `contract` | on | off | Frama-C default | all |

## 11. Troubleshooting

| symptom | fix |
|---|---|
| `failed to open /dev/fuse` (Apptainer) | use the unpacked sandbox directory (section 2) |
| `"frama-c": executable file not found` | point `FRAMAGEN_FRAMAC_IMAGE` at the image; FramGen sets the image's opam `PATH` itself |
| `nvidia-smi`: "No devices were found" on minsky | GPUs are only visible inside a SLURM job (`srun`/`sbatch --gres=gpu:...`) |
| `Could not connect to a running Ollama instance` | start `ollama serve`, and set `OLLAMA_HOST` or `--ollama-hosts` |
| job log shows "model returned no structured answer; re-prompting" | normal for qwen3:8b; FramGen re-prompts twice, then falls back to a C code block in the reply |
| one generation runs for many minutes | lower `--max-output-tokens` (qwen3 thinking can loop) |
| `rejected_candidate_NN.c` | the model edited C code or the harness; the reason is in `verification_NN.json` → `diagnostics` |
| `Non-natural loop detected` | WP does not support that goto-based control flow; such SV-COMP tasks are excluded |
| `FileExistsError` for a job folder | use a new `--job-id`/`--run-name`, or resume the run (eval does this automatically) |
