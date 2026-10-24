# FramGen

An LLM agent that writes ACSL specifications for C programs, checks them with
Frama-C/WP, and refines them using the verifier's feedback.

```
task.c ──► LLM agent (deepagents + skills) ──► candidate_NN.c
              ▲                                   │
              │                          guards: C tokens unchanged, harness intact,
              │                                  no admit / ghost / (axioms)
              │                                   │
              └──── unproved goals, smoke tests ◄─ Frama-C/WP (-wp-smoke-tests [-wp-rte])
```

The LLM only writes annotations; Frama-C decides the verdict.

**Step-by-step instructions: [docs/RUNNING.md](docs/RUNNING.md)** (install, Frama-C, Ollama, single files, SV-COMP on SLURM, outputs, troubleshooting).

## Setup

```bash
uv sync                       # Python 3.11–3.13
uv sync --extra anthropic     # optional: anthropic:* models (also: --extra openai)
```

Frama-C runs in the official image `framac/frama-c:32.1` (Frama-C 32.1 Germanium, Alt-Ergo 2.5.4).

| Machine | Configuration |
|---|---|
| Laptop with Docker | nothing to set (`docker` is the default backend) |
| HPC with Apptainer | `apptainer pull frama-c_32.1.sif docker://framac/frama-c:32.1`, then `export FRAMAGEN_FRAMAC_BACKEND=apptainer FRAMAGEN_FRAMAC_IMAGE=/path/frama-c_32.1.sif`. If `/dev/fuse` is unavailable, unpack it first: `apptainer build --sandbox frama-c_32.1 frama-c_32.1.sif` and point the variable at the directory. |
| Native opam install | `export FRAMAGEN_FRAMAC_BACKEND=native` |

## Datasets

```bash
framagen fetch                       # SV-COMP @ svcomp26 (sparse), frama-c-problems @ d7d6daa
framagen build --calibrate --jobs 32 # -> benchmarks/manifest.jsonl, benchmarks/splits/demo.txt
framagen list --split demo
```

`benchmarks/manifest.jsonl` has one row per task: upstream path, commit, and SHA-256; the
prepared `task.c`; verification settings; control-flow class; WP verdicts for the unannotated
task and for the reference; and an exclusion reason if the task is unusable.

### SV-COMP (loop invariants for a fixed property)

`sv2acsl` ([src/framagen/benchmarks/sv2acsl.py](src/framagen/benchmarks/sv2acsl.py)) replaces
the SV-COMP harness with ACSL contracts, in the style of Frama-C-SV:

| harness function | ACSL contract | effect |
|---|---|---|
| `reach_error()` | `requires \false` | each call must be proved unreachable |
| `__VERIFIER_assert(c)` | `requires c != 0` | each call is a proof goal |
| `assume_abort_if_not(c)` | `ensures c != 0` | assumption |
| `abort()`, `exit()` | `ensures \false` | no normal return |
| `__VERIFIER_nondet_T()` | `assigns \nothing` | arbitrary value of `T` |

The rest of the program is unchanged. The harness is fenced by markers, and a candidate that
edits it is rejected.

Selection: `unreach-call` with expected verdict `true`, from the categories `loop-invariants`,
`loop-simple`, `loop-lit`, `loop-zilu`, `nla-digbench` and `loop-invgen`, at tag `svcomp26`.

| step | tasks |
|---|---|
| eligible YAML tasks (73 use a `.i` file) | 128 |
| − machine-generated `*-reducer.c` variants | −16 |
| − WP rejects with "non-natural loop" (`string_concat-noarr`) | −1 |
| − loops abstracted away (`*_abstracted`, trivially proved) | −5 |
| **usable** | **106**: 42 single-path, 43 multi-path, 21 nested |

None of the usable tasks is proved without generated invariants.

WP settings: `-machdep gcc_x86_32` (the tasks are ILP32); no `-wp-rte`, because the property is
reachability and many tasks are expected to overflow; termination is not checked, because
`unreach-call` is a safety property; smoke tests check preconditions and loop invariants
(dead-code probes are off, since proving the error branch dead is the point).

**Caveat:** without `-wp-rte`, WP treats paths with signed overflow as infeasible, so SV-COMP
proofs assume no overflow occurs.

### frama-c-problems (full contracts)

51 programs with hand-written ACSL. The task is the program with all ACSL removed; the original
is kept as `reference.c` and the agent never sees it. Classes: 5 sequential, 7 branched,
10 pointer, 17 single-path loops, 11 multi-path loops, 1 nested.

WP settings: `-wp-rte`, smoke tests, no termination goals (the references predate WP's default
`terminates` obligations). Under these settings 15/51 references verify, 23 are inconclusive
(mostly overflow), and 13 fail (5 do not parse; the others violate their own `main` or are vacuous).
A reference that fails does not make its task infeasible: a correct spec may need extra
preconditions.

### Demo split (24 tasks, 4 per class)

`splits/demo.txt` is chosen deterministically: the smallest tasks per class, round-robin across
upstream categories, preferring tasks whose reference verifies. Straight-line, branched and
pointer programs come from frama-c-problems; loops come from SV-COMP. Edit the file freely, or
write more `splits/<name>.txt` files.

## Running

```bash
framagen run prog.c                                   # agent writes ACSL, Frama-C/WP checks it
framagen run prog.c --property contract --timeout 30  # full contracts, 30 s per proof goal
framagen run sv_task.c --property unreach-call        # SV-COMP file: auto-converted with sv2acsl
framagen run --task svcomp/loop-zilu/benchmark32_linear
framagen verify annotated.c [--task ID] [--json]      # verifier only, no LLM
framagen properties                                   # what each target property means
framagen eval --split svcomp --run-name svcomp-qwen3 --timeout 30
framagen summarize runs/svcomp-qwen3
```

Defaults: `--model ollama:qwen3:8b --seed 42 --max-refinements 3`. `--property auto` picks
`unreach-call` when the file uses `__VERIFIER_assert`/`reach_error`, otherwise `contract`.
`--model` takes any LangChain `provider:model` string; for Ollama, `OLLAMA_HOST` or
`--ollama-hosts` selects the server(s), and the context window defaults to 32k tokens
(`--num-ctx`; Ollama's own default truncates the agent prompt without an error).
`--self-check` gives the agent a `check_candidate` tool so it can run WP itself before answering.

If the model ends without calling the structured-output tool, it is re-prompted (twice by default);
if it printed the complete annotated program in a code block instead, that program is used
(recorded as `generation: recovered`).

### On the minsky GPU node (SLURM)

```bash
sbatch scripts/slurm_eval.sh --split svcomp --run-name svcomp-qwen3-8b-s42-t30 \
    --model ollama:qwen3:8b --seed 42 --timeout 30
```

The script requests one L40S (the `gpu-limited` QOS allows 1 GPU per user), starts 4 Ollama
servers on it with `OLLAMA_NUM_PARALLEL=1` (one request at a time keeps seeded sampling
reproducible), and runs one task per server.
Resubmitting the same command resumes from `results.jsonl` after the 8 h limit.

### Target properties and the cross-check

| property | used by | target goals |
|---|---|---|
| `unreach-call` | SV-COMP | the call-site preconditions of every `__VERIFIER_assert(...)` / `reach_error()` call |
| `contract` | frama-c-problems, plain C | every WP goal (contracts, loop annotations, RTE if on) |

After WP, each property site in the candidate is mapped to its WP goal by line, and the
outcome is classified (`property.json`):

| status | meaning |
|---|---|
| `verified` | all property sites proved, every supporting annotation proved, smoke tests pass |
| `conditional` | all sites proved, but only under some unproved annotation |
| `unproved` | some site not proved |
| `vacuous` | smoke tests show inconsistent annotations |
| `missing-goals` | a site has no WP goal (encoding did not apply) |
| `not-checked` | parse error, rejected candidate, or tool error |

For SV-COMP the result is compared with the task's expected verdict (`unreach-call: true`):
`matches_expected` is `true` for `verified`, `false` for `vacuous`, and `null` otherwise (an
unproved goal is not a counterexample).

### Outputs

Each job writes `outputs/<job>/output/`:

- `candidate_NN.c` / `rejected_candidate_NN.c`: every attempt, never overwritten
- `verification_NN.json`, `wp_report_NN.json`, `parse_NN.log`, `verification_NN.log`: Frama-C results and raw logs
- `property_NN.json`: the target-property cross-check
- `transcript_NN.json`: the agent conversation (messages, tool calls, reasoning)
- `llm_usage.jsonl`: one record per generation (latency, token usage, LLM calls, status), collected
  with the `LLMRunMetrics` callback from `llm/callbacks`
- `job.log`: timestamped log of the job; `job.json`: settings, per-attempt results, totals
  (`input_tokens`, `output_tokens`, `llm_calls`, `llm_seconds`, `verify_seconds`, `seconds`)

An eval run writes `runs/<name>/results.jsonl` (resumable), `summary.md`/`summary.json` (by
control-flow class and suite, including property verified/conditional, agreement with the
expected verdict, tokens and time), and `run.log` (all jobs).

### Verdicts

| verdict | meaning |
|---|---|
| `VERIFIED` | every WP goal proved and no smoke test failed |
| `FAILED` | parse error, rejected edit, vacuous spec (a smoke test failed), or a goal refuted |
| `INCONCLUSIVE` | some goal unknown or timed out (Alt-Ergo cannot refute; matches SpecPylot's INCONCLUSIVE) |
| `ERROR` | tooling failure |

`VERIFIED` means the annotations are consistent and proved. It does not mean the contract is
complete: a weak but true `ensures` also verifies. For frama-c-problems, compare the result
with `reference.c` before claiming the spec is faithful.

## Tests

```bash
uv run --group dev pytest                      # offline tests; Frama-C tests run if a backend is set
```
