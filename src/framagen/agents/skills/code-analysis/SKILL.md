---
name: code-analysis
description: Inspect an assigned C program and optional verification property to identify function behavior, control flow, memory accesses, loops, call-site obligations, and evidence for ACSL specifications. Use before generating contracts and when a proof failure requires reexamining source semantics.
---

# C Program Analysis

## Procedure
1. Inspect only the source and explicit property inputs assigned to the current job. Do not read sibling solutions, patched sources, expected verdicts, or earlier output candidates.
2. Inventory functions, parameters, return values, global state, called library functions, pointer operations, allocations, and loops.
3. For each function, map relevant branches, reads, writes, dereferences, bounds, null termination, integer arithmetic, and termination conditions.
4. Identify call sites and the conditions callers must establish. Do not assume callee preconditions hold automatically.
5. Separate three kinds of information: **stated property** (explicit input), **source observation** (evidence from program text), and **assumption** (an additional condition that must be justified).
6. Highlight ambiguity and undefined behavior without repairing the program. If the program has custom benchmark intrinsics or verifier harness constructs, identify their semantics or mark them unsupported.
7. Record line ranges and function names supporting each proposed property. Identify risky or unsupported claims for later review.

## Output
Provide an analysis record containing source identity, function inventory, loops, memory operations, call sites, explicit properties, candidate conditions with provenance, and unresolved questions.

Do not infer a benchmark's expected verdict from its filename. Do not treat compiler success or runtime success as a formal proof.
