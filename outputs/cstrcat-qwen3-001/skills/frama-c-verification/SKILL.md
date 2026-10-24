---
name: frama-c-verification
description: Validate ACSL parsing and attempt deductive verification of an annotated C program using the registered Frama-C Docker tool. Use whenever an ACSL candidate is ready to check or verification logs need to be interpreted and summarized.
---

# Frama-C Verification

## Procedure
1. Verify that the candidate is a versioned annotated copy of the assigned original source and that executable statements are unchanged.
2. Invoke the **registered deterministic Frama-C verification tool**, if available. The tool should handle Docker, mounting, resource limits, the configured Frama-C image, parsing, WP, runtime-error obligations, and log capture. Do not compose arbitrary host shell commands.
3. Run parsing before proof attempts. On a parser/type error, record its exact source location and return the candidate for a targeted ACSL fix.
4. If parsing succeeds, invoke WP with runtime-error obligations as configured for the job. Record prover version, options, timeout, resource limits, source hash, and the raw output.
5. Attribute outcomes to specific obligations when possible: proved, unknown, timeout, failure, unreachable, unsupported, or tool error. Preserve the verifier's exact counts; do not convert timeouts into failures or confirmed bugs.
6. Summarize obligation types (e.g., requires at call sites, ensures, assigns, invariants, memory access, termination) without treating all goals as equivalent.
7. Distinguish (a) ACSL parsing, (b) formal proof under assumptions, (c) benchmark-property verification, and (d) fidelity to user requirements. They are not interchangeable.
8. Return exact diagnostics to the refinement workflow; if the tool is unavailable, mark the candidate `not_checked` and do not invent logs.

## Safety and reproducibility
Use a restricted container with no unnecessary network access, read-only original inputs, limited CPU/memory/runtime, and a dedicated writable output area. Do not execute the benchmark binary. Keep every run's logs and settings.

The SKILL.md is procedural guidance, not a substitute for implementing and registering the actual verification tool.
