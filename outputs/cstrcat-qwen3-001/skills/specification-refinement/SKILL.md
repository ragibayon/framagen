---
name: specification-refinement
description: Refine generated ACSL annotations in response to Frama-C parse errors, unproved conditions, missing loop invariants, frame warnings, and prover diagnostics. Use after a verification attempt when meaningful, justified specification changes may improve the candidate.
---

# Specification Refinement

## Procedure
1. Read the annotated candidate, original source analysis, exact verifier diagnostics, and prior refinement decisions.
2. Classify each unresolved obligation: ACSL syntax/type issue; missing or weak invariant; incorrect or missing `assigns`; missing termination argument; unproved call-site precondition; unsupported model; insufficient prover resources; or likely invalid requirement/assumption.
3. Prefer a targeted annotation edit over regenerating the full specification. Make one clearly motivated set of changes per iteration.
4. Preserve justified requirements and safety conditions. Do not drop difficult postconditions, weaken assertions, or add stronger preconditions merely to make goals pass.
5. Never change executable C code. Do not conceal the benchmark's intended defect. For unsafe callers, retain and report unproved callee preconditions rather than assuming them away.
6. Save a new, numbered candidate and a refinement record showing changed clauses, reason, supporting diagnostic, and the intended proof effect. Never overwrite an earlier candidate.
7. Rerun parsing and verification through the registered tool. Compare outcomes by property/obligation, not only raw number of proved goals (the total number of goals may change).
8. Stop when all required obligations are proved, the configured iteration limit is reached, no justified improvement remains, or verification is unavailable.

## Reporting
State which claims were actually proved, which remain unresolved, whether assumptions changed, and whether the generated annotations are faithful to the supplied property. An increase in proof count alone is not evidence of correctness.
