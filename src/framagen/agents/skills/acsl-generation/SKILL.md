---
name: acsl-generation
description: Generate ACSL function contracts and loop annotations for a C source, including preconditions, postconditions, assigns clauses, loop invariants, loop assigns, and loop variants. Use after code analysis or when an ACSL candidate needs a targeted syntax or specification correction.
---

# ACSL Specification Generation

## Procedure
1. Start from the program analysis and the explicitly supplied property. Formalize relevant behavior and safety obligations; label extra environmental assumptions.
2. Generate function-level `requires`, `ensures`, and `assigns` clauses. Use named behaviors and `assumes` when justified by distinct functional cases. Describe return-value dependencies only when supported.
3. For pointer and array operations, specify readable/writable ranges, memory separation where needed, frame conditions, and relevant integer/bounds constraints.
4. For loops, propose inductive invariants, a precise `loop assigns` footprint, and `loop variant` when termination must be proved. Preserve facts from loop entry that are required to prove postconditions.
5. Use ACSL constructs with correct state semantics: `\result`, `\old`, `\at`, `\valid`, `\valid_read`, `\separated`, and quantified predicates as applicable. Do not use `\old` inside loop annotations; use appropriate `\at` labels instead.
6. Maintain the original executable C statements, declarations, includes, and control flow. Add ACSL comments only. Do not add helper functions or ghost code without explicit authorization.
7. Ensure generated preconditions do not hide an original unsafe call. The verifier must attempt to prove each precondition at its call site.
8. Do not introduce tautological postconditions, contradictory preconditions, undefined helper predicates, or broad frame conditions when a precise description is possible.
9. Generate a clause inventory linking every annotation to a stated property, source observation, or justified assumption; identify uncertainty openly.

## Review checklist
- ACSL can be parsed by the targeted Frama-C version.
- Annotations are meaningful and not vacuous.
- Function calls, array bounds, null termination, and mutable memory are accounted for.
- Loop invariants are expected to hold initially and be preserved; variants should decrease and remain nonnegative.
- Source behavior has not been modified.

A plausible contract is not a verified contract. Send the candidate to the verification tool and revise only on evidence.
