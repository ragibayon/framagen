---
name: loop-invariant-synthesis
description: Synthesize inductive loop invariants, loop assigns and (optionally) loop variants so that Frama-C/WP can prove a fixed safety property, such as the call-site preconditions of an SV-COMP harness (reach_error / __VERIFIER_assert). Use for loop-annotation tasks and whenever WP reports an unproved "established", "preserved" or call-site goal after a loop.
---

# Loop Invariant Synthesis

## Harness semantics (sv2acsl)
- `reach_error()` carries `requires \false`; a call is proved only if WP shows it is unreachable.
- `__VERIFIER_assert(c)` carries `requires c != 0`; each call is a goal on `c`.
- `assume_abort_if_not(c)` ensures `c != 0` on return; `abort()` never returns. They restrict inputs.
- `__VERIFIER_nondet_T()` returns any value of type `T` and assigns nothing.
The harness block is part of the task; never edit it or add contracts to these functions.

## Procedure
1. Locate each property site (`__VERIFIER_assert`, `reach_error`) and the loops that precede it.
2. Write the property as a formula over program variables at the loop exit. The invariant `I`
   together with the negated loop condition must imply it: `I && !cond ==> P`.
3. Collect candidate facts:
   - variable ranges from initial values and the loop condition (`0 <= i <= n`);
   - equalities preserved by the body (`x + y == \at(x, LoopEntry) + \at(y, LoopEntry)`,
     `y == 2 * i`, polynomial relations such as `x == i * i` for nonlinear tasks);
   - facts established before the loop that the body does not change (repeat them: WP forgets
     everything about modified variables that the invariant does not state);
   - for multi-path bodies, a relation that every branch preserves (often a disjunction or an
     inequality rather than an equality).
4. Check each candidate by hand: (a) holds when the loop is first reached, (b) preserved by one
   iteration of every path through the body, including `break`/`continue`.
5. Add `loop assigns` listing exactly the variables (and memory) the loop writes.
6. Nested loops: annotate every loop. The inner invariant must carry what the outer invariant needs
   (outer counters, accumulated sums), because the inner loop's `loop assigns` hides other facts.
7. Infinite loops written `while (1)` with `break`: the property after the loop is reached only by
   `break`; the invariant must hold at every check of the condition.
8. Prefer several small `loop invariant` clauses over one large conjunction: WP reports each one
   separately, which makes refinement targeted.

## Reading WP feedback
- `*_loop_invariant_*_established`: false on entry; weaken it or fix the initial-value reasoning.
- `*_loop_invariant_*_preserved`: not inductive; strengthen with the missing relation, or split cases.
- `*_call___VERIFIER_assert_requires` / `*_call_reach_error_requires`: the invariant is too weak to
  imply the property at exit; add the relation the assertion needs.
- `*_loop_assigns`: a written variable is missing from `loop assigns`.
- Smoke test `dead_loop` failed: the invariants contradict each other; the spec is vacuous.

Never use `admit`, axioms, ghost code, or `requires` on `main` to make a goal disappear.
