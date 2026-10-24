"""Tests that need neither Frama-C nor an LLM."""

from framagen.agents.guards import check_candidate, normalize_candidate
from framagen.benchmarks.frama_c_problems import strip_all_acsl
from framagen.benchmarks.manifest import classify
from framagen.benchmarks.sv2acsl import HARNESS_BEGIN, HARNESS_END, convert, harness_block

SV_TASK = """extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void reach_error() { __assert_fail("0", "t.c", 3, "reach_error"); }
extern int __VERIFIER_nondet_int(void);
void __VERIFIER_assert(int cond) {
  if (!(cond)) {
  ERROR: {reach_error();abort();}
  }
  return;
}
int main() {
  int x = __VERIFIER_nondet_int();
  if (!(x == 1)) return 0;
  while (x < 10) {
    x++;
  }
  __VERIFIER_assert(x == 10);
  return 0;
}
"""


def test_convert_replaces_harness():
    conv = convert(SV_TASK)
    text = conv.text
    assert text.startswith(HARNESS_BEGIN) and HARNESS_END in text
    block = harness_block(text)
    assert "requires cond != 0;" in block
    # reach_error/abort were only used inside __VERIFIER_assert's body.
    assert "reach_error" not in block and "abort" not in block
    assert "extern int __VERIFIER_nondet_int(void);" in block
    body = text[text.index(HARNESS_END):]
    assert "__assert_fail" not in body  # unused glibc prototype dropped
    assert "void __VERIFIER_assert(int cond) {" not in text
    assert "while (x < 10)" in body


def test_convert_direct_reach_error():
    src = (
        'extern void __assert_fail(const char *, const char *, unsigned int, const char *);\n'
        'void reach_error() { __assert_fail("0", "t.c", 2, "reach_error"); }\n'
        "int main() { int a; for (a = 0; a < 6; ++a) {} if (!(a == 6)) { reach_error(); } return 1; }\n"
    )
    block = harness_block(convert(src).text)
    assert "requires \\false;" in block and "void reach_error(void);" in block


def test_guard_accepts_added_loop_annotations():
    task = convert(SV_TASK).text
    cand = task.replace("  while (x < 10)", "  /*@ loop invariant 1 <= x <= 10; loop assigns x; */\n  while (x < 10)")
    assert check_candidate(task, cand, allow_axioms=False).ok


def test_guard_rejects_cheats():
    task = convert(SV_TASK).text
    cheats = {
        "harness": task.replace("requires cond != 0;", "requires \\true;"),
        "code": task.replace("x == 10", "x >= 0"),
        "admit": task.replace("  __VERIFIER_assert", "  /*@ admit x == 10; */\n  __VERIFIER_assert"),
        "ghost": task.replace("  return 0;\n}", "  /*@ ghost int g = 0; */\n  return 0;\n}"),
        "axiom": task.replace("int main", "/*@ axiomatic A { axiom a: \\false; } */\nint main"),
        "braces": task.replace("{\n    x++;\n  }", "x++;"),
    }
    for name, cand in cheats.items():
        assert not check_candidate(task, cand, allow_axioms=False).ok, name


def test_axioms_allowed_for_full_contracts():
    src = "int f(int n) { return n; }\n"
    cand = "/*@ axiomatic A { logic integer g(integer n); axiom g0: g(0) == 0; } */\n" + src
    report = check_candidate(src, cand, allow_axioms=True)
    assert report.ok and report.notes


def test_normalize_strips_markdown_fence():
    text, notes = normalize_candidate("```c\nint x;\n```")
    assert text == "int x;\n" and notes


def test_strip_all_acsl():
    src = "/*@ requires x > 0; */\nint f(int x) {\n  //@ assert x > 0;\n  return x;\n}\n"
    out = strip_all_acsl(src)
    assert "@" not in out and "return x;" in out


def test_classify_control_flow():
    assert classify("int f(int a){ return a+1; }")[0] == "sequential"
    assert classify("int f(int a){ if (a) return 1; return 0; }")[0] == "branched"
    assert classify("void f(int *p){ *p = 0; }")[0] == "pointer"
    assert classify("void f(){ int i=0; while(i<3) i++; }")[0] == "single-path-loop"
    assert classify("void f(){ int i=0; while(i<3){ if(i) i++; else i+=2; } }")[0] == "multi-path-loop"
    assert classify("void f(){ for(int i=0;i<3;i++) for(int j=0;j<3;j++){} }")[0] == "nested-loop"
    # A __VERIFIER_assert call inside a loop is not a branch.
    assert classify("void f(){ int i=0; while(i<3){ __VERIFIER_assert(i>=0); i++; } }")[0] == "single-path-loop"
    # do-while
    assert classify("void f(){ int i=0; do { i++; } while(i<3); }") == ("single-path-loop", 1, 1)
