"""Integration tests against a real Frama-C (skipped unless configured).

    FRAMAGEN_FRAMAC_BACKEND=apptainer FRAMAGEN_FRAMAC_IMAGE=~/containers/frama-c_32.1 pytest
"""

import os
import shutil

import pytest

from framagen.benchmarks.svcomp import SVCOMP_VERIFY
from framagen.benchmarks.sv2acsl import convert
from framagen.utils.frama_c import FramaCRunner, VerifyOptions, verify_candidate

from test_offline import SV_TASK

backend = os.environ.get("FRAMAGEN_FRAMAC_BACKEND")
pytestmark = pytest.mark.skipif(
    not backend and not shutil.which("docker"), reason="no Frama-C backend configured"
)


def _verify(tmp_path, text, options):
    path = tmp_path / "candidate_00.c"
    path.write_text(text)
    return verify_candidate(path, runner=FramaCRunner.from_env(), options=options)


def test_svcomp_harness_needs_invariant(tmp_path):
    task = convert(SV_TASK).text
    options = SVCOMP_VERIFY.to_options(prover_timeout_seconds=5)
    assert _verify(tmp_path, task, options).verdict == "INCONCLUSIVE"
    good = task.replace("  while (x < 10)", "  /*@ loop invariant 1 <= x <= 10; loop assigns x; */\n  while (x < 10)")
    assert _verify(tmp_path, good, options).verdict == "VERIFIED"


def test_smoke_tests_flag_vacuous_spec(tmp_path):
    src = "/*@ requires x > 0 && x < 0; assigns \\nothing; ensures \\result == 42; */\nint f(int x) { return 1; }\n"
    result = _verify(tmp_path, src, VerifyOptions(prover_timeout_seconds=5, check_termination=False))
    assert result.status == "vacuous" and result.verdict == "FAILED"


def test_parse_error(tmp_path):
    result = _verify(tmp_path, "/*@ ensures \\result == ; */\nint f(void) { return 0; }\n", VerifyOptions())
    assert result.status == "parse_error"
