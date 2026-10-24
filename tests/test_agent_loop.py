"""Agent loop with a scripted chat model (needs a Frama-C backend, no LLM)."""

import json
import os
import shutil
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from framagen.agents.agent import run_job
from framagen.benchmarks.svcomp import SVCOMP_VERIFY
from framagen.benchmarks.sv2acsl import convert

from test_offline import SV_TASK

pytestmark = pytest.mark.skipif(
    not os.environ.get("FRAMAGEN_FRAMAC_BACKEND") and not shutil.which("docker"),
    reason="no Frama-C backend configured",
)


class Scripted(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _final(code: str, n: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "CandidateSpecification", "id": f"c{n}",
                     "args": {"summary": "s", "annotated_c": code}}],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


def test_refinement_nudge_and_property(tmp_path):
    task = tmp_path / "task.c"
    task.write_text(convert(SV_TASK).text)
    text = task.read_text()
    weak = text.replace("  while (x < 10)", "  /*@ loop assigns x; */\n  while (x < 10)")
    good = text.replace("  while (x < 10)", "  /*@ loop invariant 1 <= x <= 10; loop assigns x; */\n  while (x < 10)")
    model = Scripted(responses=[
        _final(text.replace("x == 10", "x >= 0"), 0),                     # attempt 0: edits code -> rejected
        AIMessage(content="I think the invariant is x <= 10."),           # attempt 1: forgets the tool...
        _final(weak, 1),                                                  # ...nudged -> weak -> unproved
        AIMessage(content="Here it is:\n```c\n" + good.rstrip("\n") + "\n```"),  # attempt 2: plain text
    ])
    result = run_job(
        source_file=task, output_root=tmp_path / "out", model=model, job_id="t",
        target_property="unreach-call", expected_verdict=True, goal="loop-annotations",
        max_refinements=3, max_nudges=1,
        verify_options=SVCOMP_VERIFY.to_options(prover_timeout_seconds=5),
    )
    attempts = result["attempts"]
    assert [a["status"] for a in attempts] == ["rejected", "partially_proved", "proved"]
    assert attempts[1]["nudges"] == 1 and attempts[2]["generation"] == "recovered"
    assert [a["property_status"] for a in attempts] == ["not-checked", "unproved", "verified"]
    assert result["property"]["matches_expected"] is True
    out = Path(result["output_dir"])
    usage = [json.loads(line) for line in (out / "llm_usage.jsonl").read_text().splitlines()]
    assert len(usage) == 3 and all(u["latency_seconds"] is not None for u in usage)
    assert usage[0]["token_usage"]["input_tokens"] == 100
    job = json.loads((out / "job.json").read_text())
    assert job["usage"]["input_tokens"] >= 200 and job["usage"]["llm_seconds"] >= 0
    assert "property unreach-call" in (out / "job.log").read_text()
    assert (out / "transcript_01.json").is_file()
