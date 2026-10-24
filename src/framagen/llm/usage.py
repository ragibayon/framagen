# One record per agent invocation, appended to a JSONL file.

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMCallRecord:
    stage: str
    provider: str
    model: str
    latency_seconds: float | None
    token_usage: dict[str, Any] | None
    status: str  # accepted | no_structured_response | rejected | error
    error: str | None = None
    call_count: int = 0
    attempt: int | None = None
    job_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JSONLUsageRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, record: LLMCallRecord) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
