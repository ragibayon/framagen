# Small shared Rich logger; integrations can disable it per call.
#
# Console output is unchanged from the original ContractFixLogger. In addition,
# every message can be persisted: attach log files (one per job, one per run)
# with `logger.to_file(path)`; each line is timestamped, leveled and tagged
# with the job that emitted it.

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterator, TextIO

from rich.console import Console
from rich.markup import escape

_STYLES = {
    "info": ("cyan", "ℹ"),
    "success": ("green", "✓"),
    "warning": ("yellow", "⚠"),
    "error": ("red", "✗"),
}


class FramaGenLogger:
    def __init__(self) -> None:
        self.console = Console(stderr=True)
        self.console_enabled = True
        self._sinks: dict[Path, tuple[TextIO, str | None]] = {}
        self._lock = threading.Lock()
        self._local = threading.local()

    # -- levels -----------------------------------------------------------
    def info(self, message: str, *, console: bool = True) -> None:
        self._emit("info", message, console)

    def success(self, message: str, *, console: bool = True) -> None:
        self._emit("success", message, console)

    def warning(self, message: str, *, console: bool = True) -> None:
        self._emit("warning", message, console)

    def error(self, message: str, *, console: bool = True) -> None:
        self._emit("error", message, console)

    @contextmanager
    def timed(self, label: str, *, console: bool = True) -> Iterator[dict[str, float]]:
        """Log start/finish of a block; the yielded dict receives ``seconds``."""
        timing: dict[str, float] = {}
        start = perf_counter()
        self.info(f"{label} started", console=console)
        try:
            yield timing
        except BaseException:
            timing["seconds"] = perf_counter() - start
            self.error(f"{label} failed after {timing['seconds']:.2f}s", console=console)
            raise
        else:
            timing["seconds"] = perf_counter() - start
            self.success(f"{label} finished in {timing['seconds']:.2f}s", console=console)

    # -- persistence ------------------------------------------------------
    @contextmanager
    def to_file(self, path: str | Path, *, job: str | None = None) -> Iterator[Path]:
        """Also write messages to ``path`` while the block runs.

        With ``job``, only messages emitted under :meth:`job_context` with the
        same name are written, so parallel jobs keep separate logs.
        """
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open("a", encoding="utf-8")
        with self._lock:
            self._sinks[target] = (handle, job)
        try:
            yield target
        finally:
            with self._lock:
                self._sinks.pop(target, None)
            handle.close()

    @contextmanager
    def job_context(self, job: str) -> Iterator[None]:
        previous = getattr(self._local, "job", None)
        self._local.job = job
        try:
            yield
        finally:
            self._local.job = previous

    def _emit(self, level: str, message: str, console: bool) -> None:
        job = getattr(self._local, "job", None)
        color, icon = _STYLES[level]
        if console and self.console_enabled:
            prefix = f"[dim]{escape(job)}[/dim] " if job else ""
            self.console.print(f"{prefix}[{color}]{icon} {escape(message)}[/{color}]")
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        line = f"{stamp} {level.upper():7s} {job or '-'} {message}\n"
        with self._lock:
            for handle, only_job in self._sinks.values():
                if only_job is None or only_job == job:
                    handle.write(line)
                    handle.flush()


logger = FramaGenLogger()
