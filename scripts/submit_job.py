"""Run FramGen on a single C file or benchmark task.

Thin wrapper around `framagen run`, kept for the original prototype workflow:

    python scripts/submit_job.py data/cstrcat_unsafe.c --model ollama:qwen3:8b
    python scripts/submit_job.py --task svcomp/loop-zilu/benchmark04_conjunctive --model ollama:qwen3:8b
"""

import sys

from framagen.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
