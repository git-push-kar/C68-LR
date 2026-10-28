"""Deprecated legacy entry point for benchmark preparation.

The previous implementation called ``DatasetNormalizer.normalize_logicbench`` /
``normalize_multi_logieval`` / ``normalize_logicnli`` on whole release files
(containers such as ``{"samples": [...]}`` or ``{"0": {...}, ...}``) instead of
on the individual examples inside them.  That produced one example per file,
with default labels (all ``True`` or all ``Uncertain``), duplicate ids and empty
questions.  Those files are invalid and must not be reused.

The canonical pipeline is ``scripts/bench/build_real_benchmarks.py``, which
transcribes the official LogicBench, Multi-LogiEval and LogicNLI releases from
``data/benchmarks_real/<name>/raw`` and records the source commit in each
README.  This entry point now delegates to it rather than writing placeholder
or degenerate data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.bench.build_real_benchmarks import BENCH_ROOT, main as build_main  # noqa: E402


def main() -> None:
    required = [
        BENCH_ROOT / "LogicBench" / "raw",
        BENCH_ROOT / "Multi-LogiEval" / "raw",
        BENCH_ROOT / "LogicNLI" / "raw",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Official benchmark clones are missing. This script no longer writes "
            "placeholder data. Clone the upstream repositories into:\n  "
            + "\n  ".join(missing)
            + "\nThen re-run this script."
        )
    print(
        "[prepare_benchmarks] deprecated wrapper: delegating to "
        "scripts/bench/build_real_benchmarks.py"
    )
    build_main()


if __name__ == "__main__":
    main()
