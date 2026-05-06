#!/usr/bin/env python3
"""Run one TeamPlasma benchmark solve in a fresh Python process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir
from submissions.team_plasma.placer import TeamPlasmaPlacer


def main() -> int:
    parser = argparse.ArgumentParser(description="TeamPlasma isolated benchmark worker")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bench_root = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / args.benchmark
    benchmark, plc = load_benchmark_from_dir(str(bench_root))

    placer = TeamPlasmaPlacer(config_path=args.config, seed=args.seed)
    if hasattr(placer, "_plc_cache"):
        placer._plc_cache[benchmark.name] = plc

    placement = placer.place(benchmark).detach().cpu().float()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(placement, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
