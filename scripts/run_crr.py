#!/usr/bin/env python3
"""Run testfiles/multi/crr.wyn end-to-end.

Compiles the Cox-Ross-Rubinstein 32-step American-option pricer,
dispatches it via `viz compute`, and prints a per-option table pairing
each input row (spot, strike, kind, tte, sigma) with the computed price
from the output buffer.

Usage:
    ./scripts/run_crr.py [--now 0.0] [--rfr 0.05]

Inputs come from testfiles/multi/crr_fixtures/*.json (8 sample options).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WYN_BIN = REPO_ROOT / "target" / "release" / "wyn"
VIZ_BIN = REPO_ROOT / "extra" / "viz" / "target" / "release" / "viz"
SHADER = REPO_ROOT / "testfiles" / "multi" / "crr.wyn"
FIXTURES = REPO_ROOT / "testfiles" / "multi" / "crr_fixtures"
SPV = Path("/tmp/crr.spv")


def ensure_built() -> None:
    if not WYN_BIN.exists():
        print("Building wyn (release)...")
        subprocess.run(["cargo", "build", "--release", "-q"], cwd=REPO_ROOT, check=True)
    if not VIZ_BIN.exists():
        print("Building viz (release)...")
        subprocess.run(
            ["cargo", "build", "--release", "-q"],
            cwd=REPO_ROOT / "extra" / "viz",
            check=True,
        )


def compile_shader() -> None:
    print(f"Compiling {SHADER.relative_to(REPO_ROOT)} -> {SPV}")
    subprocess.run(
        [str(WYN_BIN), "compile", str(SHADER), "-o", str(SPV)],
        cwd=REPO_ROOT,
        check=True,
    )


def dispatch(now: float, rfr: float) -> str:
    print(f"Dispatching with now={now}, rfr={rfr}\n")
    args = [
        str(VIZ_BIN), "compute", str(SPV),
        "--entry", "price_options",
        "--workgroups-x", "1",
        "--storage", f"0:0:8:f32:{FIXTURES}/spot.json",
        "--storage", f"0:1:8:f32:{FIXTURES}/strike.json",
        "--storage", f"0:2:8:i32:{FIXTURES}/opt_type.json",
        "--storage", f"0:3:8:f32:{FIXTURES}/tte.json",
        "--storage", f"0:4:8:f32:{FIXTURES}/sigma.json",
        "--storage", "0:5:8:f32",
        "--uniform", f"1:0:f32={now}",
        "--uniform", f"1:1:f32={rfr}",
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def parse_prices(viz_output: str) -> list[float]:
    """Extract the f32 values printed for binding 5 (the output buffer)."""
    m = re.search(
        r"Storage Buffer \(set 0, binding 5,[^)]*\)[^\[]*\[\s*0\]:\s*([0-9.\-eE \t]+)",
        viz_output,
    )
    if not m:
        sys.exit("could not find binding-5 output in viz stdout")
    return [float(x) for x in m.group(1).split()]


def load(name: str) -> list:
    with (FIXTURES / f"{name}.json").open() as f:
        return json.load(f)


def print_table(now: float, rfr: float, prices: list[float]) -> None:
    spot = load("spot")
    strike = load("strike")
    ot = load("opt_type")
    tte = load("tte")
    sigma = load("sigma")
    n = min(len(spot), len(prices))

    print(f"now={now}  rfr={rfr}")
    print(f"  {'spot':>7}  {'strike':>7}  {'kind':>4}  {'tte':>5}  {'sigma':>5}     price")
    print(f"  {'----':>7}  {'------':>7}  {'----':>4}  {'---':>5}  {'-----':>5}     -----")
    for i in range(n):
        kind = "call" if ot[i] == 1 else "put"
        print(
            f"  {spot[i]:7.2f}  {strike[i]:7.2f}  {kind:>4}  "
            f"{tte[i]:5.2f}  {sigma[i]:5.2f}    {prices[i]:7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now", type=float, default=0.0, help="years already elapsed (default 0.0)")
    parser.add_argument("--rfr", type=float, default=0.05, help="risk-free rate (default 0.05)")
    args = parser.parse_args()

    ensure_built()
    compile_shader()
    viz_output = dispatch(args.now, args.rfr)
    prices = parse_prices(viz_output)
    print_table(args.now, args.rfr, prices)


if __name__ == "__main__":
    main()
