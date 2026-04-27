#!/usr/bin/env python3
"""Run testfiles/multi/iv.wyn end-to-end.

Inverts the CRR pricer: given a market price per option, finds the σ
that the 32-step CRR model would have used to produce that price.
Inputs come from testfiles/multi/crr_fixtures/* (spot/strike/type/tte)
plus testfiles/multi/iv_fixtures/market.json. Defaults assume the
market prices were produced by ./scripts/run_crr.py at now=0, rfr=0.05;
the recovered σ values should round-trip back to the σ that crr used.

Usage:
    ./scripts/run_iv.py [--now 0.0] [--rfr 0.05]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WYN_BIN = REPO_ROOT / "target" / "release" / "wyn"
VIZ_BIN = REPO_ROOT / "extra" / "viz" / "target" / "release" / "viz"
SHADER = REPO_ROOT / "testfiles" / "multi" / "iv.wyn"
CRR_FIX = REPO_ROOT / "testfiles" / "multi" / "crr_fixtures"
IV_FIX = REPO_ROOT / "testfiles" / "multi" / "iv_fixtures"
SPV = Path("/tmp/iv.spv")


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
        "--entry", "implied_vols",
        "--workgroups-x", "1",
        "--storage", f"0:0:8:f32:{CRR_FIX}/spot.json",
        "--storage", f"0:1:8:f32:{CRR_FIX}/strike.json",
        "--storage", f"0:2:8:i32:{CRR_FIX}/opt_type.json",
        "--storage", f"0:3:8:f32:{CRR_FIX}/tte.json",
        "--storage", f"0:4:8:f32:{IV_FIX}/market.json",
        "--storage", "0:5:8:f32",
        "--uniform", f"1:0:f32={now}",
        "--uniform", f"1:1:f32={rfr}",
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def parse_ivs(viz_output: str) -> list[float]:
    m = re.search(
        r"Storage Buffer \(set 0, binding 5,[^)]*\)[^\[]*\[\s*0\]:\s*([0-9.\-eE \t]+)",
        viz_output,
    )
    if not m:
        sys.exit("could not find binding-5 output in viz stdout")
    return [float(x) for x in m.group(1).split()]


def load(fix_dir: Path, name: str) -> list:
    with (fix_dir / f"{name}.json").open() as f:
        return json.load(f)


def print_table(now: float, rfr: float, ivs: list[float]) -> None:
    spot = load(CRR_FIX, "spot")
    strike = load(CRR_FIX, "strike")
    ot = load(CRR_FIX, "opt_type")
    tte = load(CRR_FIX, "tte")
    market = load(IV_FIX, "market")
    # Original σ from sigma.json — present alongside recovered IV so
    # the round-trip is visible in one row per option.
    sigma_orig = load(CRR_FIX, "sigma")
    n = min(len(spot), len(ivs))

    print(f"now={now}  rfr={rfr}")
    print(
        f"  {'spot':>7}  {'strike':>7}  {'kind':>4}  {'tte':>5}  "
        f"{'market':>7}    {'sigma_in':>8}  {'iv_out':>7}"
    )
    print(
        f"  {'----':>7}  {'------':>7}  {'----':>4}  {'---':>5}  "
        f"{'------':>7}    {'--------':>8}  {'------':>7}"
    )
    for i in range(n):
        kind = "call" if ot[i] == 1 else "put"
        print(
            f"  {spot[i]:7.2f}  {strike[i]:7.2f}  {kind:>4}  "
            f"{tte[i]:5.2f}  {market[i]:7.3f}    {sigma_orig[i]:8.3f}  {ivs[i]:7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now", type=float, default=0.0)
    parser.add_argument("--rfr", type=float, default=0.05)
    args = parser.parse_args()

    ensure_built()
    compile_shader()
    viz_output = dispatch(args.now, args.rfr)
    ivs = parse_ivs(viz_output)
    print_table(args.now, args.rfr, ivs)


if __name__ == "__main__":
    main()
