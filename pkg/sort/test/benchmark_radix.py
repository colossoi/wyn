"""Build, verify, benchmark, and plot 2-/4-bit CPU/GPU radix sorting.

Run from any directory: python3 pkg/sort/test/benchmark_radix.py
Requires rustc/Cargo, the built Wyn compiler, Matplotlib, and a Vulkan GPU.
Results, raw samples, and the PNG/SVG chart go to target/radix-bench/results.
Run without competing compilation or GPU workloads.
"""
import csv
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "target/radix-bench"
RESULTS = OUT / "results"
SIZES = [(64 * 8**power) // 4 for power in range(7)]
RUNS = 5
THREADS = 8


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=ROOT, check=True, **kwargs)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    command(["rustc", "--edition=2021", "-O", "-C", "target-cpu=native",
             "pkg/sort/radix_cpu.rs", "-o", OUT / "radix-cpu"])
    command(["cargo", "build", "--release", "--manifest-path", "pkg/sort/cpu-bench/Cargo.toml", "--offline"],
            env={**os.environ, "RUSTFLAGS": "-C target-cpu=native"})
    command([ROOT / "target/debug/wyn", "build", "-O", "--target", "wgsl", "--target-double", "rust-wgpu",
             "--max-warnings", "0", "pkg/sort/test/radix_bench.wyn", "-o", OUT / "radix.wgsl"])
    shutil.copyfile(Path(__file__).with_name("radix_bench.rs"), OUT / "main.rs")
    (OUT / "Cargo.toml").write_text('''[package]
name = "wyn-radix-benchmark"
version = "0.1.0"
edition = "2021"
[workspace]
[[bin]]
name = "radix-benchmark"
path = "main.rs"
[dependencies]
wgpu = "27"
pollster = "0.3"
''')
    command(["cargo", "build", "--release", "--manifest-path", OUT / "Cargo.toml",
             "--target-dir", ROOT / "target/gpu-check", "--offline"])
    samples = []
    for method, binary in [("CPU single thread", OUT / "radix-cpu"),
                           (f"CPU Rayon ({THREADS} threads)", ROOT / "pkg/sort/cpu-bench/target/release/radix-rayon")]:
        for n in SIZES:
            for bits in [2, 4]:
                result = command([binary, n, RUNS, bits], capture_output=True, text=True,
                                 env={**os.environ, "RAYON_NUM_THREADS": str(THREADS)})
                (RESULTS / f"{binary.name}-{bits}-{n}.txt").write_text(result.stdout)
                for run, milliseconds in re.findall(r"run=(\d+) sort_ms=([\d.]+)", result.stdout):
                    if int(run):
                        samples.append(dict(method=method, bits=bits, n=n, run=int(run), ms=float(milliseconds)))
    gpu = command([ROOT / "target/gpu-check/release/radix-benchmark"], capture_output=True, text=True, timeout=1800)
    (RESULTS / "gpu.csv").write_text(gpu.stdout)
    (RESULTS / "gpu-device.txt").write_text(gpu.stderr)
    for row in csv.DictReader(gpu.stdout.splitlines()):
        for method, field in [("GPU resident", "resident_ms"), ("GPU with transfers", "end_to_end_ms")]:
            samples.append(dict(method=method, bits=int(row["bits"]), n=int(row["n"]),
                                run=int(row["run"]), ms=float(row[field])))
    with (RESULTS / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "bits", "n", "run", "ms"])
        writer.writeheader()
        writer.writerows(samples)
    medians = []
    for method in dict.fromkeys(s["method"] for s in samples):
        for bits in [2, 4]:
            for n in SIZES:
                times = [s["ms"] for s in samples if (s["method"], s["bits"], s["n"]) == (method, bits, n)]
                if not times:
                    continue
                medians.append(dict(method=method, bits=bits, n=n, ms=statistics.median(times),
                                    min_ms=min(times), max_ms=max(times)))
    (RESULTS / "medians.json").write_text(json.dumps(medians, indent=2) + "\n")
    metadata = dict(cpu=subprocess.check_output(["lscpu"], text=True), gpu=gpu.stderr,
                    runs=RUNS, warmups=1, rayon_threads=THREADS, gpu_memory_budget_bytes=5*1024**3,
                    input="signed i32 xorshift32; seed 0x12345678; ascending sort; identical input each run",
                    timing="CPU includes input copy and scratch allocation. GPU resident includes host submission, carried-input copy, and completion; excludes upload/readback. GPU with transfers includes upload, readback and CPU vector copy. Pipeline setup and output verification excluded.")
    (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(medians)

def plot(medians):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wyn-radix-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    colors = {"CPU single thread": "#2878b5", f"CPU Rayon ({THREADS} threads)": "#27946b", "GPU resident": "#d65a31", "GPU with transfers": "#d65a31"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, gpu_method, title in zip(axes, ["GPU resident", "GPU with transfers"],
                                    ["GPU input already resident", "GPU includes upload and readback"]):
        for method in ["CPU single thread", f"CPU Rayon ({THREADS} threads)", gpu_method]:
            for bits in [2, 4]:
                rows = [r for r in medians if r["method"] == method and r["bits"] == bits]
                ax.plot([4*r["n"] for r in rows], [r["ms"] for r in rows],
                        label=f"{method} · {bits}-bit", color=colors[method],
                        linestyle="--" if bits == 2 else "-", marker="o" if bits == 2 else "s", markersize=4)
        ax.set_xscale("log")
        ax.set_xticks([4*n for n in SIZES])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1024**3:g} GiB" if x >= 1024**3 else (f"{x/1024**2:g} MiB" if x >= 1024**2 else (f"{x/1024:g} KiB" if x >= 1024 else f"{x:g} B"))))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Input size (signed 32-bit integers)")
        ax.grid(True, which="both", alpha=.18)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Median wall time (ms, lower is better)")
    fig.suptitle("Radix sort: two-bit vs four-bit passes", fontsize=17, fontweight="bold")
    fig.text(.5, .025, "Ryzen 7 2700X · GTX 1660 Ti / Vulkan · 1 warm-up + 5 verified runs · pipeline setup excluded\nCPU timings include input copy + scratch allocation; GPU timings include submission + completion. Both axes logarithmic. Inputs: 64 B–16 MiB, increasing eightfold.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .94))
    for ext in ["png", "svg"]:
        fig.savefig(RESULTS / f"radix-comparison.{ext}", dpi=180)
    print(RESULTS / "radix-comparison.png", flush=True)


if __name__ == "__main__":
    main()
