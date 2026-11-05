"""Check projected/captured scalar reads and snapshots on SPIR-V and WGSL.

Requires release wyn/viz builds and spirv-val. WYN and VIZ override executables.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
WYN = os.environ.get("WYN", str(ROOT / f"target/release/wyn{EXE}"))
VIZ = os.environ.get("VIZ", str(ROOT / f"extra/viz/target/release/viz{EXE}"))


def flatten(value):
    if isinstance(value, dict):
        return flatten(value["backing_buffer"])
    if isinstance(value, list):
        return [item for child in value for item in flatten(child)]
    return [value]


def write_words(path, words):
    path.write_bytes(b"".join(struct.pack("<i", word) for word in words))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-validate", action="store_true",
                        help="disable viewer GPU layers; still validate compiler SPIR-V with spirv-val")
    options = parser.parse_args()
    if options.no_validate:
        print("Viewer GPU validation disabled; compiler SPIR-V validation remains enabled.", flush=True)
    validator = shutil.which("spirv-val")
    if validator is None:
        raise RuntimeError("spirv-val is required")
    cases = 0
    with tempfile.TemporaryDirectory(prefix="wyn_view_reads_") as temporary:
        directory = Path(temporary)
        rows, indices, output = (directory / name for name in ("rows.bin", "indices.bin", "output.json"))
        for target in ("spirv", "wgsl"):
            for optimized in (False, True):
                shader = directory / ("reads.spv" if target == "spirv" else "reads.wgsl")
                command = [WYN, "build", str(ROOT / "tests/compiler/cases/projected_index_reads.wyn"),
                           "--max-warnings", "0", "-t", target, "-o", str(shader)]
                subprocess.run(command + (["-O"] if optimized else []), check=True, timeout=60)
                if target == "spirv":
                    subprocess.run([validator, "--target-env", "vulkan1.3", str(shader)], check=True)
                for entry in ("tuple_read", "captured_read", "snapshot_read"):
                    words = [value for row in range(128)
                             for value in ([row * 2048 + i - 17 for i in range(64)]
                                           + ([] if entry == "snapshot_read" else [123]))]
                    write_words(rows, words)
                    lengths = (0, 4, 60, 64, 68, 100000) if entry == "captured_read" else (0, 1, 63, 64, 65, 100000)
                    for n in lengths:
                        pairs = [((i * 17) % 128, (i * 31) % 64) for i in range(n)]
                        write_words(indices, (word for pair in pairs for word in pair))
                        expected = [row * 2048 + col - 17 for row, col in pairs]
                        if entry == "snapshot_read":
                            expected = [word for value in expected for word in (value, value + 1)]
                        output.unlink(missing_ok=True)
                        result = subprocess.run(
                            [VIZ, "pipeline", str(shader), "--headless", "--no-config", "--entry", entry,
                             "--input", f"rows:{rows}", "--input", f"indices:{indices}",
                             "--output", f"{entry}:{output}"] + (["--no-validate"] if options.no_validate else []),
                            capture_output=True, text=True, timeout=60)
                        if result.returncode or "Validation Error" in result.stdout + result.stderr:
                            raise RuntimeError(f"{target}, optimized={optimized}, {entry}, n={n}:\n"
                                               + result.stdout + result.stderr)
                        actual = flatten(json.loads(output.read_text()))
                        assert actual == expected, (target, optimized, entry, n)
                        cases += 1
                print(f"{target}, optimized={optimized}: projected reads and snapshots passed", flush=True)
    print(f"{cases} GPU cases passed")


if __name__ == "__main__":
    main()
