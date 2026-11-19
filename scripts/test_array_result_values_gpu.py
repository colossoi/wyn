"""Check handwritten gather order and published tuple values on both backends."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile

from test_dispatch_loop_gpu import ROOT, VIZ, WYN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-validate", action="store_true",
                        help="disable viewer GPU layers while keeping compiler SPIR-V validation")
    options = parser.parse_args()
    gathers = {
        "gather_literal_rows": [10*r + 2*c + k for k in range(2) for c in range(2) for r in range(3)],
        "gather_ranked_map": [100*r + 10*c + k for k in range(4) for c in range(3) for r in range(2)],
        "gather_repeated_cells": [0, 11, 20, 31, 40, 51, 60, 71] * 2,
    }
    tuple_entry = "projected_tuple_7_kept"
    inputs = [3, -2, 1, 0, 7]
    expected_tuple = [[x, [x > 1, x + 1]] for x in inputs if x > 0]
    fixtures = [
        ("library/handwritten_gathers.wyn", [], gathers),
        ("cases/tuple_array_operations_projected_tuple_7.wyn", ["--graphics"], {tuple_entry: expected_tuple}),
    ]
    with tempfile.TemporaryDirectory(prefix="wyn-array-values-") as temporary:
        directory = Path(temporary)
        data = directory / "input.bin"
        data.write_bytes(struct.pack(f"<{len(inputs)}i", *inputs))
        for target, suffix in (("spirv", "spv"), ("wgsl", "wgsl")):
            for fixture, flags, cases in fixtures:
                shader = directory / f"array-values.{suffix}"
                subprocess.run(
                    [WYN, "build", str(ROOT / "tests/compiler" / fixture), *flags,
                     "-t", target, "-o", str(shader)], check=True, timeout=60,
                )
                if target == "spirv":
                    subprocess.run(["spirv-val", "--target-env", "vulkan1.3", str(shader)],
                                   check=True, timeout=60)
                for entry, expected in cases.items():
                    output = directory / "result.json"
                    output.unlink(missing_ok=True)
                    command = [VIZ, "pipeline", str(shader), "--headless", "--no-config",
                               "--entry", entry, "--output", f"{entry}:{output}"]
                    if entry == tuple_entry:
                        command += ["--input", f"projected_tuple_7_xs:{data}"]
                    if options.no_validate:
                        command.append("--no-validate")
                    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    diagnostic = run.stdout + run.stderr
                    if run.returncode or "Validation Error" in diagnostic:
                        raise RuntimeError(f"{target} {entry}:\n{diagnostic}")
                    result = json.loads(output.read_text(encoding="utf-8"))
                    actual = result["backing_buffer"]
                    assert actual == expected, (target, entry, actual, expected)
                    print(f"{target} {entry}: values match", flush=True)


if __name__ == "__main__":
    main()
