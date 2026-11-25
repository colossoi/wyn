"""Check handwritten gathers, tiled outputs, and published tuple values on both backends."""

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
    tiles = {
        7: [0, 1, 2, 3, 1, 2, 3, 4, 4, 5, 6, -1, 5, 6, 7, -1],
        8: [0, 1, 2, 3, 1, 2, 3, 4, 4, 5, 6, 7, 5, 6, 7, 8],
    }
    tiled = {}
    for count, values in tiles.items():
        metadata = list(range(2, count + 2))
        rows = [[i, i + 1, i + 2] for i in range(count)]
        for kind, other in (("mapped", metadata), ("reduced", sum(metadata))):
            entry = f"tiled_{kind}_{count}"
            results = {"result_0": values, "result_1": other}
            tiled[entry] = results
            tiled[f"{entry}_kept"] = results | {"result_2": rows}
    scratch_entry = "scratch_scatter_after_update"
    tiled[scratch_entry] = {
        "result_0": [2, 4, 3, 1, 12, 14, 13, 11],
        "result_1": [7, 27],
    }
    fixtures = [
        ("library/handwritten_gathers.wyn", [], {name: {name: values} for name, values in gathers.items()}),
        ("cases/tuple_array_operations_projected_tuple_7.wyn",
         ["--graphics"], {tuple_entry: {tuple_entry: expected_tuple}}),
        ("library/tiled_outputs.wyn", [], tiled),
    ]
    with tempfile.TemporaryDirectory(prefix="wyn-array-values-") as temporary:
        directory = Path(temporary)
        data = directory / "input.bin"
        data.write_bytes(struct.pack(f"<{len(inputs)}i", *inputs))
        rows = directory / "rows.bin"
        rows.write_bytes(struct.pack("<8i", 1, 2, 3, 4, 11, 12, 13, 14))
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
                for entry, expected_results in cases.items():
                    command = [VIZ, "pipeline", str(shader), "--headless", "--no-config",
                               "--entry", entry]
                    for name in expected_results:
                        output = directory / f"{name}.json"
                        output.unlink(missing_ok=True)
                        command += ["--output", f"{name}:{output}"]
                    if entry == tuple_entry:
                        command += ["--input", f"projected_tuple_7_xs:{data}"]
                    if entry == scratch_entry:
                        command += ["--input", f"xs:{rows}"]
                    if options.no_validate:
                        command.append("--no-validate")
                    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    diagnostic = run.stdout + run.stderr
                    if run.returncode or "Validation Error" in diagnostic:
                        raise RuntimeError(f"{target} {entry}:\n{diagnostic}")
                    for name, expected in expected_results.items():
                        result = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
                        actual = result["backing_buffer"] if isinstance(result, dict) else result
                        assert actual == expected, (target, entry, name, actual, expected)
                    print(f"{target} {entry}: values match", flush=True)


if __name__ == "__main__":
    main()
