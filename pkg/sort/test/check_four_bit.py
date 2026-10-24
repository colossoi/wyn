"""Run the four-bit Wyn regressions on both GPU backends and compare exact results.

From the repository root: python3 pkg/sort/test/check_four_bit.py
Build target/debug/wyn and extra/viz/target/debug/viz first, or set WYN and VIZ.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
WYN = Path(os.environ.get("WYN", ROOT / "target/debug/wyn"))
VIZ = Path(os.environ.get("VIZ", ROOT / "extra/viz/target/debug/viz"))


def flatten(value):
    if isinstance(value, dict):
        return flatten(value["backing_buffer"])
    if isinstance(value, list):
        return [item for child in value for item in flatten(child)]
    return [value]


def check(source, expected, directory, target, needle=None):
    shader = directory / (source.stem + (".spv" if target == "spirv" else ".wgsl"))
    subprocess.run([str(WYN), "build", str(source), "-t", target, "-o", str(shader)], check=True)
    command = [str(VIZ), "pipeline", str(shader)]
    if needle is not None:
        input_path = directory / "needle.bin"
        input_path.write_bytes(struct.pack("<i", needle))
        command.extend(["--input", f"needle:{input_path}"])
    result = subprocess.run(command,
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{source.name} ({target}): {result.stderr}")
    outputs = json.loads(result.stdout)
    actual = [flatten(outputs[key]) for key in sorted(outputs)]
    assert actual == expected, f"{source.name} ({target}): {actual} != {expected}"
    print(f"{target}: {source.name} passed", flush=True)


def main():
    pairs = sorted([((i % 97) * 2654435761 & 0xffffffff, i) for i in range(513)],
                   key=lambda pair: pair[0])
    cases = {
        "radix_four_bit.wyn": [[0, 3, 1, 1, 1, 2, 0xffffffff, 0]],
        "radix_four_bit_edges.wyn": [[0], [0xffffffff, 42]],
        "radix_four_bit_large.wyn": [[item for pair in pairs for item in pair]],
    }
    with tempfile.TemporaryDirectory() as directory:
        for target in ("spirv", "wgsl"):
            for name, expected in cases.items():
                check(Path(__file__).with_name(name), expected, Path(directory), target)
            for needle in [0, 16]:
                expected = [] if needle == 0 else [v for i in range(1, 19) for v in (i, 18-i)]
                check(Path(__file__).with_name("radix_bounded_loop.wyn"), [expected],
                      Path(directory), target, needle)


if __name__ == "__main__":
    main()
