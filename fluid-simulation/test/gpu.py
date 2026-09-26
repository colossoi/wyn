"""Build with the checkout compiler and read typed results from the WHL runner."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WYN = Path(os.environ.get('WYN', ROOT / 'target/release/wyn'))
VIZ = Path(os.environ.get('VIZ', ROOT / 'extra/viz/target/release/viz'))


def flatten(value):
    if isinstance(value, dict):
        return flatten(value['backing_buffer'])
    if isinstance(value, list):
        return [item for part in value for item in flatten(part)]
    return [value]


def run(source, directory, target='spirv'):
    shader = directory / (source.stem + ('.spv' if target == 'spirv' else '.wgsl'))
    subprocess.run([str(WYN), 'build', str(source), '-t', target, '-o', str(shader)], check=True)
    result = subprocess.run([str(VIZ), 'pipeline', str(shader)],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        result.check_returncode()
    outputs = json.loads(result.stdout)
    # Tuple results use result_0, result_1, ...; single results use the entry name.
    names = sorted(outputs, key=lambda name: int(name[7:]) if name.startswith('result_') else 0)
    return [flatten(outputs[name]) for name in names]
