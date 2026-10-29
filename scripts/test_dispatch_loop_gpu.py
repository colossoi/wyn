"""Check simultaneous product state, zero trips and nested GPU dispatch loops."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXE = '.exe' if os.name == 'nt' else ''
WYN = os.environ.get('WYN', str(ROOT / f'target/release/wyn{EXE}'))
VIZ = os.environ.get('VIZ', str(ROOT / f'extra/viz/target/release/viz{EXE}'))


def flatten(value):
    if isinstance(value, dict):
        return flatten(value['backing_buffer'])
    if isinstance(value, list):
        return [word for item in value for word in flatten(item)]
    return [value]


def reference(entry, limit, values=None):
    entry = entry.removesuffix('_runtime')
    a = [1, 2, 3, 4] if values is None else values
    b, total, changed = [x*10 for x in a], 0, True
    for i in range(limit):
        if entry == 'conditional':
            if not changed:
                break
            a = [sum(a[:j+1]) for j in range(len(a))]
            changed = any(x < 100 for x in a)
            total += 1
        elif entry == 'counted':
            a, b, total = [x+i for x in b], [x+1 for x in a], total+sum(a)
        elif entry == 'nested':
            a, total = [x+3 for x in a], total+1
        else:
            a, total = [x+1 for x in a], total+1
    return dict(conditional=[a, [total], [changed]], counted=[a, b, [total]],
                nested=[a, [total]], runtime=[a, [total]])[entry]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='wyn-dispatch-loops-') as temp:
        directory = Path(temp)
        for target in ['wgsl', 'spirv']:
            shader = directory / ('loops.wgsl' if target == 'wgsl' else 'loops.spv')
            subprocess.run([WYN, 'build', str(ROOT/'tests/compiler/library/dispatch_loop_products.wyn'),
                            '-t', target, '-o', str(shader)], check=True, timeout=120)
            if target == 'spirv':
                subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(shader)], check=True)
            for entry in ['conditional', 'counted', 'nested']:
                for limit in ([0, 1, 3, 128] if entry == 'nested' else [0, 1, 3]):
                    expected = reference(entry, limit)
                    outputs = [directory/f'{entry}-{i}.json' for i in range(len(expected))]
                    command = [VIZ, 'pipeline', str(shader), '--entry', entry,
                               '--headless', '--no-config', '--push-constant', f'limit:i32={limit}']
                    for i, path in enumerate(outputs):
                        path.unlink(missing_ok=True)
                        command += ['--output', f'result_{i}:{path}']
                    if args.no_validate:
                        command.append('--no-validate')
                    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    if run.returncode or 'Validation Error' in run.stdout + run.stderr:
                        raise RuntimeError(run.stdout + run.stderr)
                    actual = [flatten(json.loads(path.read_text())) for path in outputs]
                    assert actual == expected, (target, entry, limit, expected, actual)
            for values in [[], [1], list(range(65)), [100, 200, 300, 400]]:
                length = len(values)
                input_path = directory / 'input.bin'
                input_path.write_bytes(struct.pack(f'<{length}i', *values))
                for entry in ['runtime', 'conditional_runtime', 'counted_runtime', 'nested_runtime']:
                    for limit in [0, 1, 3]:
                        expected = reference(entry, limit, values)
                        outputs = [directory/f'{entry}-{i}.json' for i in range(len(expected))]
                        command = [VIZ, 'pipeline', str(shader), '--entry', entry,
                                   '--headless', '--no-config', '--push-constant', f'limit:i32={limit}',
                                   '--input', f'xs:{input_path}']
                        for i, path in enumerate(outputs):
                            path.unlink(missing_ok=True)
                            command += ['--output', f'result_{i}:{path}']
                        if args.no_validate:
                            command.append('--no-validate')
                        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                        if run.returncode or 'Validation Error' in run.stdout + run.stderr:
                            raise RuntimeError(run.stdout + run.stderr)
                        actual = [flatten(json.loads(path.read_text())) for path in outputs]
                        assert actual == expected, (target, entry, values, limit, expected, actual)
            print(target, 'zero-trip, counted/conditional products, runtime extents and nested loops passed', flush=True)
