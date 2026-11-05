"""Validate adaptive scans/totals with an associative, noncommutative operator."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXE = '.exe' if os.name == 'nt' else ''
WYN = os.environ.get('WYN', str(ROOT / f'target/release/wyn{EXE}'))
VIZ = os.environ.get('VIZ', str(ROOT / f'extra/viz/target/release/viz{EXE}'))
SIZES = [0, 1, 31, 32, 33, 255, 256, 257, 8193, 65537]


def flatten(value):
    if isinstance(value, dict):
        return flatten(value['backing_buffer'])
    if isinstance(value, list):
        return [word for child in value for word in flatten(child)]
    return [value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-validate', action='store_true',
                        help='disable viewer layers; compiler SPIR-V still passes spirv-val')
    options = parser.parse_args()
    cases = 0
    with tempfile.TemporaryDirectory(prefix='wyn_collective_launch_') as temp:
        directory = Path(temp)
        source = directory / 'scan.wyn'
        # Encode x -> a*x+b in one u32: a is 0/1, b occupies the low 16 bits.
        # Resets every 67 inputs keep every intermediate b below 2^16.
        source.write_text('def compose(a:u32,b:u32) u32 = '
                          '((a>>16u32)*(b>>16u32)<<16u32)+(a&65535u32)*(b>>16u32)+(b&65535u32)\n' +
                          '\n'.join(f'entry scan{n}(xs:[{n or ""}]u32) ([{n or ""}]u32,u32) = '
                                    '(scan(compose,65536u32,xs),reduce(compose,65536u32,xs))' for n in SIZES))
        for target in ['spirv', 'wgsl']:
            for optimized in [False, True]:
                shader = directory / ('scan.spv' if target == 'spirv' else 'scan.wgsl')
                subprocess.run([WYN, 'build', str(source), '--max-warnings', '0', '-t', target,
                                '-o', str(shader)] + (['-O'] if optimized else []), check=True, timeout=120)
                if target == 'spirv':
                    subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(shader)], check=True)
                for n in SIZES:
                    values = [(int(i % 67 != 0), i % 7) for i in range(n)]
                    (directory / 'input.bin').write_bytes(b''.join(struct.pack('<I', a * 65536 + b) for a, b in values))
                    prefix, total = [], (1, 0)
                    for a, b in values:
                        total = (total[0] * a, total[1] * a + b)
                        prefix.append(total[0] * 65536 + total[1])
                    outputs = [directory / 'prefix.json', directory / 'total.json']
                    for output in outputs:
                        output.unlink(missing_ok=True)
                    result = subprocess.run([VIZ, 'pipeline', str(shader), '--entry', f'scan{n}',
                                             '--headless', '--no-config', '--input', f'xs:{directory / "input.bin"}',
                                             '--output', f'result_0:{outputs[0]}', '--output', f'result_1:{outputs[1]}']
                                            + (['--no-validate'] if options.no_validate else []),
                                            capture_output=True, text=True, timeout=60)
                    if result.returncode or 'Validation Error' in result.stdout + result.stderr:
                        raise RuntimeError(f'{target}, optimized={optimized}, n={n}:\n' + result.stdout + result.stderr)
                    for output, expected in zip(outputs, [prefix, [total[0] * 65536 + total[1]]]):
                        assert flatten(json.loads(output.read_text())) == expected, (target, optimized, n, output.name)
                    cases += 1
                print(f'{target}, optimized={optimized}: all scan/totals cases passed', flush=True)
    print(f'{cases} GPU cases passed; viewer layers {"disabled" if options.no_validate else "enabled"}')


if __name__ == '__main__':
    main()
