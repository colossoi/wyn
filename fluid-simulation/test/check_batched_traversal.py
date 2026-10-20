"""Check the cooperative traversal probe against an independent scalar BFS.

Compilation failure is reported without rewriting the Wyn to avoid it.
Numeric success does not assert cooperative scheduling; inspect generated code.
WYN and VIZ may select binaries. Run from any directory.
"""
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile

from gpu import ROOT, VIZ, WYN, flatten, run


def oracle(hits):
    pending = [0]
    leaves = []
    while pending:
        node = pending.pop(0)
        if not hits[node]:
            continue
        if node >= 9:
            leaves.append(node)
        else:
            first = 1 if node == 0 else 9 + (node - 1) * 8
            pending.extend(range(first, first + 8))
    return leaves + [-1] * (64 - len(leaves)) + [len(leaves)]


def cases():
    rows = [[0] * 73, [1] * 73, [1] + [0] * 72]
    for parent in range(1, 9):
        row = [0] * 73
        row[0] = row[parent] = 1
        first = 9 + (parent - 1) * 8
        row[first:first + 8] = [1] * 8
        rows.append(row)
    rng = random.Random(20261020)
    rows.extend([[rng.randrange(2) for _ in range(73)] for _ in range(65)])
    return rows


def main():
    rows = cases()
    expected = [value for row in rows for value in oracle(row)]
    with tempfile.TemporaryDirectory(prefix='wyn-traversal-') as temporary:
        directory = Path(temporary)
        for target in ['spirv', 'wgsl']:
            actual, = run(ROOT / 'fluid-simulation/test/repro_loop_scatter_snapshot.wyn', directory, target)
            assert actual == [10, 11, 12, 0, 20, 21, 22, 0], f'{target}: stale scatter snapshot: {actual}'
            print(f'{target}: loop-carried scatter preserves all writes', flush=True)
            actual, = run(ROOT / 'fluid-simulation/test/repro_shared_loop_swap.wyn', directory, target)
            assert actual == [1,2,11,21,3,4,11,21], f'{target}: loop array swap aliases its source: {actual}'
            actual, = run(ROOT / 'fluid-simulation/test/repro_branch_array_layout.wyn', directory, target)
            assert actual == [6,10], f'{target}: inferred array boundary: {actual}'
            print(f'{target}: shared loop swaps and inferred array boundaries passed', flush=True)
        inputs = directory / 'intersections.bin'
        values = [value for row in rows for value in row]
        inputs.write_bytes(struct.pack(f'<{len(values)}i', *values))
        for target in ['spirv', 'wgsl']:
            shader = directory / ('traversal.spv' if target == 'spirv' else 'traversal.wgsl')
            subprocess.run([str(WYN), 'build', str(ROOT / 'fluid-simulation/test/batched_traversal.wyn'),
                            '-t', target, '-o', str(shader)], check=True)
            try:
                result = subprocess.run([str(VIZ), 'pipeline', str(shader), '--verbose',
                                         '--input', f'intersections:{inputs}'],
                                        check=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired as error:
                diagnostic = error.stderr or b''
                print(diagnostic.decode() if isinstance(diagnostic, bytes) else diagnostic, flush=True)
                raise
            print(result.stderr, end='')
            outputs = json.loads(result.stdout)
            actual = flatten(next(iter(outputs.values())))
            assert actual == expected, f'{target}: BFS order, pruning, padding, or counts differ'
            print(f'{target}: {len(rows)} queries match BFS oracle')


if __name__ == '__main__':
    main()
