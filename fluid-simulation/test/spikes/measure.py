"""Validate and time prepared /tmp/spike-* shaders with a frame-timing viz build.

Usage: python3 measure.py /path/to/fluid-timing-viz
Run patch_generated.py first. This does not modify compiler or simulator code.
"""
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys

root = Path('/tmp')
(root / 'spike-positions.bin').write_bytes(b''.join(
    struct.pack('<4f', .25*(i%64), 10.*(i//64), 0, .35) for i in range(4096)))
(root / 'spike-xs.bin').write_bytes(b''.join(
    struct.pack('<I', i%17) for i in range(32768*32)))

def flat(value):
    if isinstance(value, dict):
        return flat(value['backing_buffer'])
    if isinstance(value, list):
        return [x for child in value for x in flat(child)]
    return [value]

expected_scan = []
for row in range(32768):
    total = 0
    for lane in range(32):
        total += (row*32+lane)%17
        expected_scan.append(total)

report = {'load_before': os.getloadavg(), 'measurements': {}}
names = ['neighbor', 'neighbor-reuse', 'neighbor-fused', 'subgroup', 'subgroup-native']
# Reverse the second round to expose simple ordering/clock effects.
for round_id, order in enumerate([names, names[::-1]]):
    for name in order:
        neighbor = name.startswith('neighbor')
        entry, arg = ('baseline', 'positions') if neighbor else ('main', 'xs')
        stem = root / f'spike-{name}-round{round_id}'
        with stem.with_suffix('.json').open('w') as out, stem.with_suffix('.log').open('w') as err:
            subprocess.run([sys.argv[1], 'pipeline', str(root/f'spike-{name}.wgsl'),
                '--entry', entry, '--input', f'{arg}:/tmp/spike-{arg}.bin',
                '--headless', '--max-frames', '16'], stdout=out, stderr=err, check=True, timeout=90)
        output = json.loads(stem.with_suffix('.json').read_text())
        if neighbor:
            values, counts = flat(output['result_0']), flat(output['result_1'])
            for i in range(4096):
                ids = [j for j in range(i//64*64, (i//64+1)*64) if abs(i-j)<4]
                assert counts[i] == len(ids), (name, i)
                assert [values[i//32*3072+k*32+i%32] for k in range(96)] == ids+[-1]*(96-len(ids)), (name, i)
        else:
            assert flat(output['main']) == expected_scan, name
        times = [float(t) for t in re.findall(r'\[frame timing\] \d+ ([\d.]+) ms', stem.with_suffix('.log').read_text())][2:]
        assert len(times) == 14
        result = {'median_ms': statistics.median(times), 'samples_ms': times, 'correct': True}
        report['measurements'].setdefault(name, []).append(result)
        print(name, round_id, result['median_ms'], flush=True)
report['load_after'] = os.getloadavg()
(root / 'spike-results.json').write_text(json.dumps(report, indent=2)+'\n')
