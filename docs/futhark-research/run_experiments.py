#!/usr/bin/env python3
"""Reproduce the Futhark inlining/fusion experiment (Linux, official 0.27.1).
Usage: python3 run_experiments.py /path/to/futhark /tmp/futhark-results
The resource cap applies to each compiler process. No GPU is needed.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

compiler = str(Path(sys.argv[1]).resolve())
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
fixtures = Path(__file__).resolve().parent
before_fusion = ['--simplify', '--inline-conservatively', '--simplify',
                 '--inline-aggressively', '--simplify', '--cse', '--simplify']
version = subprocess.check_output([compiler, '--version'], text=True)
results = []
for fixture in sorted(fixtures.glob('radix-*.fut')):
    for stage, flags in [('before-fusion', before_fusion), ('standard', ['--standard']), ('gpu', ['--gpu'])]:
        name = fixture.stem + '-' + stage
        ir_path, log_path, time_path = [out / (name + ext) for ext in ['.ir', '.log', '.time']]
        command = ['/usr/bin/time', '-f', '%e %M', '-o', str(time_path),
                   'prlimit', '--as=3221225472', '--core=0', '--', 'timeout', '120',
                   compiler, 'dev', *flags, '-v', str(fixture), '+RTS', '-N1', '-M2G', '-RTS']
        with ir_path.open('w') as stdout, log_path.open('w') as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr)
        ir, log = ir_path.read_text(), log_path.read_text()
        wall, rss = time_path.read_text().splitlines()[-1].split()
        # Verbose elapsed times belong to the PREVIOUS marker, not the next pass.
        events = re.findall(r'^\[\s*\+\s*([0-9.]+)\] (.*)$', log, re.M)
        timings = [{'pass': events[i][1].removeprefix('Running pass: '),
                    'seconds': float(events[i+1][0])}
                   for i in range(len(events)-1) if events[i][1].startswith('Running pass: ')]
        row = dict(fixture=fixture.name, stage=stage, exit_code=process.returncode,
                   wall_seconds=float(wall), max_rss_kib=int(rss), ir_bytes=ir_path.stat().st_size,
                   radix_bit_calls=len(re.findall(r'\bapply radix_bit_\w*\(', ir)),
                   soac_nodes=len(re.findall(r'\b(?:map|reduce|scan|scanomap|redomap|scanomapper|screma|stream|hist)\(', ir)),
                   scan_nodes=len(re.findall(r'\b(?:scan|scanomap|scanomapper|screma)\(', ir)),
                   pass_timings=timings)
        results.append(row)
        (out / 'results.json').write_text(json.dumps(dict(version=version, results=results), indent=2)+'\n')
        print(name, process.returncode, wall+'s', rss+' KiB', row['scan_nodes'], 'scans', flush=True)
        if process.returncode:
            print(log[-1500:], flush=True)
