"""Check view publication and parallel copies against elementwise CPU results."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile

from test_dispatch_loop_gpu import ROOT, VIZ, WYN, flatten


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()
    values = list(range(300))
    cases = [('prefix', {}, [values[:3]]),
             ('prefixes', {}, [values, values[:3], values[:5]]),
             ('compacted', {}, [values[2::2]]),
             ('preserve', {}, [values[1:], [-7]+values[1:]])]
    cases += [('conditional', dict(cutoff=0), [[], [], []]),
              ('conditional', dict(cutoff=len(values)), [values, values, values[1:]])]
    for count in [0, 1, 65]:
        cases.append(('rows', dict(start=2, count=count), [values[6:6+count*3]]))
    for start, count in [(0, 0), (7, 0), (3, 1), (2, 65), (17, 257)]:
        sliced = values[start:start+count]
        params = dict(start=start, count=count)
        cases += [('shifted', params, [sliced]),
                  ('produced', params, [[x+7 for x in sliced]]),
                  ('mixed', params, [values, sliced, [sum(values)]]),
                  ('range', dict(count=count), [list(range(count))])]
    fixtures = [('output_copies', cases),
                ('nested_component_copies', [
                    ('projected_rows', dict(flag=flag),
                     [[x+flag-1 for x in values], [x+flag+999 for x in values]])
                    for flag in [0, 1]])]
    with tempfile.TemporaryDirectory(prefix='wyn-output-copies-') as temp:
        directory = Path(temp)
        data = directory/'input.bin'
        data.write_bytes(struct.pack(f'<{len(values)}i', *values))
        for target in ['wgsl', 'spirv']:
            for fixture, fixture_cases in fixtures:
                shader = directory/(fixture + ('.wgsl' if target == 'wgsl' else '.spv'))
                subprocess.run([WYN, 'build', str(ROOT/f'tests/compiler/library/{fixture}.wyn'),
                                '-t', target, '-o', str(shader)], check=True, timeout=120)
                if target == 'spirv':
                    subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(shader)], check=True)
                for entry, params, expected in fixture_cases:
                    outputs = [directory/f'result-{i}.json' for i in range(len(expected))]
                    command = [VIZ, 'pipeline', str(shader), '--entry', entry, '--headless', '--no-config']
                    if entry != 'range':
                        command += ['--input', f'xs:{data}']
                    for key, value in params.items():
                        command += ['--push-constant', f'{key}:i32={value}']
                    for i, path in enumerate(outputs):
                        path.unlink(missing_ok=True)
                        name = entry if len(outputs) == 1 else f'result_{i}'
                        command += ['--output', f'{name}:{path}']
                    if args.no_validate:
                        command.append('--no-validate')
                    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    if run.returncode or 'Validation Error' in run.stdout + run.stderr:
                        raise RuntimeError(f'{target} {entry} {params}:\n' + run.stdout + run.stderr)
                    actual = [flatten(json.loads(path.read_text())) for path in outputs]
                    assert actual == expected, (target, entry, params, expected, actual)
            print(target, sum(len(cases) for _, cases in fixtures), 'output copy cases passed', flush=True)


if __name__ == '__main__':
    main()
