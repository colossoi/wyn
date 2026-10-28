"""Check compiler tiled-output and native-subgroup lowering against CPU results.

Build wyn and viz first, then run from the repository root. WYN/VIZ override them.
"""
from itertools import accumulate
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from gpu import WYN, VIZ, flatten


def execute(shader, entry, directory, values=None, kind='i'):
    command = [str(VIZ), 'pipeline', str(shader), '--entry', entry,
               '--headless', '--max-frames', '1']
    if values is not None:
        binary = directory / 'input.bin'
        binary.write_bytes(struct.pack(f'<{len(values)}{kind}', *values))
        name = 'floats' if entry=='float17' else 'short' if entry=='max17' else 'xs'
        command += ['--input', f'{name}:{binary}']
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    output = json.loads(result.stdout)
    return [flatten(output[key]) for key in sorted(output)]


def main():
    tiled = [(a//8)*4+a%4+(a%8)//4 for a in range(16)]
    rows = [v for i in range(8) for v in [i,i+1,i+2]]
    padded = [tiled[a] if (a//8)*4+a%4<7 else -1 for a in range(16)]
    with tempfile.TemporaryDirectory(prefix='fluid-kernels-') as tmp:
        directory = Path(tmp)
        for target in ['spirv', 'wgsl']:
            extension = '.spv' if target == 'spirv' else '.wgsl'
            for fixture in ['subgroups', 'tiled_output']:
                shader = directory / (fixture + extension)
                subprocess.run([str(WYN), 'build', str(Path(__file__).with_name(fixture+'.wyn')),
                                '--subgroups', '-t', target, '-o', str(shader)], check=True)
                if fixture == 'tiled_output':
                    for entry, expected in [('mapped',[tiled,list(range(2,10))]),
                                            ('reduced',[tiled,[44]]), ('retained',[tiled,rows]),
                                            ('padded',[padded])]:
                        assert execute(shader,entry,directory) == expected, (target,entry)
                else:
                    for entry, width, kind in [('scan65',65,'i'),('sum65',65,'i'),
                                                ('float17',17,'f'),('max17',17,'i')]:
                        values = [(i*13)%19-9 for i in range(3*width)]
                        expected = []
                        for start in range(0,len(values),width):
                            row = values[start:start+width]
                            expected += ([sum(row)] if entry=='sum65' else
                                         list(accumulate(row,max)) if entry=='max17' else
                                         list(accumulate(row)))
                        assert execute(shader,entry,directory,values,kind) == [expected], (target,entry)
                print(target, fixture, 'passed', flush=True)


if __name__ == '__main__':
    main()
