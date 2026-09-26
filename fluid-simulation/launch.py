"""Run the generated WHL host program with persistent fluid feedback."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def command(shader):
    host = shader.with_suffix('.wynhost')
    if not host.is_file():
        raise FileNotFoundError(f'Missing generated host program: {host}')
    # Keep these capacities in sync with count/tree_capacity in src/main.wyn.
    # The host program allocates intermediate buffers; only feedback inputs
    # need caller-provided storage on the first frame.
    return [str(ROOT.parent / 'extra/viz/target/release/viz'),
            'pipeline', str(shader), '--entry', 'fluid',
            '--config', str(ROOT / 'fluid.viz.json'), '--size', '640x480',
            '--storage-bytes', f'previous_positions:{4320 * 16}',
            '--storage-bytes', f'previous_velocities:{4320 * 16}',
            '--storage-bytes', f'previous_tree:{4097 * 4}']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shader', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preset', choices=['colliding-blocks', 'double-dam-break', 'rotating-block'],
                        default='colliding-blocks')
    playback = parser.add_mutually_exclusive_group()
    playback.add_argument('--running', dest='running', action='store_true', default=True)
    playback.add_argument('--paused', dest='running', action='store_false')
    options, extra = parser.parse_known_args()
    args = command(options.shader.resolve())
    preset = ['colliding-blocks', 'double-dam-break', 'rotating-block'].index(options.preset)
    args += ['--uniform', f'preset:i32={preset}',
             '--uniform', f'running:i32={int(options.running)}']
    if options.prepare_only:
        print('Compiled shader and host program; no window opened.')
    else:
        print(f'Preset: {options.preset}; starts {"running" if options.running else "paused"}. '
              'Space: pause/resume. R: reset. Left mouse: orbit.', flush=True)
        subprocess.run(args + extra, check=True)
