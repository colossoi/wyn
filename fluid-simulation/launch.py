"""Run the generated WHL host program with persistent fluid feedback."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def command(shader):
    host = shader.with_suffix('.wynhost')
    if not host.is_file():
        raise FileNotFoundError(f'Missing generated host program: {host}')
    metadata = shader.with_suffix('.fluid.json')
    config = json.loads(metadata.read_text()) if metadata.exists() else dict(particles=4320, tree_capacity=4097)
    executable = 'viz.exe' if os.name == 'nt' else 'viz'
    return [str(ROOT.parent / 'extra/viz/target/release' / executable),
            'pipeline', str(shader), '--entry', 'fluid',
            '--config', str(ROOT / 'fluid.viz.json'), '--size', '640x480',
            '--storage-bytes', f'previous_positions:{config["particles"] * 16}',
            '--storage-bytes', f'previous_velocities:{config["particles"] * 16}',
            '--storage-bytes', f'previous_tree:{config["tree_capacity"] * 4}']


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
    args += ['--push-constant', f'preset:i32={preset}',
             '--push-constant', f'running:i32={int(options.running)}']
    if options.prepare_only:
        print('Compiled shader and host program; no window opened.')
    else:
        print(f'Preset: {options.preset}; starts {"running" if options.running else "paused"}. '
              'Space: pause/resume. R: reset. Left mouse: orbit.', flush=True)
        subprocess.run(args + extra, check=True)
