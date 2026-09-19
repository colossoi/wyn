"""Allocate descriptor-published scratch buffers before running viz."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def command(shader):
    descriptor = json.loads(shader.with_suffix('.json').read_text())
    config = json.loads((ROOT / 'fluid.viz.json').read_text())
    feedback_slots = set()
    for pair in config['feedback']:
        matches = [s for s in descriptor['source_results']
                   if s['entry'] == pair['entry'] and s['result'] == pair['result']]
        if len(matches) != 1:
            raise ValueError('Compiler stage layout changed; update fluid.viz.json')
        result = matches[0]
        feedback_slots.add((result['set'], result['binding']))
    args = [str(ROOT.parent / 'extra/viz/target/release/viz'), 'pipeline', str(shader),
            '--config', str(ROOT / 'fluid.viz.json'), '--size', '640x480']
    buffers = {}
    for pipeline in descriptor['pipelines']:
        for binding in pipeline['bindings']:
            if binding['type'] != 'storage_buffer' or binding['access'] == 'read_only':
                continue
            if (binding['set'], binding['binding']) in feedback_slots:
                continue
            length = binding.get('length') or {}
            if length.get('kind') != 'fixed':
                raise ValueError(f"Missing static buffer size: {binding['name']}")
            buffers[binding['name']] = length['bytes']
    for name, size in buffers.items():
        args += ['--buffer-init', f'{name}:0', '--storage-bytes', f'{name}:{size}']
    return args


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
    args += ['--uniform', f'preset:i32x4={preset},0,0,0',
             '--uniform', f'running:i32x4={int(options.running)},0,0,0']
    if options.prepare_only:
        print('Compiled and validated launch configuration; no window opened.')
    else:
        print(f'Preset: {options.preset}; starts {"running" if options.running else "paused"}. '
              'Space: pause/resume. R: reset. Left mouse: orbit.', flush=True)
        subprocess.run(args + extra, check=True)
