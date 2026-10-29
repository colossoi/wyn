"""Build one specialization of the shared fluid source, without editing it."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
EXE = '.exe' if os.name == 'nt' else ''


def configuration(particles=4320, tree_capacity=None):
    if not 1 <= particles <= 1_000_000:
        raise ValueError('particles must be between 1 and 1,000,000')
    leaf_threshold = 32
    # Match upstream Octree::max_leaves, retaining the default scene's reserve.
    # Round up to whole sibling groups and include the sentinel below.
    desired = max(4096, 24 * particles // (leaf_threshold + 1))
    capacity = tree_capacity if tree_capacity is not None else 7 * math.ceil((desired - 1) / 7) + 2
    if capacity < 2 or capacity > 2**24:
        raise ValueError('tree capacity must be between 2 and 16,777,216')
    scale = max(1.0, (particles / 4320) ** (1 / 3))
    return dict(particles=particles, tree_capacity=capacity, leaf_threshold=leaf_threshold,
                bounds=[coordinate * scale for coordinate in (56, 35, 30)])


def build(compiler, output, particles=4320, tree_capacity=None, target='spirv', cost_profile=None):
    config = configuration(particles, tree_capacity)
    output = Path(output).resolve()
    package = output / 'package'
    (package / 'src').mkdir(parents=True, exist_ok=True)
    for source in (ROOT / 'src').rglob('*.wyn'):
        destination = package / 'src' / source.relative_to(ROOT / 'src')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    manifest = (ROOT / 'wyn.toml').read_text()
    manifest = re.sub(r'path = "([^"\n]+)"',
                      lambda match: 'path = ' + json.dumps(Path(os.path.relpath(
                          (ROOT / match[1]).resolve(), package)).as_posix()), manifest)
    (package / 'wyn.toml').write_text(manifest)
    (package / 'src/config.wyn').write_text(
        f'def count: i32 = {particles}\n'
        f'def tree_capacity: i32 = {config["tree_capacity"]}\n'
        f'def leaf_threshold: i32 = {config["leaf_threshold"]}\n'
        'def bounds: vec3f32 = @[' + ','.join(f'{x:.8f}' for x in config['bounds']) + ']\n')
    shader = output / ('fluid.spv' if target == 'spirv' else 'fluid.wgsl')
    command = [str(Path(compiler).resolve()), 'build', str(package), '--graphics', '--subgroups',
               '-t', target, '-o', str(shader)]
    if cost_profile:
        command += ['--gpu-cost-profile', str(Path(cost_profile).resolve())]
    env = {**os.environ, 'RUST_MIN_STACK': '67108864'}
    with (output / 'build.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
    config.update(target=target, compiler_sha256=hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
                  shader_sha256=hashlib.sha256(shader.read_bytes()).hexdigest())
    shader.with_suffix('.fluid.json').write_text(json.dumps(config, indent=2) + '\n')
    return shader, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, default=4320)
    parser.add_argument('--tree-capacity', type=int)
    parser.add_argument('--target', choices=['spirv', 'wgsl'], default='spirv')
    parser.add_argument('--compiler', type=Path, default=ROOT.parent / f'target/release/wyn{EXE}')
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'tmp/fluid-simulation')
    parser.add_argument('--gpu-cost-profile', type=Path)
    args = parser.parse_args()
    shader, _ = build(args.compiler, args.output, args.particles, args.tree_capacity, args.target, args.gpu_cost_profile)
    print(shader)
