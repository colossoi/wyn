"""Compile the complete Wyn application for the unchanged Viz runtime."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'generated'

def build():
    OUT.mkdir(exist_ok=True)
    shader = OUT / 'fold.wgsl'
    shader.unlink(missing_ok=True)
    shader.with_suffix('.wynhost').unlink(missing_ok=True)
    environment = dict(os.environ)
    environment.setdefault('RUST_MIN_STACK', '67108864')
    subprocess.run([environment.get('WYN', str(ROOT.parent / 'target/release/wyn')),
                    'build', str(ROOT / 'src/main.wyn'), '--graphics', '-t', 'wgsl',
                    '--max-warnings', '0', '-o', str(shader)],
                   check=True, env=environment)

if __name__ == '__main__':
    build()
