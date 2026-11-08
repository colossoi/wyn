"""Dispatch the generated Fold application through the unchanged Viz binary."""
import argparse
from pathlib import Path
import sys
from deadline import run

ROOT = Path(__file__).resolve().parent

def command(size):
    shader = ROOT / 'generated/fold.wgsl'
    if not shader.is_file() or not shader.with_suffix('.wynhost').is_file():
        raise FileNotFoundError('Build the Wyn application first: python3 fold/build.py')
    return [str(ROOT.parent / 'extra/viz/target/release/viz'),
            'pipeline', str(shader), '--entry', 'fold',
            '--config', str(ROOT / 'fold.viz.json'), '--size', size,
            '--storage-bytes', 'previous:83312']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', default='900')
    parser.add_argument('--deadline', type=float, default=5, help='Process-group SIGKILL deadline in seconds (maximum 30)')
    options, extra = parser.parse_known_args()
    size = options.size if 'x' in options.size else f'{options.size}x{options.size}'
    try:
        run(command(size) + extra, seconds=options.deadline)
    except TimeoutError as error:
        print(error, file=sys.stderr)
        sys.exit(124)
