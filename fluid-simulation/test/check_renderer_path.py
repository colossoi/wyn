"""Check indirect indexed spheres against an orthographic CPU pixel reference.

Run: WYN="$HOME/.cargo/bin/wyn" python3 fluid-simulation/test/check_renderer_path.py
Requires Pillow for PNG readback. Tests culling at 3 and 129 runtime particles.
"""
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image
from gpu import ROOT, WYN, VIZ


def check(directory, shader, shift, particle_count):
    picture = directory / f'spheres-{particle_count}-{shift}.png'
    result = subprocess.run([
        str(VIZ), 'pipeline', str(shader), '--headless', '--size', '64x64',
        '--max-frames', '1', '--uniform', f'shift:f32={shift}',
        '--uniform', f'particle_count:i32={particle_count}',
        '--dump-texture', f'screen:{picture}',
    ], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    particles = [(-.15 + shift, .55, .4, (255, 0, 0)),
                 (.15 + shift, .45, .2, (0, 255, 0))]
    particles += [(3 + shift, .5, .2, (0, 255, 0))] * (particle_count-2)
    visible = [i for i, (cx, _, radius, _) in enumerate(particles)
               if abs(cx) < 1 + radius]
    ids = json.loads(result.stdout)['result_1']['backing_buffer']
    assert ids == visible + [-1] * (particle_count - len(visible)), (shift, ids)
    image = Image.open(picture).convert('RGB')
    curved_overlap = 0
    for y in range(64):
        for x in range(64):
            px, py = 2 * (x + .5) / 64 - 1, 1 - 2 * (y + .5) / 64
            hits = []
            for i in visible:
                cx, depth, radius, color = particles[i]
                r2 = ((px - cx)**2 + py**2) / radius**2
                if r2 <= 1:
                    hits.append((depth - radius * (1 - r2)**.5, color))
            expected = min(hits)[1] if hits else (0, 0, 0)
            actual = image.getpixel((x, y))
            assert actual == expected, (shift, x, y, actual, expected)
            curved_overlap += len(hits) == 2 and expected == (255, 0, 0)
    if shift == 0:
        assert curved_overlap > 0, 'fixture must distinguish sphere depth from quad depth'
    print(f'{particle_count} particles, shift={shift}: {len(visible)} visible; all 4096 pixels match')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='fluid-renderer-') as tmp:
        directory = Path(tmp)
        shader = directory / 'renderer.spv'
        subprocess.run([str(WYN), 'build',
                        str(ROOT / 'fluid-simulation/test/renderer_path.wyn'),
                        '--graphics', '-o', str(shader)], check=True)
        for particle_count in (3, 129):
            for shift in (0, 2, -3):
                check(directory, shader, shift, particle_count)
