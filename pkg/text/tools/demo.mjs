// No npm dependencies. Node >=18; WYN and VIZ select existing local executables.
// node tools/demo.mjs [output-directory] [--interactive]
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = resolve(process.argv.slice(2).find(a => !a.startsWith('--')) ?? resolve(root, 'build/demo'));
mkdirSync(out, { recursive: true });
const font = JSON.parse(readFileSync(resolve(root, 'assets/aileron-mtsdf.json')));
const glyphs = new Map(font.glyphs.map(g => [g.unicode, g]));
const ink = [0.91, 0.95, 0.98], muted = [0.43, 0.55, 0.64], mint = [0.42, 0.96, 0.76];
const instances = [], styles = [];
function line(text, x, y, size, mode = 1, color = ink) {
  for (const ch of text) {
    const cp = ch.codePointAt(0), g = glyphs.get(cp) ?? glyphs.get(63);
    if (g.planeBounds) { instances.push(x, y, size, cp); styles.push(...color, mode); }
    x += g.advance * size;
  }
}
line('WYN / TEXT', 52, 51, 16, 1, mint);
line('Same font. Three renderers.', 48, 143, 70);
line('Aileron Regular  /  CC0  /  196 glyphs', 52, 195, 22, 1, muted);
line('01  SDF', 72, 303, 20, 1, mint);
line('02  MSDF', 504, 303, 20, 1, mint);
line('03  SLUG', 936, 303, 20, 3, mint);
line('True distance / alpha', 72, 340, 18, 1, muted);
line('Median distance / RGB', 504, 340, 18, 1, muted);
line('Direct curve coverage', 936, 340, 18, 3, muted);
for (const [x, mode] of [[72, 0], [504, 1], [936, 3]]) {
  line('AaW', x - 2, 515, 136, mode);
  line('AVMW / 0123', x, 585, 36, mode);
  line('The same font, scaled.', x, 637, 23, mode);
  line('Café • naïve • €48.00', x, 683, 18, mode, muted);
}
line('SLUG / QUADRATIC OUTLINES + DYNAMIC HALF-PIXEL DILATION', 76, 786, 16, 3, mint);
line('Curves, without a distance atlas.', 76, 865, 56, 3);
if (instances.length / 4 !== 353) throw new Error(`Update the fixed draw count in examples/demo.wyn: ${instances.length/4}`);
const binary = values => { const b = Buffer.alloc(values.length * 4); values.forEach((v,i)=>b.writeFloatLE(v,i*4)); return b; };
writeFileSync(resolve(out, 'instances.bin'), binary(instances));
writeFileSync(resolve(out, 'styles.bin'), binary(styles));
function run(exe, args) {
  const r = spawnSync(exe, args, { stdio: 'inherit', cwd: root });
  if (r.error) throw r.error;
  if (r.status !== 0) throw new Error(`${exe} exited ${r.status}`);
}
run(process.env.WYN ?? 'wyn', ['build', resolve(root, 'examples/demo.wyn'), '--graphics',
  '--max-warnings', '0', '-t', 'wgsl', '-o', resolve(out, 'demo.wgsl')]);
const args = ['pipeline', resolve(out, 'demo.wgsl'), '--size', '1360x940',
  '--image', `atlas:${resolve(root, 'assets/aileron-mtsdf.png')}`,
  '--input', `instances:${resolve(out, 'instances.bin')}`,
  '--input', `styles:${resolve(out, 'styles.bin')}`];
args.push('--input', `curves:${resolve(root, 'assets/aileron-slug-curves.bin')}`,
  '--input', `bands:${resolve(root, 'assets/aileron-slug-bands.bin')}`,
  '--input', `indices:${resolve(root, 'assets/aileron-slug-indices.bin')}`);
if (!process.argv.includes('--interactive')) args.push('--headless', '--output', `screen:${resolve(out, 'demo.png')}`);
run(process.env.VIZ ?? 'viz', args);
console.log(`Demo: ${out}`);
