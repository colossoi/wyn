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
line('Distance, rendered.', 48, 143, 76);
line('Aileron Regular  /  CC0  /  196 glyphs', 52, 195, 22, 1, muted);
line('01  SDF', 76, 303, 20, 1, mint);
line('02  MSDF', 724, 303, 20, 1, mint);
line('True distance / alpha', 76, 340, 18, 1, muted);
line('Sharp corners / RGB', 724, 340, 18, 1, muted);
for (const [x, mode] of [[76, 0], [724, 1]]) {
  line('AaW', x - 2, 522, 180, mode);
  line('AVMW / 0123', x, 593, 42, mode);
  line('Scales from one atlas.', x, 641, 24, mode);
  line('Café • naïve • €48.00', x, 683, 18, mode, muted);
}
line('03  MTSDF / MSDF FILL + SDF OUTLINE', 76, 786, 16, 1, mint);
line('One texture. Both distances.', 76, 865, 56, 2);
if (instances.length / 4 !== 247) throw new Error('Update the fixed draw count in examples/demo.wyn.');
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
if (!process.argv.includes('--interactive')) args.push('--headless', '--output', `screen:${resolve(out, 'demo.png')}`);
run(process.env.VIZ ?? 'viz', args);
console.log(`Demo: ${out}`);
