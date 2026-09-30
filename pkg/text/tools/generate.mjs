// Offline, deterministic atlas build. Requires Node >=18 and msdf-atlas-gen 1.4.
// Usage: node tools/generate.mjs /path/to/msdf-atlas-gen[.exe]
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { generateFont } from './generate-font.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const generator = process.argv[2];
if (!generator) throw new Error('Pass the path to msdf-atlas-gen 1.4.0.');
const help = spawnSync(generator, ['-help'], { encoding: 'utf8' });
if (!help.stdout?.includes('v1.4.0')) throw new Error('Expected msdf-atlas-gen v1.4.0');
const font = resolve(root, 'assets/Aileron-Regular.otf');
const sha256 = path => createHash('sha256').update(readFileSync(path)).digest('hex');
if (sha256(font) !== '2762f4fc2ebad8323264aea52ffa2260b86c9677493d3ce2dc4f34e5851d2aa2')
  throw new Error('Bundled Aileron Regular 0.102 checksum mismatch.');
// These Latin-1 codepoints are absent in the author's 0.102 font.
const absent = new Set([0xa0, 0xaa, 0xad, 0xb2, 0xb3, 0xb5, 0xb9, 0xba, 0xbc, 0xbd, 0xbe]);
const codepoints = [...Array.from({ length: 95 }, (_, i) => i + 32),
  ...Array.from({ length: 96 }, (_, i) => i + 160).filter(cp => !absent.has(cp)),
  0x152, 0x153, 0x160, 0x161, 0x178, 0x2013, 0x2014, 0x2018, 0x2019,
  0x201c, 0x201d, 0x2022, 0x2026, 0x20ac, 0x2122, 0x2212];
const charset = codepoints.map(cp => `0x${cp.toString(16)}`).join(', ') + '\n';
writeFileSync(resolve(root, 'assets/charset.txt'), charset);
const args = ['-font', font, '-charset', resolve(root, 'assets/charset.txt'),
  '-type', 'mtsdf', '-size', '48', '-pxrange', '8', '-square4', '-yorigin', 'top',
  '-seed', '0', '-threads', '1', '-format', 'png',
  '-imageout', resolve(root, 'assets/aileron-mtsdf.png'),
  '-json', resolve(root, 'assets/aileron-mtsdf.json')];
const run = spawnSync(generator, args, { encoding: 'utf8' });
process.stdout.write(run.stdout ?? '');
process.stderr.write(run.stderr ?? '');
if (run.status !== 0 || /Missing \d+ codepoints/.test(run.stdout + run.stderr))
  throw new Error(`Atlas generation failed (${run.status}).`);
const data = JSON.parse(readFileSync(resolve(root, 'assets/aileron-mtsdf.json'), 'utf8'));
if (data.atlas.yOrigin !== 'top' || data.atlas.type !== 'mtsdf' || data.glyphs.length !== codepoints.length)
  throw new Error('Unexpected atlas metadata.');
const { width, height } = data.atlas;
const glyphs = data.glyphs.sort((a, b) => a.unicode - b.unicode);
writeFileSync(resolve(root, 'assets/aileron-mtsdf.json'), JSON.stringify(data, null, 2) + '\n');
const manifest = {
  font: 'Aileron Regular', version: '0.102', author: 'Sora Sagano (dot colon)',
  license: 'CC0-1.0', source: 'https://dotcolon.net/fonts/aileron/',
  download: 'https://dotcolon.net/files/fonts/aileron_0102.zip',
  generator: 'msdf-atlas-gen 1.4.0 / MSDFgen 1.13.0',
  generatorSource: 'https://github.com/Chlumsky/msdf-atlas-gen/releases/tag/v1.4',
  options: ['-type mtsdf', '-size 48', '-pxrange 8', '-square4', '-yorigin top', '-seed 0', '-threads 1'],
  glyphCount: glyphs.length, missingLatin1: [...absent].sort((a,b)=>a-b),
  files: Object.fromEntries(['Aileron-Regular.otf','charset.txt','aileron-mtsdf.png','aileron-mtsdf.json']
    .map(name => [name, {sha256: sha256(resolve(root, 'assets', name))}]))
};
writeFileSync(resolve(root, 'assets/provenance.json'), JSON.stringify(manifest, null, 2) + '\n');
generateFont(root);
console.log(`Generated ${glyphs.length} glyphs in ${width}x${height} MTSDF (RGB=MSDF, A=SDF).`);
