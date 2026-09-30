// GPU numerical regression checks. Node >=18, current wyn and viz on PATH.
// All build products go to a temporary directory. No compiler changes/builds.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { deflateSync } from 'node:zlib';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = mkdtempSync(resolve(tmpdir(), 'wyn-text-'));
const wyn = process.env.WYN ?? 'wyn', viz = process.env.VIZ ?? 'viz';
function run(exe, args) {
  const r = spawnSync(exe, args, { encoding: 'utf8', timeout: 180000 });
  if (r.error) throw r.error;
  assert.equal(r.status, 0, `${exe}: ${r.stdout}\n${r.stderr}`);
  return r.stdout;
}
const font = JSON.parse(readFileSync(resolve(root, 'assets/aileron-mtsdf.json')));
const provenance = JSON.parse(readFileSync(resolve(root, 'assets/provenance.json')));
for (const [name, {sha256}] of Object.entries(provenance.files))
  assert.equal(createHash('sha256').update(readFileSync(resolve(root, 'assets', name))).digest('hex'), sha256);
assert.equal(font.atlas.type, 'mtsdf');
assert.equal(font.atlas.yOrigin, 'top');
assert.equal(font.glyphs.length, 196);
assert.deepEqual(font.kerning, []); // No kerning exported by this upstream release.
const glyphs = new Map(font.glyphs.map(g => [g.unicode, g]));
for (let cp = 32; cp < 127; cp++) assert.ok(glyphs.has(cp));
for (const g of font.glyphs) if (g.atlasBounds) {
  const a = g.atlasBounds, p = g.planeBounds;
  assert.ok(a.left >= 0 && a.top >= 0 && a.right <= font.atlas.width && a.bottom <= font.atlas.height);
  assert.ok(a.left < a.right && a.top < a.bottom && p.top < p.bottom);
  assert.ok(Math.abs((a.right-a.left) / (p.right-p.left) - font.atlas.size) < 1e-6);
}
const cap = v => Math.min(1, Math.max(0, v + 0.5));
function close(actual, expected, label) {
  assert.equal(actual.length, expected.length, `${label} length`);
  actual.forEach((v, i) => assert.ok(Number.isFinite(v) && Math.abs(v - expected[i]) <= 5e-5 * Math.max(1, Math.abs(expected[i])),
    `${label}[${i}]: got ${v}, expected ${expected[i]}`));
}
function flatten(data) {
  if (Array.isArray(data)) return data.flat(Infinity);
  if (Array.isArray(data?.backing_buffer)) return data.backing_buffer.flat(Infinity);
  throw new Error(`Unexpected GPU result: ${JSON.stringify(data).slice(0,200)}`);
}
// Tiny independent RGBA PNG fixture; no image-library dependency. Nonzero RGB
// at alpha zero also catches accidental premultiplication during image upload.
const texels = [[0,64,128,255],[255,128,0,128],[64,255,192,64],[128,0,255,0]];
function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let i=0;i<8;i++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
function chunk(name, bytes) {
  const tag = Buffer.from(name), size = Buffer.alloc(4), crc = Buffer.alloc(4);
  size.writeUInt32BE(bytes.length); crc.writeUInt32BE(crc32(Buffer.concat([tag,bytes])));
  return Buffer.concat([size,tag,bytes,crc]);
}
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(2); ihdr.writeUInt32BE(2,4); ihdr[8]=8; ihdr[9]=6;
const texture = resolve(out,'fixture.png');
writeFileSync(texture,Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',ihdr),
  chunk('IDAT',deflateSync(Buffer.from([0,...texels[0],...texels[1],0,...texels[2],...texels[3]]))),
  chunk('IEND',Buffer.alloc(0))]));
let checks = 0;
for (const [target, ext] of [['wgsl','wgsl'],['spirv','spv']]) {
  const dir = resolve(out, target); mkdirSync(dir);
  const shader = resolve(dir, `text.${ext}`);
  run(wyn, ['build', resolve(root, 'test/text.wyn'), '--max-warnings','0','-t',target,'-o',shader]);
  function gpu(entry, name, values, integer = false, extra = []) {
    const b = Buffer.alloc(values.length * 4);
    values.forEach((v,i) => integer ? b.writeInt32LE(v,i*4) : b.writeFloatLE(v,i*4));
    const input = resolve(dir, `${entry}.bin`), output = resolve(dir, `${entry}.json`);
    writeFileSync(input, b);
    run(viz, ['pipeline',shader,'--entry',entry,'--headless','--input',`${name}:${input}`,
      '--output',`${entry}:${output}`,...extra]);
    checks++;
    return flatten(JSON.parse(readFileSync(output)));
  }
  const samples = [];
  // Every permutation catches accidentally using one channel or the average.
  for (const rgb of [[.1,.5,.9],[.1,.9,.5],[.5,.1,.9],[.5,.9,.1],[.9,.1,.5],[.9,.5,.1]])
    samples.push(...rgb, 8);
  for (const x of [0,.25,.4375,.49,.5,.51,.5625,.75,1]) samples.push(x,x,x,8);
  const expected = [];
  for (let i=0;i<samples.length;i+=4) {
    const [r,g,b,range] = samples.slice(i,i+4), m = [r,g,b].sort((a,b)=>a-b)[1];
    expected.push(r-.5, m-.5, cap((m-.5)*range), cap((r-.5)*range+2)-cap((r-.5)*range));
  }
  close(gpu('distances','samples',samples), expected, `${target} signed distance/coverage/outline`);
  const rs = [1/512,0,0,1/256, 0,1/256,-1/512,0, .25,0,0,.25, 0,0,0,0];
  const re = [];
  for (let i=0;i<rs.length;i+=4) {
    const [x,y,z,w] = rs.slice(i,i+4);
    re.push(Math.max(.5*((8/512)/Math.max(Math.hypot(x,z),1e-12)+(8/256)/Math.max(Math.hypot(y,w),1e-12)),1),
      Math.max(8*x/48,1),8/512,8/256);
  }
  close(gpu('ranges','samples',rs), re, `${target} rotation/minification/degenerate derivatives`);
  const steps = [], positions = []; let x=0,y=0;
  // Cross several scan workgroups and include consecutive/leading/trailing breaks.
  for (let i=0;i<777;i++) {
    const advance = [0,.25,.75,-.125][i%4], br = i%97 === 0 || i%97 === 1 || i===776;
    steps.push(advance,br?1:0); positions.push(x,y);
    if (br) {x=0;y+=24;} else x+=advance*20;
  }
  close(gpu('layout','steps',steps), positions, `${target} segmented layout`);
  close(gpu('layout','steps',[.5,0]), [0,0], `${target} singleton layout`);
  // The current viz allocator cannot bind a zero-byte runtime vec2 input.
  // Exercise empty layout inside Wyn instead, without an external buffer.
  const empty = resolve(dir, 'empty.json');
  run(viz, ['pipeline', shader, '--entry', 'empty_layout', '--headless',
    '--output', `empty_layout:${empty}`]);
  assert.equal(JSON.parse(readFileSync(empty)), 0); checks++;
  const cps = [65,86,32,10,10,233,13,10,9,160,0x1f600,-1,0x10ffff];
  const fpos = []; x=0;y=0;
  function lookup(cp) {
    if (cp===10 || cp===13) return {advance:0};
    if (cp===9) return {advance:4*glyphs.get(32).advance};
    return glyphs.get(cp===160?32:cp) ?? glyphs.get(63);
  }
  for (const cp of cps) {
    fpos.push(x,y);
    if (cp===10) {x=0;y+=24;} else x+=lookup(cp).advance*20;
  }
  close(gpu('font_layout','codepoints',cps,true),fpos,`${target} newline/CRLF/TAB/NBSP/fallback`);
  const all = [...glyphs.keys(),...cps,0,0xaa];
  close(gpu('metrics','codepoints',all,true),all.flatMap(cp=>{
    const g=lookup(cp);return [g.advance,g.planeBounds?.top??0,g.planeBounds?.bottom??0,glyphs.has(cp)?1:0];
  }),`${target} all glyphs and missing glyphs`);
  const a=glyphs.get(65), p=a.planeBounds, b=a.atlasBounds;
  close(gpu('quads','indices',[0,1,2,3,4,5],true),[[0,0],[1,0],[0,1],[0,1],[1,0],[1,1]].flatMap(([r,d])=>
    [100+40*(r?p.right:p.left),80+40*(d?p.bottom:p.top),
      (r?b.right:b.left)/font.atlas.width,(d?b.bottom:b.top)/font.atlas.height]),`${target} quad winding and UV origin`);
  const colors=[1,.2,.4,0, 1,.2,.4,.5, 1,.2,.4,1];
  close(gpu('colors','samples',colors),[0,.5,1].flatMap(alpha=>{
    const a=alpha*.5;return [1*a+.04*(1-a),.2*a+.08*(1-a),.4*a+.12*(1-a),a+.4*(1-a)];
  }),`${target} premultiplied source-over`);
  const uvs = [0,0, .25,.25, .75,.25, .25,.75, .75,.75, .5,.5, 1,1, -.25,.5, 1.25,.5];
  const sampled = [];
  for (let i=0;i<uvs.length;i+=2) {
    const x = Math.min(1,Math.max(0,uvs[i]*2-.5));
    const y = Math.min(1,Math.max(0,uvs[i+1]*2-.5));
    for (let c=0;c<4;c++) sampled.push(((1-y)*((1-x)*texels[0][c]+x*texels[1][c])
      +y*((1-x)*texels[2][c]+x*texels[3][c]))/255);
  }
  close(gpu('sampling','uvs',uvs,false,['--image',`atlas:${texture}`]),sampled,
    `${target} bilinear interpolation/texel centers/clamped edges/alpha preservation`);
  console.log(`${target}: numeric GPU checks passed.`);
}
console.log(`PASS: assets and ${checks} GPU cases; results in ${out}`);
