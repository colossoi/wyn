// Independent geometry checks and GPU execution on both supported targets.
// Node >=18; existing WYN/VIZ executables. No npm dependencies/compiler edits.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = mkdtempSync(resolve(tmpdir(),'wyn-slug-'));
const wyn = process.env.WYN ?? 'wyn', viz = process.env.VIZ ?? 'viz';
function run(exe,args) {
  const r=spawnSync(exe,args,{encoding:'utf8',timeout:180000});
  if(r.error) throw r.error;
  assert.equal(r.status,0,`${exe}: ${r.stdout}\n${r.stderr}`);
}
const data=JSON.parse(readFileSync(resolve(root,'assets/aileron-slug.json')));
const decode=(name,integer=false)=>{
  const bytes=readFileSync(resolve(root,'assets',name));
  return Array.from({length:bytes.length/4},(_,i)=>integer?bytes.readInt32LE(i*4):bytes.readFloatLE(i*4));
};
const curves=decode('aileron-slug-curves.bin'), bands=decode('aileron-slug-bands.bin',true), indices=decode('aileron-slug-indices.bin',true);
for(const [name,{sha256}] of Object.entries(data.files))
  assert.equal(createHash('sha256').update(readFileSync(resolve(root,'assets',name))).digest('hex'),sha256);
assert.equal(data.fontSha256,createHash('sha256').update(readFileSync(resolve(root,'assets/Aileron-Regular.otf'))).digest('hex'));
assert.equal(data.glyphs.length,196);
assert.equal(curves.length,data.curveCount*8);
assert.equal(bands.length,data.bandCount*2);
assert.equal(indices.length,data.indexCount);
assert.ok(curves.every(Number.isFinite));
function curve(id) { const q=curves.slice(id*8,id*8+6); return [[q[0],q[1]],[q[2],q[3]],[q[4],q[5]]]; }
for(const g of data.glyphs) {
  for(const [axis,header,count] of [[1,g.bands[0],g.bands[1]],[0,g.bands[2],g.bands[3]]]) {
    for(let b=0;b<count;b++) {
      const start=bands[(header+b)*2], n=bands[(header+b)*2+1];
      assert.ok(start>=0&&start+n<=indices.length);
      let previous=Infinity;
      for(const id of indices.slice(start,start+n)) {
        assert.ok(id>=g.firstCurve&&id<g.firstCurve+g.curveCount);
        const q=curve(id), max=Math.max(...q.map(p=>p[1-axis]));
        assert.ok(max<=previous); previous=max;
        assert.ok(!q.every(p=>p[axis]===q[0][axis]));
      }
      // Every curve whose conservative extent intersects a band must be listed.
      const lo=g.bounds[axis]+(g.bounds[axis+2]-g.bounds[axis])*b/count-data.bandOverlapEm;
      const hi=g.bounds[axis]+(g.bounds[axis+2]-g.bounds[axis])*(b+1)/count+data.bandOverlapEm;
      const present=new Set(indices.slice(start,start+n));
      for(let id=g.firstCurve;id<g.firstCurve+g.curveCount;id++) {
        const q=curve(id), a=q.map(p=>p[axis]);
        if(!a.every(v=>v===a[0]) && Math.min(...a)<=hi && Math.max(...a)>=lo) assert.ok(present.has(id));
      }
    }
  }
}
// Independent f64 ray/curve intersection using t in [0,1). This does not use
// Slug's sign-bit table, confidence weights, band data, or early exit.
function crossings(q,p,axis) {
  const v=q.map(v=>v[axis]-p[axis]);
  const a=v[0]-2*v[1]+v[2], b=2*(v[1]-v[0]), c=v[0];
  let ts=[];
  if(Math.abs(a)<1e-16) { if(b!==0) ts=[-c/b]; }
  else { const d=b*b-4*a*c; if(d>0) ts=[(-b-Math.sqrt(d))/(2*a),(-b+Math.sqrt(d))/(2*a)]; }
  return ts.filter(t=>t>=0&&t<1).map(t=>({
    distance:(1-t)**2*q[0][1-axis]+2*t*(1-t)*q[1][1-axis]+t*t*q[2][1-axis]-p[1-axis],
    direction:Math.sign(2*a*t+b)
  }));
}
let seed=17;
const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
const fontSamples=[], fontExpected=[];
for(const g of data.glyphs) {
  if(!g.curveCount) {fontSamples.push(0,0,1/131072,g.unicode);fontExpected.push(0);continue;}
  const qs=Array.from({length:g.curveCount},(_,i)=>curve(g.firstCurve+i));
  let accepted=0;
  for(let attempt=0;attempt<150&&accepted<24;attempt++) {
    const x=Math.fround(g.bounds[0]+(g.bounds[2]-g.bounds[0])*(random()*1.4-.2));
    const y=Math.fround(g.bounds[1]+(g.bounds[3]-g.bounds[1])*(random()*1.4-.2));
    const h=qs.flatMap(q=>crossings(q,[x,y],1)), v=qs.flatMap(q=>crossings(q,[x,y],0));
    // Keep this binary winding test well away from the antialias fringe.
    if([...h,...v].some(r=>Math.abs(r.distance)<1e-4)) continue;
    const winding=h.filter(r=>r.distance>0).reduce((a,r)=>a+r.direction,0);
    fontSamples.push(x,y,1/131072,g.unicode);fontExpected.push(winding===0?0:1);accepted++;
  }
  assert.equal(accepted,24,`sampling glyph ${g.unicode}`);
}
function encode(values,integer=false) {
  const b=Buffer.alloc(values.length*4);
  values.forEach((v,i)=>integer?b.writeInt32LE(v,i*4):b.writeFloatLE(v,i*4));return b;
}
function flatten(result) {return Array.isArray(result)?result.flat(Infinity):result.backing_buffer.flat(Infinity);}
function close(actual,expected,label,tol=2e-4) {
  assert.equal(actual.length,expected.length,`${label} length`);
  actual.forEach((v,i)=>assert.ok(Number.isFinite(v)&&Math.abs(v-expected[i])<=tol,`${label}[${i}]: ${v} vs ${expected[i]}`));
}
function polygon(points) {return points.map((p,i)=>[p,points[(i+1)%points.length],points[(i+1)%points.length]]);}
const square=[[0,0],[1,0],[1,1],[0,1]];
function shapeBuffers(qs) {
  const c=qs.flatMap(q=>[...q.flat(),0,0]), ix=[];
  const h=qs.map((_,i)=>i).filter(i=>!qs[i].every(p=>p[1]===qs[i][0][1])).sort((a,b)=>Math.max(...qs[b].map(p=>p[0]))-Math.max(...qs[a].map(p=>p[0])));
  const v=qs.map((_,i)=>i).filter(i=>!qs[i].every(p=>p[0]===qs[i][0][0])).sort((a,b)=>Math.max(...qs[b].map(p=>p[1]))-Math.max(...qs[a].map(p=>p[1])));
  ix.push(...h,...v);
  return {curves:[c,false],bands:[[0,h.length,h.length,v.length],true],indices:[ix,true],metadata:[[0,0,1,1,0,0,0,0,0,1,1,1],false]};
}
let cases=0;
for(const [target,ext] of [['wgsl','wgsl'],['spirv','spv']]) {
  const dir=resolve(out,target);mkdirSync(dir);
  const shader=resolve(dir,`slug.${ext}`);
  run(wyn,['build',resolve(root,'test/slug.wyn'),'--max-warnings','0','-t',target,'-o',shader]);
  function gpu(entry,inputs) {
    const args=['pipeline',shader,'--entry',entry,'--headless'];
    for(const [name,[values,integer]] of Object.entries(inputs)) {
      const path=resolve(dir,`${entry}-${name}.bin`);writeFileSync(path,encode(values,integer));
      args.push('--input',`${name}:${path}`);
    }
    const output=resolve(dir,`${entry}.json`);args.push('--output',`${entry}:${output}`);
    run(viz,args);cases++;return flatten(JSON.parse(readFileSync(output)));
  }
  const rootSamples=[];
  for(const magnitude of [1,0]) for(let i=0;i<8;i++)
    rootSamples.push((i&1)?-magnitude:magnitude,(i&2)?-magnitude:magnitude,(i&4)?-magnitude:magnitude,0);
  close(gpu('roots',{samples:[rootSamples,false]}),[0,256,257,256,1,257,1,0,0,256,257,256,1,257,1,0],`${target} root eligibility including signed zero`,0);
  const samples=[], expected=[];
  for(const delta of [-.02,-.005,-.0025,0,.0025,.005,.02]) {
    const alpha=Math.min(1,Math.max(0,delta/.01+.5));
    for(const [x,y] of [[delta,.5],[1-delta,.5],[.5,delta],[.5,1-delta]]) {
      samples.push(x,y,.01,.01);expected.push(alpha);
    }
  }
  for(const orientation of [square,[...square].reverse()])
    close(gpu('shape',{...shapeBuffers(polygon(orientation)),samples:[samples,false]}),expected,`${target} square AA/winding reversal`);
  const hole=polygon(square).concat(polygon([[.25,.25],[.25,.75],[.75,.75],[.75,.25]]));
  close(gpu('shape',{...shapeBuffers(hole),samples:[[.1,.5,.01,.01,.5,.5,.01,.01,.9,.5,.01,.01,.25,.5,.01,.01],false]}),[1,0,1,.5],`${target} nonzero hole`);
  const lens=[[[-1,0],[0,-1],[1,0]],[[1,0],[0,1],[-1,0]]];
  // At the exact tangent, the reference's horizontal double-root confidence
  // combines zero horizontal coverage with 0.5 vertical coverage -> 0.25.
  // Slug's two-ray filter is not an exact area box filter at this degeneracy.
  close(gpu('shape',{...shapeBuffers(lens),samples:[[0,0,.01,.01,0,-.5,.01,.01,0,-.52,.01,.01,0,-.48,.01,.01],false]}),[1,.25,0,1],`${target} curved edge and double roots`);
  close(gpu('font',{curves:[curves,false],bands:[bands,true],indices:[indices,true],samples:[fontSamples,false]}),fontExpected,
    `${target} all-glyph independent winding reference`,1e-3);
  for(const m of [
    [[.2,0,0,-.2],[0,-.3,0,.1],[0,0,0,0],[0,0,0,1]],
    [[.2,.07,0,-.2],[.04,-.3,0,.1],[0,0,0,0],[.05,-.1,0,1]]]) {
    const s=[.2,.3,1,0,.2,.3,-1,0,.2,.3,0,1,.2,.3,0,-1];
    const result=gpu('dilation',{samples:[s,false],matrix:[m.flat(),false]});
    const project=p=>[400*(m[0][0]*p[0]+m[0][1]*p[1]+m[0][3])/(m[3][0]*p[0]+m[3][1]*p[1]+m[3][3]),
                       300*(m[1][0]*p[0]+m[1][1]*p[1]+m[1][3])/(m[3][0]*p[0]+m[3][1]*p[1]+m[3][3])];
    const before=project([.2,.3]);
    for(let i=0;i<result.length;i+=4) {
      const after=project(result.slice(i,i+2));
      assert.ok(Math.abs(Math.hypot(after[0]-before[0],after[1]-before[1])-.5)<1e-4,`${target} half-pixel projective dilation`);
      close(result.slice(i+2,i+4),[after[0]/400,after[1]/300],`${target} consistent expanded coordinates`);
    }
  }
  console.log(`${target}: Slug synthetic and ${fontExpected.length} font samples passed.`);
}
console.log(`PASS: Slug buffers, ${cases} GPU cases, both shader targets. Results: ${out}`);
