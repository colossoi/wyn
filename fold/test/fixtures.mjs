// Generate upstream numerical oracles without a browser or GPU.
import {mkdirSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {ClothSimulation} from '../upstream/cloth-simulation.js';

const directory = process.argv[2];
if (!directory) throw new Error('Expected output directory');
mkdirSync(directory, {recursive:true});
for (const name of ['hanging','contact-grab-walls','retirement']) {
  const sim = new ClothSimulation();
  sim.activeRows = 12;
  sim.retainedRows = name === 'retirement' ? 14 : 191;
  sim.feed = 0.13;
  sim.weights.fill(1);
  sim.weights.fill(0,0,26);
  const count = sim.activeRows*sim.stride;
  for (let i=0;i<count;i++) {
    const row=Math.floor(i/13), col=i%13;
    sim.positions.set([
      col*sim.spacing-1.9,
      name==='hanging' ? 3-row*sim.spacing : 0.12+(row%4)*0.14,
      name==='hanging' ? 0.01*Math.sin(row+col) : 0.04*Math.floor(row/4)+0.01*Math.sin(col)
    ],i*3);
  }
  sim.previous.set(sim.positions);
  if (name!=='hanging') {
    sim.setViewportWalls(new Float64Array([0,0,1,0.18,0,0,-1,0.18,0,1,0,0]));
    sim.beginGrab(6*13+6);
    sim.moveGrab([0.2,0.7,0.1]);
  }
  const dt=1/120, thickness=0.35, stiffness=0.5;
  for(let row=0;row<sim.activeRows;row++) sim.rowVisibility[row]=sim.visibilityAt(row);
  sim.prepareGrab(dt,thickness);
  const p=new Float32Array(2483*4), old=new Float32Array(2483*4);
  for(let i=0;i<sim.count;i++) {
    p.set(sim.positions.subarray(i*3,i*3+3),i*4);p[i*4+3]=sim.weights[i];
    old.set(sim.previous.subarray(i*3,i*3+3),i*4);
  }
  const parameters=new Float32Array(44), grabs=new Float32Array(36);
  parameters.set(sim.wallPlanes);
  parameters.set([count,dt,thickness,stiffness,sim.feed,sim.retainedRows,sim.wallCount,0],12);
  if(sim.grab) {
    parameters.set(sim.grab.anchor,20);parameters[23]=sim.grab.particles.length;
    sim.grab.particles.forEach((particle,i)=>{
      grabs.set(particle.offset,i*4);
      parameters.set([particle.index,particle.compliance],24+i*2);
    });
  }
  // Round transported state to the shader's input precision. Upstream then
  // evaluates in binary64; comparisons allow the shader's binary32 arithmetic.
  for(let i=0;i<count;i++) for(let axis=0;axis<3;axis++) {
    sim.positions[i*3+axis]=p[i*4+axis];sim.previous[i*3+axis]=old[i*4+axis];
  }
  sim.edges=Float64Array.from(new Float32Array(sim.edges));
  if(sim.grab) sim.grab.particles.forEach((particle,i)=>{
    particle.offset=Array.from(grabs.subarray(i*4,i*4+3));
    particle.compliance=parameters[24+i*2+1];
  });
  if(sim.grab) sim.grab.anchor=Array.from(parameters.subarray(20,23));
  for(let iteration=0;iteration<4;iteration++) {
    sim.solveEdges(count,parameters[13],0.00002*10**((1-parameters[15])*3),iteration%2===1);
    sim.solveGrab(parameters[13]);
    if(iteration%2===1) sim.solveContacts(count,parameters[14]);
    sim.solveFloor(count,parameters[14]);sim.solveWalls(count,parameters[14]);
  }
  const path=join(directory,name);mkdirSync(path,{recursive:true});
  for(const [key,value] of Object.entries({input_p:p,old,edges:new Float32Array(sim.edges),parameters,grabs}))
    writeFileSync(join(path,key+'.bin'),Buffer.from(value.buffer));
  writeFileSync(join(path,'expected.json'),JSON.stringify({count,positions:Array.from(sim.positions.slice(0,count*3))}));
}
