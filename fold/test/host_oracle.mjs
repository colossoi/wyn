// Upstream host-lifecycle oracle. Constraint solves are bypassed in both hosts
// to isolate feeding, moving walls, damping, grab smoothing, and retirement.
import {ClothSimulation} from '../upstream/cloth-simulation.js';
const input=JSON.parse(process.argv[2]);
const sim=new ClothSimulation();
sim.setViewportWalls(input.walls.flat());
sim.targetRetainedRows=input.retained;
sim.reset({tipHeight:0.175,speed:2});
const walls=sim.solveWalls;
sim.solveEdges=()=>{};sim.solveGrab=()=>{};sim.solveContacts=()=>{};sim.solveFloor=()=>{};
sim.solveWalls=function(count,thickness,moving=false) {if(moving)walls.call(this,count,thickness,true);};
for(let i=0;i<90;i++) {
 if(i===10) {sim.beginGrab(10*13+6);sim.moveGrab([0.2,3.5,0.7]);}
 if(i===40)sim.endGrab();
 if(i===50) {sim.setViewportWalls(input.resized.flat());sim.targetRetainedRows=input.resizedRetained;}
 sim.step(1/120,2,0.35,0.5);
}
console.log(JSON.stringify({positions:Array.from(sim.positions),velocities:Array.from(sim.velocities),birth:Array.from(sim.birth),edges:Array.from(new Float32Array(sim.edges)),active:sim.activeRows,feed:sim.feed,time:sim.time,retained:sim.retainedRows,emitted:sim.emittedRows}));
