import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createXPBDCage,stepXPBDCage,xpbdSignedVolume} from '../kernel/xpbd.mjs';
import {dynamicBondFrame,dynamicBondLocal,dynamicBondWorld,createDynamicBondSimulation,advanceDynamicBond,createHostContact} from '../kernel/dynamicbond.mjs';
const cage={vertices:[[0,0,0],[20,0,0],[20,20,0],[0,20,0],[0,0,20],[20,0,20],[20,20,20],[0,20,20]],faces:[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]};
const frame=dynamicBondFrame([0,0,0],[1,0,0],[0,1,0]);
const state=pinned=>createXPBDCage({positions:cage.vertices.flat(),faces:cage.faces,pinned});
const near=(a,b,t=1e-8)=>assert.ok(Math.abs(a-b)<t,`${a} vs ${b}`);
test('rest cage stays exactly at rest without gravity',()=>{const s=state([0]);for(let i=0;i<240;i++)stepXPBDCage(s,1/240,{preserveVolume:true});assert.deepEqual([...s.x],cage.vertices.flat());});
test('gravity moves free vertices while pin targets stay exact',()=>{const s=state([0,1]);const targets=s.x.slice();for(let i=0;i<120;i++)stepXPBDCage(s,1/240,{gravity:[0,0,-1000],targets});assert.deepEqual([...s.x.slice(0,6)],[...targets.slice(0,6)]);assert.ok(s.x[8]<0);assert.ok([...s.x].every(Number.isFinite));});
test('all-pinned and coincident cages never divide by zero',()=>{const s=state([0,1,2,3,4,5,6,7]);stepXPBDCage(s,1/240,{compliance:0,preserveVolume:true});assert.deepEqual([...s.x],cage.vertices.flat());const flat=createXPBDCage({positions:[0,0,0,0,0,0,0,0,0],faces:[[0,1,2]]});stepXPBDCage(flat,1/240,{compliance:0});assert.ok([...flat.x].every(Number.isFinite));});
test('reject invalid timestep, topology, targets and mass',()=>{assert.throws(()=>stepXPBDCage(state([]),0));assert.throws(()=>createXPBDCage({positions:[0,0,0,1,1,1],faces:[[0,1,5]]}));assert.throws(()=>createXPBDCage({positions:cage.vertices.flat(),faces:cage.faces,mass:0}));const s=state([0]);const t=s.x.slice();t[0]=NaN;assert.throws(()=>stepXPBDCage(s,1/240,{targets:t}));});
test('failed numerical step restores both positions and velocity',()=>{const s=state([]),x=[...s.x],v=[...s.velocity];assert.throws(()=>stepXPBDCage(s,1/240,{gravity:[1e308,0,0],preserveVolume:true}));assert.deepEqual([...s.x],x);assert.deepEqual([...s.velocity],v);assert.equal(s.time,0);});
test('signed volume distinguishes reflection and survives large translations',()=>{const s=state([]);assert.equal(s.closed,true);const reflected=s.x.map((v,i)=>i%3===2?-v:v);near(xpbdSignedVolume(reflected,s.triangles),-s.volume);const moved=s.x.map(v=>v+1e8);near(xpbdSignedVolume(moved,s.triangles),s.volume);});
test('open sheet has no volume constraint',()=>{const s=createXPBDCage({positions:cage.vertices.slice(0,4).flat(),faces:[[0,1,2,3]],pinned:[0]});assert.equal(s.closed,false);stepXPBDCage(s,1/240,{preserveVolume:true,gravity:[0,0,-1000]});assert.ok([...s.x].every(Number.isFinite));});
test('anchor frame is orthonormal despite skewed/unequal parameter tangents',()=>{const f=dynamicBondFrame([4,5,6],[20,0,0],[3,2,0]);assert.deepEqual(f.axes,[[1,0,0],[0,1,0],[0,0,1]]);assert.deepEqual(dynamicBondWorld(dynamicBondLocal([8,9,10],f),f),[8,9,10]);assert.throws(()=>dynamicBondFrame([0,0,0],[0,0,0],[0,1,0]));assert.throws(()=>dynamicBondFrame([0,0,0],[1,0,0],[2,0,0]));});
test('static-host trajectories agree at 30/60/120 Hz render cadence',()=>{const run=hz=>{const s=createDynamicBondSimulation({cage,frame,pinned:[0]});for(let i=0;i<hz;i++)advanceDynamicBond(s,1/hz,frame,{gravity:[0,0,-1000]});return s;};const a=run(30),b=run(60),c=run(120);near(a.solver.time,1);for(let i=0;i<a.solver.x.length;i++){near(a.solver.x[i],b.solver.x[i]);near(a.solver.x[i],c.solver.x[i]);}});
test('moving pins interpolate consistently across presentation rates',()=>{const run=hz=>{const s=createDynamicBondSimulation({cage,frame,pinned:[0]});for(let i=0;i<hz;i++)advanceDynamicBond(s,1/hz,dynamicBondFrame([20*(i+1)/hz,0,0],[1,0,0],[0,1,0]));return s.solver.x;};const a=run(30),b=run(120);for(let i=0;i<a.length;i++)near(a[i],b[i],1e-7);near(a[0],20);});
test('a stationary face-ring settle may use a coarser fixed step without changing the form materially',()=>{const run=step=>{const s=createDynamicBondSimulation({cage,frame,pinned:[0,1,2,3],step});for(let t=0;t<3;t+=1/60)advanceDynamicBond(s,1/60,frame,{gravity:[0,0,-9810],compliance:.00009,damping:2});return s.solver.x;};const fine=run(1/240),coarse=run(1/120);for(let i=0;i<fine.length;i++)near(coarse[i],fine[i],.1);});
test('a simulation step must be finite and within the solver stability limit',()=>{for(const step of [0,-1,Infinity,1/20])assert.throws(()=>createDynamicBondSimulation({cage,frame,pinned:[0],step}));});
test('long background gap does not catch up a large impulse',()=>{const s=createDynamicBondSimulation({cage,frame,pinned:[0]});assert.equal(advanceDynamicBond(s,10,frame),0);assert.equal(s.solver.time,0);});
test('rigid frame changes preserve offsets and do not introduce shear',()=>{const f=dynamicBondFrame([10,20,30],[0,2,0],[-5,1,0]);const p=dynamicBondWorld([2,3,4],f);near(p[0],7);near(p[1],22);near(p[2],34);assert.deepEqual(dynamicBondLocal(p,f),[2,3,4]);});

// A flat host at z=0 facing +z; the count says how often the tree was asked.
const plane=()=>{const calls={n:0};const closest=(p,max)=>{calls.n++;return Math.abs(p[2])<=max?{point:[p[0],p[1],0],normal:[0,0,1]}:null;};return {closest,calls};};
const lifted=dz=>({vertices:cage.vertices.map(([x,y,z])=>[x,y,z+dz]),faces:cage.faces});
test('a free cage dropped on its host comes to rest on it, never through it',()=>{
 const c=lifted(10),s=createXPBDCage({positions:c.vertices.flat(),faces:c.faces,pinned:[]});
 const {closest}=plane();const hc=createHostContact({closest,cage:c,margin:0.3,reach:40});
 assert.equal(hc.side,1);
 for(let i=0;i<480;i++)stepXPBDCage(s,1/240,{gravity:[0,0,-1000],contact:hc.contact});
 let lo=Infinity;for(let i=2;i<s.x.length;i+=3)lo=Math.min(lo,s.x[i]);
 assert.ok(lo>=0.3-1e-9,`lowest vertex ${lo} crossed the host`);assert.ok(lo<0.3+0.5,`lowest vertex ${lo} never landed`);
 near(hc.clearance(s.x),lo,1e-9);
});
test('a cage that starts under the host stays under it',()=>{
 const c=lifted(-30),s=createXPBDCage({positions:c.vertices.flat(),faces:c.faces,pinned:[]});
 const hc=createHostContact({closest:plane().closest,cage:c,margin:0.3,reach:40});
 assert.equal(hc.side,-1);
 for(let i=0;i<480;i++)stepXPBDCage(s,1/240,{gravity:[0,0,1000],contact:hc.contact});
 for(let i=2;i<s.x.length;i+=3)assert.ok(s.x[i]<=-0.3+1e-9,`vertex z ${s.x[i]} crossed upward`);
});
test('a vertex hanging in the air asks the tree nothing until it could reach the host',()=>{
 const c=lifted(1000),s=createXPBDCage({positions:c.vertices.flat(),faces:c.faces,pinned:[]});
 const {closest,calls}=plane();const hc=createHostContact({closest,cage:c,margin:0.3,reach:50});
 const afterBuild=calls.n;
 stepXPBDCage(s,1/240,{contact:hc.contact});const afterFirst=calls.n;
 assert.equal(afterFirst-afterBuild,8);
 for(let i=0;i<100;i++)stepXPBDCage(s,1/240,{contact:hc.contact});
 assert.equal(calls.n,afterFirst);
});
test('a pinned vertex is never pushed by contact and a bound cage settles onto its host',()=>{
 const c=lifted(-5);const hc=createHostContact({closest:plane().closest,cage:c,pinOffset:[0,0,1],margin:0.3,reach:40});
 assert.equal(hc.side,1);
 const sim=createDynamicBondSimulation({cage:c,frame,pinned:[4]});
 for(let i=0;i<120;i++)advanceDynamicBond(sim,1/60,frame,{gravity:[0,0,-1000],contact:hc.contact});
 near(sim.solver.x[14],15,1e-9);
 for(let i=0;i<8;i++)if(i!==4)assert.ok(sim.solver.x[3*i+2]>=0.3-1e-9,`vertex ${i} at z ${sim.solver.x[3*i+2]}`);
});
