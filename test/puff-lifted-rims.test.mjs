// A hole drawn out of the outline's plane is an eyelet at its own height and
// angle. puffLiftRims gives each rim vertex the height of its drawn loop and
// extends it harmonically over both sheets, so the cage rim sits at the drawn
// height (times the limit gain), the limit rim lands on the drawn curve, and
// no vertex leaves the range of the rim heights.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildAnnularPuff, buildMultiHolePuff, puffInvariants, puffLiftRims } from '../kernel/puff.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';

const circle = (cx, cy, r, n) => Array.from({ length: n }, (_, i) => [cx + r * Math.cos((i / n) * Math.PI * 2), cy + r * Math.sin((i / n) * Math.PI * 2)]);
const outer = circle(0, 0, 100, 40);
const hole = circle(40, 0, 20, 16);
const flat = (L) => L.map(() => 0);
// A hole tilted about its own y axis: z = tan(tilt) · (x - cx).
const tilted = (L, cx, deg) => L.map((p) => Math.tan((deg * Math.PI) / 180) * (p[0] - cx));

function rimVertices(p, loops, tol = 1e-6) {
  // Rim vertices are the ones the lift fixed, recovered from the field
  // itself: a vertex whose z equals its lift / 0.924.
  const out = [];
  for (let i = 0; i < p.nv; i++) {
    const x = p.positions[3 * i], y = p.positions[3 * i + 1], z = p.positions[3 * i + 2];
    if (Math.abs(z - p.lift[i] / 0.924) < tol) out.push(i);
  }
  return out;
}
const limitZ = (p, i, edges, faces) => {
  const n = edges[i].size; let mz = 0, cz = 0;
  for (const j of edges[i]) mz += (p.positions[3 * i + 2] + p.positions[3 * j + 2]) / 2;
  for (const f of faces[i]) cz += (p.positions[3 * p.quads[f] + 2] + p.positions[3 * p.quads[f + 1] + 2] + p.positions[3 * p.quads[f + 2] + 2] + p.positions[3 * p.quads[f + 3] + 2]) / 4;
  return (n * n * p.positions[3 * i + 2] + 4 * mz + cz) / (n * (n + 5));
};
function graph(p) {
  const edges = Array.from({ length: p.nv }, () => new Set()), faces = Array.from({ length: p.nv }, () => []);
  for (let f = 0; f + 3 < p.quads.length; f += 4) {
    const a = p.quads[f], b = p.quads[f + 1], c = p.quads[f + 2], d = p.quads[f + 3];
    edges[a].add(b); edges[a].add(d); edges[b].add(a); edges[b].add(c); edges[c].add(b); edges[c].add(d); edges[d].add(c); edges[d].add(a);
    faces[a].push(f); faces[b].push(f); faces[c].push(f); faces[d].push(f);
  }
  return { edges, faces };
}
const drawnZ = (loops, lifts, x, y) => {
  let bd = Infinity, bh = 0;
  loops.forEach((L, k) => { const m = L.length; for (let i = 0; i < m; i++) { const j = (i + 1) % m, ax = L[i][0], ay = L[i][1], dx = L[j][0] - ax, dy = L[j][1] - ay, L2 = dx * dx + dy * dy; let t = L2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / L2 : 0; t = Math.max(0, Math.min(1, t)); const d = Math.hypot(ax + dx * t - x, ay + dy * t - y); if (d < bd) { bd = d; bh = lifts[k][i] * (1 - t) + lifts[k][j] * t; } } });
  return bh;
};

test('puffLiftRims: a flat lift list leaves the cage byte-identical', () => {
  const a = buildAnnularPuff(outer, [hole], { density: 2 });
  const b = buildAnnularPuff(outer, [hole], { density: 2, lifts: [flat(outer), flat(hole)] });
  assert.ok(a.ok && b.ok);
  assert.deepEqual(Array.from(b.positions), Array.from(a.positions));
  assert.equal(b.lifted, undefined);
});

test('puffLiftRims: a hole tilted 30° — the rim sits at the drawn height, the field stays within the rim range, the shell stays closed', () => {
  const lifts = [flat(outer), tilted(hole, 40, 30)];
  const p = buildAnnularPuff(outer, [hole], { density: 2, lifts });
  assert.ok(p.ok && p.lifted, 'built and lifted');
  const inv = puffInvariants(p);
  assert.ok(inv.ok && inv.euler === 0, `closed genus-1 (${JSON.stringify(inv)})`);
  const rims = rimVertices(p, [outer, hole]);
  assert.ok(rims.length >= 16, `rim vertices found (${rims.length})`);
  let worstRim = 0, hiLift = Math.max(...lifts[1]), loLift = Math.min(...lifts[1]);
  for (const i of rims) {
    const want = drawnZ([outer, hole], lifts, p.positions[3 * i], p.positions[3 * i + 1]);
    worstRim = Math.max(worstRim, Math.abs(p.lift[i] - want));
  }
  assert.ok(worstRim < 1e-9, `every rim vertex carries its loop's drawn height (worst miss ${worstRim.toExponential(2)})`);
  for (let i = 0; i < p.nv; i++) assert.ok(p.lift[i] >= loLift - 1e-9 && p.lift[i] <= hiLift + 1e-9, `vertex ${i} lift ${p.lift[i]} inside [${loLift}, ${hiLift}]`);
  // The limit rim: the Catmull-Clark limit of each rim vertex lands on the drawn height.
  const { edges, faces } = graph(p);
  let worstLimit = 0;
  for (const i of rims) {
    const want = drawnZ([outer, hole], lifts, p.positions[3 * i], p.positions[3 * i + 1]);
    worstLimit = Math.max(worstLimit, Math.abs(limitZ(p, i, edges, faces) - want));
  }
  const tiltRange = hiLift - loLift;
  assert.ok(worstLimit < 0.06 * tiltRange, `the limit rim is within 6 % of the drawn range of the drawn height (worst ${worstLimit.toFixed(2)} of ${tiltRange.toFixed(1)})`);
  // The mirror through the rim stays exact: front and back twins carry the same lift.
  const twin = new Map();
  for (let i = 0; i < p.nv; i++) { const k = `${p.positions[3 * i].toFixed(6)},${p.positions[3 * i + 1].toFixed(6)}`; if (!twin.has(k)) twin.set(k, []); twin.get(k).push(i); }
  let mirrorMiss = 0;
  for (const ids of twin.values()) if (ids.length === 2) mirrorMiss = Math.max(mirrorMiss, Math.abs(p.lift[ids[0]] - p.lift[ids[1]]));
  assert.ok(mirrorMiss < 1e-9, `front and back sheets share one lift (${mirrorMiss.toExponential(2)})`);
  assert.doesNotThrow(() => subdivideCatmullClark({ vertices: Array.from({ length: p.nv }, (_, i) => [p.positions[3 * i], p.positions[3 * i + 1], p.positions[3 * i + 2]]), faces: Array.from({ length: p.quads.length / 4 }, (_, f) => [p.quads[4 * f], p.quads[4 * f + 1], p.quads[4 * f + 2], p.quads[4 * f + 3]]) }));
});

test('puffLiftRims: a raised hole is a lifted grommet — its rim at the drawn height, the outer rim on the plane, nothing above the grommet', () => {
  const lifts = [flat(outer), hole.map(() => 12)];
  const p = buildAnnularPuff(outer, [hole], { density: 3, lifts });
  assert.ok(p.ok && p.lifted);
  assert.ok(Math.abs(p.liftPeak - 12) < 1e-9, `peak lift is the grommet's (${p.liftPeak})`);
  for (let i = 0; i < p.nv; i++) assert.ok(p.lift[i] >= -1e-9 && p.lift[i] <= 12 + 1e-9);
  const rims = rimVertices(p, [outer, hole]);
  const onOuter = rims.filter((i) => Math.hypot(p.positions[3 * i], p.positions[3 * i + 1]) > 60);
  const onHole = rims.filter((i) => Math.hypot(p.positions[3 * i] - 40, p.positions[3 * i + 1]) < 30);
  assert.ok(onOuter.length && onHole.length);
  assert.ok(onOuter.every((i) => Math.abs(p.lift[i]) < 1e-9), 'the outer rim stays on the plane');
  assert.ok(onHole.every((i) => Math.abs(p.lift[i] - 12) < 1e-9), 'the hole rim sits at 12');
  assert.ok(puffInvariants(p).ok);
});

test('puffLiftRims: two holes through the multi-hole builder, one raised, one tilted; and a saddle outline with a hole follows its stroke', () => {
  const h1 = circle(-45, 0, 18, 16), h2 = circle(45, 0, 18, 16);
  const lifts = [flat(outer), h1.map(() => 8), tilted(h2, 45, 25)];
  const p = buildMultiHolePuff(outer, [h1, h2], { density: 2, lifts });
  assert.ok(p.ok && p.lifted, `built (${p.reason || ''})`);
  const inv = puffInvariants(p);
  assert.ok(inv.ok && inv.euler === -2, `closed genus-2 (${JSON.stringify(inv)})`);
  const rims = rimVertices(p, [outer, h1, h2]);
  let worst = 0;
  for (const i of rims) worst = Math.max(worst, Math.abs(p.lift[i] - drawnZ([outer, h1, h2], lifts, p.positions[3 * i], p.positions[3 * i + 1])));
  assert.ok(worst < 1e-9, `every rim vertex carries its own loop's height (${worst.toExponential(2)})`);
  const lo = Math.min(...lifts.flat()), hi = Math.max(...lifts.flat());
  for (let i = 0; i < p.nv; i++) assert.ok(p.lift[i] >= lo - 1e-9 && p.lift[i] <= hi + 1e-9);
  // A saddle outline (z = 15 cos 2θ) with a flat hole: the outline is followed.
  const saddle = outer.map((q) => 15 * Math.cos(2 * Math.atan2(q[1], q[0])));
  const s = buildAnnularPuff(outer, [hole], { density: 2, lifts: [saddle, flat(hole)] });
  assert.ok(s.ok && s.lifted);
  const sr = rimVertices(s, [outer, hole]).filter((i) => Math.hypot(s.positions[3 * i], s.positions[3 * i + 1]) > 60);
  let worstS = 0;
  for (const i of sr) worstS = Math.max(worstS, Math.abs(s.lift[i] - drawnZ([outer, hole], [saddle, flat(hole)], s.positions[3 * i], s.positions[3 * i + 1])));
  assert.ok(worstS < 1e-9 && sr.length >= 8, `the saddle outline's rim carries its stroke (${sr.length} rim vertices, worst ${worstS.toExponential(2)})`);
});

test('puffLiftRims: called directly on a cage with no rim does nothing', () => {
  const p = { nv: 0, positions: new Float32Array(0), quads: new Uint32Array(0) };
  assert.equal(puffLiftRims(p, [], []), p);
});
