// An annular puff is a closed tube around a hole, and the cage is the claim.
//
// A puff with a hole is genus 1 — Euler V-E+F = 0, not 2 — and it must still be all quads,
// watertight, and consistently wound, or it cannot be subdivided, shelled or exported and nothing
// names the failure. Every vertex must sit inside the outer rim and outside the hole; both rims sit
// at z = 0 so the top meets its mirror there; and the classifier must tell an annulus from two
// separate shapes and from a shape-inside-a-hole. See kernel/puff.mjs buildAnnularPuff /
// classifyPuffLoops.
import { strict as assert } from 'node:assert';
import { buildAnnularPuff, classifyPuffLoops, offsetLoop } from '../kernel/puff.mjs';

const circle = (r, n = 64, cx = 0, cy = 0) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; p.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } return p; };
const pointInPoly = (P, x, y) => { let c = false; for (let i = 0, j = P.length - 1; i < P.length; j = i++) { const xi = P[i][0], yi = P[i][1], xj = P[j][0], yj = P[j][1]; if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) c = !c; } return c; };

// classifyPuffLoops
assert.equal(classifyPuffLoops([circle(5)]).kind, 'single', 'one loop is single');

const anC = classifyPuffLoops([circle(10), circle(4)]);
assert.ok(anC.ok && anC.kind === 'annular' && anC.holes.length === 1, 'outer + one inner is annular');

const an2 = classifyPuffLoops([circle(20), circle(3, 40, -8, 0), circle(3, 40, 8, 0)]);
assert.ok(an2.ok && an2.kind === 'annular' && an2.holes.length === 2, 'outer + two disjoint inner is annular with 2 holes');

assert.equal(classifyPuffLoops([circle(3, 64, -20, 0), circle(3, 64, 20, 0)]).ok, false, 'two side-by-side shapes are refused');
assert.equal(classifyPuffLoops([circle(3, 64, -20, 0), circle(3, 64, 20, 0)]).kind, 'multi', '...as multi-region');

const island = classifyPuffLoops([circle(20), circle(10), circle(4)]); // 4 inside 10 inside 20
assert.equal(island.ok, false, 'a loop inside a hole (island) is refused');
assert.equal(island.kind, 'island', '...as island');

const overlap = classifyPuffLoops([circle(20), circle(6, 64, -3, 0), circle(6, 64, 3, 0)]); // two holes that intersect
assert.equal(overlap.ok, false, 'overlapping holes are refused');

// buildAnnularPuff: the ring cage
const outer = circle(10, 64), hole = circle(4, 48);
const r = buildAnnularPuff(outer, [hole], { count: 64, bands: 6 });
assert.ok(r.ok, `a concentric ring must build: ${r.why || ''}`);

// quads only, four distinct corners
assert.equal(r.quads.length % 4, 0, 'the index buffer is whole quads');
for (let q = 0; q < r.quads.length; q += 4) {
  const s = new Set([r.quads[q], r.quads[q + 1], r.quads[q + 2], r.quads[q + 3]]);
  assert.equal(s.size, 4, `quad ${q / 4} has a repeated corner — a degenerate quad`);
}

// manifold + orientation: every directed edge appears exactly once, and its reverse exactly once
const dir = new Map();
for (let q = 0; q < r.quads.length; q += 4) {
  const f = [r.quads[q], r.quads[q + 1], r.quads[q + 2], r.quads[q + 3]];
  for (let k = 0; k < 4; k++) { const key = `${f[k]}_${f[(k + 1) % 4]}`; dir.set(key, (dir.get(key) || 0) + 1); }
}
let worstDir = 0, unpaired = 0;
const undirected = new Set();
for (const [key, n] of dir) {
  worstDir = Math.max(worstDir, n);
  const [a, b] = key.split('_');
  if (!dir.has(`${b}_${a}`)) unpaired++;
  undirected.add(+a < +b ? `${a}_${b}` : `${b}_${a}`);
}
assert.equal(worstDir, 1, 'every directed edge is used exactly once — consistent winding, no non-manifold edge');
assert.equal(unpaired, 0, 'every edge is shared by two oppositely-wound faces — watertight');

// Euler: genus 1 (a torus) is V - E + F = 0
const V = r.positions.length / 3, E = undirected.size, F = r.quads.length / 4;
assert.equal(V - E + F, 0, `annular puff must be closed genus 1 (V-E+F=0), got ${V - E + F} (V${V} E${E} F${F})`);

// both rims at z = 0; a real dome in between
let maxZ = 0, rimMax = 0;
for (let i = 0; i < 2 * r.N; i++) rimMax = Math.max(rimMax, Math.abs(r.positions[i * 3 + 2])); // first 2N verts are the two equators
for (let i = 0; i < V; i++) maxZ = Math.max(maxZ, Math.abs(r.positions[i * 3 + 2]));
assert.ok(rimMax < 1e-9, `both rims must sit at z = 0, worst ${rimMax}`);
assert.ok(maxZ > 0.1, `the ring must actually dome, peak |z| = ${maxZ.toFixed(3)}`);

// symmetric top/bottom (bottomScale default 1): for every +z vertex there is a −z twin at the same xy
let asym = 0;
const keyXY = (i) => `${r.positions[i * 3].toFixed(4)}_${r.positions[i * 3 + 1].toFixed(4)}`;
const byXY = new Map();
for (let i = 0; i < V; i++) { const k = keyXY(i); if (!byXY.has(k)) byXY.set(k, []); byXY.get(k).push(r.positions[i * 3 + 2]); }
for (const zs of byXY.values()) if (zs.length === 2 && Math.abs(zs[0] + zs[1]) > 1e-6) asym++;
assert.equal(asym, 0, 'top and bottom mirror across z = 0');

// containment: every vertex sits inside the outer and outside the hole (2D)
let outCount = 0, inHole = 0;
for (let i = 0; i < V; i++) { const x = r.positions[i * 3], y = r.positions[i * 3 + 1]; if (!pointInPoly(outer, x, y)) outCount++; if (pointInPoly(hole, x, y)) inHole++; }
// the rim vertices lie on the loops; allow a tiny count from on-boundary rounding
assert.ok(outCount <= r.N, `cage vertices stay inside the outer rim (${outCount} outside, rim is ${r.N})`);
assert.ok(inHole <= r.N, `cage vertices stay out of the hole (${inHole} inside, rim is ${r.N})`);

// Refusals
assert.equal(buildAnnularPuff(outer, []).ok, false, 'no hole is refused');
assert.equal(buildAnnularPuff(outer, [circle(4), circle(2, 32, 3, 0)]).ok, false, 'more than one hole is refused (buildMultiHolePuff handles that case)');
assert.equal(buildAnnularPuff(circle(4), [circle(6)]).ok, false, 'a hole larger than the outer is refused');

// offsetLoop: the Annular-toggle backend
const polyArea = (P) => { let a = 0; for (let i = 0; i < P.length; i++) { const p = P[i], q = P[(i + 1) % P.length]; a += p[0] * q[1] - q[0] * p[1]; } return Math.abs(a / 2); };

// inward offset of a circle is a smaller concentric circle
const inC = offsetLoop(circle(10, 64), 3);
assert.ok(inC.ok, `an inward offset of a circle must succeed: ${inC.why || ''}`);
{
  let rmin = Infinity, rmax = 0; for (const p of inC.loop) { const r = Math.hypot(p[0], p[1]); rmin = Math.min(rmin, r); rmax = Math.max(rmax, r); }
  assert.ok(rmax < 10 && rmin > 6 && (rmax - rmin) < 0.1, `inward offset is a concentric ~r7 ring (r ${rmin.toFixed(2)}..${rmax.toFixed(2)})`);
}
// a square offsets to a smaller square (area shrinks by the right order)
const square = [[-10, -10], [10, -10], [10, 10], [-10, 10]];
const inSq = offsetLoop(square, 2);
assert.ok(inSq.ok && Math.abs(polyArea(inSq.loop) - 16 * 16) < 1, `a square insets to 16x16, got area ${inSq.ok ? polyArea(inSq.loop).toFixed(1) : 'refused'}`);
// outward offset (dist < 0) grows the loop
const outC = offsetLoop(circle(5, 64), -2);
assert.ok(outC.ok && polyArea(outC.loop) > polyArea(circle(5, 64)), 'a negative offset grows the loop (outward)');
// a thin shape offset past half its width is refused, not silently inverted
const thin = [[-10, -1], [10, -1], [10, 1], [-10, 1]]; // 20 x 2
assert.equal(offsetLoop(thin, 2).ok, false, 'offsetting a thin shape past half its width is refused (collapsed)');
assert.equal(offsetLoop(thin, 2).reason, 'collapsed', '...named collapsed');
// a neck narrower than the wall is refused as a self-intersection
const dumbbell = [[-10, -3], [-1, -3], [-1, -0.4], [1, -0.4], [1, -3], [10, -3], [10, 3], [1, 3], [1, 0.4], [-1, 0.4], [-1, 3], [-10, 3]];
assert.equal(offsetLoop(dumbbell, 1).ok, false, 'offsetting past a thin neck is refused (self-intersect)');

// Compose: offset the outer to synthesize the hole, then puff (the Annular button end to end)
const oOuter = circle(12, 64);
const oHole = offsetLoop(oOuter, 4); // wall width 4 -> hole ~r8
assert.ok(oHole.ok, 'annular offset produced a hole');
const ring2 = buildAnnularPuff(oOuter, [oHole.loop], { count: 64, bands: 6 });
assert.ok(ring2.ok, `an offset-synthesized ring must puff: ${ring2.why || ''}`);
{
  const dirs = new Map(); const und = new Set();
  for (let q = 0; q < ring2.quads.length; q += 4) { const fq = [ring2.quads[q], ring2.quads[q + 1], ring2.quads[q + 2], ring2.quads[q + 3]]; for (let k = 0; k < 4; k++) { const a = fq[k], b = fq[(k + 1) % 4]; dirs.set(`${a}_${b}`, (dirs.get(`${a}_${b}`) || 0) + 1); und.add(a < b ? `${a}_${b}` : `${b}_${a}`); } }
  const V2 = ring2.positions.length / 3, F2 = ring2.quads.length / 4;
  assert.equal(V2 - und.size + F2, 0, 'the offset-synthesized ring is closed genus 1 too');
}

// Robustness: the hole need not be concentric or circular
const ellipse = (rx, ry, n = 48, cx = 0, cy = 0) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; p.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]); } return p; };
const genus1 = (res) => { // returns true if a valid closed genus-1 quad cage
  if (!res.ok) return false;
  const dir = new Map(); const und = new Set();
  for (let q = 0; q < res.quads.length; q += 4) { const f = [res.quads[q], res.quads[q + 1], res.quads[q + 2], res.quads[q + 3]]; if (new Set(f).size !== 4) return false; for (let k = 0; k < 4; k++) { const a = f[k], b = f[(k + 1) % 4]; dir.set(`${a}_${b}`, (dir.get(`${a}_${b}`) || 0) + 1); und.add(a < b ? `${a}_${b}` : `${b}_${a}`); } }
  for (const n of dir.values()) if (n !== 1) return false;                 // consistent winding
  const V = res.positions.length / 3, E = und.size, F = res.quads.length / 4;
  return V - E + F === 0;                                                  // closed genus 1
};
assert.ok(genus1(buildAnnularPuff(circle(10, 64), [circle(3, 48, 4, 0)], { count: 64, bands: 6 })), 'an off-center hole still builds a valid genus-1 ring');
assert.ok(genus1(buildAnnularPuff(circle(10, 64), [circle(3, 48, 6, 0)], { count: 64, bands: 6 })), 'a hole near the outer edge still builds a valid genus-1 ring');
assert.ok(genus1(buildAnnularPuff(circle(12, 64), [ellipse(6, 3, 48)], { count: 64, bands: 6 })), 'a NON-circular (elliptic) hole still builds a valid genus-1 ring');

// Per-rim falloff: the outer and the hole can carry different profiles
// Height along spoke 0: front ring k (k=1..K-1) sits at index 2N+(k-1)N. Ring 1 is nearest the
// outer rim, ring K-1 nearest the hole. A steeper (Lamé) rim rises faster near its own edge.
const spokeHeights = (res) => { const N = res.N, K = res.K, h = []; for (let k = 1; k < K; k++) h.push(res.positions[(2 * N + (k - 1) * N) * 3 + 2]); return h; };
{
  const O = circle(10, 64), H = circle(4, 48);
  const tangentOuterBulgeHole = buildAnnularPuff(O, [H], { count: 64, bands: 8, rimProfiles: [{ family: 'tangent', profile: 0.5 }, { family: 'lame', profile: 0.5 }] });
  const bulgeOuterTangentHole = buildAnnularPuff(O, [H], { count: 64, bands: 8, rimProfiles: [{ family: 'lame', profile: 0.5 }, { family: 'tangent', profile: 0.5 }] });
  assert.ok(tangentOuterBulgeHole.ok && bulgeOuterTangentHole.ok, 'both per-rim rings build');
  const a = spokeHeights(tangentOuterBulgeHole), b = spokeHeights(bulgeOuterTangentHole);
  // a: outer tangent (gentle near outer, low h[0]), hole lame (steep near hole, high h[last])
  assert.ok(a[0] < a[a.length - 1], `tangent outer + Lamé hole: the HOLE side rises faster (outer h ${a[0].toFixed(2)} < hole h ${a[a.length - 1].toFixed(2)})`);
  // b: the mirror — bulged outer, tangent hole
  assert.ok(b[0] > b[b.length - 1], `Lamé outer + tangent hole: the OUTER side rises faster (outer h ${b[0].toFixed(2)} > hole h ${b[b.length - 1].toFixed(2)})`);
  // and it is still a valid closed ring
  assert.ok(genus1(tangentOuterBulgeHole), 'a per-rim ring is still genus-1 watertight');
}
// Underside: bottomScale negative and flip support
{
  const O = circle(10, 64), H = circle(4, 48);
  const flatBottom = buildAnnularPuff(O, [H], { count: 32, bands: 4, bottomScale: 0 });
  assert.ok(flatBottom.ok && genus1(flatBottom), 'bottomScale=0 builds valid genus-1');
  const flatTop = buildAnnularPuff(O, [H], { count: 32, bands: 4, bottomScale: -1 });
  assert.ok(flatTop.ok && genus1(flatTop), 'bottomScale=-1 builds valid genus-1');
  const flipped = buildAnnularPuff(O, [H], { count: 32, bands: 4, bottomScale: 0, flip: true });
  assert.ok(flipped.ok && genus1(flipped), 'bottomScale=0 with flip=true builds valid genus-1');

  // Verify z extents
  let fbMaxZ = -Infinity, fbMinZ = Infinity;
  for (let i = 0; i < flatBottom.positions.length; i += 3) {
    const z = flatBottom.positions[i + 2];
    if (z > fbMaxZ) fbMaxZ = z;
    if (z < fbMinZ) fbMinZ = z;
  }
  assert.ok(fbMaxZ > 0.1 && Math.abs(fbMinZ) < 1e-6, `flatBottom has positive dome and flat bottom (max ${fbMaxZ.toFixed(2)}, min ${fbMinZ.toFixed(4)})`);

  let ftMaxZ = -Infinity, ftMinZ = Infinity;
  for (let i = 0; i < flatTop.positions.length; i += 3) {
    const z = flatTop.positions[i + 2];
    if (z > ftMaxZ) ftMaxZ = z;
    if (z < ftMinZ) ftMinZ = z;
  }
  assert.ok(Math.abs(ftMaxZ) < 1e-6 && ftMinZ < -0.1, `flatTop has flat top and negative dome (max ${ftMaxZ.toFixed(4)}, min ${ftMinZ.toFixed(2)})`);

  let flMaxZ = -Infinity, flMinZ = Infinity;
  for (let i = 0; i < flipped.positions.length; i += 3) {
    const z = flipped.positions[i + 2];
    if (z > flMaxZ) flMaxZ = z;
    if (z < flMinZ) flMinZ = z;
  }
  assert.ok(Math.abs(flMaxZ) < 1e-6 && flMinZ < -0.1, `flipped flatBottom has flat top and negative dome (max ${flMaxZ.toFixed(4)}, min ${flMinZ.toFixed(2)})`);
}
