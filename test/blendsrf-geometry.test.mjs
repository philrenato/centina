// Blend and Patch on the shapes a user builds first: two whole tubes, a T,
// an uneven Y, stars with unequal setbacks, sheets lifted off the origin.
// Each test measures the picture's geometry — clearance from the axis, the
// sign of the Jacobian over a dense grid, the normals against the parents —
// rather than the construction's own bookkeeping.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { blendSurfaceBetweenEdges, blendSurfaceJunction, patchFromSurfaceEdges } from '../kernel/blendsrf.mjs';
import { surfacePoint, surfacePointAndPartials, closestPointOnSurface } from '../kernel/surface.mjs';
import { makeCircle, makeArc, extrude, revolve, makeLine } from '../kernel/primitives.mjs';
import { rationalCurveDerivs } from '../kernel/curve.mjs';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sc = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const nrm = (a) => { const L = len(a); return L > 0 ? sc(a, 1 / L) : [0, 0, 0]; };
const rad = (d) => d * Math.PI / 180;
const dom = (s) => ({ u0: s.knotsU[s.degU], u1: s.knotsU[s.knotsU.length - 1 - s.degU], v0: s.knotsV[s.degV], v1: s.knotsV[s.knotsV.length - 1 - s.degV] });
const edgeUV = (s, edge, t) => { const d = dom(s); if (edge === 'u0') return [d.u0, d.v0 + t * (d.v1 - d.v0)]; if (edge === 'u1') return [d.u1, d.v0 + t * (d.v1 - d.v0)]; if (edge === 'v0') return [d.u0 + t * (d.u1 - d.u0), d.v0]; return [d.u0 + t * (d.u1 - d.u0), d.v1]; };
const normalAt = (s, u, v) => { const P = surfacePointAndPartials(s, u, v); return { p: P.point, n: nrm(cross(P.su, P.sv)), j: cross(P.su, P.sv) }; };

// A tube whose rim (edge v0) is the circle 25 out along the direction at
// `deg` degrees in the xy plane, running outward 40.
const tubeAt = (deg, r = 10, setback = 25, runs = 1) => { const t = rad(deg), a = [Math.cos(t), Math.sin(t), 0], b = [-Math.sin(t), Math.cos(t), 0]; return { srf: extrude(makeCircle(sc(a, setback), b, [0, 0, runs], r, 4), a, 40), edge: 'v0' }; };
// A flat sheet whose edge u0 faces the origin from `R` out at `deg` degrees.
function sheetAt(R, deg, w = 40, length = 50, z = 0) {
  const t = rad(deg), ax = [Math.cos(t), Math.sin(t), 0], ay = [-Math.sin(t), Math.cos(t), 0], n = 4, ctrlNet = [];
  for (let i = 0; i < n; i++) { const row = []; for (let j = 0; j < n; j++) { const a = i / (n - 1), b = j / (n - 1); const p = add(add(sc(ax, R + a * length), sc(ay, (b - 0.5) * w)), [0, 0, z]); row.push([...p, 1]); } ctrlNet.push(row); }
  const K = [0, 0, 0, 0, 1, 1, 1, 1];
  return { srf: { degU: 3, degV: 3, knotsU: K.slice(), knotsV: K.slice(), ctrlNet }, edge: 'u0' };
}

test('two whole tubes: the blend keeps its radius between them at every angle and never pinches onto the axis', () => {
  for (const d2 of [90, 100, 120, 140, 160, 170, 180]) for (const runs of [1, -1]) {
    // B's rim runs either way round, its seam on either side.
    const A = tubeAt(0), B = tubeAt(d2, 10, 25, runs);
    const r = blendSurfaceBetweenEdges(A, B, { continuity: 2 });
    assert.ok(r.ok, r.reason);
    const D = dom(r.srf), row = Array.from({ length: 200 }, (_, i) => surfacePoint(r.srf, (D.u0 + D.u1) / 2, D.v0 + i / 200 * (D.v1 - D.v0)));
    // The middle row is a loop round the blend's axis; twisted, it collapses onto it.
    const c = sc(row.reduce(add, [0, 0, 0]), 1 / row.length);
    const clearance = Math.min(...row.map((p) => len(sub(p, c))));
    assert.ok(clearance > 0.6 * 10, `0/${d2} (B runs ${runs}): the middle row comes within ${clearance.toFixed(3)} of the axis (r 10)`);
  }
});

// Samples of a patch whose Jacobian turns against the patch's own mean
// normal: a cell folded over its neighbors.
function reversals(srf, n = 48) {
  const D = dom(srf), J = [];
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) J.push(normalAt(srf, D.u0 + (D.u1 - D.u0) * (0.002 + 0.996 * i / n), D.v0 + (D.v1 - D.v0) * (0.002 + 0.996 * j / n)).j);
  const mean = J.reduce(add, [0, 0, 0]);
  return J.filter((x) => dot(x, mean) < 0).length;
}
const tubeR = (deg, r = 10, setback = 25) => { const t = rad(deg), D = [Math.cos(t), Math.sin(t), 0]; const side = nrm(cross(D, [0, 0, 1])); return { srf: revolve(makeLine(add(sc(D, setback), sc(side, r)), add(sc(D, setback + 40), sc(side, r))), [0, 0, 0], D, 0, 2 * Math.PI), edge: 'u0' }; };

test('the cap does not fold where the arms are unevenly spaced — the Jacobian keeps one sign over every patch', () => {
  const cases = {
    'flat T, 20 wide': [0, 90, 180].map((a) => sheetAt(25, a, 20)),
    'flat T, 40 wide': [0, 90, 180].map((a) => sheetAt(40, a, 40)),
    'uneven Y (0, 35, 180)': [0, 35, 180].map((a) => sheetAt(40, a, 20)),
    'four arms at 0, 50, 150, 220': [0, 50, 150, 220].map((a) => sheetAt(40, a, 30)),
    '5-arm star, unequal setbacks': [0, 72, 144, 216, 288].map((a, i) => sheetAt([20, 45, 75, 30, 60][i], a, 40)),
    '4-arm cross, unequal setbacks': [0, 90, 180, 270].map((a, i) => sheetAt([20, 45, 75, 30][i], a, 40)),
    '5-arm star, unequal widths': [0, 72, 144, 216, 288].map((a, i) => sheetAt(40, a, [15, 40, 70, 25, 55][i])),
    'flat T, 10 wide': [0, 90, 180].map((a) => sheetAt(40, a, 10)),
    '5-arm star, even': [0, 72, 144, 216, 288].map((a) => sheetAt(40, a, 40)),
    'T of whole tubes, exact web': [0, 90, 180].map((a) => tubeR(a)),
  };
  for (const [name, parts] of Object.entries(cases)) {
    const r = blendSurfaceJunction(parts, { continuity: 2 });
    assert.ok(r.ok, `${name}: ${r.reason}`);
    const bad = r.centre.map((q) => reversals(q));
    assert.ok(bad.every((b) => b === 0), `${name}: reversed samples per patch ${bad.join(' ')}`);
  }
});

// Normals of every output face against the parents it meets: samples along
// each face's border that lie on a parent's edge, counted agreeing (+) and
// opposing (-). A trimmed patch is sampled along its trim.
const borderSamples = (s, n = 24) => { const out = []; for (const e of ['u0', 'u1', 'v0', 'v1']) for (let i = 2; i <= n - 2; i++) { const [u, v] = edgeUV(s, e, i / n); out.push(normalAt(s, u, v)); } return out; };
function facing(samples, parts, tol) {
  const ref = parts.flatMap((p) => Array.from({ length: 601 }, (_, i) => { const [u, v] = edgeUV(p.srf, p.edge, i / 600); return normalAt(p.srf, u, v); }));
  let agree = 0, oppose = 0;
  for (const x of samples) {
    let best = null, bd = Infinity; for (const y of ref) { const d = len(sub(x.p, y.p)); if (d < bd) { bd = d; best = y; } }
    if (bd < tol) { if (dot(x.n, best.n) > 0) agree++; else oppose++; }
  }
  return { agree, oppose };
}
const halfTube = (deg, z = 0) => { const t = rad(deg), a = [Math.cos(t), Math.sin(t), 0], b = [-Math.sin(t), Math.cos(t), 0]; return { srf: extrude(makeArc([25 * a[0], 25 * a[1], z], b, [0, 0, 1], 10, 0, Math.PI, 2), a, 40), edge: 'v0' }; };

test('every face of a blend or patch faces the way its parent surfaces do', () => {
  const junctions = {
    'half-tube cross': [0, 90, 180, 270].map((a) => halfTube(a)),
    'half-tube cross lifted 50': [0, 90, 180, 270].map((a) => halfTube(a, 50)),
    'uneven Y': [0, 35, 180].map((a) => sheetAt(40, a, 20)),
    'even Y': [0, 120, 240].map((a) => sheetAt(40, a, 40)),
    'tube T (web)': [0, 90, 180].map((a) => tubeR(a)),
    'tube cross (web)': [0, 90, 180, 270].map((a) => tubeAt(a)),
  };
  for (const [name, parts] of Object.entries(junctions)) {
    const r = blendSurfaceJunction(parts, { continuity: 2 });
    assert.ok(r.ok, `${name}: ${r.reason}`);
    const f = facing(r.centre.flatMap((q) => borderSamples(q)), parts, 1e-3);
    assert.ok(f.agree > 0 && f.oppose === 0, `${name}: ${f.agree} agree, ${f.oppose} oppose`);
    if (r.tube) {
      const n = r.centre.length / 2, top = facing(r.centre.slice(0, n).flatMap((q) => borderSamples(q)), r.halves.up, 1e-3), bottom = facing(r.centre.slice(n).flatMap((q) => borderSamples(q)), r.halves.down, 1e-3);
      assert.ok(top.oppose === 0 && bottom.oppose === 0 && top.agree > 0 && bottom.agree > 0, `${name}: top ${top.agree}/${top.oppose}, bottom ${bottom.agree}/${bottom.oppose}`);
    }
  }
  // Walls meeting at a post: the arms face their walls, the ribbons their arms.
  const wall = (deg) => { const t = rad(deg), ax = [Math.cos(t), Math.sin(t), 0], ctrlNet = []; for (let i = 0; i < 4; i++) { const row = []; for (let j = 0; j < 4; j++) row.push([ax[0] * (30 + 50 * i / 3), ax[1] * (30 + 50 * i / 3), (j / 3 - 0.5) * 40, 1]); ctrlNet.push(row); } const K = [0, 0, 0, 0, 1, 1, 1, 1]; return { srf: { degU: 3, degV: 3, knotsU: K, knotsV: K, ctrlNet }, edge: 'u0' }; };
  const walls = [0, 90, 180].map(wall), rw = blendSurfaceJunction(walls, { continuity: 2 });
  assert.ok(rw.ok, rw.reason);
  const fa = facing(rw.arms.flatMap((q) => borderSamples(q)), walls, 1e-3), fc = facing(rw.centre.flatMap((q) => borderSamples(q)), rw.arms.map((srf) => ({ srf, edge: 'u1' })), 1e-3);
  assert.ok(fa.oppose === 0 && fa.agree > 0 && fc.oppose === 0 && fc.agree > 0, `walls: arms ${fa.agree}/${fa.oppose}, ribbons ${fc.agree}/${fc.oppose}`);
  // A two-edge blend faces A, and B where the two agree.
  for (const [A, B] of [[sheetAt(20, 180), sheetAt(20, 0)], [tubeAt(0), tubeAt(180)], [tubeAt(0), tubeAt(90)]]) {
    const r = blendSurfaceBetweenEdges(A, B, {});
    assert.ok(r.ok, r.reason);
    const f = facing(borderSamples(r.srf), [A, B], 1e-3);
    assert.ok(f.agree > 0 && f.oppose === 0, `two-edge blend: ${f.agree} agree, ${f.oppose} oppose`);
  }
  // Patch between two facing sheets, and over a flat Y, either way up.
  const down = (q) => ({ ...q, srf: { ...q.srf, ctrlNet: q.srf.ctrlNet.map((row) => row.slice().reverse()), knotsV: q.srf.knotsV.map((x) => 1 - x).reverse() } });
  for (const parts of [[sheetAt(20, 180), sheetAt(20, 0)], [0, 120, 240].map((a) => sheetAt(40, a, 40))]) for (const p of [parts, parts.map(down)]) {
    const r = patchFromSurfaceEdges(p, { grid: 40 });
    assert.ok(r.ok, r.reason);
    const f = facing(r.trimLoop.map(([u, v]) => normalAt(r.srf, u, v)), p, 0.1);
    assert.ok(f.agree > 0 && f.oppose === 0, `patch: ${f.agree} agree, ${f.oppose} oppose`);
  }
});

// Above 1 the cap's free edges are held at their fillet bound, as the exact
// web's are, so Bulge acts below 1.
test('one surface over a junction answers Bulge: a flatter patch, still on its edges', () => {
  const parts = [0, 120, 240].map((a) => tubeR(a));
  const nets = [0.6, 1, 0.35].map((bulge) => { const r = blendSurfaceJunction(parts, { onePatch: true, bulge }); assert.ok(r.ok, r.reason); return r; });
  const moved = (a, b) => { let w = 0; a.centre.forEach((s, f) => { for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) w = Math.max(w, len(sub(surfacePoint(s, 0.1 + 0.08 * i, 0.1 + 0.08 * j), surfacePoint(b.centre[f], 0.1 + 0.08 * i, 0.1 + 0.08 * j)))); }); return w; };
  assert.ok(moved(nets[0], nets[1]) > 0.5 && moved(nets[2], nets[1]) > 0.5, `bulge moved the surface ${moved(nets[0], nets[1]).toFixed(3)} and ${moved(nets[2], nets[1]).toFixed(3)} mm`);
  for (const r of nets) assert.ok(r.patched.edgeGap < 0.1, `edge gap ${r.patched.edgeGap}`);
});

// The largest distance from a blend's boundary row (u0 or u1) to the edge it
// was built on, by closest point on a dense sample refined by golden section.
function boundaryGap(srf, side, part, n = 400) {
  const at = (t) => { const [u, v] = edgeUV(part.srf, part.edge, t); return surfacePoint(part.srf, u, v); };
  const S = Array.from({ length: 2001 }, (_, i) => at(i / 2000));
  let worst = 0;
  for (let i = 0; i <= n; i++) {
    const [u, v] = edgeUV(srf, side, i / n), p = surfacePoint(srf, u, v);
    let k = 0; for (let j = 1; j < S.length; j++) if (len(sub(S[j], p)) < len(sub(S[k], p))) k = j;
    let lo = Math.max(0, (k - 1) / 2000), hi = Math.min(1, (k + 1) / 2000);
    for (let it = 0; it < 60; it++) { const a = lo + (hi - lo) / 3, b = hi - (hi - lo) / 3; if (len(sub(at(a), p)) < len(sub(at(b), p))) hi = b; else lo = a; }
    worst = Math.max(worst, len(sub(at((lo + hi) / 2), p)));
  }
  return worst;
}

test('a two-edge blend lies on its edges within the join tolerance, and a wide closed rim builds', () => {
  const JOIN = 0.001, app = { continuity: 2, stations: 24, adaptive: true, tolerance: 0.01, edgeTolerance: JOIN / 4, maxStations: 97 };
  const sector = (r0, r1, z = 0) => revolve(makeLine([r0, 0, z], [r1, 0, z]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const cases = {
    'two tubes, 180': [tubeAt(0), tubeAt(180)],
    'two tubes, 90': [tubeAt(0), tubeAt(90)],
    'rim r150': [{ srf: sector(50, 150), edge: 'u1' }, { srf: sector(195, 300, 25), edge: 'u0' }],
    'rim r300': [{ srf: sector(100, 300), edge: 'u1' }, { srf: sector(390, 600, 50), edge: 'u0' }],
  };
  for (const [name, [A, B]] of Object.entries(cases)) {
    const r = blendSurfaceBetweenEdges(A, B, app);
    assert.ok(r.ok, `${name}: ${r.reason}`);
    const gA = boundaryGap(r.srf, 'u0', A), gB = boundaryGap(r.srf, 'u1', B);
    assert.ok(gA < JOIN / 2 && gB < JOIN / 2, `${name}: the boundary rows lie ${gA.toExponential(2)} and ${gB.toExponential(2)} mm off the edges`);
    assert.ok(r.toleranceMet && r.edgeDeviation <= JOIN / 4, `${name}: edge deviation ${r.edgeDeviation} after ${r.stations} stations`);
  }
});

test('Patch on surfaces builds the flat stars Blend builds: an X, 5 and 6 arms, two facing annulus sectors', () => {
  const inside = (pt, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const a = poly[i], b = poly[j]; if ((a[1] > pt[1]) !== (b[1] > pt[1]) && pt[0] < (b[0] - a[0]) * (pt[1] - a[1]) / (b[1] - a[1]) + a[0]) c = !c; } return c; };
  const sector = (r0, r1) => revolve(makeLine([r0, 0, 0], [r1, 0, 0]), [0, 0, 0], [0, 0, 1], 0, Math.PI / 2);
  const cases = {
    'X of 40 mm sheets': [0, 90, 180, 270].map((a) => sheetAt(40, a, 40)),
    '5-arm star': [0, 72, 144, 216, 288].map((a) => sheetAt(40, a, 30)),
    '6-arm star': [0, 60, 120, 180, 240, 300].map((a) => sheetAt(40, a, 30)),
    'two facing annulus sectors': [{ srf: sector(10, 30), edge: 'u1' }, { srf: sector(45, 60), edge: 'u0' }],
  };
  for (const [name, parts] of Object.entries(cases)) {
    const r = patchFromSurfaceEdges(parts, { grid: 48 });
    assert.ok(r.ok, `${name}: ${r.reason}`);
    let up = 0, down = 0;
    for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) { const uv = [i / 60, j / 60]; if (!inside(uv, r.trimLoop)) continue; const z = normalAt(r.srf, uv[0], uv[1]).j[2]; if (z > 0) up++; else if (z < 0) down++; }
    assert.ok(up > 100 && down === 0, `${name}: ${up} samples face up, ${down} fold under, inside the trim`);
    assert.ok(r.edgeGap < 0.05 && r.edgeAngleInner < 0.5, `${name}: edge gap ${r.edgeGap}, angle ${r.edgeAngleInner}`);
  }
});

test('one surface over five tubes meets their rims within 2.4 degrees (a fill that folds steps its cross-tangents down in small rungs)', () => {
  const tubes = [0, 72, 144, 216, 288].map((a) => { const t = rad(a), d = [Math.cos(t), Math.sin(t), 0], b = [-Math.sin(t), Math.cos(t), 0]; return { b, ctr: sc(d, 25), srf: extrude(makeCircle(sc(d, 25), b, [0, 0, 1], 8, 4), d, 40), edge: 'v0' }; });
  const r = blendSurfaceJunction(tubes, { onePatch: true });
  assert.ok(r.ok, r.reason);
  let worst = 0;
  for (const tb of tubes) for (let i = 0; i < 32; i++) {
    const th = 2 * Math.PI * i / 32; if (Math.abs(Math.sin(th)) < 0.25) continue;
    const tn = add(sc(tb.b, Math.cos(th)), [0, 0, Math.sin(th)]), p = add(tb.ctr, sc(tn, 8));
    let best = null; for (const f of r.centre) { const c = closestPointOnSurface(f, p); if (!best || c.distance < best.c.distance) best = { f, c }; }
    const n = normalAt(best.f, best.c.u, best.c.v).n;
    worst = Math.max(worst, Math.acos(Math.min(1, Math.abs(dot(n, tn)))) * 180 / Math.PI);
  }
  assert.ok(worst < 2.4, `the patch meets the rims at up to ${worst.toFixed(2)} degrees`);
});

// A cap's free edge leaves one sheet's side edge tangentially and meets the
// next sheet's the same way, turning one way between — wherever the two side
// lines meet ahead of both corners, so that such an edge exists. Where they
// meet behind a corner the chord lies outside the turn and one reversal is
// the least any tangent edge can have.
function freeEdgeShape(crv) {
  const K = crv.knots, a = K[0], b = K[K.length - 1], ks = [], heads = [];
  for (let i = 0; i <= 400; i++) { const [, d1, d2] = rationalCurveDerivs(crv, a + (b - a) * i / 400, 2), s = Math.hypot(d1[0], d1[1]); ks.push((d1[0] * d2[1] - d1[1] * d2[0]) / (s * s * s)); heads.push(Math.atan2(d1[1], d1[0])); }
  const kmax = Math.max(...ks.map(Math.abs));
  let reversals = 0, prev = 0; for (const k of ks) { if (!(Math.abs(k) > 1e-3 * kmax) || kmax < 1e-9) continue; const sg = Math.sign(k); if (prev && sg !== prev) reversals++; prev = sg; }
  let total = 0, net = 0; for (let i = 1; i < heads.length; i++) { let d = heads[i] - heads[i - 1]; d -= 2 * Math.PI * Math.round(d / (2 * Math.PI)); total += Math.abs(d); net += d; }
  const p0 = rationalCurveDerivs(crv, a, 1), p1 = rationalCurveDerivs(crv, b, 1);
  return { reversals, total: total * 180 / Math.PI, net: net * 180 / Math.PI, start: p0, end: p1 };
}
test('a cap\'s free edges leave the side edges tangentially and turn one way wherever one can', () => {
  const cases = {
    '5-arm star, unequal setbacks': [0, 72, 144, 216, 288].map((a, i) => sheetAt([20, 45, 75, 30, 60][i], a, 40)),
    'flat T': [0, 90, 180].map((a) => sheetAt(25, a, 20)),
    'uneven Y (0, 35, 180)': [0, 35, 180].map((a) => sheetAt(40, a, 20)),
    '5-arm star, even': [0, 72, 144, 216, 288].map((a) => sheetAt(40, a, 40)),
  };
  for (const [name, parts] of Object.entries(cases)) {
    const r = blendSurfaceJunction(parts, { continuity: 2 });
    assert.ok(r.ok, `${name}: ${r.reason}`);
    // Every sheet corner with its side edge's direction there.
    const corners = parts.flatMap((p) => [0, 1].map((t) => { const [u, v] = edgeUV(p.srf, p.edge, t), P = surfacePointAndPartials(p.srf, u, v); return { p: P.point, d: nrm(P.su) }; }));
    const sideAt = (q) => corners.reduce((best, c) => (len(sub(c.p, q)) < len(sub(best.p, q)) ? c : best));
    r.rails.forEach((crv, i) => {
      const f = freeEdgeShape(crv), A = f.start[0], B = f.end[0], tA = nrm(f.start[1]), tB = nrm(f.end[1]);
      // G1 to both side edges.
      for (const [q, t] of [[A, tA], [B, tB]]) { const c = sideAt(q); assert.ok(len(sub(c.p, q)) < 1e-6 && Math.abs(Math.abs(dot(c.d, t)) - 1) < 1e-8, `${name}, free edge ${i}: leaves its corner at ${(Math.acos(Math.min(1, Math.abs(dot(c.d, t)))) * 180 / Math.PI).toFixed(4)} degrees off the side edge`); }
      // Where the side lines meet: X = A + s tA = B + t tB, s > 0 and t < 0 when ahead of both.
      const w = sub(B, A), cr = tA[0] * tB[1] - tA[1] * tB[0];
      const s = Math.abs(cr) > 1e-9 ? (w[0] * tB[1] - w[1] * tB[0]) / cr : NaN, t = Math.abs(cr) > 1e-9 ? (w[0] * tA[1] - w[1] * tA[0]) / cr : NaN, chord = len(w);
      const oneWay = !Number.isFinite(s) || (s > 0.02 * chord && -t > 0.02 * chord);
      if (oneWay) assert.ok(f.reversals === 0 && Math.abs(f.total - Math.abs(f.net)) < 1, `${name}, free edge ${i}: ${f.reversals} curvature reversals, turns ${f.total.toFixed(1)} degrees for a net ${Math.abs(f.net).toFixed(1)}`);
      else assert.ok(f.reversals <= 1, `${name}, free edge ${i}: ${f.reversals} curvature reversals where one is the least`);
    });
  }
});

test('sheets that overlap where they meet are refused, naming the two and the fix, never capped with a fold', () => {
  const cases = {
    'four arms at 0, 40, 150, 220, 40 wide': [0, 40, 150, 220].map((a) => sheetAt(40, a, 40)),
    'two arms 20 degrees apart, 20 wide': [0, 20, 190].map((a) => sheetAt(40, a, 20)),
    'six arms, one 70 wide beside one 40 wide': [0, 60, 120, 180, 240, 300].map((a, i) => sheetAt(40, a, [15, 40, 70, 25, 55, 10][i])),
  };
  for (const [name, parts] of Object.entries(cases)) {
    const r = blendSurfaceJunction(parts, { continuity: 2 });
    if (r.ok) { const bad = r.centre.map((q) => reversals(q)); assert.ok(bad.every((b) => b === 0), `${name}: built with reversed samples ${bad.join(' ')}`); continue; }
    assert.match(r.reason, /edges \d+ and \d+ overlap.*move them apart/, `${name}: ${r.reason}`);
    assert.ok(r.reason.length <= 62, `${name}: the reason fits the status line after "Blend — " (${r.reason.length})`);
  }
});

// One surface over tubes is a fitted stand-in for the exact web — the
// construction that meets every rim at G2 and keeps the tubes' radius across
// the junction. So its shape is measured against the web: every sample of the
// trimmed surface within 0.02 of a tube radius of the web, and every sample of
// the web as near the surface. A pinch toward the axes' plane or a bulb past
// the tubes is a departure of the order of the radius.
test('one surface over a T and a cross keeps the exact web\'s shape (no pinch, no bulb)', () => {
  const inside = (pt, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const a = poly[i], b = poly[j]; if ((a[1] > pt[1]) !== (b[1] > pt[1]) && pt[0] < (b[0] - a[0]) * (pt[1] - a[1]) / (b[1] - a[1]) + a[0]) c = !c; } return c; };
  const grid = (faces, trims, n) => faces.flatMap((f, i) => { const D = dom(f), out = []; for (let a = 0; a <= n; a++) for (let b = 0; b <= n; b++) { const u = D.u0 + (D.u1 - D.u0) * a / n, v = D.v0 + (D.v1 - D.v0) * b / n; if (trims && trims[i] && !inside([u, v], trims[i])) continue; out.push(surfacePoint(f, u, v)); } return out; });
  const farthest = (pts, faces) => Math.max(...pts.map((p) => Math.min(...faces.map((f) => closestPointOnSurface(f, p).distance))));
  for (const [name, parts] of Object.entries({ 'T': [90, 180, 270].map((a) => tubeAt(a)), 'cross': [0, 90, 180, 270].map((a) => tubeAt(a)) })) {
    const one = blendSurfaceJunction(parts, { onePatch: true }), web = blendSurfaceJunction(parts, {});
    assert.ok(one.ok && web.ok, `${name}: ${one.reason || web.reason}`);
    const a = farthest(grid(one.centre, one.centreTrims, 10), web.centre), b = farthest(grid(web.centre, null, 3), one.centre);
    assert.ok(a < 0.2 && b < 0.2, `${name}: one surface lies up to ${a.toFixed(3)} mm off the web, the web up to ${b.toFixed(3)} mm off it (r 10)`);
  }
});

test('a Flip reverses that edge\'s run and the cap is built, not refused', () => {
  const parts = [0, 120, 240].map((a) => sheetAt(40, a, 40));
  const plain = blendSurfaceJunction(parts, { continuity: 2 }), flipped = blendSurfaceJunction(parts, { continuity: 2, flips: [false, true, false] });
  assert.ok(plain.ok && flipped.ok, flipped.reason);
  assert.ok(flipped.reversed[1] !== plain.reversed[1] && flipped.reversed[0] === plain.reversed[0], 'edge 2 runs the other way, edge 1 does not');
});
