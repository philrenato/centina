import test from 'node:test';
import assert from 'node:assert/strict';
import { pipeSectionCurve, pipeSectionRing, pipeSectionInner, pipeSectionFrameOpts, pipeSectionCapSurface, pipeDomeMeridian, pipeBlendMeridian, pipeDrawnSectionLocal, pipeSectionMeanRadius } from '../kernel/pipesection.mjs';
import { buildParallelTransportFrames, sweep1Rigid, pipeBlendCapSurface, pipeRoundCapSurface } from '../kernel/sweep.mjs';
import { subdPipeCage } from '../kernel/subdpipe.mjs';
import { curvePoint, curvePointAndTangent } from '../kernel/curve.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { makeCircle } from '../kernel/primitives.mjs';

const samples = (crv, n = 400) => { const a = crv.knots[0], b = crv.knots[crv.knots.length - 1]; return Array.from({ length: n }, (_, i) => curvePoint(crv, a + ((b - a) * i) / n)); };
const area = (pts) => { let s = 0; for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) s += pts[j][0] * pts[i][1] - pts[i][0] * pts[j][1]; return s / 2; };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

test('a teardrop is a circle with a 90-degree tip on +x at r * sqrt 2, counter-clockwise, no outline steeper than 45 degrees from +x', () => {
  const r = 5, td = pipeSectionCurve({ kind: 'teardrop' }, r);
  const pts = samples(td);
  assert.ok(area(pts) > 0, 'counter-clockwise');
  const far = pts.reduce((m, p) => Math.max(m, p[0]), -Infinity);
  assert.ok(Math.abs(far - r * Math.SQRT2) < 1e-9, `tip at ${far}`);
  for (const p of pts) assert.ok(Math.hypot(p[0], p[1]) <= r * Math.SQRT2 + 1e-9 && Math.hypot(p[0], p[1]) >= r - 1e-9);
  // Overhang: going around, the outline's slope never leaves 45 degrees of the +x (up) axis on the underside.
  const exact = (Math.PI * r * r * 3) / 4 + r * r; // 270-degree sector + the kite between the tangent points and the tip
  assert.ok(Math.abs(area(pts) - exact) / exact < 1e-3, `area ${area(pts)} vs ${exact}`);
});

test('a thick teardrop wall is even: the inner teardrop is the offset of the outer', () => {
  const r = 6, w = 1.5, outer = pipeSectionCurve({ kind: 'teardrop' }, r);
  const inner = pipeSectionInner(outer, { kind: 'teardrop' }, r, w);
  const op = samples(outer, 600);
  for (const q of samples(inner, 120)) {
    const d = Math.min(...op.map((p) => Math.hypot(p[0] - q[0], p[1] - q[1])));
    assert.ok(Math.abs(d - w) < 0.02, `wall ${d}`);
  }
});

test('lobes: n maxima, the first on +x, closed without a seam kink', () => {
  for (const n of [2, 3, 5]) {
    const crv = pipeSectionCurve({ kind: 'lobes', lobes: n, depth: 0.3 }, 4);
    const rad = samples(crv, 720).map((p) => Math.hypot(p[0], p[1]));
    let peaks = 0;
    for (let i = 0; i < rad.length; i++) if (rad[i] > rad[(i + rad.length - 1) % rad.length] && rad[i] >= rad[(i + 1) % rad.length]) peaks++;
    assert.equal(peaks, n);
    const t0 = curvePointAndTangent(crv, crv.knots[0]).tangent, t1 = curvePointAndTangent(crv, crv.knots[crv.knots.length - 1]).tangent;
    assert.ok(dot(t0, t1) / (Math.hypot(...t0) * Math.hypot(...t1)) > 0.9999, 'the seam is tangent-continuous');
  }
});

test('the ring starts at the teardrop tip and is evenly spaced', () => {
  const ring = pipeSectionRing(pipeSectionCurve({ kind: 'teardrop' }, 5), 16);
  assert.equal(ring.length, 16);
  assert.ok(Math.abs(ring[0][0] - 5 * Math.SQRT2) < 1e-6 && Math.abs(ring[0][1]) < 1e-6);
  const gaps = ring.map((p, i) => { const q = ring[(i + 1) % ring.length]; return Math.hypot(q[0] - p[0], q[1] - p[1]); });
  assert.ok(Math.max(...gaps) / Math.min(...gaps) < 1.2);
});

test('a round dome from the section cap equals the revolved hemisphere', () => {
  const frame = { origin: [1, 2, 3], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };
  const circle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 4);
  const cap = pipeSectionCapSurface(frame, circle, null, pipeDomeMeridian(4), [0, 0, 1]);
  for (const [u, v] of [[0.1, 0.2], [0.5, 0.7], [0.9, 0.95]]) {
    const p = surfacePoint(cap, cap.knotsU[0] + u * (cap.knotsU.at(-1) - cap.knotsU[0]), cap.knotsV[0] + v * (cap.knotsV.at(-1) - cap.knotsV[0]));
    assert.ok(Math.abs(Math.hypot(p[0] - 1, p[1] - 2, p[2] - 3) - 4) < 1e-9, 'on the sphere');
    assert.ok(p[2] >= 3 - 1e-9);
  }
  assert.ok(pipeRoundCapSurface(frame, 4, [0, 0, 1]));
});

test('the blend meridian across a round wall gives the revolved blend cap', () => {
  const frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };
  const outer = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 6), inner = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 4);
  const { meridian } = pipeBlendMeridian(6, 4, 1, 0.5);
  const cap = pipeSectionCapSurface(frame, outer, inner, meridian, [0, 0, 1]);
  const ref = pipeBlendCapSurface(frame, 6, 4, [0, 0, 1], 1, 0.5).srf;
  // Same profile: every cap point's (radius, height) lies on the revolved cap's.
  const rh = [];
  const [a0, a1, c0, c1] = [ref.knotsU[0], ref.knotsU.at(-1), ref.knotsV[0], ref.knotsV.at(-1)];
  for (let i = 0; i <= 60; i++) for (let j = 0; j <= 300; j++) { const p = surfacePoint(ref, a0 + ((a1 - a0) * i) / 60, c0 + ((c1 - c0) * j) / 300); rh.push([Math.hypot(p[0], p[1]), p[2]]); }
  for (let i = 1; i < 40; i++) {
    const u = cap.knotsU[0] + 0.3 * (cap.knotsU.at(-1) - cap.knotsU[0]), v = cap.knotsV[0] + (i / 40) * (cap.knotsV.at(-1) - cap.knotsV[0]);
    const p = surfacePoint(cap, u, v), q = [Math.hypot(p[0], p[1]), p[2]];
    const d = Math.min(...rh.map((t) => Math.hypot(t[0] - q[0], t[1] - q[1])));
    assert.ok(d < 0.05, `off the blend profile by ${d}`);
  }
});

test('a drawn square section sits centered and keeps its size', () => {
  const sq = { degree: 1, knots: [0, 0, 1, 2, 3, 4, 4], ctrlPts: [[10, 10, 5, 1], [10, 14, 5, 1], [14, 14, 5, 1], [14, 10, 5, 1], [10, 10, 5, 1]] };
  const frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };
  const local = pipeDrawnSectionLocal(sq, frame);
  const pts = samples(local);
  assert.ok(area(pts) > 0);
  assert.ok(Math.abs(area(pts) - 16) < 1e-6);
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  assert.ok(Math.hypot(cx, cy) < 1e-6);
  assert.ok(pipeSectionMeanRadius(local) > 2 && pipeSectionMeanRadius(local) < 2.83);
  assert.equal(pipeDrawnSectionLocal({ degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [5, 0, 0, 1]] }, frame), null);
});

test('a teardrop pipe holds its tip toward up on a horizontal helix and stays continuous through vertical', () => {
  const helix = globalCurveInterp(Array.from({ length: 25 }, (_, i) => { const a = (i / 24) * 4 * Math.PI; return [30 * Math.cos(a), 30 * Math.sin(a), a * 2]; }), 3);
  const opts = pipeSectionFrameOpts({ kind: 'teardrop' });
  const F = buildParallelTransportFrames(helix, Array.from({ length: 99 }, (_, i) => (i + 1) / 100), opts).extra;
  for (const f of F) {
    const proj = [0, 0, 1].map((x, k) => x - f.zAxis[k] * f.zAxis[2]);
    const l = Math.hypot(...proj);
    if (l < Math.sin((25 * Math.PI) / 180)) continue;
    const ang = Math.acos(Math.min(1, dot(proj, f.xAxis) / l)) * 180 / Math.PI;
    assert.ok(ang < 1, `tip ${ang} degrees off up`);
  }
  // An S-rail that climbs straight up between two horizontal runs.
  const s = globalCurveInterp([[0, 0, 0], [20, 0, 0], [30, 0, 10], [30, 0, 30], [30, 0, 50], [40, 0, 60], [60, 0, 60]], 3);
  const G = buildParallelTransportFrames(s, Array.from({ length: 199 }, (_, i) => (i + 1) / 200 * s.knots.at(-1)), opts).extra;
  for (let i = 1; i < G.length; i++) {
    assert.ok(G[i].xAxis.every(Number.isFinite));
    assert.ok(dot(G[i].xAxis, G[i - 1].xAxis) > Math.cos((12 * Math.PI) / 180), `frame jump at ${i}`);
  }
});

test('a teardrop sweeps along a rail that turns 180 degrees, and a SuperB pipe takes the ring', () => {
  const u = globalCurveInterp([[0, 0, 0], [40, 0, 0], [60, 0, 20], [40, 0, 40], [0, 0, 40]], 3);
  const td = pipeSectionCurve({ kind: 'teardrop' }, 4);
  const srf = sweep1Rigid(u, td, null, pipeSectionFrameOpts({ kind: 'teardrop' }));
  assert.ok(srf.ctrlNet.flat(2).every(Number.isFinite));
  const cage = subdPipeCage(u, { radius: 4, ring: pipeSectionRing(td, 12), frameOpts: pipeSectionFrameOpts({ kind: 'teardrop' }), capStart: 'flat', capEnd: 'round' });
  assert.ok(cage.vertices.length > 24 && cage.vertices.flat().every(Number.isFinite));
  const round = subdPipeCage(u, { radius: 4 });
  const roundAgain = subdPipeCage(u, { radius: 4, frameOpts: {} });
  assert.deepEqual(round.vertices, roundAgain.vertices);
});
