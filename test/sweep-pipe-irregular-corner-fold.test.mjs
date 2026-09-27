// Pipe rounded-corner fold on an irregular rail.
//
// `sweep1RigidResampled` (kernel/sweep.mjs) is reached whenever a rail's raw
// control points are not guaranteed on-curve — always true for a
// `'rounded'`-cornerStyle Pipe, since `pipeRailForSweep` turns the rail into
// a degree-2 composed curve via `filletSegmentsToCurve`. A single global
// cubic B-spline fitted through the dense, arc-length-sampled frame origins
// of the whole rail rings wherever the rail has a short segment next to much
// longer neighbors (5mm next to 200mm): the short segment's tiny v-fraction
// footprint next to its neighbors' large ones is a Runge's-phenomenon /
// Gibbs-ringing setup. On the fixture below such a fit folds the tessellated
// surface back on itself (worst adjacent-face-normal angle 149-173 degrees
// per span, measured at a dense per-span sampling rate — see
// `perSpanWorstFoldAngle`) and swings the swept cross-section radius
// 4.88-5.25mm on a nominal 5mm pipe.
//
// `railHardBreakParams` (kernel/sweep.mjs) finds every C0-or-worse joint in
// the rail (full-multiplicity knots only — an ordinary smooth spline's simple
// interior knots are left alone, so a plain SketchCurve/Circle/Arc rail with
// no internal joint keeps the single global fit, covered by
// test/sweep-rational-rail.test.mjs and test/sweep-sparse-rail.test.mjs), and
// `sweep1RigidResampled` fits a separate local degree-3 interpolation through
// each span's own dense samples only, stitched with `concatTwoC0`
// (kernel/knots.mjs) — the join primitive `filletSegmentsToCurve` uses to
// build the rail itself.
//
// Known limitation, asserted below: a plain C0 join only guarantees position
// continuity; on a severe fixture this can leave a tiny (sub-micron
// amplitude, three orders of magnitude below the fold) direction reversal
// immediately at one joint, for a profile control point far off the tube's
// own axis. It is asserted to stay small.

import test from 'node:test';
import assert from 'node:assert/strict';
import { filletOpenPolyline, filletSegmentsToCurve, makeCircle } from '../kernel/primitives.mjs';
import { sweep1Rigid, buildParallelTransportFrames } from '../kernel/sweep.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { curvePointAndTangent } from '../kernel/curve.mjs';

// the reproduction fixture (irregular polygon: 200/40/60/50/~99mm
// legs, a 2mm fillet radius small next to every neighbor) — a single global
// fit folds on this geometry.
function irregularFixtureRail(cornerRadius = 2, midLen = 40) {
  const pts = [[0, 0, 0], [200, 0, 0], [200, midLen, 0], [260, midLen, 0], [260, 90, 0], [330, 150, 0]];
  const res = filletOpenPolyline(pts, cornerRadius, { closed: false });
  assert.equal(res.ok, true, res.reason);
  return { rail: filletSegmentsToCurve(res.segments), segments: res.segments };
}

// Rail-span boundaries in the same v-fraction the swept surface's own
// `knotsV` domain uses — derived independently from the rail's own segment
// list (never from kernel/sweep.mjs's internals), so this is an outside
// check.
function railSpanVBoundaries(rail, segments) {
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const total = (() => {
    let acc = 0;
    for (const seg of segments) acc += segLength(seg);
    return acc;
  })();
  const bounds = [0];
  let acc = 0;
  for (const seg of segments) { acc += segLength(seg); bounds.push(acc / total); }
  return bounds;
}
function segLength(seg) {
  if (seg.type === 'line') return Math.hypot(seg.b[0] - seg.a[0], seg.b[1] - seg.a[1], seg.b[2] - seg.a[2]);
  // arc: approximate true length via the apex/weight construction — fine
  // for splitting v-fraction windows, doesn't need to be exact.
  const n = 64;
  let acc = 0; let prev = seg.p0;
  for (let k = 1; k <= n; k++) {
    const t = k / n;
    const w = seg.weight;
    const b0 = (1 - t) * (1 - t), b1 = 2 * (1 - t) * t * w, b2 = t * t;
    const denom = b0 + b1 + b2;
    const pt = [0, 1, 2].map((i) => (b0 * seg.p0[i] + b1 * seg.apex[i] + b2 * seg.p2[i]) / denom);
    acc += Math.hypot(pt[0] - prev[0], pt[1] - prev[1], pt[2] - prev[2]);
    prev = pt;
  }
  return acc;
}

function ring(srf, v, M = 16) {
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const pts = [];
  for (let i = 0; i <= M; i++) pts.push(surfacePoint(srf, uMin + (uMax - uMin) * i / M, v));
  return pts;
}
function triNormal(a, b, c) {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const len = Math.hypot(...n);
  return len < 1e-15 ? [0, 0, 0] : n.map((x) => x / len);
}
function angleBetween(n1, n2) {
  const d = Math.max(-1, Math.min(1, n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]));
  return Math.acos(d) * 180 / Math.PI;
}

// The fold metric, sampled at a per-span dense rate (300 stations across
// each rail span) rather than one coarse full-domain grid — a fixed global
// step either badly over-samples a long span or badly under-samples a short
// one on an irregular rail, an aliasing trap independent of whether the
// surface itself folds.
function perSpanWorstFoldAngle(srf, vBoundaries, samplesPerSpan = 300) {
  let worst = 0;
  for (let s = 0; s < vBoundaries.length - 1; s++) {
    const vLo = vBoundaries[s], vHi = vBoundaries[s + 1];
    const step = (vHi - vLo) / samplesPerSpan;
    const rows = [];
    for (let j = 0; j <= samplesPerSpan; j++) rows.push(ring(srf, vLo + j * step));
    for (let j = 1; j < samplesPerSpan; j++) {
      for (let i = 0; i < rows[0].length - 1; i++) {
        const n1 = triNormal(rows[j - 1][i], rows[j - 1][i + 1], rows[j][i]);
        const n2 = triNormal(rows[j][i], rows[j][i + 1], rows[j + 1][i]);
        const ang = angleBetween(n1, n2);
        if (ang > worst) worst = ang;
      }
    }
  }
  return worst;
}

function crossSectionRadiusRange(srf, trueRadius, stations = 400) {
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  let minR = Infinity, maxR = -Infinity;
  for (let j = 0; j <= stations; j++) {
    const v = vMin + (vMax - vMin) * j / stations;
    const M = 8; const pts = []; let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < M; i++) {
      const p = surfacePoint(srf, uMin + (uMax - uMin) * i / M, v);
      pts.push(p); cx += p[0]; cy += p[1]; cz += p[2];
    }
    cx /= M; cy /= M; cz /= M;
    for (const p of pts) {
      const r = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
      if (r < minR) minR = r; if (r > maxR) maxR = r;
    }
  }
  return { minR, maxR };
}

const PIPE_RADIUS = 5;

test('sweep1Rigid on the irregular fixture: the rounded-corner surface is fold-free per rail span (worst adjacent-face-normal angle under 1 degree; a single global fit folds at 149-173 degrees)', () => {
  const { rail, segments } = irregularFixtureRail(2);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const vBounds = railSpanVBoundaries(rail, segments);
  assert.ok(vBounds.length > 3, 'sanity: the irregular fixture has multiple rail spans');
  const worst = perSpanWorstFoldAngle(srf, vBounds);
  assert.ok(worst < 1, `worst per-span adjacent-face-normal angle ${worst.toFixed(4)} degrees should be under 1 degree (fold-free) — a single global fit measures 149-173 degrees on this fixture`);
});

test('sweep1Rigid on the irregular fixture: the swept cross-section radius holds within 0.001mm of the true 5mm profile radius everywhere (a single global fit swings 4.88-5.25mm, a ~7% ringing distortion)', () => {
  const { rail } = irregularFixtureRail(2);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const { minR, maxR } = crossSectionRadiusRange(srf, PIPE_RADIUS);
  assert.ok(Math.abs(minR - PIPE_RADIUS) < 0.001 && Math.abs(maxR - PIPE_RADIUS) < 0.001,
    `cross-section radius range [${minR.toFixed(6)}, ${maxR.toFixed(6)}] should stay within 0.001mm of the true ${PIPE_RADIUS}mm radius`);
});

test('sweep1Rigid on the irregular fixture: a milder disparity (10mm short leg next to 200/260mm legs) is also fold-free', () => {
  const { rail, segments } = irregularFixtureRail(2, 10);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const vBounds = railSpanVBoundaries(rail, segments);
  const worst = perSpanWorstFoldAngle(srf, vBounds);
  assert.ok(worst < 1, `worst per-span adjacent-face-normal angle ${worst.toFixed(4)} degrees should be under 1 degree at the milder disparity too`);
});

test('residual at the C0 joins: any direction change on the severe fixture stays under 0.01mm in absolute terms — well below the 0.23mm defect of a single global fit', () => {
  const { rail } = irregularFixtureRail(2);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  // Interior knots with full (>= degV) multiplicity are the span joins;
  // probe a small absolute window around each and measure the worst
  // (mm) positional non-monotonicity of the surface's own outermost u
  // stations (the rational, off-axis profile control points this residual
  // is specific to) — an absolute-distance check, not an angle, since an
  // angle alone can't distinguish "invisible sub-micron wobble" from "a
  // fold".
  const mult = new Map();
  for (const k of srf.knotsV) mult.set(k, (mult.get(k) || 0) + 1);
  const joins = [...mult.entries()].filter(([k, m]) => k > vMin + 1e-9 && k < vMax - 1e-9 && m >= srf.degV).map(([k]) => k);
  assert.ok(joins.length > 0, 'sanity: the composed rail has interior C0 joins to check');
  let worstWiggleMm = 0;
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const uStations = 24;
  for (const vB of joins) {
    for (let i = 0; i <= uStations; i++) {
      const u = uMin + (uMax - uMin) * i / uStations;
      const h = 1e-6;
      const a = surfacePoint(srf, u, vB - 2 * h);
      const b = surfacePoint(srf, u, vB - h);
      const c = surfacePoint(srf, u, vB);
      const d = surfacePoint(srf, u, vB + h);
      const e = surfacePoint(srf, u, vB + 2 * h);
      // "wiggle" = how far c sticks out beyond the envelope of its 4
      // neighbors along the dominant local displacement axis — near-zero
      // for a monotonic/smooth run, bounded-but-nonzero only right at a
      // reversal.
      const axis = [d[0] - b[0], d[1] - b[1], d[2] - b[2]];
      const axisLen = Math.hypot(...axis);
      if (axisLen < 1e-12) continue;
      const proj = (p) => (p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2]) / axisLen;
      const [pa, pb, pc, pd, pe] = [a, b, c, d, e].map(proj);
      const lo = Math.min(pa, pb, pd, pe), hi = Math.max(pa, pb, pd, pe);
      const overshoot = Math.max(0, pc - hi, lo - pc);
      if (overshoot > worstWiggleMm) worstWiggleMm = overshoot;
    }
  }
  assert.ok(worstWiggleMm < 0.01, `worst measured C0-join positional wiggle ${worstWiggleMm.toFixed(6)}mm should stay under 0.01mm (the single-global-fit defect is 0.23mm)`);
});

test('sweep1Rigid: the composed rounded-corner rail has interior full-multiplicity knots, the condition that engages the per-span fit', () => {
  const { rail: irregularRail } = irregularFixtureRail(2);
  // The composed rail has multiple interior full-multiplicity knots (the
  // condition that engages the per-span path).
  const mult = new Map();
  for (const k of irregularRail.knots) mult.set(k, (mult.get(k) || 0) + 1);
  const hardBreaks = [...mult.values()].filter((m) => m >= irregularRail.degree);
  assert.ok(hardBreaks.length > 2, 'sanity: the composed rail has internal joints');
});

// `buildParallelTransportFrames`'s own `extraParams` dedupe (kernel/sweep.mjs):
// an extra dense sample landing within `DEDUPE_TOL` of a control point's own
// Greville parameter may reuse that raw control point's frame only where the
// control point is on the curve — an interior control point of a degree<=1
// rail, or the two domain endpoints of any degree. For a degree>=2 rail (a
// fillet arc), an interior control point (its own weighted "apex") sits off
// the curve. The case below uses a symmetric arc, whose apex Greville
// parameter is exactly the midpoint of its own local domain ([5,6] -> 5.5),
// which a dense arc-length-fair sample can land on exactly.
test('buildParallelTransportFrames: an extraParams sample exactly on a symmetric arc\'s own interior apex Greville parameter gets a true curve evaluation, not the off-curve control point', () => {
  // A single 90-degree symmetric arc via filletCornerArc's own construction
  // path (filletOpenPolyline on a right-angle corner produces exactly this
  // shape), which filletSegmentsToCurve rescales onto a domain such as [5,6]
  // in a composed rail.
  const pts = [[0, 0, 0], [10, 0, 0], [10, 10, 0]];
  const res = filletOpenPolyline(pts, 3, { closed: false });
  assert.equal(res.ok, true);
  const arcSeg = res.segments.find((s) => s.type === 'arc');
  assert.ok(arcSeg, 'sanity: an arc segment exists');
  const arcCrv = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...arcSeg.p0, 1], [...arcSeg.apex, arcSeg.weight], [...arcSeg.p2, 1]] };
  const trueMid = curvePointAndTangent(arcCrv, 0.5).point;
  // The apex control point is off the curve (the reason a reused frame
  // would be wrong) — checked directly.
  const distApexToTrueMid = Math.hypot(arcSeg.apex[0] - trueMid[0], arcSeg.apex[1] - trueMid[1], arcSeg.apex[2] - trueMid[2]);
  assert.ok(distApexToTrueMid > 0.1, `sanity: the arc's own apex control point (${arcSeg.apex}) is off the true curve midpoint (${trueMid}), distance ${distApexToTrueMid.toFixed(4)} — otherwise this test can't distinguish the bug from the fix`);

  const frames = buildParallelTransportFrames(arcCrv, [0.5]);
  const gotOrigin = frames.extra[0].origin;
  const distToApex = Math.hypot(gotOrigin[0] - arcSeg.apex[0], gotOrigin[1] - arcSeg.apex[1], gotOrigin[2] - arcSeg.apex[2]);
  const distToTrueMid = Math.hypot(gotOrigin[0] - trueMid[0], gotOrigin[1] - trueMid[1], gotOrigin[2] - trueMid[2]);
  assert.ok(distToTrueMid < 1e-9, `the extra sample at u=0.5 (the apex's own Greville parameter) returns the true curve point (${trueMid}), got ${gotOrigin} (distance ${distToTrueMid.toExponential(3)}) — not the off-curve apex control point (which would measure ~${distApexToTrueMid.toFixed(4)} away)`);
  assert.ok(distToApex > 0.1, 'the returned origin is not the off-curve apex control point');
});

test('buildParallelTransportFrames: the same dedupe substitution fires for a degree<=1 rail (interior control points are exact)', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 2, 2], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1], [10, 10, 0, 1]] };
  const frames = buildParallelTransportFrames(rail, [1]); // u=1 is exactly control point index 1's own Greville parameter
  assert.deepEqual(frames.extra[0].origin, [10, 0, 0], 'a degree<=1 rail\'s interior control point is on-curve — its frame is reused directly');
});

test('buildParallelTransportFrames: the dedupe substitution fires at the two domain endpoints of a degree>=2 rail (always exact, any degree)', () => {
  const pts = [[0, 0, 0], [10, 0, 0], [10, 10, 0]];
  const res = filletOpenPolyline(pts, 3, { closed: false });
  const arcSeg = res.segments.find((s) => s.type === 'arc');
  const arcCrv = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...arcSeg.p0, 1], [...arcSeg.apex, arcSeg.weight], [...arcSeg.p2, 1]] };
  const frames = buildParallelTransportFrames(arcCrv, [0]); // u=0 is the domain start, exactly control point 0
  assert.deepEqual(frames.extra[0].origin, arcCrv.ctrlPts[0].slice(0, 3), 'the domain start reuses its own control point exactly, any degree');
});

test('end to end: a rail dragged into a symmetric corner sweeps without the 178-degree spike an off-curve apex frame produces', () => {
  // Rail handle positions from a node drag in the app (a symmetric corner),
  // reproduced at the kernel level.
  const pts = [[-200, 0, 0], [0, 0, 0], [23.3766233766234, 20.584423511059246, 0], [60, 5, 0], [60, 60, 0], [110, 110, 0]];
  const res = filletOpenPolyline(pts, 2, { closed: false });
  assert.equal(res.ok, true);
  const rail = filletSegmentsToCurve(res.segments);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const vBounds = railSpanVBoundaries(rail, res.segments);
  const worst = perSpanWorstFoldAngle(srf, vBounds);
  assert.ok(worst < 5, `worst per-span adjacent-face-normal angle ${worst.toFixed(4)} degrees should be under 5 degrees (reusing the off-curve apex frame for a dense sample landing exactly on a symmetric arc's apex Greville parameter gives a 178.48 degree spike on this fixture)`);
});
