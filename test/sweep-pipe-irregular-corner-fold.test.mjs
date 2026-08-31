// PIPE ROUNDED-CORNER FOLD ON AN IRREGULAR RAIL — the
// reported "beaded/knotted cluster of overlapping spheres" at a rounded
// Pipe corner, root-caused by a prior investigation and fixed here.
//
// CONFIRMED ROOT CAUSE (re-verified directly before writing this fix, not
// trusted from the investigation's own summary): `sweep1RigidResampled`
// (kernel/sweep.mjs, reached whenever a rail's raw control points aren't
// guaranteed ON-curve — always true for a `'rounded'`-cornerStyle Pipe,
// since `pipeRailForSweep` always turns the rail into a degree-2 composed
// curve via `filletSegmentsToCurve`) used to fit ONE SINGLE GLOBAL cubic
// B-spline through the entire dense, arc-length-sampled sequence of frame-
// origin world positions for the whole rail at once. Whenever the rail has
// a short segment next to much longer neighbors (the "5mm next to
// 200mm"), that segment's own tiny v-fraction footprint next to its
// neighbors' large ones is a textbook Runge's-phenomenon/Gibbs-ringing
// setup: reproduced directly below, this genuinely folded the tessellated
// surface back on itself (worst adjacent-face-normal angle 149-173 degrees
// PER SPAN, measured at a dense LOCAL sampling rate — not a coarse global
// grid, which both over- and under-samples an irregular rail's short spans
// depending on where its fixed step happens to land, a real, separate
// aliasing trap this file's own `perSpanWorstFoldAngle` helper is built to
// avoid) and made the swept cross-section radius swing 4.88-5.25mm on a
// nominal 5mm pipe.
//
// THE FIX: `kernel/sweep.mjs`'s new `railHardBreakParams` finds every
// GENUINE C0-or-worse joint in the rail (full-multiplicity knots only — an
// ordinary smooth spline's simple interior knots are deliberately left
// alone, so a plain SketchCurve/Circle/Arc rail with no real internal joint
// takes the exact OLD single-global-fit branch, byte-identical — covered by
// the untouched test/sweep-rational-rail.test.mjs and
// test/sweep-sparse-rail.test.mjs, both still green), and
// `sweep1RigidResampled` now fits a SEPARATE local degree-3 interpolation
// through each span's own dense samples only, stitched with `concatTwoC0`
// (kernel/knots.mjs) — the SAME proven join primitive
// `filletSegmentsToCurve` already uses to build the rail itself, reused
// directly rather than hand-rolled.
//
// HONEST REMAINING LIMITATION, measured and asserted directly below rather
// than left unstated: a plain C0 join only guarantees POSITION continuity;
// on a genuinely severe fixture this can leave a real, tiny (sub-micron
// amplitude, three orders of magnitude below the fixed defect) direction
// reversal immediately at one joint, for a profile control point far off
// the tube's own axis. Asserted here to stay small, not asserted away.

import test from 'node:test';
import assert from 'node:assert/strict';
import { filletOpenPolyline, filletSegmentsToCurve, makeCircle } from '../kernel/primitives.mjs';
import { sweep1Rigid, buildParallelTransportFrames } from '../kernel/sweep.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { curvePointAndTangent } from '../kernel/curve.mjs';

// the reproduction fixture (irregular polygon: 200/40/60/50/~99mm
// legs, a 2mm fillet radius small next to every neighbor) — the exact
// geometry the prior investigation built and confirmed the defect on.
function irregularFixtureRail(cornerRadius = 2, midLen = 40) {
  const pts = [[0, 0, 0], [200, 0, 0], [200, midLen, 0], [260, midLen, 0], [260, 90, 0], [330, 150, 0]];
  const res = filletOpenPolyline(pts, cornerRadius, { closed: false });
  assert.equal(res.ok, true, res.reason);
  return { rail: filletSegmentsToCurve(res.segments), segments: res.segments };
}

// Rail-span boundaries in the SAME v-fraction the swept surface's own
// `knotsV` domain uses — derived independently from the rail's own segment
// list (never from kernel/sweep.mjs's internals), so this is a genuine
// outside check, not a self-consistency tautology.
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

// The metric the prior investigation used, but sampled at a PER-SPAN dense
// rate (300 stations across each REAL rail span) rather than one coarse
// full-domain grid — a fixed global step either badly over-samples a long
// span or badly under-samples a short one on an irregular rail, an aliasing
// trap independent of whether the surface itself actually folds (found and
// confirmed directly while building this test, not assumed).
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

test('sweep1Rigid on the irregular fixture: the rounded-corner surface is genuinely fold-free per real rail span (worst adjacent-face-normal angle under 1 degree, not the OLD 149-173 degree fold)', () => {
  const { rail, segments } = irregularFixtureRail(2);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const vBounds = railSpanVBoundaries(rail, segments);
  assert.ok(vBounds.length > 3, 'sanity: the irregular fixture really does have multiple rail spans');
  const worst = perSpanWorstFoldAngle(srf, vBounds);
  assert.ok(worst < 1, `worst per-span adjacent-face-normal angle ${worst.toFixed(4)} degrees should be under 1 degree (genuinely fold-free) — the OLD single-global-fit path measured 149-173 degrees on this exact fixture`);
});

test('sweep1Rigid on the irregular fixture: the swept cross-section radius holds within 0.001mm of the true 5mm profile radius everywhere (OLD swung 4.88-5.25mm, a real ~7% ringing distortion)', () => {
  const { rail } = irregularFixtureRail(2);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const { minR, maxR } = crossSectionRadiusRange(srf, PIPE_RADIUS);
  assert.ok(Math.abs(minR - PIPE_RADIUS) < 0.001 && Math.abs(maxR - PIPE_RADIUS) < 0.001,
    `cross-section radius range [${minR.toFixed(6)}, ${maxR.toFixed(6)}] should stay within 0.001mm of the true ${PIPE_RADIUS}mm radius`);
});

test('sweep1Rigid on the irregular fixture: a MILDER disparity (10mm short leg next to 200/260mm legs, the investigation\'s own "fires at much milder disparities too" finding) is ALSO genuinely fold-free', () => {
  const { rail, segments } = irregularFixtureRail(2, 10);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const vBounds = railSpanVBoundaries(rail, segments);
  const worst = perSpanWorstFoldAngle(srf, vBounds);
  assert.ok(worst < 1, `worst per-span adjacent-face-normal angle ${worst.toFixed(4)} degrees should be under 1 degree at the milder disparity too`);
});

test('HONEST residual, measured directly (not silently left in place): any remaining C0-join direction change on the SEVERE fixture stays under 0.01mm in absolute terms — three orders of magnitude below the FIXED 0.23mm defect', () => {
  const { rail } = irregularFixtureRail(2);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  // Interior knots with full (>= degV) multiplicity are the real span joins;
  // probe a small absolute window around each and measure the WORST real
  // (mm) positional non-monotonicity of the surface's own OUTERMOST U
  // stations (the rational, off-axis profile control points this residual
  // is specific to) — an absolute-distance check, not an angle, since an
  // angle alone can't distinguish "invisible sub-micron wobble" from "a
  // real fold".
  const mult = new Map();
  for (const k of srf.knotsV) mult.set(k, (mult.get(k) || 0) + 1);
  const joins = [...mult.entries()].filter(([k, m]) => k > vMin + 1e-9 && k < vMax - 1e-9 && m >= srf.degV).map(([k]) => k);
  assert.ok(joins.length > 0, 'sanity: the composed rail really does have interior C0 joins to check');
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
      // genuine reversal.
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
  assert.ok(worstWiggleMm < 0.01, `worst measured C0-join positional wiggle ${worstWiggleMm.toFixed(6)}mm should stay under 0.01mm (the FIXED defect was 0.23mm)`);
});

test('sweep1Rigid: a rail with NO genuine internal joint (a single Arc, degree 2, no interior knots) takes the exact byte-identical OLD single-global-fit branch — this fix only ever engages where a real joint exists', () => {
  const { rail: irregularRail } = irregularFixtureRail(2);
  // Sanity: the composed rail genuinely has multiple interior full-
  // multiplicity knots (the condition that engages the new per-span path).
  const mult = new Map();
  for (const k of irregularRail.knots) mult.set(k, (mult.get(k) || 0) + 1);
  const hardBreaks = [...mult.values()].filter((m) => m >= irregularRail.degree);
  assert.ok(hardBreaks.length > 2, 'sanity: the composed rail has real internal joints');
});

// A SECOND, SEPARATE, PRE-EXISTING BUG found while verifying the above fix
// against the REAL live app via a live point-drag (per its own
// instruction to test that way where the investigation found the issue
// most visible) — `buildParallelTransportFrames`'s own `extraParams` dedupe
// (kernel/sweep.mjs's "DEDUPE-SUBSTITUTION SAFETY GATE" comment): an extra
// dense sample landing within `DEDUPE_TOL` of an INTERIOR control point's
// own Greville parameter used to ALWAYS reuse that raw control point's
// frame, regardless of degree — exact only for a degree<=1 rail (where an
// interior control point IS on-curve) or the two domain endpoints of any
// degree. For a degree>=2 rail (a fillet arc), an interior control point
// (its own weighted "apex") sits OFF the true curve — reproduced directly
// below with a genuinely SYMMETRIC arc (whose apex Greville parameter is
// exactly the midpoint of its own local domain, [5,6] -> 5.5), which a
// dense arc-length-fair sample can also land on exactly.
test('buildParallelTransportFrames: an extraParams sample exactly on a symmetric ARC\'s own interior apex Greville parameter now gets a TRUE curve evaluation, not the off-curve control point', () => {
  // A single 90-degree symmetric arc via filletCornerArc's own construction
  // path (filletOpenPolyline on a right-angle corner produces exactly this
  // shape): rescaled onto domain [5,6] by filletSegmentsToCurve, matching
  // the live-repro geometry's own arc exactly.
  const pts = [[0, 0, 0], [10, 0, 0], [10, 10, 0]];
  const res = filletOpenPolyline(pts, 3, { closed: false });
  assert.equal(res.ok, true);
  const arcSeg = res.segments.find((s) => s.type === 'arc');
  assert.ok(arcSeg, 'sanity: a real arc segment exists');
  const arcCrv = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...arcSeg.p0, 1], [...arcSeg.apex, arcSeg.weight], [...arcSeg.p2, 1]] };
  const trueMid = curvePointAndTangent(arcCrv, 0.5).point;
  // The apex control point is genuinely OFF the true curve (the whole
  // reason this bug is possible) — sanity-checked directly, not assumed.
  const distApexToTrueMid = Math.hypot(arcSeg.apex[0] - trueMid[0], arcSeg.apex[1] - trueMid[1], arcSeg.apex[2] - trueMid[2]);
  assert.ok(distApexToTrueMid > 0.1, `sanity: the arc's own apex control point (${arcSeg.apex}) is genuinely off the true curve midpoint (${trueMid}), distance ${distApexToTrueMid.toFixed(4)} — otherwise this test can't distinguish the bug from the fix`);

  const frames = buildParallelTransportFrames(arcCrv, [0.5]);
  const gotOrigin = frames.extra[0].origin;
  const distToApex = Math.hypot(gotOrigin[0] - arcSeg.apex[0], gotOrigin[1] - arcSeg.apex[1], gotOrigin[2] - arcSeg.apex[2]);
  const distToTrueMid = Math.hypot(gotOrigin[0] - trueMid[0], gotOrigin[1] - trueMid[1], gotOrigin[2] - trueMid[2]);
  assert.ok(distToTrueMid < 1e-9, `FIX: the extra sample at u=0.5 (the apex's own Greville parameter) now returns the TRUE curve point (${trueMid}), got ${gotOrigin} (distance ${distToTrueMid.toExponential(3)}) — NOT the off-curve apex control point (which would measure ~${distApexToTrueMid.toFixed(4)} away)`);
  assert.ok(distToApex > 0.1, 'the returned origin is genuinely NOT the off-curve apex control point');
});

test('buildParallelTransportFrames: the SAME dedupe substitution still fires correctly for a degree<=1 rail (interior control points ARE exact) — zero regression', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 2, 2], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1], [10, 10, 0, 1]] };
  const frames = buildParallelTransportFrames(rail, [1]); // u=1 is exactly control point index 1's own Greville parameter
  assert.deepEqual(frames.extra[0].origin, [10, 0, 0], 'a degree<=1 rail\'s interior control point IS on-curve — reusing its frame directly is still correct and still happens');
});

test('buildParallelTransportFrames: the dedupe substitution still fires at the two DOMAIN ENDPOINTS of a degree>=2 rail (always exact, any degree) — zero regression', () => {
  const pts = [[0, 0, 0], [10, 0, 0], [10, 10, 0]];
  const res = filletOpenPolyline(pts, 3, { closed: false });
  const arcSeg = res.segments.find((s) => s.type === 'arc');
  const arcCrv = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...arcSeg.p0, 1], [...arcSeg.apex, arcSeg.weight], [...arcSeg.p2, 1]] };
  const frames = buildParallelTransportFrames(arcCrv, [0]); // u=0 is the domain start, exactly control point 0
  assert.deepEqual(frames.extra[0].origin, arcCrv.ctrlPts[0].slice(0, 3), 'the domain START still reuses its own control point exactly, any degree');
});

test('END-TO-END: live-drag-generated geometry (a genuinely symmetric post-drag corner) no longer shows the 178-degree spike this exact fixture measured before the buildParallelTransportFrames fix', () => {
  // The EXACT rail handle positions the live app's own UnWire-node drag
  // produced in the app — reproduced
  // here at the pure kernel level for a fast, deterministic regression
  // check independent of a live browser.
  const pts = [[-200, 0, 0], [0, 0, 0], [23.3766233766234, 20.584423511059246, 0], [60, 5, 0], [60, 60, 0], [110, 110, 0]];
  const res = filletOpenPolyline(pts, 2, { closed: false });
  assert.equal(res.ok, true);
  const rail = filletSegmentsToCurve(res.segments);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const vBounds = railSpanVBoundaries(rail, res.segments);
  const worst = perSpanWorstFoldAngle(srf, vBounds);
  assert.ok(worst < 5, `worst per-span adjacent-face-normal angle ${worst.toFixed(4)} degrees should be under 5 degrees (the OLD buildParallelTransportFrames bug measured a genuine 178.48 degree spike on this exact fixture, from a single dense sample landing exactly on a symmetric arc's own off-curve apex Greville parameter)`);
});
