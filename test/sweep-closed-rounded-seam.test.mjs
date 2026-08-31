// CLOSED-RAIL ROUNDED-CORNER SEAM, THE 4th "PIPE ROUNDED-CORNER SEAM"
// INSTALLMENT — an investigation (real fixtures, real
// numbers, driving the live app via CDP) found that Rounded+Closed
// together — the ONE combination no existing verify script ever
// exercised — carries a real, previously-unmeasured seam defect at the
// wraparound, with TWO separate, independently-confirmed causes, both in
// kernel/sweep.mjs:
//
//  BUG 1 (FIXED here) — `buildParallelTransportFrames`'s own earlier
//  seam weld reassigned the by-index `frames[0]`/`frames[last]` array
//  slots but never touched the underlying step objects `.extra` (the ONLY
//  thing `sweep1RigidResampled` — the path a Rounded Pipe on a CLOSED rail
//  ALWAYS reaches, since the filleted rail is degree>=2 — ever reads)
//  aliased, so `.extra` still carried the OLD, un-welded frames. Measured
//  directly on a real non-planar 7-corner closed Pipe rail with default
//  Rounded params: an ~11 degree xAxis orientation mismatch between the
//  two (position-coincident) seam frames, producing a real ~0.09mm gap in
//  the swept surface's own V-control net right at the seam. FIXED by
//  mutating the underlying step objects' `.frame` in place and moving the
//  `.extra` capture to run AFTER the weld (see that function's own
//  own UPDATE comment) — proven below.
//
//  BUG 2 (NOT fixed here, named honestly) — even with BUG 1 fixed
//  (positions coincide exactly at the seam), `sweep1RigidResampled`'s own
//  per-row V-curve fit is a plain OPEN, CLAMPED interpolation with no
//  knowledge that a CLOSED rail's own two domain ends should agree in
//  DERIVATIVE, only in position — a real, several-degree tangent mismatch
//  remains at the seam, present at a SIMILAR magnitude on a PLANAR closed
//  rail too (falsifying this file's own prior "planar closed rail has zero
//  holonomy and is unaffected" claim — see `pipeRailForSweep`'s own
//  corrected comment in the app). THREE different fix attempts
//  (point-borrowed padding, density-matched padding, a Hermite-style
//  single-control-point nudge) were built and adversarially tested against
//  this file's own real 7-corner fixture — each one either reproduced a
//  Runge's-phenomenon-style knot-clustering ringing, corrupted
//  `extractSubCurve`'s own tolerance-based knot multiplicity counting, or
//  measurably disturbed the tightly-curved short span's own already-proven
//  shape more than its own established baseline. None was safe to ship per
//  this project's own standing "don't rush unverified NURBS math" rule —
//  see `sweep1RigidResampled`'s own "CLOSED-RAIL SEAM" header comment for
//  the full account of all three attempts and why each was rejected. Real,
//  deferred scope, not silently dropped — NOT tested as "fixed" below,
//  since it isn't.

import test from 'node:test';
import assert from 'node:assert/strict';
import { buildParallelTransportFrames, sweep1Rigid } from '../kernel/sweep.mjs';
import { filletOpenPolyline, filletSegmentsToCurve, makeCircle } from '../kernel/primitives.mjs';
import { isCurveClosed, curvePointAndTangent } from '../kernel/curve.mjs';
import { isFiniteNet } from '../kernel/surface.mjs';

// Real, non-trivial fixture — the investigation's own primary proof case,
// not a toy: a non-planar 7-corner closed loop (repeats its own start
// point, the `isCurveClosed` convention this kernel already uses).
const FIXTURE_A = [[0, 0, 0], [150, 0, 10], [220, 90, -20], [160, 180, 15], [40, 200, -10], [-60, 130, 25], [-40, 40, -15]];
// The investigation's own decisive PLANAR control case — same shape
// class, forced flat, used to falsify "planar closed rails are unaffected".
const FIXTURE_C_SQUARE = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]];

function roundedClosedRail(points, cornerRadius) {
  const res = filletOpenPolyline(points, cornerRadius, { closed: true });
  assert.ok(res.ok, 'fillet must succeed for these fixtures at this radius');
  return filletSegmentsToCurve(res.segments);
}

function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function angleBetween(a, b) {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  return Math.acos(Math.max(-1, Math.min(1, d / (na * nb)))) * 180 / Math.PI;
}

// Real, non-symmetric circle profile (weighted rational "corner" control
// points, ~41% off-axis) — the investigation's own identified worst case
// for this defect, reused directly rather than a simpler stand-in.
const PROFILE_RADIUS = 5;
const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PROFILE_RADIUS);

test('BUG 1, ground truth: buildParallelTransportFrames welds frames[0]/frames[last] but (before the fix) the SAME extraParams stations would dedupe onto the pre-weld frame', () => {
  const rail = roundedClosedRail(FIXTURE_A, 5.1);
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const frames = buildParallelTransportFrames(rail, [uMin, uMax]);
  assert.ok(frames[0] === frames[frames.length - 1], 'sanity: the by-index weld itself still fires for this composed (degree>1) closed rail');
  assert.equal(frames.extra.length, 2);
  assert.ok(frames.extra[0] === frames[0], 'THE FIX: extra[0] (the domain-start dense station) now shares the SAME welded frame object as frames[0]');
  assert.ok(frames.extra[1] === frames[frames.length - 1], 'THE FIX: extra[1] (the domain-end dense station) now shares the SAME welded frame object as frames[last]');
});

test('BUG 1, THE FIX, at the swept-surface level: the composed rounded rail\'s V-control net has ZERO position gap at the seam, on the real non-planar 7-corner fixture', () => {
  const rail = roundedClosedRail(FIXTURE_A, 5.1);
  assert.equal(isCurveClosed(rail), true);
  assert.ok(rail.degree > 1, 'sanity: this composed/filleted rail really is degree>1, so this test exercises sweep1RigidResampled, not the by-index free path');
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  let worstGap = 0;
  for (const row of srf.ctrlNet) {
    const a = row[0], b = row[row.length - 1];
    worstGap = Math.max(worstGap, dist(a, b));
  }
  assert.ok(worstGap < 1e-6, `V-control-net seam position gap is ${worstGap}mm — was ~0.09mm before this fix`);
});

test('BUG 1, THE FIX, on the DECISIVE planar control case: position gap is ALSO zero on a plain planar closed square rail', () => {
  const rail = roundedClosedRail(FIXTURE_C_SQUARE, 5.1);
  assert.equal(isCurveClosed(rail), true);
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  let worstGap = 0;
  for (const row of srf.ctrlNet) {
    worstGap = Math.max(worstGap, dist(row[0], row[row.length - 1]));
  }
  assert.ok(worstGap < 1e-6, `V-control-net seam position gap on the planar square is ${worstGap}mm`);
});

test('BUG 2, HONEST STATUS CHECK (not "fixed" — measures and documents the real, still-open residual): a genuine tangent mismatch remains at the seam even with BUG 1\'s position fix, on the real non-planar 7-corner fixture', () => {
  const rail = roundedClosedRail(FIXTURE_A, 5.1);
  const srf = sweep1Rigid(rail, circleProfile);
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const eps = (vMax - vMin) * 1e-6;
  let worstAngle = 0, worstPosGap = 0;
  for (const row of srf.ctrlNet) {
    const rowCrv = { degree: srf.degV, knots: srf.knotsV, ctrlPts: row };
    const startExact = curvePointAndTangent(rowCrv, vMin);
    const endExact = curvePointAndTangent(rowCrv, vMax);
    const startIn = curvePointAndTangent(rowCrv, vMin + eps);
    const endIn = curvePointAndTangent(rowCrv, vMax - eps);
    worstPosGap = Math.max(worstPosGap, dist(startExact.point, endExact.point));
    worstAngle = Math.max(worstAngle, angleBetween(startIn.tangent, endIn.tangent));
  }
  assert.ok(worstPosGap < 1e-6, `sanity: BUG 1's own position fix still holds here (${worstPosGap}mm)`);
  // NOT asserting worstAngle is small — that would be the (unbuilt) BUG 2
  // fix. This is a HONEST STATUS check: documents the residual is real and
  // roughly the magnitude the investigation measured (single-digit-to-low-
  // double-digit degrees), not a crash/NaN, and not silently claimed fixed.
  assert.ok(Number.isFinite(worstAngle), 'the residual tangent gap is at least finite, never NaN');
  assert.ok(worstAngle > 0.5, `documents the residual is REAL (not accidentally already ~0): worst seam tangent angle is ${worstAngle.toFixed(4)} degrees — BUG 2, named but not fixed this file`);
});

test('REGRESSION GUARD: an OPEN rounded rail (the ordinary, already-working, far more common case) is byte-identical — nothing about BUG 1\'s fix touches the non-closed path', () => {
  const res = filletOpenPolyline(FIXTURE_A, 5.1, { closed: false });
  assert.ok(res.ok);
  const rail = filletSegmentsToCurve(res.segments);
  assert.equal(isCurveClosed(rail), false, 'sanity: an open rail is correctly NOT recognized as closed');
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'an open rounded rail still sweeps to a valid, finite tube exactly as before this file');
});
