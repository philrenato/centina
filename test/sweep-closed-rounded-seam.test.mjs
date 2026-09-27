// The seam of a rounded (filleted) Pipe swept along a closed rail, where the
// rail wraps around. Two effects, both in kernel/sweep.mjs:
//
//  Seam frames. buildParallelTransportFrames welds frames[0] and frames[last]
//  on a closed rail. sweep1RigidResampled — the path a rounded Pipe on a
//  closed rail always takes, since the filleted rail is degree >= 2 — reads
//  the dense stations in `frames.extra`, so those must alias the welded frame
//  objects. If they carry the pre-weld frames, the two position-coincident
//  seam frames differ by about 11 degrees in xAxis on the non-planar 7-corner
//  fixture below with default rounded parameters, and the swept surface's
//  V control net opens a ~0.09 mm gap at the seam.
//
//  Seam tangent (known limitation). sweep1RigidResampled fits each V row
//  with an open, clamped interpolation, which makes a closed rail's two
//  domain ends agree in position but not in derivative. A tangent mismatch
//  of several degrees remains at the seam, at a similar magnitude on a planar
//  closed rail. See the known-limitation note above sweep1RigidResampled.
//  It is measured below, not asserted fixed.

import test from 'node:test';
import assert from 'node:assert/strict';
import { buildParallelTransportFrames, sweep1Rigid } from '../kernel/sweep.mjs';
import { filletOpenPolyline, filletSegmentsToCurve, makeCircle } from '../kernel/primitives.mjs';
import { isCurveClosed, curvePointAndTangent } from '../kernel/curve.mjs';
import { isFiniteNet } from '../kernel/surface.mjs';

// A non-planar 7-corner closed loop.
const FIXTURE_A = [[0, 0, 0], [150, 0, 10], [220, 90, -20], [160, 180, 15], [40, 200, -10], [-60, 130, 25], [-40, 40, -15]];
// Planar control: a flat closed square.
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

// A rational circle profile (weighted "corner" control points, ~41%
// off-axis): the worst case for this seam.
const PROFILE_RADIUS = 5;
const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PROFILE_RADIUS);

test('seam frames: the extraParams stations at both domain ends alias the welded frames[0]/frames[last]', () => {
  const rail = roundedClosedRail(FIXTURE_A, 5.1);
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const frames = buildParallelTransportFrames(rail, [uMin, uMax]);
  assert.ok(frames[0] === frames[frames.length - 1], 'sanity: the by-index weld itself still fires for this composed (degree>1) closed rail');
  assert.equal(frames.extra.length, 2);
  assert.ok(frames.extra[0] === frames[0], 'extra[0] (the domain-start dense station) shares the same welded frame object as frames[0]');
  assert.ok(frames.extra[1] === frames[frames.length - 1], 'extra[1] (the domain-end dense station) shares the same welded frame object as frames[last]');
});

test('seam frames, at the swept-surface level: the composed rounded rail\'s V-control net has zero position gap at the seam, on the non-planar 7-corner fixture', () => {
  const rail = roundedClosedRail(FIXTURE_A, 5.1);
  assert.equal(isCurveClosed(rail), true);
  assert.ok(rail.degree > 1, 'sanity: this composed/filleted rail is degree>1, so this test exercises sweep1RigidResampled, not the by-index free path');
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  let worstGap = 0;
  for (const row of srf.ctrlNet) {
    const a = row[0], b = row[row.length - 1];
    worstGap = Math.max(worstGap, dist(a, b));
  }
  assert.ok(worstGap < 1e-6, `V-control-net seam position gap is ${worstGap}mm`);
});

test('seam frames, planar control: the position gap is also zero on a planar closed square rail', () => {
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

test('seam tangent, known limitation: a tangent mismatch remains at the seam with zero position gap, on the non-planar 7-corner fixture', () => {
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
  assert.ok(worstPosGap < 1e-6, `sanity: the seam position gap is zero here too (${worstPosGap}mm)`);
  // Not asserting worstAngle is small: the open, clamped row fit does not
  // match derivatives at the seam. This records that the residual is present
  // (single-digit to low double-digit degrees) and finite, not NaN.
  assert.ok(Number.isFinite(worstAngle), 'the residual tangent gap is at least finite, never NaN');
  assert.ok(worstAngle > 0.5, `the residual is present (not already ~0): worst seam tangent angle is ${worstAngle.toFixed(4)} degrees`);
});

test('an open rounded rail sweeps to a finite tube — the seam weld does not touch the non-closed path', () => {
  const res = filletOpenPolyline(FIXTURE_A, 5.1, { closed: false });
  assert.ok(res.ok);
  const rail = filletSegmentsToCurve(res.segments);
  assert.equal(isCurveClosed(rail), false, 'sanity: an open rail is not recognized as closed');
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'an open rounded rail sweeps to a valid, finite tube');
});
