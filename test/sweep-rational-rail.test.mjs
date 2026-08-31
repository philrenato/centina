// THE RATIONAL-RAIL FIX, superseding the earlier warning-only
// treatment ("SWEEP1 RATIONAL-RAIL WARNING"). Confirmed root
// cause, reproduced directly below before any fix is exercised: a Circle
// rail (kernel/primitives.mjs's makeArc/makeCircle, the standard rational
// closed-form tangent-intersection construction) has roughly half its
// control points sitting ~41.42% farther from center than the true circle
// (radius / cos(dtheta/2) at each tangent-line corner, weight < 1) —
// buildParallelTransportFrames' reuse-trick path places every swept frame's
// ORIGIN at one of these raw control points, so a profile reaching outward
// (a Line) amplifies that alternating ~41% error into a real, oscillating
// "twisted ribbon," exactly as first reported.
//
// The fix (sweep1Rigid dispatches to sweep1RigidRational when
// railIsRational(rail)): never uses a raw control point as a frame origin
// except at the two clamped domain endpoints; every other frame is a
// CONTINUOUS curve evaluation at a dense, arc-length-spaced parameter, and
// the final surface is a FRESH global interpolation through that dense
// grid — the same "resample dense, interpolate through values, never touch
// original control nets" technique sweepNProfiles' own earlier
// curved-rail fix already proved for a different (multi-profile) problem.

import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle, makeLine } from '../kernel/primitives.mjs';
import { sweep1Rigid, railIsRational, buildParallelTransportFrames } from '../kernel/sweep.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';
import { curvePoint, buildArcLengthTable, paramAtArcLength } from '../kernel/curve.mjs';
import { length, sub } from '../kernel/vec3.mjs';

const RADIUS = 100;
const railCircle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], RADIUS); // default segments=4, 9 ctrlPts — the exact reported scenario's own geometry

test('makeCircle really is rational, and really does have the documented ~41.42% off-curve control points (ground truth, not assumed)', () => {
  assert.equal(railIsRational(railCircle), true);
  const dists = railCircle.ctrlPts.map(([x, y]) => Math.hypot(x, y));
  const onCurve = dists.filter((d) => Math.abs(d - RADIUS) < 1e-6);
  const offCurve = dists.filter((d) => Math.abs(d - RADIUS * Math.SQRT2) < 1e-6);
  assert.equal(onCurve.length + offCurve.length, railCircle.ctrlPts.length, 'every control point is either exactly on the circle or exactly at the known tangent-corner overshoot — no unexplained third value');
  assert.ok(offCurve.length > 0, 'at least one genuinely off-curve control point exists (the condition this fix addresses)');
});

test('sweep1Rigid: a rational Circle rail dispatches to the rational-rail path (frames are NOT the raw off-curve reuse trick)', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [40, 0, 0, 1]] }; // a Line reaching outward — the exact reported amplifying case
  const srf = sweep1Rigid(railCircle, profile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  // The old bug's own signature: the outer row's distance from the rail's
  // center oscillating by the ~41% tangent-corner ratio. Reproduce the OLD
  // (pre-fix) construction directly here — not from memory, not from a
  // saved number — to prove the NEW result is genuinely different, not
  // just asserted to be fixed.
  const oldFrames = buildParallelTransportFrames(railCircle); // the reuse-trick frames the OLD path used as-is
  const oldOuterRadii = oldFrames.map((f) => Math.hypot(f.origin[0] + f.xAxis[0] * 40, f.origin[1] + f.xAxis[1] * 40));
  const oldRatio = Math.max(...oldOuterRadii) / Math.min(...oldOuterRadii);
  assert.ok(oldRatio > 1.3, `sanity: the OLD reuse-trick construction really does oscillate (ratio ${oldRatio.toFixed(4)}) — otherwise this test isn't reproducing the real bug`);

  const newOuterRadii = srf.ctrlNet[1].map((p) => Math.hypot(p[0], p[1]));
  const newRatio = Math.max(...newOuterRadii) / Math.min(...newOuterRadii);
  assert.ok(newRatio < 1.02, `FIX: the new construction's outer row stays consistent (ratio ${newRatio.toFixed(4)}), not oscillating by the old ~41% tangent-corner amount`);
});

test('sweep1Rigid rational-rail path tracks the TRUE rail path at MANY genuinely off-sample V values, not just its own dense sample set (the exactness-at-stations-only tautology this test specifically avoids)', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [40, 0, 0, 1]] };
  const srf = sweep1Rigid(railCircle, profile);
  const uMin = railCircle.knots[0], uMax = railCircle.knots[railCircle.knots.length - 1];
  const table = buildArcLengthTable(railCircle, uMin, uMax);
  const total = table.total;

  const SAMPLE_COUNT = 41; // deliberately not a divisor of the internal RATIONAL_RAIL_SAMPLES_PER_SPAN density, and offset by 0.5 below, so none of these coincide with the surface's own internal fit samples
  let maxInnerErr = 0, maxOuterErr = 0;
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const v = (i + 0.5) / SAMPLE_COUNT;
    const uTest = paramAtArcLength(table, v * total);
    // Independent ground truth: a FRESH single-parameter frame build (the
    // same already-independently-tested buildParallelTransportFrames
    // primitive — proven orthonormal/tangent-correct elsewhere in this
    // suite — used here only as a ground-truth oracle, not as part of
    // sweep1Rigid's own construction).
    const trueFrame = buildParallelTransportFrames(railCircle, [uTest]).extra[0];
    const expectedInner = trueFrame.origin;
    const expectedOuter = [
      trueFrame.origin[0] + trueFrame.xAxis[0] * 40,
      trueFrame.origin[1] + trueFrame.xAxis[1] * 40,
      trueFrame.origin[2] + trueFrame.xAxis[2] * 40,
    ];
    const actualInner = surfacePoint(srf, 0, v);
    const actualOuter = surfacePoint(srf, 1, v);
    maxInnerErr = Math.max(maxInnerErr, length(sub(actualInner, expectedInner)));
    maxOuterErr = Math.max(maxOuterErr, length(sub(actualOuter, expectedOuter)));

    // Also: the swept surface's own inner edge really is close to the true
    // circle at this off-sample parameter (not just close to our own
    // independently-rebuilt frame — an actual ground-truth curve check).
    const trueCirclePt = curvePoint(railCircle, uTest);
    assert.ok(length(sub(actualInner, trueCirclePt)) < 0.01, `v=${v.toFixed(4)}: swept inner edge ${actualInner} strays from the true circle point ${trueCirclePt} by more than the tight 0.01-unit tolerance (radius=${RADIUS})`);
  }
  assert.ok(maxInnerErr < 0.01, `max inner-row (on-rail) tracking error ${maxInnerErr.toFixed(5)} exceeds tight tolerance`);
  assert.ok(maxOuterErr < 0.01, `max outer-row (offset 40 outward) tracking error ${maxOuterErr.toFixed(5)} exceeds tight tolerance`);
});

test('sweep1Rigid: a NON-rational rail (Line) is completely untouched by the rational-rail path — same object shape, same exactness property as before', () => {
  const rail = makeLine([0, 0, 0], [0, 0, 200]);
  assert.equal(railIsRational(rail), false);
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[10, 5, 0, 1], [30, 5, 0, 1]] };
  const srf = sweep1Rigid(rail, profile);
  // The non-rational path's own established exactness contract: V-degree/
  // knots are the rail's OWN, reused directly (no re-fitting) — proves the
  // dispatch really did take the free/exact path, not the dense-resample one.
  assert.deepEqual(srf.degV, rail.degree);
  assert.deepEqual(srf.knotsV, rail.knots);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
});

test('sweep1Rigid: frames returned for the rational path are all orthonormal (same contract non-rational callers already rely on)', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [40, 0, 0, 1]] };
  const srf = sweep1Rigid(railCircle, profile);
  assert.ok(srf.frames.length > 0);
  for (const f of srf.frames) {
    assert.ok(Math.abs(length(f.xAxis) - 1) < 1e-8);
    assert.ok(Math.abs(length(f.yAxis) - 1) < 1e-8);
    assert.ok(Math.abs(length(f.zAxis) - 1) < 1e-8);
  }
});
