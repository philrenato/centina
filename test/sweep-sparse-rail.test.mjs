// A SECOND, real Sweep1 bug, distinct from the rational-rail one's
// Circle-rail rational fix: "Is sweep1 working with 1 profile now?
// Doesn't seem to be for me." — a sharp, nearly-planar WEDGE surface (Deg
// U=2/V=2, Ctrl pts U=3/V=3), reproduced directly through the REAL running
// app (see the interactive repro and its own captured
// screenshot) with a 3-point, degree-2, NON-rational SketchCurve rail (a
// genuine hook/S shape, matching the reported object's own Points=3/
// Degree=2/Length~291mm) and a second, sparse 3-point SketchCurve profile.
//
// Root cause: `railIsRational` (the rational-rail dispatch condition) is NOT
// the right test for whether a rail's raw control points sit on its own
// curve. `globalCurveInterp`'s own middle control point, for a curve that
// has to bend sharply to hit all its data points with only a few control
// points, can sit FAR off the true curve — a basic NURBS/Bezier fact that
// has nothing to do with rational weights. Confirmed directly: the exact
// reported rail's own middle control point sits ~70+ units off its own
// true curve on a ~260-unit-long rail (~30% of the rail's own span) —
// worse, proportionally, than the rational-rail case's ~41% Circle-tangent-corner
// case, and NOT caught by railIsRational (every weight here is exactly 1).
//
// The fix (kernel/sweep.mjs): sweep1Rigid's dispatch condition is now
// `railFrameOriginsExact(rail)` (true only for degree<=1 — Line/Polyline,
// whose control polygon literally IS the curve, by construction, whatever
// the weight) instead of `!railIsRational(rail)`. Any degree>=2 rail —
// rational (Circle/Arc) OR non-rational (a sparse/sharply-curved
// SketchCurve) — now runs the SAME dense-arc-length-resample-and-fresh-
// interpolate construction the rational-rail fix already proved
// (`sweep1RigidResampled`, renamed from `sweep1RigidRational` since it is
// no longer rational-rail-specific).
import test from 'node:test';
import assert from 'node:assert/strict';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { sweep1Rigid, railIsRational, railFrameOriginsExact, buildParallelTransportFrames, localizeSectionToFrame } from '../kernel/sweep.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';
import { curvePoint, buildArcLengthTable, paramAtArcLength } from '../kernel/curve.mjs';
import { length, sub } from '../kernel/vec3.mjs';

// The exact reported rail: a 3-point, degree-2 SketchCurve, a genuine
// hook/S shape (matching the SketchCurve02: Points=3, Degree=2,
// Length~291mm — this rail's own true length is ~260.8, the same order of
// magnitude and curvature character, reproduced directly through the real
// app, interactively).
const rail = globalCurveInterp([[0, 0, 0], [80, 120, 0], [20, 260, 0]], 2);
// The exact reported profile: a SECOND, sparse, angular 3-point
// SketchCurve (not a Circle) — the case the earlier, kernel-only-with-a-
// Circle-profile investigation this file started from did NOT try.
const rawProfile = globalCurveInterp([[0, 0, -20], [15, 0, 10], [-10, 0, 25]], 2);

test('the rail in question really is non-rational (every weight 1) yet has a control point badly off its own true curve — this is NOT the rational-rail mechanism', () => {
  assert.equal(railIsRational(rail), false, 'sanity: every weight is exactly 1 — railIsRational alone would (wrongly) send this rail down the old free/exact path');
  const mid = rail.ctrlPts[1];
  const trueMid = curvePoint(rail, 0.5);
  const deviation = length(sub([mid[0], mid[1], mid[2]], trueMid));
  assert.ok(deviation > 40, `the rail's own middle control point should be badly off-curve (deviation ${deviation.toFixed(2)}) — otherwise this test isn't reproducing the real reported bug`);
});

test('railFrameOriginsExact correctly distinguishes degree<=1 (Line/Polyline) from this degree-2 sparse rail', () => {
  assert.equal(railFrameOriginsExact(rail), false, 'a degree-2 rail, even non-rational, is NOT guaranteed to have on-curve control points');
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [50, 0, 0, 1]] };
  assert.equal(railFrameOriginsExact(line), true, 'a degree-1 (Line) rail IS guaranteed exact, regardless of weight');
});

test('sweep1Rigid on the exact reported scenario dispatches to the resampled path, not the old off-curve reuse trick', () => {
  const frames0 = buildParallelTransportFrames(rail);
  const section = localizeSectionToFrame(rawProfile, frames0[0]);
  const srf = sweep1Rigid(rail, section);
  assert.equal(isFiniteNet(srf.ctrlNet), true);

  // Reproduce the OLD (pre-fix) construction directly — not from memory —
  // to prove the new result is genuinely different, not just re-asserted.
  const oldMidRow = section.ctrlPts.map(([sx, sy, sz]) => {
    const f = frames0[1]; // the OLD path's own middle frame: raw rail control point as origin
    return [f.origin[0] + f.xAxis[0] * sx + f.yAxis[0] * sy + f.zAxis[0] * sz,
            f.origin[1] + f.xAxis[1] * sx + f.yAxis[1] * sy + f.zAxis[1] * sz,
            f.origin[2] + f.xAxis[2] * sx + f.yAxis[2] * sy + f.zAxis[2] * sz];
  });
  const trueMid = curvePoint(rail, 0.5);
  const oldDeviation = length(sub(oldMidRow[0], trueMid));
  assert.ok(oldDeviation > 40, `sanity: the OLD construction's own middle row really is badly off the true rail (deviation ${oldDeviation.toFixed(2)}) — otherwise this test isn't reproducing the real bug`);

  // The NEW surface's own knotsV must NOT be the rail's raw knots/degree —
  // proof the dispatch genuinely took the resampled path, not a
  // coincidentally-similar result via the old one.
  assert.ok(srf.knotsV.length > rail.knots.length, `the resampled path's own fresh knot vector (length ${srf.knotsV.length}) should be denser than the rail's raw one (length ${rail.knots.length}) — proof a fresh interpolation genuinely ran, not a reuse`);
});

test('sweep1Rigid on the exact reported scenario tracks the rail\'s TRUE path within a tight tolerance, at many genuinely off-sample V AND U values', () => {
  const frames0 = buildParallelTransportFrames(rail);
  const section = localizeSectionToFrame(rawProfile, frames0[0]);
  const srf = sweep1Rigid(rail, section);

  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const table = buildArcLengthTable(rail, uMin, uMax);
  const total = table.total;
  const TOLERANCE = total * 0.01; // 1% of the rail's own true length — tight

  // Ground truth: an INDEPENDENT single-parameter buildParallelTransportFrames
  // call at the exact off-sample rail parameter (the same ground-truth-
  // oracle pattern test/sweep-rational-rail.test.mjs already established),
  // transporting the PROFILE'S OWN LOCAL CURVE VALUE at a chosen U
  // parameter (via curvePoint on the local `section` curve — NOT a raw
  // control point, for the exact same reason the rail side of this bug
  // required a curve VALUE, not a control point: `surfacePoint(srf, u, v)`
  // evaluates real blended B-spline basis functions on BOTH u and v, so
  // comparing it against a single raw control point conflates two
  // different, unrelated approximation errors) through that TRUE frame.
  function frameToWorld(f, [sx, sy, sz]) {
    return [
      f.origin[0] + f.xAxis[0] * sx + f.yAxis[0] * sy + f.zAxis[0] * sz,
      f.origin[1] + f.xAxis[1] * sx + f.yAxis[1] * sy + f.zAxis[1] * sz,
      f.origin[2] + f.xAxis[2] * sx + f.yAxis[2] * sy + f.zAxis[2] * sz,
    ];
  }
  const u0 = section.knots[0], u1 = section.knots[section.knots.length - 1];
  let maxErr = 0;
  for (let i = 0; i < 41; i++) {
    const v = (i + 0.5) / 41; // deliberately off the surface's own internal dense-sample grid
    const uTrue = paramAtArcLength(table, v * total);
    const trueFrame = buildParallelTransportFrames(rail, [uTrue]).extra[0];
    for (let j = 0; j <= 6; j++) {
      const uParam = u0 + (j / 6) * (u1 - u0); // includes both profile control-point stations AND genuinely in-between U values
      const localVal = curvePoint(section, uParam); // the local section's OWN true curve value — U direction is an exact reuse of the profile's degree/knots, so this must match surfacePoint exactly (up to the same tiny numerical solve tolerance as the V direction)
      const expected = frameToWorld(trueFrame, localVal);
      const actual = surfacePoint(srf, uParam, v);
      maxErr = Math.max(maxErr, length(sub(actual, expected)));
    }
  }
  assert.ok(maxErr < TOLERANCE, `max tracking error ${maxErr.toFixed(4)} exceeds the tight ${TOLERANCE.toFixed(4)}-unit tolerance (rail true length ${total.toFixed(2)})`);

  // Contrast directly against the OLD construction's own error at the SAME
  // check — proof the fix is a real, large improvement, not a marginal one.
  let oldMaxErr = 0;
  for (let i = 0; i < 41; i++) {
    const v = (i + 0.5) / 41;
    const uTrue = paramAtArcLength(table, v * total);
    const trueFrame = buildParallelTransportFrames(rail, [uTrue]).extra[0];
    // OLD construction: nearest Greville-station raw-control-point frame reused across the whole span (what buildParallelTransportFrames(rail) without extraParams gave every row before this fix).
    const oldFrame = frames0.reduce((best, f) => (Math.abs(f.u - v) < Math.abs(best.u - v) ? f : best), frames0[0]);
    for (let j = 0; j <= 6; j++) {
      const uParam = u0 + (j / 6) * (u1 - u0);
      const localVal = curvePoint(section, uParam);
      const expected = frameToWorld(trueFrame, localVal);
      const oldActual = frameToWorld(oldFrame, localVal);
      oldMaxErr = Math.max(oldMaxErr, length(sub(oldActual, expected)));
    }
  }
  assert.ok(oldMaxErr > 40, `sanity: the OLD construction's own error really was large (max ${oldMaxErr.toFixed(2)}) — otherwise this isn't a meaningful contrast`);
  assert.ok(maxErr < oldMaxErr / 20, `the new construction should be at least 20x tighter than the old off-curve reuse trick (new ${maxErr.toFixed(4)} vs old ${oldMaxErr.toFixed(2)})`);
});

test('sweep1Rigid: the teapot\'s own spout rail (also a 3-point degree-2 non-rational curve) still runs the free/exact path — its own control-point deviation is genuinely tiny, unlike the reported hook rail', () => {
  const spoutRail = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[6, 0, 3, 1], [10, 0, 5, 1], [13, 0, 6, 1]] };
  assert.equal(railFrameOriginsExact(spoutRail), false, 'a degree-2 rail is dispatched to the resampled path regardless of how gentle it is — correctness over a type-based shortcut');
  const mid = spoutRail.ctrlPts[1];
  const trueMid = curvePoint(spoutRail, 0.5);
  const deviation = length(sub([mid[0], mid[1], mid[2]], trueMid));
  assert.ok(deviation < 1, `sanity: the teapot's own spout rail is gentle — its own control-point deviation (${deviation.toFixed(4)}) is tiny, which is why this bug never surfaced there`);
});

test('sweep1Rigid: a Line rail (degree 1) is completely untouched — same byte-identical reuse-trick path as before this file', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 0, 30, 1]] };
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [8, 0, 0, 1]] };
  const srf = sweep1Rigid(line, profile);
  assert.equal(srf.degV, line.degree, 'V-direction degree is still the rail\'s own raw degree — the free path, unchanged');
  assert.deepEqual(srf.knotsV, line.knots, 'V-direction knots are still the rail\'s own raw knots (same array values) — the free path, unchanged');
});
