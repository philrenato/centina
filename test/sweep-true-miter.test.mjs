// True miter for 'sharp' corners. Sharp stays an option and stays the
// default.
//
// A sharp corner's ring mitered to the bisector tangent makes a ruled span
// between two non-parallel same-radius circles, and that span contracts
// mid-segment — 4.6194mm on a nominal 5mm pipe at a 90-degree corner, a 7.6%
// error. The 'rounded' corner style avoids it by filleting the rail before
// sweeping (test/fillet-open-polyline.test.mjs); this file covers 'sharp'.
//
// The mechanism (kernel/sweep.mjs's `railInteriorCorners`/`applyTrueMiterStretch`/
// `applyMiterLimitFallback`, all exercised here through the public
// `sweep1Rigid` entry point only — this file deliberately never pokes at
// ctrlNet internals directly, matching this kernel's own "measure real
// geometry, don't assert on internals" discipline): at a corner of turn
// angle theta, the geometrically correct shared cross-section is the
// ellipse the bisector plane cuts from either adjacent segment's own true
// cylinder — unchanged radius perpendicular to the bend plane, stretched by
// sec(theta/2) along the bend-plane direction (the "Stretch Around Turns"
// behavior documented for Rhino and for Houdini's Sweep 2.0, where it is on
// by default). A degenerate near-fold-back corner (sec(theta/2) exceeding
// `PIPE_MITER_LIMIT`, Houdini's own "Max Stretch" default of 10) falls back
// to a small faceted rounded fillet for just that one corner instead,
// reusing `filletOpenPolyline`'s own per-corner selection machinery
// (through its `cornerFilter` option — see kernel/primitives.mjs).
//
// Measurement: every test below measures the real swept surface via
// `surfacePoint`, never asserts on ctrlNet/frame internals directly. The
// single 90-degree planar L-corner fixture (the same rail as
// test/fillet-open-polyline.test.mjs) measures exactly 5.000000mm,
// hand-derivable — but a general, non-planar, multi-corner rail (two
// different-angle corners sandwiching one shared middle segment) measures a
// real, small residual (~0.035mm, ~0.7%, at the worst sampled station) — the
// "~1e-3 level" residual expected from the rational circle's off-curve
// control points interacting with the stretch.

import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle } from '../kernel/primitives.mjs';
import { sweep1Rigid, railInteriorCorners, PIPE_MITER_LIMIT } from '../kernel/sweep.mjs';
import { isFiniteNet } from '../kernel/surface.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { curvePoint, buildArcLengthTable, paramAtArcLength, isCurveClosed } from '../kernel/curve.mjs';

function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function minMaxRadiusAt(srf, center, v, uSampleCount = 48) {
  const uSpan = srf.knotsU[srf.knotsU.length - 1] - srf.knotsU[0];
  let minD = Infinity, maxD = 0;
  for (let k = 0; k < uSampleCount; k++) {
    const u = srf.knotsU[0] + (k / uSampleCount) * uSpan;
    const d = dist(surfacePoint(srf, u, v), center);
    minD = Math.min(minD, d);
    maxD = Math.max(maxD, d);
  }
  return { minD, maxD };
}
function openPolylineRail(points) {
  const ctrlPts = points.map(([x, y, z]) => [x, y, z, 1]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}
function closedPolylineRail(points) {
  const ctrlPts = points.map(([x, y, z]) => [x, y, z, 1]);
  ctrlPts.push(ctrlPts[0]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}

test('railInteriorCorners: a collinear rail has no corners at all', () => {
  const rail = openPolylineRail([[0, 0, 0], [50, 0, 0], [100, 0, 0], [150, 0, 0]]);
  assert.equal(railInteriorCorners(rail).length, 0);
});

test('railInteriorCorners: a real 90-degree L-corner reports theta=90deg and stretch=sec(45deg) exactly', () => {
  const rail = openPolylineRail([[0, 0, 0], [10, 0, 0], [10, 10, 0]]);
  const corners = railInteriorCorners(rail);
  assert.equal(corners.length, 1);
  assert.equal(corners[0].ringIndices.length, 1);
  assert.equal(corners[0].ringIndices[0], 1);
  assert.ok(Math.abs(corners[0].theta - Math.PI / 2) < 1e-9, `theta should be exactly 90deg (measured ${(corners[0].theta * 180 / Math.PI).toFixed(4)})`);
  assert.ok(Math.abs(corners[0].stretch - 1 / Math.cos(Math.PI / 4)) < 1e-9, `stretch should be exactly sec(45deg) = ${(1 / Math.cos(Math.PI / 4)).toFixed(6)} (measured ${corners[0].stretch.toFixed(6)})`);
});

test('True miter: sweep1Rigid on the default (sharp, unrounded) 90-degree L-corner rail measures exactly 5.0000mm mid-span, not the 4.6194mm of an unstretched miter, with no rounding applied', () => {
  const trueRadius = 5;
  const rail = openPolylineRail([[0, 0, 0], [10, 0, 0], [10, 10, 0]]); // the same fixture as fillet-open-polyline.test.mjs
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], trueRadius);
  const srf = sweep1Rigid(rail, profile);
  assert.equal(srf.degV, 1, 'stays on the free (discrete) path — the true miter never rounds the rail, corners stay crisp');
  for (const segStart of [0, 1]) {
    const { minD } = minMaxRadiusAt(srf, curvePoint(rail, segStart + 0.5), segStart + 0.5);
    assert.ok(Math.abs(minD - trueRadius) < 1e-6, `segment ${segStart} mid-span min radius measured ${minD.toFixed(6)}, expected exactly ${trueRadius} (an unstretched miter gives 4.6194 = cos(22.5deg)*5)`);
  }
});

test('The ellipse, proven not assumed: the corner ring itself is a real ellipse — min radius (perpendicular to the bend) stays exactly the true radius, max radius (along the bend) grows to exactly radius*sec(theta/2)', () => {
  const trueRadius = 5;
  const rail = openPolylineRail([[0, 0, 0], [10, 0, 0], [10, 10, 0]]);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], trueRadius);
  const srf = sweep1Rigid(rail, profile);
  const { minD, maxD } = minMaxRadiusAt(srf, curvePoint(rail, 1), 1);
  assert.ok(Math.abs(minD - trueRadius) < 1e-6, `corner min radius should stay exactly ${trueRadius} (measured ${minD.toFixed(6)})`);
  const expectedMax = trueRadius / Math.cos(Math.PI / 4);
  assert.ok(Math.abs(maxD - expectedMax) < 1e-6, `corner max radius should be exactly radius*sec(45deg) = ${expectedMax.toFixed(6)} (measured ${maxD.toFixed(6)}) — proves the ring is a real, deliberate ellipse, not silently still a circle`);
});

test('an ordinary straight rail (no corners at all) is completely byte-identical to plain sweep1Rigid — applyMiterLimitFallback/applyTrueMiterStretch are true no-ops here', () => {
  const rail = openPolylineRail([[0, 0, 0], [50, 0, 0], [100, 0, 0], [150, 0, 0]]);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const srf = sweep1Rigid(rail, profile);
  assert.equal(srf.degV, rail.degree);
  assert.deepEqual(srf.knotsV, rail.knots, 'V knots byte-identical to the original rail knots — applyMiterLimitFallback returned the same rail object unchanged, not a rebuilt one');
  for (let i = 0; i < rail.ctrlPts.length; i++) {
    const { minD, maxD } = minMaxRadiusAt(srf, curvePoint(rail, i), i);
    assert.ok(Math.abs(minD - 5) < 1e-9 && Math.abs(maxD - 5) < 1e-9, `station ${i}: still a plain circle (min===max===5), never stretched — no corner exists here`);
  }
});

test('Closed pentagon rail: the seam weld and every interior true-miter stretch coexist correctly — still a zero-gap closed tube', () => {
  const PENTAGON = [[-80, 40, 0], [0, 90, 0], [90, 30, 0], [50, -70, 0], [-70, -60, 0]];
  const rail = closedPolylineRail(PENTAGON);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const srf = sweep1Rigid(rail, profile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  const firstRing = srf.ctrlNet.map((row) => row[0]);
  const lastRing = srf.ctrlNet.map((row) => row[row.length - 1]);
  let maxRingDiff = 0;
  for (let i = 0; i < firstRing.length; i++) maxRingDiff = Math.max(maxRingDiff, dist(firstRing[i].slice(0, 3), lastRing[i].slice(0, 3)));
  assert.ok(maxRingDiff < 1e-9, `the seam still closes with zero gap after the true-miter stretch is applied to it (measured max diff ${maxRingDiff})`);
});

test('Miter-limit fallback, required not optional: a near-180-degree fold-back corner exceeds PIPE_MITER_LIMIT and falls back to a small faceted fillet — stays finite, no NaN/Infinity, and the rail gained extra points (the fallback fired)', () => {
  const rail = openPolylineRail([[0, 0, 0], [100, 0, 0], [0.5, 3, 0]]); // ~178-degree turn — a near-exact fold-back
  const corners = railInteriorCorners(rail);
  assert.equal(corners.length, 1);
  assert.ok(corners[0].stretch > PIPE_MITER_LIMIT, `this fixture's own stretch (${corners[0].stretch.toFixed(2)}) must exceed PIPE_MITER_LIMIT (${PIPE_MITER_LIMIT}) — otherwise this test isn't testing the fallback at all`);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const srf = sweep1Rigid(rail, profile);
  assert.equal(srf.degV, 1, 'the fallback stays degree-1 (a faceted polyline substitute, not a rational arc) — it never forces the rail through the resampled path');
  assert.ok(srf.knotsV.length > rail.knots.length, `the effective rail gained extra V-stations near the fold-back corner (fallback fired) — original had ${rail.knots.length} knots, swept surface has ${srf.knotsV.length}`);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'stays fully finite even at this extreme fold-back corner — never NaN/Infinity');
});

test('Mixed rail: one sane 90-degree corner true-mitered and one real near-fold-back corner fillet-fallback, on the same rail — both regions measure the correct uniform radius', () => {
  const trueRadius = 5;
  // [0,0,0]->[50,0,0]->[50,50,0]: a sane 90deg corner (index 1).
  // [50,50,0]->[50.5,47,0]: reverses back almost exactly along -Y — a real near-180 fold-back corner (index 2).
  const rail = openPolylineRail([[0, 0, 0], [50, 0, 0], [50, 50, 0], [50.5, 47, 0]]);
  const corners = railInteriorCorners(rail);
  assert.equal(corners.length, 2);
  assert.ok(corners[0].stretch <= PIPE_MITER_LIMIT, 'corner 1 (the sane 90deg turn) should be true-mitered, not fallback');
  assert.ok(corners[1].stretch > PIPE_MITER_LIMIT, 'corner 2 (the near-fold-back turn) should exceed the miter limit and fall back');

  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], trueRadius);
  const srf = sweep1Rigid(rail, profile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  assert.equal(srf.degV, 1, 'the whole mixed rail stays degree-1 — the fallback never elevates the rail as a whole');

  const table = buildArcLengthTable(rail, rail.knots[0], rail.knots[rail.knots.length - 1]);
  // Sample well clear of both corners: near the start (before corner 1),
  // and in the middle segment between corner 1 (sane) and corner 2 (fallback).
  for (const frac of [0.1, 0.3, 0.45]) {
    const u = paramAtArcLength(table, frac * table.total);
    const center = curvePoint(rail, u);
    const { minD } = minMaxRadiusAt(srf, center, u);
    assert.ok(Math.abs(minD - trueRadius) < 1e-4, `mixed rail at fraction ${frac}: measured min radius ${minD.toFixed(6)}, expected ~${trueRadius} (both the true-mitered corner and the fallback-filleted corner should compose correctly here)`);
  }
});

test('Variable radius composes correctly with the true-miter stretch: the corner\'s own interpolated radius (not the base radius) becomes the ellipse\'s minor axis, and the major axis scales with it too', () => {
  const rail = openPolylineRail([[0, 0, 0], [10, 0, 0], [10, 10, 0]]);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const radiusOpts = { radiusPoints: [{ t: 0, radius: 5 }, { t: 1, radius: 10 }], baseRadius: 5 };
  const srf = sweep1Rigid(rail, profile, radiusOpts);
  const { minD, maxD } = minMaxRadiusAt(srf, curvePoint(rail, 1), 1);
  const expectedRadiusAtCorner = 7.5; // t=0.5 (the corner sits halfway along this 2-segment rail's domain), linearly interpolated between 5 and 10
  assert.ok(Math.abs(minD - expectedRadiusAtCorner) < 1e-4, `corner min radius (ellipse minor axis) should be the INTERPOLATED radius-at-t ${expectedRadiusAtCorner} (measured ${minD.toFixed(4)}), not the base radius 5`);
  const expectedMax = expectedRadiusAtCorner / Math.cos(Math.PI / 4);
  assert.ok(Math.abs(maxD - expectedMax) < 1e-4, `corner max radius (ellipse major axis) should be the interpolated radius times sec(45deg) = ${expectedMax.toFixed(4)} (measured ${maxD.toFixed(4)})`);
});

test('cornerStyle "rounded" (an already-filleted, degree>1 rail) is unaffected by the true miter — no interior corners left for this mechanism to touch; it dispatches straight to the resampled path', async () => {
  const { filletOpenPolyline, filletSegmentsToCurve } = await import('../kernel/primitives.mjs');
  const railPts = [[0, 0, 0], [10, 0, 0], [10, 10, 0]];
  const filletRes = filletOpenPolyline(railPts, 3, { closed: false });
  assert.equal(filletRes.ok, true);
  const filletedRail = filletSegmentsToCurve(filletRes.segments);
  assert.ok(filletedRail.degree > 1, 'a rounded rail is degree>1');
  assert.equal(railInteriorCorners(filletedRail).length, 0, 'railInteriorCorners is only ever meaningful on a degree<=1 rail\'s own raw control points — a rounded rail has none of the kind this mechanism looks for');
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const srf = sweep1Rigid(filletedRail, profile);
  assert.ok(srf.degV >= 2, 'still dispatches to sweep1RigidResampled, exactly as before this file — completely untouched code path');
});

test('Residual, measured not assumed: a general non-planar rail with two different-angle corners sandwiching one shared middle segment shows a real, small residual — not claimed bit-exact where the true miter\'s "axis-direction correspondence" assumption is less clean than in the single-corner case', () => {
  const trueRadius = 5;
  const rail = openPolylineRail([[0, 0, 0], [20, 0, 0], [20, 15, 8], [5, 25, 20]]); // non-planar zigzag: corner 1 (90deg) and corner 2 (~48deg), sharing the middle segment
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], trueRadius);
  const srf = sweep1Rigid(rail, profile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  const table = buildArcLengthTable(rail, rail.knots[0], rail.knots[rail.knots.length - 1]);
  let worstMin = Infinity;
  for (let i = 1; i < 100; i++) {
    const frac = i / 100;
    const u = paramAtArcLength(table, frac * table.total);
    const { minD } = minMaxRadiusAt(srf, curvePoint(rail, u), u);
    worstMin = Math.min(worstMin, minD);
  }
  // Measured: worst-case min radius on this fixture is ~4.9648mm at a 1/200
  // sampling (a ~0.035mm / ~0.7% residual) — about 11x smaller than the 7.6%
  // (0.38mm on 5mm) of an unstretched miter, and not zero.
  assert.ok(worstMin > trueRadius * 0.99, `worst-case measured min radius ${worstMin.toFixed(6)} should be within ~1% of the true radius ${trueRadius} — a real, small-tolerated residual on this harder (non-planar, two-corners-sharing-a-segment) fixture, not the 7.6% waist of an unstretched miter`);
  assert.ok(worstMin < trueRadius, 'the residual is a real (if small) undershoot, not overshoot — reported, not rounded away');
});
