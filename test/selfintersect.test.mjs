// PROFILE SELF-INTERSECTION — Phase C of the self-intersection guards,
// planar tier and the profile-validity consumer it names as the highest-value
// application of it.
//
// The gate stated for this phase, tested literally:
//   "a figure-eight planar SketchCurve is caught; a non-planar curve merely
//    passing near itself is NOT reported as intersecting."
//
// Every fixture is built from coordinates whose crossing (or lack of one) is
// obvious by inspection, so a failure here is a failure of the code and never
// an argument about what the fixture was supposed to be.

import test from 'node:test';
import assert from 'node:assert/strict';
import { curveSelfIntersects, polylineSelfIntersects2D, bestFitPlane } from '../kernel/selfintersect.mjs';
import { makeCircle } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';

const X = [1, 0, 0], Y = [0, 1, 0], O = [0, 0, 0];

// A degree-1 curve through the given points — the exact shape a Polyline
// profile reaches `getProfileCrv` as, so these fixtures are the real thing
// rather than a stand-in.
function polylineCrv(points, closed) {
  const pts = closed ? [...points, points[0]] : points;
  const ctrlPts = pts.map((p) => [...p, 1]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}

// -------------------------------------------------------- THE 2D PRIMITIVE

test('polylineSelfIntersects2D respects open vs closed rather than assuming a loop', () => {
  // Three sides of a square, OPEN. There is no crossing — but a loop-assuming
  // test would invent the closing segment and could report one.
  const openL = [[0, 0], [10, 0], [10, 10], [0, 10]];
  assert.equal(polylineSelfIntersects2D(openL, false), false);
  assert.equal(polylineSelfIntersects2D(openL, true), false, 'a plain square is clean closed too');

  // A genuine planar figure-eight (a bow-tie), the canonical crossing case.
  const bowtie = [[0, 0], [10, 10], [10, 0], [0, 10]];
  assert.equal(polylineSelfIntersects2D(bowtie, true), true);
  assert.equal(polylineSelfIntersects2D(bowtie, false), true, 'the crossing is between two INTERIOR segments, so it is there open too');
});

test('a closed polyline is not reported as crossing itself at its own seam', () => {
  // The one-directional-adjacency bug this guards: on a closed ring the FIRST
  // and LAST segments are adjacent, and excluding only j === i+1 would report
  // every closed curve in the app as self-intersecting.
  for (const n of [3, 4, 5, 8, 16]) {
    const ring = [];
    for (let i = 0; i < n; i++) ring.push([Math.cos((2 * Math.PI * i) / n) * 20, Math.sin((2 * Math.PI * i) / n) * 20]);
    assert.equal(polylineSelfIntersects2D(ring, true), false, `a regular ${n}-gon must be clean`);
  }
});

// ---------------------------------------------------------------- PLANES

test('bestFitPlane refuses a collinear ring rather than normalizing noise', () => {
  assert.equal(bestFitPlane([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]), null);
});

// THE REASON bestFitPlane DOES NOT USE NEWELL'S METHOD, pinned as a test so
// a future "simplify this to the same technique capProfilePlaneCheck uses"
// fails loudly instead of silently disabling the guard.
//
// Newell's sums SIGNED per-edge contributions — an area-weighted normal. A
// figure-eight's two lobes wind in OPPOSITE directions, so they cancel and
// the normal comes out zero-length. Newell's therefore reports the canonical
// self-intersecting profile as having no plane at all, and the whole guard
// declines to judge the one shape it exists for.
test('Newell\'s method cancels to zero on a figure-eight — bestFitPlane must not use it', () => {
  const bowtie = [[0, 0, 0], [40, 40, 0], [40, 0, 0], [0, 40, 0]];
  const newell = [0, 0, 0];
  for (let i = 0; i < bowtie.length; i++) {
    const a = bowtie[i], b = bowtie[(i + 1) % bowtie.length];
    newell[0] += (a[1] - b[1]) * (a[2] + b[2]);
    newell[1] += (a[2] - b[2]) * (a[0] + b[0]);
    newell[2] += (a[0] - b[0]) * (a[1] + b[1]);
  }
  const newellLen = Math.hypot(...newell);
  assert.ok(newellLen < 1e-9, `Newell's must genuinely cancel here for this test to mean anything (got ${newellLen})`);

  // The shipped method finds the plane regardless — this is the whole point.
  const plane = bestFitPlane(bowtie);
  assert.ok(plane, 'bestFitPlane must still find the plane a figure-eight obviously lies in');
  assert.ok(Math.abs(Math.abs(plane.normal[2]) - 1) < 1e-9, 'and it is the Z plane, as drawn');
});

test('bestFitPlane recovers a known tilted plane exactly', () => {
  // A square lying in the plane spanned by X and (Y+Z)/sqrt(2): its normal is
  // (0, -1, 1)/sqrt(2) up to sign, known independently of the function.
  const s = Math.SQRT1_2;
  const pts = [[0, 0, 0], [10, 0, 0], [10, 10 * s, 10 * s], [0, 10 * s, 10 * s]];
  const plane = bestFitPlane(pts);
  assert.ok(plane);
  const expected = [0, -s, s];
  const d = Math.abs(plane.normal[0] * expected[0] + plane.normal[1] * expected[1] + plane.normal[2] * expected[2]);
  assert.ok(Math.abs(d - 1) < 1e-9, `normal should be parallel to the known one, |dot| = ${d}`);
});

// ------------------------------------------------------- THE STATED GATE

test('GATE: a planar figure-eight profile is caught', () => {
  const crv = polylineCrv([[0, 0, 0], [40, 40, 0], [40, 0, 0], [0, 40, 0]], true);
  const r = curveSelfIntersects(crv);
  assert.equal(r.tested, true);
  assert.equal(r.planar, true);
  assert.equal(r.selfIntersects, true);
});

test('GATE: a non-planar curve merely passing near itself is NOT reported as intersecting', () => {
  // The SAME figure-eight footprint, but lifted so the two branches pass at
  // genuinely different heights — a real 3D crossing-over, not a crossing.
  // The lift is large relative to the curve's own size, so this is
  // unambiguously a space curve, not a near-planar one.
  const crv = polylineCrv([[0, 0, 0], [40, 40, 30], [40, 0, 0], [0, 40, -30]], true);
  const r = curveSelfIntersects(crv);
  assert.equal(r.selfIntersects, false, 'must not claim an intersection');
  assert.equal(r.tested, false, 'and must say plainly that it did not judge it, rather than reporting a clean pass');
  assert.equal(r.planar, false);
  assert.match(r.reason, /not planar/);
});

// ----------------------------------------------------- ORDINARY PROFILES
// The regression half: every profile a student actually draws must stay
// clean, or this guard becomes noise the moment it ships.

test('ordinary profiles are clean — a circle, a polygon, a smooth open curve', () => {
  const circle = curveSelfIntersects(makeCircle(O, X, Y, 25));
  assert.equal(circle.tested, true);
  assert.equal(circle.planar, true);
  assert.equal(circle.selfIntersects, false, 'a circle must never read as self-intersecting');

  const hex = [];
  for (let i = 0; i < 6; i++) hex.push([Math.cos((2 * Math.PI * i) / 6) * 30, Math.sin((2 * Math.PI * i) / 6) * 30, 0]);
  const poly = curveSelfIntersects(polylineCrv(hex, true));
  assert.equal(poly.selfIntersects, false);

  // A smooth interpolated OPEN curve, the SketchCurve case.
  const smooth = globalCurveInterp([[0, 0, 0], [20, 15, 0], [40, -10, 0], [60, 5, 0]], 3);
  const sm = curveSelfIntersects(smooth);
  assert.equal(sm.tested, true);
  assert.equal(sm.selfIntersects, false);
});

test('a NEARLY-planar crossing profile is still caught — the case that matters', () => {
  // The figure-eight again, with one point nudged off-plane by well under the
  // planarity tolerance. Its exact 3D crossing is gone, but the tolerance-
  // level one that genuinely breaks Extrude/boolean/export remains — so this
  // must still be caught, not excused as "a space curve".
  const size = 40 * Math.SQRT2;
  const nudge = size * 0.002; // comfortably inside PLANARITY_TOL_FRAC (1%)
  const crv = polylineCrv([[0, 0, 0], [40, 40, nudge], [40, 0, 0], [0, 40, 0]], true);
  const r = curveSelfIntersects(crv);
  assert.equal(r.planar, true, 'a sub-tolerance nudge must not demote the curve to "not judged"');
  assert.equal(r.selfIntersects, true);
});

test('a self-crossing OPEN profile is caught without inventing a closing segment', () => {
  // An open curve whose own two interior stretches genuinely cross.
  const crossing = polylineCrv([[0, 0, 0], [40, 40, 0], [40, 0, 0], [0, 40, 0]], false);
  assert.equal(curveSelfIntersects(crossing).selfIntersects, true);

  // An open curve that only LOOKS like it would cross if a phantom closing
  // segment were added — three sides of a Z. The closing chord from the last
  // point back to the first would cross nothing here either, so the sharper
  // check is the C shape below.
  const cShape = polylineCrv([[0, 0, 0], [40, 0, 0], [40, 40, 0], [0, 40, 0]], false);
  assert.equal(curveSelfIntersects(cShape).selfIntersects, false);
});
