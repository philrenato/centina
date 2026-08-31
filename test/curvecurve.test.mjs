// The Phase 5 gate is a TYPE MATRIX — one case per real curve type this
// app actually produces (line, rational arc, sketch curve, joined polycurve)
// — deliberately not four variations on a line, because a line is the one
// case where degree-1 control points ARE the curve and every step of the
// search is trivially exact. The rational and multi-span cases are where a
// subdivision/refinement scheme actually has to be right.
//
// Ground truth throughout is ANALYTIC where the fixture allows it: a line
// through a circle's center crosses the circle at exactly `radius`, and that
// number comes from the geometry, not from the function under test.

import test from 'node:test';
import assert from 'node:assert/strict';
import { intersectCurves2D } from '../kernel/curvecurve.mjs';
import { makeLine, makeCircle, makeArc } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { joinCurvesC0 } from '../kernel/knots.mjs';
import { curvePoint } from '../kernel/curve.mjs';

const X = [1, 0, 0], Y = [0, 1, 0];

// Every reported point must genuinely lie on BOTH curves at the parameters
// reported for it — checked against the curves themselves, never against the
// intersector's own returned point, which would be a tautology.
function assertOnBothCurves(crvA, crvB, hit, tol, label) {
  const a = curvePoint(crvA, hit.uA);
  const b = curvePoint(crvB, hit.uB);
  assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) < tol,
    `${label}: the two curves do not actually meet at the reported parameters (gap ${Math.hypot(a[0] - b[0], a[1] - b[1]).toExponential(3)})`);
  assert.ok(Math.hypot(a[0] - hit.point[0], a[1] - hit.point[1]) < tol,
    `${label}: the reported point is not on curve A at its own reported parameter`);
}

test('TYPE MATRIX 1 — line x line crosses at the exact analytic point', () => {
  // Deliberately not axis-aligned: an axis-aligned pair makes several
  // independent bugs (a swapped x/y, a dropped term) invisible.
  const a = makeLine([-10, -10, 0], [10, 10, 0]);
  const b = makeLine([-10, 10, 0], [10, -10, 0]);
  const r = intersectCurves2D(a, b);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.points.length, 1);
  const p = r.points[0].point;
  assert.ok(Math.abs(p[0]) < 1e-9 && Math.abs(p[1]) < 1e-9,
    `expected the crossing at the origin, got (${p[0]}, ${p[1]})`);
  assertOnBothCurves(a, b, r.points[0], 1e-9, 'line x line');
});

test('TYPE MATRIX 2 — line x RATIONAL circle gives both crossings at exactly the radius', () => {
  // The load-bearing case for the rational half of the algorithm: a circle's
  // control points are NOT on the curve (half sit at radius*sqrt(2) on the
  // tangent corners), so a scheme that forgot to divide through by w would
  // land measurably off and this test would catch it.
  const R = 25;
  const circ = makeCircle([0, 0, 0], X, Y, R);
  const line = makeLine([-60, 0, 0], [60, 0, 0]); // straight through the center
  const r = intersectCurves2D(line, circ);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.points.length, 2, `a line through the center must cross a full circle exactly twice, got ${r.points.length}`);
  for (const hit of r.points) {
    const d = Math.hypot(hit.point[0], hit.point[1]);
    assert.ok(Math.abs(d - R) < 1e-7,
      `crossing must sit at exactly the circle's radius ${R}, got ${d}`);
    assertOnBothCurves(line, circ, hit, 1e-7, 'line x circle');
  }
  // The two crossings are the genuinely opposite ends of a diameter.
  const [p0, p1] = r.points.map((h) => h.point);
  assert.ok(Math.abs(p0[0] + p1[0]) < 1e-7, 'the two crossings must be diametrically opposite');
});

test('an OFF-ORIGIN rational circle is bounded by its EUCLIDEAN control points', () => {
  // This fixture exists because the obvious one does not discriminate. A
  // circle centered at the ORIGIN survives a bug that bounds pieces by their
  // HOMOGENEOUS control points instead of their euclidean ones, because
  // multiplying by w<1 shrinks the corners toward the origin by almost exactly
  // the amount the corners overshoot — the wrong box happens to still contain
  // the curve. Move the circle away from the origin and that coincidence
  // breaks: homogeneous scaling pulls every control point toward the origin,
  // the box lands somewhere the curve is not, and the search rejects a range
  // that genuinely contains a crossing.
  //
  // The convex hull property is stated for the EUCLIDEAN control points, so
  // that is the only bound a rejection may be proven with.
  const C = [100, 60, 0], R = 25;
  const circ = makeCircle(C, X, Y, R);
  // Crosses well off the center line, so neither crossing lands on a control
  // point — a crossing that coincides with one is found even by a broken box.
  const yCut = C[1] + 0.6 * R;
  const line = makeLine([C[0] - 3 * R, yCut, 0], [C[0] + 3 * R, yCut, 0]);
  const r = intersectCurves2D(line, circ);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.points.length, 2, `expected 2 crossings on an off-origin circle, got ${r.points.length}`);
  for (const hit of r.points) {
    const d = Math.hypot(hit.point[0] - C[0], hit.point[1] - C[1]);
    assert.ok(Math.abs(d - R) < 1e-7, `crossing must sit at the circle's own radius, got ${d}`);
    assertOnBothCurves(line, circ, hit, 1e-7, 'off-origin circle');
  }
});

test('TYPE MATRIX 3 — sketch curve (degree 3, multi-span) x line', () => {
  // globalCurveInterp through 5 points gives a genuinely multi-span curve with
  // real interior knots, so decomposeToBezier produces several pieces and the
  // piece-pairing loop is actually exercised rather than short-circuited.
  const sketch = globalCurveInterp([
    [-40, 0, 0], [-20, 30, 0], [0, -30, 0], [20, 30, 0], [40, 0, 0],
  ], 3);
  const line = makeLine([-50, 0, 0], [50, 0, 0]);
  const r = intersectCurves2D(sketch, line);
  assert.equal(r.ok, true, r.reason);
  // The curve is interpolatory, so it passes through (-40,0) and (40,0)
  // exactly (both endpoints, on the line) and weaves across y=0 in between.
  assert.ok(r.points.length >= 3,
    `a curve weaving across the line should cross it at least three times, got ${r.points.length}`);
  for (const hit of r.points) {
    assert.ok(Math.abs(hit.point[1]) < 1e-6,
      `every crossing with y=0 must have y=0, got ${hit.point[1]}`);
    assertOnBothCurves(sketch, line, hit, 1e-6, 'sketch x line');
  }
});

test('TYPE MATRIX 4 — joined MIXED-DEGREE polycurve x line', () => {
  // A joined chain of a straight segment and a rational arc: degree-elevated
  // to a common degree by joinCurvesC0, so the resulting curve is rational in
  // some spans and (elevated) straight in others. This is the shape a filleted
  // profile actually has in this app.
  const seg = makeLine([-40, 20, 0], [0, 20, 0]);
  const arc = makeArc([0, 0, 0], X, Y, 20, Math.PI / 2, -Math.PI / 2);
  const joined = joinCurvesC0([seg, arc]);
  const line = makeLine([-50, 10, 0], [50, 10, 0]);
  const r = intersectCurves2D(joined, line);
  assert.equal(r.ok, true, r.reason);
  assert.ok(r.points.length >= 1, 'the arc descends through y=10 and must be found');
  for (const hit of r.points) {
    assert.ok(Math.abs(hit.point[1] - 10) < 1e-6,
      `every crossing with y=10 must have y=10, got ${hit.point[1]}`);
    assertOnBothCurves(joined, line, hit, 1e-6, 'polycurve x line');
    // On the arc portion the crossing must also sit on the true circle.
    if (hit.point[0] > 1e-6) {
      const d = Math.hypot(hit.point[0], hit.point[1]);
      assert.ok(Math.abs(d - 20) < 1e-6, `an arc-portion crossing must sit on the arc's own radius, got ${d}`);
    }
  }
});

test('curves that genuinely do not meet return an EMPTY answer, not a refusal', () => {
  // ok:true with zero points is a real answer. Collapsing it into a refusal
  // would make "no intersection" indistinguishable from "could not tell".
  const a = makeLine([-10, 0, 0], [10, 0, 0]);
  const b = makeLine([-10, 50, 0], [10, 50, 0]);
  const r = intersectCurves2D(a, b);
  assert.equal(r.ok, true);
  assert.equal(r.points.length, 0);
  assert.equal(r.reason, undefined);
});

test('a shared endpoint is a real intersection, not an error', () => {
  // Trim loops are built from chains that meet exactly at their endpoints, so
  // this is the single most common case the face-splitting caller will hit.
  const a = makeLine([-20, 0, 0], [0, 0, 0]);
  const b = makeLine([0, 0, 0], [0, 20, 0]);
  const r = intersectCurves2D(a, b);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.points.length, 1);
  assert.ok(Math.hypot(r.points[0].point[0], r.points[0].point[1]) < 1e-9);
});

test('a TANGENTIAL meeting is refused by name rather than given a number', () => {
  // A line grazing a circle at its top: they touch at exactly one point but do
  // not cross. The crossing angle is zero, so no transversal intersection is
  // defined and the honest output is a named refusal.
  const R = 25;
  const circ = makeCircle([0, 0, 0], X, Y, R);
  const tangent = makeLine([-40, R, 0], [40, R, 0]);
  const r = intersectCurves2D(tangent, circ);
  assert.equal(r.ok, false, 'a tangency must not be reported as a clean transversal crossing');
  assert.equal(r.tangential, true);
  assert.match(r.reason, /tangential/i);
});

test('an OVERLAP is refused by name rather than returning an arbitrary point', () => {
  // A curve against itself shares its whole length. Every leaf survives
  // subdivision, which is exactly the signature this case is detected by.
  const a = makeLine([-30, -10, 0], [30, 40, 0]);
  const r = intersectCurves2D(a, a);
  assert.equal(r.ok, false);
  assert.equal(r.overlapping, true);
  assert.match(r.reason, /share a region|overlap/i);
});

test('a NON-POSITIVE weight is refused, because it breaks the hull proof', () => {
  // The convex hull rejection is the only reason a discarded parameter range
  // is safe to discard. A negative weight makes the curve leave its control
  // points' hull, so the search would prove nothing — refused rather than run.
  const bad = makeLine([-10, 0, 0], [10, 0, 0]);
  bad.ctrlPts = bad.ctrlPts.map((p) => p.slice());
  bad.ctrlPts[1][3] = -1;
  const r = intersectCurves2D(bad, makeLine([0, -10, 0], [0, 10, 0]));
  assert.equal(r.ok, false);
  assert.match(r.reason, /weight/i);
});

test('the result is INDEPENDENT of argument order', () => {
  // Swapping the operands must return the same crossings with uA/uB swapped —
  // an order-dependent answer would mean one curve is being treated as
  // privileged somewhere in the subdivision or the refinement.
  const R = 25;
  const circ = makeCircle([0, 0, 0], X, Y, R);
  const line = makeLine([-60, 7, 0], [60, 7, 0]);
  const fwd = intersectCurves2D(line, circ);
  const rev = intersectCurves2D(circ, line);
  assert.equal(fwd.ok && rev.ok, true);
  assert.equal(fwd.points.length, rev.points.length);
  const key = (pts) => pts.map((h) => `${h.point[0].toFixed(9)},${h.point[1].toFixed(9)}`).sort().join('|');
  assert.equal(key(fwd.points), key(rev.points),
    'the same two curves must produce the same crossing points regardless of which is passed first');
});

test('accuracy does not depend on the fixture SCALE', () => {
  // The same configuration at 1000x must be solved to the same RELATIVE
  // accuracy. This is what proves the tolerances are genuinely relative rather
  // than absolute constants that happen to suit one fixture's size.
  for (const R of [0.5, 25, 25000]) {
    const circ = makeCircle([0, 0, 0], X, Y, R);
    const line = makeLine([-3 * R, 0, 0], [3 * R, 0, 0]);
    const r = intersectCurves2D(line, circ);
    assert.equal(r.ok, true, `R=${R}: ${r.reason}`);
    assert.equal(r.points.length, 2, `R=${R}: expected 2 crossings`);
    for (const hit of r.points) {
      const rel = Math.abs(Math.hypot(hit.point[0], hit.point[1]) - R) / R;
      assert.ok(rel < 1e-9, `R=${R}: relative radius error ${rel.toExponential(3)} is scale-dependent`);
    }
  }
});
