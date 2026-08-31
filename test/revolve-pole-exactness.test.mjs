import test from 'node:test';
import assert from 'node:assert/strict';
import { curvePoint } from '../kernel/curve.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { makeArc, makeCircle, revolve } from '../kernel/primitives.mjs';

// THE BUG (found while building SSI test fixtures, fixed here):
// revolve()'s old pole-row branch returned a UNIFORM per-
// column weight (the raw profile weight) instead of the SAME alternating
// arc-weight pattern (1, cos(dtheta/2), 1, ...) every other row's own
// arcSpanPoints construction uses. A8.1's exactness proof requires every
// row blended at a given U to share that IDENTICAL weight-function SHAPE
// (as a function of the sweep parameter v) — a uniform weight breaks that
// identity the instant a pole row is U-blended with its non-pole
// neighbors away from a knot corner, producing the reported ~1-5%
// relative radius error. The fix makes the pole row use the SAME
// arcSpanPoints construction as every other row, with radius=0 (any
// perpendicular basis works, since it's multiplied by zero) — this is
// what's actually under test below, not a knot-insertion-based fix (a
// strategy-review lead investigated and found insufficient on its own:
// inserting a knot at the pole's own U value would still leave every
// OTHER interior U value in the same span exposed to the identical
// weight-shape mismatch, since the mismatch is a WEIGHT-FUNCTION bug, not
// a knot-placement one).

test('revolve of a pole-to-pole sphere profile (a vertical semicircle revolved 360 deg about its own diameter) is exact at NON-KNOT-CORNER (u,v) samples, not just at knot corners', () => {
  const R = 10;
  // Semicircle profile in the X-Z half-plane, south pole -> equator -> north
  // pole (the exact "canonical case" the bug report names) — degree 2,
  // rational, control points at u=0 (south pole, r=0), u=1 (equator), u=2
  // (north pole, r=0), with real tangent-point control rows in between.
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], R, -Math.PI / 2, Math.PI);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);

  // Knot corners: U at {0,1,2}, V at {0,1,2,3,4} (arcKnots for a 4-span
  // full sweep). Sample deliberately AWAY from every one of those, in both
  // the pole-adjacent span [0,1]/[1,2] and the sweep direction.
  const uVals = [0.2, 0.4, 0.5, 0.6, 0.8, 1.2, 1.4, 1.5, 1.6, 1.8];
  const vVals = [0.3, 0.7, 1.1, 1.6, 1.9, 2.3, 2.7, 3.1, 3.6, 3.9];

  let maxRelErr = 0;
  let sampleCount = 0;
  for (const u of uVals) {
    for (const v of vVals) {
      const p = surfacePoint(srf, u, v);
      // The one, trivial, exact property of a real sphere: every point's
      // distance from the CENTER (not the axis) is exactly R.
      const r = Math.hypot(p[0], p[1], p[2]);
      const relErr = Math.abs(r - R) / R;
      maxRelErr = Math.max(maxRelErr, relErr);
      sampleCount++;
    }
  }
  assert.ok(sampleCount === uVals.length * vVals.length, 'sanity: every sample actually ran');
  assert.ok(maxRelErr < 1e-6, `max relative radius error at non-corner samples was ${maxRelErr} (expected < 1e-6, tight/exact)`);
});

test('revolve of the SAME sphere profile is (of course) also exact exactly at knot corners', () => {
  const R = 10;
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], R, -Math.PI / 2, Math.PI);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  for (const u of [0, 1, 2]) {
    for (const v of [0, 1, 2, 3, 4]) {
      const p = surfacePoint(srf, u, v);
      const r = Math.hypot(p[0], p[1], p[2]);
      assert.ok(Math.abs(r - R) < 1e-9, `corner u=${u} v=${v}: r=${r}`);
    }
  }
});

test('revolve of a profile combining a pole AND a genuine sign-change of the radial direction (crossing to the axis\'s other side) is still exact at non-corner samples', () => {
  // A degree-1 polyline profile: south pole -> +X point -> -X point (a real
  // sign flip, no pole there) -> north pole. Exercises BOTH named
  // mechanisms in one net: a true pole row (r=0, weight-shape mismatch) AND
  // a profile that changes which side of the axis it's on between two
  // ordinary (non-pole) rows.
  const p0 = [0, 0, -10, 1];
  const p1 = [10, 0, -3, 1];
  const p2 = [-6, 0, 4, 1];
  const p3 = [0, 0, 10, 1];
  const profile = { degree: 1, knots: [0, 0, 1, 2, 3, 3], ctrlPts: [p0, p1, p2, p3] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);

  function trueRadiusHeight(u) {
    const p = curvePoint(profile, u); // exact for a degree-1 curve (piecewise-linear)
    return { r: Math.hypot(p[0], p[1]), z: p[2] };
  }

  let maxErr = 0;
  for (const u of [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9]) {
    const { r: trueR, z: trueZ } = trueRadiusHeight(u);
    for (const v of [0.2, 0.9, 1.6, 2.3, 3.0, 3.7, 4.4, 5.1, 5.9]) {
      const p = surfacePoint(srf, u, v);
      const r = Math.hypot(p[0], p[1]);
      maxErr = Math.max(maxErr, Math.abs(r - trueR), Math.abs(p[2] - trueZ));
    }
  }
  assert.ok(maxErr < 1e-9, `max abs error (pole + sign-flip combined case) was ${maxErr}`);
});

test('revolve of a torus-like profile that does NOT cross or touch the axis is unaffected (regression guard for the already-working non-pole case)', () => {
  // A full circle profile offset from the axis (a torus tube), degree 2,
  // rational — no control point ever has r<1e-9, so the pole branch is
  // never triggered at all; this must remain exactly as accurate as before
  // the fix (the fix only touches the r<1e-9 branch).
  const tubeRadius = 3;
  const tubeCenterDist = 10; // profile circle's own center, offset from the Z axis
  const profile = makeCircle([tubeCenterDist, 0, 0], [1, 0, 0], [0, 0, 1], tubeRadius);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);

  const uVals = [0.3, 0.7, 1.1, 1.6, 2.3, 2.9, 3.4, 3.8];
  const vVals = [0.3, 0.9, 1.4, 1.9, 2.6, 3.1, 3.6];
  let maxErr = 0;
  for (const u of uVals) {
    // A torus's own defining symmetry: at a fixed profile parameter u, the
    // (distance-from-axis, height) pair is the SAME for every sweep v,
    // and must match the profile's own true value at that u exactly.
    const profilePt = curvePoint(profile, u); // planar profile, y=0 by construction
    const trueDistFromAxis = Math.hypot(profilePt[0], profilePt[1]);
    const trueZ = profilePt[2];
    for (const v of vVals) {
      const p = surfacePoint(srf, u, v);
      const distFromAxis = Math.hypot(p[0], p[1]);
      maxErr = Math.max(maxErr, Math.abs(trueDistFromAxis - distFromAxis), Math.abs(trueZ - p[2]));
    }
  }
  assert.ok(maxErr < 1e-9, `torus (non-pole) case max error was ${maxErr} (must remain exact, unaffected by the pole fix)`);
});
