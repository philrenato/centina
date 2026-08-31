import test from 'node:test';
import assert from 'node:assert/strict';
import { surfacePoint, surfacePointAndPartials, closestPointOnSurface, surfaceClosure, wrapParam } from '../kernel/surface.mjs';
import { makeArc, revolve, extrude, makeLine, makeCircle } from '../kernel/primitives.mjs';

// The exact torus construction test/isocurve.test.mjs's own "zero naked
// edges" test already uses — closed in BOTH u and v, confirmed there via
// surfaceClosure directly, reused here rather than re-derived.
function makeTorusSrf() {
  const profile = makeCircle([5, 0, 0], [0, 0, 1], [1, 0, 0], 1, 4);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}

// Same bilinear fixture dividesrf.test.mjs already uses — a genuinely
// simple, independently-checkable surface (degree-1 in both directions,
// so surfacePoint is exactly the bilinear-blend formula).
const bilinearSrf = {
  degU: 1, degV: 1,
  knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
  ctrlNet: [
    [[0, 0, 0, 1], [0, 10, 0, 1]],
    [[10, 0, 5, 1], [10, 10, 5, 1]],
  ],
};

// Same fully-RATIONAL stress fixture isocurve.test.mjs/dividesrf.test.mjs
// already use (a 270deg-by-270deg revolve) — non-[0,1] domain in BOTH
// directions, so a test that only happened to pass for an already-[0,1]
// domain would not be a real proof.
function makeStressSurface() {
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 5, 0, 1.5 * Math.PI);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 1.5 * Math.PI);
}

test('surfacePointAndPartials point matches plain surfacePoint exactly at several (u,v)', () => {
  const srf = makeStressSurface();
  // The surface's own real domain is [0,3] in BOTH directions (3 arc
  // spans of 90deg each, per arcSpanPoints' own MAX_ARC_SPAN convention)
  // — NOT the raw 1.5*Math.PI radian sweep angle, which is a completely
  // different number. Derived from the real knot vector, not assumed.
  const uMax = srf.knotsU[srf.knotsU.length - 1];
  for (const [u, v] of [[0.3, 0.7], [1.1, 2.9], [0, 0], [uMax, uMax]]) {
    const direct = surfacePoint(srf, u, v);
    const { point } = surfacePointAndPartials(srf, u, v);
    const d = Math.hypot(point[0] - direct[0], point[1] - direct[1], point[2] - direct[2]);
    assert.ok(d < 1e-9, `surfacePointAndPartials point diverges from surfacePoint by ${d} at u=${u},v=${v}`);
  }
});

test('surfacePointAndPartials partials match a finite-difference estimate', () => {
  const srf = makeStressSurface();
  const u = 0.6, v = 1.3, h = 1e-5;
  const { su, sv } = surfacePointAndPartials(srf, u, v);
  const pU0 = surfacePoint(srf, u - h, v), pU1 = surfacePoint(srf, u + h, v);
  const fdU = [(pU1[0] - pU0[0]) / (2 * h), (pU1[1] - pU0[1]) / (2 * h), (pU1[2] - pU0[2]) / (2 * h)];
  const pV0 = surfacePoint(srf, u, v - h), pV1 = surfacePoint(srf, u, v + h);
  const fdV = [(pV1[0] - pV0[0]) / (2 * h), (pV1[1] - pV0[1]) / (2 * h), (pV1[2] - pV0[2]) / (2 * h)];
  const errU = Math.hypot(su[0] - fdU[0], su[1] - fdU[1], su[2] - fdU[2]);
  const errV = Math.hypot(sv[0] - fdV[0], sv[1] - fdV[1], sv[2] - fdV[2]);
  assert.ok(errU < 1e-4, `Su diverges from finite-difference estimate by ${errU}`);
  assert.ok(errV < 1e-4, `Sv diverges from finite-difference estimate by ${errV}`);
});

test('closestPointOnSurface finds the EXACT (u,v) back for a point taken directly off the surface', () => {
  const srf = makeStressSurface();
  // Kept inside the FIRST arc span (u,v in [0,1]) deliberately — this
  // profile's radius genuinely goes negative past u=1 (a 270deg arc
  // profile crossing the revolve axis's own radial line), which makes
  // the swept surface legitimately SELF-OVERLAPPING further out (a real
  // property of this stress fixture, confirmed directly: surfacePoint at
  // u=1.8 lands exactly on the same 3D point as u=0.2 for a suitably
  // chosen v, since a negative-radius point rotated by phi coincides
  // with the positive-radius point rotated by phi+180). That's a fact
  // about THIS fixture surface, not a bug in closestPointOnSurface — the
  // point/distance invariant (checked below) still holds for either
  // valid (u,v) preimage; this test's own domain choice just avoids
  // needing to reason about which preimage the coarse grid seed lands on.
  for (const [u0, v0] of [[0.2, 0.5], [0.6, 0.15], [0.05, 0.95]]) {
    const target = surfacePoint(srf, u0, v0);
    const { u, v, point, distance } = closestPointOnSurface(srf, target);
    assert.ok(distance < 1e-6, `expected ~0 distance for a point already on the surface, got ${distance}`);
    const d = Math.hypot(point[0] - target[0], point[1] - target[1], point[2] - target[2]);
    assert.ok(d < 1e-6, `returned point diverges from the real target by ${d}`);
    // The inverted (u,v) should reproduce the SAME real point even if it
    // isn't bit-identical to (u0,v0) (a surface point can, in principle,
    // have more than one (u,v) preimage only at a genuine pole — this
    // fixture has none at these interior parameters).
    const reeval = surfacePoint(srf, u, v);
    const d2 = Math.hypot(reeval[0] - target[0], reeval[1] - target[1], reeval[2] - target[2]);
    assert.ok(d2 < 1e-6, `re-evaluating the inverted (u,v) diverges from the target by ${d2}`);
  }
});

test('closestPointOnSurface on a bilinear surface finds the exact center for the true center point', () => {
  const target = surfacePoint(bilinearSrf, 0.5, 0.5); // [5, 5, 2.5]
  const { u, v, distance } = closestPointOnSurface(bilinearSrf, target);
  assert.ok(distance < 1e-9, `expected exact 0 distance, got ${distance}`);
  assert.ok(Math.abs(u - 0.5) < 1e-6 && Math.abs(v - 0.5) < 1e-6, `expected (0.5,0.5), got (${u},${v})`);
});

test('closestPointOnSurface on a point slightly OFF the surface finds a genuinely nearby projection, not a wild miss', () => {
  const srf = makeStressSurface();
  const onSurface = surfacePoint(srf, 0.8, 1.2);
  // Nudge it a small amount along an arbitrary direction — a real click's
  // raycast hit against the TESSELLATED mesh (a piecewise-linear
  // approximation of the true curved surface) will not sit bit-exactly on
  // the mathematical NURBS surface either, so this proves the inversion
  // is robust to that same real-world slack.
  const nudged = [onSurface[0] + 0.05, onSurface[1] - 0.03, onSurface[2] + 0.02];
  const { u, v, point, distance } = closestPointOnSurface(srf, nudged);
  assert.ok(distance < 0.1, `expected a small residual distance for a slightly-off-surface point, got ${distance}`);
  const dBack = Math.hypot(point[0] - onSurface[0], point[1] - onSurface[1], point[2] - onSurface[2]);
  assert.ok(dBack < 0.1, `expected the found point to land back near the original on-surface point, got ${dBack}`);
  assert.ok(Math.abs(u - 0.8) < 0.05 && Math.abs(v - 1.2) < 0.05, `expected (u,v) close to (0.8,1.2), got (${u},${v})`);
});

test('closestPointOnSurface handles a surface with a genuine POLE (on-axis point) without NaN/Infinity', () => {
  // A revolve of a profile touching the axis (radius 0 at one end) has a
  // degenerate control row there — the same pole case teapot.test.mjs's
  // own body/lid fixtures already exercise for revolve() itself.
  const profile = makeLine([0, 0, 0], [3, 0, 5]);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const target = surfacePoint(srf, 0.5, 1.0); // somewhere mid-profile, not at the pole itself
  const { u, v, point, distance } = closestPointOnSurface(srf, target);
  assert.ok(Number.isFinite(u) && Number.isFinite(v), 'u/v must be finite even near a pole-bearing surface');
  assert.ok(point.every(Number.isFinite), 'returned point must be finite');
  assert.ok(distance < 1e-6, `expected near-exact match away from the pole, got ${distance}`);
});

test('closestPointOnSurface on an EXTRUDE (ruled, degree-1 in V) finds the exact answer', () => {
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 8, 0, Math.PI);
  const srf = extrude(profile, [0, 0, 1], 20);
  const target = surfacePoint(srf, 0.5, 0.35);
  const { u, v, distance } = closestPointOnSurface(srf, target);
  assert.ok(distance < 1e-6, `expected near-exact match, got ${distance}`);
  assert.ok(Math.abs(u - 0.5) < 1e-4 && Math.abs(v - 0.35) < 1e-4, `expected (0.5,0.35), got (${u},${v})`);
});

// wrapParam — the shared boundary rule closestPointOnSurface/
// refineClosestPointOnSurface both depend on.
test('wrapParam clamps an open direction and wraps a closed one', () => {
  assert.equal(wrapParam(1.5, 0, 1, false), 1, 'open direction must clamp above the max');
  assert.equal(wrapParam(-0.5, 0, 1, false), 0, 'open direction must clamp below the min');
  assert.ok(Math.abs(wrapParam(1.3, 0, 1, true) - 0.3) < 1e-12, 'closed direction must wrap past the max back near the min');
  assert.ok(Math.abs(wrapParam(-0.2, 0, 1, true) - 0.8) < 1e-12, 'closed direction must wrap below the min back near the max');
});

// THE ACTUAL REPORTED BUG (Project onto a NURBS torus): a
// torus is closed in BOTH u and v (confirmed directly, not assumed). A
// target point on the OTHER SIDE of the seam from a warm-started seed must
// still converge to a near-exact match — the old hard-clamp-and-break
// behavior pinned dead against the domain edge the instant a step wanted
// to cross it, silently returning the SEED unchanged forever after.
test('closestPointOnSurface crosses a torus seam instead of pinning against it', () => {
  const srf = makeTorusSrf();
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true, 'torus must be closed in u (the tube-around direction)');
  assert.equal(closedV, true, 'torus must be closed in v (the ring-around direction)');
  const uMax = srf.knotsU[srf.knotsU.length - 1], vMax = srf.knotsV[srf.knotsV.length - 1];
  // A target genuinely near the SEAM but on the "wrapped-around" side —
  // exactly the case a straight line swept across a real torus must cross.
  for (const [u0, v0] of [[uMax * 0.02, vMax * 0.5], [uMax * 0.5, vMax * 0.02], [uMax * 0.98, vMax * 0.98]]) {
    const target = surfacePoint(srf, u0, v0);
    const { point, distance } = closestPointOnSurface(srf, target);
    assert.ok(distance < 1e-4, `expected a near-exact match crossing the torus seam near (u0=${u0},v0=${v0}), got distance=${distance}`);
    const d = Math.hypot(point[0] - target[0], point[1] - target[1], point[2] - target[2]);
    assert.ok(d < 1e-4, `returned point diverges from the real target by ${d}`);
  }
});
