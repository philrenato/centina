// TWO RAILS MEETING AT A POINT ARE ONE LONGER RAIL — proving that the
// concatenated rail's junction is REACHABLE BY THE EXISTING MITER
// MACHINERY, which is the only thing that makes this helper worth more
// than a plain join.
//
// The fixture is a 90-degree elbow at the world origin: a 100mm leg
// arriving along +X and a 100mm leg leaving along +Y. Deliberately NOT a
// two-point-per-rail minimum in every case — the reversal cases use a
// three-point polyline so a reversal is observable in the control point
// ORDER, not only in a sign.

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  concatRailsAtJunction, RAIL_JUNCTION_TOLERANCE, curvePoint, reverseCurve,
  adaptiveArcLengthSamples,
} from '../kernel/curve.mjs';
import { railInteriorCorners, sweep1Rigid, PIPE_MITER_LIMIT } from '../kernel/sweep.mjs';
import { degreeElevateCurve } from '../kernel/knots.mjs';
import { makeCircle } from '../kernel/primitives.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

// A clamped degree-1 curve through the given points — the exact shape a
// Line/Polyline rail arrives as.
function deg1(pts) {
  const ctrlPts = pts.map((p) => [p[0], p[1], p[2], 1]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}
function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function domain(crv) { return [crv.knots[0], crv.knots[crv.knots.length - 1]]; }

const LEG_IN = deg1([[-100, 0, 0], [-50, 0, 0], [0, 0, 0]]);   // travels +X, ends at the origin
const LEG_OUT = deg1([[0, 0, 0], [0, 50, 0], [0, 100, 0]]);    // starts at the origin, travels +Y
const ORIGIN = [0, 0, 0];

test('junction of two degree-1 rails IS reported as an interior corner by railInteriorCorners, at the right index and the right angle', () => {
  const res = concatRailsAtJunction(LEG_IN, LEG_OUT);
  assert.equal(res.ok, true, res.reason);
  assert.equal(res.curve.degree, 1, 'two degree-1 rails must stay degree 1 — a degree>1 result is invisible to railInteriorCorners');

  // The junction survives as a genuine interior CONTROL POINT, which is
  // exactly what railInteriorCorners reads.
  assert.equal(res.junctionIndex, 2, 'junction control point index = A.ctrlPts.length - 1 for a degree-1 pair');
  assert.ok(dist(res.curve.ctrlPts[res.junctionIndex].slice(0, 3), ORIGIN) < 1e-12);

  const corners = railInteriorCorners(res.curve);
  // The two straight legs contribute NO corners of their own (collinear
  // runs are omitted outright, not returned as theta=0), so the junction is
  // the only corner on the whole concatenated rail — the payoff, stated as
  // an exact count rather than "at least one".
  assert.equal(corners.length, 1, 'the junction is the only genuine corner on the joined rail');
  const c = corners[0];
  assert.deepEqual(c.ringIndices, [res.junctionIndex]);
  assert.ok(Math.abs(c.theta - Math.PI / 2) < 1e-12, `turn angle ${c.theta} should be exactly 90 degrees`);
  // sec(45deg) — the true-miter stretch this corner will now receive.
  assert.ok(Math.abs(c.stretch - Math.SQRT2) < 1e-12, `stretch ${c.stretch} should be sec(45deg)`);
  assert.ok(Math.abs(res.turnAngle - Math.PI / 2) < 1e-12);
});

test('the miter is actually APPLIED to the joined rail — the swept skin reaches r*sec(theta/2) from the rail at the junction', () => {
  const res = concatRailsAtJunction(LEG_IN, LEG_OUT);
  const R = 5;
  const srf = sweep1Rigid(res.curve, makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], R));
  const reach = maxSkinDistanceFromRail(srf, res.curve);
  const trueMiter = R / Math.cos(Math.PI / 4);
  assert.ok(Math.abs(reach - trueMiter) < 1e-6, `mitered elbow reaches ${reach}mm, expected r*sec(45deg) = ${trueMiter}mm`);
});

test('all four end-pairings work, reversing whichever rail needs it, and all four produce the same elbow', () => {
  const cases = [
    { name: 'A-end / B-start', a: LEG_IN, b: LEG_OUT, revA: false, revB: false },
    { name: 'A-end / B-end', a: LEG_IN, b: reverseCurve(LEG_OUT), revA: false, revB: true },
    { name: 'A-start / B-start', a: reverseCurve(LEG_IN), b: LEG_OUT, revA: true, revB: false },
    { name: 'A-start / B-end', a: reverseCurve(LEG_IN), b: reverseCurve(LEG_OUT), revA: true, revB: true },
  ];
  for (const cse of cases) {
    const res = concatRailsAtJunction(cse.a, cse.b);
    assert.equal(res.ok, true, `${cse.name}: ${res.reason}`);
    assert.equal(res.reversedA, cse.revA, `${cse.name}: reversedA`);
    assert.equal(res.reversedB, cse.revB, `${cse.name}: reversedB`);
    // Every pairing produces the SAME rail: same control points in the same
    // order, and therefore the same single 90-degree corner.
    assert.deepEqual(
      res.curve.ctrlPts.map((p) => p.slice(0, 3)),
      [[-100, 0, 0], [-50, 0, 0], [0, 0, 0], [0, 50, 0], [0, 100, 0]],
      `${cse.name}: control points`,
    );
    assert.equal(railInteriorCorners(res.curve).length, 1, `${cse.name}: one corner`);
  }
});

test('the junction point is exact, and both halves reproduce their originals exactly', () => {
  const res = concatRailsAtJunction(LEG_IN, LEG_OUT);
  // Exact at the junction parameter, not merely within a tolerance: the
  // seam knot carries full multiplicity, so the curve passes through the
  // shared control point itself.
  assert.equal(dist(curvePoint(res.curve, res.junctionParam), ORIGIN), 0);
  assert.equal(dist(res.junction, ORIGIN), 0);

  // joinCurvesC0 rescales curve i onto [i, i+1], a pure affine
  // reparametrization, so the joined curve on [0,1] must reproduce A and on
  // [1,2] must reproduce B — sampled at genuinely in-between parameters,
  // not only at the shared ends.
  const halves = [{ crv: LEG_IN, lo: 0, hi: 1 }, { crv: LEG_OUT, lo: 1, hi: 2 }];
  for (const h of halves) {
    const [uMin, uMax] = domain(h.crv);
    let worst = 0;
    for (let i = 0; i <= 40; i++) {
      const t = i / 40;
      const joined = curvePoint(res.curve, h.lo + (h.hi - h.lo) * t);
      const original = curvePoint(h.crv, uMin + (uMax - uMin) * t);
      worst = Math.max(worst, dist(joined, original));
    }
    assert.ok(worst < 1e-9, `half [${h.lo},${h.hi}] deviates by ${worst}mm from its own original`);
  }
});

test('a reversed input still reproduces its original curve, traversed the other way', () => {
  // The A-start pairing reverses LEG_IN. The joined curve's first half must
  // still be LEG_IN's own point set — read backwards.
  const res = concatRailsAtJunction(reverseCurve(LEG_IN), LEG_OUT);
  assert.equal(res.reversedA, true);
  const [uMin, uMax] = domain(LEG_IN);
  let worst = 0;
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    worst = Math.max(worst, dist(curvePoint(res.curve, t), curvePoint(LEG_IN, uMin + (uMax - uMin) * t)));
  }
  assert.ok(worst < 1e-9, `reversed-A half deviates by ${worst}mm`);
});

test('a degree>1 result is INVISIBLE to railInteriorCorners, and the junction ships the un-mitered elbow — measured, not assumed', () => {
  // Same 90-degree elbow, same geometry, only the REPRESENTATION differs:
  // one leg arrives already degree-elevated (as a curved SketchCurve rail
  // would), so joinCurvesC0 elevates the pair to a common degree 3.
  const res = concatRailsAtJunction(LEG_IN, degreeElevateCurve(LEG_OUT, 3));
  assert.equal(res.ok, true, res.reason);
  assert.equal(res.curve.degree, 3);
  assert.equal(railInteriorCorners(res.curve).length, 0, 'railInteriorCorners returns [] for any degree>1 rail — the corner gets no miter at all');
  // The turn angle itself is still reported honestly by the helper, even
  // though nothing downstream will act on it.
  assert.ok(Math.abs(res.turnAngle - Math.PI / 2) < 1e-9);

  const R = 5;
  const srf = sweep1Rigid(res.curve, makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], R));
  const reach = maxSkinDistanceFromRail(srf, res.curve);
  const trueMiter = R / Math.cos(Math.PI / 4); // 7.0710678...

  // THE HONEST NUMBER. A true miter reaches r*sec(45deg) = 7.0711mm from
  // its own rail at the corner; this reaches 5.2380mm — 74.1% of it. The
  // shortfall is real geometry (an un-mitered butt joint is narrower by
  // roughly cos(theta/2)), not sampling: the resampled sweep's own fit
  // rounds the corner slightly, which is why the measurement lands a little
  // above the ideal 5.0 rather than exactly on it. Pinned as a RANGE so a
  // future fix that genuinely closes the gap fails this test loudly instead
  // of passing silently.
  assert.ok(reach > R, `un-mitered elbow reaches ${reach.toFixed(4)}mm, expected slightly over the tube radius ${R}mm`);
  assert.ok(reach < 5.6, `un-mitered elbow reaches ${reach.toFixed(4)}mm — expected well short of a true miter`);
  assert.ok(reach < trueMiter * 0.8, `un-mitered elbow reaches ${reach.toFixed(4)}mm, only ${(100 * reach / trueMiter).toFixed(1)}% of a true miter's ${trueMiter.toFixed(4)}mm`);
});

test('refuses honestly when the two rails do not share an endpoint, and reports the real gap', () => {
  const far = deg1([[10, 0, 0], [10, 100, 0]]); // 10mm away from LEG_IN's own end
  const res = concatRailsAtJunction(LEG_IN, far);
  assert.equal(res.ok, false);
  assert.match(res.reason, /don't share an endpoint/);
  assert.ok(Math.abs(res.nearestGap - 10) < 1e-9, `nearest gap ${res.nearestGap} should be the real 10mm`);

  // A gap just inside the tolerance still joins — the tolerance is a real
  // number the caller can rely on, not a decoration.
  const nearlyTouching = deg1([[RAIL_JUNCTION_TOLERANCE * 0.5, 0, 0], [0, 100, 0]]);
  assert.equal(concatRailsAtJunction(LEG_IN, nearlyTouching).ok, true);
  const justOutside = deg1([[RAIL_JUNCTION_TOLERANCE * 2, 0, 0], [0, 100, 0]]);
  assert.equal(concatRailsAtJunction(LEG_IN, justOutside).ok, false);
});

test('refuses honestly when the junction is ambiguous — a closed rail, or two rails closing a loop', () => {
  // (a) BOTH ends of A land on B: A and B together close a loop.
  const back = deg1([[0, 100, 0], [-100, 100, 0], [-100, 0, 0]]);
  const loop = concatRailsAtJunction(LEG_OUT, back);
  assert.equal(loop.ok, true, 'control: LEG_OUT/back share exactly one endpoint pair');
  const bothEnds = concatRailsAtJunction(deg1([[-100, 0, 0], [0, 0, 0]]), deg1([[0, 0, 0], [-100, 0, 100], [-100, 0, 0]]));
  assert.equal(bothEnds.ok, false);
  assert.match(bothEnds.reason, /ambiguous/);

  // (b) B is itself CLOSED and its seam sits on A's own end: reversing B or
  // not gives two genuinely different traversals of the same loop.
  const closedB = deg1([[0, 0, 0], [0, 50, 0], [50, 50, 0], [0, 0, 0]]);
  const withClosed = concatRailsAtJunction(LEG_IN, closedB);
  assert.equal(withClosed.ok, false);
  assert.match(withClosed.reason, /ambiguous/);

  // (c) A is itself CLOSED — both of its ends are the same point, so both
  // coincide with B at once.
  const closedA = deg1([[0, 0, 0], [0, 50, 0], [50, 50, 0], [0, 0, 0]]);
  const closedFirst = concatRailsAtJunction(closedA, LEG_OUT);
  assert.equal(closedFirst.ok, false);
  assert.match(closedFirst.reason, /ambiguous/);
});

test('refuses a 180-degree fold-back, and the miter machinery genuinely cannot describe one', () => {
  const foldBack = deg1([[0, 0, 0], [-50, 0, 0]]); // leaves the junction back along -X, exactly opposite LEG_IN's arrival
  const res = concatRailsAtJunction(LEG_IN, foldBack);
  assert.equal(res.ok, false);
  assert.match(res.reason, /fold straight back/);
  assert.ok(Math.abs(res.turnAngle - Math.PI) < 1e-12);

  // WHY refusing rather than deferring: build the rail the refusal
  // prevented and confirm the miter machinery produces a finite, NaN-free,
  // silently self-intersecting tube rather than any honest signal of its
  // own. railInteriorCorners reports an effectively infinite stretch...
  const forced = { degree: 1, knots: [0, 0, 1, 2, 2], ctrlPts: [[-100, 0, 0, 1], [0, 0, 0, 1], [-50, 0, 0, 1]] };
  const corners = railInteriorCorners(forced);
  assert.equal(corners.length, 1);
  assert.ok(corners[0].stretch > PIPE_MITER_LIMIT * 1e6, 'a fold-back asks for an effectively infinite miter stretch');
  // ...whose bend direction has collapsed onto the rail's own tangent, so
  // the stretch has nothing perpendicular to act on. The swept corner ring
  // comes back a plain, untouched circle of radius r.
  const R = 5;
  const srf = sweep1Rigid(forced, makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], R));
  assert.ok(srf.ctrlNet.flat(2).every(Number.isFinite), 'no NaN — the failure is silent, which is the point');
  const cornerRing = srf.ctrlNet.map((row) => row[1]);
  // The on-curve control points of a rational circle sit at exactly R; the
  // off-curve corner points sit at R*sqrt(2). Neither is stretched.
  for (let i = 0; i < cornerRing.length; i += 2) {
    assert.ok(Math.abs(Math.hypot(...cornerRing[i].slice(0, 3)) - R) < 1e-9, 'the corner ring was never stretched at all');
  }
});

test('a turn merely CLOSE to 180 degrees is NOT refused — the miter-limit fallback handles it', () => {
  // 179 degrees: far past PIPE_MITER_LIMIT's own ~168.5-degree ceiling, so
  // the corner routes through applyMiterLimitFallback's fillet rather than
  // a true miter — which is exactly the downstream behavior to defer to,
  // not duplicate here.
  const theta = (179 * Math.PI) / 180;
  const almostBack = deg1([[0, 0, 0], [-50 * Math.cos(Math.PI - theta), 50 * Math.sin(Math.PI - theta), 0]]);
  const res = concatRailsAtJunction(LEG_IN, almostBack);
  assert.equal(res.ok, true, res.reason);
  assert.ok(Math.abs(res.turnAngle - theta) < 1e-9);
  const corners = railInteriorCorners(res.curve);
  assert.equal(corners.length, 1);
  assert.ok(corners[0].stretch > PIPE_MITER_LIMIT, 'this corner is over the miter limit, so the fallback is what handles it');
  const srf = sweep1Rigid(res.curve, makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 1));
  assert.ok(srf.ctrlNet.flat(2).every(Number.isFinite));
});

test('refuses a degenerate input rather than building a broken rail', () => {
  assert.equal(concatRailsAtJunction(null, LEG_OUT).ok, false);
  assert.match(concatRailsAtJunction(LEG_IN, { degree: 1, knots: [0, 0], ctrlPts: [[0, 0, 0, 1]] }).reason, /at least 2 control points/);
});

// How far the swept skin reaches from its own rail. For a tube of radius r
// on a straight run this is exactly r; at a TRUE miter it is r*sec(theta/2)
// (the corner ring is an ellipse whose in-bend-plane semi-axis is stretched
// by exactly that factor), and at an un-mitered butt joint it stays near r.
// One scalar, and the one that actually distinguishes the two.
function maxSkinDistanceFromRail(srf, rail) {
  const [rMin, rMax] = domain(rail);
  const poly = adaptiveArcLengthSamples(rail, rMin, rMax, 1e-6).map((s) => s.pt);
  const distToPolyline = (P) => {
    let best = Infinity;
    for (let i = 0; i < poly.length - 1; i++) {
      const a = poly[i], b = poly[i + 1];
      const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const L2 = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2;
      let t = L2 > 0 ? ((P[0] - a[0]) * ab[0] + (P[1] - a[1]) * ab[1] + (P[2] - a[2]) * ab[2]) / L2 : 0;
      t = Math.max(0, Math.min(1, t));
      best = Math.min(best, Math.hypot(P[0] - a[0] - ab[0] * t, P[1] - a[1] - ab[1] * t, P[2] - a[2] - ab[2] * t));
    }
    return best;
  };
  const [uMin, uMax] = [srf.knotsU[0], srf.knotsU[srf.knotsU.length - 1]];
  const [vMin, vMax] = [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]];
  const NU = 48, NV = 601;
  let maxD = -Infinity;
  for (let j = 0; j < NV; j++) {
    const v = vMin + (vMax - vMin) * (j / (NV - 1));
    for (let i = 0; i < NU; i++) {
      const d = distToPolyline(surfacePoint(srf, uMin + (uMax - uMin) * (i / NU), v));
      if (d > maxD) maxD = d;
    }
  }
  return maxD;
}
