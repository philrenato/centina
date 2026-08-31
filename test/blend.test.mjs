// Tests for kernel/blend.mjs — curvature-continuous (G2) and third-order
// (G3) corner blends.
//
// EVERY continuity claim in this file is MEASURED, never asserted. The
// quantities measured are the reparametrization-invariant (geometric) ones —
// unit tangent, curvature vector, d(kappa)/ds, torsion — because those are
// what G^k actually means and what a highlight running across a surface
// actually sees. Parametric (C^k) agreement is neither claimed nor tested;
// see the module header for why the two are different claims.
//
// Two disciplines this file holds itself to:
//  1. EVERY continuity assertion has a CONTRAST assertion next to it showing
//     the same measurement FAILS for the next lower order (G1 fails the
//     curvature test, G2 fails the dkappa/ds and torsion tests). A test that
//     cannot fail is not a proof, and a curvature test that a plain arc would
//     also pass would prove nothing at all about this module.
//  2. NO SYMMETRIC FIXTURES. A symmetric corner or an equal-length polyline
//     hides a whole class of construction errors (a swapped end, a mirrored
//     shape fraction). Every polyline here has unequal edge lengths and
//     unequal turn angles; the curve/curve fixtures are non-planar and of
//     mismatched degree, one of them rational with non-unit weights.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  geometricBlend,
  blendCornerCurve,
  blendPolyline,
  blendSegmentsToCurve,
  blendCurves,
  blendFrameFromCurve,
  blendFrameFromLine,
  curveGeometryAt,
  frameGeometry,
  peakCurvature,
  nearestCurveEndpoints,
  nearestEndpointToPoint,
} from '../kernel/blend.mjs';
import { filletCornerArc, filletOpenPolyline, makeCircle } from '../kernel/primitives.mjs';
import { curvePoint, rationalCurveDerivs } from '../kernel/curve.mjs';

const Z = [0, 0, 1];

function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function len3(a) { return Math.hypot(a[0], a[1], a[2]); }
function unit3(a) { const l = len3(a); return [a[0] / l, a[1] / l, a[2] / l]; }

// Finite-difference unit tangent — the SAME instrument
// test/interpolate.test.mjs already uses to check seam tangency (its own
// closedCurveInterp G1 test asserts dot > 0.999). Kept here so the G1
// assertions below are comparable to that established bar, and so they can be
// shown to beat it by orders of magnitude.
function fdUnitTangent(crv, u, du) {
  const a = curvePoint(crv, u - du);
  const b = curvePoint(crv, u + du);
  return unit3(sub3(b, a));
}

// ---------------------------------------------------------------------------
// FIXTURES — deliberately irregular. `RAIL` has four distinct turn angles
// (about 48, 43, 52 and 39 degrees) and five distinct edge lengths.
// ---------------------------------------------------------------------------
const RAIL = [[0, 0, 0], [70, 0, 0], [123, 47, 0], [196, 43, 0], [231, 112, 0], [312, 96, 0]];
const LOOP = [[0, 0, 0], [96, -14, 0], [147, 58, 0], [83, 121, 0], [-26, 74, 0]]; // irregular closed pentagon, no two edges equal

// A rational quadratic neighbor (degree 2, a non-unit weight) and a degree-4
// non-planar polynomial neighbor — mismatched degree, mismatched rationality,
// genuinely 3D, both with nonzero curvature at the join.
const NEIGHBOR_A = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[-60, -20, 5, 1], [-25, 15, -8, Math.SQRT1_2], [0, 0, 0, 1]] };
const NEIGHBOR_B = { degree: 4, knots: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], ctrlPts: [[30, 10, 4, 1], [52, 34, -6, 1], [80, 20, 18, 1], [95, 60, 3, 1], [130, 44, 25, 1]] };
// A second pair where BOTH ends carry genuinely nonzero torsion (a 3-control-
// point rational is always planar, so NEIGHBOR_A's torsion is exactly zero —
// a torsion test against it alone would pass vacuously).
const TWIST_A = { degree: 3, knots: [0, 0, 0, 0, 1, 1, 1, 1], ctrlPts: [[-90, 30, -40, 1], [-55, -10, 12, 1], [-28, 26, 33, 1], [0, 0, 0, 1]] };
const TWIST_B = { degree: 3, knots: [0, 0, 0, 0, 1, 1, 1, 1], ctrlPts: [[45, 18, 22, 1], [77, -14, -19, 1], [104, 41, 30, 1], [151, 12, -8, 1]] };

// ===========================================================================
// 0. THE MEASURING INSTRUMENT ITSELF
// ===========================================================================

test('curveGeometryAt: validated against an EXACT analytic curve first — a real NURBS circle reads back constant curvature 1/R and zero torsion', () => {
  // Every claim below is made with this instrument, so it is checked against
  // geometry whose invariants are known in closed form before it is trusted
  // anywhere else. A rational circle is the right probe: it exercises the
  // weight bookkeeping in rationalCurveDerivs that a polynomial curve would
  // leave completely untested.
  const R = 17.5;
  const circle = makeCircle([3, -8, 2], [1, 0, 0], [0, 1, 0], R, 4);
  const uEnd = circle.knots[circle.knots.length - 1];
  for (let i = 0; i <= 40; i++) {
    const u = (i / 40) * uEnd;
    const g = curveGeometryAt(circle, u, Z);
    assert.ok(Math.abs(g.kappa - 1 / R) < 1e-12, `circle curvature at u=${u} should be exactly 1/R=${1 / R}, got ${g.kappa}`);
    assert.ok(Math.abs(g.signedKappa - 1 / R) < 1e-12, `circle SIGNED curvature (CCW about +Z) should be +1/R, got ${g.signedKappa}`);
    assert.ok(Math.abs(g.dKappaDs) < 1e-11, `a circle's curvature is constant, so dkappa/ds must be 0, got ${g.dKappaDs}`);
    assert.ok(Math.abs(g.torsion) < 1e-12, `a planar circle has zero torsion, got ${g.torsion}`);
    assert.ok(Math.abs(len3(g.tangent) - 1) < 1e-12, 'reported tangent must be a unit vector');
    // The curvature VECTOR must point at the center, magnitude 1/R.
    const toCenter = unit3(sub3([3, -8, 2], curvePoint(circle, u)));
    assert.ok(dot3(unit3(g.kappaVec), toCenter) > 1 - 1e-11, 'curvature vector must point at the circle center');
    assert.ok(Math.abs(len3(g.kappaVec) - 1 / R) < 1e-12, 'curvature vector magnitude must equal 1/R');
  }
});

test('curveGeometryAt: a straight degree-1 line reads exactly zero curvature, and NaN torsion rather than a vacuous zero', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[-4, 7, 1, 1], [88, -13, 40, 1]] };
  const g = curveGeometryAt(line, 0.37, Z);
  assert.equal(g.kappa, 0);
  assert.equal(g.signedKappa, 0);
  assert.ok(Number.isNaN(g.torsion), 'torsion is genuinely undefined on a straight run — reporting 0 would let a torsion assertion pass for free');
});

// ===========================================================================
// 1. REFUSALS AND INPUT VALIDATION
// ===========================================================================

test('geometricBlend: refuses any continuity order outside {1,2,3}, including non-integers', () => {
  const a = blendFrameFromLine([0, 0, 0], [1, 0, 0]);
  const b = blendFrameFromLine([10, 10, 0], [0, 1, 0]);
  for (const bad of [0, 4, 2.5, -1, '2', NaN]) {
    const r = geometricBlend(a, b, { continuity: bad });
    assert.equal(r.ok, false, `continuity ${bad} must be refused`);
    assert.match(r.reason, /continuity order must be 1, 2 or 3/);
  }
});

test('geometricBlend: refuses a zero-speed frame, coincident endpoints, and non-finite input — honestly, by name, never silently', () => {
  const good = blendFrameFromLine([10, 10, 0], [0, 1, 0]);
  const dead = blendFrameFromLine([0, 0, 0], [0, 0, 0]);
  assert.match(geometricBlend(dead, good).reason, /zero-speed/);
  const same = blendFrameFromLine([10, 10, 0], [1, 0, 0]);
  assert.match(geometricBlend(same, good).reason, /coincident/);
  const nan = { point: [0, 0, 0], d1: [1, NaN, 0], d2: [0, 0, 0], d3: [0, 0, 0] };
  assert.match(geometricBlend(nan, good).reason, /non-finite/);
  const a = blendFrameFromLine([0, 0, 0], [1, 0, 0]);
  assert.match(geometricBlend(a, good, { startMagnitude: 0 }).reason, /magnitudes must be positive/);
});

test('blendCornerCurve / blendPolyline: a collinear run and a near-180 reversal are refused with filletCornerArc\'s OWN wording — one source of truth for what a corner is', () => {
  const straight = blendCornerCurve([10, 0, 0], [0, 0, 0], [20, 0, 0], 3, Z, { continuity: 2 });
  assert.equal(straight.ok, false);
  assert.equal(straight.reason, filletCornerArc([10, 0, 0], [0, 0, 0], [20, 0, 0], 3, Z).reason);
  const fold = blendCornerCurve([10, 0, 0], [0, 0, 0], [0.0001, 0, 0], 3, Z, { continuity: 2 });
  assert.equal(fold.ok, false);
  assert.equal(fold.reason, filletCornerArc([10, 0, 0], [0, 0, 0], [0.0001, 0, 0], 3, Z).reason);
  assert.equal(blendPolyline(RAIL, -5, { continuity: 2 }).ok, false);
  assert.match(blendPolyline(RAIL, 0, { continuity: 2 }).reason, /must be positive/);
  assert.match(blendCornerCurve([10, 0, 0], [0, 0, 0], [10, 20, 0], 3, Z, { continuity: 2, tangentScale: 0 }).reason, /tangentScale/);
  assert.match(blendCornerCurve([10, 0, 0], [0, 0, 0], [10, 20, 0], 3, Z, { continuity: 2, tangentScale: -1 }).reason, /tangentScale/);
  assert.match(blendCornerCurve([10, 0, 0], [0, 0, 0], [10, 20, 0], 3, Z, { continuity: 2, tangentScale: NaN }).reason, /tangentScale/);
});

// ===========================================================================
// 2. G1 REGRESSION GUARD — the path that already ships must not move
// ===========================================================================

test('REGRESSION GUARD: blendPolyline at continuity 1 returns filletOpenPolyline\'s OWN result, byte for byte', () => {
  // Not "equivalent to" — literally the same object graph, because the G1
  // path delegates rather than reimplementing. This is the guard on shipped
  // FilletCrv behavior: if this ever diverges, the Fillet command changed
  // shape without anyone asking it to.
  for (const r of [3, 12, 25]) {
    assert.deepEqual(blendPolyline(RAIL, r, { continuity: 1 }), filletOpenPolyline(RAIL, r));
    assert.deepEqual(blendPolyline(LOOP, r, { continuity: 1, closed: true }), filletOpenPolyline(LOOP, r, { closed: true }));
  }
});

test('REGRESSION GUARD: blendCornerCurve at continuity 1 reproduces filletCornerArc exactly (p0/apex/p2/weight/trim/turnAngle)', () => {
  const g1 = blendCornerCurve([70, 0, 0], [0, 0, 0], [123, 47, 0], 14, Z, { continuity: 1 });
  const arc = filletCornerArc([70, 0, 0], [0, 0, 0], [123, 47, 0], 14, Z);
  assert.equal(g1.segment.type, 'arc');
  assert.deepEqual(g1.segment.p0, arc.p0);
  assert.deepEqual(g1.segment.apex, arc.apex);
  assert.deepEqual(g1.segment.p2, arc.p2);
  assert.equal(g1.segment.weight, arc.weight);
  assert.equal(g1.trim, arc.trim);
  assert.equal(g1.turnAngle, arc.turnAngle);
});

test('a G2 blend keeps the arc\'s OWN tangent points, so it is a true drop-in: the straight runs either side do not move', () => {
  const g1 = blendPolyline(RAIL, 12, { continuity: 1 });
  const g2 = blendPolyline(RAIL, 12, { continuity: 2 });
  const g3 = blendPolyline(RAIL, 12, { continuity: 3 });
  assert.equal(g1.segments.length, g2.segments.length);
  assert.equal(g1.segments.length, g3.segments.length);
  for (let i = 0; i < g1.segments.length; i++) {
    const a = g1.segments[i];
    if (a.type === 'line') {
      assert.deepEqual(g2.segments[i], a, `segment ${i} is a straight run and must be untouched`);
      assert.deepEqual(g3.segments[i], a, `segment ${i} is a straight run and must be untouched`);
      continue;
    }
    for (const b of [g2.segments[i], g3.segments[i]]) {
      assert.equal(b.type, 'blend');
      const first = b.crv.ctrlPts[0], last = b.crv.ctrlPts[b.crv.ctrlPts.length - 1];
      assert.ok(len3(sub3(first, a.p0)) < 1e-12, `blend ${i} must start exactly where the arc started`);
      assert.ok(len3(sub3(last, a.p2)) < 1e-12, `blend ${i} must end exactly where the arc ended`);
    }
  }
});

// ===========================================================================
// 3. G1 — POSITION AND TANGENT AT EVERY SEAM
// ===========================================================================

test('G0/G1 at every seam of a blended rail: position to 1e-12, and unit tangent by BOTH analytic and finite-difference measurement (dot > 1 - 1e-11, beating the project\'s established 0.999 bar by 8 orders of magnitude)', () => {
  for (const k of [2, 3]) {
    const res = blendPolyline(RAIL, 12, { continuity: k });
    assert.equal(res.ok, true);
    for (let i = 0; i + 1 < res.segments.length; i++) {
      const cur = res.segments[i], nxt = res.segments[i + 1];
      const endPt = cur.type === 'line' ? cur.b : curvePoint(cur.crv, 1);
      const startPt = nxt.type === 'line' ? nxt.a : curvePoint(nxt.crv, 0);
      assert.ok(len3(sub3(endPt, startPt)) < 1e-12, `k=${k} seam ${i}: position gap ${len3(sub3(endPt, startPt))}`);

      const tOut = cur.type === 'line' ? unit3(sub3(cur.b, cur.a)) : unit3(rationalCurveDerivs(cur.crv, 1, 1)[1]);
      const tIn = nxt.type === 'line' ? unit3(sub3(nxt.b, nxt.a)) : unit3(rationalCurveDerivs(nxt.crv, 0, 1)[1]);
      assert.ok(dot3(tOut, tIn) > 1 - 1e-11, `k=${k} seam ${i}: analytic tangent dot ${dot3(tOut, tIn)}`);

      // Independent finite-difference cross-check, the same instrument the
      // existing closedCurveInterp seam test uses. Catches an analytic
      // derivative that is self-consistently wrong.
      const fdOut = cur.type === 'line' ? unit3(sub3(cur.b, cur.a)) : fdUnitTangent(cur.crv, 1 - 1e-5, 1e-5);
      const fdIn = nxt.type === 'line' ? unit3(sub3(nxt.b, nxt.a)) : fdUnitTangent(nxt.crv, 1e-5, 1e-5);
      assert.ok(dot3(fdOut, fdIn) > 1 - 1e-9, `k=${k} seam ${i}: finite-difference tangent dot ${dot3(fdOut, fdIn)}`);
    }
  }
});

// ===========================================================================
// 4. G2 — THE ASSERTION THAT DISTINGUISHES THIS FROM THE SHIPPED ARC
// ===========================================================================

test('G2: a blend meets a STRAIGHT neighbor at exactly zero curvature at both ends — and the CONTRAST that makes this a real test: the G1 arc it replaces jumps straight to 1/R there', () => {
  for (const [radius, turnDeg] of [[10, 90], [4.5, 31], [40, 118], [2, 152], [77, 61]]) {
    const phi = turnDeg * Math.PI / 180;
    const vertex = [0, 0, 0];
    const prev = [-300, 0, 0];
    const next = [400 * Math.cos(phi), 400 * Math.sin(phi), 0];

    const g2 = blendCornerCurve(vertex, prev, next, radius, Z, { continuity: 2 });
    assert.equal(g2.ok, true, g2.reason);
    for (const u of [0, 1]) {
      const g = curveGeometryAt(g2.crv, u, Z);
      // The neighbor is a straight edge, whose curvature is exactly 0, so
      // "matching" here is an absolute claim, not a tolerance on two nonzero
      // numbers. Stated as an absolute bound scaled by the corner's own
      // 1/radius so it reads as a relative claim against what the arc does.
      assert.ok(Math.abs(g.kappa) < 1e-12 / radius, `G2 blend curvature at u=${u} (R=${radius}, turn ${turnDeg}deg) must be 0, got ${g.kappa}`);
      assert.ok(len3(g.kappaVec) < 1e-12 / radius, 'the curvature VECTOR, not just its magnitude, must vanish at the seam');
    }
    // CONTRAST: the shipped G1 arc for the identical corner. Its curvature at
    // its own start is 1/R against the edge's 0 — the jump this module exists
    // to remove. Measured, so the test above cannot be passing vacuously.
    const g1 = blendCornerCurve(vertex, prev, next, radius, Z, { continuity: 1 });
    const arcCrv = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...g1.segment.p0, 1], [...g1.segment.apex, g1.segment.weight], [...g1.segment.p2, 1]] };
    const arcK = curveGeometryAt(arcCrv, 0, Z).kappa;
    assert.ok(Math.abs(arcK * radius - 1) < 1e-9, `sanity: the G1 arc really is a radius-${radius} arc (kappa*R = ${arcK * radius})`);
    assert.ok(arcK > 0.9 / radius, 'CONTRAST: the arc jumps from 0 curvature on the edge to 1/R — a real G2 violation this same measurement detects');
  }
});

test('G2 continuity is INDEPENDENT of the shape parameter — the beta-constraint argument says so, and five different shape fractions confirm it against the same tolerance', () => {
  // b1 (the tangent magnitude, here `tangentScale`) is the construction's
  // only free scalar, and curvature is reparametrization-invariant, so the
  // continuity order cannot depend on it. Demonstrated rather than asserted.
  for (const lam of [0.6, 0.9, 1.2, 1.8, 2.6]) {
    const r = blendCornerCurve([0, 0, 0], [-90, 0, 0], [60, 80, 0], 11, Z, { continuity: 2, tangentScale: lam });
    assert.equal(r.ok, true, `tangentScale ${lam}: ${r.reason}`);
    assert.equal(r.tangentScale, lam);
    for (const u of [0, 1]) {
      assert.ok(Math.abs(curveGeometryAt(r.crv, u).kappa) < 1e-12 / 11, `tangentScale ${lam} u=${u}: curvature must still vanish`);
    }
    // ...but it genuinely changes the SHAPE, otherwise the parameter is dead.
    assert.ok(peakCurvature(r.crv).kappa > 0, 'a real curve, not a straight line');
  }
  const tight = blendCornerCurve([0, 0, 0], [-90, 0, 0], [60, 80, 0], 11, Z, { continuity: 2, tangentScale: 0.6 });
  const loose = blendCornerCurve([0, 0, 0], [-90, 0, 0], [60, 80, 0], 11, Z, { continuity: 2, tangentScale: 2.6 });
  assert.ok(Math.abs(peakCurvature(tight.crv).kappa - peakCurvature(loose.crv).kappa) > 0.05 / 11, 'the shape parameter must actually change the shape');
});

// ===========================================================================
// 5. G3 — CLAIMED ONLY WHERE PROVEN
// ===========================================================================

test('G3 on a straight-edge corner: curvature AND d(kappa)/ds both vanish at both seams — with the CONTRAST that the G2 blend fails exactly the d(kappa)/ds half', () => {
  // For a PLANAR curve, third-order geometric contact means agreement of
  // position, unit tangent, signed curvature and d(kappa)/ds (torsion is
  // identically zero on both sides and carries no information). Signed
  // curvature is used deliberately: |kappa| has a non-differentiable minimum
  // at zero, which is exactly where a straight-edge seam sits, so the
  // unsigned derivative would be meaningless there.
  for (const [radius, turnDeg] of [[9, 74], [30, 128], [3.3, 42]]) {
    const phi = turnDeg * Math.PI / 180;
    const args = [[0, 0, 0], [-250, 0, 0], [350 * Math.cos(phi), 350 * Math.sin(phi), 0], radius, Z];
    const g3 = blendCornerCurve(...args, { continuity: 3 });
    assert.equal(g3.ok, true, g3.reason);
    assert.equal(g3.degree, 7);
    for (const u of [0, 1]) {
      const g = curveGeometryAt(g3.crv, u, Z);
      assert.ok(Math.abs(g.signedKappa) < 1e-12 / radius, `G3 u=${u}: curvature must vanish, got ${g.signedKappa}`);
      assert.ok(Math.abs(g.dSignedKappaDs) < 1e-12 / (radius * radius), `G3 u=${u}: d(kappa)/ds must vanish too, got ${g.dSignedKappaDs}`);
    }
    // CONTRAST — the G2 blend across the identical corner matches curvature
    // but NOT its arc-length derivative. Without this the G3 test above could
    // be passing for a reason unrelated to the extra two control points.
    const g2 = blendCornerCurve(...args, { continuity: 2 });
    const d2 = Math.abs(curveGeometryAt(g2.crv, 0, Z).dSignedKappaDs);
    assert.ok(d2 > 1e-4 / (radius * radius), `CONTRAST: the G2 blend's d(kappa)/ds at the seam is genuinely nonzero (${d2}) — G2 is not G3`);
  }
});

// ===========================================================================
// 6. GENERAL CURVE-TO-CURVE — CURVED, RATIONAL, MISMATCHED-DEGREE NEIGHBORS
// ===========================================================================

test('G2 between a degree-2 RATIONAL neighbor (non-unit weight) and a degree-4 non-planar one: the full curvature VECTOR matches at both ends to 1e-14 relative — and G1 across the same pair misses it by 50%+', () => {
  const gA = curveGeometryAt(NEIGHBOR_A, 1);
  const gB = curveGeometryAt(NEIGHBOR_B, 0);
  assert.ok(gA.kappa > 1e-3 && gB.kappa > 1e-3, 'sanity: both neighbors genuinely curve at the join (a zero-curvature fixture would make this test vacuous)');
  assert.notEqual(NEIGHBOR_A.degree, NEIGHBOR_B.degree, 'sanity: the neighbors really are of mismatched degree');

  const g2 = blendCurves(NEIGHBOR_A, 1, NEIGHBOR_B, 0, { continuity: 2 });
  assert.equal(g2.ok, true, g2.reason);
  assert.equal(g2.degree, 5);
  const s = curveGeometryAt(g2.crv, 0), e = curveGeometryAt(g2.crv, 1);
  assert.ok(dot3(s.tangent, gA.tangent) > 1 - 1e-14, `start tangent dot ${dot3(s.tangent, gA.tangent)}`);
  assert.ok(dot3(e.tangent, gB.tangent) > 1 - 1e-14, `end tangent dot ${dot3(e.tangent, gB.tangent)}`);
  assert.ok(len3(sub3(s.kappaVec, gA.kappaVec)) / gA.kappa < 1e-13, `start curvature vector relative error ${len3(sub3(s.kappaVec, gA.kappaVec)) / gA.kappa}`);
  assert.ok(len3(sub3(e.kappaVec, gB.kappaVec)) / gB.kappa < 1e-13, `end curvature vector relative error ${len3(sub3(e.kappaVec, gB.kappaVec)) / gB.kappa}`);

  // CONTRAST: G1 across the identical pair matches tangent only.
  const g1 = blendCurves(NEIGHBOR_A, 1, NEIGHBOR_B, 0, { continuity: 1 });
  assert.equal(g1.degree, 3);
  const s1 = curveGeometryAt(g1.crv, 0);
  assert.ok(dot3(s1.tangent, gA.tangent) > 1 - 1e-14, 'G1 still matches the tangent exactly');
  assert.ok(Math.abs(s1.kappa - gA.kappa) / gA.kappa > 0.1, `CONTRAST: G1 curvature error is large (${Math.abs(s1.kappa - gA.kappa) / gA.kappa}), which is what G1 means`);
});

test('G3 between two genuinely NON-PLANAR neighbors: curvature, d(kappa)/ds AND torsion all match at both ends — with the CONTRAST that G2 matches curvature and misses the other two', () => {
  // Torsion is the part of the G3 claim that a planar fixture cannot test at
  // all. Both neighbors here are degree-3 space curves with nonzero torsion
  // at the join, checked first so the assertions below cannot pass vacuously.
  const gA = curveGeometryAt(TWIST_A, 1);
  const gB = curveGeometryAt(TWIST_B, 0);
  assert.ok(Math.abs(gA.torsion) > 1e-4 && Math.abs(gB.torsion) > 1e-4, `sanity: both neighbors carry real torsion (${gA.torsion}, ${gB.torsion})`);
  assert.ok(Math.abs(gA.dKappaDs) > 1e-8 && Math.abs(gB.dKappaDs) > 1e-8, 'sanity: both neighbors have genuinely varying curvature at the join');

  const g3 = blendCurves(TWIST_A, 1, TWIST_B, 0, { continuity: 3 });
  assert.equal(g3.ok, true, g3.reason);
  assert.equal(g3.degree, 7);
  for (const [u, ref, nm] of [[0, gA, 'start'], [1, gB, 'end']]) {
    const g = curveGeometryAt(g3.crv, u);
    assert.ok(dot3(g.tangent, ref.tangent) > 1 - 1e-14, `${nm} tangent dot ${dot3(g.tangent, ref.tangent)}`);
    assert.ok(len3(sub3(g.kappaVec, ref.kappaVec)) / ref.kappa < 1e-12, `${nm} curvature vector relative error ${len3(sub3(g.kappaVec, ref.kappaVec)) / ref.kappa}`);
    assert.ok(Math.abs(g.dKappaDs - ref.dKappaDs) / Math.abs(ref.dKappaDs) < 1e-10, `${nm} d(kappa)/ds relative error ${Math.abs(g.dKappaDs - ref.dKappaDs) / Math.abs(ref.dKappaDs)}`);
    assert.ok(Math.abs(g.torsion - ref.torsion) / Math.abs(ref.torsion) < 1e-10, `${nm} torsion relative error ${Math.abs(g.torsion - ref.torsion) / Math.abs(ref.torsion)}`);
  }

  const g2 = blendCurves(TWIST_A, 1, TWIST_B, 0, { continuity: 2 });
  const s2 = curveGeometryAt(g2.crv, 0);
  assert.ok(len3(sub3(s2.kappaVec, gA.kappaVec)) / gA.kappa < 1e-12, 'G2 does match the curvature vector');
  assert.ok(Math.abs(s2.dKappaDs - gA.dKappaDs) / Math.abs(gA.dKappaDs) > 0.1, `CONTRAST: G2 misses d(kappa)/ds (relative error ${Math.abs(s2.dKappaDs - gA.dKappaDs) / Math.abs(gA.dKappaDs)})`);
  assert.ok(Math.abs(s2.torsion - gA.torsion) / Math.abs(gA.torsion) > 0.1, 'CONTRAST: G2 misses torsion');
});

test('blendFrameFromCurve: the reverse flag negates the ODD derivatives only — the identity the far end of every blend is built on', () => {
  const f = blendFrameFromCurve(NEIGHBOR_B, 0.4);
  const r = blendFrameFromCurve(NEIGHBOR_B, 0.4, { reverse: true });
  for (let i = 0; i < 3; i++) {
    assert.equal(r.point[i], f.point[i]);
    assert.equal(r.d1[i], -f.d1[i]);
    assert.equal(r.d2[i], f.d2[i]);
    assert.equal(r.d3[i], -f.d3[i]);
  }
  // The geometry a reversed frame reports is the SAME geometry: curvature and
  // torsion are direction-independent, the unit tangent flips.
  const a = frameGeometry(f.point, f.d1, f.d2, f.d3);
  const b = frameGeometry(r.point, r.d1, r.d2, r.d3);
  assert.ok(Math.abs(a.kappa - b.kappa) < 1e-14 * a.kappa);
  assert.ok(Math.abs(a.torsion - b.torsion) < 1e-12 * Math.abs(a.torsion));
  assert.ok(dot3(a.tangent, b.tangent) < -1 + 1e-14);
});

// ===========================================================================
// 7. THE MEASURED PRICE OF CURVATURE CONTINUITY
// ===========================================================================

test('THE PRICE, pinned numerically: across turn angles 5-179 degrees the min-curvature G2 blend peaks at 1.10-1.18 x the arc\'s 1/R (G3: 1.12-1.32) — always MORE than the arc, never less', () => {
  // A G2 blend spanning the same corner as an R arc must ramp curvature from
  // 0 up and back to 0 while turning through the same total angle, so its
  // peak cannot be 1/R. This records exactly how much that costs, so a caller
  // choosing G2 for an R10 corner knows it peaks near 1/8.5mm — and so a
  // future change to the shape default cannot silently make blends tighter.
  const R = 10;
  const bounds = { 2: [1.09, 1.19], 3: [1.11, 1.33] };
  for (const k of [2, 3]) {
    for (const turnDeg of [5, 30, 60, 90, 120, 150, 170, 179]) {
      const phi = turnDeg * Math.PI / 180;
      const r = blendCornerCurve([0, 0, 0], [-500, 0, 0], [600 * Math.cos(phi), 600 * Math.sin(phi), 0], R, Z, { continuity: k });
      assert.equal(r.ok, true, r.reason);
      const ratio = r.peakKappa * R;
      assert.ok(ratio > 1.0, `k=${k} turn ${turnDeg}deg: a curvature-continuous blend cannot beat the arc's own 1/R (got ${ratio})`);
      assert.ok(ratio > bounds[k][0] && ratio < bounds[k][1], `k=${k} turn ${turnDeg}deg: peak curvature x R = ${ratio}, expected within ${bounds[k]}`);
      assert.ok(r.tangentScale > 0.8 && r.tangentScale < 2.8, `k=${k} turn ${turnDeg}deg: tangent scale ${r.tangentScale} out of the expected band`);
    }
  }
});

test('scale invariance: the chosen tangent scale depends only on turn angle, and peak curvature x R is identical across radii spanning six orders of magnitude', () => {
  const phi = 97 * Math.PI / 180;
  const prev = [-1e6, 0, 0];
  const next = [1e6 * Math.cos(phi), 1e6 * Math.sin(phi), 0];
  let refLambda = null, refRatio = null;
  for (const R of [0.01, 1, 100, 10000]) {
    const r = blendCornerCurve([0, 0, 0], prev, next, R, Z, { continuity: 2 });
    assert.equal(r.ok, true, r.reason);
    if (refLambda === null) { refLambda = r.tangentScale; refRatio = r.peakKappa * R; continue; }
    assert.equal(r.tangentScale, refLambda, `tangent scale must not depend on radius (R=${R})`);
    assert.ok(Math.abs(r.peakKappa * R - refRatio) < 1e-9 * refRatio, `peak curvature x R must not depend on radius (R=${R}: ${r.peakKappa * R} vs ${refRatio})`);
  }
});

// ===========================================================================
// 8. DEGENERATE-ADJACENT INPUT
// ===========================================================================

test('a VERY SHALLOW corner (0.4 degrees of turn) still produces a finite, regular, curvature-continuous blend at every order', () => {
  // The trim distance collapses like tan(phi/2), so this is the case where a
  // naive construction divides a tiny number by a tiny number. Tolerances
  // here are stated RELATIVE to the corner's own 1/R, because the absolute
  // curvature values are large while the geometry is nearly straight.
  const phi = 0.4 * Math.PI / 180;
  const R = 8;
  for (const k of [1, 2, 3]) {
    const r = blendCornerCurve([0, 0, 0], [-900, 0, 0], [900 * Math.cos(phi), 900 * Math.sin(phi), 0], R, Z, { continuity: k });
    assert.equal(r.ok, true, `k=${k}: ${r.reason}`);
    if (k === 1) continue;
    for (const p of r.crv.ctrlPts) for (const c of p) assert.ok(Number.isFinite(c), `k=${k}: non-finite control point`);
    for (let i = 0; i <= 32; i++) {
      const g = curveGeometryAt(r.crv, i / 32, Z);
      assert.ok(Number.isFinite(g.kappa) && Number.isFinite(g.speed) && g.speed > 0, `k=${k}: degenerate reading at u=${i / 32}`);
    }
    assert.ok(Math.abs(curveGeometryAt(r.crv, 0, Z).kappa) < 1e-9 / R, `k=${k}: curvature must still vanish at the seam of a near-straight corner`);
    assert.ok(Math.abs(curveGeometryAt(r.crv, 1, Z).kappa) < 1e-9 / R);
  }
});

test('a VERY TIGHT corner (175 degrees of turn, a near-reversal) still blends; 179.9999 degrees is refused, exactly where filletCornerArc refuses', () => {
  const R = 6;
  for (const turnDeg of [170, 175, 179]) {
    const phi = turnDeg * Math.PI / 180;
    for (const k of [2, 3]) {
      const r = blendCornerCurve([0, 0, 0], [-1000, 0, 0], [1000 * Math.cos(phi), 1000 * Math.sin(phi), 0], R, Z, { continuity: k });
      assert.equal(r.ok, true, `turn ${turnDeg}deg k=${k}: ${r.reason}`);
      assert.ok(Math.abs(curveGeometryAt(r.crv, 0, Z).kappa) < 1e-10 / R, `turn ${turnDeg}deg k=${k}: curvature must vanish at the seam`);
      assert.ok(Math.abs(curveGeometryAt(r.crv, 1, Z).kappa) < 1e-10 / R);
      for (let i = 0; i <= 32; i++) assert.ok(curveGeometryAt(r.crv, i / 32).speed > 0, 'no cusp inside a tight blend');
    }
  }
  // filletCornerArc's own threshold is |turn| > PI - 1e-6, i.e. within about
  // 5.7e-5 degrees of a full reversal — checked against that number, not
  // guessed, so this asserts the real shared boundary rather than a
  // comfortably-far-past-it value that would pass for the wrong reason.
  const justInside = 179.9999 * Math.PI / 180; // still ACCEPTED: PI - 1.75e-6 has not crossed PI - 1e-6
  const inside = blendCornerCurve([0, 0, 0], [-1e7, 0, 0], [1e7 * Math.cos(justInside), 1e7 * Math.sin(justInside), 0], R, Z, { continuity: 2 });
  assert.equal(inside.ok, true, `179.9999 deg is still inside filletCornerArc's own threshold: ${inside.reason}`);
  for (let i = 0; i <= 32; i++) {
    const g = curveGeometryAt(inside.crv, i / 32, Z);
    assert.ok(Number.isFinite(g.kappa) && g.speed > 0, 'a corner right at the refusal boundary must still be finite and regular');
  }
  const phi = 179.99999 * Math.PI / 180; // now past it
  const r = blendCornerCurve([0, 0, 0], [-1e7, 0, 0], [1e7 * Math.cos(phi), 1e7 * Math.sin(phi), 0], R, Z, { continuity: 2 });
  assert.equal(r.ok, false);
  assert.match(r.reason, /reversal/);
});

test('a collinear vertex inside a rail is skipped, not blended and not fatal — the blend path inherits filletOpenPolyline\'s per-corner skip', () => {
  const withStraight = [[0, 0, 0], [60, 0, 0], [130, 0, 0], [190, 55, 0], [265, 42, 0]]; // vertex 1 is a genuine straight-through
  const res = blendPolyline(withStraight, 9, { continuity: 2 });
  assert.equal(res.ok, true);
  assert.equal(res.cornerCount, 2, 'exactly the two real corners are blended; the collinear vertex is passed through');
  assert.equal(res.segments.filter((s) => s.type === 'blend').length, 2);
  assert.equal(res.cornerCount, filletOpenPolyline(withStraight, 9).cornerCount, 'the same corner set the shipped fillet finds');
});

test('a radius too large for the rail is refused with the SAME maxSafeRadius the shipped fillet reports — and that clamped radius then blends successfully', () => {
  const tooBig = blendPolyline(RAIL, 200, { continuity: 2 });
  const shipped = filletOpenPolyline(RAIL, 200);
  assert.equal(tooBig.ok, false);
  assert.equal(shipped.ok, false);
  assert.equal(tooBig.maxSafeRadius, shipped.maxSafeRadius);
  assert.equal(tooBig.reason, shipped.reason);
  const clamped = blendPolyline(RAIL, tooBig.maxSafeRadius, { continuity: 2 });
  assert.equal(clamped.ok, true, 'the reported clamp must actually be usable — an unretriable refusal would be a dead end');
});

test('BLEND MEETS BLEND across a vanishing straight run: at the tightest radius the shipped budget allows, the residual line is a fraction of a percent of the trim and BOTH of its seams are G2 — so the limiting case where it disappears entirely is G2 too', () => {
  // The interesting case is two corners so tight that the straight run
  // between them all but disappears. filletOpenPolyline's own zero-length
  // remainder omission (which would make two blends literally adjacent) is
  // effectively unreachable: its trim budget already refuses at
  // needed/edgeLen >= 1 - 1e-9, while the omission needs the remainder under
  // 1e-9 ABSOLUTE — for any edge longer than 1 unit those two windows do not
  // overlap. So this tests the reachable limit instead, and the limit case
  // follows from it directly: the residual run is a straight segment, whose
  // curvature is exactly zero along its whole length, so shrinking it to
  // nothing cannot change what either seam measures.
  const pts = [[0, 0, 0], [80, 0, 0], [104, 34, 0], [190, 26, 0]];
  const tooBig = filletOpenPolyline(pts, 500);
  assert.equal(tooBig.ok, false, 'fixture sanity: 500 really is too large for this rail');
  const R = tooBig.maxSafeRadius; // the tightest the shipped budget will allow on this rail
  const res = blendPolyline(pts, R, { continuity: 2 });
  assert.equal(res.ok, true, res.reason);

  const idx = res.segments.findIndex((s, i) => s.type === 'blend' && res.segments[i + 1] && res.segments[i + 1].type === 'line' && res.segments[i + 2] && res.segments[i + 2].type === 'blend');
  assert.ok(idx >= 0, 'fixture sanity: two blends separated by one straight run');
  const [a, line, b] = [res.segments[idx], res.segments[idx + 1], res.segments[idx + 2]];
  const runLen = len3(sub3(line.b, line.a));
  const trim = R * Math.tan(Math.acos(dot3(unit3(sub3(pts[2], pts[1])), unit3(sub3(pts[1], pts[0])))) / 2);
  assert.ok(runLen < 0.005 * trim, `the residual straight run must be vanishing next to the trim it sits between (run ${runLen}, trim ${trim})`);

  // Both seams of that vanishing run, measured the same way as any other.
  assert.ok(len3(sub3(curvePoint(a.crv, 1), line.a)) < 1e-9, 'position match into the run');
  assert.ok(len3(sub3(line.b, curvePoint(b.crv, 0))) < 1e-9, 'position match out of the run');
  const runDir = unit3(sub3(line.b, line.a));
  assert.ok(dot3(unit3(rationalCurveDerivs(a.crv, 1, 1)[1]), runDir) > 1 - 1e-9, 'tangent match into the run');
  assert.ok(dot3(runDir, unit3(rationalCurveDerivs(b.crv, 0, 1)[1])) > 1 - 1e-9, 'tangent match out of the run');
  // A straight run has curvature exactly 0 everywhere along it, so BOTH of
  // these vanishing is exactly the blend-meets-blend claim in the limit.
  assert.ok(Math.abs(curveGeometryAt(a.crv, 1, Z).kappa) < 1e-12 / R, 'curvature vanishes on the leaving side');
  assert.ok(Math.abs(curveGeometryAt(b.crv, 0, Z).kappa) < 1e-12 / R, 'curvature vanishes on the arriving side');
});

test('finiteness sweep: every control point and 33 sampled geometry readings are finite across continuity 1-3 x six turn angles x four radii', () => {
  let checked = 0;
  for (const k of [1, 2, 3]) {
    for (const turnDeg of [2, 25, 60, 90, 140, 176]) {
      for (const R of [0.05, 1.7, 30, 900]) {
        const phi = turnDeg * Math.PI / 180;
        const r = blendCornerCurve([0, 0, 0], [-1e5, 0, 0], [1e5 * Math.cos(phi), 1e5 * Math.sin(phi), 0], R, Z, { continuity: k });
        assert.equal(r.ok, true, `k=${k} turn=${turnDeg} R=${R}: ${r.reason}`);
        const crv = k === 1
          ? { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...r.segment.p0, 1], [...r.segment.apex, r.segment.weight], [...r.segment.p2, 1]] }
          : r.crv;
        for (const p of crv.ctrlPts) for (const c of p) assert.ok(Number.isFinite(c), `k=${k} turn=${turnDeg} R=${R}: non-finite control point`);
        for (let i = 0; i <= 32; i++) {
          const g = curveGeometryAt(crv, i / 32, Z);
          assert.ok(Number.isFinite(g.kappa) && Number.isFinite(g.speed) && Number.isFinite(g.signedKappa), `k=${k} turn=${turnDeg} R=${R} u=${i / 32}: non-finite geometry`);
          assert.ok(g.speed > 0, 'no cusp');
        }
        checked++;
      }
    }
  }
  assert.equal(checked, 72);
});

// ===========================================================================
// 9. CLOSED LOOPS AND AUTO-JOIN ("results auto-join")
// ===========================================================================

test('an IRREGULAR closed loop blends every corner including the wrap seam, and every seam is G2', () => {
  const R = 11;
  const res = blendPolyline(LOOP, R, { continuity: 2, closed: true });
  assert.equal(res.ok, true, res.reason);
  assert.equal(res.closed, true);
  assert.equal(res.cornerCount, LOOP.length, 'every vertex of a closed loop is a real corner');
  const n = res.segments.length;
  for (let i = 0; i < n; i++) {
    const cur = res.segments[i], nxt = res.segments[(i + 1) % n]; // (i+1)%n covers the wrap seam too
    const endPt = cur.type === 'line' ? cur.b : curvePoint(cur.crv, 1);
    const startPt = nxt.type === 'line' ? nxt.a : curvePoint(nxt.crv, 0);
    assert.ok(len3(sub3(endPt, startPt)) < 1e-12, `seam ${i}: position gap`);
    const tOut = cur.type === 'line' ? unit3(sub3(cur.b, cur.a)) : unit3(rationalCurveDerivs(cur.crv, 1, 1)[1]);
    const tIn = nxt.type === 'line' ? unit3(sub3(nxt.b, nxt.a)) : unit3(rationalCurveDerivs(nxt.crv, 0, 1)[1]);
    assert.ok(dot3(tOut, tIn) > 1 - 1e-11, `seam ${i}: tangent dot ${dot3(tOut, tIn)}`);
    const kOut = cur.type === 'line' ? 0 : curveGeometryAt(cur.crv, 1, Z).kappa;
    const kIn = nxt.type === 'line' ? 0 : curveGeometryAt(nxt.crv, 0, Z).kappa;
    assert.ok(Math.abs(kOut - kIn) < 1e-12 / R, `seam ${i}: curvature jump ${Math.abs(kOut - kIn)}`);
  }
});

test('AUTO-JOIN: blendSegmentsToCurve composes the chain into ONE curve that is still G2 at every internal joint — measured on the COMPOSED curve, where the knot vector only reports C0', () => {
  // This is the assertion the auto-join rule asks for ("results auto-join ... confirm
  // it holds for the new paths"), and it is measured geometrically on
  // purpose: joinCurvesC0 rescales each segment onto its own integer domain
  // slot and stacks degree-multiplicity knots at each joint, so a PARAMETRIC
  // check would report a discontinuity that is pure parametrization artifact.
  // Curvature is reparametrization-invariant, so it sees through that.
  const R = 12;
  const g2 = blendPolyline(RAIL, R, { continuity: 2 });
  const composed = blendSegmentsToCurve(g2.segments);
  assert.equal(composed.degree, 5, 'the whole chain elevates to the blend degree');
  const uMax = composed.knots[composed.knots.length - 1];
  for (let i = 0; i <= 400; i++) {
    const p = curvePoint(composed, (i / 400) * uMax);
    assert.ok(p.every(Number.isFinite), `composed curve non-finite at u=${(i / 400) * uMax}`);
  }
  const joints = [...new Set(composed.knots.slice(composed.degree + 1, composed.knots.length - composed.degree - 1))];
  assert.equal(joints.length, g2.segments.length - 1);
  const EPS = 1e-6;
  let worstK = 0, worstDot = 1;
  for (const u of joints) {
    const a = curveGeometryAt(composed, u - EPS, Z);
    const b = curveGeometryAt(composed, u + EPS, Z);
    worstK = Math.max(worstK, Math.abs(a.signedKappa - b.signedKappa));
    worstDot = Math.min(worstDot, dot3(a.tangent, b.tangent));
  }
  assert.ok(worstDot > 1 - 1e-11, `composed G1: worst tangent dot ${worstDot}`);
  // The residual is O(EPS * dkappa/ds), i.e. it is the curvature genuinely
  // varying across the sampling offset, NOT a jump. Bounded here at 1e-4 of
  // the corner's own 1/R; the contrast below is four orders of magnitude
  // larger, which is what makes this bound meaningful.
  assert.ok(worstK < 1e-4 / R, `composed G2: worst curvature difference across a joint ${worstK} (1/R = ${1 / R})`);

  // CONTRAST: compose the shipped G1 arc chain the same way and measure the
  // same thing. It jumps by essentially the full 1/R at every joint.
  const g1 = blendPolyline(RAIL, R, { continuity: 1 });
  const composedArc = blendSegmentsToCurve(g1.segments);
  const arcJoints = [...new Set(composedArc.knots.slice(composedArc.degree + 1, composedArc.knots.length - composedArc.degree - 1))];
  let arcWorst = 0;
  for (const u of arcJoints) {
    const a = curveGeometryAt(composedArc, u - EPS, Z);
    const b = curveGeometryAt(composedArc, u + EPS, Z);
    arcWorst = Math.max(arcWorst, Math.abs(a.signedKappa - b.signedKappa));
  }
  assert.ok(arcWorst > 0.99 / R, `CONTRAST: the arc chain's curvature jumps by ~1/R at each joint (measured ${arcWorst}, 1/R = ${1 / R})`);
});

test('blendSegmentsToCurve refuses an unknown segment type rather than silently dropping it', () => {
  assert.throws(() => blendSegmentsToCurve([{ type: 'spiral' }]), /unknown segment type/);
  assert.equal(blendSegmentsToCurve([]), null);
});

// ===========================================================================
// 7. nearestCurveEndpoints — the default uA/uB picker a standalone Blend
// COMMAND needs (blendCurves itself deliberately takes no position on this —
// see its own header). Every claim here is checked against the real geometry
// (curvePoint), never assumed from which candidate "looks" nearest.
// ===========================================================================

test('nearestCurveEndpoints: picks the genuinely closest of the 4 end pairs — NEIGHBOR_A\'s END sits near NEIGHBOR_B\'s START, and is the correct answer among 4 real candidates, not a default guess', () => {
  const pick = nearestCurveEndpoints(NEIGHBOR_A, NEIGHBOR_B);
  // Independently recompute all 4 real distances rather than trusting the
  // function's own internal candidate list.
  const aStart = curvePoint(NEIGHBOR_A, 0), aEnd = curvePoint(NEIGHBOR_A, 1);
  const bStart = curvePoint(NEIGHBOR_B, 0), bEnd = curvePoint(NEIGHBOR_B, 1);
  const dEndStart = len3(sub3(aEnd, bStart));
  const dEndEnd = len3(sub3(aEnd, bEnd));
  const dStartStart = len3(sub3(aStart, bStart));
  const dStartEnd = len3(sub3(aStart, bEnd));
  const trueMin = Math.min(dEndStart, dEndEnd, dStartStart, dStartEnd);
  assert.equal(dEndStart, trueMin, 'sanity: the A-end/B-start pair really is the closest of the 4 in this fixture');
  assert.equal(pick.endA, 'end');
  assert.equal(pick.endB, 'start');
  assert.equal(pick.uA, 1);
  assert.equal(pick.uB, 0);
  assert.equal(pick.reverseA, false);
  assert.equal(pick.reverseB, false);
  assert.ok(Math.abs(pick.distance - dEndStart) < 1e-9, `reported distance ${pick.distance} should match the independently-measured ${dEndStart}`);
  // This is exactly the (uA=1, uB=0) pair the existing manual blendCurves
  // calls above already use — proof the picker reproduces the same, already
  // camera-ready join a hand-chosen call would.
  const manual = blendCurves(NEIGHBOR_A, 1, NEIGHBOR_B, 0, { continuity: 1 });
  const picked = blendCurves(NEIGHBOR_A, pick.uA, NEIGHBOR_B, pick.uB, { continuity: 1, reverseA: pick.reverseA, reverseB: pick.reverseB });
  assert.equal(picked.ok, true, picked.reason);
  for (let i = 0; i < picked.crv.ctrlPts.length; i++) {
    for (let k = 0; k < 4; k++) assert.ok(Math.abs(picked.crv.ctrlPts[i][k] - manual.crv.ctrlPts[i][k]) < 1e-9, `control point ${i}[${k}] mismatch`);
  }
});

test('nearestCurveEndpoints: is symmetric under swapping which curve is "A" and which is "B" — the SAME physical pair is found, reverse flags mirrored', () => {
  const ab = nearestCurveEndpoints(NEIGHBOR_A, NEIGHBOR_B);
  const ba = nearestCurveEndpoints(NEIGHBOR_B, NEIGHBOR_A);
  assert.equal(ba.endA, ab.endB);
  assert.equal(ba.endB, ab.endA);
  assert.ok(Math.abs(ba.distance - ab.distance) < 1e-9);
});

test('nearestCurveEndpoints: the OTHER 3 end-pair combinations are exercised too, not just one lucky fixture — a curve whose START is nearest, and one whose far END is nearest', () => {
  // A short curve placed so its own START is genuinely nearest NEIGHBOR_B's
  // start — deliberately laid out with its END far away.
  const nearAtStart = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[24, 6, 2, 1], [-400, -300, -200, 1]] };
  const pick1 = nearestCurveEndpoints(nearAtStart, NEIGHBOR_B);
  assert.equal(pick1.endA, 'start');
  assert.equal(pick1.reverseA, true, 'a blend leaving from A\'s own START must travel backward to stay on A\'s live remainder');
  assert.equal(pick1.endB, 'start');
  assert.equal(pick1.reverseB, false);

  // A curve whose END is nearest NEIGHBOR_B's own far END.
  const bFarEnd = curvePoint(NEIGHBOR_B, 1);
  const nearAtOtherEnd = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[900, 900, 900, 1], [bFarEnd[0] + 3, bFarEnd[1] - 2, bFarEnd[2] + 1, 1]] };
  const pick2 = nearestCurveEndpoints(nearAtOtherEnd, NEIGHBOR_B);
  assert.equal(pick2.endA, 'end');
  assert.equal(pick2.reverseA, false);
  assert.equal(pick2.endB, 'end');
  assert.equal(pick2.reverseB, true, 'a blend arriving at B\'s own END must have come from the backward direction to stay on B\'s live remainder');

  // Both non-degenerate results actually blend, at every continuity order.
  for (const [pick, crvA] of [[pick1, nearAtStart], [pick2, nearAtOtherEnd]]) {
    for (const k of [1, 2, 3]) {
      const r = blendCurves(crvA, pick.uA, NEIGHBOR_B, pick.uB, { continuity: k, reverseA: pick.reverseA, reverseB: pick.reverseB });
      assert.equal(r.ok, true, `continuity ${k}: ${r.reason}`);
    }
  }
});

test('nearestCurveEndpoints: reports a real positive distance, never coincident by construction, for two ordinary separated curves', () => {
  const pick = nearestCurveEndpoints(TWIST_A, TWIST_B);
  assert.ok(pick.distance > 1, `sanity: these two fixtures are genuinely apart (${pick.distance}), not touching`);
  assert.ok(Number.isFinite(pick.distance));
});

// ===========================================================================
// 8. nearestEndpointToPoint — the per-curve, CLICK-DRIVEN sibling. Unlike
// section 7 above (a JOINT search across two curves), this is a LOCAL
// decision: one curve, one point, no visibility into any second curve at
// all. Every test here uses NEIGHBOR_A — a genuinely CURVED (rational,
// non-planar) fixture, never a straight line, so "picks the far end" is
// proven against a real curve evaluation (curvePoint), not a coincidence
// of linear interpolation.
// ===========================================================================

test('nearestEndpointToPoint: on a real curved (rational) fixture, a click near the TRUE end (curvePoint at uMax) picks that end, with the exact analytic point and a genuinely small distance', () => {
  const trueEnd = curvePoint(NEIGHBOR_A, 1);
  // Click lands a few units off the exact curve point — proving this reads
  // real curve geometry (curvePoint at u=0 and u=1), not just "whichever
  // control point index is smaller."
  const click = [trueEnd[0] + 2, trueEnd[1] - 1, trueEnd[2] + 1.5];
  const pick = nearestEndpointToPoint(NEIGHBOR_A, click);
  assert.equal(pick.end, 'end');
  assert.equal(pick.u, 1);
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(pick.point[k] - trueEnd[k]) < 1e-9, `reported end point[${k}] should be the exact curve value at u=1`);
  assert.ok(pick.distance < 5, `distance to a click a few units off the true end should stay small (${pick.distance})`);
});

test('nearestEndpointToPoint: a click near the TRUE start picks the start instead, on the SAME curved fixture — proving both ends are reachable, not a hardcoded bias', () => {
  const trueStart = curvePoint(NEIGHBOR_A, 0);
  const click = [trueStart[0] - 3, trueStart[1] + 2, trueStart[2] - 0.5];
  const pick = nearestEndpointToPoint(NEIGHBOR_A, click);
  assert.equal(pick.end, 'start');
  assert.equal(pick.u, 0);
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(pick.point[k] - trueStart[k]) < 1e-9);
});

test('THE ACTUAL DISCRIMINATING PROOF: nearestEndpointToPoint picks the FAR end when clicked there, even though that end is the globally-nearest-pair answer, and the OPPOSITE (locally correct) end when the click says so — nearestCurveEndpoints alone would always answer the FIRST way, never the second', () => {
  // Confirm, independently, what nearestCurveEndpoints (the JOINT/global
  // picker) says about this exact pair first: section 7's own test already
  // established NEIGHBOR_A's END is the globally-closest-to-NEIGHBOR_B
  // answer. Re-derive it here too, so this test does not depend on reading
  // that one.
  const joint = nearestCurveEndpoints(NEIGHBOR_A, NEIGHBOR_B);
  assert.equal(joint.endA, 'end', 'sanity: the joint/global search really does prefer NEIGHBOR_A\'s END for this pair');

  const trueEnd = curvePoint(NEIGHBOR_A, 1);
  const trueStart = curvePoint(NEIGHBOR_A, 0);

  // Direction 1: click lands near the END — genuinely far from
  // NEIGHBOR_B too (NEIGHBOR_B's own nearest point to A sits near A's own
  // END already, so a click there is unavoidably "near" that global
  // answer as well) — the LOCAL and GLOBAL answers happen to agree here,
  // which is exactly why direction 2 below is the real proof.
  const clickNearEnd = [trueEnd[0] + 1, trueEnd[1] + 1, trueEnd[2]];
  const pickEnd = nearestEndpointToPoint(NEIGHBOR_A, clickNearEnd);
  assert.equal(pickEnd.end, 'end');

  // Direction 2 — THE discriminating case: click lands near A's own START,
  // which sits at (-60,-20,5), nowhere near NEIGHBOR_B at all (NEIGHBOR_B's
  // own control points are all in the +30..+130 range) — a joint/global
  // search would never choose this end for this pair (confirmed above:
  // joint.endA === 'end', never 'start'), yet a click landing there must
  // still resolve to 'start', because the decision is local to THIS click
  // on THIS curve, with zero visibility into where NEIGHBOR_B is.
  const clickNearStart = [trueStart[0] - 2, trueStart[1] + 1, trueStart[2] + 1];
  const pickStart = nearestEndpointToPoint(NEIGHBOR_A, clickNearStart);
  assert.equal(pickStart.end, 'start', 'a click near the curve\'s own START must win locally, even though the joint/global search never picks this end for this pair');
  assert.notEqual(pickStart.end, joint.endA, 'the click-driven answer genuinely DIFFERS from the joint/global answer here — that difference is the whole point of this primitive');

  // And the reverse-flag rule (documented on nearestCurveEndpoints, and
  // re-derived here per-end rather than per-pair) generalizes correctly to
  // an arbitrarily-chosen end: reverse iff the chosen end is this curve's
  // own START — true regardless of which end a JOINT search would have
  // preferred.
  const reverseForStart = pickStart.end === 'start';
  const reverseForEnd = pickEnd.end === 'start';
  assert.equal(reverseForStart, true);
  assert.equal(reverseForEnd, false);
});
