import test from 'node:test';
import assert from 'node:assert/strict';
import { curvePoint, curvePointAndTangent, rationalCurveDerivs, grevilleAbscissae, reverseCurve, divideByArcLength, curveLength, isCurveClosed, buildArcLengthTable, paramAtArcLength } from '../kernel/curve.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';

test('degree-1 line curve evaluates to exact linear interpolation', () => {
  const line = {
    degree: 1,
    knots: [0, 0, 1, 1],
    ctrlPts: [[0, 0, 0, 1], [10, 20, 30, 1]],
  };
  for (let u = 0; u <= 1; u += 0.1) {
    const p = curvePoint(line, u);
    assert.ok(Math.abs(p[0] - 10 * u) < 1e-10);
    assert.ok(Math.abs(p[1] - 20 * u) < 1e-10);
    assert.ok(Math.abs(p[2] - 30 * u) < 1e-10);
  }
});

test('degree-1 line curve tangent is constant and correct direction', () => {
  const line = {
    degree: 1,
    knots: [0, 0, 1, 1],
    ctrlPts: [[0, 0, 0, 1], [3, 4, 0, 1]], // length-5 direction
  };
  const { tangent } = curvePointAndTangent(line, 0.5);
  assert.ok(Math.abs(tangent[0] - 3 / 5) < 1e-10);
  assert.ok(Math.abs(tangent[1] - 4 / 5) < 1e-10);
});

// The standard textbook exact-rational unit circle: degree 2, 9 control
// points (4 quarter-arcs), weights alternating 1 / (sqrt(2)/2).
const s = Math.SQRT1_2;
const unitCircle = {
  degree: 2,
  knots: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4],
  ctrlPts: [
    [1, 0, 0, 1], [1, 1, 0, s], [0, 1, 0, 1],
    [-1, 1, 0, s], [-1, 0, 0, 1], [-1, -1, 0, s],
    [0, -1, 0, 1], [1, -1, 0, s], [1, 0, 0, 1],
  ],
};

test('unit circle: every evaluated point is at radius 1 from origin', () => {
  for (let u = 0; u <= 4; u += 0.05) {
    const [x, y, z] = curvePoint(unitCircle, u);
    const r = Math.sqrt(x * x + y * y + z * z);
    assert.ok(Math.abs(r - 1) < 1e-9, `u=${u} r=${r}`);
  }
});

test('unit circle: tangent is perpendicular to the radius vector', () => {
  for (let u = 0.1; u < 4; u += 0.31) {
    const { point, tangent } = curvePointAndTangent(unitCircle, u);
    const dot = point[0] * tangent[0] + point[1] * tangent[1] + point[2] * tangent[2];
    assert.ok(Math.abs(dot) < 1e-8, `u=${u} dot=${dot}`);
  }
});

test('curvature (2nd derivative) of the unit circle points inward, magnitude ~1', () => {
  // For a unit-speed... actually this parametrization isn't unit-speed, so
  // just check the 2nd derivative has a negative component along the radius
  // (curving toward the center) at a sample point deep inside a span.
  const [C0, , C2] = rationalCurveDerivs(unitCircle, 0.5, 2);
  const radial = C0[0] * C2[0] + C0[1] * C2[1];
  assert.ok(radial < 0, `expected inward curvature, got radial=${radial}`);
});

test('grevilleAbscissae returns one value per control point, within the domain', () => {
  const g = grevilleAbscissae(unitCircle);
  assert.equal(g.length, unitCircle.ctrlPts.length);
  for (const v of g) {
    assert.ok(v >= 0 && v <= 4);
  }
});

test('bilinear surface (degree1 x degree1) evaluates to exact bilinear interpolation', () => {
  const srf = {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [
      [[0, 0, 0, 1], [0, 10, 0, 1]],
      [[10, 0, 5, 1], [10, 10, 5, 1]],
    ],
  };
  const p = surfacePoint(srf, 0.5, 0.5);
  assert.ok(Math.abs(p[0] - 5) < 1e-10);
  assert.ok(Math.abs(p[1] - 5) < 1e-10);
  assert.ok(Math.abs(p[2] - 2.5) < 1e-10);
});

test('reverseCurve of a rational, non-uniform-knot curve (the unit circle) occupies the EXACT SAME point set, traversed backward', () => {
  const rev = reverseCurve(unitCircle);
  assert.equal(rev.degree, unitCircle.degree);
  assert.equal(rev.knots.length, unitCircle.knots.length);
  const [a, b] = [unitCircle.knots[0], unitCircle.knots[unitCircle.knots.length - 1]];
  for (let u = 0; u <= 4; u += 0.1) {
    const p1 = curvePoint(unitCircle, u);
    const p2 = curvePoint(rev, a + b - u); // the reversed curve at the mirrored parameter...
    assert.ok(Math.abs(p1[0] - p2[0]) < 1e-9, `x mismatch at u=${u}`);
    assert.ok(Math.abs(p1[1] - p2[1]) < 1e-9, `y mismatch at u=${u}`);
    assert.ok(Math.abs(p1[2] - p2[2]) < 1e-9, `z mismatch at u=${u}`);
  }
});

test('reverseCurve swaps the endpoints exactly (start<->end control points)', () => {
  const rev = reverseCurve(unitCircle);
  const n = unitCircle.ctrlPts.length;
  assert.deepEqual(rev.ctrlPts[0], unitCircle.ctrlPts[n - 1]);
  assert.deepEqual(rev.ctrlPts[n - 1], unitCircle.ctrlPts[0]);
  // domain (first/last knot) is unchanged — only the interior traversal flips
  assert.equal(rev.knots[0], unitCircle.knots[0]);
  assert.equal(rev.knots[rev.knots.length - 1], unitCircle.knots[unitCircle.knots.length - 1]);
});

test('reverseCurve applied TWICE returns the original curve (own inverse)', () => {
  const rev = reverseCurve(unitCircle);
  const back = reverseCurve(rev);
  assert.deepEqual(back.ctrlPts, unitCircle.ctrlPts);
  for (let i = 0; i < back.knots.length; i++) {
    assert.ok(Math.abs(back.knots[i] - unitCircle.knots[i]) < 1e-12, `knot ${i} mismatch`);
  }
});

test('reverseCurve flips the endpoint tangent DIRECTION (a straight line traversed backward points the other way)', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [3, 4, 0, 1]] };
  const rev = reverseCurve(line);
  const { tangent: fwd } = curvePointAndTangent(line, 0.5);
  const { tangent: back } = curvePointAndTangent(rev, 0.5);
  assert.ok(Math.abs(fwd[0] + back[0]) < 1e-10);
  assert.ok(Math.abs(fwd[1] + back[1]) < 1e-10);
  assert.ok(Math.abs(fwd[2] + back[2]) < 1e-10);
});

test('isFiniteNet rejects NaN and non-positive weight, accepts a clean net', () => {
  assert.equal(isFiniteNet([[[1, 2, 3, 1]]]), true);
  assert.equal(isFiniteNet([[[1, NaN, 3, 1]]]), false);
  assert.equal(isFiniteNet([[[1, 2, 3, 0]]]), false);
});

// DIVIDE (divideByArcLength) — a straight degree-1 line's own parametrization
// already IS arc-length-proportional (constant speed), so this is the
// simplest possible ground truth: dividing into N segments must land
// exactly on the N+1 evenly-spaced points along the line, not merely
// "on the line somewhere."
test('divideByArcLength on a straight line lands on exact evenly-spaced points', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1]] };
  const divs = divideByArcLength(line, 5);
  assert.equal(divs.length, 6);
  for (let i = 0; i <= 5; i++) {
    assert.ok(Math.abs(divs[i].point[0] - (10 * i) / 5) < 1e-6, `point ${i}: ${divs[i].point[0]}`);
    assert.ok(Math.abs(divs[i].point[1]) < 1e-9 && Math.abs(divs[i].point[2]) < 1e-9);
  }
});

test('divideByArcLength on an OPEN curve (a straight line — start/end control points genuinely distinct) always returns count+1 points including BOTH exact curve endpoints', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1]] };
  assert.equal(isCurveClosed(line), false, 'a straight line is never closed');
  const divs = divideByArcLength(line, 7);
  assert.equal(divs.length, 8);
  assert.equal(divs[0].u, line.knots[0]);
  assert.equal(divs[7].u, line.knots[line.knots.length - 1]);
  const p0 = curvePoint(line, line.knots[0]);
  const pLast = curvePoint(line, line.knots[line.knots.length - 1]);
  assert.ok(Math.abs(divs[0].point[0] - p0[0]) < 1e-12);
  assert.ok(Math.abs(divs[7].point[0] - pLast[0]) < 1e-12);
});

// The unit circle's own first/last control point are the IDENTICAL point
// ([1,0,0,1], the seam), so isCurveClosed must recognize it as closed —
// dividing it into `count` must give exactly `count` points, never count+1
// (the old behavior emitted the seam point twice: once at u=uMin, once at
// u=uMax, the same physical location on a closed curve — real Rhino's own
// Divide on a closed curve gives exactly N points, not N+1).
test('divideByArcLength on a CLOSED curve (the unit circle) returns exactly `count` points, all DISTINCT (no seam duplicate)', () => {
  assert.equal(isCurveClosed(unitCircle), true, 'the unit circle\'s own first/last control point coincide — a real seam');
  const divs = divideByArcLength(unitCircle, 7);
  assert.equal(divs.length, 7, 'a closed curve divided into 7 must give exactly 7 points, not 8');
  assert.equal(divs[0].u, unitCircle.knots[0]);
  for (let i = 0; i < divs.length; i++) {
    for (let j = i + 1; j < divs.length; j++) {
      const d = Math.hypot(divs[i].point[0] - divs[j].point[0], divs[i].point[1] - divs[j].point[1], divs[i].point[2] - divs[j].point[2]);
      assert.ok(d > 1e-6, `points ${i} and ${j} coincide (seam duplicate): ${d}`);
    }
  }
});

// The full REAL point of the closed-curve fix: the `count` points must be
// evenly spaced around the FULL closed arc length, including the
// WRAPAROUND step from the last point back to the first — not just the
// interior chain up to a dropped seam.
test('divideByArcLength on a CLOSED curve places points evenly spaced around the FULL closed length, including the wraparound step', () => {
  const divs = divideByArcLength(unitCircle, 12);
  assert.equal(divs.length, 12);
  const angles = divs.map((d) => Math.atan2(d.point[1], d.point[0]));
  for (let i = 1; i < angles.length; i++) {
    let step = angles[i] - angles[i - 1];
    if (step < 0) step += 2 * Math.PI;
    assert.ok(Math.abs(step - Math.PI / 6) < 1e-4, `step ${i}: ${step} vs ${Math.PI / 6}`);
  }
  let wrap = angles[0] - angles[angles.length - 1];
  if (wrap < 0) wrap += 2 * Math.PI;
  assert.ok(Math.abs(wrap - Math.PI / 6) < 1e-4, `wraparound step (last point back to first) ${wrap} vs ${Math.PI / 6}`);
});

// The REAL point of arc-length (not raw-parameter) division: the standard
// rational quadratic circle has non-uniform angular speed in its own
// parameter (that's WHY the spec calls out "bunches on unevenly-
// parametrized curves" as the bug this avoids) — but arc-length-even
// division of a circle must still be angularly EQUAL, since arc length and
// angle are directly proportional on a circle of constant radius. A
// raw-parameter-uniform sample would fail this check (bunches near each
// quarter-arc's own rational weight peak).
test('divideByArcLength on the rational circle produces ANGULARLY EQUAL steps (proves real arc-length, not raw-parameter bunching)', () => {
  const divs = divideByArcLength(unitCircle, 12);
  assert.equal(divs.length, 12); // closed curve: exactly `count` points, no seam duplicate to slice off
  const angles = divs.map((d) => Math.atan2(d.point[1], d.point[0]));
  for (let i = 1; i < angles.length; i++) {
    let step = angles[i] - angles[i - 1];
    if (step < 0) step += 2 * Math.PI;
    assert.ok(Math.abs(step - Math.PI / 6) < 1e-4, `step ${i}: ${step} vs ${Math.PI / 6}`);
  }
});

test('divideByArcLength produces genuinely EQUAL chord lengths between consecutive division points, including the wraparound chord back to the first point', () => {
  const divs = divideByArcLength(unitCircle, 9);
  assert.equal(divs.length, 9); // closed curve: exactly `count` points
  const dists = [];
  for (let i = 1; i < divs.length; i++) {
    const [x0, y0, z0] = divs[i - 1].point, [x1, y1, z1] = divs[i].point;
    dists.push(Math.hypot(x1 - x0, y1 - y0, z1 - z0));
  }
  const [xf, yf, zf] = divs[divs.length - 1].point, [x0, y0, z0] = divs[0].point;
  dists.push(Math.hypot(x0 - xf, y0 - yf, z0 - zf)); // the wraparound chord
  for (const d of dists) assert.ok(Math.abs(d - dists[0]) < 1e-4, `chord ${d} vs ${dists[0]}`);
});

test('curveLength of the unit circle is approximately 2*pi (a real numerical arc length, not a placeholder)', () => {
  const len = curveLength(unitCircle, unitCircle.knots[0], unitCircle.knots[unitCircle.knots.length - 1], 1e-6);
  assert.ok(Math.abs(len - 2 * Math.PI) < 1e-3, `len=${len}`);
});

test('divideByArcLength rejects a non-positive or non-integer count', () => {
  assert.throws(() => divideByArcLength(unitCircle, 0));
  assert.throws(() => divideByArcLength(unitCircle, -3));
  assert.throws(() => divideByArcLength(unitCircle, 2.5));
});

// ---------------------------------------------------------------------------
// POINT-SYMMETRIC SPANS — the case a one-sample chord-deviation test cannot
// see. On a span with C(m+s) = 2C(m) - C(m-s) the midpoint lies EXACTLY on
// the endpoint chord, so the single deviation sample reads exactly zero even
// though the curve genuinely bows away everywhere else. Guarded in
// adaptiveArcLengthSamples by DIVIDE_MIN_DEPTH.
//
// Every check below measures against an INDEPENDENT dense-sampling arc
// length, never against the table under test — comparing the table to
// itself would prove nothing about the property that actually broke.
// ---------------------------------------------------------------------------

// Independent ground truth: a very dense uniform-parameter polyline. Slow and
// crude on purpose — it shares no code path with the adaptive refinement, so
// it cannot inherit the same blind spot.
function denseArcLength(crv, u0, u1, steps = 200000) {
  let prev = curvePoint(crv, u0);
  let total = 0;
  for (let i = 1; i <= steps; i++) {
    const p = curvePoint(crv, u0 + ((u1 - u0) * i) / steps);
    total += Math.hypot(p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]);
    prev = p;
  }
  return total;
}
function wholeCurveDenseLength(crv, steps) {
  return denseArcLength(crv, crv.knots[0], crv.knots[crv.knots.length - 1], steps);
}

// A cubic through 4 points chosen so the curve is point-symmetric about its
// own span midpoint: the S runs up, back down through the chord, and up
// again by the same amounts. Interpolation gives it NO interior knots, so
// the whole curve is one span and the symmetry is exact.
const symmetricS = globalCurveInterp([[0, 0, 0], [40, 0, 20], [80, 0, 0], [120, 0, 20]], 3);
// Deliberately NOT symmetric — the control fixture. Must stay exact and must
// not get materially more expensive.
const asymmetricS = globalCurveInterp([[0, 0, 0], [80, 0, 40], [150, 0, -10], [240, 0, 50]], 3);
// NEAR-symmetric, not exactly so: one endpoint nudged by a fraction of a
// millimeter. This is where a PARTIAL version of the bug lives — the
// midpoint deviation is tiny but nonzero, so whether the old code saw it at
// all depended entirely on how it compared against the auto-tolerance.
const nearSymmetricS = globalCurveInterp([[0, 0, 0], [40, 0, 20], [80, 0, 0], [120.00002, 0, 20.00002]], 3);
// The symmetric S bracketed by two further points, so it becomes ONE SPAN
// among several inside an otherwise ordinary hand-drawn-shaped curve. Proves
// the defect is per-span, not a property of wholly symmetric curves only.
const symmetricSpanInsideLongerCurve = globalCurveInterp(
  [[-50, 0, -10], [0, 0, 0], [40, 0, 20], [80, 0, 0], [120, 0, 20], [170, 0, 30]], 3,
);

test('a POINT-SYMMETRIC curve measures its true arc length (the midpoint deviation sample reads exactly zero there, so a single-sample test terminates at depth 0 and returns the straight chord)', () => {
  const uMin = symmetricS.knots[0], uMax = symmetricS.knots[symmetricS.knots.length - 1];
  const trueLen = wholeCurveDenseLength(symmetricS);
  // The property that makes this fixture discriminating: the midpoint really
  // IS on the chord, so a one-sample test genuinely cannot tell the curve
  // from its own chord. Asserted directly so a future fixture edit that
  // accidentally breaks the symmetry fails loudly instead of passing for the
  // wrong reason.
  const a = curvePoint(symmetricS, uMin), b = curvePoint(symmetricS, uMax);
  const mid = curvePoint(symmetricS, (uMin + uMax) / 2);
  const chordMid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  assert.ok(
    Math.hypot(mid[0] - chordMid[0], mid[1] - chordMid[1], mid[2] - chordMid[2]) < 1e-9,
    'fixture sanity: this curve must be genuinely point-symmetric about its own midpoint, or it does not exercise the defect at all',
  );
  const chordLen = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  assert.ok(trueLen > chordLen * 1.1, `fixture sanity: the curve must genuinely bow away from its chord (true ${trueLen}, chord ${chordLen})`);

  const table = buildArcLengthTable(symmetricS, uMin, uMax);
  assert.ok(
    Math.abs(table.total - trueLen) / trueLen < 1e-4,
    `arc-length table total ${table.total} must match the independently dense-sampled true length ${trueLen}`,
  );
});

test('a POINT-SYMMETRIC curve inverts length->parameter to genuinely non-uniform parameters (a chord-only table returns exactly the uniform fractions, so every arc-length-even consumer silently gets parameter-even stations)', () => {
  const uMin = symmetricS.knots[0], uMax = symmetricS.knots[symmetricS.knots.length - 1];
  const table = buildArcLengthTable(symmetricS, uMin, uMax);
  let maxOffUniform = 0;
  for (let i = 1; i < 8; i++) {
    const u = paramAtArcLength(table, (table.total * i) / 8);
    maxOffUniform = Math.max(maxOffUniform, Math.abs(u - (uMin + ((uMax - uMin) * i) / 8)));
  }
  assert.ok(maxOffUniform > 1e-3, `length->parameter inversion must differ from the uniform parameter fractions, worst difference was ${maxOffUniform}`);
});

test('DIVIDE on a POINT-SYMMETRIC curve places points whose successive TRUE arc-length gaps are equal (the user-facing property arc-length-evenness exists for, measured by dense sampling rather than by asking the same table again)', () => {
  const COUNT = 8;
  const divs = divideByArcLength(symmetricS, COUNT);
  assert.equal(divs.length, COUNT + 1, 'an open curve divides into count+1 points');
  const gaps = [];
  for (let i = 1; i < divs.length; i++) gaps.push(denseArcLength(symmetricS, divs[i - 1].u, divs[i].u, 20000));
  const mean = gaps.reduce((s, g) => s + g, 0) / gaps.length;
  for (const g of gaps) {
    assert.ok(Math.abs(g - mean) / mean < 5e-3, `every true arc-length gap must match the mean ${mean}; gaps were ${gaps.map((x) => x.toFixed(4)).join(', ')}`);
  }
  // And they are genuinely NOT the parameter-even stations, which is what a
  // chord-only table would have produced.
  const uniformGaps = [];
  for (let i = 1; i <= COUNT; i++) {
    uniformGaps.push(denseArcLength(symmetricS, symmetricS.knots[0] + ((i - 1) / COUNT), symmetricS.knots[0] + (i / COUNT), 20000));
  }
  const uniformSpread = (Math.max(...uniformGaps) - Math.min(...uniformGaps)) / mean;
  assert.ok(uniformSpread > 0.05, `fixture sanity: parameter-even stations must be measurably UNEVEN in true arc length on this curve (spread ${uniformSpread}), or this test could pass without the fix`);
});

test('a NEAR-symmetric curve (one endpoint nudged a fraction of a millimeter) measures its true arc length too — the partial case, where the midpoint deviation is nonzero but below the auto-tolerance', () => {
  const uMin = nearSymmetricS.knots[0], uMax = nearSymmetricS.knots[nearSymmetricS.knots.length - 1];
  const a = curvePoint(nearSymmetricS, uMin), b = curvePoint(nearSymmetricS, uMax);
  const mid = curvePoint(nearSymmetricS, (uMin + uMax) / 2);
  const chordMid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  const midDev = Math.hypot(mid[0] - chordMid[0], mid[1] - chordMid[1], mid[2] - chordMid[2]);
  assert.ok(midDev > 0, 'fixture sanity: this curve must NOT be exactly symmetric — its midpoint deviation is small but genuinely nonzero');
  assert.ok(midDev < 1e-4, `fixture sanity: the midpoint deviation must still fall below the auto-tolerance band, got ${midDev}`);
  const trueLen = wholeCurveDenseLength(nearSymmetricS);
  const table = buildArcLengthTable(nearSymmetricS, uMin, uMax);
  assert.ok(Math.abs(table.total - trueLen) / trueLen < 1e-4, `near-symmetric total ${table.total} vs true ${trueLen}`);
});

test('a symmetric SPAN inside an otherwise ordinary multi-span curve is measured correctly (the defect is per-span, so it does not need a wholly symmetric curve to be reachable)', () => {
  const crv = symmetricSpanInsideLongerCurve;
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  assert.ok(crv.knots.length > 8, 'fixture sanity: this curve must have genuine interior knots, i.e. several spans');
  const trueLen = wholeCurveDenseLength(crv);
  const table = buildArcLengthTable(crv, uMin, uMax);
  assert.ok(Math.abs(table.total - trueLen) / trueLen < 1e-4, `multi-span total ${table.total} vs true ${trueLen}`);
});

test('a NON-symmetric curve stays exact and does not get materially more expensive (the minimum depth is free wherever a span genuinely curves — it subdivides past that depth on its own merits)', () => {
  const uMin = asymmetricS.knots[0], uMax = asymmetricS.knots[asymmetricS.knots.length - 1];
  const trueLen = wholeCurveDenseLength(asymmetricS);
  const table = buildArcLengthTable(asymmetricS, uMin, uMax);
  assert.ok(Math.abs(table.total - trueLen) / trueLen < 1e-5, `non-symmetric total ${table.total} vs true ${trueLen}`);
  // A minimum depth of n can at worst quadruple a genuinely FLAT span. This
  // curve is not flat anywhere, so the count must stay in the same league as
  // the tolerance alone demands, not multiply.
  const seedSpans = new Set([uMin, uMax]);
  for (const k of asymmetricS.knots) if (k > uMin && k < uMax) seedSpans.add(k);
  assert.ok(table.samples.length < 4000, `sample count must stay bounded, got ${table.samples.length}`);
});

test('a DEGREE-1 curve is exempt from the minimum depth and keeps its exact previous sample count — its span basis functions are non-negative and sum to one, so the chord IS the curve and subdividing can reveal nothing', () => {
  const polyPts = [];
  for (let i = 0; i <= 20; i++) polyPts.push([i * 7, (i % 2) * 3, 0]);
  const poly = globalCurveInterp(polyPts, 1);
  assert.equal(poly.degree, 1);
  const uMin = poly.knots[0], uMax = poly.knots[poly.knots.length - 1];
  const table = buildArcLengthTable(poly, uMin, uMax);
  // One sample per vertex, exactly — no forced subdivision anywhere.
  assert.equal(table.samples.length, polyPts.length, `a degree-1 polyline must sample exactly its own vertices, got ${table.samples.length}`);
  let exact = 0;
  for (let i = 1; i < polyPts.length; i++) {
    exact += Math.hypot(polyPts[i][0] - polyPts[i - 1][0], polyPts[i][1] - polyPts[i - 1][1], polyPts[i][2] - polyPts[i - 1][2]);
  }
  assert.ok(Math.abs(table.total - exact) < 1e-9, `and its length is exact, got ${table.total} vs ${exact}`);
});
