import test from 'node:test';
import assert from 'node:assert/strict';
import { globalCurveInterp, chordLengthParams, closedCurveInterp } from '../kernel/interpolate.mjs';
import { curvePoint } from '../kernel/curve.mjs';

function unitTangent(crv, u, du) {
  const a = curvePoint(crv, u - du), b = curvePoint(crv, u + du);
  const v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const len = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / len, v[1] / len, v[2] / len];
}

test('globalCurveInterp: 2 points clamps to a degree-1 line through both', () => {
  const pts = [[0, 0, 0], [10, 20, 30]];
  const crv = globalCurveInterp(pts, 3);
  assert.equal(crv.degree, 1);
  const p0 = curvePoint(crv, 0), p1 = curvePoint(crv, 1);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(p0[i] - pts[0][i]) < 1e-9);
    assert.ok(Math.abs(p1[i] - pts[1][i]) < 1e-9);
  }
});

test('globalCurveInterp: curve passes through EVERY input point exactly, at its own chord-length parameter', () => {
  const pts = [[0, 0, 0], [10, 4, 0], [25, -6, 5], [40, 0, 0], [55, 10, -5], [70, 0, 0]];
  const crv = globalCurveInterp(pts, 3);
  assert.equal(crv.degree, 3);
  for (let k = 0; k < pts.length; k++) {
    const c = curvePoint(crv, crv.paramsUsed[k]);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(c[i] - pts[k][i]) < 1e-7, `point ${k} axis ${i}: got ${c[i]} want ${pts[k][i]}`);
  }
});

test('globalCurveInterp: collinear points interpolate to a straight line (no phantom bulge)', () => {
  const pts = [[0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40]];
  const crv = globalCurveInterp(pts, 3);
  for (let u = 0; u <= 1; u += 0.05) {
    const [x, y, z] = curvePoint(crv, u);
    assert.ok(Math.abs(x - y) < 1e-6 && Math.abs(y - z) < 1e-6, `u=${u} point (${x},${y},${z}) should be on x=y=z`);
  }
});

test('globalCurveInterp: no NaN/Infinity anywhere across the sampled curve, a realistic 8-point zigzag', () => {
  const pts = [
    [0, 0, 0], [12, 30, -4], [30, 5, 10], [48, -20, 0],
    [65, 15, -8], [80, 40, 3], [95, 0, 12], [110, -10, 0],
  ];
  const crv = globalCurveInterp(pts, 3);
  for (let u = 0; u <= 1; u += 0.02) {
    const [x, y, z] = curvePoint(crv, u);
    assert.ok(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z), `u=${u} produced non-finite point`);
  }
});

test('chordLengthParams: degenerate all-coincident points falls back to uniform spacing, no divide-by-zero', () => {
  const pts = [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]];
  const ubar = chordLengthParams(pts);
  assert.deepEqual(ubar, [0, 1 / 3, 2 / 3, 1]);
});

test('closedCurveInterp: still passes through every original point exactly', () => {
  const pts = [[40, 0, 0], [20, 35, 0], [-20, 35, 0], [-40, 0, 0], [-20, -35, 0], [20, -35, 0]];
  const { crv, uStart, uEnd } = closedCurveInterp(pts, 3);
  // The kept sub-range's OWN parameter list, in order, corresponds 1:1 to
  // the original points (paramsUsed[k..k+n] on the padded curve — k is the
  // degree here, matching closedCurveInterp's own padding amount).
  const k = 3;
  for (let i = 0; i < pts.length; i++) {
    const u = crv.paramsUsed[k + i];
    assert.ok(u >= uStart - 1e-9 && u <= uEnd + 1e-9, `param for point ${i} (${u}) should fall inside [uStart,uEnd]`);
    const c = curvePoint(crv, u);
    for (let ax = 0; ax < 3; ax++) assert.ok(Math.abs(c[ax] - pts[i][ax]) < 1e-6, `point ${i} axis ${ax}: got ${c[ax]} want ${pts[i][ax]}`);
  }
});

test('closedCurveInterp: TANGENT continuity at the seam — approaching the closing point and leaving it point the same direction (unlike plain positional closure, which kinks)', () => {
  const pts = [[40, 0, 0], [20, 35, 0], [-20, 35, 0], [-40, 0, 0], [-20, -35, 0], [20, -35, 0]];
  const { crv, uStart, uEnd } = closedCurveInterp(pts, 3);
  const du = (uEnd - uStart) * 1e-4;
  const tangentLeavingStart = unitTangent(crv, uStart, du);
  const tangentApproachingEnd = unitTangent(crv, uEnd, du);
  const dot = tangentLeavingStart[0] * tangentApproachingEnd[0] + tangentLeavingStart[1] * tangentApproachingEnd[1] + tangentLeavingStart[2] * tangentApproachingEnd[2];
  assert.ok(dot > 0.999, `tangent directions should match closely across the seam (dot=${dot}, want close to 1)`);
});

test('closedCurveInterp: a REGULAR polygon-ish loop closes to something visibly rounder than its straight-chord positional counterpart (sanity: not degenerate)', () => {
  const pts = [[40, 0, 0], [20, 35, 0], [-20, 35, 0], [-40, 0, 0], [-20, -35, 0], [20, -35, 0]];
  const { crv, uStart, uEnd } = closedCurveInterp(pts, 3);
  let allFinite = true;
  for (let t = 0; t <= 1; t += 0.02) {
    const u = uStart + (uEnd - uStart) * t;
    const [x, y, z] = curvePoint(crv, u);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) allFinite = false;
  }
  assert.ok(allFinite, 'closed curve should be finite everywhere across its kept sub-range');
});

// solveLinearSystem's zero-pivot guard (Project's own torus/non-planar-
// extrusion bug): a run of duplicate consecutive sample points —
// the exact failure mode a clamped/stalled Gauss-Newton projection search
// produces — gives chordLengthParams two identical parameters, so
// interpAtParams builds two byte-identical rows in its coefficient matrix.
// Partial pivoting alone can't fix a matrix that's genuinely singular (every
// candidate row in that column is the same) — this must THROW a clear,
// catchable error instead of silently dividing by ~0 into NaN control
// points.
test('globalCurveInterp throws a clear error (not silent NaN) on duplicate consecutive through-points', () => {
  const pts = [[0, 0, 0], [10, 0, 0], [10, 0, 0], [10, 0, 0], [20, 5, 0]];
  assert.throws(() => globalCurveInterp(pts, 3), /singular|coincide/, 'expected a clear singular-matrix error, not a silent NaN result');
});

test('globalCurveInterp still succeeds normally when points are merely CLOSE but genuinely distinct', () => {
  const pts = [[0, 0, 0], [10, 0, 0], [10.0001, 0.0001, 0], [20, 5, 0], [30, 10, 0]];
  const crv = globalCurveInterp(pts, 3);
  assert.ok(crv.ctrlPts.flat().every(Number.isFinite), 'a genuinely (if narrowly) distinct point set must still fit cleanly');
});
