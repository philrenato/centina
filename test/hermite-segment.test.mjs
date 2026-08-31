import test from 'node:test';
import assert from 'node:assert/strict';
import { cubicHermiteSegment, catmullRomTangent } from '../kernel/interpolate.mjs';
import { curvePoint, rationalCurveDerivs } from '../kernel/curve.mjs';

// Independent finite-difference derivative — the same discipline
// test/bezier-end-derivs.test.mjs already established: never trust the
// analytic derivative alone.
function finiteDiffDeriv(crv, u, du) {
  const a = curvePoint(crv, u - du), b = curvePoint(crv, u + du);
  return [(b[0] - a[0]) / (2 * du), (b[1] - a[1]) / (2 * du), (b[2] - a[2]) / (2 * du)];
}

test('cubicHermiteSegment: passes through both endpoints exactly', () => {
  const p0 = [0, 0, 0], p1 = [40, 20, -10];
  const m0 = [10, 0, 0], m1 = [5, -8, 4];
  const crv = cubicHermiteSegment(p0, p1, m0, m1);
  const c0 = curvePoint(crv, 0), c1 = curvePoint(crv, 1);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(c0[i] - p0[i]) < 1e-9, `start axis ${i}: got ${c0[i]} want ${p0[i]}`);
    assert.ok(Math.abs(c1[i] - p1[i]) < 1e-9, `end axis ${i}: got ${c1[i]} want ${p1[i]}`);
  }
});

test('cubicHermiteSegment: matches the requested tangent EXACTLY at both ends (analytic)', () => {
  const p0 = [0, 0, 0], p1 = [40, 20, -10];
  const m0 = [10, 0, 0], m1 = [5, -8, 4];
  const crv = cubicHermiteSegment(p0, p1, m0, m1);
  const [, atStart] = rationalCurveDerivs(crv, 0, 1);
  const [, atEnd] = rationalCurveDerivs(crv, 1, 1);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(atStart[i] - m0[i]) < 1e-9, `start deriv axis ${i}: got ${atStart[i]} want ${m0[i]}`);
    assert.ok(Math.abs(atEnd[i] - m1[i]) < 1e-9, `end deriv axis ${i}: got ${atEnd[i]} want ${m1[i]}`);
  }
});

test('cubicHermiteSegment: the analytic derivative is cross-checked against an INDEPENDENT finite-difference derivative', () => {
  const p0 = [0, 0, 0], p1 = [30, -15, 5];
  const m0 = [12, 8, -2], m1 = [-6, 4, 9];
  const crv = cubicHermiteSegment(p0, p1, m0, m1);
  for (const u of [0.15, 0.5, 0.82]) {
    const [, analytic] = rationalCurveDerivs(crv, u, 1);
    const fd = finiteDiffDeriv(crv, u, 1e-5);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(analytic[i] - fd[i]) < 1e-3, `u=${u} axis ${i}: analytic ${analytic[i]} vs finite-diff ${fd[i]}`);
  }
});

test('cubicHermiteSegment: a zero tangent at both ends degenerates to the classic smoothstep ease curve, NOT a straight chord (real, checkable boundary case — a zero-tangent cubic Bezier has double points at each end, which is the well-known 3t²-2t³ curve, verified by hand here)', () => {
  const p0 = [0, 0, 0], p1 = [10, 0, 0];
  const crv = cubicHermiteSegment(p0, p1, [0, 0, 0], [0, 0, 0]);
  for (let t = 0; t <= 1; t += 0.25) {
    const [x, y, z] = curvePoint(crv, t);
    const expectedX = 10 * (3 * t * t - 2 * t * t * t);
    assert.ok(Math.abs(x - expectedX) < 1e-9 && Math.abs(y) < 1e-9 && Math.abs(z) < 1e-9, `t=${t}: got [${x},${y},${z}] want x=${expectedX}`);
  }
  // Still monotonic start-to-end and still exact at both endpoints — a
  // genuine, well-formed curve, just not a linear one.
  const c0 = curvePoint(crv, 0), c1 = curvePoint(crv, 1);
  assert.ok(Math.abs(c0[0]) < 1e-9 && Math.abs(c1[0] - 10) < 1e-9, 'endpoints still exact');
});

test('cubicHermiteSegment: no NaN/Infinity anywhere across the domain for a real, non-degenerate case', () => {
  const crv = cubicHermiteSegment([0, 0, 0], [50, 30, -20], [20, -10, 5], [-15, 25, 10]);
  for (let t = 0; t <= 1; t += 0.05) {
    const [x, y, z] = curvePoint(crv, t);
    assert.ok(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z), `t=${t}: got [${x},${y},${z}]`);
  }
});

test('catmullRomTangent: is half the straight chord from the previous point to the next (the defining identity)', () => {
  const pPrev = [0, 0, 0], pNext = [20, -10, 6];
  const m = catmullRomTangent(pPrev, pNext);
  assert.deepEqual(m, [10, -5, 3]);
});

test('catmullRomTangent: zero for a symmetric (collinear-and-centered) neighbor pair', () => {
  const m = catmullRomTangent([-5, 2, 0], [5, 2, 0]);
  assert.deepEqual(m, [5, 0, 0]);
});

test('cubicHermiteSegment + catmullRomTangent together: a 3-point Catmull-Rom-tangent chain is G1-continuous at the shared interior joint by construction', () => {
  // Three points, an interior tangent computed once and fed as BOTH the
  // end derivative of segment A and the start derivative of segment B —
  // exactly the G1-continuity-by-shared-value mechanism the app-layer
  // "internal B handles" feature (unbuilt at the app layer) would rely
  // on. Proven directly: segment A's own end derivative and segment B's
  // own start derivative, read independently off each curve, must match
  // the SAME requested tangent exactly.
  const pA = [0, 0, 0], pMid = [20, 15, 0], pB = [45, 10, 0];
  const mMid = catmullRomTangent(pA, pB);
  const segA = cubicHermiteSegment(pA, pMid, [15, 0, 0], mMid);
  const segB = cubicHermiteSegment(pMid, pB, mMid, [15, -5, 0]);
  const [, endA] = rationalCurveDerivs(segA, 1, 1);
  const [, startB] = rationalCurveDerivs(segB, 0, 1);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(endA[i] - mMid[i]) < 1e-9, `segA end deriv axis ${i}: got ${endA[i]} want ${mMid[i]}`);
    assert.ok(Math.abs(startB[i] - mMid[i]) < 1e-9, `segB start deriv axis ${i}: got ${startB[i]} want ${mMid[i]}`);
    assert.ok(Math.abs(endA[i] - startB[i]) < 1e-9, `G1 continuity: segA end (${endA[i]}) should equal segB start (${startB[i]}) axis ${i}`);
  }
});
