import test from 'node:test';
import assert from 'node:assert/strict';
import { globalCurveInterp, globalCurveInterpWithEndDerivs } from '../kernel/interpolate.mjs';
import { curvePoint, rationalCurveDerivs } from '../kernel/curve.mjs';

// Independent finite-difference derivative — checked against the analytic
// rationalCurveDerivs result below, so an exactness claim never rests on
// only one method computing it.
function finiteDiffDeriv(crv, u, du) {
  const a = curvePoint(crv, u - du), b = curvePoint(crv, u + du);
  return [(b[0] - a[0]) / (2 * du), (b[1] - a[1]) / (2 * du), (b[2] - a[2]) / (2 * du)];
}

const PTS = [
  [0, 0, 0], [10, 4, 0], [25, -6, 5], [40, 0, 0], [55, 10, -5], [70, 0, 0],
];

test('globalCurveInterpWithEndDerivs: passes through every input point exactly, at its own chord-length parameter', () => {
  const D0 = [8, 6, 0], Dn = [10, -6, 4];
  const crv = globalCurveInterpWithEndDerivs(PTS, D0, Dn);
  assert.equal(crv.degree, 3);
  assert.equal(crv.ctrlPts.length, PTS.length + 2); // n+3 control points, 2 more than ordinary n+1
  for (let k = 0; k < PTS.length; k++) {
    const c = curvePoint(crv, crv.paramsUsed[k]);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(c[i] - PTS[k][i]) < 1e-7, `point ${k} axis ${i}: got ${c[i]} want ${PTS[k][i]}`);
  }
});

test('globalCurveInterpWithEndDerivs: matches the REQUESTED derivative exactly at both ends (analytic rationalCurveDerivs)', () => {
  const D0 = [8, 6, 0], Dn = [10, -6, 4];
  const crv = globalCurveInterpWithEndDerivs(PTS, D0, Dn);
  const [, atStart] = rationalCurveDerivs(crv, crv.paramsUsed[0], 1);
  const [, atEnd] = rationalCurveDerivs(crv, crv.paramsUsed[PTS.length - 1], 1);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(atStart[i] - D0[i]) < 1e-6, `start deriv axis ${i}: got ${atStart[i]} want ${D0[i]}`);
    assert.ok(Math.abs(atEnd[i] - Dn[i]) < 1e-6, `end deriv axis ${i}: got ${atEnd[i]} want ${Dn[i]}`);
  }
});

test('globalCurveInterpWithEndDerivs: the analytic derivative is cross-checked against an INDEPENDENT finite-difference derivative (never trust one method alone)', () => {
  const D0 = [8, 6, 0], Dn = [10, -6, 4];
  const crv = globalCurveInterpWithEndDerivs(PTS, D0, Dn);
  const u0 = crv.paramsUsed[0], un = crv.paramsUsed[PTS.length - 1];
  const du = (un - u0) * 1e-5;
  const fdStart = finiteDiffDeriv(crv, u0 + du, du); // offset off the exact clamped boundary so both samples stay inside the domain
  const fdEnd = finiteDiffDeriv(crv, un - du, du);
  // A generous tolerance here on purpose — this is a SANITY cross-check
  // (does an independent numerical method roughly agree?), not the
  // exactness proof itself: the offset off the true boundary (needed so
  // both finite-difference samples stay inside the domain) introduces its
  // own small, expected discrepancy proportional to local curvature. The
  // analytic rationalCurveDerivs comparison above is the real exactness
  // proof, already held to 1e-6.
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(fdStart[i] - D0[i]) < 0.5, `finite-diff start axis ${i}: got ${fdStart[i]} want ~${D0[i]}`);
    assert.ok(Math.abs(fdEnd[i] - Dn[i]) < 0.5, `finite-diff end axis ${i}: got ${fdEnd[i]} want ~${Dn[i]}`);
  }
});

test('globalCurveInterpWithEndDerivs: editing ONE end\'s derivative leaves every interpolated point (both ends included) untouched — only the SHAPE between them can move', () => {
  const D0a = [8, 6, 0], Dn = [10, -6, 4];
  const D0b = [40, -30, 15]; // wildly different start tangent
  const crvA = globalCurveInterpWithEndDerivs(PTS, D0a, Dn);
  const crvB = globalCurveInterpWithEndDerivs(PTS, D0b, Dn);
  for (let k = 0; k < PTS.length; k++) {
    const cA = curvePoint(crvA, crvA.paramsUsed[k]);
    const cB = curvePoint(crvB, crvB.paramsUsed[k]);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(cA[i] - PTS[k][i]) < 1e-6);
      assert.ok(Math.abs(cB[i] - PTS[k][i]) < 1e-6, `changing only the start derivative must not break interpolation at point ${k}`);
    }
  }
  // And the shape genuinely DID change near the start — a real, provable
  // effect, not a no-op: sample just past u=paramsUsed[0] and confirm the
  // two curves diverge there.
  const uNear = crvA.paramsUsed[0] + (crvA.paramsUsed[1] - crvA.paramsUsed[0]) * 0.4;
  const pA = curvePoint(crvA, uNear), pB = curvePoint(crvB, uNear);
  const dist = Math.hypot(pA[0] - pB[0], pA[1] - pB[1], pA[2] - pB[2]);
  assert.ok(dist > 0.1, `a drastically different start tangent should visibly move the curve near the start (dist=${dist})`);
});

test('globalCurveInterpWithEndDerivs: feeding the VANILLA curve\'s own true end derivatives back in still interpolates every point exactly and stays finite everywhere (a different, higher-DOF representation of related but not necessarily identical shape — this proves it is still a well-posed, non-degenerate system, not that the two curves coincide)', () => {
  const vanilla = globalCurveInterp(PTS, 3);
  const [, d0] = rationalCurveDerivs(vanilla, vanilla.paramsUsed[0], 1);
  const [, dn] = rationalCurveDerivs(vanilla, vanilla.paramsUsed[PTS.length - 1], 1);
  const crv = globalCurveInterpWithEndDerivs(PTS, d0, dn);
  for (let k = 0; k < PTS.length; k++) {
    const c = curvePoint(crv, crv.paramsUsed[k]);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(c[i] - PTS[k][i]) < 1e-6);
  }
  for (let t = 0; t <= 1; t += 0.02) {
    const u = crv.paramsUsed[0] + (crv.paramsUsed[PTS.length - 1] - crv.paramsUsed[0]) * t;
    const [x, y, z] = curvePoint(crv, u);
    assert.ok(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z), `u=${u} produced non-finite point`);
  }
});

test('globalCurveInterpWithEndDerivs: refuses honestly with fewer than 4 points (cubic + 2 end derivatives needs n>=3)', () => {
  assert.throws(() => globalCurveInterpWithEndDerivs(PTS.slice(0, 3), [1, 0, 0], [1, 0, 0]));
});
