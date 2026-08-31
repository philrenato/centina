import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle } from '../kernel/primitives.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';
import { loft } from '../kernel/loft.mjs';

// A clamped tensor-product global-interpolation surface always touches its
// own 4 corner control points exactly at the domain's own min/max U and V
// (standard B-spline endpoint-interpolation property, compounded across
// both directions) — and each CORNER control point is, in turn, exactly
// the corresponding section's own first/last sampled point (global
// interpolation's own first/last-control-point-equals-first/last-
// through-point property). So for a 2-section loft, all 4 corners are
// exactly predictable with zero reconstruction error, regardless of the
// dense-sampling/degree-3-in-V approximation everywhere else on the
// surface — the rigorous check this test leans on.
test('loft of two straight-line profiles reproduces an exact ruled (flat) patch at all 4 corners, Y identically 0 everywhere', () => {
  const profile0 = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1]] };
  const profile1 = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 10, 1], [10, 0, 10, 1]] };
  const srf = loft([profile0, profile1]);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  assert.equal(srf.degV, 1, 'only 2 sections — degV clamps down from the requested 3, same "not enough points" clamp globalCurveInterp already does');
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const corners = {
    '(uMin,vMin)': [surfacePoint(srf, uMin, vMin), [0, 0, 0]],
    '(uMax,vMin)': [surfacePoint(srf, uMax, vMin), [10, 0, 0]],
    '(uMin,vMax)': [surfacePoint(srf, uMin, vMax), [0, 0, 10]],
    '(uMax,vMax)': [surfacePoint(srf, uMax, vMax), [10, 0, 10]],
  };
  for (const [name, [p, expected]] of Object.entries(corners)) {
    assert.ok(Math.hypot(p[0] - expected[0], p[1] - expected[1], p[2] - expected[2]) < 1e-9, `${name}: got ${p}, expected ${expected}`);
  }
  // Both input profiles have Y=0 identically — every control point solved
  // from purely-Y=0 input data must itself be exactly Y=0 (the linear
  // system's Y right-hand-side is the zero vector), so the WHOLE surface
  // stays in the Y=0 plane exactly, not just at the 4 corners.
  for (let u = 0; u <= 1; u += 0.13) {
    for (let v = 0; v <= 1; v += 0.27) {
      const p = surfacePoint(srf, uMin + u * (uMax - uMin), vMin + v * (vMax - vMin));
      assert.ok(Math.abs(p[1]) < 1e-9, `u=${u} v=${v} y=${p[1]}`);
    }
  }
});

test('loft between two circles of different radii stays close to a true circle at both ends (a real, honest reconstruction-from-samples approximation, not exact)', () => {
  const c0 = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const c1 = makeCircle([0, 0, 20], [1, 0, 0], [0, 1, 0], 12);
  const srf = loft([c0, c1], 32);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  // At v=vMin (section 0, radius 5, z=0) and v=vMax (section 1, radius 12,
  // z=20) — dense-sampled reconstruction, so a small tolerance (not exact
  // like the straight-line case above, which has zero reconstruction
  // error by construction) but tight enough to prove this is genuinely
  // modeling the two circles, not something wildly off.
  for (let t = 0; t <= 1; t += 0.05) {
    const u = uMin + t * (uMax - uMin);
    const p0 = surfacePoint(srf, u, vMin);
    const r0 = Math.hypot(p0[0], p0[1]);
    assert.ok(Math.abs(r0 - 5) < 0.05 && Math.abs(p0[2]) < 0.05, `section0 u=${u}: r=${r0} z=${p0[2]}`);
    const p1 = surfacePoint(srf, u, vMax);
    const r1 = Math.hypot(p1[0], p1[1]);
    assert.ok(Math.abs(r1 - 12) < 0.05 && Math.abs(p1[2] - 20) < 0.05, `section1 u=${u}: r=${r1} z=${p1[2]}`);
  }
});

test('loft across 4 sections (more than 2) still produces a finite, well-formed surface with the requested degrees', () => {
  const sections = [0, 5, 10, 15].map((z) => makeCircle([0, 0, z], [1, 0, 0], [0, 1, 0], 5 + z * 0.3));
  const srf = loft(sections, 20);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  assert.equal(srf.degV, 3, '4 sections is plenty for the requested degree-3 V direction, no clamping needed');
  assert.equal(srf.ctrlNet.length, 20);
  assert.equal(srf.ctrlNet[0].length, 4, 'one V control point per section');
});

test('loft throws honestly on fewer than 2 sections rather than producing a degenerate surface', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1]] };
  assert.throws(() => loft([profile]));
});
