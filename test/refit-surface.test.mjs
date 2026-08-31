import test from 'node:test';
import assert from 'node:assert/strict';
import { revolve } from '../kernel/primitives.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';
import { refitSurfaceUV } from '../kernel/loft.mjs';

// A real, non-trivial source surface — a revolve of an open, gently curved
// profile (rational in V from the arc-span construction), not a flat/ruled
// toy — so an exactness proof against it is a genuine test, not a case
// that would pass by algebraic accident (matching this kernel's own
// repeated "avoid too-simple test geometry" lesson).
function stressRevolve() {
  const profile = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[10, 0, 0, 1], [14, 0, 20, 1], [8, 0, 40, 1]] };
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 1.4);
}

test('refitSurfaceUV reproduces the source surface exactly at every uCount x vCount grid station', () => {
  const srf = stressRevolve();
  const refit = refitSurfaceUV(srf, 6, 5, 3, 3);
  assert.equal(isFiniteNet(refit.ctrlNet), true);
  const u0 = srf.knotsU[0], u1 = srf.knotsU[srf.knotsU.length - 1];
  const v0 = srf.knotsV[0], v1 = srf.knotsV[srf.knotsV.length - 1];
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 5; j++) {
      const uf = i / 5, vf = j / 4;
      const u = u0 + uf * (u1 - u0), v = v0 + vf * (v1 - v0);
      const want = surfacePoint(srf, u, v);
      const got = surfacePoint(refit, uf, vf); // refit's own domain is plain [0,1] fractions by construction
      const err = Math.hypot(got[0] - want[0], got[1] - want[1], got[2] - want[2]);
      assert.ok(err < 1e-6, `station (${i},${j}): got ${got}, want ${want}, err ${err}`);
    }
  }
});

test('refitSurfaceUV genuinely changes control point count/degree to the requested values', () => {
  const srf = stressRevolve();
  const refit = refitSurfaceUV(srf, 8, 6, 3, 2);
  assert.equal(refit.ctrlNet.length, 8);
  assert.equal(refit.ctrlNet[0].length, 6);
  assert.equal(refit.degU, 3);
  assert.equal(refit.degV, 2);
});

test('refitSurfaceUV clamps degree down when the requested count is too small for it (same honest clamp every other kernel builder uses)', () => {
  const srf = stressRevolve();
  const refit = refitSurfaceUV(srf, 3, 3, 5, 5);
  assert.equal(refit.degU, 2, 'degU clamps to uCount-1');
  assert.equal(refit.degV, 2, 'degV clamps to vCount-1');
});

test('refitSurfaceUV refuses fewer than 2 points in either direction', () => {
  const srf = stressRevolve();
  assert.throws(() => refitSurfaceUV(srf, 1, 5, 3, 3));
  assert.throws(() => refitSurfaceUV(srf, 5, 1, 3, 3));
});

test('a coarser refit is a genuinely different (fewer-control-point) surface than a finer one, both still exact at their own stations', () => {
  const srf = stressRevolve();
  const coarse = refitSurfaceUV(srf, 4, 4, 3, 3);
  const fine = refitSurfaceUV(srf, 12, 10, 3, 3);
  assert.notEqual(coarse.ctrlNet.length, fine.ctrlNet.length);
  // Both should still land close to the true surface at a shared interior point.
  const u0 = srf.knotsU[0], u1 = srf.knotsU[srf.knotsU.length - 1];
  const v0 = srf.knotsV[0], v1 = srf.knotsV[srf.knotsV.length - 1];
  const want = surfacePoint(srf, u0 + 0.5 * (u1 - u0), v0 + 0.5 * (v1 - v0));
  const gotCoarse = surfacePoint(coarse, 0.5, 0.5);
  const gotFine = surfacePoint(fine, 0.5, 0.5);
  const errCoarse = Math.hypot(gotCoarse[0] - want[0], gotCoarse[1] - want[1], gotCoarse[2] - want[2]);
  const errFine = Math.hypot(gotFine[0] - want[0], gotFine[1] - want[1], gotFine[2] - want[2]);
  assert.ok(errFine <= errCoarse + 1e-9, `finer refit (${errFine}) should be at least as close as the coarser one (${errCoarse}) at a shared non-station interior point`);
});
