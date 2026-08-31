import test from 'node:test';
import assert from 'node:assert/strict';
import { makeEllipsoidProfile, revolve } from '../kernel/primitives.mjs';
import { curvePoint } from '../kernel/curve.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { sub, cross, normalize, length } from '../kernel/vec3.mjs';

// Distance of a point from the revolve axis line through `center`.
function offAxis(pt, center, axis) {
  const rel = sub(pt, center);
  const along = rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2];
  const proj = [axis[0] * along, axis[1] * along, axis[2] * along];
  return length(sub(rel, proj));
}

test('makeEllipsoidProfile: a true half-ellipse arc, both endpoints exactly on the axis, apex at the equatorial radius', () => {
  const center = [3, -5, 2];
  const eAxis = [1, 0, 0], pAxis = [0, 0, 1];
  const eR = 40, pR = 15;
  const prof = makeEllipsoidProfile(center, eAxis, pAxis, eR, pR, 2);
  const axis = normalize(pAxis);
  const eq = normalize(eAxis);
  const cps = prof.ctrlPts;
  // both control-net endpoints sit on the axis (the two poles) to machine precision
  const p0 = curvePoint(prof, prof.knots[0]);
  const pN = curvePoint(prof, prof.knots[prof.knots.length - 1]);
  assert.ok(offAxis(p0, center, axis) < 1e-9, `start endpoint off-axis ${offAxis(p0, center, axis)}`);
  assert.ok(offAxis(pN, center, axis) < 1e-9, `end endpoint off-axis ${offAxis(pN, center, axis)}`);
  // the two poles are at ±pR along the axis
  const along0 = sub(p0, center)[2], alongN = sub(pN, center)[2];
  assert.ok(Math.abs(Math.abs(along0) - pR) < 1e-9);
  assert.ok(Math.abs(Math.abs(alongN) - pR) < 1e-9);
  // every sampled profile point satisfies the 2D half-ellipse equation
  // (equatorial/eR)^2 + (polar/pR)^2 == 1
  let worst = 0;
  for (let i = 0; i <= 200; i++) {
    const u = prof.knots[0] + (prof.knots[prof.knots.length - 1] - prof.knots[0]) * (i / 200);
    const P = curvePoint(prof, u);
    const rel = sub(P, center);
    const e = rel[0] * eq[0] + rel[1] * eq[1] + rel[2] * eq[2];
    const pol = rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2];
    worst = Math.max(worst, Math.abs((e / eR) ** 2 + (pol / pR) ** 2 - 1));
  }
  assert.ok(worst < 1e-9, `worst half-ellipse residual ${worst}`);
  // the equator (apex, angle 0) reaches exactly the equatorial radius, on the equatorial plane
  const apex = curvePoint(prof, (prof.knots[0] + prof.knots[prof.knots.length - 1]) / 2);
  assert.ok(Math.abs(offAxis(apex, center, axis) - eR) < 1e-9, `apex radius ${offAxis(apex, center, axis)}`);
  assert.ok(Math.abs(sub(apex, center)[2]) < 1e-9, 'apex on the equatorial plane');
});

// The REAL proof: revolve the profile and check the true ellipsoid equation at
// many (u,v), including at both poles (u/v = domain min/max where the pole rows are).
function checkEllipsoidExact(center, eAxis, pAxis, eR, pR, minSeg) {
  const prof = makeEllipsoidProfile(center, eAxis, pAxis, eR, pR, minSeg);
  const axis = normalize(pAxis);
  const srf = revolve(prof, center, axis, 0, 2 * Math.PI);
  const eq1 = normalize(eAxis);
  const eq2 = cross(axis, eq1); // second equatorial direction
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  let worst = 0, worstPole = 0;
  const NU = 48, NV = 48;
  for (let i = 0; i <= NU; i++) {
    for (let j = 0; j <= NV; j++) {
      const u = uMin + (uMax - uMin) * (i / NU);
      const v = vMin + (vMax - vMin) * (j / NV);
      const P = surfacePoint(srf, u, v);
      const rel = sub(P, center);
      const cx = rel[0] * eq1[0] + rel[1] * eq1[1] + rel[2] * eq1[2];
      const cy = rel[0] * eq2[0] + rel[1] * eq2[1] + rel[2] * eq2[2];
      const cz = rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2];
      const res = Math.abs((cx / eR) ** 2 + (cy / eR) ** 2 + (cz / pR) ** 2 - 1);
      worst = Math.max(worst, res);
      // pole rows: u at the two profile-domain ends (i === 0 or i === NU)
      if (i === 0 || i === NU) worstPole = Math.max(worstPole, res);
    }
  }
  return { worst, worstPole };
}

test('revolved ellipsoid satisfies the true equation exactly at every (u,v), poles included', () => {
  for (const [name, c, ea, pa, eR, pR, seg] of [
    ['sphere r=1', [0, 0, 0], [1, 0, 0], [0, 0, 1], 1, 1, 2],
    ['oblate', [3, -5, 2], [1, 0, 0], [0, 0, 1], 40, 15, 2],
    ['prolate', [0, 0, 0], [1, 0, 0], [0, 0, 1], 10, 60, 3],
    ['oblique frame', [1, 2, 3], [0.6, 0.8, 0], [0, 0, 1], 25, 50, 4],
    ['fully oblique axis', [0, 0, 0], [0.6, 0, 0.8], [-0.8, 0, 0.6], 30, 12, 3],
  ]) {
    const { worst, worstPole } = checkEllipsoidExact(c, ea, pa, eR, pR, seg);
    assert.ok(worst < 1e-8, `${name}: worst ellipsoid residual ${worst}`);
    assert.ok(worstPole < 1e-8, `${name}: worst POLE residual ${worstPole}`);
  }
});

test('makeEllipsoidProfile: sphere case (eR === pR) is a genuine circular arc; no NaN/Inf', () => {
  for (const seg of [2, 3, 4]) {
    const prof = makeEllipsoidProfile([0, 0, 0], [1, 0, 0], [0, 0, 1], 20, 20, seg);
    for (const cp of prof.ctrlPts) for (const v of cp) assert.ok(Number.isFinite(v));
    // every sampled point is exactly radius 20 from center (a circular meridian)
    for (let i = 0; i <= 100; i++) {
      const u = prof.knots[0] + (prof.knots[prof.knots.length - 1] - prof.knots[0]) * (i / 100);
      const P = curvePoint(prof, u);
      assert.ok(Math.abs(length(P) - 20) < 1e-9, `sphere meridian radius ${length(P)}`);
    }
  }
});
