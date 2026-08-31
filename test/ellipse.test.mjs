import test from 'node:test';
import assert from 'node:assert/strict';
import { makeEllipse, makeCircle } from '../kernel/primitives.mjs';
import { curvePoint } from '../kernel/curve.mjs';

// Local-frame decomposition: an ellipse point P must satisfy
//   (u/radiusX)^2 + (v/radiusY)^2 === 1
// where u = (P - center)·xHatUnit, v = (P - center)·yHatUnit (the in-plane
// components against the UNIT axes). This is the actual ellipse equation,
// tested directly against real sampled curve points — not the control net
// (whose rational middle points sit OFF the true ellipse by construction).
function ellipseResidualAt(crv, center, xHat, yHat, radiusX, radiusY, u) {
  const p = curvePoint(crv, u);
  const rel = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
  const uu = rel[0] * xHat[0] + rel[1] * xHat[1] + rel[2] * xHat[2];
  const vv = rel[0] * yHat[0] + rel[1] * yHat[1] + rel[2] * yHat[2];
  // also confirm the point genuinely lies IN the ellipse's own plane (no
  // out-of-plane drift): the component along the plane normal must be ~0.
  const nx = xHat[1] * yHat[2] - xHat[2] * yHat[1];
  const ny = xHat[2] * yHat[0] - xHat[0] * yHat[2];
  const nz = xHat[0] * yHat[1] - xHat[1] * yHat[0];
  const outOfPlane = rel[0] * nx + rel[1] * ny + rel[2] * nz;
  return { eq: (uu / radiusX) ** 2 + (vv / radiusY) ** 2, outOfPlane };
}

test('makeEllipse: every sampled point satisfies the ellipse equation exactly', () => {
  const center = [3, -5, 2];
  const xHat = [1, 0, 0], yHat = [0, 1, 0];
  const rx = 40, ry = 15;
  const crv = makeEllipse(center, xHat, yHat, rx, ry, 4);
  const knots = crv.knots;
  const uMin = knots[0], uMax = knots[knots.length - 1];
  let worst = 0, worstPlane = 0;
  for (let i = 0; i <= 200; i++) {
    const u = uMin + (uMax - uMin) * (i / 200);
    const { eq, outOfPlane } = ellipseResidualAt(crv, center, xHat, yHat, rx, ry, u);
    worst = Math.max(worst, Math.abs(eq - 1));
    worstPlane = Math.max(worstPlane, Math.abs(outOfPlane));
  }
  assert.ok(worst < 1e-9, `worst ellipse-equation residual ${worst} should be < 1e-9`);
  assert.ok(worstPlane < 1e-9, `worst out-of-plane drift ${worstPlane} should be < 1e-9`);
});

test('makeEllipse: exact on a rotated/oblique in-plane frame too', () => {
  const center = [0, 0, 0];
  // an oblique but orthonormal plane (its normal is NOT a world axis)
  const xHat = [0.6, 0.8, 0], yHat = [0, 0, 1];
  const rx = 25, ry = 60;
  const crv = makeEllipse(center, xHat, yHat, rx, ry, 6);
  const knots = crv.knots;
  const uMin = knots[0], uMax = knots[knots.length - 1];
  let worst = 0, worstPlane = 0;
  for (let i = 0; i <= 200; i++) {
    const u = uMin + (uMax - uMin) * (i / 200);
    const { eq, outOfPlane } = ellipseResidualAt(crv, center, xHat, yHat, rx, ry, u);
    worst = Math.max(worst, Math.abs(eq - 1));
    worstPlane = Math.max(worstPlane, Math.abs(outOfPlane));
  }
  assert.ok(worst < 1e-9, `worst residual ${worst} should be < 1e-9`);
  assert.ok(worstPlane < 1e-9, `worst out-of-plane drift ${worstPlane} should be < 1e-9`);
});

test('makeEllipse: radiusX === radiusY reproduces makeCircle bit-for-bit', () => {
  const center = [1, 2, 3];
  const xHat = [1, 0, 0], yHat = [0, 1, 0];
  const r = 17;
  const ell = makeEllipse(center, xHat, yHat, r, r, 4);
  const cir = makeCircle(center, xHat, yHat, r, 4);
  assert.equal(ell.degree, cir.degree);
  assert.deepEqual(ell.knots, cir.knots);
  assert.equal(ell.ctrlPts.length, cir.ctrlPts.length);
  for (let i = 0; i < ell.ctrlPts.length; i++) {
    for (let k = 0; k < 4; k++) {
      assert.ok(Math.abs(ell.ctrlPts[i][k] - cir.ctrlPts[i][k]) < 1e-12,
        `ctrlPt ${i} component ${k}: ellipse ${ell.ctrlPts[i][k]} vs circle ${cir.ctrlPts[i][k]}`);
    }
  }
});

test('makeEllipse: no NaN/Infinity anywhere in the control net, at several segment counts', () => {
  for (const seg of [4, 6, 8, 12]) {
    const crv = makeEllipse([0, 0, 0], [1, 0, 0], [0, 1, 0], 30, 7, seg);
    for (const cp of crv.ctrlPts) for (const v of cp) assert.ok(Number.isFinite(v), `non-finite ${v} at segments=${seg}`);
    for (const k of crv.knots) assert.ok(Number.isFinite(k));
  }
});

test('makeEllipse: the two radii genuinely differ the shape (not a coincidental circle)', () => {
  // The point at parameter-start (angle 0) sits at +radiusX along xHat; a
  // quarter-turn later it sits at +radiusY along yHat — different distances
  // from center, which a true circle could never produce.
  const crv = makeEllipse([0, 0, 0], [1, 0, 0], [0, 1, 0], 50, 10, 4);
  const p0 = curvePoint(crv, crv.knots[0]);              // angle 0 -> (50, 0)
  const pQuarter = curvePoint(crv, 1);                    // one full arc span (90deg) -> (0, 10)
  assert.ok(Math.abs(Math.hypot(p0[0], p0[1]) - 50) < 1e-9, `start point radius ${Math.hypot(p0[0], p0[1])} should be 50`);
  assert.ok(Math.abs(Math.hypot(pQuarter[0], pQuarter[1]) - 10) < 1e-9, `quarter point radius ${Math.hypot(pQuarter[0], pQuarter[1])} should be 10`);
});
