import test from 'node:test';
import assert from 'node:assert/strict';
import { makeSquircle2D } from '../kernel/primitives.mjs';
import { curvePoint, rationalCurveDerivs } from '../kernel/curve.mjs';

function isFiniteNet(ctrlPts) {
  for (const cp of ctrlPts) for (const v of cp) if (!Number.isFinite(v)) return false;
  return true;
}

// A closed, simple (non-self-intersecting) curve that encloses the origin is
// star-shaped iff its polar angle is monotone and it winds exactly once — a
// robust, cheap simplicity witness for these convex-ish cages.
function analyze(crv) {
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const p0 = curvePoint(crv, uMin), p1 = curvePoint(crv, uMax);
  const gap = Math.hypot(p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]);
  const N = 1200;
  let prevAng = null, monotone = true, winding = 0, minR = Infinity, minSpeed = Infinity;
  for (let i = 0; i <= N; i++) {
    const u = uMin + (uMax - uMin) * (i / N);
    const P = curvePoint(crv, u);
    minR = Math.min(minR, Math.hypot(P[0], P[1]));
    const ang = Math.atan2(P[1], P[0]);
    if (prevAng !== null) {
      let d = ang - prevAng;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      if (d < -1e-9) monotone = false;
      winding += d;
    }
    prevAng = ang;
    if (i > 0 && i < N) { // avoid the exact clamped endpoints (derivative eval edge case)
      const der = rationalCurveDerivs(crv, u, 1)[1];
      minSpeed = Math.min(minSpeed, Math.hypot(der[0], der[1], der[2]));
    }
  }
  return { gap, monotone, winding: winding / (2 * Math.PI), minR, minSpeed };
}

// squareness = (radius toward the diagonal corner) / (radius toward the +x
// edge); 1 == circle, √2 == a true square's corner/edge ratio. Only meaningful
// for a square cage (hw === hh).
function squareness(crv) {
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const radAt = (deg) => {
    let best = Infinity, r = 0;
    for (let i = 0; i <= 3000; i++) {
      const u = uMin + (uMax - uMin) * (i / 3000);
      const P = curvePoint(crv, u);
      const a = (Math.atan2(P[1], P[0]) * 180) / Math.PI;
      const d = Math.abs(a - deg);
      if (d < best) { best = d; r = Math.hypot(P[0], P[1]); }
    }
    return r;
  };
  return radAt(45) / radAt(0);
}

test('makeSquircle2D: closed, simple, cubic, isFiniteNet at every softness including extremes', () => {
  for (const s of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
    const crv = makeSquircle2D([0, 0, 0], [1, 0, 0], [0, 1, 0], 30, 30, s);
    assert.equal(crv.degree, 3, `softness ${s}: degree`);
    assert.ok(isFiniteNet(crv.ctrlPts), `softness ${s}: isFiniteNet`);
    const a = analyze(crv);
    assert.ok(a.gap < 1e-9, `softness ${s}: gap ${a.gap} (must be closed)`);
    assert.ok(a.monotone, `softness ${s}: polar angle not monotone (self-intersecting?)`);
    assert.ok(Math.abs(a.winding - 1) < 1e-6, `softness ${s}: winding ${a.winding} (must be exactly 1)`);
    assert.ok(a.minR > 1, `softness ${s}: min radius ${a.minR} collapsed toward origin`);
    assert.ok(a.minSpeed > 1e-3, `softness ${s}: min curve speed ${a.minSpeed} (a cusp/zero-derivative)`);
  }
});

test('makeSquircle2D: softness smoothly and monotonically rounds a square-ish shape toward a circle', () => {
  const sq = [];
  for (const s of [0, 0.25, 0.5, 0.75, 1]) {
    sq.push(squareness(makeSquircle2D([0, 0, 0], [1, 0, 0], [0, 1, 0], 30, 30, s)));
  }
  // s=0 is clearly square-ish (corner reaches well past the edge), s=1 is
  // near-circular (ratio near 1), and it decreases monotonically in between.
  assert.ok(sq[0] > 1.25, `softness 0 squareness ${sq[0]} should read square-ish (> 1.25)`);
  assert.ok(sq[4] < 1.06, `softness 1 squareness ${sq[4]} should read near-circular (< 1.06)`);
  for (let i = 1; i < sq.length; i++) {
    assert.ok(sq[i] < sq[i - 1] - 1e-3, `squareness not strictly decreasing at step ${i}: ${sq.join(', ')}`);
  }
});

test('makeSquircle2D: non-square aspect ratios stay closed and simple', () => {
  for (const [hw, hh, s] of [[40, 20, 0], [40, 20, 0.5], [40, 20, 1], [15, 60, 0.3], [50, 8, 0.7]]) {
    const crv = makeSquircle2D([2, -3, 5], [1, 0, 0], [0, 1, 0], hw, hh, s);
    assert.ok(isFiniteNet(crv.ctrlPts), `${hw}x${hh} s=${s}: finite`);
    const a = analyze(crv);
    assert.ok(a.gap < 1e-9, `${hw}x${hh} s=${s}: gap ${a.gap}`);
    assert.ok(a.monotone && Math.abs(a.winding - 1) < 1e-6, `${hw}x${hh} s=${s}: simple/winding`);
    assert.ok(a.minSpeed > 1e-3, `${hw}x${hh} s=${s}: min speed ${a.minSpeed}`);
  }
});

test('makeSquircle2D: honors an oblique in-plane frame (curve lies in that plane)', () => {
  const center = [1, 2, 3];
  const xAxis = [0.6, 0.8, 0], yAxis = [0, 0, 1]; // orthonormal, normal = (0.8,-0.6,0)
  const crv = makeSquircle2D(center, xAxis, yAxis, 25, 18, 0.5);
  const nx = 0.8, ny = -0.6, nz = 0;
  let worst = 0;
  for (let i = 0; i <= 300; i++) {
    const u = crv.knots[0] + (crv.knots[crv.knots.length - 1] - crv.knots[0]) * (i / 300);
    const P = curvePoint(crv, u);
    const rel = [P[0] - center[0], P[1] - center[1], P[2] - center[2]];
    worst = Math.max(worst, Math.abs(rel[0] * nx + rel[1] * ny + rel[2] * nz));
  }
  assert.ok(worst < 1e-9, `out-of-plane drift ${worst}`);
});
