import { test } from 'node:test';
import assert from 'node:assert/strict';
import { offsetCurve2D } from '../kernel/offsetcurve.mjs';
import { makeLine, makeCircle } from '../kernel/primitives.mjs';
import { curvePoint } from '../kernel/curve.mjs';

test('offsetCurve2D: a straight line offsets exactly to a parallel line (the flat-plane exactness anchor)', () => {
  const line = makeLine([0, 0, 0], [10, 0, 0]); // along +X
  const offset = offsetCurve2D(line, 3, [0, 0, 1]); // plane normal +Z -> perp is +Y
  // Every control point should have moved by exactly (0, 3, 0).
  for (let i = 0; i < line.ctrlPts.length; i++) {
    const p0 = line.ctrlPts[i], p1 = offset.ctrlPts[i];
    assert.ok(Math.abs(p1[0] - p0[0]) < 1e-12, `x unchanged at ${i}`);
    assert.ok(Math.abs(p1[1] - (p0[1] + 3)) < 1e-12, `y offset by exactly 3 at ${i}`);
    assert.ok(Math.abs(p1[2] - p0[2]) < 1e-12, `z unchanged at ${i}`);
    assert.equal(p1[3], p0[3], 'weight untouched');
  }
  // An independent check: sample the offset curve at several parameters and
  // confirm every sampled point lies on the parallel line (y === 3
  // everywhere, x spans the same 0..10 range).
  for (const u of [0, 0.25, 0.5, 0.75, 1]) {
    const pt = curvePoint(offset, u);
    assert.ok(Math.abs(pt[1] - 3) < 1e-9, `sampled point at u=${u} sits at y=3`);
  }
});

test('offsetCurve2D: offsetting the other direction (negative distance) is the exact mirror', () => {
  const line = makeLine([0, 0, 0], [10, 0, 0]);
  const offset = offsetCurve2D(line, -3, [0, 0, 1]);
  for (const u of [0, 0.5, 1]) {
    const pt = curvePoint(offset, u);
    assert.ok(Math.abs(pt[1] - -3) < 1e-9, `negative offset sits at y=-3, got ${pt[1]}`);
  }
});

test('offsetCurve2D: a circle offsets to a different-radius, still-round curve, magnitude close to |r +/- d|', () => {
  // Sign convention: the offset direction is normalize(planeNormal x
  // tangent), so it depends on the curve's own winding relative to the
  // chosen plane normal (Rhino's OffsetCrv likewise needs a direction pick
  // to disambiguate). For this circle's construction (xAxis=[1,0,0],
  // yAxis=[0,1,0], a standard CCW winding) and planeNormal [0,0,1], the
  // resolved direction is inward (radius shrinks). The test checks the
  // magnitude of the resulting offset (close to r-d or r+d), not an assumed
  // sign.
  const circle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 1);
  const offset = offsetCurve2D(circle, 2, [0, 0, 1]);
  let minR = Infinity, maxR = -Infinity;
  for (let i = 0; i <= 40; i++) {
    const u = circle.knots[circle.degree] + (i / 40) * (circle.knots[circle.knots.length - 1 - circle.degree] - circle.knots[circle.degree]);
    const pt = curvePoint(offset, u);
    const r = Math.hypot(pt[0], pt[1]);
    minR = Math.min(minR, r); maxR = Math.max(maxR, r);
  }
  const midR = (minR + maxR) / 2;
  const closeToInward = Math.abs(midR - 8) < 1;
  const closeToOutward = Math.abs(midR - 12) < 1;
  assert.ok(closeToInward || closeToOutward, `offset circle's own radius sits near either r-d=8 or r+d=12, got midpoint ${midR.toFixed(3)} (range [${minR.toFixed(3)}, ${maxR.toFixed(3)}])`);
  assert.ok(maxR - minR < 1, `offset circle stays reasonably round (max-min radius spread ${(maxR - minR).toFixed(4)} < 1)`);
});

test('offsetCurve2D: weights are preserved (a rational circle stays rational with the same weights)', () => {
  const circle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 1);
  const offset = offsetCurve2D(circle, 2, [0, 0, 1]);
  for (let i = 0; i < circle.ctrlPts.length; i++) {
    assert.equal(offset.ctrlPts[i][3], circle.ctrlPts[i][3], `weight at control point ${i} unchanged`);
  }
});

test('offsetCurve2D: refuses when the plane normal is parallel to the curve\'s own tangent everywhere', () => {
  const line = makeLine([0, 0, 0], [10, 0, 0]); // tangent is +X everywhere
  assert.throws(() => offsetCurve2D(line, 3, [1, 0, 0]), /parallel|degenerate/i);
});

test('offsetCurve2D: zero distance is the exact identity (every point unchanged)', () => {
  const circle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 1);
  const offset = offsetCurve2D(circle, 0, [0, 0, 1]);
  for (let i = 0; i < circle.ctrlPts.length; i++) {
    for (let k = 0; k < 4; k++) assert.ok(Math.abs(offset.ctrlPts[i][k] - circle.ctrlPts[i][k]) < 1e-12, `component ${k} of point ${i} unchanged at zero distance`);
  }
});

test('offsetCurve2D: result is finite everywhere (no NaN/Infinity) across a sampled curve', () => {
  const circle = makeCircle([2, 5, -1], [1, 0, 0], [0, 1, 0], 7, 2);
  const offset = offsetCurve2D(circle, 1.5, [0, 0, 1]);
  for (const p of offset.ctrlPts) for (const v of p) assert.ok(Number.isFinite(v), 'every offset control point component is finite');
});
