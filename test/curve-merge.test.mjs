import test from 'node:test';
import assert from 'node:assert/strict';
import { mergeCurves } from '../kernel/curvemerge.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { makeCircle } from '../kernel/primitives.mjs';
import { curvePoint } from '../kernel/curve.mjs';

const wave = (dy = 0, dz = 0) => globalCurveInterp(Array.from({ length: 9 }, (_, i) => [i * 10, 8 * Math.sin(i / 2) + dy, dz]), 3);
const reversed = (c) => { const a = c.knots[0], b = c.knots.at(-1); return { degree: c.degree, knots: c.knots.map((k) => a + b - k).reverse(), ctrlPts: c.ctrlPts.slice().reverse() }; };
const sampleAt = (c, t) => curvePoint(c, c.knots[0] + t * (c.knots.at(-1) - c.knots[0]));

test('two copies 0.01 mm apart merge into their middle, at the first copy\'s degree and count', () => {
  const a = wave(0, 0), b = wave(0, 0.01);
  const m = mergeCurves([a, b]);
  assert.equal(m.crv.degree, a.degree);
  assert.equal(m.crv.ctrlPts.length, a.ctrlPts.length);
  for (const t of [0, 0.3, 0.7, 1]) assert.ok(Math.abs(sampleAt(m.crv, t)[2] - 0.005) < 1e-3, `z ${sampleAt(m.crv, t)[2]}`);
  assert.ok(m.deviation < 0.006);
});

test('a reversed copy merges to the same shape, not a folded one', () => {
  const a = wave(), m = mergeCurves([a, reversed(a)]);
  assert.ok(m.deviation < 0.05, `deviation ${m.deviation}`);
  for (const t of [0, 0.25, 0.5, 0.75, 1]) { const p = sampleAt(m.crv, t), q = sampleAt(a, t); assert.ok(Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) < 0.2, `t ${t}`); }
});

test('two circles with their seams apart merge into the circle', () => {
  const a = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 10);
  const b = makeCircle([0, 0, 0], [0, 1, 0], [-1, 0, 0], 10); // the same circle, seam a quarter turn on
  const m = mergeCurves([a, b]);
  for (let i = 0; i < 24; i++) { const p = sampleAt(m.crv, i / 24); assert.ok(Math.abs(Math.hypot(p[0], p[1]) - 10) < 0.05, `radius ${Math.hypot(p[0], p[1])}`); }
  const s = sampleAt(m.crv, 0), e = sampleAt(m.crv, 1);
  assert.ok(Math.hypot(s[0] - e[0], s[1] - e[1]) < 1e-9, 'closed');
});

test('fewer than two curves is no merge', () => {
  assert.equal(mergeCurves([wave()]), null);
  assert.equal(mergeCurves([]), null);
});
