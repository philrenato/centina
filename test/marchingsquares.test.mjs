import { test } from 'node:test';
import assert from 'node:assert/strict';
import { marchingSquares } from '../kernel/marchingsquares.mjs';

// Build a flat uCount*vCount field from a callback f(i,j).
function field(uCount, vCount, f) {
  const v = new Array(uCount * vCount);
  for (let i = 0; i < uCount; i++) for (let j = 0; j < vCount; j++) v[i * vCount + j] = f(i, j);
  return v;
}

test('empty result: no cell above threshold produces ZERO contours (honest empty, not a crash)', () => {
  const v = field(20, 20, () => 0);
  const c = marchingSquares(v, 20, 20, 0.5, {});
  assert.equal(c.length, 0);
});

test('all-above result: no cell straddles the threshold produces ZERO contours', () => {
  const v = field(20, 20, () => 1);
  const c = marchingSquares(v, 20, 20, 0.5, {});
  assert.equal(c.length, 0);
});

test('a single Gaussian bump traces exactly ONE closed, circle-ish contour at a mid threshold', () => {
  // Peak 1 at (10,10), sigma 4. Contour at 0.5 is a circle of radius
  // r = sigma*sqrt(2 ln 2) ≈ 4.7096 around (10,10).
  const cx = 10, cy = 10, sigma = 4;
  const v = field(21, 21, (i, j) => Math.exp(-(((i - cx) ** 2) + ((j - cy) ** 2)) / (2 * sigma * sigma)));
  const contours = marchingSquares(v, 21, 21, 0.5, {});
  assert.equal(contours.length, 1, 'exactly one contour');
  const c = contours[0];
  assert.ok(c.closed, 'the contour is a closed loop');
  const rExpect = sigma * Math.sqrt(2 * Math.log(2));
  // Every vertex sits at ~rExpect from the center (a real, checkable
  // geometric property — this is a circle, not just "some curve came out").
  let minR = Infinity, maxR = -Infinity;
  for (const [ci, cj] of c.pts) {
    const r = Math.hypot(ci - cx, cj - cy);
    minR = Math.min(minR, r); maxR = Math.max(maxR, r);
  }
  assert.ok(Math.abs(minR - rExpect) < 0.25, `min radius ${minR.toFixed(3)} ~ ${rExpect.toFixed(3)}`);
  assert.ok(Math.abs(maxR - rExpect) < 0.25, `max radius ${maxR.toFixed(3)} ~ ${rExpect.toFixed(3)}`);
  // A closed contour must genuinely go all the way around: its vertices span
  // the full angular range, not a partial arc.
  const angs = c.pts.map(([ci, cj]) => Math.atan2(cj - cy, ci - cx));
  assert.ok(Math.max(...angs) - Math.min(...angs) > 5.5, 'the loop wraps a full 2π, not a partial arc');
});

test('a linear ramp traces a single straight iso-line at the exact constant column', () => {
  // values increase with i (column index); the 10.5 iso-line is the vertical
  // line i = 10.5, spanning every j.
  const v = field(21, 21, (i) => i);
  const contours = marchingSquares(v, 21, 21, 10.5, {});
  assert.equal(contours.length, 1, 'one iso-line');
  const c = contours[0];
  assert.ok(!c.closed, 'an open contour (a line from one edge to the other)');
  for (const [ci] of c.pts) assert.ok(Math.abs(ci - 10.5) < 1e-9, `every vertex sits at ci=10.5, got ${ci}`);
  // spans the full j range 0..20
  const js = c.pts.map((p) => p[1]);
  assert.ok(Math.min(...js) < 0.5 && Math.max(...js) > 19.5, 'the line spans the full V extent');
});

test('SEAM WRAP: a bump straddling a closed-U seam traces ONE continuous contour with wrap, TWO without', () => {
  // A bump centered ON the seam (column 0, which wraps to column uCount).
  // Distance from the seam is the WRAPPED column distance min(i, uCount-i).
  const uCount = 24, vCount = 21, cy = 10, sigma = 4;
  const v = field(uCount, vCount, (i, j) => {
    const di = Math.min(i, uCount - i);
    return Math.exp(-((di * di) + ((j - cy) ** 2)) / (2 * sigma * sigma));
  });
  const wrapped = marchingSquares(v, uCount, vCount, 0.5, { wrapU: true });
  assert.equal(wrapped.length, 1, 'with wrapU, the seam-straddling bump is ONE continuous contour');
  assert.ok(wrapped[0].closed, 'and it is a closed loop');
  // The single contour genuinely crosses the seam: it has vertices with ci
  // near 0 AND vertices with ci near uCount (the far side of the seam).
  const cis = wrapped[0].pts.map((p) => p[0]);
  assert.ok(Math.min(...cis) < 2, 'has vertices near ci=0 (one side of the seam)');
  assert.ok(Math.max(...cis) > uCount - 2, 'has vertices near ci=uCount (the OTHER side of the seam)');

  const unwrapped = marchingSquares(v, uCount, vCount, 0.5, {});
  assert.ok(unwrapped.length >= 2, `without wrap the SAME field splits into ${unwrapped.length} disconnected halves (>=2)`);
});

test('two separate bumps trace two separate contours', () => {
  const v = field(30, 20, (i, j) => {
    const g = (cx, cy) => Math.exp(-(((i - cx) ** 2) + ((j - cy) ** 2)) / (2 * 3 * 3));
    return Math.max(g(7, 10), g(22, 10));
  });
  const contours = marchingSquares(v, 30, 20, 0.5, {});
  assert.equal(contours.length, 2, 'two bumps -> two closed contours');
  assert.ok(contours.every((c) => c.closed));
});
