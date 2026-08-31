import test from 'node:test';
import assert from 'node:assert/strict';
import { findSpan, basisFuns, dersBasisFuns } from '../kernel/basis.mjs';

// P&T's own worked example (Ex2.3 / the standard textbook knot vector):
// p=2, U=[0,0,0,1,2,3,4,4,5,5,5], n=7 (8 control points, indices 0..7).
const p = 2;
const U = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5];
const n = 7;

test('findSpan matches the P&T textbook example at u=5/2', () => {
  assert.equal(findSpan(n, p, 5 / 2, U), 4);
});

test('findSpan clamps at the domain ends', () => {
  assert.equal(findSpan(n, p, 0, U), p);
  assert.equal(findSpan(n, p, 5, U), n);
});

test('basisFuns sums to 1 (partition of unity) across many u values', () => {
  for (let u = 0; u <= 5; u += 0.1) {
    const span = findSpan(n, p, u, U);
    const N = basisFuns(span, u, p, U);
    const sum = N.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-10, `u=${u} sum=${sum}`);
  }
});

test('basisFuns are non-negative everywhere in the domain', () => {
  for (let u = 0; u <= 5; u += 0.05) {
    const span = findSpan(n, p, u, U);
    const N = basisFuns(span, u, p, U);
    for (const v of N) assert.ok(v >= -1e-12, `negative basis value ${v} at u=${u}`);
  }
});

test('dersBasisFuns 1st derivative matches central-difference of basisFuns', () => {
  const h = 1e-6;
  for (let u = 0.2; u < 4.8; u += 0.37) {
    const span = findSpan(n, p, u, U);
    const ders = dersBasisFuns(span, u, p, 1, U);

    // Central difference needs the SAME span for u-h/u+h to compare the same
    // basis-function indices — pick points safely inside one span's interior.
    const spanMinus = findSpan(n, p, u - h, U);
    const spanPlus = findSpan(n, p, u + h, U);
    if (spanMinus !== span || spanPlus !== span) continue; // skip near span boundaries

    const Nminus = basisFuns(span, u - h, p, U);
    const Nplus = basisFuns(span, u + h, p, U);
    for (let j = 0; j <= p; j++) {
      const fd = (Nplus[j] - Nminus[j]) / (2 * h);
      assert.ok(Math.abs(fd - ders[1][j]) < 1e-4,
        `u=${u} j=${j} analytic=${ders[1][j]} fd=${fd}`);
    }
  }
});

test('dersBasisFuns row 0 equals basisFuns', () => {
  const u = 2.3;
  const span = findSpan(n, p, u, U);
  const N = basisFuns(span, u, p, U);
  const ders = dersBasisFuns(span, u, p, 2, U);
  for (let j = 0; j <= p; j++) {
    assert.ok(Math.abs(N[j] - ders[0][j]) < 1e-12);
  }
});
