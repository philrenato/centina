import test from 'node:test';
import assert from 'node:assert/strict';
import { fairControlNet, fairParamsFromAmount } from '../kernel/fair.mjs';

// A 5x5 flat, planar net (Z=0 everywhere) with ONE deliberately-perturbed
// interior point (a real "bump" to smooth away) and real, non-uniform
// rational weights (so a test can confirm weight is left untouched).
function bumpyNet() {
  const nu = 5, nv = 5;
  const net = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) {
      const w = 1 + 0.1 * ((i + j) % 3); // real, non-uniform weights
      row.push([i * 10, j * 10, 0, w]);
    }
    net.push(row);
  }
  net[2][2][2] = 40; // a real bump at the exact center interior point
  return { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], ctrlNet: net };
}

test('fairParamsFromAmount: amount=0 gives 0 iterations, amount=1 gives the max', () => {
  assert.equal(fairParamsFromAmount(0).iterations, 0);
  assert.equal(fairParamsFromAmount(1).iterations, 20);
  assert.equal(fairParamsFromAmount(0.5).iterations, 10);
});

test('fairControlNet(srf, 0) is a byte-identical passthrough', () => {
  const srf = bumpyNet();
  const out = fairControlNet(srf, 0);
  assert.deepEqual(out.ctrlNet, srf.ctrlNet);
});

test('fairControlNet genuinely relaxes the bumped interior point toward its own neighbor average, one real step at a time', () => {
  const srf = bumpyNet();
  const before = srf.ctrlNet[2][2][2]; // 40
  const oneIterAmount = 1 / 20; // maps to exactly 1 iteration via fairParamsFromAmount
  const oneIter = fairControlNet(srf, oneIterAmount);
  // Neighbors of (2,2): (1,2)=[10,20,0,w], (3,2)=[30,20,0,w], (2,1)=[20,10,0,w], (2,3)=[20,30,0,w] — all Z=0.
  const expectedAfterOne = before + (0 - before) * 0.5; // lambda=0.5, neighbor avg Z = 0
  assert.ok(Math.abs(oneIter.ctrlNet[2][2][2] - expectedAfterOne) < 1e-9, `expected ${expectedAfterOne}, got ${oneIter.ctrlNet[2][2][2]}`);
});

test('fairControlNet NEVER moves the boundary rows/columns — surface edges/corners stay pinned exactly', () => {
  const srf = bumpyNet();
  const out = fairControlNet(srf, 1);
  for (let j = 0; j < 5; j++) {
    assert.deepEqual(out.ctrlNet[0][j], srf.ctrlNet[0][j], `row 0, col ${j}`);
    assert.deepEqual(out.ctrlNet[4][j], srf.ctrlNet[4][j], `row 4, col ${j}`);
  }
  for (let i = 0; i < 5; i++) {
    assert.deepEqual(out.ctrlNet[i][0], srf.ctrlNet[i][0], `row ${i}, col 0`);
    assert.deepEqual(out.ctrlNet[i][4], srf.ctrlNet[i][4], `row ${i}, col 4`);
  }
});

test('fairControlNet never touches a control point\'s own rational weight, only its XYZ position', () => {
  const srf = bumpyNet();
  const out = fairControlNet(srf, 1);
  for (let i = 0; i < 5; i++) {
    for (let j = 0; j < 5; j++) {
      assert.equal(out.ctrlNet[i][j][3], srf.ctrlNet[i][j][3], `weight at (${i},${j}) must be untouched`);
    }
  }
});

test('a higher Smoothness amount relaxes the bump further than a lower one (monotonic convergence)', () => {
  const srf = bumpyNet();
  const low = fairControlNet(srf, 0.15);   // 3 iterations
  const high = fairControlNet(srf, 0.75);  // 15 iterations
  const bumpLow = Math.abs(low.ctrlNet[2][2][2]);
  const bumpHigh = Math.abs(high.ctrlNet[2][2][2]);
  assert.ok(bumpHigh < bumpLow, `higher amount (${bumpHigh}) should relax the bump further than a lower one (${bumpLow})`);
  assert.ok(bumpHigh >= 0, 'never overshoots past the true equilibrium (0) into a negative bump');
});

test('a surface with fewer than 3 rows or columns has no interior point to relax — an honest no-op, not a crash', () => {
  const twoByN = { degU: 1, degV: 3, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 0, 0, 1, 1, 1, 1], ctrlNet: [[[0, 0, 0, 1], [0, 10, 0, 1], [0, 20, 0, 1], [0, 30, 0, 1]], [[10, 0, 0, 1], [10, 10, 5, 1], [10, 20, 0, 1], [10, 30, 0, 1]]] };
  const out = fairControlNet(twoByN, 1);
  assert.deepEqual(out.ctrlNet, twoByN.ctrlNet);
});
