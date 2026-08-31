// FALLOFF — a displacement that fades toward the edges instead of running at
// full strength into them.
//
// The claim that matters is not "the numbers got smaller". It is that the
// BOUNDARY ROWS STOP MOVING, because those are the rows a Join, a Match Edge or
// a Bridge depends on — a seam that shifts because someone added noise is the
// thing this exists to prevent. And that the default is unchanged, since every
// existing surface in every existing file was made without it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { noiseControlNet, boundaryFalloffGrid } from '../kernel/noise.mjs';
import { waveControlNet } from '../kernel/wave.mjs';

function grid(nu, nv) {
  const cl = (n, d) => {
    const k = [];
    for (let q = 0; q <= d; q++) k.push(0);
    for (let q = 1; q <= n - d - 1; q++) k.push(q / (n - d));
    for (let q = 0; q <= d; q++) k.push(1);
    return k;
  };
  const rows = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) row.push([(60 * i) / (nu - 1), (60 * j) / (nv - 1), 0, 1]);
    rows.push(row);
  }
  return { degU: 3, degV: 3, knotsU: cl(nu, 3), knotsV: cl(nv, 3), ctrlNet: rows };
}
const moved = (a, b) => {
  let worst = 0;
  for (let i = 0; i < a.ctrlNet.length; i++) {
    for (let j = 0; j < a.ctrlNet[i].length; j++) {
      const p = a.ctrlNet[i][j], q = b.ctrlNet[i][j];
      worst = Math.max(worst, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    }
  }
  return worst;
};
// How far the OUTERMOST rows moved — the ones a neighbour is joined to.
const movedOnBoundary = (a, b) => {
  const nu = a.ctrlNet.length, nv = a.ctrlNet[0].length;
  let worst = 0;
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      if (i > 1 && i < nu - 2 && j > 1 && j < nv - 2) continue;
      const p = a.ctrlNet[i][j], q = b.ctrlNet[i][j];
      worst = Math.max(worst, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    }
  }
  return worst;
};

test('falloff 0 is the IDENTITY — every existing surface is untouched by this', () => {
  const g = boundaryFalloffGrid(9, 9, 0);
  for (const row of g) for (const v of row) assert.equal(v, 1);
});

test('falloff 1 fades to nothing across BOTH continuity rows, full strength inside', () => {
  const g = boundaryFalloffGrid(9, 9, 1);
  assert.ok(g[0][4] < 1e-9, `the boundary row fades to nothing, got ${g[0][4]}`);
  assert.ok(g[1][4] < 1e-9, `and so does the row behind it, which carries the tangent, got ${g[1][4]}`);
  assert.ok(g[4][4] > 0.99, `the middle keeps full strength, got ${g[4][4]}`);
  // Monotone from edge to centre — a fade that wobbles would read as a defect
  // in the surface rather than in the fade.
  for (let i = 1; i <= 4; i++) assert.ok(g[i][4] >= g[i - 1][4] - 1e-12, `row ${i} does not dip`);
});

test('⭐ NOISE with falloff leaves the boundary rows where they were', () => {
  const srf = grid(9, 9);
  const params = { style: 'value', amplitude: 6, frequency: 3, seed: 7, direction: 'world-z' };
  const plain = noiseControlNet(srf, params);
  const faded = noiseControlNet(srf, { ...params, falloff: 1 });
  assert.ok(movedOnBoundary(srf, plain) > 0.5,
    `⚠ without falloff the boundary DOES move (${movedOnBoundary(srf, plain).toFixed(3)}mm) — which is the problem`);
  assert.ok(movedOnBoundary(srf, faded) < 1e-6,
    `⭐ with falloff it does not (${movedOnBoundary(srf, faded).toFixed(6)}mm)`);
  assert.ok(moved(srf, faded) > 0.5,
    `and the middle still moves (${moved(srf, faded).toFixed(3)}mm) — a falloff that killed the whole effect would pass the check above`);
});

test('⭐ WAVE with falloff leaves the boundary rows where they were', () => {
  const srf = grid(9, 9);
  const params = { axis: 'u', amplitude: 6, frequency: 2, direction: 'world-z' };
  const plain = waveControlNet(srf, params);
  const faded = waveControlNet(srf, { ...params, falloff: 1 });
  assert.ok(movedOnBoundary(srf, plain) > 0.5,
    `⚠ without falloff the boundary DOES move (${movedOnBoundary(srf, plain).toFixed(3)}mm)`);
  assert.ok(movedOnBoundary(srf, faded) < 1e-6,
    `⭐ with falloff it does not (${movedOnBoundary(srf, faded).toFixed(6)}mm)`);
  assert.ok(moved(srf, faded) > 0.5, 'and the middle still moves');
});

test('a partial falloff is between the two, not a switch', () => {
  const srf = grid(9, 9);
  const params = { axis: 'u', amplitude: 6, frequency: 2, direction: 'world-z' };
  const half = waveControlNet(srf, { ...params, falloff: 0.5 });
  const none = waveControlNet(srf, params);
  const full = waveControlNet(srf, { ...params, falloff: 1 });
  const b = (x) => movedOnBoundary(srf, x);
  assert.ok(b(half) < b(none) && b(half) > b(full),
    `the boundary movement falls with the slider (${b(none).toFixed(3)} -> ${b(half).toFixed(3)} -> ${b(full).toFixed(6)})`);
});
