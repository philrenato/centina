// END BULGE — the claim is not "the surface changed", it is "the surface changed
// AND the edge did not move AND the surface did not tilt there".
//
// ⚠ THE SECOND HALF OF THAT IS CONDITIONAL, AND THESE TESTS SAY SO. The boundary
// row is never written, so G0 is exact on every net. The tangent plane is exact
// on POLYNOMIAL nets and on rational nets whose second-row weights are
// PROPORTIONAL to the boundary row's — which is every surface this kernel builds
// — and NOT on a rational net with non-proportional rows, where the edge stays
// put and the plane still tilts (measured: 1.73 degrees mid-edge, 0 at both
// corners, i.e. exactly between the stations a control-point check can see).
// The last three tests pin which regime is reported, because a guarantee that is
// asserted rather than measured is the thing that goes quietly wrong.
import { strict as assert } from 'node:assert';
import { endBulgeNet, edgeRows } from '../kernel/matchedge.mjs';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const L = len(a); return [a[0] / L, a[1] / L, a[2] / L]; };

// A 4x4 net with real depth, deliberately NOT flat and NOT axis-aligned in the
// depth direction, so "the direction was preserved" is a claim with content.
function net4x4() {
  const out = [];
  for (let i = 0; i < 4; i++) {
    const row = [];
    for (let j = 0; j < 4; j++) row.push([i * 10, j * 7 + i * 2, Math.sin(i + j) * 3, 1]);
    out.push(row);
  }
  return out;
}

let passed = 0, failed = 0;
function t(name, fn) {
  try { fn(); passed++; console.log(`  PASS: ${name}`); }
  catch (e) { failed++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

t('the boundary row is bit-identical — G0 is exact by construction, not by tolerance', () => {
  const base = net4x4();
  const { ok, net } = endBulgeNet(base, 'u0', 1.8);
  assert.equal(ok, true);
  const rows = edgeRows(base, 'u0');
  for (const [i, j] of rows.boundary) {
    assert.deepEqual(net[i][j], base[i][j], `boundary point ${i},${j} moved`);
  }
});

t('every column keeps its exact direction — the tangent plane cannot tilt', () => {
  const base = net4x4();
  const { net } = endBulgeNet(base, 'u0', 2.5);
  const rows = edgeRows(base, 'u0');
  for (let k = 0; k < rows.second.length; k++) {
    const [si, sj] = rows.second[k];
    const [bi, bj] = rows.boundary[k];
    const before = unit(sub(base[si][sj], base[bi][bj]));
    const after = unit(sub(net[si][sj], base[bi][bj]));
    for (let c = 0; c < 3; c++) {
      assert.ok(Math.abs(before[c] - after[c]) < 1e-12, `column ${k} turned: ${before} -> ${after}`);
    }
  }
});

t('the distance scales by exactly the factor asked for', () => {
  const base = net4x4();
  const factor = 1.75;
  const { net } = endBulgeNet(base, 'v0', factor);
  const rows = edgeRows(base, 'v0');
  for (let k = 0; k < rows.second.length; k++) {
    const [si, sj] = rows.second[k];
    const [bi, bj] = rows.boundary[k];
    const before = len(sub(base[si][sj], base[bi][bj]));
    const after = len(sub(net[si][sj], base[bi][bj]));
    assert.ok(Math.abs(after - before * factor) < 1e-9, `column ${k}: ${before} * ${factor} != ${after}`);
  }
});

t('a factor of 1 changes nothing at all', () => {
  const base = net4x4();
  const { net } = endBulgeNet(base, 'u1', 1);
  assert.deepEqual(net, base);
});

t('it REFUSES a net only two deep, naming why', () => {
  // Two control points across is a ruled/extruded surface: the "second" row IS
  // the opposite edge, so bulging would move the far side.
  const ruled = [
    [[0, 0, 0, 1], [0, 10, 0, 1]],
    [[10, 0, 0, 1], [10, 10, 0, 1]],
  ];
  const r = endBulgeNet(ruled, 'v0', 2);
  assert.equal(r.ok, false);
  assert.match(r.reason, /opposite edge|far side/);
});

t('it refuses a non-positive or non-finite fullness', () => {
  const base = net4x4();
  assert.equal(endBulgeNet(base, 'u0', 0).ok, false);
  assert.equal(endBulgeNet(base, 'u0', -1).ok, false);
  assert.equal(endBulgeNet(base, 'u0', NaN).ok, false);
});

t('a rational net keeps every weight it had', () => {
  // ctrlNet stores plain positions with the weight alongside — dividing by w
  // here would move every control point of a rational surface, and w is 1 on
  // every hand-built fixture, so the mistake stays invisible until it meets a
  // primitive sphere.
  const base = net4x4().map((row, i) => row.map((cp, j) => [cp[0], cp[1], cp[2], 1 + (i + j) * 0.25]));
  const { net } = endBulgeNet(base, 'u0', 1.4);
  for (let i = 0; i < base.length; i++) {
    for (let j = 0; j < base[0].length; j++) {
      assert.equal(net[i][j][3], base[i][j][3], `weight at ${i},${j} changed`);
    }
  }
});

t('it leaves a degenerate (pole) column exactly where it was', () => {
  const base = net4x4();
  base[1][2] = [base[0][2][0], base[0][2][1], base[0][2][2], 1]; // second row sits ON the edge
  const { net } = endBulgeNet(base, 'u0', 3);
  assert.deepEqual(net[1][2], base[1][2]);
});

t('the input net is never mutated', () => {
  const base = net4x4();
  const copy = JSON.parse(JSON.stringify(base));
  endBulgeNet(base, 'u0', 2);
  assert.deepEqual(base, copy);
});


// ── THE GUARANTEE IS REPORTED, NOT ASSUMED ──────────────────────────────────
t('a polynomial net reports the exact tangent-plane guarantee', () => {
  const r = endBulgeNet(net4x4(), 'u0', 2);
  assert.equal(r.tangentPlaneExact, true);
});

t('a rational net with PROPORTIONAL weight rows still reports exact', () => {
  // w1j = k * w0j — a sphere, a revolve, a pipe, a torus. B collapses to zero.
  const base = net4x4().map((row, i) => row.map((cp, j) => [cp[0], cp[1], cp[2], (1 + j * 0.3) * (i === 1 ? 2 : 1)]));
  const r = endBulgeNet(base, 'u0', 1.5);
  assert.equal(r.tangentPlaneExact, true);
});

t('a rational net with NON-proportional weight rows reports the weaker guarantee', () => {
  // The measured case: edge exact, tangent plane tilts 1.73 deg mid-edge and 0
  // at both corners — between the stations a control-point check can see.
  const base = net4x4();
  base[0][1] = [base[0][1][0], base[0][1][1], base[0][1][2], 0.7071];
  const r = endBulgeNet(base, 'u0', 2);
  assert.equal(r.ok, true);
  assert.equal(r.tangentPlaneExact, false);
});


t('it reports how far it can go before the net folds', () => {
  const base = net4x4();
  const r = endBulgeNet(base, 'u0', 1.2);
  assert.ok(Number.isFinite(r.maxSafeFactor), `expected a finite fold limit, got ${r.maxSafeFactor}`);
  assert.ok(r.maxSafeFactor > 1, 'a net that is not already folded must allow SOME growth');
  assert.equal(r.folds, false);
});

t('and it says so when the request would fold it', () => {
  const base = net4x4();
  const probe = endBulgeNet(base, 'u0', 1.2);
  const tooFar = endBulgeNet(base, 'u0', probe.maxSafeFactor * 1.5);
  assert.equal(tooFar.folds, true, 'a request past the fold limit must be flagged');
});

console.log(`\n${passed}/${passed + failed} checks passed.`);
if (failed) process.exit(1);
