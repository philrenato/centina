// PATCH — a surface fitted through scattered curves and points.
//
// The fixtures here are chosen so each claim has an INDEPENDENT oracle rather
// than being checked against the fitter's own output:
//   · a plane and a saddle have closed forms, so "did it fit" is a distance to a
//     formula this file writes, not to anything patch.mjs computed;
//   · a sparse cloud has fewer samples than control points in places, which is
//     the case that makes an unregularised solve singular rather than merely
//     loose — it is here to prove the stiffness is load-bearing;
//   · a folded input has no single-valued answer at all, and the refusal is the
//     correct result.
import { strict as assert } from 'node:assert';
import { fitPatch, fitPatchToTolerance, samplePatchInputs, PATCH_REFUSAL } from '../kernel/patch.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

let passed = 0, failed = 0;
function t(name, fn) {
  try { fn(); passed++; console.log(`  PASS: ${name}`); }
  catch (e) { failed++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

const gridOn = (f, n = 12, span = 100) => {
  const out = [];
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
    const x = (i / n) * span - span / 2, y = (j / n) * span - span / 2;
    out.push([x, y, f(x, y)]);
  }
  return out;
};
const worstAgainst = (r, f, n = 9, span = 100) => {
  // Sample the FITTED surface on its own domain and measure against the closed
  // form — the fit's own residual is not consulted.
  let worst = 0;
  for (let i = 1; i < n; i++) for (let j = 1; j < n; j++) {
    const s = surfacePoint(r.srf, i / n, j / n);
    worst = Math.max(worst, Math.abs(s[2] - f(s[0], s[1])));
  }
  return worst;
};

t('a plane is fitted to within a hair of exact', () => {
  const f = (x, y) => 0.3 * x - 0.2 * y + 5;
  const r = fitPatch(gridOn(f), { uCount: 6, vCount: 6, stiffness: 0.5 });
  assert.equal(r.ok, true, r.reason);
  const worst = worstAgainst(r, f);
  assert.ok(worst < 1e-6, `plane should be exact, worst ${worst}`);
});

t('a saddle is fitted closely, measured against its own formula', () => {
  const f = (x, y) => (x * x - y * y) / 200;
  const r = fitPatch(gridOn(f), { uCount: 8, vCount: 8, stiffness: 0.2 });
  assert.equal(r.ok, true, r.reason);
  const worst = worstAgainst(r, f);
  assert.ok(worst < 1.5, `saddle worst deviation ${worst} over a 100mm span`);
});

t('more control points fit a curved surface more closely', () => {
  const f = (x, y) => 12 * Math.sin(x / 22) * Math.cos(y / 22);
  const pts = gridOn(f, 16);
  const coarse = fitPatch(pts, { uCount: 5, vCount: 5, stiffness: 0.2 });
  const fine = fitPatch(pts, { uCount: 10, vCount: 10, stiffness: 0.2 });
  assert.equal(coarse.ok, true); assert.equal(fine.ok, true);
  assert.ok(fine.maxDeviation < coarse.maxDeviation,
    `fine ${fine.maxDeviation} should beat coarse ${coarse.maxDeviation}`);
});

t('⭐ the stiffness is load-bearing: a sparse cloud still resolves', () => {
  // Far fewer samples than control points, clustered so whole regions of the net
  // have nothing near them. Unregularised this is singular, not just loose.
  const f = (x, y) => 0.02 * x * y / 10;
  const pts = [];
  for (let i = 0; i < 18; i++) {
    const x = -50 + (i % 6) * 4, y = -50 + Math.floor(i / 6) * 4;
    pts.push([x, y, f(x, y)]);
  }
  pts.push([50, 50, f(50, 50)], [50, -50, f(50, -50)], [-50, 50, f(-50, 50)]);
  const r = fitPatch(pts, { uCount: 7, vCount: 7, stiffness: 1 });
  assert.equal(r.ok, true, r.reason);
  assert.ok(r.srf.ctrlNet.every((row) => row.every((cp) => cp.every(Number.isFinite))),
    'every control point must be finite — an undetermined one comes back NaN');
});

t('⛔ zero stiffness is refused, and the refusal says why', () => {
  const r = fitPatch(gridOn((x, y) => x / 10), { stiffness: 0 });
  assert.equal(r.ok, false);
  assert.equal(r.kind, PATCH_REFUSAL.BAD_REQUEST);
  assert.match(r.reason, /singular|undetermined/);
});

t('⛔ input that folds over its own plane is refused, not averaged', () => {
  // Two sheets at the same (u,v): a fit would return the average of both, a
  // surface through neither, and report half the gap as if it were accuracy.
  const pts = [];
  for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) {
    const x = i * 10 - 50, y = j * 10 - 50;
    pts.push([x, y, 0], [x, y, 60]);
  }
  const r = fitPatch(pts, { uCount: 6, vCount: 6 });
  assert.equal(r.ok, false);
  assert.equal(r.kind, PATCH_REFUSAL.FOLDED);
});

t('⛔ collinear input is refused — there is no second direction to span', () => {
  const pts = [];
  for (let i = 0; i <= 20; i++) pts.push([i * 5, 0, 0]);
  const r = fitPatch(pts, { uCount: 5, vCount: 5 });
  assert.equal(r.ok, false);
  assert.equal(r.kind, PATCH_REFUSAL.DEGENERATE_PLANE);
});

t('⛔ and an empty input is refused rather than producing an empty surface', () => {
  assert.equal(fitPatch([], {}).ok, false);
  assert.equal(fitPatch({ curves: [], points: [] }, {}).kind, PATCH_REFUSAL.NO_INPUT);
});

t('curves and loose points are sampled into one target list', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [100, 0, 0, 1]] };
  const got = samplePatchInputs({ curves: [line], points: [[50, 50, 0]] }, 10);
  assert.equal(got.length, 12, `1 loose point + 11 stations, got ${got.length}`);
  assert.ok(got.some((p) => Math.abs(p[0] - 50) < 1e-9 && Math.abs(p[1] - 50) < 1e-9));
});

t('a patch is fitted through four boundary curves that do NOT touch', () => {
  // The case BoundSrf and the n-sided patch both refuse: curves with gaps.
  const mk = (a, b) => ({ degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...a, 1], [...b, 1]] });
  const curves = [
    mk([-50, -50, 0], [50, -50, 0]),
    mk([-50, 50, 6], [50, 50, 6]),
    mk([-50, -40, 0], [-50, 40, 5]),
    mk([50, -40, 0], [50, 40, 5]),
  ];
  const r = fitPatch({ curves }, { uCount: 7, vCount: 7, stiffness: 0.05 });
  assert.equal(r.ok, true, r.reason);
  assert.ok(r.maxDeviation < 0.25, `should pass near the curves, worst ${r.maxDeviation}`);
});

t('the tolerance loop grows the net and reports every attempt', () => {
  const f = (x, y) => 10 * Math.sin(x / 25) * Math.cos(y / 25);
  const r = fitPatchToTolerance(gridOn(f, 20), { tolerance: 0.5, stiffness: 0.1, maxCount: 14 });
  assert.equal(r.ok, true, r.reason);
  assert.ok(Array.isArray(r.tried) && r.tried.length >= 1, 'every attempt must be reported');
  for (let i = 1; i < r.tried.length; i++) {
    assert.ok(r.tried[i].count > r.tried[i - 1].count, 'the net must GROW, not wander');
  }
});

t('the tolerance loop refuses to outrun its data', () => {
  // Nine samples cannot hold a 16x16 net; it must stop rather than solve a system
  // with more freedoms than constraints.
  const pts = [];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) pts.push([i * 40, j * 40, i * j]);
  const r = fitPatchToTolerance(pts, { tolerance: 1e-9, maxCount: 16, stiffness: 0.5 });
  assert.ok(!r.ok || r.uCount * r.vCount <= pts.length, `net ${r.uCount}x${r.vCount} against ${pts.length} samples`);
});

t('the reported deviation is an upper bound, never flattering', () => {
  const f = (x, y) => (x * x - y * y) / 300;
  const pts = gridOn(f, 14);
  const r = fitPatch(pts, { uCount: 7, vCount: 7, stiffness: 0.2 });
  assert.equal(r.ok, true);
  /* ⚠ THE TRUE DISTANCE IS APPROXIMATED ON A GRID, SO THE GRID'S OWN RESOLUTION
     IS THE ALLOWANCE. A grid search OVERESTIMATES the true minimum — it can only
     find the closest point it happened to sample — so comparing it to the
     reported residual without accounting for the spacing fails the test rather
     than the code. The spacing is measured here rather than assumed, so the
     allowance shrinks with the grid instead of being a magic number. */
  const G = 60;
  const grid = [];
  for (let i = 0; i <= G; i++) { const row = []; for (let j = 0; j <= G; j++) row.push(surfacePoint(r.srf, i / G, j / G)); grid.push(row); }
  let spacing = 0;
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    spacing = Math.max(spacing, Math.hypot(grid[i + 1][j][0] - grid[i][j][0], grid[i + 1][j][1] - grid[i][j][1], grid[i + 1][j][2] - grid[i][j][2]));
    spacing = Math.max(spacing, Math.hypot(grid[i][j + 1][0] - grid[i][j][0], grid[i][j + 1][1] - grid[i][j][1], grid[i][j + 1][2] - grid[i][j][2]));
  }
  let worst = 0;
  for (const q of pts) {
    let best = Infinity;
    for (let i = 0; i <= G; i++) for (let j = 0; j <= G; j++) {
      const s = grid[i][j];
      best = Math.min(best, Math.hypot(s[0] - q[0], s[1] - q[1], s[2] - q[2]));
    }
    worst = Math.max(worst, best);
  }
  assert.ok(worst <= r.maxDeviation + spacing,
    `true worst distance ${worst.toFixed(4)} must not exceed the reported ${r.maxDeviation.toFixed(4)} plus the grid's own ${spacing.toFixed(4)} spacing`);
});

console.log(`\n${passed}/${passed + failed} checks passed.`);
if (failed) process.exit(1);
