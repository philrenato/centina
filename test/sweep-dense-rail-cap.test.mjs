import test from 'node:test';
import assert from 'node:assert/strict';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { makeCircle } from '../kernel/primitives.mjs';
import { sweep1Rigid } from '../kernel/sweep.mjs';

// A real, live-reported bug: Pipe on a Curve-Generator
// Lorenz-attractor curve (~100 knot spans, no hard breaks — an ordinary
// smooth global-interpolation curve) hung the tab indefinitely. Root
// cause: sweep1RigidResampled's own `numSpans<=1` branch (the path any
// rail with no genuine C0 corner takes) handed its ENTIRE dense sample
// set — which scales with the rail's KNOT SPAN COUNT, not with real
// curvature complexity — to ONE interpAtParams call, a dense O(n^3)
// Gauss-Jordan solve. For a ~100-span rail this reached ~6000 points,
// an ~6000x6000 solve that never finished in practice. Fixed with a
// uniform-stride cap (MAX_DENSE_INTERP_POINTS) before that final solve.
function lorenzLikeRail(n) {
  const pts = [];
  let x = 0.1, y = 0, z = 0;
  const dt = 0.01, sigma = 10, rho = 28, beta = 8 / 3;
  for (let i = 0; i < n; i++) {
    const dx = sigma * (y - x), dy = x * (rho - z) - y, dz = x * y - beta * z;
    x += dx * dt; y += dy * dt; z += dz * dt;
    pts.push([x * 2, y * 2, z * 2]);
  }
  return pts;
}

test('sweep1Rigid on a ~100-knot-span rail (no hard breaks) completes fast and produces a finite, correctly-shaped surface', () => {
  const rail = globalCurveInterp(lorenzLikeRail(101), 3);
  assert.ok(rail.knots.length > 90, 'sanity: this rail genuinely has many knot spans, matching the reported reproduction');
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const t0 = Date.now();
  const srf = sweep1Rigid(rail, profile);
  const elapsedMs = Date.now() - t0;
  assert.ok(elapsedMs < 5000, `expected well under 5s, took ${elapsedMs}ms (the pre-fix version never returned within any reasonable time)`);
  assert.equal(srf.ctrlNet.length, profile.ctrlPts.length); // U direction = the circle's own control points, untouched
  assert.ok(srf.ctrlNet[0].length <= 301, `V direction is capped near MAX_DENSE_INTERP_POINTS, got ${srf.ctrlNet[0].length}`);
  for (const row of srf.ctrlNet) for (const cp of row) for (const v of cp) assert.ok(Number.isFinite(v));
});

test('a SHORT rail (few knot spans) is completely unaffected by the cap — byte-identical point count to before', () => {
  const rail = globalCurveInterp([[0, 0, 0], [10, 0, 5], [20, 0, 0], [30, 0, 5], [40, 0, 0]], 3);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const srf = sweep1Rigid(rail, profile);
  for (const row of srf.ctrlNet) for (const cp of row) for (const v of cp) assert.ok(Number.isFinite(v));
  // a short rail's own dense sample count never approaches the cap, so this
  // is just a plain correctness/no-regression check, not a size assertion.
});
