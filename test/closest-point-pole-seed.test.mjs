import test from 'node:test';
import assert from 'node:assert/strict';
import { closestPointOnSurface, surfacePoint } from '../kernel/surface.mjs';
import { refineClosestPointOnSurface } from '../kernel/trim.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { revolve, makeArc, makeCircle, makeLine, extrude } from '../kernel/primitives.mjs';

// A REVOLVE WHOSE PROFILE STARTS ON THE AXIS HAS A POLE, and a pole is where a
// closest-point search can be seeded into a place it can never leave: the
// partials collapse there, so the Gauss-Newton step has a singular Jacobian and
// the loop's own degenerate-matrix guard returns the seed untouched. On a
// strongly shaped profile the pole is also, genuinely, the nearest sample of a
// coarse parameter grid — the true minimum sits inside the first cell beside it,
// where the surface expands fastest per unit of u — so the search picks the one
// seed that cannot move and reports a point over a millimeter away from a target
// that is ON the surface.
//
// These fixtures are the ones that produced that failure in a real boolean: two
// interpolated-profile revolves whose intersection curve was refused with "sample
// 0 is 1.496296 away" while lying 3e-14 from the surface it was said to miss.
const blobSurface = (profile) =>
  revolve(globalCurveInterp(profile, 3), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);

const WAVY_A = blobSurface([[0, 0, 0], [12, 0, 6], [8, 0, 14], [15, 0, 24], [6, 0, 34], [0, 0, 40]]);
const WAVY_B_AT_ORIGIN = blobSurface([[0, 0, 0], [14, 0, 8], [9, 0, 18], [16, 0, 30], [0, 0, 38]]);
const WAVY_B = {
  ...WAVY_B_AT_ORIGIN,
  ctrlNet: WAVY_B_AT_ORIGIN.ctrlNet.map((row) => row.map(([x, y, z, w]) => [x + 10, y + 4, z + 6, w])),
};
const CYLINDER = extrude(makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 12), [0, 0, 1], 40);
const SPHERE = revolve(makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 15, -Math.PI / 2, Math.PI), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
const TORUS = revolve(makeCircle([20, 0, 0], [1, 0, 0], [0, 0, 1], 6), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
const DISC = revolve(makeLine([0, 0, 0], [10, 0, 0]), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);

// The honest answer, at a resolution no shipped call would pay for: a dense grid
// followed by the same refinement. Slow, and only ever used as the thing the
// real search is scored against.
function referenceDistance(srf, target, n = 240) {
  let best = null;
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= n; j++) {
      const p = surfacePoint(srf, i / n, j / n);
      const d = (p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2 + (p[2] - target[2]) ** 2;
      if (!best || d < best.d) best = { u: i / n, v: j / n, d };
    }
  }
  return refineClosestPointOnSurface(srf, target, best.u, best.v).distance;
}

test('closestPointOnSurface: a point on a pole-adjacent revolve is found, not missed by 1.5mm', () => {
  const target = surfacePoint(WAVY_B, 0.008859, 0.667965);
  const found = closestPointOnSurface(WAVY_B, target);
  assert.ok(found.distance < 1e-6,
    `the target is ON the surface, so the search should reach it; got ${found.distance}`);
  const back = surfacePoint(WAVY_B, found.u, found.v);
  const gap = Math.hypot(back[0] - target[0], back[1] - target[1], back[2] - target[2]);
  assert.ok(gap < 1e-6, `the returned parameters should evaluate back to the target; got ${gap}`);
});

test('closestPointOnSurface: the pole itself still answers, and answers correctly', () => {
  // Directly above the disc's center, the closest point IS the pole. A search
  // that refuses to seed there must still be able to RETURN there.
  const found = closestPointOnSurface(DISC, [0, 0, 7]);
  assert.ok(Math.abs(found.distance - 7) < 1e-6, `expected 7 above the center, got ${found.distance}`);
});

test('closestPointOnSurface: matches a dense reference across pole-carrying surfaces', () => {
  // Deterministic targets, a mix of on-surface, just-off and far-off, so the
  // sweep covers both the search's accuracy and its refusals.
  let seed = 20260811;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (const [name, srf] of [['wavy A', WAVY_A], ['wavy B', WAVY_B], ['sphere', SPHERE], ['disc', DISC]]) {
    let worst = 0, worstAt = null;
    for (let t = 0; t < 24; t++) {
      const on = surfacePoint(srf, rnd(), rnd());
      const target = t % 3 === 0 ? on : on.map((c) => c + (rnd() - 0.5) * (t % 3 === 1 ? 0.4 : 8));
      const excess = closestPointOnSurface(srf, target).distance - referenceDistance(srf, target);
      if (excess > worst) { worst = excess; worstAt = target; }
    }
    assert.ok(worst < 1e-3, `${name}: worst excess over the dense reference was ${worst} at ${worstAt}`);
  }
});

test('closestPointOnSurface: surfaces without a pole are untouched', () => {
  // A cylinder and a torus have no degenerate row at all, so nothing here may
  // change what they answer — the same values, to the bit.
  for (const [name, srf] of [['cylinder', CYLINDER], ['torus', TORUS]]) {
    for (const target of [[13, 0, 20], [0, 30, 15], [5, 5, -8], [26, 0, 0], [0, 0, 0]]) {
      const found = closestPointOnSurface(srf, target);
      const ref = referenceDistance(srf, target);
      assert.ok(found.distance - ref < 1e-6, `${name} at ${target}: got ${found.distance}, reference ${ref}`);
    }
  }
});
