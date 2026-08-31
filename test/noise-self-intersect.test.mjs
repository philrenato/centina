import test from 'node:test';
import assert from 'node:assert/strict';
import { maxSafeDisplacementScale } from '../kernel/noise.mjs';

// A minimal 2x1 control net (one U-direction edge, no V edges at all —
// nv=1 so the j+1<nv branch never fires) — the smallest fixture that can
// exercise a single adjacent-edge crossing check in isolation.
function twoPointNet(dx = 1) {
  return [[[0, 0, 0, 1]], [[dx, 0, 0, 1]]];
}

test('maxSafeDisplacementScale: an untouched (all-zero) displacement field returns Infinity — nothing to constrain', () => {
  const net = twoPointNet();
  const field = [[[0, 0, 0]], [[0, 0, 0]]];
  assert.equal(maxSafeDisplacementScale(net, field), Infinity);
});

test('maxSafeDisplacementScale: a UNIFORM (identical everywhere) displacement field returns Infinity — a pure translate can never fold an edge', () => {
  const net = twoPointNet();
  const field = [[[3, -2, 5]], [[3, -2, 5]]]; // same vector at every control point
  assert.equal(maxSafeDisplacementScale(net, field), Infinity);
});

test('maxSafeDisplacementScale: a single hand-derivable shrinking edge is clamped to the exact predicted scale', () => {
  const net = twoPointNet(1); // Pa=(0,0,0), Pb=(1,0,0), e=(1,0,0), e2=1
  // Pb moves toward Pa at unit scale (va=0, vb=(-1,0,0)): dv=(-1,0,0),
  // a = dot(dv,e) = -1 -> crit = e2/|a| = 1 -> safe = 1 * 0.98 = 0.98.
  const field = [[[0, 0, 0]], [[-1, 0, 0]]];
  const safe = maxSafeDisplacementScale(net, field);
  assert.ok(Math.abs(safe - 0.98) < 1e-12, `expected exactly 0.98, got ${safe}`);
});

test('maxSafeDisplacementScale: an edge that GROWS (a >= 0) is never constrained, even under a large displacement', () => {
  const net = twoPointNet(1);
  // Pb moves AWAY from Pa: dv=(+2,0,0), a = dot(dv,e) = +2 >= 0 -> no crossing possible.
  const field = [[[0, 0, 0]], [[2, 0, 0]]];
  assert.equal(maxSafeDisplacementScale(net, field), Infinity);
});

test('maxSafeDisplacementScale: a 3-point row with one shrinking edge and one growing edge is bound by the shrinking one alone', () => {
  // Points at x=0,1,2 (nu=3, nv=1). Middle point pulls toward BOTH neighbours
  // at unit scale: va=0 (i=0), vb=(-2,0,0) (i=1), vc=0 (i=2).
  //   edge(0,1): Pa=(0,0,0) Pb=(1,0,0), e2=1, dv=vb-va=(-2,0,0), a=-2 -> crit=1/2=0.5
  //   edge(1,2): Pa=(1,0,0) Pb=(2,0,0), e2=1, dv=vc-vb=(2,0,0),  a=+2 -> no constraint
  // Overall min crit = 0.5 -> safe = 0.5 * 0.98 = 0.49.
  const net = [[[0, 0, 0, 1]], [[1, 0, 0, 1]], [[2, 0, 0, 1]]];
  const field = [[[0, 0, 0]], [[-2, 0, 0]], [[0, 0, 0]]];
  const safe = maxSafeDisplacementScale(net, field);
  assert.ok(Math.abs(safe - 0.49) < 1e-12, `expected exactly 0.49, got ${safe}`);
});

test('maxSafeDisplacementScale: a genuine 2D grid considers BOTH U and V adjacent edges, not just one direction', () => {
  // A 2x2 unit-square grid; only the (0,0)-(0,1) V-edge is made to shrink,
  // every other edge (U-direction and the other V-edge) stays untouched.
  const net = [
    [[0, 0, 0, 1], [0, 1, 0, 1]],
    [[1, 0, 0, 1], [1, 1, 0, 1]],
  ];
  const zero = [0, 0, 0];
  const field = [
    [zero, [0, -1, 0]], // (0,0)->(0,1) edge: e=(0,1,0), e2=1, dv=(0,-1,0), a=-1 -> crit=1
    [zero, zero],
  ];
  const safe = maxSafeDisplacementScale(net, field);
  assert.ok(Math.abs(safe - 0.98) < 1e-12, `expected exactly 0.98 (only the one V-edge constrains it), got ${safe}`);
});

test('maxSafeDisplacementScale: a genuinely coincident control-point edge (e2~0, e.g. a pole row) contributes no constraint', () => {
  const net = [[[5, 5, 5, 1]], [[5, 5, 5, 1]]]; // both points identical — a pole edge
  const field = [[[0, 0, 0]], [[-100, 0, 0]]]; // even a huge displacement can't "cross" a zero-length edge
  assert.equal(maxSafeDisplacementScale(net, field), Infinity);
});
