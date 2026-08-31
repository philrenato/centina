// COMPLETE SURFACE-SURFACE INTERSECTION — the boolean pipeline's
// Phase 0 fixtures and Phase 1 gate.
//
// Every truth here is closed-form geometry written down independently of
// what the kernel returns: a cylinder of radius R about an axis is
// x^2+y^2=R^2, and where two of them cross is solvable by hand. Nothing is
// compared against "what it did last time".
//
// TWO THINGS THESE FIXTURES FOUND, in the order they surfaced:
//
// 1. THE MARCH BUDGET WAS SCALE-BLIND. `stepLen = 0.25` with
//    `maxSteps = 400` is 100 units of arc length total, at any scale, so the
//    simplest possible case — one plane cutting one cylinder into one circle
//    of circumference 188.5 — refused outright. The marching math was never
//    wrong: the same pair at a larger step closed with radius 30.0000. This
//    had to be fixed before completeness could even be MEASURED, which is
//    why it lands ahead of the seeding work otherwise sequenced first.
//
// 2. SEEDING FOUND ONE COMPONENT. Boundary seeding plus interior local
//    minima now find every component in these fixtures. That is complete BY
//    CONSTRUCTION for branches reaching a boundary, and an honest best
//    effort for interior loops — see `loopSearchProven` below.

import test from 'node:test';
import assert from 'node:assert/strict';
import { revolve, extrude, makeLine } from '../kernel/primitives.mjs';
import { intersectSurfaces, intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { curveSurfaceIntersections } from '../kernel/curvesurface.mjs';
import { extractBorderCurves } from '../kernel/isocurve.mjs';

// A cylinder of radius R about a world axis, spanning [t0,t1] along it.
// Built by revolving a straight line offset R from that axis, so its own
// implicit equation is exact and known.
function cylinder(R, axis, t0, t1) {
  const perp = axis === 'z' ? [1, 0, 0] : axis === 'x' ? [0, 1, 0] : [0, 0, 1];
  const dir = axis === 'z' ? [0, 0, 1] : axis === 'x' ? [1, 0, 0] : [0, 1, 0];
  const p = (t) => [perp[0] * R + dir[0] * t, perp[1] * R + dir[1] * t, perp[2] * R + dir[2] * t];
  return revolve(makeLine(p(t0), p(t1)), [0, 0, 0], dir, 0, 2 * Math.PI);
}
function planeAtZ(half, z) {
  const s = extrude(makeLine([-half, -half, 0], [half, -half, 0]), [0, 1, 0], 2 * half);
  for (const row of s.ctrlNet) for (const cp of row) cp[2] += z;
  return s;
}

// ------------------------------------------------ THE SCALE-BLIND BUDGET

test('a plane cutting a cylinder closes into one exact circle — the case the old fixed step refused', () => {
  const r = intersectSurfaces(planeAtZ(100, 10), cylinder(30, 'z', -50, 50));
  assert.equal(r.ok, true, r.ok ? '' : r.reason);
  assert.equal(r.closed, true, 'the intersection of a plane and a full cylinder is a closed circle');
  // The truth: radius exactly 30, at exactly z=10.
  for (const s of r.samples) {
    assert.ok(Math.abs(Math.hypot(s.point[0], s.point[1]) - 30) < 1e-4, `radius ${Math.hypot(s.point[0], s.point[1])}`);
    assert.ok(Math.abs(s.point[2] - 10) < 1e-6, `z ${s.point[2]}`);
  }
});

test('an explicit stepLen is honoured, absent a corrector failure at it', () => {
  // THE CONTRACT THIS ENCODES CHANGED, so it is rewritten to the surviving
  // invariant rather than left asserting something no longer true. It used to
  // read "honoured verbatim... existing callers unchanged". A step where the
  // corrector genuinely fails to converge is now retried shorter, so the
  // guarantee is narrower and exact: the turn-based adaptation stays disabled,
  // the supplied value is genuinely used rather than overridden by the
  // size-derived default, the run stays deterministic — and the shortening,
  // where it happens at all, applies to that one step only and does not
  // ratchet the rest of the march down.
  //
  // This fixture never triggers a retry, so it tests the ordinary path; the
  // retry path has its own test below.
  const a = intersectSurfaces(planeAtZ(100, 10), cylinder(30, 'z', -50, 50), { stepLen: 2, maxSteps: 400 });
  const b = intersectSurfaces(planeAtZ(100, 10), cylinder(30, 'z', -50, 50), { stepLen: 2, maxSteps: 400 });
  const c = intersectSurfaces(planeAtZ(100, 10), cylinder(30, 'z', -50, 50), { stepLen: 4, maxSteps: 400 });
  assert.equal(a.ok && b.ok && c.ok, true);
  assert.equal(a.samples.length, b.samples.length);
  assert.notEqual(a.samples.length, c.samples.length);
});

test('an intersection longer than the arc budget refuses by naming the budget, not a step count', () => {
  // A deliberately tiny budget on a pair that genuinely intersects: the
  // refusal must say what actually ran out.
  const r = intersectSurfaces(planeAtZ(100, 10), cylinder(30, 'z', -50, 50), { arcBudget: 5 });
  assert.equal(r.ok, false);
  assert.match(r.reason, /arc length/);
});

// -------------------------------------------- CURVE / SURFACE INTERSECTION

test('a line through a cylinder finds BOTH crossings, exactly', () => {
  const hits = curveSurfaceIntersections(makeLine([-100, 0, 10], [100, 0, 10]), cylinder(30, 'z', -50, 50));
  assert.equal(hits.length, 2, 'a line through a cylinder crosses its wall twice');
  const xs = hits.map((h) => h.point[0]).sort((a, b) => a - b);
  assert.ok(Math.abs(xs[0] + 30) < 1e-6, `first crossing at x=${xs[0]}, truth -30`);
  assert.ok(Math.abs(xs[1] - 30) < 1e-6, `second crossing at x=${xs[1]}, truth +30`);
  for (const h of hits) assert.ok(h.residual < 1e-6);
});

test('a line that misses reports nothing, and one that starts inside reports one crossing', () => {
  assert.equal(curveSurfaceIntersections(makeLine([-100, 200, 10], [100, 200, 10]), cylinder(30, 'z', -50, 50)).length, 0);
  assert.equal(curveSurfaceIntersections(makeLine([0, 0, 10], [100, 0, 10]), cylinder(30, 'z', -50, 50)).length, 1);
});

test('a closed direction\'s seam is not treated as an edge', () => {
  // The regression: a full-revolve cylinder's closest point to a line lands
  // ON the v seam either side of the far crossing (v=0 and v=vMax being the
  // same place). Treating that as a patch boundary discarded the sign there
  // and lost the crossing entirely — one hit instead of two.
  const cyl = cylinder(30, 'z', -50, 50);
  const hits = curveSurfaceIntersections(makeLine([-100, 0, 0], [100, 0, 0]), cyl);
  assert.equal(hits.length, 2);
});

test('a patch border against the opposing surface gives the boundary seeds', () => {
  // This is the seeding step itself: the cylinder's two borders each cross
  // the plane at x = +/-30, which are exactly the two components.
  const plane = planeAtZ(100, 0);
  const borders = extractBorderCurves(cylinder(30, 'y', -50, 50));
  assert.equal(borders.length, 2, 'closed in its sweep direction, so only the two ends are borders');
  for (const b of borders) {
    const hits = curveSurfaceIntersections(b.crv, plane);
    assert.equal(hits.length, 2, `border ${b.edge} crosses the plane twice`);
    const xs = hits.map((h) => h.point[0]).sort((a, b2) => a - b2);
    assert.ok(Math.abs(xs[0] + 30) < 1e-6 && Math.abs(xs[1] - 30) < 1e-6);
  }
});

// -------------------------------------------------- THE COMPLETENESS GATE

test('FIXTURE B: two open components, both reaching a boundary', () => {
  // plane z=0 X cylinder R=30 about Y. Truth: the two straight lines
  // x = +/-30, z = 0, running the cylinder's full y extent.
  const r = intersectSurfacesComplete(planeAtZ(100, 0), cylinder(30, 'y', -50, 50));
  assert.equal(r.ok, true);
  assert.equal(r.components.length, 2, 'TWO components — one seed found only one');
  for (const c of r.components) {
    assert.equal(c.closed, false);
    for (const s of c.samples) {
      assert.ok(Math.abs(Math.abs(s.point[0]) - 30) < 1e-4, `|x| ${Math.abs(s.point[0])}, truth 30`);
      assert.ok(Math.abs(s.point[2]) < 1e-4, `z ${s.point[2]}, truth 0`);
    }
  }
  const meanX = r.components.map((c) => c.samples.reduce((a, s) => a + s.point[0], 0) / c.samples.length);
  assert.ok(Math.min(...meanX) < 0 && Math.max(...meanX) > 0, 'the two components are on opposite sides, not the same one twice');
});

test('FIXTURE A: two interior loops, neither touching any boundary', () => {
  // cyl R1=30 about Z X cyl R2=20 about X, both long enough that neither
  // loop is cut. Truth: two closed loops satisfying BOTH implicit equations.
  const r = intersectSurfacesComplete(cylinder(30, 'z', -50, 50), cylinder(20, 'x', -200, 200));
  assert.equal(r.ok, true);
  assert.equal(r.components.length, 2, 'TWO closed loops');
  for (const c of r.components) {
    assert.equal(c.closed, true);
    for (const s of c.samples) {
      const [x, y, z] = s.point;
      assert.ok(Math.abs(Math.hypot(x, y) - 30) < 1e-3, `on cylinder 1: ${Math.hypot(x, y)}`);
      assert.ok(Math.abs(Math.hypot(y, z) - 20) < 1e-3, `on cylinder 2: ${Math.hypot(y, z)}`);
    }
  }
  const meanX = r.components.map((c) => c.samples.reduce((a, s) => a + s.point[0], 0) / c.samples.length);
  assert.ok(Math.min(...meanX) < 0 && Math.max(...meanX) > 0, 'one loop each side of the axis');
});

test('FIXTURE C: a loop and open branches in the same patch pair', () => {
  // The same pair, but srf2 clipped at x=-25. On the x<0 loop,
  // x = -sqrt(900 - 400 sin^2 phi), so x >= -25 needs |sin phi| >= 0.829 —
  // TWO disjoint phi ranges, so that loop is cut into TWO open arcs while
  // the x>0 loop (x in [22.36,30]) is untouched and stays closed. Three
  // components, and the count is derived here rather than observed.
  const r = intersectSurfacesComplete(cylinder(30, 'z', -50, 50), cylinder(20, 'x', -25, 200));
  assert.equal(r.ok, true);
  assert.equal(r.components.length, 3);
  assert.equal(r.components.filter((c) => c.closed).length, 1, 'exactly one survives closed');
  assert.equal(r.components.filter((c) => !c.closed).length, 2, 'the cut loop becomes two arcs, not one');
  const closed = r.components.find((c) => c.closed);
  for (const s of closed.samples) assert.ok(s.point[0] > 0, 'the surviving loop is the x>0 one');
  for (const c of r.components.filter((x) => !x.closed)) {
    for (const s of c.samples) assert.ok(s.point[0] >= -25 - 1e-3, 'the open arcs are clipped at x=-25');
  }
});

test('FIXTURE E: swapping the arguments returns the same component set', () => {
  const s1 = planeAtZ(100, 0), s2 = cylinder(30, 'y', -50, 50);
  const ab = intersectSurfacesComplete(s1, s2);
  const ba = intersectSurfacesComplete(s2, s1);
  assert.equal(ab.ok && ba.ok, true);
  assert.equal(ab.components.length, ba.components.length, 'order dependence detector');
  const key = (r) => r.components.map((c) => c.samples.reduce((a, s) => a + s.point[0], 0) / c.samples.length)
    .map((x) => Math.round(x * 100) / 100).sort((a, b) => a - b).join(',');
  assert.equal(key(ab), key(ba), 'the same curves, not merely the same count');
});

test('completeness is reported honestly: interior loops are searched, not proven', () => {
  const r = intersectSurfacesComplete(planeAtZ(100, 10), cylinder(30, 'z', -50, 50));
  assert.equal(r.ok, true);
  assert.equal(r.loopSearchProven, false,
    'boundary seeding is exhaustive by construction; the interior-loop half needs the normal-cone proof, which is NOT built — a caller must not be able to assume otherwise');
});

test('two surfaces that do not meet refuse, rather than inventing a component', () => {
  const r = intersectSurfacesComplete(planeAtZ(100, 400), cylinder(30, 'z', -50, 50));
  assert.equal(r.ok, false);
  assert.match(r.reason, /no intersection/);
});

// ------------------------------------- THE SEAM, THE MIDPOINT, THE RETRY

// EVERY MARCHED SAMPLE MUST SATISFY THE EQUATION IT WAS SOLVED FOR.
//
// The corrector's whole job is |S1(u1,v1) - S2(u2,v2)| = 0, and it reports a
// residual it is happy with. But nothing downstream re-checked that the
// parameters actually STORED on the sample still satisfy it — and they did
// not: a step crossing a CLOSED direction's seam converged with the
// parameter outside the domain (where a clamped B-spline extrapolates along
// its first span's polynomial rather than continuing around the seam), and
// the caller then wrapped it back in, silently moving the sample to a
// different real point. Measured before the fix: exactly one sample per
// component, always the seam-crossing one, sat ~1e-2 off BOTH surfaces while
// every neighbour was at machine precision. Asserting the residual at the
// stored parameters is what catches that class outright.
function worstSampleResidual(srf1, srf2, components) {
  let worst = 0;
  for (const c of components) {
    for (const s of c.samples) {
      const p = surfacePoint(srf1, s.u1, s.v1);
      const q = surfacePoint(srf2, s.u2, s.v2);
      worst = Math.max(worst, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    }
  }
  return worst;
}

test('every marched sample still satisfies S1=S2 at its own STORED parameters, seam crossings included', () => {
  // A cylinder is closed in its sweep direction, so any loop around it
  // crosses that seam — this is the ordinary case, not a contrived one.
  const pairs = [
    [cylinder(30, 'z', -50, 50), planeAtZ(100, 10)],
    [planeAtZ(100, 10), cylinder(30, 'z', -50, 50)],
    [cylinder(30, 'z', -50, 50), cylinder(18, 'x', -60, 60)],
  ];
  for (const [s1, s2] of pairs) {
    const r = intersectSurfacesComplete(s1, s2);
    assert.equal(r.ok, true, r.ok ? '' : r.reason);
    const worst = worstSampleResidual(s1, s2, r.components);
    assert.ok(worst < 1e-6, `a stored sample violates S1=S2 by ${worst.toExponential(3)}`);
  }
});

test('an emitted sample point IS the midpoint of both surfaces\' own evaluations', () => {
  // THE DIRECT TEST, and the reason it has to be direct. The obvious version
  // — measure the emitted point against both surfaces and require it close to
  // each — is VACUOUS on any well-conditioned pair: the corrector drives the
  // residual to ~1e-9 there, so srf1's own evaluation already sits far inside
  // any tolerance loose enough to be worth asserting, and reverting to
  // srf1-only passes it unchanged. The decision under test is which POINT is
  // emitted, so the assertion has to be on that decision, at machine
  // precision, where no amount of corrector convergence can hide it.
  const cyl = cylinder(30, 'z', -50, 50);
  const pln = planeAtZ(100, 10);
  const r = intersectSurfacesComplete(cyl, pln);
  assert.equal(r.ok, true, r.ok ? '' : r.reason);
  let checked = 0;
  for (const c of r.components) {
    for (const s of c.samples) {
      const a = surfacePoint(cyl, s.u1, s.v1);
      const b = surfacePoint(pln, s.u2, s.v2);
      for (let d = 0; d < 3; d++) {
        assert.ok(Math.abs(s.point[d] - (a[d] + b[d]) / 2) < 1e-12,
          `sample point is not the midpoint of the two evaluations on axis ${d}`);
      }
      checked++;
    }
  }
  assert.ok(checked > 0, 'no samples were checked, so this test proved nothing');
});

test('an emitted sample point is on BOTH surfaces, not exact on the first and off the second', () => {
  // The point emitted per sample used to be srf1's own evaluation, which is
  // exact there by construction and off srf2 by the full corrector residual
  // — so which surface carried the error depended entirely on argument
  // order. The midpoint splits it, and swapping the arguments must not move
  // the answer. Measured against BOTH surfaces so a one-sided result fails.
  const cyl = cylinder(30, 'z', -50, 50);
  const pln = planeAtZ(100, 10);
  const forward = intersectSurfacesComplete(cyl, pln);
  const swapped = intersectSurfacesComplete(pln, cyl);
  assert.equal(forward.ok && swapped.ok, true);
  for (const [label, r] of [['forward', forward], ['swapped', swapped]]) {
    let worstRadial = 0, worstZ = 0;
    for (const c of r.components) {
      for (const s of c.samples) {
        worstRadial = Math.max(worstRadial, Math.abs(Math.hypot(s.point[0], s.point[1]) - 30));
        worstZ = Math.max(worstZ, Math.abs(s.point[2] - 10));
      }
    }
    // Both are analytic truths of the fixture itself, not of the marcher:
    // the loop lies on the cylinder (radius 30) AND on the plane (z=10).
    assert.ok(worstRadial < 1e-6, `${label}: off the cylinder by ${worstRadial.toExponential(3)}`);
    assert.ok(worstZ < 1e-6, `${label}: off the plane by ${worstZ.toExponential(3)}`);
  }
});

test('a correction that fails at the offered step is retried shorter before the march refuses', () => {
  // A predictor throwing its guess a full step along the tangent can land
  // outside Newton's basin where the curve turns hard. Refusing there loses
  // a genuinely findable component. A deliberately oversized explicit step
  // on a tightly-curved pair is the reachable version of that: it must
  // still return a real curve rather than an honest-but-needless refusal.
  const big = intersectSurfaces(cylinder(30, 'z', -50, 50), cylinder(18, 'x', -60, 60), { stepLen: 40, maxSteps: 400 });
  assert.equal(big.ok, true, big.ok ? '' : `oversized step refused: ${big.reason}`);
  let worst = 0;
  for (const s of big.samples) worst = Math.max(worst, Math.abs(Math.hypot(s.point[0], s.point[1]) - 30));
  assert.ok(worst < 1e-6, `retried march left samples ${worst.toExponential(3)} off the cylinder`);
});
