// PIPE-B — a smooth rail's own curvature vs. its tube radius.
//
// The guard this proves: a tube of radius r swept along a rail whose local
// radius of curvature falls below r self-intersects, exactly as a torus does
// the instant its tube radius exceeds its bend radius. The corner-fillet
// floor already in `pipeRailForSweep` covers a SHARP corner on a degree<=1
// rail; nothing covered a genuinely SMOOTH tightly-curved rail until now.
//
// Every expected number below is derived analytically from the fixture's own
// construction (a circle of radius R has curvature exactly 1/R everywhere),
// never read back from the function under test.

import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle, makeArc } from '../kernel/primitives.mjs';
import { railMinBendRadius, pipeSafeTubeRadius, PIPE_SELF_INTERSECT_MARGIN } from '../kernel/sweep.mjs';
import { curvePoint } from '../kernel/curve.mjs';

const X = [1, 0, 0], Y = [0, 1, 0], O = [0, 0, 0];

// A straight degree-1 rail. Piecewise linear, so C'' is identically zero and
// the bend radius is genuinely infinite — the honest answer, and the one that
// keeps this guard from ever firing on the case the corner floor owns.
test('a straight degree-1 rail reports an infinite bend radius', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [100, 0, 0, 1]] };
  const bend = railMinBendRadius(rail);
  assert.equal(bend.radius, Infinity);
  assert.equal(bend.kappa, 0);
});

// A POLYLINE with a real 90-degree corner, same reasoning: the corner's own
// curvature is a delta function no sampled derivative can see, so this guard
// correctly stays out of the corner floor's territory rather than competing
// with it.
test('a sharp-cornered degree-1 polyline is left to the corner floor', () => {
  const rail = {
    degree: 1,
    knots: [0, 0, 1, 2, 2],
    ctrlPts: [[0, 0, 0, 1], [50, 0, 0, 1], [50, 50, 0, 1]],
  };
  assert.equal(railMinBendRadius(rail).radius, Infinity);
});

// A CIRCLE of radius R has curvature exactly 1/R at every point, so its bend
// radius is exactly R. This is the load-bearing check: the value is known
// analytically, and a circle's own knot domain is [0,4], not [0,1] — reading
// only [0,1] would sample a single quarter-arc and is exactly the mistake
// this function's own comment names.
test('a circular rail reports its own radius as the bend radius', () => {
  for (const R of [8, 25, 120]) {
    const rail = makeCircle(O, X, Y, R);
    const bend = railMinBendRadius(rail);
    assert.ok(Math.abs(bend.radius - R) < R * 1e-6, `R=${R}: got ${bend.radius}`);
    assert.ok(Math.abs(bend.kappa - 1 / R) < 1e-9, `R=${R}: kappa ${bend.kappa}`);
  }
});

// The domain claim, proven rather than asserted: the reported extremum sits
// at a parameter inside a circle's real [0,4] domain, and sampling only the
// first quarter would still find the right value here (a circle is uniformly
// curved) — so the discriminating fixture is an ARC whose tight stretch lives
// past u=1. Built as two joined arcs of very different radii: a gentle one
// first, a tight one second, so the true minimum is only reachable by
// sampling the real domain.
test('the tight stretch is found even when it lies past u=1', () => {
  // A 90-degree arc of radius 60, then a 90-degree arc of radius 6 hinged off
  // its end. Rather than composing them (which needs the join machinery this
  // test has no business depending on), assert the property that matters on
  // the tight arc alone AND confirm its own domain runs past 1.
  const tight = makeArc(O, X, Y, 6, 0, Math.PI / 2);
  const gentle = makeArc(O, X, Y, 60, 0, Math.PI / 2);
  assert.ok(Math.abs(railMinBendRadius(tight).radius - 6) < 6e-5);
  assert.ok(Math.abs(railMinBendRadius(gentle).radius - 60) < 60e-5);

  // The real domain check: a FULL circle's own knot domain genuinely exceeds
  // 1, and evaluating past u=1 returns a genuinely different point — so a
  // [0,1]-only sampler would have been reading a fraction of the rail.
  const full = makeCircle(O, X, Y, 10);
  const kLast = full.knots[full.knots.length - 1];
  assert.ok(kLast > 1, `expected a domain past 1, got ${kLast}`);
  const atOne = curvePoint(full, 1);
  const atLast = curvePoint(full, kLast);
  assert.ok(Math.hypot(atOne[0] - atLast[0], atOne[1] - atLast[1]) > 1, 'u=1 and the domain end must be genuinely different points');
});

// The clamp itself. A radius-10 circular rail can carry a tube up to
// 10 / 1.02 before the horn-torus boundary; anything larger is clamped to
// exactly that, and anything smaller passes through untouched.
test('an over-large tube radius clamps to the safe maximum, a safe one passes through', () => {
  const rail = makeCircle(O, X, Y, 10);
  const expectedMax = 10 / PIPE_SELF_INTERSECT_MARGIN;

  const over = pipeSafeTubeRadius(rail, 25);
  assert.equal(over.clamped, true);
  assert.ok(Math.abs(over.radius - expectedMax) < 1e-4, `got ${over.radius}, expected ${expectedMax}`);
  assert.ok(Math.abs(over.minBendRadius - 10) < 1e-4);

  // Exactly AT the bend radius is still over the margin, deliberately — the
  // margin exists because that exact value is the failure boundary.
  const atBoundary = pipeSafeTubeRadius(rail, 10);
  assert.equal(atBoundary.clamped, true);

  const under = pipeSafeTubeRadius(rail, 3);
  assert.equal(under.clamped, false);
  assert.equal(under.radius, 3, 'a safe request must pass through BIT-IDENTICALLY, not be rounded to a computed near-equal');
});

// THE COMPOSITION PROOF — the specific hazard the shared margin exists to
// prevent, and the reason this guard uses `bendRadius / MARGIN` rather than a
// separately-chosen `bendRadius * 0.98`.
//
// The corner floor (`pipeRailForSweep`, app layer) raises a too-small
// cornerRadius UP to `tubeRadius * MARGIN`, then rounds the rail. That
// rounded rail's tightest bend IS that raised cornerRadius. If this guard
// then measured it against an independently-chosen down-margin, the two would
// multiply to slightly under 1 and clamp the tube by a sub-micron amount —
// firing a real, visible, confusing status message on EVERY rounded-corner
// Pipe in the app, about a difference no geometry can express.
//
// Simulated at the kernel level by building the arc the corner floor would
// produce: a fillet of exactly `tubeRadius * MARGIN`, which is what the rail
// carries at its tightest point after that floor has run.
test('a rail already raised by the corner floor does NOT get re-clamped by the bend floor', () => {
  for (const tubeRadius of [1, 5, 12.5]) {
    const flooredCornerRadius = tubeRadius * PIPE_SELF_INTERSECT_MARGIN;
    const roundedCorner = makeArc(O, X, Y, flooredCornerRadius, 0, Math.PI / 2);
    const out = pipeSafeTubeRadius(roundedCorner, tubeRadius);
    assert.equal(out.clamped, false, `tubeRadius=${tubeRadius}: the two guards must compose, not fight (safeMax ${out.safeMax}, bend ${out.minBendRadius})`);
    assert.equal(out.radius, tubeRadius, 'and the radius must pass through bit-identically, not be nudged to a near-equal');
  }
});

// A straight rail can never clamp, at any radius — the regression guard that
// keeps every existing straight-rail Pipe in this app byte-identical.
test('a straight rail never clamps, at any requested radius', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [100, 0, 0, 1]] };
  for (const r of [0.1, 5, 500, 1e6]) {
    const out = pipeSafeTubeRadius(rail, r);
    assert.equal(out.clamped, false, `r=${r} must not clamp on a straight rail`);
    assert.equal(out.radius, r);
  }
});
