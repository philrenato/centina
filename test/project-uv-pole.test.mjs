// PROJECTING A CHAIN THAT WALKS OVER A POLE.
//
// `projectPointsToSurfaceUV` refines each sample from the PREVIOUS one. That
// warm start is what makes a long chain cheap and what keeps it on one branch
// of a closed surface — and it is a trap at a pole, where the surface's
// partials collapse: Gauss-Newton has no direction to move in, so every later
// sample stays pinned where the pole left it.
//
// This is not a hypothetical surface. A cylinder CAP in this app is a revolved
// disc (`revolvedCapPanel` — a line from center to rim, revolved), so its u=0
// parametric edge IS its center, collapsed to a single point. Any loop walking
// that face's own parametric boundary crosses the pole.
//
// ⚠ THE FAILURE LOOKED LIKE GEOMETRY AND WAS NOT. It surfaced as a boolean that
// could not be exported as a joined solid, refusing with "curve is not on the
// surface — sample N is 12.000000 away" on a radius-12 cap: exactly the
// center-to-rim distance. Every one of those points projects EXACTLY when
// seeded cold, so nothing was off the surface — the seed was.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { revolve, makeLine } from '../kernel/primitives.mjs';
import { projectPointsToSurfaceUV } from '../kernel/trim.mjs';
import { closestPointOnSurface, surfacePoint } from '../kernel/surface.mjs';

const R = 12;
const CENTRE = [0, 0, 0];
// Built the way the app builds a cylinder cap, not an idealisation of one.
const cap = revolve(makeLine(CENTRE, [R, 0, 0]), CENTRE, [0, 0, 1], 0, 2 * Math.PI);
const onRim = (a) => [R * Math.cos(a), R * Math.sin(a), 0];

test('THE FIXTURE IS THE REAL SHAPE: a cap is a revolved disc with a pole at its center', () => {
  assert.equal(cap.degU, 1);
  assert.equal(cap.degV, 2, 'the revolved direction is the rational quadratic a circle needs');
  // u = 0 is the center for EVERY v — that is what makes it a pole, and what
  // makes the partials collapse there.
  for (const v of [0, 0.7, 1.9, 3.3]) {
    const p = surfacePoint(cap, cap.knotsU[0], cap.knotsV[0] + v);
    assert.ok(Math.hypot(p[0], p[1], p[2]) < 1e-9, `u=0 must be the center at v=${v}, got ${JSON.stringify(p)}`);
  }
});

test('EVERY POINT PROJECTS EXACTLY WHEN SEEDED COLD — so nothing here is off the surface', () => {
  // The control that makes the next test a statement about the SEARCH rather
  // than about the geometry. If this ever fails, the fixture is wrong and the
  // warm-start test below is measuring the wrong thing.
  for (const p of [onRim(0), CENTRE, onRim(1), onRim(2)]) {
    const r = closestPointOnSurface(cap, p);
    const back = surfacePoint(cap, r.u, r.v);
    assert.ok(Math.hypot(back[0] - p[0], back[1] - p[1], back[2] - p[2]) < 1e-9,
      `${JSON.stringify(p)} must project exactly, got ${r.distance}`);
  }
});

test('⭐ A CHAIN THAT WALKS OVER THE POLE STILL PROJECTS — the warm start re-seeds instead of refusing', () => {
  const res = projectPointsToSurfaceUV([onRim(0), CENTRE, onRim(1), onRim(2)], cap, { tolerance: 0.001 });
  assert.ok(res.ok, `refused: ${res.reason}`);
  assert.equal(res.uv.length, 4);
  // And the recovered parameters must be real, not merely "not refused".
  res.uv.forEach(([u, v], i) => {
    const truth = [onRim(0), CENTRE, onRim(1), onRim(2)][i];
    const back = surfacePoint(cap, u, v);
    assert.ok(Math.hypot(back[0] - truth[0], back[1] - truth[1], back[2] - truth[2]) < 1e-3,
      `sample ${i} recovered a parameter that does not evaluate back to it`);
  });
});

test('THE POLE LEADS, TRAILS AND REPEATS — every arrangement, not just the one that was hit', () => {
  const cases = {
    'pole first': [CENTRE, onRim(0), onRim(1)],
    'pole last': [onRim(0), onRim(1), CENTRE],
    'pole twice': [onRim(0), CENTRE, onRim(2), CENTRE, onRim(4)],
    'pole only': [CENTRE, CENTRE, CENTRE],
  };
  for (const [name, pts] of Object.entries(cases)) {
    const res = projectPointsToSurfaceUV(pts, cap, { tolerance: 0.001 });
    assert.ok(res.ok, `${name} refused: ${res.reason}`);
  }
});

test('A POINT GENUINELY OFF THE SURFACE STILL REFUSES, and says the re-seed also missed', () => {
  // The re-seed must not turn every refusal into a pass. `closestPointOnSurface`
  // is a global search, so a point it cannot reach is a real statement about
  // the geometry — and the reason has to say that, or the next reader will
  // suspect the seed again.
  const res = projectPointsToSurfaceUV([onRim(0), [0, 0, 50], onRim(1)], cap, { tolerance: 0.001 });
  assert.equal(res.ok, false, 'a point 50 units off the plane must refuse');
  assert.match(res.reason, /re-seeded from scratch/);
});
