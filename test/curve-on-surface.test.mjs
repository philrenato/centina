// A CURVE THAT LIVES IN A SURFACE'S PARAMETER SPACE — the geometry behind
// "draw on a surface and have the line update when the surface changes".
//
// Three claims, and each has a control:
//   1. Every sample is ON the surface, and the stroke BETWEEN samples does not
//      cut corners across it. The samples are on-surface by construction, so
//      measuring them proves nothing — the chord deviation is the real number.
//   2. A stroke near a seam takes the SHORT way round. Interpolating in raw
//      parameters sends it the long way across the whole model to reach a point
//      millimeters away, and the control below measures exactly that.
//   3. Change the surface, re-evaluate the SAME stations, and the stroke
//      follows — which is the entire point of storing (u,v) rather than 3-D.
import test from 'node:test';
import assert from 'node:assert/strict';
import { surfacePoint, surfaceClosure } from '../kernel/surface.mjs';
import { makeArc, revolve } from '../kernel/primitives.mjs';
import { curveOnSurfacePoints, curveOnSurfaceUV, chordDeviation, unwrapStations } from '../kernel/curveonsurface.mjs';

// A real sphere: a half-circle profile revolved a full turn, so it is CLOSED in
// v and carries poles at both ends of u — not a flat plate, which would make
// every claim here trivially true.
function sphere(radius = 50) {
  // A meridian: a half-circle in the XZ plane running pole to pole, revolved a
  // full turn about Z. Angles are radians here, and the profile must lie in a
  // plane CONTAINING the axis or the revolve sweeps a torus instead.
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], radius, -Math.PI / 2, Math.PI);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}

// STATIONS ARE IN THE SURFACE'S OWN DOMAIN, not in 0..1 — that is what
// closestPointOnSurface hands back from a click, so it is what the stored
// stations are. A revolve's u domain is its knot range, not the unit interval.
const domainOf = (knots, degree) => [knots[degree], knots[knots.length - 1 - degree]];
function stationsFor(srf, fractions) {
  const [uMin, uMax] = domainOf(srf.knotsU, srf.degU);
  const [vMin, vMax] = domainOf(srf.knotsV, srf.degV);
  return fractions.map(([fu, fv]) => [uMin + (uMax - uMin) * fu, vMin + (vMax - vMin) * fv]);
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const polylineLength = (pts) => {
  let L = 0;
  for (let i = 0; i < pts.length - 1; i++) L += dist(pts[i], pts[i + 1]);
  return L;
};

test('a sphere fixture is genuinely closed in v — otherwise the seam tests prove nothing', () => {
  const s = sphere();
  const closure = surfaceClosure(s);
  // MEASURED, not assumed: the revolve sweeps in V (the profile is U), so the
  // seam every test below crosses is a v seam. Asserting the wrong direction
  // would leave the seam tests running down a direction that cannot wrap, where
  // they would pass by never being tested.
  assert.equal(closure.closedV, true, 'the revolve must be closed in v for any seam claim below to mean anything');
  assert.equal(closure.closedU, false, 'and open in u — the profile direction runs pole to pole');
});

test('every sample lies on the surface, and the chords between them hug it', () => {
  const s = sphere(50);
  const stations = stationsFor(s, [[0.1, 0.3], [0.3, 0.5], [0.55, 0.45], [0.8, 0.6]]);
  const res = curveOnSurfacePoints(s, stations, { degree: 3, samplesPerSpan: 24 });
  assert.equal(res.ok, true);
  assert.ok(res.points.length > 50, `expected a densified stroke, got ${res.points.length} points`);

  // On-surface by construction — asserted anyway, because it is the property
  // the whole design rests on and a regression here would be silent.
  let worstOff = 0;
  for (let i = 0; i < res.uv.length; i++) {
    const p = surfacePoint(s, res.uv[i][0], res.uv[i][1]);
    worstOff = Math.max(worstOff, dist(p, res.points[i]));
  }
  assert.ok(worstOff < 1e-9, `samples must BE surface evaluations, worst off by ${worstOff}`);

  // The real measurement: how far the drawn polyline strays between samples.
  const dev = chordDeviation(s, res.uv);
  assert.ok(dev < 0.5, `the stroke should hug a 50mm sphere, worst chord deviation ${dev.toFixed(4)}mm`);

  // AND IT IS THE SAMPLING THAT BUYS THAT. A coarse run must be measurably
  // worse, or the tolerance above is just loose enough to pass anything.
  const coarse = curveOnSurfacePoints(s, stations, { degree: 3, samplesPerSpan: 2 });
  const coarseDev = chordDeviation(s, coarse.uv);
  assert.ok(coarseDev > dev * 2,
    `denser sampling must actually reduce deviation (coarse ${coarseDev.toFixed(4)} vs fine ${dev.toFixed(4)})`);
});

test('a stroke across the seam takes the SHORT way round, not across the whole sphere', () => {
  const s = sphere(50);
  // Two stations either side of the V seam: 0.97 -> 0.03 is a step of +0.06 the
  // short way and -0.94 the long way. Held at the equator in u, so the hop is a
  // real arc rather than a pole where every u collapses to one point.
  const stations = stationsFor(s, [[0.5, 0.97], [0.5, 0.03]]);
  const res = curveOnSurfacePoints(s, stations, { degree: 1, samplesPerSpan: 32 });
  assert.equal(res.ok, true);

  const equator = 2 * Math.PI * 50;
  const short = polylineLength(res.points);
  assert.ok(short < equator * 0.15,
    `crossing the seam should be a short hop, got ${short.toFixed(2)}mm against an equator of ${equator.toFixed(2)}mm`);

  // THE CONTROL: the same two stations interpolated in RAW parameters, which is
  // what the unwrapping exists to avoid. It must be dramatically longer, or the
  // assertion above is not testing the unwrap at all.
  const rawPts = [];
  for (let i = 0; i <= 32; i++) {
    const t = i / 32;
    const v = stations[0][1] + (stations[1][1] - stations[0][1]) * t;
    const p = surfacePoint(s, stations[0][0], v);
    rawPts.push([p[0], p[1], p[2]]);
  }
  const long = polylineLength(rawPts);
  assert.ok(long > short * 5,
    `CONTROL: raw-parameter interpolation must go the long way (${long.toFixed(2)}mm vs ${short.toFixed(2)}mm)`);

  // And the unwrapped station really did leave the domain — the mechanism, not
  // just its effect.
  const path = unwrapStations(s, stations);
  const [, vMax] = domainOf(s.knotsV, s.degV);
  assert.ok(path[1][1] > vMax,
    `the second station should unwrap past the v domain end ${vMax}, got ${path[1][1]}`);
});

test('the stroke REFLOWS when the surface changes, from the same stations', () => {
  const small = sphere(50);
  const stations = stationsFor(small, [[0.1, 0.3], [0.35, 0.5], [0.6, 0.42]]);
  const before = curveOnSurfacePoints(small, stations, { degree: 3 });

  // The same surface, twice the size — a real host edit, not a nudge. Its
  // domain is identical (same profile shape, same revolve), so the SAME
  // stations address the same places on it, which is exactly the reflow claim.
  const big = sphere(100);
  const after = curveOnSurfacePoints(big, stations, { degree: 3 });

  assert.equal(before.points.length, after.points.length, 'the same stations must give the same station count');

  // It moved, and it moved BY THE RIGHT FACTOR — a curve that merely changed
  // would pass a "did it move" check while sitting nowhere near the surface.
  const rBefore = before.points.map((p) => Math.hypot(...p));
  const rAfter = after.points.map((p) => Math.hypot(...p));
  const worstBefore = Math.max(...rBefore.map((r) => Math.abs(r - 50)));
  const worstAfter = Math.max(...rAfter.map((r) => Math.abs(r - 100)));
  assert.ok(worstBefore < 1e-6, `before: every point sits on the 50mm sphere, worst off ${worstBefore}`);
  assert.ok(worstAfter < 1e-6, `after: every point sits on the 100mm sphere, worst off ${worstAfter}`);

  const moved = Math.max(...before.points.map((p, i) => dist(p, after.points[i])));
  assert.ok(moved > 10, `the stroke must actually follow the host, worst point moved ${moved.toFixed(3)}mm`);
});

test('a closed stroke returns to its first station', () => {
  const s = sphere(50);
  const stations = stationsFor(s, [[0.1, 0.4], [0.4, 0.55], [0.7, 0.4]]);
  const res = curveOnSurfacePoints(s, stations, { degree: 3, closed: true });
  assert.equal(res.ok, true);
  const gap = dist(res.points[0], res.points[res.points.length - 1]);
  assert.ok(gap < 1e-6, `a closed stroke must meet itself, gap ${gap}`);
});

test('fewer than two stations is refused, not silently drawn as nothing', () => {
  const s = sphere();
  const res = curveOnSurfacePoints(s, stationsFor(s, [[0.5, 0.5]]), {});
  assert.equal(res.ok, false);
  assert.match(res.reason, /two points/i);
});
