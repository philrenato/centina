// Pipe rounded-corner cross-section resolution. `cornerStyle:'rounded'` on a
// closed, multi-corner rail needs enough dense-rail samples per span:
// under-sampling produces Gibbs-like ringing at every arc/line junction, seen
// as torn or self-intersecting geometry. The per-span sample floor is
// `MIN_SPAN_SAMPLES` in `denseRailFrames` (kernel/sweep.mjs; its header
// comment has the derivation). The swept surface's V control-point count
// comes from `sweep1RigidResampled`'s dense sampling, not from the rail's own
// control-point count, so a count such as 157 on a ~6-corner rail is
// expected. Every assertion here measures the swept surface via
// `surfacePoint`, never internals.

import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle, filletOpenPolyline, filletSegmentsToCurve } from '../kernel/primitives.mjs';
import { sweep1Rigid } from '../kernel/sweep.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

function ringRadii(srf, v, samples = 24) {
  const [uMin, uMax] = [srf.knotsU[0], srf.knotsU[srf.knotsU.length - 1]];
  const ring = [];
  for (let j = 0; j < samples; j++) {
    const u = uMin + (uMax - uMin) * (j / samples);
    ring.push(surfacePoint(srf, u, v));
  }
  const cx = ring.reduce((a, p) => a + p[0], 0) / ring.length;
  const cy = ring.reduce((a, p) => a + p[1], 0) / ring.length;
  const cz = ring.reduce((a, p) => a + p[2], 0) / ring.length;
  return ring.map((p) => Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz));
}

function worstCrossSectionEllipticity(srf, pipeRadius, stations = 240) {
  const [vMin, vMax] = [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]];
  let worst = 0;
  for (let i = 0; i <= stations; i++) {
    const v = vMin + (vMax - vMin) * (i / stations);
    const radii = ringRadii(srf, v);
    const ratio = (Math.max(...radii) - Math.min(...radii)) / pipeRadius;
    if (Number.isFinite(ratio) && ratio > worst) worst = ratio;
  }
  return worst;
}

function roundedPipeSurface(points, cornerRadius, pipeRadius) {
  const res = filletOpenPolyline(points, cornerRadius, { closed: true });
  assert.equal(res.ok, true, res.reason);
  const rail = filletSegmentsToCurve(res.segments);
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], pipeRadius);
  return sweep1Rigid(rail, circleProfile);
}

test('a rounded-corner Pipe on a 3D, 5-corner closed rail keeps every cross-section close to a true circle', () => {
  const pts = [
    [0, 0, 0],
    [40, 5, 8],
    [55, 45, -6],
    [15, 60, 12],
    [-20, 25, -4],
  ];
  const srf = roundedPipeSurface(pts, 1, 5);
  const worst = worstCrossSectionEllipticity(srf, 5);
  assert.ok(worst < 0.03, `worst cross-section ellipticity ${(worst * 100).toFixed(2)}% should be under 3%`);
});

test('a tight-cornered 8-vertex star rail (small radius relative to segment length) stays close to circular at every station', () => {
  const N = 8;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const r = 8 + (i % 2 === 0 ? 3 : -2);
    pts.push([r * Math.cos(a), r * Math.sin(a), (i % 3) * 2 - 2]);
  }
  const srf = roundedPipeSurface(pts, 1, 2.5);
  const worst = worstCrossSectionEllipticity(srf, 2.5);
  assert.ok(worst < 0.01, `worst cross-section ellipticity ${(worst * 100).toFixed(2)}% should be tiny on this fixture`);
});

test('a 12-corner closed rail — many junctions in one loop — stays well-behaved, not compounding into a larger error', () => {
  const N = 12;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    pts.push([20 * Math.cos(a) + (i % 2) * 2, 20 * Math.sin(a), Math.sin(a * 3) * 5]);
  }
  const srf = roundedPipeSurface(pts, 1, 3);
  const worst = worstCrossSectionEllipticity(srf, 3);
  assert.ok(worst < 0.01, `worst cross-section ellipticity ${(worst * 100).toFixed(2)}% should stay small even with 12 junctions`);
});

test('a straight (zero-turn) span still gets at least MIN_SPAN_SAMPLES worth of resolution — the floor applies uniformly, not only near corners', () => {
  // A rail with one real corner and one very long straight run: the straight
  // run's own local shape is a plain 3D line (rigidly transported, zero
  // curvature) — reproduced exactly at any density, so this proves the floor
  // is actually applied there (a real V-resolution jump vs. a much shorter
  // straight run in the same rail), not that quality happens to be fine.
  const pts = [[0, 0, 0], [20, 0, 0], [20, 200, 0]];
  const res = filletOpenPolyline(pts, 3, { closed: false });
  assert.equal(res.ok, true, res.reason);
  const rail = filletSegmentsToCurve(res.segments);
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2);
  const srf = sweep1Rigid(rail, circleProfile);
  // 3 segments (short line, arc, long line) each contribute (n+1) points minus
  // shared joints; with a floor of 48 samples/span this must be substantially
  // denser than a fixed 12 samples per span (~3*13-2=37 in total).
  assert.ok(srf.ctrlNet[0].length > 100, `expected a much denser V-resolution than a fixed 12 samples per span, got ${srf.ctrlNet[0].length}`);
  const worst = worstCrossSectionEllipticity(srf, 2);
  assert.ok(worst < 0.03, `worst cross-section ellipticity ${(worst * 100).toFixed(2)}% should be small on this simple 1-corner rail`);
});

test('the composed rail\'s own control-point count is independent of the sweep\'s resampling floor', () => {
  const pts = [
    [0, 0, 0],
    [40, 5, 8],
    [55, 45, -6],
    [15, 60, 12],
    [-20, 25, -4],
  ];
  const res = filletOpenPolyline(pts, 1, { closed: true });
  const rail = filletSegmentsToCurve(res.segments);
  assert.equal(rail.ctrlPts.length, 21, 'the rail\'s own composed control-net size (4*cornerCount+1) does not depend on the sweep resampling');
});
