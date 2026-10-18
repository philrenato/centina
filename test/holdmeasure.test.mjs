import test from 'node:test';
import assert from 'node:assert';
import { planarCurveArea, closedSurfaceVolume, revolveRegionVolume, fractionToHeight } from '../kernel/holdmeasure.mjs';
import { massProperties } from '../kernel/massprops.mjs';

const S2 = Math.SQRT1_2;

// A clamped degree-1 curve through `pts`.
function polyline(pts) {
  const knots = [0, 0];
  for (let i = 1; i <= pts.length - 2; i += 1) knots.push(i);
  knots.push(pts.length - 1, pts.length - 1);
  return { degree: 1, knots, ctrlPts: pts.map((p) => [p[0], p[1], p[2] ?? 0, 1]) };
}

// The standard 9-point rational quadratic full circle in the plane spanned by
// unit vectors a, b about center c.
function circle(r, c = [0, 0, 0], a = [1, 0, 0], b = [0, 1, 0]) {
  const ctrlPts = [];
  for (let j = 0; j < 9; j += 1) {
    const t = (j * Math.PI) / 4, f = j % 2 ? Math.SQRT2 : 1, w = j % 2 ? S2 : 1;
    const x = r * f * Math.cos(t), y = r * f * Math.sin(t);
    ctrlPts.push([c[0] + a[0] * x + b[0] * y, c[1] + a[1] * x + b[1] * y, c[2] + a[2] * x + b[2] * y, w]);
  }
  return { degree: 2, knots: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], ctrlPts };
}

// A semicircle meridian in the XZ plane from the south pole to the north pole.
function semicircle(r) {
  return {
    degree: 2,
    knots: [0, 0, 0, 1, 1, 2, 2, 2],
    ctrlPts: [[0, 0, -r, 1], [r, 0, -r, S2], [r, 0, 0, 1], [r, 0, r, S2], [0, 0, r, 1]],
  };
}

// The surface of revolution of an XZ-plane meridian about Z: U the full circle,
// V the meridian (Piegl & Tiller A8.1).
function revolveZ(meridian) {
  const ctrlNet = [];
  for (let j = 0; j < 9; j += 1) {
    const t = (j * Math.PI) / 4, f = j % 2 ? Math.SQRT2 : 1, wj = j % 2 ? S2 : 1;
    ctrlNet.push(meridian.ctrlPts.map(([x, , z, w]) => [x * f * Math.cos(t), x * f * Math.sin(t), z, w * wj]));
  }
  return { degU: 2, degV: meridian.degree, knotsU: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], knotsV: meridian.knots.slice(), ctrlNet };
}

function translateSrf(srf, d) {
  return { ...srf, ctrlNet: srf.ctrlNet.map((row) => row.map(([x, y, z, w]) => [x + d[0], y + d[1], z + d[2], w])) };
}

function flipU(srf) {
  const U = srf.knotsU, a = U[0], b = U[U.length - 1];
  return { ...srf, knotsU: U.map((k) => a + b - k).reverse(), ctrlNet: srf.ctrlNet.slice().reverse() };
}

const ZAXIS = { origin: [0, 0, 0], direction: [0, 0, 1] };

test('planarCurveArea: a unit square polyline is 1, exact', () => {
  const r = planarCurveArea(polyline([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]));
  assert.strictEqual(r.reason, null);
  assert.ok(Math.abs(r.value - 1) < 1e-12, `area ${r.value}`);
  assert.strictEqual(r.exact, true);
});

test('planarCurveArea: a rational circle r = 10 is 100π within 1e-9, at the origin and 1e5 away', () => {
  const at0 = planarCurveArea(circle(10));
  assert.ok(Math.abs(at0.value - 100 * Math.PI) < 1e-9, `area ${at0.value} off by ${at0.value - 100 * Math.PI}`);
  assert.strictEqual(at0.exact, true);
  const far = planarCurveArea(circle(10, [1e5, 1e5, 0]));
  assert.ok(Math.abs(far.value - 100 * Math.PI) < 1e-9, `translated area off by ${far.value - 100 * Math.PI}`);
  // and in a tilted plane
  const tilt = planarCurveArea(circle(10, [3, -4, 5], [S2, 0, S2], [0, 1, 0]));
  assert.ok(Math.abs(tilt.value - 100 * Math.PI) < 1e-9);
});

test('planarCurveArea: winding does not set the sign', () => {
  const cw = planarCurveArea(polyline([[0, 0], [0, 2], [3, 2], [3, 0], [0, 0]]));
  assert.ok(Math.abs(cw.value - 6) < 1e-12);
});

test('planarCurveArea: a 2-point curve has no area', () => {
  const r = planarCurveArea(polyline([[0, 0], [5, 0]]));
  assert.strictEqual(r.value, null);
  assert.ok(r.reason);
});

test('planarCurveArea: a figure-eight crosses itself; a closed planar quad beside it measures', () => {
  const eight = planarCurveArea(polyline([[0, 0], [2, 2], [2, 0], [0, 2], [0, 0]]));
  assert.strictEqual(eight.value, null);
  assert.strictEqual(eight.reason, 'crosses itself');
  const quad = planarCurveArea(polyline([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]));
  assert.ok(Math.abs(quad.value - 4) < 1e-12);
});

test('planarCurveArea: an open arc is not closed; the full circle beside it measures', () => {
  const quarter = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[10, 0, 0, 1], [10, 10, 0, S2], [0, 10, 0, 1]] };
  const r = planarCurveArea(quarter);
  assert.strictEqual(r.value, null);
  assert.strictEqual(r.reason, 'not closed');
  assert.ok(planarCurveArea(circle(10)).value > 0);
});

test('planarCurveArea: a square with one corner lifted 1 mm is not planar; flat, it measures', () => {
  const lifted = planarCurveArea(polyline([[0, 0, 0], [10, 0, 0], [10, 10, 1], [0, 10, 0], [0, 0, 0]]));
  assert.strictEqual(lifted.value, null);
  assert.strictEqual(lifted.reason, 'not planar');
  const flat = planarCurveArea(polyline([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 0]]));
  assert.ok(Math.abs(flat.value - 100) < 1e-12);
});

test('closedSurfaceVolume: a revolved sphere r = 10 is 4000π/3 within 1e-8, exact', () => {
  const r = closedSurfaceVolume(revolveZ(semicircle(10)));
  assert.strictEqual(r.reason, null);
  assert.ok(Math.abs(r.value - (4000 * Math.PI) / 3) < 1e-8, `volume off by ${r.value - (4000 * Math.PI) / 3}`);
  assert.strictEqual(r.exact, true);
});

test('closedSurfaceVolume: flipped normals give the same value (sign from closure, not winding)', () => {
  const srf = revolveZ(semicircle(10));
  const a = closedSurfaceVolume(srf), b = closedSurfaceVolume(flipU(srf));
  assert.ok(Math.abs(a.value - b.value) < 1e-9, `${a.value} vs ${b.value}`);
});

test('closedSurfaceVolume: the sphere 1e5 away gives the same value', () => {
  const r = closedSurfaceVolume(translateSrf(revolveZ(semicircle(10)), [1e5, -1e5, 1e5]));
  assert.ok(Math.abs(r.value - (4000 * Math.PI) / 3) < 1e-6, `off by ${r.value - (4000 * Math.PI) / 3}`);
});

test('a torus by Pappus and by closedSurfaceVolume agree within 1e-8', () => {
  const R = 25, rr = 6;
  const meridian = circle(rr, [R, 0, 0], [1, 0, 0], [0, 0, 1]);
  const pappus = revolveRegionVolume(meridian, ZAXIS, null);
  const shell = closedSurfaceVolume(revolveZ(meridian));
  const truth = 2 * Math.PI * Math.PI * R * rr * rr;
  assert.ok(Math.abs(pappus.value - truth) < 1e-8, `Pappus off by ${pappus.value - truth}`);
  assert.ok(Math.abs(pappus.value - shell.value) < 1e-8, `${pappus.value} vs ${shell.value}`);
  assert.strictEqual(pappus.exact, true);
  assert.strictEqual(shell.exact, true);
});

test('closedSurfaceVolume: two disjoint shells sum', () => {
  const a = revolveZ(semicircle(10)), b = translateSrf(revolveZ(semicircle(4)), [40, 0, 0]);
  const r = closedSurfaceVolume([a, b]);
  const truth = (4 * Math.PI / 3) * (1000 + 64);
  assert.ok(Math.abs(r.value - truth) < 1e-8, `off by ${r.value - truth}`);
});

// A box as ONE degree-1 surface: U runs around the four sides, V from the bottom
// center (a pole) up a side to the top center (a pole).
function boxSurface(lo, hi) {
  const [x0, y0, z0] = lo, [x1, y1, z1] = hi;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
  const ctrlNet = ring.map(([x, y]) => [[cx, cy, z0, 1], [x, y, z0, 1], [x, y, z1, 1], [cx, cy, z1, 1]]);
  return { degU: 1, degV: 1, knotsU: [0, 0, 1, 2, 3, 4, 4], knotsV: [0, 0, 1, 2, 3, 3], ctrlNet };
}
function boxLoops(lo, hi) {
  const [x0, y0, z0] = lo, [x1, y1, z1] = hi;
  return [
    [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]],
    [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
    [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
    [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]],
    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
    [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
  ];
}

test('closedSurfaceVolume: a box equals massProperties within 1e-12', () => {
  const lo = [7, -13, 3], hi = [11, -7, 12];
  const r = closedSurfaceVolume(boxSurface(lo, hi));
  const m = massProperties(boxLoops(lo, hi));
  assert.ok(Math.abs(r.value - Math.abs(m.volume)) <= 1e-12 * Math.abs(m.volume), `${r.value} vs ${m.volume}`);
  assert.strictEqual(r.exact, true);
});

test('closedSurfaceVolume: an open surface is not closed; the closed sphere beside it measures', () => {
  const half = revolveZ({ degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[0, 0, -10, 1], [10, 0, -10, S2], [10, 0, 0, 1]] });
  const r = closedSurfaceVolume(half);
  assert.strictEqual(r.value, null);
  assert.strictEqual(r.reason, 'not closed');
  assert.ok(closedSurfaceVolume(revolveZ(semicircle(10))).value > 0);
});

test('exact is FALSE on a coarse step where n and 2n differ by more than 1e-9', () => {
  const coarse = closedSurfaceVolume(revolveZ(semicircle(10)), { cellsPerSpan: 1, maxCellsPerSpan: 2, order: 2 });
  assert.ok(coarse.value != null);
  assert.strictEqual(coarse.exact, false);
  assert.ok(coarse.gap > 1e-9 * coarse.value, `gap ${coarse.gap}`);
  const curve = planarCurveArea(circle(10), { cellsPerSpan: 1, maxCellsPerSpan: 2, order: 2 });
  assert.strictEqual(curve.exact, false);
});

test('revolveRegionVolume: a sphere from its semicircle meridian, either direction', () => {
  const truth = (4000 * Math.PI) / 3;
  const m = semicircle(10);
  const r = revolveRegionVolume(m, ZAXIS, null);
  assert.ok(Math.abs(r.value - truth) < 1e-8, `off by ${r.value - truth}`);
  const rev = { degree: 2, knots: m.knots.map((k) => 2 - k).reverse(), ctrlPts: m.ctrlPts.slice().reverse() };
  assert.ok(Math.abs(revolveRegionVolume(rev, ZAXIS, null).value - truth) < 1e-8);
});

test('revolveRegionVolume: a cup cavity stops at the rim and clips at a height', () => {
  // A cylinder cup r = 5, floor at z = 0, wall to z = 20 then a lip that turns
  // down: the cavity ends at the first height maximum (z = 20).
  const cup = polyline([[0, 0, 0], [5, 0, 0], [5, 0, 20], [7, 0, 18]]);
  const full = revolveRegionVolume(cup, ZAXIS, null);
  assert.ok(Math.abs(full.value - Math.PI * 25 * 20) < 1e-9, `${full.value}`);
  const half = revolveRegionVolume(cup, ZAXIS, 10);
  assert.ok(Math.abs(half.value - Math.PI * 25 * 10) < 1e-9, `${half.value}`);
  // A cone cavity, clipped: π r² h / 3 with r = h / 2.
  const cone = polyline([[0, 0, 0], [10, 0, 20]]);
  const clip = revolveRegionVolume(cone, ZAXIS, 8);
  assert.ok(Math.abs(clip.value - Math.PI * 16 * 8 / 3) < 1e-9, `${clip.value}`);
  // Starting off the axis, or leaving the axis plane, is refused; the cup above measures.
  assert.strictEqual(revolveRegionVolume(polyline([[1, 0, 0], [5, 0, 0], [5, 0, 20]]), ZAXIS, null).reason, 'not on the axis');
  assert.strictEqual(revolveRegionVolume(polyline([[0, 0, 0], [5, 0, 0], [5, 1, 20]]), ZAXIS, null).reason, 'not planar');
});

test('fractionToHeight: half a cylinder cup is at half its height; a cone at 2^(-1/3) of it', () => {
  const cup = polyline([[0, 0, 0], [5, 0, 0], [5, 0, 20]]);
  const h = fractionToHeight(cup, ZAXIS, 0.5);
  assert.ok(Math.abs(h.value - 10) < 1e-9, `${h.value}`);
  const cone = polyline([[0, 0, 0], [10, 0, 20]]);
  const hc = fractionToHeight(cone, ZAXIS, 0.5);
  assert.ok(Math.abs(hc.value - 20 * Math.cbrt(0.5)) < 1e-8, `${hc.value}`);
  assert.strictEqual(fractionToHeight(cup, ZAXIS, 1.5).value, null);
});
