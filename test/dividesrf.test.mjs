import test from 'node:test';
import assert from 'node:assert/strict';
import { surfacePoint, divideSrfGrid, surfaceClosure } from '../kernel/surface.mjs';
import { makeArc, makeLine, makeCircle, revolve, extrude } from '../kernel/primitives.mjs';
import { extractIsocurveU, extractIsocurveV } from '../kernel/isocurve.mjs';
import { curvePoint } from '../kernel/curve.mjs';

// The bilinear fixture curve-surface.test.mjs already proved surfacePoint
// against directly — reused here so a grid test isolates divideSrfGrid's
// own placement logic from surface evaluation correctness (already proven
// elsewhere).
const bilinearSrf = {
  degU: 1, degV: 1,
  knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
  ctrlNet: [
    [[0, 0, 0, 1], [0, 10, 0, 1]],
    [[10, 0, 5, 1], [10, 10, 5, 1]],
  ],
};

// A fully RATIONAL surface (both U and V direction have non-unit
// alternating weights), the SAME stress fixture isocurve.test.mjs already
// uses — deliberately non-[0,1] domain (a 270deg-by-270deg revolve), so a
// grid test that only happened to work for a domain that's already [0,1]
// wouldn't be a real proof.
function makeTestSurface() {
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 5, 0, 1.5 * Math.PI);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 1.5 * Math.PI);
}

test('divideSrfGrid on a bilinear surface lands on exact parameter-uniform grid points', () => {
  const grid = divideSrfGrid(bilinearSrf, 2, 2); // 3x3 grid
  assert.equal(grid.length, 9);
  // Center point (u=0.5, v=0.5) is the exact bilinear average.
  const center = grid.find((g) => Math.abs(g.u - 0.5) < 1e-9 && Math.abs(g.v - 0.5) < 1e-9);
  assert.ok(center, 'expected an exact (0.5, 0.5) grid point');
  assert.ok(Math.abs(center.point[0] - 5) < 1e-10);
  assert.ok(Math.abs(center.point[1] - 5) < 1e-10);
  assert.ok(Math.abs(center.point[2] - 2.5) < 1e-10);
});

test('divideSrfGrid always returns (uCount+1)*(vCount+1) points including all FOUR exact corner points', () => {
  const srf = makeTestSurface();
  const uCount = 4, vCount = 3;
  const grid = divideSrfGrid(srf, uCount, vCount);
  assert.equal(grid.length, (uCount + 1) * (vCount + 1));
  const uDomain = [srf.knotsU[0], srf.knotsU[srf.knotsU.length - 1]];
  const vDomain = [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]];
  for (const uEnd of uDomain) {
    for (const vEnd of vDomain) {
      const hit = grid.find((g) => Math.abs(g.u - uEnd) < 1e-9 && Math.abs(g.v - vEnd) < 1e-9);
      assert.ok(hit, `expected an exact corner at u=${uEnd}, v=${vEnd}`);
      const direct = surfacePoint(srf, uEnd, vEnd);
      const d = Math.hypot(hit.point[0] - direct[0], hit.point[1] - direct[1], hit.point[2] - direct[2]);
      assert.ok(d < 1e-9, `corner (${uEnd},${vEnd}) diverges from direct surfacePoint by ${d}`);
    }
  }
});

test('divideSrfGrid every returned point matches a direct surfacePoint evaluation at its own (u,v) exactly', () => {
  const srf = makeTestSurface();
  const grid = divideSrfGrid(srf, 6, 5);
  for (const { u, v, point } of grid) {
    const direct = surfacePoint(srf, u, v);
    const d = Math.hypot(point[0] - direct[0], point[1] - direct[1], point[2] - direct[2]);
    assert.ok(d < 1e-9, `(u=${u}, v=${v}) diverges by ${d}`);
  }
});

test('divideSrfGrid U/V values are evenly spaced in PARAMETER space (the real point of a parameter-uniform grid, not arc-length)', () => {
  const srf = makeTestSurface();
  const uCount = 5, vCount = 4;
  const grid = divideSrfGrid(srf, uCount, vCount);
  const uVals = [...new Set(grid.map((g) => g.u))].sort((a, b) => a - b);
  const vVals = [...new Set(grid.map((g) => g.v))].sort((a, b) => a - b);
  assert.equal(uVals.length, uCount + 1);
  assert.equal(vVals.length, vCount + 1);
  const uStep = uVals[1] - uVals[0];
  for (let i = 1; i < uVals.length; i++) assert.ok(Math.abs((uVals[i] - uVals[i - 1]) - uStep) < 1e-9, `U step ${i} unequal`);
  const vStep = vVals[1] - vVals[0];
  for (let i = 1; i < vVals.length; i++) assert.ok(Math.abs((vVals[i] - vVals[i - 1]) - vStep) < 1e-9, `V step ${i} unequal`);
});

test('divideSrfGrid\'s own u=uMin column exactly reproduces extractIsocurveU(srf, uMin) at every grid V — a real cross-check against already-proven kernel machinery, not a self-consistency tautology', () => {
  const srf = makeTestSurface();
  const grid = divideSrfGrid(srf, 3, 6);
  const uMin = srf.knotsU[0];
  const isoAtUMin = extractIsocurveU(srf, uMin);
  const columnAtUMin = grid.filter((g) => Math.abs(g.u - uMin) < 1e-9);
  assert.equal(columnAtUMin.length, 7); // vCount+1
  for (const { v, point } of columnAtUMin) {
    const onIso = curvePoint(isoAtUMin, v);
    const d = Math.hypot(point[0] - onIso[0], point[1] - onIso[1], point[2] - onIso[2]);
    assert.ok(d < 1e-9, `v=${v}: grid point diverges from extractIsocurveU's own curve by ${d}`);
  }
});

test('divideSrfGrid\'s own v=vMax row exactly reproduces extractIsocurveV(srf, vMax) at every grid U', () => {
  const srf = makeTestSurface();
  const grid = divideSrfGrid(srf, 5, 3);
  const vMax = srf.knotsV[srf.knotsV.length - 1];
  const isoAtVMax = extractIsocurveV(srf, vMax);
  const rowAtVMax = grid.filter((g) => Math.abs(g.v - vMax) < 1e-9);
  assert.equal(rowAtVMax.length, 6); // uCount+1
  for (const { u, point } of rowAtVMax) {
    const onIso = curvePoint(isoAtVMax, u);
    const d = Math.hypot(point[0] - onIso[0], point[1] - onIso[1], point[2] - onIso[2]);
    assert.ok(d < 1e-9, `u=${u}: grid point diverges from extractIsocurveV's own curve by ${d}`);
  }
});

test('divideSrfGrid produces no NaN/Infinity anywhere across a full grid on the rational stress surface', () => {
  const srf = makeTestSurface();
  const grid = divideSrfGrid(srf, 7, 6);
  for (const { point } of grid) assert.ok(point.every(Number.isFinite), JSON.stringify(point));
});

// A full 360deg revolve of an OPEN straight profile (a cylinder) wraps in
// V (the sweep) but not U (the profile itself never wraps) — the surface
// analog of divideByArcLength's own closed-curve seam case, now proven
// per-direction: the closed V direction must place exactly vCount points
// around the full loop (never repeating the v=vMin/v=vMax seam column),
// while the open U direction is completely unaffected (still uCount+1).
test('divideSrfGrid on a cylinder (closed in V, open in U) places exactly vCount DISTINCT angular columns, never the seam duplicate — U unaffected', () => {
  const profile = makeLine([5, 0, 0], [5, 0, 20]);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false, 'the straight profile itself never wraps');
  assert.equal(closedV, true, 'a full 360deg revolve wraps the sweep direction into a seam');
  const uCount = 3, vCount = 4;
  const grid = divideSrfGrid(srf, uCount, vCount);
  assert.equal(grid.length, (uCount + 1) * vCount, `expected (uCount+1)*vCount = ${(uCount + 1) * vCount}, got ${grid.length}`);
  const uVals = [...new Set(grid.map((g) => Math.round(g.u * 1e6)))];
  assert.equal(uVals.length, uCount + 1, 'U (open) direction still gets uCount+1 distinct values, unaffected by the V fix');
  const vVals = [...new Set(grid.map((g) => Math.round(g.v * 1e6)))];
  assert.equal(vVals.length, vCount, 'V (closed) direction gets exactly vCount distinct values, no seam duplicate');
  // Every point genuinely lies on the cylinder wall (radius 5 from the Z axis).
  for (const { point } of grid) assert.ok(Math.abs(Math.hypot(point[0], point[1]) - 5) < 1e-9, JSON.stringify(point));
  // No two grid points coincide (the old bug: a v=vMin and v=vMax column at
  // the same U row were the SAME physical point on the cylinder wall).
  for (let i = 0; i < grid.length; i++) {
    for (let j = i + 1; j < grid.length; j++) {
      const d = Math.hypot(grid[i].point[0] - grid[j].point[0], grid[i].point[1] - grid[j].point[1], grid[i].point[2] - grid[j].point[2]);
      assert.ok(d > 1e-6, `grid points ${i} and ${j} coincide (seam duplicate): ${d}`);
    }
  }
});

// The mirror case: an extrude of a CLOSED profile (a circle) is a tube,
// closed in U (the profile itself) and open in V (the straight extrude
// direction) — proves the fix reacts to WHICHEVER direction is actually
// closed, not a hardcoded "V is always the closed one" assumption.
test('divideSrfGrid on a tube (closed in U, open in V) places exactly uCount DISTINCT angular rows, never the seam duplicate — V unaffected', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 4, 4);
  const srf = extrude(profile, [0, 0, 1], 15);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true, 'the circle profile itself already wraps');
  assert.equal(closedV, false, 'a straight extrude never wraps');
  const uCount = 5, vCount = 2;
  const grid = divideSrfGrid(srf, uCount, vCount);
  assert.equal(grid.length, uCount * (vCount + 1), `expected uCount*(vCount+1) = ${uCount * (vCount + 1)}, got ${grid.length}`);
  const uVals = [...new Set(grid.map((g) => Math.round(g.u * 1e6)))];
  assert.equal(uVals.length, uCount, 'U (closed) direction gets exactly uCount distinct values, no seam duplicate');
  const vVals = [...new Set(grid.map((g) => Math.round(g.v * 1e6)))];
  assert.equal(vVals.length, vCount + 1, 'V (open) direction still gets vCount+1 distinct values, unaffected by the U fix');
  for (let i = 0; i < grid.length; i++) {
    for (let j = i + 1; j < grid.length; j++) {
      const d = Math.hypot(grid[i].point[0] - grid[j].point[0], grid[i].point[1] - grid[j].point[1], grid[i].point[2] - grid[j].point[2]);
      assert.ok(d > 1e-6, `grid points ${i} and ${j} coincide (seam duplicate): ${d}`);
    }
  }
});

// A torus (closed in BOTH directions) proves the fix composes cleanly
// per-axis: both U and V drop their own seam duplicate independently,
// giving a plain uCount*vCount grid with zero coincident points anywhere.
test('divideSrfGrid on a torus (closed in BOTH directions) gives a plain uCount*vCount grid, no seam duplicate in either direction', () => {
  const profile = makeCircle([5, 0, 0], [0, 0, 1], [1, 0, 0], 1, 4);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true);
  assert.equal(closedV, true);
  const uCount = 3, vCount = 4;
  const grid = divideSrfGrid(srf, uCount, vCount);
  assert.equal(grid.length, uCount * vCount, `expected uCount*vCount = ${uCount * vCount}, got ${grid.length}`);
  for (const { point } of grid) assert.ok(point.every(Number.isFinite), JSON.stringify(point));
  for (let i = 0; i < grid.length; i++) {
    for (let j = i + 1; j < grid.length; j++) {
      const d = Math.hypot(grid[i].point[0] - grid[j].point[0], grid[i].point[1] - grid[j].point[1], grid[i].point[2] - grid[j].point[2]);
      assert.ok(d > 1e-6, `grid points ${i} and ${j} coincide (seam duplicate): ${d}`);
    }
  }
});

test('divideSrfGrid rejects a non-positive or non-integer uCount/vCount', () => {
  assert.throws(() => divideSrfGrid(bilinearSrf, 0, 2));
  assert.throws(() => divideSrfGrid(bilinearSrf, 2, 0));
  assert.throws(() => divideSrfGrid(bilinearSrf, 2.5, 2));
  assert.throws(() => divideSrfGrid(bilinearSrf, 2, -1));
});
