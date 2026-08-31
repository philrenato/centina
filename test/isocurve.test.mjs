import test from 'node:test';
import assert from 'node:assert/strict';
import { curvePoint } from '../kernel/curve.mjs';
import { surfacePoint, surfaceClosure } from '../kernel/surface.mjs';
import { makeArc, makeLine, makeCircle, revolve, extrude } from '../kernel/primitives.mjs';
import { extractIsocurveU, extractIsocurveV, extractBorderCurves, extractWireframeCurves } from '../kernel/isocurve.mjs';
import { grevilleAbscissae } from '../kernel/curve.mjs';

// A fully RATIONAL surface (both U and V direction have non-unit alternating
// weights) — a revolve of a 270deg arc profile revolved 270deg itself —
// deliberately the harshest stress case: if isocurve extraction only
// happened to work for a non-rational (Line/Polyline) profile or a
// non-rational revolve sweep, that would be a much weaker proof.
function makeTestSurface() {
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 5, 0, 1.5 * Math.PI);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 1.5 * Math.PI);
}

test('extractIsocurveU: the extracted curve exactly reproduces the surface at every sampled V, for several fixed U', () => {
  const srf = makeTestSurface();
  const uDomain = [srf.knotsU[0], srf.knotsU[srf.knotsU.length - 1]];
  const vDomain = [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]];
  for (const uFrac of [0, 0.13, 0.5, 0.87, 1]) {
    const u = uDomain[0] + uFrac * (uDomain[1] - uDomain[0]);
    const iso = extractIsocurveU(srf, u);
    assert.equal(iso.degree, srf.degV, `isocurve degree should match the surface's own V degree at uFrac=${uFrac}`);
    assert.deepEqual(iso.knots, srf.knotsV, `isocurve knots should be the surface's own V knots at uFrac=${uFrac}`);
    for (let vFrac = 0; vFrac <= 1.0001; vFrac += 0.05) {
      const v = vDomain[0] + vFrac * (vDomain[1] - vDomain[0]);
      const onSurface = surfacePoint(srf, u, v);
      const onCurve = curvePoint(iso, v);
      const d = Math.hypot(onSurface[0] - onCurve[0], onSurface[1] - onCurve[1], onSurface[2] - onCurve[2]);
      assert.ok(d < 1e-9, `uFrac=${uFrac} vFrac=${vFrac}: surface/isocurve diverge by ${d}`);
    }
  }
});

test('extractIsocurveV: the extracted curve exactly reproduces the surface at every sampled U, for several fixed V', () => {
  const srf = makeTestSurface();
  const uDomain = [srf.knotsU[0], srf.knotsU[srf.knotsU.length - 1]];
  const vDomain = [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]];
  for (const vFrac of [0, 0.25, 0.6, 1]) {
    const v = vDomain[0] + vFrac * (vDomain[1] - vDomain[0]);
    const iso = extractIsocurveV(srf, v);
    assert.equal(iso.degree, srf.degU);
    assert.deepEqual(iso.knots, srf.knotsU);
    for (let uFrac = 0; uFrac <= 1.0001; uFrac += 0.05) {
      const u = uDomain[0] + uFrac * (uDomain[1] - uDomain[0]);
      const onSurface = surfacePoint(srf, u, v);
      const onCurve = curvePoint(iso, u);
      const d = Math.hypot(onSurface[0] - onCurve[0], onSurface[1] - onCurve[1], onSurface[2] - onCurve[2]);
      assert.ok(d < 1e-9, `vFrac=${vFrac} uFrac=${uFrac}: surface/isocurve diverge by ${d}`);
    }
  }
});

test('extractIsocurveU at an exact boundary (u = domain start/end) reproduces the surface\'s own edge control ROW verbatim, not an approximation', () => {
  const srf = makeTestSurface();
  const uStart = srf.knotsU[0];
  const uEnd = srf.knotsU[srf.knotsU.length - 1];
  const isoStart = extractIsocurveU(srf, uStart);
  const isoEnd = extractIsocurveU(srf, uEnd);
  const firstRow = srf.ctrlNet[0];
  const lastRow = srf.ctrlNet[srf.ctrlNet.length - 1];
  for (let j = 0; j < firstRow.length; j++) {
    for (let c = 0; c < 4; c++) {
      assert.ok(Math.abs(isoStart.ctrlPts[j][c] - firstRow[j][c]) < 1e-9, `start row ctrlPt ${j} component ${c}`);
      assert.ok(Math.abs(isoEnd.ctrlPts[j][c] - lastRow[j][c]) < 1e-9, `end row ctrlPt ${j} component ${c}`);
    }
  }
});

test('extractIsocurveU/V are consistent with each other at a shared (u,v) point', () => {
  const srf = makeTestSurface();
  const u = (srf.knotsU[0] + srf.knotsU[srf.knotsU.length - 1]) * 0.37;
  const v = (srf.knotsV[0] + srf.knotsV[srf.knotsV.length - 1]) * 0.62;
  const fromU = curvePoint(extractIsocurveU(srf, u), v);
  const fromV = curvePoint(extractIsocurveV(srf, v), u);
  const direct = surfacePoint(srf, u, v);
  const d1 = Math.hypot(fromU[0] - direct[0], fromU[1] - direct[1], fromU[2] - direct[2]);
  const d2 = Math.hypot(fromV[0] - direct[0], fromV[1] - direct[1], fromV[2] - direct[2]);
  assert.ok(d1 < 1e-9 && d2 < 1e-9, `d1=${d1} d2=${d2}`);
});

test('extractIsocurveU/V produce a curve with no NaN/Infinity anywhere across the full extracted control net', () => {
  const srf = makeTestSurface();
  const isoU = extractIsocurveU(srf, srf.knotsU[3]);
  const isoV = extractIsocurveV(srf, srf.knotsV[3]);
  for (const crv of [isoU, isoV]) {
    for (const p of crv.ctrlPts) {
      assert.ok(p.every(Number.isFinite), JSON.stringify(p));
      assert.ok(p[3] > 0, 'weight must stay positive');
    }
  }
});

// ExtractBorder — a surface open in BOTH directions (this file's own
// rational stress surface: a 270deg arc profile, itself only revolved
// 270deg) gets all 4 borders, each bit-for-bit reproducing an independent
// extractIsocurveU/V call at the exact same domain boundary — a real
// cross-check against already-proven kernel machinery, not a self-
// consistency tautology, matching DivideSrf's own established technique.
test('extractBorderCurves: a surface open in both directions returns all 4 edges, each exactly matching an independent extractIsocurveU/V call at the same boundary', () => {
  const srf = makeTestSurface();
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false);
  assert.equal(closedV, false);
  const borders = extractBorderCurves(srf);
  assert.equal(borders.length, 4, 'all 4 parametric sides are genuinely free');
  assert.deepEqual(borders.map((b) => b.edge), ['uMin', 'uMax', 'vMin', 'vMax'], 'a stable, documented edge order');
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  assert.deepEqual(borders[0].crv, extractIsocurveU(srf, uMin));
  assert.deepEqual(borders[1].crv, extractIsocurveU(srf, uMax));
  assert.deepEqual(borders[2].crv, extractIsocurveV(srf, vMin));
  assert.deepEqual(borders[3].crv, extractIsocurveV(srf, vMax));
  for (const { crv } of borders) {
    for (const p of crv.ctrlPts) assert.ok(p.every(Number.isFinite), JSON.stringify(p));
  }
});

// A FULL revolve of an OPEN profile (a cylinder) wraps in V (the sweep),
// so the vMin/vMax boundary curves would be IDENTICAL (the same seam) —
// correctly excluded rather than returned as a duplicate pair. The two
// remaining uMin/uMax edges are the cylinder's own real top/bottom rims.
test('extractBorderCurves: a full revolve of an open profile (cylinder, closed in V) returns only the 2 naked uMin/uMax rim edges, never the seam', () => {
  const profile = makeLine([5, 0, 0], [5, 0, 20]);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false, 'the straight profile itself never wraps');
  assert.equal(closedV, true, 'a full 360deg revolve wraps the sweep direction into a seam');
  const borders = extractBorderCurves(srf);
  assert.equal(borders.length, 2);
  assert.deepEqual(borders.map((b) => b.edge), ['uMin', 'uMax']);
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  assert.deepEqual(borders[0].crv, extractIsocurveU(srf, uMin));
  assert.deepEqual(borders[1].crv, extractIsocurveU(srf, uMax));
  // Each rim is a genuine circle of radius 5 at its own fixed Z (0 and 20).
  for (const [i, expectedZ] of [[0, 0], [1, 20]]) {
    const { crv } = borders[i];
    for (let t = 0; t <= 1.0001; t += 0.1) {
      const v = srf.knotsV[0] + t * (srf.knotsV[srf.knotsV.length - 1] - srf.knotsV[0]);
      const p = curvePoint(crv, v);
      assert.ok(Math.abs(p[2] - expectedZ) < 1e-9, `expected Z=${expectedZ}, got ${p[2]}`);
      assert.ok(Math.abs(Math.hypot(p[0], p[1]) - 5) < 1e-9, `expected radius 5, got ${Math.hypot(p[0], p[1])}`);
    }
  }
});

// An extrude of a CLOSED profile (a circle) is a tube, closed in U (the
// profile itself), open in V (the extrude direction) — the mirror image
// of the cylinder case above: uMin/uMax would coincide (the tube's own
// vertical seam), so only the 2 real rim-circle edges (vMin/vMax) return.
test('extractBorderCurves: extrude of a closed profile (a tube, closed in U) returns only the 2 naked vMin/vMax rim edges, never the seam', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 4, 4);
  const srf = extrude(profile, [0, 0, 1], 15);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true, 'the circle profile itself already wraps');
  assert.equal(closedV, false, 'a straight extrude never wraps');
  const borders = extractBorderCurves(srf);
  assert.equal(borders.length, 2);
  assert.deepEqual(borders.map((b) => b.edge), ['vMin', 'vMax']);
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  assert.deepEqual(borders[0].crv, extractIsocurveV(srf, vMin));
  assert.deepEqual(borders[1].crv, extractIsocurveV(srf, vMax));
});

// A surface closed in BOTH directions (a torus: a full revolve of an
// already-closed circle profile) genuinely has NO naked edges at all —
// extractBorderCurves correctly returns an empty array, not a crash or a
// spurious seam pair, matching nakedEdgeCount's own identical 0 result
// for the same real geometric reason.
test('extractBorderCurves: a torus (closed in both U and V) has zero naked edges — an honestly empty array, not a bug', () => {
  const profile = makeCircle([5, 0, 0], [0, 0, 1], [1, 0, 0], 1, 4);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true);
  assert.equal(closedV, true);
  const borders = extractBorderCurves(srf);
  assert.deepEqual(borders, []);
});

// EXTRACTWIREFRAME — the batch EXTRACTISOCURVE command, reusing
// the Wireframe overlay's own Greville-abscissae density convention (see
// extractWireframeCurves' own header comment for the full reasoning). This
// app's rational stress surface again (open in both U and V), so the
// count is exactly ctrlNet.length U-rows + ctrlNet[0].length V-rows, no
// closed-seam special case to worry about in this first test.
test('extractWireframeCurves: total curve count is exactly ctrlNet.length (U rows) + ctrlNet[0].length (V rows) for a surface open in both directions', () => {
  const srf = makeTestSurface();
  const wires = extractWireframeCurves(srf);
  assert.equal(wires.length, srf.ctrlNet.length + srf.ctrlNet[0].length);
  assert.equal(wires.filter((w) => w.dir === 'U').length, srf.ctrlNet.length);
  assert.equal(wires.filter((w) => w.dir === 'V').length, srf.ctrlNet[0].length);
});

// Cross-check against DIRECTLY calling grevilleAbscissae + extractIsocurveU/V
// ourselves, curve-for-curve, bit-for-bit — not a self-consistency
// tautology, the same "prove it against already-proven kernel machinery"
// technique extractBorderCurves' own test above already uses.
test('extractWireframeCurves: every curve exactly matches an independent grevilleAbscissae + extractIsocurveU/V call at the same parameter', () => {
  const srf = makeTestSurface();
  const wires = extractWireframeCurves(srf);
  const uVals = grevilleAbscissae({ degree: srf.degU, knots: srf.knotsU, ctrlPts: new Array(srf.ctrlNet.length) });
  const vVals = grevilleAbscissae({ degree: srf.degV, knots: srf.knotsV, ctrlPts: new Array(srf.ctrlNet[0].length) });
  assert.deepEqual(uVals.map((u) => extractIsocurveU(srf, u)), wires.filter((w) => w.dir === 'U').map((w) => w.crv));
  assert.deepEqual(vVals.map((v) => extractIsocurveV(srf, v)), wires.filter((w) => w.dir === 'V').map((w) => w.crv));
  // The edges (first/last Greville value of a clamped knot vector) land
  // exactly on the domain boundary — the same "edges included for free"
  // property the Wireframe overlay's own comment already documents.
  assert.equal(uVals[0], srf.knotsU[0]);
  assert.equal(uVals[uVals.length - 1], srf.knotsU[srf.knotsU.length - 1]);
  assert.equal(vVals[0], srf.knotsV[0]);
  assert.equal(vVals[vVals.length - 1], srf.knotsV[srf.knotsV.length - 1]);
});

// Every extracted curve stays fully finite (no NaN/Infinity), matching
// extractIsocurveU/V's own existing proof, just swept across every row
// this batch command produces instead of one hand-picked parameter.
test('extractWireframeCurves: every curve in the batch is finite everywhere, no NaN/Infinity', () => {
  const srf = makeTestSurface();
  const wires = extractWireframeCurves(srf);
  assert.ok(wires.length > 0, 'a valid surface always has at least 2x2 control points, so this can never be empty');
  for (const { crv } of wires) {
    for (const p of crv.ctrlPts) {
      assert.ok(p.every(Number.isFinite), JSON.stringify(p));
      assert.ok(p[3] > 0, 'weight must stay positive');
    }
  }
});

// A cylinder (full revolve of an open profile, closed in V) is NOT
// filtered for the seam the way extractBorderCurves would filter it —
// this function deliberately matches the Wireframe OVERLAY's own
// convention (no closure filtering at all), so the closed direction's own
// first/last Greville values (both landing on the same physical seam
// row) both still appear, a real, named, already-accepted duplicate-seam
// cosmetic quirk, not a bug.
test('extractWireframeCurves: a closed-V cylinder does NOT filter the seam (matches the Wireframe overlay\'s own no-filtering convention, not ExtractBorder\'s stricter one)', () => {
  const profile = makeLine([5, 0, 0], [5, 0, 20]);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false);
  assert.equal(closedV, true);
  const wires = extractWireframeCurves(srf);
  assert.equal(wires.length, srf.ctrlNet.length + srf.ctrlNet[0].length, 'no seam-duplicate filtering — every U row and every V row is present, including the closed direction\'s own repeated end');
  // Every V-direction row (fixed v, running along U) is a real circle at
  // its own fixed height, exactly like the border rims already proven.
  for (const { dir, crv } of wires) {
    if (dir !== 'V') continue;
    for (let t = 0; t <= 1.0001; t += 0.25) {
      const u = srf.knotsU[0] + t * (srf.knotsU[srf.knotsU.length - 1] - srf.knotsU[0]);
      const p = curvePoint(crv, u);
      assert.ok(Math.abs(Math.hypot(p[0], p[1]) - 5) < 1e-9, `expected radius 5, got ${Math.hypot(p[0], p[1])}`);
    }
  }
});
