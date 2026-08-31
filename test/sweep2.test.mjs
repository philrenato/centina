import test from 'node:test';
import assert from 'node:assert/strict';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';
import { curvePoint, buildArcLengthTable, paramAtArcLength } from '../kernel/curve.mjs';
import { sweep2, localizeSectionToFrame } from '../kernel/sweep.mjs';

const line = (p0, p1) => ({ degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...p0, 1], [...p1, 1]] });

// Two GENUINELY DIFFERENT curving rails (not straight lines — per this
// project's own logged "too-simple test geometry can hide real bugs"
// lesson: a straight-rail fixture can't distinguish a correct width-axis-
// per-station construction from a naive single-frame shortcut). Both built
// via real globalCurveInterp, different shapes, different curvature.
function railA() {
  return globalCurveInterp([[0, 0, 0], [50, 30, 0], [100, 10, 0], [150, 40, 0]], 3);
}
function railB() {
  return globalCurveInterp([[0, 0, 60], [50, 10, 60], [100, 55, 60], [150, 20, 60]], 3);
}

// A profile that is NOT degenerate — its own start/end are 10 units apart
// along X, and its middle control point is pulled 3 units out of the
// straight start-end line (a genuine "depth"/perpendicular extent to
// track), symmetric (equal chord lengths either side of the mid point, so
// globalCurveInterp's own chord-length parametrization puts the mid data
// point at EXACTLY u=0.5 — deliberately chosen so this test can predict
// exactly which resampled row corresponds to it, not guess).
function profileBow() {
  return globalCurveInterp([[0, 0, 0], [5, 3, 0], [10, 0, 0]], 2);
}

test('sweep2: reproduces the profile\'s own two endpoints EXACTLY at both rails, at EVERY station (independent cross-check against the rails\' own real curvePoint, not a self-consistency tautology)', () => {
  const rail1 = railA(), rail2 = railB();
  const profile = profileBow();
  const srf = sweep2(rail1, rail2, profile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  assert.ok(srf.vStations.length >= 4, 'should produce a genuinely dense station set');

  // Independent ground truth: fresh arc-length tables built directly in
  // this test, not read off srf's own internals.
  const u1min = rail1.knots[0], u1max = rail1.knots[rail1.knots.length - 1];
  const u2min = rail2.knots[0], u2max = rail2.knots[rail2.knots.length - 1];
  const table1 = buildArcLengthTable(rail1, u1min, u1max);
  const table2 = buildArcLengthTable(rail2, u2min, u2max);
  const len1 = table1.total, len2 = table2.total;

  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  let maxErrRail1 = 0, maxErrRail2 = 0;
  for (const v of srf.vStations) {
    const expected1 = curvePoint(rail1, paramAtArcLength(table1, v * len1));
    const expected2 = curvePoint(rail2, paramAtArcLength(table2, v * len2));
    const got1 = surfacePoint(srf, uMin, v);
    const got2 = surfacePoint(srf, uMax, v);
    maxErrRail1 = Math.max(maxErrRail1, Math.hypot(got1[0] - expected1[0], got1[1] - expected1[1], got1[2] - expected1[2]));
    maxErrRail2 = Math.max(maxErrRail2, Math.hypot(got2[0] - expected2[0], got2[1] - expected2[1], got2[2] - expected2[2]));
  }
  assert.ok(maxErrRail1 < 1e-6, `u=uMin edge should reproduce rail 1 exactly at every station, max error ${maxErrRail1}`);
  assert.ok(maxErrRail2 < 1e-6, `u=uMax edge should reproduce rail 2 exactly at every station, max error ${maxErrRail2}`);
});

test('sweep2: works with two rails of DIFFERENT type/degree (a straight Line + a genuinely curving degree-3 rail) — never needs a shared knot vector or curve type', () => {
  const rail1 = line([0, 0, 0], [150, 0, 0]); // degree 1
  const rail2 = railB(); // degree 3, curving
  const profile = profileBow();
  const srf = sweep2(rail1, rail2, profile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);

  const u2min = rail2.knots[0], u2max = rail2.knots[rail2.knots.length - 1];
  const table2 = buildArcLengthTable(rail2, u2min, u2max);
  const len2 = table2.total;
  const uMax = srf.knotsU[srf.knotsU.length - 1];
  let maxErr = 0;
  for (const v of srf.vStations) {
    const expected = curvePoint(rail2, paramAtArcLength(table2, v * len2));
    const got = surfacePoint(srf, uMax, v);
    maxErr = Math.max(maxErr, Math.hypot(got[0] - expected[0], got[1] - expected[1], got[2] - expected[2]));
  }
  assert.ok(maxErr < 1e-6, `mixed-degree rails: u=uMax edge should still reproduce rail 2 exactly, max error ${maxErr}`);
});

test('sweep2: the profile\'s own width scaling genuinely VARIES station-to-station when the rails\' separation varies (not a fixed/frozen scale)', () => {
  const rail1 = railA(), rail2 = railB();
  const profile = profileBow();
  const srf = sweep2(rail1, rail2, profile);

  const u1min = rail1.knots[0], u1max = rail1.knots[rail1.knots.length - 1];
  const u2min = rail2.knots[0], u2max = rail2.knots[rail2.knots.length - 1];
  const table1 = buildArcLengthTable(rail1, u1min, u1max);
  const table2 = buildArcLengthTable(rail2, u2min, u2max);
  const len1 = table1.total, len2 = table2.total;

  // Recompute the true rail-to-rail width at every station INDEPENDENTLY
  // (not merely reading srf.widths) and confirm it genuinely varies.
  const widths = srf.vStations.map((v) => {
    const p1 = curvePoint(rail1, paramAtArcLength(table1, v * len1));
    const p2 = curvePoint(rail2, paramAtArcLength(table2, v * len2));
    return Math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
  });
  const minW = Math.min(...widths), maxW = Math.max(...widths);
  assert.ok(maxW - minW > 1, `width should vary meaningfully station-to-station (these two rails genuinely diverge/converge), got min=${minW.toFixed(4)} max=${maxW.toFixed(4)}`);
  // Cross-check srf's own exposed widths match this independent recompute.
  for (let i = 0; i < widths.length; i++) {
    assert.ok(Math.abs(widths[i] - srf.widths[i]) < 1e-6, `srf.widths[${i}] should match independently recomputed rail separation`);
  }
});

test('sweep2: the profile\'s own perpendicular ("depth") extent scales PROPORTIONALLY with the width scale — the stated v1 scale decision, a real checkable proof not an assumption', () => {
  const rail1 = railA(), rail2 = railB();
  const profile = profileBow();
  const uSampleCount = 25; // chosen so t=12/24=0.5 lands EXACTLY on the profile's own symmetric mid data point
  const srf = sweep2(rail1, rail2, profile, { uSampleCount });
  assert.ok(Array.isArray(srf.ubar) && srf.ubar.length === uSampleCount, 'ubar should be exposed, one entry per U sample, for exactly this kind of independent verification');

  // Independently predict the local (frame-relative) offset of the
  // profile's own mid point, the SAME way sweep2 itself derives it —
  // localizeSectionToFrame is separately, independently tested elsewhere
  // (test/sweep-nprofiles.test.mjs's own frame-decomposition precedent);
  // reusing it here to compute an INDEPENDENT expected value is the same
  // "reconstruct ground truth via the kernel's own already-proven primitive,
  // not the function under test's own internal numbers" technique this
  // project already established (the Bug1 repro's own
  // chordLengthParams reconstruction).
  const start = profile.ctrlPts[0], end = profile.ctrlPts[profile.ctrlPts.length - 1];
  const width0 = Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
  const xAxisP = [(end[0] - start[0]) / width0, (end[1] - start[1]) / width0, (end[2] - start[2]) / width0];
  const ref = Math.abs(xAxisP[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const rawY = [xAxisP[1] * ref[2] - xAxisP[2] * ref[1], xAxisP[2] * ref[0] - xAxisP[0] * ref[2], xAxisP[0] * ref[1] - xAxisP[1] * ref[0]];
  const yLen = Math.hypot(rawY[0], rawY[1], rawY[2]);
  const yAxisP = [rawY[0] / yLen, rawY[1] / yLen, rawY[2] / yLen];
  const zAxisP = [xAxisP[1] * yAxisP[2] - xAxisP[2] * yAxisP[1], xAxisP[2] * yAxisP[0] - xAxisP[0] * yAxisP[2], xAxisP[0] * yAxisP[1] - xAxisP[1] * yAxisP[0]];
  const localized = localizeSectionToFrame(profile, { origin: [start[0], start[1], start[2]], xAxis: xAxisP, yAxis: yAxisP, zAxis: zAxisP });
  const midLocal = curvePoint(localized, 0.5); // profile's own symmetric mid station, by construction
  const depthLocal = Math.hypot(midLocal[1], midLocal[2]); // magnitude perpendicular to the width axis, expected 3 by construction
  assert.ok(Math.abs(depthLocal - 3) < 1e-6, `sanity: the profile's own bow depth should be exactly 3 in its local frame, got ${depthLocal}`);

  const u1min = rail1.knots[0], u1max = rail1.knots[rail1.knots.length - 1];
  const u2min = rail2.knots[0], u2max = rail2.knots[rail2.knots.length - 1];
  const table1 = buildArcLengthTable(rail1, u1min, u1max);
  const table2 = buildArcLengthTable(rail2, u2min, u2max);
  const len1 = table1.total, len2 = table2.total;

  // For every station, the surface's own sample at (ubar[12], v) — the row
  // predicted to correspond to the profile's own mid point (t=12/24=0.5) —
  // should sit at perpendicular distance depthLocal*(width(v)/width0) from
  // the R1(v)-R2(v) line, PROPORTIONAL to that station's own width, not a
  // fixed physical depth.
  const uMid = srf.ubar[12];
  const ratios = [];
  for (const v of srf.vStations) {
    const p1 = curvePoint(rail1, paramAtArcLength(table1, v * len1));
    const p2 = curvePoint(rail2, paramAtArcLength(table2, v * len2));
    const width = Math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
    const xAxisWorld = [(p2[0] - p1[0]) / width, (p2[1] - p1[1]) / width, (p2[2] - p1[2]) / width];
    const sample = surfacePoint(srf, uMid, v);
    const rel = [sample[0] - p1[0], sample[1] - p1[1], sample[2] - p1[2]];
    const along = rel[0] * xAxisWorld[0] + rel[1] * xAxisWorld[1] + rel[2] * xAxisWorld[2];
    const perp = [rel[0] - along * xAxisWorld[0], rel[1] - along * xAxisWorld[1], rel[2] - along * xAxisWorld[2]];
    const perpDist = Math.hypot(perp[0], perp[1], perp[2]);
    ratios.push(perpDist / width);
  }
  const expectedRatio = depthLocal / width0;
  const maxRatioErr = Math.max(...ratios.map((r) => Math.abs(r - expectedRatio)));
  assert.ok(maxRatioErr < 1e-3, `perpendicular depth / width should be the SAME constant ratio at every station (proportional scaling, not fixed depth), expected ${expectedRatio.toFixed(6)}, max deviation ${maxRatioErr.toFixed(6)}`);
  // And confirm this ratio is genuinely non-trivial (the depth really does
  // move as width changes, not merely "both happen to be constant").
  const widths = srf.vStations.map((v) => {
    const p1 = curvePoint(rail1, paramAtArcLength(table1, v * len1));
    const p2 = curvePoint(rail2, paramAtArcLength(table2, v * len2));
    return Math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
  });
  assert.ok(Math.max(...widths) - Math.min(...widths) > 1, 'setup: width genuinely varies across stations in this fixture too');
});

test('sweep2 honestly refuses when the two rails cross/coincide at a shared station (an undefined width axis), rather than producing a degenerate/garbage surface', () => {
  // Two straight lines that cross exactly at their own shared arc-length
  // fraction v=0.5 (both run uniformly in arc-length from t=0 to t=1, so
  // the fraction IS the linear parameter here) — deliberately picking
  // stationSamplesPerSpan=9 so v=0.5 (5/10) is an EXACT dense station, not
  // a near-miss that could pass by luck.
  const rail1 = line([0, 0, 0], [100, 0, 0]);
  const rail2 = line([0, 10, 0], [100, -10, 0]); // crosses rail1's Y=0 line exactly at x=50, v=0.5
  const profile = profileBow();
  assert.throws(
    () => sweep2(rail1, rail2, profile, { stationSamplesPerSpan: 9 }),
    /coincide \(or cross\)/,
    'should throw a real, named refusal naming the coincide/cross case'
  );
});

test('sweep2 honestly refuses a profile whose own start/end coincide (no defined rail-anchor width axis)', () => {
  const rail1 = railA(), rail2 = railB();
  const closedProfile = {
    degree: 1,
    knots: [0, 0, 0.5, 1, 1],
    ctrlPts: [[5, 0, 0, 1], [5, 3, 0, 1], [5, 0, 0, 1]], // start === end
  };
  assert.throws(
    () => sweep2(rail1, rail2, closedProfile),
    /distinct start\/end points/,
    'should throw a real, named refusal naming the coincident-endpoints case'
  );
});
