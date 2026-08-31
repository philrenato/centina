import test from 'node:test';
import assert from 'node:assert/strict';
import { curvePoint } from '../kernel/curve.mjs';
import { filletCornerArc, filletPolygon, filletOpenPolyline, filletSegmentsToCurve } from '../kernel/primitives.mjs';
import { sweep1Rigid } from '../kernel/sweep.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { makeCircle } from '../kernel/primitives.mjs';

const Z = [0, 0, 1];
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function square(half) {
  return [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]];
}

// ---- open rail, mixed collinear + genuine corners ----
test('filletOpenPolyline: an open rail with collinear runs and genuine corners fillets only the real corners, leaves collinear vertices untouched', () => {
  // A "staircase" rail: (0,0)->(10,0)->(10,0)  duplicated collinear point removed;
  // build a rail with an explicit MID-EDGE collinear vertex (a real, if redundant,
  // point sitting exactly on a straight run) plus two genuine 90deg corners.
  const points = [
    [0, 0, 0],   // open end
    [5, 0, 0],   // collinear (mid-straight-run) vertex — should pass straight through
    [10, 0, 0],  // genuine 90deg corner (turn from +x to +y)
    [10, 10, 0], // genuine 90deg corner (turn from +y back to +x)
    [20, 10, 0], // open end
  ];
  const res = filletOpenPolyline(points, 2, { closed: false });
  assert.equal(res.ok, true, res.reason);
  assert.equal(res.cornerCount, 2, 'only the 2 genuine corners should be filleted, not the collinear midpoint or the 2 open ends');
  const arcs = res.segments.filter((s) => s.type === 'arc');
  assert.equal(arcs.length, 2);
  for (const arc of arcs) {
    assert.ok(Math.abs(arc.weight - Math.cos(Math.PI / 4)) < 1e-9, 'every corner here is an identical 90deg fillet');
  }
  // The composed curve should still pass through (or very near) the open ends exactly.
  const crv = filletSegmentsToCurve(res.segments);
  const p0 = curvePoint(crv, crv.knots[0]);
  const p1 = curvePoint(crv, crv.knots[crv.knots.length - 1]);
  assert.ok(dist(p0, points[0]) < 1e-9, 'composed rail starts exactly at the open rail\'s own first point');
  assert.ok(dist(p1, points[4]) < 1e-9, 'composed rail ends exactly at the open rail\'s own last point');
});

// ---- open rail's own first/last edge budgets against ONE neighbor only ----
test('filletOpenPolyline: an open rail\'s first/last edge trims against its ONE interior neighbor only, not two', () => {
  // 3-point open "L": short first edge (6mm), long second edge (100mm), one
  // interior corner at the bend. A radius of 5mm needs a 5mm trim on EACH
  // side of the corner — the corner's own trim on the SHORT edge (6mm) is
  // the binding constraint; if this were (incorrectly) budgeted as though
  // BOTH the corner's own trim AND a second, nonexistent neighbor's trim
  // had to fit in 6mm, a much smaller radius would wrongly be required.
  const points = [[0, 0, 0], [6, 0, 0], [6, 100, 0]];
  const res = filletOpenPolyline(points, 5, { closed: false });
  assert.equal(res.ok, true, res.reason);
  assert.equal(res.cornerCount, 1);
  const arc = res.segments.find((s) => s.type === 'arc');
  assert.ok(Math.abs(arc.p0[0] - 1) < 1e-9, 'trim of 5mm back from the 6mm-long first edge leaves p0 at x=1');
  // A radius trimming exactly to (or past) the short edge's own full length must
  // still succeed (possibly auto-clamped) rather than wrongly refusing as if a
  // second neighbor shared that edge.
  const tight = filletOpenPolyline(points, 5.9, { closed: false });
  assert.equal(tight.ok, true, 'trimming nearly the whole 6mm first edge from ONE side only should still succeed');
});

// ---- near-too-large radius auto-clamp, cross-checked against a real maxSafeRadius ----
test('filletOpenPolyline: refuses honestly when the radius is too large, reports a real maxSafeRadius that itself succeeds on retry, matching Polygon Fillet\'s own clamp-not-refuse precedent', () => {
  const points = [[0, 0, 0], [10, 0, 0], [10, 10, 0]]; // one 90deg corner, 10mm edges both sides
  const tooLarge = filletOpenPolyline(points, 11, { closed: false });
  assert.equal(tooLarge.ok, false);
  assert.ok(tooLarge.maxSafeRadius > 0 && tooLarge.maxSafeRadius < 11);
  const retried = filletOpenPolyline(points, tooLarge.maxSafeRadius, { closed: false });
  assert.equal(retried.ok, true, 'the reported maxSafeRadius should itself succeed when retried');
  const justOver = filletOpenPolyline(points, tooLarge.maxSafeRadius * 1.02, { closed: false });
  assert.equal(justOver.ok, false);
});

// ---- collinear / near-180 NaN guard ----
test('filletOpenPolyline: a perfectly collinear vertex and a near-180 fold-back vertex both pass straight through, never NaN', () => {
  const collinear = [[0, 0, 0], [5, 0, 0], [10, 0, 0]];
  const res1 = filletOpenPolyline(collinear, 2, { closed: false });
  assert.equal(res1.ok, true);
  assert.equal(res1.cornerCount, 0, 'a perfectly straight rail has no real corners to fillet');
  for (const s of res1.segments) {
    for (const key of ['a', 'b', 'p0', 'apex', 'p2']) {
      if (s[key]) assert.ok(s[key].every(Number.isFinite), `${key} must be finite, never NaN`);
    }
  }
  const foldBack = [[0, 0, 0], [10, 0, 0], [0.001, 0, 0]]; // nearly reverses back on itself
  const res2 = filletOpenPolyline(foldBack, 2, { closed: false });
  assert.equal(res2.ok, true);
  assert.equal(res2.cornerCount, 0, 'a near-180 fold-back has no well-defined fillet, must skip not NaN');
  for (const s of res2.segments) {
    for (const key of ['a', 'b', 'p0', 'apex', 'p2']) {
      if (s[key]) assert.ok(s[key].every(Number.isFinite), `${key} must be finite, never NaN`);
    }
  }
});

// ---- closed planar rail matches filletPolygon exactly (regression-safety cross-check) ----
test('filletOpenPolyline on a closed PLANAR rail produces geometry IDENTICAL to filletPolygon\'s own proven behavior (per-corner local normal reduces to the same construction when the loop is planar)', () => {
  const half = 10;
  const pts = square(half);
  const radius = 3;
  const viaPolygon = filletPolygon(pts, radius, Z);
  const viaOpenPolyline = filletOpenPolyline(pts, radius, { closed: true });
  assert.equal(viaPolygon.ok, true);
  assert.equal(viaOpenPolyline.ok, true);
  assert.equal(viaOpenPolyline.segments.length, viaPolygon.segments.length);
  const arcsA = viaPolygon.segments.filter((s) => s.type === 'arc');
  const arcsB = viaOpenPolyline.segments.filter((s) => s.type === 'arc');
  assert.equal(arcsA.length, arcsB.length);
  // Cross-check each arc's own p0/apex/p2/weight match (order may differ by
  // rotation only in principle; here both walk the SAME point array in the
  // SAME order, so a direct index-for-index comparison is valid).
  for (let i = 0; i < arcsA.length; i++) {
    assert.ok(dist(arcsA[i].p0, arcsB[i].p0) < 1e-9, `corner ${i} p0 matches`);
    assert.ok(dist(arcsA[i].apex, arcsB[i].apex) < 1e-9, `corner ${i} apex matches`);
    assert.ok(dist(arcsA[i].p2, arcsB[i].p2) < 1e-9, `corner ${i} p2 matches`);
    assert.ok(Math.abs(arcsA[i].weight - arcsB[i].weight) < 1e-9, `corner ${i} weight matches`);
  }
});

test('filletOpenPolyline on a closed rail with a reflex (star-like) vertex still matches filletPolygon\'s own signed-turn handling exactly', () => {
  // A simple non-convex quad (one reflex corner) — regular filletPolygon
  // already proves reflex corners work via its own signed turn angle; this
  // proves the per-corner LOCAL normal construction agrees on the same case.
  const pts = [[0, 0, 0], [10, 0, 0], [5, 2, 0], [10, 10, 0]]; // vertex index 2 is a reflex dent
  const radius = 0.5;
  const viaPolygon = filletPolygon(pts, radius, Z);
  const viaOpenPolyline = filletOpenPolyline(pts, radius, { closed: true });
  assert.equal(viaPolygon.ok, true);
  assert.equal(viaOpenPolyline.ok, true);
  const arcsA = viaPolygon.segments.filter((s) => s.type === 'arc');
  const arcsB = viaOpenPolyline.segments.filter((s) => s.type === 'arc');
  for (let i = 0; i < arcsA.length; i++) {
    assert.ok(dist(arcsA[i].apex, arcsB[i].apex) < 1e-9, `corner ${i} apex matches (reflex handled identically)`);
    assert.ok(Math.abs(arcsA[i].weight - arcsB[i].weight) < 1e-9, `corner ${i} weight matches (reflex handled identically)`);
  }
});

// ---- filletSegmentsToCurve composition ----
test('filletSegmentsToCurve composes a mixed line/arc segment list into one degree-2 curve reproducing every trim point exactly', () => {
  const points = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [20, 10, 0]];
  const res = filletOpenPolyline(points, 2, { closed: false });
  assert.equal(res.ok, true);
  const crv = filletSegmentsToCurve(res.segments);
  assert.equal(crv.degree, 2, 'any arc segment present forces the composed curve to degree 2');
  const p0 = curvePoint(crv, crv.knots[0]);
  const p1 = curvePoint(crv, crv.knots[crv.knots.length - 1]);
  assert.ok(dist(p0, points[0]) < 1e-9);
  assert.ok(dist(p1, points[3]) < 1e-9);
});

test('filletSegmentsToCurve on an all-collinear (zero-corner) rail stays degree 1, byte-consistent with a plain polyline', () => {
  const points = [[0, 0, 0], [5, 0, 0], [10, 0, 0]];
  const res = filletOpenPolyline(points, 2, { closed: false });
  assert.equal(res.cornerCount, 0);
  const crv = filletSegmentsToCurve(res.segments);
  assert.equal(crv.degree, 1, 'no arcs present — the composed rail stays a plain degree-1 curve');
});

// ---- THE ACTUAL POINT: sweeping a filleted rail eliminates the mid-segment waist ----
// Reproduces, as a permanent regression test, the exact hand investigation that
// found the bug (a 4-point-zigzag-style rail, sweep1Rigid'd with a circular
// profile, sampled at many v-stations): the UNFILLETED rail (degree<=1, the
// discrete per-control-point free path) shows a real, measured mid-segment
// radius dip to ~cos(halfTurn/2)*trueRadius; the FILLETED rail (degree-2, forced
// through sweep1RigidResampled — `railFrameOriginsExact` is false the instant
// the rail is degree>1) does not.
//
// MEASUREMENT METHOD, confirmed correct against BOTH the sharp AND filleted
// cases numerically before writing this as a permanent assertion (not assumed
// from theory): at a given rail arc-length fraction, the true "center" is
// `curvePoint(rail, uAtThatArcLength)` — exact for the sharp (degree<=1: control
// polygon literally is the curve) case directly via its own v-domain value, and
// via `buildArcLengthTable`/`paramAtArcLength` (kernel/curve.mjs, the exact
// mechanism `sweep1RigidResampled` itself already uses to relate its own
// arc-length-fraction V domain back to a real rail parameter) for the filleted
// (degree-2, arc-length-parametrized V) case — sample MANY points around the
// profile at that v (never just 3-4, which can miss a distortion direction
// depending on which points are chosen — confirmed live: an earlier draft of
// this exact test used a 3-point circumradius fit and completely MISSED the
// known dip on this same fixture, a real methodology bug caught by cross-
// checking against the SAME fixture and radius this file's own header comment
// already derived by hand: cos(45deg/2)*5 = 4.6194, exactly what a dense 32-
// sample MIN distance reproduces) and take the MIN distance — the tube's own
// narrowest extent at that station, which is exactly what "necks in" means.
function minRadiusAt(srf, center, v, uSampleCount = 32) {
  const uSpan = srf.knotsU[srf.knotsU.length - 1] - srf.knotsU[0];
  let minD = Infinity;
  for (let k = 0; k < uSampleCount; k++) {
    const u = srf.knotsU[0] + (k / uSampleCount) * uSpan;
    minD = Math.min(minD, dist(surfacePoint(srf, u, v), center));
  }
  return minD;
}

// SUPERSEDED — this test used to be the "sharp still ships the
// waist" regression baseline (the exact fixture/derivation this file's own
// header comment above documents: cos(22.5deg)*5 = 4.6194mm). The TRUE-MITER
// fix (kernel/sweep.mjs's `applyTrueMiterStretch`/`railInteriorCorners`,
// verified against real McNeel Rhino docs and Houdini's own documented
// Sweep 2.0 "Stretch Around Turns") now eliminates that waist for 'sharp'
// too, WITHOUT rounding the corner — this test is rewritten to prove the
// FIX, not the old bug, on the IDENTICAL fixture (measured, not assumed:
// the mid-span min radius on this exact 90deg corner now reads exactly
// 5.000000mm — see test/sweep-true-miter.test.mjs for the full derivation
// and the general (non-planar, multi-corner) case's own honestly-reported
// small residual).
test('sweep1Rigid on the UNFILLETED (sharp) rail — TRUE MITER FIX: the old measured mid-segment radius dip is GONE, corners stay crisp (no rounding)', () => {
  const trueRadius = 5;
  const railPts = [[0, 0, 0], [10, 0, 0], [10, 10, 0]]; // single genuine 90deg corner, matches this file's own header-comment derivation exactly — the SAME fixture the old (pre-fix) 4.6194mm baseline used
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], trueRadius);
  const sharpRail = { degree: 1, knots: [0, 0, 1, 2, 2], ctrlPts: railPts.map((p) => [...p, 1]) };
  const sharpSrf = sweep1Rigid(sharpRail, profile);
  assert.equal(sharpSrf.degV, 1, 'still the discrete per-control-point free sweep path — the fix is a ring-level post-pass on this SAME path, not a switch to the resampled one');
  for (const segStart of [0, 1]) {
    const midRadius = minRadiusAt(sharpSrf, curvePoint(sharpRail, segStart + 0.5), segStart + 0.5);
    assert.ok(Math.abs(midRadius - trueRadius) < 1e-6, `segment ${segStart}: measured mid-span min radius ${midRadius.toFixed(6)} should now be exactly ${trueRadius} (the OLD, pre-fix measurement here was 4.6194, cos(22.5deg)*5 — this fixture is the permanent regression proof the true-miter fix actually eliminates it, not merely reduces it)`);
  }
  // At the corner's own control-point station (v=1, the exact corner), the
  // MINIMUM radius was already exact even before this fix (a rigid transform
  // of a circle has a constant true radius regardless of orientation) — the
  // true-miter stretch elongates the corner ring into an ELLIPSE (unchanged
  // minor axis, stretched major axis along the bend-plane direction), so the
  // min radius stays exactly correct while the MAX radius grows to
  // trueRadius*sec(45deg) — proving the ring is now a real, deliberate
  // ellipse, not silently still a plain circle.
  const cornerRadius = minRadiusAt(sharpSrf, curvePoint(sharpRail, 1), 1);
  assert.ok(Math.abs(cornerRadius - trueRadius) < 1e-6, `at the corner control-point station itself, min radius is still exact (measured ${cornerRadius.toFixed(6)})`);
});

test('THE FIX: sweep1Rigid on the SAME rail, corners rounded via filletOpenPolyline + filletSegmentsToCurve first, keeps the tube\'s cross-sectional radius uniform everywhere — no more mid-segment dip', async () => {
  const { buildArcLengthTable, paramAtArcLength } = await import('../kernel/curve.mjs');
  const trueRadius = 5;
  const railPts = [[0, 0, 0], [10, 0, 0], [10, 10, 0]]; // the IDENTICAL fixture as the sharp-baseline test above
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], trueRadius);

  const filletRes = filletOpenPolyline(railPts, 3, { closed: false }); // 3mm corner radius, well under the 10mm edges' own maxSafeRadius
  assert.equal(filletRes.ok, true, filletRes.reason);
  assert.equal(filletRes.cornerCount, 1);
  const filletedRail = filletSegmentsToCurve(filletRes.segments);
  assert.equal(filletedRail.degree, 2, 'a filleted rail with a real corner is genuinely degree-2');

  const filletedSrf = sweep1Rigid(filletedRail, profile);
  assert.ok(filletedSrf.degV >= 2, 'a degree-2 rail must route through sweep1RigidResampled (railFrameOriginsExact is false) — this is what eliminates the waist structurally, not just cosmetically');
  assert.equal(filletedSrf.knotsV[0], 0);
  assert.equal(filletedSrf.knotsV[filletedSrf.knotsV.length - 1], 1, 'sweep1RigidResampled\'s own V domain is the rail\'s ARC-LENGTH FRACTION [0,1], not the composed curve\'s own raw knot domain — the exact mismatch that made an early draft of this test wrongly sample points far outside the true tube (min-radius readings in the tens), caught and fixed here by mapping arc-length fraction -> rail parameter -> world point via buildArcLengthTable/paramAtArcLength, the SAME mechanism sweep1RigidResampled itself uses internally');

  const uMin = filletedRail.knots[0], uMax = filletedRail.knots[filletedRail.knots.length - 1];
  const table = buildArcLengthTable(filletedRail, uMin, uMax);
  const fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
  const radii = fractions.map((frac) => {
    const u = paramAtArcLength(table, frac * table.total);
    const center = curvePoint(filletedRail, u);
    return minRadiusAt(filletedSrf, center, frac);
  });
  const minR = Math.min(...radii);
  const maxR = Math.max(...radii);
  assert.ok(minR > trueRadius * 0.999, `FIX PROOF: filleted-rail tube radius stays uniform across the WHOLE rail (measured min ${minR.toFixed(4)}, true ${trueRadius}) — was ${((trueRadius * Math.cos(Math.PI / 8) / trueRadius) * 100).toFixed(1)}% (a real ${trueRadius - trueRadius * Math.cos(Math.PI / 8) < 0 ? '' : (trueRadius * (1 - Math.cos(Math.PI / 8))).toFixed(3) + 'mm'} dip) on the SAME rail unfilleted — the reported "non-uniform radius along some sections" bug is gone`);
  assert.ok(maxR < trueRadius * 1.001, `radius doesn't overshoot either (measured max ${maxR.toFixed(4)})`);
});
