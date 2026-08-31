import test from 'node:test';
import assert from 'node:assert/strict';
import { curvePoint, curvePointAndTangent, closestPointOnCurve, isCurveClosed } from '../kernel/curve.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { makeLine, makeArc, revolve } from '../kernel/primitives.mjs';
import { extractIsocurveU } from '../kernel/isocurve.mjs';
import { joinCurvesC0 } from '../kernel/knots.mjs';
import {
  conformCurveToCurve,
  conformCurveToSurface,
  mapPointsBaseToTarget,
} from '../kernel/conform.mjs';

// GENUINELY CURVED, NON-TRIVIAL 3D fixtures (this project's house rule
// against trivial test geometry hiding bugs — never two straight lines).
// BASE: a gentle S-curve in the XY plane.
// TARGET: a different curve that also swings in Z, so the target's own
// parallel-transport frame genuinely rotates in 3D (not a planar special
// case) — the real stress on the frame-relative mapping.
function makeBaseCurve() {
  return globalCurveInterp([
    [0, 0, 0], [10, 8, 0], [22, -4, 0], [34, 6, 0], [46, 0, 0],
  ], 3);
}
function makeTargetCurve() {
  return globalCurveInterp([
    [0, 0, 5], [12, 5, 12], [24, -6, 6], [38, 3, 16], [50, 0, 9],
  ], 3);
}

test('mapPointsBaseToTarget: OFFSET-MAGNITUDE PRESERVATION — |mapped - C_target| equals |P - C_base| EXACTLY (orthonormal frame invariant)', () => {
  const base = makeBaseCurve();
  const target = makeTargetCurve();
  // A straight object line offset well off the base curve (its closest
  // distance to the base genuinely VARIES point to point along it).
  const obj = makeLine([-2, 14, 3], [48, 12, -4]);
  const pts = [];
  for (let t = 0; t <= 1.00001; t += 0.05) pts.push(curvePoint(obj, t));

  const { baseDistances, frameOffsetMagnitudes } = mapPointsBaseToTarget(base, target, pts);
  let worst = 0;
  for (let i = 0; i < pts.length; i++) {
    worst = Math.max(worst, Math.abs(frameOffsetMagnitudes[i] - baseDistances[i]));
  }
  // The load-bearing mathematical proof: the two orthonormal frames preserve
  // the offset magnitude to machine precision.
  assert.ok(worst < 1e-9, `frame-offset magnitude vs base distance diverge by ${worst} (must be ~0)`);
  // And the distances genuinely vary (not a degenerate all-equal fixture).
  const spread = Math.max(...baseDistances) - Math.min(...baseDistances);
  assert.ok(spread > 1, `base distances should genuinely vary along the object; spread was ${spread}`);
});

test('conformCurveToCurve: OFFSET-MAGNITUDE PRESERVATION also holds against the TARGET curve itself — each mapped point sits the same distance FROM the target curve as its source did FROM the base curve', () => {
  const base = makeBaseCurve();
  const target = makeTargetCurve();
  const obj = makeLine([-2, 11, 2], [48, 9, -3]);
  const res = conformCurveToCurve(base, target, obj);
  const { mapped, baseDistances } = res.conform;
  let worst = 0;
  for (let i = 0; i < mapped.length; i++) {
    const distToTarget = closestPointOnCurve(target, mapped[i]).distance;
    worst = Math.max(worst, Math.abs(distToTarget - baseDistances[i]));
  }
  // A small target-curvature correction is expected (the closest point on a
  // CURVED target isn't exactly the frame origin) — stays well under a
  // fraction of a mm on a ~50mm-span, gently-curved fixture.
  assert.ok(worst < 0.05, `mapped point's distance-from-target vs source's distance-from-base diverge by ${worst}mm (curvature correction only)`);
});

test('conformCurveToCurve: EXACT REPRODUCTION ON BASE — conforming the base curve itself produces points that lie EXACTLY on the target curve', () => {
  const base = makeBaseCurve();
  const target = makeTargetCurve();
  // The object IS the base curve — every sampled object point lies exactly
  // on the base, so its offset is (0,0,0) and it must map onto the target.
  const res = conformCurveToCurve(base, target, base);
  const { mapped } = res.conform;
  let worst = 0;
  for (const m of mapped) {
    worst = Math.max(worst, closestPointOnCurve(target, m).distance);
  }
  assert.ok(worst < 1e-6, `mapped base-curve points diverge from the target curve by ${worst} (must be ~0)`);
  // The refit curve through those points is a real, faithful reconstruction
  // of the target (interpolates the on-target points; deviates only minutely
  // between them, like any interpolation).
  let fitWorst = 0;
  for (let t = res.knots[0]; t <= res.knots[res.knots.length - 1] + 1e-9; t += (res.knots[res.knots.length - 1] - res.knots[0]) / 40) {
    fitWorst = Math.max(fitWorst, closestPointOnCurve(target, curvePoint(res, t)).distance);
  }
  assert.ok(fitWorst < 0.5, `refit curve deviates from the target by ${fitWorst}mm between samples (interpolation residual only — ~0.5% of the ~50mm span; the mapped POINTS themselves lie on the target to <1e-6, above)`);
});

test('conformCurveToCurve: a CURVED object (not a straight line) both reproduces on base and preserves offset magnitude', () => {
  const base = makeBaseCurve();
  const target = makeTargetCurve();
  // A curved object curve (its own arc, above the base plane).
  const obj = makeArc([22, 0, 8], [1, 0, 0], [0, 1, 0], 14, 0, Math.PI);
  const res = conformCurveToCurve(base, target, obj);
  const { mapped, baseDistances, frameOffsetMagnitudes } = res.conform;
  let worst = 0;
  for (let i = 0; i < mapped.length; i++) worst = Math.max(worst, Math.abs(frameOffsetMagnitudes[i] - baseDistances[i]));
  assert.ok(worst < 1e-9, `curved-object frame-offset magnitude diverges by ${worst}`);
  assert.ok(mapped.length >= 4, 'a curved object should resample to a real point set');
  assert.equal(res.degree, obj.degree, 'refit preserves the object curve degree by default');
});

test('conformCurveToSurface (MODE B): the implicit base isocurve genuinely matches a direct extractIsocurveU call, and the conform runs against it', () => {
  // A revolve surface (cylinder-ish): profile line revolved 360deg.
  const profile = makeLine([10, 0, 0], [10, 0, 40]);
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const uMid = srf.knotsU[0] + 0.5 * (srf.knotsU[srf.knotsU.length - 1] - srf.knotsU[0]);

  const target = makeTargetCurve();
  const obj = makeLine([8, 0, 5], [8, 0, 35]);
  const res = conformCurveToSurface(srf, uMid, 0, 'u', target, obj);

  // The extracted implicit base isocurve is bit-for-bit a direct
  // extractIsocurveU call on the same (u).
  const direct = extractIsocurveU(srf, uMid);
  assert.equal(res.baseIsocurve.degree, direct.degree);
  assert.deepEqual(res.baseIsocurve.knots, direct.knots);
  assert.deepEqual(res.baseIsocurve.ctrlPts, direct.ctrlPts);

  // And the mapped result preserves offset magnitude against that base too.
  const { baseDistances, frameOffsetMagnitudes } = res.conform;
  let worst = 0;
  for (let i = 0; i < baseDistances.length; i++) worst = Math.max(worst, Math.abs(frameOffsetMagnitudes[i] - baseDistances[i]));
  assert.ok(worst < 1e-9, `Mode B frame-offset magnitude diverges by ${worst}`);
});

// ----------------------------------------------------------------
// CLOSURE, CORNERS AND SAMPLE DENSITY
// ----------------------------------------------------------------

const DEG = 180 / Math.PI;
function unit(v) { const L = Math.hypot(v[0], v[1], v[2]); return [v[0] / L, v[1] / L, v[2] / L]; }
function angleBetween(a, b) {
  const [ax, ay, az] = unit(a), [bx, by, bz] = unit(b);
  return Math.acos(Math.max(-1, Math.min(1, ax * bx + ay * by + az * bz))) * DEG;
}
// The angle the curve turns THROUGH a parameter, measured as the one-sided
// chord directions either side of it. On a curve that is smooth there this
// goes to zero with h; at a genuine C0 corner it converges on the corner
// angle instead — so a small h separates "rounded away" from "kept".
function turnAngleAt(crv, u, h) {
  const p = curvePoint(crv, u), pm = curvePoint(crv, u - h), pp = curvePoint(crv, u + h);
  return angleBetween([p[0] - pm[0], p[1] - pm[1], p[2] - pm[2]], [pp[0] - p[0], pp[1] - p[1], pp[2] - p[2]]);
}
function seamTurnAngle(crv) {
  const u0 = crv.knots[0], u1 = crv.knots[crv.knots.length - 1];
  return angleBetween(curvePointAndTangent(crv, u1).tangent, curvePointAndTangent(crv, u0).tangent);
}

test('conformCurveToCurve: a CLOSED object curve comes back CLOSED — the seam keeps its tangent continuity instead of acquiring the open refit\'s clamped-end kink', () => {
  const base = makeBaseCurve();
  const target = makeTargetCurve();
  // A genuine closed curve with EXACT tangent continuity at its seam (a full
  // circle is rational degree 2, seam turn angle exactly 0) — so any turn in
  // the conformed result is manufactured by the refit, not inherited.
  const obj = makeArc([24, 6, 3], [1, 0, 0], [0, 1, 0], 9, 0, 2 * Math.PI);
  assert.ok(isCurveClosed(obj), 'fixture must be a closed object curve');
  assert.ok(seamTurnAngle(obj) < 1e-9, `fixture seam must be exactly tangent-continuous, was ${seamTurnAngle(obj)}deg`);

  // MEASURED AT SEVERAL SAMPLE DENSITIES, because the size of the defect
  // depends on density and the default is its most flattering case: an open
  // refit's clamped end reads its tangent off the points nearest the seam,
  // so crowding samples there hides most of the error. Refitting the same
  // mapped points OPEN turns in 0.15deg at the default 8 interior samples
  // per span but 4.03deg at 2 and 10.71deg at 1; refitting them CLOSED stays
  // at the closed interpolator's own wrap-approximation floor throughout
  // (0.08deg / 0.25deg / 0.65deg).
  for (const interiorSamplesPerSpan of [8, 2, 1]) {
    const res = conformCurveToCurve(base, target, obj, { interiorSamplesPerSpan });
    assert.ok(isCurveClosed(res, 1e-6), `the conformed curve must still close on itself (at ${interiorSamplesPerSpan} interior samples per span)`);
    const seam = seamTurnAngle(res);
    assert.ok(seam < 1.0, `closed conform seam kinks by ${seam}deg at ${interiorSamplesPerSpan} interior samples per span`);
  }
});

test('conformCurveToCurve: a genuine CORNER in the object curve survives the conform instead of being smoothed away', () => {
  const base = makeBaseCurve();
  const target = makeTargetCurve();
  // Two degree-3 curves joined C0 at a sharp angle: the join knot carries
  // full multiplicity, so this is a real corner and not a tight fillet.
  const A = globalCurveInterp([[2, 10, 0], [8, 13, 1], [14, 12, 0], [20, 8, 2]], 3);
  const B = globalCurveInterp([[20, 8, 2], [22, 2, 1], [26, -4, 0], [32, -8, 1]], 3);
  const obj = joinCurvesC0([A, B]);
  const objCorner = turnAngleAt(obj, 1, 1e-5);
  assert.ok(objCorner > 40, `fixture must carry a real corner, measured ${objCorner}deg`);

  const res = conformCurveToCurve(base, target, obj);
  // Locate the corner in the RESULT geometrically (no reliance on any new
  // field): the mapped image of the object's corner param, then the result
  // parameter nearest that point.
  const idx = res.conform.sampleParams.findIndex((u) => Math.abs(u - 1) < 1e-12);
  assert.ok(idx >= 0, 'the corner param must be one of the sampled stations');
  const mappedCorner = res.conform.mapped[idx];
  const at = closestPointOnCurve(res, mappedCorner);
  assert.ok(at.distance < 1e-6, `the conformed curve must still pass through the mapped corner (off by ${at.distance})`);
  const resCorner = turnAngleAt(res, at.u, 1e-5);
  assert.ok(resCorner > 30, `the corner was rounded away: object turns ${objCorner}deg, conformed result turns only ${resCorner}deg`);
});

test('conformCurveToCurve: the sample-density cap keeps the object curve\'s own KNOT STATIONS, decimating only the interior samples', () => {
  const base = makeBaseCurve();
  const target = makeTargetCurve();
  // Enough distinct knots that the cap genuinely bites: 33 interpolated
  // points give 29 interior knot values, and 31 stations at 8 interior
  // samples per span is 271 candidate params against a cap of 80.
  const pts = [];
  for (let i = 0; i <= 32; i++) {
    const t = i / 32;
    pts.push([2 + 42 * t, 9 * Math.sin(t * 5.5) * (0.25 + t), 3 * Math.cos(t * 7.5)]);
  }
  const obj = globalCurveInterp(pts, 3);
  const uMin = obj.knots[0], uMax = obj.knots[obj.knots.length - 1];
  const stations = [...new Set(obj.knots.filter((k) => k >= uMin && k <= uMax))].sort((a, b) => a - b);
  assert.ok(stations.length > 20, `fixture must have many knot stations, had ${stations.length}`);

  const res = conformCurveToCurve(base, target, obj);
  const sampled = res.conform.sampleParams;
  const missing = stations.filter((s) => !sampled.some((u) => Math.abs(u - s) < 1e-12));
  assert.deepEqual(missing, [], `the cap threw away ${missing.length} of ${stations.length} knot stations (kept ${stations.length - missing.length}) while keeping evenly-spaced interior samples`);
});
