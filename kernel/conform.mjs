// CONFORM — map/bend a set of object curves so they follow a TARGET
// reference curve instead of the BASE reference curve they were built
// around (Rhino's own Flow command, exactly). CONFORM has two modes:
// Mode A (curve-to-curve) and Mode B
// (curve-to-surface-isocurve) BUILD; Mode C (surface-to-surface) DEFERRED
// (needs least-squares NURBS refit machinery this kernel doesn't have yet —
// mapping raw control points through a nonlinear surface map does not
// correctly map a curve between two surfaces).
//
// THE OPERATION (Mode A), stated precisely so the two provable properties
// below are unambiguous. For each point P of an object curve:
//   (1) find its CLOSEST point C_b = C_base(u_b) on the BASE curve
//       (closestPointOnCurve — already proven/shipped), express u_b as an
//       arc-length FRACTION f along the base curve's own length, and record
//       the OFFSET P - C_b decomposed into (a, b, c) in the base curve's own
//       parallel-transport frame AT u_b (buildParallelTransportFrames — the
//       exact machinery Sweep1 already uses: a continuously-varying
//       orthonormal frame along a curve);
//   (2) reconstruct the mapped point by looking up the SAME fraction f on
//       the TARGET curve (u_t = its own param at that arc-length fraction),
//       reading the target's own parallel-transport frame there, and
//       re-applying the SAME local offset (a, b, c) in the TARGET frame:
//       C_target(u_t) + a·xAxis_t + b·yAxis_t + c·zAxis_t.
//
// TWO PROVABLE PROPERTIES (verified numerically in test/conform.test.mjs on
// real CURVED base/target fixtures, never two straight lines):
//   - EXACT REPRODUCTION ON BASE: an object point that lies EXACTLY ON the
//     base curve has offset (a,b,c) ≈ (0,0,0), so it maps to C_target(u_t) —
//     a point exactly on the target curve. So conforming the base curve
//     itself reproduces the target curve. (Frame-INDEPENDENT: holds by
//     construction regardless of how the two transport frames are seeded.)
//   - OFFSET-MAGNITUDE PRESERVATION: because both frames are ORTHONORMAL,
//     |mapped - C_target| = sqrt(a²+b²+c²) = |P - C_base| EXACTLY. The local
//     offset MAGNITUDE (a point's distance from its reference curve) is
//     preserved even though the offset DIRECTION rotates with the frame.
//     (Additionally, since the closest point makes P-C_b ⊥ the base tangent,
//     c ≈ 0, so the mapped point's own closest-distance to the TARGET curve
//     also matches its original closest-distance to the BASE curve, up to a
//     small target-curvature correction — reported, not over-tightened.)
//
// The mapped POINT SET is refit into a fresh NURBS curve by interpolation —
// the SAME "resample, map continuously, refit" pattern sweepNProfiles/loft
// already use, not a new one. THREE PROPERTIES OF THE INPUT SURVIVE THAT
// REFIT, because a single open interpolation through the whole point set
// destroys all three:
//   - CLOSURE. A closed object curve is refit with closedCurveInterp, whose
//     cyclic wrap gives real tangent continuity across the seam; a clamped
//     open interpolation leaves a kink there instead. On a conformed circle
//     the open refit turns 0.15deg at the default sample density and 10.7deg
//     at one interior sample per span, where the closed refit stays at the
//     wrap approximation's own floor (0.08deg and 0.65deg). The defect hides
//     at high density precisely because a clamped end reads its tangent off
//     the samples nearest the seam.
//   - CORNERS. An interior knot at full multiplicity MAY carry a genuine
//     tangent break, and an open interpolation through sampled points cannot
//     know that and smooths it away. The curve is split at its real corners,
//     each run is conformed on its own, and the runs are chained with
//     joinCurvesC0 — the knot multiplicity the join leaves at each corner is
//     what makes the corner sharp. This is contourToCurve's (kernel/text.mjs)
//     split-fit-join shape, applied to the same problem.
//     ⚠ MULTIPLICITY ALONE IS NOT A CORNER: a rational arc/circle carries
//     interior knots at multiplicity == degree and is perfectly smooth
//     across them. Every candidate is confirmed by comparing the one-sided
//     chord directions either side of it before it is treated as a corner.
//   - DETAIL. The density cap is spent on the interior samples, never on the
//     curve's own knot stations, so a curve whose detail is concentrated in a
//     few dense spans keeps those spans rather than being decimated evenly
//     into its flat parts.

import { add, sub, scale, dot, length } from './vec3.mjs';
import {
  curvePoint,
  closestPointOnCurve,
  buildArcLengthTable,
  arcLengthAtParam,
  paramAtArcLength,
  adaptiveArcLengthSamples,
  isCurveClosed,
} from './curve.mjs';
import { globalCurveInterp, closedCurveInterp } from './interpolate.mjs';
import { extractSubCurve, joinCurvesC0, rescaleCurveDomain } from './knots.mjs';
import { buildParallelTransportFrames } from './sweep.mjs';
import { extractIsocurveU, extractIsocurveV } from './isocurve.mjs';

// Default number of extra samples inserted per object-curve knot span (on
// top of the curve's own distinct knot values) before mapping+refitting —
// enough to capture a curved object's shape faithfully, matching the
// "resample at a reasonable density" spirit of loft/sweepNProfiles.
const DEFAULT_INTERIOR_SAMPLES_PER_SPAN = 8;
// Density cap: a genuinely wild object curve's adaptive resample can produce
// far more points than a refit needs — a huge linear system for no shape
// benefit. Applied PER RUN, since each run between two corners is its own
// independent interpolation and its own linear system, which is the cost the
// cap exists to bound (the refit degree is preserved either way).
const MAX_REFIT_POINTS = 80;
// A tangent break of more than this at a full-multiplicity interior knot is
// treated as a real corner and split at. Small, because a NURBS corner is
// exact rather than estimated from traced points: the question is only
// whether the break is genuine or floating-point noise around zero.
const DEFAULT_CORNER_ANGLE_DEG = 0.5;

// Map an arbitrary set of world-space points from the BASE curve's frame
// field into the TARGET curve's frame field (the core Mode-A transform).
// Returns { mapped, baseDistances, frameOffsetMagnitudes } — `mapped` the
// mapped points, `baseDistances[i]` the i-th point's own closest-distance to
// the base curve (|P - C_base|), and `frameOffsetMagnitudes[i]` the i-th
// mapped point's own frame-offset magnitude |mapped - C_target| (equal to
// baseDistances[i] by orthonormality — the load-bearing invariant, exposed
// here so a verify script can prove it directly rather than recompute it).
export function mapPointsBaseToTarget(baseCrv, targetCrv, points) {
  if (!points.length) return { mapped: [], baseDistances: [], frameOffsetMagnitudes: [] };

  const baseUMax = baseCrv.knots[baseCrv.knots.length - 1];
  const baseUMin = baseCrv.knots[0];
  const targetUMax = targetCrv.knots[targetCrv.knots.length - 1];
  const targetUMin = targetCrv.knots[0];

  const baseTable = buildArcLengthTable(baseCrv, baseUMin, baseUMax);
  const targetTable = buildArcLengthTable(targetCrv, targetUMin, targetUMax);
  const baseLen = baseTable.total;
  const targetLen = targetTable.total;

  // (1) closest point on base for each object point
  const cps = points.map((P) => closestPointOnCurve(baseCrv, P));
  const baseUs = cps.map((cp) => cp.u);

  // one parallel-transport frame per object point, read at that point's own
  // closest base parameter (extraParams — the exact arbitrary-parameter frame
  // mechanism Sweep1's N-profiles already established)
  const baseFrames = buildParallelTransportFrames(baseCrv, baseUs).extra;

  // (2) same arc-length fraction on the target -> its own parameter -> frame
  const targetUs = baseUs.map((u) => {
    const s = arcLengthAtParam(baseTable, u);
    const f = baseLen > 1e-12 ? s / baseLen : 0;
    return paramAtArcLength(targetTable, f * targetLen);
  });
  const targetFrames = buildParallelTransportFrames(targetCrv, targetUs).extra;

  const mapped = [];
  const baseDistances = [];
  const frameOffsetMagnitudes = [];
  for (let i = 0; i < points.length; i++) {
    const bf = baseFrames[i];
    const off = sub(points[i], bf.origin);
    const a = dot(off, bf.xAxis);
    const b = dot(off, bf.yAxis);
    const c = dot(off, bf.zAxis);
    const tf = targetFrames[i];
    const m = add(add(add(tf.origin, scale(tf.xAxis, a)), scale(tf.yAxis, b)), scale(tf.zAxis, c));
    mapped.push(m);
    // TRUE closest-distance |P - C_base|, computed from the closest curve
    // POINT itself (which is also the frame origin, exactly) — NOT
    // closestPointOnCurve's own `.distance` field, which reports the coarse
    // polyline-projection distance whenever its Newton refinement rejects
    // its first step (a real, pre-existing quirk of that shipped function:
    // its `.point` is the true curve point but `.distance` can be the
    // polyline approximation, the two differing by the chord deviation).
    // Using the frame origin here keeps the orthonormal invariant
    // (frameOffsetMagnitude == baseDistance) EXACT, since both are |P -
    // bf.origin| computed the same way.
    baseDistances.push(length(off));
    frameOffsetMagnitudes.push(Math.sqrt(a * a + b * b + c * c));
  }
  return { mapped, baseDistances, frameOffsetMagnitudes };
}

// Even decimation down to the cap, keeping both ends. Correct for a list
// that is ALREADY density-weighted (an adaptive arc-length resample puts its
// samples where the curvature is, so thinning it evenly BY INDEX keeps that
// weighting); wrong for a list of uniform samples, which is why the uniform
// path below never reaches for it while it still has interior samples to
// spend instead.
function decimateEven(list, cap) {
  if (list.length <= cap) return list;
  const out = [];
  for (let i = 0; i < cap; i++) out.push(list[Math.round((i / (cap - 1)) * (list.length - 1))]);
  return [...new Set(out)].sort((x, y) => x - y);
}

// Stations plus `interior` evenly-spaced samples inside each gap between
// consecutive stations.
function stationSamples(stations, interior) {
  const params = [];
  for (let k = 0; k < stations.length; k++) {
    params.push(stations[k]);
    if (k < stations.length - 1) {
      const s0 = stations[k], s1 = stations[k + 1];
      for (let s = 1; s <= interior; s++) params.push(s0 + (s1 - s0) * (s / (interior + 1)));
    }
  }
  return params;
}

// THE DENSITY CAP IS SPENT ON THE INTERIOR SAMPLES, NOT ON THE STATIONS.
// The stations are the curve's own knot values — where its detail is
// declared to be — and an even decimation across the combined list drops
// them at exactly the same rate as the filler samples between them, so a
// curve whose knots cluster around its interesting region loses that region
// and keeps its flat parts. Interior density is uniform per span and is
// therefore the part that can be traded away; the largest per-span count
// that still fits under the cap is used instead. Only when the stations
// ALONE exceed the cap is there anything left to decimate, and then it is
// the stations, evenly, because nothing better is available.
function cappedStationSamples(stations, interior) {
  const params = stationSamples(stations, interior);
  if (params.length <= MAX_REFIT_POINTS) return params;
  if (stations.length >= MAX_REFIT_POINTS) return decimateEven(stations, MAX_REFIT_POINTS);
  const allowed = Math.floor((MAX_REFIT_POINTS - stations.length) / (stations.length - 1));
  return stationSamples(stations, Math.max(0, allowed));
}

// Build the dense object-curve resample parameter set over [u0, u1]: the
// range ends plus every distinct knot value strictly inside it (real span
// boundaries — never smoothed over) plus `interiorSamplesPerSpan`
// evenly-spaced interior samples per span, capped as above.
function objectSampleParams(objectCrv, interiorSamplesPerSpan, u0, u1) {
  const uMin = u0 ?? objectCrv.knots[0];
  const uMax = u1 ?? objectCrv.knots[objectCrv.knots.length - 1];
  const knotSet = new Set([uMin, uMax]);
  for (const k of objectCrv.knots) if (k > uMin && k < uMax) knotSet.add(k);
  const stations = [...knotSet].sort((x, y) => x - y);
  return cappedStationSamples(stations, interiorSamplesPerSpan);
}

// Distinct interior knot values, at or above full multiplicity — the ONLY
// parameters at which a NURBS curve is allowed to have a tangent break.
function fullMultiplicityInteriorKnots(crv) {
  const knots = crv.knots;
  const uMin = knots[0], uMax = knots[knots.length - 1];
  const span = uMax - uMin;
  const tol = Math.max(span * 1e-12, 1e-12);
  const out = [];
  for (const u of knots) {
    if (!(u > uMin + tol && u < uMax - tol)) continue;
    if (out.some((v) => Math.abs(v - u) <= tol)) continue;
    let mult = 0;
    for (const k of knots) if (Math.abs(k - u) <= tol) mult++;
    if (mult >= crv.degree) out.push(u);
  }
  return out.sort((x, y) => x - y);
}

// The angle, in degrees, between the chord directions arriving at and
// leaving `u` — the one-sided tangent directions, read geometrically so a
// curve whose knot ALLOWS a break but does not take one (a rational
// arc/circle: interior knots at multiplicity == degree, perfectly smooth
// across them) is not mistaken for a corner.
function turnAngleDeg(crv, u, h) {
  const c = curvePointXYZ(crv, u);
  const before = sub(c, curvePointXYZ(crv, u - h));
  const after = sub(curvePointXYZ(crv, u + h), c);
  const lb = length(before), la = length(after);
  if (lb < 1e-14 || la < 1e-14) return 0;
  const cosA = Math.max(-1, Math.min(1, dot(before, after) / (lb * la)));
  return Math.acos(cosA) * 180 / Math.PI;
}

// The object curve's GENUINE corners: full-multiplicity interior knots that
// actually break tangency there by more than `cornerAngleDeg`. A degree-1
// curve is excluded outright — it IS its own control polygon and the refit
// is degree 1 too, so its corners survive by construction and splitting at
// every vertex would only produce two-point runs.
function objectCornerParams(objectCrv, cornerAngleDeg) {
  if (objectCrv.degree < 2) return [];
  const uMin = objectCrv.knots[0], uMax = objectCrv.knots[objectCrv.knots.length - 1];
  const h = (uMax - uMin) * 1e-6;
  return fullMultiplicityInteriorKnots(objectCrv).filter((u) => turnAngleDeg(objectCrv, u, h) > cornerAngleDeg);
}

// MODE A — conform one object curve from BASE onto TARGET, returning a fresh
// NURBS curve fit through the mapped points. The result also carries
// `.conform = { mapped, sampleParams, baseDistances, frameOffsetMagnitudes }`
// for verification (the raw mapped point set and per-point invariants above).
export function conformCurveToCurve(baseCrv, targetCrv, objectCrv, opts = {}) {
  const interior = opts.interiorSamplesPerSpan ?? DEFAULT_INTERIOR_SAMPLES_PER_SPAN;
  const uMin = objectCrv.knots[0];
  const uMax = objectCrv.knots[objectCrv.knots.length - 1];
  const refitDegree = opts.refitDegree ?? objectCrv.degree;

  // adaptive resample for genuinely wild curves, falling back to the
  // knot-span sampling for ordinary ones — both are "resample at a
  // reasonable density"; the adaptive path catches a high-curvature object
  // between sparse knots the uniform path would under-sample.
  const sampleRange = (a, b) => (opts.adaptive
    ? decimateEven(adaptiveArcLengthSamples(objectCrv, a, b, opts.adaptiveTol).map((x) => x.u), MAX_REFIT_POINTS)
    : objectSampleParams(objectCrv, interior, a, b));

  // A closed object curve refits as a closed curve; a corner in it splits it
  // into runs. Both are decided from the object curve's own geometry, never
  // asked of the caller.
  const closed = isCurveClosed(objectCrv);
  const corners = objectCornerParams(objectCrv, opts.cornerAngleDeg ?? DEFAULT_CORNER_ANGLE_DEG);
  const h = (uMax - uMin) * 1e-6;
  // A closed curve's SEAM is a corner candidate too, and it is not an
  // interior knot, so it is asked the same geometric question separately:
  // does the tangent leaving u=uMin match the one arriving at u=uMax.
  const seamIsCorner = closed && objectCrv.degree >= 2 && (() => {
    const p0 = curvePointXYZ(objectCrv, uMin);
    const arriving = sub(p0, curvePointXYZ(objectCrv, uMax - h));
    const leaving = sub(curvePointXYZ(objectCrv, uMin + h), p0);
    const la = length(arriving), ll = length(leaving);
    if (la < 1e-14 || ll < 1e-14) return false;
    const cosA = Math.max(-1, Math.min(1, dot(arriving, leaving) / (la * ll)));
    return Math.acos(cosA) * 180 / Math.PI > (opts.cornerAngleDeg ?? DEFAULT_CORNER_ANGLE_DEG);
  })();
  // The seam of a closed curve can only be carried through a run's INTERIOR
  // (where an ordinary interpolation is smooth) when the seam isn't itself a
  // corner; otherwise the chain simply starts and ends there, and the join
  // leaves the corner where it belongs.
  const wrapAtSeam = closed && !seamIsCorner && corners.length > 0;
  let closedRefit = closed && corners.length === 0 && !seamIsCorner && refitDegree >= 2;
  // Whether the chain of runs shuts on itself (its last point IS its first).
  const chainCloses = closed && corners.length > 0;

  const runs = [];
  if (wrapAtSeam) {
    for (let i = 0; i < corners.length; i++) {
      const a = corners[i], b = corners[(i + 1) % corners.length];
      if (i < corners.length - 1) runs.push(sampleRange(a, b));
      // the wrapping run: from the last corner over the seam to the first.
      // u=uMax and u=uMin are the SAME physical point, so the second half
      // contributes everything but its first sample.
      else runs.push([...sampleRange(a, uMax), ...sampleRange(uMin, b).slice(1)]);
    }
  } else if (corners.length) {
    const cuts = [uMin, ...corners, uMax];
    for (let i = 0; i < cuts.length - 1; i++) runs.push(sampleRange(cuts[i], cuts[i + 1]));
  } else {
    const params = sampleRange(uMin, uMax);
    // a closed curve refit closed must NOT repeat its seam point: the wrap
    // closedCurveInterp does is cyclic over the points it is given, and it
    // needs at least three of them to have anything to wrap. Too few (an
    // opts.interiorSamplesPerSpan of 0 on a two-station curve) falls back to
    // the open refit rather than throwing out of the closed interpolator.
    if (closedRefit && params.length >= 4) runs.push(params.slice(0, -1));
    else { closedRefit = false; runs.push(params); }
  }

  // ONE mapping pass over the whole object, with each run's shared endpoint
  // contributing a SINGLE mapped point that both neighboring runs then
  // interpolate exactly — which is what makes the chain watertight at every
  // corner and what keeps `mapped[i]` aligned with `sampleParams[i]` for the
  // two invariants this module is gated on.
  const sampleParams = [];
  const runIndices = [];
  for (let r = 0; r < runs.length; r++) {
    const idx = [];
    for (let j = 0; j < runs[r].length; j++) {
      if (r > 0 && j === 0) { idx.push(sampleParams.length - 1); continue; }
      if (chainCloses && r === runs.length - 1 && j === runs[r].length - 1) { idx.push(0); continue; }
      sampleParams.push(runs[r][j]);
      idx.push(sampleParams.length - 1);
    }
    runIndices.push(idx);
  }
  // sample the object curve at those params (curvePoint via the same helper
  // closestPointOnCurve/etc. use); mapPointsBaseToTarget wants plain [x,y,z]
  const objPoints = sampleParams.map((u) => curvePointXYZ(objectCrv, u));
  const { mapped, baseDistances, frameOffsetMagnitudes } = mapPointsBaseToTarget(baseCrv, targetCrv, objPoints);

  let fit;
  if (closedRefit) {
    const { crv, uStart, uEnd } = closedCurveInterp(mapped, refitDegree);
    fit = rescaleCurveDomain(extractSubCurve(crv, uStart, uEnd), 0, 1);
  } else if (runIndices.length === 1) {
    fit = globalCurveInterp(mapped, refitDegree);
  } else {
    const segments = [];
    for (const idx of runIndices) {
      if (idx.length < 2) continue;
      segments.push(globalCurveInterp(idx.map((i) => mapped[i]), refitDegree));
    }
    fit = segments.length === 1 ? segments[0] : joinCurvesC0(segments);
  }
  fit.conform = {
    mapped, sampleParams, baseDistances, frameOffsetMagnitudes,
    closed, closedRefit, cornerParams: corners, seamIsCorner, runCount: runIndices.length,
  };
  return fit;
}

// MODE B — conform one object curve onto a surface's own ISOCURVE (the
// "genuine Rhino-beating win": Rhino's Flow needs an explicit curve for both
// ends; here the base curve is IMPLICITLY the isocurve running through a
// clicked surface point, extracted live rather than requiring a separate
// ExtractIsocurve step first). Reuses extractIsocurveU/V directly for the
// implicit base-curve extraction, then the identical Mode-A mapping runs
// against that extracted curve.
//   direction: 'u' -> the base curve is the FIXED-U isocurve (runs along V)
//              'v' -> the base curve is the FIXED-V isocurve (runs along U)
// The returned curve also carries `.baseIsocurve` (the extracted implicit
// base) so a verify script can prove it genuinely matches a direct
// extractIsocurveU/V call on the same (u,v).
export function conformCurveToSurface(srf, pickU, pickV, direction, targetCrv, objectCrv, opts = {}) {
  const baseIsocurve = direction === 'u'
    ? extractIsocurveU(srf, pickU)
    : extractIsocurveV(srf, pickV);
  const fit = conformCurveToCurve(baseIsocurve, targetCrv, objectCrv, opts);
  fit.baseIsocurve = baseIsocurve;
  return fit;
}

// Object-curve point as plain [x,y,z] (curvePoint already returns exactly
// that; the copy keeps mapPointsBaseToTarget's inputs a clean array of
// fresh triples).
function curvePointXYZ(crv, u) {
  const p = curvePoint(crv, u);
  return [p[0], p[1], p[2]];
}
