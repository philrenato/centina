// Sweep1: a single cross-section (planar or not — its full 3D shape is
// preserved) carried rigidly along a rail curve by a parallel-transport
// frame. Known limitations: no Roadlike/Freeform orientation styles and no
// multi-section Refit/Rebuild.
//
// Parallel transport rather than Frenet: Frenet frames flip sign or are
// undefined wherever curvature vanishes (a straight run in the rail).
// Parallel transport carries the previous frame's normal forward by
// projecting it into the new tangent's perpendicular plane and
// re-normalizing, which is well-defined wherever the tangent is, and keeps
// every frame orthonormal by construction.

import { curvePointAndTangent, curvePoint, grevilleAbscissae, closestPointOnCurve, buildArcLengthTable, paramAtArcLength, arcLengthAtParam, isCurveClosed, rationalCurveDerivs } from './curve.mjs';
import { add, sub, scale, dot, cross, normalize, anyPerpendicular, length } from './vec3.mjs';
import { chordLengthParams, averagingKnotVector, interpAtParams } from './interpolate.mjs';
import { concatTwoC0 } from './knots.mjs';
import { denseWithInterior } from './loft.mjs';
import { filletOpenPolyline, filletCornerArc, filletSegmentsToCurve, makeArc, revolve } from './primitives.mjs';

// A profile curve is picked in world coordinates, but sweep1Rigid (below)
// expects it as local (x,y,z) offsets from a frame origin. This re-projects
// the curve's control points onto the frame's X/Y/Z axes by dot product; the
// basis is orthonormal, so decomposing and reconstructing through the same
// frame is an exact round-trip. The full 3D shape is carried, including any
// depth along the frame's tangent (zAxis): a profile need not lie in the
// rail's perpendicular plane, and rigid transport preserves its shape.
export function localizeSectionToFrame(sectionCrv, frame) {
  const ctrlPts = sectionCrv.ctrlPts.map(([x, y, z, w]) => {
    const rel = sub([x, y, z], frame.origin);
    return [dot(rel, frame.xAxis), dot(rel, frame.yAxis), dot(rel, frame.zAxis), w];
  });
  return { degree: sectionCrv.degree, knots: sectionCrv.knots, ctrlPts };
}

// Frame origins are the rail's control points, not curve values at the
// Greville parameters. A curve value used as a control point of the swept
// surface's V direction (same knots, same degree) does not reproduce that
// curve in general (only degree-1 or clamped-endpoint cases coincide), and
// the mismatch compounds into twisting or ballooning on a sharply curved
// rail. Control points are exact: a swept surface built from them (same
// knotsV/degV as the rail) reproduces the rail exactly as one of its edges
// wherever a profile control point coincides with a frame's origin. The
// tangent still comes from the smooth curve at the Greville parameter; it is
// a direction estimate and needs no positional exactness. Known limitation:
// this holds only while the control points lie on the curve, i.e. for a
// degree<=1 rail; see `railFrameOriginsExact` and sweep1RigidResampled.
//
// `extraParams`: optional rail parameters that need not be control-point
// stations (sweepNProfiles stations each profile where it is closest to the
// rail, almost never at a control point). Every requested station,
// control-point and extra, is merged into one strictly ordered sequence, and
// the same incremental parallel-transport recurrence runs through all of
// them in rail-parameter order, so an extra station is one more link in the
// same chain. With `extraParams` empty the result is identical to the
// control-point-only chain (test/sweep-nprofiles.test.mjs).
//
// Origins at merged stations:
//  - A control-point station always uses the rail's raw control point. Where
//    two control points share one Greville parameter (a knot-multiplicity
//    case), each keeps its own origin; control-point stations are never
//    merged with one another.
//  - An extra station uses the curve evaluation at its parameter, unless it
//    coincides (within DEDUPE_TOL, relative to the rail's domain) with a
//    control-point station, in which case it reuses that station's frame
//    rather than building a twin one ulp away. This keeps a profile placed at
//    the rail's start or end (always control-point stations) reproducing the
//    rail exactly, as in sweep1Rigid's single-profile case. The substitution
//    is restricted further below.
//
// Returned as `frames` (one per rail control point, in order) with a plain
// `.extra` array attached (one frame per `extraParams` entry, same order);
// `frames[i]` and `frames.length` are unaffected by it.
export function buildParallelTransportFrames(rail, extraParams = []) {
  const gParams = grevilleAbscissae(rail);
  const domainSpan = Math.max(1e-12, Math.abs(rail.knots[rail.knots.length - 1] - rail.knots[0]));
  const DEDUPE_TOL = domainSpan * 1e-9;
  const lastCtrlIndex = rail.ctrlPts.length - 1;

  const steps = gParams.map((u, i) => ({ u, ctrlIndex: i }));
  const extraStep = new Array(extraParams.length);
  extraParams.forEach((u, j) => {
    // Substitution is exact only where the control point lies on the curve:
    // on a degree<=1 rail, or at either domain endpoint of a clamped rail of
    // any degree. On a degree>=2 rail an interior control point sits off the
    // curve (a fillet arc's weighted apex especially). A dense arc-length
    // sample on a symmetric arc can land exactly on the apex's Greville
    // parameter (u=5.5 on a [5,6] arc whose apex Greville parameter is
    // (5+6)/2), and substituting the off-curve apex there puts a sharp spike
    // in the surface. So substitution is limited to those two exact cases.
    const hit = steps.find((s) => Math.abs(s.u - u) <= DEDUPE_TOL
      && (rail.degree <= 1 || s.ctrlIndex === 0 || s.ctrlIndex === lastCtrlIndex));
    if (hit) { extraStep[j] = hit; return; }
    const s = { u, ctrlIndex: null };
    steps.push(s);
    extraStep[j] = s;
  });
  steps.sort((a, b) => a.u - b.u);

  const frames = [];
  let prevNormal = null;
  let prevRawTangent = null; // the previous step's un-mitered tangent: an interior corner's incoming edge direction
  for (let k = 0; k < steps.length; k++) {
    const step = steps[k];
    const { tangent: rawTangent } = curvePointAndTangent(rail, step.u);
    let tangent = rawTangent;

    // Interior-corner miter. `curvePointAndTangent` at a control point's
    // Greville parameter is a one-sided derivative (FindSpan, P&T A2.1,
    // resolves an interior knot u to the outgoing span [u, next)), so on a
    // degree<=1 rail `rawTangent` is only the outgoing edge's direction, and a
    // ring there would sit perpendicular to one of its two edges. The tangent
    // is replaced by the bisector of the outgoing and incoming edge tangents,
    // the construction the closed-rail seam weld (below) uses, and
    // `prevNormal` chains through it as one more link of the same recurrence.
    // The incoming tangent is `prevRawTangent`: on a degree<=1 rail the
    // segment from control point i-1 to i is both i-1's outgoing edge and i's
    // incoming edge.
    //
    // Applied only at an interior control-point join: degree<=1 (a higher
    // degree has a continuous tangent, nothing to miter); `ctrlIndex` strictly
    // between first and last (index 0/lastCtrlIndex are the closed-rail seam,
    // handled by the weld after this loop, or an open rail's ends); and the
    // preceding step is control point ctrlIndex-1 with no `extraParams` sample
    // between them, since otherwise `prevRawTangent` is not this corner's
    // incoming edge. In that last case (sweepNProfiles with extra samples on
    // a degree<=1 rail) the corner is not mitered.
    const isInteriorCorner = rail.degree <= 1 && step.ctrlIndex !== null
      && step.ctrlIndex > 0 && step.ctrlIndex < lastCtrlIndex
      && k > 0 && steps[k - 1].ctrlIndex === step.ctrlIndex - 1;
    if (isInteriorCorner) {
      const tOut = rawTangent;
      const tIn = prevRawTangent;
      // Collinear no-op, tested before any bisector math: where the rail does
      // not turn, `tangent` is left untouched rather than recomputed to the
      // same value.
      if (dot(tOut, tIn) < 1 - 1e-9) {
        // Degenerate corner, as in the seam weld: a near-180-degree fold-back
        // (bisector sum length ~0) falls back to the outgoing tangent.
        const sum = add(tOut, tIn);
        const sumLen = length(sum);
        tangent = sumLen > 1e-9 ? scale(sum, 1 / sumLen) : tOut;
      }
    }

    // The same guarded projection (with an `anyPerpendicular` fallback) the
    // seam weld uses for its miter normal, applied at every station. Where the
    // projection is well-defined it computes the identical
    // `normalize(sub(...))`, so it changes nothing there.
    //
    // It matters where the projection vanishes, and that case is ordinary. A
    // degree<=1 rail's tangent changes discontinuously at a corner, so an
    // extra (arc-length) station just past one inherits `prevNormal` from
    // before the turn — and if the rail turns into that normal's direction,
    // the projection is exactly zero. A rail along X and then straight up Z
    // does it: `anyPerpendicular([1,0,0])` is [0,0,1], the second leg's
    // tangent. The interior-corner miter cannot cover it, because an extra
    // station is not a control point.
    let normal;
    if (prevNormal) {
      const raw = sub(prevNormal, scale(tangent, dot(prevNormal, tangent)));
      normal = length(raw) > 1e-9 ? normalize(raw) : anyPerpendicular(tangent);
    } else {
      normal = anyPerpendicular(tangent);
    }

    const binormal = cross(tangent, normal);
    const origin = step.ctrlIndex !== null
      ? [rail.ctrlPts[step.ctrlIndex][0], rail.ctrlPts[step.ctrlIndex][1], rail.ctrlPts[step.ctrlIndex][2]]
      : curvePoint(rail, step.u);
    step.frame = { u: step.u, origin, xAxis: normal, yAxis: binormal, zAxis: tangent };
    if (step.ctrlIndex !== null) frames[step.ctrlIndex] = step.frame;
    prevNormal = normal;
    prevRawTangent = rawTangent;
  }
  // Closed-rail seam weld. On a closed rail (first and last control points
  // coincide — `isCurveClosed`), `frames[0]` and `frames[frames.length-1]`
  // are the same physical vertex but are built by independent steps of the
  // loop: frame[0] is seeded with `anyPerpendicular` off the outgoing edge's
  // tangent, frame[last] carries the normal the whole chain accumulated, off
  // the incoming edge's tangent. Used as the swept surface's v=0 and v=vMax
  // rings they are two differently oriented rings on one point, and the tube
  // is open at the seam.
  //
  // The loop is welded with one shared miter frame, used as both `frames[0]`
  // and `frames[frames.length-1]`, so the first and last V rings are
  // identical. Its tangent is the bisector of the outgoing and incoming edge
  // tangents (their normalized sum); its normal continues frame[0]'s `xAxis`
  // one more parallel-transport step, projected into the bisector's
  // perpendicular plane and renormalized. Degenerate fallbacks (never NaN):
  // nearly opposite edge tangents (sum length ~0) fall back to the outgoing
  // tangent; an xAxis nearly parallel to the bisector falls back to
  // `anyPerpendicular`, the chain's own first seed.
  //
  // This touches only `frames[0]`/`frames[frames.length-1]`, which the
  // interior-corner miter excludes by construction, so the two compose: a
  // closed pentagon rail gets its 4 interior corners mitered in the loop and
  // its seam vertex welded here (test/sweep-interior-corner-miter.test.mjs).
  //
  // The weld also rewrites the underlying step objects' `.frame` in place,
  // and `.extra` is captured after it. Extra stations that deduped onto the
  // start/end control point (a closed rail's two domain-endpoint dense
  // samples always do) refer to those step objects; with only the array
  // slots reassigned they would read the un-welded frames, which on a
  // non-planar 7-corner closed Pipe rail differ by ~11 degrees in xAxis and
  // leave a ~0.09 mm gap in the V control net at the seam.
  // `denseRailFrames`/`sweep1RigidResampled` (the only path a 'rounded'
  // cornerStyle Pipe on a closed rail reaches) read `.extra` exclusively;
  // sweep1Rigid's free (by-index) path never reads it.
  if (frames.length >= 2 && isCurveClosed(rail)) {
    const first = frames[0];
    const last = frames[frames.length - 1];
    const tOut = first.zAxis;
    const tIn = last.zAxis;
    const sum = add(tOut, tIn);
    const sumLen = length(sum);
    const seamTangent = sumLen > 1e-9 ? scale(sum, 1 / sumLen) : tOut;
    const proj = dot(first.xAxis, seamTangent);
    const rawNormal = sub(first.xAxis, scale(seamTangent, proj));
    const seamNormal = length(rawNormal) > 1e-9 ? normalize(rawNormal) : anyPerpendicular(seamTangent);
    const seamBinormal = cross(seamTangent, seamNormal);
    const seamFrame = { u: first.u, origin: first.origin, xAxis: seamNormal, yAxis: seamBinormal, zAxis: seamTangent };
    frames[0] = seamFrame;
    frames[frames.length - 1] = seamFrame;
    const firstStepObj = steps.find((s) => s.ctrlIndex === 0);
    const lastStepObj = steps.find((s) => s.ctrlIndex === lastCtrlIndex);
    if (firstStepObj) firstStepObj.frame = seamFrame;
    if (lastStepObj) lastStepObj.frame = seamFrame;
  }

  // Captured after the weld, so an extra station deduped onto the start/end
  // control point reads the welded seam frame.
  frames.extra = extraStep.map((s) => s.frame);

  return frames;
}

// Whether any control-point weight differs from 1. A Circle/Arc from makeArc
// (kernel/primitives.mjs, a tangent-line-intersection construction) is
// rational: about half its control points are off-curve tangent-line corners
// (radius / cos(dtheta/2), ~41% farther from center than the curve for a
// 90-degree span) with weight below 1. This is not the frame-origin dispatch
// condition (see `railFrameOriginsExact`): rationality is one cause of
// off-curve control points, not the only one.
export function railIsRational(rail) {
  return rail.ctrlPts.some((cp) => Math.abs(cp[3] - 1) > 1e-9);
}

// Whether a rail's raw control points are guaranteed to lie on its curve:
// true only for degree<=1 (Line, Polyline), whose control polygon is the
// curve regardless of weight. This, not `railIsRational`, is the
// precondition of sweep1Rigid's free path. A non-rational curve is not
// exempt: the middle control point of a 3-point degree-2 `globalCurveInterp`
// rail that bends sharply to hit its data with one quadratic span sits 70-98
// units off the curve on a ~260-unit rail, and the swept surface becomes a
// nearly planar wedge instead of following the rail. A gentle 3-point
// degree-2 rail deviates only ~0.13 units, so severity scales with how
// sharply the rail bends relative to its control-point count, not with the
// rail's type.
export function railFrameOriginsExact(rail) {
  return rail.degree <= 1;
}

// A smooth rail's tightest bend, as a radius of curvature, sampled over the
// rail's own knot domain.
//
// Why this measures a tube: a tube of radius r swept along a rail behaves
// locally like a torus whose major radius is the rail's local radius of
// curvature R = 1/kappa and whose minor radius is r — the spindle/horn torus
// self-intersection `pipeRailForSweep`'s corner floor reasons about from the
// other direction. A torus self-intersects once its tube radius exceeds its
// bend radius, so `r > R` anywhere along the rail means the swept surface
// passes through itself there. This is geometry, not a tessellation
// artifact: no sampling density removes it, and a denser mesh shows it more
// clearly.
//
// It partitions against the corner floor by construction. A degree<=1 rail
// (Line/Polyline) is piecewise straight, so its second derivative is zero
// and this returns Infinity — a sharp corner carries its curvature as a
// delta function no sampled derivative can see, which is what the
// corner-fillet floor exists for. A rail rounded by that floor has fillet
// arcs of radius >= the tube radius by construction, so their curvature
// 1/cornerRadius is already inside this bound: the two guards agree rather
// than double-clamping.
//
// It samples the actual knot domain, not a presumed [0,1]. A Circle rail
// (`makeArc`/`makeCircle`) has domain [0, narcs] — [0,4] for a full circle —
// and a fillet-composed rail has its spans on a sequential integer domain.
// Sampling [0,1] would read only the first quarter of a circular rail.
export function railMinBendRadius(rail, samples = 257) {
  const k = rail.knots;
  const uMin = k[rail.degree];
  const uMax = k[k.length - 1 - rail.degree];
  if (!(uMax > uMin)) return { radius: Infinity, kappa: 0, u: uMin };
  let kappaMax = 0, at = uMin;
  for (let i = 0; i <= samples; i++) {
    const u = uMin + ((uMax - uMin) * i) / samples;
    const [, C1, C2] = rationalCurveDerivs(rail, u, 2);
    const speed = length(C1);
    if (!(speed > 0)) continue; // a degenerate station contributes nothing rather than an Infinity/NaN kappa
    const kappa = length(cross(C1, C2)) / (speed * speed * speed);
    if (Number.isFinite(kappa) && kappa > kappaMax) { kappaMax = kappa; at = u; }
  }
  return { radius: kappaMax > 0 ? 1 / kappaMax : Infinity, kappa: kappaMax, u: at };
}

// The margin both self-intersection guards keep off the horn-torus boundary,
// named once so they compose exactly.
//
// A margin is needed because clamping to exactly the bend radius sits on
// that boundary, where the tube touches itself at a single point and
// discrete tessellation lands adjacent samples on both sides of the contact,
// producing a spurious ~180 degree fold indistinguishable from
// self-intersection.
//
// One constant, not two: the corner floor raises a too-small cornerRadius to
// `tubeRadius * PIPE_SELF_INTERSECT_MARGIN`; this lowers a too-large
// tubeRadius to `bendRadius / PIPE_SELF_INTERSECT_MARGIN`. On a
// corner-floored rail, whose bend radius is that raised cornerRadius, the two
// cancel to the original tube radius, so the second guard never re-clamps
// what the first made safe. Independent margins (say 1.02 up and 0.98 down)
// would multiply to 0.9996 and fire a sub-micron clamp on every
// rounded-corner Pipe. test/pipe-bend-radius.test.mjs composes the two and
// asserts no clamp.
export const PIPE_SELF_INTERSECT_MARGIN = 1.02;

export function pipeSafeTubeRadius(rail, requested, samples = 257) {
  const bend = railMinBendRadius(rail, samples);
  const safeMax = bend.radius / PIPE_SELF_INTERSECT_MARGIN;
  // A relative tolerance, not a bare `>`: the composition above cancels
  // algebraically but not necessarily to the last bit in floating point, and a
  // clamp firing on a 1e-16 overshoot would report a difference no geometry
  // can express.
  if (!(requested > safeMax * (1 + 1e-9))) return { radius: requested, clamped: false, safeMax, minBendRadius: bend.radius, u: bend.u };
  return { radius: safeMax, clamped: true, safeMax, minBendRadius: bend.radius, u: bend.u };
}

// Dense arc-length-spaced rail parameters (every knot-span boundary exactly,
// plus in-between samples per span — the station-plus-interior-samples
// pattern sweepNProfiles uses) and the continuously evaluated
// parallel-transport frame at each one (buildParallelTransportFrames'
// `extraParams`; a raw control point is used only at the two clamped domain
// endpoints, where curve value and control point coincide regardless of
// degree or rationality). `vDense` is each sample's rail arc-length fraction
// (0..1), the V parametrization sweep1RigidResampled (below) fits its final
// interpolation to.
//
// Per-span sample count. The swept surface is exact only at the dense
// samples and interpolates between them. The composed rail is
// tangent-continuous at every arc/line junction (a fillet guarantees G1),
// but its curvature is not: a fillet arc turns at a constant nonzero rate
// and its straight neighbor at zero. A global cubic (C2) interpolation
// through data with such a discontinuity rings near the junction
// (Gibbs-like), and the ringing needs resolution on both sides of the
// junction: on synthetic 5-, 8- and 12-corner closed rails, raising every
// span from 12 to 48 samples took worst-case cross-section ellipticity from
// ~19% to under 1.5%, while raising only the arc span's count left it at
// ~18%.
//
// So `MIN_SPAN_SAMPLES` is the floor for every span, straight ones included,
// and `adaptiveSpanSampleCount` raises a span's count with its measured
// tangent-turn angle (`spanTurnAngle`) past that floor, capped at
// `MAX_SPAN_SAMPLES` so a near-180-degree fold-back cannot grow the
// control-point count without bound. An explicit `samplesPerSpan` bypasses
// the measurement.
const MIN_SPAN_SAMPLES = 48;
const MAX_SPAN_SAMPLES = 64;
const TARGET_ANGLE_STEP_RAD = 2 * Math.PI / 180; // ~2 degrees per dense sample
function spanTurnAngle(rail, u0, u1) {
  // Probe just inside the span, not at its boundary: a span boundary is often
  // a C0 knot (an arc/line join in a filleted rail) where the tangent is
  // one-sided, and this wants the span's own interior turn, not the
  // neighboring span's tangent.
  const eps = Math.max(1e-9, (u1 - u0) * 1e-6);
  const t0 = curvePointAndTangent(rail, u0 + eps).tangent;
  const t1 = curvePointAndTangent(rail, u1 - eps).tangent;
  const d = Math.max(-1, Math.min(1, dot(t0, t1)));
  return Math.acos(d);
}
function adaptiveSpanSampleCount(rail, u0, u1) {
  const angle = spanTurnAngle(rail, u0, u1);
  const bySpin = Math.ceil(angle / TARGET_ANGLE_STEP_RAD);
  return Math.min(MAX_SPAN_SAMPLES, Math.max(MIN_SPAN_SAMPLES, bySpin));
}
// railHardBreakParams (below) returns structural breaks only. A rail's
// `knots` mix two things: a simple (multiplicity-1) interior knot of a
// smooth global interpolation (a SketchCurve rail, say), where the curve is
// C^(p-1) as anywhere else on its span, and a C0-or-worse joint, which
// carries full multiplicity (= degree) by construction
// (`filletSegmentsToCurve`'s `joinCurvesC0`/`concatTwoC0`, and a Circle/Arc's
// 2-per-arc tangent-corner knots). Only the second is a seam the dense
// sampling treats specially; treating the first as one would fragment a
// smooth spline rail at every interior knot. The same distinction as
// `tessellateSurface`'s `m < srf.degU` test, in the rail (V) direction.
// rescaleKnotsOnly rescales only a knot vector's numeric domain, for
// sweep1RigidResampled's per-span local fits; `rescaleCurveDomain`
// (kernel/knots.mjs) rescales a whole curve.
function rescaleKnotsOnly(knots, newMin, newMax) {
  const oldMin = knots[0], oldMax = knots[knots.length - 1];
  const span = oldMax - oldMin;
  if (span < 1e-12) return knots.map(() => newMin);
  return knots.map((k) => newMin + ((k - oldMin) / span) * (newMax - newMin));
}
function railHardBreakParams(rail) {
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const eps = Math.max(1e-9, (uMax - uMin) * 1e-9);
  const seen = new Set();
  const breaks = [];
  for (const k of rail.knots) {
    if (k <= uMin + eps || k >= uMax - eps || seen.has(k)) continue;
    seen.add(k);
    let mult = 0;
    for (const k2 of rail.knots) if (Math.abs(k2 - k) < eps) mult++;
    if (mult >= rail.degree) breaks.push(k);
  }
  return breaks.sort((a, b) => a - b);
}
function denseRailFrames(rail, samplesPerSpan = null) {
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const knotSet = new Set([uMin, uMax]);
  for (const k of rail.knots) if (k > uMin && k < uMax) knotSet.add(k);
  const spanBoundaries = [...knotSet].sort((a, b) => a - b);
  const hardBreaks = railHardBreakParams(rail);

  const railTable = buildArcLengthTable(rail, uMin, uMax);
  const railLen = railTable.total;

  const hardBreakSet = new Set(hardBreaks);
  const uDense = [];
  const hardBreakIdx = []; // indices into uDense/vDense/denseFrames of each hard-break span boundary, including the two domain ends — length = hardBreaks.length + 2
  for (let i = 0; i < spanBoundaries.length; i++) {
    if (i === 0 || i === spanBoundaries.length - 1 || hardBreakSet.has(spanBoundaries[i])) hardBreakIdx.push(uDense.length);
    uDense.push(spanBoundaries[i]);
    if (i < spanBoundaries.length - 1) {
      const l0 = arcLengthAtParam(railTable, spanBoundaries[i]);
      const l1 = arcLengthAtParam(railTable, spanBoundaries[i + 1]);
      const n = samplesPerSpan != null ? samplesPerSpan : adaptiveSpanSampleCount(rail, spanBoundaries[i], spanBoundaries[i + 1]);
      for (let k = 1; k <= n; k++) {
        const t = k / (n + 1);
        uDense.push(paramAtArcLength(railTable, l0 + (l1 - l0) * t));
      }
    }
  }
  const vDense = uDense.map((u) => arcLengthAtParam(railTable, u) / railLen);
  const denseFrames = buildParallelTransportFrames(rail, uDense).extra;
  return { denseFrames, vDense, hardBreakIdx };
}

// Variable radius along a Sweep1 rail, within one curve: a single Pipe's rail
// carries several radius breakpoints. Per-curve radius profiles across a
// MultiPipe network are not built; `sweepNProfiles` and `sweep2` take no
// radius options.
//
// Data model: `radiusPoints` is a list of `{t, radius}` breakpoints, `t` a
// normalized rail-parameter fraction (0 = rail start, 1 = rail end), not an
// arc-length fraction — the convention of the Split operator's `splitFrac`
// (`dom[0] + frac*(dom[1]-dom[0])` on a raw knot domain), applied to a rail.
// Linearly interpolated between the two breakpoints straddling `t`, and
// clamped to the nearest end breakpoint's radius outside [minT, maxT], never
// extrapolated.
//
// Known limitation: a parameter fraction only approximates physical position
// where spans have unequal length relative to their parameter width. A
// multi-segment Polyline rail has uniform per-segment parameter width
// (getProfileCrv builds `[0,0,1,2,...,n-1,n-1]`, not chord-length spacing),
// and a rail rounded by `pipeRailForSweep`/`filletSegmentsToCurve` gives
// every Line/Arc segment an equal 1-unit span (`joinCurvesC0`'s
// `rescaleCurveDomain`), so a short fillet arc has the same parameter width
// as a long straight run beside it. `t` is read from the rail actually swept
// (`frame.u` in `sweep1Rigid`/`sweep1RigidResampled`), so t=0 and t=1 land
// exactly on the rail's ends (fillets never move them) and interior
// breakpoints map by the same convention on a sharp or rounded rail. An
// arc-length-exact mapping would need `buildArcLengthTable`/
// `paramAtArcLength` threaded through both paths.
// test/sweep-variable-radius.test.mjs checks the rounded-corner combination:
// finite everywhere, endpoints exact, interior values approximate.
export function radiusAtT(radiusPoints, t) {
  if (!radiusPoints || radiusPoints.length === 0) return null;
  if (radiusPoints.length === 1) return radiusPoints[0].radius;
  const sorted = radiusPoints.slice().sort((a, b) => a.t - b.t);
  if (t <= sorted[0].t) return sorted[0].radius;
  const last = sorted[sorted.length - 1];
  if (t >= last.t) return last.radius;
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i], b = sorted[i + 1];
    if (t >= a.t && t <= b.t) {
      const span = b.t - a.t;
      const frac = span > 1e-12 ? (t - a.t) / span : 0;
      return a.radius + frac * (b.radius - a.radius);
    }
  }
  return last.radius; // unreachable given the bounds checks above
}

// Builds a per-frame scale-factor function for `sweep1Rigid`/
// `sweep1RigidResampled`, or `null` when there is nothing to scale (no
// radiusOpts, or 0-1 breakpoints). On `null` the caller skips the scale
// multiply entirely rather than multiplying by 1.0, so a constant-radius
// sweep is bit-identical to an unscaled one.
function variableRadiusScaler(rail, radiusOpts) {
  const pts = radiusOpts && radiusOpts.radiusPoints;
  const baseRadius = radiusOpts && radiusOpts.baseRadius;
  if (!pts || pts.length <= 1 || !(baseRadius > 1e-9)) return null;
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const span = Math.max(1e-12, uMax - uMin);
  return (frame) => radiusAtT(pts, (frame.u - uMin) / span) / baseRadius;
}

// Resampled path, for any rail whose control points are not guaranteed to
// lie on the curve (see `railFrameOriginsExact`). The free path (sweep1Rigid
// reusing the rail's control points and knots) is exact only when they do
// (degree<=1); for a degree>=2 rail, rational (Circle/Arc) or not (a sparse
// SketchCurve interpolation), reusing raw control points as frame origins can
// carry them well off the rail's path, at every V station.
//
// This path uses a raw control point as a frame origin only at the two
// clamped domain endpoints (where curve value and control point coincide);
// every other frame comes from a continuous curve evaluation at a dense,
// arc-length-spaced parameter (denseRailFrames, via
// buildParallelTransportFrames' `extraParams`). It is the technique loft()
// and sweepNProfiles use to combine curves whose knot/degree structures are
// not directly combinable — resample to a shared dense parameter set, then
// interpolate through values without touching the original control nets —
// applied to one profile's rigid transport.
//
// Tradeoff: the free path needs no solve and is exact at every rail
// parameter, but only for a degree<=1 rail; this path costs one
// global-interpolation solve per section control point and is exact only at
// the dense samples (test/sweep-rational-rail.test.mjs and
// test/sweep-sparse-rail.test.mjs check that it tracks the rail's path
// within a tight tolerance at in-between V values too).
//
// Per-rail-span composed fit. A single global cubic interpolation through
// the whole dense sample set rings (Gibbs-like) wherever the rail has a C0
// joint (a fillet arc) whose arc-length footprint is tiny next to its
// straight neighbors': with a 5 mm fillet next to a 200 mm run, adjacent
// tessellated face normals reach 178-180 degrees (a fold) and the
// centerline carries a ~0.23 mm non-monotonic wiggle on a 5 mm pipe. So no
// spline is fit across a structural joint: `railHardBreakParams` finds every
// C0-or-worse break (full-multiplicity knots only), a separate local
// degree-3 interpolation is fit through each span's dense samples, and the
// pieces are stitched with `concatTwoC0` (kernel/knots.mjs), the join
// `filletSegmentsToCurve` uses to build the rail. Sampled at 300 stations
// per span, the worst adjacent face-normal angle is then under 0.0003
// degrees (149-173 degrees with the single fit) and the swept radius holds
// 4.999999-5.000000 mm on a nominal 5 mm pipe (4.88-5.25 mm with the single
// fit). With `numSpans<=1` (no internal joint) the single global fit is
// used.
//
// Known limitation: `concatTwoC0` joins at C0, position only (the composed
// rail is likewise formally C0 and measured G1; see
// `filletSegmentsToCurve`). A severe case — a very short arc span between
// much longer straight runs, with a rational profile control point far off
// the tube axis (a circle's weighted corner points, ~41% farther out than
// the on-curve ones) — can carry a sub-micron direction reversal at a joint
// (~0.0003 mm, 0.006% of a 5 mm profile radius). An exact fix would solve
// each span with a shared derivative constraint at every interior joint
// (`globalCurveInterpWithEndDerivs`, generalized to a per-boundary
// derivative estimate, with the derivative magnitude derived correctly, not
// only its direction).
//
// Known limitation on a closed rail: the position gap at the seam is closed
// by the weld in `buildParallelTransportFrames` (the `.extra` frames this
// function reads carry the welded seam frame). The per-row V-curve fit
// (single or per-span) is still an open, clamped interpolation that does not
// know the rail's two ends must agree in derivative as well as position, so
// the worst profile control point (a circle profile's rational corner point)
// can show a several-degree tangent mismatch at the seam. It occurs at a
// similar magnitude on a planar closed rail, so it does not scale with
// non-planarity or corner angle.
//
// Two numerical traps for a fix. Padding each seam-touching span's local fit
// with samples from its neighbor fails when the two spans have very
// different arc-length footprints (a short fillet arc next to a long
// straight run): the borrowed samples cluster the knots and reproduce the
// ringing the per-span fit avoids (a near-180-degree chord-direction
// reversal within the first ~2% of the receiving span). Fresh padding at the
// receiving span's own sample spacing avoids that, but puts the padding on
// the grid the span's samples already sit on, so the padded curve's
// Eq. 9.8-style interior-knot averaging can land exactly (up to float noise)
// on the trim boundary, defeating `extractSubCurve`'s tolerance-based
// multiplicity count and corrupting control points near it; a half-step
// phase offset avoids that. A periodic global curve interpolation, rather
// than padding and trimming, is the likely correct construction.
//
// Size cap. interpAtParams solves a dense O(n^3) Gauss-Jordan system, sized
// for sketched-curve control-point counts (tens, not thousands).
// denseRailFrames' sampling scales with the rail's knot-span count, not its
// curvature: a smooth rail with many simple interior knots (a degree-3
// interpolation through many points, such as a Curve Generator Lorenz or
// Random-Walk curve) has no hard breaks, takes the `numSpans<=1` branch, and
// hands its whole dense set (up to MAX_SPAN_SAMPLES per knot span) to one
// interpAtParams call: ~6000 points for a ~100-span Lorenz curve, a
// 6000x6000 solve that takes minutes. The set is subsampled by uniform
// stride to MAX_DENSE_INTERP_POINTS, both domain endpoints kept exactly, as
// other counts unbounded by construction are capped (MAX_LSYSTEM_SEGMENTS,
// ArrayLinear's 500 copies). 300 points is far denser than the rail
// direction needs.
const MAX_DENSE_INTERP_POINTS = 300;
function cappedDenseIndices(n) {
  if (n <= MAX_DENSE_INTERP_POINTS) return null; // no cap needed — caller uses the original arrays untouched
  const idx = [];
  const step = (n - 1) / (MAX_DENSE_INTERP_POINTS - 1);
  for (let i = 0; i < MAX_DENSE_INTERP_POINTS; i++) idx.push(i === MAX_DENSE_INTERP_POINTS - 1 ? n - 1 : Math.round(i * step));
  return idx;
}
function sweep1RigidResampled(rail, section, radiusOpts = null) {
  const { denseFrames, vDense, hardBreakIdx } = denseRailFrames(rail);
  const scaler = variableRadiusScaler(rail, radiusOpts);
  const numSpans = hardBreakIdx.length - 1;
  const capIdx = numSpans <= 1 ? cappedDenseIndices(vDense.length) : null;
  const vDenseCapped = capIdx ? capIdx.map((i) => vDense[i]) : vDense;
  let sharedDegV = null, sharedKnotsV = null;
  const ctrlNet = section.ctrlPts.map(([sx, sy, sz, sw]) => {
    const worldRow = denseFrames.map((f) => {
      if (!scaler) return frameToWorld(f, [sx, sy, sz]);
      const k = scaler(f);
      return frameToWorld(f, [sx * k, sy * k, sz * k]);
    });
    if (numSpans <= 1) {
      const worldRowCapped = capIdx ? capIdx.map((i) => worldRow[i]) : worldRow;
      const dV = Math.min(3, vDenseCapped.length - 1);
      const knotsV = averagingKnotVector(vDenseCapped, dV);
      const localCtrl = interpAtParams(worldRowCapped, dV, vDenseCapped, knotsV);
      if (sharedDegV === null) { sharedDegV = dV; sharedKnotsV = knotsV; }
      return localCtrl.map(([x, y, z]) => [x, y, z, sw]);
    }
    let acc = null;
    for (let s = 0; s < numSpans; s++) {
      const i0 = hardBreakIdx[s], i1 = hardBreakIdx[s + 1];
      const localVRaw = vDense.slice(i0, i1 + 1);
      const localPts = worldRow.slice(i0, i1 + 1);
      // `averagingKnotVector` fixes ubar[0]===0/ubar[n]===1 (Eq 9.8's
      // clamped-end convention), so this span's slice of the global vDense is
      // shifted and scaled to [0,1] for that call, and its knots are then
      // rescaled back to the span's arc-length-fraction sub-range
      // (`rescaleKnotsOnly`) so the composed `knotsV` stays in [0,1].
      // Consumers (`tessellateSurface`, cap building, measure/pick) read
      // `knotsV[0]`/`knotsV[last]` rather than assuming 0/1.
      const spanMin = localVRaw[0], spanMax = localVRaw[localVRaw.length - 1];
      const spanLen = Math.max(1e-12, spanMax - spanMin);
      const localVNorm = localVRaw.map((v) => (v - spanMin) / spanLen);
      const dVLocal = Math.min(3, localPts.length - 1);
      const knotsNorm = averagingKnotVector(localVNorm, dVLocal);
      const ctrlLocal = interpAtParams(localPts, dVLocal, localVNorm, knotsNorm);
      const knotsTrue = rescaleKnotsOnly(knotsNorm, spanMin, spanMax);
      const spanCrv = { degree: dVLocal, knots: knotsTrue, ctrlPts: ctrlLocal };
      acc = acc ? concatTwoC0(acc, spanCrv, dVLocal) : spanCrv;
    }
    if (sharedDegV === null) { sharedDegV = acc.degree; sharedKnotsV = acc.knots; }
    return acc.ctrlPts.map(([x, y, z]) => [x, y, z, sw]);
  });
  return {
    degU: section.degree, knotsU: section.knots,
    degV: sharedDegV, knotsV: sharedKnotsV,
    ctrlNet,
    frames: denseFrames, // exposed for verification, same orthonormal-frame contract as the free path
  };
}

// True miter for 'sharp' corners ('sharp' remains the default). The
// interior-corner miter orients each corner's ring, but with a circular ring
// the ruled spans on either side of the corner neck: 4.6194 mm on a nominal
// 5 mm pipe at a 90-degree corner (test/fillet-open-polyline.test.mjs).
// 'rounded' corners avoid this by filleting the rail.
//
// The correct shared cross-section at a corner of turn angle theta is not a
// circle in the bisector frame but the ellipse the bisector (miter) plane
// cuts from either adjacent segment's cylinder: radius unchanged
// perpendicular to the bend plane, stretched by sec(theta/2) along the
// bend-plane direction (as documented for McNeel Rhino, and for Houdini's
// Sweep 2.0 "Stretch Around Turns", on by default). A ruled span between a
// segment's perpendicular ring and this ellipse is exactly that segment's
// cylinder, so the necking is compensated exactly. For the 90-degree
// L-corner the stretched point matches the cylinder-ellipse point
// algebraically for every profile angle; test/sweep-true-miter.test.mjs
// checks the general case numerically.
//
// theta and the bend direction need no new geometry: for unit tangents
// tIn/tOut, theta = angle(tIn,tOut), and `tOut - tIn` is perpendicular to
// `tOut + tIn` (the bisector tangent), since dot(a-b,a+b) = |a|^2-|b|^2 = 0
// for unit vectors. So `normalize(tOut-tIn)` is the in-plane bend
// direction, computed without cross products and already in the corner
// ring's plane.
//
// Degenerate case: sec(theta/2) -> infinity as theta -> 180 degrees. A
// corner needing more than `PIPE_MITER_LIMIT` stretch (10x, ~168.5 degrees of
// turn; Houdini's "Max Stretch" default) falls back to a small rounded fillet
// for that corner only, the other corners on the rail staying true-mitered.
// The limit is a constant, not a per-object parameter.
export const PIPE_MITER_LIMIT = 10; // Houdini Sweep 2.0's documented "Max Stretch" default
const PIPE_MITER_FALLBACK_FACETS = 8; // straight facets approximating a fallback corner's rounded arc; each facet turns a small fraction of the original corner, far under PIPE_MITER_LIMIT (test/sweep-true-miter.test.mjs)

// Every non-collinear corner of a degree<=1 rail, open or closed, with its
// turn angle (`theta`, between the incoming and outgoing unit edge
// directions; collinear runs are omitted, not returned as theta=0), the
// bend-plane unit direction (`dHat`, see above), and the true-miter stretch
// factor `sec(theta/2)`. `ringIndices` is `[i]` for an interior
// control-point corner, or `[0, n-1]` for a closed rail's seam vertex, where
// the seam weld makes `frames[0]` and `frames[n-1]` the same frame.
export function railInteriorCorners(rail) {
  if (rail.degree > 1) return []; // corners are defined on a degree<=1 rail's control points; a higher-degree rail's control points are not generally on the curve (sweep1Rigid never passes one here; see railFrameOriginsExact)
  const pts = rail.ctrlPts.map((cp) => [cp[0], cp[1], cp[2]]);
  const n = pts.length;
  if (n < 3) return [];
  const closed = isCurveClosed(rail);
  const corners = [];
  const consider = (prevPt, vPt, nextPt, ringIndices) => {
    const tIn = normalize(sub(vPt, prevPt));
    const tOut = normalize(sub(nextPt, vPt));
    const c = Math.max(-1, Math.min(1, dot(tIn, tOut)));
    if (c > 1 - 1e-9) return; // collinear: no turn, nothing to stretch
    const theta = Math.acos(c);
    const dRaw = sub(tOut, tIn);
    const dLen = length(dRaw);
    if (dLen < 1e-12) return; // defensive only — algebraically unreachable once c<=1-1e-9 above
    const dHat = scale(dRaw, 1 / dLen);
    corners.push({ ringIndices, theta, dHat, stretch: 1 / Math.cos(theta / 2) });
  };
  const lastIdx = n - 1;
  for (let i = 1; i < lastIdx; i++) consider(pts[i - 1], pts[i], pts[i + 1], [i]);
  if (closed) consider(pts[lastIdx - 1], pts[0], pts[1], [0, lastIdx]);
  return corners;
}

function degree1RailFromPoints(points) {
  const ctrlPts = points.map((p) => [p[0], p[1], p[2], 1]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}

// Miter-limit fallback: replaces each corner of a degree<=1 rail that needs
// more than `PIPE_MITER_LIMIT` stretch with a small faceted rounded fillet,
// using `filletOpenPolyline`'s per-corner selection and trim-budget math
// (via `cornerFilter`, kernel/primitives.mjs). The fillet's rational arc is
// flattened into straight facets, rather than composed at degree 2 by
// `filletSegmentsToCurve`, so the rail stays degree<=1: every other corner
// still gets the true-miter stretch, and the facet joints turn far less than
// the limit (test/sweep-true-miter.test.mjs, "mixed rail" case). Returns the
// same rail object unchanged when no corner exceeds the limit, so an
// ordinary rail's degV/knotsV are untouched.
function applyMiterLimitFallback(rail) {
  const corners = railInteriorCorners(rail);
  const overLimit = corners.filter((c) => c.stretch > PIPE_MITER_LIMIT);
  if (overLimit.length === 0) return rail;

  const closed = isCurveClosed(rail);
  const allPts = rail.ctrlPts.map((cp) => [cp[0], cp[1], cp[2]]);
  const pts = closed ? allPts.slice(0, -1) : allPts;
  const n = pts.length;
  const overLimitIdx = new Set();
  for (const c of overLimit) for (const idx of c.ringIndices) overLimitIdx.add(idx % n);

  // Fallback fillet radius from rail geometry alone: a quarter of the shorter
  // adjacent segment at each flagged corner, minimized across all flagged
  // corners so it is safe for every one; `filletOpenPolyline`'s auto-clamp
  // is a further, independent limit.
  let radius = Infinity;
  for (const idx of overLimitIdx) {
    const prev = pts[(idx - 1 + n) % n], next = pts[(idx + 1) % n];
    const dPrev = length(sub(pts[idx], prev)), dNext = length(sub(next, pts[idx]));
    radius = Math.min(radius, Math.max(1e-6, Math.min(dPrev, dNext) * 0.25));
  }

  let res = filletOpenPolyline(pts, radius, { closed, cornerFilter: overLimitIdx });
  if (!res.ok && res.maxSafeRadius > 1e-6) res = filletOpenPolyline(pts, res.maxSafeRadius, { closed, cornerFilter: overLimitIdx });
  if (!res.ok) return rail; // leaves the corner true-mitered at its extreme rather than propagating a refusal; the stretch clamp in applyTrueMiterStretch keeps it finite

  const flat = [];
  const pushPt = (p) => { if (flat.length === 0 || length(sub(p, flat[flat.length - 1])) > 1e-9) flat.push(p); };
  for (const seg of res.segments) {
    if (seg.type === 'line') { pushPt(seg.a); pushPt(seg.b); continue; }
    const arcCrv = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...seg.p0, 1], [...seg.apex, seg.weight], [...seg.p2, 1]] };
    for (let k = 0; k <= PIPE_MITER_FALLBACK_FACETS; k++) pushPt(curvePoint(arcCrv, k / PIPE_MITER_FALLBACK_FACETS));
  }
  // Usually a no-op: the segment chain already traverses the full closed loop
  // back to its start, and `pushPt`'s adjacent dedupe catches it. Kept to
  // re-close against float drift.
  if (closed) pushPt(flat[0]);
  if (flat.length < (closed ? 4 : 2)) return rail; // defensive — should be unreachable given the checks above
  return degree1RailFromPoints(flat);
}

// Applies the elliptical stretch (see above) to every interior corner ring of
// `ctrlNet`, in place — a post-pass on the free-path control net only. Each
// corner's stretch is clamped to `PIPE_MITER_LIMIT` so it stays finite;
// `applyMiterLimitFallback` has already removed every over-limit corner from
// the rail before this runs.
function applyTrueMiterStretch(rail, frames, ctrlNet) {
  const corners = railInteriorCorners(rail);
  if (corners.length === 0) return;
  const profileCount = ctrlNet.length;
  for (const corner of corners) {
    const stretch = Math.min(corner.stretch, PIPE_MITER_LIMIT);
    if (Math.abs(stretch - 1) < 1e-12) continue; // no-op, nothing to stretch
    for (const idx of corner.ringIndices) {
      const origin = frames[idx].origin;
      for (let p = 0; p < profileCount; p++) {
        const pt = ctrlNet[p][idx];
        const offset = sub(pt, origin);
        const proj = dot(offset, corner.dHat);
        const stretched = add(offset, scale(corner.dHat, (stretch - 1) * proj));
        const world = add(origin, stretched);
        ctrlNet[p][idx] = [world[0], world[1], world[2], pt[3]];
      }
    }
  }
}

// `section`'s control points are local (x,y,z) offsets from the frame origin
// (localizeSectionToFrame's output), not world coordinates; the full 3D
// shape is carried rigidly. When `railFrameOriginsExact(rail)` (degree<=1 —
// Line, Polyline), this is the free path: frames sit at the rail's Greville
// parameters and the V direction reuses the rail's degree/knots exactly (U
// reuses the section's), with no refitting, plus the true-miter stretch at
// every interior corner. A rail with no corners gets an empty corner list and
// the same rail object back, so nothing is stretched. For any degree>=2 rail
// — rational (Circle/Arc) or not (a sparse, sharply curved SketchCurve) —
// this dispatches to sweep1RigidResampled (see its header for why the reuse
// is not valid there; a 'rounded' cornerStyle rail is already degree>=2 via
// `pipeRailForSweep` and has no corners for the miter to touch).
export function sweep1Rigid(rail, section, radiusOpts = null) {
  if (!railFrameOriginsExact(rail)) return sweep1RigidResampled(rail, section, radiusOpts);
  const effectiveRail = applyMiterLimitFallback(rail);
  const frames = buildParallelTransportFrames(effectiveRail);
  const scaler = variableRadiusScaler(effectiveRail, radiusOpts);
  const ctrlNet = section.ctrlPts.map(([sx, sy, sz, sw]) => frames.map((f) => {
    let x = sx, y = sy, z = sz;
    if (scaler) { const k = scaler(f); x *= k; y *= k; z *= k; }
    const world = add(f.origin, add(add(scale(f.xAxis, x), scale(f.yAxis, y)), scale(f.zAxis, z)));
    return [...world, sw];
  }));
  applyTrueMiterStretch(effectiveRail, frames, ctrlNet);
  return {
    degU: section.degree, knotsU: section.knots,
    degV: effectiveRail.degree, knotsV: effectiveRail.knots,
    ctrlNet,
    frames, // exposed for verification and rail-frame ghost display
  };
}

// Round cap for Pipe: a hemisphere. `makeArc` (P&T Ch.7) builds a
// quarter-circle profile in the plane of the end frame's radial direction
// (`frame.xAxis`) and axial direction (`axisDir`, the frame's tangent,
// signed so the dome bulges the requested way; see the caller contract
// below), from the rim (radius r, in the frame's origin plane) to the pole
// (on the axis, r along axisDir). `revolve` (A8.1) sweeps it 360 degrees
// around the same axis, giving an exact hemisphere.
//
// Seamless by construction: the hemisphere's U=0 row (the profile's rim
// control point, local angle 0, weight 1) is built by `revolve`'s per-row
// arcSpanPoints call with the same (origin, xHat=frame.xAxis,
// yHat=cross(axisDir,frame.xAxis), radius) as `sweep1Rigid`'s tube ring at
// that frame (when axisDir === frame.zAxis), so the two rings are the same
// control points, not the same circle approximated twice, and the cap sits
// flush on the tube's end ring. (Pipe's flat cap likewise reads the real
// edge, with extractIsocurveV.)
//
// Caller contract: `frame` is `frames[0]` (rail start) or
// `frames[frames.length-1]` (rail end) of `sweep1Rigid`/
// `sweep1RigidResampled`; both guarantee that ordering on either internal
// path. `radius` is the effective rim radius at that frame, read off the
// built tube's ctrlNet (e.g. `ctrlNet[0][vIndex]`'s distance from
// `frame.origin`), never assumed constant, so a variable-radius Pipe's two
// ends each get a correctly sized dome. `axisDir` is the frame's zAxis at
// the tube's outgoing end (the dome continues past the end) or its negation
// at the start (the dome extends before it); the sign is the caller's choice.
export function pipeRoundCapSurface(frame, radius, axisDir) {
  const profile = makeArc(frame.origin, frame.xAxis, axisDir, radius, 0, Math.PI / 2);
  return revolve(profile, frame.origin, axisDir, 0, 2 * Math.PI);
}

// Blended inner/outer rim cap for a thick pipe: a rounded end shaped as a
// blend between the inner and outer walls — half a torus between them, or
// separately sized fillets at the inner and outer edges. It is revolved onto
// the two concentric walls of a Thick pipe (the app's
// `convertPipeChildToThick`).
//
// Design choice: the two walls' end rings (B at radius=outerRadius, C at
// radius=innerRadius, both at axial=0 — the tube's rim, where the flat and
// round caps read their edge) are never moved or shortened. The cap bulges
// outward from those two points, a rolled-rim bead tangent to both walls,
// with no overlap against the tube panels (the relationship
// `pipeRoundCapSurface`'s hemisphere has with its tube). Filleting inward,
// receding into the wall like an edge break, is also valid but would need
// the tube panels' V domain to stop short of the rim
// (arc-length-along-the-rail bookkeeping), which is not built.
//
// The arc construction is `filletCornerArc`'s, aimed at a virtual vertex
// placed `radius` outward along the axis from the rim, so the fillet's
// trimmed start lands exactly on the rim. For the outer rim, vertex
// `Vout = B + rOut*axisDir`, arriving direction `dIn = normalize(Vout-B) =
// axisDir` (the wall's tangent), leaving direction `dOut = -frame.xAxis`
// (radially inward, the direction the middle section runs). `frame.xAxis`
// and `axisDir` are orthonormal, so the turn is 90 degrees, `halfPhi=45deg`
// and `trim = rOut*tan(45deg) = rOut`, which puts `filletCornerArc`'s
// `p0 = vertex - dIn*trim = (B+rOut*axisDir) - axisDir*rOut = B` exactly on
// the rim. The inner rim mirrors this (vertex `Vin = C + rIn*axisDir`,
// arriving along `+frame.xAxis`, leaving toward the rim along `-axisDir`),
// landing its `p2` exactly on C.
//
// Auto-clamp, not refusal, as for every other fillet caller: each corner's
// middle-facing point sits at radius `outerRadius-rOut` (outer) /
// `innerRadius+rIn` (inner). If `rOut+rIn` exceeds the wall thickness
// `outerRadius-innerRadius`, the middle section would fold back on itself, so
// both radii are scaled down proportionally, keeping their ratio, to 0.999
// of the wall. Near that limit the two middle-facing points share a radius
// but sit at axial heights rOut and rIn, so they meet in a single point (a
// torus segment) only when the radii are equal. The applied radii are
// returned so a caller can write back the value actually used (as with
// `pipeRailForSweep`'s `cornerRadius` clamp).
//
// A radius of 0 on either rim leaves that corner sharp (the middle section
// runs straight to the rim point), e.g. a rolled lip on the outer edge only.
// Both radii 0 give a single straight line from B to C, a flat annular disk.
export function pipeBlendCapSurface(frame, outerRadius, innerRadius, axisDir, outerFilletRadius, innerFilletRadius) {
  const wall = Math.max(0, outerRadius - innerRadius);
  let rOut = Math.max(0, outerFilletRadius || 0);
  let rIn = Math.max(0, innerFilletRadius || 0);
  const sum = rOut + rIn;
  if (sum > 0 && sum > wall * 0.999) {
    const k = (wall * 0.999) / sum;
    rOut *= k;
    rIn *= k;
  }
  const at = (r, a) => add(frame.origin, add(scale(frame.xAxis, r), scale(axisDir, a)));
  const B = at(outerRadius, 0);
  const C = at(innerRadius, 0);
  const crossVec = cross(frame.xAxis, axisDir);
  const crossLen = length(crossVec);
  const planeNormal = crossLen > 1e-12 ? scale(crossVec, 1 / crossLen) : anyPerpendicular(axisDir);
  const segments = [];
  let midStart = B;
  if (rOut > 1e-9) {
    const Vout = at(outerRadius, rOut);
    const auxOut = at(outerRadius - 1, rOut); // any distinct point further -xAxis at the same axial height; only its direction from Vout matters
    const cb = filletCornerArc(Vout, B, auxOut, rOut, planeNormal);
    segments.push({ type: 'arc', p0: cb.p0, apex: cb.apex, p2: cb.p2, weight: cb.weight });
    midStart = cb.p2;
  }
  let midEnd = C;
  let innerArc = null;
  if (rIn > 1e-9) {
    const Vin = at(innerRadius, rIn);
    const auxIn = at(innerRadius + 1, rIn); // any distinct point further +xAxis at the same axial height; only its direction into Vin matters
    innerArc = filletCornerArc(Vin, auxIn, C, rIn, planeNormal);
    midEnd = innerArc.p0;
  }
  if (length(sub(midEnd, midStart)) > 1e-9) segments.push({ type: 'line', a: midStart, b: midEnd });
  if (innerArc) segments.push({ type: 'arc', p0: innerArc.p0, apex: innerArc.apex, p2: innerArc.p2, weight: innerArc.weight });
  if (!segments.length) segments.push({ type: 'line', a: B, b: C }); // both radii ~0: a flat annular disk
  const profile = filletSegmentsToCurve(segments);
  const srf = revolve(profile, frame.origin, axisDir, 0, 2 * Math.PI);
  return { srf, appliedOuterFilletRadius: rOut, appliedInnerFilletRadius: rIn };
}

// Sweep1 with N profiles (N>=2): a rail plus several cross-section curves,
// each positioned near the rail station it belongs to, blended in between,
// the result following the rail's curved shape between stations. Three
// design questions:
//
// Q1 (profile-to-rail correspondence): each profile's station is the closest
// point on the rail to the profile's centroid (an unweighted average of its
// control points — a representative anchor, not its geometric centroid).
// Cross-sections are drawn near their intended station, and this needs
// nothing beyond `closestPointOnCurve` (kernel/curve.mjs). A rail that loops
// back near itself can make this ambiguous (two rail parameters equally
// close to one profile); `closestPointOnCurve`'s ambiguity flag becomes a
// named error rather than a possibly wrong pick.
//
// Q2 (a frame at an arbitrary rail parameter): handled by
// `buildParallelTransportFrames`'s `extraParams`; this function passes the
// profile stations and a dense set of in-between rail parameters (see Q3).
//
// Q3 (how the cross-section varies between profiles). Two constructions
// fail. (1) Running loft()'s global-interpolation solve directly on each
// profile's world-space points, using the rail only to set each station's V
// parameter, is not a sweep: with only the 2 endpoint profiles the
// V-direction curve is degree 1, a straight line between the two world
// points. On a 90-degree, radius-100 arc rail that puts the v=0.5
// cross-section at the chord's midpoint (51, 53) instead of the rail's
// curved midpoint (70.71, 70.71).
// (2) Lofting in local frame-relative space at each profile's position and
// transporting every row of the interpolated control net through a
// continuously varying frame is wrong for N>=3: pairing solved control point
// k with a frame at profile k's station parameter assumes the control point
// sits at that parameter, and it does not. Only the solved curve, evaluated
// at a data parameter, reproduces that data point — true for N=2, a degree-1
// clamped spline whose 2 control points are the 2 data points, but false for
// a higher-degree curve's interior control points.
//
// The construction below never treats a solved control point as if it sat
// at a station parameter:
//  1. Each profile is localized into its own station's frame
//     (`localizeSectionToFrame`), a rigid, exact decomposition.
//  2. For each U-sample row, a local NURBS curve is interpolated (loft()'s
//     global-interpolation technique, on local offsets instead of world
//     points) through the n stations' local data at their vbar parameters.
//     Only its value is used, never its control points; interpolation
//     guarantees that value reproduces the local data at each vbar station.
//  3. That local curve is evaluated at a dense set of V parameters — every
//     station exactly, plus in-between samples per gap — and each sample is
//     transformed to world space by the rail frame at that same parameter
//     (`buildParallelTransportFrames`'s `extraParams`, arc-length-inverted
//     via `buildArcLengthTable`/`paramAtArcLength`, kernel/curve.mjs).
//     Station samples reuse the frame object used for localization (one
//     merged `buildParallelTransportFrames` call), so at v=vbar_j the dense
//     world sample is frame_j(localCurve(vbar_j)) = frame_j(local station
//     data) = the original profile's world point, exactly.
//  4. A second, final global interpolation (loft()'s technique, on this
//     dense world-space grid) builds the returned control net. The data
//     include each station's exact world point, so the final surface
//     reproduces every profile exactly at its station, while the path between
//     stations comes from the rail frame sampled at intermediate rail
//     parameters rather than a straight blend of neighboring world points.
//     Exactness at stations holds because it rests on curve values at data
//     parameters; curvature between stations holds because the frame is
//     sampled continuously along the rail.
//
// Known limitation: the local blend (step 2) interpolates each station's
// frame-relative offsets without regard to rotational "clocking". Two
// similar profiles stationed with very different rotational alignment
// relative to their (parallel-transported, twist-minimizing) frames can
// blend into an unintuitive mix of the two orientations rather than a
// smooth rotation — the seam/clocking problem Sweep tools address with a
// seam-alignment UI. Simple or circular profiles on twist-minimizing frames
// do not reach it.
//
// Inherited from loft(): U-direction (cross-section point) correspondence
// is by relative parameter fraction, so profiles with very different point
// distributions can loft oddly across U.
const SWEEP_N_STATION_REL_TOL = 1e-6; // minimum relative rail-parameter gap between two profile stations
const SWEEP_N_INTERIOR_SAMPLES_PER_SPAN = 8; // extra rail-frame samples between each pair of consecutive profile stations, so the surface follows the rail's curvature there instead of a straight blend
function frameToWorld(frame, [sx, sy, sz]) {
  return add(frame.origin, add(add(scale(frame.xAxis, sx), scale(frame.yAxis, sy)), scale(frame.zAxis, sz)));
}
export function sweepNProfiles(rail, profiles, uSampleCount = 24, degU = 3, degVmax = 3, interiorSamplesPerSpan = SWEEP_N_INTERIOR_SAMPLES_PER_SPAN) {
  const n = profiles.length;
  if (n < 2) throw new Error('sweepNProfiles needs at least 2 profile curves (use sweep1Rigid for a single profile)');
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];

  const stationed = profiles.map((crv, idx) => {
    const sum = crv.ctrlPts.reduce((acc, [x, y, z]) => [acc[0] + x, acc[1] + y, acc[2] + z], [0, 0, 0]);
    const centroid = sum.map((s) => s / crv.ctrlPts.length);
    const hit = closestPointOnCurve(rail, centroid);
    if (hit.ambiguous) {
      throw new Error(`sweepNProfiles: profile ${idx + 1}'s closest point on the rail is AMBIGUOUS (nearly equidistant from rail parameters ${hit.u.toFixed(6)} and ${hit.ambiguousWith.toFixed(6)}) — the rail passes near itself more than once, so there is no single honest station for this profile; reposition it further from the rail's other pass, or use a rail that doesn't loop back near itself`);
    }
    return { crv, idx, u: hit.u };
  });

  stationed.sort((a, b) => a.u - b.u);
  for (let i = 1; i < stationed.length; i++) {
    if (stationed[i].u - stationed[i - 1].u < (uMax - uMin) * SWEEP_N_STATION_REL_TOL) {
      throw new Error(`sweepNProfiles: profile ${stationed[i - 1].idx + 1} and profile ${stationed[i].idx + 1} both station at (nearly) the same rail parameter (~${stationed[i].u.toFixed(6)}) — two cross-sections can't occupy the same point along the rail; reposition one of them`);
    }
  }

  // Rail arc-length fraction: a Sweep has a rail to measure position along
  // (a Loft has only chord length of grid points; see kernel/loft.mjs). One
  // shared dense arc-length table drives both directions (a station's
  // fraction via `arcLengthAtParam`, an interior sample's rail parameter via
  // `paramAtArcLength` below). A separately derived coarse length for either
  // side leaves the last station's vbar slightly short of 1.0, pushing the
  // surface's v=1 edge into extrapolation past its last data point.
  const railTable = buildArcLengthTable(rail, uMin, uMax);
  const railLen = railTable.total;
  const vbar = stationed.map((s) => arcLengthAtParam(railTable, s.u) / railLen);
  for (let i = 1; i < vbar.length; i++) {
    if (vbar[i] - vbar[i - 1] < 1e-9) {
      throw new Error('sweepNProfiles: two profile stations map to the same rail arc-length position despite different rail parameters — the rail has a zero-length span between them');
    }
  }

  // Dense V-sample sequence: every station's own vbar (exact; the exactness
  // argument depends on it) plus `interiorSamplesPerSpan` arc-length-spaced
  // samples strictly between consecutive stations, which make the surface
  // follow the rail's curvature there (see the header comment). An interior
  // sample's rail parameter is the arc-length inversion of its fraction,
  // approximate to `railTable`'s tolerance, which suffices for a sample that
  // is not a station.
  const vDense = [];
  const uForDense = []; // rail parameter for each vDense entry; station entries reuse the exact station parameter
  const stationDenseIndex = new Array(n);
  for (let j = 0; j < n; j++) {
    stationDenseIndex[j] = vDense.length;
    vDense.push(vbar[j]);
    uForDense.push(stationed[j].u);
    if (j < n - 1) {
      const v0 = vbar[j], v1 = vbar[j + 1];
      for (let k = 1; k <= interiorSamplesPerSpan; k++) {
        const t = k / (interiorSamplesPerSpan + 1);
        const v = v0 + (v1 - v0) * t;
        vDense.push(v);
        uForDense.push(paramAtArcLength(railTable, v * railLen));
      }
    }
  }

  // One merged frame call for the station frames (used for localization and
  // for reproducing each profile exactly) and the dense in-between frames
  // (used for curvature), so all of them lie on one parallel-transport chain.
  const framesAll = buildParallelTransportFrames(rail, uForDense);
  const denseFrames = framesAll.extra; // one per vDense entry, same order
  const stationFrames = stationDenseIndex.map((idx) => denseFrames[idx]);
  const localizedProfiles = stationed.map((s, i) => localizeSectionToFrame(s.crv, stationFrames[i]));

  // U grid in local (frame-relative) space. The chord-length parametrization
  // is the same as with world points (a rigid per-profile transform preserves
  // within-profile distances), but the interpolated values are comparable
  // offsets from a rail frame rather than raw world positions.
  const localGrid = [];
  for (let i = 0; i < uSampleCount; i++) {
    const t = i / (uSampleCount - 1);
    localGrid.push(localizedProfiles.map((crv) => {
      const u0 = crv.knots[0], u1 = crv.knots[crv.knots.length - 1];
      return curvePoint(crv, u0 + t * (u1 - u0));
    }));
  }

  const dV = Math.min(degVmax, n - 1); // degree of the local scaffold curve (step 2 above); its control points are never read directly
  const dU = Math.min(degU, uSampleCount - 1);

  const ubarSum = new Array(uSampleCount).fill(0);
  for (let j = 0; j < n; j++) {
    const col = localGrid.map((row) => [...row[j], 1]);
    const u = chordLengthParams(col);
    for (let i = 0; i < uSampleCount; i++) ubarSum[i] += u[i];
  }
  const ubar = ubarSum.map((s) => s / n);
  const knotsU = averagingKnotVector(ubar, dU);
  const knotsV = averagingKnotVector(vbar, dV);

  const R_local = Array.from({ length: uSampleCount }, () => new Array(n));
  for (let j = 0; j < n; j++) {
    const col = localGrid.map((row) => [...row[j], 1]);
    const ctrl = interpAtParams(col, dU, ubar, knotsU);
    for (let i = 0; i < uSampleCount; i++) R_local[i][j] = ctrl[i];
  }

  // Final V degree/knots over the dense sample sequence, not just the n
  // stations, so the final surface carries the rail's curvature between
  // stations.
  const denseCount = vDense.length;
  const dV2 = Math.min(degVmax, denseCount - 1);
  const knotsVDense = averagingKnotVector(vDense, dV2);

  const ctrlNet = R_local.map((row) => {
    // Step 2: the local scaffold curve for this U-row, solved by
    // interpAtParams through the n stations' local data at their vbar.
    // `row` is data, not control points; only this curve's value is read
    // below, and it reproduces `row` exactly at each vbar station.
    const localCtrl = interpAtParams(row, dV, vbar, knotsV);
    const localRowCrv = { degree: dV, knots: knotsV, ctrlPts: localCtrl };
    // Steps 3 and 4: evaluate the local curve at every dense V sample,
    // transform through the frame at that parameter, then interpolate
    // through every dense sample (stations included) for this row's
    // world-space V control points.
    const worldRow = vDense.map((v, m) => [...frameToWorld(denseFrames[m], curvePoint(localRowCrv, v)), 1]);
    return interpAtParams(worldRow, dV2, vDense, knotsVDense);
  });

  return {
    degU: dU, knotsU, degV: dV2, knotsV: knotsVDense, ctrlNet,
    stations: stationed.map((s, i) => ({ profileIndex: s.idx, u: s.u, v: vbar[i] })),
    frames: stationFrames,
    localizedProfiles,
    ubar, // exposed for verification: surfacePoint(srf, ubar[i], v) reproduces each profile's resample exactly at that profile's station v
  };
}

// Sweep2: a two-rail sweep with one profile, whose two endpoints ride along
// two independent rail curves, its width and orientation adjusting station
// by station to the rails' separation and relative position. This is not a
// degenerate Gordon surface: a network surface needs a family of 2+
// profiles, and a single profile is one constraint, not a family.
//
// Stationing: the rails are reparametrized to a shared dense station set
// rather than combined as raw control nets (as in loft()/sweepNProfiles/
// gordonNetworkSurface). Each rail is arc-length-parametrized
// (buildArcLengthTable) and sampled at the same relative arc-length
// fractions v in [0,1] — R1(v) via paramAtArcLength(table1, v*len1), R2(v)
// likewise on rail 2 — so the rails need not share knot vector, degree or
// curve type. The station set is the union of both rails' knot-span
// boundaries (each converted to its own arc-length fraction) plus
// `stationSamplesPerSpan` interior samples per gap (denseWithInterior,
// kernel/loft.mjs), so a sharp feature in either rail gets a station.
//
// Local frame per station: origin = R1(v), anchoring at the rail side as
// Sweep1 does; the choice does not affect exactness, since the profile's
// other endpoint reaches R2 through the width axis. xAxis (the width axis) =
// normalize(R2(v) - R1(v)). The y/z axes are parallel-transported along v by
// buildParallelTransportFrames' recurrence: seed with anyPerpendicular at
// the first station, then project the previous normal into the new xAxis's
// perpendicular plane and re-normalize. It is driven by the width axis
// instead of a rail tangent, since the R2-R1 direction can rotate between
// stations as a tangent can.
//
// Profile anchors: the profile's two endpoints (first/last control points,
// exact on a clamped NURBS curve) are the points made to land on R1(v)/R2(v)
// at every station. The profile is localized once into its own static frame
// (localizeSectionToFrame) — origin = its start point, xAxis =
// normalize(end-start) — so its localized start is (0,0,0) and its localized
// end is (width0,0,0).
//
// Scale: at each station the profile's local (a,b,c) offsets are scaled by
// one factor, width(v)/width0, on all three local axes — a uniform scale,
// not "width stretches, depth stays a fixed size". As the rails converge
// (width(v) -> small), a fixed-depth profile would still stand a full-size
// cross-section out of an almost-single point; a uniformly scaled one
// shrinks toward a point, as a cone's cross-section does toward its apex.
// (Rhino exposes a "maintain height" option for the other behavior; it is
// not built here.)
//
// Exactness: the localized start/end are (0,0,0)/(width0,0,0), and the
// per-station transform maps them to R1(v) and R1(v) + width(v)*xAxis =
// R1(v) + (R2(v)-R1(v)) = R2(v) (xAxis is normalize(R2(v)-R1(v))), at every
// dense station. The final interpolation reproduces every dense sample, so
// the returned surface's u=0 and u=1 edges pass through the rails at every
// station.
//
// Refused: a profile whose start and end coincide (no width axis; a closed
// profile has no anchor pair), and rails that cross or coincide at a station
// (the width axis is undefined there). Each is a named error, not a
// zero-width sliver or a NaN surface.
//
// Known limitation: exactly one profile. An N-profile Sweep2 (several
// cross-sections, each anchored to both rails, blended between) is not
// built.
const SWEEP2_STATION_SAMPLES_PER_SPAN = 8;
const SWEEP2_MIN_WIDTH = 1e-6;
export function sweep2(rail1, rail2, profile, opts = {}) {
  const stationSamplesPerSpan = opts.stationSamplesPerSpan ?? SWEEP2_STATION_SAMPLES_PER_SPAN;
  const uSampleCount = opts.uSampleCount ?? 24;
  const degU = opts.degU ?? 3, degV = opts.degV ?? 3;

  const u1min = rail1.knots[0], u1max = rail1.knots[rail1.knots.length - 1];
  const u2min = rail2.knots[0], u2max = rail2.knots[rail2.knots.length - 1];
  const table1 = buildArcLengthTable(rail1, u1min, u1max);
  const table2 = buildArcLengthTable(rail2, u2min, u2max);
  const len1 = table1.total, len2 = table2.total;

  // Shared dense station fractions: the union of both rails' knot-span
  // boundaries (each converted to its own arc-length fraction), plus
  // interior samples per gap.
  const fracSet = new Set([0, 1]);
  const addBoundaries = (rail, table, len) => {
    const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
    for (const k of rail.knots) {
      if (k > uMin && k < uMax) fracSet.add(arcLengthAtParam(table, k) / len);
    }
  };
  addBoundaries(rail1, table1, len1);
  addBoundaries(rail2, table2, len2);
  const boundaries = [...fracSet].sort((a, b) => a - b);
  const vStations = denseWithInterior(boundaries, stationSamplesPerSpan);

  // Per-station rail points, width axis/magnitude, and the degenerate
  // (coincident/crossing rails) check.
  const stationData = vStations.map((v, k) => {
    const p1 = curvePoint(rail1, paramAtArcLength(table1, v * len1));
    const p2 = curvePoint(rail2, paramAtArcLength(table2, v * len2));
    const widthVec = sub(p2, p1);
    const width = length(widthVec);
    if (width < SWEEP2_MIN_WIDTH) {
      throw new Error(`sweep2: rail 1 and rail 2 coincide (or cross) at rail-fraction v=${v.toFixed(6)} (station ${k + 1}/${vStations.length}) — the width axis between them is undefined there; the two rails must stay genuinely separated along their full shared length`);
    }
    return { v, p1, p2, width, xAxis: normalize(widthVec) };
  });

  // Parallel-transport the width axis's own perpendicular frame along v —
  // structurally identical to buildParallelTransportFrames, driven by
  // xAxis (the width direction) instead of a single rail's tangent.
  let prevNormal = null;
  for (const s of stationData) {
    const normal = prevNormal
      ? normalize(sub(prevNormal, scale(s.xAxis, dot(prevNormal, s.xAxis))))
      : anyPerpendicular(s.xAxis);
    s.yAxis = normal;
    s.zAxis = cross(s.xAxis, normal);
    prevNormal = normal;
  }

  // The profile's static local frame and localization
  // (localizeSectionToFrame); its start/end control points become
  // (0,0,0)/(width0,0,0) in local coordinates.
  const start = [profile.ctrlPts[0][0], profile.ctrlPts[0][1], profile.ctrlPts[0][2]];
  const endCp = profile.ctrlPts[profile.ctrlPts.length - 1];
  const end = [endCp[0], endCp[1], endCp[2]];
  const width0Vec = sub(end, start);
  const width0 = length(width0Vec);
  if (width0 < SWEEP2_MIN_WIDTH) {
    throw new Error('sweep2: the profile curve needs distinct start/end points (its own two endpoints define the rail-anchor width axis) — this profile\'s start and end are coincident or nearly so (a closed/periodic profile has no defined anchor pair)');
  }
  const xAxisP = normalize(width0Vec);
  const yAxisP = anyPerpendicular(xAxisP);
  const zAxisP = cross(xAxisP, yAxisP);
  const localizedProfile = localizeSectionToFrame(profile, { origin: start, xAxis: xAxisP, yAxis: yAxisP, zAxis: zAxisP });

  // Dense U (profile-direction) samples of the localized profile, by
  // loft()'s relative-parameter-fraction resampling convention.
  const lp0 = localizedProfile.knots[0], lp1 = localizedProfile.knots[localizedProfile.knots.length - 1];
  const localSamples = [];
  for (let i = 0; i < uSampleCount; i++) {
    const t = i / (uSampleCount - 1);
    localSamples.push(curvePoint(localizedProfile, lp0 + t * (lp1 - lp0)));
  }

  // Per-(u,v) world sample grid: one uniform scale factor (width(v)/width0,
  // see above) on all three local axes, then the station's frame. Local
  // (0,0,0) maps to R1(v) and (width0,0,0) to R2(v) at every station.
  const grid = localSamples.map((lpt) => stationData.map((s) => {
    const k = s.width / width0;
    return [
      s.p1[0] + k * (lpt[0] * s.xAxis[0] + lpt[1] * s.yAxis[0] + lpt[2] * s.zAxis[0]),
      s.p1[1] + k * (lpt[0] * s.xAxis[1] + lpt[1] * s.yAxis[1] + lpt[2] * s.zAxis[1]),
      s.p1[2] + k * (lpt[0] * s.xAxis[2] + lpt[1] * s.yAxis[2] + lpt[2] * s.zAxis[2]),
    ];
  }));

  // One final global surface interpolation (loft()'s two-pass interpAtParams
  // technique) through the dense grid. U: chord-length parameters averaged
  // across all v-rows (loft()'s convention; the profile's internal point
  // distribution needs a shared knot vector). V: the shared
  // arc-length-fraction stations directly (as in sweepNProfiles and
  // gordonNetworkSurface), since a station parameter already exists.
  const n = vStations.length;
  const dV = Math.min(degV, n - 1);
  const dU = Math.min(degU, uSampleCount - 1);
  const vbar = vStations;
  const knotsV = averagingKnotVector(vbar, dV);

  const ubarSum = new Array(uSampleCount).fill(0);
  for (let j = 0; j < n; j++) {
    const col = grid.map((row) => [...row[j], 1]);
    const u = chordLengthParams(col);
    for (let i = 0; i < uSampleCount; i++) ubarSum[i] += u[i];
  }
  const ubar = ubarSum.map((s) => s / n);
  const knotsU = averagingKnotVector(ubar, dU);

  const R = Array.from({ length: uSampleCount }, () => new Array(n));
  for (let j = 0; j < n; j++) {
    const col = grid.map((row) => [...row[j], 1]);
    const ctrl = interpAtParams(col, dU, ubar, knotsU);
    for (let i = 0; i < uSampleCount; i++) R[i][j] = ctrl[i];
  }
  const ctrlNet = R.map((row) => interpAtParams(row, dV, vbar, knotsV));

  return {
    degU: dU, knotsU, degV: dV, knotsV, ctrlNet,
    // Exposed for verification: surfacePoint(srf, knotsU[0]/knotsU[last], vStations[k])
    // reproduces rail1/rail2's curvePoint at that station; `widths` shows the
    // profile's scale varying by station; `ubar` (the U chord-length
    // parameter of row i) identifies which resampled row corresponds to a
    // given profile-domain sample, as sweepNProfiles' `ubar` does.
    vStations, width0,
    widths: stationData.map((s) => s.width),
    ubar,
  };
}
