// Sweep1, P0-scoped: a single cross-section (planar or not — its full 3D
// shape is preserved) carried rigidly along a rail curve by a parallel-
// transport frame. This is Sweep1 at the fidelity the teapot
// integration test needs (a tube-like spout/handle); full Sweep1
// (Roadlike/Freeform orientation styles, multi-section Refit/Rebuild,
// closed-rail seam handling) is P4 work, not P0.
//
// Parallel transport (not Frenet) deliberately: Frenet frames flip sign or
// go undefined wherever curvature vanishes (a straight run in the rail), the
// exact shape a spout/handle rail can have. Parallel transport carries the
// previous frame's normal forward by projecting it into the new tangent's
// perpendicular plane and re-normalizing — well-defined everywhere a tangent
// is well-defined, and "all frames orthonormal" (the teapot test's own
// acceptance line) holds by construction, not by luck.

import { curvePointAndTangent, curvePoint, grevilleAbscissae, closestPointOnCurve, buildArcLengthTable, paramAtArcLength, arcLengthAtParam, isCurveClosed, rationalCurveDerivs } from './curve.mjs';
import { add, sub, scale, dot, cross, normalize, anyPerpendicular, length } from './vec3.mjs';
import { chordLengthParams, averagingKnotVector, interpAtParams } from './interpolate.mjs';
import { concatTwoC0 } from './knots.mjs';
import { denseWithInterior } from './loft.mjs';
import { filletOpenPolyline, filletCornerArc, filletSegmentsToCurve, makeArc, revolve } from './primitives.mjs';

// A profile curve is picked anywhere in the document, in world-space
// coordinates — but sweep1Rigid (below) expects it as LOCAL (x,y,z)
// offsets from a frame origin. Re-projects the picked curve's own
// control points onto a frame's own X/Y/Z axes (a plain dot-product
// decomposition) — since the frame's basis is orthonormal, decompose-
// then-reconstruct through THIS SAME frame is an exact round-trip
// regardless of what the frame's own axes actually point in (see
// test/loft.test.mjs's own sibling reasoning for the analogous corner-
// exactness property). Carries the FULL 3D shape, including whatever
// depth the profile has along the frame's own tangent (zAxis) — a
// profile isn't required to be flat/planar-perpendicular to the rail;
// real rigid transport preserves its actual shape. (Earlier versions of
// this function dropped the tangent-direction component entirely,
// silently flattening any non-planar profile into the rail's own
// perpendicular plane — found from a real swept surface that didn't
// hug its own profile curve's shape.)
export function localizeSectionToFrame(sectionCrv, frame) {
  const ctrlPts = sectionCrv.ctrlPts.map(([x, y, z, w]) => {
    const rel = sub([x, y, z], frame.origin);
    return [dot(rel, frame.xAxis), dot(rel, frame.yAxis), dot(rel, frame.zAxis), w];
  });
  return { degree: sectionCrv.degree, knots: sectionCrv.knots, ctrlPts };
}

// Frame ORIGINS are the rail's own CONTROL POINTS, not curve values sampled
// at the Greville parameters — found from a real bug: using a curve value
// as a new control point of the swept surface's V-direction (same knots,
// same degree) does NOT reproduce that curve in general (only degree-1/
// clamped-endpoint cases happen to coincide) — the mismatch is invisible
// on a gentle rail but compounds into visible twisting/ballooning on a
// sharply curved one. Control points ARE exact: a swept surface built
// from them (same knotsV/degV as the rail) reproduces the rail EXACTLY
// as one of its own edges, wherever a profile control point coincides
// with a frame's origin — the "two edges of a 4-edge patch trace the
// two touching input curves exactly" property a NURBS patch should have.
// Tangent direction still comes from the smooth curve AT that Greville
// parameter (a direction estimate only, doesn't need positional
// exactness). Honest limitation: this uses each control point's raw
// (x,y,z), ignoring weight — exact for a non-rational rail (Line/
// Polyline/SketchCurve), an approximation for a rational one (Circle
// used as a rail, whose control points sit off the true curve) — full
// per-row rational V-direction support is real P4 scope, not attempted
// here.
// `extraParams`: optional rail parameters that are NOT necessarily one of
// the rail's own control points (Sweep1-N-PROFILES' own real need — a
// profile stations itself at wherever it's closest to the rail, almost
// never exactly a control point). This is the Sweep1 N-profiles
// open design question #2 ("a frame at an ARBITRARY rail
// parameter is a real, new function to write... but the transport chain
// itself must be re-derived carefully") — resolved by EXTENDING the
// existing chain rather than re-deriving a second one: every requested
// station (control-point AND extra) is merged into ONE strictly-ordered
// sequence and the SAME incremental parallel-transport recurrence threads
// through all of them in rail-parameter order, so an extra station simply
// gets one more link in the identical chain, not a separately-derived
// formula. When `extraParams` is empty this reduces BYTE-FOR-BYTE to the
// original control-point-only behavior (proven in
// test/sweep-nprofiles.test.mjs) — existing callers (`sweep1Rigid`, the
// single-profile Sweep1 command) are completely untouched.
//
// Two real, separate rules for an origin at a merged station:
//  - A CONTROL-POINT station always uses the rail's own raw control point
//    (unchanged from before) — even where two DIFFERENT control points
//    happen to share an identical Greville parameter (a real, if rare,
//    knot-multiplicity case), each keeps its OWN distinct origin; control-
//    point stations are never merged with one another.
//  - An EXTRA station uses the actual curve evaluation at that parameter —
//    UNLESS it coincides (within a tiny tolerance, relative to the rail's
//    own parameter domain) with an existing control-point station, in
//    which case it reuses THAT control point's own frame exactly rather
//    than building a numerically-separate twin one ULP away. This is what
//    keeps a profile placed exactly at the rail's own start/end (always a
//    control-point station) reproducing the rail exactly, the identical
//    property sweep1Rigid's own single-profile case already proves.
//
// Returned as `frames` (unchanged array shape/order, one per rail control
// point) with a non-enumerable-in-spirit-but-plain `.extra` array attached
// (one entry per `extraParams`, same order) — arrays are objects in JS, so
// this is a genuine additive extension, not a shape change existing
// `frames[i]`/`frames.length` callers can observe.
export function buildParallelTransportFrames(rail, extraParams = []) {
  const gParams = grevilleAbscissae(rail);
  const domainSpan = Math.max(1e-12, Math.abs(rail.knots[rail.knots.length - 1] - rail.knots[0]));
  const DEDUPE_TOL = domainSpan * 1e-9;
  const lastCtrlIndex = rail.ctrlPts.length - 1;

  const steps = gParams.map((u, i) => ({ u, ctrlIndex: i }));
  const extraStep = new Array(extraParams.length);
  extraParams.forEach((u, j) => {
    // DEDUPE-SUBSTITUTION SAFETY GATE — found directly while
    // verifying the per-rail-span Pipe-corner fix (this file's own "PER-
    // RAIL-SPAN COMPOSED FIT" comment) via a real live point-drag: an
    // `extraParams` sample that lands within `DEDUPE_TOL` of an INTERIOR
    // control point's own Greville parameter used to ALWAYS reuse that raw
    // control point's frame — correct reasoning ONLY for a degree<=1 rail
    // (this file's own header: "control points ARE exact... an
    // approximation for a rational one"), or for the two DOMAIN ENDPOINTS
    // of ANY-degree rail (a clamped curve's first/last control point always
    // equals its own curve value there, full clamping's own standard
    // property — not degree-specific). For a degree>=2 (rational/arc) rail,
    // an INTERIOR control point sits OFF the true curve — a fillet arc's
    // own weighted "apex" control point especially so. `denseRailFrames`
    // (below) samples arc-length-fair points densely through EVERY span for
    // exactly this reason (never reusing a raw off-curve control point
    // except at the two clamped ends) — but a genuinely symmetric arc's own
    // dense arc-length sampling can land EXACTLY on that same apex control
    // point's Greville parameter (proven directly: reproduced live via a
    // drag-generated symmetric corner, u=5.5 for a [5,6]-domain arc whose
    // apex Greville parameter is exactly (5+6)/2=5.5) — which, unguarded,
    // silently substituted the off-curve apex position as if it were the
    // true curve point there, corrupting one dense sample into a real,
    // sharp spike. Gated here to the ONLY cases where substitution is
    // actually exact: `rail.degree <= 1` (interior control points ARE
    // on-curve) or the station IS one of the two domain endpoints (always
    // exact, any degree). Every other caller (`sweep1Rigid`'s own
    // degree<=1 free path, `sweepNProfiles`) is untouched — this only ever
    // changes behavior for a degree>=2 rail's INTERIOR coincidence, a case
    // that was silently wrong before, never intentionally relied upon.
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
  let prevRawTangent = null; // the RAW (un-mitered) tangent of the PREVIOUS step — the physical segment direction it actually is, needed below as an interior corner's own "incoming edge" tangent
  for (let k = 0; k < steps.length; k++) {
    const step = steps[k];
    const { tangent: rawTangent } = curvePointAndTangent(rail, step.u);
    let tangent = rawTangent;

    // INTERIOR-CORNER MITER — Pipe
    // (and especially MultiPipe/T-Splines-style multipipe) still needs real
    // corner awareness... this is pretty much just a 'polypipe'." The
    // CLOSED-RAIL SEAM WELD above fixed the ONE
    // vertex where a closed rail's own repeated first/last control point
    // gets two independently-oriented frames — but explicitly deferred
    // every OTHER interior corner of a degree<=1 rail, which has the exact
    // same root cause: `curvePointAndTangent` at a control point's own
    // Greville parameter is a ONE-SIDED derivative (P&T's own FindSpan,
    // A2.1 — an interior knot u resolves to the OUTGOING span [u, next),
    // confirmed directly against kernel/basis.mjs, not assumed), so this
    // step's own `rawTangent` is only ever the OUTGOING edge's direction —
    // the ring at this vertex sat perpendicular to just one of its two
    // adjacent edges, creasing/skewing against the other. FIX: the SAME
    // technique the seam weld already uses — miter with the angle bisector
    // of the outgoing and incoming edge tangents — generalized to run
    // INSIDE this same per-step loop at every interior control-point
    // station, so `prevNormal` chains through each miter as one more link
    // in the identical parallel-transport recurrence, not a new mechanism.
    // `tIn` (the incoming edge's tangent) needs no separate curve
    // evaluation: it's simply `prevRawTangent`, the PREVIOUS step's own
    // rawTangent — for a degree<=1 rail, the segment from control point
    // i-1 to i is BOTH point i-1's outgoing edge and point i's incoming
    // edge, the same physical direction either way.
    //
    // Gated to fire ONLY at a genuine interior control-point join: degree<=1
    // (the only case a raw control point join is an actual C0 corner —
    // higher degree gives a smooth, already-continuous tangent from the
    // curve itself, nothing to miter); `ctrlIndex` strictly between the
    // rail's first and last (index 0/lastCtrlIndex are either the closed-
    // rail seam, handled by the weld block below, AFTER this loop, or an
    // open rail's own genuine open ends — not a corner either way); and the
    // immediately PRECEDING step in this loop's own u-sorted order is
    // exactly control point ctrlIndex-1 with nothing (no inserted
    // `extraParams` sample) between them — the one case `prevRawTangent`
    // would NOT be this corner's true incoming edge. (`sweepNProfiles` can
    // in principle call this function with `extraParams` on a degree<=1
    // rail; when an extra sample lands between two control points, this
    // check correctly declines to miter rather than mitering against the
    // wrong tangent — an honest, narrow scope-hold, not a silent bug.)
    const isInteriorCorner = rail.degree <= 1 && step.ctrlIndex !== null
      && step.ctrlIndex > 0 && step.ctrlIndex < lastCtrlIndex
      && k > 0 && steps[k - 1].ctrlIndex === step.ctrlIndex - 1;
    if (isInteriorCorner) {
      const tOut = rawTangent;
      const tIn = prevRawTangent;
      // COLLINEAR NO-OP, deliberately checked BEFORE any bisector math runs
      // at all (not merely relied on to reduce to the same value
      // algebraically): a vertex where the rail doesn't actually turn
      // (tOut and tIn already equal) must leave `tangent` completely
      // untouched — the single most important regression guard for this
      // fix, proving it changes nothing where there's nothing to fix.
      if (dot(tOut, tIn) < 1 - 1e-9) {
        // Same two named, honest degenerate-corner fallbacks as the seam
        // weld, verbatim: a near-180-degree fold-back (edges point nearly
        // exactly opposite, bisector sum length ~0) falls back to the
        // outgoing tangent rather than a degenerate bisector direction.
        const sum = add(tOut, tIn);
        const sumLen = length(sum);
        tangent = sumLen > 1e-9 ? scale(sum, 1 / sumLen) : tOut;
      }
    }

    // The SAME guarded projection (with an `anyPerpendicular` fallback) the
    // seam weld already established for its own miter normal, applied at
    // EVERY station rather than only at an interior-corner one. In the
    // non-degenerate case it computes the identical `normalize(sub(...))`
    // an unguarded projection would, with one extra (equal-valued) length
    // check first — so this is a no-op everywhere the projection is
    // well-defined.
    //
    // IT IS NOT A NO-OP WHERE IT ISN'T, and that case is ordinary rather
    // than exotic. A degree<=1 rail's tangent changes DISCONTINUOUSLY at a
    // corner, so an EXTRA (arc-length) station landing just past one
    // inherits `prevNormal` from before the turn — and if the rail turns
    // into that normal's own direction, the projection is exactly zero.
    // Reached by a rail as ordinary as one running along X and then
    // straight up Z: `anyPerpendicular([1,0,0])` is [0,0,1], which is the
    // second leg's own tangent. The interior-corner guard could not cover
    // it, because an extra station is by definition not a control point.
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
  // CLOSED-RAIL SEAM WELD — a
  // closed pentagon Polyline rail piped at a default radius produced a
  // hollow, disconnected-looking gap at one point around the loop. Root
  // cause, confirmed numerically (see this kernel's matching test):
  // for a CLOSED rail (its own first
  // and last control points coincide — `isCurveClosed`, the same
  // convention `getProfileCrv`'s closed-Polyline branch and
  // `getCapProfilePoints` already use — the whole reason a repeated final
  // control point exists at all), `frames[0]` and `frames[frames.length-1]`
  // are the SAME physical vertex (their origins already coincide exactly,
  // proven above) but were built by two completely independent steps of
  // this loop: frame[0] seeds fresh with `anyPerpendicular` off the
  // OUTGOING edge's tangent (the first segment leaving that vertex);
  // frame[last] carries whatever normal the ENTIRE chain accumulated,
  // off the INCOMING edge's tangent (the last segment arriving at that
  // same vertex — a different edge of the polygon). Reusing these two
  // DIFFERENT frames as the swept surface's own v=0 and v=vMax rings (as
  // `sweep1Rigid`'s free path does, with zero closure awareness) produces
  // two independently-oriented, uncapped rings sitting on top of the same
  // 3D point — a genuinely OPEN tube at that one seam, not a rendering
  // artifact.
  //
  // THE FIX: weld the loop shut with ONE shared MITER frame at that single
  // corner, reused as BOTH `frames[0]` and `frames[frames.length-1]` — so
  // the swept surface's first and last V-rings become IDENTICAL 3D rings,
  // closing the tube with zero gap. The miter frame's tangent is the
  // angle bisector of the outgoing and incoming edge tangents (normalize
  // their sum) — the standard, well-reasoned construction for a shared
  // cross-section straddling two edges meeting at a corner. Its normal
  // continues frame[0]'s OWN original seed (`xAxis`, whatever
  // `anyPerpendicular`/the chain picked there) one more step, projected
  // into the new bisector tangent's perpendicular plane and renormalized
  // — the exact same "project + renormalize" parallel-transport step this
  // function already runs at every OTHER vertex, just one more link in
  // the identical chain rather than an arbitrary fresh pick. Two named,
  // honest degenerate-corner fallbacks (never a NaN): if the two edge
  // tangents are nearly exactly opposite (a spike/fold-back corner, sum
  // length ~0), the bisector falls back to the outgoing tangent; if
  // frame[0]'s own xAxis is itself nearly parallel to the resulting
  // bisector tangent (an extreme corner angle), the normal falls back to
  // `anyPerpendicular`, the same seed the chain's own first step uses.
  //
  // SCOPE, as originally shipped (this fix): welds only the ONE
  // seam corner (the rail's own start/end, the single vertex a closed
  // profile-curve-as-rail repeats). Every OTHER interior corner of the rail
  // kept its existing single-adjacent-tangent frame at the time — a real,
  // separate, more general robustness gap, deliberately not attempted in
  // this one.
  //
  // UPDATE: that more general gap is now also
  // fixed — see "INTERIOR-CORNER MITER" above, in the main per-step loop
  // this block runs after. The two fixes are independent and compose
  // cleanly: the loop above now miters every OTHER interior control-point
  // corner of a degree<=1 rail (a closed pentagon rail gets its 4 interior
  // corners mitered there AND its one seam vertex welded here, both correct
  // in the same tube — see test/sweep-interior-corner-miter.test.mjs's own
  // "closed pentagon" case); this block still only ever touches
  // `frames[0]`/`frames[frames.length-1]`, which the loop above never
  // treats as an interior corner (index 0/last are excluded from
  // `isInteriorCorner` by construction), so neither fix can clobber the
  // other's work.
  //
  // EXTRA-FRAME UPDATE — this weld used to ALSO silently miss `.extra`
  // entirely, which was NOT a safe non-regression the way the paragraph
  // above once assumed: `denseRailFrames`/`sweep1RigidResampled` (the ONLY
  // path a 'rounded'-cornerStyle Pipe on a CLOSED rail can reach, since
  // that composed rail is degree>=2 and fails `railFrameOriginsExact`) read
  // `.extra` exclusively — and `.extra` was captured (see the old comment
  // here, and the dedupe-substitution block above) by literally storing
  // `step.frame` object REFERENCES for whichever step each `extraParams`
  // station deduped onto, captured BEFORE this weld ran at all. Reassigning
  // `frames[0]`/`frames[frames.length-1]` (plain array-slot writes) never
  // touched those SAME underlying step objects, so any extra/dense station
  // that deduped onto the rail's own start/end control point (every closed
  // rail's own two domain-endpoint dense samples always do — see the
  // dedupe gate's own `s.ctrlIndex === 0 || s.ctrlIndex === lastCtrlIndex`
  // clause) still read the OLD, un-welded seed/accumulated frame, not this
  // weld's shared miter frame. Measured directly (not assumed) on a real
  // non-planar 7-corner closed Pipe rail with default Rounded-corner
  // params: the two dense seam frames disagreed by ~11 degrees in xAxis
  // orientation even though their origins already coincided exactly,
  // producing a real ~0.09mm gap in the swept surface's own V-control net
  // right at the seam. Fixed by mutating the underlying step objects'
  // `.frame` IN PLACE (not just the output `frames` array slots) and moving
  // the `.extra` capture to run AFTER this weld, so every `extraStep` entry
  // that dedupe-substituted onto the rail's start/end control point picks
  // up the SAME welded frame those array slots now hold — this is a
  // pure widening of what the weld already reaches, not a new mechanism;
  // `sweep1Rigid`'s own free (by-index) path is completely unaffected
  // (byte-identical), since it never reads `.extra` at all.
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

  // Captured HERE, after the weld above (not before it, as originally
  // shipped — see that block's own "EXTRA-FRAME UPDATE" note) so
  // that any dedupe-substituted extra/dense station at the rail's own
  // start/end control point reflects the welded seam frame too.
  frames.extra = extraStep.map((s) => s.frame);

  return frames;
}

// Whether a rail's own control-point weights deviate from 1 — a Circle/Arc
// built via this kernel's standard closed-form tangent-line-intersection
// construction (kernel/primitives.mjs's makeArc) is RATIONAL: roughly half
// its control points are genuine off-curve tangent-line corners (radius /
// cos(dtheta/2), ~41% farther from center than the true curve for a
// 90-degree span), carrying a weight below 1. Kept as its own named
// predicate even though it is NO LONGER
// the dispatch condition below (see `railFrameOriginsExact`) — rationality
// is A cause of off-curve control points, not the ONLY one (a real,
// separate bug — see that function's own comment).
export function railIsRational(rail) {
  return rail.ctrlPts.some((cp) => Math.abs(cp[3] - 1) > 1e-9);
}

// Whether a rail's own raw control points are GUARANTEED to sit exactly on
// its own curve — true ONLY for degree<=1 (Line, Polyline: a piecewise-
// linear curve's control polygon literally IS the curve, by construction,
// regardless of weight). This is the REAL condition sweep1Rigid's free
// reuse-trick path needs, not `railIsRational` above  —
// from a second, different failure than the
// rational-rail fix: a hand-drawn 3-point, degree-2, NON-rational
// SketchCurve rail (a genuine hook/S shape) produced a sharp, nearly-planar
// WEDGE surface, not a twisted ribbon — a different symptom because it's a
// different mechanism. Reproduced and measured directly (not assumed),
// through the real running app: `globalCurveInterp`'s own middle control
// point, for a 3-point curve that has to bend sharply to hit all 3 data
// points with only a single quadratic span, sat 70-98 units off the TRUE
// curve on a ~260-unit rail (roughly 30% of the whole curve's own span) —
// a non-rational curve (every weight exactly 1) is not exempt from this at
// all; "rational" and "control points off-curve" are two DIFFERENT
// properties that happened to coincide for the Circle-rail
// repro, which is why that fix's own `railIsRational` dispatch, while
// correct for THAT bug, was never a general enough condition for the free
// reuse-trick path's own actual precondition. A gentle, densely-sampled,
// or genuinely straight rail keeps this deviation tiny (the teapot's own
// spout rail — also a 3-point degree-2 non-rational curve — deviates only
// ~0.13 units, confirmed directly, which is why it never surfaced this
// bug); severity scales with how sharply the rail has to bend relative to
// how many control points it has, not with any single rail TYPE.
export function railFrameOriginsExact(rail) {
  return rail.degree <= 1;
}

// A SMOOTH rail's own tightest bend, as a RADIUS OF CURVATURE, sampled over
// the rail's OWN knot domain.
//
// WHY THIS IS THE RIGHT MEASURE FOR A TUBE. A tube of radius r swept along a
// rail behaves locally exactly like a torus whose major radius is the rail's
// own local radius of curvature R = 1/kappa and whose minor radius is r —
// the SAME "spindle/horn torus" self-intersection `pipeRailForSweep`'s own
// corner floor already reasons about, arriving from the other direction. A
// torus self-intersects the instant its tube radius exceeds its bend radius,
// so `r > R` anywhere along the rail means the swept surface passes through
// itself there. This is real geometry, not a tessellation artifact: no
// sampling density can render it away, and a denser mesh makes it look
// worse, not better.
//
// PARTITIONS CLEANLY AGAINST THE CORNER FLOOR, by construction rather than by
// agreement. A degree<=1 rail (Line/Polyline) is piecewise straight, so its
// second derivative is identically zero and this returns Infinity — a sharp
// corner carries its curvature as a delta function no sampled derivative can
// see, and that case is exactly what the corner-fillet floor exists for. A
// rail already rounded by that floor has fillet arcs of radius >= the tube
// radius by that floor's own construction, so their curvature 1/cornerRadius
// is already inside this bound too: the two guards agree rather than
// double-clamping.
//
// SAMPLES THE REAL KNOT DOMAIN, not a presumed [0,1]. A rail can arrive here
// as a Circle (`makeArc`/`makeCircle` build a domain of [0, narcs] — [0,4]
// for a full circle) or as a fillet-composed curve whose own spans were
// concatenated onto a sequential integer domain. Sampling [0,1] would read
// only the first quarter of a circular rail and report a bend radius for a
// stretch of curve the tube never travels.
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
    if (!(speed > 0)) continue; // a genuinely degenerate station contributes nothing rather than a spurious Infinity/NaN kappa
    const kappa = length(cross(C1, C2)) / (speed * speed * speed);
    if (Number.isFinite(kappa) && kappa > kappaMax) { kappaMax = kappa; at = u; }
  }
  return { radius: kappaMax > 0 ? 1 / kappaMax : Infinity, kappa: kappaMax, u: at };
}

// The margin BOTH self-intersection guards keep off the horn-torus boundary,
// named once so they compose exactly rather than by coincidence.
//
// The measured reason for having a margin at all: clamping to EXACTLY the
// bend radius sits ON that boundary, where the tube touches itself at a
// single point and real discrete tessellation still lands adjacent samples on
// both sides of that razor-thin contact, producing a spurious ~180 degree
// fold indistinguishable from genuine self-intersection.
//
// WHY ONE CONSTANT AND NOT TWO. The corner floor raises a too-small
// cornerRadius UP to `tubeRadius * PIPE_SELF_INTERSECT_MARGIN`; this lowers a
// too-large tubeRadius DOWN to `bendRadius / PIPE_SELF_INTERSECT_MARGIN`.
// Composed on an already-corner-floored rail, whose bend radius IS that
// raised cornerRadius, those cancel to exactly the original tube radius — so
// the second guard never re-clamps what the first just made safe. Two
// independently-chosen margins (say 1.02 up and 0.98 down) would multiply to
// 0.9996 and fire a spurious sub-micron clamp with a confusing message on
// every rounded-corner Pipe in the app. Proven directly rather than reasoned
// about: test/pipe-bend-radius.test.mjs composes the two and asserts no
// clamp.
export const PIPE_SELF_INTERSECT_MARGIN = 1.02;

export function pipeSafeTubeRadius(rail, requested, samples = 257) {
  const bend = railMinBendRadius(rail, samples);
  const safeMax = bend.radius / PIPE_SELF_INTERSECT_MARGIN;
  // A RELATIVE tolerance, not a bare `>`: the composition above cancels
  // algebraically but not necessarily to the last bit in floating point, and
  // a clamp firing on a 1e-16 overshoot would be a real, visible status
  // message about a difference no geometry can express.
  if (!(requested > safeMax * (1 + 1e-9))) return { radius: requested, clamped: false, safeMax, minBendRadius: bend.radius, u: bend.u };
  return { radius: safeMax, clamped: true, safeMax, minBendRadius: bend.radius, u: bend.u };
}

// Dense arc-length-spaced rail parameters (every knot-span boundary
// exactly, plus several genuinely in-between samples per span — the same
// "station exactly, plus interior samples for curvature" pattern
// sweepNProfiles already established) and the CONTINUOUSLY-evaluated
// parallel-transport frame at each one (buildParallelTransportFrames'
// own `extraParams`, never a raw control point except at the two clamped
// domain endpoints, where curve value and control point coincide exactly
// regardless of degree/rationality). `vDense` is each sample's own true
// rail arc-length fraction (0..1) — the fresh V-parametrization
// sweep1RigidResampled (below) fits its final interpolation to.
//
// PER-SPAN SAMPLE COUNT — the fix for "a rounded corner on a closed,
// multi-corner rail produces visibly torn geometry", which is a resolution
// problem and NOT the closed-rail holonomy gap this file names and defers
// elsewhere.
//
// A HIGH V CONTROL-POINT COUNT IS NOT THE SYMPTOM. A rail's own control
// points and the swept SURFACE's V control points are different numbers:
// `sweep1RigidResampled` (below) builds a separate, denser V-resolution via
// this function, and under the OLD fixed `samplesPerSpan=12` an M-corner
// closed rail produces exactly `26*M+1` of them. Comparing the two is an
// apples-to-oranges mismatch, not evidence of a compounding bug.
//
// THE REAL DEFECT:
// a synthetic 5-corner closed rail (1mm fillet radius) measured ~19%
// cross-section ellipticity (a real, visible distortion) at the OLD fixed
// 12-samples-per-span density. This function is exact ONLY at each dense
// sample it builds (the interpolation's own data points) and merely
// INTERPOLATES between them. The composed rail is TANGENT-continuous at
// every arc/line junction (the fillet's own construction guarantees G1) but
// its CURVATURE RATE is not — a fillet arc turns at a constant nonzero rate
// while its neighboring straight run turns at exactly zero, a genuine jump
// in the THIRD derivative right at the junction. A global cubic (C2)
// interpolation through data with a derivative discontinuity of this kind
// always rings (Gibbs-phenomenon-like) near the junction — proven directly,
// not assumed: boosting ONLY the arc span's own sample count while leaving
// its straight neighbors at the old fixed count left the ellipticity
// virtually unchanged (still ~18%), but raising the floor UNIFORMLY (both
// sides of every junction) shrank it monotonically and reliably — 12->48
// samples/span dropped worst-case ellipticity from ~19% to under 1.5% on
// every synthetic rail tried (5/8/12 corners, tight-to-near-clamped fillet
// radii) — confirming genuine under-resolution of a real ringing artifact,
// not a frame-orientation/holonomy defect.
//
// FIX: `MIN_SPAN_SAMPLES` raises the floor for every span (including a
// perfectly straight one, since the ringing needs resolution on ITS side
// of the junction too, not just the curved side); `adaptiveSpanSampleCount`
// additionally scales a span's OWN count up with its real measured tangent-
// turn angle (`spanTurnAngle`) past that floor, capped at `MAX_SPAN_SAMPLES`
// so a pathological near-180-degree fold-back (already a named, separate
// degenerate case elsewhere in this file) can't run away the control-point
// count unboundedly. `samplesPerSpan` remains a caller override (still used
// nowhere today, kept for the same reason the old constant's own default
// parameter was) — passing it explicitly bypasses this measurement
// entirely, matching the exact prior behavior byte-for-byte when supplied.
const MIN_SPAN_SAMPLES = 48;
const MAX_SPAN_SAMPLES = 64;
const TARGET_ANGLE_STEP_RAD = 2 * Math.PI / 180; // ~2 degrees per dense sample
function spanTurnAngle(rail, u0, u1) {
  // Probe fractionally inside the span, not exactly at its own boundary —
  // a span boundary is frequently a C0 knot (an interior arc/line join in
  // a filleted rail), where the tangent itself is one-sided/discontinuous;
  // this function wants the span's OWN interior curvature, not a
  // neighboring span's tangent leaking in from across the joint.
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
// GENUINE STRUCTURAL BREAKS ONLY — a rail's raw `knots` array
// mixes two entirely different things: an ordinary SIMPLE (multiplicity-1)
// interior knot of an otherwise-smooth global interpolation (a SketchCurve
// rail, say) is NOT a real join at all — the curve is fully C^(p-1) THERE,
// same as anywhere else on its own span — versus a genuine C0-or-worse
// JOINT, which always carries FULL multiplicity (= degree) by construction
// (`filletSegmentsToCurve`'s own `joinCurvesC0`/`concatTwoC0`, and a
// standard 2-per-arc Circle/Arc's own tangent-corner knots). Only the
// second kind is a real seam a fresh dense sampling pass should ever treat
// specially — conflating the two would wrongly fragment an ordinary smooth
// spline rail at every one of its harmless interior knots. Mirrors
// `tessellateSurface`'s own already-shipped `m < srf.degU` gate for the
// exact same distinction, one direction over (profile/U there, rail/V
// here).
// Rescales JUST a knot vector's own numeric domain — no control points to
// touch, the narrower need `sweep1RigidResampled`'s own per-span local fits
// have versus `rescaleCurveDomain` (kernel/knots.mjs), which rescales a
// whole curve object.
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
  const hardBreakIdx = []; // indices into uDense/vDense/denseFrames of each GENUINE (hard-break) span boundary, including the two domain ends — length = hardBreaks.length + 2
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

// VARIABLE RADIUS ALONG A SWEEP1 RAIL — added directly after the
// corner-rounding fix shipped.
// Scoped to WITHIN one curve only — a single Pipe's
// own rail carries several radius breakpoints. MultiPipe's own "per curve"
// case (each independent tube in a network getting its OWN radius profile)
// is a deliberately staged, separate follow-up round, not attempted here —
// `sweepNProfiles`/`sweep2` (N-profile Sweep1, two-rail Sweep2) are
// completely untouched, on purpose.
//
// DATA MODEL: `radiusPoints` is a list of `{t, radius}` breakpoints, `t` a
// NORMALIZED RAIL-PARAMETER FRACTION (0 = rail start, 1 = rail end) — NOT
// an arc-length fraction. This matches an EXISTING convention already in
// this app, not a new invention: the Split operator's own `splitFrac`
// resolves via `dom[0] + frac*(dom[1]-dom[0])` on the surface's own raw
// knot domain, a parameter-domain fraction, the identical idea applied here
// to a rail curve instead of a surface direction. Linearly interpolated
// between the two breakpoints straddling a given `t` (this kernel's own
// "resample and interpolate through values" precedent, e.g. loft()'s
// chord-length approach) and CLAMPED to the nearest end breakpoint's own
// radius outside [minT, maxT] — never extrapolated past the ends.
//
// HONEST LIMITATION, stated plainly (not silently shipped): because `t` is
// a PARAMETER-domain fraction, not a true arc-length fraction, it only
// approximates physical position along the rail whenever the rail's own
// segments/spans have unequal length relative to their PARAMETER width —
// already true of a plain multi-segment Polyline rail today (getProfileCrv
// builds its knot vector with UNIFORM per-segment parameter width
// `[0,0,1,2,...,n-1,n-1]`, not chord-length spacing), and MORE pronounced
// once a rail is rounded via `pipeRailForSweep`/`filletSegmentsToCurve`
// (each Line/Arc segment of the composed rail gets an equal 1-unit
// parameter span via `joinCurvesC0`'s own `rescaleCurveDomain`, regardless
// of the segment's own real length — a short fillet arc gets the SAME
// parameter width as a long straight run next to it). This is why `t`
// deliberately reads from whichever RAIL is actually being swept at
// evaluate() time (see `sweep1Rigid`/`sweep1RigidResampled` below, both of
// which read `frame.u`, the real per-station rail parameter, straight off
// whatever `rail` argument was actually passed in) — a breakpoint's t=0/t=1
// ALWAYS lands EXACTLY at the rail's true start/end (fillets never move the
// rail's own endpoints), and every interior breakpoint maps CONSISTENTLY
// (same convention, same approximation character) whether the rail is
// sharp or rounded — not a NEW bug introduced by combining the two
// features, just this kernel's existing "parameter fraction, not
// arc-length fraction" characteristic, now also visible on a rounded rail.
// A true arc-length-exact remapping is real, separate, harder scope (it
// would need `buildArcLengthTable`/`paramAtArcLength`, threaded through
// BOTH the free and resampled paths) — not attempted; verified
// directly in test/sweep-variable-radius.test.mjs's own "combined with
// corner-rounding" case (finite everywhere, endpoints exact, interior
// values land APPROXIMATELY at the intended physical fraction — not
// asserted to be exact there).
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
  return last.radius; // unreachable given the bounds checks above — a defensive fallback, never actually hit
}

// Builds a per-frame SCALE-FACTOR function for `sweep1Rigid`/
// `sweep1RigidResampled` below, or `null` when there's nothing to scale —
// the SINGLE choke point both functions call, so "no radiusOpts, or exactly
// 0-1 breakpoints" (the default, constant-radius case every existing Pipe
// and every plain single-profile Sweep1 already is) returns `null` and the
// caller's own per-station loop skips the scale multiply ENTIRELY, not just
// multiplies by a computed 1.0 — the strongest form of "byte-identical to
// today's constant-radius tube," the single most important regression
// guard for this whole feature.
function variableRadiusScaler(rail, radiusOpts) {
  const pts = radiusOpts && radiusOpts.radiusPoints;
  const baseRadius = radiusOpts && radiusOpts.baseRadius;
  if (!pts || pts.length <= 1 || !(baseRadius > 1e-9)) return null;
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const span = Math.max(1e-12, uMax - uMin);
  return (frame) => radiusAtT(pts, (frame.u - uMin) / span) / baseRadius;
}

// RESAMPLED PATH (originally the Circle-rail fix;
// BROADENED to any rail whose control points aren't
// guaranteed on-curve — see `railFrameOriginsExact` above for the full
// derivation and numeric proof on both
// bugs). The reuse trick below (the free path) is exact ONLY when a rail's
// own raw control points are guaranteed to sit exactly on its curve
// (`railFrameOriginsExact` — degree<=1) — for any degree>=2 rail, rational
// (Circle/Arc) OR non-rational (a sparse SketchCurve interpolation), that
// premise can be false, so reusing those raw control points as frame
// origins can carry them substantially off the rail's true path into the
// swept surface, at every V station.
//
// This path never uses a raw control point as a frame origin except at
// the two clamped domain endpoints (where curve value and control point
// coincide exactly either way) — every other frame comes from a
// CONTINUOUS curve evaluation at a dense, arc-length-spaced parameter
// (denseRailFrames above, via buildParallelTransportFrames' own
// `extraParams`). This is the SAME "resample to a shared dense parameter
// set, then run a fresh global interpolation through VALUES, never touch
// original control nets" technique this kernel's own loft()/
// sweepNProfiles already use to combine curves whose own knot/degree
// structure isn't directly combinable (a network-surface research finding;
// sweepNProfiles' curved-rail fix above
// is the direct model this was ported from) — applied here to a SINGLE
// profile's rigid transport rather than a blend across several.
//
// Honest tradeoff, stated plainly (matching loft()'s/sweepNProfiles' own
// precedent): the free path is free (no solve) and exact at every
// continuous rail parameter, but ONLY for a degree<=1 rail; this path
// costs one real global-interpolation solve per section control point and
// is exact only at the dense sample set built above (proven in
// test/sweep-rational-rail.test.mjs and test/sweep-sparse-rail.test.mjs to
// track the rail's TRUE path within a tight tolerance at many genuinely
// in-between V values, not just at the samples) — not literally exact at
// every continuous parameter the way the free path is. Worth it
// specifically because the free path's "exact everywhere" property was
// never actually true for a degree>=2 rail to begin with, rational or not.
//
// PER-RAIL-SPAN COMPOSED FIT — fixes a real, confirmed
// geometric defect at a rounded Pipe corner on an IRREGULAR rail (a
// own "5mm fillet next to a 200mm run" report, chased to ground with real
// numbers, not assumed): the single GLOBAL cubic interpolation this
// function used to fit through the ENTIRE dense sample set at once is a
// textbook Runge's-phenomenon setup whenever the rail has a genuine C0
// joint (a fillet arc) whose own arc-length footprint is tiny next to its
// straight neighbors' — confirmed directly: the built surface's own
// tessellated adjacent-face-normal angle hit 178-180 degrees (a literal
// fold) on that irregular fixture, and the true continuous
// centerline showed a real (not tessellation-artifact) ~0.23mm
// non-monotonic wiggle on a 5mm-radius pipe.
//
// THE FIX: never fit ONE global spline across a genuine structural joint.
// `railHardBreakParams` (above) finds every GENUINE C0-or-worse break in
// the rail (full-multiplicity knots only — an ordinary smooth spline's
// simple interior knots are deliberately left alone, see that function's
// own header) and this function fits a SEPARATE local degree-3
// interpolation through EACH span's own dense samples only, then stitches
// the pieces with `concatTwoC0` (kernel/knots.mjs) — the SAME proven join
// primitive `filletSegmentsToCurve` already uses to build the rail itself
// in the first place, reused directly rather than hand-rolled. Verified
// directly (not assumed) on that irregular fixture, per-span, at a
// dense LOCAL sampling rate (300 stations per real rail span — the
// resolution that actually resolves a span this narrow, unlike a single
// coarse full-domain grid, which both over- and under-samples narrow
// spans depending on where its fixed step happens to land): the worst
// adjacent-face-normal angle in every span drops from 149-173 degrees (OLD)
// to under 0.0003 degrees (NEW) — genuinely fold-free — and the swept
// cross-section radius, which used to swing 4.88-5.25mm on a nominal 5mm
// pipe (OLD), now holds 4.999999-5.000000mm (NEW) everywhere. `numSpans<=1`
// (a plain Line/Arc/SketchCurve/loft rail with no genuine internal joint)
// takes the exact OLD single-global-fit branch, byte-identical — this
// change only ever engages for a rail that actually HAS an internal C0
// break to protect against.
//
// HONEST REMAINING LIMITATION, found and named directly rather than
// silently left in place: because `concatTwoC0` joins at plain C0 (matching
// position only, matching the SAME "formally C0, measured G1 in practice"
// contract this kernel's own composed RAIL already carries — see
// `filletSegmentsToCurve`), a genuinely severe fixture (a very short arc
// span sandwiched between much longer straight runs, combined with a wide
// RATIONAL profile control point sitting far off the tube's own axis — a
// circle's own off-curve weighted "corner" control points, ~41% farther out
// than the on-curve ones) can carry a real, tiny, sub-micron-amplitude
// direction reversal immediately AT one of these joints (measured directly:
// ~0.0003mm, 0.006% of a 5mm profile radius, versus the FIXED defect's own
// 0.23mm/4.6% — three orders of magnitude smaller). An exact fix would mean
// explicitly solving each span WITH a shared derivative constraint at every
// interior joint (`globalCurveInterpWithEndDerivs`, already in this kernel,
// generalized to a per-boundary derivative estimate) — real, separate,
// harder kernel work with its own new derivation to get the derivative
// MAGNITUDE right (not just direction), deliberately NOT attempted this
// round per this project's own standing "don't rush unverified NURBS math"
// rule; named here as real, deferred scope, not silently dropped.
//
// CLOSED-RAIL SEAM — a 4th real root cause was found and
// PARTIALLY fixed; see `buildParallelTransportFrames`'s own
// "EXTRA-FRAME UPDATE" comment for the part that IS fixed here (the
// `.extra` frame array used by THIS function now correctly reflects the
// by-index seam weld, closing a real ~0.09mm V-control-net position gap at
// the seam on a real non-planar 7-corner closed Pipe rail — confirmed
// directly, not assumed). A SECOND, SEPARATE, genuinely HARDER part of the
// same investigation was found but is EXPLICITLY NOT fixed here, named
// plainly rather than shipped half-working: even with the position gap
// closed, this function's own per-row V-curve fit (both the numSpans<=1
// global fit and the numSpans>1 per-span composed fit above) is built as a
// plain OPEN, CLAMPED interpolation with no knowledge that a CLOSED rail's
// own two domain ends should agree in DERIVATIVE, not just position —
// measured directly: the worst profile control point on the 7-corner
// fixture (a circle profile's own off-curve RATIONAL "corner" point) still
// shows a genuine several-degree tangent mismatch right at the seam after
// the position fix, present at a SIMILAR magnitude on a PLANAR closed rail
// too (directly falsifying an earlier claim that "a planar
// closed rail has zero holonomy and is unaffected" — the defect does not
// scale with non-planarity or corner angle, so that was never the
// relevant property).
//
// NOT FIXED HERE, and the two numerical traps any future fix will hit.
// Padding each seam-touching span's local fit from its neighbor is UNSAFE
// the instant the two spans have very different arc-length footprints (a
// short fillet arc next to a long straight run): the borrowed samples pack
// into a knot-density cluster and reproduce the SAME class of Runge's-
// phenomenon ringing the "PER-RAIL-SPAN COMPOSED FIT" fix above already had
// to solve once — measured as a near-180-degree chord-direction reversal
// within the first ~2% of the receiving span. Computing FRESH padding at
// the receiving span's own natural per-sample spacing fixes that, but puts
// the padding on the SAME arithmetic grid the span's dense samples already
// sit on, so the padded curve's Eq-9.8-style interior-knot averaging can
// land EXACTLY (up to float noise) back on the trim boundary, defeating
// `extractSubCurve`'s tolerance-based multiplicity counting and corrupting
// control points near that boundary — a half-step phase offset avoids it.
//
// The position fix above IS shipped, because it is fully proven. The
// tangent-only residual is NOT, and is named here rather than silently
// left for a future reader to rediscover. A genuinely periodic (not
// padded-and-trimmed) global curve interpolation, derived from first
// principles rather than adapted from `closedCurveInterp`'s point-count-
// based technique, is the most likely correct next attempt — real,
// separate, harder kernel work.
// SANITY CAP — interpAtParams' own Gauss-Jordan solve is a genuine dense
// O(n^3) linear solve ("full reduction to identity is fine at the
// control-point counts a sketched curve actually reaches — tens, not
// thousands", interpolate.mjs's own comment). denseRailFrames' own
// per-span dense sampling scales with the rail's own KNOT SPAN COUNT, not
// with genuine curvature complexity — an ORDINARY smooth rail with many
// SIMPLE interior knots (any degree-3 global-interpolation curve through
// more than a handful of points — e.g. a Curve Generator Lorenz/Random-
// Walk curve, or any hand-sketched curve with many picks) has ZERO hard
// breaks, so it collapses to the `numSpans<=1` branch below and hands its
// ENTIRE dense sample set (up to MAX_SPAN_SAMPLES per knot span) to ONE
// interpAtParams call. Confirmed live and reachable, not hypothetical: a
// Pipe run on a ~100-knot-span Lorenz curve builds a ~6000-point dense
// array there, an ~6000x6000 dense solve that hangs the tab for minutes.
// Fixed the same way this kernel already bounds every other "unbounded by
// construction, not by user intent" blowup (L-System's own
// MAX_LSYSTEM_SEGMENTS, ArrayLinear's own 500-copy cap): a uniform
// stride-based subsample down to a safe cap, BOTH domain endpoints always
// kept exactly — 300 points is already far denser than this rail
// direction needs for any real geometric/visual fidelity.
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
      // `averagingKnotVector` hardcodes ubar[0]===0/ubar[n]===1 (Eq 9.8's
      // own clamped-end convention) — this span's own slice of the GLOBAL
      // vDense must be shifted+scaled to a local [0,1] domain before that
      // call, then rescaled back to its TRUE arc-length-fraction sub-range
      // afterward (`rescaleKnotsOnly`, knots only — no control points to
      // touch) so the final composed `knotsV` stays genuinely in the true
      // [0,1] range every existing v-consuming caller already assumes
      // (`tessellateSurface`, cap-building, measure/pick all read
      // `knotsV[0]`/`knotsV[last]` dynamically, never hardcode 0/1 — found
      // and confirmed directly, not assumed, before relying on it here).
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

// TRUE MITER FOR 'sharp' CORNERS — sharp stays an option, and stays
// the default. The interior-corner miter orients
// each corner's ring correctly, but the RULED SPANS on either side of it
// still neck — the "waist" bug (fixed for ROUNDED corners by filleting the
// rail entirely). 'sharp' itself, unchanged, still ships that
// exact waist (measured 4.6194mm on a nominal 5mm pipe at a 90-degree
// corner — test/fillet-open-polyline.test.mjs's own regression baseline).
//
// THE FIX, verified against real McNeel Rhino docs and Houdini's own
// documented Sweep 2.0 "Stretch Around Turns" (default ON) behavior by
// that research: the geometrically CORRECT shared cross-section at a
// corner of turn angle theta is not a plain circle in the bisector frame —
// it is the ELLIPSE the bisector (miter) plane cuts from either adjacent
// segment's own true cylinder: unchanged radius perpendicular to the bend
// plane, stretched by sec(theta/2) along the bend-plane direction. A ruled
// span between a segment's own exact perpendicular ring and this miter
// ellipse IS that segment's exact uniform cylinder (the ellipse is, by
// definition, cylinder-intersect-bisector-plane) — the necking stops being
// an approximation to accept and becomes exactly compensated. Hand-derived
// and independently confirmed (not merely assumed) against the shipped
// `buildParallelTransportFrames` miter construction for the exact 90-degree
// L-corner fixture this kernel's own tests already use: the stretched world
// point matches the true cylinder-ellipse point exactly, algebraically, for
// every profile angle — see test/sweep-true-miter.test.mjs's own numeric
// proof for the general (not hand-derived) case.
//
// theta and the in-plane "bend direction" need no new geometry: for unit
// tangents tIn/tOut, theta = angle(tIn,tOut) (the SAME angle the interior-
// corner miter's own bisector construction already implies), and
// `tOut - tIn` is ALWAYS exactly perpendicular to `tOut + tIn` (the
// bisector/miter tangent) for unit vectors — dot(a-b,a+b) = |a|^2-|b|^2 = 0,
// an identity, not an approximation — so `normalize(tOut-tIn)` is exactly
// the bend-plane's own in-plane direction, computable with zero cross
// products and guaranteed to already lie in the corner ring's own
// cross-sectional plane (perpendicular to the bisector tangent).
//
// DEGENERATE CASE, required not optional (that same research, matching
// Houdini's documented "Max Stretch", default 10 — a corner needing more
// than 10x cross-section stretch, ~168.5 degrees of turn, falls back to a
// small rounded fillet for JUST that one corner, every OTHER corner on the
// same rail staying true-mitered): sec(theta/2) -> infinity as theta -> 180
// degrees (a near-fold-back corner). `PIPE_MITER_LIMIT` below is that same
// 10x default, named plainly rather than left a magic number — not exposed
// as a live per-object param (Houdini itself buries its
// equivalent too; a real, small, deferred v2 if a model ever needs it
// tuned).
export const PIPE_MITER_LIMIT = 10; // Houdini Sweep 2.0's own documented "Max Stretch" default
const PIPE_MITER_FALLBACK_FACETS = 8; // straight micro-facets approximating a fallback corner's own rounded arc; each facet's own turn angle is a small fraction of the original corner's — always far under PIPE_MITER_LIMIT itself (proven in test/sweep-true-miter.test.mjs, not assumed)

// Every genuine (non-collinear) corner of a degree<=1 rail, open OR closed,
// with its own turn angle (`theta`, the angle between the incoming and
// outgoing unit edge directions — a collinear run has none and is omitted
// entirely, not returned as theta=0), the bend-plane unit direction
// (`dHat`, see header comment above), and the true-miter stretch factor
// (`sec(theta/2)`) that ring needs. `ringIndices` is `[i]` for an ordinary
// interior control-point corner, or `[0, n-1]` for a CLOSED rail's own
// shared seam vertex — `buildParallelTransportFrames`' seam weld already
// makes `frames[0]` and `frames[n-1]` the SAME frame object, so stretching
// one ring column stretches both by construction.
export function railInteriorCorners(rail) {
  if (rail.degree > 1) return []; // this function's whole contract is "a degree<=1 rail's own raw control points" — a degree>1 rail's control points aren't generally ON the curve at all (a rational arc's own middle point, for instance), so there is no meaningful "corner" to report here; defensive, matching this kernel's own gating discipline (sweep1Rigid's free path never reaches this function with a degree>1 rail in the first place — see railFrameOriginsExact)
  const pts = rail.ctrlPts.map((cp) => [cp[0], cp[1], cp[2]]);
  const n = pts.length;
  if (n < 3) return [];
  const closed = isCurveClosed(rail);
  const corners = [];
  const consider = (prevPt, vPt, nextPt, ringIndices) => {
    const tIn = normalize(sub(vPt, prevPt));
    const tOut = normalize(sub(nextPt, vPt));
    const c = Math.max(-1, Math.min(1, dot(tIn, tOut)));
    if (c > 1 - 1e-9) return; // collinear: no turn, nothing to stretch — the single most important regression guard, matching the interior-corner-miter fix's own identical collinear no-op discipline
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

// MITER-LIMIT FALLBACK: for whichever corner(s) of a degree<=1 rail need
// MORE than `PIPE_MITER_LIMIT` stretch, replaces JUST that corner with a
// small faceted (straight-micro-segment) rounded fillet — reusing
// `filletOpenPolyline`'s own existing per-corner selection/trim-budget math
// (generalized with `cornerFilter`, see its own header comment
// in kernel/primitives.mjs) rather than re-deriving corner selection from
// scratch. Deliberately FLATTENS the fillet's own rational arc into
// straight facets (not `filletSegmentsToCurve`'s degree-2 composition) so
// the returned rail STAYS degree<=1 overall — every OTHER (sane) corner
// keeps going through this same free path's own per-corner true-miter
// stretch below, unaffected, and the new facet joints (tiny turn angles by
// construction) are automatically, provably far under the miter limit
// themselves (verified numerically, not assumed — see
// test/sweep-true-miter.test.mjs's own "mixed rail" case). Returns the SAME
// rail object, completely unchanged, the instant no corner exceeds the
// limit — the overwhelmingly common case, and the single most important
// regression guard for this whole mechanism (byte-identical degV/knotsV
// for every ordinary Pipe/Sweep1 rail that never gets anywhere near a
// 168-degree turn).
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

  // A generous, purely rail-geometry-derived fallback fillet radius (no new
  // Pipe-specific param needed) — a quarter of the shorter adjacent segment
  // at each flagged corner, then the SMALLEST such request across every
  // flagged corner (conservative, safe for all of them at once);
  // `filletOpenPolyline`'s own auto-clamp is a further, independent safety
  // net regardless.
  let radius = Infinity;
  for (const idx of overLimitIdx) {
    const prev = pts[(idx - 1 + n) % n], next = pts[(idx + 1) % n];
    const dPrev = length(sub(pts[idx], prev)), dNext = length(sub(next, pts[idx]));
    radius = Math.min(radius, Math.max(1e-6, Math.min(dPrev, dNext) * 0.25));
  }

  let res = filletOpenPolyline(pts, radius, { closed, cornerFilter: overLimitIdx });
  if (!res.ok && res.maxSafeRadius > 1e-6) res = filletOpenPolyline(pts, res.maxSafeRadius, { closed, cornerFilter: overLimitIdx });
  if (!res.ok) return rail; // honest fallback: leaves the corner true-mitered-but-extreme rather than propagating a refusal — the defensive stretch clamp below still keeps it finite

  const flat = [];
  const pushPt = (p) => { if (flat.length === 0 || length(sub(p, flat[flat.length - 1])) > 1e-9) flat.push(p); };
  for (const seg of res.segments) {
    if (seg.type === 'line') { pushPt(seg.a); pushPt(seg.b); continue; }
    const arcCrv = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...seg.p0, 1], [...seg.apex, seg.weight], [...seg.p2, 1]] };
    for (let k = 0; k <= PIPE_MITER_FALLBACK_FACETS; k++) pushPt(curvePoint(arcCrv, k / PIPE_MITER_FALLBACK_FACETS));
  }
  // Closing the loop back to its own start is usually a no-op here (the
  // segment chain above already traverses the FULL closed loop and returns
  // to its own starting point — `pushPt`'s own adjacent-only dedupe catches
  // it); kept as a defensive re-close in case of any float drift.
  if (closed) pushPt(flat[0]);
  if (flat.length < (closed ? 4 : 2)) return rail; // defensive — should be unreachable given the checks above
  return degree1RailFromPoints(flat);
}

// Applies the elliptical stretch (see header comment above) to every
// genuine interior corner ring of `ctrlNet`, IN PLACE — a pure post-pass on
// the already-built free-path control net, touching nothing else (no
// profile localization, no transport chain, no resampled path). A corner
// whose OWN stretch is (already, defensively) within `PIPE_MITER_LIMIT` is
// clamped there anyway — a hard safety net that should never actually
// trigger in practice (`applyMiterLimitFallback` above already removes
// every over-limit corner from `rail` before this runs), matching this
// kernel's "never NaN/Infinity" posture elsewhere.
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

// `section`'s own control points are LOCAL (x,y,z) offsets from the frame
// origin (localizeSectionToFrame's own output shape), not world
// coordinates — the full 3D shape is carried rigidly, not just an XY
// slice. When `railFrameOriginsExact(rail)` (degree<=1 — Line, Polyline),
// frames are placed at the rail's own Greville parameters and the swept
// surface's V direction reuses the rail's own degree/knots exactly (no
// re-fitting; U direction reuses the section's degree/knots) — this is
// exact, unchanged since the P0 teapot test, and BYTE-IDENTICAL to every
// prior round's own results (proven by full regression, not merely
// asserted) EXCEPT for the true-miter stretch this adds at every
// genuine interior corner (see above) — a rail with NO corners at all (a
// straight multi-point Polyline) sees `railInteriorCorners` return an empty
// list and `applyMiterLimitFallback` return the SAME rail object, so this
// round changes NOTHING for that case, still byte-identical. For any
// degree>=2 rail — rational (Circle/Arc) OR non-rational (a
// sparse/sharply-curved SketchCurve) — dispatches
// to sweep1RigidResampled above instead (completely untouched by this
// round — see that function's own header comment for why the reuse trick
// isn't valid there, and why a 'rounded' cornerStyle rail, already
// degree>=2 via `pipeRailForSweep`, has no interior corners left for this
// mechanism to touch in the first place).
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
    frames, // exposed for verification + later rail-frame ghost display
  };
}

// ROUND CAP FOR PIPE (Stage 4 — "Round cap = a hemisphere: revolve a
// quarter-arc in the end
// frame's own plane... small, self-contained"). Genuinely zero new curve
// math: `makeArc` (P&T Ch.7, already exact/proven) builds a quarter-circle
// profile lying in the plane spanned by the end frame's own radial
// direction (`frame.xAxis`) and its axial direction (`axisDir`, the frame's
// own tangent, SIGNED so the dome bulges the requested way — see the two
// call-site conventions below), running from the RIM (radius r, at the
// frame's own origin plane) to the POLE (on the axis, offset by r along
// axisDir). `revolve` (A8.1, already exact/proven) sweeps that quarter-arc
// the full 360 degrees around the SAME axis, producing an exact hemisphere.
//
// SEAMLESS BY CONSTRUCTION, not by matching afterward: the hemisphere's own
// U=0 row (the profile's own rim control point, at local angle 0, weight 1)
// is built by `revolve`'s internal per-row arcSpanPoints call using EXACTLY
// the same (origin, xHat=frame.xAxis, yHat=cross(axisDir,frame.xAxis),
// radius) as `sweep1Rigid`'s own tube-ring construction at that same frame
// (when axisDir === frame.zAxis) — the two rings are the IDENTICAL set of
// control points, not merely the same circle approximated twice. This is
// what lets the cap sit flush against the tube's own real end ring with no
// visible seam, matching this app's own established "read the real edge,
// never re-derive it independently" discipline (extractIsocurveV's use in
// Pipe's existing flat-cap code is the sibling precedent).
//
// CALLER CONTRACT: `frame` is one of `sweep1Rigid`/`sweep1RigidResampled`'s
// own exposed `frames[0]` (rail start) or `frames[frames.length-1]` (rail
// end) — both functions guarantee this ordering regardless of which
// internal path built the tube (see denseRailFrames' own header comment).
// `radius` should be the REAL effective rim radius at that exact frame —
// callers read it directly off the already-built tube's own ctrlNet (e.g.
// `ctrlNet[0][vIndex]`'s distance from `frame.origin`), never assumed
// constant, so a variable-radius Pipe's two different end radii each get
// their own correctly-sized dome for free, with no separate bookkeeping
// here. `axisDir` is the frame's OWN zAxis for the tube's genuine outgoing
// end (the dome continues past where the tube stops), or its negation for
// the tube's start (the dome extends before the tube begins) — the sign is
// entirely the caller's choice, this function has no opinion on which end
// it's building.
export function pipeRoundCapSurface(frame, radius, axisDir) {
  const profile = makeArc(frame.origin, frame.xAxis, axisDir, radius, 0, Math.PI / 2);
  return revolve(profile, frame.origin, axisDir, 0, 2 * Math.PI);
}

// BLENDED INNER/OUTER RIM CAP for a THICK pipe — a rounded end for a thick
// pipe, shaped as a blend between the inner and outer walls: essentially
// half a torus between them, or editable fillets at the inner and
// outer edges. Picked up now that a plain Thick pipe with a flat/round
// annular cap already exists (the app's own `convertPipeChildToThick`
// twin) — staged deliberately, because there was
// nothing to revolve a rim fillet onto until those two concentric walls
// and their real rim edges existed as actual objects.
//
// A REAL DESIGN CHOICE, not the only one, made explicit here: the two
// walls' real end rings (B at radius=outerRadius, C at radius=innerRadius,
// both at axial=0 — the tube's own TRUE rim, exactly where the flat/round
// caps already read their edge from) are NEVER moved or shortened. This
// cap bulges OUTWARD from those two exact points instead of receding back
// into the wall — a genuine "rolled rim" bead, tangent-matched to both
// real walls with ZERO overlap against the tube panels already rendered
// for the outer/inner tube surfaces (the same non-overlapping relationship
// `pipeRoundCapSurface`'s own hemisphere already has with its tube). The
// alternative (fillet the corner INWARD, receding into the existing wall
// material, matching a plain edge-break/chamfer) is also geometrically
// valid but would require re-tessellating the outer/inner tube panels'
// own V-domain to stop short of the true rim — real, separate, harder
// engineering (arc-length-along-the-rail bookkeeping) not attempted here.
//
// THE ARC CONSTRUCTION, still exactly `filletCornerArc`'s own math (doc
// 47's own literal framing), just aimed at a VIRTUAL vertex placed
// `radius` OUTWARD along the axis from the true rim, rather than at the
// rim itself — the trick that makes the fillet's own trimmed start point
// land EXACTLY back on the true rim with zero algebra to verify by eye
// (confirmed directly, not assumed): for the outer rim, vertex
// `Vout = B + rOut*axisDir`, arriving direction `dIn = normalize(Vout-B) =
// axisDir` (the wall's own true tangent), leaving direction `dOut =
// -frame.xAxis` (pure radially inward, the flat direction the middle
// section runs in) — a genuine 90-degree turn (`frame.xAxis` and
// `axisDir` are orthonormal by construction), so `halfPhi=45deg` and
// `trim = rOut*tan(45deg) = rOut` exactly — which makes
// `filletCornerArc`'s own `p0 = vertex - dIn*trim = (B+rOut*axisDir) -
// axisDir*rOut = B` land EXACTLY on the true rim, no epsilon-fudging
// needed. The inner rim's own corner mirrors this exactly (vertex
// `Vin = C + rIn*axisDir`, arriving from the middle section's own
// direction `+frame.xAxis`, leaving toward the true rim along
// `-axisDir`), landing its own `p2` exactly on C the identical way.
//
// AUTO-CLAMP, not refusal (the named degenerate case, matching
// every other fillet caller's established clamp-not-refuse precedent):
// each corner's OWN trimmed "middle-facing" point sits at radius
// `outerRadius-rOut` (outer) / `innerRadius+rIn` (inner) — if the two
// fillets together would push the outer one's radius below the inner
// one's (`rOut+rIn > outerRadius-innerRadius`, i.e. more than the pipe's
// real wall thickness), the middle section would fold back on itself.
// Both requested radii are scaled down PROPORTIONALLY (never
// independently, so their RATIO — the actual shape asked for — survives
// the clamp) until they land exactly at that shared boundary: "the two
// fillets meeting exactly in the middle (the
// half-torus case) is the boundary where clamping both to their shared
// maximum lands — not a special case, just what the clamp naturally
// produces at its own limit" (note: with UNEQUAL radii even at this exact
// boundary, the two corners' own middle-facing points share the same
// radius but sit at two DIFFERENT axial heights — rOut and rIn
// respectively — so "meeting exactly" only collapses to a single shared
// point when the two radii are also equal; this is an honest correction
// to the more optimistic "genuine torus-segment" framing for the
// general unequal-radii case, not silently glossed over). Applied fillet
// radii are returned so a caller can write the real, honest value back
// (matching `pipeRailForSweep`'s own `cornerRadius` auto-clamp
// write-back precedent) rather than silently doing something smaller
// than requested.
//
// A requested radius of exactly 0 on either rim degenerates that corner
// to a plain sharp, unfilleted rim (no arc at all — the middle section
// runs straight to the true rim point) — the "editable fillets at inner
// and outer edges... a rolled lip on the outer edge only, leaving the
// inner edge sharp" case, falling out of this same
// construction for free, not a special case. Both radii 0 degenerates to
// a single straight line from B to C — a plain flat annular disk.
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
    const auxOut = at(outerRadius - 1, rOut); // any distinct point further -xAxis at the same axial height — only its DIRECTION from Vout matters
    const cb = filletCornerArc(Vout, B, auxOut, rOut, planeNormal);
    segments.push({ type: 'arc', p0: cb.p0, apex: cb.apex, p2: cb.p2, weight: cb.weight });
    midStart = cb.p2;
  }
  let midEnd = C;
  let innerArc = null;
  if (rIn > 1e-9) {
    const Vin = at(innerRadius, rIn);
    const auxIn = at(innerRadius + 1, rIn); // any distinct point further +xAxis at the same axial height — only its DIRECTION into Vin matters
    innerArc = filletCornerArc(Vin, auxIn, C, rIn, planeNormal);
    midEnd = innerArc.p0;
  }
  if (length(sub(midEnd, midStart)) > 1e-9) segments.push({ type: 'line', a: midStart, b: midEnd });
  if (innerArc) segments.push({ type: 'arc', p0: innerArc.p0, apex: innerArc.apex, p2: innerArc.p2, weight: innerArc.weight });
  if (!segments.length) segments.push({ type: 'line', a: B, b: C }); // both radii ~0 — a plain flat annular disk, the honest degenerate case
  const profile = filletSegmentsToCurve(segments);
  const srf = revolve(profile, frame.origin, axisDir, 0, 2 * Math.PI);
  return { srf, appliedOuterFilletRadius: rOut, appliedInnerFilletRadius: rIn };
}

// SWEEP1, N PROFILES (N>=2) — real Rhino's own Sweep1: a rail plus SEVERAL
// cross-section curves, each already positioned near the rail station it
// belongs to, blended smoothly in between, the RESULT actually following
// the rail's own curved shape between those stations. The Sweep1 N-profiles
// scoping named 3 real open design questions before
// any code was written; this is where each was actually answered:
//
// Q1 (profile-to-rail correspondence): each profile's station is the
// CLOSEST POINT on the rail to the profile's own centroid (a plain
// unweighted average of its control points — a representative anchor for
// "roughly where this profile sits," not a claim about its true geometric
// centroid). This matches how a user actually works (rail first, then
// cross-sections drawn near their own intended station) and needs no new
// interaction concept beyond `closestPointOnCurve` (kernel/curve.mjs).
// A rail that loops back near itself can make this genuinely AMBIGUOUS
// (two rail parameters equally close to one profile); `closestPointOnCurve`'s
// own ambiguity flag is checked and turned into an honest, named error
// here — never a silent, possibly-wrong pick.
//
// Q2 (a frame at an arbitrary rail parameter): resolved entirely inside
// `buildParallelTransportFrames`'s own `extraParams` argument above — this
// function needs ZERO separate frame-transport math of its own, only calls
// with the parameters it needs (profile stations AND, new, a
// dense set of IN-BETWEEN rail parameters — see Q3).
//
// Q3 (how the cross-section actually varies between profiles).
//
// ⚠ TWO TRAPS, stated first, because between them they are what makes this
// hard. (1) Running loft()'s global-interpolation solve DIRECTLY against
// each profile's raw WORLD-SPACE points, using the rail only to weight the
// V-parameter each profile's station sits at, is provably NOT a sweep:
// with only the 2 endpoint profiles carrying any data, the resulting
// V-direction curve is degree-1 — a straight line between the two WORLD
// points. On a 90-degree, radius-100 arc rail that puts the v=0.5
// cross-section at the STRAIGHT CHORD's midpoint (51, 53) instead of the
// rail's own true curved midpoint (70.71, 70.71).
// (2) The obvious repair — loft in the LOCAL frame-relative space at each
// profile's own position, then transport EVERY row of the resulting
// interpolated control net through a continuously-varying frame — is ALSO
// wrong for N>=3: pairing SOLVED interpolation control points (row index
// k) with a per-row frame at profile k's own station parameter assumes the
// control point sits at that parameter, and it does not. Only the SOLVED
// CURVE, evaluated at a data parameter, is guaranteed to reproduce that
// data point — true for N=2, a degree-1 clamped spline, where the 2
// control points coincide with the 2 data points, but false for a
// higher-degree curve's interior control points.
//
// The construction below avoids both — it never treats a SOLVED CONTROL
// POINT as if it sat at a station parameter. Instead:
//  1. Each profile is localized into ITS OWN station's frame FIRST
//     (`localizeSectionToFrame`, unchanged) — a rigid, exact decomposition,
//     not an approximation.
//  2. For each U-sample row, a SMALL local NURBS curve is built (the SAME
//     global-interpolation technique loft() uses, just on
//     LOCAL offsets instead of world points) through the n stations' own
//     local data at their true vbar parameters. This curve's CONTROL
//     POINTS are never used directly — only its CONTINUOUS VALUE,
//     evaluated at chosen parameters, which the standard interpolation
//     guarantee DOES promise reproduces the original local data exactly at
//     each vbar station (curve VALUE at a data parameter, not curve
//     CONTROL POINT — exactly the distinction the second trap above
//     turns on).
//  3. That local curve is evaluated at a DENSE set of V parameters — every
//     profile's own station (exactly, so exactness is preserved) PLUS
//     several genuinely in-between samples per gap — and EACH dense
//     sample is transformed to world space by the rail frame evaluated AT
//     THAT SAME parameter (built via `buildParallelTransportFrames`'s
//     `extraParams`, arc-length-inverted via the new `buildArcLengthTable`/
//     `paramAtArcLength`, kernel/curve.mjs). Station samples reuse the
//     EXACT SAME frame object used for localization (one merged
//     `buildParallelTransportFrames` call, not two), so the already-proven
//     "localized profile round-trips exactly through its own frame"
//     identity carries straight through: at v=vbar_j, this dense world
//     sample equals frame_j(localCurve(vbar_j)) = frame_j(local station
//     data) = the ORIGINAL profile's own world point, exactly.
//  4. A SECOND, final global interpolation (again loft()'s technique, now
//     on this dense WORLD-space grid) builds the actual returned control
//     net. Because the DATA already includes each station's exact
//     world point, the standard interpolation guarantee proves the FINAL
//     surface reproduces every profile exactly at its own station — the
//     same guarantee the current tests below already check — while the
//     path BETWEEN stations now comes from the rail's own frame sampled
//     at real intermediate rail parameters, not a straight blend of the
//     two neighboring world points. This is why it resolves BOTH bugs at
//     once: exactness at stations (the n>=3 case that sinks the naive
//     per-row-frame approach) survives because it is proven via
//     CURVE VALUES at data parameters (always true), while curvature
//     between stations survives because the frame is now sampled
//     CONTINUOUSLY along the rail rather than only at the two profiles.
//
// Honest remaining limitation: the LOCAL blend (step 2) interpolates each
// station's own (x,y,z) frame-relative offsets independently of the
// others' orientation "clocking" — if two neighboring profiles are similar
// shapes stationed with very different rotational alignment relative to
// their own (parallel-transported, so twist-minimizing by construction)
// frames, the in-between cross-section can look like an unintuitive blend
// of the two orientations rather than a smooth rotation, the same
// "seam/clocking" family of issues real Sweep tools flag with an explicit
// seam-alignment UI. Not reachable in this app's typical use (simple/
// circular profiles, parallel-transport frames that already minimize
// twist) and not attempted here — same honesty standard as loft()'s own
// stated U-direction correspondence limitation below.
//
// Inherited, NOT fixed (stated honestly, matching loft()'s own
// header comment): U-direction (cross-section point) correspondence is
// still Loft's own "relative parameter fraction" convention — profiles
// with very different point distributions can still loft oddly across U,
// same known limitation, not attempted here.
const SWEEP_N_STATION_REL_TOL = 1e-6; // minimum relative rail-parameter gap between two profile stations
const SWEEP_N_INTERIOR_SAMPLES_PER_SPAN = 8; // extra rail-frame samples inserted BETWEEN each pair of consecutive profile stations, so the surface tracks the rail's real curvature there instead of a straight blend
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

  // TRUE rail arc-length fraction — a Sweep has a real rail to measure true
  // position along (unlike a plain Loft, which only has chord-length-of-
  // grid-points to go by — see kernel/loft.mjs's own header comment). ONE
  // shared dense arc-length table drives BOTH directions (a station's own
  // arc-length fraction via `arcLengthAtParam`, and an interior sample's
  // rail parameter via `paramAtArcLength` below) — mixing a fine table for
  // one side with a separately, coarsely re-derived length for the other
  // was a real bug: the last station's vbar came out slightly
  // short of 1.0, silently pushing the surface's own v=1 edge into
  // extrapolation beyond its last real data point.
  const railTable = buildArcLengthTable(rail, uMin, uMax);
  const railLen = railTable.total;
  const vbar = stationed.map((s) => arcLengthAtParam(railTable, s.u) / railLen);
  for (let i = 1; i < vbar.length; i++) {
    if (vbar[i] - vbar[i - 1] < 1e-9) {
      throw new Error('sweepNProfiles: two profile stations map to the same rail arc-length position despite different rail parameters — the rail has a zero-length span between them');
    }
  }

  // Dense V-sample sequence: every station's OWN vbar (exact, drives the
  // exactness proof) plus `interiorSamplesPerSpan` evenly-arc-length-spaced
  // samples strictly BETWEEN each pair of consecutive stations (this is
  // what actually makes the surface follow the rail's curvature there —
  // see the header comment above). Each interior sample's rail parameter
  // is the arc-length inversion of its own global fraction (approximate,
  // to `railTable`'s tolerance — fine, it's only used for an in-between
  // sample, not a station).
  const vDense = [];
  const uForDense = []; // rail parameter for each vDense entry, station entries EXACT (reused, not re-derived)
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

  // ONE merged frame call for BOTH the station frames (load-bearing for
  // localization AND for reproducing each profile exactly) and the dense
  // in-between frames (load-bearing for curvature) — the same "extend the
  // one incremental parallel-transport chain" discipline
  // buildParallelTransportFrames' own extraParams already documents.
  const framesAll = buildParallelTransportFrames(rail, uForDense);
  const denseFrames = framesAll.extra; // one per vDense entry, same order
  const stationFrames = stationDenseIndex.map((idx) => denseFrames[idx]);
  const localizedProfiles = stationed.map((s, i) => localizeSectionToFrame(s.crv, stationFrames[i]));

  // U grid, in LOCAL (frame-relative) space — mathematically identical
  // chord-length parametrization to using world points would give (a rigid
  // per-profile rotation+translation preserves within-profile distances),
  // but now the actual VALUES being interpolated are comparable "offset
  // from a rail frame" triples, not raw world positions.
  const localGrid = [];
  for (let i = 0; i < uSampleCount; i++) {
    const t = i / (uSampleCount - 1);
    localGrid.push(localizedProfiles.map((crv) => {
      const u0 = crv.knots[0], u1 = crv.knots[crv.knots.length - 1];
      return curvePoint(crv, u0 + t * (u1 - u0));
    }));
  }

  const dV = Math.min(degVmax, n - 1); // degree of the SCAFFOLD local curve (step 2 above) — its control points are never read directly
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

  // Final V degree/knots, over the DENSE sample sequence (not just the n
  // stations) — this is what lets the final surface carry real curvature
  // information between stations rather than only n data points to blend.
  const denseCount = vDense.length;
  const dV2 = Math.min(degVmax, denseCount - 1);
  const knotsVDense = averagingKnotVector(vDense, dV2);

  const ctrlNet = R_local.map((row) => {
    // Step 2: the LOCAL scaffold curve for this U-row — an actual SOLVE
    // (interpAtParams, the same global-interpolation core loft() uses)
    // through the n stations' own local data at their true vbar, NOT a
    // reuse of the raw data as control points (`row` is DATA, not control
    // points — control points never get read directly below, only this
    // curve's continuous VALUE, which the interpolation guarantee DOES
    // promise reproduces `row` exactly at each vbar station).
    const localCtrl = interpAtParams(row, dV, vbar, knotsV);
    const localRowCrv = { degree: dV, knots: knotsV, ctrlPts: localCtrl };
    // Steps 3+4: evaluate that local curve at every dense V sample,
    // transform through the frame at THAT SAME parameter, then re-
    // interpolate (exactly, through every dense sample including every
    // station) to get this row's FINAL world-space V control points.
    const worldRow = vDense.map((v, m) => [...frameToWorld(denseFrames[m], curvePoint(localRowCrv, v)), 1]);
    return interpAtParams(worldRow, dV2, vDense, knotsVDense);
  });

  return {
    degU: dU, knotsU, degV: dV2, knotsV: knotsVDense, ctrlNet,
    stations: stationed.map((s, i) => ({ profileIndex: s.idx, u: s.u, v: vbar[i] })),
    frames: stationFrames,
    localizedProfiles,
    ubar, // exposed for verification: surfacePoint(srf, ubar[i], v) reproduces each profile's own resample exactly at v=that profile's own station
  };
}

// SWEEP2 — real Rhino's own two-rail sweep: the
// "Tier D" combination (2+ rails, exactly 1 profile),
// explicitly deferred and out of scope for
// the Gordon/network-surface work (a network surface needs a
// real 2+ profile FAMILY; a lone profile is one constraint, not a family —
// structurally a DIFFERENT, well-known construction, not a degenerate case
// of Gordon). Built for real here, per the live use case (2 rails +
// 1 profile): a single cross-section profile whose own two ENDPOINTS ride
// along two independent rail curves, its own width/orientation adjusting
// station-by-station to match the rails' varying separation and relative
// position along their length.
//
// STATIONING: reuses this kernel's own established "reparametrize to a
// SHARED, dense set of stations, never combine raw control nets" discipline
// (loft()/sweepNProfiles/gordonNetworkSurface's own precedent). Each rail is
// independently arc-length-parametrized (buildArcLengthTable) and sampled
// at the SAME shared set of relative arc-length FRACTIONS v in [0,1] —
// R1(v) via paramAtArcLength(table1, v*len1), R2(v) the same way on rail
// 2 — so the two rails never need to share a knot vector, degree, or even
// curve type. The station set is the union of BOTH rails' own knot-span
// boundaries (each converted to ITS OWN arc-length fraction) plus
// `stationSamplesPerSpan` interior samples per gap (denseWithInterior,
// kernel/loft.mjs's own already-proven densifier, reused not reinvented,
// exactly the "reuse interiorSamplesPerSpan-style dense sampling" this
// construction was asked to follow) — so a sharp feature in EITHER rail
// gets a real station, not just a guess.
//
// LOCAL FRAME, PER STATION: origin = R1(v) (an arbitrary but consistent
// choice between R1/R2/a derived midpoint; R1 is picked to match Sweep1's
// own existing convention of anchoring a profile's local frame at the RAIL
// side, not a derived point — which rail is "origin" has no effect on the
// result's exactness, since the profile's OTHER endpoint reaches R2 exactly
// through the width axis below regardless of this choice). xAxis (the width
// axis) = normalize(R2(v) - R1(v)). The y/z axes are PARALLEL-TRANSPORTED
// along v — buildParallelTransportFrames' own algorithm, reused
// structurally: seed with anyPerpendicular at the first station, then carry
// the previous normal forward by projecting it into the new xAxis's own
// perpendicular plane and re-normalizing — driven by the WIDTH AXIS
// direction changing along v instead of a single rail's tangent, the same
// anti-flip reasoning buildParallelTransportFrames already solves for one
// rail's tangent, applied here to the (equally capable of rotating
// unpredictably between stations) R2-R1 direction.
//
// PROFILE ANCHOR POINTS: the profile's own two ENDPOINTS (its first/last
// control point — exact on a clamped NURBS curve, the same property
// localizeSectionToFrame's own existing callers already rely on) are the
// two points made to land on R1(v)/R2(v) at every station — the natural,
// honest v1 choice, matching how a user actually draws a profile spanning
// from near one rail to near the other. The profile is localized ONCE into
// its OWN static frame (localizeSectionToFrame, unchanged, reused verbatim)
// — origin = its own start point, xAxis = normalize(end-start) — so its
// localized start is exactly (0,0,0) and its localized end is exactly
// (width0,0,0) by construction (end-start is, by definition, purely along
// its own xAxis).
//
// THE SCALE DECISION (a real, considered v1 choice, stated plainly, not
// silently assumed — matching this kernel's own "correspondence by relative
// parameter fraction" precedent of naming its own honest simplification):
// at each station, the profile's local (a,b,c) offsets are scaled by ONE
// factor, width(v)/width0, applied to ALL THREE local axes — a UNIFORM
// scale within the local frame, not "width stretches to fit, depth (the
// profile's own perpendicular extent) stays a fixed physical size." Chosen
// over the fixed-depth alternative for a concrete reason: if the two rails
// converge toward each other (width(v) -> small), a fixed-depth profile
// would still stick a full-size cross-section out of an almost-single
// point — visibly wrong, and actively worse the closer the rails get. A
// uniformly-scaled profile shrinks smoothly toward a point instead, the
// same intuitive behavior a cone's cross-section has shrinking toward its
// apex — not the only choice a future round could make (real Rhino itself
// exposes a scale-lock/"maintain height" option for the other behavior),
// but a genuine, justified v1 default rather than an arbitrary pick.
//
// EXACTNESS: because the localized profile's start/end are EXACTLY
// (0,0,0)/(width0,0,0), and the per-station transform maps local (0,0,0) to
// world origin=R1(v) and local (width0,0,0) to world origin +
// width(v)*xAxis = R1(v) + (R2(v)-R1(v)) = R2(v) EXACTLY (xAxis is, by
// construction, normalize(R2(v)-R1(v)), so width(v)*xAxis = R2(v)-R1(v)
// exactly), this holds at EVERY one of the dense stations, unconditionally
// — the same "the final surface reproduces every dense sample exactly"
// guarantee loft()/sweepNProfiles/gordonNetworkSurface already rely on,
// applied here to the two rail-anchor edges (u=0 and u=1) of the returned
// surface.
//
// DEGENERATE CASES, refused honestly rather than producing garbage
// geometry: the profile's own start/end coincide (no defined width axis to
// anchor with at all — a closed/periodic profile has no natural anchor
// pair); the two rails cross or coincide at some shared station (the width
// axis there is undefined) — both are real, distinct, named refusals, not
// a silent zero-width sliver or a NaN surface.
//
// SCOPE, exactly as asked: exactly ONE profile curve, matching this
// construction's own real use case exactly. A genuine N-profile Sweep2
// (several cross-sections, each still anchored to both rails, blended
// between) is real, separate, further scope — NOT attempted here.
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

  // Shared dense station fractions — union of both rails' own knot-span
  // boundaries (each converted to ITS OWN arc-length fraction), plus
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

  // The profile's own static local frame + localization (localizeSectionToFrame,
  // unchanged, reused verbatim) — its own start/end control points become
  // exactly (0,0,0)/(width0,0,0) in local coordinates by construction.
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

  // Dense U (profile-direction) samples of the LOCALIZED profile, matching
  // loft()'s own "relative parameter fraction" resampling convention.
  const lp0 = localizedProfile.knots[0], lp1 = localizedProfile.knots[localizedProfile.knots.length - 1];
  const localSamples = [];
  for (let i = 0; i < uSampleCount; i++) {
    const t = i / (uSampleCount - 1);
    localSamples.push(curvePoint(localizedProfile, lp0 + t * (lp1 - lp0)));
  }

  // Per-(u,v) world sample grid: ONE uniform scale factor (width(v)/width0
  // — the scale decision above) applied to all three local axes, then
  // transformed through the station's own frame. At lp=(0,0,0) this is
  // exactly R1(v); at lp=(width0,0,0) this is exactly R2(v) (see the
  // EXACTNESS comment above) — for EVERY station v.
  const grid = localSamples.map((lpt) => stationData.map((s) => {
    const k = s.width / width0;
    return [
      s.p1[0] + k * (lpt[0] * s.xAxis[0] + lpt[1] * s.yAxis[0] + lpt[2] * s.zAxis[0]),
      s.p1[1] + k * (lpt[0] * s.xAxis[1] + lpt[1] * s.yAxis[1] + lpt[2] * s.zAxis[1]),
      s.p1[2] + k * (lpt[0] * s.xAxis[2] + lpt[1] * s.yAxis[2] + lpt[2] * s.zAxis[2]),
    ];
  }));

  // ONE final global surface interpolation (loft()'s own two-pass
  // interpAtParams technique) through the dense grid. U: chord-length-
  // averaged across all v-rows (loft()'s own convention — the profile's own
  // internal point distribution needs a shared, well-distributed knot
  // vector). V: the TRUE shared arc-length-fraction stations directly
  // (sweepNProfiles'/gordonNetworkSurface's own convention — a real station
  // parameter already exists here, no need to re-derive one from chord
  // length of the sample grid).
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
    // should reproduce rail1/rail2's own real curvePoint at that station;
    // `widths` proves the profile's own scale genuinely varies station-to-
    // station rather than being frozen; `ubar` (the U-direction chord-
    // length parameter used for row i) lets a caller identify exactly which
    // resampled row corresponds to a specific profile-domain sample,
    // mirroring sweepNProfiles' own `ubar` exposure for the same purpose.
    vStations, width0,
    widths: stationData.map((s) => s.width),
    ubar,
  };
}
