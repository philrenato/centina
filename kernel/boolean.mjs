// THE THREE OPERATORS — Phase 8 of the boolean pipeline,
// second half, and the point every phase before it was building toward.
// Union, Difference and Intersect are ONE machine: split every face where the
// two solids meet, decide which fragments survive, sew the survivors back
// into a closed shell. Only the keep-rule differs, and that difference is one
// function call (kernel/booleansew.mjs's `keepFragments`).
//
// WHAT THIS OWNS, AND WHAT IT DELIBERATELY DOES NOT. Every hard step already
// exists and is tested on its own: `intersectSurfacesComplete` marches the
// curves, `projectPointsToSurfaceUV` puts them in a face's own parameters,
// `splitFaceByCurves` cuts the face up, `classifyFragment` decides inside from
// outside, `sewFragments` welds the result. This module is the assembly, and
// assembling is where a boolean usually goes wrong quietly — so every step
// that cannot be completed refuses by name, naming the FACE it failed on,
// rather than dropping a fragment and returning a plausible solid.
//
// CURVES ARRIVE TAGGED WITH THEIR OWN FACE PAIR, and that is a real decision
// rather than an interface convenience. An intersection curve lies on exactly
// the two faces that produced it; blind-projecting every curve onto every
// face would ask `projectPointsToSurfaceUV` to answer "is this curve on this
// surface" as a side effect of a tolerance check, which it is not built to
// decide and would get wrong for two faces that happen to be near-coincident.
// The caller already knows the pairing — a face-pair sweep is how the curves
// were computed — so it passes it through instead of having it re-derived.
//
// A FACE WITH NO INTERSECTION CURVE IS NOT SPLIT AT ALL. It lies wholly
// inside or wholly outside the other solid, so its own trim loop IS its
// single fragment and one classification decides it. This is not an
// optimisation: running an empty split just to get one fragment back would
// route the commonest face in any real boolean through the machinery most
// likely to refuse.
//
// ORIENTATION IS THE SEW'S JOB, said plainly because it looks like a gap.
// Nothing here reverses B's faces for a Difference. `buildBrepSolid`'s own
// `orientLoops` pass re-winds every face into one consistent orientation as
// part of welding, so a per-operand flip here would be undone or fought. What
// that pass guarantees is CONSISTENCY, not that the result faces outward —
// a global inside-out shell is a real possible outcome and is checked by
// signed volume, not assumed away.

import { trivialTrimLoop, projectPointsToSurfaceUV, seamCrossingSpine, seamStraddleChains, seamDoubleStraddleChains, seamOpenChains, seamCrossingUVPoints } from './trim.mjs';
import { splitFaceByCurves } from './facesplit.mjs';
import { keepFragments, sewFragments } from './booleansew.mjs';
import { keepRuleFor } from './classify.mjs';
import { closestPointOnSurface, surfacePoint, surfaceClosure } from './surface.mjs';

/**
 * Union, Difference or Intersect, from two solids and the curves they share.
 *
 * `solidA` / `solidB` — { faces: [{ srf, trimLoop?, trimHoles? }], triangles }
 *   `triangles` is that solid's own tessellation, used only for classifying
 *   the OTHER solid's fragments. An untrimmed face may omit `trimLoop`; its
 *   full parametric rectangle is used.
 *
 * `curves` — [{ samples: [[x,y,z], ...], faceA, faceB }, ...] where faceA and
 *   faceB index into each solid's own `faces`. This is what a face-pair SSI
 *   sweep already produces.
 *
 * `operation` — 'union' | 'difference' | 'intersect'. Difference is A minus B.
 *
 * Returns { ok, solid, fragments, stats, verdict, keptCount, faceReports } on
 * success, or { ok: false, reason, faceReports } naming the first face that
 * could not be resolved.
 *
 * `fragments` is the kept set itself — [{ srf, outer, holes }, ...] — and is
 * what a caller BUILDS from. `solid` is the welded Brep, which proves the
 * result closed and carries its own topology numbers, but a Brep face here is
 * a boundary polyline, not a surface: handing back only that would leave a
 * caller able to verify a boolean and unable to draw one. The two are
 * different readings of the same kept set, so both are returned.
 */
export function booleanSolids(solidA, solidB, curves, operation, opts = {}) {
  if (!keepRuleFor(operation === 'difference' ? 'intersect' : operation)) {
    return { ok: false, reason: `unknown boolean operation "${operation}"` };
  }
  if (!solidA?.faces?.length || !solidB?.faces?.length) {
    return { ok: false, reason: 'both operands need at least one face' };
  }
  if (!solidA.triangles?.length || !solidB.triangles?.length) {
    return { ok: false, reason: 'both operands need a tessellation to classify against' };
  }

  const faceReports = [];
  const kept = [];
  const shared = shareTriplePoints(curves || [], solidA, solidB, opts);
  const seamShared = shareSeamCrossings(shared.curves, solidA, solidB, opts);
  const workCurves = seamShared.curves;

  for (const side of ['a', 'b']) {
    const mine = side === 'a' ? solidA : solidB;
    const other = side === 'a' ? solidB : solidA;
    const key = side === 'a' ? 'faceA' : 'faceB';

    for (let fi = 0; fi < mine.faces.length; fi++) {
      const face = mine.faces[fi];
      const label = `${side.toUpperCase()} face ${fi}`;
      const mineCurves = workCurves.filter((c) => c[key] === fi);

      const res = resolveFace(face, mineCurves, other.triangles, operation, side, label, opts);
      faceReports.push(res.report);
      if (!res.ok) return { ok: false, reason: res.reason, faceReports };
      kept.push(...res.kept);
    }
  }

  if (!kept.length) {
    return {
      ok: false,
      reason: `every fragment was discarded — the operands may not overlap in the way ${operation} needs`,
      faceReports,
    };
  }

  const sewn = sewFragments(kept, opts);
  return sewn.ok
    ? { ok: true, solid: sewn.solid, fragments: kept, stats: sewn.stats, verdict: sewn.verdict, worstSharedGap: sewn.worstSharedGap, keptCount: kept.length, triplePoints: shared.inserted, seamPoints: seamShared.inserted, faceReports }
    : { ok: false, reason: `the kept fragments do not sew into a closed solid — ${sewn.verdict}`, verdict: sewn.verdict, stats: sewn.stats ?? null, worstSharedGap: sewn.worstSharedGap, nakedEdgePoints: sewn.nakedEdgePoints ?? null, fragments: kept, keptCount: kept.length, triplePoints: shared.inserted, seamPoints: seamShared.inserted, faceReports };
}

// A SEAM CROSSING IS A CORNER OF THE ANSWER, AND EACH FACE WAS INVENTING ITS
// OWN. THIS COMPUTES THEM ALL FROM THE SHARED CURVE AND GIVES EVERY FACE THE
// SAME ONES.
//
// A cut curve running across a closed surface's seam has to be broken there:
// the domain rectangle's two ends are the same physical place, so the chain is
// re-expressed as pieces that reach the edge (seamCrossingSpine,
// seamStraddleChains, seamOpenChains, all via seamPointAt). The crossing's
// position is interpolated between the two samples straddling the domain edge,
// IN THAT SURFACE'S OWN PARAMETERS.
//
// Two faces cut by the same curve therefore compute the corner twice. While
// their seams are in different places that is harmless — the two corners are
// different points and each is a corner of one face only. When the seams pass
// through the SAME place it is not: both faces put a vertex at what is
// geometrically one point, from two different interpolations, and the answers
// differ. Measured on two revolved balls whose offset points straight along
// both their seam meridians, one surface each and one intersection circle: the
// two corners land 1.3e-2 apart against a 1e-4 weld, and the shell comes back
// with a four-edge sliver quad — the two faces' corners joined by the two
// curve samples either side, enclosing nothing. Refining the march does not
// close it. The separation falls linearly with the sample spacing while the
// naked count stays at exactly 4, because the disagreement is not an accuracy
// problem: no sampling makes two independent interpolations agree to a
// tolerance.
//
// So every face's crossing is computed here, BEFORE anything is projected or
// split, and spliced into the curve every face reads. Each face then finds its
// own corner sitting exactly on a sample it already carries and paves the
// other face's corner as an ordinary interior point, so the two boundaries run
// through the same vertices in the same order and weld by construction. It is
// the argument shareTriplePoints makes, for the other kind of corner.
//
// IDEMPOTENT BY CONSTRUCTION, which is what makes the splice safe to do for
// every face at once rather than in some order. A point placed exactly on a
// surface's own domain edge re-projects to that edge, so the jump either side
// of it interpolates with a fraction of 0 or 1 and returns the inserted point
// unchanged — a face's second look at its own corner cannot move it.
//
// THE CORNER IS SOLVED, NOT INTERPOLATED, and that is the difference between
// this working and merely moving the error. seamPointAt's own answer is a
// LINEAR interpolation across the jump, so it is exact on the seam and off the
// true intersection curve by roughly one sample spacing — measured at 9.7e-3
// on the ball pair, which `projectPointsToSurfaceUV` rightly refuses as "not
// on the surface" the moment the paired face reads it. Sharing that point
// would hand the other face a corner it cannot accept.
//
// The exact corner is available and cheap. A seam is an ISOCURVE of its own
// surface — the domain edge a = aEdge, traced by the other parameter — so the
// crossing is where that isocurve meets the paired surface: one unknown, one
// equation, bracketed by the two samples that straddle the jump. Solving it
// gives a point exactly on the seam AND on both surfaces, which is what both
// faces need and neither could interpolate.
//
// Two faces whose seams pass through one place then solve the SAME equation
// and land on the same point to machine precision, so they weld with nothing
// left over. Where the seams are genuinely apart the two solves return two
// genuinely different corners, each spliced in and paved by both faces — which
// is right, and is why this shares every crossing rather than trying to detect
// coincident seams.
function shareSeamCrossings(curves, solidA, solidB, opts = {}) {
  const out = curves.map((c) => ({ ...c, samples: c.samples.slice() }));
  const weld = opts.tolerance ?? 1e-4;
  let inserted = 0;

  for (const c of out) {
    if (c.samples.length < 3) continue;
    const srfA = solidA.faces[c.faceA]?.srf;
    const srfB = solidB.faces[c.faceB]?.srf;
    if (!srfA || !srfB) continue;
    // A surface with no closed direction has no seam, so nothing here can
    // apply to it. Checked before the projection rather than after, because
    // projecting every sample of every curve onto every face is the expensive
    // part and a box, a plane and a prism wall are all open in both
    // directions — the commonest faces in any real boolean.
    const closureA = surfaceClosure(srfA), closureB = surfaceClosure(srfB);
    if (!closureA.closedU && !closureA.closedV && !closureB.closedU && !closureB.closedV) continue;
    const cyclic = closedIn3D(c.samples);
    // How far this curve's own samples read from the two surfaces they lie on
    // — the projector's noise floor for this pair, and what a solved corner
    // has to match to count as being on the curve at all.
    let floor = null;
    const adds = [];
    for (const [srf, other, closure] of [[srfA, srfB, closureA], [srfB, srfA, closureB]]) {
      if (!closure.closedU && !closure.closedV) continue;
      const proj = projectPointsToSurfaceUV(c.samples, srf, opts);
      // A curve this face cannot be projected onto has no seam crossing to
      // contribute; resolveFace refuses it later by name, and refusing it
      // twice here would only move the message.
      if (!proj.ok || proj.uv.length !== c.samples.length) continue;
      for (const x of seamCrossingUVPoints(proj.uv, srf, cyclic)) {
        // The crossing lies between samples `seg` and `seg + 1`. One on the
        // closing step of a cyclic chain has no segment to be spliced into —
        // on a marched closed curve that step is the degenerate one joining
        // the repeated endpoint, so there is nothing there to place.
        if (x.seg + 1 >= c.samples.length) continue;
        const oi = 1 - x.axisIndex;
        const p = solveSeamCrossing(srf, other, x, proj.uv[x.seg][oi], proj.uv[x.seg + 1][oi]);
        if (!p) continue;
        if (floor === null) floor = medianResidual(c.samples, srfA, srfB) * TRIPLE_POINT_RESIDUAL_SLACK;
        // ON THE CURVE, judged against the curve's own samples rather than a
        // chosen number. A solve that converged somewhere else — the isocurve
        // grazing the other surface, or never reaching it inside the bracket —
        // fails this by orders, and inserting it would put a corner on the
        // boundary where the geometry has none.
        //
        // THE MEDIAN RESIDUAL, NOT THE WORST, and the difference is not
        // cosmetic. `closestPointOnSurface` converges poorly near a POLE, where
        // a whole domain row collapses to one point and the parameters stop
        // separating — so one sample of a curve that passes near one can read
        // 9e-2 while the rest read 1e-13. A worst-case floor takes that
        // straight into the acceptance test and admits a corner 1.45 away from
        // the surface it is supposed to be on, which then fails projection as
        // "not on the surface" and takes the whole boolean down. The median is
        // the same measurement with the projector's own bad cases outvoted.
        if (closestPointOnSurface(other, p).distance > Math.max(floor, weld * 1e-2)) continue;
        const a = c.samples[x.seg], b = c.samples[x.seg + 1];
        const span = dist3(a, b);
        if (!(span > 0)) continue;
        adds.push({ seg: x.seg + 1, t: dist3(a, p) / span, p });
      }
    }
    if (!adds.length) continue;
    const before = c.samples.length;
    c.samples = spliceIntoPolyline(c.samples, adds, weld);
    inserted += c.samples.length - before;
  }
  return { curves: out, inserted };
}

// The typical distance this curve's own samples read from the two surfaces
// they lie on. Median rather than maximum, so a single sample the projector
// handled badly cannot set the standard every other measurement is judged by.
function medianResidual(samples, srfA, srfB) {
  const step = Math.max(1, Math.floor(samples.length / TRIPLE_POINT_CALIBRATION_SAMPLES));
  const ds = [];
  for (let i = 0; i < samples.length; i += step) {
    ds.push(closestPointOnSurface(srfA, samples[i]).distance);
    ds.push(closestPointOnSurface(srfB, samples[i]).distance);
  }
  if (!ds.length) return 0;
  ds.sort((a, b) => a - b);
  return ds[ds.length >> 1];
}

// Enough thirds to drive the bracket to the double's own resolution: the
// interval shrinks by 2/3 each round, so 80 takes any real parameter span
// below 1e-14 and the limit becomes the evaluation rather than the search.
const SEAM_SOLVE_ITERATIONS = 80;
// The point where `srf`'s seam meets `other`, as a ternary search on the
// distance along the seam isocurve.
//
// TERNARY RATHER THAN NEWTON, and rather than a signed bisection. Distance to
// a surface is non-negative with a minimum of zero at the crossing, so it has
// no sign to bisect on and no derivative that stays well conditioned as it
// approaches the root — while a unimodal minimum is exactly what a ternary
// search is for. It cannot diverge, needs no initial slope, and the bracket is
// already in hand from the two samples that straddle the jump.
//
// THE BRACKET IS GROWN BEFORE IT IS SEARCHED, and skipping that step is what
// made the first version of this find nothing. The two samples straddling the
// jump bound the crossing only as well as the polyline does, and on a marched
// curve they can sit 2.7e-5 apart in parameter with the true root just outside
// — a search confined to them stalls against its own endpoint and returns a
// point still 6.3e-4 off the paired surface, which the acceptance test below
// then rightly refuses. So the interval doubles about its own midpoint until
// the cost at both ends exceeds the cost in the middle, which is what makes it
// a bracket rather than a guess, and only then is it searched.
//
// Growth stops at the domain, and a bracket that reaches both ends without
// ever enclosing a minimum is reported as no crossing: the seam does not meet
// the other surface anywhere along its length, so there is no corner to share.
const SEAM_BRACKET_GROWTH_STEPS = 40;
function solveSeamCrossing(srf, other, crossing, oFrom, oTo) {
  const ai = crossing.axisIndex, oi = 1 - ai;
  const oKnots = oi === 0 ? srf.knotsU : srf.knotsV;
  const oMin = oKnots[0], oMax = oKnots[oKnots.length - 1];
  const aEdge = crossing.uv[ai];

  const at = (o) => {
    const uv = [0, 0];
    uv[ai] = aEdge; uv[oi] = o;
    return surfacePoint(srf, uv[0], uv[1]);
  };
  const cost = (o) => {
    const p = at(o);
    return p.every((v) => Number.isFinite(v)) ? closestPointOnSurface(other, p).distance : Infinity;
  };

  const mid = (oFrom + oTo) / 2;
  // A degenerate pair gives no width to grow from, so the search starts at the
  // domain's own resolution instead of at zero.
  let half = Math.max(Math.abs(oTo - oFrom) / 2, (oMax - oMin) * 1e-9);
  let lo = mid, hi = mid, bracketed = false;
  const cMid = cost(mid);
  for (let i = 0; i < SEAM_BRACKET_GROWTH_STEPS; i++) {
    lo = Math.max(oMin, mid - half); hi = Math.min(oMax, mid + half);
    if (cost(lo) > cMid && cost(hi) > cMid) { bracketed = true; break; }
    if (lo <= oMin && hi >= oMax) break;
    half *= 2;
  }
  if (!bracketed || !(hi > lo)) return null;

  for (let i = 0; i < SEAM_SOLVE_ITERATIONS && hi - lo > 0; i++) {
    const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
    if (cost(m1) < cost(m2)) hi = m2; else lo = m1;
  }
  const p = at((lo + hi) / 2);
  return p.every((v) => Number.isFinite(v)) ? p : null;
}

// A TRIPLE POINT IS ALREADY COMPUTED EXACTLY. IT IS JUST MISSING FROM ONE OF
// THE CURVES THAT PASSES THROUGH IT, AND THAT IS WHY A MULTI-FACE OPERAND
// LEAVES SLIVERS.
//
// Where three faces meet, two intersection curves pass through one point. On a
// multi-face operand that is ordinary rather than exotic: a prism's side face
// and its cap share an edge, so the other solid's surface crosses all three at
// once. One of those curves generally ENDS there — the side face's march runs
// out of its own domain exactly on the cap plane — and that endpoint lies on
// all three surfaces to machine precision. Measured on a revolved blob against
// a 12-face star prism: the endpoint sits 2.8e-15 from the blob and 5.0e-15
// from the cap plane, against 6.5e-15 and 7.3e-15 for the cap curve's OWN
// samples. It is the exact answer, already in hand.
//
// The other curve — the blob against the cap plane — simply has no sample
// there; the marcher stepped past it, nearest sample 9.4e-2 away. So the cap
// FACE, which is split by that curve against its own star trim loop, finds the
// crossing on a CHORD instead, 1.9e-3 off the true point. The blob face, which
// is split by both curves, lands on the exact one. Two faces sharing an edge
// therefore disagree by 22x the weld tolerance, and the shell comes back with
// one sliver triangle per triple point — 20 of them, 60 naked edges.
//
// So this inserts the point that is already known into the curve that is
// missing it, BEFORE anything is projected or split. Every face then paves the
// same 3D point and the fragments weld by construction, which is the same
// argument densifyOnCutLines makes for a seam cut line, one stage earlier.
//
// WHY EXACTLY ONE SHARED FACE, and both halves of that are load-bearing.
//
// A curve's endpoint only needs to be a vertex of another curve if some FACE is
// split by both — that is exactly when their crossing has to be paved
// consistently. Curves with no face in common are never seen together by any
// arrangement, and forcing a vertex between them would invent a corner nothing
// asked for.
//
// Curves sharing BOTH faces are excluded for the opposite reason: they are
// separate COMPONENTS of one surface-surface intersection, so they are disjoint
// by construction and an endpoint of one is never a point of the other. The
// surface test below cannot see that — every sample of either component lies on
// both surfaces, so the test passes vacuously and would splice one component's
// end into whichever chord of the other it happened to pass nearest. Measured
// on an off-axis star prism, which has two components against two of its side
// faces: that admitted false points and turned an open shell into a refusal.
//
// THE ACCEPTANCE TEST IS A MEASUREMENT AGAINST THE CURVE'S OWN SAMPLES, not a
// chosen tolerance. A candidate is inserted only if it sits on BOTH of that
// curve's surfaces no worse than the curve's own samples do, read with the same
// projector. That comparison is not close: on the fixture above a genuine
// triple point measures 5e-15 against the curve's own 7e-15, while the same
// curve's other endpoint — a point on the blob but nowhere near the cap plane —
// measures 18. Fifteen orders separate the two answers, so the slack factor
// below cannot decide anything; it exists only so a projector that converges
// slightly worse on one sample than another does not throw away a real point.
const TRIPLE_POINT_RESIDUAL_SLACK = 16;
// Enough spread samples to read the projector's own floor on this curve without
// walking a dense march end to end. The floor is a property of the surfaces and
// the projector, not of any one sample, so a bounded sample of it is exact
// enough for a test with fifteen orders of headroom.
const TRIPLE_POINT_CALIBRATION_SAMPLES = 24;
export function shareTriplePoints(curves, solidA, solidB, opts = {}) {
  const out = curves.map((c) => ({ ...c, samples: c.samples.slice() }));
  if (out.length < 2) return { curves: out, inserted: 0 };
  const weld = opts.tolerance ?? 1e-4;

  // Only an OPEN curve has endpoints to share. A closed loop's first and last
  // samples are the same place, so neither is a corner of anything.
  const ends = [];
  for (let i = 0; i < out.length; i++) {
    const s = out[i].samples;
    if (s.length < 2 || closedIn3D(s)) continue;
    ends.push({ from: i, p: s[0] });
    ends.push({ from: i, p: s[s.length - 1] });
  }
  if (!ends.length) return { curves: out, inserted: 0 };

  let inserted = 0;
  for (let ci = 0; ci < out.length; ci++) {
    const c = out[ci];
    const srfA = solidA.faces[c.faceA]?.srf;
    const srfB = solidB.faces[c.faceB]?.srf;
    if (!srfA || !srfB) continue;

    const adds = [];
    let floor = null; // computed once, and only if this curve has a candidate
    for (const e of ends) {
      if (e.from === ci) continue;
      const o = out[e.from];
      const sharesA = o.faceA === c.faceA, sharesB = o.faceB === c.faceB;
      if (sharesA === sharesB) continue; // neither face in common, or both
      const hit = nearestOnPolyline(e.p, c.samples);
      if (hit.seg < 1) continue;
      // A point OF this curve, falling between two of its samples, sits at most
      // its own chord's sagitta off that chord — and the polyline measures that
      // for itself, so nothing here has to be chosen. See segmentDeviation.
      if (hit.off > segmentDeviation(c.samples, hit.seg)) continue;
      const a = c.samples[hit.seg - 1], b = c.samples[hit.seg];
      // A point landing within the weld tolerance of a sample this curve
      // already carries is that sample as far as the sew is concerned;
      // inserting it would only add an edge shorter than a vertex is wide.
      if (dist3(e.p, a) <= weld || dist3(e.p, b) <= weld) continue;
      if (floor === null) {
        floor = calibrationFloor(c.samples, srfA, srfB) * TRIPLE_POINT_RESIDUAL_SLACK;
      }
      if (closestPointOnSurface(srfA, e.p).distance > floor) continue;
      if (closestPointOnSurface(srfB, e.p).distance > floor) continue;
      adds.push({ seg: hit.seg, t: hit.t, p: e.p });
    }
    if (!adds.length) continue;
    const before = c.samples.length;
    c.samples = spliceIntoPolyline(c.samples, adds, weld);
    // What actually went in, not what was proposed: the splice drops a point
    // that would land within the weld tolerance of one already placed, and a
    // count that reported the proposals instead would be a number nothing
    // downstream could check against the curves it describes.
    inserted += c.samples.length - before;
  }
  return { curves: out, inserted };
}

// The same closed-in-3D test the seam unwrap uses: a closed chain's two ends
// meet within half of its own largest step, which no genuinely open chain can
// do by coincidence at any sampling density.
function closedIn3D(samples) {
  let span = 0;
  for (let i = 1; i < samples.length; i++) span = Math.max(span, dist3(samples[i], samples[i - 1]));
  return dist3(samples[0], samples[samples.length - 1]) <= span * 0.5 + 1e-9;
}

function dist3(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// How far this curve's OWN samples read from the two surfaces they lie on —
// the projector's noise floor for this pair, which is what a candidate has to
// match to count as being on the same curve.
function calibrationFloor(samples, srfA, srfB) {
  const step = Math.max(1, Math.floor(samples.length / TRIPLE_POINT_CALIBRATION_SAMPLES));
  let worst = 0;
  for (let i = 0; i < samples.length; i += step) {
    worst = Math.max(worst, closestPointOnSurface(srfA, samples[i]).distance);
    worst = Math.max(worst, closestPointOnSurface(srfB, samples[i]).distance);
  }
  return worst;
}

// HOW FAR OFF ITS OWN CHORD A POINT OF THIS CURVE CAN SIT, measured from the
// polyline rather than assumed.
//
// A sample straddled by its two neighbours is a point of the TRUE curve, and
// its distance from the chord joining those neighbours is that two-segment
// chord's sagitta, directly observed. A chord half as long has a quarter the
// sagitta, so one segment's own bound is about a quarter of what is measured
// here — returning the measured two-segment figure therefore keeps a fourfold
// margin over the bound it stands for, which is deliberate: this is a
// plausibility guard behind an exact surface test, and the two failure
// directions are not symmetric. Too tight drops a genuine triple point and
// leaves the naked edges that were already there; too loose cannot admit a
// wrong point on its own, because a point off this curve still has to be on
// both of its surfaces, and that test separates the cases by fifteen orders.
//
// A curve too short to have an interior sample falls back to its own chord
// length, which is the only scale it carries.
function segmentDeviation(samples, seg) {
  let worst = 0;
  for (const j of [seg - 1, seg]) {
    if (j < 1 || j + 1 >= samples.length) continue;
    const a = samples[j - 1], b = samples[j + 1], p = samples[j];
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if (!(l2 > 0)) continue;
    const t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / l2;
    worst = Math.max(worst, dist3(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]));
  }
  return worst > 0 ? worst : dist3(samples[seg - 1], samples[seg]);
}

// The nearest point of a polyline, as the segment it fell in and the fraction
// along it. `seg` is the index of that segment's SECOND sample, so 0 means no
// segment was usable.
function nearestOnPolyline(p, poly) {
  let best = Infinity, seg = 0, bestT = 0;
  for (let i = 1; i < poly.length; i++) {
    const a = poly[i - 1], b = poly[i];
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if (!(l2 > 0)) continue;
    let t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / l2;
    t = Math.max(0, Math.min(1, t));
    const d = dist3(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]);
    if (d < best) { best = d; seg = i; bestT = t; }
  }
  return { seg, t: bestT, off: best };
}

// Rebuild a sample chain with the accepted points placed in their own segments,
// in order along each. Several triple points can land in one segment — three
// faces of a prism can meet the same chord — so they are grouped and sorted
// rather than inserted one at a time.
function spliceIntoPolyline(samples, adds, weld) {
  const byseg = new Map();
  for (const a of adds) {
    if (!byseg.has(a.seg)) byseg.set(a.seg, []);
    byseg.get(a.seg).push(a);
  }
  for (const list of byseg.values()) list.sort((x, y) => x.t - y.t);
  const out = [samples[0]];
  for (let i = 1; i < samples.length; i++) {
    for (const a of byseg.get(i) ?? []) {
      if (dist3(a.p, out[out.length - 1]) > weld) out.push(a.p);
    }
    out.push(samples[i]);
  }
  return out;
}

// A cut curve that goes ALL THE WAY AROUND a closed direction is a closed
// loop in 3D but NOT a closed loop in (u,v) — it runs off one edge of the
// parametric rectangle and resumes at the other, because u=uMin and u=uMax
// are the same physical place. Handed to the arrangement as-is, the samples
// read as an out-and-back that stops short of the domain edge, which Stage
// 3 correctly prunes as a dangling spur — the face comes back unsplit and
// the boolean quietly builds from one fragment where there should be
// several. Re-expressing it as the open spine that genuinely reaches both
// edges is what makes the cut land.
//
// Gated on the curve being CLOSED IN 3D, not on the UV samples alone: an
// open curve's own first and last samples can sit on opposite sides of the
// seam by coincidence, and the cyclic jump test cannot tell that apart from
// a real wrap.
//
// A REFUSAL FROM seamCrossingSpine IS NOT INTERCHANGEABLE WITH SUCCESS;
// treating it as one produces silently wrong solids. Returning the raw UV
// for a chain that genuinely jumps the domain hands the arrangement a
// phantom chord spanning the whole parametric rectangle; it dutifully splits
// the face along that phantom and the boolean builds a plausible-looking
// solid out of fragments that are not real. Measured on a sphere pair
// straddling its own seam: six fragments where the geometry has two, then an
// open shell with 59 naked edges. The `code` field is what lets this
// function tell the cases apart without matching on prose:
//
//   no-seam-crossing / too-few-points — the loop never jumps the domain at
//   all. This is the ordinary interior cut, the overwhelmingly common case,
//   and the raw UV chain is exactly right. Pass it through.
//
//   seam-straddle — the loop crosses the seam an EVEN number of times, so it
//   is an ordinary region sitting ON the seam rather than a wrap. It is
//   genuinely SEVERAL pieces in this surface's parameters, and
//   seamStraddleChains returns one boundary-reaching chain per piece. A
//   single curve therefore contributes several pcurves here, which the
//   arrangement handles natively — they are just more cuts.
//
//   double-straddle — the surface is closed in BOTH directions and the loop
//   sits over the corner where its two seams meet, crossing each and winding
//   around neither. Contractible, ordinary geometry in awkward parameters;
//   seamDoubleStraddleChains splits it on both seams in turn.
//
//   double-wrap / wrap-and-straddle / multi-wrap — the chain genuinely winds
//   around a closed direction in a way neither routine re-expresses. REFUSE BY
//   NAME. The distinction from the case above is NET WINDING, not which
//   directions carry jumps: presence and winding are different questions.
//
// The straddle is the reachable case, and it is generic rather than exotic:
// for two solids of revolution, each operand's seam meridian typically
// points straight through the other body. Whether a given pair works is
// therefore a matter of which way its seams happen to face — the same
// sphere pair closes cleanly with its seams rotated off the cut.
//
// The two UV copies of a crossing point are the same place in 3D, so the
// fragments either side of the seam weld to each other in the ordinary sew
// with no seam-specific step.
function unwrapSeamCut(uv, samples3d, srf) {
  if (!samples3d || samples3d.length < 3) return { ok: true, uvs: [uv] };
  const a = samples3d[0], b = samples3d[samples3d.length - 1];
  let span = 0;
  for (let i = 1; i < samples3d.length; i++) {
    const p = samples3d[i], q = samples3d[i - 1];
    span = Math.max(span, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
  }
  const gap = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  if (!(gap <= span * 0.5 + 1e-9)) return unwrapOpenSeamCut(uv, srf);
  // A closed sample chain usually repeats its own first point last; the
  // wrap arithmetic wants each point once.
  const loop = uv.slice();
  const f = loop[0], l = loop[loop.length - 1];
  if (Math.abs(f[0] - l[0]) < 1e-12 && Math.abs(f[1] - l[1]) < 1e-12) loop.pop();
  const cut = seamCrossingSpine(loop, srf);
  if (cut.ok) return { ok: true, uvs: [cut.spine] };
  if (cut.code === 'no-seam-crossing' || cut.code === 'too-few-points') return { ok: true, uvs: [uv] };
  if (cut.code === 'seam-straddle') {
    const s = seamStraddleChains(loop, srf);
    if (s.ok) return { ok: true, uvs: s.chains };
    return { ok: false, code: s.code, reason: s.reason };
  }
  // A doubly-closed surface's two seams meet at the domain's corners, and a
  // region sitting over one of those corners crosses both while winding around
  // neither. It is the straddle case in both directions at once, not a wrap.
  if (cut.code === 'double-straddle') {
    const d = seamDoubleStraddleChains(loop, srf);
    if (d.ok) return { ok: true, uvs: d.chains };
    return { ok: false, code: d.code, reason: d.reason };
  }
  return { ok: false, code: cut.code, reason: cut.reason };
}

// THE OPEN-CURVE HALF, and the case the closed-in-3D gate above does not
// catch.
//
// That gate is correct about what it tests: an open curve's two ends can
// land either side of a seam by coincidence, and the CYCLIC jump test
// cannot tell that apart from a real once-around wrap, so an open curve
// must never reach seamCrossingSpine. What it must NOT do is fall through
// — returning the raw uv is only valid for a curve that never touches the
// seam at all. An open arc crossing the seam in its MIDDLE (a sphere cut
// by a box PANEL: two spheres always meet in a closed circle, so no
// closed-surface fixture can produce this) would otherwise skip every seam
// path in the kernel and hand the arrangement a phantom chord across
// almost the whole face.
//
// seamOpenChains is the sibling that owns this case: split at the interior
// crossings into open sub-chains, each reaching the domain edge exactly
// where the curve genuinely leaves it. A chain that never crosses is the
// ordinary interior cut and passes through raw.
function unwrapOpenSeamCut(uv, srf) {
  const open = seamOpenChains(uv, srf);
  if (open.ok) return { ok: true, uvs: open.chains };
  // The two benign codes mean the chain is not entangled with a seam at
  // all — the overwhelmingly common case, and the raw chain is exactly
  // right. Everything else is a topology this does not re-express, and a
  // refusal is the honest answer rather than a phantom chord.
  if (open.code === 'no-seam-crossing' || open.code === 'too-few-points') return { ok: true, uvs: [uv] };
  return { ok: false, code: open.code, reason: open.reason };
}

// One face: project its own curves into its parameters, split, keep.
// A fragment's share of its surface's own parameter domain, by the shoelace
// area of its outer loop minus its holes. Reported alongside the region so a
// classification can be judged: "kept 2 of 5" says nothing until you know
// whether the three dropped ones were slivers or most of the surface.
function fragmentDomainFraction(srf, fragment) {
  const shoelace = (loop) => {
    let a = 0;
    for (let i = 0; i < loop.length; i++) {
      const p = loop[i], q = loop[(i + 1) % loop.length];
      a += p[0] * q[1] - q[0] * p[1];
    }
    return Math.abs(a / 2);
  };
  const outer = fragment.outer || [];
  if (outer.length < 3) return 0;
  const uSpan = srf.knotsU[srf.knotsU.length - 1] - srf.knotsU[0];
  const vSpan = srf.knotsV[srf.knotsV.length - 1] - srf.knotsV[0];
  const domain = Math.abs(uSpan * vSpan);
  if (!(domain > 0)) return 0;
  let area = shoelace(outer);
  for (const h of fragment.holes || []) if (h.length >= 3) area -= shoelace(h);
  return Math.max(0, area) / domain;
}
function resolveFace(face, mineCurves, otherTriangles, operation, operand, label, opts) {
  const outer = face.trimLoop ?? trivialTrimLoop(face.srf);
  const holes = face.trimHoles ?? [];

  const pcurves = [];
  for (let ci = 0; ci < mineCurves.length; ci++) {
    const proj = projectPointsToSurfaceUV(mineCurves[ci].samples, face.srf, opts);
    if (!proj.ok) {
      return {
        ok: false,
        reason: `${label}: an intersection curve tagged as lying on it does not — ${proj.reason}`,
        report: { label, curves: mineCurves.length, error: 'projection' },
      };
    }
    const unwrapped = unwrapSeamCut(proj.uv, mineCurves[ci].samples, face.srf);
    if (!unwrapped.ok) {
      return {
        ok: false,
        reason: `${label}: an intersection curve ${unwrapped.reason}`,
        report: { label, curves: mineCurves.length, error: 'seam', code: unwrapped.code },
      };
    }
    // One curve can contribute several cuts: a region straddling the seam is
    // genuinely several pieces in this face's own parameters.
    pcurves.push(...unwrapped.uvs);
  }

  // No curve crosses this face, so it is wholly one side of the other solid
  // and its own trim loop is its single fragment.
  const fragments = pcurves.length ? null : [{ outer, holes }];
  let split = null;
  if (pcurves.length) {
    split = splitFaceByCurves({ outer, holes }, pcurves, opts);
    if (!split.ok) {
      return {
        ok: false,
        reason: `${label}: could not be split — ${split.reason}`,
        report: { label, curves: pcurves.length, error: 'split' },
      };
    }
  }

  // A CUT CURVE THAT CONTRIBUTED NO EDGE IS THE SILENT FAILURE THIS WHOLE
  // FILE'S HISTORY TURNS ON, AND IT MUST NOT BE DROPPED ON THE FLOOR.
  //
  // `splitFaceByCurves` already measures it: a curve was handed in, the
  // arrangement pruned it as a dangling spur, and the face came back with
  // its own trim loop as the single "fragment" — an UNCUT face, returned
  // with ok:true because nothing in the split itself was malformed. Every
  // downstream check then passes over a solid that was never split.
  //
  // That is exactly how the clipped-SSI shortfall stayed invisible: the run
  // stopped short of the trim boundary, the arc never reached the loop it
  // was meant to cut along, and the boolean reported success. The signal to
  // catch it existed the whole time and this function discarded it.
  //
  // Reported, NOT refused. A dangling curve is a real diagnostic about the
  // input, but it is not by itself proof the answer is wrong — a face pair
  // can legitimately be tagged with a curve that grazes rather than crosses
  // it. Turning this into a refusal without knowing which of those it is
  // would trade a silent wrong answer for a loud wrong refusal. The count
  // rides on the report so a caller (and the suite) can see it.
  const dangling = split ? (split.danglingCurves ?? 0) : 0;

  const toClassify = fragments ?? split.fragments;
  const decision = keepFragments(face.srf, toClassify, otherTriangles, operation, { ...opts, operand });
  if (!decision.ok) {
    return {
      ok: false,
      reason: `${label}: ${decision.reason}`,
      report: { label, curves: pcurves.length, fragments: toClassify.length, dangling, error: 'classify' },
    };
  }

  return {
    ok: true,
    kept: decision.kept,
    report: {
      label,
      curves: pcurves.length,
      fragments: toClassify.length,
      kept: decision.kept.length,
      regions: decision.classifications.map((c) => c.region),
      // HOW BIG EACH FRAGMENT IS, next to what it was called. A region list
      // alone cannot distinguish a correct classification from a badly wrong
      // one: dropping a sliver and dropping half the surface both read as one
      // fewer `kept`. As a fraction of the surface's own parameter domain,
      // because that is comparable across fragments of one face and needs no
      // evaluation — a fragment holding 40% of the domain that gets discarded
      // is a hole the size of the model, which is exactly the failure the
      // banked torus pair shows (about six missing faces, naked edges up to
      // 197 long on a model spanning 437).
      domainFractions: toClassify.map((f) => fragmentDomainFraction(face.srf, f)),
      split: !!pcurves.length,
      dangling,
      alongBoundary: split ? (split.alongBoundary ?? 0) : 0,
    },
  };
}
