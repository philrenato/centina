// The three operators — Phase 8 of the boolean pipeline, second half.
// Union, Difference and Intersect are one machine: split every face where the
// two solids meet, decide which fragments survive, sew the survivors back
// into a closed shell. Only the keep-rule differs, and that difference is one
// function call (kernel/booleansew.mjs's `keepFragments`).
//
// What this owns, and what it does not. Every hard step exists and is tested
// on its own: `intersectSurfacesComplete` marches the
// curves, `projectPointsToSurfaceUV` puts them in a face's own parameters,
// `splitFaceByCurves` cuts the face up, `classifyFragment` decides inside from
// outside, `sewFragments` welds the result. This module is the assembly, and
// assembling is where a boolean usually goes wrong quietly — so every step
// that cannot be completed refuses by name, naming the face it failed on,
// rather than dropping a fragment and returning a plausible solid.
//
// Curves arrive tagged with their own face pair. An intersection curve lies
// on exactly the two faces that produced it; blind-projecting every curve onto
// every face would ask `projectPointsToSurfaceUV` to answer "is this curve on
// this surface" as a side effect of a tolerance check, which it is not built
// to decide and would get wrong for two faces that happen to be
// near-coincident. The caller already knows the pairing — a face-pair sweep is
// how the curves were computed — so it passes it through instead of having it
// re-derived.
//
// A face with no intersection curve is not split at all. It lies wholly
// inside or wholly outside the other solid, so its own trim loop is its
// single fragment and one classification decides it. This is not an
// optimization: running an empty split just to get one fragment back would
// route the commonest face in any real boolean through the machinery most
// likely to refuse.
//
// Orientation is the sew's job. Nothing here reverses B's faces for a
// Difference. `buildBrepSolid`'s own `orientLoops` pass re-winds every face
// into one consistent orientation as part of welding, so a per-operand flip
// here would be undone or fought. What that pass guarantees is consistency,
// not that the result faces outward — a global inside-out shell is a real
// possible outcome and is checked by signed volume, not assumed away.

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
 *   the other solid's fragments. An untrimmed face may omit `trimLoop`; its
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
 * what a caller builds from. `solid` is the welded Brep, which proves the
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
    : { ok: false, reason: `the kept fragments do not sew into a closed solid — ${sewn.verdict}`, verdict: sewn.verdict, stats: sewn.stats ?? null, worstSharedGap: sewn.worstSharedGap, nakedEdgePoints: sewn.nakedEdgePoints ?? null, nonManifoldEdgePoints: sewn.nonManifoldEdgePoints ?? null, nonManifoldEdgeUses: sewn.nonManifoldEdgeUses ?? null, fragments: kept, keptCount: kept.length, triplePoints: shared.inserted, seamPoints: seamShared.inserted, faceReports };
}

// A seam crossing is a corner of the answer. Every seam crossing is computed
// here from the shared curve and every face is given the same ones.
//
// A cut curve running across a closed surface's seam has to be broken there:
// the domain rectangle's two ends are the same physical place, so the chain is
// re-expressed as pieces that reach the edge (seamCrossingSpine,
// seamStraddleChains, seamOpenChains, all via seamPointAt). The crossing's
// position is interpolated between the two samples straddling the domain edge,
// in that surface's own parameters.
//
// Two faces cut by the same curve would therefore compute the corner twice.
// While their seams are in different places that is harmless — the two corners
// are different points and each is a corner of one face only. When the seams
// pass through the same place it is not: both faces put a vertex at what is
// geometrically one point, from two different interpolations, and the answers
// differ. For two revolved balls whose offset points straight along both their
// seam meridians, one surface each and one intersection circle, the two
// corners land 1.3e-2 apart against a 1e-4 weld, and the shell comes back
// with a four-edge sliver quad — the two faces' corners joined by the two
// curve samples either side, enclosing nothing. Refining the march does not
// close it. The separation falls linearly with the sample spacing while the
// naked count stays at exactly 4, because the disagreement is not an accuracy
// problem: no sampling makes two independent interpolations agree to a
// tolerance.
//
// So every face's crossing is computed here, before anything is projected or
// split, and spliced into the curve every face reads. Each face then finds its
// own corner sitting exactly on a sample it already carries and paves the
// other face's corner as an ordinary interior point, so the two boundaries run
// through the same vertices in the same order and weld by construction. It is
// the argument shareTriplePoints makes, for the other kind of corner.
//
// Idempotent by construction, which is what makes the splice safe to do for
// every face at once rather than in some order. A point placed exactly on a
// surface's own domain edge re-projects to that edge, so the jump either side
// of it interpolates with a fraction of 0 or 1 and returns the inserted point
// unchanged — a face's second look at its own corner cannot move it.
//
// The corner is solved, not interpolated. seamPointAt's own answer is a
// linear interpolation across the jump, so it is exact on the seam and off the
// true intersection curve by roughly one sample spacing — 9.7e-3 on the ball
// pair, which `projectPointsToSurfaceUV` refuses as "not on the surface" the
// moment the paired face reads it. Sharing that point would hand the other
// face a corner it cannot accept.
//
// The exact corner is available and cheap. A seam is an isocurve of its own
// surface — the domain edge a = aEdge, traced by the other parameter — so the
// crossing is where that isocurve meets the paired surface: one unknown, one
// equation, bracketed by the two samples that straddle the jump. Solving it
// gives a point exactly on the seam and on both surfaces, which is what both
// faces need and neither could interpolate.
//
// Two faces whose seams pass through one place then solve the same equation
// and land on the same point to machine precision, so they weld with nothing
// left over. Where the seams are apart the two solves return two different
// corners, each spliced in and paved by both faces — which is why this shares
// every crossing rather than trying to detect coincident seams.
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
        // On the curve, judged against the curve's own samples rather than a
        // chosen number. A solve that converged somewhere else — the isocurve
        // grazing the other surface, or never reaching it inside the bracket —
        // fails this by orders, and inserting it would put a corner on the
        // boundary where the geometry has none.
        //
        // The median residual, not the worst. `closestPointOnSurface` converges
        // poorly near a pole, where a whole domain row collapses to one point and
        // the parameters stop separating — so one sample of a curve that passes near
        // one can read 9e-2 while the rest read 1e-13. A worst-case floor takes that
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
// Ternary rather than Newton, and rather than a signed bisection. Distance to
// a surface is non-negative with a minimum of zero at the crossing, so it has
// no sign to bisect on and no derivative that stays well conditioned as it
// approaches the root — while a unimodal minimum is exactly what a ternary
// search is for. It cannot diverge, needs no initial slope, and the bracket is
// already in hand from the two samples that straddle the jump.
//
// The bracket is grown before it is searched. The two samples straddling the
// jump bound the crossing only as well as the polyline does, and on a marched
// curve they can sit 2.7e-5 apart in parameter with the true root just outside
// — a search confined to them stalls against its own endpoint and returns a
// point still 6.3e-4 off the paired surface, which the acceptance test
// refuses. So the interval doubles about its own midpoint until the cost at
// both ends exceeds the cost in the middle, which is what makes it a bracket
// rather than a guess, and only then is it searched.
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

// A triple point is already computed exactly; it is missing from one of the
// curves that pass through it, and that is why a multi-face operand leaves
// slivers.
//
// Where three faces meet, two intersection curves pass through one point. On a
// multi-face operand that is ordinary rather than exotic: a prism's side face
// and its cap share an edge, so the other solid's surface crosses all three at
// once. One of those curves generally ends there — the side face's march runs
// out of its own domain exactly on the cap plane — and that endpoint lies on
// all three surfaces to machine precision. For a revolved blob against a
// 12-face star prism, the endpoint sits 2.8e-15 from the blob and 5.0e-15
// from the cap plane, against 6.5e-15 and 7.3e-15 for the cap curve's own
// samples. It is the exact answer, already in hand.
//
// The other curve — the blob against the cap plane — simply has no sample
// there; the marcher stepped past it, nearest sample 9.4e-2 away. So the cap
// face, which is split by that curve against its own star trim loop, finds the
// crossing on a chord instead, 1.9e-3 off the true point. The blob face, which
// is split by both curves, lands on the exact one. Two faces sharing an edge
// therefore disagree by 22x the weld tolerance, and the shell comes back with
// one sliver triangle per triple point.
//
// So this inserts the point that is already known into the curve that is
// missing it, before anything is projected or split. Every face then paves the
// same 3D point and the fragments weld by construction, which is the same
// argument densifyOnCutLines makes for a seam cut line, one stage earlier.
//
// Why exactly one shared face, and both halves of that are load-bearing.
//
// A curve's endpoint only needs to be a vertex of another curve if some face is
// split by both — that is exactly when their crossing has to be paved
// consistently. Curves with no face in common are never seen together by any
// arrangement, and forcing a vertex between them would invent a corner nothing
// asked for.
//
// Curves sharing both faces are excluded for the opposite reason: they are
// separate components of one surface-surface intersection, so they are disjoint
// by construction and an endpoint of one is never a point of the other. The
// surface test below cannot see that — every sample of either component lies on
// both surfaces, so the test passes vacuously and would splice one component's
// end into whichever chord of the other it happened to pass nearest. An
// off-axis star prism has two components against two of its side faces, and
// admitting them turns an open shell into a refusal.
//
// The acceptance test is a measurement against the curve's own samples, not a
// chosen tolerance. A candidate is inserted only if it sits on both of that
// curve's surfaces no worse than the curve's own samples do, read with the same
// projector. That comparison is not close: on the prism above a true
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

  // Only an open curve has endpoints to share. A closed loop's first and last
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
    let replaceStart = null, replaceEnd = null;
    let floor = null; // computed once, and only if this curve has a candidate
    for (const e of ends) {
      if (e.from === ci) continue;
      const o = out[e.from];
      const sharesA = o.faceA === c.faceA, sharesB = o.faceB === c.faceB;
      if (sharesA === sharesB) continue; // neither face in common, or both
      const hit = nearestOnPolyline(e.p, c.samples);
      if (hit.seg < 1) continue;
      const atStart = hit.seg === 1 && hit.t === 0;
      const atEnd = hit.seg === c.samples.length - 1 && hit.t === 1;
      // A point of this curve, falling between two of its samples, sits at most
      // its own chord's sagitta off that chord — and the polyline measures that
      // for itself, so nothing here has to be chosen. See segmentDeviation.
      if (!atStart && !atEnd && hit.off > segmentDeviation(c.samples, hit.seg)) continue;
      const a = c.samples[hit.seg - 1], b = c.samples[hit.seg];
      // A point landing within the weld tolerance of a sample this curve
      // already carries is that sample as far as the sew is concerned;
      // inserting it would only add an edge shorter than a vertex is wide.
      if (!atStart && !atEnd && (dist3(e.p, a) <= weld || dist3(e.p, b) <= weld)) continue;
      if (floor === null) {
        floor = calibrationFloor(c.samples, srfA, srfB) * TRIPLE_POINT_RESIDUAL_SLACK;
      }
      const residualLimit = (atStart || atEnd) ? Math.max(floor, weld * 1e-2) : floor;
      if (closestPointOnSurface(srfA, e.p).distance > residualLimit) continue;
      if (closestPointOnSurface(srfB, e.p).distance > residualLimit) continue;
      if (atStart || atEnd) {
        // A marcher clipped independently on the adjacent face pair may stop
        // just short of this exact triple point. Extend only by a fraction of
        // its neighboring real step; the two-surface residual checks above
        // prove this is on the same intersection, not merely nearby.
        const edgeStep = atStart ? dist3(c.samples[0], c.samples[1])
          : dist3(c.samples[c.samples.length - 2], c.samples[c.samples.length - 1]);
        if (!(edgeStep > 0) || hit.off > edgeStep * 0.25) continue;
        const slot = atStart ? replaceStart : replaceEnd;
        if (!slot || hit.off < slot.off) {
          const candidate = { p: e.p, off: hit.off };
          if (atStart) replaceStart = candidate; else replaceEnd = candidate;
        }
        continue;
      }
      adds.push({ seg: hit.seg, t: hit.t, p: e.p });
    }
    if (!adds.length && !replaceStart && !replaceEnd) continue;
    const before = c.samples.length;
    c.samples = spliceIntoPolyline(c.samples, adds, weld);
    if (replaceStart) c.samples[0] = replaceStart.p;
    if (replaceEnd) c.samples[c.samples.length - 1] = replaceEnd.p;
    // What went in, not what was proposed: the splice drops a point that
    // would land within the weld tolerance of one already placed, and a count of
    // the proposals would not match the curves it describes.
    inserted += c.samples.length - before + (replaceStart ? 1 : 0) + (replaceEnd ? 1 : 0);
  }
  return { curves: out, inserted };
}

// The same closed-in-3D test the seam unwrap uses: a closed chain's two ends
// meet within half of its own largest step, which no open chain can do by
// coincidence at any sampling density.
function closedIn3D(samples) {
  let span = 0;
  for (let i = 1; i < samples.length; i++) span = Math.max(span, dist3(samples[i], samples[i - 1]));
  return dist3(samples[0], samples[samples.length - 1]) <= span * 0.5 + 1e-9;
}

function dist3(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// How far this curve's own samples read from the two surfaces they lie on —
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

// How far off its own chord a point of this curve can sit, measured from the
// polyline rather than assumed.
//
// A sample straddled by its two neighbors is a point of the true curve, and
// its distance from the chord joining those neighbors is that two-segment
// chord's sagitta, directly observed. A chord half as long has a quarter the
// sagitta, so one segment's own bound is about a quarter of what is measured
// here — returning the measured two-segment figure therefore keeps a fourfold
// margin over the bound it stands for: this is a plausibility guard behind an
// exact surface test, and the two failure directions are not symmetric. Too
// tight drops a true triple point and leaves the naked edges that were
// already there; too loose cannot admit a wrong point on its own, because a
// point off this curve still has to be on both of its surfaces, and that test
// separates the cases by fifteen orders.
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
// along it. `seg` is the index of that segment's second sample, so 0 means no
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

// A cut curve that goes all the way around a closed direction is a closed
// loop in 3D but not a closed loop in (u,v) — it runs off one edge of the
// parametric rectangle and resumes at the other, because u=uMin and u=uMax
// are the same physical place. Handed to the arrangement as-is, the samples
// read as an out-and-back that stops short of the domain edge, which Stage
// 3 correctly prunes as a dangling spur — the face comes back unsplit and
// the boolean quietly builds from one fragment where there should be
// several. Re-expressing it as the open spine that reaches both edges is
// what makes the cut land.
//
// Gated on the curve being closed in 3D, not on the UV samples alone: an
// open curve's own first and last samples can sit on opposite sides of the
// seam by coincidence, and the cyclic jump test cannot tell that apart from
// a real wrap.
//
// A refusal from seamCrossingSpine is not interchangeable with success.
// Returning the raw UV for a chain that jumps the domain hands the
// arrangement a phantom chord spanning the whole parametric rectangle; it
// splits the face along that phantom and the boolean builds a
// plausible-looking solid out of fragments that are not real (for a sphere
// pair straddling its own seam: six fragments where the geometry has two,
// then an open shell with 59 naked edges). The `code` field is what lets
// this function tell the cases apart without matching on prose:
//
//   no-seam-crossing / too-few-points — the loop never jumps the domain at
//   all. This is the ordinary interior cut, the overwhelmingly common case,
//   and the raw UV chain is exactly right. Pass it through.
//
//   seam-straddle — the loop crosses the seam an even number of times, so it
//   is an ordinary region sitting on the seam rather than a wrap. It is
//   several pieces in this surface's parameters, and
//   seamStraddleChains returns one boundary-reaching chain per piece. A
//   single curve therefore contributes several pcurves here, which the
//   arrangement handles natively — they are just more cuts.
//
//   double-straddle — the surface is closed in both directions and the loop
//   sits over the corner where its two seams meet, crossing each and winding
//   around neither. Contractible, ordinary geometry in awkward parameters;
//   seamDoubleStraddleChains splits it on both seams in turn.
//
//   double-wrap / wrap-and-straddle / multi-wrap — the chain winds around a
//   closed direction in a way neither routine re-expresses. Refused by name.
//   The distinction from the case above is net winding, not which directions
//   carry jumps: presence and winding are different questions.
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

// The open-curve half, and the case the closed-in-3D check above does not
// catch.
//
// That check is correct about what it tests: an open curve's two ends can
// land either side of a seam by coincidence, and the cyclic jump test
// cannot tell that apart from a real once-around wrap, so an open curve
// must never reach seamCrossingSpine. It must not fall through either —
// returning the raw uv is only valid for a curve that never touches the
// seam at all. An open arc crossing the seam in its middle (a sphere cut
// by a box panel: two spheres always meet in a closed circle, so a sphere
// pair cannot produce this) would otherwise skip every seam
// path in the kernel and hand the arrangement a phantom chord across
// almost the whole face.
//
// seamOpenChains is the sibling that owns this case: split at the interior
// crossings into open sub-chains, each reaching the domain edge exactly
// where the curve leaves it. A chain that never crosses is the
// ordinary interior cut and passes through raw.
function unwrapOpenSeamCut(uv, srf) {
  const open = seamOpenChains(uv, srf);
  if (open.ok) return { ok: true, uvs: open.chains };
  // The two benign codes mean the chain is not entangled with a seam at
  // all — the overwhelmingly common case, and the raw chain is exactly
  // right. Everything else is a topology this does not re-express, and it is
  // refused rather than handed on as a phantom chord.
  if (open.code === 'no-seam-crossing' || open.code === 'too-few-points') return { ok: true, uvs: [uv] };
  return { ok: false, code: open.code, reason: open.reason };
}

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

// An open intersection chain must reach the face boundary to cut the face.
// Marching stops at its last accepted 3-D sample, which can sit one sample
// short of a trim boundary even though the underlying intersection continues
// to it. In the arrangement that is a dangling spur and silently contributes
// no edge. Complete only that final, measured shortfall: an endpoint may snap
// to the nearest trim segment when the gap is at most two of the chain's own
// adjacent sample steps. Interior loops and interior open curves remain
// untouched.
function completePcurveToTrimBoundary(curve, outer, holes, adoptExact = false, protectedPoints = []) {
  if (!curve || curve.length < 2) return curve;
  const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const closedStep = Math.max(d2(curve[0], curve[1]), d2(curve[curve.length - 2], curve[curve.length - 1]));
  if (d2(curve[0], curve[curve.length - 1]) <= closedStep * 0.5) return curve;
  const boundaries = [outer, ...(holes || [])];
  const nearest = (p) => {
    let best = null;
    for (const loop of boundaries) {
      for (let i = 0; i < loop.length; i++) {
        const a = loop[i], b = loop[(i + 1) % loop.length];
        const dx = b[0] - a[0], dy = b[1] - a[1], den = dx * dx + dy * dy;
        let t = den > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den : 0;
        t = Math.max(0, Math.min(1, t));
        const q = [a[0] + dx * t, a[1] + dy * t], dist = d2(p, q);
        if (!best || dist < best.dist) best = { q, dist, loop, index: i };
      }
    }
    return best;
  };
  const out = curve.map((p) => p.slice());
  for (const end of [0, out.length - 1]) {
    if (protectedPoints.some((p) => d2(out[end], p) <= 1e-9)) continue;
    const neighbor = end === 0 ? 1 : out.length - 2;
    const step = d2(out[end], out[neighbor]);
    const hit = nearest(out[end]);
    if (hit && step > 0 && hit.dist <= step * 2 + 1e-12) {
      if (adoptExact && hit.dist > 1e-12 && hit.dist <= step * 0.01) {
        // A shared triple point can be more accurate than a polygonal trim
        // chord sampled from the same curved rim. In that near-zero case the
        // topology adopts the exact point instead of moving the intersection
        // back onto the approximation.
        hit.loop.splice(hit.index + 1, 0, out[end].slice());
      } else {
        out[end] = hit.q;
      }
    }
  }
  return out;
}

// One face: project its own curves into its parameters, split, keep.
function resolveFace(face, mineCurves, otherTriangles, operation, operand, label, opts) {
  const outer = (face.trimLoop ?? trivialTrimLoop(face.srf)).map((p) => p.slice());
  const holes = (face.trimHoles ?? []).map((h) => h.map((p) => p.slice()));

  const pcurves = [];
  const projected = [];
  for (let ci = 0; ci < mineCurves.length; ci++) {
    const proj = projectPointsToSurfaceUV(mineCurves[ci].samples, face.srf, opts);
    if (!proj.ok) {
      return {
        ok: false,
        reason: `${label}: an intersection curve tagged as lying on it does not — ${proj.reason}`,
        report: { label, curves: mineCurves.length, error: 'projection' },
      };
    }
    projected.push({ uv: proj.uv, samples: mineCurves[ci].samples });
  }

  // The same 3-D triple point has one UV address on this face. Inverted per
  // curve, closest-point iteration on a planar fitted cap converges to
  // different parameters for one world point; the arrangement then sees two
  // open cuts where there is one junction and keeps a fragment spanning both
  // sides of the cutter. Shared 3-D endpoints are pooled and every copy takes
  // the inverse with the smaller surface residual.
  const endpointGroups = [];
  const { closedU: endpointClosedU, closedV: endpointClosedV } = surfaceClosure(face.srf);
  const weld = opts.weldTolerance ?? 1e-4;
  const trimLoops = [outer, ...holes];
  const uvSpan = Math.max(...outer.map((p) => p[0])) - Math.min(...outer.map((p) => p[0]))
    + Math.max(...outer.map((p) => p[1])) - Math.min(...outer.map((p) => p[1]));
  const trimDistance = (p) => {
    let best = Infinity;
    for (const loop of trimLoops) for (let i = 0; i < loop.length; i++) {
      const a = loop[i], b = loop[(i + 1) % loop.length];
      const dx = b[0] - a[0], dy = b[1] - a[1], den = dx * dx + dy * dy;
      const t = den ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den)) : 0;
      best = Math.min(best, Math.hypot(p[0] - a[0] - dx * t, p[1] - a[1] - dy * t));
    }
    return best;
  };
  if (!endpointClosedU && !endpointClosedV) for (const rec of projected) for (const index of [0, rec.uv.length - 1]) {
    const p = rec.samples[index], uv = rec.uv[index];
    let group = endpointGroups.find((g) => dist3(g.p, p) <= weld);
    const q = surfacePoint(face.srf, uv[0], uv[1]);
    const residual = dist3(q, p);
    if (!group) { group = { p, refs: [], uv, residual }; endpointGroups.push(group); }
    group.refs.push({ rec, index });
    if (residual < group.residual) { group.uv = uv; group.residual = residual; }
  }
  const sharedPcurveEndpoints = [];
  for (const group of endpointGroups) {
    if (group.refs.length < 2) continue;
    for (const ref of group.refs) ref.rec.uv[ref.index] = group.uv.slice();
    if (trimDistance(group.uv) > Math.max(uvSpan * 1e-5, 1e-10)) sharedPcurveEndpoints.push(group.uv.slice());
  }

  for (const rec of projected) {
    const unwrapped = unwrapSeamCut(rec.uv, rec.samples, face.srf);
    if (!unwrapped.ok) {
      return {
        ok: false,
        reason: `${label}: an intersection curve ${unwrapped.reason}`,
        report: { label, curves: mineCurves.length, error: 'seam', code: unwrapped.code },
      };
    }
    // One curve can contribute several cuts: a region straddling the seam is
    // several pieces in this face's own parameters.
    pcurves.push(...unwrapped.uvs);
  }

  const joinedPcurves = pcurves.map((uv) => completePcurveToTrimBoundary(uv, outer, holes, !!face.srf.sharedBoundaryTrim, sharedPcurveEndpoints));
  pcurves.length = 0;
  pcurves.push(...joinedPcurves);

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

  // A cut curve that contributed no edge must be reported.
  //
  // `splitFaceByCurves` measures it: a curve was handed in, the arrangement
  // pruned it as a dangling spur, and the face came back with its own trim
  // loop as the single "fragment" — an uncut face, returned with ok:true
  // because nothing in the split itself was malformed. Every downstream check
  // then passes over a solid that was never split. A marched intersection that
  // stops short of the trim boundary produces exactly this: the arc never
  // reaches the loop it was meant to cut along, and the boolean reports
  // success.
  //
  // Reported, not refused. A dangling curve is a real diagnostic about the
  // input, but it is not by itself proof the answer is wrong — a face pair
  // can legitimately be tagged with a curve that grazes rather than crosses
  // it. Turning this into a refusal without knowing which of those it is
  // would trade a silent wrong answer for a loud wrong refusal. The count
  // rides on the report so a caller can see it.
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
  for (const fragment of decision.kept) fragment.booleanFaceLabel = label;
  return {
    ok: true,
    kept: decision.kept,
    report: {
      label,
      curves: pcurves.length,
      fragments: toClassify.length,
      kept: decision.kept.length,
      regions: decision.classifications.map((c) => c.region),
      // How big each fragment is, next to what it was called. A region list
      // alone cannot distinguish a correct classification from a badly wrong
      // one: dropping a sliver and dropping half the surface both read as one
      // fewer `kept`. As a fraction of the surface's own parameter domain,
      // because that is comparable across fragments of one face and needs no
      // evaluation — a fragment holding 40% of the domain that gets discarded
      // is a hole the size of the model.
      domainFractions: toClassify.map((f) => fragmentDomainFraction(face.srf, f)),
      split: !!pcurves.length,
      dangling,
      alongBoundary: split ? (split.alongBoundary ?? 0) : 0,
    },
  };
}
