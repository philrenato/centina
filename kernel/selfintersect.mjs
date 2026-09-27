// Self-intersection of a curve — the curve tier of the self-intersection
// guards, and the check a profile needs before it is consumed.
//
// Extrude, Revolve, Loft, Cap and Sweep all consume a profile curve, and each
// of them turns a self-intersecting profile into a self-intersecting surface.
// A figure-eight extruded gives a solid that looks plausible and then fails
// every downstream boolean, export and CAM step with no diagnostic.
//
// Three decisions:
//
// 1. Which plane. The best-fit plane through the curve's own dense samples,
//    found from the most spread sample pair about the centroid (see
//    `bestFitPlane` for why not Newell's method).
//
// 2. What planarity tolerance. Relative to the curve's own size (its sample
//    bounding-box diagonal), never an absolute millimeter count — otherwise
//    a 5mm curve and a 500mm curve are judged by different standards,
//    and this kernel has no document scale to appeal to. Generous (1%),
//    because the near-planar case is the one that matters: a planar sketch
//    with one point nudged off-plane loses its exact 3D crossing while
//    keeping the tolerance-level one that breaks downstream operations. A
//    curve that is planar to within 1% of its own size and crosses itself in
//    that plane will still produce a broken surface, so it is worth
//    catching; a spatial curve is not.
//
// 3. Which path. Planar -> an exact 2D segment-intersection test. Non-planar
//    -> reported as not tested (`planar: false`), never as "clean": a space
//    curve passing near itself is not an intersection, and claiming a clean
//    result for a case this tier cannot judge would be worse than saying
//    nothing.
//
// Not built here: a space-curve minimum self-distance report (a clearance
// number with a parametric-neighborhood exclusion for the s ~= t pairs that
// are trivially zero). It needs its own decision about what an actionable
// clearance number is. This module is the planar half.

import { curvePoint, adaptiveArcLengthSamples, isCurveClosed } from './curve.mjs';
import { segmentsIntersect } from './trim.mjs';
import { sub, cross, dot, normalize, length } from './vec3.mjs';

// How far off its own best-fit plane a curve may sit and still be judged
// planar, as a fraction of its own sample bounding-box diagonal. See decision
// 2 above for why this is relative and why it is generous.
export const PLANARITY_TOL_FRAC = 0.01;

// The plane the samples lie in, found by the most spread pair about the
// centroid rather than by Newell's method, which is why this does not reuse
// `capProfilePlaneCheck`'s.
//
// Newell's method cancels to zero on exactly the shapes this module exists to
// catch. It sums signed per-edge contributions, i.e. it is an area-weighted
// normal — and a figure-eight's two lobes wind in opposite directions, so
// their contributions cancel and the "normal" comes out zero-length. Newell's
// then reports the canonical self-intersecting profile as having no plane at
// all, and the guard declines to judge the one case it was built for.
//
// Taking the single most spread pair instead is exact for coplanar points
// (every pair's cross product is parallel to the true normal, so the largest
// is simply the best-conditioned one), independent of winding, and independent
// of the order the samples arrive in. `bridgeEdgeRunsHub` uses the same
// construction against the same failure: a sum of signed cross products
// canceling on symmetric input. Sign is not meaningful here — a
// self-intersection is a topological fact about the projected polygon, and
// which way the normal points cannot change it.
export function bestFitPlane(points) {
  const n = points.length;
  if (n < 3) return null;
  const centroid = [0, 0, 0];
  for (const p of points) { centroid[0] += p[0]; centroid[1] += p[1]; centroid[2] += p[2]; }
  centroid[0] /= n; centroid[1] /= n; centroid[2] /= n;
  // Anchor on the sample furthest from the centroid — the longest available
  // lever arm, so the cross products below are as well-conditioned as this
  // point set allows.
  let anchor = null, bestR = -1;
  for (const p of points) { const r = length(sub(p, centroid)); if (r > bestR) { bestR = r; anchor = p; } }
  if (!(bestR > 0)) return null; // every sample sits on the centroid
  const a = sub(anchor, centroid);
  let nrm = null, bestMag = 0;
  for (const p of points) {
    const c = cross(a, sub(p, centroid));
    const m = length(c);
    if (m > bestMag) { bestMag = m; nrm = c; }
  }
  // Scale-aware: the cross product's magnitude grows with the square of the
  // point set's own size, so an absolute floor would refuse a legitimately
  // small curve and accept noise on a large one.
  if (!nrm || !(bestMag > bestR * bestR * 1e-9)) return null; // collinear or coincident samples have no plane — refused rather than normalized into noise
  return { origin: centroid, normal: normalize(nrm) };
}

// Sample bounding-box diagonal — the size the planarity tolerance is relative
// to, and the curve's size for a curve that has no other notion of scale.
function sampleDiagonal(points) {
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of points) for (let i = 0; i < 3; i++) { if (p[i] < lo[i]) lo[i] = p[i]; if (p[i] > hi[i]) hi[i] = p[i]; }
  return length(sub(hi, lo));
}

// Self-intersection of a 2D polyline, adjacency-aware in both directions.
//
// `closed` is load-bearing, not cosmetic. `polylineSelfIntersects`
// (kernel/trim.mjs) is written for a trim loop and unconditionally wraps the
// final segment back to the start — correct there, and wrong for an open
// profile, where it would invent a closing segment the curve does not have
// and report a crossing against geometry that is not on screen.
export function polylineSelfIntersects2D(pts, closed) {
  const n = pts.length;
  const segCount = closed ? n : n - 1;
  if (segCount < 3) return false; // fewer than three segments cannot cross without sharing an endpoint
  for (let i = 0; i < segCount; i++) {
    const a0 = pts[i], a1 = pts[(i + 1) % n];
    for (let j = i + 1; j < segCount; j++) {
      // Adjacent segments legitimately share an endpoint; on a closed
      // polyline the first and last are adjacent too. Excluded in both
      // directions rather than only forward — a one-directional check reports
      // every closed curve as self-intersecting at its own seam.
      if (j === i + 1) continue;
      if (closed && i === 0 && j === segCount - 1) continue;
      if (segmentsIntersect(a0, a1, pts[j], pts[(j + 1) % n])) return true;
    }
  }
  return false;
}

// The entry point. Returns a result shape rather than a bare boolean, so a
// caller can distinguish
// "checked, clean", "checked, crosses itself" and "could not be checked at
// this tier" instead of collapsing the third into the first.
//
//   { tested, planar, selfIntersects, reason, planarityError, size }
//
// `tested: false` is never a pass. Every caller must treat it as "no verdict".
export function curveSelfIntersects(crv, opts = {}) {
  const tolFrac = opts.planarityTolFrac ?? PLANARITY_TOL_FRAC;
  const k = crv.knots;
  const uMin = k[crv.degree];
  const uMax = k[k.length - 1 - crv.degree];
  if (!(uMax > uMin)) return { tested: false, planar: false, selfIntersects: false, reason: 'degenerate parameter domain', planarityError: NaN, size: 0 };

  // Two-stage sampling, and the first stage is not optional.
  //
  // `adaptiveArcLengthSamples` refines against a chord-deviation tolerance in
  // the curve's own coordinate units, and this kernel has no document scale
  // to pick one from — passing `undefined` does not fall back to a default,
  // it makes every `deviation > tolerance` comparison false and silently
  // returns the knot seeds alone (four points for a four-segment profile,
  // which is not enough to see anything). So: a coarse uniform pass first,
  // purely to learn how big the curve is, then a real adaptive pass at a
  // tolerance relative to that size. A tight loop then earns the samples that
  // make its own crossing visible while a gentle curve does not pay for them.
  const COARSE = 48;
  const coarse = [];
  for (let i = 0; i <= COARSE; i++) coarse.push(curvePoint(crv, uMin + ((uMax - uMin) * i) / COARSE));
  const coarseSize = sampleDiagonal(coarse);
  if (!(coarseSize > 0)) return { tested: false, planar: false, selfIntersects: false, reason: 'zero-extent curve', planarityError: 0, size: 0 };
  const chordTol = coarseSize * (opts.chordTolFrac ?? 1e-3);
  // `adaptiveArcLengthSamples` returns a {u, pt} chain, not bare points.
  let pts = adaptiveArcLengthSamples(crv, uMin, uMax, chordTol).map((s) => s.pt);
  if (!pts || pts.length < 4) pts = coarse;
  const closed = isCurveClosed(crv);
  // A closed curve's sample chain repeats its own start as its last point.
  // Left in place, that duplicate becomes a zero-length final segment, which
  // is both meaningless and a source of spurious adjacency.
  if (closed && pts.length > 1 && length(sub(pts[pts.length - 1], pts[0])) < 1e-9) pts = pts.slice(0, -1);

  const size = sampleDiagonal(pts);
  if (!(size > 0)) return { tested: false, planar: false, selfIntersects: false, reason: 'zero-extent curve', planarityError: 0, size: 0 };

  const plane = bestFitPlane(pts);
  if (!plane) return { tested: false, planar: false, selfIntersects: false, reason: 'no well-defined plane (the samples are collinear)', planarityError: NaN, size };

  let maxDev = 0;
  for (const p of pts) maxDev = Math.max(maxDev, Math.abs(dot(sub(p, plane.origin), plane.normal)));
  const planar = maxDev <= size * tolFrac;
  if (!planar) {
    return { tested: false, planar: false, selfIntersects: false, reason: 'the curve is not planar — a space curve passing near itself is not an intersection, and this tier does not judge it', planarityError: maxDev, size };
  }

  // Project onto the plane's own orthonormal basis. Any two perpendicular
  // in-plane axes work — a self-intersection is a topological fact about the
  // projected polygon, invariant to which basis is chosen.
  const nz = plane.normal;
  const seed = Math.abs(nz[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const ax = normalize(cross(nz, seed));
  const ay = cross(nz, ax);
  const flat = pts.map((p) => { const d = sub(p, plane.origin); return [dot(d, ax), dot(d, ay)]; });

  return {
    tested: true,
    planar: true,
    selfIntersects: polylineSelfIntersects2D(flat, closed),
    reason: null,
    planarityError: maxDev,
    size,
  };
}
