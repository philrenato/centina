// Flip and seam — a shape's own parametrization, never its position in space.
//
// Both operations here change how a shape is described and leave the set of
// points it occupies exactly where it was. That is the whole of their
// contract: sample before, sample after, require the same point set. Nothing
// here is a mirror, a rotation or any other transform — kernel/transform.mjs
// owns those.
//
//   Flip    reverse the direction a curve runs; reverse the direction a
//           surface's normal points. Rhino calls these `Flip` and `Dir`.
//   Seam    move where a closed shape starts and ends — the start/end point
//           of a closed curve, the seam of a closed surface. Rhino calls
//           these `CrvSeam` and `SrfSeam`.
//
// Not to be confused with kernel/seam.mjs, which uses the same word for a
// different operation. That module cuts a triangulation open along a seam so
// kernel/flatten.mjs's LSCM can accept a disk; it duplicates vertices and
// re-indexes faces and never touches a knot vector. This file moves the seam
// of a NURBS description and never touches a mesh. Neither calls the other.
//
// Flipping a curve is kernel/curve.mjs's `reverseCurve`: exact (P&T §6.5),
// self-inverse, and used by extend/loft/sweep. This file does not wrap it in
// a second name; for curves it adds only the seam move.
//
// Flipping a surface normal necessarily reparametrizes. A tensor-product
// surface's normal is N = Su x Sv. There is no way to negate it without
// changing the order of one of the two directions (or swapping the two, which
// negates it as well and swaps the domains along with it). So a normal flip on
// a NURBS surface is always a reparametrization. The direction reversed here
// is u — the same choice kernel/offset.mjs's `reverseSurfaceU` makes when it
// builds the far cap of a thickened solid, so the kernel flips a normal one
// way rather than two.
//
// Consequence: anything stored against this surface's u parameter — a trim
// loop's uv points, a split's fraction along u, a curve pinned to the surface
// by uv — is expressed in the old u after this runs. `remapReversedParam`
// below is the map those consumers need (u' = a + b - u).
//
// Moving a seam is split and rejoin, which is exact. A closed clamped curve on
// [a,b] with a new seam wanted at t: cut it into [t,b] and [a,t] and
// concatenate them in that order. Both cuts are `extractSubCurve` (knots.mjs,
// P&T A5.1 knot insertion to multiplicity degree+1, shape-preserving by
// construction) and the join is `joinCurvesC0` (knots.mjs), whose C0 joint is
// legitimate here because the two pieces are two halves of one curve and their
// shared endpoint is the same point evaluated twice.
//
// The point set is preserved exactly. The description is not: the old seam
// becomes an interior knot of multiplicity `degree`, so the curve is only
// C0-described there, though still geometrically smooth (knot insertion moves
// nothing). Moving a seam costs description size and continuity class, never
// shape. Rhino's SrfSeam/CrvSeam have the same property.
//
// A surface's seam moves by doing the same to every one of its curves in the
// closed direction. Every such curve carries the same knot vector and degree,
// so every one takes the same sequence of insertions and comes back with the
// same knot vector — which is what lets the result reassemble into a valid
// tensor-product net. It is the curve case run nv times through the
// net<->curves bookkeeping (`surfaceDirCurves`/`surfaceFromDirCurves`,
// surfaceknots.mjs) that surfaceInsertKnot uses.
//
// Refused with a reason:
//   * moving the seam of an open curve or an open surface direction. An open
//     shape has a start and an end that are different points; there is no
//     seam to move, and "reparametrize so it begins elsewhere" would tear it.
//   * a seam parameter at or outside the domain ends. At an end it is the
//     seam already there; outside it names no point on the shape.

import { curveDomain, curvePoint, isCurveClosed } from './curve.mjs';
import { extractSubCurve, joinCurvesC0, rescaleCurveDomain } from './knots.mjs';
import { surfaceClosure } from './surface.mjs';
import { surfaceDirCurves, surfaceFromDirCurves } from './surfaceknots.mjs';

// Flip

// Reverse one parametric direction of a surface: the control rows (or columns)
// in that direction are reversed and its knot vector re-based onto the same
// domain as k' = a + b - k, exactly the algebra reverseCurve uses on a curve.
// The surface occupies the same points; its normal points the other way.
// Self-inverse: a + b - (a + b - k) = k, and reversing a reversed list is the
// original list.
export function reverseSurfaceDirection(srf, dir) {
  if (dir !== 'u' && dir !== 'v') throw new Error(`reverseSurfaceDirection: dir must be 'u' or 'v', got '${dir}'`);
  if (!srf || !Array.isArray(srf.ctrlNet) || !srf.ctrlNet.length) throw new Error('reverseSurfaceDirection: needs a surface with a control net');
  const knots = dir === 'u' ? srf.knotsU : srf.knotsV;
  const a = knots[0], b = knots[knots.length - 1];
  const newKnots = knots.map((k) => a + b - k).reverse();
  const net = dir === 'u'
    ? srf.ctrlNet.slice().reverse().map((row) => row.map((cp) => cp.slice()))
    : srf.ctrlNet.map((row) => row.slice().reverse().map((cp) => cp.slice()));
  return dir === 'u'
    ? { ...srf, knotsU: newKnots, knotsV: srf.knotsV.slice(), ctrlNet: net }
    : { ...srf, knotsU: srf.knotsU.slice(), knotsV: newKnots, ctrlNet: net };
}

// The normal flip. The choice of which direction to reverse is made here,
// once, rather than at every call site (u, matching offset.mjs).
export function flipSurfaceNormals(srf) {
  return reverseSurfaceDirection(srf, 'u');
}

// The parameter map a reversed direction imposes on anything stored against
// it: trim loops, split fractions and on-surface curves.
export function remapReversedParam(t, knots) {
  const a = knots[0], b = knots[knots.length - 1];
  return a + b - t;
}

// Seam — closed curve

// Whether this curve has a seam to move at all; null if it does, otherwise the
// reason as user-facing text. Callers draw a seam control only when this is
// null.
export function curveSeamRefusal(crv, tol = 1e-6) {
  if (!crv || !Array.isArray(crv.ctrlPts) || crv.ctrlPts.length < 2) return 'this is not a curve with control points';
  if (!isCurveClosed(crv, tol)) return 'this curve is open — its start and end are different points, so there is no seam to move';
  return null;
}

// Move a closed curve's start/end point to the point at parameter `t`.
// The returned curve occupies the same points, in the same direction, and
// begins at crv(t). Its domain is the original one (the split-and-rejoin
// works on joinCurvesC0's own [0,2] convention and is rescaled back, an
// affine reparametrization that moves nothing).
export function moveCurveSeam(crv, t, opts = {}) {
  const tol = opts.tol ?? 1e-6;
  const refusal = curveSeamRefusal(crv, tol);
  if (refusal) throw new Error(`moveCurveSeam: ${refusal}`);
  const [min, max] = curveDomain(crv);
  const span = max - min;
  const eps = opts.paramEps ?? span * 1e-9;
  if (!(t > min + eps) || !(t < max - eps)) {
    throw new Error(`moveCurveSeam: the new seam parameter ${t} must lie strictly inside the curve's domain [${min}, ${max}] — an end IS the seam already there`);
  }
  const tail = extractSubCurve(crv, t, max);
  const head = extractSubCurve(crv, min, t);
  const joined = joinCurvesC0([tail, head]);
  return rescaleCurveDomain(joined, min, max);
}

// Move a closed curve's seam to the point on it closest to a picked 3-D point.
// The closest point is found by sampling (opts.samples, default 720); the
// seam move itself is exact.
export function moveCurveSeamToPoint(crv, point, opts = {}) {
  const tol = opts.tol ?? 1e-6;
  const refusal = curveSeamRefusal(crv, tol);
  if (refusal) throw new Error(`moveCurveSeamToPoint: ${refusal}`);
  const [min, max] = curveDomain(crv);
  const samples = opts.samples ?? 720;
  let bestU = min, bestD = Infinity;
  for (let i = 0; i <= samples; i++) {
    const u = min + (max - min) * (i / samples);
    const p = curvePoint(crv, u);
    const d = (p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2 + (p[2] - point[2]) ** 2;
    if (d < bestD) { bestD = d; bestU = u; }
  }
  const eps = (max - min) * 1e-6;
  const t = Math.min(max - eps, Math.max(min + eps, bestU));
  return moveCurveSeam(crv, t, opts);
}

// Seam — closed surface

// Which of a surface's two directions are closed, and therefore have a seam.
// A pass-through to surfaceClosure, renamed to { u, v }.
export function surfaceSeamDirections(srf, tol = 1e-6) {
  const c = surfaceClosure(srf, tol);
  return { u: c.closedU, v: c.closedV };
}

export function surfaceSeamRefusal(srf, dir, tol = 1e-6) {
  if (dir !== 'u' && dir !== 'v') return `'${dir}' is not a surface direction — it is u or v`;
  if (!srf || !Array.isArray(srf.ctrlNet) || !srf.ctrlNet.length) return 'this is not a surface with a control net';
  const closed = surfaceSeamDirections(srf, tol);
  if (!closed[dir]) return `this surface is open in ${dir} — that direction has two different edges, not a seam`;
  return null;
}

// Move a closed surface's seam in one direction to the parameter `t`.
// Runs moveCurveSeam over every curve in that direction. They all share a knot
// vector and a degree, so they all take the same insertions and come back
// sharing a knot vector again — which is what lets the net reassemble.
export function moveSurfaceSeam(srf, dir, t, opts = {}) {
  const tol = opts.tol ?? 1e-6;
  const refusal = surfaceSeamRefusal(srf, dir, tol);
  if (refusal) throw new Error(`moveSurfaceSeam: ${refusal}`);
  const knots = dir === 'u' ? srf.knotsU : srf.knotsV;
  const min = knots[0], max = knots[knots.length - 1];
  const eps = (max - min) * 1e-9;
  if (!(t > min + eps) || !(t < max - eps)) {
    throw new Error(`moveSurfaceSeam: the new seam parameter ${t} must lie strictly inside the ${dir} domain [${min}, ${max}] — an end IS the seam already there`);
  }
  /* The closure test is on the net; the seam move is on each curve. A
     degenerate curve in that direction (a sphere's pole row, every control
     point in the same place) must not be refused by a per-curve closure
     test, so the per-curve refusal is skipped here and the direction's
     closure, established above on the net, governs. Splitting and rejoining a
     constant curve returns a constant curve, which is correct for a pole. */
  const curves = surfaceDirCurves(srf, dir);
  const moved = curves.map((c) => {
    const tail = extractSubCurve(c, t, max);
    const head = extractSubCurve(c, min, t);
    return rescaleCurveDomain(joinCurvesC0([tail, head]), min, max);
  });
  return surfaceFromDirCurves(srf, dir, moved);
}
