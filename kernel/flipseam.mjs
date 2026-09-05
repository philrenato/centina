// FLIP AND SEAM — a shape's OWN parametrization, never its position in space.
//
// ================================================================
// THE TWO IDEAS, AND WHY THEY ARE ONE FILE
// ================================================================
// Both operations here change how a shape is DESCRIBED and leave the set of
// points it occupies exactly where it was. That is the whole of their
// contract, and it is what the tests assert: sample before, sample after,
// require the same point set. Nothing here is a mirror, a rotation or any
// other transform — kernel/transform.mjs owns those, and a reader who wants
// the shape to MOVE wants that file, not this one.
//
//   FLIP    reverse the direction a curve runs; reverse the direction a
//           surface's normal points. Rhino calls these `Flip` and `Dir`.
//   SEAM    move where a closed shape starts and ends — the start/end point
//           of a closed curve, the seam of a closed surface. Rhino calls
//           these `CrvSeam` and `SrfSeam`.
//
// ⚠ NOT TO BE CONFUSED WITH kernel/seam.mjs, which is a different thing under
// a word this app uses twice. That module CUTS a triangulation open along a
// seam so kernel/flatten.mjs's LSCM can accept a disk; it duplicates vertices
// and re-indexes faces and never touches a knot vector. This file moves the
// seam of a NURBS description and never touches a mesh. Neither calls the
// other and neither is a generalisation of the other.
//
// ================================================================
// FLIPPING A CURVE ALREADY EXISTS — kernel/curve.mjs's `reverseCurve`
// ================================================================
// It is exact (P&T 2.5), self-inverse and already used by extend/loft/sweep.
// This file deliberately does NOT wrap it in a second name. The one thing it
// adds for curves is the SEAM, below.
//
// ================================================================
// FLIPPING A SURFACE NORMAL — WHY IT NECESSARILY REPARAMETRIZES
// ================================================================
// A tensor-product surface's normal is N = Su x Sv. There is no way to negate
// it without changing the order of one of the two directions (or swapping the
// two, which negates it as well and swaps the domains along with it). So a
// normal flip on a NURBS surface is ALWAYS a reparametrization; a kernel that
// claims otherwise is carrying an orientation flag its tessellator reads, not
// flipping the surface. This file does the real thing and says so, and the
// direction it reverses is u — the same choice kernel/offset.mjs's own
// `reverseSurfaceU` already makes when it builds the far cap of a thickened
// solid, so the whole kernel flips a normal one way rather than two.
//
// ⚠ THE CONSEQUENCE, STATED RATHER THAN DISCOVERED LATER: anything stored
// against this surface's u parameter — a trim loop's uv points, a split's
// fraction along u, a curve pinned to the surface by uv — is expressed in the
// OLD u after this runs. `remapReversedParam` below is the one-line map those
// consumers need (u' = a + b - u), exported so nobody re-derives it.
//
// ================================================================
// MOVING A SEAM — SPLIT AND REJOIN, WHICH IS EXACT
// ================================================================
// A closed clamped curve on [a,b] with a new seam wanted at t: cut it into
// [t,b] and [a,t] and concatenate them in that order. Both cuts are
// `extractSubCurve` (knots.mjs, P&T A5.1 knot insertion to multiplicity
// degree+1, shape-preserving by construction) and the join is `joinCurvesC0`
// (knots.mjs), whose C0 joint is legitimate here because the two pieces are
// two halves of ONE curve and their shared endpoint is literally the same
// point evaluated twice.
//
// ⚠ WHAT IS AND IS NOT PRESERVED. The POINT SET is preserved exactly — that
// is the invariant, and it is asserted by sampling. The DESCRIPTION is not
// preserved: the old seam becomes an interior knot of multiplicity `degree`,
// so the curve is now only C0-DESCRIBED where it was smooth before, even
// though it is still geometrically smooth there (the control points arrange
// to keep it so — this is knot insertion, which moves nothing). The honest
// summary is that moving a seam costs description size and continuity CLASS,
// never shape. Rhino's own SrfSeam/CrvSeam have exactly this property.
//
// A SURFACE'S seam moves by doing precisely that to every one of its curves
// in the closed direction at once. Every such curve carries the same knot
// vector and the same degree, so every one of them takes the same sequence of
// insertions and comes back with the same knot vector — which is what makes
// the result reassemble into a valid tensor-product net rather than a ragged
// one. That is why this is not harder than the curve case: it is the curve
// case run nv times through the same net<->curves bookkeeping
// (`surfaceDirCurves`/`surfaceFromDirCurves`, surfaceknots.mjs) that
// surfaceInsertKnot already uses.
//
// REFUSED BY NAME, not silently ignored:
//   * moving the seam of an OPEN curve or an open surface direction. An open
//     shape has a start and an end that are DIFFERENT points; there is no
//     seam to move, and "reparametrize so it begins elsewhere" would tear it.
//     This is a permanent property of the geometry, not an unbuilt case.
//   * a seam parameter at or outside the domain ends. At an end it is the
//     seam already there; outside it names no point on the shape.

import { curveDomain, curvePoint, isCurveClosed } from './curve.mjs';
import { extractSubCurve, joinCurvesC0, rescaleCurveDomain } from './knots.mjs';
import { surfaceClosure } from './surface.mjs';
import { surfaceDirCurves, surfaceFromDirCurves } from './surfaceknots.mjs';

// ============================================================================
// FLIP
// ============================================================================

// Reverse ONE parametric direction of a surface: the control rows (or columns)
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

// THE NORMAL FLIP. One name for the operation a caller actually wants, so the
// choice of which direction to reverse is made ONCE, here, rather than at
// every call site (see the header: u, matching offset.mjs).
export function flipSurfaceNormals(srf) {
  return reverseSurfaceDirection(srf, 'u');
}

// The parameter map a reversed direction imposes on anything stored against
// it. Exported because trim loops, split fractions and on-surface curves all
// need the same three-term expression and it must not be written four times.
export function remapReversedParam(t, knots) {
  const a = knots[0], b = knots[knots.length - 1];
  return a + b - t;
}

// ============================================================================
// SEAM — CLOSED CURVE
// ============================================================================

// Whether this curve has a seam to move at all, and if not, why not — in the
// words a person should read. Callers draw a seam control only when this says
// ok, so no dead affordance is ever rendered.
export function curveSeamRefusal(crv, tol = 1e-6) {
  if (!crv || !Array.isArray(crv.ctrlPts) || crv.ctrlPts.length < 2) return 'this is not a curve with control points';
  if (!isCurveClosed(crv, tol)) return 'this curve is open — its start and end are different points, so there is no seam to move';
  return null;
}

// Move a CLOSED curve's start/end point to the point at parameter `t`.
// The returned curve occupies the same points, in the same direction, and
// begins at crv(t). Its domain is the original one (the split-and-rejoin
// works on joinCurvesC0's own [0,2] convention and is rescaled back, an
// affine reparametrization that moves nothing) so a caller that stored a
// domain alongside the curve is not silently handed a different one.
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

// Move a closed curve's seam to the point on it CLOSEST to a picked 3-D point.
// Separated from moveCurveSeam so the parameter form stays usable on its own
// and so the closest-point search (which is iterative) is not buried inside an
// operation that is otherwise exact.
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

// ============================================================================
// SEAM — CLOSED SURFACE
// ============================================================================

// Which of a surface's two directions are closed, and therefore have a seam.
// A thin, named pass-through to surfaceClosure so a caller asking about SEAMS
// does not have to know that the closure test is spelled closedU/closedV.
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
  /* ⚠ THE CLOSURE TEST IS ON THE NET, THE SEAM MOVE IS ON EACH CURVE. A
     degenerate curve in that direction (a sphere's pole row, every control
     point the same place) is closed in the net sense but `isCurveClosed`'s
     tolerance is the only thing standing between it and a refusal, so the
     per-curve refusal is turned off here and the DIRECTION's closure — already
     established above, on the net the caller actually has — is what governs.
     Splitting and rejoining a constant curve returns a constant curve, which
     is correct and is what a pole should do. */
  const curves = surfaceDirCurves(srf, dir);
  const moved = curves.map((c) => {
    const tail = extractSubCurve(c, t, max);
    const head = extractSubCurve(c, min, t);
    return rescaleCurveDomain(joinCurvesC0([tail, head]), min, max);
  });
  return surfaceFromDirCurves(srf, dir, moved);
}
