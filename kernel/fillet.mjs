/* Rolling-ball edge blending — the cross-section, and the conditions under
   which it exists at all.

   Prior art (read for technique, not transcribed):

     · Rossignac, J. R. & Requicha, A. A. G., "Constant-Radius Blending in
       Solid Modeling." ASME Computers in Mechanical Engineering (CIME) 3,
       pp. 65-73, 1984. The rolling-sphere formulation itself.
     · Choi, B. K. & Ju, S. Y., "Constant-radius blending in surface
       modeling." Computer-Aided Design 21(4), pp. 213-220, 1989.
       DOI 10.1016/0010-4485(89)90046-8. The feasibility precondition, from
       its own abstract: two surfaces can be blended at radius r as long as
       their offset surfaces at distance r are smooth, so that the
       intersection between those offsets is well defined. That intersection
       is the spine.
     · Peternell, M. & Pottmann, H., "Computing Rational Parametrizations of
       Canal Surfaces." Journal of Symbolic Computation 23(2-3), pp. 255-266,
       1997. DOI 10.1006/jsco.1996.0087. The variable-radius existence
       condition used in `variableRadiusFeasible` below.

   This module does not compute the maximum buildable radius.
   The true bound is the smaller principal radius of curvature on the concave
   side (Wallner, Sakkalis, Maekawa, Pottmann & Yu, "Self-Intersections of
   Offset Curves and Surfaces", Int. J. Shape Modeling 7(1), 2001), which needs second derivatives of the supporting
   surfaces; this kernel computes first partials only. Anything this module
   returns about size is a necessary condition, never a sufficient one, and it
   says so in the field names.

   Plain data throughout: a point is [x, y, z], and nothing here imports a
   vector library or reaches the DOM. */

import { findSpan, basisFuns } from './basis.mjs';
import { solveLinearSystem, averagingKnotVector } from './interpolate.mjs';

const EPS = 1e-12;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
function norm(a) {
  const L = len(a);
  if (!(L > EPS)) return null; // a direction that does not exist is null, never [0,0,0] pretending to be one
  return [a[0] / L, a[1] / L, a[2] / L];
}

/**
 * The cross-section of a rolling-ball blend at one point along an edge.
 *
 * The setback is governed by the angle between the co-normals, not by theta;
 * written against theta it would be negative on every concave edge.
 *
 *     phi = acos(coNormalA . coNormalB)          always in [0, pi]
 *     setback d     = r / tan(phi / 2)
 *     center offset = r / sin(phi / 2)
 *
 * theta is the angle measured through the material and runs to 2*pi; phi is the
 * angle between the two directions the faces extend in. They agree on
 * a convex edge and are supplementary about 2*pi on a concave one, so
 * tan(theta/2) flips sign at theta = pi while the geometry does not.
 *
 * Worked both ways:
 *   · Box edge, theta = phi = 90 degrees, ball of radius r in a right-angled
 *     corner at the origin with cA = +y and cB = +x. Tangency at (0, r) and
 *     (r, 0), center at (r, r). So d = r and the center sits r*sqrt(2) along the
 *     45-degree bisector. r/tan(45) = r and r/sin(45) = r*sqrt(2). Agree.
 *   · Concave edge of an L-shaped solid, theta = 270 degrees, cA = +x and
 *     cB = +y so phi is still 90. The ball rounds the inside corner: tangency
 *     at (edge + r*cA) and (edge + r*cB), center at edge + (r, r). The setback
 *     is +r, exactly as in the convex case. Against theta it would have been
 *     r/tan(135) = -r, putting both tangency points on the wrong side of the
 *     edge and the blend inside the solid.
 *
 * theta is still carried, and still needed: it is what says whether the blend
 * removes material or adds it, which decides which side gets trimmed.
 *
 * The setback is why theta has to be carried along the edge rather than
 * sampled once. Where theta varies — every S-shaped or blob-derived edge — d
 * varies with it, so the two tangency curves are not constant-distance offsets
 * of the edge, and a blend built by offsetting the edge by a fixed distance is
 * wrong everywhere the angle moves.
 *
 * Returns null, with a reason, rather than a degenerate section:
 *   · theta at or below `minAngle`: the faces are nearly tangent, the setback
 *     diverges (d -> infinity as theta -> 0) and there is nothing for a ball to
 *     sit in.
 *   · theta at or above 2*pi - minAngle: the same degeneracy from the reflex side.
 *   · co-normals parallel, so no bisector plane exists.
 */
export function rollingBallSection({ point, coNormalA, coNormalB, theta, radius, minAngleRad = 0.5 * Math.PI / 180 }) {
  if (!(radius > 0)) return { ok: false, reason: 'radius must be positive', radius };
  // No separate guard on a degenerate angle value. Every way theta can be a
  // degenerate angle shows up in phi and is caught below, with a message that
  // says what happened: theta near 0 is a knife edge (phi near 0), theta near
  // pi is a smooth junction (phi near pi), theta near 2*pi is a thin slot (phi
  // near 0 again). A second guard on theta would label a knife edge "too
  // nearly tangent", which is the opposite condition.
  //
  // Theta being absent or in the wrong units is not an angle degeneracy, and
  // phi cannot see it: phi is derived from the co-normals alone. `convex` is
  // read from theta and from nothing else, so an omitted theta, a NaN, or a
  // value in degrees would classify every convex edge as concave, telling the
  // caller to add material where it must remove it. So this is a refusal
  // rather than a default.
  if (!Number.isFinite(theta)) return { ok: false, reason: 'the dihedral angle theta is missing or not a number, and the convex/concave decision is read from it' };
  if (!(theta > 0 && theta < 2 * Math.PI)) return { ok: false, reason: `the dihedral angle theta must be in radians within (0, 2*pi); got ${theta}` };
  const cA = norm(coNormalA), cB = norm(coNormalB);
  if (!cA || !cB) return { ok: false, reason: 'a co-normal has no direction' };
  const phi = Math.acos(Math.max(-1, Math.min(1, dot(cA, cB))));
  if (!(phi > minAngleRad)) return { ok: false, reason: 'the faces fold back on each other — a knife edge holds no ball', phi };
  if (!(phi < Math.PI - minAngleRad)) return { ok: false, reason: 'the faces are tangent here — there is no edge to blend', phi };
  const half = phi / 2;
  const tanHalf = Math.tan(half), sinHalf = Math.sin(half);
  if (!(Math.abs(tanHalf) > EPS) || !(Math.abs(sinHalf) > EPS)) return { ok: false, reason: 'degenerate half-angle' };
  const setback = radius / tanHalf;
  const centreOffset = radius / sinHalf;
  const bisector = norm(add(cA, cB));
  if (!bisector) return { ok: false, reason: 'co-normals are opposed — the faces double back and there is no bisector' };
  return {
    ok: true,
    phi, theta,
    setback, centreOffset,
    // The two points where the ball touches, one on each face's tangent plane.
    // A curved face needs these projected onto the surface afterwards; on a
    // planar face they are already exact.
    tangencyA: add(point, mul(cA, setback)),
    tangencyB: add(point, mul(cB, setback)),
    centre: add(point, mul(bisector, centreOffset)),
    bisector,
    // Convex blends remove material, concave blends add it. The caller needs
    // this to know which side to trim, and it falls straight out of theta.
    convex: theta < Math.PI,
  };
}

/**
 * The necessary condition on radius that is checkable here.
 *
 * Choi & Ju's precondition is about the offset surfaces, and the part of it
 * reachable without second derivatives is this: the setback must not exceed
 * how much face there is to set back into. A ball whose tangency point would
 * land beyond the far side of a face has rolled off it — the "ball falls off
 * the surface rails" case, the most common blend failure and one a local
 * algorithm cannot see.
 *
 * `widthA`/`widthB` are how far each face extends from the edge, measured
 * along its own co-normal. Returns the largest radius that still fits, so a
 * refusal can name a number instead of a class of problem.
 */
export function maxRadiusForSetback({ phi, widthA, widthB }) {
  // phi comes from an acos and is therefore in [0, pi]; anything else is a
  // caller error, which `Math.abs(tan(phi/2))` would turn into a plausible
  // answer.
  if (!(phi > 0) || !(phi < Math.PI)) return { ok: false, reason: `phi must lie strictly between 0 and pi — got ${phi}`, phi };
  const tanHalf = Math.tan(phi / 2);
  if (!(Math.abs(tanHalf) > EPS)) return { ok: false, reason: 'degenerate half-angle', phi };
  const w = Math.min(widthA, widthB);
  if (!(w > 0)) return { ok: false, reason: 'a supporting face has no width', widthA, widthB };
  // d = r / tan(phi/2) <= w  =>  r <= w * tan(phi/2)
  const rMax = w * Math.abs(tanHalf);
  return {
    ok: true,
    rMax,
    // This bounds the radius by how much face there is,
    // not by curvature. A face wide enough to hold the setback can still be too
    // curved to hold the ball, and that bound needs second derivatives this
    // kernel does not compute.
    bound: 'setback-only',
    limitedBy: widthA <= widthB ? 'A' : 'B',
  };
}

/**
 * Variable radius — when the envelope exists at all.
 *
 * Peternell & Pottmann 1997: a canal surface swept by a ball of varying radius
 * r(t) along a spine m(t) has a real envelope exactly if
 *
 *     |m'(t)|^2 - r'(t)^2  >=  0
 *
 * — if the radius changes faster than the spine advances, there is no surface
 * to build, and at equality the characteristic circle degenerates to a point.
 * Unlike the curvature bound this is computable from first derivatives, so
 * this kernel can refuse on it.
 *
 * `spine` is a polyline of ball centers, `radii` the radius at each. Returns
 * the worst margin and where it occurs, so a refusal can point at the span that
 * is asking too much rather than at the whole edge.
 */
export function variableRadiusFeasible(spine, radii) {
  if (!Array.isArray(spine) || !Array.isArray(radii) || spine.length < 2 || spine.length !== radii.length) {
    // `radii` is checked before its length is read, so a missing array
    // returns {ok, reason} rather than throwing a TypeError.
    return { ok: false, reason: 'spine and radii must be matching arrays of at least two samples' };
  }
  let worst = Infinity, worstAt = -1;
  for (let i = 0; i + 1 < spine.length; i++) {
    const ds = len(sub(spine[i + 1], spine[i]));
    const dr = radii[i + 1] - radii[i];
    // Both sides are per-step, so the parameterization cancels and no
    // derivative estimate is needed beyond the differences themselves.
    const margin = ds * ds - dr * dr;
    if (margin < worst) { worst = margin; worstAt = i; }
  }
  return {
    ok: worst >= 0,
    worstMargin: worst,
    worstAt,
    reason: worst >= 0 ? null : 'the radius changes faster than the spine advances, so no envelope exists over that span',
  };
}

/**
 * The arc a rolling ball cuts at one cross-section, as an exact rational
 * quadratic Bezier — the standard conic form, with the middle weight cos of the
 * half sweep. Exact for any sweep below 180 degrees, which every blend
 * cross-section is: the sweep is pi - theta for a convex edge and theta - pi
 * for a concave one, both strictly under pi for any theta a ball can sit in.
 */
export function sectionArc(section) {
  if (!section || !section.ok) return null;
  const { centre, tangencyA, tangencyB } = section;
  const rA = sub(tangencyA, centre), rB = sub(tangencyB, centre);
  const uA = norm(rA), uB = norm(rB);
  if (!uA || !uB) return null;
  const cosSweep = Math.max(-1, Math.min(1, dot(uA, uB)));
  const sweep = Math.acos(cosSweep);
  if (!(sweep > EPS)) return null;
  const w = Math.cos(sweep / 2);
  if (!(w > EPS)) return null; // a half-sweep at or past 90 degrees is not representable as one rational quadratic
  // The middle control point is where the two end tangents meet, which for a
  // circular arc lies on the bisector at radius / cos(halfSweep).
  const bis = norm(add(uA, uB));
  if (!bis) return null;
  const R = len(rA);
  const mid = add(centre, mul(bis, R / w));
  return {
    degree: 2,
    knots: [0, 0, 0, 1, 1, 1],
    /* Cartesian coordinates with the weight appended, not premultiplied.
       This project's evaluators do the premultiplication themselves —
       `curvePoint`/`surfacePoint` read [x, y, z, w] and accumulate x*w — and
       `makeArc` stores `[...point, weight]` to match. Premultiplying here
       would apply the weight twice and displace the middle control point by a
       factor of w.

       A perpendicular-distance check against a spine parallel to z cannot
       see that displacement, since it runs along the spine: at z = 0 the two
       forms are numerically identical. */
    ctrlPts: [
      [tangencyA[0], tangencyA[1], tangencyA[2], 1],
      [mid[0], mid[1], mid[2], w],
      [tangencyB[0], tangencyB[1], tangencyB[2], 1],
    ],
    sweep,
    radius: R,
  };
}

/**
 * Skin a run of section arcs into the blend surface.
 *
 * Every section produced by `sectionArc` is a rational quadratic on the same
 * degree and the same knot vector, which is what makes this exact rather than a
 * fit: the sections' control points are the surface's control net in U, so the
 * blend's cross-section stays a true circular arc everywhere instead of a
 * polynomial approximation of one. The general `loft` in kernel/loft.mjs
 * resamples each section onto a shared parameterization, which is the right
 * thing for arbitrary hand-picked curves and the wrong thing here — it would
 * discard the exact conic form of the rolling-ball section.
 *
 * This is a sliding-disc construction, not the exact rolling-ball envelope,
 * and the two differ wherever the dihedral varies. The sections are laid
 * in planes perpendicular to the edge; the true envelope's characteristic circle
 * lies perpendicular to the spine. Those coincide only while the spine runs
 * parallel to the edge, which it does exactly when the dihedral angle is
 * constant — as theta varies, the center moves within the cross-section plane
 * too, and the spine tilts away.
 *
 * What this construction does guarantee, exactly and at every section:
 *   · the cross-section is a true circular arc of the requested radius;
 *   · it meets each supporting face tangentially, because the arc's radius at
 *     each tangency point runs along that face's normal by construction;
 *   · the tangency curves lie in the faces, which is what the supports must be
 *     trimmed back to.
 * What it does not guarantee is that every point is exactly `radius` from the
 * spine between sections. On a straight edge of constant angle that holds to
 * machine precision; across a varying dihedral the departure falls at fourth
 * order in the section spacing with cubic interpolation in V, which is why
 * `blendRadiusDeviation` measures what was achieved rather than assuming it.
 *
 * U is the cross-section (degree 2, rational, 3 control points).
 * V runs along the edge, degree `degV` clamped to what the section count
 * supports, with a chord-length knot vector so an unevenly sampled edge is not
 * reparameterized.
 *
 * V interpolates the sections: the surface passes through every section and
 * is approximate between them. On a straight edge of constant dihedral every
 * section is a translate of its neighbors, so the result is a true
 * cylindrical patch; on a curved edge or a varying angle the error between
 * sections falls with section count and is what `blendRadiusDeviation`
 * measures.
 */
export function blendSurfaceFromSections(arcs, degV = 3) {
  if (!Array.isArray(arcs) || arcs.length < 2) return { ok: false, reason: 'a blend needs at least two sections' };
  /* One shape of section per surface, whatever that shape is: a rational
     quadratic for a rolling ball or a chamfer's chord, a quintic for a
     curvature-continuous section. What has to hold is that the sections
     agree — they are interpolated control point by control point, so a
     surface cannot be made from sections with different degrees or different
     counts of them. */
  const first = arcs[0];
  if (!first || !Array.isArray(first.ctrlPts) || first.ctrlPts.length < 3) {
    return { ok: false, reason: 'a section needs at least three control points' };
  }
  const secDeg = first.degree, secPts = first.ctrlPts.length;
  if (arcs.some((a) => !a || a.degree !== secDeg || a.ctrlPts.length !== secPts)) {
    return { ok: false, reason: `every section must have the same degree and control point count — this run mixes them (expected degree ${secDeg} with ${secPts} points)` };
  }
  const n = arcs.length;
  const dV = Math.min(degV, n - 1);
  // Chord length along the edge, from the sections' own midpoints, so a run
  // sampled densely at one end and sparsely at the other keeps its shape.
  /* No division by the weight. Control points here are stored cartesian with
     the weight appended — `sectionArc` says so and `surfacePoint` premultiplies
     for itself — so dividing by w would deproject something that was never
     projected and displace every arc's midpoint radially by 1/w. With varying
     weights, that can make distinct sections' midpoints coincide and produce
     duplicate parameters that `solveLinearSystem` cannot solve. */
  // A consistent interior point per section, used only to space the sections
  // along the edge by chord length; which one it is matters less than that it
  // is the same one on every section.
  const midIdx = secPts >> 1;
  const mid = (a) => [a.ctrlPts[midIdx][0], a.ctrlPts[midIdx][1], a.ctrlPts[midIdx][2]];
  const chords = [0];
  for (let j = 1; j < n; j++) chords.push(chords[j - 1] + len(sub(mid(arcs[j]), mid(arcs[j - 1]))));
  const total = chords[n - 1];
  if (!(total > EPS)) return { ok: false, reason: 'the sections are all at the same place — there is no edge to run along' };
  const t = chords.map((c) => c / total);
  // Averaged knots (Piegl & Tiller's own recipe for a clamped vector matching a
  // parameter set), so the basis is well conditioned rather than uniform over
  // an uneven set.
  const knotsV = averagingKnotVector(t, dV);
  /* Interpolated in V, not approximated. Taking the section control points as
     the net makes a degree-3 surface that does not pass through the sections
     between the ends, and it is less accurate than degree 1 through the same
     sections (0.094% against 0.011% at 97 sections). Solving for control
     points that make the surface pass through every section gets both —
     smooth in V and exact at each section.

     Solved in homogeneous coordinates (x*w, y*w, z*w, w) and projected back, so
     the weights are interpolated as part of the geometry rather than separately;
     interpolating cartesian points and weights independently does not reproduce
     the rational curve. kernel/interpolate.mjs's own `interpAtParams` is 3-D and
     forces w = 1, which is why the system is assembled here. */
  const knotsForSolve = averagingKnotVector(t, dV);
  const A = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let k = 0; k < n; k++) {
    const span = findSpan(n - 1, dV, t[k], knotsForSolve);
    const N = basisFuns(span, t[k], dV, knotsForSolve);
    for (let i = 0; i <= dV; i++) A[k][span - dV + i] = N[i];
  }
  const ctrlNet = [];
  for (let i = 0; i < secPts; i++) {
    const w = arcs.map((a2) => a2.ctrlPts[i][3]);
    const rhs = [0, 1, 2].map((c) => arcs.map((a2, j) => a2.ctrlPts[i][c] * w[j]));
    rhs.push(w.slice());
    const sol = solveLinearSystem(A.map((r) => r.slice()), rhs);
    const row = [];
    for (let j = 0; j < n; j++) {
      const wj = sol[3][j];
      /* Positive, not merely non-zero. Data weights are cos(halfSweep) and lie
         in (0,1], but the interpolant is not bounded by its data: sections whose
         sweeps alternate sharply overshoot to control weights like
         [0.940, -1.516, 3.399, -1.523, 0.940]. An |w| > EPS test would accept
         those, while this project's own `isFiniteNet` rejects the net for
         w <= 0. A negative control weight also
         destroys the convex-hull and variation-diminishing properties the rest
         of the evaluation assumes. */
      if (!(wj > EPS)) return { ok: false, reason: `the weight interpolation overshot to ${wj.toFixed(4)} — sections whose sweep changes too sharply between neighbors cannot be interpolated at this degree; add sections or lower degV` };
      row.push([sol[0][j] / wj, sol[1][j] / wj, sol[2][j] / wj, wj]);
    }
    ctrlNet.push(row);
  }
  return {
    ok: true,
    srf: { degU: secDeg, degV: dV, knotsU: (first.knots || [0, 0, 0, 1, 1, 1]).slice(), knotsV, ctrlNet },
    sections: n,
    // The two tangency curves, which are the surface's own U=0 and U=1 borders
    // and the curves the supporting faces have to be trimmed back to.
    tangencyCurveA: { degree: dV, knots: knotsV.slice(), ctrlPts: ctrlNet[0].map((p) => p.slice()) },
    tangencyCurveB: { degree: dV, knots: knotsV.slice(), ctrlPts: ctrlNet[secPts - 1].map((p) => p.slice()) },
  };
}

/**
 * How far the built surface strays from the ball that defined it.
 *
 * The defining property of a constant-radius blend is that every point on it is
 * exactly `radius` from the spine — the locus of ball centers.
 *
 * Measured to the nearest point on the spine, not to the spine "at the same
 * parameter". Only the first is the definition. The surface carries a
 * chord-length knot vector, so its v and a spine sampled evenly by index do
 * not name the same place along the edge, and comparing them reports a large
 * error on an accurate blend. The parameterization is free; what must hold
 * is the distance.
 *
 * `spine` is a polyline of ball centers. `evalSrf(srf, u, v)` is the caller's
 * own surface evaluator, kept as a parameter so this module stays independent
 * of the rest of the kernel. `vFrom`/`vTo` trim the sampled span, because the
 * nearest point on a finite polyline clamps at its ends and would report an end
 * effect as a surface defect.
 */
function distToPolyline(p, poly) {
  let best = Infinity;
  for (let i = 0; i + 1 < poly.length; i++) {
    const a = poly[i], b = poly[i + 1];
    const ab = sub(b, a);
    const L2 = dot(ab, ab);
    let t = L2 > EPS ? dot(sub(p, a), ab) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const d = len(sub(p, add(a, mul(ab, t))));
    if (d < best) best = d;
  }
  return best;
}
export function blendRadiusDeviation(srf, spine, radius, evalSrf, uSteps = 9, vSteps = 17, vFrom = 0.05, vTo = 0.95) {
  if (!Array.isArray(spine) || spine.length < 2) return { worst: Infinity, worstAt: null, reason: 'a spine needs at least two centers' };
  /* The spine polyline is the ruler, and its own error is second order.

     Distance is measured to a polyline through the ball centers, and a polyline
     cuts the chord off a curving spine by roughly sagitta = L^2 / (8*rho) per
     segment. Sample the spine at the same density as the sections and that
     chord error dominates: every surface, at every V degree, measures O(h^2) —
     the ruler's convergence reported as the surface's. With a spine sampled
     well below the section spacing, the observed order of cubic interpolation
     is about 4.

     `spineSagitta` is returned so a caller can see when it is instrument-bound:
     if it is not far below `worst`, the number is about the ruler. */
  let sag = 0;
  for (let i = 1; i + 1 < spine.length; i++) {
    const a0 = spine[i - 1], b0 = spine[i], c0 = spine[i + 1];
    const mid = [(a0[0] + c0[0]) / 2, (a0[1] + c0[1]) / 2, (a0[2] + c0[2]) / 2];
    const d = len(sub(b0, mid));
    if (d > sag) sag = d;
  }
  let worst = 0, worstAt = null;
  for (let j = 0; j < vSteps; j++) {
    const v = vFrom + (vTo - vFrom) * (vSteps > 1 ? j / (vSteps - 1) : 0);
    for (let i = 0; i < uSteps; i++) {
      const u = i / (uSteps - 1);
      const p = evalSrf(srf, u, v);
      const d = Math.abs(distToPolyline(p, spine) - radius);
      if (d > worst) { worst = d; worstAt = { u, v }; }
    }
  }
  return { worst, worstAt, spineSagitta: sag, instrumentBound: sag > worst * 0.25 };
}

/**
 * The section, built from the face normals rather than from a setback.
 *
 * `rollingBallSection` locates the tangency points by measuring a setback along
 * each co-normal from the edge. That is correct, and it puts the section in the
 * plane perpendicular to the edge — which is the sliding-disc construction, and
 * departs from the true envelope wherever the dihedral varies.
 *
 * This does the same job the other way round and gets the envelope directly. A
 * sphere of radius r centered at m is tangent to a face with outward unit normal
 * n exactly where it touches, and that point is
 *
 *     tangency = m + r * n
 *
 * (outward, so the touch point is on the far side of the center from the
 * material — checked on a box: material in x>0, y>0, center (r, r), face x = 0
 * whose outward normal is -x, giving (r,r) + r*(-1,0) = (0, r), which is where
 * the ball touches).
 *
 * The two tangency points and the center then span the section plane, and that
 * plane is the characteristic one: the envelope condition for a constant-radius
 * canal surface is (p - m) . m' = 0, so the contact points lie in the plane
 * through m perpendicular to the spine tangent. Building the section from the
 * normals therefore lands in the right plane without ever computing m' — the
 * derivative is implied by the geometry rather than estimated from samples.
 *
 * No setback appears here: the setback is a way of finding the tangency from
 * the edge, and finding it from the normal instead removes the approximation
 * that makes the disc slide.
 */
export function envelopeSection({ centre, radius, toTouchA, toTouchB, normalA, normalB }) {
  if (!(radius > 0)) return { ok: false, reason: 'radius must be positive' };
  /* These point from the center toward the touch point, and they are not
     "the face's outward normal" in general — the parameters are named for the
     direction rather than for the normal because a radius check cannot tell
     the two apart.

     On a convex edge the ball sits inside the material, so center-to-touch and
     the outward normal coincide and either name works. On a concave edge the
     ball sits in the void and they are opposite, so passing an outward normal
     puts the tangency point 2r away on the far side of the ball. It is still
     exactly `radius` from the center, so a check that only measures the radius
     reports a perfect blend that touches neither face.

     `normalA`/`normalB` are accepted as alternate names; `toTouchA`/`toTouchB`
     are the contract. */
  const nA = norm(toTouchA || normalA), nB = norm(toTouchB || normalB);
  if (!nA || !nB) return { ok: false, reason: 'a face normal has no direction' };
  const cosBetween = Math.max(-1, Math.min(1, dot(nA, nB)));
  // Normals parallel means the two faces are the same plane — no edge; opposed
  // means they double back on each other and no ball sits between them.
  if (cosBetween > 1 - 1e-12) return { ok: false, reason: 'the faces are tangent here — there is no edge to blend' };
  if (cosBetween < -1 + 1e-12) return { ok: false, reason: 'the faces are opposed — a ball cannot touch both' };
  const tangencyA = add(centre, mul(nA, radius));
  const tangencyB = add(centre, mul(nB, radius));
  // The spine tangent, implied rather than estimated: it is perpendicular to
  // the plane the two normals span, which is exactly the characteristic plane.
  const spineTangent = norm(cross(nA, nB));
  if (!spineTangent) return { ok: false, reason: 'the normals span no plane' };
  return { ok: true, centre, radius, tangencyA, tangencyB, spineTangent, sweep: Math.acos(cosBetween) };
}

/**
 * The same rational-quadratic arc as `sectionArc`, from an envelope section.
 * Kept separate rather than overloaded because the two carry different fields,
 * and accepting either would let one be used where the other was meant.
 */
export function envelopeSectionArc(section) {
  if (!section || !section.ok) return null;
  return sectionArc({ ok: true, centre: section.centre, tangencyA: section.tangencyA, tangencyB: section.tangencyB });
}

/**
 * Build to a tolerance, and report what was achieved.
 *
 * The departure of a skinned blend from the true tube falls at fourth order in
 * the section spacing, with cubic interpolation in V. Density is the control,
 * set by aiming, measuring, and correcting rather than by picking a number.
 *
 * From err ~ C*h^4 with h ~ 1/N, one measurement predicts the count that would
 * meet the target: N_needed = N * (err / target)^(1/4). That prediction is then
 * verified by measuring again — the law aims and never certifies. A
 * blend that cannot reach the target inside `maxRounds` returns its best effort
 * with the deviation it achieved, so a caller can say "accurate to 0.004mm"
 * instead of implying an exactness it does not have.
 *
 * `sectionAt(t)` for t in [0,1] returns { centre, radius, normalA, normalB } —
 * the caller owns the geometry, this owns the density.
 *
 * Two hooks. The aim-measure-correct loop is not specific to a constant
 * radius, but everything it does inside the loop is:
 *   · `opts.sectionArcFor(spec)` -> { ok, arc } | { ok: false, reason } replaces
 *     the envelope section builder, for a section that is not a great circle of
 *     the ball — a variable-radius contact circle is offset along the spine and
 *     shrunk (kernel/varradius.mjs).
 *   · `opts.measure(srf, n)` -> { worst, instrumentBound, floor } replaces the
 *     deviation measure, and supplying it lifts the one-radius refusal
 *     below. The refusal stays the default: a caller who did not mean
 *     to vary the radius must still be told, rather than handed a number
 *     measured against whichever radius came last.
 */
export function blendSurfaceToTolerance(sectionAt, tolerance, opts = {}) {
  const startSections = opts.startSections || 17;
  const maxSections = opts.maxSections || 401;
  const maxRounds = opts.maxRounds || 4;
  const degV = opts.degV != null ? opts.degV : 3;
  const evalSrf = opts.evalSrf;
  if (typeof evalSrf !== 'function') return { ok: false, reason: 'an evaluator is required to measure what was built' };
  const measure = typeof opts.measure === 'function' ? opts.measure : null;
  /* A second deviation the loop must also drive down. `measure` replaces how the
     blend judges itself — a chamfer swaps radius error for flatness error. This
     one is additive and orthogonal: a caller that knows something the builder
     cannot see, typically how far the blend's borders have drifted off the faces
     they must stay tangent to, contributes it here and refinement answers to
     both. Returning a plain number keeps it distinct from `measure`'s record. */
  const also = typeof opts.alsoMeasure === 'function' ? opts.alsoMeasure : null;
  const fold = (r) => {
    if (!also || !r || r.failed || !r.built) return r;
    const x = also(r.built.srf, r.n);
    if (!Number.isFinite(x)) return { failed: 'the supplementary deviation measure returned no number' };
    return x > r.dev ? { ...r, dev: x, limitedBy: 'supports' } : r;
  };
  const arcFor = typeof opts.sectionArcFor === 'function' ? opts.sectionArcFor : (spec) => {
    const e = envelopeSection(spec);
    if (!e.ok) return { ok: false, reason: e.reason };
    const arc = envelopeSectionArc(e);
    return arc ? { ok: true, arc } : { ok: false, reason: 'a section produced no arc' };
  };
  /* A closed run is sampled on a cycle, not on an interval.
     `i / (n - 1)` walks 0 to 1 inclusive, which is right for an edge that has
     two ends and wrong for one that has none: on a loop, t=0 and t=1 name the
     same station, so the run either duplicates a section there or leaves the
     wrap unsampled — and the band cannot close, leaving a gap at the seam that
     filler faces would have to bridge.
     On a cycle the stations are `i / n`, so no station is repeated, and the
     first section is appended again as the last so the skin closes on itself
     exactly rather than nearly. `closed` is the caller's to declare: only the
     caller knows whether the edge it is walking comes back to where it began. */
  const closedRun = !!opts.closed;
  const attempt = (n) => {
    const arcs = [], centres = [];
    let radius = null;
    for (let i = 0; i < n; i++) {
      const spec = sectionAt(closedRun ? i / n : i / (n - 1));
      if (!spec) return null;
      const made = arcFor(spec);
      if (!made || !made.ok) return { failed: (made && made.reason) || 'a section produced no arc' };
      arcs.push(made.arc); centres.push(spec.centre);
      /* Constant radius only unless a measure was supplied, and it refuses
         rather than averaging. Keeping the last section's radius and measuring
         the whole surface against it would report a meaningless deviation for a
         variable-radius run. The default deviation measure's premise is that
         one radius describes
         the surface, so varying it is caller error until the caller has replaced
         that measure with one that can express a varying radius. */
      if (measure) continue;
      if (radius == null) radius = spec.radius;
      else if (Math.abs(spec.radius - radius) > 1e-12) {
        return { failed: `this builder measures against ONE radius and the sections vary (${radius} then ${spec.radius}) — a variable-radius blend needs its own deviation measure` };
      }
    }
    /* The wrap section, on a closed run only: the first station again, so the
       skin's last column is its first and the band closes exactly rather than
       within a tolerance. Appended rather than sampled, because sampling t=1
       would re-solve the same ball and could land slightly off it. */
    if (closedRun && arcs.length) { arcs.push(arcs[0]); centres.push(centres[0]); }
    const built = blendSurfaceFromSections(arcs, degV);
    if (!built.ok) return { failed: built.reason };
    if (measure) {
      const m = measure(built.srf, n);
      /* A measure that could not measure says so, and its reason is the one
         reported. Flattening every such case to "returned no number"
         loses the distinction between a measure that is absent and a surface
         that could not be sampled at all. */
      if (!m || !Number.isFinite(m.worst)) return { failed: (m && m.reason) || 'the supplied deviation measure returned no number' };
      return fold({ built, dev: m.worst, n, instrumentBound: !!m.instrumentBound, measureFloor: m.floor, deviationSigned: m.signed });
    }
    /* Measured against a spine sampled far finer than the sections. Using the
       section centers themselves makes the ruler's own chord error the thing
       being reported — see blendRadiusDeviation. A 12x denser spine puts the
       instrument roughly two orders below the surface it is judging. */
    const fine = [];
    const FINE = 12;
    for (let i = 0; i < (n - 1) * FINE + 1; i++) {
      const spec = sectionAt(i / ((n - 1) * FINE));
      if (!spec) break;
      fine.push(spec.centre);
    }
    const spineForMeasure = fine.length >= n ? fine : centres;
    const dev = blendRadiusDeviation(built.srf, spineForMeasure, radius, evalSrf, 9, Math.min(81, 2 * n + 1));
    return fold({ built, dev: dev.worst, n, instrumentBound: dev.instrumentBound, spineSagitta: dev.spineSagitta });
  };
  let n = startSections;
  let best = null;
  /* What refinement did, kept on the result. A builder that can return
     "closer than this is not reachable" owes the caller the sequence it tried:
     a deviation that falls and stops short is a section ceiling, one that rises
     is a construction that does not converge, and the two want opposite fixes. */
  const trace = [];
  for (let round = 0; round < maxRounds; round++) {
    const a = attempt(n);
    if (!a) return { ok: false, reason: 'the caller could not supply a section' };
    if (a.failed) return { ok: false, reason: a.failed, trace };
    trace.push({ n, dev: a.dev });
    if (!best || a.dev < best.dev) best = a;
    if (a.dev <= tolerance) {
      return { ok: true, srf: a.built.srf, tangencyCurveA: a.built.tangencyCurveA, tangencyCurveB: a.built.tangencyCurveB,
        sections: a.n, deviation: a.dev, tolerance, metTolerance: true, rounds: round + 1, trace,
        instrumentBound: !!a.instrumentBound, spineSagitta: a.spineSagitta, measureFloor: a.measureFloor,
        deviationSigned: a.deviationSigned,
        /* A closed run has no ends, and a caller cannot see that from the
           surface. The two boundary rows of a closed band are the same row, so
           a caller that treats them as two open ends caps a hole that is not
           there. Reported rather than re-derived, because the caller would have
           to compare rows against an arbitrary tolerance to recover it. */
        closed: closedRun,
        /* What the certificate covers. The deviation is sampled over the
           middle of the run, because the nearest point on a finite spine
           polyline clamps at its ends and would report an end effect as a
           surface defect. So `metTolerance` says nothing about the outer 5% at
           each end — where, on an edge with neighbors, a corner patch takes
           over anyway. Stated in the result. */
        certifiedSpan: [0.05, 0.95] };
    }
    // Aim with the convergence law, then verify by measuring again.
    // Fourth order, not second: err ~ C*h^4 with cubic interpolation in V, so the
    // count that meets the target is n * (err/target)^(1/4). Aiming with a
    // second-order law overshoots by a large factor.
    const predicted = Math.ceil(n * Math.pow(a.dev / tolerance, 0.25));
    const next = Math.min(maxSections, Math.max(n + 2, predicted));
    if (next === n) break;
    n = next;
  }
  return {
    ok: true,
    srf: best.built.srf, tangencyCurveA: best.built.tangencyCurveA, tangencyCurveB: best.built.tangencyCurveB,
    sections: best.n, deviation: best.dev, tolerance, trace,
    instrumentBound: !!best.instrumentBound, measureFloor: best.measureFloor,
    deviationSigned: best.deviationSigned,
    closed: closedRun,
    // Not a failure, and not a silent pass. The surface is usable; it is not
    // as close as was asked for, and the caller is handed the number.
    metTolerance: false, rounds: maxRounds,
  };
}

/**
 * Splice a tangency chain into a face's trim loop.
 *
 * Trimming a face back to a blend means replacing part of its boundary with the
 * tangency curve. Both live in the same (u,v) space, and the chain's two ends
 * land on the existing loop — so the operation is: find where each end meets the
 * loop, then keep one of the two arcs between those points and replace the other
 * with the chain.
 *
 * Which arc is dropped is not a guess, and the contract is narrow.
 * `dropNear` must lie in the span of loop being removed — between
 * the two places the chain lands, measured along the loop. The arc containing it
 * is the one that goes.
 *
 * "A point on the edge being filleted" is not sufficient. A chain landing at
 * 0.3 and 0.7 along the bottom of a unit square, with a reference at 0.05 —
 * still on that same edge — names the arc going the long way round, so the
 * splice keeps the 0.04 notch and discards the 0.96 face. The result is a
 * correctly-formed, inverted trim. Nothing about the inputs is
 * malformed; the reference simply points at loop the blend is not replacing.
 *
 * The two arcs partition the loop, so a reference outside the removed span
 * cannot be rescued — it names the other side. What a caller wants is a point
 * on the removed span, and for a fillet that is the midpoint of the edge
 * between its two tangency landings. Deciding by area or by
 * winding instead would be a heuristic that fails on the first L-shaped face.
 *
 * A reference too close to being equidistant between the two arcs is refused
 * with the margin, rather than tie-broken by segment index — a point
 * at the center of a square would otherwise take segment 0 and drop the bottom.
 *
 * Returns the new loop, and the two splice parameters so a caller can tell
 * whether the ends landed where it expected.
 */
/* Where two segments cross, in (u,v), or null. Endpoints count as crossings, so
   a run that merely touches the loop is one too. */
function segCrossUV(p0, p1, a0, a1) {
  const rx = p1[0] - p0[0], ry = p1[1] - p0[1];
  const sx = a1[0] - a0[0], sy = a1[1] - a0[1];
  const denom = rx * sy - ry * sx;
  if (!(Math.abs(denom) > 1e-18)) return null;      // parallel, or a zero-length run
  const qx = a0[0] - p0[0], qy = a0[1] - p0[1];
  const t = (qx * sy - qy * sx) / denom;
  const u = (qx * ry - qy * rx) / denom;
  if (t < -1e-12 || t > 1 + 1e-12 || u < -1e-12 || u > 1 + 1e-12) return null;
  return { t, point: [p0[0] + rx * t, p0[1] + ry * t] };
}
/* A tangency run may cross its face's boundary rather than end on it, and
   that is the ordinary case wherever the boundary it runs into is curved.
   The run spans its own edge, so where the neighboring boundary curves away
   from that edge's end the run carries on past it, and its endpoint does not
   reach the loop.
   Such a run is not floating free. It crosses the loop, and where it crosses
   is the junction. Clipping it there puts its ends on the loop, so the
   caller's tolerance stays as tight as it was, and a run that floats free
   still refuses.
   Returns null unless the run crosses at both ends, so anything that already
   met its loop takes the unclipped path. */
function clipChainToLoop(loop, chain, reach) {
  const n = loop.length;
  /* Extended first, then clipped, because a run can miss its boundary in
     either direction and the two are the same question. Where the neighboring
     boundary curves away the run overshoots and wants clipping; where it curves
     toward it (a boundary bulging past the end of the edge being rounded), the
     run stops short and wants extending.
     Both are answered by growing each end along its own direction by the reach
     the caller allows — the blend's own width, which is how far a run
     may legitimately be from where its edge ended — and then taking the
     outermost crossing. A run that meets its loop already crosses inside the
     original span and is unaffected. */
  if (reach > 0 && chain.length >= 2) {
    const grow = (from, to) => {
      const dx = to[0] - from[0], dy = to[1] - from[1];
      const len = Math.hypot(dx, dy);
      if (!(len > 0)) return null;
      return [to[0] + (dx / len) * reach, to[1] + (dy / len) * reach];
    };
    const pre = grow(chain[1], chain[0]);
    const post = grow(chain[chain.length - 2], chain[chain.length - 1]);
    if (pre && post) chain = [pre, ...chain.map((q) => q.slice()), post];
  }
  const crossingsOn = (ci) => {
    const out = [];
    for (let i = 0; i < n; i++) {
      const x = segCrossUV(chain[ci], chain[ci + 1], loop[i], loop[(i + 1) % n]);
      if (x) out.push(x);
    }
    out.sort((a, b) => a.t - b.t);
    return out;
  };
  let head = null, tail = null;
  for (let ci = 0; ci + 1 < chain.length && !head; ci++) {
    const xs = crossingsOn(ci);
    if (xs.length) head = { ci, t: xs[0].t, point: xs[0].point };
  }
  for (let ci = chain.length - 2; ci >= 0 && !tail; ci--) {
    const xs = crossingsOn(ci);
    if (xs.length) tail = { ci, t: xs[xs.length - 1].t, point: xs[xs.length - 1].point };
  }
  if (!head || !tail) return null;
  if (head.ci > tail.ci || (head.ci === tail.ci && !(tail.t > head.t))) return null;
  const out = [head.point];
  for (let k = head.ci + 1; k <= tail.ci; k++) out.push(chain[k].slice());
  out.push(tail.point);
  return out.length >= 2 ? out : null;
}
export function spliceLoopWithChain(loop, chainIn, dropNear, opts = {}) {
  let chain = chainIn;
  if (!Array.isArray(loop) || loop.length < 3) return { ok: false, reason: 'a trim loop needs at least three points' };
  if (!Array.isArray(chain) || chain.length < 2) return { ok: false, reason: 'a tangency chain needs at least two points' };
  const n = loop.length;
  /* The ends must meet the loop. `nearest()` always returns a segment, so
     without a gap test a chain floating free in the middle of the face would
     splice anyway — both ends snapping to whichever walls are closest,
     cutting the face along a line the chain never described. The gap is the
     test, not only a reported field. */
  const meetTol = opts.meetTolerance != null ? opts.meetTolerance : 1e-6;
  // Nearest point on the loop to a given (u,v), as {seg, t, d}.
  const nearest = (p) => {
    let best = { seg: -1, t: 0, d: Infinity };
    for (let i = 0; i < n; i++) {
      const a = loop[i], b = loop[(i + 1) % n];
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const L2 = dx * dx + dy * dy;
      let t = L2 > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const qx = a[0] + dx * t - p[0], qy = a[1] + dy * t - p[1];
      const d = Math.hypot(qx, qy);
      if (d < best.d) best = { seg: i, t, d };
    }
    return best;
  };
  /* A polyline covers more than one place — measured from its first point over
     all of them, so a run that closes on itself is judged by where it went and
     not by where it finished. `meetTol` is this function's own standard for two
     points being the same place, so no new constant is introduced. */
  const spansTwoPlaces = (pts) => {
    if (!pts || pts.length < 2) return false;
    for (let i = 1; i < pts.length; i++) {
      if (Math.hypot(pts[i][0] - pts[0][0], pts[i][1] - pts[0][1]) > meetTol) return true;
    }
    return false;
  };
  // Clipped to where it meets the boundary, when it runs past it.
  const clipped = clipChainToLoop(loop, chain, opts.reach || 0);
  /* A clip is accepted on its extent, not on its point count. `length >= 2`
     counts array entries, and a clip can come back as several copies of one
     place. If such a clip replaced the chain, `head` and `tail` would land on
     the same loop parameter, the forward interval would be empty,
     `dropIsForward` false, the kept arc empty, and the else branch would
     discard the face's entire boundary, leaving a zero-area trim loop. */
  if (spansTwoPlaces(clipped)) chain = clipped;
  const head = nearest(chain[0]);
  const tail = nearest(chain[chain.length - 1]);
  if (head.d > meetTol || tail.d > meetTol) {
    return { ok: false, reason: `a chain end does not reach the loop — ${Math.max(head.d, tail.d).toExponential(3)} away, past the ${meetTol} join tolerance`, headGap: head.d, tailGap: tail.d };
  }
  // A position along the loop as one number, so the two arcs are easy to name.
  const pos = (h) => h.seg + h.t;
  const pHead = pos(head), pTail = pos(tail);
  /* dropNear is a position along the loop, not a point near the face. "A
     point on the edge being filleted" is only right when that point falls in
     the sub-span between the two splice parameters. A chain landing at 0.3 and
     0.7 along the bottom edge with a reference at 0.05 — still on the filleted
     edge — keeps 0.04 of a unit square instead of 0.96, and the face inverts.
     Same-segment chains are what every partial-edge fillet produces, so that
     input is the ordinary one, not an exotic one.

     Refused when the reference does not land inside either arc unambiguously,
     and the margin to the runner-up is returned so a caller can see a near-tie.
     A reference equidistant from every side of a square would otherwise take
     segment 0. */
  const drop = nearest(dropNear);
  const pDrop = pos(drop);
  {
    /* The two segments meeting at a vertex are not two sides to choose from.
       This margin exists so a reference equidistant from every side of a square
       is refused rather than assigned to segment 0. But a reference sitting on
       a loop vertex is equidistant from the two segments that share it by
       construction, and picking either yields the same position along the
       loop — so there is nothing ambiguous about it.

       That is the ordinary case, not a corner one: a reference is sampled at a
       point of the edge being dropped, and where that edge is curved the loop
       carries it as many short segments, so the sample lands on one of their
       shared vertices. A straight edge is one long segment and the sample falls
       in its interior, which is why only a curved edge reaches this.

       Neighbors are therefore skipped and every other segment still counted,
       so the square keeps its refusal: its opposite side is not adjacent. */
    let second = Infinity;
    const adjacent = (i) => i === drop.seg || (i + 1) % n === drop.seg || (drop.seg + 1) % n === i;
    for (let i = 0; i < n; i++) {
      if (adjacent(i)) continue;
      const a0 = loop[i], b0 = loop[(i + 1) % n];
      const dx = b0[0] - a0[0], dy = b0[1] - a0[1];
      const L2 = dx * dx + dy * dy;
      let t = L2 > 0 ? ((dropNear[0] - a0[0]) * dx + (dropNear[1] - a0[1]) * dy) / L2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const d = Math.hypot(a0[0] + dx * t - dropNear[0], a0[1] + dy * t - dropNear[1]);
      if (d < second) second = d;
    }
    const margin = second - drop.d;
    if (!(margin > (opts.dropMargin != null ? opts.dropMargin : 1e-9))) {
      return { ok: false, reason: `the reference point is ${margin.toExponential(2)} from being equidistant between two arcs — which side to drop is ambiguous`, dropMargin: margin };
    }
  }
  // Walking forward from head to tail wraps or does not; the dropped arc is
  // whichever of the two contains pDrop.
  const inForward = (x) => (pHead <= pTail ? (x >= pHead && x <= pTail) : (x >= pHead || x <= pTail));
  const dropIsForward = inForward(pDrop);
  // Keep the arc that does not contain the drop point, walking from tail back
  // round to head (or head round to tail), then close with the chain.
  const kept = [];
  const pushVertexRange = (from, to) => {
    // Vertices strictly inside the kept arc, in order.
    let i = Math.ceil(from + 1e-12);
    const limit = to < from ? to + n : to;
    for (let k = i; k < limit; k++) kept.push(loop[k % n].slice());
  };
  const chainFwd = chain.map((p) => p.slice());
  const chainRev = chainFwd.slice().reverse();
  let out;
  if (dropIsForward) {
    // Keep tail -> head, then chain head -> tail closes it.
    out = [];
    out.push(pointAtLoop(loop, tail));
    pushVertexRange(pTail, pHead);
    out.push(...kept);
    out.push(pointAtLoop(loop, head));
    out.push(...chainFwd.slice(1, -1));
  } else {
    out = [];
    out.push(pointAtLoop(loop, head));
    pushVertexRange(pHead, pTail);
    out.push(...kept);
    out.push(pointAtLoop(loop, tail));
    out.push(...chainRev.slice(1, -1));
  }
  /* The collapse guard asks the same question. `out.length < 3` counts points,
     and a degenerate output can have exactly three of them, one from `head`,
     one from `tail` and one from the reversed chain. A loop needs three
     distinct places, not three entries. */
  if (out.length < 3 || !spansTwoPlaces(out)) return { ok: false, reason: 'the splice collapsed the loop' };
  return { ok: true, loop: out, headAt: pHead, tailAt: pTail, droppedForward: dropIsForward, headGap: head.d, tailGap: tail.d };
}
function pointAtLoop(loop, h) {
  const a = loop[h.seg], b = loop[(h.seg + 1) % loop.length];
  return [a[0] + (b[0] - a[0]) * h.t, a[1] + (b[1] - a[1]) * h.t];
}

/**
 * The chamfer cross-section — the same section, cut straight instead of round.
 *
 * A chamfer takes the arc's two endpoints and joins them with a line. That is
 * the whole difference, and building it from the same `envelopeSection` is what
 * makes toggling between the two non-destructive: identical tangency points,
 * identical setback, identical footprint on both supporting faces, so switching
 * a fillet to a chamfer never moves where the feature meets the rest of the
 * solid. Only the shape spanning the gap changes.
 *
 * That equivalence is the reason a fillet and a chamfer are one tool with a
 * switch rather than two commands: at a given drag distance they occupy exactly
 * the same ground.
 *
 * Emitted as a degree-1 NURBS with the same 3-control-point layout as the arc,
 * so a run of chamfer sections skins through `blendSurfaceFromSections`
 * unchanged — the middle point sits at the true midpoint with weight 1, which is
 * what makes it a line rather than a flattened conic.
 */
export function chamferSectionArc(section) {
  if (!section || !section.ok) return null;
  const { tangencyA, tangencyB } = section;
  const mid = mul(add(tangencyA, tangencyB), 0.5);
  const span = len(sub(tangencyB, tangencyA));
  if (!(span > EPS)) return null;
  return {
    degree: 2, // held at 2 so the net matches an arc's exactly; weight 1 at the
    // middle makes the quadratic degenerate to the straight line through it.
    knots: [0, 0, 0, 1, 1, 1],
    ctrlPts: [
      [tangencyA[0], tangencyA[1], tangencyA[2], 1],
      [mid[0], mid[1], mid[2], 1],
      [tangencyB[0], tangencyB[1], tangencyB[2], 1],
    ],
    sweep: 0,
    radius: section.radius,
    straight: true,
    span,
  };
}

/**
 * The chamfer's section builder, as a `blendSurfaceToTolerance` hook
 * (`chamferSectionArcFor`, below).
 *
 * The completed section has to be built first and cut second: `chamferSectionArc`
 * needs `tangencyA`/`tangencyB`, which only `envelopeSection` derives, while the
 * generator handed to the builder yields a raw ball spec. Composing them in the
 * wrong order returns null for every station, which the builder reports as "the
 * caller could not supply a section".
 */
/**
 * A curvature-continuous section — the "smooth" third state beside the rolling
 * ball and the chamfer.
 *
 * A circular fillet is G1: the blend meets each flat with a matching tangent and
 * a curvature that jumps from 0 on the face to 1/r on the blend, at a line you
 * can find on any reflective surface. G2 removes that jump by giving the section
 * zero curvature where it lands on each face.
 *
 * A rational quadratic cannot do it — a conic's end curvature is never zero — so
 * the section is a quintic. The condition is purely a statement about control
 * points: a Bezier's curvature at an end vanishes exactly when the first three
 * (or last three) control points are collinear, so
 *
 *     P0 = A,  P1 = A + a*t,  P2 = A + 2a*t
 *     P5 = B,  P4 = B + b*u,  P3 = B + 2b*u
 *
 * is G2 against both faces by construction rather than by fitting, with `t` and
 * `u` the in-plane directions from each tangency point toward the sharp corner
 * the blend replaces.
 *
 * The surface stays G2 between the sections, which is what allows it to be
 * skinned. `blendSurfaceFromSections` interpolates each control row
 * independently along the edge. Collinearity survives that because it is
 * linear in the data: row2 - row0 = 2*(row1 - row0) holds at every
 * station, the two differences are interpolated by the same basis over the same
 * knots, and a basis applied to twice a data set gives twice the result. So the
 * relation holds at every v, exactly, not to a tolerance.
 *
 * The footprint is the ball's. `A` and `B` are the same tangency points a
 * rolling ball of that radius would produce, so "radius" keeps meaning the same
 * thing on screen — how far onto each face the blend reaches — and only the
 * profile between them changes.
 */

/**
 * How full the profile is; it is not one number.
 *
 * `alpha` places the inner control points along each leg toward the corner, and
 * the value that makes the quintic sit closest to the circular arc of the same
 * radius — which is what keeps a smooth blend the same size on screen as the
 * fillet it replaces — depends on how sharp the corner is. Searched at 0.001
 * resolution against the exact circle, and the result is scale free: the same
 * alpha wins at r = 1, 10 and 100, and the residual scales with r exactly.
 *
 *   sweep    best alpha   worst departure from the arc, as a fraction of r
 *    30 deg     0.308                 0.0007
 *    60         0.291                 0.0029
 *    90         0.259                 0.0072
 *   120         0.207                 0.0148
 *   150         0.127                 0.0274
 *
 * A single constant is visibly worse where the corner is wide: 0.4 measures
 * 0.104 of r at 150 degrees against 0.027 for the fitted value, which is a
 * blend that reads as the wrong size rather than as a smoother one. The
 * quadratic below reproduces every row above to within 0.005 of alpha.
 *
 * A wide corner gets a thinner profile, although a flatter corner might be
 * expected to need more filling. It follows from the footprint being
 * fixed: at 150 degrees the tangency points are already most of the way around
 * the ball, so the legs toward the corner are long and a large alpha throws the
 * curve far outside the circle it is meant to match.
 */
export function smoothSectionAlpha(sweepRad) {
  const x = Math.max(0, Math.min(Math.PI, sweepRad));
  const a = 0.3014 + 0.0324 * x - 0.0378 * x * x;
  return Math.max(0.05, Math.min(0.49, a));
}
export function smoothSectionArc(section, alpha) {
  if (!section || !section.ok) return null;
  const c = section.centre;
  const A = section.tangencyA, B = section.tangencyB;
  const nA = norm(sub(A, c)), nB = norm(sub(B, c));
  if (!nA || !nB) return null;
  const cosT = Math.max(-1, Math.min(1, dot(nA, nB)));
  const theta = Math.acos(cosT);
  // A sweep at or past 180 degrees has no corner to aim at, and one at zero has
  // no section: both are refused rather than divided by.
  if (!(theta > 1e-9) || !(theta < Math.PI - 1e-9)) return null;
  // In-plane, perpendicular to each face's own normal, pointing at the corner.
  const tA = norm(sub(nB, mul(nA, dot(nB, nA))));
  const tB = norm(sub(nA, mul(nB, dot(nA, nB))));
  if (!tA || !tB) return null;
  const al = Number.isFinite(alpha) ? alpha : smoothSectionAlpha(theta);
  // Distance from a tangency point to the sharp corner it replaces: for a
  // rolling ball that is exactly r*tan(sweep/2), which is r on a right angle.
  const r = section.radius != null ? section.radius : len(sub(A, c));
  const d = r * Math.tan(theta / 2);
  if (!(d > 0) || !Number.isFinite(d)) return null;
  const a1 = al * d, a2 = 2 * al * d;
  const P = [
    A,
    add(A, mul(tA, a1)),
    add(A, mul(tA, a2)),
    add(B, mul(tB, a2)),
    add(B, mul(tB, a1)),
    B,
  ];
  return {
    degree: 5,
    knots: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    // Non-rational: every weight 1, so the homogeneous interpolation the surface
    // builder does carries the collinearity through untouched.
    ctrlPts: P.map((p) => [p[0], p[1], p[2], 1]),
  };
}

/** The section builder hook, matching `chamferSectionArcFor`'s contract. */
export function smoothSectionArcFor(spec, alpha) {
  const e = envelopeSection(spec);
  if (!e.ok) return { ok: false, reason: e.reason };
  const arc = smoothSectionArc(e, alpha);
  return arc ? { ok: true, arc } : { ok: false, reason: 'a section produced no curvature-continuous profile' };
}

/**
 * How far a built smooth blend departs from its own exact sections.
 *
 * Same reasoning as `chamferFlatnessDeviation`, and for the same reason: the
 * default radius measure reports |distance-to-spine - radius|, and a smooth
 * section is not at constant distance from the spine, so a perfect
 * one measures as a large failure that refinement can never reduce.
 *
 * Measured against the surface's own isocurve, not against a section looked up
 * by parameter — sections are interpolated at chord-length parameters, so a
 * caller's `sectionAt(t)` and the surface's own v do not name the same station.
 * Everything the exact section needs is recoverable from the isocurve itself:
 * its two endpoints, and its two end tangents, which are the directions the
 * supporting faces impose. So the section is rebuilt from the surface and the
 * surface is compared against it, and neither the parameterization nor the
 * caller's generator is in the measure.
 */
export function smoothProfileDeviation(srf, evalSrf, alpha, uSteps = 9, vSteps = 33, vFrom = 0.05, vTo = 0.95) {
  if (!srf || typeof evalSrf !== 'function') {
    return { ok: false, reason: 'a smooth-profile measure needs a surface and an evaluator' };
  }
  const ku = srf.knotsU, kv = srf.knotsV;
  const uLo = ku[0], uHi = ku[ku.length - 1];
  const vLo = kv[0], vHi = kv[kv.length - 1];
  const hU = (uHi - uLo) * 1e-4;
  let worst = 0, worstAt = null, measured = 0;
  for (let j = 0; j < vSteps; j++) {
    const f = vFrom + (vTo - vFrom) * (vSteps === 1 ? 0.5 : j / (vSteps - 1));
    const v = vLo + (vHi - vLo) * f;
    const A = evalSrf(srf, uLo, v), B = evalSrf(srf, uHi, v);
    const A1 = evalSrf(srf, uLo + hU, v), B1 = evalSrf(srf, uHi - hU, v);
    if (!A || !B || !A1 || !B1) continue;
    const tA = norm(sub(A1, A)), tB = norm(sub(B1, B));
    if (!tA || !tB) continue;
    // The corner the two end tangents aim at, from the leg that is best
    // conditioned: solve |A + s*tA - (B + q*tB)| = 0 in the plane they span.
    const w = sub(B, A);
    const d1 = dot(tA, tB), den = 1 - d1 * d1;
    if (!(Math.abs(den) > 1e-12)) continue;
    const sPar = (dot(w, tA) - d1 * dot(w, tB)) / den;
    if (!(sPar > 0)) continue;
    const Q = add(A, mul(tA, sPar));
    const dA = len(sub(Q, A)), dB = len(sub(Q, B));
    if (!(dA > EPS) || !(dB > EPS)) continue;
    // The same rule the section was built by, from the sweep this isocurve
    // implies: the corner's interior angle is pi minus the turn between the two
    // end tangents, and the ball's sweep is pi minus that again.
    const turn = Math.acos(Math.max(-1, Math.min(1, -d1)));
    const al = Number.isFinite(alpha) ? alpha : smoothSectionAlpha(Math.PI - turn);
    const P = [
      A, add(A, mul(tA, al * dA)), add(A, mul(tA, 2 * al * dA)),
      add(B, mul(tB, 2 * al * dB)), add(B, mul(tB, al * dB)), B,
    ];
    for (let i = 1; i + 1 < uSteps; i++) {
      const u = uLo + (uHi - uLo) * (i / (uSteps - 1));
      const p = evalSrf(srf, u, v);
      if (!p) continue;
      // Nearest point on the exact quintic, sampled finely enough that the
      // sampling is well below what is being measured.
      let best = Infinity;
      for (let k = 0; k <= 200; k++) {
        const tt = k / 200;
        const q = bezierPoint5(P, tt);
        const dd = len(sub(p, q));
        if (dd < best) best = dd;
      }
      measured++;
      if (best > worst) { worst = best; worstAt = { u, v }; }
    }
  }
  /* Same rule as `chamferFlatnessDeviation`'s own: an unmeasured surface must
     not certify as an exact one. Every station here can `continue` too — a
     tangent that will not normalize, two end tangents too nearly parallel to
     locate a corner, a corner behind the start — and `worst` left at its initial
     0 would report a perfect match off no samples, which stops refinement. */
  if (measured === 0) {
    return { ok: false, reason: 'no station on this surface could be measured — no isocurve yielded a usable pair of end tangents, so its departure from the exact profile is UNKNOWN rather than zero' };
  }
  return { ok: true, worst, worstAt, stations: measured };
}

function bezierPoint5(P, t) {
  const s = 1 - t;
  const b = [s * s * s * s * s, 5 * s * s * s * s * t, 10 * s * s * s * t * t,
    10 * s * s * t * t * t, 5 * s * t * t * t * t, t * t * t * t * t];
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < 6; i++) { x += P[i][0] * b[i]; y += P[i][1] * b[i]; z += P[i][2] * b[i]; }
  return [x, y, z];
}

export function chamferSectionArcFor(spec) {
  const e = envelopeSection(spec);
  if (!e.ok) return { ok: false, reason: e.reason };
  const arc = chamferSectionArc(e);
  return arc ? { ok: true, arc } : { ok: false, reason: 'a section produced no chord' };
}

/**
 * How far a built chamfer departs from its own exact chords.
 *
 * The default deviation measure cannot judge a chamfer. `blendRadiusDeviation`
 * reports |distance-to-spine - radius|; a chamfer's chord midpoint sits at
 * r*cos(sweep/2) from the ball center, so an exact chamfer on a 90 degree edge
 * measures r*(1-cos45) ~ 0.293r of "deviation" — about 1.46mm on a 5mm chamfer
 * against a 0.01mm tolerance. The refinement loop then chases a number that can
 * never fall, ramps to `maxSections`, and returns its best effort with
 * `metTolerance: false` and a figure that means nothing.
 *
 * What is exact for a chamfer: at every station the true surface is the straight
 * segment between the two tangency points. A single section is therefore exact by
 * construction (degree 2, unit weights, midpoint at the true midpoint), and the
 * only error left is the interpolation between stations — which is the same
 * quantity the radius measure captures for a fillet, and the only one worth
 * reporting.
 *
 * So the measure is the distance from each sampled surface point to the exact
 * chord at its own station. The ruler is analytic, so unlike the radius measure
 * there is no instrument floor to subtract.
 *
 * Sampled over the middle of the run for the same reason the radius measure is:
 * a corner patch takes over the outer ends, and an end effect is not a surface
 * defect.
 */
export function chamferFlatnessDeviation(srf, evalSrf, uSteps = 9, vSteps = 33, vFrom = 0.05, vTo = 0.95) {
  if (!srf || typeof evalSrf !== 'function') {
    return { ok: false, reason: 'a chamfer measure needs a surface and an evaluator' };
  }
  /* Measured against the surface's own section, not against a section looked
     up by parameter. Asking the caller's `sectionAt(t)` for the chord and
     evaluating the surface at that same t compares the same station only
     while the two parameterizations agree. They do not: sections are
     interpolated at chord-length parameters, so on any run where the stations
     are unevenly spaced that would compare one place on the surface against
     the chord belonging to another — an error that does not shrink with
     refinement, because it is not in the surface.

     A section is flat if it is straight between its own two ends, which is a
     question about the surface alone. Taking the endpoints from the same isocurve
     being sampled removes the parameterization from the measure entirely, and
     removes the caller's section generator from it too. */
  const ku = srf.knotsU, kv = srf.knotsV;
  const uLo = ku[0], uHi = ku[ku.length - 1];
  const vLo = kv[0], vHi = kv[kv.length - 1];
  let worst = 0, measured = 0;
  for (let j = 0; j < vSteps; j++) {
    const f = vFrom + (vTo - vFrom) * (vSteps === 1 ? 0.5 : j / (vSteps - 1));
    const v = vLo + (vHi - vLo) * f;
    const a = evalSrf(srf, uLo, v);
    const b = evalSrf(srf, uHi, v);
    if (!a || !b) continue;
    const ab = sub(b, a);
    const abLen2 = dot(ab, ab);
    if (!(abLen2 > EPS)) continue;
    for (let i = 1; i + 1 < uSteps; i++) {
      const u = uLo + (uHi - uLo) * (i / (uSteps - 1));
      const p = evalSrf(srf, u, v);
      if (!p) continue;
      const q = [p[0], p[1], p[2]];
      // Distance to the segment, clamped: past either end the nearest point on
      // the true chamfer is its endpoint, and an unclamped line distance would
      // read zero for a point that has run off the end of the chord entirely.
      let sPar = dot(sub(q, a), ab) / abLen2;
      sPar = sPar < 0 ? 0 : sPar > 1 ? 1 : sPar;
      const d = len(sub(q, add(a, mul(ab, sPar))));
      measured++;
      if (d > worst) worst = d;
    }
  }
  /* A surface that could not be measured is not a flat one. Every station
     above can `continue` — an evaluator that returns nothing, a chord that
     degenerates — and with `worst` still at its initial 0 the loop would
     report `ok: true, worst: 0` from zero samples. `blendSurfaceToTolerance`
     would read that as tolerance met on the first attempt and stop refining.
     Unknown must not read as perfect. */
  if (measured === 0) {
    return { ok: false, reason: 'no station on this surface could be measured — every sample either failed to evaluate or had a degenerate chord, so its flatness is UNKNOWN rather than perfect' };
  }
  return { ok: true, worst, instrumentBound: false, floor: 0, stations: measured };
}

/**
 * Is a built blend round or flat? The check a record cannot provide.
 *
 * A fillet and a chamfer differ only in the shape spanning the gap, and every
 * other observable — tangency points, setback, footprint, the record's own type
 * field — is identical between them. So a check that reads the record, or the
 * parameters, or the footprint, passes for both and distinguishes neither.
 *
 * This reads the surface. Three points across one section determine a circle;
 * the residual of the remaining samples against that circle says whether the
 * section is an arc, and the fitted radius says which arc. A chord returns
 * a vanishing curvature and a radius that runs away to infinity.
 *
 * Returns `curvature` (1/radius, zero for a straight section) rather than the
 * radius itself, because the flat case is the one being tested for and a
 * predicate on a finite number is easier to assert than one on an infinity.
 */
export function blendSectionCurvature(srf, evalSrf, v = 0.5, samples = 9) {
  if (!srf || typeof evalSrf !== 'function') return { ok: false, reason: 'a curvature oracle needs a surface and an evaluator' };
  const pts = [];
  for (let i = 0; i < samples; i++) {
    const p = evalSrf(srf, i / (samples - 1), v);
    if (p) pts.push([p[0], p[1], p[2]]);
  }
  if (pts.length < 3) return { ok: false, reason: 'the section could not be sampled' };
  const a = pts[0], b = pts[Math.floor((pts.length - 1) / 2)], c = pts[pts.length - 1];
  const ab = sub(b, a), ac = sub(c, a);
  const n = cross(ab, ac);
  const nLen = len(n);
  const chord = len(sub(c, a));
  // Three collinear points span no plane, which is the flat answer rather than a
  // failure — a chamfer's section is exactly this case.
  if (!(nLen > EPS) || !(chord > EPS)) return { ok: true, curvature: 0, radius: Infinity, residual: 0, flat: true };
  // Circumradius from the triangle: R = |ab||ac||bc| / (4*area).
  const bc = len(sub(c, b));
  const R = (len(ab) * len(ac) * bc) / (2 * nLen);
  if (!Number.isFinite(R) || !(R > EPS)) return { ok: true, curvature: 0, radius: Infinity, residual: 0, flat: true };
  // The center lies in the plane of the three points, at the intersection of two
  // perpendicular bisectors — solved directly rather than iterated.
  const nHat = mul(n, 1 / nLen);
  const abMid = mul(add(a, b), 0.5), acMid = mul(add(a, c), 0.5);
  const dirAB = cross(nHat, ab), dirAC = cross(nHat, ac);
  const w = sub(acMid, abMid);
  const denom = dirAB[0] * dirAC[1] - dirAB[1] * dirAC[0];
  let centre = null;
  if (Math.abs(denom) > 1e-9) {
    const s = (w[0] * dirAC[1] - w[1] * dirAC[0]) / denom;
    centre = add(abMid, mul(dirAB, s));
  } else {
    const d2 = dirAB[1] * dirAC[2] - dirAB[2] * dirAC[1];
    if (Math.abs(d2) > 1e-9) {
      const s = (w[1] * dirAC[2] - w[2] * dirAC[1]) / d2;
      centre = add(abMid, mul(dirAB, s));
    }
  }
  if (!centre) return { ok: true, curvature: 1 / R, radius: R, residual: Infinity, flat: false };
  let residual = 0;
  for (const p of pts) residual = Math.max(residual, Math.abs(len(sub(p, centre)) - R));
  // A section whose samples do not sit on their own circumcircle is not an arc,
  // whatever the three chosen points implied.
  const flat = residual > R * 1e-3 || R > chord * 1e6;
  return { ok: true, curvature: flat ? 0 : 1 / R, radius: flat ? Infinity : R, residual, flat, chord };
}

/**
 * The area of a corner patch, by two independent routes.
 *
 * The spherical triangle bounded by three touch directions has area
 * r^2 * (A + B + C - pi) — Girard's excess — where A, B, C are the triangle's
 * interior angles.
 *
 * The interior angle is not the dihedral, and a cube cannot tell the
 * difference. On three perpendicular faces both are 90 degrees, so an octant
 * result confirms nothing about which quantity the formula wants. On a skewed
 * trihedron they separate plainly: interior angles 88.52 / 68.99 / 100.87
 * against dihedrals 78.92 / 87.38 / 111.12. The interior angle is measured
 * between the two arcs leaving a vertex — each other direction projected into
 * the tangent plane there — which is what this computes and what Girard needs.
 *
 * The dihedral is the supplement of the angle between two touch directions, and
 * is a different number again; it belongs to the edge, not to the corner.
 *
 * Checked against van Oosterom & Strackee, "The solid angle of a plane
 * triangle", IEEE Transactions on Biomedical Engineering BME-30(2):125-126,
 * 1983 — tan(omega/2) = |a.(b x c)| / (1 + a.b + b.c + c.a), which shares no
 * algebra with the excess-of-angles route. They agree to the last digit on both
 * a cube corner and a skewed one, which is why the disagreement is returned
 * rather than assumed: a construction that drifts will say so here first.
 */
export function sphericalTriangleArea(dirs, radius) {
  if (!Array.isArray(dirs) || dirs.length !== 3) return { ok: false, reason: 'a spherical triangle needs exactly three directions' };
  const d = dirs.map(norm);
  if (d.some((x) => !x)) return { ok: false, reason: 'a direction has no length' };
  const interior = [];
  for (let i = 0; i < 3; i++) {
    const a = d[i], b = d[(i + 1) % 3], c = d[(i + 2) % 3];
    const u1 = norm(sub(b, mul(a, dot(b, a))));
    const u2 = norm(sub(c, mul(a, dot(c, a))));
    if (!u1 || !u2) return { ok: false, reason: 'two directions coincide — the triangle has no area' };
    interior.push(Math.acos(Math.max(-1, Math.min(1, dot(u1, u2)))));
  }
  const excess = interior.reduce((s, v) => s + v, 0) - Math.PI;
  const num = Math.abs(dot(d[0], cross(d[1], d[2])));
  const den = 1 + dot(d[0], d[1]) + dot(d[1], d[2]) + dot(d[2], d[0]);
  const vanOosterom = 2 * Math.atan2(num, den);
  return {
    ok: true,
    interiorAngles: interior,
    excess,
    area: radius * radius * excess,
    // The second opinion, and the gap between them: the two share no
    // algebra, so a drift in either shows up here before it shows up in a
    // surface.
    solidAngle: vanOosterom,
    agreement: Math.abs(excess - vanOosterom),
  };
}
