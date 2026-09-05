/* ROLLING-BALL EDGE BLENDING — the cross-section, and the conditions under
   which it exists at all.
   ==================================================================

   PRIOR ART, cited before this file was written, per the standing rule that
   a citation lands before the code — read for TECHNIQUE, never transcribed:

     · Rossignac, J. R. & Requicha, A. A. G., "Constant-Radius Blending in
       Solid Modeling." ASME Computers in Mechanical Engineering (CIME) 3,
       pp. 65-73, 1984. The rolling-sphere formulation itself.
     · Choi, B. K. & Ju, S. Y., "Constant-radius blending in surface
       modeling." Computer-Aided Design 21(4), pp. 213-220, 1989.
       DOI 10.1016/0010-4485(89)90046-8. The FEASIBILITY precondition, from
       its own abstract: two surfaces can be blended at radius r as long as
       their offset surfaces at distance r are smooth, so that the
       intersection between those offsets is well defined. That intersection
       is the spine.
     · Peternell, M. & Pottmann, H., "Computing Rational Parametrizations of
       Canal Surfaces." Journal of Symbolic Computation 23(2-3), pp. 255-266,
       1997. DOI 10.1006/jsco.1996.0087. The variable-radius existence
       condition used in `variableRadiusFeasible` below.

   WHAT THIS MODULE IS NOT. It does not compute the maximum buildable radius.
   The true bound is the smaller principal radius of curvature on the concave
   side (Wallner et al. 2001), which needs SECOND derivatives of the supporting
   surfaces; this kernel computes first partials only, by an earlier deliberate
   decision. Anything this module returns about size is a NECESSARY condition,
   never a sufficient one, and it says so in the field names.

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
 * THE CROSS-SECTION OF A ROLLING-BALL BLEND AT ONE POINT ALONG AN EDGE.
 *
 * ⚠ THE SETBACK IS GOVERNED BY THE ANGLE BETWEEN THE CO-NORMALS, NOT BY THETA,
 * and writing it against theta gives a NEGATIVE setback on every concave edge.
 *
 *     phi = acos(coNormalA . coNormalB)          always in [0, pi]
 *     setback d     = r / tan(phi / 2)
 *     center offset = r / sin(phi / 2)
 *
 * theta is the angle measured through the MATERIAL and runs to 2*pi; phi is the
 * angle between the two directions the faces actually extend in. They agree on
 * a convex edge and are supplementary about 2*pi on a concave one, so
 * tan(theta/2) flips sign at theta = pi while the geometry does not.
 *
 * Worked both ways by hand, which is how the sign error surfaced:
 *   · BOX EDGE, theta = phi = 90 degrees, ball of radius r in a right-angled
 *     corner at the origin with cA = +y and cB = +x. Tangency at (0, r) and
 *     (r, 0), center at (r, r). So d = r and the center sits r*sqrt(2) along the
 *     45-degree bisector. r/tan(45) = r and r/sin(45) = r*sqrt(2). Agree.
 *   · CONCAVE EDGE of an L-shaped solid, theta = 270 degrees, cA = +x and
 *     cB = +y so phi is still 90. The ball rounds the inside corner: tangency
 *     at (edge + r*cA) and (edge + r*cB), center at edge + (r, r). The setback
 *     is +r, exactly as in the convex case. Against theta it would have been
 *     r/tan(135) = -r, putting both tangency points on the wrong side of the
 *     edge and the blend inside the solid.
 *
 * theta is still carried, and still needed: it is what says whether the blend
 * REMOVES material or ADDS it, which decides which side gets trimmed.
 *
 * THE SETBACK IS WHY theta HAS TO BE CARRIED ALONG THE EDGE RATHER THAN
 * SAMPLED ONCE. Where theta varies — every S-shaped or blob-derived edge — d
 * varies with it, so the two tangency curves are NOT constant-distance offsets
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
  // NO SEPARATE GUARD ON A DEGENERATE ANGLE VALUE. Every way theta can be a
  // degenerate ANGLE shows up in phi and is caught below, with a message that
  // says what actually happened: theta near 0 is a knife edge (phi near 0),
  // theta near pi is a smooth junction (phi near pi), theta near 2*pi is a thin
  // slot (phi near 0 again). Guarding those here as well duplicated the cases
  // and described them wrongly — a knife edge comes back from that duplicate
  // path labeled "too nearly tangent", which is the opposite condition.
  //
  // ⚠ BUT THETA BEING ABSENT OR IN THE WRONG UNITS IS NOT AN ANGLE DEGENERACY,
  // and phi genuinely cannot see it: phi is derived from the co-normals alone.
  // `convex` is read from theta and from nothing else, so an omitted theta, a
  // NaN, or a value handed over in DEGREES all returned ok:true with
  // convex:false — every convex edge silently classified concave, which tells
  // the caller to add material where it must remove it. Wrong and quiet is the
  // worst pair, so this is a refusal rather than a default.
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
    // A CURVED face needs these projected onto the surface afterwards; on a
    // planar face they are already exact.
    tangencyA: add(point, mul(cA, setback)),
    tangencyB: add(point, mul(cB, setback)),
    centre: add(point, mul(bisector, centreOffset)),
    bisector,
    // Convex blends REMOVE material, concave blends ADD it. The caller needs
    // this to know which side to trim, and it falls straight out of theta.
    convex: theta < Math.PI,
  };
}

/**
 * THE NECESSARY CONDITION ON RADIUS THAT IS ACTUALLY CHECKABLE HERE.
 *
 * Choi & Ju's precondition is about the OFFSET surfaces, and the honest part of
 * it reachable without second derivatives is this: the setback must not exceed
 * how much face there is to set back into. A ball whose tangency point would
 * land beyond the far side of a face has rolled off it — the "ball falls off
 * the surface rails" case, which is the single most common blend failure and
 * the one a local algorithm cannot see at all.
 *
 * `widthA`/`widthB` are how far each face extends from the edge, measured
 * along its own co-normal. Returns the largest radius that still fits, so a
 * refusal can name a number instead of a class of problem.
 */
export function maxRadiusForSetback({ phi, widthA, widthB }) {
  // phi comes from an acos and is therefore in [0, pi]; anything else is a
  // caller error, and `Math.abs(tan(phi/2))` was laundering it into a plausible
  // answer (phi = 4.2 returned rMax = 17.1, ok:true).
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
    // NAMED AS WHAT IT IS. This bounds the radius by how much FACE there is,
    // not by curvature. A face wide enough to hold the setback can still be too
    // curved to hold the ball, and that bound needs second derivatives this
    // kernel does not compute.
    bound: 'setback-only',
    limitedBy: widthA <= widthB ? 'A' : 'B',
  };
}

/**
 * VARIABLE RADIUS — WHEN THE ENVELOPE EXISTS AT ALL.
 *
 * Peternell & Pottmann 1997: a canal surface swept by a ball of varying radius
 * r(t) along a spine m(t) has a real envelope exactly if
 *
 *     |m'(t)|^2 - r'(t)^2  >=  0
 *
 * — if the radius changes faster than the spine advances, there is no surface
 * to build, and at equality the characteristic circle degenerates to a point.
 * Unlike the curvature bound this is computable from FIRST derivatives, so it
 * is a refusal this app can actually make honestly.
 *
 * `spine` is a polyline of ball centers, `radii` the radius at each. Returns
 * the worst margin and where it occurs, so a refusal can point at the span that
 * is asking too much rather than at the whole edge.
 */
export function variableRadiusFeasible(spine, radii) {
  if (!Array.isArray(spine) || !Array.isArray(radii) || spine.length < 2 || spine.length !== radii.length) {
    // ⚠ `radii` is checked BEFORE its length is read. Reading it first threw a
    // TypeError straight through a function whose whole contract is to return
    // {ok, reason} — a refusal path that crashes is not a refusal path.
    return { ok: false, reason: 'spine and radii must be matching arrays of at least two samples' };
  }
  let worst = Infinity, worstAt = -1;
  for (let i = 0; i + 1 < spine.length; i++) {
    const ds = len(sub(spine[i + 1], spine[i]));
    const dr = radii[i + 1] - radii[i];
    // Both sides are per-step, so the parameterisation cancels and no
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
  // The middle control point is where the two end TANGENTS meet, which for a
  // circular arc lies on the bisector at radius / cos(halfSweep).
  const bis = norm(add(uA, uB));
  if (!bis) return null;
  const R = len(rA);
  const mid = add(centre, mul(bis, R / w));
  return {
    degree: 2,
    knots: [0, 0, 0, 1, 1, 1],
    /* ⚠ CARTESIAN COORDINATES WITH THE WEIGHT APPENDED, NOT PREMULTIPLIED.
       This project's evaluators do the premultiplication themselves —
       `curvePoint`/`surfacePoint` read [x, y, z, w] and accumulate x*w — and
       `makeArc` stores `[...point, weight]` to match. Premultiplying here
       applied the weight twice and displaced the middle control point by a
       factor of w.

       It survived the straight-edge test because that test measured PERPENDICULAR
       distance to a spine running parallel to z, and the displacement is along
       the very axis such a measurement cannot see: at z = 0 the two forms are
       numerically identical, and at z = 10 the point moved to 7.07 without
       changing its distance from the line by anything at all. The instrument
       could not express the defect. */
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
 * SKIN A RUN OF SECTION ARCS INTO THE BLEND SURFACE.
 *
 * Every section produced by `sectionArc` is a rational quadratic on the SAME
 * degree and the SAME knot vector, which is what makes this exact rather than a
 * fit: the sections' control points ARE the surface's control net in U, so the
 * blend's cross-section stays a true circular arc everywhere instead of a
 * polynomial approximation of one. The general `loft` in kernel/loft.mjs
 * resamples each section onto a shared parameterisation, which is the right
 * thing for arbitrary hand-picked curves and the wrong thing here — it would
 * discard the exact conic form the rolling ball just handed us.
 *
 * ⚠⚠ THIS IS A SLIDING-DISC CONSTRUCTION, NOT THE EXACT ROLLING-BALL ENVELOPE,
 * and the difference is real wherever the dihedral varies. The sections are laid
 * in planes perpendicular to the EDGE; the true envelope's characteristic circle
 * lies perpendicular to the SPINE. Those coincide only while the spine runs
 * parallel to the edge, which it does exactly when the dihedral angle is
 * constant — as theta varies, the center moves within the cross-section plane
 * too, and the spine tilts away.
 *
 * What this construction DOES guarantee, exactly and at every section:
 *   · the cross-section is a true circular arc of the requested radius;
 *   · it meets each supporting face tangentially, because the arc's radius at
 *     each tangency point runs along that face's normal by construction;
 *   · the tangency curves lie in the faces, which is what the supports must be
 *     trimmed back to.
 * What it does NOT guarantee is that every point is exactly `radius` from the
 * spine BETWEEN sections. On a straight edge of constant angle that holds to
 * machine precision; across a varying dihedral the departure falls at FOURTH
 * order in the section spacing with cubic interpolation in V, which is why
 * `blendRadiusDeviation` measures what was achieved rather than assuming it.
 *
 * ⚠ AN EARLIER VERSION OF THIS PARAGRAPH SAID THE DEPARTURE WAS ABOUT 3% AND
 * "CONVERGES RATHER THAN FALLING WITH SECTION COUNT". Both were the measuring
 * instrument, not the surface: distance was taken to a POLYLINE through the ball
 * centers, whose own chord error is second order, so every surface at every
 * degree reported O(h^2) and appeared to stall. Struck rather than softened,
 * because a stale claim in a comment outlives the code it describes and leaves
 * the next reader choosing between two paragraphs that disagree.
 *
 * U is the cross-section (degree 2, rational, 3 control points).
 * V runs along the edge, degree `degV` clamped to what the section count
 * supports, with a chord-length knot vector so an unevenly sampled edge is not
 * silently reparameterised.
 *
 * ⚠ V IS AN APPROXIMATION AND SAYS SO. Taking the section control points as the
 * net makes the surface pass exactly through the FIRST and LAST sections and
 * approximate the ones between. On a straight edge of constant dihedral every
 * section is a translate of its neighbors, so the approximation is exact and
 * the result is a true cylindrical patch; on a curved edge or a varying angle
 * the error falls with section count and is what
 * `blendRadiusDeviation` measures. Nothing here claims it is zero.
 */
export function blendSurfaceFromSections(arcs, degV = 3) {
  if (!Array.isArray(arcs) || arcs.length < 2) return { ok: false, reason: 'a blend needs at least two sections' };
  /* ⚠ ONE SHAPE OF SECTION PER SURFACE, WHATEVER THAT SHAPE IS. This required a
     rational quadratic outright, which is what `sectionArc` produces and is
     right for a rolling ball and for a chamfer's chord. It is not right for a
     CURVATURE-CONTINUOUS section, which needs a quintic to carry zero curvature
     at both ends, and the restriction was the only thing standing between this
     builder and one. What actually has to hold is that the sections agree —
     they are interpolated control point by control point, so a surface cannot
     be made from sections with different degrees or different counts of them. */
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
  /* ⚠ NO DIVISION BY THE WEIGHT. Control points here are stored CARTESIAN with
     the weight appended — `sectionArc` says so and `surfacePoint` premultiplies
     for itself — so dividing by w deprojects something that was never projected
     and displaces every arc's midpoint radially by 1/w.

     It was invisible to every test because `blendRadiusDeviation` is
     parameterisation-free and the straight-edge fixture has CONSTANT weight,
     where a uniform chord scaling cancels in the normalization. What it broke
     was real: two sections 2.9992 apart whose distorted midpoints coincided were
     refused as "all at the same place", and three such sections produced
     duplicate parameters that made `solveLinearSystem` THROW straight through
     the ok/reason contract. */
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
  /* INTERPOLATED IN V, NOT APPROXIMATED — and the difference runs the opposite
     way to intuition, which is why it is worth stating. Taking the section
     control points AS the net makes a degree-3 surface that does not pass
     through the sections between the ends, and it measured consistently WORSE
     than degree 1: 0.094% against 0.011% on the same fixture at 97 sections. A
     higher degree approximating is beaten by a lower degree interpolating.
     Solving for control points that make the surface pass through every section
     gets both — smooth in V and exact at each section.

     Solved in HOMOGENEOUS coordinates (x*w, y*w, z*w, w) and projected back, so
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
      /* ⚠ POSITIVE, not merely non-zero. Data weights are cos(halfSweep) and lie
         in (0,1], but the INTERPOLANT is not bounded by its data: sections whose
         sweeps alternate sharply overshoot to control weights like
         [0.940, -1.516, 3.399, -1.523, 0.940]. An |w| > EPS test accepts those,
         and the surface comes back ok:true while this project's own
         `isFiniteNet` rejects the net for w <= 0. A negative control weight also
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
 * HOW FAR THE BUILT SURFACE STRAYS FROM THE BALL THAT DEFINED IT.
 *
 * The defining property of a constant-radius blend is that every point on it is
 * exactly `radius` from the SPINE — the locus of ball centers. Anything else is
 * a surface that merely looks like a fillet.
 *
 * ⚠ MEASURED TO THE NEAREST POINT ON THE SPINE, not to the spine "at the same
 * parameter". Those are different questions and only the first one is the
 * definition. The surface carries a chord-length knot vector, so its v and a
 * spine sampled evenly by index do not name the same place along the edge; a
 * first version of this compared them anyway and reported a 37% error on a
 * blend whose real error is a small fraction of a percent. The parameterisation
 * is free to be whatever it likes — what must hold is the distance.
 *
 * `spine` is a polyline of ball centers. `evalSrf(srf, u, v)` is the caller's
 * own surface evaluator, kept as a parameter so this module stays independent
 * of the rest of the kernel. `vFrom`/`vTo` trim the sampled span, because the
 * nearest point on a FINITE polyline clamps at its ends and would report an end
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
  /* ⚠⚠ THE SPINE POLYLINE IS THE RULER, AND ITS OWN ERROR IS SECOND ORDER.

     Distance is measured to a POLYLINE through the ball centers, and a polyline
     cuts the chord off a curving spine by roughly sagitta = L^2 / (8*rho) per
     segment. Sample the spine at the same density as the sections and that
     chord error DOMINATES: every surface, at every V degree, measures O(h^2) —
     which is the ruler's convergence being reported as the surface's.

     It cost a wrong conclusion. Sampling the spine 40x finer on the SAME
     surfaces turned 1.796e-3mm at 97 sections into 1.176e-6mm, and the observed
     order from 2.01 into roughly 4 — cubic interpolation behaving exactly as
     cubic interpolation should. The earlier reading that "raising the V degree
     cannot fix a second-order departure" was measuring the instrument.

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
 * THE SECTION, BUILT FROM THE FACE NORMALS RATHER THAN FROM A SETBACK.
 *
 * `rollingBallSection` locates the tangency points by measuring a setback along
 * each co-normal from the edge. That is correct, and it puts the section in the
 * plane perpendicular to the EDGE — which is the sliding-disc construction, and
 * departs from the true envelope wherever the dihedral varies.
 *
 * This does the same job the other way round and gets the envelope for free. A
 * sphere of radius r centered at m is tangent to a face with OUTWARD unit normal
 * n exactly where it touches, and that point is simply
 *
 *     tangency = m + r * n
 *
 * (outward, so the touch point is on the far side of the center from the
 * material — checked on a box: material in x>0, y>0, center (r, r), face x = 0
 * whose outward normal is -x, giving (r,r) + r*(-1,0) = (0, r), which is where
 * the ball actually touches).
 *
 * The two tangency points and the center then span the section plane, and that
 * plane is the characteristic one: the envelope condition for a constant-radius
 * canal surface is (p - m) . m' = 0, so the contact points lie in the plane
 * through m perpendicular to the spine tangent. Building the section from the
 * normals therefore lands in the right plane WITHOUT ever computing m' — the
 * derivative is implied by the geometry rather than estimated from samples.
 *
 * No setback appears anywhere here, which is the point: the setback was only
 * ever a way of finding the tangency from the edge, and finding it from the
 * normal instead removes the one approximation that made the disc slide.
 */
export function envelopeSection({ centre, radius, toTouchA, toTouchB, normalA, normalB }) {
  if (!(radius > 0)) return { ok: false, reason: 'radius must be positive' };
  /* ⚠ THESE POINT FROM THE CENTER TOWARDS THE TOUCH POINT, and they are NOT
     "the face's outward normal" in general — the parameters are named for the
     direction rather than for the normal precisely because that distinction is
     invisible when it is wrong.

     On a CONVEX edge the ball sits inside the material, so center-to-touch and
     the outward normal coincide and either name works. On a CONCAVE edge the
     ball sits in the void and they are OPPOSITE, so passing an outward normal
     puts the tangency point 2r away on the far side of the ball. It is still
     exactly `radius` from the center, so any check that only measures the radius
     reports a perfect blend that touches neither face — measured here at 6mm
     from the plane it was meant to sit on, with the radius reading 3.000000000.

     `normalA`/`normalB` stay accepted as the older spelling so no caller breaks
     silently, but the new names are the contract. */
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
 * Kept separate rather than overloaded because the two carry different fields
 * and silently accepting either is how one gets used where the other was meant.
 */
export function envelopeSectionArc(section) {
  if (!section || !section.ok) return null;
  return sectionArc({ ok: true, centre: section.centre, tangencyA: section.tangencyA, tangencyB: section.tangencyB });
}

/**
 * BUILD TO A TOLERANCE, AND REPORT WHAT WAS ACHIEVED.
 *
 * The departure of a skinned blend from the true tube falls at FOURTH order in
 * the section spacing, with cubic interpolation in V. Density is the control,
 * and the honest way to set it is to aim, measure, and correct rather than to
 * pick a number.
 *
 * From err ~ C*h^4 with h ~ 1/N, one measurement predicts the count that would
 * meet the target: N_needed = N * (err / target)^(1/4). That prediction is then
 * VERIFIED by measuring again — the law is used to aim, never to certify. A
 * blend that cannot reach the target inside `maxRounds` returns its best effort
 * WITH the deviation it actually achieved, so a caller can say "accurate to
 * 0.004mm" instead of implying an exactness it does not have.
 *
 * `sectionAt(t)` for t in [0,1] returns { centre, radius, normalA, normalB } —
 * the caller owns the geometry, this owns the density.
 *
 * TWO HOOKS, AND WHAT THEY ARE FOR. The aim-measure-correct loop is not specific
 * to a constant radius, but everything it does INSIDE the loop is:
 *   · `opts.sectionArcFor(spec)` -> { ok, arc } | { ok: false, reason } replaces
 *     the envelope section builder, for a section that is not a great circle of
 *     the ball — a variable-radius contact circle is offset along the spine and
 *     shrunk (kernel/varradius.mjs).
 *   · `opts.measure(srf, n)` -> { worst, instrumentBound, floor } replaces the
 *     deviation measure, and supplying it is what LIFTS the one-radius refusal
 *     below. The refusal stays the default on purpose: a caller who did not mean
 *     to vary the radius must still be told, rather than handed a number
 *     measured against whichever radius happened to come last.
 */
export function blendSurfaceToTolerance(sectionAt, tolerance, opts = {}) {
  const startSections = opts.startSections || 17;
  const maxSections = opts.maxSections || 401;
  const maxRounds = opts.maxRounds || 4;
  const degV = opts.degV != null ? opts.degV : 3;
  const evalSrf = opts.evalSrf;
  if (typeof evalSrf !== 'function') return { ok: false, reason: 'an evaluator is required to measure what was built' };
  const measure = typeof opts.measure === 'function' ? opts.measure : null;
  /* A SECOND DEVIATION THE LOOP MUST ALSO DRIVE DOWN. `measure` replaces how the
     blend judges ITSELF — a chamfer swaps radius error for flatness error. This
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
  /* ⭐⭐ A CLOSED RUN IS SAMPLED ON A CYCLE, NOT ON AN INTERVAL.
     `i / (n - 1)` walks 0 to 1 inclusive, which is right for an edge that has
     two ends and wrong for one that has none: on a loop, t=0 and t=1 name the
     SAME station, so the run either duplicates a section there or leaves the
     wrap unsampled — and the band cannot close.
     Measured on a filleted extruded circle before this existed: the band spanned
     about 1,237mm of a 1,257mm circumference and THREE small filler faces
     bridged the remaining 20mm at the seam, every one of them lying flat in the
     seam plane with all its control columns at the same angle. That trio is what
     a reader sees as a dart in the round, reported four times, and it is a B-rep
     defect rather than a shading one — which is why it survived every change of
     material, environment, mesh density and build.
     On a cycle the stations are `i / n`, so no station is repeated, and the
     FIRST section is appended again as the last so the skin closes on itself
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
      /* ⚠ CONSTANT RADIUS ONLY UNLESS A MEASURE WAS SUPPLIED, AND IT REFUSES
         RATHER THAN AVERAGING. Keeping the LAST section's radius and measuring
         the whole surface against it would report a meaningless deviation for a
         variable-radius run — the number would look fine and mean nothing. The
         default deviation measure's whole premise is that one radius describes
         the surface, so varying it is caller error until the caller has replaced
         that measure with one that can express a varying radius. */
      if (measure) continue;
      if (radius == null) radius = spec.radius;
      else if (Math.abs(spec.radius - radius) > 1e-12) {
        return { failed: `this builder measures against ONE radius and the sections vary (${radius} then ${spec.radius}) — a variable-radius blend needs its own deviation measure` };
      }
    }
    /* THE WRAP SECTION, on a closed run only: the first station again, so the
       skin's last column IS its first and the band closes exactly rather than
       within a tolerance. Appended rather than sampled, because sampling t=1
       would re-solve the same ball and could land a hair off it — and a hair is
       the whole defect this exists to remove. */
    if (closedRun && arcs.length) { arcs.push(arcs[0]); centres.push(centres[0]); }
    const built = blendSurfaceFromSections(arcs, degV);
    if (!built.ok) return { failed: built.reason };
    if (measure) {
      const m = measure(built.srf, n);
      /* A MEASURE THAT COULD NOT MEASURE SAYS SO, AND ITS REASON IS THE ONE
         WORTH REPORTING. Flattening every such case to "returned no number"
         loses the distinction between a measure that is absent and a surface
         that could not be sampled at all. */
      if (!m || !Number.isFinite(m.worst)) return { failed: (m && m.reason) || 'the supplied deviation measure returned no number' };
      return fold({ built, dev: m.worst, n, instrumentBound: !!m.instrumentBound, measureFloor: m.floor, deviationSigned: m.signed });
    }
    /* MEASURED AGAINST A SPINE SAMPLED FAR FINER THAN THE SECTIONS. Using the
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
  /* WHAT REFINEMENT ACTUALLY DID, kept on the result. A builder that can return
     "closer than this is not reachable" owes the caller the sequence it tried:
     a deviation that FALLS and stops short is a section ceiling, one that RISES
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
        /* ⚠ A CLOSED RUN HAS NO ENDS, AND A CALLER CANNOT SEE THAT FROM THE
           SURFACE. The two boundary rows of a closed band are the same row, so
           a caller that treats them as two open ends caps a hole that is not
           there. Reported rather than re-derived, because the caller would have
           to compare rows against a tolerance nobody chose to get back the
           answer this function was handed. */
        closed: closedRun,
        /* ⚠ WHAT THE CERTIFICATE ACTUALLY COVERS. The deviation is sampled over
           the middle of the run, because the nearest point on a FINITE spine
           polyline clamps at its ends and would report an end effect as a
           surface defect. So `metTolerance` says nothing about the outer 5% at
           each end — where, on a real edge, a corner patch takes over anyway.
           Stated in the result rather than left in a comment nobody reads. */
        certifiedSpan: [0.05, 0.95] };
    }
    // Aim with the second-order law, then verify by measuring again.
    // FOURTH ORDER, not second: err ~ C*h^4 with cubic interpolation in V, so the
    // count that meets the target is n * (err/target)^(1/4). Aiming with the
    // second-order law overshot by a large factor — 138 sections where about 30
    // reach the same tolerance.
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
    // NOT A FAILURE, AND NOT A SILENT PASS EITHER. The surface is real and
    // usable; it simply is not as close as was asked for, and the caller is
    // handed the number so it can say so rather than imply otherwise.
    metTolerance: false, rounds: maxRounds,
  };
}

/**
 * SPLICE A TANGENCY CHAIN INTO A FACE'S TRIM LOOP.
 *
 * Trimming a face back to a blend means replacing part of its boundary with the
 * tangency curve. Both live in the same (u,v) space, and the chain's two ends
 * land ON the existing loop — so the operation is: find where each end meets the
 * loop, then keep one of the two arcs between those points and replace the other
 * with the chain.
 *
 * WHICH ARC IS DROPPED IS NOT A GUESS, AND THE CONTRACT IS NARROWER THAN IT
 * FIRST LOOKS. `dropNear` must lie IN THE SPAN OF LOOP BEING REMOVED — between
 * the two places the chain lands, measured along the loop. The arc containing it
 * is the one that goes.
 *
 * ⚠ "A POINT ON THE EDGE BEING FILLETED" IS NOT SUFFICIENT. A chain landing at
 * 0.3 and 0.7 along the bottom of a unit square, with a reference at 0.05 —
 * still on that same edge — names the arc going the LONG way round, so the
 * splice keeps the 0.04 notch and discards the 0.96 face. The result is a
 * correctly-formed, completely inverted trim. Nothing about the inputs is
 * malformed; the reference simply points at loop the blend is not replacing.
 *
 * The two arcs partition the loop, so a reference outside the removed span
 * cannot be rescued by being cleverer — it genuinely names the other side. What
 * a caller wants is a point on the removed span, and for a fillet that is the
 * midpoint of the edge BETWEEN its two tangency landings. Deciding by area or by
 * winding instead would be a heuristic that fails on the first L-shaped face.
 *
 * A reference too close to being equidistant between the two arcs is refused
 * with the margin, rather than tie-broken by segment index in silence — a point
 * at the center of a square would otherwise take segment 0 and drop the bottom.
 *
 * Returns the new loop, and the two splice parameters so a caller can tell
 * whether the ends landed where it expected rather than trusting that they did.
 */
/* WHERE TWO SEGMENTS CROSS, in (u,v), or null. Endpoints count as crossings, so
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
/* ⚠⚠ A TANGENCY RUN MAY CROSS ITS FACE'S BOUNDARY RATHER THAN END ON IT, and
   that is the ORDINARY case wherever the boundary it runs into is curved.
   The run spans its own EDGE, so where the neighboring boundary curves away
   from that edge's end the run simply carries on past it. Measured on the
   plainest solid a person can draw with one curved side: a 4mm round left the
   run 1.172mm beyond the cap's boundary, and every rim of the shape refused with
   "a chain end does not reach the loop" — a true sentence about the run's
   endpoint and the wrong conclusion about the geometry.
   Such a run is not floating free. It CROSSES the loop, and where it crosses IS
   the junction. Clipping it there puts its ends on the loop, which is what the
   caller's tolerance was always asking for — so that tolerance stays exactly as
   tight as it was, and a run that genuinely floats free still refuses.
   Returns null unless the run crosses at BOTH ends, so anything that already
   met its loop takes the same path it always did. */
function clipChainToLoop(loop, chain, reach) {
  const n = loop.length;
  /* ⚠ EXTENDED FIRST, THEN CLIPPED, because a run can miss its boundary in
     EITHER direction and the two are the same question. Where the neighboring
     boundary curves AWAY the run overshoots and wants clipping; where it curves
     TOWARDS, the run stops short and wants extending — measured at 1.172mm short
     on a cap whose boundary bulges past the end of the edge being rounded.
     Both are answered by growing each end along its own direction by the reach
     the caller allows — the blend's own width, which is exactly how far a run
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
  /* ⚠ THE ENDS MUST ACTUALLY MEET THE LOOP. `nearest()` always returns a
     segment, so the old `head.seg < 0` guard was dead code and a chain floating
     free in the middle of the face spliced anyway — both ends snapping to
     whichever walls happened to be closest, cutting the face along a line the
     chain never described. It reported headGap 0.4 on a unit square and nobody
     read it. The gap is now the test, not a field. */
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
  /* A POLYLINE COVERS MORE THAN ONE PLACE — measured from its first point over
     ALL of them, so a run that closes on itself is judged by where it went and
     not by where it finished. `meetTol` is this function's own standard for two
     points being the same place, so no new constant is introduced. */
  const spansTwoPlaces = (pts) => {
    if (!pts || pts.length < 2) return false;
    for (let i = 1; i < pts.length; i++) {
      if (Math.hypot(pts[i][0] - pts[0][0], pts[i][1] - pts[0][1]) > meetTol) return true;
    }
    return false;
  };
  // Clipped to where it actually meets the boundary, when it runs past it.
  const clipped = clipChainToLoop(loop, chain, opts.reach || 0);
  /* ⚠⚠ A CLIP IS ACCEPTED ON ITS EXTENT, NOT ON ITS POINT COUNT. `length >= 2`
     reads as "this is still a segment" — but it counts ARRAY ENTRIES, and a
     clip can come back as three copies of ONE place: measured at 4.16e-17
     across, and it replaced a real 111-point chain. Everything after follows
     mechanically. `head` and `tail` land on the same loop parameter, so the
     forward interval is empty, `dropIsForward` is false, the kept arc is empty,
     and the else branch discards the face's entire boundary — leaving a
     zero-area trim loop and taking its four neighboring faces naked with it. */
  if (spansTwoPlaces(clipped)) chain = clipped;
  const head = nearest(chain[0]);
  const tail = nearest(chain[chain.length - 1]);
  if (head.d > meetTol || tail.d > meetTol) {
    return { ok: false, reason: `a chain end does not reach the loop — ${Math.max(head.d, tail.d).toExponential(3)} away, past the ${meetTol} join tolerance`, headGap: head.d, tailGap: tail.d };
  }
  // A position along the loop as one number, so the two arcs are easy to name.
  const pos = (h) => h.seg + h.t;
  const pHead = pos(head), pTail = pos(tail);
  /* ⚠ dropNear IS A POSITION ALONG THE LOOP, NOT A POINT NEAR THE FACE. "A
     point on the edge being filleted" is only right when that point falls in
     the SUB-SPAN between the two splice parameters. A chain landing at 0.3 and
     0.7 along the bottom edge with a reference at 0.05 — still on the filleted
     edge — keeps 0.04 of a unit square instead of 0.96, and the face inverts.
     Same-segment chains are what every partial-edge fillet produces, so that
     input is the ordinary one, not an exotic one.

     Refused when the reference does not land inside either arc unambiguously,
     and the MARGIN to the runner-up is returned so a caller can see a near-tie
     rather than be handed a coin toss. A reference equidistant from every side
     of a square would otherwise take segment 0 in silence. */
  const drop = nearest(dropNear);
  const pDrop = pos(drop);
  {
    /* ⚠⚠ THE TWO SEGMENTS MEETING AT A VERTEX ARE NOT TWO SIDES TO CHOOSE FROM.
       This margin exists so a reference equidistant from every side of a square
       is refused rather than silently assigned to segment 0. But a reference
       sitting ON a loop vertex is equidistant from the two segments that share
       it BY CONSTRUCTION, and picking either yields the SAME position along the
       loop — so there is nothing ambiguous about it and nothing downstream can
       tell the two choices apart.

       That is the ordinary case, not a corner one: a reference is sampled at a
       point of the edge being dropped, and where that edge is CURVED the loop
       carries it as many short segments, so the sample lands on one of their
       shared vertices. A straight edge is one long segment and the sample falls
       in its interior, which is why only a curved edge reaches this. Measured at
       9.49e-18 against a 1e-7 margin — a tie to the last bit, refused as a
       coin toss.

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
  // Walking FORWARD from head to tail wraps or does not; the dropped arc is
  // whichever of the two contains pDrop.
  const inForward = (x) => (pHead <= pTail ? (x >= pHead && x <= pTail) : (x >= pHead || x <= pTail));
  const dropIsForward = inForward(pDrop);
  // Keep the arc that does NOT contain the drop point, walking from tail back
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
  /* ⚠ AND THE COLLAPSE GUARD HAS TO ASK THE SAME QUESTION. `out.length < 3`
     is the last net here, and it counts points too — the degenerate output above
     has exactly three of them, one from `head`, one from `tail` and one from the
     reversed chain, so `3 < 3` never fired. A loop needs three distinct PLACES,
     not three entries. */
  if (out.length < 3 || !spansTwoPlaces(out)) return { ok: false, reason: 'the splice collapsed the loop' };
  return { ok: true, loop: out, headAt: pHead, tailAt: pTail, droppedForward: dropIsForward, headGap: head.d, tailGap: tail.d };
}
function pointAtLoop(loop, h) {
  const a = loop[h.seg], b = loop[(h.seg + 1) % loop.length];
  return [a[0] + (b[0] - a[0]) * h.t, a[1] + (b[1] - a[1]) * h.t];
}

/**
 * THE CHAMFER CROSS-SECTION — the same section, cut straight instead of round.
 *
 * A chamfer takes the ARC's two endpoints and joins them with a line. That is
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
 * THE CHAMFER'S SECTION BUILDER, AS A `blendSurfaceToTolerance` HOOK.
 *
 * The completed section has to be built FIRST and cut second: `chamferSectionArc`
 * needs `tangencyA`/`tangencyB`, which only `envelopeSection` derives, while the
 * generator handed to the builder yields a raw ball SPEC. Composing them in the
 * wrong order returns null for every station, which the builder reports as "the
 * caller could not supply a section" — a total refusal that looks nothing like the
 * missing shape it actually is.
 */
/**
 * A CURVATURE-CONTINUOUS SECTION — the "smooth" third state beside the rolling
 * ball and the chamfer.
 *
 * A circular fillet is G1: the blend meets each flat with a matching TANGENT and
 * a curvature that jumps from 0 on the face to 1/r on the blend, at a line you
 * can find on any reflective surface. G2 removes that jump by giving the section
 * ZERO curvature where it lands on each face.
 *
 * A rational quadratic cannot do it — a conic's end curvature is never zero — so
 * the section is a QUINTIC. The condition is purely a statement about control
 * points: a Bezier's curvature at an end vanishes exactly when the first three
 * (or last three) control points are COLLINEAR, so
 *
 *     P0 = A,  P1 = A + a*t,  P2 = A + 2a*t
 *     P5 = B,  P4 = B + b*u,  P3 = B + 2b*u
 *
 * is G2 against both faces by construction rather than by fitting, with `t` and
 * `u` the in-plane directions from each tangency point toward the sharp corner
 * the blend replaces.
 *
 * ⚠ AND THE SURFACE STAYS G2 BETWEEN THE SECTIONS, which is not obvious and is
 * the reason this can be skinned at all. `blendSurfaceFromSections` interpolates
 * each control ROW independently along the edge. Collinearity survives that
 * because it is LINEAR in the data: row2 - row0 = 2*(row1 - row0) holds at every
 * station, the two differences are interpolated by the same basis over the same
 * knots, and a basis applied to twice a data set gives twice the result. So the
 * relation holds at every v, exactly, not to a tolerance.
 *
 * THE FOOTPRINT IS THE BALL'S. `A` and `B` are the same tangency points a
 * rolling ball of that radius would produce, so "radius" keeps meaning the same
 * thing on screen — how far onto each face the blend reaches — and only the
 * profile between them changes.
 */

/**
 * HOW FULL THE PROFILE IS, and it is NOT one number.
 *
 * `alpha` places the inner control points along each leg toward the corner, and
 * the value that makes the quintic sit closest to the circular arc of the same
 * radius — which is what keeps a smooth blend the same SIZE on screen as the
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
 * ⚠ A WIDE CORNER GETS A THINNER PROFILE, which is the opposite of the intuition
 * that a flatter corner needs more filling. It falls out of the footprint being
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
 * HOW FAR A BUILT SMOOTH BLEND DEPARTS FROM ITS OWN EXACT SECTIONS.
 *
 * Same reasoning as `chamferFlatnessDeviation`, and for the same reason: the
 * default radius measure reports |distance-to-spine - radius|, and a smooth
 * section is deliberately NOT at constant distance from the spine, so a perfect
 * one measures as a large failure that refinement can never reduce.
 *
 * ⚠ MEASURED AGAINST THE SURFACE'S OWN ISOCURVE, not against a section looked up
 * by parameter — sections are interpolated at chord-length parameters, so a
 * caller's `sectionAt(t)` and the surface's own v do not name the same station.
 * Everything the exact section needs is recoverable from the isocurve itself:
 * its two endpoints, and its two end tangents, which are the directions the
 * supporting faces impose. So the section is rebuilt from the surface and the
 * surface is compared against it, and neither the parameterisation nor the
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
  /* ⚠ SAME RULE AS `chamferFlatnessDeviation`'s OWN: an unmeasured surface must
     not certify as an exact one. Every station here can `continue` too — a
     tangent that will not normalise, two end tangents too nearly parallel to
     locate a corner, a corner behind the start — and `worst` left at its initial
     0 would report a smooth blend that matches its exact profile perfectly, off
     no samples at all, which stops refinement dead. */
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
 * HOW FAR A BUILT CHAMFER DEPARTS FROM ITS OWN EXACT CHORDS.
 *
 * ⚠ THE DEFAULT DEVIATION MEASURE CANNOT JUDGE A CHAMFER, and it fails in the
 * direction that looks like success. `blendRadiusDeviation` reports
 * |distance-to-spine - radius|; a chamfer's chord midpoint sits at
 * r*cos(sweep/2) from the ball center, so a PERFECT chamfer on a 90 degree edge
 * measures r*(1-cos45) ~ 0.293r of "deviation" — about 1.46mm on a 5mm chamfer
 * against a 0.01mm tolerance. The refinement loop then chases a number that can
 * never fall, ramps to `maxSections`, and returns its best effort with
 * `metTolerance: false` and a figure that means nothing. Any panel reporting that
 * figure calls a correct chamfer a failed one.
 *
 * What IS exact for a chamfer: at every station the true surface is the straight
 * segment between the two tangency points. A single section is therefore exact by
 * construction (degree 2, unit weights, midpoint at the true midpoint), and the
 * only error left is the interpolation BETWEEN stations — which is the same
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
  /* ⚠⚠ MEASURED AGAINST THE SURFACE'S OWN SECTION, NOT AGAINST A SECTION LOOKED
     UP BY PARAMETER. This asked the caller's `sectionAt(t)` for the chord to
     compare against and then evaluated the SURFACE at that same t — which is only
     the same station while the two parameterisations agree. They do not: sections
     are interpolated at CHORD-LENGTH parameters, so on any run where the stations
     are unevenly spaced the measure compares one place on the surface against the
     chord belonging to another.

     The error that produces does not shrink with refinement, because it is not an
     error in the surface: a freeform rim floored at 2.6mm through 106, 426 and
     1601 sections while the fillet on the same rim reached 0.0098mm. It reads
     exactly like a chamfer that will not converge.

     A section is flat if it is straight between ITS OWN two ends, which is a
     question about the surface alone. Taking the endpoints from the same isocurve
     being sampled removes the parameterisation from the measure entirely, and
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
      // Distance to the SEGMENT, clamped: past either end the nearest point on
      // the true chamfer is its endpoint, and an unclamped line distance would
      // read zero for a point that has run off the end of the chord entirely.
      let sPar = dot(sub(q, a), ab) / abLen2;
      sPar = sPar < 0 ? 0 : sPar > 1 ? 1 : sPar;
      const d = len(sub(q, add(a, mul(ab, sPar))));
      measured++;
      if (d > worst) worst = d;
    }
  }
  /* ⚠ A SURFACE NOBODY COULD MEASURE IS NOT A FLAT ONE. Every station above can
     `continue` — an evaluator that returns nothing, a chord that degenerates —
     and with `worst` still sitting at its initial 0 the loop then falls out and
     reports `ok: true, worst: 0`: a PERFECTLY flat chamfer, certified, from zero
     samples. `blendSurfaceToTolerance` reads that as tolerance met on the first
     attempt and stops refining, so the one surface that most needed another look
     is the only one that gets none. Unknown must not read as perfect. */
  if (measured === 0) {
    return { ok: false, reason: 'no station on this surface could be measured — every sample either failed to evaluate or had a degenerate chord, so its flatness is UNKNOWN rather than perfect' };
  }
  return { ok: true, worst, instrumentBound: false, floor: 0, stations: measured };
}

/**
 * IS A BUILT BLEND ROUND OR FLAT? THE ORACLE A RECORD CANNOT PROVIDE.
 *
 * A fillet and a chamfer differ only in the shape spanning the gap, and every
 * other observable — tangency points, setback, footprint, the record's own type
 * field — is identical between them. So a check that reads the record, or the
 * parameters, or the footprint, passes for both and distinguishes neither.
 *
 * This reads the SURFACE. Three points across one section determine a circle;
 * the residual of the remaining samples against that circle says whether the
 * section is really an arc, and the fitted radius says which arc. A chord returns
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
  // Three collinear points span no plane, which IS the flat answer rather than a
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
 * THE AREA OF A CORNER PATCH, BY TWO INDEPENDENT ROUTES.
 *
 * The spherical triangle bounded by three touch directions has area
 * r^2 * (A + B + C - pi) — Girard's excess — where A, B, C are the triangle's
 * INTERIOR angles.
 *
 * ⚠ THE INTERIOR ANGLE IS NOT THE DIHEDRAL, and a cube cannot tell the
 * difference. On three perpendicular faces both are 90 degrees, so an octant
 * result confirms nothing about which quantity the formula wants. On a skewed
 * trihedron they separate plainly: interior angles 88.52 / 68.99 / 100.87
 * against dihedrals 78.92 / 87.38 / 111.12. The interior angle is measured
 * between the two arcs LEAVING a vertex — each other direction projected into
 * the tangent plane there — which is what this computes and what Girard needs.
 *
 * The dihedral is the supplement of the angle between two touch directions, and
 * is a different number again; it belongs to the edge, not to the corner.
 *
 * CHECKED AGAINST VAN OOSTEROM & STRACKEE, "The solid angle of a plane
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
    // The second opinion, and the gap between them. Not a formality: the two
    // share no algebra, so a drift in either shows up here before it shows up
    // in a surface.
    solidAngle: vanOosterom,
    agreement: Math.abs(excess - vanOosterom),
  };
}
