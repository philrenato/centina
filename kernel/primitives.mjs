// Exact-rational primitives — Piegl & Tiller Ch. 7 (conic arcs) and A8.1
// (surface of revolution), plus the simple ruled/extruded surface (Ch. 8).
//
// The circle/arc technique: split the sweep into spans of at most 90°, and
// for each span use the closed-form tangent-intersection point — no general
// line-line intersection needed, because in the span's own local (xHat,yHat)
// basis the middle control point is always P0 + r*tan(dtheta/2)*tangentAtP0,
// a fact that's rotation-invariant so it works for every 90°-or-less span at
// any starting angle. This is the same construction that produces the
// standard 9-control-point unit circle (degree 2, weights 1 and root2/2
// alternating), checked against curve.mjs in test/curve-surface.test.mjs.

import { add, sub, scale, dot, cross, length, normalize, anyPerpendicular } from './vec3.mjs';
import { degreeElevateCurve, joinCurvesC0, insertKnot } from './knots.mjs';
import { globalCurveInterp } from './interpolate.mjs';

const MAX_ARC_SPAN = Math.PI / 2;

function arcSpanPoints(center, xHat, yHat, radius, angleStart, dtheta) {
  const cosA = Math.cos(angleStart), sinA = Math.sin(angleStart);
  const p0 = add(center, add(scale(xHat, radius * cosA), scale(yHat, radius * sinA)));
  const tangentAtStart = add(scale(xHat, -sinA), scale(yHat, cosA)); // unit
  const angleEnd = angleStart + dtheta;
  const cosB = Math.cos(angleEnd), sinB = Math.sin(angleEnd);
  const p2 = add(center, add(scale(xHat, radius * cosB), scale(yHat, radius * sinB)));
  const w1 = Math.cos(dtheta / 2);
  const p1 = add(p0, scale(tangentAtStart, radius * Math.tan(dtheta / 2)));
  return { p0, p1, p2, w1 };
}

function arcKnots(narcs) {
  const knots = [0, 0, 0];
  for (let k = 1; k < narcs; k++) { knots.push(k, k); }
  knots.push(narcs, narcs, narcs);
  return knots;
}

export function makeLine(p0, p1) {
  return { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...p0, 1], [...p1, 1]] };
}

// Exact rational arc: center + orthonormal in-plane basis (xAxis, yAxis) +
// radius + start angle + signed sweep (radians). A full circle is sweep=2*PI.
// `minSegments` forces more arc spans than the sweep alone would need (each
// span is still <=90 deg, so the closed-form construction stays exact
// regardless of span size). This is how a circle is rebuilt with more control
// points without leaving the exact rational representation: more spans means
// more control points, not an approximating refit. Default 1: no forced extra
// subdivision.
export function makeArc(center, xAxis, yAxis, radius, angleStart, sweep, minSegments = 1) {
  const narcs = Math.max(minSegments, Math.ceil(Math.abs(sweep) / MAX_ARC_SPAN));
  const dtheta = sweep / narcs;
  const ctrlPts = [];
  let angle = angleStart;
  const first = arcSpanPoints(center, xAxis, yAxis, radius, angle, dtheta);
  ctrlPts.push([...first.p0, 1]);
  for (let k = 0; k < narcs; k++) {
    const seg = arcSpanPoints(center, xAxis, yAxis, radius, angle, dtheta);
    ctrlPts.push([...seg.p1, seg.w1]);
    ctrlPts.push([...seg.p2, 1]);
    angle += dtheta;
  }
  return { degree: 2, knots: arcKnots(narcs), ctrlPts };
}

// `segments` (default 4, the minimum for a full 360 deg sweep at
// MAX_ARC_SPAN=90 deg) can be raised to rebuild the circle with more control
// points — still an exact circle at any segment count, so it stays usable as
// a Sweep1/Extrude/Loft/Revolve profile at any rebuild level.
export function makeCircle(center, xAxis, yAxis, radius, segments = 4) {
  return makeArc(center, xAxis, yAxis, radius, 0, 2 * Math.PI, segments);
}

// Ellipse — a non-uniform scale of a unit circle. A NURBS rational curve is
// exactly preserved under any affine map applied per control point: the
// rational basis functions sum to 1, so C(t) = sum(R_i(t) P_i) is an affine combination of its control
// points, and any affine T satisfies T(sum(R_i P_i)) = sum(R_i T(P_i)).
// makeArc/makeCircle builds each control point as
//   center + xHat*r*cos + yHat*r*sin  (positions)  and  weight = cos(dtheta/2)
// with r=1 here, so passing a non-unit in-plane basis (xAxis scaled by
// radiusX, yAxis scaled by radiusY) applies exactly the affine map
// (u,v) -> center + u*radiusX*xAxis + v*radiusY*yAxis to every unit-circle
// control point — the exact ellipse. arcSpanPoints'
// tangent-line intersection point p1 (a control point, not a curve
// point) transforms correctly for the same reason (an affine map preserves
// the tangent-line intersection). The radiusX===radiusY case is bit-for-bit
// identical to makeCircle (test/ellipse.test.mjs). The weight formula does
// not assume |xAxis|==|yAxis|: those magnitudes scale only the position
// terms, never the (dtheta-only) weights.
export function makeEllipse(center, xAxis, yAxis, radiusX, radiusY, segments = 4) {
  return makeCircle(center, scale(xAxis, radiusX), scale(yAxis, radiusY), 1, segments);
}

// Ellipsoid profile — the half-ellipse meridian arc (pole to pole) whose
// surface of revolution around `polarAxis` is an exact ellipsoid, by the same
// per-control-point affine map as makeEllipse (a meridian is a non-uniform
// scale of a unit half-circle). Reuses makeArc's
// exact conic-arc construction for a half sweep (angleStart=-PI/2, sweep=PI),
// with the arc's own local x-axis (scaled by `equatorialRadius`) mapped to the
// equatorial/radial direction and its local y-axis (scaled by `polarRadius`)
// to the polar direction (the revolve axis). Because a NURBS curve is exactly
// preserved under any per-control-point affine map (the same identity
// makeEllipse relies on), passing the two scaled axes to makeArc applies the
// exact map (u,v) -> center + u*eR*equatorialAxis + v*pR*polarAxis to every
// control point — the true half-ellipse, never an approximation.
//
// Both endpoints land on the revolve axis (the two poles) to machine
// precision: the equatorial component of an endpoint is r*cos(-PI/2) and
// r*cos(+PI/2), each ~1e-16*r (cos of a floating-point PI/2), well under
// revolve()'s own 1e-9 pole-detection threshold, so revolve collapses each
// end-row exactly onto the axis. The apex (angle 0) reaches exactly the
// equatorial radius. Revolving 2*PI: for axis=z, equatorialAxis=x, a surface
// point is (eR*cosθ*cosφ, eR*cosθ*sinφ, pR*sinθ), satisfying the true
// ellipsoid equation (x/eR)^2+(y/eR)^2+(z/pR)^2 = cos²θ+sin²θ = 1 exactly at
// every (θ,φ) including the poles (test/ellipsoid.test.mjs).
export function makeEllipsoidProfile(center, equatorialAxis, polarAxis, equatorialRadius, polarRadius, minSegments = 2) {
  return makeArc(center, scale(equatorialAxis, equatorialRadius), scale(polarAxis, polarRadius), 1, -Math.PI / 2, Math.PI, minSegments);
}

// Squircle (2D) — a closed, degree-3 NURBS curve from an 8-control-point
// corner-pull cage: a superellipse-like family parameterized by
// `softness` ∈ [0,1], from a square-ish rounded shape (0) toward a
// near-ellipse (1). It is a periodic uniform cubic B-spline over a control
// cage, where the curve stays inside its cage and rounds the corners, not an
// exact conic arc. The 8 cage points are the 4 edge midpoints (±hw,0),
// (0,±hh) plus the 4 corners at (±cs*hw, ±cs*hh), where `cs` (the corner
// control-point radial scale) is the single softness knob:
//   cs = 1.25 - 0.5*softness  (softness 0 -> 1.25, softness 1 -> 0.75)
// A larger cs pushes the corner control points further out along the
// diagonal, so the curve reaches nearer the true square corner (square-ish);
// a smaller cs pulls them in toward the round of a circle. cs=1.25 places the
// corner curve-point at ~the true square corner (the most square-reading
// smooth member without corner overshoot); cs≈0.71 would be a numeric circle.
// The 0.75 range keeps every member convex, simple (no self-intersection),
// and cusp-free at both extremes.
//
// Closed-and-smooth representation: a periodic B-spline's seam must stay C2
// and its two ends must coincide exactly, but this kernel evaluates only
// clamped curves. So the cage is wrapped into an over-padded uniform periodic
// B-spline (n+4p control points, so one full central period has a full valid
// p-control-point neighborhood on both sides — clamping at an unclamped
// curve's own validity boundary pulls in the invalid tail control points and
// produces a zero-speed seam cusp), then that central period is clamped at
// both its interior boundaries via insertKnot (Boehm A5.1,
// geometry-preserving) up to multiplicity degree+1 and sliced out as a
// standalone clamped curve, its domain renormalized to [0, n]. Because the
// source is periodic, the extracted curve's first control point equals its
// last (closed, zero gap) and the seam carries the same tangent/curvature as
// everywhere else. test/squircle.test.mjs checks closure, simplicity,
// cusp-freedom and a monotone square→circle progression across softness.
function squircleCageToClosedCubic(Q) {
  const p = 3, n = Q.length;
  const M = n + 4 * p; // heavy both-sides padding so the extracted period has full valid support
  const ctrlPts = [];
  for (let i = 0; i < M; i++) { const q = Q[((i - p) % n + n) % n]; ctrlPts.push([q[0], q[1], q[2], 1]); }
  const knots = [];
  for (let i = 0; i < M + p + 1; i++) knots.push(i);
  let c = { degree: p, knots, ctrlPts };
  const a = 2 * p, b = a + n; // one full central period, well inside [p, M]
  c = insertKnot(c, a, p); // uniform interior knot has mult 1 -> +p reaches mult p+1 (clamped)
  c = insertKnot(c, b, p);
  const K = c.knots, P = c.ctrlPts;
  const i0 = K.findIndex((v) => Math.abs(v - a) < 1e-9);
  let i1 = -1;
  for (let i = K.length - 1; i >= 0; i--) if (Math.abs(K[i] - b) < 1e-9) { i1 = i; break; }
  const subKnots = K.slice(i0, i1 + 1);
  const ncp = subKnots.length - p - 1;
  const ctrl = P.slice(i0, i0 + ncp).map((pt) => pt.slice());
  const k0 = subKnots[0], span = subKnots[subKnots.length - 1] - k0;
  const nk = subKnots.map((v) => ((v - k0) / span) * n);
  return { degree: p, knots: nk, ctrlPts: ctrl };
}
export function makeSquircle2D(center, xAxis, yAxis, halfWidth, halfHeight, softness = 0.5) {
  const s = Math.min(1, Math.max(0, softness));
  const cs = 1.25 - 0.5 * s;
  const hw = halfWidth, hh = halfHeight;
  const local = [
    [hw, 0], [cs * hw, cs * hh], [0, hh], [-cs * hw, cs * hh],
    [-hw, 0], [-cs * hw, -cs * hh], [0, -hh], [cs * hw, -cs * hh],
  ];
  const Q = local.map(([lx, ly]) => add(add([...center], scale(xAxis, lx)), scale(yAxis, ly)));
  return squircleCageToClosedCubic(Q);
}

// Surface of revolution (A8.1). `profile` is a NurbsCrv (degree/knots become
// the surface's U direction); axisPoint+axisDir define the rotation axis;
// angleStart/sweep are radians. Handles profile control points that lie on
// the axis (a degenerate "pole" row, e.g. a revolve profile that touches its
// own axis, such as a dome apex).
//
// Exactness at a pole row. A8.1's exactness holds only if every row's
// V-direction weight function (as a function of the sweep parameter v,
// ignoring the row's radius) has the identical shape for every row blended
// together at a given U — the alternating (1, cos(dtheta/2), 1,
// cos(dtheta/2), ...) column pattern `arcSpanPoints` builds for an ordinary
// row. A pole row with a uniform weight at every column evaluates correctly
// in isolation but breaks the surface's row-blend identity between knot
// corners (errors of ~1-5%). So the pole row runs through the same
// `arcSpanPoints` construction with radius=0 (any perpendicular basis works;
// it is multiplied by zero). A zero-radius arc collapses every column to the
// pole point exactly, and carries the alternating weight shape, so
// `S(u,v) = O(u) + Rotate(Q(u), v)` (Q(u) the profile's radial offset at u)
// holds at every (u,v). The identity does not depend on rows sharing a local
// basis direction, so a profile that also crosses to the axis's other side
// needs no separate handling. test/revolve-pole-exactness.test.mjs checks
// the analytic sphere to 1e-9 relative error at non-corner samples.
export function revolve(profile, axisPoint, axisDir, angleStart, sweep) {
  const axis = normalize(axisDir);
  const narcs = Math.max(1, Math.ceil(Math.abs(sweep) / MAX_ARC_SPAN));
  const dtheta = sweep / narcs;
  const knotsV = arcKnots(narcs);

  const ctrlNet = profile.ctrlPts.map(([px, py, pz, pw]) => {
    const P = [px, py, pz];
    const rel = sub(P, axisPoint);
    const along = dot(rel, axis);
    const O = add(axisPoint, scale(axis, along));
    const X = sub(P, O);
    let r = length(X);
    let xHat;
    if (r < 1e-9) {
      // Pole: point is on the axis. Radius 0 collapses every column of the
      // construction below to the pole point exactly, regardless of the
      // basis direction chosen here.
      r = 0;
      xHat = anyPerpendicular(axis);
    } else {
      xHat = scale(X, 1 / r);
    }
    const yHat = cross(axis, xHat); // unit: axis and xHat are orthonormal
    const row = [];
    let angle = angleStart;
    const first = arcSpanPoints(O, xHat, yHat, r, angle, dtheta);
    row.push([...first.p0, pw]);
    for (let k = 0; k < narcs; k++) {
      const seg = arcSpanPoints(O, xHat, yHat, r, angle, dtheta);
      row.push([...seg.p1, seg.w1 * pw]);
      row.push([...seg.p2, pw]);
      angle += dtheta;
    }
    return row;
  });

  return { degU: profile.degree, knotsU: profile.knots, degV: 2, knotsV, ctrlNet };
}

// Ruled/extruded surface (Ch. 8): profile translated along a direction.
// Degree-1 in V — a straight translation needs no arcs, and the weight of a
// translated point is unchanged (translation is an affine, not projective,
// move on the homogeneous form). Still a ruled surface (straight lines
// between corresponding U-parameter points on the two rows) with a nonzero
// draftAngleDeg — only the top row's coordinates change — so degV stays 1.
//
// draftAngleDeg (default 0): a molding-style taper, simplified. A true draft
// angle offsets the profile by a perpendicular curve offset before extruding
// it, which is exact for any profile shape including concave ones. This
// instead grows/shrinks each control point radially from the profile's
// centroid by distance*tan(angle) — exact for a circle/regular polygon
// (every point is equidistant from the centroid, so a uniform radial grow is
// the true offset), close for most convex profiles, but not a true offset
// for a concave or highly irregular curve.
//
// `vDegree` (default 1: exactly 2 control points per row, a literal ruled
// surface). When higher, each U-row's straight bottom-to-top ruling line is
// degree-elevated via kernel/knots.mjs's degreeElevateCurve — exact and
// shape-preserving (elevating a straight line's degree adds control points
// along the same line) — giving draggable intermediate control points along
// the extrusion height.
export function extrude(profile, direction, distance, draftAngleDeg = 0, vDegree = 1) {
  const d = scale(normalize(direction), distance);
  let topPts = profile.ctrlPts;
  if (draftAngleDeg) {
    const n = profile.ctrlPts.length;
    const cx = profile.ctrlPts.reduce((s, p) => s + p[0], 0) / n;
    const cy = profile.ctrlPts.reduce((s, p) => s + p[1], 0) / n;
    const cz = profile.ctrlPts.reduce((s, p) => s + p[2], 0) / n;
    const delta = distance * Math.tan(draftAngleDeg * Math.PI / 180);
    topPts = profile.ctrlPts.map(([x, y, z, w]) => {
      const rx = x - cx, ry = y - cy, rz = z - cz;
      const r = Math.hypot(rx, ry, rz);
      if (r < 1e-9) return [x, y, z, w]; // a control point at the centroid has no radial direction to grow along — left untapered
      const s = (r + delta) / r;
      return [cx + rx * s, cy + ry * s, cz + rz * s, w];
    });
  }
  if (vDegree <= 1) {
    const ctrlNet = profile.ctrlPts.map(([x, y, z, w], i) => [
      [x, y, z, w],
      [topPts[i][0] + d[0], topPts[i][1] + d[1], topPts[i][2] + d[2], w],
    ]);
    return { degU: profile.degree, knotsU: profile.knots, degV: 1, knotsV: [0, 0, 1, 1], ctrlNet };
  }
  let knotsV = null;
  const ctrlNet = profile.ctrlPts.map(([x, y, z, w], i) => {
    const bottom = [x, y, z, w];
    const top = [topPts[i][0] + d[0], topPts[i][1] + d[1], topPts[i][2] + d[2], w];
    const elevated = degreeElevateCurve({ degree: 1, knots: [0, 0, 1, 1], ctrlPts: [bottom, top] }, vDegree);
    knotsV = elevated.knots; // identical for every row (same input degree/knots each time)
    return elevated.ctrlPts;
  });
  return { degU: profile.degree, knotsU: profile.knots, degV: vDegree, knotsV, ctrlNet };
}

// Fillet (polygon corner rounding) — the same closed-form conic-arc
// construction as arcSpanPoints/makeArc above (P&T Ch.7: two tangent points,
// their tangent-line intersection point, weight = cos(halfSweep)), solved
// from the other set of knowns. arcSpanPoints starts from a center and start
// angle and derives the tangent-line intersection (`p1`); at a polygon corner
// that intersection is the vertex itself, since the two tangent lines to the
// fillet arc are the two polygon edges. So this goes apex + the two edge
// directions -> the two tangent points + the rational weight, with trim =
// radius*tan(halfPhi).
//
// Works for both convex and reflex corners (a Star polygon's inner vertices)
// via the signed turn angle (atan2 of a cross/dot against the polygon's plane
// normal) rather than an interior-angle formula that only holds for a convex
// turn. `planeNormal` must be a unit vector normal to the polygon's plane
// (e.g. cross(xAxis,yAxis)); it fixes only the sign convention for a left vs
// right turn. dIn/dOut/vertex are assumed to lie in that plane.
export function filletCornerArc(vertex, prevPt, nextPt, radius, planeNormal) {
  const dIn = normalize(sub(vertex, prevPt)); // direction of travel arriving at vertex
  const dOut = normalize(sub(nextPt, vertex)); // direction of travel leaving vertex
  const sinPhi = dot(cross(dIn, dOut), planeNormal); // signed sine of the turn angle
  const cosPhi = dot(dIn, dOut);
  const phi = Math.atan2(sinPhi, cosPhi); // signed turn angle in (-PI, PI]; >0 = convex/left turn, <0 = reflex/right turn (CCW loop convention)
  const halfPhi = Math.abs(phi) / 2;
  if (halfPhi < 1e-7) return { ok: false, reason: 'the path is straight here — nothing to round' };
  if (Math.abs(phi) > Math.PI - 1e-6) return { ok: false, reason: 'a near-180° reversal has no well-defined fillet' };
  const trim = radius * Math.tan(halfPhi); // same tangent length as arcSpanPoints' p1 offset
  const weight = Math.cos(halfPhi); // same rational weight as arcSpanPoints' `w1`
  const p0 = sub(vertex, scale(dIn, trim)); // trimmed back along the incoming edge
  const p2 = add(vertex, scale(dOut, trim)); // trimmed forward along the outgoing edge
  return { ok: true, p0, apex: vertex, p2, weight, trim, turnAngle: phi };
}

// Builds a filleted closed vertex loop (a regular/star Polygon's vertex
// array, in order) — every corner rounded by the same radius — as a segment
// list, not one NURBS curve: `{type:'line', a, b}` /
// `{type:'arc', p0, apex, p2, weight}` in path order, forming a closed loop.
// Refuses when the radius is too large for the polygon's edge lengths and
// turn angles: two neighboring corners sharing one edge would trim past each
// other (or exactly meet). `maxSafeRadius` is reported with the refusal so a
// caller can clamp.
export function filletPolygon(points, radius, planeNormal) {
  const n = points.length;
  if (n < 3) return { ok: false, reason: 'need at least 3 points to fillet a closed loop' };
  if (!(radius > 0)) return { ok: false, reason: 'fillet radius must be positive' };
  const corners = points.map((v, i) => filletCornerArc(v, points[(i - 1 + n) % n], points[(i + 1) % n], radius, planeNormal));
  for (const c of corners) if (!c.ok) return { ok: false, reason: c.reason };
  const edgeLens = points.map((v, i) => length(sub(points[(i + 1) % n], v)));
  let worstRatio = 0; // needed/edgeLen, across every edge — > 1 means this radius is too large somewhere
  for (let i = 0; i < n; i++) {
    const needed = corners[i].trim + corners[(i + 1) % n].trim;
    worstRatio = Math.max(worstRatio, needed / edgeLens[i]);
  }
  /* `maxSafeRadius` is null, not 0, when nothing could be measured. A
     zero-length edge makes `needed / edgeLen` Infinity, and `radius / Infinity`
     is 0, which would read as a measured largest radius of 0. null means there
     is no retriable radius; callers' `> 1e-6` / `> 0` clamp guards act on it. */
  if (worstRatio >= 1 - 1e-9) {
    return { ok: false, reason: 'fillet radius is too large for this polygon — neighboring corners would overlap', maxSafeRadius: Number.isFinite(worstRatio) ? radius / worstRatio * 0.999 : null };
  }
  const segments = [];
  for (let i = 0; i < n; i++) {
    const c = corners[i];
    segments.push({ type: 'arc', p0: c.p0, apex: c.apex, p2: c.p2, weight: c.weight });
    const next = corners[(i + 1) % n];
    segments.push({ type: 'line', a: c.p2, b: next.p0 });
  }
  return { ok: true, segments };
}

// Open-rail fillet. A degree<=1 rail's sweep path (kernel/sweep.mjs,
// `sweep1Rigid`) is a ruled (linear) blend between two rings per
// control-point span. A mitered interior corner sets both end rings of a span
// to a shared bisector orientation, and a linear blend between two
// same-radius circles at mutually tilted orientations contracts mid-span by
// ~cos(half the relative tilt) (test/sweep-interior-corner-miter.test.mjs
// measures a 4.58/5.0 waist). No choice of miter frame avoids this while the
// corner is a C0 join inside one ruled surface. Rounding the corner into the
// rail before sweeping gives a degree-2 rail, which routes through
// `sweep1RigidResampled` (the arc-length-resample path; see
// `railFrameOriginsExact`) and has no shared-frame C0 join anywhere.
//
// A separate function from `filletPolygon` because that one assumes (1)
// closed-loop indexing (`(i-1+n)%n`/`(i+1)%n` on every vertex, no open end);
// (2) all-or-nothing refusal (any one collinear corner aborts the whole
// polygon — routine on an arbitrary rail, never on a regular/star Polygon);
// (3) one shared `planeNormal`, correct only for a planar vertex set — a
// non-planar rail needs a different plane normal at each corner.
//
// Per-corner local normal: `planeNormal = normalize(cross(dIn, dOut))`,
// computed at each corner from its own two edge directions. This makes
// `filletCornerArc`'s `sinPhi = dot(cross(dIn,dOut), planeNormal)` reduce to
// `|crossVec| = sin(turnAngle)`, the non-negative turn-angle magnitude, at
// every corner. On a planar closed loop it produces the same arc geometry
// (p0/apex/p2/weight/trim) as `filletPolygon`'s shared normal:
// `filletCornerArc` builds the arc from `Math.abs(phi)/2` only, so a flipped
// normal at a reflex corner flips only the sign of the reported `turnAngle`
// (test/fillet-open-polyline.test.mjs cross-checks against `filletPolygon`).
//
// Collinear / near-180 guard, checked before any fillet math. The per-corner
// normal needs `normalize(cross(dIn,dOut))`, and a collinear vertex (turn=0)
// or a near-180 fold-back (turn=~PI) makes that cross product nearly zero,
// so its normalization would be NaN. Both cases are detected by
// `length(cross(dIn,dOut))` against a small epsilon, and the vertex passes
// through unfilleted rather than aborting the rail.
//
// Open vs closed edge budget: an open rail's first/last vertex has one
// adjacent edge and is never a corner (it passes through, `{p0:v, p2:v}`,
// zero trim). Every edge's trim-budget check (`needed = tanHalf[i] +
// tanHalf[i+1]` at radius=1, scaled by the requested radius) therefore
// budgets an end edge against its one interior corner with no separate
// branch. A closed rail reduces to `filletPolygon`'s two-corner-per-edge
// budget.
//
// Auto-clamp: on a too-large radius, reports a retriable `maxSafeRadius`
// (`radius/worstRatio*0.999`, as `filletPolygon`); shrinking toward 0 always
// eventually fits.
//
// Zero-length remainder: on an open rail an end edge has only one corner
// trimming it, so its remaining straight run can be zero-length even at a
// safe radius. Any line segment shorter than FILLET_ZERO_LEN_EPS is omitted
// from the segment chain rather than passed to `joinCurvesC0` /
// `filletSegmentsToCurve` as a zero-length span.
//
// Scope: one radius for the whole rail, as `filletPolygon` applies one
// radius to every corner; no per-corner radius.
//
// `opts.cornerFilter`: an optional `Set` of vertex indices; a corner not in
// the set passes through unfilleted, through the same branch as the
// collinear skip. Default: fillet every corner. `sweep1Rigid`'s miter-limit
// fallback (kernel/sweep.mjs) uses it to fillet only the corners whose
// true-miter stretch would exceed the limit, leaving the rest to be
// true-mitered.
const FILLET_COLLINEAR_EPS = 1e-9; // on |cross(dIn,dOut)|, both unit vectors — catches turn=0 (collinear) and turn=PI (fold-back) alike
const FILLET_ZERO_LEN_EPS = 1e-9;
export function filletOpenPolyline(points, radius, opts = {}) {
  const closed = !!opts.closed;
  const cornerFilter = opts.cornerFilter || null;
  const n = points.length;
  if (closed) {
    if (n < 3) return { ok: false, reason: 'need at least 3 points to fillet a closed loop' };
  } else if (n < 2) {
    return { ok: false, reason: 'need at least 2 points to fillet an open rail' };
  }
  if (!(radius > 0)) return { ok: false, reason: 'fillet radius must be positive' };

  // Pass 1 — radius-independent geometry: which vertices are corners (two
  // neighbors, a real turn), which pass straight through (an open rail's
  // endpoints, or a collinear/near-180 vertex), and each corner's `tanHalf`
  // (trim per unit radius, i.e. `filletCornerArc`'s `trim` at radius=1).
  const eff = points.map((v, i) => {
    if (!closed && (i === 0 || i === n - 1)) {
      return { isCorner: false, p0: v, p2: v, tanHalf: 0 };
    }
    if (cornerFilter && !cornerFilter.has(i)) {
      // Excluded from this fillet pass (e.g. handled by true-miter instead);
      // same shape as the collinear/near-180 skip below.
      return { isCorner: false, p0: v, p2: v, tanHalf: 0 };
    }
    const prev = points[(i - 1 + n) % n];
    const next = points[(i + 1) % n];
    const dIn = normalize(sub(v, prev));
    const dOut = normalize(sub(next, v));
    const crossVec = cross(dIn, dOut);
    const crossLen = length(crossVec);
    if (crossLen < FILLET_COLLINEAR_EPS) {
      // Collinear (turn~0) or a near-180 fold-back (turn~PI) — either way
      // `cross(dIn,dOut)` is too close to the zero vector to normalize
      // into a well-defined local plane normal. Both pass straight
      // through unfilleted.
      return { isCorner: false, p0: v, p2: v, tanHalf: 0 };
    }
    const planeNormal = scale(crossVec, 1 / crossLen);
    const unit = filletCornerArc(v, prev, next, 1, planeNormal); // radius=1 probe: unit.trim === tan(halfPhi), the shape-only quantity
    if (!unit.ok) return { isCorner: false, p0: v, p2: v, tanHalf: 0 }; // defensive: the crossLen guard above covers every case filletCornerArc refuses
    return { isCorner: true, v, prev, next, planeNormal, tanHalf: unit.trim };
  });

  // Pass 2 — the trim-budget check, `filletPolygon`'s worstRatio
  // construction, generalized so an edge missing one of its two corners (an
  // open rail's first/last edge) budgets against the one it has (the missing
  // side's `tanHalf` is 0).
  const edgeCount = closed ? n : n - 1;
  let worstRatio = 0;
  for (let i = 0; i < edgeCount; i++) {
    const a = points[i], b = points[(i + 1) % n];
    const edgeLen = length(sub(b, a));
    const tA = eff[i].tanHalf, tB = eff[(i + 1) % n].tanHalf;
    const needed = radius * (tA + tB);
    if (needed <= 0) continue; // no adjacent corner eats into this edge at all
    if (edgeLen < 1e-12) { worstRatio = Infinity; continue; }
    worstRatio = Math.max(worstRatio, needed / edgeLen);
  }
  if (worstRatio >= 1 - 1e-9) {
    /* As in `filletPolygon`: a zero-length edge sets worstRatio to Infinity,
       and 0 returned for it would read as a measured largest radius. null
       means no radius fits here. */
    return {
      ok: false,
      reason: 'fillet radius is too large for this rail — neighboring corners would overlap',
      maxSafeRadius: Number.isFinite(worstRatio) ? radius / worstRatio * 0.999 : null,
    };
  }

  // Pass 3 — build the arcs at the requested radius (safe after pass 2),
  // then interleave with the trimmed straight runs, omitting any that
  // collapse to zero length (see the header comment).
  const corners = eff.map((c) => {
    if (!c.isCorner) return { p0: c.p0, p2: c.p2 };
    const real = filletCornerArc(c.v, c.prev, c.next, radius, c.planeNormal);
    return real; // .ok is true here: pass 2 established this radius fits every corner
  });

  const segments = [];
  let cornerCount = 0;
  for (let i = 0; i < edgeCount; i++) {
    if (eff[i].isCorner) {
      const c = corners[i];
      segments.push({ type: 'arc', p0: c.p0, apex: c.apex, p2: c.p2, weight: c.weight });
      cornerCount++;
    }
    const nextIdx = (i + 1) % n;
    const startPt = corners[i].p2;
    const endPt = corners[nextIdx].p0;
    if (length(sub(endPt, startPt)) > FILLET_ZERO_LEN_EPS) {
      segments.push({ type: 'line', a: startPt, b: endPt });
    }
  }
  return { ok: true, segments, closed, cornerCount };
}

// Converts a `filletPolygon`/`filletOpenPolyline` segment list
// (`{type:'line', a, b}` / `{type:'arc', p0, apex, p2, weight}`) into one
// C0-joined NurbsCrv via `joinCurvesC0`, the kernel-side equivalent of the
// app's PolyCurve -> `joinCurvesC0` path, for kernel callers such as
// Pipe/MultiPipe. An all-line list (every corner skipped) takes the same
// path; `joinCurvesC0` reduces to plain concatenation there.
export function filletSegmentsToCurve(segments) {
  if (!segments.length) return null;
  const crvs = segments.map((s) => (
    s.type === 'line'
      ? makeLine(s.a, s.b)
      : { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...s.p0, 1], [...s.apex, s.weight], [...s.p2, 1]] }
  ));
  return joinCurvesC0(crvs);
}

// Gear (spur) and rack — involute-tooth mechanical primitives. Helical,
// internal-ring, bevel and worm gears are not built. All 2D profiles in the
// local XY plane, z=0; the app layer maps them into a picked frame and
// extrudes them (extrude()) into a solid.

// The involute of a circle — the standard gear-tooth flank curve. Exact
// closed form for base-circle radius `baseRadius` and involute parameter `t`
// (radians), unwound off the base circle, rotated by `startAngle`, with
// `handed` selecting the base involute (+1) or its mirror image across the
// generating radial (-1, the opposite-turning flank of a tooth):
//   x0(t) = rb*(cos t + t*sin t)
//   y0(t) = rb*(sin t - t*cos t)   (negated when handed = -1)
// then rotated by startAngle. This is the true involute, not an
// approximation. Two identities (checked in test/gear.test.mjs):
// |P| == rb*sqrt(1+t^2), and the normal to the involute at any point is at
// distance exactly rb from the base center (tangent to the base circle — the
// taut-string property; for this parametrization it is the normal, not the
// tangent, that touches the base circle, since dP/dt = rb*t*(cos t, sin t)).
export function involutePoint(baseRadius, t, startAngle = 0, handed = 1) {
  const x0 = baseRadius * (Math.cos(t) + t * Math.sin(t));
  const y0 = handed * baseRadius * (Math.sin(t) - t * Math.cos(t));
  const ca = Math.cos(startAngle), sa = Math.sin(startAngle);
  return [x0 * ca - y0 * sa, x0 * sa + y0 * ca, 0];
}

// makeInvoluteFlank — sample the analytic involute over a parameter range
// and interpolate a NURBS curve through the samples (globalCurveInterp,
// A9.1): exact at the sample points only. An involute is not an exact NURBS
// curve the way an arc is, so this is an approximation; test/gear.test.mjs
// bounds its deviation between samples.
// Returns { crv, points, tParams } — `points` are the raw analytic samples
// (exactly on the involute), which buildSpurGearProfile threads into the
// gear outline; `crv` is the fitted flank.
export function makeInvoluteFlank(baseRadius, startAngle, tParams, handed = 1, degree = 3) {
  const points = tParams.map((t) => involutePoint(baseRadius, t, startAngle, handed));
  const crv = globalCurveInterp(points, Math.min(degree, points.length - 1));
  return { crv, points, tParams };
}

// Standard AGMA spur-gear metrics from module + tooth count + pressure angle.
//   pitch diameter d = module * teethCount   -> pitch radius rp = d/2
//   base circle    rb = rp * cos(pressureAngle)
//   addendum (tip) ra = rp + module
//   dedendum(root) rf = rp - 1.25*module     (clamped >0 for tiny gears)
// invAlpha = the involute function inv(a) = tan(a) - a at the design pressure
// angle — the classic quantity that positions the tooth thickness.
export function gearMetrics(module, teethCount, pressureAngleDeg = 20) {
  const alpha = pressureAngleDeg * Math.PI / 180;
  const rp = module * teethCount / 2;
  const rb = rp * Math.cos(alpha);
  const ra = rp + module;
  const rf = Math.max(rp - 1.25 * module, 0.05 * module);
  return { module, teethCount, alpha, rp, rb, ra, rf, invAlpha: Math.tan(alpha) - alpha };
}

// Sample a rational quadratic (a filletCornerArc result) from p0 to p2 through
// its control apex with the given weight — n subdivisions, endpoints included.
function sampleRationalArc(f, n) {
  const out = [];
  for (let i = 0; i <= n; i++) {
    const s = i / n;
    const b0 = (1 - s) * (1 - s), b1 = 2 * s * (1 - s) * f.weight, b2 = s * s;
    const w = b0 + b1 + b2;
    out.push([
      (b0 * f.p0[0] + b1 * f.apex[0] + b2 * f.p2[0]) / w,
      (b0 * f.p0[1] + b1 * f.apex[1] + b2 * f.p2[1]) / w,
      0,
    ]);
  }
  return out;
}

// buildSpurGearProfile — assemble N repeated involute-tooth profiles into one
// closed 2D curve (degree-3, interpolated through a dense ordered ring of
// boundary points). Each tooth = involute flank out (makeInvoluteFlank
// samples), an addendum-circle tip arc, the mirrored flank in, and a root
// fillet to the next tooth built with filletCornerArc.
// Returns { crv, ring, metrics }.
export function buildSpurGearProfile(module, teethCount, pressureAngleDeg = 20, opts = {}) {
  const g = gearMetrics(module, teethCount, pressureAngleDeg);
  const { teethCount: N, rb, ra, rf, invAlpha } = g;
  const halfBaseAngle = Math.PI / (2 * N) + invAlpha; // tooth half-angle at the base circle
  const rStart = Math.max(rb, rf);
  const tStart = Math.sqrt(Math.max((rStart / rb) ** 2 - 1, 0));
  const tTip = Math.sqrt((ra / rb) ** 2 - 1);
  const NF = opts.flankSamples ?? 12;
  const NTIP = opts.tipSamples ?? 6;
  const NROOT = opts.rootSamples ?? 3;
  // Cosine (Chebyshev-like) spacing clusters flank samples toward both ends,
  // where the flank meets the root and the addendum arc at a corner. A single
  // global cubic through a corner otherwise overshoots there; dense samples
  // bracketing each corner hold the outline to the involute
  // (test/gear.test.mjs: whole-outline deviation under 0.03mm, the isolated
  // flank fit under 0.01mm).
  const tParams = [];
  for (let i = 0; i < NF; i++) { const s = (1 - Math.cos(Math.PI * i / (NF - 1))) / 2; tParams.push(tStart + (tTip - tStart) * s); }
  // half tooth-angle at radius r (>= rb), from the involute function
  const invAt = (r) => { const a = Math.acos(Math.min(rb / r, 1)); return Math.tan(a) - a; };
  const psiTip = Math.PI / (2 * N) + invAlpha - invAt(ra);
  const rootFillet = Math.min(0.38 * module, 0.45 * Math.max(rb - rf, 1e-4));
  const planeN = [0, 0, 1];
  const ring = [];
  for (let k = 0; k < N; k++) {
    const tc = 2 * Math.PI * k / N;
    // Right flank (base involute), base -> tip, angle increasing:
    const rightPts = makeInvoluteFlank(rb, tc - halfBaseAngle, tParams, +1).points;
    for (const p of rightPts) ring.push(p);
    // Tip arc across the addendum circle, right tip -> left tip (interior samples):
    for (let i = 1; i < NTIP; i++) { const a = (tc - psiTip) + 2 * psiTip * (i / NTIP); ring.push([ra * Math.cos(a), ra * Math.sin(a), 0]); }
    // Left flank (mirror involute), tip -> base, angle increasing:
    const leftPts = makeInvoluteFlank(rb, tc + halfBaseAngle, tParams, -1).points;
    for (let i = leftPts.length - 1; i >= 0; i--) ring.push(leftPts[i]);
    // Root / gap to the next tooth's right flank base:
    const leftBase = leftPts[0];
    const leftBaseAng = tc + halfBaseAngle;
    const nextRightBaseAng = 2 * Math.PI * (k + 1) / N - halfBaseAngle;
    const nextRightBase = involutePoint(rb, tStart, nextRightBaseAng, +1);
    if (rf < rb - 1e-9 && rootFillet > 1e-6) {
      const cornerL = [rf * Math.cos(leftBaseAng), rf * Math.sin(leftBaseAng), 0];
      const cornerR = [rf * Math.cos(nextRightBaseAng), rf * Math.sin(nextRightBaseAng), 0];
      const fL = filletCornerArc(cornerL, leftBase, cornerR, rootFillet, planeN);
      const fR = filletCornerArc(cornerR, cornerL, nextRightBase, rootFillet, planeN);
      if (fL.ok) for (const p of sampleRationalArc(fL, NROOT)) ring.push(p); else ring.push(cornerL);
      if (fR.ok) for (const p of sampleRationalArc(fR, NROOT)) ring.push(p); else ring.push(cornerR);
    } else {
      // rf >= rb (very high tooth counts): the flank already reaches the root
      // circle; join adjacent bases with a plain dedendum arc.
      for (let i = 1; i < NTIP; i++) { const a = leftBaseAng + (nextRightBaseAng - leftBaseAng) * (i / NTIP); ring.push([rStart * Math.cos(a), rStart * Math.sin(a), 0]); }
    }
  }
  // Interpolate one closed degree-3 curve through the ring. Duplicating the
  // first point at the end makes it a clamped cubic whose whole knot domain is
  // the closed loop (start == end, zero gap) — directly usable by extrude(),
  // the same shape makeCircle's output has (start point == end point).
  const crv = globalCurveInterp([...ring, ring[0]], 3);
  return { crv, ring, metrics: g };
}

// buildRackProfile — a rack is a spur gear of infinite radius. The involute
// of an infinite-radius base circle degenerates to a straight line inclined
// at the pressure angle (test/gear.test.mjs checks the finite-gear flank
// approaching it as radius grows), so a rack tooth is a straight-sided
// trapezoid. Lays `teethLength` teeth (a count) along +x, pitch line at y=0,
// teeth pointing +y; closes the toothed top edge into a solid bar
// cross-section (flat bottom) so it can be extruded.
// Returns { crv, ring, metrics }.
export function buildRackProfile(module, teethLength, pressureAngleDeg = 20, opts = {}) {
  const m = module, N = Math.max(1, Math.round(teethLength)), alpha = pressureAngleDeg * Math.PI / 180;
  const p = Math.PI * m;               // circular pitch (tooth spacing along the pitch line)
  const add = m, ded = 1.25 * m;       // addendum / dedendum
  const tanA = Math.tan(alpha);
  const halfTip = p / 4 - add * tanA;  // half tooth thickness at the tip land
  const back = opts.backHeight ?? 2 * m;
  const yTop = add, yRoot = -ded, yBottom = -ded - back;
  // Top toothed edge, +x order. Tooth k centered at xc = k*p; flanks:
  //   left flank  x = xc - p/4 + y*tanA,  right flank x = xc + p/4 - y*tanA.
  const ring = [];
  for (let k = 0; k < N; k++) {
    const xc = k * p;
    ring.push([xc - p / 4 - ded * tanA, yRoot, 0]); // root-left corner
    ring.push([xc - halfTip, yTop, 0]);             // tip-left
    ring.push([xc + halfTip, yTop, 0]);             // tip-right
    ring.push([xc + p / 4 + ded * tanA, yRoot, 0]); // root-right corner
  }
  const xStart = ring[0][0], xEnd = ring[ring.length - 1][0];
  // Close into a bar cross-section: down the right end, across the flat bottom,
  // up the left end (back to the start corner).
  ring.push([xEnd, yBottom, 0]);
  ring.push([xStart, yBottom, 0]);
  // Degree-1 interpolation keeps every corner sharp (a rack is faceted).
  const crv = globalCurveInterp([...ring, ring[0]], 1);
  return { crv, ring, metrics: { module: m, teethCount: N, alpha, pitch: p, addendum: add, dedendum: ded } };
}
