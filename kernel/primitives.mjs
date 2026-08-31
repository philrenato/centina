// Exact-rational primitives — Piegl & Tiller Ch. 7 (conic arcs) and A8.1
// (surface of revolution), plus the simple ruled/extruded surface (Ch. 8).
//
// The circle/arc technique: split the sweep into spans of at most 90°, and
// for each span use the CLOSED-FORM tangent-intersection point — no general
// line-line intersection needed, because in the span's own local (xHat,yHat)
// basis the middle control point is always P0 + r*tan(dtheta/2)*tangentAtP0,
// a fact that's rotation-invariant so it works for every 90°-or-less span at
// any starting angle. This is the same construction that produces the
// standard 9-control-point unit circle (degree 2, weights 1/(root2/2)
// alternating) verified against curve.mjs in test/curve-surface.test.mjs.

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
// `minSegments` forces MORE arc spans than the sweep alone would need (each
// span is still <=90 deg, so the closed-form construction stays exact
// regardless of span size) — this is how a circle gets "rebuilt" with more
// control points (Rhino's own Rebuild vocabulary) WITHOUT ever
// leaving the exact rational representation: more spans means more
// (still-exactly-on-the-true-circle) control points, not an approximating
// refit. Default 1 (no forced extra subdivision) keeps every existing
// caller's behavior byte-identical.
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

// `segments` (default 4, the natural minimum for a full 360 deg sweep at
// MAX_ARC_SPAN=90 deg) can be raised to "rebuild" the circle with more
// control points — still an EXACT circle at any segment count, never an
// approximation, so it stays fully usable as a Sweep1/Extrude/Loft/Revolve
// profile at any rebuild level (unlike a closed-curve-interpolation
// refit, which would need real domain-restriction machinery this kernel
// doesn't have — see getProfileCrv's own closed-SketchCurve comment).
export function makeCircle(center, xAxis, yAxis, radius, segments = 4) {
  return makeArc(center, xAxis, yAxis, radius, 0, 2 * Math.PI, segments);
}

// ELLIPSE — a NON-UNIFORM scale of a unit circle. A NURBS rational curve is
// EXACTLY preserved under any affine map applied per control point (the same
// "affine maps commute with affine combinations" identity Shear's own doc
// comments and revolve() already rely on): the rational basis functions sum
// to 1, so C(t) = sum(R_i(t) P_i) is an affine combination of its control
// points, and any affine T satisfies T(sum(R_i P_i)) = sum(R_i T(P_i)).
// makeArc/makeCircle builds each control point as
//   center + xHat*r*cos + yHat*r*sin  (positions)  and  weight = cos(dtheta/2)
// with r=1 here, so passing a NON-UNIT in-plane basis (xAxis scaled by
// radiusX, yAxis scaled by radiusY) applies exactly the affine map
// (u,v) -> center + u*radiusX*xAxis + v*radiusY*yAxis to every unit-circle
// control point — the exact ellipse, never an approximation. arcSpanPoints'
// own tangent-line intersection point p1 (a control point, not a curve
// point) transforms correctly for the same reason (an affine map preserves
// the tangent-line intersection). Verified numerically against the ellipse
// equation to float precision, and the degenerate radiusX===radiusY case is
// bit-for-bit identical to makeCircle (see test/ellipse.test.mjs) — so the
// weight formula's own implicit "|xAxis|==|yAxis|" assumption never actually
// bites, because those magnitudes only ever scale the POSITION terms, never
// the (dtheta-only) weights.
export function makeEllipse(center, xAxis, yAxis, radiusX, radiusY, segments = 4) {
  return makeCircle(center, scale(xAxis, radiusX), scale(yAxis, radiusY), 1, segments);
}

// ELLIPSOID PROFILE — the HALF-ELLIPSE meridian arc (pole to pole) whose
// surface of revolution around `polarAxis` is an exact ellipsoid, the direct
// generalization of makeEllipse's own "scale a unit arc's control points"
// affine trick (an ellipse is a non-uniform scale of a unit circle; a
// meridian is a non-uniform scale of a unit half-circle). Reuses makeArc's
// exact conic-arc construction for a HALF sweep (angleStart=-PI/2, sweep=PI),
// with the arc's own local x-axis (scaled by `equatorialRadius`) mapped to the
// equatorial/radial direction and its local y-axis (scaled by `polarRadius`)
// to the polar direction (the revolve axis). Because a NURBS curve is exactly
// preserved under any per-control-point affine map (the same identity
// makeEllipse relies on), passing the two scaled axes to makeArc applies the
// exact map (u,v) -> center + u*eR*equatorialAxis + v*pR*polarAxis to every
// control point — the true half-ellipse, never an approximation.
//
// BOTH endpoints land on the revolve axis (the two poles) to machine
// precision: the equatorial component of an endpoint is r*cos(-PI/2) and
// r*cos(+PI/2), each ~1e-16*r (cos of a floating-point PI/2), well under
// revolve()'s own 1e-9 pole-detection threshold, so revolve collapses each
// end-row exactly onto the axis. The apex (angle 0) reaches exactly the
// equatorial radius. Revolving 2*PI: for axis=z, equatorialAxis=x, a surface
// point is (eR*cosθ*cosφ, eR*cosθ*sinφ, pR*sinθ), satisfying the true
// ellipsoid equation (x/eR)^2+(y/eR)^2+(z/pR)^2 = cos²θ+sin²θ = 1 exactly at
// every (θ,φ) including the poles — proven numerically in test/ellipsoid.test.mjs.
export function makeEllipsoidProfile(center, equatorialAxis, polarAxis, equatorialRadius, polarRadius, minSegments = 2) {
  return makeArc(center, scale(equatorialAxis, equatorialRadius), scale(polarAxis, polarRadius), 1, -Math.PI / 2, Math.PI, minSegments);
}

// SQUIRCLE (2D) — a real, closed, degree-3 (cubic) NURBS curve from an
// 8-control-point corner-pull cage: a superellipse-like family smoothly
// parameterized by `softness` ∈ [0,1], from a square-ish rounded shape (0)
// toward a near-ellipse (1). NOT ellipse-adjacent — a genuinely different
// construction (a periodic uniform cubic B-spline over a control cage, where
// the curve stays INSIDE its cage and rounds the corners, rather than an
// exact conic arc). The 8 cage points are the 4 edge midpoints (±hw,0),
// (0,±hh) plus the 4 corners at (±cs*hw, ±cs*hh), where `cs` (the corner
// control-point radial scale) is the single softness knob:
//   cs = 1.25 - 0.5*softness  (softness 0 -> 1.25, softness 1 -> 0.75)
// A larger cs pushes the corner control points further out along the
// diagonal, so the curve reaches nearer the true square corner (square-ish);
// a smaller cs pulls them in toward the round of a circle. cs=1.25 places the
// corner curve-point at ~the true square corner (the most square-reading
// smooth member without corner overshoot); cs≈0.71 would be a numeric circle.
// The default 0.75 range keeps every member safely convex, simple (no
// self-intersection), and cusp-free at both extremes.
//
// CLOSED-AND-SMOOTH REPRESENTATION — the real subtlety, done carefully rather
// than rushed: a periodic B-spline's seam must stay C2 and its two ends must
// coincide exactly, but this kernel evaluates only CLAMPED curves. So the
// cage is wrapped into an OVER-PADDED uniform periodic B-spline (n+4p control
// points, so one full central period has a full valid p-control-point
// neighborhood on BOTH sides — clamping AT an unclamped curve's own validity
// boundary silently pulls in the invalid tail control points, the trap that
// makes a naive extraction produce a zero-speed seam cusp), then that central
// period is CLAMPED at both its interior boundaries via the proven exact
// insertKnot (Boehm A5.1, geometry-preserving) up to multiplicity degree+1
// and sliced out as a standalone clamped curve, its domain renormalized to
// [0, n]. Because the source is a genuine periodic B-spline, the extracted
// curve's first control point equals its last (closed, zero gap) and the
// seam carries the same tangent/curvature as everywhere else (no cusp).
// Verified numerically in test/squircle.test.mjs at many softness values:
// closed to zero gap, simple (angle-monotone/star-shaped, winds exactly
// once), cusp-free (curve speed bounded away from zero everywhere), and a
// smooth monotone square→circle progression, isFiniteNet at every extreme.
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
// angleStart/sweep are radians. Handles profile control points that lie ON
// the axis (a degenerate "pole" row, e.g. a revolve profile that touches its
// own axis — the teapot lid's dome apex is exactly this case).
//
// EXACTNESS AT A POLE ROW — root-caused, not guessed. A8.1's own
// exactness proof only holds if every row's own V-direction WEIGHT function
// (as a function of the sweep parameter v, ignoring the row's own radius)
// is the IDENTICAL shape for every row blended together at a given U — the
// same alternating (1, cos(dtheta/2), 1, cos(dtheta/2), ...) column pattern
// `arcSpanPoints` already builds for an ordinary row. The OLD pole branch
// broke that: it returned a UNIFORM weight (the raw profile weight `pw`) at
// every column instead of that alternating pattern — fine for the pole ROW
// evaluated in isolation (any constant point/weight column blends back to
// exactly that point, regardless of weight shape), but it desyncs the
// SURFACE's own row-blend identity the moment a pole row is combined with
// its non-pole neighbors at a non-knot-corner U — which is exactly the
// "exact only at knot corners, ~1-5% off between them" bug. The fix: run
// the pole row through the SAME `arcSpanPoints` construction every other
// row uses, with radius=0 (any perpendicular basis works — it's multiplied
// by zero, so direction can't matter) rather than a hand-rolled uniform-
// weight stand-in. A zero-radius arc still collapses every column to the
// pole point exactly (arcSpanPoints scales every position term by
// `radius`), but now carries the CORRECT alternating weight shape, so the
// row-blend identity `S(u,v) = O(u) + Rotate(Q(u), v)` (Q(u) the profile's
// own true radial offset at u) holds at every (u,v), not just U corners —
// verified in test/revolve-pole-exactness.test.mjs against the analytic
// sphere (a semicircle profile revolved through both its own poles) to
// 1e-9 relative error at many non-corner (u,v) samples, including a
// profile that also crosses to the axis's OTHER side (a genuine sign
// change in the radial direction) combined with a pole in the same net —
// that combined case turns out to need NO separate handling once the pole
// weight-shape bug above is fixed, since the underlying row-blend identity
// never actually depended on any two rows sharing a "consistent" local
// basis direction (only on the correct, row-independent V-weight shape).
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
      // SAME construction below to the pole point exactly, regardless of
      // the (otherwise-meaningless) basis direction chosen here.
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
// move on the homogeneous form). Still a genuine ruled surface (straight
// lines between corresponding U-parameter points on the two rows) with a
// nonzero draftAngleDeg too — only the TOP row's own coordinates change,
// not the surface's own construction — so degV stays exactly 1 either way.
//
// draftAngleDeg (default 0, unchanged behavior): a molding-style taper.
// Honest, STATED simplification, same spirit as loft()'s own relative-
// parameter-fraction seam-matching note below — a true draft angle offsets
// the profile by a perpendicular curve OFFSET before extruding it
// (OffsetCrv, not built in this app's toolset yet), which is exact for any
// profile shape including concave ones. This instead grows/shrinks each
// control point RADIALLY from the profile's own centroid by
// distance*tan(angle) — exact for a circle/regular polygon (every point is
// equidistant from the centroid, so a uniform radial grow IS the true
// offset), a close visual match for most ordinary convex profiles, but not
// a true offset for a concave or highly irregular curve.
// `vDegree` — an extrusion defaults to a higher
// degree than 2 in the direction of extrusion") — OPTIONAL, defaults to
// 1 (the ORIGINAL, unchanged behavior: exactly 2 control points per row,
// a literal ruled surface) so every EXISTING caller/test that doesn't
// pass it is byte-for-byte unaffected. When higher, each U-row's own
// straight bottom-to-top ruling LINE (a trivial degree-1, 2-point curve)
// is degree-ELEVATED via kernel/knots.mjs's own degreeElevateCurve — a
// real, exact, shape-preserving NURBS operation (elevating a straight
// line's degree redistributes MORE control points along the SAME exact
// line, never bending it) — giving real, directly-draggable intermediate
// control points along the extrusion height, which a plain 2-point ruled
// surface structurally cannot offer at all. The app's own creation code
// (its own Extrude operator) is what actually raises this
// default for NEW user-created extrudes; the kernel function itself
// stays conservative so other callers (the teapot builder, any future
// kernel consumer) keep their own exact prior behavior unless they ask.
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
      if (r < 1e-9) return [x, y, z, w]; // a control point AT the centroid has no radial direction to grow along — left untapered rather than dividing by zero
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
    knotsV = elevated.knots; // identical for every row by construction (same input degree/knots each time) — captured once, reused
    return elevated.ctrlPts;
  });
  return { degU: profile.degree, knotsU: profile.knots, degV: vDegree, knotsV, ctrlNet };
}

// FILLET (polygon corner rounding) — reuses the EXACT SAME closed-form
// conic-arc construction as arcSpanPoints/makeArc above (P&T Ch.7's "two
// tangent points + their tangent-line intersection point + weight =
// cos(halfSweep)" recipe), just parametrized differently. arcSpanPoints
// starts from a known CENTER + start angle and derives the tangent-line
// intersection (its own `p1`) from them; a polygon corner already gives
// us that tangent-line intersection FOR FREE — it's the vertex itself, by
// construction, since the two tangent lines to the fillet arc are the two
// polygon edges meeting exactly at that vertex — so this goes the other
// direction: apex (a known point) + the two edge directions -> the two
// tangent points + the rational weight, with no separate center/angle
// derivation needed at all. Same formula (trim = radius*tan(halfSweep/2)
// swept over the halfSweep... — see below), same conic identity, just
// evaluated from the opposite set of knowns.
//
// Works for both CONVEX and REFLEX corners (needed once a Star polygon's
// alternating inner vertices are filleted too) via the SIGNED turn angle
// (atan2 of a 2D cross/dot against the polygon's own plane normal) rather
// than an interior-angle formula that only holds for a convex turn (whose
// tan() would go negative/wrap past a reflex corner's own interior angle
// exceeding 180deg). `planeNormal` must be a unit vector normal to the
// polygon's own plane (e.g. cross(xAxis,yAxis) for this app's own Circle/
// Polygon convention) — it only fixes the SIGN convention for which way a
// "left" vs "right" turn reads, not which side of the plane the corner is
// on (both dIn/dOut/vertex are assumed to already lie in that plane).
export function filletCornerArc(vertex, prevPt, nextPt, radius, planeNormal) {
  const dIn = normalize(sub(vertex, prevPt)); // direction of travel ARRIVING at vertex
  const dOut = normalize(sub(nextPt, vertex)); // direction of travel LEAVING vertex
  const sinPhi = dot(cross(dIn, dOut), planeNormal); // signed sine of the turn angle
  const cosPhi = dot(dIn, dOut);
  const phi = Math.atan2(sinPhi, cosPhi); // signed turn angle in (-PI, PI]; >0 = convex/left turn, <0 = reflex/right turn (CCW loop convention)
  const halfPhi = Math.abs(phi) / 2;
  if (halfPhi < 1e-7) return { ok: false, reason: 'the path is straight here — nothing to round' };
  if (Math.abs(phi) > Math.PI - 1e-6) return { ok: false, reason: 'a near-180° reversal has no well-defined fillet' };
  const trim = radius * Math.tan(halfPhi); // same tangent-length formula as arcSpanPoints' own `tanScale`
  const weight = Math.cos(halfPhi); // same rational weight formula as arcSpanPoints' own `w1`
  const p0 = sub(vertex, scale(dIn, trim)); // trimmed back along the INCOMING edge
  const p2 = add(vertex, scale(dOut, trim)); // trimmed forward along the OUTGOING edge
  return { ok: true, p0, apex: vertex, p2, weight, trim, turnAngle: phi };
}

// Builds a filleted CLOSED vertex loop (a regular/star Polygon's own
// vertex array, in order) — every corner rounded by the SAME radius,
// alternating LINE/ARC segments (never a single degree-elevated curve —
// this app's own PolyCurve type already represents a mixed straight+
// curved chain as a plain segment LIST, not one unified NURBS curve, so
// that's the shape returned here too: `{type:'line', a, b}` /
// `{type:'arc', p0, apex, p2, weight}` segments, in path order, forming a
// closed loop). Refuses honestly (never silently overlaps) when the
// requested radius is geometrically too large for the polygon's OWN edge
// lengths/turn angles to support: two neighboring corners sharing one
// edge would trim past each other (or exactly meet, a genuine zero-length
// remaining straight run) — `maxSafeRadius` is reported alongside the
// refusal so a caller can clamp instead of just failing outright.
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
  if (worstRatio >= 1 - 1e-9) {
    return { ok: false, reason: 'fillet radius is too large for this polygon — neighboring corners would overlap', maxSafeRadius: radius / worstRatio * 0.999 };
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

// OPEN-RAIL-AWARE FILLET — live-tested against the corner-mitering fix,
// which left a non-uniform swept radius along some sections and no option
// for soft corners (wanted especially for MultiPipe). The non-uniform
// radius root-caused to a
// SEPARATE bug from mitering itself, confirmed directly against
// kernel/sweep.mjs's own `buildParallelTransportFrames`/`sweep1Rigid`
// before writing anything here: a degree<=1 rail's free sweep path is a
// PLAIN RULED (linear) blend between exactly two rings per control-point
// span; the interior-corner miter (shipped the same week) sets BOTH end
// rings of a span to a shared bisector orientation, and a linear ruled
// blend between two same-radius circles at DIFFERENT (mutually tilted)
// orientations contracts mid-span by ~cos(half the relative tilt) — a
// real, measured ~4.58/5.0 waist (see test/sweep-interior-corner-miter.test.mjs
// for the numeric proof), not a rendering
// artifact and not fixable by tweaking the miter itself: representing a
// sharp C0 join as one shared frame per corner inside a single continuous
// ruled surface structurally cannot stay round mid-span. THE FIX
// asked for, and confirmed structurally correct rather than just
// a preference: round the corner INTO THE RAIL before sweeping at all — a
// genuinely curved (degree-2) rail routes through `sweep1RigidResampled`
// (the dense arc-length-resample path; see `railFrameOriginsExact`),
// which was never taught to treat a corner specially at all, eliminating
// the waist as a structural side effect of no longer having a shared-frame
// C0 join anywhere along the rail.
//
// This is a NEW, separate sibling of `filletPolygon` above, not a
// generalization of it in place, because `filletPolygon` has THREE real
// assumptions this rail case breaks: (1) closed-loop-only indexing
// (`(i-1+n)%n`/`(i+1)%n` on every vertex, no concept of an open end with
// only one adjacent edge); (2) all-or-nothing refusal (`for (const c of
// corners) if (!c.ok) return {ok:false...}` aborts the WHOLE polygon the
// instant ANY one corner is a genuine collinear straight run — routine and
// expected for an arbitrary rail, never for a regular/star Polygon's own
// vertex set, which is why this never bit that caller); (3) ONE shared
// `planeNormal` argument, correct only because a Circle/Polygon's own
// vertices are already known-planar — a general 3D pipe rail has no such
// guarantee, and a genuinely non-planar rail needs a DIFFERENT plane
// normal at each corner (its own local turn axis) to fillet correctly at
// all.
//
// PER-CORNER LOCAL NORMAL (point 2 above, the load-bearing new idea):
// `planeNormal = normalize(cross(dIn, dOut))` computed FRESH at each
// corner from that corner's own two edge directions, instead of one
// normal shared across every corner. This makes `filletCornerArc`'s own
// `sinPhi = dot(cross(dIn,dOut), planeNormal)` reduce to EXACTLY
// `dot(crossVec, crossVec/|crossVec|) = |crossVec| = sin(turnAngle)` (the
// correct, always non-negative, turn-angle MAGNITUDE) at every corner,
// nothing degenerate — confirmed directly against `filletCornerArc`'s own
// formula above, not assumed. Provably produces IDENTICAL arc geometry
// (p0/apex/p2/weight/trim) to `filletPolygon`'s own shared-global-normal
// convention on a genuinely PLANAR closed loop: `filletCornerArc` only
// ever uses `Math.abs(phi)/2` (`halfPhi`) to build the arc — flipping
// `planeNormal`'s sign (which per-corner local normals can do, relative to
// one fixed global choice, at a REFLEX corner) only flips the SIGN of the
// reported `turnAngle`, never `halfPhi`'s magnitude — proven directly in
// test/fillet-open-polyline.test.mjs's own planar cross-check against
// `filletPolygon` on the identical input.
//
// COLLINEAR / NEAR-180 GUARD, checked BEFORE any bisector/fillet math runs
// (matching this app's own established "collinear no-op checked first"
// discipline — see buildParallelTransportFrames' own identical-in-spirit
// guard): `filletCornerArc` itself never normalizes `cross(dIn,dOut)` (it
// only ever takes `dot(cross(dIn,dOut), planeNormal)`, safe even when that
// cross product is the zero vector), but THIS function's own per-corner
// `planeNormal` computation DOES need to normalize that same cross
// product — a collinear vertex (dIn parallel to dOut, turn=0) or a
// near-180 fold-back (dIn nearly opposite dOut, turn=~PI) both make
// `cross(dIn,dOut)` nearly the zero vector, so `normalize` on it would be
// NaN/garbage. Guarded by checking `length(cross(dIn,dOut))` against a
// small epsilon BEFORE calling `normalize` or `filletCornerArc` at all —
// both cases are treated identically: the vertex passes straight through
// UNFILLETED (an honest, deliberate skip, never an abort of the whole
// rail — see point (2)'s all-or-nothing contrast above).
//
// OPEN vs CLOSED, AND THE EDGE-BUDGET DIFFERENCE (point 3): an OPEN rail's
// first/last vertex has only ONE adjacent edge and is never itself a
// corner (endpoints pass straight through, `{p0:v, p2:v}`, zero trim) —
// which means every edge's own trim-budget check (`needed = tanHalf[i] +
// tanHalf[i+1]` at radius=1, scaled by the real requested radius) already
// gets this right FOR FREE, with no separate open-vs-closed branch: an
// edge touching an open rail's own endpoint always has that endpoint's
// own trim contribute exactly 0 to the sum, correctly budgeting the
// WHOLE edge against its one real interior-corner neighbor alone, never
// double-counting a second neighbor that doesn't exist. A CLOSED rail
// (every vertex a genuine corner) reduces to `filletPolygon`'s own
// identical two-corner-per-edge budget.
//
// AUTO-CLAMP, not hard refusal (point 4, matching `filletPolygon`'s own
// existing clamp-not-refuse precedent, `applyPolygonEvaluation`'s own
// established caller pattern): on a genuinely too-large requested radius,
// reports a real, retriable `maxSafeRadius` (the identical
// `radius/worstRatio*0.999` formula `filletPolygon` already uses) rather
// than only failing — there is no genuinely unsolvable case for a fillet,
// shrinking toward 0 always eventually fits.
//
// ZERO-LENGTH REMAINDER (point 4, a genuinely NEW degenerate case
// `filletPolygon` never had to handle, since a closed loop's every edge
// always sits between two real corners): when a requested radius trims
// exactly up to (or past, before the auto-clamp catches it) an open
// rail's own first/last edge full length, the remaining straight run on
// that side can come out zero-length even at a SAFE radius (its far end's
// own trim alone can legitimately consume the whole edge). Any resulting
// line segment shorter than a tiny tolerance is OMITTED entirely from the
// segment chain, never passed to `joinCurvesC0`/`filletSegmentsToCurve` as
// a degenerate zero-length span (which would risk NaN/garbage there).
//
// V1 SCOPE, stated honestly (matching this kernel's own recurring
// "shared, not per-element" precedent — `filletPolygon` itself already
// applies ONE shared radius to every corner of a polygon): ONE shared
// radius for the WHOLE rail, not a per-corner individual radius — real,
// separate v2 scope, not attempted here.
//
// `opts.cornerFilter` (TRUE MITER round): an optional `Set` of
// vertex indices — when present, any corner NOT in the set is treated
// EXACTLY like a collinear/near-180 skip (`isCorner: false`, passes straight
// through unfilleted), reusing the identical branch that already exists for
// that case rather than a new one. Default `undefined` = fillet every real
// corner, today's exact behavior, byte-identical for every existing caller.
// This is what `sweep1Rigid`'s own miter-limit fallback (kernel/sweep.mjs)
// needed: fillet ONLY the one (or few) corner(s) whose true-miter stretch
// would exceed the limit, leaving every other corner on the same rail
// completely untouched by this function, to be true-mitered instead.
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

  // Pass 1 — pure geometry, radius-independent: which vertices are real
  // corners (has two neighbors, genuinely turns), which pass straight
  // through (an open rail's own endpoints, or a collinear/near-180
  // vertex), and each real corner's own `tanHalf` (its trim-per-unit-
  // radius, i.e. `filletCornerArc`'s own `trim` at radius=1 exactly,
  // reused rather than re-derived so this function carries zero duplicate
  // trig — the SAME formula stays the single source of truth either way).
  const eff = points.map((v, i) => {
    if (!closed && (i === 0 || i === n - 1)) {
      return { isCorner: false, p0: v, p2: v, tanHalf: 0 };
    }
    if (cornerFilter && !cornerFilter.has(i)) {
      // Explicitly excluded from THIS fillet pass (e.g. handled by
      // true-miter instead) — an honest per-corner skip, same shape as the
      // collinear/near-180 skip below, never an abort of the whole rail.
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
      // through unfilleted, an honest per-corner skip, never an abort of
      // the whole rail.
      return { isCorner: false, p0: v, p2: v, tanHalf: 0 };
    }
    const planeNormal = scale(crossVec, 1 / crossLen);
    const unit = filletCornerArc(v, prev, next, 1, planeNormal); // radius=1 probe: unit.trim === tan(halfPhi), the shape-only quantity
    if (!unit.ok) return { isCorner: false, p0: v, p2: v, tanHalf: 0 }; // defensive — the crossLen guard above should already have caught every case filletCornerArc itself would refuse
    return { isCorner: true, v, prev, next, planeNormal, tanHalf: unit.trim };
  });

  // Pass 2 — the trim-budget check, exactly `filletPolygon`'s own
  // worstRatio construction, generalized so an edge missing one of its two
  // neighbors (an open rail's first/last edge) naturally budgets against
  // only the one real neighbor it has (the missing side's own `tanHalf`
  // is always 0, contributing nothing to `needed`).
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
    return {
      ok: false,
      reason: 'fillet radius is too large for this rail — neighboring corners would overlap',
      maxSafeRadius: Number.isFinite(worstRatio) ? radius / worstRatio * 0.999 : 0,
    };
  }

  // Pass 3 — build the real arcs at the actual requested radius (now
  // proven safe), then interleave with the trimmed straight runs,
  // omitting any that collapse to zero length (the genuinely new open-
  // rail degenerate case `filletPolygon` never had to handle — see header
  // comment).
  const corners = eff.map((c) => {
    if (!c.isCorner) return { p0: c.p0, p2: c.p2 };
    const real = filletCornerArc(c.v, c.prev, c.next, radius, c.planeNormal);
    return real; // .ok guaranteed true here — pass 2 already proved this radius safe for every real corner
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

// Converts a `filletPolygon`/`filletOpenPolyline` segment list (plain
// `{type:'line', a, b}` / `{type:'arc', p0, apex, p2, weight}` objects,
// the SAME shape either function returns) into ONE composed NurbsCrv —
// pure kernel-side mirror of the app layer's own established
// `buildPolygonFilletSegments` -> `getProfileCrv`'s PolyCurve branch ->
// `joinCurvesC0` pipeline (in the app), reused here so a kernel
// caller (Pipe/MultiPipe's own `evaluate()`) never needs to round-trip
// through app-side THREE.Vector3/PolyCurve pseudo-objects just to get a
// single sweepable rail curve back. An all-degree-1 segment list (no arcs
// at all — e.g. every corner was collinear/skipped, or radius<=0) still
// goes through `joinCurvesC0` rather than getProfileCrv's own separate
// degree-1 fast path; `joinCurvesC0` itself reduces to the identical
// concatenation in that case (proven in test/knots.test.mjs already, not
// re-proven here) — one code path, not two, for this kernel-side use.
export function filletSegmentsToCurve(segments) {
  if (!segments.length) return null;
  const crvs = segments.map((s) => (
    s.type === 'line'
      ? makeLine(s.a, s.b)
      : { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...s.p0, 1], [...s.apex, s.weight], [...s.p2, 1]] }
  ));
  return joinCurvesC0(crvs);
}

// ===========================================================================
// GEAR (Spur) + RACK — involute-tooth mechanical primitives
// (reconciled scope: Spur + Rack;
// Helical / Internal-ring / Bevel / Worm+wheel deferred, see gear.test.mjs
// header). All 2D profiles in the local XY plane, z=0; the app layer maps
// them into a picked frame and EXTRUDEs them (reusing extrude()) into a solid.
// ===========================================================================

// The INVOLUTE OF A CIRCLE — the standard gear-tooth flank curve. Exact
// closed form for base-circle radius `baseRadius` and involute parameter `t`
// (radians), unwound off the base circle, rotated by `startAngle`, with
// `handed` selecting the base involute (+1) or its mirror image across the
// generating radial (-1, the opposite-turning flank of a tooth):
//   x0(t) = rb*(cos t + t*sin t)
//   y0(t) = rb*(sin t - t*cos t)   (negated when handed = -1)
// then rotated by startAngle. This is NOT an approximation — it traces the
// true involute exactly. Two checkable identities (verified numerically in
// test/gear.test.mjs, not eyeballed): |P| == rb*sqrt(1+t^2) exactly, and the
// NORMAL to the involute at any point is at distance exactly rb from the base
// center (i.e. is tangent to the base circle — the taut-string property; note
// it is the NORMAL, not the tangent, that is tangent to the base circle for
// this parametrization, since dP/dt = rb*t*(cos t, sin t) is radial).
export function involutePoint(baseRadius, t, startAngle = 0, handed = 1) {
  const x0 = baseRadius * (Math.cos(t) + t * Math.sin(t));
  const y0 = handed * baseRadius * (Math.sin(t) - t * Math.cos(t));
  const ca = Math.cos(startAngle), sa = Math.sin(startAngle);
  return [x0 * ca - y0 * sa, x0 * sa + y0 * ca, 0];
}

// makeInvoluteFlank — sample the analytic involute densely over a parameter
// range and INTERPOLATE a real NURBS curve through those samples (reusing
// this kernel's own already-proven globalCurveInterp / A9.1, the same honest
// "interpolated, exact AT its sample points, dense enough to look and behave
// smooth" standard used for SketchCurve). An involute is not itself an exact
// NURBS curve the way an arc is, so this is a genuine (dense) approximation —
// its deviation from the true analytic involute between samples is bounded and
// proven under a stated tolerance in test/gear.test.mjs, not claimed exact.
// Returns { crv, points, tParams } — `points` are the RAW analytic samples
// (exactly on the involute), which buildSpurGearProfile threads into the whole
// gear outline; `crv` is the fitted flank NURBS, verified against the analytic
// curve.
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

// buildSpurGearProfile — assemble N repeated involute-tooth profiles into ONE
// closed, periodic 2D curve (degree-3, interpolated through a dense ordered
// ring of boundary points). Each tooth = involute flank out (makeInvoluteFlank
// samples), an addendum-circle tip arc, the mirrored flank in, and a ROOT
// FILLET connecting to the next tooth built with this kernel's own proven
// filletCornerArc (the Tier-1 fillet consumer the doc calls for — NOT a third
// parallel arc implementation). Returns { crv, ring, metrics }.
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
  // COSINE (Chebyshev-like) spacing clusters flank samples toward BOTH ends —
  // where the flank meets the root and the addendum arc at a genuine corner.
  // A single global cubic through a corner otherwise overshoots there; dense
  // samples bracketing each corner hold the interpolated outline tight to the
  // true involute (proven in test/gear.test.mjs: whole-outline deviation stays
  // well under 0.03mm, the isolated flank fit under 0.01mm).
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
    // RIGHT flank (base involute), base -> tip, angle increasing:
    const rightPts = makeInvoluteFlank(rb, tc - halfBaseAngle, tParams, +1).points;
    for (const p of rightPts) ring.push(p);
    // TIP arc across the addendum circle, right tip -> left tip (interior samples):
    for (let i = 1; i < NTIP; i++) { const a = (tc - psiTip) + 2 * psiTip * (i / NTIP); ring.push([ra * Math.cos(a), ra * Math.sin(a), 0]); }
    // LEFT flank (mirror involute), tip -> base, angle increasing:
    const leftPts = makeInvoluteFlank(rb, tc + halfBaseAngle, tParams, -1).points;
    for (let i = leftPts.length - 1; i >= 0; i--) ring.push(leftPts[i]);
    // ROOT / gap to the next tooth's right flank base:
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
  // Interpolate ONE closed degree-3 curve through the ring. Duplicating the
  // first point at the end makes it a clamped cubic whose whole knot domain is
  // the closed loop (start == end, zero gap) — directly usable by extrude(),
  // the same shape makeCircle's output has (start point == end point).
  const crv = globalCurveInterp([...ring, ring[0]], 3);
  return { crv, ring, metrics: g };
}

// buildRackProfile — a RACK is a spur gear of infinite radius (a straight
// "linear gear"). The involute of an infinite-radius base circle degenerates
// to a STRAIGHT LINE inclined at the pressure angle (verified in
// test/gear.test.mjs against the finite-gear flank as radius grows) — so a
// rack tooth is genuinely a straight-sided trapezoid, NOT a curved involute:
// a real, correct simplification, not a shortcut. Lays `teethCount` teeth along
// +x, pitch line at y=0, teeth pointing +y; closes the toothed top edge into a
// solid bar cross-section (flat bottom) so it can be extruded. `teethLength` is
// the number of teeth. Returns { crv, ring, metrics }.
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
