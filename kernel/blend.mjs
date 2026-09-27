// Curvature-continuous corner blends — the G2/G3 alternative to the tangent
// arc that `filletCornerArc` builds.
//
// `filletCornerArc` builds a circular arc tangent to both edges: G1 (the unit
// tangent is continuous across each seam) but not G2 — curvature jumps from 0
// on the straight edge to 1/R where the arc starts, and back to 0 at the far
// seam. That discontinuity is invisible in a wireframe and visible in a
// reflection/zebra image on any surface swept along or filleted from such a
// curve. This module supplies curvature-continuous blends; it does not change
// or wrap the arc path, which remains the answer whenever a constant-radius
// round is wanted (a machined fillet, a pipe corner).
//
// G-continuity vs C-continuity — the claims below depend on the distinction.
//   C^k (parametric) : the first k derivatives with respect to the curve
//                      parameter agree across the seam.
//   G^k (geometric)  : there exists a regular reparametrization under which
//                      they agree — equivalently, the reparametrization-
//                      invariant quantities agree: unit tangent (G1), plus
//                      curvature vector (G2), plus d(kappa)/ds and torsion
//                      (G3).
// G^k is the weaker, and the one that governs how a highlight runs across a
// surface. Everything this module claims is G^k, measured through those
// invariants; nothing here claims C^k. The composed PolyCurve produced by
// `blendSegmentsToCurve` is only C^0 in its own knot vector (`joinCurvesC0`
// rescales every segment onto its own integer domain slot and stacks
// full-multiplicity knots at each joint) while being G2/G3 in its geometry.
// Parametric derivatives across that joint would report a discontinuity that
// is an artifact of the parametrization; test/blend.test.mjs measures the
// geometric invariants on both sides of the joint instead.
//
// The construction: a Bezier blend built from the beta-constraint (connection
// matrix) characterization of geometric continuity.
//
// Barsky & DeRose ("Geometric Continuity of Parametric Curves: Three
// Equivalent Characterizations", IEEE Computer Graphics & Applications 9(6),
// 1989, pp. 60-68) prove the equivalence this module rests on: two
// regular curve segments join with G^k continuity if and only if their
// one-sided derivatives at the joint are related by the beta-constraints
//     C'   = b1 A'
//     C''  = b1^2 A'' + b2 A'
//     C''' = b1^3 A''' + 3 b1 b2 A'' + b3 A'
// for some b1 > 0 and free b2, b3 (see also Farin, "Curves and Surfaces for
// CAGD", 5th ed., the geometric-continuity/Beta-spline chapter, for the same
// relations derived as the chain rule of a reparametrization u = phi(t)):
// b1 = phi', b2 = phi'', b3 = phi'''.
//
// This module takes the simplest valid choice, b2 = b3 = 0, which is a
// linear reparametrization phi(t) = uJoin + b1*t. Under it the blend has
// C^k contact with A o phi at the joint, so G^k follows directly and the
// tests can be strict rather than approximate. b1 > 0 remains free: it is
// the blend's only shape knob, and because curvature and torsion are
// reparametrization-invariant, changing b1 cannot change the continuity
// order, only the interior shape. The tests exercise several values of it
// against the same tolerances.
//
// Turning the derivative data into control points is then the ordinary Bezier
// endpoint-derivative relation (Piegl & Tiller, "The NURBS Book", 2nd ed.,
// §1.3 — the k-th derivative at an end equals n!/(n-k)! times
// the k-th forward difference of the k+1 control points at that end):
//     C^(j)(0) = n!/(n-j)! * Delta^j b_0        (and the mirror at t = 1)
// inverted as b_j = sum_i binom(j,i) Delta^i b_0. Fixing derivatives 0..k at
// both ends pins k+1 control points at each end, so the minimal single-
// segment polynomial degree is n = 2k+1:
//     G1 -> cubic  (but see below: the corner API uses the arc for G1)
//     G2 -> quintic
//     G3 -> degree 7
// Rational curve derivatives on the neighbor side come from the kernel's
// `rationalCurveDerivs` (P&T A4.2), so a rational neighbor — a Circle, an
// existing fillet arc — is handled with no special case.
//
// Why not a cubic. The continuity orders offered are G1 (the degree-2
// rational arc), G2 and G3. A G2 blend of literal curve degree 3 cannot be
// built as a single Bezier segment: a G2 blend between two straight edges
// must have zero curvature at both ends, which forces b0,b1,b2 collinear at
// the start and b_n,b_n-1,b_n-2 collinear at the end. For a cubic (n = 3)
// those two conditions share b1 and b2 and force all four control points
// onto one line — a straight segment. Degree 4 is achievable only in the
// special case where the two triples share their middle point (which for
// straight edges means the corner vertex itself), and admits no analog once
// a neighbor has nonzero curvature. The quintic is therefore the minimal
// degree that works uniformly. A cubic representation would be a cubic
// B-spline with interior knots fitted to this curve, which is not built
// here; `blendSegmentsToCurve` returns the quintic/septic directly.
//
// The cost of curvature continuity. Given the same two tangent points a G1
// arc of radius R would use, a G2 blend's curvature must rise from 0 and fall
// back to 0 while turning through the same total angle, so its peak
// curvature is necessarily greater than 1/R — the blend is sharper in the
// middle and softer at the ends than the arc. With the shape knob tuned to
// minimize that peak (the default, `tangentScale: 'min-curvature'` below) the
// cost across the admissible turn-angle range is a factor of 1.10 to 1.18
// for G2 and 1.12 to 1.32 for G3 — a G2-blended "R10" corner peaks near
// 1/8.5mm. This is inherent to the continuity order; test/blend.test.mjs
// pins the numbers. A caller that needs a constant-radius round uses the arc.
//
// Scope:
//
//  * Exact: given end frames (point + derivatives), the emitted control
//    points satisfy the beta-constraints to floating-point rounding. The G^k
//    claim is exact in the same sense a knot insertion is exact — no fitting,
//    no iteration, no tolerance in the construction. The residuals measured
//    in the tests are round-off, not method error.
//
//  * The shape knob is numerically optimized; the construction is not. The
//    default `tangentScale: 'min-curvature'` runs a golden-section search
//    over the one free scalar, evaluating sampled peak curvature. That search
//    affects shape only: any positive value yields the same continuity order.
//
//  * Not here — the general curve/curve fillet. `blendCurves` blends two
//    curves at parameters the caller supplies. Finding the pair of parameters
//    at which a blend of a requested radius is tangent to two arbitrary NURBS
//    curves is a separate nonlinear solve this kernel does not have
//    (`filletCornerArc`/`filletPolygon`/`filletOpenPolyline` are all
//    straight-edge corner constructions). This module supplies the blend once
//    the two join points are chosen; it does not choose them.
//
//  * No self-intersection search. The blend is checked for regularity (no
//    vanishing speed / cusp) by sampling, and refused if it fails, but a
//    blend that loops back over itself without cusping is not caught. The
//    tests sweep every turn angle `filletCornerArc` accepts at the default
//    shape fraction without reaching it; a caller forcing an extreme
//    `tangentScale` could.
//
// Curvature/torsion formulas are the standard ones (do Carmo, "Differential
// Geometry of Curves and Surfaces", Ch. 1.5; the torsion below takes the
// sign convention b' = -tau n, opposite to do Carmo's b' = tau n):
//     kappa = |C' x C''| / |C'|^3        tau = (C' x C'') . C''' / |C' x C''|^2
// with the arc-length derivative of curvature obtained by the quotient rule
// and divided by |C'| once more to convert d/dt into d/ds.

import { rationalCurveDerivs, curvePoint, assertCurve } from './curve.mjs';
import { filletCornerArc, filletOpenPolyline } from './primitives.mjs';
import { joinCurvesC0 } from './knots.mjs';
import { add, sub, scale, dot, cross, length, normalize } from './vec3.mjs';

// Geometric invariants — the quantities every claim in this module is stated
// in terms of, and that the tests measure. Exported for curvature-comb and
// continuity-report overlays.

// Full third-order geometric report at one parameter of one curve. `d3` is
// only meaningful when the curve's degree is >= 3 (P&T's derivative algorithm
// returns zero beyond the degree, which is the true value). `planeNormal`,
// when supplied, adds a signed planar curvature and its arc-length
// derivative — signed curvature keeps dkappa/ds well-defined through an
// inflection (|C' x C''| has a non-differentiable minimum at zero, so the
// unsigned dkappa/ds is undefined exactly at straight-edge seams).
export function curveGeometryAt(crv, u, planeNormal = null) {
  const [C0, C1, C2, C3] = rationalCurveDerivs(crv, u, 3);
  return frameGeometry(C0, C1, C2, C3, planeNormal);
}

// The same report computed from raw derivative vectors, so a caller holding
// derivatives (this module's end frames) need not synthesize a curve to
// measure them.
export function frameGeometry(C0, C1, C2, C3, planeNormal = null) {
  const speed = length(C1);
  const w = cross(C1, C2); // C' x C''
  const wLen = length(w);
  const s3 = speed * speed * speed;
  const kappa = s3 > 0 ? wLen / s3 : NaN;
  const tangent = speed > 0 ? scale(C1, 1 / speed) : [NaN, NaN, NaN];
  // Curvature vector: the component of C''/|C'|^2 perpendicular to the
  // tangent. This is the quantity G2 requires to match — matching only the
  // scalar |kappa| would accept a blend curving away on the wrong side of its
  // neighbor.
  const inv2 = speed > 0 ? 1 / (speed * speed) : NaN;
  const along = speed > 0 ? dot(C2, tangent) : NaN;
  const kappaVec = speed > 0
    ? scale(sub(C2, scale(tangent, along)), inv2)
    : [NaN, NaN, NaN];
  // d(kappa)/dt by the quotient rule on |C' x C''| / |C'|^3, then /|C'| to
  // reach d(kappa)/ds. d|w|/dt = (w . (C' x C''')) / |w|; the C'' x C'' term
  // of the product rule vanishes identically.
  const dwdt = cross(C1, C3);
  const dLen = wLen > 0 ? dot(w, dwdt) / wLen : NaN;
  const dSpeed = speed > 0 ? dot(C1, C2) / speed : NaN;
  const dKappaDt = s3 > 0 ? dLen / s3 - (3 * wLen * dSpeed) / (s3 * speed) : NaN;
  const dKappaDs = speed > 0 ? dKappaDt / speed : NaN;
  // Torsion. Undefined where the curve is locally straight (|C' x C''| = 0),
  // reported as NaN rather than 0 so a torsion comparison there cannot pass
  // vacuously.
  const torsion = wLen > 1e-300 ? dot(w, C3) / (wLen * wLen) : NaN;
  const out = { point: C0, tangent, speed, kappa, kappaVec, dKappaDs, torsion };
  if (planeNormal) {
    const n = normalize(planeNormal);
    const sk = s3 > 0 ? dot(w, n) / s3 : NaN;
    const dSkDt = s3 > 0 ? dot(dwdt, n) / s3 - (3 * dot(w, n) * dSpeed) / (s3 * speed) : NaN;
    out.signedKappa = sk;
    out.dSignedKappaDs = speed > 0 ? dSkDt / speed : NaN;
  }
  return out;
}

// End frames

// A blend end frame is { point, d1, d2, d3 } — the position and the first
// three derivatives with respect to some regular parametrization of the
// neighbor, in the neighbor's direction of travel through the blend. The
// blend always runs start-frame -> end-frame, so the start frame's `d1` must
// point into the blend and the end frame's `d1` must point out of it (i.e.
// both agree with the direction the finished chain is traversed). That
// convention is what makes b1 > 0 the correct sign at both ends.
export function blendFrameFromCurve(crv, u, opts = {}) {
  const [C0, C1, C2, C3] = rationalCurveDerivs(crv, u, 3);
  const s = opts.reverse ? -1 : 1;
  // Reversing the direction of travel negates the odd derivatives only
  // (d/d(-t) applied j times contributes (-1)^j) — the same identity
  // `geometricBlend` uses internally to build its far end.
  return { point: C0, d1: scale(C1, s), d2: scale(C2, s * s), d3: scale(C3, s * s * s) };
}

// A straight neighbor: every derivative above the first is exactly zero. Used
// by the corner API, where the two neighbors are polyline edges.
export function blendFrameFromLine(point, direction) {
  return { point: point.slice(0, 3), d1: direction.slice(0, 3), d2: [0, 0, 0], d3: [0, 0, 0] };
}

// The core blend

function binom(n, k) {
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
}

function allFinite(v) { return Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]); }

// Build the k+1 control points at one end of a degree-n Bezier from that
// end's derivative data, using P&T Eq. 1.9 inverted. `derivs[j]` is C^(j) at
// that end (already scaled by b1^j by the caller).
function endControlPoints(point, derivs, n, k) {
  // Delta^j b_0 = C^(j)(0) / (n!/(n-j)!)
  const deltas = [point.slice(0, 3)];
  for (let j = 1; j <= k; j++) {
    let fall = 1;
    for (let i = 0; i < j; i++) fall *= (n - i); // n!/(n-j)!
    deltas.push(scale(derivs[j], 1 / fall));
  }
  const pts = [];
  for (let j = 0; j <= k; j++) {
    let acc = [0, 0, 0];
    for (let i = 0; i <= j; i++) acc = add(acc, scale(deltas[i], binom(j, i)));
    pts.push(acc);
  }
  return pts;
}

// geometricBlend(startFrame, endFrame, opts)
//
//   opts.continuity      1 | 2 | 3   (default 2) -> degree 3 | 5 | 7
//   opts.startMagnitude  |C'(0)| for the blend (default: the chord length)
//   opts.endMagnitude    |C'(1)| for the blend (default: the chord length)
//
// Returns { ok, crv, degree, continuity, chord } or { ok:false, reason }.
// Refuses on a frame whose tangent has collapsed,
// on coincident endpoints, on a non-integer/out-of-range continuity order, on
// non-finite input, and on a result that fails a sampled regularity check.
export function geometricBlend(startFrame, endFrame, opts = {}) {
  const k = opts.continuity === undefined ? 2 : opts.continuity;
  if (!Number.isInteger(k) || k < 1 || k > 3) {
    return { ok: false, reason: `blend continuity order must be 1, 2 or 3 (got ${opts.continuity})` };
  }
  for (const [nm, f] of [['start', startFrame], ['end', endFrame]]) {
    if (!f || !allFinite(f.point) || !allFinite(f.d1) || !allFinite(f.d2) || !allFinite(f.d3)) {
      return { ok: false, reason: `the ${nm} frame contains non-finite values` };
    }
  }
  const s1 = length(startFrame.d1);
  const e1 = length(endFrame.d1);
  if (!(s1 > 1e-12) || !(e1 > 1e-12)) {
    return { ok: false, reason: 'a blend end frame has a degenerate (zero-speed) tangent, cannot blend' };
  }
  const chord = length(sub(endFrame.point, startFrame.point));
  if (!(chord > 1e-12)) {
    return { ok: false, reason: 'blend endpoints are coincident, nothing to blend' };
  }
  const mStart = opts.startMagnitude === undefined ? chord : opts.startMagnitude;
  const mEnd = opts.endMagnitude === undefined ? chord : opts.endMagnitude;
  if (!(mStart > 0) || !(mEnd > 0) || !Number.isFinite(mStart) || !Number.isFinite(mEnd)) {
    return { ok: false, reason: 'blend tangent magnitudes must be positive and finite' };
  }

  const n = 2 * k + 1;
  // b1 (the beta-constraint's first coefficient) is exactly the ratio between
  // the blend's requested end speed and the neighbor's own. b2 = b3 = 0, so
  // C^(j) = b1^j * (neighbor's j-th derivative) — a pure linear
  // reparametrization of the neighbor at the joint. See the header.
  const bStart = mStart / s1;
  const startDerivs = [
    startFrame.point,
    scale(startFrame.d1, bStart),
    scale(startFrame.d2, bStart * bStart),
    scale(startFrame.d3, bStart * bStart * bStart),
  ];
  const head = endControlPoints(startFrame.point, startDerivs, n, k);

  // The far end via the reversal identity C~(t) = C(1-t), for which
  // C~^(j)(0) = (-1)^j C^(j)(1): negate the odd derivatives, run the same
  // routine, then lay the result down from b_n backwards.
  const bEnd = mEnd / e1;
  const endDerivs = [
    endFrame.point,
    scale(endFrame.d1, -bEnd),
    scale(endFrame.d2, bEnd * bEnd),
    scale(endFrame.d3, -(bEnd * bEnd * bEnd)),
  ];
  const tail = endControlPoints(endFrame.point, endDerivs, n, k);

  const ctrlPts = new Array(n + 1);
  for (let i = 0; i <= k; i++) {
    ctrlPts[i] = [...head[i], 1];
    ctrlPts[n - i] = [...tail[i], 1];
  }
  for (const p of ctrlPts) {
    if (!allFinite(p)) return { ok: false, reason: 'blend produced a non-finite control point' };
  }
  const knots = [];
  for (let i = 0; i <= n; i++) knots.push(0);
  for (let i = 0; i <= n; i++) knots.push(1);
  const crv = { degree: n, knots, ctrlPts };

  // Regularity guard. A Bezier whose speed vanishes somewhere in the interior
  // has a cusp there, and every geometric invariant this module reports is
  // undefined at it. Not reached at the default shape fraction (swept over
  // turn angle in the tests) but reachable if a caller forces an extreme
  // magnitude.
  const SPEED_FLOOR = 1e-7;
  for (let i = 0; i <= 64; i++) {
    const [, C1] = rationalCurveDerivs(crv, i / 64, 1);
    if (!(length(C1) > SPEED_FLOOR * chord)) {
      return { ok: false, reason: 'blend shape parameters produce a cusp (vanishing speed) inside the blend' };
    }
  }
  return { ok: true, crv, degree: n, continuity: k, chord };
}

// Sampled curvature extremum — used by the shape optimizer and reported as a
// diagnostic (the blend's tightest radius).
export function peakCurvature(crv, samples = 129) {
  let best = 0, at = 0;
  for (let i = 0; i <= samples; i++) {
    const u = i / samples;
    const [, C1, C2] = rationalCurveDerivs(crv, u, 2);
    const sp = length(C1);
    if (!(sp > 0)) continue;
    const kv = length(cross(C1, C2)) / (sp * sp * sp);
    if (Number.isFinite(kv) && kv > best) { best = kv; at = u; }
  }
  return { kappa: best, u: at };
}

// Corner blend — the drop-in G2/G3 sibling of `filletCornerArc`

// The corner case pins both neighbors to straight edges, so all curvature
// data is zero and the whole blend is determined by one scalar per end — the
// blend's own end speed |C'(0)|, which for a straight neighbor spaces the
// collinear run of control points evenly along the edge:
//
//     b_j = b_0 + j * (L/n) * dIn      for j = 0..k, with L = |C'(0)|
//
// L is expressed as a multiple of the chord between the two tangent points
// (`tangentScale`), not as a fraction of the arc's trim distance: trim =
// R*tan(phi/2) grows without bound as the turn approaches 180 degrees (a
// "radius 6" corner at 179.9999 degrees trims nearly 7 million units back
// along each edge, as the arc does too), while the chord = 2*R*sin(phi/2)
// stays bounded by 2R at every angle. A trim-relative knob would need a very
// different numeric range at each end of the turn-angle sweep and
// degenerates near a reversal; a chord-relative one lands in [0.86, 2.68]
// across the admissible range, so a single search bracket covers every
// corner `filletCornerArc` accepts.
//
// The default is the peak-curvature-minimizing scale, found per corner by
// golden-section search. A constant would not do: the optimum moves with
// turn angle (about 0.9 at a shallow turn, about 2.7 at a near-reversal)
// while the resulting peak curvature barely does — 1.10 to 1.18 times the
// arc's 1/R for G2, 1.12 to 1.32 for G3. Putting b_k on the corner vertex
// (the analog of the arc, whose single interior control point is the vertex)
// clusters k+1 of the 2k+2 control points at the vertex and pulls the blend
// into the corner: 3x (30 degree turn) to 35x (170 degree turn) the arc's
// 1/R.
//
// The optimum depends only on (k, turn angle), never on radius: scaling R
// scales the chord and every control point by the same factor, so curvature
// scales as 1/R and the optimal scale is unchanged. Memoized on that pair
// (the invariance is asserted in the tests).
const TANGENT_SCALE_CACHE = new Map();
const TANGENT_SCALE_LO = 0.35;
const TANGENT_SCALE_HI = 3.4;
const TANGENT_SCALE_SCAN = 48;

function blendFromCornerTrim(p0, dIn, p2, dOut, chord, k, tangentScale) {
  const mag = tangentScale * chord;
  return geometricBlend(
    blendFrameFromLine(p0, dIn),
    blendFrameFromLine(p2, dOut),
    { continuity: k, startMagnitude: mag, endMagnitude: mag },
  );
}

// Coarse scan, then golden-section refine within the winning sample's two
// neighbors. Golden section alone is not enough: sampled peak curvature is
// not unimodal across the full bracket. Past roughly 3x the chord the
// blend's control polygon reaches so far past the corner that the curve
// doubles back on itself, and the sampled peak drops again into a second,
// spurious basin (at a 90-degree turn: 1.22 at scale 1.0, 12671 at 3.0, then
// back down to 5.1 at 5.0). A pure golden-section search can land in that
// basin and return a scale an order of magnitude too large. Bracketing by a
// scan over the region where the blend is well-behaved avoids it. A
// bracketing failure would cost shape only, never continuity — every
// positive scale yields the same continuity order — and the result is
// clamped into range.
function optimalTangentScale(k, phi) {
  const key = `${k}:${phi.toFixed(9)}`;
  const hit = TANGENT_SCALE_CACHE.get(key);
  if (hit !== undefined) return hit;
  // Canonical unit-radius corner, built from the chord rather than the trim
  // so the search stays numerically clean at a near-reversal (where the
  // actual corner's tangent points sit millions of units from a vertex
  // that is effectively at infinity). Chord along +x, the two edge directions
  // symmetric about it by half the turn angle each.
  const half = phi / 2;
  const chord = 2 * Math.sin(half);
  const dIn = [Math.cos(half), -Math.sin(half), 0];
  const dOut = [Math.cos(half), Math.sin(half), 0];
  const p0 = [-chord / 2, 0, 0];
  const p2 = [chord / 2, 0, 0];
  const f = (scale) => {
    const b = blendFromCornerTrim(p0, dIn, p2, dOut, chord, k, scale);
    if (!b.ok) return Infinity;
    return peakCurvature(b.crv, 96).kappa;
  };
  const step = (TANGENT_SCALE_HI - TANGENT_SCALE_LO) / TANGENT_SCALE_SCAN;
  let bestI = 0, bestF = Infinity;
  for (let i = 0; i <= TANGENT_SCALE_SCAN; i++) {
    const v = f(TANGENT_SCALE_LO + i * step);
    if (v < bestF) { bestF = v; bestI = i; }
  }
  const g = (Math.sqrt(5) - 1) / 2;
  let a = TANGENT_SCALE_LO + Math.max(0, bestI - 1) * step;
  let b = TANGENT_SCALE_LO + Math.min(TANGENT_SCALE_SCAN, bestI + 1) * step;
  let c = b - g * (b - a), d = a + g * (b - a);
  let fc = f(c), fd = f(d);
  for (let i = 0; i < 40; i++) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
    else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
  }
  const best = Math.min(TANGENT_SCALE_HI, Math.max(TANGENT_SCALE_LO, (a + b) / 2));
  TANGENT_SCALE_CACHE.set(key, best);
  return best;
}

// blendCornerCurve(vertex, prevPt, nextPt, radius, planeNormal, opts)
//
// Signature-compatible with `filletCornerArc` plus an `opts`. Returns the
// same segment shapes the fillet functions emit:
//   continuity 1 -> { type:'arc',   p0, apex, p2, weight }   (delegated
//                    verbatim to filletCornerArc — same numbers, same
//                    refusals, no reimplementation)
//   continuity 2 -> { type:'blend', crv }  degree 5
//   continuity 3 -> { type:'blend', crv }  degree 7
// alongside the diagnostics `trim`, `turnAngle`, `tangentScale` and (for a
// blend) `peakKappa`.
//
// The tangent points p0/p2 are identical to the G1 arc's for the same
// radius, so the blend is a drop-in for the arc inside an existing chain
// (the neighboring straight runs do not move). The blend does not have
// radius R anywhere; it spans the same corner an R arc would. Once
// continuity > 1, `radius` selects the size of the corner treatment, not a
// constant radius of curvature.
export function blendCornerCurve(vertex, prevPt, nextPt, radius, planeNormal, opts = {}) {
  const k = opts.continuity === undefined ? 2 : opts.continuity;
  if (!Number.isInteger(k) || k < 1 || k > 3) {
    return { ok: false, reason: `blend continuity order must be 1, 2 or 3 (got ${opts.continuity})` };
  }
  const arc = filletCornerArc(vertex, prevPt, nextPt, radius, planeNormal);
  if (!arc.ok) return arc; // filletCornerArc decides what a fillet-able corner is
  if (k === 1) {
    return {
      ok: true,
      continuity: 1,
      segment: { type: 'arc', p0: arc.p0, apex: arc.apex, p2: arc.p2, weight: arc.weight },
      degree: 2, // the arc is a rational quadratic — reported for API parity with the polynomial blends below
      trim: arc.trim,
      turnAngle: arc.turnAngle,
    };
  }
  return blendFromArcSegment(arc, k, opts);
}

// Shared by blendCornerCurve and blendPolyline: given a validated
// `filletCornerArc` result, build the higher-continuity replacement across
// the same tangent points.
function blendFromArcSegment(arc, k, opts) {
  const dIn = normalize(sub(arc.apex, arc.p0));
  const dOut = normalize(sub(arc.p2, arc.apex));
  const phi = Math.abs(arc.turnAngle);
  const chord = length(sub(arc.p2, arc.p0)); // = 2*R*sin(phi/2), bounded at every turn angle — see the header
  let scale = opts.tangentScale === undefined ? 'min-curvature' : opts.tangentScale;
  if (scale === 'min-curvature') scale = optimalTangentScale(k, phi);
  if (!Number.isFinite(scale) || !(scale > 0)) {
    return { ok: false, reason: `blend tangentScale must be a positive number or 'min-curvature' (got ${opts.tangentScale})` };
  }
  const b = blendFromCornerTrim(arc.p0, dIn, arc.p2, dOut, chord, k, scale);
  if (!b.ok) return b;
  return {
    ok: true,
    continuity: k,
    segment: { type: 'blend', crv: b.crv },
    crv: b.crv,
    degree: b.degree,
    trim: arc.trim,
    turnAngle: arc.turnAngle,
    tangentScale: scale,
    peakKappa: peakCurvature(b.crv).kappa,
  };
}

// Polyline / rail blend

// blendPolyline(points, radius, opts) mirrors `filletOpenPolyline`: same
// `opts.closed` / `opts.cornerFilter`, same segment-list return shape, same
// refusal + `maxSafeRadius` auto-clamp report.
//
// It is implemented on top of `filletOpenPolyline`, not beside it. The trim
// budget, the collinear and near-180 skips, the open-vs-closed edge
// accounting, the zero-length remainder omission and the too-large-radius
// clamp (test/fillet-open-polyline.test.mjs) all depend only on the tangent
// points, which this module keeps identical to the arc's. So: run
// `filletOpenPolyline`, then swap each emitted 'arc' for its
// higher-continuity replacement across the same p0/p2. `continuity: 1`
// returns `filletOpenPolyline`'s result unchanged (asserted in the tests).
//
// Where two corners are close enough that the straight run between them all
// but disappears, two blends are separated only by a vanishing line. Both
// have zero end curvature (straight-edge neighbors) and so does the run, so
// every seam there is G2/G3 with no special case. Two blends cannot become
// literally adjacent: `filletOpenPolyline`'s zero-length-remainder omission
// needs the remainder under 1e-9 absolute while its trim budget refuses at
// needed/edgeLen >= 1 - 1e-9, and for any edge longer than one unit those two
// windows do not overlap. The tests measure the reachable limit (a residual
// run a fraction of a percent of the trim).
export function blendPolyline(points, radius, opts = {}) {
  const k = opts.continuity === undefined ? 2 : opts.continuity;
  if (!Number.isInteger(k) || k < 1 || k > 3) {
    return { ok: false, reason: `blend continuity order must be 1, 2 or 3 (got ${opts.continuity})` };
  }
  const base = filletOpenPolyline(points, radius, opts);
  if (!base.ok) return base;
  if (k === 1) return base; // the arc fillet, unchanged

  const segments = [];
  for (const s of base.segments) {
    if (s.type !== 'arc') { segments.push(s); continue; }
    // Reconstruct the corner's turn from the arc segment itself: the trim
    // length is |apex - p0| by construction, and the turn angle follows from
    // the two edge directions.
    const dIn = normalize(sub(s.apex, s.p0));
    const dOut = normalize(sub(s.p2, s.apex));
    const trim = length(sub(s.apex, s.p0));
    const phi = Math.atan2(length(cross(dIn, dOut)), dot(dIn, dOut)); // magnitude only; the arc's own sign convention is irrelevant to the blend
    const rep = blendFromArcSegment({ ok: true, p0: s.p0, apex: s.apex, p2: s.p2, trim, turnAngle: phi }, k, opts);
    if (!rep.ok) return rep;
    segments.push(rep.segment);
  }
  return { ok: true, segments, closed: base.closed, cornerCount: base.cornerCount, continuity: k };
}

// Composes a blend segment list into one NurbsCrv, the analog of
// `filletSegmentsToCurve` (which does not handle the 'blend' type). Uses
// `joinCurvesC0`, so the result is one PolyCurve.
//
// `joinCurvesC0` degree-elevates every segment to the highest degree present
// (5 or 7 here) and stacks full-multiplicity knots at each joint, so the
// composed curve is C0 in its knot vector. The geometry is unchanged by
// elevation and rescaling, so the curve is still G2/G3 at those joints;
// test/blend.test.mjs measures curvature on both sides of each joint.
export function blendSegmentsToCurve(segments) {
  if (!segments.length) return null;
  const crvs = segments.map((s) => {
    if (s.type === 'line') return { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...s.a, 1], [...s.b, 1]] };
    if (s.type === 'arc') return { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...s.p0, 1], [...s.apex, s.weight], [...s.p2, 1]] };
    if (s.type === 'blend') return s.crv;
    throw new Error(`blendSegmentsToCurve: unknown segment type '${s.type}'`);
  });
  return joinCurvesC0(crvs);
}

// General curve-to-curve blend

// blendCurves(crvA, uA, crvB, uB, opts) — a G^k blend from a point on one
// curve to a point on another, matching each neighbor's position,
// tangent direction, curvature (G2) and dkappa/ds + torsion (G3) at the join.
// Neighbor degree, rationality and planarity are all irrelevant to the
// construction: it reads derivatives, nothing else. A degree-2 rational arc
// blended into a degree-4 polynomial is the same code path as two lines.
//
// `opts.reverseA` / `opts.reverseB` flip which way each neighbor is traversed
// through the blend; the defaults assume the chain runs
//     ...crvA up to uA -> blend -> crvB onward from uB...
// Trimming the two neighbors back to uA/uB is the caller's job
// (`extractSubCurve` in kernel/knots.mjs does it) — this returns the blend
// alone.
//
// `opts.startMagnitude` / `opts.endMagnitude` default to the chord length
// between the two join points, the standard Hermite choice. There is no
// min-curvature search here: unlike the corner case, peak curvature is not
// an evident objective once the neighbors themselves are curved.
export function blendCurves(crvA, uA, crvB, uB, opts = {}) {
  assertCurve(crvA, 'blendCurves'); assertCurve(crvB, 'blendCurves');
  const start = blendFrameFromCurve(crvA, uA, { reverse: !!opts.reverseA });
  const end = blendFrameFromCurve(crvB, uB, { reverse: !!opts.reverseB });
  return geometricBlend(start, end, opts);
}

// nearestCurveEndpoints(crvA, crvB) — default parameters for `blendCurves`
// when no point along either curve was picked: blend from whichever end of
// each curve sits nearest the other. Checks the 4 combinations of (A's
// start/end, B's start/end) and returns the closest pair as
// uA/uB/reverseA/reverseB for `blendCurves`.
//
// The reverse-flag rule, from blendCurves' convention ("...crvA up to uA ->
// blend -> crvB onward from uB..."): a frame's derivative must point along
// the remaining part of its own curve. At A's end (domain max) that is A's
// forward tangent — unreversed. At A's start (domain min) the remaining part
// of A lies at higher u, so the blend must leave backward — reversed. B is
// the mirror image: unreversed at its start, reversed at its end. So:
// reverse the A candidate iff it is A's start; reverse the B candidate iff
// it is B's end.
export function nearestCurveEndpoints(crvA, crvB) {
  const uMinA = crvA.knots[0], uMaxA = crvA.knots[crvA.knots.length - 1];
  const uMinB = crvB.knots[0], uMaxB = crvB.knots[crvB.knots.length - 1];
  const aStart = curvePoint(crvA, uMinA), aEnd = curvePoint(crvA, uMaxA);
  const bStart = curvePoint(crvB, uMinB), bEnd = curvePoint(crvB, uMaxB);
  const dist = (p, q) => length(sub(p, q));
  const candidates = [
    { uA: uMaxA, endA: 'end', reverseA: false, uB: uMinB, endB: 'start', reverseB: false, d: dist(aEnd, bStart) },
    { uA: uMaxA, endA: 'end', reverseA: false, uB: uMaxB, endB: 'end', reverseB: true, d: dist(aEnd, bEnd) },
    { uA: uMinA, endA: 'start', reverseA: true, uB: uMinB, endB: 'start', reverseB: false, d: dist(aStart, bStart) },
    { uA: uMinA, endA: 'start', reverseA: true, uB: uMaxB, endB: 'end', reverseB: true, d: dist(aStart, bEnd) },
  ];
  candidates.sort((a, b) => a.d - b.d);
  const best = candidates[0];
  return { uA: best.uA, endA: best.endA, reverseA: best.reverseA, uB: best.uB, endB: best.endB, reverseB: best.reverseB, distance: best.d };
}

// nearestEndpointToPoint(crv, point) — which of a curve's two ends is nearer
// a given 3D point (typically where a click landed on that curve). Unlike
// nearestCurveEndpoints, which searches the 4 combinations across two curves,
// this looks at one curve only, so a click-driven two-curve blend calls it
// once per curve and each curve's end choice is independent of where the
// other was clicked. The point need not come from a curve pick.
export function nearestEndpointToPoint(crv, point) {
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const startPt = curvePoint(crv, uMin);
  const endPt = curvePoint(crv, uMax);
  const dStart = length(sub(startPt, point));
  const dEnd = length(sub(endPt, point));
  return dStart <= dEnd
    ? { end: 'start', u: uMin, point: startPt, distance: dStart }
    : { end: 'end', u: uMax, point: endPt, distance: dEnd };
}
