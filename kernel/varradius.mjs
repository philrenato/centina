/* VARIABLE-RADIUS EDGE BLENDING — a radius that changes along the edge, and the
   conditions under which the surface it asks for exists.
   ==================================================================

   PRIOR ART:

     · Peternell, M. & Pottmann, H., "Computing Rational Parametrizations of
       Canal Surfaces." Journal of Symbolic Computation 23(2-3), pp. 255-266,
       1997. DOI 10.1006/jsco.1996.0087. Two things are taken from here: the
       existence condition |m'|^2 - r'^2 >= 0, which `variableRadiusFeasible` in
       fillet.mjs already encodes and which this module ENFORCES; and the
       characteristic circle of the sphere family, which is what a section of a
       variable-radius blend actually is.
     · Fritsch, F. N. & Carlson, R. E., "Monotone Piecewise Cubic
       Interpolation." SIAM Journal on Numerical Analysis 17(2), pp. 238-246,
       1980. DOI 10.1137/0717021. The shape-preserving cubic used for the radius
       profile, and the reason it is used rather than a smoother spline.
     · Lukacs, G., "Differential geometry of G1 variable radius rolling ball
       blend surfaces." Computer Aided Geometric Design 15(6), pp. 585-613,
       1998. DOI 10.1016/S0167-8396(98)00006-5. Names the REGRESSIVE points a
       variable-radius blend develops. NOT implemented here — see the scope note
       at the bottom of this header.

   THE ONE THING A VARIABLE RADIUS CHANGES ABOUT THE SECTION, and it is not
   obvious: THE CONTACT CIRCLE IS NO LONGER A GREAT CIRCLE OF THE BALL.

   A ball of radius r(t) centred at m(t) touches the finished surface along its
   characteristic circle, the set of points where the neighbouring balls of the
   family agree. Differentiating |p - m(t)|^2 = r(t)^2 gives

       (p - m) . m' + r r' = 0,

   so every contact point p = m + r n satisfies n . T = -r'/|m'| =: tilt, with T
   the unit spine tangent. The contact points therefore lie on a SMALL circle:
   its centre is m + r*tilt*T, offset from the ball centre ALONG the spine, and
   its radius is r*sqrt(1 - tilt^2), shrunk. Both reduce to the ball centre and r
   when r' = 0, which is why a constant radius never exposes this.

   Skinning great-circle arcs instead — which is what a constant-radius section
   builder does — puts the mid-arc a fixed fraction of r off the true envelope,
   on the order of 8% of r at |dr/ds| = 0.2 on a right-angled edge, and that
   error is STRUCTURAL: it does not fall with section count, because every
   section is individually in the wrong plane. It is also invisible to any
   instrument that only asks "is this point r from the ball centre", since a
   great-circle arc lies exactly on the ball. What it is not invisible to is the
   signed canal measure below, which asks the stronger question: is this point on
   the boundary of the union of balls, or inside it.

   WHAT THIS MODULE DOES NOT DO. It does not detect REGRESSIVE points (Lukacs
   1998) — the cusps and local self-intersections a variable-radius blend can
   develop where the tube folds back on itself even though the existence
   condition holds pointwise. The feasibility test here is the Peternell &
   Pottmann inequality, which is NECESSARY and not sufficient, and every field
   name says so.

   Plain data throughout: a point is [x, y, z]. Nothing here imports a vector
   library or reaches the DOM. */

import { sectionArc, blendSurfaceToTolerance, variableRadiusFeasible } from './fillet.mjs';

const EPS = 1e-12;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
function norm(a) {
  const L = len(a);
  if (!(L > EPS)) return null;
  return [a[0] / L, a[1] / L, a[2] / L];
}

/* ─────────────────────────────────────────────────────────────────────────
   THE RADIUS PROFILE
   ───────────────────────────────────────────────────────────────────────── */

/**
 * A RADIUS PROFILE: an ordered set of STOPS, each a parameter t in [0,1] with a
 * radius, evaluated to a smooth radius function r(t).
 *
 * ⚠ SHAPE-PRESERVING, NOT SMOOTHEST. The interpolant is the monotone piecewise
 * cubic Hermite of Fritsch & Carlson 1980, and the choice is load-bearing rather
 * than a matter of taste:
 *
 *   · A NATURAL or Catmull-Rom cubic is C2 and OVERSHOOTS. Overshoot in a radius
 *     profile is not cosmetic — it is a fillet that grows larger than any radius
 *     the user asked for, and on a profile that drops steeply it dips BELOW
 *     ZERO, which is a ball of negative radius and a blend inverted through its
 *     own spine. The overshoot also invents |r'| that the stops never contained,
 *     so a profile whose stops are comfortably feasible can be refused because
 *     of a bulge nobody asked for.
 *   · A LINEAR profile has no overshoot but is only C0, so r' jumps at every
 *     stop. r' sets the tilt of every section (see `variableRadiusSection`), so
 *     a jump in r' is a visible crease in the finished blend at each stop.
 *   · Fritsch & Carlson is C1 and shape-preserving: on every segment the
 *     interpolant is monotone between its two stops, so the whole function is
 *     bounded by min and max of the stop radii — positive stops give a positive
 *     radius everywhere, with no test needed to hope for it — and at an interior
 *     stop that is a local extremum of the data the slope is set to zero, so the
 *     waist of a thick-thin-thick profile lands exactly on its stop.
 *
 * WHAT IS GIVEN UP: C1 and not C2, so the blend's curvature steps at each stop.
 * Curvature continuity would need the overshoot back, or more stops.
 *
 * TWO STOPS ARE A STRAIGHT TAPER, exactly and by construction: with no interior
 * stop the end slopes are both the single segment slope, and the Hermite cubic
 * through two points with equal end slopes IS the straight line. "Starts at X,
 * ends at Y" is therefore a closed form, not an approximation of one.
 *
 * Stops are `[t, radius]` pairs or `{ t, radius }` records. The first must sit
 * at t = 0 and the last at t = 1: a profile that covers only part of the edge
 * would have to invent the rest, and a constant extrapolation would put a slope
 * discontinuity — a crease — at the first and last stop.
 */
export function radiusProfile(stops) {
  if (!Array.isArray(stops) || stops.length < 2) {
    return { ok: false, reason: 'a radius profile needs at least two stops — one radius alone is a constant-radius fillet' };
  }
  const t = [], r = [];
  for (let i = 0; i < stops.length; i++) {
    const s = stops[i];
    const ti = Array.isArray(s) ? s[0] : (s ? s.t : undefined);
    const ri = Array.isArray(s) ? s[1] : (s ? s.radius : undefined);
    if (!Number.isFinite(ti) || !Number.isFinite(ri)) {
      return { ok: false, reason: `stop ${i} is not a finite (t, radius) pair` };
    }
    if (!(ri > 0)) return { ok: false, reason: `stop ${i} asks for radius ${ri}; a rolling ball has a positive radius` };
    if (ti < 0 || ti > 1) return { ok: false, reason: `stop ${i} sits at t = ${ti}, outside the edge's own [0,1]` };
    if (i > 0 && !(ti > t[i - 1])) {
      // Equal parameters are the trap here, not reversed ones: two stops at the
      // same t divide by a zero segment length and would emit Infinity slopes
      // rather than refusing.
      return { ok: false, reason: `stops must be strictly ordered along the edge; stop ${i} is at t = ${ti} and stop ${i - 1} is at t = ${t[i - 1]}` };
    }
    t.push(ti); r.push(ri);
  }
  if (t[0] !== 0 || t[t.length - 1] !== 1) {
    return { ok: false, reason: `a profile must span the whole edge — the first stop at t = ${t[0]} and the last at t = ${t[t.length - 1]} leave part of it with no radius` };
  }
  const n = t.length;
  const h = [], del = [];
  for (let i = 0; i + 1 < n; i++) {
    h.push(t[i + 1] - t[i]);
    del.push((r[i + 1] - r[i]) / h[i]);
  }
  const d = new Array(n).fill(0);
  for (let i = 1; i + 1 < n; i++) {
    // A sign change in the data means an interior extremum: the slope is zeroed
    // so the interpolant's extremum lands ON the stop rather than past it.
    if (del[i - 1] * del[i] <= 0) { d[i] = 0; continue; }
    const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1];
    d[i] = (w1 + w2) / (w1 / del[i - 1] + w2 / del[i]);
  }
  const endSlope = (h1, h2, d1, d2) => {
    if (h2 === undefined) return d1; // two stops: the segment slope, so the cubic IS the line
    let s = ((2 * h1 + h2) * d1 - h1 * d2) / (h1 + h2);
    if (s * d1 <= 0) s = 0;
    else if (d1 * d2 <= 0 && Math.abs(s) > Math.abs(3 * d1)) s = 3 * d1;
    return s;
  };
  d[0] = endSlope(h[0], h[1], del[0], del[1]);
  d[n - 1] = endSlope(h[n - 2], h[n - 3], del[n - 2], del[n - 3]);

  const seg = (x) => {
    let i = 0;
    while (i + 2 < n && x >= t[i + 1]) i++;
    return i;
  };
  const radiusAt = (x) => {
    const u = x < 0 ? 0 : x > 1 ? 1 : x;
    const i = seg(u);
    const s = (u - t[i]) / h[i], s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * r[i] + (s3 - 2 * s2 + s) * h[i] * d[i]
      + (-2 * s3 + 3 * s2) * r[i + 1] + (s3 - s2) * h[i] * d[i + 1];
  };
  const slopeAt = (x) => {
    const u = x < 0 ? 0 : x > 1 ? 1 : x;
    const i = seg(u);
    const s = (u - t[i]) / h[i], s2 = s * s;
    return (6 * s2 - 6 * s) * r[i] / h[i] + (3 * s2 - 4 * s + 1) * d[i]
      + (-6 * s2 + 6 * s) * r[i + 1] / h[i] + (3 * s2 - 2 * s) * d[i + 1];
  };
  return {
    ok: true,
    kind: 'monotone-cubic-hermite',
    stops: t.map((ti, i) => ({ t: ti, radius: r[i] })),
    slopes: d.slice(),
    radiusAt,
    slopeAt,
    // Exact, not sampled: a shape-preserving interpolant attains its extremes at
    // the stops, so these are the stop radii themselves.
    minRadius: Math.min(...r),
    maxRadius: Math.max(...r),
  };
}

/**
 * THE STATED DEFAULT FOR A CHAIN: thick at both ends, thin in the middle.
 *
 * Three stops, the two ends equal, the middle smaller. Named rather than left to
 * the caller to assemble because it is the shape the app offers by default, and
 * because the shape-preserving interpolant gives it a property worth relying on:
 * the waist sits exactly at `at` with exactly `waist`, with zero slope there, and
 * the radius never exceeds `ends` anywhere.
 *
 * `waistFraction` is the one-knob form — waist = ends * fraction — so the UI can
 * expose a single 0..1 control instead of two coupled lengths.
 */
export function thickThinThickProfile({ ends, waist, waistFraction, at = 0.5 } = {}) {
  if (!(ends > 0)) return { ok: false, reason: `the end radius must be positive; got ${ends}` };
  let w = waist;
  if (w == null && Number.isFinite(waistFraction)) w = ends * waistFraction;
  if (!(w > 0)) return { ok: false, reason: `the waist radius must be positive; got ${w}` };
  if (!(w < ends)) {
    // A waist equal to the ends is a constant profile wearing this function's
    // name. Refused rather than silently built, so a caller cannot believe it
    // asked for a waist and get none.
    return { ok: false, reason: `a waist of ${w} is not thinner than the ends at ${ends} — that is a constant radius, and radiusProfile builds it` };
  }
  if (!(at > 0) || !(at < 1)) return { ok: false, reason: `the waist must sit strictly inside the edge; got t = ${at}` };
  const p = radiusProfile([[0, ends], [at, w], [1, ends]]);
  if (!p.ok) return p;
  return { ...p, shape: 'thick-thin-thick', ends, waist: w, waistAt: at };
}

/* ─────────────────────────────────────────────────────────────────────────
   THE SECTION
   ───────────────────────────────────────────────────────────────────────── */

/**
 * ONE CROSS-SECTION OF A VARIABLE-RADIUS BLEND — the characteristic circle of
 * the sphere family, not a great circle of the ball.
 *
 * `radiusRate` is dr/ds: how fast the radius changes per unit of SPINE ARC
 * LENGTH, which is the scale-free form of the Peternell & Pottmann condition —
 * |m'|^2 >= r'^2 is exactly |dr/ds| <= 1. It is refused at 1 and above, where
 * the contact circle has shrunk to a point and there is no section left.
 *
 * ⚠ THE TILT IS AN INVARIANT OF THE INPUT, NOT A FREE PARAMETER, and checking it
 * is the only way this function can tell whether the caller's spine, radius and
 * normals describe a ball that is really rolling. For ANY ball kept tangent to a
 * fixed surface, differentiating m = p - r n with n . p' = 0 and n . n' = 0
 * gives n . m' = -r' identically. So both touch directions must have the SAME
 * component along the spine tangent, equal to -dr/ds. A caller who supplies a
 * spine that does not keep the ball in contact violates that, and the residual
 * is the size of the violation.
 *
 * With tilt = 0 every line below reduces to the constant-radius construction in
 * fillet.mjs, exactly and not merely closely — the offset along the tangent is
 * multiplied by zero and the contact radius by one.
 */
export function variableRadiusSection({ centre, radius, toTouchA, toTouchB, spineTangent, radiusRate, maxTiltMismatch = 1e-6 }) {
  if (!(radius > 0)) return { ok: false, reason: `radius must be positive; got ${radius}` };
  if (!Number.isFinite(radiusRate)) return { ok: false, reason: 'the radius rate dr/ds is missing or not a number — with no rate there is no tilt and the section would silently be built as a constant-radius one' };
  const nA = norm(toTouchA), nB = norm(toTouchB);
  if (!nA || !nB) return { ok: false, reason: 'a touch direction has no direction' };
  const T = norm(spineTangent);
  if (!T) return { ok: false, reason: 'the spine tangent has no direction' };
  if (!(Math.abs(radiusRate) < 1)) {
    return {
      ok: false,
      reason: `the radius changes at ${Math.abs(radiusRate).toFixed(4)} per unit of spine advance, and 1 is the ceiling — at or past it the contact circle has collapsed to a point and no envelope exists`,
      radiusRate,
    };
  }
  const tilt = -radiusRate;
  const misA = Math.abs(dot(nA, T) - tilt), misB = Math.abs(dot(nB, T) - tilt);
  const tiltMismatch = Math.max(misA, misB);
  if (!(tiltMismatch <= maxTiltMismatch)) {
    return {
      ok: false,
      reason: `the ball is not rolling in contact here: a touch direction leans ${tiltMismatch.toExponential(3)} away from the -dr/ds the spine and radius imply, past the ${maxTiltMismatch} tolerance — the spine, the radius and the touch directions do not describe the same ball`,
      tiltMismatch,
    };
  }
  const cosBetween = Math.max(-1, Math.min(1, dot(nA, nB)));
  if (cosBetween > 1 - 1e-12) return { ok: false, reason: 'the faces are tangent here — there is no edge to blend' };
  if (cosBetween < -1 + 1e-12) return { ok: false, reason: 'the faces are opposed — a ball cannot touch both' };
  const tangencyA = add(centre, mul(nA, radius));
  const tangencyB = add(centre, mul(nB, radius));
  /* THE ONE PLACE EXACTNESS AT tilt = 0 IS DELIBERATE. A constant profile must
     reproduce the constant-radius blend BIT FOR BIT, not to within a rounding —
     that is the closed form this construction is checked against. Adding
     `radius * 0 * T` component-wise would be exact for every finite coordinate
     but turns a -0 into a +0, so the branch is taken explicitly. */
  const offset = radius * tilt;
  const contactCentre = offset === 0 ? centre.slice() : add(centre, mul(T, offset));
  const contactRadius = tilt === 0 ? radius : radius * Math.sqrt(1 - tilt * tilt);
  // The sweep is measured at the CONTACT circle's centre, and a tilt opens it:
  // cos(sweep) = (nA.nB - tilt^2) / (1 - tilt^2), which exceeds a right angle
  // wherever the tilt is large and the faces are already near-tangent.
  const cosSweep = Math.max(-1, Math.min(1, (cosBetween - tilt * tilt) / (1 - tilt * tilt)));
  const sweep = Math.acos(cosSweep);
  if (!(sweep < Math.PI - 1e-9)) {
    return { ok: false, reason: `the tilted contact arc sweeps ${(sweep * 180 / Math.PI).toFixed(2)} degrees, which one rational quadratic cannot carry — the radius is changing too fast for how flat the edge is here`, sweep };
  }
  return {
    ok: true,
    centre, radius,
    tangencyA, tangencyB,
    spineTangent: T,
    // The circle the ball actually touches along, offset along the spine and
    // shrunk. Equal to (centre, radius) exactly when the radius is not changing.
    contactCentre, contactRadius,
    tilt, radiusRate, tiltMismatch, sweep,
  };
}

/**
 * The same exact rational quadratic `sectionArc` builds, taken about the CONTACT
 * circle rather than about the ball centre. Feeding it the ball centre instead
 * is the great-circle error described in this module's header: the arc still
 * passes through both tangency points and still measures `radius` from the ball
 * centre, so only a signed envelope measure can see that it is wrong.
 */
export function variableRadiusSectionArc(section) {
  if (!section || !section.ok) return null;
  return sectionArc({ ok: true, centre: section.contactCentre, tangencyA: section.tangencyA, tangencyB: section.tangencyB });
}

/* ─────────────────────────────────────────────────────────────────────────
   THE SPINE FRAME
   ───────────────────────────────────────────────────────────────────────── */

/**
 * THE SPINE TANGENT AND dr/ds AT ONE PARAMETER.
 *
 * ⚠ THE SPINE IS NOT AN INPUT — IT DEPENDS ON THE RADIUS. A ball of radius r in
 * a right-angled corner sits at (r, r); change r and the centre moves. So the
 * centre path is m(t) = c(t, r(t)) and its derivative is
 *
 *     dm/dt = dc/dt + r'(t) * dc/dr,
 *
 * with both partials taken of a SMOOTH function of two arguments. Differencing
 * the composed path directly instead would be differencing across the profile's
 * own C1-but-not-C2 joints, where a central difference degrades from second
 * order to first and lands right on the stops a user is most likely to place a
 * section at. Splitting it moves every difference onto smooth ground and takes
 * r' from the profile in closed form.
 *
 * `ballAt(t, radius)` returns `{ centre, toTouchA, toTouchB }` and may also
 * return `dCentreDt` and `dCentreDr` in closed form, in which case no
 * differencing happens at all.
 */
export function spineFrame(ballAt, profile, t, opts = {}) {
  const hT = opts.hT || 1e-5;
  const radius = profile.radiusAt(t);
  const drdt = profile.slopeAt(t);
  const b = ballAt(t, radius);
  if (!b || !Array.isArray(b.centre)) return { ok: false, reason: `the caller supplied no ball centre at t = ${t}` };
  let dcdt = b.dCentreDt, dcdr = b.dCentreDr;
  if (!dcdt) {
    // Second-order everywhere: central inside, three-point one-sided at the ends
    // rather than a first-order forward difference that would quietly halve the
    // accuracy of the two sections most exposed to it.
    const at = (x) => { const q = ballAt(x, radius); return q && q.centre; };
    if (t - hT >= 0 && t + hT <= 1) {
      const p1 = at(t + hT), m1 = at(t - hT);
      if (!p1 || !m1) return { ok: false, reason: 'the caller could not supply a neighbouring ball centre' };
      dcdt = mul(sub(p1, m1), 1 / (2 * hT));
    } else {
      const s = t - hT < 0 ? 1 : -1;
      const f0 = at(t), f1 = at(t + s * hT), f2 = at(t + 2 * s * hT);
      if (!f0 || !f1 || !f2) return { ok: false, reason: 'the caller could not supply a neighbouring ball centre' };
      dcdt = mul(add(add(mul(f0, -3), mul(f1, 4)), mul(f2, -1)), s / (2 * hT));
    }
  }
  if (!dcdr) {
    const hR = Math.min(opts.hR || 1e-5, radius * 0.25);
    const p1 = ballAt(t, radius + hR), m1 = ballAt(t, radius - hR);
    if (!p1 || !m1 || !p1.centre || !m1.centre) return { ok: false, reason: 'the caller could not supply a ball centre at a neighbouring radius' };
    dcdr = mul(sub(p1.centre, m1.centre), 1 / (2 * hR));
  }
  const dm = add(dcdt, mul(dcdr, drdt));
  const speed = len(dm);
  if (!(speed > EPS)) {
    return { ok: false, reason: `the ball centre is stationary at t = ${t} — a spine that does not advance has no tangent and no envelope`, speed };
  }
  return {
    ok: true,
    t, radius,
    centre: b.centre, toTouchA: b.toTouchA, toTouchB: b.toTouchB,
    spineTangent: mul(dm, 1 / speed),
    speed,
    dRadiusDt: drdt,
    radiusRate: drdt / speed,
    analytic: !!(b.dCentreDt && b.dCentreDr),
  };
}

/* ─────────────────────────────────────────────────────────────────────────
   FEASIBILITY, ENFORCED
   ───────────────────────────────────────────────────────────────────────── */

/**
 * WHETHER THIS PROFILE CAN BE BUILT ON THIS EDGE AT ALL, AND WHERE IT CANNOT.
 *
 * The verdict comes from `variableRadiusFeasible` in fillet.mjs — the module of
 * record for the Peternell & Pottmann condition — sampled along the spine the
 * profile actually produces. What is added here is the part a refusal needs to
 * be usable: WHERE along the edge it fails and BY HOW MUCH.
 *
 * ⚠ THE MARGIN AND THE RATE DISAGREE ABOUT "WORST", AND THE RATE IS THE ONE TO
 * QUOTE. The margin ds^2 - dr^2 is an area and scales with how fast the spine
 * happens to be sampled, so on a uniform t-sampling of a non-uniform spine the
 * smallest margin drifts towards wherever the spine is SLOW rather than towards
 * wherever the radius is out of control. |dr|/ds is scale-free, converges to the
 * true |dr/ds| as the sampling refines, and crosses 1 at exactly the same place
 * the margin crosses 0 — so the two never disagree about the verdict, only about
 * which span to name.
 *
 * ⚠⚠ WHERE THIS REFUSAL IS ACTUALLY REACHABLE, because it is not where it looks.
 * For a ball genuinely rolling in CONTACT with fixed faces, n . m' = -r' is an
 * identity (see `variableRadiusSection`), so |r'| = |n . m'| <= |m'| always: the
 * condition cannot be violated, whatever profile is asked for, because the
 * contact constraint drags the spine along at least as fast as the radius grows.
 * A right-angled edge is stronger still — the centre moves in both faces'
 * normal directions at once, capping the rate at 1/sqrt(2) no matter how steep
 * the taper.
 *
 * What violates it is a PRESCRIBED spine: a centre path that does not move when
 * the radius does. That is a variable-radius pipe, and it is also what an app
 * produces the moment it takes the spine from an offset of the edge curve, or
 * reuses a spine sampled from an earlier constant-radius build, and then applies
 * a profile to it. Those are the inputs this refuses, and they are common enough
 * that "the fillet cannot violate it" is not a reason to skip the check.
 *
 * NECESSARY, NOT SUFFICIENT. A profile that passes here can still fold back on
 * itself at a regressive point (Lukacs 1998), which this does not look for.
 */
export function profileFeasibility({ ballAt, profile, samples = 257 } = {}) {
  if (typeof ballAt !== 'function') return { ok: false, reason: 'a ball placement function is required' };
  if (!profile || !profile.ok || typeof profile.radiusAt !== 'function') {
    return { ok: false, reason: `the radius profile is not usable${profile && profile.reason ? ': ' + profile.reason : ''}` };
  }
  const N = Math.max(9, samples | 0);
  const spine = [], radii = [], ts = [];
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const r = profile.radiusAt(t);
    const b = ballAt(t, r);
    if (!b || !Array.isArray(b.centre)) return { ok: false, reason: `no ball centre at t = ${t.toFixed(4)}` };
    ts.push(t); spine.push(b.centre); radii.push(r);
  }
  const verdict = variableRadiusFeasible(spine, radii);
  let worstRate = 0, worstSpan = -1;
  for (let i = 0; i + 1 < N; i++) {
    const ds = len(sub(spine[i + 1], spine[i]));
    const dr = Math.abs(radii[i + 1] - radii[i]);
    const rate = ds > EPS ? dr / ds : (dr > 0 ? Infinity : 0);
    if (rate > worstRate) { worstRate = rate; worstSpan = i; }
  }
  const i = worstSpan < 0 ? 0 : worstSpan;
  const ds = len(sub(spine[i + 1], spine[i]));
  const dr = Math.abs(radii[i + 1] - radii[i]);
  const where = { from: ts[i], to: ts[i + 1], at: 0.5 * (ts[i] + ts[i + 1]) };
  if (verdict.ok) {
    return {
      ok: true,
      worstRate, worstAt: where, samples: N,
      worstMargin: verdict.worstMargin,
      // The headroom a caller can quote: how much steeper the profile could get
      // before it stops existing.
      headroom: 1 - worstRate,
      bound: 'canal-existence-only',
    };
  }
  return {
    ok: false,
    reason: `the radius outruns the edge near t = ${where.at.toFixed(3)}: it changes ${dr.toFixed(4)} while the ball centre moves only ${ds.toFixed(4)}, a rate of ${worstRate.toFixed(3)} against a ceiling of 1 — no envelope exists there. Move the stops ${(worstRate).toFixed(2)}x further apart along the edge, or bring their radii within ${ds.toFixed(4)} of each other.`,
    worstRate, worstAt: where, samples: N,
    worstMargin: verdict.worstMargin,
    excess: dr - ds,
    bound: 'canal-existence-only',
  };
}

/* ─────────────────────────────────────────────────────────────────────────
   MEASURING WHAT WAS BUILT
   ───────────────────────────────────────────────────────────────────────── */

/**
 * HOW FAR A VARIABLE-RADIUS SURFACE STRAYS FROM THE BALLS THAT DEFINED IT.
 *
 * ⚠ THE CONSTANT-RADIUS TEST DOES NOT GENERALISE, and the way it fails is
 * silent. "Every point is r from the spine" becomes "every point is r(t) from
 * m(t) for SOME t", and taking the unsigned min of | |p - m(t)| - r(t) | over t
 * gives zero for any point lying on any sphere of the family — including every
 * point of a great-circle arc built in the wrong plane. That measure reports a
 * perfect blend for the one error a variable radius introduces.
 *
 * What the envelope actually is: the boundary of the union of the balls. So the
 * quantity is the SIGNED
 *
 *     g(p) = min over t of ( |p - m(t)| - r(t) ),
 *
 * which is zero on the envelope, negative for a point swallowed inside a
 * neighbouring ball — a blend that has cut too deep — and positive for a point
 * standing off the balls entirely. Its sign is returned as well as its size,
 * because those two failures need opposite fixes.
 *
 * ⚠ AND THE RULER MEASURES ITSELF. g has a stationary minimum at the contact
 * parameter, so a SAMPLED minimum is second order in the sample spacing and
 * always overstates g — a coarse sample reports a better surface than exists,
 * and by far more than the surface error being judged: at 257 spine samples the
 * sampled minimum on a fixture whose true worst is 4.0e-6 reads 6.6e-3, three
 * orders too kind. Sampling alone is therefore not an instrument at this scale
 * at any density anyone would pay for.
 *
 * So each sampled minimum is REFINED by golden section against the caller's own
 * continuous `at`, which removes the sampling error rather than shrinking it:
 * the same surface measures 4.033441e-6 from 33 samples and from 1025.
 * `sampleFloor` is how far the refinement had to move the answer at the point
 * that set `worst` — the error the unrefined ruler would have reported there —
 * and `refinedBracket` is the parameter width the search closed to, which is
 * what `instrumentBound` is read from. A caller that turns refinement off gets
 * the sampled number and the honest flag that goes with it.
 */
export function canalDeviation(srf, at, evalSrf, opts = {}) {
  const spineSamples = Math.max(9, opts.spineSamples || 257);
  const uSteps = opts.uSteps || 9, vSteps = opts.vSteps || 33;
  const vFrom = opts.vFrom != null ? opts.vFrom : 0.05;
  const vTo = opts.vTo != null ? opts.vTo : 0.95;
  const refine = opts.refine !== false;
  const grid = [];
  for (let i = 0; i < spineSamples; i++) {
    const t = i / (spineSamples - 1);
    const s = at(t);
    if (!s || !Array.isArray(s.centre) || !(s.radius > 0)) return { ok: false, reason: `the spine has no ball at t = ${t}` };
    grid.push([t, s.centre, s.radius]);
  }
  let pCur = null;
  const g = (t) => {
    const s = at(t);
    return len(sub(pCur, s.centre)) - s.radius;
  };
  let worst = 0, worstSigned = 0, worstAt = null, floor = 0, bracket = 0;
  const PHI = (Math.sqrt(5) - 1) / 2;
  for (let j = 0; j < vSteps; j++) {
    const v = vFrom + (vTo - vFrom) * (vSteps > 1 ? j / (vSteps - 1) : 0);
    for (let i = 0; i < uSteps; i++) {
      const u = uSteps > 1 ? i / (uSteps - 1) : 0;
      pCur = evalSrf(srf, u, v);
      let bestI = 0, best = Infinity;
      for (let k = 0; k < grid.length; k++) {
        const d = len(sub(pCur, grid[k][1])) - grid[k][2];
        if (d < best) { best = d; bestI = k; }
      }
      let val = best, moved = 0, width = 0;
      if (refine) {
        // Golden section on the bracketing triple. g is smooth and has a
        // stationary minimum inside it, so this converges to the true minimum
        // rather than to a finer sample of the same overstatement.
        let lo = grid[Math.max(0, bestI - 1)][0], hi = grid[Math.min(grid.length - 1, bestI + 1)][0];
        let x1 = hi - PHI * (hi - lo), x2 = lo + PHI * (hi - lo);
        let f1 = g(x1), f2 = g(x2);
        for (let it = 0; it < 60 && hi - lo > 1e-14; it++) {
          if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - PHI * (hi - lo); f1 = g(x1); }
          else { lo = x1; x1 = x2; f1 = f2; x2 = lo + PHI * (hi - lo); f2 = g(x2); }
        }
        val = Math.min(best, f1, f2);
        moved = Math.abs(best - val);
        width = hi - lo;
      }
      if (Math.abs(val) > worst) {
        worst = Math.abs(val); worstSigned = val; worstAt = { u, v };
        // Both recorded AT THE WORST POINT, not as maxima over the grid: a
        // correction the ruler made somewhere the surface is fine says nothing
        // about the number being reported.
        floor = moved; bracket = width;
      }
    }
  }
  return {
    ok: true,
    worst, worstSigned, worstAt,
    // What the ruler was worth at the point it was read: how far refinement had
    // to move the sampled answer, and how tightly the search closed.
    sampleFloor: floor,
    refinedBracket: refine ? bracket : null,
    refined: refine,
    instrumentBound: refine ? !(bracket < 1e-9) : floor > worst * 0.25,
    inside: worstSigned < 0,
    spineSamples,
  };
}

/* ─────────────────────────────────────────────────────────────────────────
   THE BLEND
   ───────────────────────────────────────────────────────────────────────── */

/**
 * A VARIABLE-RADIUS BLEND, BUILT TO A TOLERANCE AND REFUSED WHEN IT CANNOT
 * EXIST.
 *
 * The section count is not chosen here: it is handed to `blendSurfaceToTolerance`
 * in fillet.mjs, which aims, MEASURES, and corrects. What this supplies are the
 * two things that builder cannot know for a varying radius — how to build a
 * tilted section, and how to measure a surface that has no single radius to be
 * measured against.
 *
 * REFUSES BEFORE IT BUILDS. `profileFeasibility` runs first and a failure comes
 * back as a refusal naming the parameter and the rate, not as a number attached
 * to a surface. A second, independent net sits inside `variableRadiusSection`,
 * which refuses the same condition pointwise at whatever parameters the section
 * count actually lands on.
 *
 * `ballAt(t, radius)` is the caller's geometry: where a ball of that radius sits
 * at that parameter and which way it touches each face.
 */
export function variableRadiusBlend({ ballAt, profile, tolerance = 0.01, evalSrf, feasibilitySamples = 257, ...opts } = {}) {
  if (typeof ballAt !== 'function') return { ok: false, reason: 'a ball placement function is required' };
  if (typeof evalSrf !== 'function') return { ok: false, reason: 'an evaluator is required to measure what was built' };
  if (!profile || !profile.ok) {
    return { ok: false, reason: `the radius profile is not usable${profile && profile.reason ? ': ' + profile.reason : ''}` };
  }
  const feas = profileFeasibility({ ballAt, profile, samples: feasibilitySamples });
  if (!feas.ok) return { ok: false, reason: feas.reason, feasibility: feas };

  const frameOpts = { hT: opts.hT, hR: opts.hR };
  const sectionAt = (t) => {
    const f = spineFrame(ballAt, profile, t, frameOpts);
    return f.ok ? f : null;
  };
  const sectionArcFor = (spec) => {
    const s = variableRadiusSection({
      centre: spec.centre, radius: spec.radius,
      toTouchA: spec.toTouchA, toTouchB: spec.toTouchB,
      spineTangent: spec.spineTangent, radiusRate: spec.radiusRate,
      maxTiltMismatch: opts.maxTiltMismatch != null ? opts.maxTiltMismatch : 1e-6,
    });
    if (!s.ok) return { ok: false, reason: `at t = ${spec.t.toFixed(4)}: ${s.reason}` };
    const arc = variableRadiusSectionArc(s);
    if (!arc) return { ok: false, reason: `at t = ${spec.t.toFixed(4)}: the contact arc could not be formed` };
    return { ok: true, arc };
  };
  const at = (t) => {
    const r = profile.radiusAt(t);
    const b = ballAt(t, r);
    return b && b.centre ? { centre: b.centre, radius: r } : null;
  };
  const measure = (srf) => {
    const d = canalDeviation(srf, at, evalSrf, {
      spineSamples: opts.spineSamples || 257,
      uSteps: opts.uSteps || 9,
      vSteps: opts.vSteps || 33,
    });
    return d.ok ? { worst: d.worst, instrumentBound: d.instrumentBound, floor: d.sampleFloor, signed: d.worstSigned } : null;
  };
  const built = blendSurfaceToTolerance(sectionAt, tolerance, {
    ...opts, evalSrf, sectionArcFor, measure,
  });
  if (!built.ok) return built;
  return {
    ...built,
    profile: { kind: profile.kind, stops: profile.stops, minRadius: profile.minRadius, maxRadius: profile.maxRadius },
    feasibility: feas,
    // The deviation above is a SIGNED envelope distance, not a distance from one
    // radius, and a caller quoting it should say which.
    deviationMeasure: 'signed distance to the boundary of the union of balls',
  };
}
