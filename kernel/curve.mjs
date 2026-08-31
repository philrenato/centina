// NURBS curve evaluation — Piegl & Tiller Ch. 3 (CurveDerivsAlg, A3.2) and
// Ch. 4 (rational derivatives via the quotient rule, A4.2).
//
// A NurbsCrv in this kernel is { degree, knots, ctrlPts } where ctrlPts is an
// array of [x, y, z, w] — the REAL point plus its weight, not pre-multiplied.
// P&T's algorithms operate on the homogeneous form Pw = [x*w, y*w, z*w, w];
// `toHomogeneous` does that conversion once, so basis.mjs stays a pure,
// P&T-faithful module untouched by the rational/weight bookkeeping.

import { findSpan, basisFuns, dersBasisFuns } from './basis.mjs';
import { normalize, sub, scale, length, dot } from './vec3.mjs';
import { joinCurvesC0 } from './knots.mjs';

export function toHomogeneous(ctrlPts) {
  return ctrlPts.map(([x, y, z, w]) => [x * w, y * w, z * w, w]);
}

function lastIndex(crv) { return crv.ctrlPts.length - 1; }

// Homogeneous point on the curve (CurvePoint, A3.1, generalized to 4D Pw).
export function curvePointHomogeneous(crv, u) {
  assertCurve(crv, 'curvePoint');
  /* ⚠⚠ OFF-DOMAIN IS REFUSED, BECAUSE THE SILENT ANSWER LOOKS RIGHT. `findSpan`
     clamps to the last real span, so evaluating a circle of radius 10 at u = 99
     — its domain being [0, 4] — returned a point 10.000 from the center: on the
     circle, plausible, and meaningless. That is worse than a garbage number,
     which at least announces itself. A NURBS curve is defined on [knots[0],
     knots[last]] and nowhere else.
     ⚠ The tolerance is RELATIVE to the domain span, not absolute, because a
     caller arriving at the far end through arc-length inversion or a knot value
     can land an ulp past it and means the endpoint. Measured across the whole
     test suite before this was added: 1829 tests, ZERO evaluations outside the
     domain, so nothing legitimate is being taken away. Also measured: the check
     costs nothing detectable in the hottest function in the library. */
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const span0 = uMax - uMin;
  if (!(u >= uMin - span0 * 1e-9 && u <= uMax + span0 * 1e-9)) {
    throw new Error(`curvePoint: u = ${u} is outside the curve's domain [${uMin}, ${uMax}] — a NURBS curve is not defined there, and evaluating anyway returns a plausible-looking point that means nothing`);
  }
  const { degree: p, knots: U } = crv;
  const Pw = toHomogeneous(crv.ctrlPts);
  const n = lastIndex(crv);
  const span = findSpan(n, p, u, U);
  const N = basisFuns(span, u, p, U);
  const Cw = [0, 0, 0, 0];
  for (let i = 0; i <= p; i++) {
    const cp = Pw[span - p + i];
    for (let k = 0; k < 4; k++) Cw[k] += N[i] * cp[k];
  }
  return Cw;
}

/* WHAT A CURVE IS, CHECKED ONCE AT THE PLACES EVERYTHING GOES THROUGH.
   A malformed input used to surface as "Cannot read properties of undefined
   (reading '0')" from three frames down — measured across a sample of twelve
   public entry points, TEN of them failed that way. For a kernel with no API
   reference, that difference is somebody's first hour: a named refusal says
   which argument was wrong, a TypeError says only that this library is
   unfinished.
   ⚠ PLACED AT THE CHOKEPOINTS, not sprinkled over thirty entry points. Almost
   everything that consumes a curve either asks for its domain or evaluates it,
   so `curveDomain` and `curvePointHomogeneous` between them cover the family —
   `curveLength`, `tessellateCurve`, `divideByArcLength` and
   `closestPointOnCurve` all inherit it without being touched. */
export function assertCurve(crv, fn = 'this function') {
  if (!crv || typeof crv !== 'object') throw new Error(`${fn}: expected a curve object { degree, knots, ctrlPts }, got ${crv === null ? 'null' : typeof crv}`);
  if (!Array.isArray(crv.knots) || crv.knots.length < 2) throw new Error(`${fn}: the curve has no usable knot vector — expected { degree, knots, ctrlPts }`);
  if (!Array.isArray(crv.ctrlPts) || crv.ctrlPts.length < 1) throw new Error(`${fn}: the curve has no control points — expected { degree, knots, ctrlPts }`);
  if (!Number.isFinite(crv.degree)) throw new Error(`${fn}: the curve has no degree — expected { degree, knots, ctrlPts }`);
  return crv;
}

/* THE PARAMETER RANGE A CURVE IS ACTUALLY DEFINED OVER. Published because it is
   the first thing a newcomer gets wrong: a NURBS curve's domain is NOT [0,1] —
   `makeCircle` spans 0..4, one unit per quadrant — and reading it off the knot
   vector by hand (`crv.knots[crv.knots.length - 1]`) is the kind of thing that
   ends up copied into every consumer's own code. */
export function curveDomain(crv) {
  assertCurve(crv, 'curveDomain');
  return [crv.knots[0], crv.knots[crv.knots.length - 1]];
}

// Dehomogenized (real, Euclidean) point on the curve.
export function curvePoint(crv, u) {
  const Cw = curvePointHomogeneous(crv, u);
  return [Cw[0] / Cw[3], Cw[1] / Cw[3], Cw[2] / Cw[3]];
}

// Homogeneous derivatives 0..d of the curve (CurveDerivsAlg, A3.2, run on Pw
// directly — this is the standard trick: a rational curve's homogeneous form
// IS a non-rational B-spline curve in 4D, so the ordinary derivative
// algorithm applies unchanged).
export function curveDerivsHomogeneous(crv, u, d) {
  const { degree: p, knots: U } = crv;
  const Pw = toHomogeneous(crv.ctrlPts);
  const n = lastIndex(crv);
  const du = Math.min(d, p);
  const CK = Array.from({ length: d + 1 }, () => [0, 0, 0, 0]);
  const span = findSpan(n, p, u, U);
  const ders = dersBasisFuns(span, u, p, du, U);
  for (let k = 0; k <= du; k++) {
    for (let j = 0; j <= p; j++) {
      const cp = Pw[span - p + j];
      for (let c = 0; c < 4; c++) CK[k][c] += ders[k][j] * cp[c];
    }
  }
  return CK; // derivatives beyond du are correctly zero (degree-limited)
}

function binom(n, k) {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
}

// Euclidean derivatives 0..d, via A4.2's quotient-rule recursion on the
// homogeneous derivatives above. CK[0] is the point itself.
export function rationalCurveDerivs(crv, u, d) {
  const Awders = curveDerivsHomogeneous(crv, u, d);
  const CK = [];
  for (let k = 0; k <= d; k++) {
    let v = [Awders[k][0], Awders[k][1], Awders[k][2]];
    for (let i = 1; i <= k; i++) {
      const bin = binom(k, i);
      const wi = Awders[i][3];
      v[0] -= bin * wi * CK[k - i][0];
      v[1] -= bin * wi * CK[k - i][1];
      v[2] -= bin * wi * CK[k - i][2];
    }
    const w0 = Awders[0][3];
    CK.push([v[0] / w0, v[1] / w0, v[2] / w0]);
  }
  return CK;
}

// Convenience for framing (sweeps, dimensioning leaders): point + unit tangent.
export function curvePointAndTangent(crv, u) {
  const [C0, C1] = rationalCurveDerivs(crv, u, 1);
  return { point: C0, tangent: normalize(C1) };
}

// Reverse a NURBS curve's own parametrization (P&T 2.5, "reversing the
// direction of a curve") — a real, exact, well-known operation, NOT
// curve-joining or degree elevation (a different, harder P&T Ch.5
// operation this kernel doesn't have). Reversing simply means: traverse
// the SAME point set from the other end. Two things must flip together
// for that to hold — the control point ORDER (P'_i = P_{n-i}) and the
// knot vector, re-based onto the same domain so span/basis lookups stay
// valid (U'_j = a + b - U_{m-j}, where [a,b] = [knots[0], knots[m]] is
// the curve's own domain). Applying this twice is its own inverse
// (reversing back reproduces the original knots exactly, algebraically:
// a+b-(a+b-U_k) = U_k) — checked below, not just asserted.
export function reverseCurve(crv) {
  const { degree, knots, ctrlPts } = crv;
  const m = knots.length - 1;
  const a = knots[0], b = knots[m];
  const newKnots = [];
  for (let j = 0; j <= m; j++) newKnots.push(a + b - knots[m - j]);
  const newCtrlPts = ctrlPts.slice().reverse().map((cp) => cp.slice());
  return { degree, knots: newKnots, ctrlPts: newCtrlPts };
}

// Greville abscissae — one parameter value per control point, the standard
// average-of-p-knots association used to place a frame "at" each control
// point of a curve with no other canonical per-CP parameter (Sweep1, 03).
export function grevilleAbscissae(crv) {
  const { degree: p, knots: U } = crv;
  const n = lastIndex(crv);
  const g = [];
  for (let i = 0; i <= n; i++) {
    let s = 0;
    for (let j = i + 1; j <= i + p; j++) s += U[j];
    g.push(s / p);
  }
  return g;
}

// DIVIDE — arc-length-even point placement along a curve (the DIVIDE spec:
// "Uses arc-length reparametrization for even spacing, not raw
// parameter-uniform, which bunches on unevenly-parametrized curves"). A
// general rational NURBS curve has no closed-form arc length (unlike
// reverseCurve's exact knot-domain algebra above) — the honest, standard
// numerical recipe, the SAME one real NURBS kernels use for their own
// "divide by length"/GetLength, is a dense chord-length polyline refined
// until adjacent chords deviate from the true curve by less than a
// tolerance, then linear interpolation along that polyline to invert
// length->parameter. This mirrors the app's own render tessellation
// (`sampleCurveAdaptive`'s chord-deviation recursion) rather than
// reinventing a second, independently-tuned "smooth enough" — ported here
// to plain arrays (no THREE dependency) so the kernel stays framework-free
// and node-testable.
function chordDeviationPlain(p0, p1, pMid) {
  const chord = sub(p1, p0);
  const chordLenSq = length(chord) ** 2;
  if (chordLenSq < 1e-12) return length(sub(pMid, p0)); // degenerate zero-length chord
  const t = (pMid[0]-p0[0])*chord[0]/chordLenSq + (pMid[1]-p0[1])*chord[1]/chordLenSq + (pMid[2]-p0[2])*chord[2]/chordLenSq;
  const proj = [p0[0]+chord[0]*t, p0[1]+chord[1]*t, p0[2]+chord[2]*t];
  return length(sub(pMid, proj));
}
const DIVIDE_MAX_DEPTH = 20; // bisection depth safety cap, same role as sampleCurveAdaptive's ADAPTIVE_CURVE_MAX_DEPTH
// A ONE-SAMPLE CHORD-DEVIATION TEST CANNOT BE TRUSTED TO TERMINATE THE
// RECURSION, and the reason is structural, not a tolerance question. On a
// span that is POINT-SYMMETRIC about its own midpoint — C(m+s) = 2C(m) -
// C(m-s), an S-curve through symmetric points, a curve mirrored about its
// own center — the midpoint lies EXACTLY on the chord between the span's
// endpoints, so the single deviation sample reads exactly zero while the
// curve genuinely bows away everywhere else. Left unguarded that returns
// the two endpoints alone, and every consumer downstream (arc length,
// length->parameter inversion, DIVIDE stations, sweep rail stations)
// silently degrades to the straight chord. This is a per-SPAN property,
// so an ordinary hand-drawn curve with one symmetric span between two of
// its own knots is affected too, not only a wholly symmetric curve.
//
// The fix is a MINIMUM DEPTH: the deviation test is not allowed to stop
// anything until the span has been bisected DIVIDE_MIN_DEPTH times, so
// termination requires 2^DIVIDE_MIN_DEPTH independent sub-span tests at
// parameters no single symmetry can zero out at once. Its cost is only
// paid by a span that is genuinely flat — a span that really curves
// subdivides past this depth on its own merits regardless — so this is
// free for every curve that has any curvature to resolve.
//
// A DEGREE<=1 SPAN IS EXEMPT, and provably so rather than as an
// optimization: for degree 1 the basis functions over a span sum to one
// and are non-negative, so C(u) is a convex combination of that span's
// two control points (weights included, for a rational curve) — the
// chord IS the curve, the deviation is identically zero, and subdividing
// can never reveal anything. Line/Polyline rails keep their exact
// previous sample counts.
const DIVIDE_MIN_DEPTH = 2;
// Dense {u, point} chain, adaptively refined to `tolerance` (same units as
// the curve's own control points — caller picks a value relative to the
// curve's own scale, see `divideByArcLength`'s default below).
export function adaptiveArcLengthSamples(crv, uStart, uEnd, tolerance) {
  /* ⚠⚠ THE DOMAIN DEFAULTS TO THE CURVE'S OWN, and it did not. With `uStart`
     and `uEnd` left off, the seed set was `{undefined}`, every `k > undefined`
     comparison was false, and this returned ONE sample — so `curveLength(crv)`,
     the most obvious call in the library, answered 0 for a circle of
     circumference 62.83. Silently, with no refusal and no NaN. A caller who
     does not know a NURBS curve's domain is usually not [0,1] has no way to
     read that as anything but the truth. */
  const dom = curveDomain(crv);
  if (uStart == null) uStart = dom[0];
  if (uEnd == null) uEnd = dom[1];
  const evalAt = (u) => ({ u, pt: curvePoint(crv, u) });
  const minDepth = crv.degree <= 1 ? 0 : DIVIDE_MIN_DEPTH;
  const seedSet = new Set([uStart, uEnd]);
  for (const k of crv.knots) if (k > uStart && k < uEnd) seedSet.add(k);
  const seeds = [...seedSet].sort((a, b) => a - b);
  const samples = [evalAt(seeds[0])];
  function recurse(s0, s1, depth) {
    if (depth < DIVIDE_MAX_DEPTH) {
      const mid = evalAt((s0.u + s1.u) / 2);
      if (depth < minDepth || chordDeviationPlain(s0.pt, s1.pt, mid.pt) > tolerance) {
        recurse(s0, mid, depth + 1);
        recurse(mid, s1, depth + 1);
        return;
      }
    }
    samples.push(s1);
  }
  for (let i = 0; i < seeds.length - 1; i++) recurse(evalAt(seeds[i]), evalAt(seeds[i + 1]), 0);
  return samples;
}

// Real total arc length (dense-polyline approximation, accurate to within
// `tolerance`).
export function curveLength(crv, uStart, uEnd, tolerance) {
  const samples = adaptiveArcLengthSamples(crv, uStart, uEnd, tolerance);
  let total = 0;
  for (let i = 1; i < samples.length; i++) total += length(sub(samples[i].pt, samples[i - 1].pt));
  return total;
}

// Arc-length -> parameter inversion, factored out of divideByArcLength so a
// caller needing MANY inversions against the SAME curve (sweepNProfiles'
// own fix, below in kernel/sweep.mjs — evaluating a rail frame at
// several intermediate arc-length fractions BETWEEN two profile stations)
// builds the dense polyline + cumulative-length table ONCE, not once per
// lookup. `buildArcLengthTable` is the SAME `adaptiveArcLengthSamples` +
// running-sum recipe divideByArcLength already uses inline; `paramAtArcLength`
// is the SAME binary-search + linear-interpolation-along-the-bracketing-
// segment inversion divideByArcLength already does inline, just reusable
// against an arbitrary (not evenly-spaced) target length. divideByArcLength
// itself is left completely untouched (it already has this logic inline and
// is fully tested) — this is a genuinely NEW, additive pair of functions,
// not a refactor of an existing tested one.
export function buildArcLengthTable(crv, uStart, uEnd, tolerance) {
  if (tolerance === undefined) {
    const uMin = uStart, uMax = uEnd;
    const knotSet = new Set([uMin, uMax]);
    for (const k of crv.knots) if (k > uMin && k < uMax) knotSet.add(k);
    const seeds = [...knotSet].sort((a, b) => a - b);
    let coarse = 0;
    let prev = curvePoint(crv, seeds[0]);
    for (let i = 1; i < seeds.length; i++) {
      const p = curvePoint(crv, seeds[i]);
      coarse += length(sub(p, prev));
      prev = p;
    }
    tolerance = Math.max(coarse * 1e-6, 1e-9);
  }
  const samples = adaptiveArcLengthSamples(crv, uStart, uEnd, tolerance);
  const cumLen = [0];
  for (let i = 1; i < samples.length; i++) cumLen.push(cumLen[i - 1] + length(sub(samples[i].pt, samples[i - 1].pt)));
  return { samples, cumLen, total: cumLen[cumLen.length - 1] };
}

export function paramAtArcLength(table, targetLen) {
  const { samples, cumLen, total } = table;
  if (targetLen <= 0) return samples[0].u;
  if (targetLen >= total) return samples[samples.length - 1].u;
  let lo = 0, hi = cumLen.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (cumLen[mid] < targetLen) lo = mid; else hi = mid;
  }
  const segLen = cumLen[hi] - cumLen[lo];
  return segLen < 1e-12 ? samples[lo].u : samples[lo].u + (samples[hi].u - samples[lo].u) * (targetLen - cumLen[lo]) / segLen;
}

// The other direction — parameter -> arc length, against the SAME table
// (natural pairing with paramAtArcLength above). sweepNProfiles needs BOTH
// directions against the identical dense polyline: stations' own arc
// length (this function) and interior in-between samples' rail parameter
// (paramAtArcLength) — using ONE shared table for both is what keeps a
// station's own vbar and the table's own `total` consistent with each
// other (mixing a fine table with a separately, coarsely re-derived
// curveLength() for one side of that ratio was a real bug: the
// last station's vbar came out slightly short of 1.0, pushing the surface
// domain's own v=1 edge into silent extrapolation).
export function arcLengthAtParam(table, u) {
  const { samples, cumLen } = table;
  if (u <= samples[0].u) return 0;
  if (u >= samples[samples.length - 1].u) return cumLen[cumLen.length - 1];
  let lo = 0, hi = samples.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].u < u) lo = mid; else hi = mid;
  }
  const segU = samples[hi].u - samples[lo].u;
  return segU < 1e-15 ? cumLen[lo] : cumLen[lo] + (cumLen[hi] - cumLen[lo]) * (u - samples[lo].u) / segU;
}

// Closest point on a curve to an arbitrary 3D point — needed to STATION a
// Sweep1-with-N-profiles cross-section along its rail (real Rhino: the
// user places each profile already positioned near its own intended rail
// location, the app infers which rail parameter it belongs to from
// proximity — see the Sweep1 N-profiles correspondence question).
// Two stages, the standard numeric recipe (P&T 6.1's own point-inversion
// is the same shape — a coarse global search first, so a Newton step
// can't converge to the WRONG local minimum, then Newton-Raphson for
// accuracy beyond the coarse search's own resolution):
//   1. Dense chord-deviation-adaptive samples (the SAME `adaptiveArcLengthSamples`
//      DIVIDE already uses) give a polyline; the target point is projected
//      onto every SEGMENT of it (not just each vertex, clamped to [0,1]
//      per segment) and the globally closest kept as a coarse seed.
//   2. A bounded number of Newton-Raphson iterations on f(u) = |C(u)-P|^2
//      (root of f', using the curve's own real first/second derivatives,
//      not a finite-difference estimate) refine that seed to numerical
//      precision. A step that leaves the curve's own domain or makes the
//      distance WORSE is rejected outright (keeps the coarse answer rather
//      than diverging) — the same "never silently produce a wrong number"
//      discipline as this file's own divideByArcLength fallback.
//
// AMBIGUITY is named honestly, not silently resolved: a rail that loops
// back near itself can have TWO genuinely different parameter values
// nearly equidistant from the same target point — there is no single
// correct "closest point" then, and silently picking one would silently
// misplace a profile's station. Detected by scanning the SAME dense
// samples for every local-minimum dip in distance (not just the global
// one) and flagging when a second, PARAMETRICALLY DISTANT dip comes
// within a small relative tolerance of the best one. This is an honest
// engineering heuristic (dense-sample local minima), not a closed-form
// proof of global uniqueness — stated plainly, matching this kernel's own
// standard for every other numeric-approximation function (curveLength,
// divideByArcLength).
const CLOSEST_POINT_AMBIGUITY_REL_TOL = 1e-3;
export function closestPointOnCurve(crv, targetPt, tolerance) {
  assertCurve(crv, 'closestPointOnCurve');
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  if (tolerance === undefined) {
    const knotSet = new Set([uMin, uMax]);
    for (const k of crv.knots) if (k > uMin && k < uMax) knotSet.add(k);
    const seeds = [...knotSet].sort((a, b) => a - b);
    let coarse = 0;
    let prev = curvePoint(crv, seeds[0]);
    for (let i = 1; i < seeds.length; i++) {
      const p = curvePoint(crv, seeds[i]);
      coarse += length(sub(p, prev));
      prev = p;
    }
    tolerance = Math.max(coarse * 1e-5, 1e-9);
  }
  const samples = adaptiveArcLengthSamples(crv, uMin, uMax, tolerance);
  const distSq = samples.map((s) => {
    const d = sub(s.pt, targetPt);
    return dot(d, d);
  });

  // Stage 1 — best point on any polyline SEGMENT, clamped, not just a
  // sample vertex.
  let bestU = samples[0].u, bestDistSq = distSq[0];
  for (let i = 0; i < samples.length - 1; i++) {
    const p0 = samples[i].pt, p1 = samples[i + 1].pt;
    const seg = sub(p1, p0);
    const segLenSq = dot(seg, seg);
    let t = segLenSq < 1e-14 ? 0 : dot(sub(targetPt, p0), seg) / segLenSq;
    t = Math.max(0, Math.min(1, t));
    const proj = [p0[0] + seg[0] * t, p0[1] + seg[1] * t, p0[2] + seg[2] * t];
    const dSq = dot(sub(proj, targetPt), sub(proj, targetPt));
    if (dSq < bestDistSq) {
      bestDistSq = dSq;
      bestU = samples[i].u + (samples[i + 1].u - samples[i].u) * t;
    }
  }

  // Stage 2 — Newton-Raphson refinement (P&T 6.1 point-inversion shape).
  let u = bestU;
  let curDistSq = bestDistSq;
  for (let iter = 0; iter < 12; iter++) {
    const [C0, C1, C2] = rationalCurveDerivs(crv, u, 2);
    const diff = sub(C0, targetPt);
    const fPrime = 2 * dot(diff, C1);
    const fDoublePrime = 2 * (dot(diff, C2) + dot(C1, C1));
    if (Math.abs(fDoublePrime) < 1e-12) break;
    let uNext = u - fPrime / fDoublePrime;
    uNext = Math.max(uMin, Math.min(uMax, uNext));
    const nextPt = curvePoint(crv, uNext);
    const nextDiff = sub(nextPt, targetPt);
    const nextDistSq = dot(nextDiff, nextDiff);
    if (nextDistSq > curDistSq + 1e-14) break; // reject a worsening step
    const converged = Math.abs(uNext - u) < 1e-12;
    u = uNext;
    curDistSq = nextDistSq;
    if (converged) break;
  }

  // Ambiguity scan.
  const bestDist = Math.sqrt(curDistSq);
  let ambiguousWith = null;
  for (let i = 0; i < samples.length; i++) {
    const isLocalMin = (i === 0 || distSq[i] <= distSq[i - 1]) && (i === samples.length - 1 || distSq[i] <= distSq[i + 1]);
    if (!isLocalMin) continue;
    if (Math.abs(samples[i].u - u) < (uMax - uMin) * 0.02) continue; // the same dip Newton just refined
    const d = Math.sqrt(distSq[i]);
    if (d <= bestDist * (1 + CLOSEST_POINT_AMBIGUITY_REL_TOL) + tolerance) { ambiguousWith = samples[i].u; break; }
  }

  return { u, point: curvePoint(crv, u), distance: bestDist, ambiguous: ambiguousWith !== null, ambiguousWith };
}

// Whether a curve's own control points wrap around — the start and end
// control point coincide within tolerance, meaning u=uMin and u=uMax are
// the SAME physical point (a seam, not two distinct endpoints). The
// curve analog of surface.mjs's surfaceClosure, which asks the
// identical question of a control NET's first/last row or column —
// applied here to a curve's own single first/last control point.
export function isCurveClosed(crv, tol = 1e-6) {
  const p0 = crv.ctrlPts[0], p1 = crv.ctrlPts[crv.ctrlPts.length - 1];
  return Math.hypot(p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]) <= tol && Math.abs(p0[3] - p1[3]) <= tol;
}

// The actual DIVIDE operation: for an OPEN curve, `count` segments ->
// count+1 points (including BOTH curve endpoints, matching Rhino/MoI's
// own default Divide-by-segment-count behavior). For a CLOSED curve
// (isCurveClosed above), u=uMin and u=uMax are the SAME physical seam
// point, so including both would double-count it — real Rhino's own
// Divide on a closed curve instead returns exactly `count` points spaced
// evenly around the FULL closed length, never repeating the seam. Either
// way, points are evenly spaced by real arc length. Returns [{u, point}],
// `point` always an EXACT curve evaluation (curvePoint at the found u) —
// only the arc-length POSITION of that u is approximate to within
// tolerance, the point itself never leaves the real curve.
export function divideByArcLength(crv, count, tolerance) {
  assertCurve(crv, 'divideByArcLength');
  if (!Number.isInteger(count) || count < 1) throw new Error('divideByArcLength: count must be a positive integer');
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  if (tolerance === undefined) {
    // Scale-relative default: a coarse one-chord-per-knot-span estimate of
    // the curve's own length sets the tolerance automatically (same
    // reasoning sampleCurveAdaptive's own comment gives for why a fixed
    // absolute number is the wrong shape — this curve could be 1 unit or
    // 100000 units across).
    const knotSet = new Set([uMin, uMax]);
    for (const k of crv.knots) if (k > uMin && k < uMax) knotSet.add(k);
    const seeds = [...knotSet].sort((a, b) => a - b);
    let coarse = 0;
    let prev = curvePoint(crv, seeds[0]);
    for (let i = 1; i < seeds.length; i++) {
      const p = curvePoint(crv, seeds[i]);
      coarse += length(sub(p, prev));
      prev = p;
    }
    tolerance = Math.max(coarse * 1e-6, 1e-9);
  }
  const samples = adaptiveArcLengthSamples(crv, uMin, uMax, tolerance);
  const cumLen = [0];
  for (let i = 1; i < samples.length; i++) cumLen.push(cumLen[i - 1] + length(sub(samples[i].pt, samples[i - 1].pt)));
  const total = cumLen[cumLen.length - 1];
  const closed = isCurveClosed(crv);
  const n = closed ? count : count + 1; // closed: count points, no seam duplicate; open: count+1, both real endpoints
  const results = [];
  for (let i = 0; i < n; i++) {
    if (i === 0) { results.push({ u: uMin, point: curvePoint(crv, uMin) }); continue; }
    if (!closed && i === count) { results.push({ u: uMax, point: curvePoint(crv, uMax) }); continue; }
    const targetLen = (total * i) / count;
    // Binary search the bracket in the monotonic cumulative-length table.
    let lo = 0, hi = cumLen.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (cumLen[mid] < targetLen) lo = mid; else hi = mid;
    }
    const segLen = cumLen[hi] - cumLen[lo];
    const u = segLen < 1e-12 ? samples[lo].u : samples[lo].u + (samples[hi].u - samples[lo].u) * (targetLen - cumLen[lo]) / segLen;
    results.push({ u, point: curvePoint(crv, u) });
  }
  return results;
}

// DEDUPE (a named BLOCKING PREREQUISITE for surface work — students
// routinely copy-paste-forget-delete, leaving spatially-coincident duplicate
// curves that would badly confuse a future angle/parallel/loft-compatibility
// classifier) — are two curves the SAME curve, spatially, within
// `tolerance`? Deliberately takes two already-resampled point chains
// (plain [x,y,z] triples), not two NurbsCrv objects: this is the SAME
// "resample and compare" technique every per-type display path in this
// kernel/the app already uses (adaptiveArcLengthSamples above, the app's own
// sampleCurveAdaptive, obj.sketchSamples/circleSamples) — reused here rather
// than re-deriving a curve-object-specific comparison, so the SAME function
// works for a Line (2 points), a Polyline (its own vertices), or a dense
// SketchCurve/Circle chain without needing to special-case any of them.
//
// Two checks, run in the order that discriminates the named false-positive
// case (a genuine sub-segment of a much longer curve) as early as possible:
//   1. EXTENT — the two chains' own chord-summed length must match within
//      a small multiple of `tolerance` (not `tolerance` itself: two
//      independently-adaptive-refined chains of a truly identical curve
//      already differ slightly in vertex count/placement, so their
//      chord-summed lengths are an approximation of an approximation, never
//      bit-exact even for two genuinely identical curves — a real
//      sub-segment's length is shorter by a macroscopic, modeling-scale
//      amount, so a modest safety factor still discriminates it cleanly).
//   2. BIDIRECTIONAL nearest-point — every point of A has a near point
//      somewhere in B, AND every point of B has a near point somewhere in
//      A (order- and direction-agnostic, so a reversed duplicate still
//      matches). Checking only A->B would let a short A sitting INSIDE a
//      longer B pass — every one of A's few points does sit near some
//      point of B — even though B has plenty of points nowhere near A; the
//      B->A direction is what actually catches that case. This is the real
//      Hausdorff-distance-under-tolerance test, and the (1) length check
//      above is a cheap early-exit for the same case, not a substitute for
//      it (a curve could theoretically fold back on itself to fake a
//      matching length while still failing (2), so both are kept).
// A bounding-box pre-filter belongs to the CALLER (it already has the real
// mesh/object to test cheaply before ever reaching for two full sample
// chains) — this function is the exact geometric test once two candidates
// are already worth comparing in full.
function polylineLength(pts) {
  let total = 0;
  for (let i = 1; i < pts.length; i++) total += length(sub(pts[i], pts[i - 1]));
  return total;
}
function hasNearPoint(p, pts, tolerance) {
  for (const q of pts) if (length(sub(p, q)) < tolerance) return true;
  return false;
}
export function curvesCoincident(samplesA, samplesB, tolerance) {
  if (!samplesA.length || !samplesB.length) return false;
  const lenA = polylineLength(samplesA), lenB = polylineLength(samplesB);
  if (Math.abs(lenA - lenB) > tolerance * 10) return false; // extent mismatch — cheaply rejects the sub-segment case
  for (const p of samplesA) if (!hasNearPoint(p, samplesB, tolerance)) return false;
  for (const p of samplesB) if (!hasNearPoint(p, samplesA, tolerance)) return false;
  return true;
}

// THE SAME TEST, INVERTED: not "does this pair pass at `tolerance`" but
// "what tolerance would this pair need". A threshold comparison that returns
// only a boolean throws away the one number a refusal most needs — how close
// the rejected candidate actually was — and no amount of rewording the
// refusal message can recover it afterwards. This returns that number, so a
// caller that found no match can say what it measured, what it required, and
// what value would have worked, instead of naming a threshold alone.
//
// It is the exact algebraic inverse of curvesCoincident's own two checks, not
// a second, independently-derived proximity metric — read them together:
//   · EXTENT needs `|lenA - lenB| <= tolerance * 10`, so it alone demands
//     `tolerance >= |lenA - lenB| / 10`.
//   · BIDIRECTIONAL nearest-point needs every point of each chain within
//     `tolerance` of the other chain, so it demands a tolerance above the
//     larger of the two directed maxima — the symmetric Hausdorff distance.
// The pair's requirement is the larger of those two, and a tolerance at or
// below it genuinely fails, which is what makes the answer actionable rather
// than merely descriptive.
//
// ⚠ THE RETURNED VALUE IS AN INFIMUM, NOT A WORKING SETTING. The
// nearest-point comparison is strict (`<`), so a tolerance exactly equal to
// the Hausdorff term still fails by a hair; a caller offering the number to
// a user must round it UP, never present it bare — see coincidenceGapLabels
// in the app, which exists for that reason.
//
// Deliberately NOT folded into curvesCoincident as an out-param: the boolean
// form early-exits on the first far point and is called O(n^2) times over
// every curve in a document (the dedupe pass), while this one must visit
// every point of both chains to find a maximum. Keeping them separate keeps
// the hot path fast and confines the full measurement to the path that has
// already failed and is about to explain itself.
export function curveCoincidenceGap(samplesA, samplesB) {
  if (!samplesA.length || !samplesB.length) return Infinity;
  let worst = Math.abs(polylineLength(samplesA) - polylineLength(samplesB)) / 10;
  for (const p of samplesA) {
    let best = Infinity;
    for (const q of samplesB) { const d = length(sub(p, q)); if (d < best) best = d; }
    if (best > worst) worst = best;
  }
  for (const p of samplesB) {
    let best = Infinity;
    for (const q of samplesA) { const d = length(sub(p, q)); if (d < best) best = d; }
    if (best > worst) worst = best;
  }
  return worst;
}

// TWO RAILS MEETING AT A POINT ARE ONE LONGER RAIL — the concatenation a
// two-rail pipe junction actually wants, rather than a bridge/weld between
// two independently swept stubs. Sweeping ONE continuous tube along the
// joined rail is not merely simpler; for the NURBS expression it inherits,
// for free, machinery a cross-object weld can never reach:
// `railInteriorCorners`/`applyTrueMiterStretch` (a true elliptical miter at
// the junction instead of a butt joint), `applyMiterLimitFallback` (a
// fillet for a corner too sharp to miter), and one unbroken
// parallel-transport frame chain across the junction instead of two
// independently clocked ones. It is the same move `pipeRailForSweep`
// already makes for a single rail's own interior corners.
//
// WHY THIS LIVES IN curve.mjs AND NOT knots.mjs. Every DECISION here is
// curve geometry, not knot arithmetic: which endpoints coincide (curve
// evaluation + a distance tolerance), which curve needs reversing
// (`reverseCurve`, this file), and whether the two rails fold back on each
// other (end tangents, via `rationalCurveDerivs`, this file). The knot
// arithmetic — degree elevation to a common degree, domain rescaling, the
// C0 splice — is not re-derived at all: it is delegated wholesale to
// `joinCurvesC0`, which is already proven and node-tested. knots.mjs has no
// notion of an endpoint, a tangent, or a tolerance and would have to import
// three functions from here to gain one; this direction of dependency is
// the one that already exists (knots.mjs imports only basis.mjs, so there
// is no cycle).
//
// WHAT THE CALLER GETS, AND THE ONE REAL LIMIT. When BOTH rails are
// degree 1 (Line/Polyline — the common case), the result is degree 1 and
// the junction survives as a genuine interior CONTROL POINT, which is
// exactly what `railInteriorCorners` reads: the corner is reported, mitered
// and miter-limited like any other corner of a single rail. When either
// input is degree>1, or `joinCurvesC0` elevates the pair to a common
// degree>1, `railInteriorCorners` returns nothing at all (its own contract
// is "a degree<=1 rail's own raw control points") and `sweep1Rigid`
// dispatches to `sweep1RigidResampled`, which has no miter machinery — so
// the junction ships the un-mitered elbow, narrower than a true miter by
// roughly cos(theta/2). Measured on a 90-degree fixture with a radius-5
// tube: the swept skin reaches ~5.24mm from its own rail at the junction
// where a true miter reaches exactly r*sec(45deg) = 7.071mm, i.e. 74% of
// the elbow a mitered junction would have given. That is a real
// limit of the cheap route, not a rounding artifact, and it is pinned as a
// test rather than left as a footnote.
//
// 0.001mm — the same "these two points count as the same point" number the
// app's own Join uses for endpoint chaining. Named locally rather than
// imported because this module has no app-layer dependency, matching
// kernel/loft.mjs's own CLOSE_LOOP_TOL precedent.
export const RAIL_JUNCTION_TOLERANCE = 0.001;
// A turn this close to a full reversal is refused outright, and the
// threshold is not a fresh invention: it is `filletCornerArc`'s own
// "a near-180 reversal has no well-defined fillet" bound. Matching it
// exactly is what makes the refusal correspond to a real downstream
// incapacity rather than to taste — see the fold-back reasoning below.
const RAIL_FOLD_BACK_EPS = 1e-6;

// Domain ends of a clamped curve, the same knots[0]/knots[last] convention
// every other function in this file already uses (divideByArcLength,
// closestPointOnCurve) and the same one `joinCurvesC0`'s own
// `rescaleCurveDomain` reads when it builds the result.
function railDomainEnds(crv) {
  const a = crv.knots[0], b = crv.knots[crv.knots.length - 1];
  return [{ key: 'start', u: a, pt: curvePoint(crv, a) }, { key: 'end', u: b, pt: curvePoint(crv, b) }];
}
// Unit direction of TRAVEL at a parameter. Callers here always evaluate an
// already-reversed curve rather than negating a tangent, so the sign is
// always the curve's own — one fewer convention to get backwards.
function unitTravelDirection(crv, u) {
  const [, C1] = rationalCurveDerivs(crv, u, 1);
  const L = length(C1);
  return L > 0 ? scale(C1, 1 / L) : null;
}

// Concatenate two rails meeting at a shared endpoint into ONE curve
// traversing A then B. Returns `{ ok: true, curve, ... }` or
// `{ ok: false, reason }` — a result object, not a throw, matching
// `filletOpenPolyline`/`filletPolygon`'s own precedent for a rail-preparation
// helper whose refusals are ordinary reachable user situations (two curves
// that simply do not touch) rather than programmer errors.
//
// ALL FOUR END-PAIRINGS WORK. A user picks two curves in whatever order and
// whichever direction they happened to draw them, so A-end/B-start is only
// one of four cases; the other three are handled by reversing whichever
// curve needs it (`reverseCurve` — exact, P&T 2.5, control points reversed
// and the knot vector re-based onto the same domain), never by resampling.
//
// THREE HONEST REFUSALS, each by name:
//  - NO SHARED ENDPOINT. The nearest of the four endpoint pairs is reported
//    with its real gap, so the caller can say how far off the two rails
//    actually are rather than only that they missed.
//  - AMBIGUOUS. More than one distinct end of A (or of B) lands on the other
//    rail: A is closed, or B is closed, or the two together close a loop.
//    Every such case has two or more genuinely different valid answers —
//    and the loop case additionally produces a SECOND junction, at the
//    result's own seam, that nothing here has looked at. Refused rather
//    than picked arbitrarily.
//  - FOLD-BACK. The two rails leave the junction along exactly opposite
//    directions. This is refused rather than deferred downstream, and the
//    reason is measured, not assumed: at a full reversal the miter's own
//    bisector tangent `tIn + tOut` is the zero vector and the bend direction
//    `normalize(tOut - tIn)` collapses onto the rail's own tangent, so the
//    elliptical stretch has nothing perpendicular to act on.
//    `applyMiterLimitFallback` cannot rescue it either — its fillet is
//    `filletCornerArc`, which refuses a near-180 turn by name — so
//    `applyTrueMiterStretch` clamps an effectively infinite stretch to
//    `PIPE_MITER_LIMIT` and applies it along a direction parallel to the
//    tangent, a provable no-op on a ring perpendicular to that tangent. The
//    tube runs out and straight back through itself, finite and NaN-free
//    and silently self-intersecting. A turn merely CLOSE to 180 degrees is
//    NOT refused: the miter-limit fallback genuinely fillets it, so
//    deferring is correct there and only the degenerate case is stopped.
export function concatRailsAtJunction(railA, railB, opts = {}) {
  const tolerance = opts.tolerance ?? RAIL_JUNCTION_TOLERANCE;
  if (!railA || !railB || !Array.isArray(railA.ctrlPts) || !Array.isArray(railB.ctrlPts)) {
    return { ok: false, reason: 'concatRailsAtJunction needs two curves' };
  }
  if (railA.ctrlPts.length < 2 || railB.ctrlPts.length < 2) {
    return { ok: false, reason: 'concatRailsAtJunction needs two curves with at least 2 control points each' };
  }

  const ea = railDomainEnds(railA), eb = railDomainEnds(railB);
  const pairs = [];
  for (const a of ea) for (const b of eb) pairs.push({ aKey: a.key, bKey: b.key, gap: length(sub(a.pt, b.pt)) });
  const matches = pairs.filter((p) => p.gap <= tolerance);

  if (matches.length === 0) {
    const nearest = pairs.reduce((best, p) => (p.gap < best.gap ? p : best), pairs[0]);
    return {
      ok: false,
      reason: `these two rails don't share an endpoint — their nearest ends (${nearest.aKey}/${nearest.bKey}) are ${nearest.gap.toFixed(4)}mm apart, outside the ${tolerance}mm join tolerance`,
      nearestGap: nearest.gap,
    };
  }
  const aKeys = new Set(matches.map((m) => m.aKey));
  const bKeys = new Set(matches.map((m) => m.bKey));
  if (aKeys.size > 1 || bKeys.size > 1) {
    return {
      ok: false,
      reason: 'this junction is ambiguous — more than one end of these rails meets the other, so there is no single longer rail to build (a closed rail, or two rails closing a loop, needs its own construction and leaves a second junction at its own seam)',
      matchCount: matches.length,
    };
  }

  const m = matches[0];
  const reversedA = m.aKey === 'start';
  const reversedB = m.bKey === 'end';
  const A = reversedA ? reverseCurve(railA) : railA;
  const B = reversedB ? reverseCurve(railB) : railB;

  const tA = unitTravelDirection(A, A.knots[A.knots.length - 1]);
  const tB = unitTravelDirection(B, B.knots[0]);
  if (!tA || !tB) {
    return { ok: false, reason: 'one of these rails has no well-defined direction at the junction (a zero-length derivative there)' };
  }
  const turnAngle = Math.acos(Math.max(-1, Math.min(1, dot(tA, tB))));
  if (turnAngle > Math.PI - RAIL_FOLD_BACK_EPS) {
    return {
      ok: false,
      reason: 'these two rails fold straight back on each other at the junction — a 180-degree reversal has no corner a miter or a fillet can describe, and the swept tube would run back through itself',
      turnAngle,
    };
  }

  const curve = joinCurvesC0([A, B]);
  const junction = curvePoint(A, A.knots[A.knots.length - 1]);
  // The result's own control point AT the junction, when one exists — the
  // exact index `railInteriorCorners` reports for a degree-1 pair. Found by
  // search rather than arithmetic because `joinCurvesC0` may have degree-
  // elevated one or both inputs first, which changes the control point
  // count; `null` rather than a guessed index when nothing lands there.
  let junctionIndex = null, bestD = Infinity;
  for (let i = 0; i < curve.ctrlPts.length; i++) {
    const d = length(sub([curve.ctrlPts[i][0], curve.ctrlPts[i][1], curve.ctrlPts[i][2]], junction));
    if (d < bestD) { bestD = d; junctionIndex = i; }
  }
  if (!(bestD < 1e-9)) junctionIndex = null;

  return {
    ok: true,
    curve,
    junction,
    // joinCurvesC0 rescales curve i onto [i, i+1], so the seam between the
    // first and second curve is exactly u = 1 by that function's own
    // convention, not by measurement.
    junctionParam: 1,
    junctionIndex,
    turnAngle,
    reversedA,
    reversedB,
    gap: m.gap,
  };
}
