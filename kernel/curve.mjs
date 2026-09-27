// NURBS curve evaluation — Piegl & Tiller Ch. 3 (CurveDerivsAlg, A3.2) and
// Ch. 4 (rational derivatives via the quotient rule, A4.2).
//
// A NurbsCrv in this kernel is { degree, knots, ctrlPts } where ctrlPts is an
// array of [x, y, z, w] — the point plus its weight, not pre-multiplied.
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
  /* Off-domain evaluation is refused. `findSpan` clamps to the last span, so
     evaluating outside [knots[0], knots[last]] would return a point on the
     curve that belongs to no parameter value — plausible, and meaningless.
     A NURBS curve is defined on that interval and nowhere else.
     The tolerance is relative to the domain span, because a caller reaching the
     far end through arc-length inversion or a knot value can land an ulp past
     it and means the endpoint. */
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

/* Structural check of a curve, placed where everything passes through.
   A malformed input otherwise surfaces as a TypeError several frames down; a
   named refusal says which argument was wrong. Almost everything that consumes
   a curve either asks for its domain or evaluates it, so `curveDomain` and
   `curvePointHomogeneous` between them cover the family — `curveLength`,
   `tessellateCurve`, `divideByArcLength` and `closestPointOnCurve` all inherit
   the check. */
export function assertCurve(crv, fn = 'this function') {
  if (!crv || typeof crv !== 'object') throw new Error(`${fn}: expected a curve object { degree, knots, ctrlPts }, got ${crv === null ? 'null' : typeof crv}`);
  if (!Array.isArray(crv.knots) || crv.knots.length < 2) throw new Error(`${fn}: the curve has no usable knot vector — expected { degree, knots, ctrlPts }`);
  if (!Array.isArray(crv.ctrlPts) || crv.ctrlPts.length < 1) throw new Error(`${fn}: the curve has no control points — expected { degree, knots, ctrlPts }`);
  if (!Number.isFinite(crv.degree)) throw new Error(`${fn}: the curve has no degree — expected { degree, knots, ctrlPts }`);
  return crv;
}

/* The parameter range a curve is defined over. A NURBS curve's domain is not
   in general [0,1] — `makeCircle` spans 0..4, one unit per quadrant. */
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

// Reverse a NURBS curve's parametrization (P&T §6.5, "Curve and Surface
// Reversal"): traverse the same point set from the other end. The control
// point order flips (P'_i = P_{n-i}) and the knot vector is re-based onto the
// same domain so span/basis lookups stay valid (U'_j = a + b - U_{m-j}, where
// [a,b] = [knots[0], knots[m]]). The operation is its own inverse:
// a+b-(a+b-U_k) = U_k.
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
// average-of-p-knots association that places a frame "at" each control
// point of a curve with no other canonical per-CP parameter (Sweep1).
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

// Divide: points evenly spaced by arc length rather than by parameter, which
// bunches on unevenly parametrized curves. A general rational NURBS curve has
// no closed-form arc length, so the standard numerical recipe is used: a dense
// chord-length polyline refined until each chord deviates from the curve by
// less than a tolerance, then linear interpolation along that polyline to
// invert length -> parameter. The recursion mirrors the app's render
// tessellation (`sampleCurveAdaptive`), on plain arrays so the kernel has no
// THREE dependency.
function chordDeviationPlain(p0, p1, pMid) {
  const chord = sub(p1, p0);
  const chordLenSq = length(chord) ** 2;
  if (chordLenSq < 1e-12) return length(sub(pMid, p0)); // degenerate zero-length chord
  const t = (pMid[0]-p0[0])*chord[0]/chordLenSq + (pMid[1]-p0[1])*chord[1]/chordLenSq + (pMid[2]-p0[2])*chord[2]/chordLenSq;
  const proj = [p0[0]+chord[0]*t, p0[1]+chord[1]*t, p0[2]+chord[2]*t];
  return length(sub(pMid, proj));
}
const DIVIDE_MAX_DEPTH = 20; // bisection depth safety cap, same role as sampleCurveAdaptive's ADAPTIVE_CURVE_MAX_DEPTH
// A one-sample chord-deviation test cannot be trusted to terminate the
// recursion. On a span that is point-symmetric about its own midpoint —
// C(m+s) = 2C(m) - C(m-s), e.g. an S-curve through symmetric points — the
// midpoint lies exactly on the chord, so the single deviation sample reads
// zero while the curve bows away everywhere else. Unguarded, that returns the
// two endpoints alone and every consumer (arc length, length -> parameter
// inversion, Divide stations, sweep rail stations) degrades to the straight
// chord. This is a per-span property, so one symmetric span between two knots
// of an ordinary curve is affected too.
//
// Hence a minimum depth: the deviation test may not stop anything until the
// span has been bisected DIVIDE_MIN_DEPTH times, so termination requires
// 2^DIVIDE_MIN_DEPTH sub-span tests at parameters no single symmetry can zero
// out at once. Only a flat span pays for it; a curved span subdivides past
// this depth anyway.
//
// A degree <= 1 span is exempt: its basis functions are non-negative and sum
// to one, so C(u) is a convex combination of the span's two control points
// (weights included) — the chord is the curve and subdividing reveals
// nothing.
const DIVIDE_MIN_DEPTH = 2;
// Dense {u, point} chain, adaptively refined to `tolerance` (same units as
// the curve's own control points — caller picks a value relative to the
// curve's own scale, see `divideByArcLength`'s default below).
export function adaptiveArcLengthSamples(crv, uStart, uEnd, tolerance) {
  /* The domain defaults to the curve's own. A NURBS curve's domain is
     usually not [0,1], so there is no safe fixed default. */
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

// Total arc length (dense-polyline approximation, accurate to within
// `tolerance`).
export function curveLength(crv, uStart, uEnd, tolerance) {
  const samples = adaptiveArcLengthSamples(crv, uStart, uEnd, tolerance);
  let total = 0;
  for (let i = 1; i < samples.length; i++) total += length(sub(samples[i].pt, samples[i - 1].pt));
  return total;
}

// Arc-length -> parameter inversion, factored out of divideByArcLength so a
// caller needing many inversions against the same curve (sweepNProfiles in
// kernel/sweep.mjs evaluates rail frames at several arc-length fractions
// between two profile stations) builds the dense polyline and cumulative-length
// table once rather than per lookup. `buildArcLengthTable` is the same
// sampling + running-sum recipe divideByArcLength uses inline;
// `paramAtArcLength` is the same binary search + linear interpolation along
// the bracketing segment, against an arbitrary target length.
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

// Parameter -> arc length, against the same table as paramAtArcLength.
// sweepNProfiles needs both directions against one dense polyline: stations'
// arc length (this function) and in-between samples' rail parameter
// (paramAtArcLength). One shared table keeps a station's vbar consistent with
// the table's `total`; deriving one side of that ratio from a separate, coarser
// curveLength() leaves the last station's vbar short of 1.0 and pushes the
// surface's v=1 edge into extrapolation.
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

// Closest point on a curve to an arbitrary 3D point — for stationing a
// Sweep1-with-N-profiles cross-section along its rail: each profile is placed
// near its intended rail location and its rail parameter is inferred from
// proximity. Two stages, the standard numeric recipe (the shape of P&T §6.1
// point inversion — a coarse global search first, so Newton cannot converge to
// the wrong local minimum, then Newton-Raphson for accuracy):
//   1. Dense chord-deviation-adaptive samples (`adaptiveArcLengthSamples`)
//      give a polyline; the target point is projected onto every segment
//      (clamped to [0,1] per segment) and the closest kept as the seed.
//   2. A bounded number of Newton-Raphson iterations on f(u) = |C(u)-P|^2
//      (root of f', using the curve's analytic first/second derivatives)
//      refine the seed. A step that leaves the domain or increases the
//      distance is rejected, keeping the coarse answer.
//
// Ambiguity is reported rather than resolved: a rail that loops back near
// itself can have two different parameter values nearly equidistant from the
// target, and picking one would misplace a profile's station. Detected by
// scanning the same dense samples for every local-minimum dip in distance and
// flagging when a second, parametrically distant dip comes within a small
// relative tolerance of the best one. This is a heuristic on dense-sample local
// minima, not a proof of global uniqueness.
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

  // Stage 1 — best point on any polyline segment, clamped, not just a
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

// Whether a curve's start and end control points coincide within tolerance,
// making u=uMin and u=uMax the same physical point (a seam, not two
// endpoints). The curve analog of surface.mjs's surfaceClosure.
export function isCurveClosed(crv, tol = 1e-6) {
  const p0 = crv.ctrlPts[0], p1 = crv.ctrlPts[crv.ctrlPts.length - 1];
  return Math.hypot(p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]) <= tol && Math.abs(p0[3] - p1[3]) <= tol;
}

// Divide: for an open curve, `count` segments -> count+1 points including both
// endpoints (Rhino/MoI's Divide-by-segment-count convention). For a closed
// curve (isCurveClosed) u=uMin and u=uMax are the same seam point, so exactly
// `count` points are returned, evenly spaced around the full length without
// repeating the seam. Returns [{u, point}]; `point` is an exact curve
// evaluation at the found u — only the arc-length position of u is
// approximate to within tolerance.
export function divideByArcLength(crv, count, tolerance) {
  assertCurve(crv, 'divideByArcLength');
  if (!Number.isInteger(count) || count < 1) throw new Error('divideByArcLength: count must be a positive integer');
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  if (tolerance === undefined) {
    // Scale-relative default: a coarse one-chord-per-knot-span estimate of
    // the curve's length sets the tolerance, since the curve could be 1 unit
    // or 100000 units across.
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
  const n = closed ? count : count + 1; // closed: count points, no seam duplicate; open: count+1 including both endpoints
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

// Dedupe: are two curves the same curve, spatially, within `tolerance`?
// Coincident duplicates would confuse downstream angle/parallel/
// loft-compatibility classification. Takes two resampled point chains (plain
// [x,y,z] triples) rather than two NurbsCrv objects, so the same function works
// for a Line (2 points), a Polyline (its vertices), or a dense SketchCurve or
// Circle chain without special cases.
//
// Two checks, ordered to reject the main false positive (a sub-segment of a
// much longer curve) early:
//   1. Extent — the chains' chord-summed lengths must match within a small
//      multiple of `tolerance`. Not `tolerance` itself: two independently
//      refined chains of an identical curve differ slightly in vertex count and
//      placement, while a sub-segment is shorter by a modeling-scale amount.
//   2. Bidirectional nearest-point — every point of A has a near point in B,
//      and every point of B has a near point in A (order- and
//      direction-agnostic, so a reversed duplicate matches). A->B alone would
//      pass a short A lying inside a longer B; B->A catches it. This is the
//      Hausdorff-distance-under-tolerance test; check (1) is a cheap early
//      exit, not a substitute, since a curve folding back on itself can match
//      the length and still fail (2).
// A bounding-box pre-filter belongs to the caller, which has the objects to
// test cheaply before building two sample chains.
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

// The same test, inverted: the tolerance this pair would need, rather than a
// boolean at a given tolerance. A caller that found no match can then report
// what it measured, what it required, and what value would have worked.
//
// The algebraic inverse of curvesCoincident's two checks:
//   · Extent needs `|lenA - lenB| <= tolerance * 10`, so it alone demands
//     `tolerance >= |lenA - lenB| / 10`.
//   · Bidirectional nearest-point needs every point of each chain within
//     `tolerance` of the other chain, so it demands a tolerance above the
//     larger of the two directed maxima — the symmetric Hausdorff distance.
// The pair's requirement is the larger of the two, and a tolerance at or below
// it fails.
//
// The returned value is an infimum, not a working setting: the nearest-point
// comparison is strict (`<`), so a tolerance exactly equal to the Hausdorff
// term still fails. A caller showing it to a user must round it up (see
// coincidenceGapLabels in the app).
//
// Kept separate from curvesCoincident: the boolean form early-exits on the
// first far point and runs O(n^2) times over a document's curves, while this
// one must visit every point of both chains to find a maximum.
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

// Two rails meeting at a point, joined into one longer rail — the
// concatenation a two-rail pipe junction wants, rather than a weld between two
// independently swept stubs. Sweeping one continuous tube along the joined rail
// brings `railInteriorCorners`/`applyTrueMiterStretch` (an elliptical miter
// at the junction instead of a butt joint), `applyMiterLimitFallback` (a
// fillet for a corner too sharp to miter), and one unbroken parallel-transport
// frame chain across the junction. It is the same move `pipeRailForSweep`
// makes for a single rail's interior corners.
//
// Why this lives in curve.mjs and not knots.mjs: every decision here is curve
// geometry — which endpoints coincide (evaluation + a distance tolerance),
// which curve needs reversing (`reverseCurve`), and whether the rails fold
// back on each other (end tangents via `rationalCurveDerivs`). The knot
// arithmetic — degree elevation to a common degree, domain rescaling, the C0
// splice — is delegated to `joinCurvesC0`. knots.mjs imports only basis.mjs,
// so this direction of dependency has no cycle.
//
// Limit: when both rails are degree 1 (Line/Polyline), the result is degree 1
// and the junction is an interior control point, which `railInteriorCorners`
// reports, miters and miter-limits like any other corner. When either input
// is degree > 1, or `joinCurvesC0` elevates the pair to a common degree > 1,
// `railInteriorCorners` returns nothing (its contract is a degree <= 1 rail's
// raw control points) and `sweep1Rigid` dispatches to `sweep1RigidResampled`,
// which has no miter machinery — the junction gets an un-mitered elbow,
// narrower than a true miter by roughly cos(theta/2). On a 90-degree junction
// with a radius-5 tube the skin reaches ~5.24 from the rail where a true miter
// reaches r*sec(45deg) = 7.071, i.e. 74%. A test pins this.
//
// 0.001 mm — the same coincident-point tolerance the app's Join uses for
// endpoint chaining. Defined locally because this module has no app-layer
// dependency, as kernel/loft.mjs does with CLOSE_LOOP_TOL.
export const RAIL_JUNCTION_TOLERANCE = 0.001;
// A turn this close to a full reversal is refused. The threshold is
// `filletCornerArc`'s bound for "a near-180 reversal has no well-defined
// fillet", so the refusal matches a real downstream incapacity — see the
// fold-back case below.
const RAIL_FOLD_BACK_EPS = 1e-6;

// Domain ends of a clamped curve (knots[0] / knots[last]), the convention
// `joinCurvesC0`'s `rescaleCurveDomain` also reads.
function railDomainEnds(crv) {
  const a = crv.knots[0], b = crv.knots[crv.knots.length - 1];
  return [{ key: 'start', u: a, pt: curvePoint(crv, a) }, { key: 'end', u: b, pt: curvePoint(crv, b) }];
}
// Unit direction of travel at a parameter. Callers evaluate an
// already-reversed curve rather than negating a tangent, so the sign is always
// the curve's own.
function unitTravelDirection(crv, u) {
  const [, C1] = rationalCurveDerivs(crv, u, 1);
  const L = length(C1);
  return L > 0 ? scale(C1, 1 / L) : null;
}

// Concatenate two rails meeting at a shared endpoint into one curve
// traversing A then B. Returns `{ ok: true, curve, ... }` or
// `{ ok: false, reason }` — a result object, not a throw, as in
// `filletOpenPolyline`/`filletPolygon`, because the refusals are ordinary user
// situations (two curves that do not touch) rather than programmer errors.
//
// All four end-pairings work: the other three are handled by reversing
// whichever curve needs it (`reverseCurve`, exact, P&T §6.5), never by
// resampling.
//
// Three refusals:
//  - No shared endpoint. The nearest of the four endpoint pairs is reported
//    with its gap.
//  - Ambiguous. More than one distinct end of A (or of B) lands on the other
//    rail: A is closed, or B is closed, or the two together close a loop.
//    Each such case has two or more valid answers, and the loop case also
//    produces a second junction at the result's seam that nothing here
//    examines.
//  - Fold-back. The two rails leave the junction in exactly opposite
//    directions. At a full reversal the miter's bisector tangent `tIn + tOut`
//    is the zero vector and the bend direction `normalize(tOut - tIn)`
//    collapses onto the rail's tangent, so the elliptical stretch has nothing
//    perpendicular to act on. `applyMiterLimitFallback` cannot rescue it — its
//    fillet is `filletCornerArc`, which refuses a near-180 turn — so
//    `applyTrueMiterStretch` would clamp the stretch to `PIPE_MITER_LIMIT` and
//    apply it along the tangent, a no-op on a ring perpendicular to it: the
//    tube runs out and straight back through itself. A turn merely close to
//    180 degrees is not refused, because the miter-limit fallback fillets
//    it.
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
  // The result's control point at the junction, when one exists — the index
  // `railInteriorCorners` reports for a degree-1 pair. Found by search because
  // `joinCurvesC0` may have degree-elevated the inputs, changing the control
  // point count; `null` when nothing lands there.
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
    // first and second curve is exactly u = 1 by that function's convention.
    junctionParam: 1,
    junctionIndex,
    turnAngle,
    reversedA,
    reversedB,
    gap: m.gap,
  };
}
