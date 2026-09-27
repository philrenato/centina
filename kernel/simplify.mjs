// Point-count decimation and insertion — the two operations behind the
// point-count knob, a live, non-destructive alternative to a Rebuild command.
// Operates on plain point arrays ([x,y,z] triples), never on a control net
// directly, as Divide/Sweep auto-stationing do (curves are combined as
// continuous values, never as raw control points).
//
// Decimation is rank-based — Visvalingam & Whyatt 1993's "effective area"
// metric — rather than epsilon-driven Ramer-Douglas-Peucker. V-W scores every
// point by the area of the triangle it forms with its two immediate neighbors;
// a small area means the point is nearly collinear with its neighbors and
// contributes little visual detail. The lowest-scoring point is removed first,
// one at a time, with its new neighbors' scores recomputed after each removal
// (the V-W iterative loop; the paper also holds a neighbor's score at least at
// the eliminated point's area, which this does not). Removing one at a time
// maps directly onto "keep exactly N points"; RDP's epsilon would need a
// binary search to hit an exact target count.
//
// The surviving points feed into globalCurveInterp/closedCurveInterp
// (kernel/interpolate.mjs); only the point selection is new.

import { sub, dot, length } from './vec3.mjs';
import { globalCurveInterp, closedCurveInterp } from './interpolate.mjs';
import { curvePoint, adaptiveArcLengthSamples } from './curve.mjs';

function triangleArea(a, b, c) {
  const ab = sub(b, a), ac = sub(c, a);
  const cx = ab[1] * ac[2] - ab[2] * ac[1];
  const cy = ab[2] * ac[0] - ab[0] * ac[2];
  const cz = ab[0] * ac[1] - ab[1] * ac[0];
  return 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
}

// Decimate an open point sequence down to exactly `targetCount` points via
// iterative Visvalingam-Whyatt. Both endpoints are never removed (infinite
// effective area, as divideByArcLength/adaptiveArcLengthSamples keep both
// endpoints exact). O(n^2): simple and correct at the point counts a sketched
// curve reaches (tens, not hundreds). A caller with several hundred raw points
// should use a min-heap (the standard O(n log n) V-W implementation).
export function decimateOpenToCount(points, targetCount) {
  const n = points.length;
  if (targetCount >= n) return points.map((p) => [...p]);
  if (targetCount < 2) throw new Error('decimateOpenToCount needs a target of at least 2 (both endpoints)');
  const prev = new Array(n), next = new Array(n);
  const alive = new Array(n).fill(true);
  for (let i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1; }
  next[n - 1] = -1;
  const areaOf = (i) => (prev[i] === -1 || next[i] === -1) ? Infinity : triangleArea(points[prev[i]], points[i], points[next[i]]);
  let aliveCount = n;
  while (aliveCount > targetCount) {
    let minIdx = -1, minArea = Infinity;
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      const a = areaOf(i);
      if (a < minArea) { minArea = a; minIdx = i; }
    }
    alive[minIdx] = false;
    if (prev[minIdx] !== -1) next[prev[minIdx]] = next[minIdx];
    if (next[minIdx] !== -1) prev[next[minIdx]] = prev[minIdx];
    aliveCount--;
  }
  const result = [];
  for (let i = 0; i < n; i++) if (alive[i]) result.push(points[i]);
  return result;
}

// The farthest-apart pair of points in a closed point set — the anchor
// pair a closed decimation splits on (see decimateClosedToCount below).
// O(n^2), same scale reasoning as decimateOpenToCount above.
export function farthestPointPairIndices(points) {
  const n = points.length;
  let best = { i: 0, j: Math.min(1, n - 1), dist: -1 };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = length(sub(points[i], points[j]));
      if (d > best.dist) best = { i, j, dist: d };
    }
  }
  return [best.i, best.j];
}

function arcSlice(points, from, to) {
  const n = points.length;
  const result = [];
  let idx = from;
  while (true) {
    result.push(points[idx]);
    if (idx === to) break;
    idx = (idx + 1) % n;
  }
  return result;
}

// Decimate a closed point sequence down to exactly `targetCount` points.
// A closed loop has no natural anchor pair the way an open sequence's two
// endpoints are — plain Visvalingam-Whyatt run on it directly would privilege
// whichever point sits first/last in the stored array, an artifact of capture
// order rather than a geometric feature. So the loop is split at its
// farthest-apart point pair first — an order-independent geometric anchor —
// each resulting arc is decimated independently as an open sequence (each
// keeping its two endpoints, which are the anchor points shared between the
// arcs), and the arcs are recombined.
export function decimateClosedToCount(points, targetCount) {
  const n = points.length;
  if (targetCount >= n) return points.map((p) => [...p]);
  if (targetCount < 3) throw new Error('decimateClosedToCount needs a target of at least 3');
  const [i, j] = farthestPointPairIndices(points);
  const arcA = arcSlice(points, i, j);
  const arcB = arcSlice(points, j, i);
  const innerA = arcA.length - 2, innerB = arcB.length - 2; // interior points, excluding each arc's own 2 shared endpoints
  const totalInner = Math.max(0, innerA) + Math.max(0, innerB);
  const targetInner = Math.max(0, targetCount - 2);
  const targetInnerA = totalInner > 0 ? Math.round(targetInner * Math.max(0, innerA) / totalInner) : 0;
  const targetInnerB = targetInner - targetInnerA;
  const decA = decimateOpenToCount(arcA, Math.max(2, targetInnerA + 2));
  const decB = decimateOpenToCount(arcB, Math.max(2, targetInnerB + 2));
  return [...decA, ...decB.slice(1, -1)]; // decB's two endpoints (j, i) are already the seam with decA — dropped to avoid duplicating
}

export function decimateToCount(points, targetCount, closed) {
  return closed ? decimateClosedToCount(points, targetCount) : decimateOpenToCount(points, targetCount);
}

// Perpendicular distance from a curve sample to the chord between the span's
// two endpoint values (not the stored points — the curve may not pass exactly
// through every stored point once a manual edit has reshaped the working set,
// so this measures against the curve's evaluated endpoints).
function chordDeviation(pt, chordA, chordB) {
  const ab = sub(chordB, chordA);
  const abLenSq = dot(ab, ab);
  if (abLenSq < 1e-18) return length(sub(pt, chordA));
  let t = dot(sub(pt, chordA), ab) / abLenSq;
  t = Math.max(0, Math.min(1, t));
  const closest = [chordA[0] + ab[0] * t, chordA[1] + ab[1] * t, chordA[2] + ab[2] * t];
  return length(sub(pt, closest));
}

// The point on the curve, within [u0,u1], that deviates furthest from the
// straight chord between the curve's values at u0/u1 — a fixed-sample scan
// (not adaptive like adaptiveArcLengthSamples, since this only needs a
// maximum good enough to rank spans against each other).
function highestDeviationInSpan(crv, u0, u1, samples = 16) {
  const chordA = curvePoint(crv, u0), chordB = curvePoint(crv, u1);
  let bestU = (u0 + u1) / 2, bestPt = curvePoint(crv, bestU), bestDev = chordDeviation(bestPt, chordA, chordB);
  for (let i = 1; i < samples; i++) {
    const u = u0 + (u1 - u0) * i / samples;
    const pt = curvePoint(crv, u);
    const dev = chordDeviation(pt, chordA, chordB);
    if (dev > bestDev) { bestDev = dev; bestU = u; bestPt = pt; }
  }
  return { u: bestU, point: bestPt, deviation: bestDev };
}

// Insert `extraCount` new points into a working point set, one at a time,
// each at the current fit curve's highest-chord-deviation location (the
// highest-curvature location, the inverse of Illustrator's Simplify) — a point
// on the curve, evaluated from the interpolation, never a straight-line
// midpoint (which would flatten the detail this exists to add back). Re-fits
// the curve after each single insertion so each new point responds to the
// shape after its predecessor was added — fine at the small insertion counts
// a slider tick requests.
export function insertByHighestDeviation(points, extraCount, degree, closed) {
  let current = points.map((p) => [...p]);
  for (let k = 0; k < extraCount; k++) {
    const n = current.length;
    let crv, paramAt, spanCount;
    if (closed) {
      const built = closedCurveInterp(current, degree);
      crv = built.crv;
      paramAt = (i) => crv.paramsUsed[degree + i]; // i in [0,n]; i===n wraps back to point 0 (uEnd)
      spanCount = n; // wraps: span n-1 goes from point n-1 back to point 0
    } else {
      crv = globalCurveInterp(current, degree);
      paramAt = (i) => crv.paramsUsed[i];
      spanCount = n - 1;
    }
    let bestSpan = -1, bestDev = -1, bestPt = null;
    for (let i = 0; i < spanCount; i++) {
      const u0 = paramAt(i), u1 = paramAt(i + 1);
      if (!(u1 > u0)) continue; // a degenerate/duplicate-parameter span has nothing to search
      const found = highestDeviationInSpan(crv, u0, u1);
      if (found.deviation > bestDev) { bestDev = found.deviation; bestSpan = i; bestPt = found.point; }
    }
    if (bestSpan === -1) break; // every span degenerate — nothing left to refine
    current.splice(bestSpan + 1, 0, bestPt);
  }
  return current;
}

// Uniform spacing mode (the alternative to the rank-based adaptive
// decimation/insertion above): resample the current fit curve at exactly
// `count` points, evenly by arc length — the technique of Divide
// (divideByArcLength, kernel/curve.mjs), generalized to an explicit [u0,u1]
// sub-range rather than the full knot domain, so it also works on a
// closedCurveInterp result, whose useful domain is [uStart,uEnd], not its full
// wrap-padded knot range. `includeEnd` mirrors divideByArcLength's
// open-vs-closed distinction: true for an open curve (both endpoints
// included, as in Rhino's Divide), false for a closed one (u0/u1 are the same
// seam point — including both would duplicate it).
export function resampleUniformInRange(crv, u0, u1, count, includeEnd, tolerance) {
  if (!Number.isInteger(count) || count < 1) throw new Error('resampleUniformInRange: count must be a positive integer');
  if (tolerance === undefined) {
    let coarse = 0, prev = curvePoint(crv, u0);
    const STEPS = 20;
    for (let i = 1; i <= STEPS; i++) {
      const u = u0 + (u1 - u0) * i / STEPS;
      const p = curvePoint(crv, u);
      coarse += length(sub(p, prev));
      prev = p;
    }
    tolerance = Math.max(coarse * 1e-6, 1e-9);
  }
  const samples = adaptiveArcLengthSamples(crv, u0, u1, tolerance);
  const cumLen = [0];
  for (let i = 1; i < samples.length; i++) cumLen.push(cumLen[i - 1] + length(sub(samples[i].pt, samples[i - 1].pt)));
  const total = cumLen[cumLen.length - 1];
  const denom = includeEnd ? count - 1 : count;
  const results = [];
  for (let i = 0; i < count; i++) {
    if (i === 0) { results.push(curvePoint(crv, u0)); continue; }
    if (includeEnd && i === count - 1) { results.push(curvePoint(crv, u1)); continue; }
    const targetLen = denom > 0 ? (total * i) / denom : 0;
    let lo = 0, hi = cumLen.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (cumLen[mid] < targetLen) lo = mid; else hi = mid;
    }
    const segLen = cumLen[hi] - cumLen[lo];
    const u = segLen < 1e-12 ? samples[lo].u : samples[lo].u + (samples[hi].u - samples[lo].u) * (targetLen - cumLen[lo]) / segLen;
    results.push(curvePoint(crv, u));
  }
  return results;
}

// The entry point the app layer calls: given a curve's current working point
// set, its pointCount/degree/spacing params, and whether it is closed, produce
// the new working point set the knobs request. Dispatches adaptive
// (rank-based decimate/insert, preserves as many original points as possible)
// vs uniform (a fresh even-arc-length resample of the fit curve, discarding
// original point identity) — both on existing interpolation machinery.
export function regeneratePointSet(workingPoints, targetCount, degree, closed, spacing) {
  if (spacing === 'uniform') {
    if (closed) {
      const { crv, uStart, uEnd } = closedCurveInterp(workingPoints, degree);
      return resampleUniformInRange(crv, uStart, uEnd, targetCount, false);
    }
    const crv = globalCurveInterp(workingPoints, degree);
    const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
    return resampleUniformInRange(crv, uMin, uMax, targetCount, true);
  }
  if (targetCount === workingPoints.length) return workingPoints.map((p) => [...p]);
  if (targetCount < workingPoints.length) return decimateToCount(workingPoints, targetCount, closed);
  return insertByHighestDeviation(workingPoints, targetCount - workingPoints.length, degree, closed);
}
