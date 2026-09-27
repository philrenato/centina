// Fit a NURBS curve through sampled points, to a stated tolerance.
// Piegl & Tiller §9.4.4, "Approximation to Within a Specified Accuracy"
// (least-squares with a control-point count raised
// until the measured deviation clears the bound), with the endpoints
// interpolated exactly (P&T Eq. 9.63-9.67).
//
// Purpose: this kernel's boolean produces its cut curves by marching — an SSI
// component arrives as a few hundred sampled points, and every trim boundary
// downstream of it is a polyline of those samples. That suffices for
// tessellating and classifying, but not as geometry: a half-edge reserves a
// `pcurve` slot and an edge reserves `curve3d` + `tolerance`, which map
// one-to-one onto a B-rep trim and its edge. This module fills them.
//
// It does not replace the polyline. The polyline is what trims, sews and
// tessellates; a fitted curve is the exact record alongside it, which is what
// the nullable `curve3d` slot is for.
//
// A fit is not a measurement of itself. `fitCircle` reports its own residual,
// and a least-squares solve reports nothing; neither answers "how far is the
// worst input point from the curve about to be returned". So every path here
// measures the returned curve against the original points by the same
// conservative rule (see `maxDeviationFromCurve`) and refuses rather than
// return a curve that misses its bound.
import { findSpan, basisFuns } from './basis.mjs';
import { chordLengthParams, solveLinearSystem, averagingKnotVector, interpAtParams } from './interpolate.mjs';
import { curvePoint, reverseCurve } from './curve.mjs';
import { fitLine, fitCircle, fitEllipse } from './refit.mjs';
import { extractSubCurve, rescaleCurveDomain } from './knots.mjs';
import { makeLine, makeCircle, makeEllipse } from './primitives.mjs';

// Deviation, measured so it can only over-report, and without a resolution
// floor. Two stages:
//   1. Bracket on a sampled polyline — which segment is nearest.
//   2. Refine by ternary search on |C(t) - Q| over that bracket, evaluating
//      the curve, so the answer is the distance to a point on the curve.
//
// Stage 2 is required because a polyline sampled through a curve is inscribed
// in it, so distances measured to that polyline are too large by the sagitta
// of the sampling — on a radius-25 circle at 960 segments that is 1.3e-4, a
// floor no request for 1e-6 can get under. A measure dominated by its own
// discretization reports its step size, not the deviation.
//
// The conservative property survives refinement: whatever t the search lands
// on, |C(t) - Q| is the distance to an actual point of the curve, so it is
// never less than the true minimum distance. The search tightens the bound and
// never makes it optimistic.
// It is one-sided on purpose — samples to curve, never curve back to the input
// polyline. A two-sided (Hausdorff) measure is wrong for this job: the
// polyline is inscribed in the shape its samples came from, so a smooth curve
// is necessarily about one sagitta away from those chords. An exact circle
// through 120 samples of radius 25 scores 8.6e-3 against itself that way — the
// measure penalizing the fit for being smoother than the data.
// The risk a two-sided measure addresses — a curve threading its samples and
// swinging between them — is handled by `corridorExcess` below, which measures
// the other direction with the sagitta the data implies subtracted.
export function maxDeviationFromCurve(points, crv, opts = {}) {
  const samplesPerPoint = opts.samplesPerPoint ?? 4;
  const p = crv.degree, U = crv.knots;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  const n = Math.max(64, points.length * samplesPerPoint);
  const ts = [], poly = [];
  for (let i = 0; i <= n; i++) { const t = t0 + (t1 - t0) * (i / n); ts.push(t); poly.push(curvePoint(crv, t)); }
  const distSq = (t, q) => { const c = curvePoint(crv, t); const dx = c[0] - q[0], dy = c[1] - q[1], dz = c[2] - q[2]; return dx * dx + dy * dy + dz * dz; };
  let worst = 0;

  // Bracket on the sampled polyline, then refine on the curve itself.
  for (const q of points) {
    let bestSeg = 0, best = Infinity;
    for (let i = 0; i < poly.length - 1; i++) {
      const d = pointSegmentDistanceSq(q, poly[i], poly[i + 1]);
      if (d < best) { best = d; bestSeg = i; }
    }
    let lo = ts[Math.max(0, bestSeg - 1)], hi = ts[Math.min(n, bestSeg + 2)];
    for (let it = 0; it < 60 && hi - lo > 1e-15; it++) {
      const a = lo + (hi - lo) / 3, b = hi - (hi - lo) / 3;
      if (distSq(a, q) < distSq(b, q)) hi = b; else lo = a;
    }
    const d = Math.sqrt(distSq(0.5 * (lo + hi), q));
    if (d > worst) worst = d;
  }
  return worst;
}

function pointSegmentDistanceSq(q, a, b) {
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const aqx = q[0] - a[0], aqy = q[1] - a[1], aqz = q[2] - a[2];
  const len2 = abx * abx + aby * aby + abz * abz;
  let t = len2 > 0 ? (aqx * abx + aqy * aby + aqz * abz) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const dx = aqx - abx * t, dy = aqy - aby * t, dz = aqz - abz * t;
  return dx * dx + dy * dy + dz * dz;
}

// How far the curve may wander between two samples, derived from the data.
//
// `maxDeviationFromCurve` is one-sided because the data polyline is inscribed
// in the shape it was sampled from, so an uncorrected two-sided (Hausdorff)
// measure penalizes a fit for being smoother than its chords by about one
// sagitta of the sampling. That sagitta is a property of the data, and can be
// computed from it.
//
// For each chord Q_i..Q_{i+1}, take the circle through it and each of its two
// neighboring points, and keep the larger of the two sagittas that circle cuts
// over that chord. That is the second-order reconstruction of what the samples
// imply happens between them — zero where three points are collinear, and the
// shape's own bulge where they are not. The curve is then allowed to sit
// `tolerance + h_i` from chord i and no further.
//
// `tolerance` is the caller's, and `h_i` is measured off the caller's points;
// there is no tuned constant.
//
// This catches what the deviation cannot. An adaptive sampler puts its points
// far apart where the shape is straight, so the longest chords have no sample
// in the middle to hold the curve down — and a fit that swings out there scores
// a perfect deviation while bowing a straight stem.
function chordSagittas(points, closed) {
  const n = points.length;
  const count = closed ? n : n - 1;
  const out = new Array(Math.max(0, count)).fill(0);
  if (count < 1) return out;
  const at = (i) => points[((i % n) + n) % n];
  const inRange = (i) => closed || (i >= 0 && i < n);
  for (let i = 0; i < count; i++) {
    let h = 0;
    for (const k of [i - 1, i + 2]) {
      if (!inRange(k)) continue;
      h = Math.max(h, sagitta(at(i), at(i + 1), at(k)));
    }
    out[i] = h;
  }
  return out;
}

// The sagitta the circle through a, b, c cuts over the chord a..b. Collinear
// (or coincident) triples have no circle and no bulge, which is the answer a
// straight run needs.
function sagitta(a, b, c) {
  const ab = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const bc = Math.hypot(c[0] - b[0], c[1] - b[1], c[2] - b[2]);
  const ca = Math.hypot(a[0] - c[0], a[1] - c[1], a[2] - c[2]);
  const s = (ab + bc + ca) / 2;
  const areaSq = s * (s - ab) * (s - bc) * (s - ca);
  if (!(areaSq > 0)) return 0;
  const radius = (ab * bc * ca) / (4 * Math.sqrt(areaSq));
  if (!Number.isFinite(radius) || !(radius > ab / 2)) return ab / 2;
  return radius - Math.sqrt(Math.max(0, radius * radius - (ab / 2) * (ab / 2)));
}

// How far outside that corridor the curve gets, at its worst. Zero or negative
// is inside. Sampled on the curve, because the excursion happens at parameters
// no data point owns.
function corridorExcess(points, crv, closed, tolerance) {
  const n = points.length;
  const segs = closed ? n : n - 1;
  if (segs < 1) return 0;
  const h = chordSagittas(points, closed);
  const p = crv.degree, U = crv.knots;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  const samples = Math.max(64, points.length * 2);
  let worst = -Infinity;
  for (let i = 0; i <= samples; i++) {
    const c = curvePoint(crv, t0 + ((t1 - t0) * i) / samples);
    let excess = Infinity;
    for (let j = 0; j < segs; j++) {
      const d = Math.sqrt(pointSegmentDistanceSq(c, points[j], points[(j + 1) % n]));
      const e = d - (tolerance + h[j]);
      if (e < excess) excess = e;
    }
    if (excess > worst) worst = excess;
  }
  return worst;
}

// The spacing of the samples is not the shape of the samples. Chord-length
// parametrization (P&T Eq. 9.5) spends parameter in proportion to distance.
// That is right when the points are evenly spread and wrong when they are not —
// and this kernel's samplers are adaptive: they put their points far apart
// where the shape is straight. A marched intersection takes long steps through
// low curvature; a Douglas-Peucker simplification deletes every interior point
// of a straight run. Either way one leg of the data can carry most of the
// parameter domain while all the shape change is crowded into the rest, so the
// curve has to turn inside a sliver of parameter — which it can only do by
// throwing a control point far out, and the resulting bulge lands on the long
// straight leg, where there is no sample to constrain it.
//
// Centripetal parametrization (P&T Eq. 9.6, after Lee, Computer-Aided Design 21(6), 1989) takes
// the square root of each chord. It is the standard answer to this case: it
// damps the ratio between the longest and the shortest leg without discarding
// the spacing information, as uniform parametrization would.
export function centripetalParams(points) {
  const n = points.length - 1;
  const ubar = new Array(points.length).fill(0);
  ubar[n] = 1;
  if (n === 0) return ubar;
  const roots = [];
  let total = 0;
  for (let k = 1; k <= n; k++) {
    const a = points[k - 1], b = points[k];
    const d = Math.sqrt(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
    roots.push(d);
    total += d;
  }
  // Same degenerate case chordLengthParams guards: all-coincident points have
  // no spacing to read, so uniform is the only answer that is not a division
  // by zero.
  if (!(total > 0)) {
    for (let k = 1; k < n; k++) ubar[k] = k / n;
    return ubar;
  }
  let acc = 0;
  for (let k = 1; k < n; k++) { acc += roots[k - 1]; ubar[k] = acc / total; }
  return ubar;
}

// Which parametrization suits the data is decided by measurement. Every fit
// below is built both ways against the same bound, and of the candidates that
// meet the bound the shorter curve is returned.
//
// That is a comparison, not a threshold: no length is tested against a
// constant. Two curves that each sit within `tolerance` of every sample agree
// with the data equally well and differ only between the samples, which the
// samples cannot settle. The shorter one adds no shape the data does not ask
// for — the fairness argument for preferring the lower-energy curve when the
// residual does not separate them.
//
// It also sees the failure `maxDeviationFromCurve` cannot. Deviation is
// one-sided, sample to curve; an excursion between two samples moves no sample
// and costs it nothing, but it always costs arc length.
function sampledLength(crv, samples = 256) {
  const p = crv.degree, U = crv.knots;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  let L = 0, prev = curvePoint(crv, t0);
  for (let i = 1; i <= samples; i++) {
    const c = curvePoint(crv, t0 + ((t1 - t0) * i) / samples);
    L += Math.hypot(c[0] - prev[0], c[1] - prev[1], c[2] - prev[2]);
    prev = c;
  }
  return L;
}

// Chord-length is asked first and therefore wins an exact tie: it is P&T's
// default and the parametrization every other fit and loft in this kernel
// uses.
const PARAMETRISATIONS = [chordLengthParams, centripetalParams];

// A9.1's three ingredients assembled against a stated parametrization rather
// than a derived one, via `interpAtParams`. With `chordLengthParams` this is
// `globalCurveInterp` exactly; writing it out lets the closed and open
// interpolations below run under either parametrization through one code
// path.
function interpolateWith(points, requestedDegree, paramsOf) {
  const n = points.length - 1;
  if (n < 1) throw new Error('interpolation needs at least 2 points');
  const p = Math.min(requestedDegree, n);
  const ubar = paramsOf(points);
  const knots = averagingKnotVector(ubar, p);
  return { degree: p, knots, ctrlPts: interpAtParams(points, p, ubar, knots), paramsUsed: ubar };
}

// The closed counterpart, `closedCurveInterp`'s construction: wrap `k`
// points cyclically off each end, interpolate the padded sequence openly, and
// keep the middle sub-range, so the clamped-end artifacts land in the padding
// and the seam carries a tangent instead of a kink.
function interpolateClosedWith(points, k, paramsOf) {
  const n = points.length;
  if (n < 3) throw new Error('closed interpolation needs at least 3 points');
  const extended = [];
  for (let i = 0; i < n + 2 * k; i++) extended.push(points[(((i - k) % n) + n) % n]);
  const crv = interpolateWith(extended, k, paramsOf);
  return { crv, uStart: crv.paramsUsed[k], uEnd: crv.paramsUsed[k + n] };
}

// The approximation knot vector is not the interpolation one. P&T Eq. 9.68-
// 9.69: with n+1 control points spread over m+1 points, interior knots are
// taken by averaging the parameters at evenly spaced fractional positions
// through the parameter list, so each knot span covers a similar number of
// samples. `averagingKnotVector` (interpolate.mjs) solves the different
// problem where those counts are equal, and produces a singular system here.
function approximationKnotVector(ubar, p, n) {
  const m = ubar.length - 1;
  const U = new Array(n + p + 2);
  for (let i = 0; i <= p; i++) U[i] = 0;
  for (let i = n + 1; i <= n + p + 1; i++) U[i] = 1;
  const d = (m + 1) / (n - p + 1);
  for (let j = 1; j <= n - p; j++) {
    const i = Math.floor(j * d);
    const alpha = j * d - i;
    const lo = ubar[Math.max(0, Math.min(m, i - 1))];
    const hi = ubar[Math.max(0, Math.min(m, i))];
    U[p + j] = (1 - alpha) * lo + alpha * hi;
  }
  return U;
}

// P&T Eq. 9.63-9.67. The two end control points are fixed to the first and
// last input point rather than solved for, so a fitted boundary meets its
// neighbors exactly at the corners the topology already agreed on — a gap at
// a shared corner is a naked edge, which outweighs a marginally lower
// residual.
export function leastSquaresFit(points, p, n, ubar) {
  const m = points.length - 1;
  const U = approximationKnotVector(ubar, p, n);
  const Q0 = points[0], Qm = points[m];
  // R_k = Q_k - N_{0,p}(u_k) Q_0 - N_{n,p}(u_k) Q_m, for the interior points.
  const Rk = [];
  const Nrows = [];
  for (let k = 1; k <= m - 1; k++) {
    const span = findSpan(n, p, ubar[k], U);
    const N = basisFuns(span, ubar[k], p, U);
    const row = new Array(n + 1).fill(0);
    for (let i = 0; i <= p; i++) row[span - p + i] = N[i];
    Nrows.push(row);
    Rk.push([
      points[k][0] - row[0] * Q0[0] - row[n] * Qm[0],
      points[k][1] - row[0] * Q0[1] - row[n] * Qm[1],
      points[k][2] - row[0] * Q0[2] - row[n] * Qm[2],
    ]);
  }
  // Normal equations over the free control points P_1..P_{n-1}.
  const size = n - 1;
  const A = Array.from({ length: size }, () => new Array(size).fill(0));
  const b = [new Array(size).fill(0), new Array(size).fill(0), new Array(size).fill(0)];
  for (let r = 0; r < Nrows.length; r++) {
    const row = Nrows[r];
    for (let i = 1; i <= n - 1; i++) {
      if (row[i] === 0) continue;
      for (let j = 1; j <= n - 1; j++) {
        if (row[j] === 0) continue;
        A[i - 1][j - 1] += row[i] * row[j];
      }
      for (let c = 0; c < 3; c++) b[c][i - 1] += row[i] * Rk[r][c];
    }
  }
  const [Px, Py, Pz] = solveLinearSystem(A, b);
  const ctrlPts = [[Q0[0], Q0[1], Q0[2], 1]];
  for (let i = 0; i < size; i++) ctrlPts.push([Px[i], Py[i], Pz[i], 1]);
  ctrlPts.push([Qm[0], Qm[1], Qm[2], 1]);
  return { degree: p, knots: U, ctrlPts };
}

// A closed loop is fitted by wrapping, as closedCurveInterp does and for the
// same reason: an open fit's clamped end behavior lands in padding that is
// then discarded, so the kept range behaves as if the curve continues
// periodically instead of showing a kink at the seam. This kernel has no
// periodic B-spline approximation; wrapping reaches the same result.
function wrapClosed(points, pad) {
  const n = points.length;
  const out = [];
  for (let i = n - pad; i < n; i++) out.push(points[i]);
  for (const q of points) out.push(q);
  for (let i = 0; i <= pad; i++) out.push(points[i % n]);
  return out;
}

// Least-squares does not interpolate the seam, so trimming the wrap padding
// leaves a loop whose two ends are near each other rather than at each other
// (on a 24-point closed superellipse of perimeter 72.30, a residual of 8.90e-2
// — 0.29% of the bbox diagonal, 18% of the fit tolerance). `isCurveClosed`
// answers at 1e-6: a nominally open loop makes a renderer draw a closing
// chord, and makes downstream extrude, offset and trim treat a closed profile
// as open.
//
// The two ends are moved to their midpoint, not to the data point at the seam.
// Both moves are the same size (on that superellipse: arc length ends at
// 1.00129x the data perimeter snapping to the midpoint, 1.00114x snapping to
// the sample, against 1.00252x with no snap), so the choice is about what the
// seam means. Snapping to the sample would make it the only exactly
// interpolated sample on an otherwise least-squares loop, biased onto a single
// possibly noisy measurement — the asymmetry `wrapClosed` exists to remove.
// The midpoint of the two computed ends is still a least-squares answer.
//
// The move happens before the deviation is measured. The caller's tolerance is
// a claim about the curve that is returned, so a snapped curve that misses its
// bound must fail the test and let the loop raise the control point count,
// which shortens the gap it then has to close.
function closeSeamExactly(crv) {
  const cp = crv.ctrlPts;
  const n = cp.length;
  if (n < 2) return crv;
  const a = cp[0], b = cp[n - 1];
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, (a[3] + b[3]) / 2];
  const out = cp.map((q) => q.slice());
  out[0] = mid.slice();
  out[n - 1] = mid.slice();
  return { ...crv, ctrlPts: out };
}

// Exact where the shape is exact. A plane cutting a cylinder or a sphere gives
// a circle or ellipse, and those are the commonest boolean cuts. Emitting the
// rational primitive means Rhino re-reads the boundary as a circle rather than
// as a spline that looks round. The recognizer's own residual is not trusted:
// the primitive is built and then measured against the original points like
// any other candidate.
function tryPrimitive(points, tolerance, closed, opts = {}) {
  const out = [];
  if (closed) {
    const c = fitCircle(points, {});
    if (c && c.ok && Number.isFinite(c.radius)) {
      out.push({ kind: 'circle', curve: makeCircle(c.center, c.xAxis, c.yAxis, c.radius) });
    }
    const e = fitEllipse(points, {});
    if (e && e.ok && Number.isFinite(e.radiusX) && Number.isFinite(e.radiusY)) {
      out.push({ kind: 'ellipse', curve: makeEllipse(e.center, e.xAxis, e.yAxis, e.radiusX, e.radiusY) });
    }
  } else {
    const l = fitLine(points, {});
    if (l && l.ok && l.start && l.end) out.push({ kind: 'line', curve: makeLine(l.start, l.end) });
  }
  let best = null;
  for (let cand of out) {
    if (!cand.curve || !cand.curve.ctrlPts) continue;
    // A primitive does not interpolate its endpoints. fitLine returns the
    // input projected onto the fitted line, so an open fit can move the first
    // and last points by the residual — harmless for a display curve, but a
    // trim's ends must meet its neighbors exactly: a pcurve starting at
    // u = -0.000754 instead of 0 is outside the domain, and OpenNURBS rejects
    // the loop for not joining. A primitive that moves an endpoint is rejected
    // here and the least-squares path, which fixes its endpoints by
    // construction, takes over.
    if (!closed && opts.exactEndpoints) {
      const q0 = points[0], qn = points[points.length - 1];
      const ends = (crv) => {
        const cp = crv.ctrlPts;
        const a = cp[0], b = cp[cp.length - 1];
        return Math.hypot(a[0] - q0[0], a[1] - q0[1], a[2] - q0[2])
          + Math.hypot(b[0] - qn[0], b[1] - qn[1], b[2] - qn[2]);
      };
      // A primitive can come back running the other way; it is reversed rather
      // than rejected. `fitLine` canonicalizes its direction (largest
      // component positive) so that near-identical input cannot flicker
      // between opposite directions, which means a run traveling in -x returns
      // start and end swapped. Reversal is exact and its own inverse.
      if (ends(cand.curve) > 1e-12) {
        const flipped = reverseCurve(cand.curve);
        if (ends(flipped) > 1e-12) continue;
        cand = { ...cand, curve: flipped };
      }
    }
    let dev;
    try { dev = maxDeviationFromCurve(points, cand.curve, { closed }); } catch { continue; }
    if (!Number.isFinite(dev)) continue;
    if (!best || dev < best.maxDeviation) best = { ...cand, maxDeviation: dev };
  }
  return best && best.maxDeviation <= tolerance ? best : null;
}

// Entry point.
//   points     — ordered [x,y,z]; for a closed loop do not repeat the first
//   tolerance  — the bound the returned curve is guaranteed to meet, measured
//                conservatively (see maxDeviationFromCurve)
//   degree     — requested; clamped down when there are too few points
//   closed     — the loop wraps
// Returns { ok, kind, curve, maxDeviation, ctrlPtCount, triedCounts } or
// { ok:false, reason } — never a curve that missed its bound.
export function fitCurveToPoints(points, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-3;
  const requestedDegree = opts.degree ?? 3;
  const closed = !!opts.closed;
  if (!Array.isArray(points) || points.length < 2) {
    return { ok: false, reason: 'fitCurveToPoints needs at least 2 points' };
  }
  if (!(tolerance > 0)) return { ok: false, reason: 'fitCurveToPoints needs a positive tolerance' };

  const prim = tryPrimitive(points, tolerance, closed, opts);
  if (prim) {
    return { ok: true, kind: prim.kind, curve: prim.curve, maxDeviation: prim.maxDeviation, ctrlPtCount: prim.curve.ctrlPts.length, triedCounts: [] };
  }

  const p = Math.min(requestedDegree, points.length - 1);
  const work = closed ? wrapClosed(points, p) : points;
  const m = work.length - 1;
  const ubars = PARAMETRISATIONS.map((paramsOf) => paramsOf(work));

  // Raise the count until it clears, rather than guessing one. Growth is
  // geometric so a curve needing many spans is reached in a few solves, and
  // the ceiling is m (at which point the fit has as many freedoms as points
  // and any remaining error is the parametrization, not the count).
  // The ceiling is n < m, not n <= m, as a conditioning limit. As the free
  // control points approach the interior point count the normal equations lose
  // rank — neighboring samples share a parameter to within the solver's pivot
  // threshold — and the solve throws. Stopping one short keeps every refusal a
  // statement about the data rather than about the matrix.
  const nMax = Math.max(p + 1, m - 1);
  const triedCounts = [];
  let n = Math.max(p + 1, Math.min(nMax, p + 2));
  let singularAt = null;
  // The shortest candidate that met the bound but left the corridor, kept so a
  // search that finds nothing better still answers with the fit it had.
  let wandering = null;
  for (;;) {
    let best = null;
    for (const ubar of ubars) {
      let candidate = null;
      try { candidate = leastSquaresFit(work, p, n, ubar); }
      catch { singularAt = n + 1; continue; }
      /* The wrap padding is cut off again. `wrapClosed` prepends `p` points
         and appends `p+1` so the fit's clamped ends land outside the loop;
         without the trim the returned curve runs past its own start and
         retraces (a closed superellipse of perimeter 72.358 comes back at arc
         length 91.734, 27% too long).
         The tolerance cannot detect this: deviation is measured from the data
         to the curve, and every point of a retrace is still on the shape. */
      const trimmed = closed
        ? closeSeamExactly(rescaleCurveDomain(extractSubCurve(candidate, ubar[p], ubar[p + points.length]), 0, 1))
        : candidate;
      let dev;
      try { dev = maxDeviationFromCurve(points, trimmed, { closed }); } catch { continue; }
      const excess = corridorExcess(points, trimmed, closed, tolerance);
      triedCounts.push({ ctrlPts: n + 1, deviation: dev, corridorExcess: excess });
      if (!(Number.isFinite(dev) && dev <= tolerance)) continue;
      const length = sampledLength(trimmed);
      const cand = { curve: trimmed, maxDeviation: dev, length };
      if (excess > 0) {
        if (!wandering || length < wandering.length) wandering = cand;
        continue;
      }
      if (!best || length < best.length) best = cand;
    }
    if (best) {
      return { ok: true, kind: 'nurbs', curve: best.curve, maxDeviation: best.maxDeviation, ctrlPtCount: best.curve.ctrlPts.length, triedCounts };
    }
    if (n >= nMax) break;
    n = Math.min(nMax, Math.max(n + 1, Math.ceil(n * 1.6)));
  }
  // Last resort: interpolate. Least-squares is capped at n < m for
  // conditioning, so a short, coarsely sampled chain (e.g. 6-9 points spanning
  // ~300 units and turning 30 degrees a step, where six control points cannot
  // follow seven samples) can never reach the curve that passes through its
  // points. Interpolation is exactly determined and uses the averaging knot
  // vector, so it is well-conditioned where the least-squares normal equations
  // are not, and it is where P&T's bounded approximation converges anyway.
  //
  // Its deviation at the samples is zero by construction, so this branch
  // proves nothing about the curve between samples — which is unknowable from
  // the samples alone, for any method. It is reported as its own kind so a
  // caller can tell a fit from a curve that threaded the points.
  //
  // It is bounded to short chains. Interpolation always scores zero at the
  // samples, so an unbounded fallback would certify any bound on any data —
  // including 60 points of noise at 1e-9 — and the tolerance would stop being
  // a claim. A long chain that least-squares cannot fit says something about
  // the data, and threading it is not an answer.
  //
  // A short chain can still overshoot. The count bounds how much oscillation
  // can be spent; it does not bound an excursion, and the spacing does. Four
  // points off a letter's stem — three a fraction of a unit apart and the
  // fourth seventeen units away — interpolate under chord-length
  // parametrization into a curve 2.10x the length of its data, swinging
  // fourteen units clear of a nearly straight run. So this branch builds both
  // parametrizations too, and answers with the shorter.
  const INTERP_MAX_POINTS = 12;
  const interpolants = [];
  let wanderingInterp = null;
  if (points.length <= INTERP_MAX_POINTS) {
    for (const paramsOf of PARAMETRISATIONS) {
      let interp = null;
      try {
        /* The open and closed interpolations return different shapes: the
           open one returns a curve; the closed one returns
           `{ crv, uStart, uEnd }` — a periodic curve plus the sub-domain that
           is the closed loop. Normalized here as `conform.mjs` normalizes the
           same call, so the two consumers agree on what it returns. */
        const raw = closed ? interpolateClosedWith(points, p, paramsOf) : interpolateWith(points, p, paramsOf);
        interp = (closed && raw && raw.crv)
          ? rescaleCurveDomain(extractSubCurve(raw.crv, raw.uStart, raw.uEnd), 0, 1)
          : raw;
      } catch { continue; /* coincident points — falls through to the refusal below, which says more */ }
      if (!interp || !interp.ctrlPts) continue;
      let dev;
      try { dev = maxDeviationFromCurve(points, interp, { closed }); } catch { continue; }
      const excess = corridorExcess(points, interp, closed, tolerance);
      triedCounts.push({ ctrlPts: interp.ctrlPts.length, deviation: dev, corridorExcess: excess, interpolated: true });
      if (!(Number.isFinite(dev) && dev <= tolerance)) continue;
      const cand = { curve: interp, maxDeviation: dev, length: sampledLength(interp) };
      if (excess > 0) { if (!wanderingInterp || cand.length < wanderingInterp.length) wanderingInterp = cand; continue; }
      interpolants.push(cand);
    }
  }
  if (interpolants.length) {
    let best = interpolants[0];
    for (const cand of interpolants) if (cand.length < best.length) best = cand;
    return { ok: true, kind: 'interpolated', curve: best.curve, maxDeviation: best.maxDeviation, ctrlPtCount: best.curve.ctrlPts.length, triedCounts };
  }
  // Nothing stayed inside the corridor, so the answer is the shortest curve
  // that did meet the caller's bound rather than a refusal.
  //
  // The corridor steers the search; it is not a second tolerance. It is built
  // on a three-point circle, a second-order reading of data that may be
  // sampled too coarsely for second order to hold: a seven-point chain
  // spanning a hundred units and turning thirty degrees a step has a
  // legitimate interpolation sitting 2.36 outside its own corridor, and
  // refusing it would reject the case the interpolation fallback exists to
  // serve. So a candidate that leaves the corridor is deprioritized, never
  // rejected.
  const strayed = wandering && wanderingInterp
    ? (wandering.length <= wanderingInterp.length ? { c: wandering, k: 'nurbs' } : { c: wanderingInterp, k: 'interpolated' })
    : wandering ? { c: wandering, k: 'nurbs' } : wanderingInterp ? { c: wanderingInterp, k: 'interpolated' } : null;
  if (strayed) {
    return { ok: true, kind: strayed.k, curve: strayed.c.curve, maxDeviation: strayed.c.maxDeviation, ctrlPtCount: strayed.c.curve.ctrlPts.length, triedCounts };
  }

  // A bound below the samples' own accuracy cannot be met by any curve, and
  // the refusal says so. These points came from a process — a marched
  // intersection, a projected boundary — and asking a fit to sit closer to
  // them than that process was accurate asks it to reproduce the noise.
  const best = triedCounts.length ? Math.min(...triedCounts.map((t) => t.deviation)) : null;
  return {
    ok: false,
    reason: `no curve within ${tolerance} fits these ${points.length} points`
      + (best != null ? ` — closest was ${best.toExponential(3)} at up to ${nMax + 1} control points` : '')
      + (singularAt != null ? `, and the system became singular at ${singularAt}` : '')
      + (points.length > 12 ? ' (too long to fall back on interpolation)' : '')
      + `. If the bound is below the accuracy of whatever produced these points, no curve can meet it.`,
    bestDeviation: best,
    triedCounts,
  };
}
