// FIT A NURBS CURVE THROUGH SAMPLED POINTS, TO A STATED TOLERANCE
// ================================================================
// Piegl & Tiller §9.4.1, "Global Curve Approximation to within a Bound"
// (Algorithm A9.6's shape: least-squares with a control-point count raised
// until the measured deviation clears the bound), with the endpoints
// interpolated exactly (P&T Eq. 9.63-9.67).
//
// WHY THIS EXISTS. This kernel's boolean produces its cut curves by MARCHING
// — an SSI component arrives as a few hundred sampled points, and every trim
// boundary downstream of it is a POLYLINE of those samples. That is fine for
// tessellating and for classifying, and it is not fine as geometry: a
// half-edge here reserves a `pcurve` slot and an edge reserves `curve3d` +
// `tolerance`, which map one-to-one onto a B-rep trim and its edge, and both
// sit empty. Until something fills them, the exact form of a boolean's own
// boundary does not exist anywhere in this kernel — measured on the banked
// torus pair, a union carries 526 edges and not one of them has a curve.
//
// ⚠ THIS DOES NOT REPLACE THE POLYLINE. The polyline is what trims, sews and
// tessellates; a fitted curve is the EXACT RECORD ALONGSIDE it, which is
// exactly what the nullable `curve3d` slot was reserved for. Swapping the
// working representation would put a fit in the path of every operation that
// currently succeeds.
//
// ⚠ AND A FIT IS NOT A MEASUREMENT OF ITSELF. `fitCircle` reports its own
// residual, and a least-squares solve reports nothing at all; neither answers
// "how far is the worst input point from the curve I am about to return".
// So every path here measures the returned curve against the ORIGINAL points
// by the same conservative rule (see `maxDeviationFromCurve`) and refuses
// rather than returning a curve that misses its bound.
import { findSpan, basisFuns } from './basis.mjs';
import { chordLengthParams, solveLinearSystem, averagingKnotVector, interpAtParams } from './interpolate.mjs';
import { curvePoint, reverseCurve } from './curve.mjs';
import { fitLine, fitCircle, fitEllipse } from './refit.mjs';
import { extractSubCurve, rescaleCurveDomain } from './knots.mjs';
import { makeLine, makeCircle, makeEllipse } from './primitives.mjs';

// DEVIATION, MEASURED SO IT CAN ONLY OVER-REPORT, AND WITHOUT A RESOLUTION
// FLOOR. Two stages, and the second is what makes the first usable:
//   1. BRACKET on a sampled polyline — which segment is nearest.
//   2. REFINE by ternary search on |C(t) - Q| over that bracket, evaluating
//      the CURVE, so the answer is the distance to a real point on the curve.
//
// ⚠ STAGE 2 IS NOT OPTIONAL, and the reason is the same inscribed-polygon
// effect this project has now paid for twice. A polyline sampled through a
// curve is INSIDE it, so distances measured to that polyline are too large by
// the sagitta of the sampling — on a radius-25 circle at 960 segments that is
// 1.3e-4, which is a floor no amount of asking for 1e-6 can get under. A
// measure whose own discretisation dominates the quantity it reports cannot
// certify a tolerance at all; it just reports its own step size.
//
// The conservative property survives refinement for free: whatever t the
// search lands on, |C(t) - Q| is the distance to an ACTUAL point of the
// curve, so it is never less than the true minimum distance. The search
// makes the bound tighter, never optimistic.
// ⚠⚠ AND IT IS ONE-SIDED ON PURPOSE — samples to curve, never curve back to
// the input polyline. The two-sided (Hausdorff) version was tried and is
// WRONG for this job: the polyline is INSCRIBED in the shape its samples came
// from, so a perfectly smooth curve is necessarily about one sagitta away from
// those chords. An exact circle through 120 samples of radius 25 scores
// 8.6e-3 against itself that way — the measure punishing the fit for being
// smoother than the data, which is the entire thing it was asked to be. Third
// appearance of the same inscribed-polygon effect in this project: once in the
// geometry, once in the ruler, once in the reference.
// The real risk that motivated it — a curve threading its samples and swinging
// between them — is real, and it is answered by `corridorExcess` below, which
// measures the other direction with the sagitta the data itself implies
// SUBTRACTED. That is the correction the plain Hausdorff version lacks, and it
// is what lets the second measurement exist without rejecting good fits.
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

// HOW FAR THE CURVE MAY WANDER BETWEEN TWO SAMPLES, WITHOUT INVENTING A
// CONSTANT TO SAY SO.
//
// `maxDeviationFromCurve` is one-sided by necessity and its own note explains
// why: the data polyline is INSCRIBED in the shape it was sampled from, so a
// two-sided (Hausdorff) measure punishes a fit for being smoother than its
// chords, by about one sagitta of the sampling. That objection is correct, and
// it is an objection to an UNCORRECTED two-sided measure — not to measuring the
// other direction at all. The sagitta it names is a property of the DATA, and
// the data can be asked for it.
//
// So: for each chord Q_i..Q_{i+1}, take the circle through it and each of its
// two neighbouring points, and keep the larger of the two sagittas that circle
// cuts over that chord. That is the ordinary second-order reconstruction of
// what the samples imply happens between them — zero where three points are
// collinear, and exactly the shape's own bulge where they are not. The curve is
// then allowed to sit `tolerance + h_i` from chord i and no further.
//
// Both halves of that bound are given: `tolerance` is the caller's, and `h_i`
// is measured off the caller's own points. Nothing here is chosen.
//
// ⚠ WHAT THIS CATCHES IS THE ONE THING THE DEVIATION CANNOT. An adaptive
// sampler puts its points far apart where the shape is straight, so the longest
// chords are the ones with no sample in the middle to hold the curve down — and
// a fit that swings out there scores a perfect deviation while drawing a
// letter's straight stem as a banana.
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
// is inside. Sampled on the CURVE, because the excursion is a fact about the
// curve at parameters no data point owns.
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

// THE SPACING OF THE SAMPLES IS NOT THE SHAPE OF THE SAMPLES, and a fit has
// to decide which of the two it believes. Chord-length parametrisation (P&T
// Eq. 9.5) spends parameter in proportion to distance. That is right when the
// points are evenly spread and wrong when they are not — and the samplers this
// kernel actually has are ADAPTIVE, which means they put their points far
// apart exactly where the shape is straight. A marched intersection takes long
// steps through low curvature; a Douglas-Peucker simplification deletes every
// interior point of a straight run. Either way one leg of the data can carry
// most of the parameter domain while all the shape change is crowded into the
// rest, so the curve has to turn inside a sliver of parameter — which it can
// only do by throwing a control point a long way out, and the bulge that
// leaves lands on the long straight leg, where there is no sample to object.
//
// CENTRIPETAL parametrisation (P&T Eq. 9.6, after Lee, CAGD 6(2), 1989) takes
// the SQUARE ROOT of each chord. It is the standard answer to exactly this
// case: it damps the ratio between the longest and the shortest leg without
// throwing the spacing information away, which is what uniform parametrisation
// would do.
function centripetalParams(points) {
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

// WHICH PARAMETRISATION SUITS THIS DATA IS A QUESTION ABOUT THIS DATA, so it
// is measured rather than declared. Every fit below is built BOTH ways against
// the same bound, and of the candidates that MEET the bound the SHORTER curve
// is returned.
//
// ⚠ THAT IS A COMPARISON, NOT A THRESHOLD. No length is ever tested against a
// constant; nothing here has a number in it to tune. Two curves that each sit
// within `tolerance` of every sample agree with the data equally well, and
// differ only in what they do BETWEEN the samples — which the samples cannot
// settle, for either curve. The shorter of the two is the one that added no
// shape the data never asked for, which is the ordinary fairness argument for
// preferring the lower-energy curve when the residual does not separate them.
//
// It is also the one comparison that can see the failure `maxDeviationFromCurve`
// structurally cannot. Deviation is one-sided, sample to curve, and it has to
// be (see its own note); an excursion between two samples moves no sample and
// therefore costs it nothing. An excursion always costs ARC LENGTH.
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

// The two are ordered so that chord-length is asked first and therefore wins
// an exact tie: it is P&T's own default and the one every other fit and loft
// in this kernel parametrises with, so a change of answer is always a change
// this comparison actually paid for.
const PARAMETRISATIONS = [chordLengthParams, centripetalParams];

// A9.1's three ingredients assembled against a STATED parametrisation rather
// than a derived one — which is what `interpAtParams` was factored out of
// `globalCurveInterp` to allow. With `chordLengthParams` this is
// `globalCurveInterp` exactly; the point of writing it out is that the closed
// and open interpolations below can then be run under either parametrisation
// through one code path instead of two that could drift apart.
function interpolateWith(points, requestedDegree, paramsOf) {
  const n = points.length - 1;
  if (n < 1) throw new Error('interpolation needs at least 2 points');
  const p = Math.min(requestedDegree, n);
  const ubar = paramsOf(points);
  const knots = averagingKnotVector(ubar, p);
  return { degree: p, knots, ctrlPts: interpAtParams(points, p, ubar, knots), paramsUsed: ubar };
}

// The closed counterpart, `closedCurveInterp`'s own construction: wrap `k`
// points cyclically off each end, interpolate the padded sequence openly, and
// keep the middle sub-range, so the clamped-end artifacts land in the padding
// and the seam carries a real tangent instead of a kink.
function interpolateClosedWith(points, k, paramsOf) {
  const n = points.length;
  if (n < 3) throw new Error('closed interpolation needs at least 3 points');
  const extended = [];
  for (let i = 0; i < n + 2 * k; i++) extended.push(points[(((i - k) % n) + n) % n]);
  const crv = interpolateWith(extended, k, paramsOf);
  return { crv, uStart: crv.paramsUsed[k], uEnd: crv.paramsUsed[k + n] };
}

// THE APPROXIMATION KNOT VECTOR IS NOT THE INTERPOLATION ONE. P&T Eq. 9.68-
// 9.69: with n+1 control points spread over m+1 points, interior knots are
// taken by averaging the parameters at evenly spaced FRACTIONAL positions
// through the parameter list, so each knot span ends up covering a similar
// number of samples. `averagingKnotVector` (interpolate.mjs) solves the
// different problem where those counts are equal, and produces a singular
// system here.
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

// P&T Eq. 9.63-9.67. The two end control points are FIXED to the first and
// last input point rather than solved for, so a fitted boundary still meets
// its neighbours exactly at the corners the topology already agreed on —
// which matters more here than a marginally lower residual, because a gap at
// a shared corner is a naked edge.
function leastSquaresFit(points, p, n, ubar) {
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

// A CLOSED LOOP IS FITTED BY WRAPPING, the same device closedCurveInterp
// already uses and for the same reason: an open fit's clamped end behaviour
// lands in padding that is then discarded, so the kept range behaves as if
// the curve continues periodically instead of showing a kink at the seam.
// Genuinely periodic B-spline approximation (P&T §9.4.2) is machinery this
// kernel does not have, and this reaches the same practical result.
function wrapClosed(points, pad) {
  const n = points.length;
  const out = [];
  for (let i = n - pad; i < n; i++) out.push(points[i]);
  for (const q of points) out.push(q);
  for (let i = 0; i <= pad; i++) out.push(points[i % n]);
  return out;
}

// LEAST-SQUARES DOES NOT INTERPOLATE THE SEAM, so trimming the wrap padding
// leaves a loop whose two ends are near each other rather than at each other —
// measured on a 24-point closed superellipse of perimeter 72.30, a residual
// 8.90e-2, which is 0.29% of the bbox diagonal and 18% of the fit tolerance
// that produced it. That is small and it is not zero, and `isCurveClosed`
// answers a yes/no question at 1e-6: a nominally open loop makes a renderer
// draw a closing chord, and it makes every downstream extrude, offset and
// trim treat a closed profile as an open one.
//
// THE TWO ENDS ARE MOVED TO THEIR MIDPOINT, NOT TO THE DATA POINT AT THE
// SEAM. Both are the same size of movement (measured on that superellipse:
// arc length ends at 1.00129x the data perimeter snapping to the midpoint,
// 1.00114x snapping to the sample, against 1.00252x with no snap at all), so
// the choice is not about magnitude. It is about what the seam then MEANS.
// Snapping to the sample would make that one point the only exactly
// interpolated sample on an otherwise least-squares loop — the seam becomes
// special, biased onto a single possibly-noisy measurement, which is the
// exact asymmetry `wrapClosed` exists to remove. The midpoint of the two
// computed ends is still a least-squares answer, so the seam keeps being an
// ordinary point of the loop.
//
// ⚠ THE MOVE HAPPENS BEFORE THE DEVIATION IS MEASURED, deliberately. The
// caller's tolerance is a claim about the curve that is RETURNED, so a
// snapped curve that no longer meets its bound must fail the test and let the
// loop raise the control point count — which shortens the gap it then has to
// close. Snapping after the measurement would certify a curve nobody measured.
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

// EXACT WHERE THE SHAPE IS ACTUALLY EXACT. A plane cutting a cylinder or a
// sphere gives a genuine circle or ellipse, and the commonest booleans a
// student runs are exactly those. Recognising one and emitting the RATIONAL
// primitive is not an optimisation — it is the difference between a boundary
// Rhino re-reads as a circle and one it re-reads as a spline that happens to
// look round. The recogniser's own residual is not trusted for this: the
// primitive is built and then measured against the original points like any
// other candidate.
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
    // ⚠ A PRIMITIVE DOES NOT INTERPOLATE ITS ENDPOINTS. fitLine returns the
    // input projected ONTO the fitted line, so an open fit can move the first
    // and last points by the residual — harmless for a display curve and fatal
    // for a TRIM, whose ends must meet its neighbours exactly. Measured as
    // u = -0.000754 on a pcurve that should have started at 0: outside the
    // domain, and OpenNURBS rejects the loop for not joining. So a primitive
    // that moves an endpoint is rejected here and the least-squares path takes
    // over, which fixes its endpoints by construction.
    if (!closed && opts.exactEndpoints) {
      const q0 = points[0], qn = points[points.length - 1];
      const ends = (crv) => {
        const cp = crv.ctrlPts;
        const a = cp[0], b = cp[cp.length - 1];
        return Math.hypot(a[0] - q0[0], a[1] - q0[1], a[2] - q0[2])
          + Math.hypot(b[0] - qn[0], b[1] - qn[1], b[2] - qn[2]);
      };
      // ⚠ A PRIMITIVE CAN COME BACK RUNNING THE OTHER WAY, and rejecting it for
      // that discards an exact answer over a convention. `fitLine`
      // CANONICALIZES its direction (largest component positive) so that
      // near-identical input cannot flicker between opposite directions — a
      // property worth having, and one that means a run travelling in -x
      // returns start and end swapped. Measured: two of a square's four sides,
      // every one of them perfectly straight, fell through to a 6-control-point
      // least-squares spline purely because of which way the loop happened to
      // run around them. Reversal is exact and its own inverse, so the fix is
      // to turn the candidate round rather than to give up on it.
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

// THE ENTRY POINT.
//   points     — ordered [x,y,z]; for a closed loop do NOT repeat the first
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

  // RAISE THE COUNT UNTIL IT CLEARS, rather than guessing one. Growth is
  // geometric so a curve needing many spans is reached in a few solves, and
  // the ceiling is m (at which point the fit has as many freedoms as points
  // and any remaining error is the parametrisation, not the count).
  // ⚠ THE CEILING IS n < m, NOT n <= m, and it is a conditioning limit rather
  // than a formality. As the free control points approach the interior point
  // count the normal equations lose rank — neighbouring samples share a
  // parameter to within the solver's pivot threshold — and the solve throws.
  // Stopping one short keeps every refusal a statement about the DATA instead
  // of about the matrix.
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
      /* ⚠⚠ THE PADDING HAS TO BE CUT OFF AGAIN, AND FOR A LONG TIME IT WAS NOT.
         `wrapClosed` prepends `p` points and appends `p+1` so the fit's clamped
         ends land outside the loop; the comment on it says the padding "is then
         discarded" and this branch never discarded it. The returned curve
         therefore ran past its own start and RETRACED — measured on a drawn
         letter O, back at its starting point at 84% of its domain and still
         going, ending 40% of the glyph's diagonal away, with three
         self-crossings. A closed superellipse of perimeter 72.358 came back at
         arc length 91.734: 27% too long.
         ⚠ AND THE TOLERANCE COULD NOT SEE IT. Deviation is measured from the
         DATA to the curve, and every point of a retrace is still on the shape —
         so the fit reported 2.46e-1 against a tolerance of 0.5 and called itself
         good. The interpolation branch below already did this trim; this one
         never did, which is why round glyphs broke and cornered ones did not. */
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
  // LAST RESORT: INTERPOLATE. Least-squares is capped at n < m for
  // conditioning, so a SHORT, COARSELY SAMPLED chain can never reach the one
  // curve that certainly passes through its points — measured on the torus
  // pair, nine chains of 6-9 points spanning ~300 units and turning 30 degrees
  // a step, where six control points cannot follow seven samples. Interpolation
  // is exactly determined and uses the averaging knot vector, so it is
  // well-conditioned precisely where the least-squares normal equations are
  // not, and it is where P&T's own bounded approximation converges anyway.
  //
  // ⚠ ITS DEVIATION AT THE SAMPLES IS ZERO BY CONSTRUCTION, so this branch
  // buys no reduction and proves nothing about the curve BETWEEN samples —
  // which is unknowable from the samples alone, for any method. It is
  // reported as its own kind so a caller can tell a genuine fit from a curve
  // that simply threaded the points.
  //
  // ⚠⚠ AND IT IS BOUNDED TO SHORT CHAINS, which is the guard that makes the
  // whole tolerance mean something again. Interpolation always scores zero at
  // the samples, so an unbounded fallback would certify ANY bound on ANY data
  // — including 60 points of noise at 1e-9 — and the tolerance would stop
  // being a claim. A handful of points spanning a long chord is the case that
  // genuinely needs it; a long chain that least-squares cannot fit is telling
  // you something about the data, and threading it is not an answer.
  //
  // ⚠⚠⚠ BUT A SHORT CHAIN IS NOT A SAFE ONE. The count bounds how much
  // oscillation can be spent; it does not bound an EXCURSION, and the spacing
  // does. Four points off a letter's stem — three of them a fraction of a unit
  // apart and the fourth seventeen units away — interpolate under chord-length
  // parametrisation into a curve 2.10x the length of its own data, swinging
  // fourteen units clear of a run that is very nearly straight. So this branch
  // builds both parametrisations too, and answers with the shorter.
  const INTERP_MAX_POINTS = 12;
  const interpolants = [];
  let wanderingInterp = null;
  if (points.length <= INTERP_MAX_POINTS) {
    for (const paramsOf of PARAMETRISATIONS) {
      let interp = null;
      try {
        /* ⚠⚠ THE OPEN AND CLOSED INTERPOLATIONS DO NOT RETURN THE SAME SHAPE,
           and the guard below is what made that matter. The open one returns a
           curve; the closed one returns `{ crv, uStart, uEnd }` — a PERIODIC
           curve plus the sub-domain that is the closed loop — so
           `interp.ctrlPts` was undefined on the closed branch and the guard was
           always false. The closed interpolation therefore never once fired:
           every closed contour silently fell through to the least-squares path
           above. Normalised here exactly as `conform.mjs` normalises the same
           call, so the two consumers cannot disagree about what it returns. */
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
  // NOTHING STAYED INSIDE THE CORRIDOR — so the corridor has run out of things
  // to steer towards, and the answer is the shortest curve that DID meet the
  // caller's bound rather than a refusal.
  //
  // ⚠ THE CORRIDOR STEERS THE SEARCH; IT IS NOT A SECOND TOLERANCE. It is built
  // on a three-point circle, which is a second-order reading of data that may be
  // sampled far too coarsely for second order to hold: a seven-point chain
  // spanning a hundred units and turning thirty degrees a step has a legitimate
  // interpolation sitting 2.36 outside its own corridor, and refusing that would
  // reject the exact case the interpolation fallback exists to serve. So a
  // candidate that leaves the corridor is deprioritised, never rejected.
  const strayed = wandering && wanderingInterp
    ? (wandering.length <= wanderingInterp.length ? { c: wandering, k: 'nurbs' } : { c: wanderingInterp, k: 'interpolated' })
    : wandering ? { c: wandering, k: 'nurbs' } : wanderingInterp ? { c: wanderingInterp, k: 'interpolated' } : null;
  if (strayed) {
    return { ok: true, kind: strayed.k, curve: strayed.c.curve, maxDeviation: strayed.c.maxDeviation, ctrlPtCount: strayed.c.curve.ctrlPts.length, triedCounts };
  }

  // A BOUND BELOW THE SAMPLES' OWN ACCURACY CANNOT BE MET BY ANY CURVE, and
  // saying so is the useful half of the refusal. These points came from
  // somewhere — a marched intersection, a projected boundary — and asking a
  // fit to sit closer to them than that process was accurate is asking it to
  // reproduce the noise, which costs control points without buying fidelity.
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
