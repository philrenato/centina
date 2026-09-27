// Knot insertion (P&T Ch.5, Algorithm A5.1, CurveKnotIns) + degree
// elevation (Ch.5, via the Bezier-decomposition equivalent of Algorithm
// A5.9). Two consumers need them: a closed SketchCurve's padded-domain
// curve (kernel/interpolate.mjs's closedCurveInterp) is not a valid
// standalone profile (its shape exists only over a sub-range of its knot
// domain), and a mixed-degree PolyCurve (a filleted Polygon's Line+Arc chain,
// or a planar-arrangement smooth-segment output) cannot become one curve
// until its segments share a degree.
//
// Every function here operates on the NurbsCrv shape the rest of this kernel
// uses ({degree, knots, ctrlPts}, ctrlPts an array of [x,y,z,w] — the point
// plus its weight, never pre-multiplied). Insertion and elevation work in
// homogeneous space internally (the toHomogeneous form curve.mjs's evaluation
// uses), so a rational curve (an Arc/Circle-derived fillet segment) elevates
// and inserts exactly.

import { findSpan } from './basis.mjs';

function toHomogeneousPts(ctrlPts) {
  return ctrlPts.map(([x, y, z, w]) => [x * w, y * w, z * w, w]);
}
function fromHomogeneousPts(Pw) {
  return Pw.map(([xw, yw, zw, w]) => [xw / w, yw / w, zw / w, w]);
}

function knotMultiplicity(knots, u, tol = 1e-9) {
  let s = 0;
  for (const k of knots) if (Math.abs(k - u) < tol) s++;
  return s;
}

// A5.1, r=1 — Boehm's single-knot-insertion formula. Shape-preserving by
// construction: a knot insertion changes only the curve's control-point/knot
// representation, never its geometry. test/knots.test.mjs samples
// before/after.
export function insertKnotOnce(crv, u) {
  const { degree: p, knots: U } = crv;
  const Pw = toHomogeneousPts(crv.ctrlPts);
  const n = Pw.length - 1;
  const k = findSpan(n, p, u, U);
  const s = knotMultiplicity(U, u);
  const UQ = new Array(U.length + 1);
  for (let i = 0; i <= k; i++) UQ[i] = U[i];
  UQ[k + 1] = u;
  for (let i = k + 1; i < U.length; i++) UQ[i + 1] = U[i];
  const Qw = new Array(n + 2);
  for (let i = 0; i <= k - p; i++) Qw[i] = Pw[i];
  for (let i = k - s; i <= n; i++) Qw[i + 1] = Pw[i];
  for (let i = k - p + 1; i <= k - s; i++) {
    const alpha = (u - U[i]) / (U[i + p] - U[i]);
    Qw[i] = [0, 1, 2, 3].map((d) => alpha * Pw[i][d] + (1 - alpha) * Pw[i - 1][d]);
  }
  return { degree: p, knots: UQ, ctrlPts: fromHomogeneousPts(Qw) };
}

// r repeated single insertions of the same knot value — algebraically
// identical to P&T's multi-insertion A5.1 (r>1), built from the r=1 case
// rather than transcribing A5.1's multi-insertion indexing.
export function insertKnot(crv, u, r = 1) {
  let c = crv;
  for (let i = 0; i < r; i++) c = insertKnotOnce(c, u);
  return c;
}

// The knot already in the vector that `u` means, when there is one within `tol`.
function knotValueAt(knots, u, tol = 1e-9) {
  for (const k of knots) if (Math.abs(k - u) < tol) return k;
  return u;
}
/* Snap to the knot that is already there, then insert copies of that value.
   `knotMultiplicity` counts within `tol`, so a `u` a floating-point hair away
   from an existing knot is counted at that knot's multiplicity; inserting the
   copies at `u` itself would leave a span of width ~1e-16 between the two
   values. That span's control points are coincident: speed 0, curvature
   ~7e+25, which every downstream consumer reads (a comb draws a quill to
   infinity, tessellation divides by it, Extrude and Revolve carry it into a
   surface). `extractSubCurve` on a closed interpolation hits this, since its
   [uStart,uEnd] can land ~1e-16 from a knot. Counting and inserting must agree
   about which knot they mean. */
function insertKnotToMultiplicity(crv, u, targetMult, tol = 1e-9) {
  let c = crv;
  const at = knotValueAt(c.knots, u, tol);
  let s = knotMultiplicity(c.knots, at, tol);
  while (s < targetMult) { c = insertKnotOnce(c, at); s++; }
  return c;
}

function distinctInteriorKnotValues(knots, uMin, uMax, tol = 1e-9) {
  const out = [];
  for (const k of knots) {
    if (k <= uMin + tol || k >= uMax - tol) continue;
    if (!out.some((v) => Math.abs(v - k) < tol)) out.push(k);
  }
  return out.sort((a, b) => a - b);
}

// DecomposeCurve (the target of A5.6) — insert every interior knot value up
// to full multiplicity `degree` (not degree+1: an interior Bezier breakpoint
// stays part of the same curve, sharing its one boundary control point with
// its neighbor; only the curve's two ends are p+1/clamped). Returns an ordered
// array of Bezier pieces, each { ctrlPts: [degree+1 [x,y,z,w] points], u0, u1 }.
// A curve with zero interior knots (a single Bezier span — a Line, or one
// smooth SketchCurve span) returns exactly one piece, unchanged.
export function decomposeToBezier(crv) {
  const p = crv.degree;
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const interiorVals = distinctInteriorKnotValues(crv.knots, uMin, uMax);
  let c = crv;
  for (const u of interiorVals) c = insertKnotToMultiplicity(c, u, p);
  const breakpoints = [uMin, ...interiorVals, uMax];
  const pieces = [];
  for (let s = 0; s < breakpoints.length - 1; s++) {
    const start = s * p;
    pieces.push({ ctrlPts: c.ctrlPts.slice(start, start + p + 1).map((pt) => pt.slice()), u0: breakpoints[s], u1: breakpoints[s + 1] });
  }
  return pieces;
}

// Bezier degree elevation by 1 (Farin; P&T closed form:
// Q_i = (i/(p+1))*P_{i-1} + (1-i/(p+1))*P_i, boundary terms dropped rather
// than indexed out of range) — run in homogeneous space so a rational Bezier
// (an Arc/Circle segment) elevates exactly.
function elevateBezierOnce(ctrlPts) {
  const p = ctrlPts.length - 1;
  const Pw = toHomogeneousPts(ctrlPts);
  const out = [];
  for (let i = 0; i <= p + 1; i++) {
    const a = i / (p + 1);
    const prev = i - 1 >= 0 ? Pw[i - 1] : [0, 0, 0, 0];
    const cur = i <= p ? Pw[i] : [0, 0, 0, 0];
    out.push([0, 1, 2, 3].map((d) => a * prev[d] + (1 - a) * cur[d]));
  }
  return fromHomogeneousPts(out);
}

// Reassembles an ordered chain of same-degree Bezier pieces (contiguous
// domains, piece[i].u1 === piece[i+1].u0) into one clamped NurbsCrv — the
// inverse of decomposeToBezier above.
export function assembleBezierChain(pieces, degree) {
  const p = degree;
  const knots = [];
  for (let i = 0; i <= p; i++) knots.push(pieces[0].u0);
  const ctrlPts = pieces[0].ctrlPts.map((pt) => pt.slice());
  for (let s = 1; s < pieces.length; s++) {
    for (let i = 0; i < p; i++) knots.push(pieces[s].u0);
    ctrlPts.push(...pieces[s].ctrlPts.slice(1).map((pt) => pt.slice()));
  }
  for (let i = 0; i <= p; i++) knots.push(pieces[pieces.length - 1].u1);
  return { degree: p, knots, ctrlPts };
}

// Degree elevation (the target of A5.9, reached via the equivalent
// construction rather than A5.9's bezalfs-table pseudocode): decompose to
// Bezier, elevate every piece by the same amount, reassemble at full
// multiplicity everywhere. This is exact (decomposition and per-Bezier
// elevation are each shape-preserving) but not minimal: an interior knot that
// started below full multiplicity (a smooth SketchCurve span's simple knots)
// comes back at full multiplicity `targetDegree` rather than the minimal
// s_orig+t, so the knot vector reports less smoothness than the geometry has.
// The shape is unchanged (test/knots.test.mjs samples before/after at many
// parameters, derivatives included). The only input joinCurvesC0 below
// elevates is a Line/Polyline segment (degree 1, already at full
// multiplicity everywhere — see getProfileCrv's Polyline branch), for which
// the result is minimal anyway.
export function degreeElevateCurve(crv, targetDegree) {
  const p = crv.degree;
  if (targetDegree === p) return { degree: p, knots: crv.knots.slice(), ctrlPts: crv.ctrlPts.map((c) => c.slice()) };
  if (targetDegree < p) throw new Error(`degreeElevateCurve: targetDegree (${targetDegree}) must be >= the curve's own degree (${p})`);
  const t = targetDegree - p;
  const pieces = decomposeToBezier(crv).map((piece) => {
    let ctrlPts = piece.ctrlPts;
    for (let step = 0; step < t; step++) ctrlPts = elevateBezierOnce(ctrlPts);
    return { ...piece, ctrlPts };
  });
  return assembleBezierChain(pieces, targetDegree);
}

// Affine reparametrization of a curve's knot values (control points
// untouched) — shape-preserving, because a B-spline's geometry depends only on
// the relative spacing of its knot vector.
export function rescaleCurveDomain(crv, newMin, newMax) {
  const oldMin = crv.knots[0], oldMax = crv.knots[crv.knots.length - 1];
  const span = oldMax - oldMin;
  const knots = span < 1e-12
    ? crv.knots.map(() => newMin)
    : crv.knots.map((k) => newMin + ((k - oldMin) / span) * (newMax - newMin));
  return { degree: crv.degree, knots, ctrlPts: crv.ctrlPts.map((p) => p.slice()) };
}

// Extract sub-curve — trims a closed SketchCurve's padded domain
// (closedCurveInterp's [uStart,uEnd] sub-range is not a valid curve until the
// domain is trimmed to it): insert both boundary parameters up to full clamped
// multiplicity (degree+1, as at any curve endpoint — not decomposeToBezier's
// interior multiplicity=degree, since these become new ends, not an internal
// C0 breakpoint), then slice the isolated index range. Exact by construction:
// knot insertion does not change the shape, and slicing off a fully clamped,
// non-overlapping index range discards the padding without touching the kept
// part. Tested against the same curve's sampled points over [uStart,uEnd],
// before vs. after extraction.
function firstIndexOfValue(arr, v, tol = 1e-9) {
  for (let i = 0; i < arr.length; i++) if (Math.abs(arr[i] - v) < tol) return i;
  return -1;
}
function lastIndexOfValue(arr, v, tol = 1e-9) {
  for (let i = arr.length - 1; i >= 0; i--) if (Math.abs(arr[i] - v) < tol) return i;
  return -1;
}
export function extractSubCurve(crv, uStart, uEnd) {
  const p = crv.degree;
  let c = insertKnotToMultiplicity(crv, uStart, p + 1);
  c = insertKnotToMultiplicity(c, uEnd, p + 1);
  const i0 = firstIndexOfValue(c.knots, uStart);
  const i1 = lastIndexOfValue(c.knots, uEnd);
  const knots = c.knots.slice(i0, i1 + 1);
  const ctrlPts = c.ctrlPts.slice(i0, i1 - p).map((pt) => pt.slice());
  return { degree: p, knots, ctrlPts };
}

// Do not clamp an unclamped curve with these functions. insertKnotOnce
// implements P&T A5.1, whose precondition is that the inserted knot stays
// within multiplicity <= degree. extractSubCurve goes one further (to degree+1)
// to isolate a sub-range, which is exact for the clamped curves it is written
// for — at multiplicity p the control point being duplicated lies on the
// curve, so duplicating it splits the curve without moving it. On an unclamped
// curve that point is not on the curve, the step duplicates it anyway with no
// blend, and the result is a different curve (0.37 units on a plain uniform
// cubic spanning ~3). decomposeToBezier assumes clamped input too, and returns
// NaN pieces for an unclamped one.
// The exact conversion for the one unclamped shape this kernel produces — a
// uniform bicubic patch — lives in subdlimit.mjs next to the construction it
// undoes, where the knot vector it assumes is defined.

// C0-only concatenation of two same-degree clamped curves whose domains are
// adjacent (A's end === B's start) — drops one of A's (degree+1) trailing
// end-knot copies (leaving `degree` copies, the interior-joint multiplicity
// for a positional-only seam) and skips all of B's (degree+1) leading copies,
// plus drops B's first control point (the shared joint, coincident with A's
// last one — Join's JOIN_TOLERANCE match guarantees that coincidence
// upstream). Each curve's internal knot structure, and therefore its internal
// continuity class, is untouched; only the shared boundary knot's
// multiplicity is affected.
export function concatTwoC0(A, B, degree) {
  const p = degree;
  const knots = A.knots.slice(0, A.knots.length - 1).concat(B.knots.slice(p + 1));
  const ctrlPts = A.ctrlPts.map((pt) => pt.slice()).concat(B.ctrlPts.slice(1).map((pt) => pt.slice()));
  return { degree: p, knots, ctrlPts };
}

// Join curves at C0 — turns a mixed-degree PolyCurve (a filleted Polygon's
// Line+Arc chain) into one curve: every segment is degree-elevated (above) to
// the highest degree present, each rescaled onto its own sequential integer
// domain slot [i,i+1] (rescaleCurveDomain — a simple consistent convention;
// extrude()/revolve() need a well-formed curve, not particular parameter
// values), then chained via concatTwoC0 into one curve. A degree-1 chain
// needs no elevation and reduces to pure concatenation — the same
// knot/control-point shape as getProfileCrv's degree-1 fast path (checked in
// test/knots.test.mjs).
export function joinCurvesC0(curves) {
  if (!curves.length) throw new Error('joinCurvesC0: need at least one curve');
  const targetDegree = Math.max(...curves.map((c) => c.degree));
  const prepped = curves.map((crv, i) => {
    const rescaled = rescaleCurveDomain(crv, i, i + 1);
    return rescaled.degree < targetDegree ? degreeElevateCurve(rescaled, targetDegree) : rescaled;
  });
  return prepped.reduce((acc, c) => (acc ? concatTwoC0(acc, c, targetDegree) : c), null);
}
