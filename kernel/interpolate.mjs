// Global Curve Interpolation — Piegl & Tiller Ch. 9.2.1, Algorithm A9.1.
// Given n+1 through-points, find a degree-p non-rational B-spline curve
// that passes through every one of them exactly (SketchCurve/InterpCrv —
// distinct from Curve, whose control points the curve does not touch).
//
// Three ingredients, each a distinct P&T technique:
//  1. Chord-length parameters ubar_k (Eq 9.5) — one parameter per point,
//     spaced by how far apart the points are, not uniformly; this keeps the
//     curve from bulging near closely spaced points.
//  2. The averaging knot vector (Eq 9.8) — built from the parameters, which
//     guarantees the resulting coefficient matrix is nonsingular and banded,
//     unlike a uniform knot vector.
//  3. A linear solve: the interpolation condition C(ubar_k) = Q_k for every
//     k is exactly (n+1) linear equations in the (n+1) unknown control
//     points, same coefficient matrix for all three of x/y/z — solved
//     once per coordinate via Gauss-Jordan elimination.

import { findSpan, basisFuns, dersBasisFuns } from './basis.mjs';

// Eq 9.5 — chord-length parametrization. Falls back to uniform spacing
// only in the degenerate all-coincident-points case (zero total chord
// length), so a caller never divides by zero.
export function chordLengthParams(points) {
  const n = points.length - 1;
  const ubar = new Array(points.length).fill(0);
  ubar[n] = 1;
  if (n === 0) return ubar;
  const chords = [];
  let total = 0;
  for (let k = 1; k <= n; k++) {
    const [ax, ay, az] = points[k - 1], [bx, by, bz] = points[k];
    const d = Math.hypot(bx - ax, by - ay, bz - az);
    chords.push(d);
    total += d;
  }
  if (total < 1e-12) {
    for (let k = 1; k < n; k++) ubar[k] = k / n;
    return ubar;
  }
  let acc = 0;
  for (let k = 1; k < n; k++) { acc += chords[k - 1]; ubar[k] = acc / total; }
  return ubar;
}

// Eq 9.8 — the averaging knot vector: each interior knot is the mean of p
// consecutive parameters, so every ubar_k lies under p+1 nonzero basis
// functions (Schoenberg-Whitney, satisfied by construction this way).
export function averagingKnotVector(ubar, p) {
  const n = ubar.length - 1;
  const m = n + p + 1;
  const U = new Array(m + 1).fill(0);
  for (let i = m - p; i <= m; i++) U[i] = 1;
  for (let j = 1; j <= n - p; j++) {
    let s = 0;
    for (let i = j; i <= j + p - 1; i++) s += ubar[i];
    U[j + p] = s / p;
  }
  return U;
}

// Gauss-Jordan elimination with partial pivoting, one shared coefficient
// matrix A against several right-hand-side columns at once (here: x, y, z)
// — full reduction to identity is fine at the control-point counts a
// sketched curve reaches (tens, not thousands).
export function solveLinearSystem(A, rhsCols) {
  const n = A.length;
  const numB = rhsCols.length;
  const M = A.map((row, i) => [...row, ...rhsCols.map((b) => b[i])]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (piv !== col) { const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp; }
    const pivVal = M[col][col];
    // Partial pivoting picks the largest available magnitude in this column;
    // if that is still ~0, every candidate row is degenerate (typically two
    // through-points close enough that chordLengthParams/averagingKnotVector
    // give them the same parameter, so basisFuns produces two identical
    // rows). Throw a catchable error rather than divide by ~0 and emit
    // Infinity/NaN control points.
    if (Math.abs(pivVal) < 1e-10) {
      throw new Error('solveLinearSystem: matrix is singular — two or more input points coincide (or share a parameter), so no unique curve fits them');
    }
    for (let c = col; c < n + numB; c++) M[col][c] /= pivVal;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (factor === 0) continue;
      for (let c = col; c < n + numB; c++) M[r][c] -= factor * M[col][c];
    }
  }
  const xs = Array.from({ length: numB }, () => new Array(n));
  for (let i = 0; i < n; i++) for (let b = 0; b < numB; b++) xs[b][i] = M[i][n + b];
  return xs;
}

// The shared core of A9.1 — solve for control points given an already chosen
// degree, parameters and knot vector, rather than deriving them from
// `points`. Used by callers that need several interpolations to share one
// parametrization: Loft's global surface interpolation (kernel/loft.mjs, P&T
// 9.2.5) requires every row of a lofted net to use one common knot vector per
// direction, or the result is not a tensor-product surface.
export function interpAtParams(points, degree, ubar, knots) {
  const n = points.length - 1;
  const p = Math.min(degree, n);
  const size = n + 1;
  const A = Array.from({ length: size }, () => new Array(size).fill(0));
  for (let k = 0; k <= n; k++) {
    const span = findSpan(n, p, ubar[k], knots);
    const N = basisFuns(span, ubar[k], p, knots);
    for (let i = 0; i <= p; i++) A[k][span - p + i] = N[i];
  }
  const Bx = points.map((pt) => pt[0]);
  const By = points.map((pt) => pt[1]);
  const Bz = points.map((pt) => pt[2]);
  const [Px, Py, Pz] = solveLinearSystem(A, [Bx, By, Bz]);
  return Px.map((x, i) => [x, Py[i], Pz[i], 1]);
}

// A9.1 itself. `requestedDegree` is clamped down to n (point count - 1)
// when there are not enough points for it, so a 2-point SketchCurve is a
// degree-1 line rather than an error.
export function globalCurveInterp(points, requestedDegree = 3) {
  const n = points.length - 1;
  if (n < 1) throw new Error('globalCurveInterp needs at least 2 points');
  const p = Math.min(requestedDegree, n);
  const ubar = chordLengthParams(points);
  const U = averagingKnotVector(ubar, p);
  const ctrlPts = interpAtParams(points, p, ubar, U);
  return { degree: p, knots: U, ctrlPts, paramsUsed: ubar };
}

// P&T 9.2.2 — Global Curve Interpolation with End Derivatives Specified,
// specialized to cubic (degree 3) only. Adding 2 control points beyond the
// ordinary n+1 (to carry the 2 derivative constraints) needs n-p+2 interior
// knots; for p=3 that is exactly n-1, the number of interior chord-length
// parameters (ubar_1..ubar_{n-1}), so the interior knots are assigned directly
// from those parameters with no averaging formula. The count match is specific
// to p=3; other degrees need a different interior-knot rule, which is not
// implemented, so there is no `requestedDegree` parameter.
//
// Given n+1 points and two derivative vectors D0/Dn (magnitude matters — a
// longer vector produces a more pronounced bulge near that end, as with a
// Bezier/Hermite tangent handle), this builds a cubic curve with n+3 control
// points that interpolates every point exactly and matches C'(ubar_0)=D0,
// C'(ubar_n)=Dn exactly. The linear system is built as for ordinary global
// interpolation: one row per constraint, basis values from basisFuns for a
// position row, basis derivative values from dersBasisFuns (A2.2/A2.3) for a
// derivative row.
export function globalCurveInterpWithEndDerivs(points, D0, Dn) {
  const n = points.length - 1;
  if (n < 3) throw new Error('globalCurveInterpWithEndDerivs needs at least 4 points (cubic + 2 end derivatives)');
  const p = 3;
  const ubar = chordLengthParams(points);
  // Knot vector: clamped p+1 at each end; interior knots are the interior
  // chord-length parameters themselves — see the header for why this count
  // matches exactly for degree 3 with 2 extra control points.
  const U = [];
  for (let i = 0; i <= p; i++) U.push(ubar[0]);
  for (let i = 1; i <= n - 1; i++) U.push(ubar[i]);
  for (let i = 0; i <= p; i++) U.push(ubar[n]);
  const N = n + 3; // control point count (2 more than ordinary interpolation's n+1)
  const A = Array.from({ length: N }, () => new Array(N).fill(0));
  const bx = new Array(N).fill(0), by = new Array(N).fill(0), bz = new Array(N).fill(0);
  // Row 0 — start derivative constraint.
  {
    const span = findSpan(N - 1, p, ubar[0], U);
    const ders = dersBasisFuns(span, ubar[0], p, 1, U);
    for (let i = 0; i <= p; i++) A[0][span - p + i] = ders[1][i];
    bx[0] = D0[0]; by[0] = D0[1]; bz[0] = D0[2];
  }
  // Rows 1..n+1 — one position constraint per input point.
  for (let k = 0; k <= n; k++) {
    const span = findSpan(N - 1, p, ubar[k], U);
    const Nb = basisFuns(span, ubar[k], p, U);
    for (let i = 0; i <= p; i++) A[k + 1][span - p + i] = Nb[i];
    bx[k + 1] = points[k][0]; by[k + 1] = points[k][1]; bz[k + 1] = points[k][2];
  }
  // Row n+2 — end derivative constraint.
  {
    const span = findSpan(N - 1, p, ubar[n], U);
    const ders = dersBasisFuns(span, ubar[n], p, 1, U);
    for (let i = 0; i <= p; i++) A[n + 2][span - p + i] = ders[1][i];
    bx[n + 2] = Dn[0]; by[n + 2] = Dn[1]; bz[n + 2] = Dn[2];
  }
  const [Px, Py, Pz] = solveLinearSystem(A, [bx, by, bz]);
  const ctrlPts = Px.map((x, i) => [x, Py[i], Pz[i], 1]);
  return { degree: p, knots: U, ctrlPts, paramsUsed: ubar };
}

// Cubic Hermite segment — for tangent handles at interior through-points.
// globalCurveInterpWithEndDerivs solves one global curve through all points;
// adding interior tangent constraints to that system (P&T 9.2.4) is not
// implemented. This is the simpler closed-form alternative: the cubic
// Hermite-to-Bezier control-point conversion for a single 2-point segment with
// two independent end tangents, with no linear solve — B0=p0, B1=p0+m0/3,
// B2=p1-m1/3, B3=p1 (differentiate the degree-3 Bezier basis: C(0)=B0,
// C(1)=B3, C'(0)=3(B1-B0)=m0, C'(1)=3(B3-B2)=m1, exact by construction).
// A PolyCurve chain of these segments gives a per-interior-point tangent
// handle: each interior point's tangent feeds the end derivative of the
// segment before it and the start derivative of the segment after it, giving
// G1 continuity by construction without solving a global system. The app-layer
// wiring (PolyCurve conversion, handle UI) is not part of this kernel.
export function cubicHermiteSegment(p0, p1, m0, m1) {
  const b0 = p0;
  const b1 = [p0[0] + m0[0] / 3, p0[1] + m0[1] / 3, p0[2] + m0[2] / 3];
  const b2 = [p1[0] - m1[0] / 3, p1[1] - m1[1] / 3, p1[2] - m1[2] / 3];
  const b3 = p1;
  return {
    degree: 3,
    knots: [0, 0, 0, 0, 1, 1, 1, 1],
    ctrlPts: [[b0[0], b0[1], b0[2], 1], [b1[0], b1[1], b1[2], 1], [b2[0], b2[1], b2[2], 1], [b3[0], b3[1], b3[2], 1]],
  };
}

// Catmull-Rom (uniform) tangent — the standard default tangent at an interior
// point of a piecewise curve: the average of the two chord vectors on either
// side (equivalently, half the chord from the previous point to the next).
// This is the smooth-anchor default an interior handle shows before it is
// dragged: symmetric, antiparallel handles on either side, independently
// draggable afterward.
export function catmullRomTangent(pPrev, pNext) {
  return [(pNext[0] - pPrev[0]) / 2, (pNext[1] - pPrev[1]) / 2, (pNext[2] - pPrev[2]) / 2];
}

// Closed-curve interpolation with tangent continuity at the seam — distinct
// from re-interpolating through a duplicated start point, which gives only
// positional (C0) closure and a kink where the loop shuts.
//
// The textbook answer is periodic B-spline interpolation, which builds
// periodic basis functions; this kernel does not have it. This function
// reaches the same result — a closed loop whose tangent matches across the
// seam — by wrapping `degree` points from each end of the closed point
// sequence cyclically, running the ordinary open A9.1 interpolation on the
// padded sequence, and keeping only the middle sub-range. The clamped-end
// artifacts land in the padding, outside the kept region, so the kept region
// behaves as if it continues periodically. test/interpolate.test.mjs compares
// the tangent approaching the seam to the tangent leaving it.
export function closedCurveInterp(points, requestedDegree = 3) {
  const n = points.length;
  if (n < 3) throw new Error('closedCurveInterp needs at least 3 points');
  const k = requestedDegree;
  const extended = [];
  for (let i = 0; i < n + 2 * k; i++) {
    const j = (((i - k) % n) + n) % n;
    extended.push(points[j]);
  }
  const crv = globalCurveInterp(extended, requestedDegree);
  return { crv, uStart: crv.paramsUsed[k], uEnd: crv.paramsUsed[k + n] };
}
