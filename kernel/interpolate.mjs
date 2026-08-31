// Global Curve Interpolation — Piegl & Tiller Ch. 9.2.1, Algorithm A9.1.
// Given n+1 through-points, find a degree-p NON-RATIONAL B-spline curve
// that passes through every one of them exactly (SketchCurve/InterpCrv —
// distinct from CURVE, whose control points the curve does NOT touch).
//
// Three ingredients, each a distinct P&T technique:
//  1. Chord-length parameters ubar_k (Eq 9.5) — one parameter per point,
//     spaced by how far apart the points actually are, not uniformly; this
//     is what keeps the curve from bulging near closely-spaced points.
//  2. The averaging knot vector (Eq 9.8) — built FROM the parameters, the
//     standard technique that guarantees the resulting coefficient matrix
//     is nonsingular and banded (every basis function's support is well-
//     behaved), unlike a naive uniform knot vector.
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
// sketched curve actually reaches (tens, not thousands).
export function solveLinearSystem(A, rhsCols) {
  const n = A.length;
  const numB = rhsCols.length;
  const M = A.map((row, i) => [...row, ...rhsCols.map((b) => b[i])]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (piv !== col) { const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp; }
    const pivVal = M[col][col];
    // Partial pivoting above picks the LARGEST available magnitude in this
    // column — if that's still ~0, every candidate row is genuinely
    // degenerate here (the classic cause: two through-points close enough
    // that chordLengthParams/averagingKnotVector give them the identical
    // parameter, so basisFuns produces two byte-identical rows). Dividing
    // by ~0 would silently produce Infinity/NaN control points instead of
    // failing — throw a clear, catchable error instead, so a caller can
    // report an honest refusal rather than commit degenerate geometry.
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

// The shared core of A9.1 — solve for control points GIVEN an already-
// chosen degree/parameters/knot vector, rather than deriving them from
// `points` itself. Factored out of globalCurveInterp (below) so a caller
// that needs SEVERAL interpolations to all share the SAME parametrization
// (Loft's own global SURFACE interpolation, kernel/loft.mjs — P&T 9.2.5:
// every row of a lofted net must use one common knot vector per direction,
// not each row/column independently chord-length-parametrized, or the
// result isn't a genuine tensor-product surface) can reuse this directly.
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
// when there aren't enough points for it — a 2-point SketchCurve still
// works (as a degree-1 line) instead of erroring, same "honest fallback,
// not a hard stop" spirit as REBUILD's live deviation readout.
export function globalCurveInterp(points, requestedDegree = 3) {
  const n = points.length - 1;
  if (n < 1) throw new Error('globalCurveInterp needs at least 2 points');
  const p = Math.min(requestedDegree, n);
  const ubar = chordLengthParams(points);
  const U = averagingKnotVector(ubar, p);
  const ctrlPts = interpAtParams(points, p, ubar, U);
  return { degree: p, knots: U, ctrlPts, paramsUsed: ubar };
}

// Closed-curve interpolation with TANGENT continuity at the seam — distinct
// from just re-interpolating through a duplicated start point (which only
// gives positional/C0 closure, a visible kink where the loop shuts).
//
// P&T's own answer to this (9.2.5, "Global Curve Interpolation with
// Periodic B-splines") builds genuinely periodic basis functions from
// scratch — real machinery this kernel doesn't have. This function reaches
// the same practical result — a closed loop whose tangent matches across
// the seam — via a well-known, easier-to-verify-correct alternative: WRAP
// `degree` points from each end of the closed point sequence cyclically,
// run the ordinary OPEN A9.1 interpolation (above) on that padded
// sequence, and keep only the middle sub-range. The clamped/open artifacts
// a plain interpolation always has at ITS two ends land in the padding,
// outside the kept region — so the kept region behaves exactly as if it
// continues periodically, with no seam kink. Verified directly (not just
// asserted) in test/interpolate.test.mjs by comparing the tangent
// approaching the seam to the tangent leaving it.
// P&T 9.2.2 — Global Curve Interpolation with End Derivatives Specified,
// specialized to CUBIC (degree 3) only — a real, deliberate scope cut, not
// an oversight. Adding 2 control points beyond the ordinary n+1 (to carry
// the 2 new derivative constraints) needs n-p+2 interior knots; for p=3
// that's exactly n-1, which is exactly how many interior chord-length
// parameters (ubar_1..ubar_{n-1}) are already on hand — so the interior
// knots can be assigned directly from those parameters with no further
// averaging/redistribution formula needed. This exact count match is
// specific to p=3; other degrees would need a different (unbuilt here)
// interior-knot rule, so this function refuses anything but the cubic
// case by simply not offering a `requestedDegree` parameter at all.
//
// Given n+1 points and two derivative VECTORS D0/Dn (magnitude matters —
// a longer vector produces a more pronounced bulge near that end, the
// same convention a Bezier/Hermite tangent handle already carries), this
// builds a cubic curve with n+3 control points that interpolates every
// point exactly AND matches C'(ubar_0)=D0, C'(ubar_n)=Dn exactly. The
// linear system is built the same way ordinary global interpolation
// builds its own (one row per constraint, basis VALUES from basisFuns for
// a position row, basis DERIVATIVE values from dersBasisFuns for a
// derivative row) — genuinely new only in which rows are asked for, not
// in how any single row is computed; both basis functions are the same
// already-proven A2.2/A2.3 machinery every other curve in this kernel
// already relies on.
export function globalCurveInterpWithEndDerivs(points, D0, Dn) {
  const n = points.length - 1;
  if (n < 3) throw new Error('globalCurveInterpWithEndDerivs needs at least 4 points (cubic + 2 end derivatives)');
  const p = 3;
  const ubar = chordLengthParams(points);
  // Knot vector: clamped p+1 at each end; interior knots are the raw
  // interior chord-length parameters themselves — see the header comment
  // above for why this count matches exactly for degree 3 with 2 extra
  // control points.
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

// CUBIC HERMITE SEGMENT — the real building block "internal B handles"
// needs (task: extend Bezier tangent-handle editing from a SketchCurve's
// two ENDS to every INTERIOR through-point too). globalCurveInterpWithEndDerivs
// above only ever solves ONE global curve through ALL points at once —
// genuinely new interior-derivative-constraint kernel math (P&T 9.2.4,
// mixed-multiplicity interior knots) would be needed to add a THIRD,
// interior tangent constraint to that same system, and this project's own
// standing rule ("don't rush unverified NURBS math without the reference
// in hand") already named that specific generalization too risky to
// attempt blind. This is deliberately a DIFFERENT, much simpler, well-
// known piece instead: the classic cubic Hermite-to-Bezier control-point
// conversion for a single 2-point segment with two independent end
// tangents, needing no linear solve at all (unlike every other function
// in this file) — B0=p0, B1=p0+m0/3, B2=p1-m1/3, B3=p1 is the standard
// closed-form identity (differentiate the degree-3 Bezier basis: C(0)=B0,
// C(1)=B3, C'(0)=3(B1-B0)=m0, C'(1)=3(B3-B2)=m1 — exact by construction,
// not an approximation). A PolyCurve chain of these segments (reusing
// this app's own already-proven "segments are plain-data pseudo-objects"
// container Join/Fillet already established) is the tractable path to a
// real per-interior-point tangent handle: each interior point's own
// tangent feeds the END derivative of the segment before it and the
// START derivative of the segment after it, giving G1 continuity by
// construction without ever solving a global system. App-layer wiring
// (the PolyCurve conversion, the interior handle UI, drag/commit/undo)
// is real, separate, NOT attempted — this is the kernel-only
// proof, matching this project's own "prove the math first" pattern.
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

// CATMULL-ROM (uniform) TANGENT — the standard, well-known default tangent
// at an interior point of a piecewise curve: the average of the two chord
// vectors on either side (equivalently, half the chord from the previous
// point to the next). This is the smooth-anchor default a "B handle" at
// an interior point would show before ever being dragged — matches the
// real Points app's own smooth-anchor convention (researched for this
// project's original Bezier Handles round): symmetric, antiparallel
// handles on either side, independently draggable afterward.
export function catmullRomTangent(pPrev, pNext) {
  return [(pNext[0] - pPrev[0]) / 2, (pNext[1] - pPrev[1]) / 2, (pNext[2] - pPrev[2]) / 2];
}

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
