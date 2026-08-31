// Knot insertion (P&T Ch.5, Algorithm A5.1, CurveKnotIns) + degree
// elevation (Ch.5, Algorithm A5.9's own well-known Bezier-decomposition
// equivalent) — the real, general kernel machinery two long-standing gaps
// needed: a closed SketchCurve's own
// padded-domain curve (kernel/interpolate.mjs's closedCurveInterp) is not
// a valid standalone profile on its own (its true shape only exists over
// a SUB-RANGE of its raw knot domain), and a mixed-degree PolyCurve (a
// filleted Polygon's Line+Arc chain, or a planar-arrangement smooth-segment
// output) has no way to become ONE curve at all when its segments don't
// already share a degree. Both need this same real P&T Ch.5 machinery —
// built once here, consumed by the app's own two profile refusal sites.
//
// Every function here operates on the SAME NurbsCrv shape the rest of
// this kernel already uses ({degree, knots, ctrlPts}, ctrlPts an array of
// [x,y,z,w] — the real point plus its weight, never pre-multiplied) —
// insertion/elevation both do their real work in HOMOGENEOUS space
// internally (the same toHomogeneous trick curve.mjs's own evaluation
// already relies on), so a RATIONAL curve (an Arc/Circle-derived fillet
// segment) elevates/inserts exactly, not just an approximation.

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

// A5.1, r=1 — Boehm's single-knot-insertion formula. Exact and shape-
// preserving by construction (a knot insertion NEVER changes a curve's
// geometry, only its own control-point/knot REPRESENTATION of it) —
// verified directly in test/knots.test.mjs by sampling before/after.
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

// r repeated single insertions of the SAME knot value — algebraically
// identical to P&T's own multi-insertion A5.1 (r>1), just built from the
// already-proven r=1 case rather than transcribing the denser bezalfs-
// style multi-insertion indexing a second, riskier way.
export function insertKnot(crv, u, r = 1) {
  let c = crv;
  for (let i = 0; i < r; i++) c = insertKnotOnce(c, u);
  return c;
}

function insertKnotToMultiplicity(crv, u, targetMult, tol = 1e-9) {
  let c = crv;
  let s = knotMultiplicity(c.knots, u, tol);
  while (s < targetMult) { c = insertKnotOnce(c, u); s++; }
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

// DecomposeCurve (A5.6's own real target) — insert every interior knot
// value up to full multiplicity `degree` (NOT degree+1 — an INTERIOR
// Bezier breakpoint stays part of the same curve, sharing its one
// boundary control point with its neighbor; only the curve's own two true
// ENDS are ever p+1/clamped). Returns an ordered array of Bezier pieces,
// each { ctrlPts: [degree+1 real [x,y,z,w] points], u0, u1 }. A curve with
// zero interior knots (already a single Bezier span — a Line, or one
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

// Standard Bezier degree-elevation-by-1 (Farin/P&T's own well-known
// closed-form: Q_i = (i/(p+1))*P_{i-1} + (1-i/(p+1))*P_i, boundary terms
// dropped rather than indexed out of range) — run in homogeneous space so
// a rational Bezier (an Arc/Circle segment) elevates exactly, matching
// every other rational-aware routine in this kernel.
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

// Reassembles an ordered chain of ALREADY-SAME-DEGREE Bezier pieces
// (contiguous domains, piece[i].u1 === piece[i+1].u0) into one combined
// clamped NurbsCrv — the shared inverse of decomposeToBezier above.
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

// Degree elevation (A5.9's own real target, reached via the well-known
// simpler equivalent construction rather than transcribing A5.9's own
// dense bezalfs-table pseudocode a second, riskier way from memory):
// decompose to Bezier, elevate every piece by the same amount, reassemble
// at FULL multiplicity everywhere. This is exact (both decomposition and
// per-Bezier elevation are individually shape-preserving) but honestly
// NOT minimal — an interior knot that started below full multiplicity
// (a smooth SketchCurve span's own simple knots) comes back at full
// multiplicity `targetDegree` rather than the theoretically-minimal
// s_orig+t, i.e. the elevated curve reports less smoothness in its own
// knot vector than its real, sampled geometry actually has. This never
// changes the curve's real SHAPE (verified directly in test/knots.test.mjs
// by sampling before/after at many parameters, derivatives included) —
// only its representational minimality — and it's the exact case this
// kernel's own actual callers need: a Line/Polyline segment (degree 1,
// ALREADY at full multiplicity everywhere by construction — see
// getProfileCrv's own Polyline branch) is the only thing ever elevated
// by joinCurvesC0 below, so the "non-minimal" cost never actually bites
// in practice, only the guaranteed-exact shape matters.
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

// Pure affine reparametrization of a curve's own knot VALUES (control
// points untouched) — shape-preserving by construction (a B-spline's
// geometry depends only on the RELATIVE spacing of its knot vector, never
// the absolute numbers labeling each parameter).
export function rescaleCurveDomain(crv, newMin, newMax) {
  const oldMin = crv.knots[0], oldMax = crv.knots[crv.knots.length - 1];
  const span = oldMax - oldMin;
  const knots = span < 1e-12
    ? crv.knots.map(() => newMin)
    : crv.knots.map((k) => newMin + ((k - oldMin) / span) * (newMax - newMin));
  return { degree: crv.degree, knots, ctrlPts: crv.ctrlPts.map((p) => p.slice()) };
}

// EXTRACT SUB-CURVE — the real fix for a closed SketchCurve's own padded
// domain (closedCurveInterp's [uStart,uEnd] sub-range is not itself a
// valid curve until the domain is actually TRIMMED down to it): insert
// BOTH boundary parameters up to full clamped multiplicity (degree+1,
// same as any curve's real endpoint — NOT decomposeToBezier's own
// interior-multiplicity=degree, since these become genuine new ends, not
// an internal C0 breakpoint), then slice the now-isolated index range.
// Exact by construction (knot insertion never changes the curve's real
// shape; slicing off a fully-clamped, non-overlapping index range simply
// discards the surrounding padding without touching the kept part at
// all) — verified directly against the SAME curve's own sampled points
// over [uStart,uEnd], before vs. after extraction.
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

// A NOTE FOR ANYONE CLAMPING AN UNCLAMPED CURVE WITH THESE FUNCTIONS: don't.
// insertKnotOnce implements P&T A5.1, whose own precondition is that the knot
// being inserted stays within multiplicity <= degree. extractSubCurve below
// deliberately goes one further (to degree+1) to isolate a sub-range, and that
// is exact for the CLAMPED curves it was written for — at multiplicity p the
// control point being duplicated genuinely lies ON the curve, so duplicating it
// splits the curve without moving it. On an UNCLAMPED curve that point is not
// on the curve, the same step silently duplicates it anyway with no blend, and
// the result is a DIFFERENT curve (measured: 0.37 units on a plain uniform
// cubic, against a curve spanning ~3). decomposeToBezier assumes clamped input
// too, and returns NaN pieces for an unclamped one.
// The exact conversion for the one unclamped shape this kernel actually
// produces — a uniform bicubic patch — lives in subdlimit.mjs next to the
// construction it undoes, where the knot vector it assumes is defined.

// C0-only concatenation of two ALREADY-SAME-DEGREE clamped curves whose
// domains are adjacent (A's own end === B's own start) — drops exactly
// ONE of A's own (degree+1) trailing end-knot copies (leaving `degree`
// copies, the standard interior-joint multiplicity for a positional-only,
// non-smooth seam) and skips ALL of B's own (degree+1) leading copies
// outright (redundant once A's own trailing copies already supply the
// needed multiplicity), plus drops B's own first control point (the
// shared joint, geometrically coincident with A's own last one — Join's
// own JOIN_TOLERANCE match is what already guarantees that coincidence
// upstream). Each curve's own INTERNAL knot structure — and therefore its
// own internal continuity class — is left completely untouched; only the
// one shared boundary knot's multiplicity is affected.
export function concatTwoC0(A, B, degree) {
  const p = degree;
  const knots = A.knots.slice(0, A.knots.length - 1).concat(B.knots.slice(p + 1));
  const ctrlPts = A.ctrlPts.map((pt) => pt.slice()).concat(B.ctrlPts.slice(1).map((pt) => pt.slice()));
  return { degree: p, knots, ctrlPts };
}

// JOIN CURVES AT C0 — the real fix for a mixed-degree PolyCurve (a
// filleted Polygon's Line+Arc chain, or any future mixed-segment chain):
// every segment is degree-ELEVATED (above) up to the highest degree
// present, each rescaled onto its own sequential integer domain slot
// [i,i+1] (rescaleCurveDomain — an arbitrary but simple, consistent
// convention; extrude()/revolve() only ever need a well-formed curve,
// never care what its real parameter values mean), then chained via
// concatTwoC0 into ONE combined curve. A degree-1 chain (every segment
// already the same degree) hits zero elevation calls and reduces to pure
// concatenation — algebraically the exact same knot/control-point shape
// getProfileCrv's own pre-existing degree-1-only fast path already
// produced (checked directly in test/knots.test.mjs, not just assumed).
export function joinCurvesC0(curves) {
  if (!curves.length) throw new Error('joinCurvesC0: need at least one curve');
  const targetDegree = Math.max(...curves.map((c) => c.degree));
  const prepped = curves.map((crv, i) => {
    const rescaled = rescaleCurveDomain(crv, i, i + 1);
    return rescaled.degree < targetDegree ? degreeElevateCurve(rescaled, targetDegree) : rescaled;
  });
  return prepped.reduce((acc, c) => (acc ? concatTwoC0(acc, c, targetDegree) : c), null);
}
