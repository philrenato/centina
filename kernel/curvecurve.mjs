// CURVE-CURVE INTERSECTION IN THE PLANE.
//
// WHY 2D, AND WHY THAT IS THE RIGHT QUESTION: this exists to gate trim-loop
// validity and to pave face boundaries for face splitting, and both of those
// live in a surface's own UV domain. Two general curves in 3D generically do
// not meet at all, so a 3D version would answer a question nobody is asking.
// Callers hand in curves whose x/y ARE the coordinates they care about; z is
// ignored outright rather than projected, so a caller that has not already
// flattened its input gets a wrong answer loudly rather than a plausible one
// quietly.
//
// EVERY STEP IS A PROOF OR A REFUSAL, not an approximation:
//
//   1. decomposeToBezier (kernel/knots.mjs) turns each curve into Bezier
//      pieces, because the convex hull property below is stated for a Bezier,
//      not for a general B-spline.
//
//   2. CONVEX HULL REJECTION. A rational Bezier with POSITIVE weights is a
//      convex combination of its own euclidean control points: the rational
//      basis R_i = w_i*B_i / sum(w_j*B_j) is non-negative and sums to one. So
//      the piece lies inside its control points' bounding box, and two pieces
//      with disjoint boxes provably cannot cross. A rejection here is a proof,
//      not a heuristic, which is what lets the recursion discard whole
//      parameter ranges outright.
//
//      The positive-weight precondition is CHECKED, never assumed. A negative
//      weight breaks the convex combination and with it the entire rejection
//      argument, so it is refused by name rather than silently producing a
//      confidently wrong answer.
//
//   3. SUBDIVISION is de Casteljau at the midpoint, run in HOMOGENEOUS space,
//      which is exact for a rational piece — an arc or circle segment
//      subdivides without drift, matching every other rational-aware routine
//      in this kernel.
//
//   4. NEWTON REFINEMENT. Subdivision alone converges linearly and would need
//      dozens of levels to reach machine precision. Each surviving leaf seeds
//      a 2x2 Newton solve on (Ax(uA)-Bx(uB), Ay(uA)-By(uB)) against the
//      curves' own analytic derivatives, which converges quadratically and
//      lands on the TRUE curves rather than on a subdivided approximation of
//      them. A step that fails to reduce the residual is rejected rather than
//      taken, so a bad seed degrades to its own starting accuracy instead of
//      wandering.
//
//   5. TANGENCY AND OVERLAP ARE REFUSED, NOT GUESSED. At a transversal
//      crossing the Jacobian [A'(uA), -B'(uB)] is well conditioned; at a
//      tangency it is singular, and along an overlap every leaf survives
//      subdivision. Both are reported by name. Transversal-only is the stated
//      scope, so an honest refusal is the correct output, not a fallback.

import { decomposeToBezier } from './knots.mjs';
import { curvePoint, rationalCurveDerivs } from './curve.mjs';

// A leaf's box must shrink to this fraction of the input's own size before it
// is accepted as a root seed. Relative, not absolute: an absolute floor would
// refuse to converge on a large curve and over-subdivide a small one.
const SEED_TOL_FRAC = 1e-4;
// Recursion depth cap. Each level halves a box, so this bounds the seed's own
// accuracy independently of SEED_TOL_FRAC; Newton is what actually delivers
// precision, so this only has to be deep enough to separate distinct roots.
const MAX_DEPTH = 40;
// Surviving leaves beyond this mean the two curves share a region rather than
// crossing at points — the overlap case, which is refused by name.
const MAX_LEAVES = 4096;
const NEWTON_STEPS = 24;
// Relative conditioning floor for the 2x2 Jacobian. |det| is compared against
// the product of the two tangent magnitudes, which makes it a sine of the
// crossing angle and therefore scale-free.
const TANGENT_SIN_FLOOR = 1e-6;

function domainOf(crv) {
  const k = crv.knots;
  return [k[crv.degree], k[k.length - 1 - crv.degree]];
}

function toHomogPts(pts) {
  return pts.map(([x, y, z, w]) => [x * w, y * w, z * w, w]);
}

// de Casteljau split at t, in homogeneous space. Returns both halves' control
// points. Exact for a rational piece because the homogeneous coordinates are
// what the recurrence is linear in.
function splitHomog(ptsW, t) {
  const n = ptsW.length;
  const tri = [ptsW.map((p) => p.slice())];
  for (let r = 1; r < n; r++) {
    const prev = tri[r - 1];
    const row = [];
    for (let i = 0; i + r < n; i++) {
      row.push([0, 1, 2, 3].map((d) => (1 - t) * prev[i][d] + t * prev[i + 1][d]));
    }
    tri.push(row);
  }
  const left = [], right = [];
  for (let r = 0; r < n; r++) {
    left.push(tri[r][0].slice());
    right.push(tri[n - 1 - r][r].slice());
  }
  return { left, right };
}

// Bounding box of the EUCLIDEAN control points — the convex hull bound from
// step 2. Homogeneous points are divided through by w here precisely because
// the hull property is about the euclidean points, not the homogeneous ones.
function box2(ptsW) {
  let xlo = Infinity, ylo = Infinity, xhi = -Infinity, yhi = -Infinity;
  for (const p of ptsW) {
    const w = p[3];
    const x = p[0] / w, y = p[1] / w;
    if (x < xlo) xlo = x;
    if (x > xhi) xhi = x;
    if (y < ylo) ylo = y;
    if (y > yhi) yhi = y;
  }
  return { xlo, ylo, xhi, yhi, size: Math.max(xhi - xlo, yhi - ylo) };
}

function boxesOverlap(a, b, tol) {
  return a.xlo - tol <= b.xhi && b.xlo - tol <= a.xhi
    && a.ylo - tol <= b.yhi && b.ylo - tol <= a.yhi;
}

// Every weight must be strictly positive for the convex hull argument to
// hold. Checked rather than assumed — see the header.
function weightsPositive(crv) {
  return crv.ctrlPts.every((p) => p[3] > 0);
}

// 2x2 Newton against the true curves. Returns null when the system is
// singular at the seed (a tangency, handled by the caller) or when no step
// improves the residual.
function refine(crvA, crvB, uA0, uB0, domA, domB) {
  let uA = uA0, uB = uB0;
  const clamp = (u, d) => Math.min(d[1], Math.max(d[0], u));
  // Seeded from the incoming guess rather than left null: a NaN residual on
  // the very first evaluation (a degenerate input curve) would otherwise leave
  // this null and turn an honest refusal into a thrown TypeError one line
  // later. A seeded best degrades to "no better than where we started", which
  // the caller's own residual gate then discards.
  let best = { uA, uB }, bestRes = Infinity;
  for (let it = 0; it < NEWTON_STEPS; it++) {
    const dA = rationalCurveDerivs(crvA, uA, 1);
    const dB = rationalCurveDerivs(crvB, uB, 1);
    const fx = dA[0][0] - dB[0][0];
    const fy = dA[0][1] - dB[0][1];
    const res = Math.hypot(fx, fy);
    if (res < bestRes) { bestRes = res; best = { uA, uB }; }
    // J = [ Ax'  -Bx' ; Ay'  -By' ]
    const a11 = dA[1][0], a12 = -dB[1][0];
    const a21 = dA[1][1], a22 = -dB[1][1];
    const det = a11 * a22 - a12 * a21;
    const magA = Math.hypot(dA[1][0], dA[1][1]);
    const magB = Math.hypot(dB[1][0], dB[1][1]);
    // |det| / (|A'||B'|) is the sine of the crossing angle — scale-free, so
    // this floor means the same thing on a 1mm curve and a 1000mm one.
    if (!(Math.abs(det) > TANGENT_SIN_FLOOR * magA * magB)) {
      return { singular: true, uA: best.uA, uB: best.uB, residual: bestRes };
    }
    const du = (-fx * a22 + fy * a12) / det;
    const dv = (-a11 * fy + a21 * fx) / det;
    const nA = clamp(uA + du, domA);
    const nB = clamp(uB + dv, domB);
    const pA = curvePoint(crvA, nA), pB = curvePoint(crvB, nB);
    const nres = Math.hypot(pA[0] - pB[0], pA[1] - pB[1]);
    // A step that does not reduce the residual is rejected rather than taken:
    // a clamped step at a domain end can otherwise oscillate forever.
    if (!(nres < res)) break;
    uA = nA; uB = nB;
    if (nres < 1e-14) { best = { uA, uB }; bestRes = nres; break; }
  }
  return { singular: false, uA: best.uA, uB: best.uB, residual: bestRes };
}

/**
 * Intersect two planar NURBS curves, using x/y only.
 *
 * Returns { ok, points, reason?, tangential?, overlapping? } where each point
 * is { uA, uB, point:[x,y,z] }. `ok:false` is always accompanied by a reason
 * naming the case — an empty `points` array with `ok:true` means the curves
 * genuinely do not meet, which is a real answer and not a failure.
 */
export function intersectCurves2D(crvA, crvB, opts = {}) {
  if (!weightsPositive(crvA) || !weightsPositive(crvB)) {
    return { ok: false, points: [], reason: 'a control point weight is zero or negative, so the convex hull bound this search proves its rejections with does not hold' };
  }
  const domA = domainOf(crvA), domB = domainOf(crvB);
  const piecesA = decomposeToBezier(crvA);
  const piecesB = decomposeToBezier(crvB);

  // Tolerances are relative to the two curves' combined extent, so the same
  // call behaves identically on a shape scaled up or down.
  const allBox = box2(toHomogPts([...crvA.ctrlPts, ...crvB.ctrlPts]));
  const scale = Math.max(allBox.size, 1e-12);
  const seedTol = opts.tolerance ?? scale * SEED_TOL_FRAC;
  const mergeTol = opts.mergeTolerance ?? scale * 1e-6;

  const leaves = [];
  let overflowed = false;

  const recurse = (a, b, depth) => {
    if (overflowed) return;
    const ba = box2(a.ptsW), bb = box2(b.ptsW);
    // The provable rejection. Everything else in this function is bookkeeping
    // around this one line.
    if (!boxesOverlap(ba, bb, seedTol)) return;
    if ((ba.size <= seedTol && bb.size <= seedTol) || depth >= MAX_DEPTH) {
      if (leaves.length >= MAX_LEAVES) { overflowed = true; return; }
      leaves.push({ uA: (a.u0 + a.u1) / 2, uB: (b.u0 + b.u1) / 2 });
      return;
    }
    // Split whichever piece is currently larger, so both shrink over the
    // recursion rather than one stalling while the other is refined forever.
    if (ba.size >= bb.size) {
      const { left, right } = splitHomog(a.ptsW, 0.5);
      const mid = (a.u0 + a.u1) / 2;
      recurse({ ptsW: left, u0: a.u0, u1: mid }, b, depth + 1);
      recurse({ ptsW: right, u0: mid, u1: a.u1 }, b, depth + 1);
    } else {
      const { left, right } = splitHomog(b.ptsW, 0.5);
      const mid = (b.u0 + b.u1) / 2;
      recurse(a, { ptsW: left, u0: b.u0, u1: mid }, depth + 1);
      recurse(a, { ptsW: right, u0: mid, u1: b.u1 }, depth + 1);
    }
  };

  for (const pa of piecesA) {
    for (const pb of piecesB) {
      recurse(
        { ptsW: toHomogPts(pa.ctrlPts), u0: pa.u0, u1: pa.u1 },
        { ptsW: toHomogPts(pb.ctrlPts), u0: pb.u0, u1: pb.u1 },
        0,
      );
      if (overflowed) break;
    }
    if (overflowed) break;
  }

  if (overflowed) {
    return {
      ok: false, points: [], overlapping: true,
      reason: 'the two curves share a region rather than crossing at isolated points — an overlap, which this transversal-only intersection refuses rather than returning an arbitrary point from',
    };
  }

  const out = [];
  let sawTangency = false;
  for (const leaf of leaves) {
    const r = refine(crvA, crvB, leaf.uA, leaf.uB, domA, domB);
    if (r.singular) { sawTangency = true; continue; }
    // A leaf survives box overlap without necessarily containing a root; the
    // residual is what decides whether one is actually there.
    if (!(r.residual <= mergeTol * 10 + 1e-9 * scale)) continue;
    const pA = curvePoint(crvA, r.uA);
    const pB = curvePoint(crvB, r.uB);
    const pt = [(pA[0] + pB[0]) / 2, (pA[1] + pB[1]) / 2, (pA[2] + pB[2]) / 2];
    // Dedupe on the POINT, not the parameters: subdivision routinely lands
    // several leaves on one root, and a closed curve reaches the same point
    // from two very different parameters at its own seam.
    if (out.some((e) => Math.hypot(e.point[0] - pt[0], e.point[1] - pt[1]) <= mergeTol)) continue;
    out.push({ uA: r.uA, uB: r.uB, point: pt });
  }
  out.sort((a, b) => a.uA - b.uA);

  if (!out.length && sawTangency) {
    return {
      ok: false, points: [], tangential: true,
      reason: 'the two curves meet tangentially rather than crossing — the crossing angle is below this search\'s own conditioning floor, so no transversal intersection point is defined here',
    };
  }
  return { ok: true, points: out, tangential: sawTangency || undefined };
}
