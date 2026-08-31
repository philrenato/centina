// SURFACE-SURFACE INTERSECTION (SSI) — the genuinely hard piece of the
// trimmed-surface work, built for the first time. Standard
// Barnhill-Kersey-style marching in JOINT parameter space (u1,v1,u2,v2), per
// the spec: seed by coarse search + Gauss-Newton snap, then march
// with a predictor (tangent = N1 x N2) / corrector (Newton on the joint
// parameter vector, closed by a marching-plane constraint) pair, honestly
// refusing at genuine tangency rather than attempting to resolve it.
//
// SCOPE, STATED PLAINLY (matching this kernel's own standing "name the
// limitation, don't silently half-solve it" discipline): this module finds
// ONE intersection curve component — whichever the seed search happens to
// land nearest — either a closed interior loop or an open curve running to
// one surface's own parametric boundary. It does NOT enumerate every
// component two surfaces might share (a genuinely separate, harder search
// problem — a loop smaller than the seeding facet size can
// be missed) and it REFUSES outright at a near-tangent
// region rather than attempting to resolve degenerate/higher-multiplicity
// crossings. This is Tier C's real numerical core
// for the bounded, honest v1 case: two ordinary, transversally
// intersecting surfaces, one real curve component.

import { surfacePoint, surfacePointAndPartials, assertSurface } from './surface.mjs';
import { surfaceClosure } from './surface.mjs';
import { closestPointOnSurface } from './surface.mjs';
import { add, sub, scale, dot, cross, length } from './vec3.mjs';
import { extractBorderCurves } from './isocurve.mjs';
import { curveSurfaceIntersections } from './curvesurface.mjs';

// Small dense linear solve (Gauss-Jordan with partial pivoting) — every
// system this module builds is at most 4x4 (the joint-parameter Newton
// correction), so a general library is unwarranted; this is the same class
// of "small, local, well-known, hand-rollable" primitive as basis.mjs's own
// findSpan, not new algorithmic risk. Returns null for a singular system
// (caller's own responsibility to treat that as "correction failed," never
// silently divide by ~0).
export function solveSquareSystem(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-13) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const pivVal = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= pivVal;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}

function safeNormalize(v) {
  const l = length(v);
  return l < 1e-10 ? null : scale(v, 1 / l);
}

// THE EMITTED SAMPLE POINT IS THE MIDPOINT OF BOTH SURFACES' OWN
// EVALUATIONS, NOT srf1's ALONE — a real correctness decision, not a
// cosmetic average. The corrector drives |S1(u1,v1) - S2(u2,v2)| below its
// own residual tolerance, never to zero: the two evaluations genuinely
// disagree by that residual. Emitting S1 alone makes every sample exact on
// srf1 and off srf2 by the FULL residual, so which surface gets the error
// depends entirely on which one the caller happened to pass first — a
// silently order-dependent result. The midpoint splits the disagreement
// evenly and is order-independent by construction.
function pairPoint(srf1, u1, v1, srf2, u2, v2) {
  const a = surfacePoint(srf1, u1, v1);
  const b = surfacePoint(srf2, u2, v2);
  return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
}

function domainOf(srf) {
  return {
    uMin: srf.knotsU[0], uMax: srf.knotsU[srf.knotsU.length - 1],
    vMin: srf.knotsV[0], vMax: srf.knotsV[srf.knotsV.length - 1],
  };
}

// Seed finding (step 1, simplified to the single-nearest-approach
// case rather than a full triangle-mesh facet search — the coarse grid here
// plays the same "get within the true intersection's Newton basin" role a
// tessellate-and-facet-test pass would, at the cost of the same honest
// limitation already named above: a component whose closest approach isn't
// found by this grid can be missed). Stage 1 samples srf1 on a coarse grid
// and calls the ALREADY-BUILT closestPointOnSurface (surface.mjs) against
// srf2 for each sample — reusing proven point-inversion machinery rather
// than re-deriving a second one. Stage 2 is a genuine 4-unknown Gauss-Newton
// snap (minimize |S1(u1,v1)-S2(u2,v2)|^2 over all four parameters at once —
// the natural generalization of closestPointOnSurface's own 2-unknown
// Gauss-Newton to two independent surfaces) that pins the coarse seed onto
// the TRUE zero-distance intersection point before marching starts.
export function refineSeedSnap(srf1, srf2, seed) {
  const d1 = domainOf(srf1), d2 = domainOf(srf2);
  let { u1, v1, u2, v2 } = seed;
  let curDist = length(sub(surfacePoint(srf1, u1, v1), surfacePoint(srf2, u2, v2)));
  for (let iter = 0; iter < 30; iter++) {
    const a = surfacePointAndPartials(srf1, u1, v1);
    const b = surfacePointAndPartials(srf2, u2, v2);
    const r = sub(a.point, b.point);
    const cols = [a.su, a.sv, scale(b.su, -1), scale(b.sv, -1)];
    const JTJ = Array.from({ length: 4 }, () => Array(4).fill(0));
    const JTr = Array(4).fill(0);
    for (let ii = 0; ii < 4; ii++) {
      for (let jj = 0; jj < 4; jj++) JTJ[ii][jj] = dot(cols[ii], cols[jj]);
      JTr[ii] = dot(cols[ii], r);
    }
    const trace = JTJ[0][0] + JTJ[1][1] + JTJ[2][2] + JTJ[3][3];
    const lambda = Math.max(trace * 1e-8, 1e-12);
    for (let k = 0; k < 4; k++) JTJ[k][k] += lambda;
    const delta = solveSquareSystem(JTJ, JTr.map((x) => -x));
    if (!delta) break;
    const nu1 = Math.max(d1.uMin, Math.min(d1.uMax, u1 + delta[0]));
    const nv1 = Math.max(d1.vMin, Math.min(d1.vMax, v1 + delta[1]));
    const nu2 = Math.max(d2.uMin, Math.min(d2.uMax, u2 + delta[2]));
    const nv2 = Math.max(d2.vMin, Math.min(d2.vMax, v2 + delta[3]));
    const nd = length(sub(surfacePoint(srf1, nu1, nv1), surfacePoint(srf2, nu2, nv2)));
    if (nd > curDist + 1e-10) break;
    u1 = nu1; v1 = nv1; u2 = nu2; v2 = nv2; curDist = nd;
    if (nd < 1e-12) break;
  }
  return { u1, v1, u2, v2, distance: curDist, point: pairPoint(srf1, u1, v1, srf2, u2, v2) };
}

export function seedSurfaceIntersection(srf1, srf2, opts = {}) {
  const d1 = domainOf(srf1), d2 = domainOf(srf2);
  const gridU1 = opts.gridU1 ?? Math.max(8, Math.min(24, srf1.ctrlNet.length * 2));
  const gridV1 = opts.gridV1 ?? Math.max(6, Math.min(16, srf1.ctrlNet[0].length * 2));
  let best = null;
  for (let i = 0; i <= gridU1; i++) {
    const u1 = d1.uMin + (d1.uMax - d1.uMin) * (i / gridU1);
    for (let j = 0; j <= gridV1; j++) {
      const v1 = d1.vMin + (d1.vMax - d1.vMin) * (j / gridV1);
      const p1 = surfacePoint(srf1, u1, v1);
      const cp = closestPointOnSurface(srf2, p1);
      if (!best || cp.distance < best.distance) best = { u1, v1, u2: cp.u, v2: cp.v, distance: cp.distance };
    }
  }
  let { u1, v1, u2, v2 } = best;
  let curDist = best.distance;
  for (let iter = 0; iter < 30; iter++) {
    const a = surfacePointAndPartials(srf1, u1, v1);
    const b = surfacePointAndPartials(srf2, u2, v2);
    const r = sub(a.point, b.point);
    const cols = [a.su, a.sv, scale(b.su, -1), scale(b.sv, -1)];
    const JTJ = Array.from({ length: 4 }, () => Array(4).fill(0));
    const JTr = Array(4).fill(0);
    for (let ii = 0; ii < 4; ii++) {
      for (let jj = 0; jj < 4; jj++) JTJ[ii][jj] = dot(cols[ii], cols[jj]);
      JTr[ii] = dot(cols[ii], r);
    }
    // Levenberg-Marquardt damping, not plain Gauss-Newton: this system is
    // FUNDAMENTALLY rank-deficient once the seed is genuinely close to a
    // true intersection curve, not a numerical accident of this particular
    // fixture. The residual r=S1-S2 is 3 equations in 4 unknowns — near any
    // real point ON the intersection curve, r=0 holds along a whole
    // 1-parameter FAMILY of (u1,v1,u2,v2) (the curve's own tangent
    // direction), not just one isolated point, so J^T J is singular exactly
    // along that direction (confirmed directly while building this: plain
    // Gauss-Newton hit a genuinely singular normal-equation matrix on a
    // real cylinder-cylinder seed, not a degenerate fixture choice). A
    // small diagonal damping term regularizes that flat direction the same
    // way LM damping always does, converging normally elsewhere — the
    // standard, well-known fix for exactly this shape of problem.
    const trace = JTJ[0][0] + JTJ[1][1] + JTJ[2][2] + JTJ[3][3];
    const lambda = Math.max(trace * 1e-8, 1e-12);
    for (let k = 0; k < 4; k++) JTJ[k][k] += lambda;
    const delta = solveSquareSystem(JTJ, JTr.map((x) => -x));
    if (!delta) break;
    const nu1 = Math.max(d1.uMin, Math.min(d1.uMax, u1 + delta[0]));
    const nv1 = Math.max(d1.vMin, Math.min(d1.vMax, v1 + delta[1]));
    const nu2 = Math.max(d2.uMin, Math.min(d2.uMax, u2 + delta[2]));
    const nv2 = Math.max(d2.vMin, Math.min(d2.vMax, v2 + delta[3]));
    const nd = length(sub(surfacePoint(srf1, nu1, nv1), surfacePoint(srf2, nu2, nv2)));
    if (nd > curDist + 1e-10) break; // reject a worsening step, matching closestPointOnSurface's own guard
    u1 = nu1; v1 = nv1; u2 = nu2; v2 = nv2; curDist = nd;
    if (nd < 1e-12) break;
  }
  return { u1, v1, u2, v2, distance: curDist, point: pairPoint(srf1, u1, v1, srf2, u2, v2) };
}

const TANGENT_SINE_THRESHOLD = 0.05; // |N1 x N2| below this = near-tangent, refuse rather than resolve
const CLAMP_EPS = 1e-9;

// MARCH STEP — DERIVED FROM THE GEOMETRY, NOT A CONSTANT.
//
// The original fixed `stepLen = 0.25` with `maxSteps = 400` is a hard
// budget of 100 UNITS OF ARC LENGTH, total, at any scale. That is not a
// robustness nicety to tune later: a plane cutting a 30-unit-radius
// cylinder produces a circle of circumference 188.5, so the single
// simplest intersection there is — one circle, one component, no
// tangency, no boundary — refused outright with "did not close or reach a
// boundary within 400 steps". Measured directly, and the marching math was
// never at fault: the SAME pair at stepLen 1/2/3 closes correctly with
// radius 30.0000 every time. A millimetre-scale CAD document reaches this
// immediately, so boolean completeness cannot even be MEASURED
// until it is fixed — which is why this lands ahead of the seeding work
// that is otherwise sequenced first.
//
// The step now starts as a fraction of the two surfaces' OWN size and then
// adapts to the curve's own turning, which is the quantity that actually
// governs both risks: too large a step invites the component jumping
// Krishnan & Manocha name (landing on a different branch and marching on
// happily), and too small a step just burns budget. Bounding the TURN PER
// STEP bounds the chord's deviation from the true curve directly, and it
// costs nothing extra — the unit tangent is already computed every step.
//
// An explicit `opts.stepLen` DISABLES the turn-based adaptation entirely.
// It is honoured verbatim on every step EXCEPT one where the corrector
// genuinely fails to converge at it: that single step is retried shorter (see
// SSI_CORRECTOR_RETRIES) and the caller's own value is restored for the next
// one. The shortening is a per-step remedy for one hard corner, never a
// permanent downgrade of a step the caller asked for.
const SSI_STEP_FRAC_INIT = 1 / 200;   // opening step, as a fraction of the pair's own size
const SSI_STEP_MIN_FRAC = 1 / 20000;  // floor, so a tight pinch cannot stall the march at zero
const SSI_STEP_MAX_FRAC = 1 / 40;     // ceiling, so a near-straight run cannot stride past a branch
const SSI_SAMPLES_PER_SPAN = 12;       // minimum samples across the tightest knot span either surface has
const SSI_TARGET_TURN_RAD = 5 * Math.PI / 180;
const SSI_MAX_ARC_FRAC = 80;          // refuse past this multiple of the pair's own size of marched arc
const SSI_MAX_STEPS = 20000;          // backstop only — the arc budget above is the real stop
const SSI_CORRECTOR_RETRIES = 8;      // halvings a failed correction is allowed before the march honestly refuses

// The pair's own size: the diagonal of the combined control-net bounding
// box. A control net bounds its own surface (the convex-hull property), so
// this is a genuine upper bound on the geometry's extent, not a sample-
// based estimate that could miss a bulge.
export function ssiPairScale(srf1, srf2) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const srf of [srf1, srf2]) {
    for (const row of srf.ctrlNet) {
      for (const cp of row) {
        for (let k = 0; k < 3; k++) { if (cp[k] < lo[k]) lo[k] = cp[k]; if (cp[k] > hi[k]) hi[k] = cp[k]; }
      }
    }
  }
  const d = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return d > 0 ? d : 1;
}

// SPAN-AWARE STEP CEILING — the step must never stride across a whole knot
// span of either surface without sampling inside it.
//
// The adaptive step above keys on the CURVE's own turning, which is exactly
// the right signal for a smooth run and exactly the wrong one at a curvature
// BREAK: a rational multi-span wall (a filleted profile, a swept rail) is
// nearly straight through each span and turns sharply only where two spans
// meet, so a turn-driven step happily grows to its ceiling and then strides
// clean over a fillet arc — the samples land on either side and the fitted
// curve cuts the corner, with no turn ever measured to react to.
//
// The cure is the same one this kernel already uses in two other places for
// the same reason: tessellationVSamples' own "guarantee N samples per span"
// floor (surface.mjs) and denseRailFrames' own MIN_SPAN_SAMPLES (sweep.mjs).
// Here it lands as a CEILING on the marching step rather than a floor on a
// sample count, because the march has no span index to count against — but
// it is the same guarantee: at least SSI_SAMPLES_PER_SPAN samples across the
// tightest span either surface has.
//
// A single-span pair (a plain box face, an untrimmed plane) has spanCount 1,
// which puts this ceiling far ABOVE the ordinary SSI_STEP_MAX_FRAC one, so
// the min() below leaves those cases byte-identical.
function distinctSpanCount(knots, degree) {
  const interior = knots.slice(degree + 1, knots.length - degree - 1);
  let distinct = 0;
  for (let i = 0; i < interior.length; i++) {
    if (i === 0 || interior[i] > interior[i - 1] + 1e-12) distinct++;
  }
  return distinct + 1;
}

export function ssiMaxSpanCount(srf1, srf2) {
  let m = 1;
  for (const srf of [srf1, srf2]) {
    m = Math.max(
      m,
      distinctSpanCount(srf.knotsU, srf.degU),
      distinctSpanCount(srf.knotsV, srf.degV),
    );
  }
  return m;
}

function clampToBoundary(u, v, d, tolClosedU, tolClosedV) {
  // Returns { u, v, hit } — hit is null, or {which:'u'|'v', edge:'min'|'max'}
  // if this step genuinely exits a NON-closed direction's domain (a real
  // surface boundary, as opposed to a closed direction's own internal seam,
  // which is wrapped instead — see wrapParam below).
  let hit = null;
  if (!tolClosedU) {
    if (u < d.uMin - CLAMP_EPS) { u = d.uMin; hit = { which: 'u', edge: 'min' }; }
    else if (u > d.uMax + CLAMP_EPS) { u = d.uMax; hit = { which: 'u', edge: 'max' }; }
  }
  if (!hit && !tolClosedV) {
    if (v < d.vMin - CLAMP_EPS) { v = d.vMin; hit = { which: 'v', edge: 'min' }; }
    else if (v > d.vMax + CLAMP_EPS) { v = d.vMax; hit = { which: 'v', edge: 'max' }; }
  }
  return { u, v, hit };
}

function wrapParam(x, min, max) {
  const span = max - min;
  if (span <= 0) return x;
  while (x < min) x += span;
  while (x > max) x -= span;
  return x;
}

// One predictor/corrector march step (step 2). `fixed` optionally
// pins ONE of the four joint parameters (used to solve an exact boundary
// crossing — see marchDirection below): when set, that parameter is held
// constant and the marching-plane constraint row is DROPPED entirely, since
// fixing one unknown already leaves exactly 3 unknowns for the 3 real
// position equations S1-S2=0 (no additional constraint needed or wanted —
// adding the plane row back would over-determine a perfectly well-posed
// system).
//
// `wrapParams` (optional) folds each CLOSED direction's parameter back into
// its own domain after every Newton step. This is not tidiness — without it
// the corrector can converge OUTSIDE the domain, where a clamped B-spline
// EXTRAPOLATES along its first (or last) span's polynomial rather than
// continuing around the seam. The residual it reports there is genuinely
// small, so the step is accepted; the caller then wraps the parameter, and
// the sample silently lands on a completely different real point. Found by
// measurement, not inspection: on a closed extruded wall exactly one sample
// per component — always the one whose step crossed the seam — sat ~2e-2 off
// BOTH surfaces while every neighbour was at machine precision. Wrapping
// inside the loop keeps every evaluation on the real surface, so the answer
// the corrector converges to is the answer the caller keeps.
function correctorStep(srf1, srf2, u1, v1, u2, v2, Qpred, T, fixed, wrapParams) {
  const idx = { u1: 0, v1: 1, u2: 2, v2: 3 };
  let params = [u1, v1, u2, v2];
  if (wrapParams) wrapParams(params);
  const fixedIdx = fixed ? idx[fixed.which] : -1;
  for (let iter = 0; iter < 20; iter++) {
    const [pu1, pv1, pu2, pv2] = params;
    const a = surfacePointAndPartials(srf1, pu1, pv1);
    const b = surfacePointAndPartials(srf2, pu2, pv2);
    const posR = sub(a.point, b.point);
    const cols = [a.su, a.sv, scale(b.su, -1), scale(b.sv, -1)];
    let rows, r, freeIdx;
    if (fixedIdx >= 0) {
      freeIdx = [0, 1, 2, 3].filter((i) => i !== fixedIdx);
      // 3x3: rows[eqI][unknownI] = cols[unknownI][eqI] — one row per real
      // position equation (S1-S2=0), one column per remaining free unknown.
      rows = [0, 1, 2].map((eqI) => freeIdx.map((unkI) => cols[unkI][eqI]));
      r = [posR[0], posR[1], posR[2]];
    } else {
      freeIdx = [0, 1, 2, 3];
      const planeRow = [dot(a.su, T), dot(a.sv, T), 0, 0];
      rows = [
        [cols[0][0], cols[1][0], cols[2][0], cols[3][0]],
        [cols[0][1], cols[1][1], cols[2][1], cols[3][1]],
        [cols[0][2], cols[1][2], cols[2][2], cols[3][2]],
        planeRow,
      ];
      r = [posR[0], posR[1], posR[2], dot(sub(a.point, Qpred), T)];
    }
    const delta = solveSquareSystem(rows, r.map((x) => -x));
    if (!delta) return null;
    for (let k = 0; k < freeIdx.length; k++) params[freeIdx[k]] += delta[k];
    if (wrapParams) wrapParams(params);
    const resid = length(sub(
      surfacePoint(srf1, params[0], params[1]),
      surfacePoint(srf2, params[2], params[3]),
    ));
    if (resid < 1e-9) break;
  }
  const finalResid = length(sub(
    surfacePoint(srf1, params[0], params[1]),
    surfacePoint(srf2, params[2], params[3]),
  ));
  return { u1: params[0], v1: params[1], u2: params[2], v2: params[3], residual: finalResid };
}

// Distance from a point to the SEGMENT a..b in 3D. The spatial twin of the
// projection `jointDistanceToComponent` already does in joint-parameter space
// further down this file, and it exists for the same measured reason: asking
// how near a point is to a marched curve by testing its SAMPLES makes the
// answer depend on how long the step happened to be.
function distPointToSegment(p, a, b) {
  const e = sub(b, a);
  const den = dot(e, e);
  if (den <= 1e-30) return length(sub(p, a)); // degenerate segment
  const t = Math.max(0, Math.min(1, dot(sub(p, a), e) / den));
  return length(sub(p, [a[0] + e[0] * t, a[1] + e[1] * t, a[2] + e[2] * t]));
}

// March from `seed` in ONE direction (sign +1/-1 picks which way along the
// tangent the walk goes). Stops on: (a) closed-loop return to the seed
// (interior loop, the first case), (b) a genuine surface boundary
// exit on a NON-closed direction (open curve, the second case —
// resolved to an EXACT boundary-crossing point via a reduced 3-unknown
// corrector, not just clamped-and-stopped), (c) near-tangent degeneracy
// (honest refusal at the near-tangent wall), (d) a step-budget cap (refusal
// — a genuinely runaway or pathological march, never silently truncated and
// called done).
function marchDirection(srf1, srf2, seed, sign, opts) {
  const d1 = domainOf(srf1), d2 = domainOf(srf2);
  const { closedU: c1u, closedV: c1v } = surfaceClosure(srf1);
  const { closedU: c2u, closedV: c2v } = surfaceClosure(srf2);
  // Fold every CLOSED direction back into its own domain after each Newton
  // step, so the corrector never converges in extrapolated territory it will
  // then be wrapped out of — see correctorStep's own note.
  const wrapParams = (p) => {
    if (c1u) p[0] = wrapParam(p[0], d1.uMin, d1.uMax);
    if (c1v) p[1] = wrapParam(p[1], d1.vMin, d1.vMax);
    if (c2u) p[2] = wrapParam(p[2], d2.uMin, d2.uMax);
    if (c2v) p[3] = wrapParam(p[3], d2.vMin, d2.vMax);
  };
  // `pairScale`, not `scale` — `scale` is vec3.mjs's own vector helper,
  // imported at the top of this file and used throughout marchDirection.
  const pairScale = opts.pairScale ?? ssiPairScale(srf1, srf2);
  const explicitStep = opts.stepLen != null;
  const stepMin = pairScale * SSI_STEP_MIN_FRAC;
  const stepMax = Math.min(
    pairScale * SSI_STEP_MAX_FRAC,
    pairScale / (ssiMaxSpanCount(srf1, srf2) * SSI_SAMPLES_PER_SPAN),
  );
  // The opening step obeys the same ceiling — a surface dense enough in
  // spans can want a first step finer than the size-derived default.
  let stepLen = explicitStep ? opts.stepLen : Math.min(pairScale * SSI_STEP_FRAC_INIT, stepMax);
  const maxSteps = opts.maxSteps ?? SSI_MAX_STEPS;
  const arcBudget = opts.arcBudget ?? pairScale * SSI_MAX_ARC_FRAC;
  const minStepsForClosure = 3;
  let arcUsed = 0;

  let u1 = seed.u1, v1 = seed.v1, u2 = seed.u2, v2 = seed.v2;
  let prevT = null;
  const samples = [{ u1, v1, u2, v2, point: seed.point }];

  for (let step = 0; step < maxSteps; step++) {
    if (arcUsed > arcBudget) {
      return { status: 'error', reason: `the intersection curve did not close or reach a boundary within ${SSI_MAX_ARC_FRAC}x the surfaces' own size of marched arc length — refusing rather than guessing this is done`, samples };
    }
    const a = surfacePointAndPartials(srf1, u1, v1);
    const b = surfacePointAndPartials(srf2, u2, v2);
    const n1 = safeNormalize(cross(a.su, a.sv));
    const n2 = safeNormalize(cross(b.su, b.sv));
    if (!n1 || !n2) return { status: 'error', reason: 'surface normal is undefined here (a pole) — SSI refuses rather than guessing a marching direction', samples };
    const crossN = cross(n1, n2);
    if (length(crossN) < TANGENT_SINE_THRESHOLD) {
      return { status: 'tangent', reason: 'the two surfaces are nearly tangent here — SSI refuses rather than attempting to resolve a degenerate crossing', samples };
    }
    let T = safeNormalize(crossN);
    T = scale(T, sign);
    if (prevT && dot(T, prevT) < 0) T = scale(T, -1);
    // ADAPT ON THE TURN JUST TAKEN. Measured against the PREVIOUS step's
    // tangent, so the very first step of a march uses the opening size-
    // derived guess and every step after it is informed by real curvature.
    if (!explicitStep && prevT) {
      const turn = Math.acos(Math.max(-1, Math.min(1, dot(T, prevT))));
      if (turn > SSI_TARGET_TURN_RAD) stepLen = Math.max(stepMin, stepLen * Math.max(0.5, SSI_TARGET_TURN_RAD / turn));
      else if (turn < SSI_TARGET_TURN_RAD * 0.25) stepLen = Math.min(stepMax, stepLen * 1.5);
    }
    prevT = T;
    // A FAILED CORRECTION IS A STEP-SIZE QUESTION FIRST, A REFUSAL SECOND.
    // The predictor throws the guess a full step along the tangent; where the
    // curve turns hard, or the joint Jacobian is momentarily ill-conditioned,
    // that guess can land outside Newton's basin. Giving up there refuses a
    // genuinely findable component — measured directly: an organic pair's
    // third real intersection read as a refusal at the ordinary step
    // and marched cleanly at a quarter of it. Halving and retrying is the
    // standard remedy and costs nothing on a step that converges first try.
    let corrected = null;
    let attemptStep = stepLen;
    let takenStep = stepLen;
    for (let attempt = 0; attempt < SSI_CORRECTOR_RETRIES; attempt++) {
      const Qpred = add(a.point, scale(T, attemptStep));
      const c = correctorStep(srf1, srf2, u1, v1, u2, v2, Qpred, T, null, wrapParams);
      if (c && c.residual <= 1e-4) { corrected = c; takenStep = attemptStep; break; }
      if (attemptStep <= stepMin) break;
      attemptStep = Math.max(stepMin, attemptStep * 0.5);
    }
    if (!corrected) {
      return { status: 'error', reason: 'march correction failed to converge (singular Jacobian or step too large), even after retrying at a shorter step', samples };
    }
    // THE SHORTENING IS PER-STEP, NOT A RATCHET. On the adaptive path the
    // shortened step becomes the new working step and the turn-based rule
    // above regrows it. On an EXPLICIT step there is no regrowth rule, so
    // carrying the shortening forward would mean one hard corner silently
    // downgrading every remaining step of the march — the caller's own value
    // is restored instead, and only the step that actually failed is short.
    if (!explicitStep) stepLen = takenStep;
    // Closure tolerance rides the step ACTUALLY TAKEN rather than a constant
    // — once the step is measured, everything derived from
    // it has to be measured too, or the two silently disagree about what
    // "close enough to have come back" means. Rides `takenStep`, not
    // `stepLen`: a retry legitimately shortens the step this iteration took,
    // and an explicit step has already been restored by the line above.
    const closureTol = opts.closureTol ?? takenStep * 0.5;

    let { u1: nu1, v1: nv1, u2: nu2, v2: nv2 } = corrected;
    // Wrap a CLOSED direction's own seam instead of treating it as a boundary.
    if (c1u) nu1 = wrapParam(nu1, d1.uMin, d1.uMax);
    if (c1v) nv1 = wrapParam(nv1, d1.vMin, d1.vMax);
    if (c2u) nu2 = wrapParam(nu2, d2.uMin, d2.uMax);
    if (c2v) nv2 = wrapParam(nv2, d2.vMin, d2.vMax);

    const clamp1 = clampToBoundary(nu1, nv1, d1, c1u, c1v);
    const clamp2 = clampToBoundary(nu2, nv2, d2, c2u, c2v);
    if (clamp1.hit || clamp2.hit) {
      // Resolve the EXACT boundary crossing: fix the offending parameter at
      // its true boundary value and re-solve the other 3 (reduced corrector).
      const onSrf1 = !!clamp1.hit;
      const which = onSrf1 ? (clamp1.hit.which === 'u' ? 'u1' : 'v1') : (clamp2.hit.which === 'u' ? 'u2' : 'v2');
      // Only `which` is passed, and that is deliberate rather than an
      // omission: the reduced corrector uses it solely to drop that parameter
      // from the free set, and the parameter's VALUE already arrives inside
      // the guess below (the clamped coordinate is what the guess is built
      // from). A `value` field here would be read by nothing and would have
      // to be kept in step with the guess by hand — two ways of saying the
      // same number is how the two quietly come to disagree.
      const guess = onSrf1
        ? { u1: clamp1.u, v1: clamp1.v, u2: nu2, v2: nv2 }
        : { u1: nu1, v1: nv1, u2: clamp2.u, v2: clamp2.v };
      const exact = correctorStep(srf1, srf2, guess.u1, guess.v1, guess.u2, guess.v2, null, null, { which }, wrapParams);
      if (exact && exact.residual < 1e-4) {
        samples.push({ u1: exact.u1, v1: exact.v1, u2: exact.u2, v2: exact.v2, point: pairPoint(srf1, exact.u1, exact.v1, srf2, exact.u2, exact.v2) });
      } else {
        samples.push({ u1: nu1, v1: nv1, u2: nu2, v2: nv2, point: pairPoint(srf1, nu1, nv1, srf2, nu2, nv2) });
      }
      return { status: 'boundary', exitEdge: onSrf1 ? { surface: 1, ...clamp1.hit } : { surface: 2, ...clamp2.hit }, samples };
    }

    u1 = nu1; v1 = nv1; u2 = nu2; v2 = nv2;
    arcUsed += stepLen;
    const prevPt = samples[samples.length - 1].point;
    const pt = pairPoint(srf1, u1, v1, srf2, u2, v2);
    samples.push({ u1, v1, u2, v2, point: pt });

    // Closure asks whether the marched POLYLINE came back past the seed, not
    // whether a SAMPLE happened to land near it. Testing the sample alone is
    // marginal BY CONSTRUCTION: with samples a step apart, the nearest one to
    // the seed can sit a full half-step away, and the tolerance IS half a
    // step — so a seed falling midway between two samples is a dead tie
    // decided by float noise. Measured on a real three-sphere fixture: the
    // two samples straddling the seed read 0.6140 and 0.6145 against a
    // tolerance of 0.6141 and 0.6142. The march missed, kept going, and
    // closed a full lap later — a doubly-traced loop that self-overlaps in
    // UV, which the face arrangement then cuts into hundreds of fragments
    // and the sew reports as NON-MANIFOLD, three stages from the cause.
    // Segment distance has no such aliasing: a polyline passing the seed
    // comes within its own chord sagitta of it regardless of where the
    // samples fall. This can only ever close EARLIER than the old test
    // (segment distance <= endpoint distance, always), never later.
    if (step >= minStepsForClosure && distPointToSegment(seed.point, prevPt, pt) < closureTol) {
      samples[samples.length - 1] = { ...samples[0] }; // snap to the exact seed for a bit-exact closed loop
      return { status: 'closed', samples };
    }
  }
  return { status: 'error', reason: `did not close or reach a boundary within ${maxSteps} march steps — refusing rather than guessing this is done`, samples };
}

// The real entry point (Tier C's numerical core). Returns:
//   { ok:true, closed:true,  samples }               — one interior loop
//   { ok:true, closed:false, samples, startEdge, endEdge } — one open curve
//   { ok:false, reason }                               — honest refusal
export function intersectSurfaces(srf1, srf2, opts = {}) {
  assertSurface(srf1, 'intersectSurfaces'); assertSurface(srf2, 'intersectSurfaces');
  const seedTolerance = opts.seedTolerance ?? 1e-4;
  const pairScale = ssiPairScale(srf1, srf2);
  opts = { ...opts, pairScale };
  const seed = seedSurfaceIntersection(srf1, srf2, opts);
  if (seed.distance > seedTolerance) {
    return { ok: false, reason: `no intersection found — the closest approach found between the two surfaces is ${seed.distance.toFixed(6)}, above tolerance ${seedTolerance}` };
  }
  const forward = marchDirection(srf1, srf2, seed, +1, opts);
  if (forward.status === 'error') return { ok: false, reason: forward.reason };
  if (forward.status === 'tangent') return { ok: false, reason: forward.reason };
  if (forward.status === 'closed') return { ok: true, closed: true, samples: forward.samples };

  const backward = marchDirection(srf1, srf2, seed, -1, opts);
  if (backward.status === 'error') return { ok: false, reason: backward.reason };
  if (backward.status === 'tangent') return { ok: false, reason: backward.reason };
  if (backward.status === 'closed') return { ok: true, closed: true, samples: backward.samples };

  const backSamples = backward.samples.slice(1).reverse(); // drop the duplicated seed, reverse to run start->seed
  const samples = [...backSamples, ...forward.samples];
  return { ok: true, closed: false, samples, startEdge: backward.exitEdge, endEdge: forward.exitEdge };
}

// ============================================================================
// COMPLETE INTERSECTION — every component, not just the nearest one.
//
// `intersectSurfaces` above seeds from a coarse grid over srf1 and keeps
// whichever point is globally CLOSEST, so it finds ONE component and reports
// success. That is the quiet failure a boolean cannot survive: a missed
// component does not produce a slightly wrong solid, it produces one that
// fails three stages later as an unclosed shell, with nothing at the failure
// site pointing back at the curve that was never found.
//
// Sederberg & Meyers 1988 splits completeness in two. Every branch that
// reaches a patch BOUNDARY is found by intersecting the four boundary curves
// against the opposing surface (curvesurface.mjs) — that half is exhaustive
// by construction, because a branch reaching a boundary must cross one of
// those curves. A branch that reaches NO boundary is a closed interior loop,
// and only a normal-cone argument can prove none exists.
//
// WHAT IS BUILT HERE IS THE FIRST HALF, PLUS AN HONEST BEST EFFORT AT THE
// SECOND. Boundary seeding is complete for boundary-reaching branches.
// Interior loops are sought by keeping every LOCAL MINIMUM of the coarse
// grid rather than only the global best — a real improvement over one seed,
// and enough to find loops the grid resolves, but NOT a proof. A loop
// smaller than the grid spacing can still be missed, and the returned
// `loopSearchProven: false` says so rather than letting a caller assume
// otherwise. The normal-cone test and its subdivision (Phase 1b/1c)
// are what would upgrade that flag, and they are not built.
// ============================================================================

const SSI_SEED_LOCAL_MIN_FRAC = 0.02; // a grid sample counts as a candidate when its closest approach is under this fraction of the pair's own size
const SSI_COMPONENT_DEDUPE = 0.03;    // normalized joint-parameter distance below which a seed lies on a known component

function wrapDelta(d, span, closed) {
  if (!closed || !(span > 0)) return d;
  let x = d % span;
  if (x > span / 2) x -= span;
  if (x < -span / 2) x += span;
  return x;
}

// The offset from `b` to `a` in joint parameter space, normalized per
// direction and wrapped in any CLOSED direction. Deliberately joint-
// PARAMETER, not 3D: two genuinely different sheets can pass within microns
// in space while being far apart in (u1,v1,u2,v2), and merging those would
// silently drop a real component.
function jointDelta(a, b, d1, d2, c1, c2) {
  const su1 = d1.uMax - d1.uMin, sv1 = d1.vMax - d1.vMin;
  const su2 = d2.uMax - d2.uMin, sv2 = d2.vMax - d2.vMin;
  return [
    wrapDelta(a.u1 - b.u1, su1, c1.closedU) / (su1 || 1),
    wrapDelta(a.v1 - b.v1, sv1, c1.closedV) / (sv1 || 1),
    wrapDelta(a.u2 - b.u2, su2, c2.closedU) / (su2 || 1),
    wrapDelta(a.v2 - b.v2, sv2, c2.closedV) / (sv2 || 1),
  ];
}

// How far a seed sits from a marched component, measured against the
// component's POLYLINE rather than its sample points.
//
// Comparing to points alone makes the answer depend on marching density: a
// seed landing midway between two samples reads as far away purely because
// the step was long. Found by fixture — a plane/cylinder pair returned the
// same two curves TWICE, once from a boundary seed and once from an interior
// one, at a measured 0.031 against a 0.03 threshold, entirely an artifact of
// sample spacing. Distance to the SEGMENT is spacing-independent, so the
// tolerance can stay tight and mean what it says.
function jointDistanceToComponent(seed, comp, d1, d2, c1, c2) {
  let best = Infinity;
  const deltas = comp.samples.map((smp) => jointDelta(seed, smp, d1, d2, c1, c2));
  for (let i = 0; i < deltas.length; i++) {
    const a = deltas[i];
    const da = Math.hypot(a[0], a[1], a[2], a[3]);
    if (da < best) best = da;
    const b = deltas[i + 1];
    if (!b) continue;
    // Distance from the origin (the seed) to the segment a..b.
    let num = 0, den = 0;
    for (let k = 0; k < 4; k++) { const e = b[k] - a[k]; num += -a[k] * e; den += e * e; }
    if (den <= 1e-30) continue;
    const t = Math.max(0, Math.min(1, num / den));
    let d2sum = 0;
    for (let k = 0; k < 4; k++) { const x = a[k] + (b[k] - a[k]) * t; d2sum += x * x; }
    const d = Math.sqrt(d2sum);
    if (d < best) best = d;
  }
  return best;
}

// A border curve's own parameter IS one of its surface's parameters:
// extractIsocurveU fixes u and returns a curve over knotsV, so its t is v;
// extractIsocurveV is the mirror. No inversion needed — that identity is
// what makes boundary seeding cheap.
function borderParamToSurfaceUV(edge, srf, t) {
  const d = domainOf(srf);
  if (edge === 'uMin') return { u: d.uMin, v: t };
  if (edge === 'uMax') return { u: d.uMax, v: t };
  if (edge === 'vMin') return { u: t, v: d.vMin };
  return { u: t, v: d.vMax };
}

export function collectSeeds(srf1, srf2, opts = {}) {
  const d1 = domainOf(srf1), d2 = domainOf(srf2);
  const pairScale = opts.pairScale ?? ssiPairScale(srf1, srf2);
  const tol = opts.seedTolerance ?? 1e-4;
  const seeds = [];

  // --- BOUNDARY SEEDS, both directions -------------------------------------
  for (const [self, other, selfIsFirst] of [[srf1, srf2, true], [srf2, srf1, false]]) {
    for (const b of extractBorderCurves(self)) {
      let hits = [];
      try { hits = curveSurfaceIntersections(b.crv, other, { tolerance: tol }); } catch { hits = []; }
      for (const h of hits) {
        const own = borderParamToSurfaceUV(b.edge, self, h.t);
        seeds.push(selfIsFirst
          ? { u1: own.u, v1: own.v, u2: h.u, v2: h.v, origin: `border ${b.edge} of srf1` }
          : { u1: h.u, v1: h.v, u2: own.u, v2: own.v, origin: `border ${b.edge} of srf2` });
      }
    }
  }

  // --- INTERIOR SEEDS: every local minimum of the coarse grid ---------------
  // The existing single-seed search computes exactly this grid and then
  // throws all but the global best away. Keeping the local minima costs one
  // extra pass over an array already built.
  const gridU = opts.gridU1 ?? Math.max(12, Math.min(32, srf1.ctrlNet.length * 2));
  const gridV = opts.gridV1 ?? Math.max(10, Math.min(24, srf1.ctrlNet[0].length * 2));
  const grid = [];
  for (let i = 0; i <= gridU; i++) {
    const row = [];
    const u1 = d1.uMin + (d1.uMax - d1.uMin) * (i / gridU);
    for (let j = 0; j <= gridV; j++) {
      const v1 = d1.vMin + (d1.vMax - d1.vMin) * (j / gridV);
      const cp = closestPointOnSurface(srf2, surfacePoint(srf1, u1, v1));
      row.push({ u1, v1, u2: cp.u, v2: cp.v, distance: cp.distance });
    }
    grid.push(row);
  }
  // THE GRID ALREADY KNOWS HOW FAR APART THEY ARE. Every cell holds a real
  // point-to-surface distance, and the `nearFrac` gate below discards all of
  // them whenever nothing is close enough to seed — which is exactly the case
  // where the caller has nothing to say and must refuse. Carrying the minimum
  // out costs one comparison over an array already built, and it is the whole
  // difference between "no intersection found" and a distance the user can
  // close. It rides on the returned array as a property, the way a regex
  // match array carries `index`, so that no consumer of the seeds themselves
  // has to change.
  let gridMin = Infinity;
  const nearFrac = pairScale * SSI_SEED_LOCAL_MIN_FRAC;
  for (let i = 0; i <= gridU; i++) {
    for (let j = 0; j <= gridV; j++) {
      const c = grid[i][j];
      if (c.distance < gridMin) gridMin = c.distance;
      if (!(c.distance < nearFrac)) continue;
      let isMin = true;
      for (let di = -1; di <= 1 && isMin; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          if (di === 0 && dj === 0) continue;
          const n = grid[i + di] && grid[i + di][j + dj];
          if (n && n.distance < c.distance) { isMin = false; break; }
        }
      }
      if (isMin) seeds.push({ u1: c.u1, v1: c.v1, u2: c.u2, v2: c.v2, origin: 'interior grid local minimum' });
    }
  }
  // A SAMPLED approach, not a proven minimum — both endpoints genuinely lie
  // on their surfaces, so this is a real distance the pair achieves, but the
  // true minimum can only be smaller. Every message built from it says
  // "found" for that reason.
  seeds.closestApproachFound = Number.isFinite(gridMin) ? gridMin : null;
  return seeds;
}

// Returns:
//   { ok:true, components:[{ closed, samples, startEdge?, endEdge?, origin }],
//     loopSearchProven:false, refusals:[...] }
//   { ok:false, reason }   — nothing found at all
export function intersectSurfacesComplete(srf1, srf2, opts = {}) {
  const d1 = domainOf(srf1), d2 = domainOf(srf2);
  const c1 = surfaceClosure(srf1), c2 = surfaceClosure(srf2);
  const pairScale = ssiPairScale(srf1, srf2);
  const marchOpts = { ...opts, pairScale };
  const seedTolerance = opts.seedTolerance ?? 1e-4;

  const raw = collectSeeds(srf1, srf2, { ...opts, pairScale });
  const components = [];
  const refusals = [];

  // THE CLOSEST SEED THIS LOOP TURNED AWAY. Every rejected candidate has
  // already been refined to its true closest approach, so the number that
  // says how nearly these two surfaces met is computed here and, without
  // this, discarded by a bare `continue` — leaving the caller able to say
  // only "no intersection found", which is the one thing the user could
  // already see. The single-surface path below this file's own
  // intersectSurfaces has named both numbers for as long as it has existed;
  // this is the same report for the multi-component path.
  let closestRejected = Infinity;

  for (const cand of raw) {
    const seed = refineSeedSnap(srf1, srf2, cand);
    if (seed.distance > seedTolerance) {
      if (seed.distance < closestRejected) closestRejected = seed.distance;
      continue;
    }
    // Already on a component we have marched? Compare against every sample of
    // every known component, not just their seeds — a boundary seed and an
    // interior seed on the SAME curve are far apart along it.
    let covered = false;
    for (const comp of components) {
      if (jointDistanceToComponent(seed, comp, d1, d2, c1, c2) < SSI_COMPONENT_DEDUPE) { covered = true; break; }
    }
    if (covered) continue;

    const forward = marchDirection(srf1, srf2, seed, +1, marchOpts);
    if (forward.status === 'closed') { components.push({ closed: true, samples: forward.samples, origin: cand.origin }); continue; }
    if (forward.status === 'error' || forward.status === 'tangent') { refusals.push({ origin: cand.origin, reason: forward.reason }); continue; }
    const backward = marchDirection(srf1, srf2, seed, -1, marchOpts);
    if (backward.status === 'closed') { components.push({ closed: true, samples: backward.samples, origin: cand.origin }); continue; }
    if (backward.status === 'error' || backward.status === 'tangent') { refusals.push({ origin: cand.origin, reason: backward.reason }); continue; }
    const samples = [...backward.samples.slice(1).reverse(), ...forward.samples];
    components.push({ closed: false, samples, startEdge: backward.exitEdge, endEdge: forward.exitEdge, origin: cand.origin });
  }

  if (!components.length) {
    // Three genuinely different failures, and collapsing them costs the
    // reader the only fact that distinguishes what to do next: seeds were
    // found and marching failed; seeds were found and all were too far
    // apart (say how far — that is a distance the user can close); or no
    // candidate was generated at all, which is not a near miss and must
    // not be dressed up as one.
    // A refined seed beats a sampled grid cell where both exist, since it has
    // been driven to a true local minimum; the grid value is the fallback for
    // the case that produced no candidate at all, which is precisely the case
    // with no refined seed to offer.
    const approach = Number.isFinite(closestRejected) ? closestRejected : raw.closestApproachFound;
    if (refusals.length) return { ok: false, reason: `every intersection seed found was refused — ${refusals[0].reason}`, closestApproach: approach ?? null, seedTolerance };
    if (approach != null) {
      return { ok: false, reason: `no intersection found — the closest approach found between the two surfaces is ${approach.toFixed(6)}, above tolerance ${seedTolerance}`, closestApproach: approach, seedTolerance };
    }
    return { ok: false, reason: 'no intersection found between these two surfaces — no seed candidate formed anywhere', closestApproach: null, seedTolerance };
  }
  return { ok: true, components, refusals, loopSearchProven: false };
}
