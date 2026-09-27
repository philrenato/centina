// Surface-surface intersection (SSI). Standard Barnhill-Kersey-style
// marching in joint parameter space (u1,v1,u2,v2): seed by coarse search +
// Gauss-Newton snap, then march with a predictor (tangent = N1 x N2) /
// corrector (Newton on the joint parameter vector, closed by a
// marching-plane constraint) pair, refusing at tangency rather than
// attempting to resolve it.
//
// Scope: intersectSurfaces finds one intersection curve component —
// whichever the seed search lands nearest — either a closed interior loop
// or an open curve running to one surface's own parametric boundary.
// intersectSurfacesComplete (below) seeds every component it can find; a
// loop smaller than the seeding grid can still be missed. Both refuse
// outright at a near-tangent region rather than attempting to resolve
// degenerate/higher-multiplicity crossings.

import { surfacePoint, surfacePointAndPartials, assertSurface } from './surface.mjs';
import { surfaceClosure } from './surface.mjs';
import { closestPointOnSurface } from './surface.mjs';
import { add, sub, scale, dot, cross, length } from './vec3.mjs';
import { extractBorderCurves } from './isocurve.mjs';
import { curveSurfaceIntersections } from './curvesurface.mjs';

// Small dense linear solve (Gauss-Jordan with partial pivoting) — every
// system this module builds is at most 4x4 (the joint-parameter Newton
// correction). Returns null for a singular system; the caller treats that
// as "correction failed" rather than dividing by ~0.
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

// The emitted sample point is the midpoint of both surfaces' own
// evaluations, not srf1's alone. The corrector drives
// |S1(u1,v1) - S2(u2,v2)| below its own residual tolerance, never to zero:
// the two evaluations disagree by that residual. Emitting S1 alone makes
// every sample exact on srf1 and off srf2 by the full residual, so which
// surface gets the error depends on argument order. The midpoint splits
// the disagreement evenly and is order-independent by construction.
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

// Seed finding, simplified to the single-nearest-approach case rather than
// a full triangle-mesh facet search — the coarse grid here plays the same
// "get within the true intersection's Newton basin" role a
// tessellate-and-facet-test pass would, with the limitation named above: a
// component whose closest approach isn't found by this grid can be missed.
// Stage 1 samples srf1 on a coarse grid and calls closestPointOnSurface
// (surface.mjs) against srf2 for each sample. Stage 2 is a 4-unknown
// Gauss-Newton snap, lightly damped (Levenberg-Marquardt) (minimize
// |S1(u1,v1)-S2(u2,v2)|^2 over all four parameters at once — the
// generalization of closestPointOnSurface's 2-unknown Gauss-Newton to two
// independent surfaces) that pins the coarse seed onto the true
// zero-distance intersection point before marching starts.
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
    // rank-deficient once the seed is close to an intersection curve. The
    // residual r=S1-S2 is 3 equations in 4 unknowns — near any point on the
    // intersection curve, r=0 holds along a whole 1-parameter family of
    // (u1,v1,u2,v2) (the curve's own tangent direction), not just one
    // isolated point, so J^T J is singular along that direction (plain
    // Gauss-Newton meets a singular normal-equation matrix on an ordinary
    // cylinder-cylinder seed). A small diagonal damping term regularizes
    // that flat direction and converges normally elsewhere.
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
    if (nd > curDist + 1e-10) break; // reject a worsening step, matching closestPointOnSurface's guard
    u1 = nu1; v1 = nv1; u2 = nu2; v2 = nv2; curDist = nd;
    if (nd < 1e-12) break;
  }
  return { u1, v1, u2, v2, distance: curDist, point: pairPoint(srf1, u1, v1, srf2, u2, v2) };
}

const TANGENT_SINE_THRESHOLD = 0.05; // |N1 x N2| below this = near-tangent, refuse rather than resolve
const CLAMP_EPS = 1e-9;

// March step — derived from the geometry, not a constant.
//
// A fixed step with a fixed step count is a fixed budget of arc length at
// any scale: a plane cutting a 30-unit-radius cylinder produces a circle of
// circumference 188.5, which a 0.25 step and 400 steps cannot close. So the
// step starts as a fraction of the two surfaces' own size and then
// adapts to the curve's own turning, which is the quantity that
// governs both risks: too large a step invites the component jumping
// Krishnan & Manocha name (landing on a different branch and marching on),
// and too small a step burns budget. Bounding the turn per
// step bounds the chord's deviation from the true curve directly, and it
// costs nothing extra — the unit tangent is already computed every step.
//
// An explicit `opts.stepLen` disables the turn-based adaptation entirely.
// It is honored verbatim on every step except one where the corrector
// fails to converge at it: that single step is retried shorter (see
// SSI_CORRECTOR_RETRIES) and the caller's own value is restored for the next
// one. The shortening is a per-step remedy for one hard corner, never a
// permanent change to the step the caller asked for.
const SSI_STEP_FRAC_INIT = 1 / 200;   // opening step, as a fraction of the pair's own size
const SSI_STEP_MIN_FRAC = 1 / 20000;  // floor, so a tight pinch cannot stall the march at zero
const SSI_STEP_MAX_FRAC = 1 / 40;     // ceiling, so a near-straight run cannot stride past a branch
const SSI_SAMPLES_PER_SPAN = 12;       // minimum samples across the tightest knot span either surface has
const SSI_TARGET_TURN_RAD = 5 * Math.PI / 180;
const SSI_MAX_ARC_FRAC = 80;          // refuse past this multiple of the pair's own size of marched arc
const SSI_MAX_STEPS = 20000;          // backstop only — the arc budget above is the real stop
const SSI_CORRECTOR_RETRIES = 8;      // halvings a failed correction is allowed before the march refuses

// The pair's own size: the diagonal of the combined control-net bounding
// box. A control net bounds its own surface (the convex-hull property), so
// this is an upper bound on the geometry's extent, not a sample-
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

// Span-aware step ceiling — the step must never stride across a whole knot
// span of either surface without sampling inside it.
//
// The adaptive step above keys on the curve's own turning, which is
// the right signal for a smooth run and the wrong one at a curvature
// break: a rational multi-span wall (a filleted profile, a swept rail) is
// nearly straight through each span and turns sharply only where two spans
// meet, so a turn-driven step grows to its ceiling and then strides
// over a fillet arc — the samples land on either side and the fitted
// curve cuts the corner, with no turn ever measured to react to.
//
// This is the same guarantee as tessellationVSamples' "N samples per span"
// floor (surface.mjs) and denseRailFrames' MIN_SPAN_SAMPLES (sweep.mjs).
// Here it is a ceiling on the marching step rather than a floor on a
// sample count, because the march has no span index to count against — but
// it is the same guarantee: at least SSI_SAMPLES_PER_SPAN samples across the
// tightest span either surface has.
//
// A single-span pair (a plain box face, an untrimmed plane) has spanCount 1,
// which puts this ceiling far above the ordinary SSI_STEP_MAX_FRAC one, so
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
  // if this step exits a non-closed direction's domain (a real
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

// One predictor/corrector march step. `fixed` optionally
// pins one of the four joint parameters (for solving an exact boundary
// crossing — see marchDirection below): when set, that parameter is held
// constant and the marching-plane constraint row is dropped entirely, since
// fixing one unknown already leaves exactly 3 unknowns for the 3 real
// position equations S1-S2=0 (no additional constraint needed or wanted —
// adding the plane row back would over-determine a perfectly well-posed
// system).
//
// `wrapParams` (optional) folds each closed direction's parameter back into
// its own domain after every Newton step. Without it the corrector can
// converge outside the domain, where a clamped B-spline extrapolates along
// its first (or last) span's polynomial rather than continuing around the
// seam. The residual it reports there is small, so the step is accepted;
// the caller then wraps the parameter, and the sample lands on a different
// point (on a closed extruded wall, the seam-crossing sample sits ~2e-2
// off both surfaces). Wrapping inside the loop keeps every evaluation on
// the surface, so the answer the corrector converges to is the answer the
// caller keeps.
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

// Distance from a point to the segment a..b in 3D. The spatial counterpart
// of the projection `jointDistanceToComponent` does in joint-parameter
// space further down this file, for the same reason: asking how near a
// point is to a marched curve by testing its samples makes the answer
// depend on the step length.
function distPointToSegment(p, a, b) {
  const e = sub(b, a);
  const den = dot(e, e);
  if (den <= 1e-30) return length(sub(p, a)); // degenerate segment
  const t = Math.max(0, Math.min(1, dot(sub(p, a), e) / den));
  return length(sub(p, [a[0] + e[0] * t, a[1] + e[1] * t, a[2] + e[2] * t]));
}

// March from `seed` in one direction (sign +1/-1 picks which way along the
// tangent the walk goes). Stops on: (a) closed-loop return to the seed
// (interior loop), (b) a surface boundary exit on a non-closed direction
// (open curve — resolved to an exact boundary-crossing point via a reduced
// 3-unknown corrector, not just clamped-and-stopped), (c) near-tangent
// degeneracy (refusal), (d) a step-budget cap (refusal — a runaway march,
// never silently truncated and called done).
function marchDirection(srf1, srf2, seed, sign, opts) {
  const d1 = domainOf(srf1), d2 = domainOf(srf2);
  const { closedU: c1u, closedV: c1v } = surfaceClosure(srf1);
  const { closedU: c2u, closedV: c2v } = surfaceClosure(srf2);
  // Fold every closed direction back into its own domain after each Newton
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
    // Adapt on the turn just taken. Measured against the previous step's
    // tangent, so the first step of a march uses the opening size-derived
    // guess and every step after it is informed by curvature.
    if (!explicitStep && prevT) {
      const turn = Math.acos(Math.max(-1, Math.min(1, dot(T, prevT))));
      if (turn > SSI_TARGET_TURN_RAD) stepLen = Math.max(stepMin, stepLen * Math.max(0.5, SSI_TARGET_TURN_RAD / turn));
      else if (turn < SSI_TARGET_TURN_RAD * 0.25) stepLen = Math.min(stepMax, stepLen * 1.5);
    }
    prevT = T;
    // A failed correction is treated as a step-size problem before it is a
    // refusal. The predictor throws the guess a full step along the tangent;
    // where the curve turns hard, or the joint Jacobian is momentarily
    // ill-conditioned, that guess can land outside Newton's basin, though
    // the component marches cleanly at a shorter step. Halving and retrying
    // costs nothing on a step that converges first try.
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
    // The shortening is per-step, not a ratchet. On the adaptive path the
    // shortened step becomes the new working step and the turn-based rule
    // above regrows it. On an explicit step there is no regrowth rule, so
    // carrying the shortening forward would shorten every remaining step of
    // the march — the caller's own value is restored instead, and only the
    // step that failed is short.
    if (!explicitStep) stepLen = takenStep;
    // Closure tolerance follows the step taken rather than a constant, so
    // the two agree about what "close enough to have come back" means.
    // Uses `takenStep`, not `stepLen`: a retry shortens the step this
    // iteration took, and an explicit step has already been restored by the
    // line above.
    const closureTol = opts.closureTol ?? takenStep * 0.5;

    let { u1: nu1, v1: nv1, u2: nu2, v2: nv2 } = corrected;
    // Wrap a closed direction's own seam instead of treating it as a boundary.
    if (c1u) nu1 = wrapParam(nu1, d1.uMin, d1.uMax);
    if (c1v) nv1 = wrapParam(nv1, d1.vMin, d1.vMax);
    if (c2u) nu2 = wrapParam(nu2, d2.uMin, d2.uMax);
    if (c2v) nv2 = wrapParam(nv2, d2.vMin, d2.vMax);

    const clamp1 = clampToBoundary(nu1, nv1, d1, c1u, c1v);
    const clamp2 = clampToBoundary(nu2, nv2, d2, c2u, c2v);
    if (clamp1.hit || clamp2.hit) {
      // Resolve the exact boundary crossing: fix the offending parameter at
      // its true boundary value and re-solve the other 3 (reduced corrector).
      const onSrf1 = !!clamp1.hit;
      const which = onSrf1 ? (clamp1.hit.which === 'u' ? 'u1' : 'v1') : (clamp2.hit.which === 'u' ? 'u2' : 'v2');
      // Only `which` is passed: the reduced corrector uses it solely to drop
      // that parameter from the free set, and the parameter's value arrives
      // inside the guess below (the clamped coordinate is what the guess is
      // built from).
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

    // Closure asks whether the marched polyline came back past the seed, not
    // whether a sample landed near it. Testing the sample alone is
    // marginal by construction: with samples a step apart, the nearest one to
    // the seed can sit a full half-step away, and the tolerance is half a
    // step — so a seed falling midway between two samples is a tie
    // decided by float noise. A miss closes a full lap later — a
    // doubly-traced loop that self-overlaps in UV, which the face
    // arrangement cuts into fragments and the sew reports as non-manifold.
    // Segment distance has no such aliasing: a polyline passing the seed
    // comes within its own chord sagitta of it regardless of where the
    // samples fall. It closes no later than a sample-distance test would
    // (segment distance <= endpoint distance, always).
    if (step >= minStepsForClosure && distPointToSegment(seed.point, prevPt, pt) < closureTol) {
      samples[samples.length - 1] = { ...samples[0] }; // snap to the exact seed for a bit-exact closed loop
      return { status: 'closed', samples };
    }
  }
  return { status: 'error', reason: `did not close or reach a boundary within ${maxSteps} march steps — refusing rather than guessing this is done`, samples };
}

// Single-component entry point. Returns:
//   { ok:true, closed:true,  samples }               — one interior loop
//   { ok:true, closed:false, samples, startEdge, endEdge } — one open curve
//   { ok:false, reason }                               — refusal
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

// Complete intersection — every component, not just the nearest one.
//
// `intersectSurfaces` above seeds from a coarse grid over srf1 and keeps
// whichever point is globally closest, so it finds one component and reports
// success. A boolean needs every component: a missed one produces a solid
// that fails later as an unclosed shell, with nothing at the failure
// site pointing back at the curve that was never found.
//
// Sederberg & Meyers 1988 splits completeness in two. Every branch that
// reaches a patch boundary is found by intersecting the four boundary curves
// against the opposing surface (curvesurface.mjs) — that half is exhaustive
// by construction, because a branch reaching a boundary must cross one of
// those curves. A branch that reaches no boundary is a closed interior loop,
// and only a normal-cone argument can prove none exists.
//
// Built here: the first half, plus a best effort at the second. Boundary
// seeding is complete for boundary-reaching branches. Interior loops are
// sought by keeping every local minimum of the coarse grid rather than only
// the global best — enough to find loops the grid resolves, but not a
// proof. A loop smaller than the grid spacing can still be missed, and the
// returned `loopSearchProven: false` says so. Known limitation: the
// normal-cone test and its subdivision, which would make that flag true,
// are not implemented.

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
// direction and wrapped in any closed direction. Joint-parameter, not 3D:
// two different sheets can pass within microns in space while being far
// apart in (u1,v1,u2,v2), and merging those would drop a component.
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
// component's polyline rather than its sample points.
//
// Comparing to points alone makes the answer depend on marching density: a
// seed landing midway between two samples reads as far away because
// the step was long, and the same curve is then marched twice (once from a
// boundary seed, once from an interior one). Distance to the segment is
// spacing-independent, so the tolerance can stay tight.
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

// A border curve's own parameter is one of its surface's parameters:
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
  const d1 = domainOf(srf1);
  const pairScale = opts.pairScale ?? ssiPairScale(srf1, srf2);
  const tol = opts.seedTolerance ?? 1e-4;
  const seeds = [];

  // Boundary seeds, both directions
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

  // Interior seeds: every local minimum of the coarse grid
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
  // Every grid cell holds a point-to-surface distance, and the `nearFrac`
  // filter below discards all of them whenever nothing is close enough to
  // seed — the case where the caller must refuse. The minimum is carried
  // out so the refusal can report a distance rather than only "no
  // intersection found". It rides on the returned array as a property, the
  // way a regex match array carries `index`, so consumers of the seeds
  // themselves are unaffected.
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
  // A sampled approach, not a proven minimum — both endpoints lie
  // on their surfaces, so this is a distance the pair achieves, but the
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

  // The closest seed this loop turned away. Every rejected candidate has
  // been refined to its true closest approach, so this records how nearly
  // the two surfaces met, for the refusal message — the same report
  // intersectSurfaces gives for the single-component path.
  let closestRejected = Infinity;

  for (const cand of raw) {
    const seed = refineSeedSnap(srf1, srf2, cand);
    if (seed.distance > seedTolerance) {
      if (seed.distance < closestRejected) closestRejected = seed.distance;
      continue;
    }
    // Already on a component we have marched? Compare against every sample of
    // every known component, not just their seeds — a boundary seed and an
    // interior seed on the same curve are far apart along it.
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
    // Three different failures, reported apart because each calls for a
    // different next step: seeds were found and marching failed; seeds were
    // found and all were too far apart (say how far); or no candidate was
    // generated at all, which is not a near miss.
    // A refined seed beats a sampled grid cell where both exist, since it has
    // been driven to a true local minimum; the grid value is the fallback for
    // the case that produced no candidate at all, which has no refined seed
    // to offer.
    const approach = Number.isFinite(closestRejected) ? closestRejected : raw.closestApproachFound;
    if (refusals.length) return { ok: false, reason: `every intersection seed found was refused — ${refusals[0].reason}`, closestApproach: approach ?? null, seedTolerance };
    if (approach != null) {
      return { ok: false, reason: `no intersection found — the closest approach found between the two surfaces is ${approach.toFixed(6)}, above tolerance ${seedTolerance}`, closestApproach: approach, seedTolerance };
    }
    return { ok: false, reason: 'no intersection found between these two surfaces — no seed candidate formed anywhere', closestApproach: null, seedTolerance };
  }
  return { ok: true, components, refusals, loopSearchProven: false };
}
