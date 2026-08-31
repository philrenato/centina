// PATCH — A SURFACE FITTED THROUGH SCATTERED CURVES AND POINTS.
//
// This is the one command in the taught set that could not be assembled out of
// what was already here, and it is worth being precise about why, because two
// nearby things look like they should have done the job:
//
//   · `nSidedTangentPatch` (tangentpatch.mjs) fills a hole. It requires N
//     boundary curves that already CLOSE INTO A LOOP, refuses anything else, and
//     INTERPOLATES its boundary exactly.
//   · `boundSurfaceFromLoop` (loft.mjs) is a Coons fill over exactly four
//     touching curves — the app ships it as BoundSrf, and its own tooltip says
//     "Rhino: EdgeSrf".
//
// Patch is the opposite contract on every axis: the input is scattered, need not
// touch, need not close, and may be bare points; the output APPROXIMATES it to a
// tolerance rather than passing through it. So the fit is genuinely new: a coarse
// control net, a least-squares solve against sampled targets, and a stiffness
// term without which the whole thing is ill-posed the moment the data is sparse
// or lopsided.
//
// ⚠⚠ THE STIFFNESS IS NOT A QUALITY KNOB, IT IS WHAT MAKES THE SOLVE POSSIBLE.
// A control point with no sample near it appears in no data row, so its column of
// the normal equations is all zeros and the matrix is singular — the solve does
// not merely fit badly, it fails. The fairness rows give every control point a
// relationship to its neighbours, so an unconstrained one is pulled to the smooth
// continuation of the ones around it instead of being undetermined. That is why
// the default is non-zero and why zero is refused.
//
// ⚠ IT FITS IN THE PLANE THE DATA ITSELF PICKS. The samples are projected onto
// their own best-fit plane (PCA, `fitPlane`) and that projection IS the surface's
// parameter domain. This is the standard choice and it has an honest limit: data
// that folds back over that plane — a C-shape seen edge-on, anything with two
// points at the same (u,v) — cannot be represented by a single-valued height
// field over it, and the refusal below names that rather than returning a
// confident average of the two branches.
import { fitPlane } from './refit.mjs';
import { solveLinearSystem } from './interpolate.mjs';
import { findSpan, basisFuns } from './basis.mjs';
import { surfacePoint } from './surface.mjs';

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len3 = (a) => Math.hypot(a[0], a[1], a[2]);

export const PATCH_REFUSAL = Object.freeze({
  NO_INPUT: 'no-input',
  TOO_FEW: 'too-few-points',
  DEGENERATE_PLANE: 'degenerate-plane',
  FOLDED: 'folded-over-its-own-plane',
  SINGULAR: 'singular-solve',
  BAD_REQUEST: 'bad-request',
});

function refuse(kind, reason) { return { ok: false, kind, reason }; }

// A clamped uniform knot vector for `count` control points at `degree`. Clamped
// so the surface starts and ends at its own boundary rows, which is what makes
// the fitted patch's edge mean something.
function clampedUniformKnots(count, degree) {
  const U = [];
  for (let k = 0; k < count + degree + 1; k++) {
    if (k <= degree) U.push(0);
    else if (k >= count) U.push(1);
    else U.push((k - degree) / (count - degree));
  }
  return U;
}

// SAMPLE THE INPUT INTO TARGETS. A curve contributes points along its length; a
// bare point contributes itself. Everything downstream sees one flat list, so
// "fit through these curves" and "fit through this cloud" are the same problem
// and cannot drift apart.
export function samplePatchInputs({ curves = [], points = [] } = {}, perCurve = 24) {
  const out = [];
  for (const p of points) if (Array.isArray(p) && p.length >= 3 && p.every((c, i) => i > 2 || Number.isFinite(c))) out.push([p[0], p[1], p[2]]);
  for (const crv of curves) {
    if (!crv || !Array.isArray(crv.knots) || !crv.knots.length) continue;
    const u0 = crv.knots[0], u1 = crv.knots[crv.knots.length - 1];
    if (!(u1 > u0)) continue;
    const n = Math.max(2, Math.round(perCurve));
    for (let i = 0; i <= n; i++) {
      const t = u0 + (u1 - u0) * (i / n);
      const pt = curvePointLocal(crv, t);
      if (pt) out.push(pt);
    }
  }
  return out;
}

// A curve evaluator local to this module so Patch does not depend on the app's
// own. Rational, matching the ctrlPts-carry-their-weight convention used
// everywhere else in this kernel.
function curvePointLocal(crv, u) {
  const p = crv.degree, U = crv.knots, P = crv.ctrlPts;
  if (!Array.isArray(P) || !P.length) return null;
  const n = P.length - 1;
  const span = findSpan(n, p, u, U);
  const N = basisFuns(span, u, p, U);
  let x = 0, y = 0, z = 0, w = 0;
  for (let i = 0; i <= p; i++) {
    const cp = P[span - p + i];
    const wi = cp[3] === undefined ? 1 : cp[3];
    const c = N[i] * wi;
    x += cp[0] * c; y += cp[1] * c; z += cp[2] * c; w += c;
  }
  if (!(Math.abs(w) > 1e-15)) return null;
  const q = [x / w, y / w, z / w];
  return q.every(Number.isFinite) ? q : null;
}

// THE FIT ITSELF.
//
// Unknowns are the uCount x vCount control points. Each target contributes one
// row of tensor-product basis values; each interior control point contributes two
// fairness rows (a second difference along u and along v) whose right-hand side
// is zero. Stacking them and forming the normal equations solves both objectives
// at once, weighted by `stiffness` — no separate regularisation matrix to get out
// of step with the data rows.
export function fitPatch(inputs, opts = {}) {
  const degree = Math.max(1, Math.min(3, Math.round(opts.degree ?? 3)));
  const uCount = Math.max(degree + 1, Math.round(opts.uCount ?? 6));
  const vCount = Math.max(degree + 1, Math.round(opts.vCount ?? 6));
  /* ⚠ THE DEFAULT IS DELIBERATELY LOW, AND IT WAS MEASURED RATHER THAN CHOSEN.
     The fairness rows are normalised against the data rows, so `stiffness` is
     roughly "how much fairness is worth relative to fidelity" — and at 0.5 that
     is already strong enough to hold a surface visibly off its own input: four
     boundary curves fitted at 0.05mm with 0.02 came back at 3.58mm with 0.5, on
     the same data and the same net. Low enough to follow the input, high enough
     that a control point no sample reaches is still determined. */
  const stiffness = opts.stiffness ?? 0.1;
  if (!(stiffness > 0)) {
    return refuse(PATCH_REFUSAL.BAD_REQUEST, 'stiffness must be greater than zero — with no fairness rows a control point that no sample reaches is undetermined and the solve is singular, not merely loose');
  }
  const targets = Array.isArray(inputs) ? inputs : samplePatchInputs(inputs, opts.samplesPerCurve ?? 24);
  if (!targets.length) return refuse(PATCH_REFUSAL.NO_INPUT, 'there is nothing to fit a surface through');
  const nCtrl = uCount * vCount;
  if (targets.length < Math.max(4, degree + 1)) {
    return refuse(PATCH_REFUSAL.TOO_FEW, `${targets.length} point(s) cannot define a surface`);
  }

  const plane = fitPlane(targets);
  if (!plane || !plane.ok) return refuse(PATCH_REFUSAL.DEGENERATE_PLANE, 'the input has no best-fit plane to lay a patch over');
  const { origin, xAxis, yAxis, normal } = plane;

  // Parameterise by the in-plane coordinates, normalised to the data's own
  // extent. The domain is the data's bounding box in that plane, so a patch is
  // never larger than what it was asked to cover.
  const su = [], sv = [];
  for (const q of targets) {
    const d = sub3(q, origin);
    su.push(dot3(d, xAxis));
    sv.push(dot3(d, yAxis));
  }
  const u0 = Math.min(...su), u1 = Math.max(...su);
  const v0 = Math.min(...sv), v1 = Math.max(...sv);
  const du = u1 - u0, dv = v1 - v0;
  if (!(du > 1e-9) || !(dv > 1e-9)) {
    return refuse(PATCH_REFUSAL.DEGENERATE_PLANE, 'the input is collinear in its own plane, so there is no second direction to span');
  }
  const params = targets.map((q, k) => [
    Math.min(1, Math.max(0, (su[k] - u0) / du)),
    Math.min(1, Math.max(0, (sv[k] - v0) / dv)),
  ]);

  /* ⚠ REFUSE DATA THAT FOLDS OVER ITS OWN PLANE. Two targets at the same (u,v)
     with a real gap between them cannot both lie on a single-valued patch, and a
     least-squares fit will happily return the average of the two branches — a
     surface through neither, reported with a deviation that looks like half the
     gap rather than like a contradiction. Detected on a coarse grid, so the cost
     is linear and the message can say how far apart the offenders were. */
  {
    const g = 24;
    const cell = new Map();
    let worstFold = 0;
    for (let k = 0; k < params.length; k++) {
      const key = `${Math.floor(params[k][0] * g)},${Math.floor(params[k][1] * g)}`;
      const prev = cell.get(key);
      if (prev) {
        const gap = Math.abs(dot3(sub3(targets[k], prev), normal));
        if (gap > worstFold) worstFold = gap;
      } else cell.set(key, targets[k]);
    }
    const span = Math.max(du, dv);
    if (worstFold > span * 0.25) {
      return refuse(PATCH_REFUSAL.FOLDED, `this input folds back over its own best-fit plane (two points ${worstFold.toFixed(3)} apart across it land in the same place on the patch), so no single surface can pass near both`);
    }
  }

  const U = clampedUniformKnots(uCount, degree);
  const V = clampedUniformKnots(vCount, degree);
  const idx = (i, j) => i * vCount + j;

  // Data rows.
  const rows = [], rhs = [[], [], []];
  for (let k = 0; k < targets.length; k++) {
    const [uu, vv] = params[k];
    const uspan = findSpan(uCount - 1, degree, uu, U);
    const vspan = findSpan(vCount - 1, degree, vv, V);
    const Nu = basisFuns(uspan, uu, degree, U);
    const Nv = basisFuns(vspan, vv, degree, V);
    const row = new Float64Array(nCtrl);
    for (let a = 0; a <= degree; a++) {
      for (let b = 0; b <= degree; b++) {
        row[idx(uspan - degree + a, vspan - degree + b)] += Nu[a] * Nv[b];
      }
    }
    rows.push(row);
    rhs[0].push(targets[k][0]); rhs[1].push(targets[k][1]); rhs[2].push(targets[k][2]);
  }

  /* THE FAIRNESS ROWS. A second difference across three consecutive control
     points, weighted by `stiffness`, with a right-hand side of zero: the solve is
     asked to make the net as close to locally straight as the data allows. Scaled
     by the count so the same stiffness number means the same thing on a coarse
     net and a fine one — without that, raising the point count would silently
     stiffen the result. */
  const w = stiffness * Math.sqrt(targets.length / Math.max(1, nCtrl));
  const penalty = (a, b, c) => {
    const row = new Float64Array(nCtrl);
    row[a] += w; row[b] -= 2 * w; row[c] += w;
    rows.push(row);
    rhs[0].push(0); rhs[1].push(0); rhs[2].push(0);
  };
  for (let i = 1; i < uCount - 1; i++) for (let j = 0; j < vCount; j++) penalty(idx(i - 1, j), idx(i, j), idx(i + 1, j));
  for (let i = 0; i < uCount; i++) for (let j = 1; j < vCount - 1; j++) penalty(idx(i, j - 1), idx(i, j), idx(i, j + 1));

  // Normal equations: (B^T B) X = B^T Q, one solve for three right-hand sides.
  const A = Array.from({ length: nCtrl }, () => new Array(nCtrl).fill(0));
  for (const row of rows) {
    for (let a = 0; a < nCtrl; a++) {
      const ra = row[a];
      if (ra === 0) continue;
      for (let b = a; b < nCtrl; b++) {
        const rb = row[b];
        if (rb === 0) continue;
        A[a][b] += ra * rb;
      }
    }
  }
  for (let a = 0; a < nCtrl; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
  const btq = [0, 1, 2].map((c) => {
    const out = new Array(nCtrl).fill(0);
    for (let r = 0; r < rows.length; r++) {
      const val = rhs[c][r];
      if (val === 0) continue;
      const row = rows[r];
      for (let a = 0; a < nCtrl; a++) if (row[a] !== 0) out[a] += row[a] * val;
    }
    return out;
  });

  let sol;
  try { sol = solveLinearSystem(A.map((r) => [...r]), btq); } catch (e) { sol = null; }
  if (!sol || sol.some((col) => col.some((x) => !Number.isFinite(x)))) {
    return refuse(PATCH_REFUSAL.SINGULAR, 'the fit did not resolve — try a coarser point count or more stiffness');
  }

  const ctrlNet = [];
  for (let i = 0; i < uCount; i++) {
    const row = [];
    for (let j = 0; j < vCount; j++) {
      const k = idx(i, j);
      row.push([sol[0][k], sol[1][k], sol[2][k], 1]);
    }
    ctrlNet.push(row);
  }
  const srf = { degU: degree, degV: degree, knotsU: U, knotsV: V, ctrlNet };

  /* WHAT THE FIT ACTUALLY ACHIEVED, measured at each target's own parameter.
     ⚠ NAMED HONESTLY: this is the residual at the fit parameters, not the true
     point-to-surface distance, which would need a projection per point. It is an
     UPPER BOUND on that distance — the real surface may pass closer somewhere
     else — so reporting it cannot flatter the fit. */
  let worst = 0, sum = 0;
  for (let k = 0; k < targets.length; k++) {
    const [uu, vv] = params[k];
    const s = surfacePoint(srf, uu, vv);
    const d = len3(sub3(s, targets[k]));
    if (d > worst) worst = d;
    sum += d * d;
  }
  return {
    ok: true,
    srf,
    maxDeviation: worst,
    rmsDeviation: Math.sqrt(sum / targets.length),
    sampleCount: targets.length,
    uCount,
    vCount,
    degree,
    stiffness,
    plane: { origin, xAxis, yAxis, normal },
  };
}

// FIT TO A TOLERANCE, by growing the net until the residual clears it.
//
// ⚠ IT GROWS RATHER THAN GUESSING, and it stops for a reason it can name: the
// tolerance was met, the net reached the cap, or the data ran out — a net with
// more control points than the samples that constrain it is a fit with nothing
// holding it. Every attempt is reported in `tried`, so a refusal says what it
// tried rather than only that it failed.
export function fitPatchToTolerance(inputs, opts = {}) {
  const tolerance = opts.tolerance ?? null;
  const maxCount = Math.max(4, Math.round(opts.maxCount ?? 20));
  const targets = Array.isArray(inputs) ? inputs : samplePatchInputs(inputs, opts.samplesPerCurve ?? 24);
  const tried = [];
  let best = null;
  for (let n = Math.max(4, Math.round(opts.uCount ?? 4)); n <= maxCount; n = Math.ceil(n * 1.5)) {
    if (n * n > targets.length) break; // more freedoms than data: nothing would hold the net
    const r = fitPatch(targets, { ...opts, uCount: n, vCount: n });
    if (!r.ok) return { ...r, tried };
    tried.push({ count: n, maxDeviation: r.maxDeviation });
    best = r;
    if (tolerance != null && r.maxDeviation <= tolerance) return { ...r, tried, metTolerance: true };
  }
  if (!best) return refuse(PATCH_REFUSAL.TOO_FEW, `${targets.length} sample(s) cannot hold even the coarsest patch`);
  return { ...best, tried, metTolerance: tolerance == null ? null : false };
}
