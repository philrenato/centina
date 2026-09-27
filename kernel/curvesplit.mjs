// Curve-curve split and trim — where two curves in space cut each other, and
// what is left when a piece is removed.
//
// Relation to kernel/curvecurve.mjs: `intersectCurves2D` solves the hard half
// exactly — Bezier decomposition, a convex-hull rejection that is a proof,
// exact homogeneous de Casteljau subdivision, and a 2x2 Newton that lands on
// the curves themselves. It is planar, because two general curves in 3D
// generically do not meet. A split tool asks a different question — where the
// modeler intends two curves to meet — which has a stated, checkable rule.
//
// What makes the 3D case exact rather than sampled: a NURBS curve is
// affine-invariant, so an orthographic projection of the curve is exactly the
// curve of the projected control points — same knots, same weights, same
// parameterization. Projecting into a plane and calling `intersectCurves2D` is
// therefore the exact solution of the projected problem, and the projected
// parameters uA/uB are the same parameters on the original 3D curves. Nothing
// is resampled and no polyline stands in for a curve anywhere in this file.
//
// Two kinds of intersection, and the caller is expected to report which one it
// used:
//
//   true      the two curves pass within `tolerance` of each other in 3D at
//             that parameter pair. `gap <= tolerance`. Direction-independent:
//             a 3D crossing shows up under every non-degenerate projection, so
//             the projection direction cannot invent or destroy one.
//
//   apparent  the two curves cross when viewed along the projection direction
//             but miss each other in space. `gap > tolerance`. This is Rhino's
//             apparent intersection and it is view-dependent by construction —
//             a different view direction gives different apparent crossings,
//             which is why the direction is reported back rather than assumed.
//
// A caller that will not accept inference passes `infer:false` and gets true
// events only, plus — on a refusal — the measured closest approach, so the
// refusal can quote a number.
//
// Tangential contact is found, not refused. `intersectCurves2D` refuses a
// tangency because its Jacobian is singular there and no transversal crossing
// is defined. But a line touching a circle does split that line, so this
// module runs a second, independent search for local minima of the 3D distance
// between the two curves (grid seeds, then a damped 2x2 Newton on the squared
// distance, whose Hessian is well conditioned exactly where the intersector's
// is not). A minimum at or below tolerance is a touch and becomes an event; a
// minimum above it is the number the refusal quotes.
//
// What Trim removes, and what it leaves whole. Rhino's Trim deletes the
// clicked section and leaves the rest of the curve intact — it does not split
// the remainder at every other crossing. So `trimCurveAtParam` returns the
// complement as maximal runs: at most two pieces from an open curve, exactly
// one from a closed one. Split is the other disposition of the same
// intersection set ("trim deletes the picked portion, split keeps
// everything").

import { curvePoint, curveDomain, rationalCurveDerivs, isCurveClosed } from './curve.mjs';
import { extractSubCurve, joinCurvesC0 } from './knots.mjs';
import { intersectCurves2D } from './curvecurve.mjs';
import { bestFitPlane } from './selfintersect.mjs';
import { sub, cross, dot, normalize, length, anyPerpendicular } from './vec3.mjs';

// The kernel's default document tolerance. Absolute, not relative: whether the
// modeler meant two curves to meet is a question about model units, and a
// relative tolerance would mean something different on a 1mm curve than on a
// 1000mm one.
export const CURVE_SPLIT_DEFAULT_TOLERANCE = 0.001;

// How far a parameter pair may move under the distance-minimizing Newton
// before two seeds are considered to have landed on the same minimum.
const MIN_MERGE_PARAM_FRAC = 1e-7;
const NEWTON_MIN_STEPS = 40;

// The fourth slot is a weight, not a homogeneous divisor. This kernel stores a
// control point as euclidean x,y,z plus w (see toHomogeneous in curve.mjs,
// which multiplies through when a homogeneous form is wanted). Dividing by w
// here would deform every rational curve — a circle's 90-degree span control
// points pulled in by their 0.707 weight, and the projected circle crossing a
// chord at four places instead of two.
function euclidCtrlPts(crv) {
  return crv.ctrlPts.map((p) => [p[0], p[1], p[2]]);
}

function ctrlDiagonal(crvs) {
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const c of crvs) for (const p of euclidCtrlPts(c)) for (let i = 0; i < 3; i++) {
    if (p[i] < lo[i]) lo[i] = p[i];
    if (p[i] > hi[i]) hi[i] = p[i];
  }
  const d = length(sub(hi, lo));
  return Number.isFinite(d) && d > 0 ? d : 1;
}

/**
 * The exact orthographic projection of a NURBS curve into the plane spanned by
 * e1/e2. Exact because a NURBS curve is affine-invariant: the projected curve
 * is the curve of the projected control points, with the knots, the weights
 * and the parameterization all unchanged, so a parameter found in the
 * projection names the same point on the 3D curve.
 */
export function projectCurveToFrame(crv, e1, e2) {
  return {
    degree: crv.degree,
    knots: [...crv.knots],
    // x,y,z are euclidean and w rides along untouched — a projection acts on
    // the point, never on its weight.
    ctrlPts: crv.ctrlPts.map((p) => [dot(p, e1), dot(p, e2), 0, p[3]]),
  };
}

// An orthonormal pair perpendicular to `dir`, so that (e1, e2, dir) is a
// right-handed frame. Uses the kernel's `anyPerpendicular` for the first axis,
// so every projected frame is built the same way.
function frameFor(dir) {
  const d = normalize(dir);
  const e1 = normalize(anyPerpendicular(d));
  const e2 = cross(d, e1);
  return { d, e1, e2 };
}

/**
 * Is this pair of curves planar, and if so what is the plane's normal?
 *
 * Measured on the control points, not on samples: a curve lies in a plane if
 * its control points do (the rational basis is non-negative and sums to one,
 * so every curve point is a convex combination of them and cannot leave their
 * affine hull). That makes this an exact test, at the cost of one pass over
 * the nets.
 *
 * `planarTolerance` is relative to the pair's size, unlike the coincidence
 * tolerance: whether two curves lie in a plane is a question about shape, and
 * a 1000mm layout drawn 0.5mm out of plane is planar in every practical sense,
 * while the same 0.5mm on a 2mm curve is not.
 */
export function coplanarNormal(crvA, crvB, planarFrac = 1e-4) {
  const pts = [...euclidCtrlPts(crvA), ...euclidCtrlPts(crvB)];
  const plane = bestFitPlane(pts);
  if (!plane) return null; // collinear or coincident nets define no plane
  const diag = ctrlDiagonal([crvA, crvB]);
  const tol = diag * planarFrac;
  for (const p of pts) if (Math.abs(dot(sub(p, plane.origin), plane.normal)) > tol) return null;
  return plane.normal;
}

// Closest approach — the second, independent search.
//
// Seeds come from a coarse parameter grid; every local minimum of the grid
// (strictly less than all 8 neighbors, boundary-aware) is refined by a damped
// Newton on g(u,v) = |A(u) - B(v)|^2, whose gradient and Hessian are written
// out below from the curves' analytic derivatives. Where the transversal
// intersector's Jacobian goes singular — a tangency — this Hessian is best
// conditioned, so the two searches together cover cases neither covers
// alone.

function gridResolution(crv) {
  // Enough seeds that no local minimum between two control points is stepped
  // over. Tied to the control count because that is what bounds how many
  // wiggles the curve can have.
  return Math.max(64, Math.min(512, crv.ctrlPts.length * 24));
}

function refineClosestApproach(crvA, crvB, u0, v0, domA, domB) {
  let u = u0, v = v0;
  const clamp = (t, d) => Math.min(d[1], Math.max(d[0], t));
  let best = { u, v, dist: Infinity };
  for (let it = 0; it < NEWTON_MIN_STEPS; it++) {
    const A = rationalCurveDerivs(crvA, u, 2);
    const B = rationalCurveDerivs(crvB, v, 2);
    const w = sub(A[0], B[0]);
    const dist = length(w);
    if (dist < best.dist) best = { u, v, dist };
    // g   = w.w
    // g_u = 2 w.A'      g_v = -2 w.B'
    const gu = 2 * dot(w, A[1]);
    const gv = -2 * dot(w, B[1]);
    const h11 = 2 * (dot(A[1], A[1]) + dot(w, A[2]));
    const h12 = -2 * dot(A[1], B[1]);
    const h22 = 2 * (dot(B[1], B[1]) - dot(w, B[2]));
    const det = h11 * h22 - h12 * h12;
    let du, dv;
    if (Math.abs(det) > 1e-300 && h11 > 0 && det > 0) {
      // Newton, taken only when the Hessian is positive definite — anywhere
      // else it points at a saddle or a maximum, not at the minimum being
      // looked for.
      du = (-gu * h22 + gv * h12) / det;
      dv = (-h11 * gv + h12 * gu) / det;
    } else {
      // Steepest descent, scaled so the step is a fraction of the domain
      // rather than of the (unit-dependent) gradient magnitude.
      const mag = Math.hypot(gu, gv);
      if (!(mag > 0)) break;
      const s = 0.01 * Math.max(domA[1] - domA[0], domB[1] - domB[0]) / mag;
      du = -gu * s;
      dv = -gv * s;
    }
    // Backtracking: a step that does not reduce the distance is halved rather
    // than taken, so a bad Newton step degrades to a short descent step
    // instead of throwing the seed away.
    let taken = false;
    for (let k = 0; k < 12; k++) {
      const nu = clamp(u + du, domA), nv = clamp(v + dv, domB);
      const nd = length(sub(curvePoint(crvA, nu), curvePoint(crvB, nv)));
      if (nd < dist) { u = nu; v = nv; taken = true; break; }
      du *= 0.5; dv *= 0.5;
    }
    if (!taken) break;
  }
  const A0 = curvePoint(crvA, best.u), B0 = curvePoint(crvB, best.v);
  return { u: best.u, v: best.v, dist: length(sub(A0, B0)) };
}

/**
 * Every local minimum of the 3D distance between two curves, refined.
 *
 * Returns [{ uA, uB, gap }] sorted by gap. Used for two different jobs: it
 * finds tangential contact, which the transversal intersector refuses, and
 * its smallest entry is the closest approach a refusal quotes.
 */
export function closestApproaches(crvA, crvB) {
  const domA = curveDomain(crvA), domB = curveDomain(crvB);
  const na = gridResolution(crvA), nb = gridResolution(crvB);
  const ptsA = [], ptsB = [];
  for (let i = 0; i <= na; i++) ptsA.push(curvePoint(crvA, domA[0] + (domA[1] - domA[0]) * i / na));
  for (let j = 0; j <= nb; j++) ptsB.push(curvePoint(crvB, domB[0] + (domB[1] - domB[0]) * j / nb));
  const at = (i, j) => {
    const d = sub(ptsA[i], ptsB[j]);
    return dot(d, d);
  };
  const seeds = [];
  for (let i = 0; i <= na; i++) {
    for (let j = 0; j <= nb; j++) {
      const c = at(i, j);
      let isMin = true;
      for (let di = -1; di <= 1 && isMin; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          if (!di && !dj) continue;
          const ii = i + di, jj = j + dj;
          if (ii < 0 || jj < 0 || ii > na || jj > nb) continue;
          if (at(ii, jj) < c) { isMin = false; break; }
        }
      }
      if (isMin) seeds.push([domA[0] + (domA[1] - domA[0]) * i / na, domB[0] + (domB[1] - domB[0]) * j / nb]);
    }
  }
  const out = [];
  const mergeA = (domA[1] - domA[0]) * MIN_MERGE_PARAM_FRAC * 1e3;
  const mergeB = (domB[1] - domB[0]) * MIN_MERGE_PARAM_FRAC * 1e3;
  for (const [su, sv] of seeds) {
    const r = refineClosestApproach(crvA, crvB, su, sv, domA, domB);
    if (!Number.isFinite(r.dist)) continue;
    if (out.some((e) => Math.abs(e.uA - r.u) <= mergeA && Math.abs(e.uB - r.v) <= mergeB)) continue;
    out.push({ uA: r.u, uB: r.v, gap: r.dist });
  }
  out.sort((a, b) => a.gap - b.gap);
  return out;
}

// The intersection

/**
 * Where two 3D curves cut each other, with the rule that was applied named in
 * the result rather than left for the caller to guess.
 *
 * opts:
 *   tolerance   absolute 3D distance at or below which a crossing is true.
 *   infer       accept apparent (projected-only) crossings as well. When
 *               false, only true events come back and `nearest` carries the
 *               measured closest approach so a refusal can quote it.
 *   direction   the direction to look along for apparent crossings. Omitted,
 *               the pair's best-fit-plane normal is used, which is exact for
 *               a coplanar pair and a reasonable generic direction otherwise;
 *               a caller with a viewport should pass the view direction,
 *               because apparent means apparent from that view.
 *
 * Returns { ok, events, coplanar, direction, directionSource, nearest, reason }.
 * `ok:true` with `events:[]` means the curves do not meet, and `nearest` says
 * by how much they miss.
 */
export function curveCurveEvents(crvA, crvB, opts = {}) {
  const tolerance = opts.tolerance ?? CURVE_SPLIT_DEFAULT_TOLERANCE;
  const infer = opts.infer !== false;
  const normal = coplanarNormal(crvA, crvB, opts.planarFrac);
  const coplanar = !!normal;
  let dirSource, dir;
  if (opts.direction && length(opts.direction) > 0) { dir = opts.direction; dirSource = 'given'; }
  else if (normal) { dir = normal; dirSource = 'plane normal'; }
  else {
    // A pair with no plane (two collinear nets, or skew nets whose best fit
    // is not a plane) still has to be viewed from somewhere. The pair's
    // diagonal is non-degenerate for at least one of them.
    const b = bestFitPlane([...euclidCtrlPts(crvA), ...euclidCtrlPts(crvB)]);
    dir = b ? b.normal : [0, 0, 1];
    dirSource = b ? 'best-fit normal' : 'world Z';
  }
  /* A projection can degenerate. Viewed exactly end-on, a curve projects to a
     point or a segment, and the 2D search then reports the pair as
     overlapping — true of the projection and false of the curves. A caller
     passing a view direction hits this whenever the view looks straight down
     a line. True intersections are direction-independent, so any other
     direction answers the same question: the pair's best-fit normal is tried
     next, then a direction perpendicular to the first. Only if all three
     degenerate is the overlap reported. */
  const tryDirs = [dir];
  if (normal) tryDirs.push(normal);
  tryDirs.push(frameFor(dir).e1);
  let flat = null, chosen = frameFor(dir), usedFallback = false;
  for (let k = 0; k < tryDirs.length; k++) {
    const fr = frameFor(tryDirs[k]);
    const r = intersectCurves2D(projectCurveToFrame(crvA, fr.e1, fr.e2), projectCurveToFrame(crvB, fr.e1, fr.e2));
    flat = r; chosen = fr;
    if (r.ok) { usedFallback = k > 0; break; }
    if (!r.overlapping) break; // a tangency is a real answer, not a degenerate view
  }
  if (usedFallback) dirSource = `${dirSource} (the first direction looked straight down a curve, so another was used)`;
  const { d } = chosen;

  const events = [];
  const push = (uA, uB, kind) => {
    const A = curvePoint(crvA, uA), B = curvePoint(crvB, uB);
    const gap = length(sub(A, B));
    // Dedupe on the 3D point, not the parameters: a closed curve reaches the
    // same place from two different parameters at its seam, and the two
    // searches often find the same event.
    const merge = Math.max(tolerance, ctrlDiagonal([crvA, crvB]) * 1e-7);
    if (events.some((e) => length(sub(e.pointA, A)) <= merge && length(sub(e.pointB, B)) <= merge)) return;
    events.push({ uA, uB, pointA: A, pointB: B, gap, kind: gap <= tolerance ? 'true' : kind });
  };

  if (flat.ok) for (const p of flat.points) push(p.uA, p.uB, 'apparent');

  // Tangential contact, and the refusal's number, from the independent
  // distance search. Only minima at or under tolerance become events — a
  // distant local minimum is the nearest miss, not a crossing.
  const mins = closestApproaches(crvA, crvB);
  /* An overlap must not be harvested as many touches. Where two curves lie on
     top of each other, every local minimum of the distance between them is
     zero, so the tangency search would return one intersection per grid seed
     for a question that has no defined answer. The 2D search has already
     established the overlap (every leaf survives subdivision); that verdict
     stands, and the minima are discarded. */
  const overlapping = !flat.ok && !!flat.overlapping;
  if (!overlapping) for (const m of mins) if (m.gap <= tolerance) push(m.uA, m.uB, 'true');

  const kept = overlapping ? [] : (infer ? events : events.filter((e) => e.kind === 'true'));
  kept.sort((a, b) => a.uA - b.uA);
  const nearest = mins.length ? mins[0].gap : null;

  const out = {
    ok: true, events: kept, coplanar,
    direction: [d[0], d[1], d[2]], directionSource: dirSource,
    nearest, tolerance,
    apparentCount: kept.filter((e) => e.kind === 'apparent').length,
  };
  if (!kept.length) {
    if (overlapping) {
      out.ok = false;
      out.overlapping = true;
      out.reason = 'these two curves lie on top of each other over a whole stretch rather than crossing at points — there is no single place to cut';
    } else if (nearest !== null) {
      out.reason = `these two curves do not meet: their closest approach is ${nearest.toPrecision(4)}, and the tolerance is ${tolerance}`;
    } else {
      out.reason = 'these two curves do not meet anywhere this search can find';
    }
  }
  return out;
}

// Splitting

/**
 * The parameters at which a curve can be cut, given raw event
 * parameters on it.
 *
 * Two filters, both geometric rather than parametric, because a parametric
 * epsilon means different distances on different curves:
 *   · a crossing at a curve end is not a cut — there is nothing on one side of
 *     it — so it is dropped rather than producing a zero-length piece;
 *   · two crossings closer together than the tolerance are one crossing.
 */
export function cuttableParams(crv, params, tolerance = CURVE_SPLIT_DEFAULT_TOLERANCE) {
  const [u0, u1] = curveDomain(crv);
  const closed = isCurveClosed(crv, Math.max(tolerance, 1e-9));
  const pStart = curvePoint(crv, u0), pEnd = curvePoint(crv, u1);
  const sorted = [...params].filter((u) => Number.isFinite(u) && u >= u0 && u <= u1).sort((a, b) => a - b);
  const out = [];
  for (const u of sorted) {
    const p = curvePoint(crv, u);
    // On a closed curve the seam is an ordinary point, so a crossing there is
    // a cut and is kept; on an open one it is an end.
    if (!closed && (length(sub(p, pStart)) <= tolerance || length(sub(p, pEnd)) <= tolerance)) continue;
    if (out.length && length(sub(curvePoint(crv, out[out.length - 1]), p)) <= tolerance) continue;
    out.push(u);
  }
  return out;
}

/**
 * Cut a curve at every given parameter.
 *
 * Open curve, k cuts -> k+1 pieces, in order.
 * Closed curve, k cuts -> k pieces, because the seam is not a cut: the last
 * piece runs from the last cut, through the seam, round to the first. That
 * wrap piece is built by extracting both halves and joining them C0 — exact,
 * not refitted. A single cut on a closed curve therefore yields one piece: the
 * curve opened at that point, as in Rhino.
 */
export function splitCurveAtParams(crv, params, opts = {}) {
  const tolerance = opts.tolerance ?? CURVE_SPLIT_DEFAULT_TOLERANCE;
  const [u0, u1] = curveDomain(crv);
  const closed = opts.closed ?? isCurveClosed(crv, Math.max(tolerance, 1e-9));
  const cuts = cuttableParams(crv, params, tolerance);
  if (!cuts.length) return { ok: false, pieces: [], reason: 'no cut parameter falls strictly inside this curve' };
  const pieces = [];
  if (!closed) {
    const bounds = [u0, ...cuts, u1];
    for (let i = 0; i + 1 < bounds.length; i++) pieces.push(extractSubCurve(crv, bounds[i], bounds[i + 1]));
  } else {
    for (let i = 0; i + 1 < cuts.length; i++) pieces.push(extractSubCurve(crv, cuts[i], cuts[i + 1]));
    // The wrap piece. Either half can be empty when a cut lands exactly on the
    // seam, in which case the other half alone is the piece.
    const tailReal = (u1 - cuts[cuts.length - 1]) > (u1 - u0) * 1e-12;
    const headReal = (cuts[0] - u0) > (u1 - u0) * 1e-12;
    const tail = tailReal ? extractSubCurve(crv, cuts[cuts.length - 1], u1) : null;
    const head = headReal ? extractSubCurve(crv, u0, cuts[0]) : null;
    if (tail && head) pieces.push(joinCurvesC0([tail, head]));
    else if (tail) pieces.push(tail);
    else if (head) pieces.push(head);
  }
  return { ok: true, pieces, cuts, closed };
}

/**
 * Mutual split of N curves: every curve is cut wherever it meets every other
 * curve in the set, in one pass, rather than N one-against-one splits.
 *
 * Returns { ok, results, pairs, reason }, where `results[i]` is
 * { index, cuts, pieces, eventCount } for every curve that was cut, and
 * `pairs` records what each pair produced (including the pairs that produced
 * nothing, and why) so the caller can report it.
 */
export function splitCurveNetwork(curves, opts = {}) {
  const tolerance = opts.tolerance ?? CURVE_SPLIT_DEFAULT_TOLERANCE;
  if (!Array.isArray(curves) || curves.length < 2) {
    return { ok: false, results: [], pairs: [], reason: 'splitting curves against each other needs at least two curves' };
  }
  const perCurve = curves.map(() => []);
  const pairs = [];
  let anyEvent = false;
  let sawApparent = false;
  for (let i = 0; i < curves.length; i++) {
    for (let j = i + 1; j < curves.length; j++) {
      const r = curveCurveEvents(curves[i], curves[j], opts);
      pairs.push({ i, j, ok: r.ok, count: r.events.length, apparent: r.apparentCount, nearest: r.nearest, coplanar: r.coplanar, overlapping: !!r.overlapping, reason: r.reason });
      if (!r.events.length) continue;
      anyEvent = true;
      if (r.apparentCount) sawApparent = true;
      for (const e of r.events) { perCurve[i].push(e.uA); perCurve[j].push(e.uB); }
    }
  }
  if (!anyEvent) {
    const nearest = pairs.reduce((m, p) => (p.nearest !== null && p.nearest !== undefined && (m === null || p.nearest < m) ? p.nearest : m), null);
    // An overlap is a different refusal from a miss; quoting a closest
    // approach of 0 for it would read as "they touch but nothing happened".
    const overlaps = pairs.filter((p) => p.overlapping).length;
    return {
      ok: false, results: [], pairs,
      reason: overlaps
        ? `${overlaps === 1 ? 'two of these curves lie' : `${overlaps} pairs of these curves lie`} on top of each other rather than crossing, so there is no single place to cut`
        : nearest !== null
          ? `none of these curves reach each other: the closest any two come is ${nearest.toPrecision(4)}, and the tolerance is ${tolerance}`
          : 'none of these curves meet each other',
    };
  }
  const results = [];
  for (let i = 0; i < curves.length; i++) {
    if (!perCurve[i].length) continue;
    const s = splitCurveAtParams(curves[i], perCurve[i], { tolerance });
    if (!s.ok || s.pieces.length < 2) {
      // One cut on an open curve at its own end, or a closed curve cut once,
      // can produce fewer than two pieces. A closed curve opened at one point
      // is a change and is kept; an open curve that gained no piece is not.
      if (s.ok && s.closed && s.pieces.length === 1) { results.push({ index: i, cuts: s.cuts, pieces: s.pieces, opened: true }); continue; }
      continue;
    }
    results.push({ index: i, cuts: s.cuts, pieces: s.pieces, closed: s.closed });
  }
  if (!results.length) {
    return { ok: false, results: [], pairs, reason: 'every crossing found lands on a curve end, so nothing can be cut off' };
  }
  return { ok: true, results, pairs, sawApparent };
}

// Trimming

/**
 * Which section of a curve a click landed in, given the parameters that bound
 * the sections. Returns { a, b, wraps, index } — the section's own parameter
 * bounds, and whether it is the closed curve's wrap section.
 */
export function sectionAtParam(crv, params, uClick, opts = {}) {
  const tolerance = opts.tolerance ?? CURVE_SPLIT_DEFAULT_TOLERANCE;
  const [u0, u1] = curveDomain(crv);
  const closed = opts.closed ?? isCurveClosed(crv, Math.max(tolerance, 1e-9));
  const cuts = cuttableParams(crv, params, tolerance);
  if (!cuts.length) return null;
  if (!closed) {
    const bounds = [u0, ...cuts, u1];
    for (let i = 0; i + 1 < bounds.length; i++) {
      if (uClick >= bounds[i] && uClick <= bounds[i + 1]) return { a: bounds[i], b: bounds[i + 1], wraps: false, index: i, count: bounds.length - 1 };
    }
    return null;
  }
  for (let i = 0; i + 1 < cuts.length; i++) {
    if (uClick >= cuts[i] && uClick <= cuts[i + 1]) return { a: cuts[i], b: cuts[i + 1], wraps: false, index: i, count: cuts.length };
  }
  // Outside every interior section means inside the one that wraps the seam.
  return { a: cuts[cuts.length - 1], b: cuts[0], wraps: true, index: cuts.length - 1, count: cuts.length };
}

/**
 * Remove the section containing `uClick` and keep the rest — as maximal runs,
 * not as every section separately.
 *
 * This is the difference between Trim and Split: a line crossed by two
 * cutters, trimmed at its left end, comes back as one curve running from the
 * first cutter to the far end, with the second crossing left uncut. Rhino
 * behaves this way, and it makes Trim a cleanup tool rather than a shredder.
 *
 * Returns { ok, pieces, removed:{a,b}, reason }. `pieces` is empty when the
 * whole curve was the clicked section — a valid outcome (the curve is
 * deleted), reported as `wholeCurve:true` rather than as a failure.
 */
export function trimCurveAtParam(crv, params, uClick, opts = {}) {
  const tolerance = opts.tolerance ?? CURVE_SPLIT_DEFAULT_TOLERANCE;
  const [u0, u1] = curveDomain(crv);
  const closed = opts.closed ?? isCurveClosed(crv, Math.max(tolerance, 1e-9));
  const sec = sectionAtParam(crv, params, uClick, { tolerance, closed });
  if (!sec) return { ok: false, pieces: [], reason: 'nothing crosses this curve, so there is no section to take away' };
  const span = u1 - u0;
  const eps = span * 1e-12;
  const pieces = [];
  if (!closed) {
    if (sec.a - u0 > eps) pieces.push(extractSubCurve(crv, u0, sec.a));
    if (u1 - sec.b > eps) pieces.push(extractSubCurve(crv, sec.b, u1));
    if (!pieces.length) return { ok: true, pieces: [], wholeCurve: true, removed: { a: sec.a, b: sec.b } };
    return { ok: true, pieces, removed: { a: sec.a, b: sec.b } };
  }
  if (sec.wraps) {
    // The removed section is [a, u1] + [u0, b] with a > b, so the kept run is
    // the single interior stretch from b up to a — no seam crossing and no
    // join.
    if (sec.a - sec.b > eps) pieces.push(extractSubCurve(crv, sec.b, sec.a));
  } else {
    // Kept run goes from the section's end, through the seam, round to its
    // start. Both halves are exact sub-curves; the join is C0 at the seam,
    // where the closed curve's own continuity already was.
    const tail = (u1 - sec.b > eps) ? extractSubCurve(crv, sec.b, u1) : null;
    const head = (sec.a - u0 > eps) ? extractSubCurve(crv, u0, sec.a) : null;
    if (tail && head) pieces.push(joinCurvesC0([tail, head]));
    else if (tail) pieces.push(tail);
    else if (head) pieces.push(head);
  }
  if (!pieces.length) return { ok: true, pieces: [], wholeCurve: true, removed: { a: sec.a, b: sec.b } };
  return { ok: true, pieces, removed: { a: sec.a, b: sec.b } };
}

/**
 * Every parameter on `crv` where any curve in `others` crosses it — the input
 * a Trim click needs before it can name a section. Collected in one place so
 * the caller does not have to re-derive the pairing rule.
 */
export function crossingParamsOn(crv, others, opts = {}) {
  const params = [];
  const notes = [];
  for (let k = 0; k < others.length; k++) {
    const r = curveCurveEvents(crv, others[k], opts);
    notes.push({ index: k, count: r.events.length, apparent: r.apparentCount, nearest: r.nearest, reason: r.reason });
    for (const e of r.events) params.push(e.uA);
  }
  return { params, notes };
}
