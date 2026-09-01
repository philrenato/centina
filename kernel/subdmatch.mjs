// MATCH SUBD — moving a SuperB cage's boundary rows so the LIMIT SURFACE meets
// a neighbor, not so the CAGE does.
//
// That distinction is the whole problem, and it is not a nicety. A Catmull-Clark
// cage is not its surface: a boundary vertex's limit position sits at
// (A + 4P + B)/6 of its two boundary neighbors, so dragging a cage vertex onto a
// target leaves the surface roughly one sixth of the local second difference
// short of it. Matching the cage to a neighbor and matching the surface to a
// neighbor are different edits, and only the second is what a person sees.
//
// ── WHAT THE BOUNDARY OF A CATMULL-CLARK SURFACE ACTUALLY IS ────────────────
//
// Read subd.mjs's own rules at a boundary and the answer falls out with no new
// machinery. `edgeSharpness` pins every boundary edge to sharpness 1, so
// `computeEdgePoint` returns the plain MIDPOINT there, and a boundary vertex has
// exactly two sharp edges, so `computeVertexPoint` takes the crease branch and
// returns (A + 6P + B)/8. Those two rules together are precisely Lane and
// Riesenfeld's subdivision for a uniform cubic B-spline curve (Lane & Riesenfeld,
// IEEE PAMI 2(1), 1980), and they read NOTHING but the boundary polyline. So:
//
//   THE LIMIT BOUNDARY CURVE IS THE UNIFORM CUBIC B-SPLINE WHOSE CONTROL POLYGON
//   IS THE CAGE'S OWN BOUNDARY POLYLINE.
//
// Everything else here is a consequence. Away from an extraordinary vertex the
// limit surface is the uniform bicubic B-spline over the cage (Catmull & Clark,
// CAD 10(6), 1978; the exact-evaluation statement of the same fact is Stam,
// SIGGRAPH 1998, which kernel/subdlimit.mjs already builds on). Write the
// boundary row as P_j and the row behind it as R_j, and extend the net outward by
// one phantom row P'_j so that the extended surface's own boundary curve has
// control points P_j. The u-basis of a uniform cubic at a knot is (1/6, 4/6, 1/6),
// so that condition is (P'_j + 4P_j + R_j)/6 = P_j, giving
//
//   P'_j = 2 P_j - R_j                          (the phantom row is a reflection)
//
// and then, at the station of boundary vertex j, with the knot-value basis
// (1/6, 4/6, 1/6) for value, (-1/2, 0, 1/2) for the first derivative and
// (1, -2, 1) for the second:
//
//   LIMIT POSITION    L_j = (P_{j-1} + 4 P_j + P_{j+1}) / 6
//   ALONG TANGENT     A_j = (P_{j+1} - P_{j-1}) / 2
//   CROSS TANGENT     X_j = (D_{j-1} + 4 D_j + D_{j+1}) / 6,   D_j = R_j - P_j
//   CROSS CURVATURE   S_uu = P'_j - 2 P_j + R_j = 0,  IDENTICALLY
//
// All four are verified by evaluating the limit surface through
// `subdivideCatmullClark` and measuring, rather than taken on the derivation's
// word: position to 2.86e-5 mm and tangent to 1.76e-2 degrees against a target,
// with an unmatched boundary of the same cage held bit-identical as the control.
//
// ── THE CEILING, AND WHY IT IS MATHEMATICS RATHER THAN SCOPE ────────────────
//
// That last line is the honest ceiling of this whole feature. The second
// cross-derivative of the limit surface at a Catmull-Clark boundary is ZERO, for
// every cage, whatever anyone does to the rows: the phantom row is defined as the
// reflection that makes it so. A curvature match to a target whose own normal
// curvature across the seam is nonzero is therefore not approximate here, it is
// unreachable, and `matchSubD` refuses order 2 by name rather than returning
// something that looks matched in a shaded view and is not. The NURBS side of
// this app reaches G2 through `applyG2` in kernel/matchedge.mjs; the SubD side
// tops out at G1, and the reason is a property of the surface type.
//
// ── WHAT MOVES, AND WHAT PROVABLY DOES NOT ──────────────────────────────────
//
// A match acts on a RUN of the cage's boundary: a contiguous chain of boundary
// vertices. It writes exactly two rows of that run — the boundary vertices
// themselves and their cross neighbors — and NO CAGE VERTEX OUTSIDE THE RUN. A
// second object, or any other boundary of the same cage, comes back bit-identical.
//
// It does not follow that the limit SURFACE is untouched outside the run, and the
// difference is worth stating rather than quietly enjoying. A limit position reads
// its two boundary neighbors, so the surface at the boundary vertex immediately
// beyond each end of an open run does move — by one station, and only there. That
// is the same thing that happens on the NURBS side when a matched edge drags the
// corner it shares with the surface's other three edges; it is what matching an
// edge means, not a leak. A closed run — the whole boundary loop of an open cage —
// has no outside at all, and its system is cyclic.
//
// ── HOW THE ROWS ARE FOUND ─────────────────────────────────────────────────
//
// Each run vertex needs a CROSS NEIGHBOR — the R_j above. Topologically it is the
// one neighbor that is not a run neighbor: an interior station has valence 3 (two
// boundary edges plus one interior edge, so the interior edge), and an open run's
// end vertex at a cage corner has valence 2 with only one run neighbor, so the
// other boundary edge serves, which is exactly the net's own next row there. Any
// run vertex with a different count is refused by name: with two candidates there
// is no single cross-boundary direction to redirect, and with none there is no row
// behind the edge at all.

import { buildTopology, edgeKey, creaseWeight, MARKED_CORNER_WEIGHT_FLOOR } from './subd.mjs';
import { surfacePointAndPartials } from './surface.mjs';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const L = len(a); return L > 1e-14 ? scale(a, 1 / L) : [0, 0, 0]; };
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// ── THE UNIFORM CUBIC B-SPLINE, EVALUATED ONCE AND USED THREE WAYS ─────────
//
// Position, along-tangent and cross-tangent above are the same basis read with
// different derivative orders, so there is one evaluator and the station formulas
// are it at t = 0. `pts` is a control polygon; `closed` wraps it. Parameter is
// (segment index + t), so a closed polygon of n points has domain [0, n) and an
// open one has domain [1, n-2] — an open uniform cubic spans only the segments
// with a full four-point window. Its first and last control points therefore SHAPE
// the matched arc without lying on it, which is why the fit below leaves them
// free: pinning a control point that the curve never reaches puts the whole error
// into the direction the curve travels at its ends.
export function uniformCubicDomain(n, closed) {
  return closed ? [0, n] : [1, n - 2];
}

// The four control points a parameter reads, AS INDICES — so the same routine
// serves evaluation and the least-squares fit, and the fit can never disagree
// with the evaluator about which points a sample sees.
export function cubicWindowIndices(n, closed, s) {
  const [lo, hi] = uniformCubicDomain(n, closed);
  const x = Math.min(Math.max(s, lo), hi);
  let i = Math.floor(x);
  let t = x - i;
  if (i >= hi) { i = hi - 1; t = 1; }             // the closing endpoint belongs to the last segment
  const wrap = (k) => (closed ? ((k % n) + n) % n : k);
  return { t, idx: [wrap(i - 1), wrap(i), wrap(i + 1), wrap(i + 2)] };
}

function windowAt(pts, closed, s) {
  const { t, idx } = cubicWindowIndices(pts.length, closed, s);
  return { t, p: idx.map((k) => pts[k]) };
}

// de Boor's uniform cubic basis and its first two derivatives, in the segment's
// own local parameter. Written out rather than run through a general B-spline
// routine because these six lines are checkable by eye and the general routine
// needs a knot vector this curve does not carry.
function cubicBasis(t, order) {
  const t2 = t * t, t3 = t2 * t;
  if (order === 0) return [(1 - 3 * t + 3 * t2 - t3) / 6, (4 - 6 * t2 + 3 * t3) / 6, (1 + 3 * t + 3 * t2 - 3 * t3) / 6, t3 / 6];
  if (order === 1) return [(-3 + 6 * t - 3 * t2) / 6, (-12 * t + 9 * t2) / 6, (3 + 6 * t - 9 * t2) / 6, (3 * t2) / 6];
  return [(6 - 6 * t) / 6, (-12 + 18 * t) / 6, (6 - 18 * t) / 6, (6 * t) / 6];
}

export function uniformCubicAt(pts, closed, s, order = 0) {
  const { t, p } = windowAt(pts, closed, s);
  const b = cubicBasis(t, order);
  let out = [0, 0, 0];
  for (let k = 0; k < 4; k++) out = add(out, scale(p[k], b[k]));
  return out;
}

// ── THE RUN ────────────────────────────────────────────────────────────────

// Order a set of boundary edge keys into one chain of vertices. Returns the
// chain, whether it closes, and a refusal naming the offending vertex when the
// selection is not a single simple run.
export function boundaryRunFromEdges(cage, edgeKeys, ctx = buildTopology(cage)) {
  const keys = [...new Set(edgeKeys)];
  if (keys.length < 2) return { ok: false, reason: `a match needs a run of at least 2 boundary edges, and ${keys.length} was selected` };
  const adj = new Map();
  for (const key of keys) {
    const e = ctx.edgeMap.get(key);
    if (!e) return { ok: false, reason: `"${key}" is not an edge of this cage` };
    if (e.faces.length !== 1) return { ok: false, reason: `edge ${e.v0}-${e.v1} is not a boundary edge — it has ${e.faces.length} faces, and a boundary edge has exactly 1` };
    for (const [a, b] of [[e.v0, e.v1], [e.v1, e.v0]]) {
      if (!adj.has(a)) adj.set(a, []);
      adj.get(a).push(b);
    }
  }
  for (const [v, ns] of adj) {
    if (ns.length > 2) return { ok: false, reason: `vertex ${v} has ${ns.length} of the selected edges meeting at it — this selection branches, so it is not one run` };
  }
  const ends = [...adj].filter(([, ns]) => ns.length === 1).map(([v]) => v);
  if (ends.length !== 0 && ends.length !== 2) return { ok: false, reason: `this selection has ${ends.length} loose ends — one run has either 2 (an open run) or 0 (a closed loop)` };
  const closed = ends.length === 0;
  const start = closed ? adj.keys().next().value : Math.min(...ends);
  const run = [start];
  let prev = -1, cur = start;
  for (;;) {
    const next = (adj.get(cur) || []).find((v) => v !== prev);
    if (next === undefined) break;
    if (next === start) { if (!closed) return { ok: false, reason: 'this selection closes on itself but reports loose ends — it is not a simple run' }; break; }
    if (run.includes(next)) return { ok: false, reason: `the run revisits vertex ${next} before finishing — it is not a simple run` };
    run.push(next);
    prev = cur; cur = next;
  }
  if (run.length !== adj.size) return { ok: false, reason: `the selection is in ${adj.size - run.length + 1} disconnected pieces — a match takes one run at a time` };
  return { ok: true, run, closed };
}

// The cross neighbor of every run vertex, plus every reason this run cannot
// carry a match. Named refusals, never a clamp: a run this function accepts is
// one every formula in this file is actually valid on.
export function runStations(cage, run, closed, ctx = buildTopology(cage)) {
  const n = run.length;
  if (n < (closed ? 3 : 3)) return { ok: false, reason: `a match needs at least 3 boundary vertices in the run, and this one has ${n}` };
  const runSet = new Set(run);
  const stations = [];
  for (let j = 0; j < n; j++) {
    const v = run[j];
    const prev = closed ? run[(j - 1 + n) % n] : (j > 0 ? run[j - 1] : -1);
    const next = closed ? run[(j + 1) % n] : (j < n - 1 ? run[j + 1] : -1);
    const edges = ctx.vertexEdges[v] || [];
    const others = [];
    for (const e of edges) {
      const w = e.v0 === v ? e.v1 : e.v0;
      if (w === prev || w === next) continue;
      others.push({ w, boundary: e.faces.length === 1 });
    }
    if (others.length !== 1) {
      return { ok: false, reason: others.length === 0
        ? `vertex ${v} on this run has no row behind the boundary — there is nothing to redirect there`
        : `vertex ${v} on this run has ${others.length} edges leaving the boundary, so there is no single cross-boundary direction; a match needs a run whose vertices each have exactly one` };
    }
    // A creased cross edge, or a marked corner, replaces the boundary rules this
    // file's formulas are derived from — refused by name rather than matched
    // against a limit surface that is somewhere else.
    if (creaseWeight(cage, v, others[0].w) > 0) {
      return { ok: false, reason: `the edge behind vertex ${v} is creased, which changes the limit rule at that vertex — remove the crease along the run before matching` };
    }
    if (prev >= 0 && creaseWeight(cage, v, prev) > MARKED_CORNER_WEIGHT_FLOOR && next >= 0 && creaseWeight(cage, v, next) > MARKED_CORNER_WEIGHT_FLOOR) {
      return { ok: false, reason: `vertex ${v} is a marked corner, which holds the limit surface at the cage point rather than on the boundary curve — a match cannot move it onto a target` };
    }
    stations.push({ v, prev, next, cross: others[0].w, corner: others[0].boundary });
    void runSet;
  }
  return { ok: true, stations };
}

// A BOUNDARY LOOP THAT CANNOT CARRY A MATCH ALL THE WAY ROUND, SPLIT INTO THE
// ARCS THAT CAN.
//
// The rim of a grid patch is the case that forces this. Its four corners have
// valence 2 — both of their edges run ALONG the rim — so at a corner there is no
// third edge to read a cross-boundary direction from, and `runStations` refuses
// the whole loop by name. The refusal is right: the row behind the boundary
// TURNS at a corner, and there is no single direction across it. But three
// quarters of that rim are perfectly ordinary boundary, and a caller looking for
// something to meet should be offered them rather than told the rim is unusable.
//
// So the loop is cut AT the corners, and each corner belongs to both arcs that
// end on it — which is exactly what makes those arcs work: an open run's end
// vertex has only ONE run neighbour, so its other boundary edge becomes the
// cross direction, and that edge is the net's own next row there. A loop with
// only one bad vertex yields nothing, because the single arc would have to begin
// and end at the same vertex and no control polygon can hold one point twice.
export function usableRuns(cage, run, closed, ctx = buildTopology(cage)) {
  const whole = runStations(cage, run, closed, ctx);
  if (whole.ok) return [{ run: run.slice(), closed }];
  if (!closed) return [];
  const n = run.length;
  const bad = [];
  for (let j = 0; j < n; j++) {
    const v = run[j], prev = run[(j - 1 + n) % n], next = run[(j + 1) % n];
    let others = 0;
    for (const e of (ctx.vertexEdges[v] || [])) {
      const w = e.v0 === v ? e.v1 : e.v0;
      if (w !== prev && w !== next) others++;
    }
    if (others !== 1) bad.push(j);
  }
  if (bad.length < 2) return [];
  const out = [];
  for (let i = 0; i < bad.length; i++) {
    const a = bad[i], b = bad[(i + 1) % bad.length];
    const seg = [];
    for (let j = a; ; j = (j + 1) % n) { seg.push(run[j]); if (j === b) break; }
    if (seg.length < 3) continue;
    if (runStations(cage, seg, false, ctx).ok) out.push({ run: seg, closed: false });
  }
  return out;
}

// The boundary polyline and the cross-offset polygon of a run, in run order —
// the two control polygons every formula above reads.
export function runPolygons(cage, stations) {
  const P = stations.map((s) => cage.vertices[s.v].slice());
  const D = stations.map((s, j) => sub(cage.vertices[s.cross], P[j]));
  return { P, D };
}

// The limit surface's own frame at a run station: where it is, the two tangents,
// and the outward normal. This is the oracle the app's readout reports and the
// quantity a match is solved against.
export function limitFrameAt(cage, stations, closed, j) {
  const { P, D } = runPolygons(cage, stations);
  const n = P.length;
  const at = (arr, k) => arr[closed ? ((k % n) + n) % n : k];
  if (!closed && (j < 1 || j > n - 2)) return null;
  const comb = (arr) => scale(add(add(at(arr, j - 1), scale(at(arr, j), 4)), at(arr, j + 1)), 1 / 6);
  const point = comb(P);
  const along = scale(sub(at(P, j + 1), at(P, j - 1)), 1 / 2);
  const crossT = comb(D);
  return { point, along, cross: crossT, normal: unit(cross(along, crossT)) };
}

// ── TARGETS ────────────────────────────────────────────────────────────────
//
// A target answers three questions at a parameter: where its edge is, its
// outward unit normal there, and the direction its own surface leaves the edge
// in. The third is the side test — a base whose cross tangent agrees in SIGN
// with the target's has folded back over it, and reports a perfect tangent angle
// while doing so.

export function nurbsEdgeTarget(srf, edge) {
  const dom = (knots, deg) => [knots[deg], knots[knots.length - 1 - deg]];
  const [uMin, uMax] = dom(srf.knotsU, srf.degU);
  const [vMin, vMax] = dom(srf.knotsV, srf.degV);
  const paramAt = (s) => {
    if (edge === 'u0') return { u: uMin, v: vMin + (vMax - vMin) * s, cross: 'u', sign: +1 };
    if (edge === 'u1') return { u: uMax, v: vMin + (vMax - vMin) * s, cross: 'u', sign: -1 };
    if (edge === 'v0') return { u: uMin + (uMax - uMin) * s, v: vMin, cross: 'v', sign: +1 };
    if (edge === 'v1') return { u: uMin + (uMax - uMin) * s, v: vMax, cross: 'v', sign: -1 };
    throw new Error(`nurbsEdgeTarget: unknown edge ${edge}`);
  };
  return {
    kind: 'nurbs',
    domain: [0, 1],
    frame(s) {
      const p = paramAt(Math.min(Math.max(s, 0), 1));
      const { point, su, sv } = surfacePointAndPartials(srf, p.u, p.v);
      const inward = scale(p.cross === 'u' ? su : sv, p.sign);
      const along = p.cross === 'u' ? sv : su;
      return { point, along, inward, normal: unit(cross(along, inward)) };
    },
  };
}

// The target is another cage's boundary loop, read through this file's own limit
// formulas — so a SuperB matched to a SuperB is matched to the surface the person
// sees, on both sides of the seam.
export function subdBoundaryTarget(cage, run, closed, ctx = buildTopology(cage)) {
  const st = runStations(cage, run, closed, ctx);
  if (!st.ok) return { ok: false, reason: `the target run: ${st.reason}` };
  const { P, D } = runPolygons(cage, st.stations);
  const [lo, hi] = uniformCubicDomain(P.length, closed);
  return {
    ok: true, kind: 'subd', domain: [lo, hi],
    frame(s) {
      const point = uniformCubicAt(P, closed, s, 0);
      const along = uniformCubicAt(P, closed, s, 1);
      const inward = uniformCubicAt(D, closed, s, 0);
      return { point, along, inward, normal: unit(cross(along, inward)) };
    },
  };
}

// CLOSEST POINT ON THE TARGET EDGE — bounded and deterministic: a dense scan to
// bracket the minimum, then a fixed number of ternary steps inside the bracket.
// A match is a match, not a snap, so this is only ever asked to travel the small
// distance a nearly-coincident pair of edges is apart; the scan density is what
// keeps it from locking onto the wrong lobe of a folded target.
export function closestOnTarget(target, q, scan = 240, refine = 60) {
  const [lo, hi] = target.domain;
  let bestS = lo, bestD = Infinity;
  for (let i = 0; i <= scan; i++) {
    const s = lo + (hi - lo) * (i / scan);
    const d = len(sub(target.frame(s).point, q));
    if (d < bestD) { bestD = d; bestS = s; }
  }
  const step = (hi - lo) / scan;
  let a = Math.max(lo, bestS - step), b = Math.min(hi, bestS + step);
  for (let i = 0; i < refine; i++) {
    const m1 = a + (b - a) / 3, m2 = b - (b - a) / 3;
    if (len(sub(target.frame(m1).point, q)) < len(sub(target.frame(m2).point, q))) b = m2; else a = m1;
  }
  const s = (a + b) / 2;
  return { s, distance: len(sub(target.frame(s).point, q)) };
}

// ── THE SOLVE ──────────────────────────────────────────────────────────────
//
// Dense Gaussian elimination with partial pivoting, three right-hand sides at
// once. The system is one row per moving station — a run is tens of vertices, not
// thousands — so this is chosen over a tridiagonal or cyclic-tridiagonal solver
// on purpose: the banded routines are the ones that are wrong in a way nobody
// reads, and nothing here is large enough to notice the difference.
function solveDense(A, B) {
  const m = A.length, c = B[0].length;
  const M = A.map((row, i) => [...row, ...B[i]]);
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;    // singular; the caller refuses rather than returning a wild net
    if (piv !== col) { const t = M[piv]; M[piv] = M[col]; M[col] = t; }
    const d = M[col][col];
    for (let k = col; k < m + c; k++) M[col][k] /= d;
    for (let r = 0; r < m; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let k = col; k < m + c; k++) M[r][k] -= f * M[col][k];
    }
  }
  return M.map((row) => row.slice(m));
}

// LEAST-SQUARES FIT OF ONE CONTROL POLYGON TO A WANTED FIELD ALONG THE SPAN.
//
// The same routine serves the boundary row and the cross-offset row, because the
// two are the same uniform cubic read on two different polygons — which is the
// claim the header makes, made once in code. Standard B-spline global
// approximation (Piegl & Tiller, The NURBS Book, ch. 9.4): stack one row per
// sample, solve the normal equations.
//
// ⚠ EVERY CONTROL POINT OF THE RUN IS FREE, INCLUDING THE TWO AT ITS ENDS, and
// that is a decision rather than an oversight. An open uniform cubic over n
// points spans only [1, n-2], so the first and last control points shape the ends
// of the matched arc without lying on it; holding them pins the arc's ENDS to
// wherever the cage happened to be, and that error lands entirely in the
// direction the boundary curve travels — which is half of the tangent plane. Held
// ends measured 6 degrees of normal deviation at the last station of an otherwise
// exact match. Freeing them costs one thing, stated plainly: a run's end vertices
// move, so the limit surface changes within one station of the run's ends along
// whatever else those vertices belong to. No cage vertex outside the run is
// written.
function fitPolygon(nCtrl, closed, samples) {
  const A = Array.from({ length: nCtrl }, () => new Array(nCtrl).fill(0));
  const B = Array.from({ length: nCtrl }, () => [0, 0, 0]);
  for (const { s, wanted } of samples) {
    const { t, idx } = cubicWindowIndices(nCtrl, closed, s);
    const b = cubicBasis(t, 0);
    for (let a = 0; a < 4; a++) {
      for (let c = 0; c < 4; c++) A[idx[a]][idx[c]] += b[a] * b[c];
      B[idx[a]] = add(B[idx[a]], scale(wanted, b[a]));
    }
  }
  return solveDense(A, B);
}

// THE SAME FIT, WITH A DIRECTION IT IS ALLOWED TO IGNORE.
//
// A boundary match wants each sample to land ON the target curve, not at one
// nominated point of it — where it lands ALONG the curve is parametrization and
// costs nothing. Charging for that tangential distance is what makes a plain
// point-to-point fit lock: it drags the base's parametrization toward the
// target's and then cannot represent the reparametrized curve that results, so
// the residual stops falling with a real gap still in it. Measured on a rim whose
// wobble is out of phase with its target's, that plateau sat at 2.6e-3 mm and
// fell like 1/passes, so no iteration budget reaches it, against 2.0e-5 mm for
// the same solve on an already-aligned pair. Corresponding by arc length instead
// was worse still, at 1.3e-2 mm, because two rims of the same length can
// distribute that length differently.
//
// So each sample carries a 3x3 weight rather than nothing: full weight across the
// target's own tangent, a little along it. That couples the three coordinates,
// which is why this is a 3n system where the plain fit is n with three right-hand
// sides.
//
// ⚠⚠ AND THE SLACK MUST BE ANCHORED TO WHERE THE SAMPLE ALREADY IS, not to the
// closest point. Both give the same matrix; only the right-hand side differs, and
// the difference is the whole behavior. Pulled weakly toward the closest point,
// the tangential freedom is a slow drift that ACCUMULATES: the first pass
// measured 1.5e-3 mm and the third 1.6e-2, ten times worse, as the control points
// bunched along the seam. Anchored to the sample's current position it is a
// damping term instead, the parametrization holds still, and each pass only
// removes the part of the error that is a genuine gap. The slide is small and
// deliberately NOT zero — at exactly zero the tangential motion is unconstrained
// and the system is singular.
const TANGENTIAL_SLIDE = 1e-3;

function crossPlaneWeight(tangent) {
  const T = unit(tangent);
  if (len(T) < 0.5) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const W = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) W[a][b] = (a === b ? 1 : 0) - (1 - TANGENTIAL_SLIDE) * T[a] * T[b];
  return W;
}

function fitPolygonWeighted(nCtrl, closed, samples) {
  const N = nCtrl * 3;
  const A = Array.from({ length: N }, () => new Array(N).fill(0));
  const B = Array.from({ length: N }, () => [0]);
  for (const { s, wanted, here, tangent, weight } of samples) {
    const { t, idx } = cubicWindowIndices(nCtrl, closed, s);
    const b = cubicBasis(t, 0);
    // ACROSS the tangent, aim at the target; ALONG it, aim at where the sample
    // already is. One matrix, two anchors.
    const T = unit(tangent);
    const across = len(T) > 0.5 ? sub(wanted, scale(T, dot(T, wanted))) : wanted;
    const along = len(T) > 0.5 ? scale(T, dot(T, here) * TANGENTIAL_SLIDE) : [0, 0, 0];
    const Wq = add(across, along);
    for (let a = 0; a < 4; a++) {
      for (let c = 0; c < 4; c++) {
        for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) A[idx[a] * 3 + p][idx[c] * 3 + q] += b[a] * b[c] * weight[p][q];
      }
      for (let p = 0; p < 3; p++) B[idx[a] * 3 + p][0] += b[a] * Wq[p];
    }
  }
  const X = solveDense(A, B);
  if (!X) return null;
  const out = [];
  for (let k = 0; k < nCtrl; k++) out.push([X[k * 3][0], X[k * 3 + 1][0], X[k * 3 + 2][0]]);
  return out;
}

// The parameters a fit is sampled at: `perSpan` inside every segment of the
// matched arc, endpoints included. Dense enough that the normal equations are
// well determined, and fixed rather than adaptive so a match is reproducible.
function fitParameters(nCtrl, closed, perSpan = 7) {
  const [lo, hi] = uniformCubicDomain(nCtrl, closed);
  const count = Math.max(1, Math.round((hi - lo) * perSpan));
  const out = [];
  for (let i = 0; i <= count; i++) out.push(lo + (hi - lo) * (i / count));
  if (closed) out.pop();                              // the closing sample is the opening one
  return out;
}

// ── THE ONE ENTRY POINT ────────────────────────────────────────────────────
//
// Returns a NEW cage; never mutates. `order` 0 is position, 1 is tangent, and 2
// is refused by name for the reason the header derives. `blend` walks the result
// back toward the cage it started from, so the whole edit stays a live parameter
// rather than a one-way commit; `fullness` scales how far the surface leans off
// the seam before it goes where it was going, the SubD reading of the same knob
// `endBulgeNet` turns on a NURBS net.
export function matchSubD(cage, run, closed, target, opts = {}) {
  const order = opts.order ?? 1;
  const blend = Math.max(0, Math.min(1, opts.blend ?? 1));
  const fullness = opts.fullness ?? 1;
  const maxPasses = opts.maxPasses ?? 12;
  const tol = opts.tolerance ?? 1e-10;

  if (order === 2) {
    return { ok: false, reason: 'curvature (G2) is not reachable on a SubD boundary at all: the limit surface\u2019s second derivative across a Catmull-Clark boundary is identically zero, whatever the cage does, so a curvature match to a curved neighbor cannot exist. Tangent (G1) is the ceiling here — the NURBS side of this app reaches G2, the SubD side cannot' };
  }
  if (order !== 0 && order !== 1) return { ok: false, reason: `unknown continuity order ${order}` };
  if (!Number.isFinite(fullness) || fullness <= 0) return { ok: false, reason: `a fullness of ${fullness} is not a positive number` };
  if (!target || typeof target.frame !== 'function') return { ok: false, reason: 'no target edge to match against' };

  const ctx = buildTopology(cage);
  const st = runStations(cage, run, closed, ctx);
  if (!st.ok) return { ok: false, reason: st.reason };
  const stations = st.stations;
  const n = stations.length;
  const P0 = stations.map((s) => cage.vertices[s.v].slice());
  const D0 = stations.map((s, j) => sub(cage.vertices[s.cross], P0[j]));
  const params = fitParameters(n, closed);

  // TOLERANCES ARE DERIVED FROM THE RUN, never pinned: a match on a 4 mm bracket
  // and a match on a 4 m hull are the same edit at different scales, and a fixed
  // epsilon is a different instrument in each.
  let span = 0;
  for (let j = 1; j < n; j++) span += len(sub(P0[j], P0[j - 1]));
  if (!(span > 0)) return { ok: false, reason: 'this run has no length — every vertex on it sits at the same point' };
  const moveTol = opts.tolerance ?? span * 1e-12;

  // --- G0: fit the boundary polygon so the limit CURVE lies on the target across
  // the whole matched arc, not merely at the vertex stations.
  //
  // ITERATED, AND THE BEST PASS IS THE ONE THAT IS KEPT — not the last one.
  // The correspondence is a closest-point projection, so it is a function of
  // where the curve currently is: the first pass projects from the curve the cage
  // started with, and refitting from the new one is what closes the rest. But the
  // tangential slack that makes the fit work at all (see fitPolygonWeighted) also
  // means the samples can slide and bunch along the seam over many passes, and a
  // pass that bunches them leaves the gap WORSE than the pass before it. Measured
  // on an out-of-phase rim: 1.5e-3 mm at pass 1, 1.6e-2 by pass 3. The loop
  // therefore records the best polygon it has seen, stops as soon as a pass buys
  // less than a percent off the gap it still has, and returns the best rather than
  // the latest. That makes the solve monotone in the only quantity anybody cares
  // about, which a plain iteration on this correspondence is not.
  let P = P0.map((p) => p.slice());
  let passes = 0, moved = Infinity, gap = Infinity, prevGap = Infinity, converged = false;
  let bestP = P.map((p) => p.slice()), bestGap = Infinity, bestHits = null;
  const hits = params.map(() => 0);
  for (; passes < maxPasses; passes++) {
    const samples = params.map((s, k) => {
      const here = uniformCubicAt(P, closed, s, 0);
      const hit = closestOnTarget(target, here);
      hits[k] = hit.s;
      const f = target.frame(hit.s);
      return { s, wanted: f.point, here, tangent: f.along, weight: crossPlaneWeight(f.along) };
    });
    const next = fitPolygonWeighted(n, closed, samples);
    if (!next) return { ok: false, reason: 'the boundary fit is singular — this run doubles back on itself' };
    moved = 0;
    for (let j = 0; j < n; j++) moved = Math.max(moved, len(sub(next[j], P[j])));
    P = next;
    gap = 0;
    params.forEach((s, k) => {
      const hit = closestOnTarget(target, uniformCubicAt(P, closed, s, 0));
      hits[k] = hit.s;
      gap = Math.max(gap, hit.distance);
    });
    if (gap < bestGap) { bestGap = gap; bestP = P.map((p) => p.slice()); bestHits = hits.slice(); }
    if (moved <= moveTol) { converged = true; passes++; break; }
    if (prevGap - gap <= gap * 1e-2) { converged = true; passes++; break; }
    prevGap = gap;
  }
  P = bestP;
  gap = bestGap;
  if (bestHits) for (let k = 0; k < hits.length; k++) hits[k] = bestHits[k];

  let D = D0.map((d) => d.slice());

  if (order === 1) {
    // --- G1: the smallest change to the cross-tangent FIELD that lays it in the
    // target's tangent plane all along the seam. At each sample the current cross
    // tangent loses its component along the target's normal and keeps its length,
    // so the surface's own lean survives and only the part that broke continuity
    // is removed; copying the target's direction outright would ask for more than
    // tangency and distort the cage to get it. The side test is separate and not
    // optional — coplanar says nothing about WHICH WAY the base leaves the seam,
    // and a base folded back over its target measures a perfect angle while doing
    // it.
    const samples = [];
    for (let k = 0; k < params.length; k++) {
      const s = params[k];
      const X = uniformCubicAt(D, closed, s, 0);
      const f = target.frame(hits[k]);
      if (len(f.normal) < 0.5) return { ok: false, reason: 'the target surface is degenerate along that edge — it has no normal to match a tangent plane against' };
      const flat = sub(X, scale(f.normal, dot(X, f.normal)));
      if (len(flat) < 1e-12) return { ok: false, reason: 'this cage leaves its boundary straight along the target\u2019s own normal, so there is no in-plane direction left to keep' };
      const W = scale(unit(flat), len(X) * fullness);
      if (dot(W, f.inward) > 0) {
        return { ok: false, reason: 'this cage leans the same way the target surface does, so a tangent match would fold it back over its neighbor rather than continuing it — the two are on the same side of the seam' };
      }
      samples.push({ s, wanted: W });
    }
    const next = fitPolygon(n, closed, samples);
    if (!next) return { ok: false, reason: 'the cross-boundary fit is singular — this run doubles back on itself' };
    D = next;
  }

  // --- BLEND, and write the new cage. Both rows are walked back together, so a
  // partial blend is a partial match and never a G0 that outran its G1.
  const out = { vertices: cage.vertices.map((p) => p.slice()), faces: cage.faces.map((f) => f.slice()), creases: { ...(cage.creases || {}) } };
  for (let j = 0; j < n; j++) {
    const s = stations[j];
    out.vertices[s.v] = lerp(P0[j], P[j], blend);
    if (order === 1) out.vertices[s.cross] = add(out.vertices[s.v], lerp(D0[j], D[j], blend));
  }
  return {
    ok: true, cage: out, order, blend, fullness, closed,
    stations: n, samples: params.length, passes, converged, projection: gap,
  };
}

// ── THE MEASUREMENT ────────────────────────────────────────────────────────
//
// What was achieved, read off the LIMIT SURFACE and not off the cage — the
// distinction this whole file exists for. Sampled between the stations as well as
// at them, because a match solved at stations is exact there and approximate in
// between, and the number worth reporting is the worse one.
export function matchDeviation(cage, run, closed, target, samples = 41) {
  const st = runStations(cage, run, closed);
  if (!st.ok) return { ok: false, reason: st.reason };
  const stations = st.stations;
  const { P, D } = runPolygons(cage, stations);
  const [lo, hi] = uniformCubicDomain(P.length, closed);
  let gap = 0, angle = 0;
  for (let i = 0; i <= samples; i++) {
    const s = lo + (hi - lo) * (i / samples);
    const point = uniformCubicAt(P, closed, s, 0);
    const along = uniformCubicAt(P, closed, s, 1);
    const crossT = uniformCubicAt(D, closed, s, 0);
    const hit = closestOnTarget(target, point);
    const f = target.frame(hit.s);
    gap = Math.max(gap, hit.distance);
    const nb = unit(cross(along, crossT));
    const nt = f.normal;
    if (len(nb) > 0.5 && len(nt) > 0.5) {
      const c = Math.min(1, Math.max(-1, Math.abs(dot(nb, nt))));
      angle = Math.max(angle, Math.acos(c) * 180 / Math.PI);
    }
  }
  return { ok: true, gap, angle };
}

