// MATCH EDGE — reshaping one surface's boundary to meet its neighbour, at a
// stated order of continuity. The read-only sibling (the continuity overlay)
// shows a student the crease; this is the half that closes it.
//
// TIERS, AND WHAT IS AND IS NOT BUILT HERE:
//   G0 position  — the base's boundary control ROW becomes the target's. Exact
//                  and not a fit: the boundary row of a clamped NURBS surface
//                  IS its edge curve's own control polygon, so this is literal
//                  substitution.
//   G1 tangent   — the base's SECOND row is REDIRECTED so the cross-boundary
//                  derivative continues the target's through the seam, with the
//                  base's own row spacing preserved. Direction only, magnitude
//                  kept: cloning the magnitude would visibly balloon or flatten
//                  an otherwise reasonable surface, and Rhino's MatchSrf keeps
//                  it for the same reason.
//   G2 curvature — BUILT, and deliberately NOT inside `matchEdge`, which still
//                  refuses order 2 by name. It is a separate pass, `applyG2`,
//                  layered on a net that has already reached G1: the normal
//                  curvature at a boundary station is linear in the THIRD row's
//                  position, so the coefficient is measured rather than
//                  transcribed from a knot-spacing expression nobody can check
//                  by eye. Two rows deep across the seam is refused by name.
//   G3 and beyond — not built. The same construction extends — the curvature
//                  DERIVATIVE across the seam is linear in the fourth row — but
//                  it needs a third-derivative evaluator, which `curvature.mjs`
//                  does not have, and four rows of depth on both surfaces.
//
// WEIGHTS TRAVEL WITH THE POSITIONS THEY BELONG TO. G0 is a substitution, so it
// takes the target's weight along with the target's point — keeping the base's
// own weight there would reproduce the target's position under a different
// rational parametrisation, which is a different curve through the same control
// points. G1 only ever redirects, so it keeps the weight it found.
import { surfacePointAndPartials } from './surface.mjs';

export const EDGES = ['u0', 'u1', 'v0', 'v1'];

// ⚠ `ctrlNet` STORES PLAIN POSITIONS with the weight alongside — it is NOT
// pre-multiplied (see toHomogeneousNet in surface.mjs, which is what does the
// multiplying). Dividing by w here would move every control point of a rational
// surface, and w is 1 on every hand-built fixture, so the mistake is invisible
// until it meets a primitive sphere or cylinder.
const euclid = (cp) => [cp[0], cp[1], cp[2]];
const homog = (p, w) => [p[0], p[1], p[2], w];
const weightOf = (cp) => (cp[3] === undefined ? 1 : cp[3]);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const L = len(a); return L > 1e-14 ? [a[0] / L, a[1] / L, a[2] / L] : [0, 0, 0]; };
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// The control points lying ON an edge, and the row one step in from it. Which
// index moves and which runs along the edge depends on the edge, so this is the
// single place that mapping is written down.
export function edgeRows(ctrlNet, edge) {
  const nu = ctrlNet.length, nv = ctrlNet[0].length;
  if (edge === 'u0') return { boundary: ctrlNet[0].map((_, j) => [0, j]), second: ctrlNet[1].map((_, j) => [1, j]), along: nv, depth: nu };
  if (edge === 'u1') return { boundary: ctrlNet[nu - 1].map((_, j) => [nu - 1, j]), second: ctrlNet[nu - 2].map((_, j) => [nu - 2, j]), along: nv, depth: nu };
  if (edge === 'v0') return { boundary: ctrlNet.map((_, i) => [i, 0]), second: ctrlNet.map((_, i) => [i, 1]), along: nu, depth: nv };
  if (edge === 'v1') return { boundary: ctrlNet.map((_, i) => [i, nv - 1]), second: ctrlNet.map((_, i) => [i, nv - 2]), along: nu, depth: nv };
  throw new Error(`edgeRows: unknown edge ${edge}`);
}

// END BULGE — CHANGE HOW FULL A SURFACE IS NEAR ONE EDGE, WITHOUT MOVING THE
// EDGE OR TILTING THE SURFACE THERE.
//
// This is matchEdge's G1 branch turned inside out. That one HOLDS the magnitude
// of each column's boundary-to-second-row vector and REDIRECTS it to agree with
// a neighbour. This one holds the DIRECTION — the surface's own — and scales the
// magnitude. So:
//   · the boundary row is not written at all, so G0 is exact by construction and
//     costs nothing to verify;
//   · every column's cross-boundary direction is unchanged, so the tangent PLANE
//     along the edge is exact too, and a surface matched to a neighbour stays
//     matched.
// What changes is how far the second row sits from the edge, which is what
// "fullness" means: the surface leaves the edge on the same heading and then
// bellies out more or less before it gets where it was going.
//
// ⚠⚠ AND THE TANGENT-PLANE GUARANTEE IS NARROWER THAN "always" — MEASURED, NOT
// ASSUMED. Scaling every column about its boundary point gives a rational
// cross-boundary derivative
//     S_u' = (c/w)(f*T + B),   B = SUM_j N_j(v) * (w1j - w0j) * (P0j - S(0,v))
// so the tangent plane survives exactly when B vanishes. It does in two regimes,
// which between them cover every surface this kernel currently builds:
//   · POLYNOMIAL nets (all weights equal) — exact at every parameter along the
//     edge, including between control-point stations, at any knot vector.
//   · RATIONAL nets whose second-row weights are PROPORTIONAL to the boundary
//     row's (w1j = k*w0j) — B collapses to k*(S*w - S*w) = 0. Spheres, revolves,
//     pipes and tori are all product-weight nets, and knot insertion and degree
//     elevation both preserve that structure.
// It does NOT hold on a rational net with non-proportional weight rows. Measured
// on a sphere-octant variant at factor 2: the edge does not move at all and the
// tangent plane still tilts 1.73 degrees mid-edge, falling to exactly 0 at both
// corners — which is to say, precisely BETWEEN the stations a control-point check
// looks at. A surface matched to a rational neighbour is exactly this shape,
// because the G0 blend writes the target's boundary weights over an unchanged
// second row. So the result reports which regime it is in rather than promising
// the stronger one everywhere.
//
// ⚠ IT IS THE PARAMETRIC SPEED THAT MOVES, NOT THE GEOMETRY OF THE EDGE. Scaling
// a first-derivative magnitude preserves the tangent plane (G1) while changing
// the derivative itself (C1).
//
// ⛔ AND IT REFUSES ON A NET ONLY TWO DEEP, which is not a formality: with two
// control points across, the "second" row IS the opposite boundary, so scaling it
// would drag the far edge of the surface around while claiming to adjust the near
// one. A plain ruled or extruded surface is exactly this shape.
//
// `factor` is a multiplier on the existing distance — 1 changes nothing, below 1
// flattens toward the edge, above 1 fills out. Returns { ok, net } or
// { ok: false, reason }, never a net that quietly did something else.
export function endBulgeNet(ctrlNet, edge, factor) {
  if (!Array.isArray(ctrlNet) || !ctrlNet.length || !Array.isArray(ctrlNet[0])) {
    return { ok: false, reason: 'there is no control net to bulge' };
  }
  if (!Number.isFinite(factor) || factor <= 0) {
    return { ok: false, reason: `a fullness of ${factor} is not a positive number` };
  }
  const rows = edgeRows(ctrlNet, edge);
  if (rows.depth < 3) {
    return { ok: false, reason: `this surface is only ${rows.depth} control points across at that edge, so the row behind the edge IS the opposite edge — bulging it would move the far side` };
  }
  /* ⚠ HOW FAR CAN THIS GO BEFORE IT FOLDS? The second row travels along its own
     direction, and once it passes the THIRD row the net doubles back on itself and
     the surface ripples near the edge. Noise and Wave in the same chain both carry
     a self-intersection guard (`maxSafeDisplacementScale`, kernel/noise.mjs, with
     its own 0.98 margin); this had none, so a high fullness could quietly fold a
     surface and report success. Computed per column and reported as
     `maxSafeFactor` — the caller clamps and says why rather than discovering it in
     a shading mode. Columns whose third row lies BEHIND the second impose no
     limit: that direction is already travelling away from the net. */
  const SAFETY = 0.98;
  const depthDir = (edge === 'u0' || edge === 'u1') ? 'u' : 'v';
  let maxSafeFactor = Infinity;
  for (let k = 0; k < rows.second.length; k++) {
    const [si, sj] = rows.second[k];
    const [bi, bj] = rows.boundary[k];
    const ti = depthDir === 'u' ? (edge === 'u0' ? 2 : ctrlNet.length - 3) : si;
    const tj = depthDir === 'v' ? (edge === 'v0' ? 2 : ctrlNet[0].length - 3) : sj;
    const third = ctrlNet[ti] && ctrlNet[ti][tj];
    if (!third) continue;
    const P0 = euclid(ctrlNet[bi][bj]);
    const away = sub(euclid(ctrlNet[si][sj]), P0);
    const L = len(away);
    if (L < 1e-14) continue;
    const dir = unit(away);
    const reach = sub(euclid(third), P0);
    const t = reach[0] * dir[0] + reach[1] * dir[1] + reach[2] * dir[2];
    if (t <= L) continue; // the third row is not further out along this column
    const limit = (t / L) * SAFETY;
    if (limit < maxSafeFactor) maxSafeFactor = limit;
  }
  const net = ctrlNet.map((row) => row.map((cp) => cp.slice()));
  let moved = 0;
  for (let k = 0; k < rows.second.length; k++) {
    const [si, sj] = rows.second[k];
    const [bi, bj] = rows.boundary[k];
    const P0 = euclid(ctrlNet[bi][bj]);
    const P1 = euclid(ctrlNet[si][sj]);
    const away = sub(P1, P0);
    const L = len(away);
    // A column whose second row sits ON the edge — a pole, a collapsed corner —
    // has no direction to preserve, so it is left exactly where it is rather
    // than being given an invented one.
    if (L < 1e-14) continue;
    const dir = unit(away);
    const d = L * factor;
    net[si][sj] = homog([P0[0] + dir[0] * d, P0[1] + dir[1] * d, P0[2] + dir[2] * d], weightOf(ctrlNet[si][sj]));
    moved++;
  }
  if (!moved) return { ok: false, reason: 'every column at that edge is degenerate, so there is no fullness to change' };
  /* WHICH REGIME THIS NET IS IN, decided from the weights themselves rather than
     from the caller's expectations. `tangentPlaneExact` false does not mean the
     result is wrong — it means the guarantee is the weaker one and the caller
     should measure rather than promise. */
  let tangentPlaneExact = true;
  let ratio = null;
  for (let k = 0; k < rows.second.length && tangentPlaneExact; k++) {
    const [si, sj] = rows.second[k];
    const [bi, bj] = rows.boundary[k];
    const w0 = weightOf(ctrlNet[bi][bj]), w1 = weightOf(ctrlNet[si][sj]);
    if (!(Math.abs(w0) > 1e-12)) { tangentPlaneExact = false; break; }
    const r = w1 / w0;
    if (ratio === null) ratio = r;
    else if (Math.abs(r - ratio) > 1e-9) tangentPlaneExact = false;
  }
  return { ok: true, net, moved, tangentPlaneExact, maxSafeFactor, folds: factor > maxSafeFactor };
}

const cloneNet = (net) => net.map((row) => row.map((cp) => [...cp]));

// Does this edge run in the same direction as the other's, or reversed? Two
// surfaces meeting at a seam are just as likely to have been built in opposite
// directions, and matching row k to row k blindly would twist the base into a
// bowtie while every count check still passed.
export function edgeOrientation(baseNet, baseEdge, targetNet, targetEdge) {
  const b = edgeRows(baseNet, baseEdge), t = edgeRows(targetNet, targetEdge);
  if (b.along !== t.along) return null;
  const bp = b.boundary.map(([i, j]) => euclid(baseNet[i][j]));
  const tp = t.boundary.map(([i, j]) => euclid(targetNet[i][j]));
  let same = 0, flipped = 0;
  for (let k = 0; k < bp.length; k++) {
    same += len(sub(bp[k], tp[k]));
    flipped += len(sub(bp[k], tp[tp.length - 1 - k]));
  }
  return flipped < same ? 'reversed' : 'aligned';
}

// How far apart the two edges currently are — the coincidence check, so a match
// is refused when the edges do not actually meet rather than silently dragging
// one surface across the scene to the other.
export function edgeGap(baseNet, baseEdge, targetNet, targetEdge, orientation) {
  const b = edgeRows(baseNet, baseEdge), t = edgeRows(targetNet, targetEdge);
  if (b.along !== t.along) return Infinity;
  const rev = orientation === 'reversed';
  let worst = 0;
  for (let k = 0; k < b.boundary.length; k++) {
    const [bi, bj] = b.boundary[k];
    const [ti, tj] = t.boundary[rev ? b.boundary.length - 1 - k : k];
    worst = Math.max(worst, len(sub(euclid(baseNet[bi][bj]), euclid(targetNet[ti][tj]))));
  }
  return worst;
}

// The one entry point. Returns a NEW control net for the base; never mutates.
export function matchEdge(base, baseEdge, target, targetEdge, opts = {}) {
  const order = opts.order ?? 1;
  const blend = Math.max(0, Math.min(1, opts.blend ?? 1));
  if (order === 2) {
    return { ok: false, reason: 'G2 (curvature) is not built yet — it constrains a third control row through a relation this kernel has not re-derived and tested', net: null };
  }
  if (order !== 0 && order !== 1) return { ok: false, reason: `unknown continuity order ${order}`, net: null };

  const baseNet = base.ctrlNet, targetNet = target.ctrlNet;
  const b = edgeRows(baseNet, baseEdge);
  const t = edgeRows(targetNet, targetEdge);
  if (b.along !== t.along) {
    return { ok: false, reason: `the two edges carry different control counts (${b.along} and ${t.along}) — they need a shared knot vector along the seam before a row edit means anything`, net: null };
  }
  if (b.depth < 2) return { ok: false, reason: 'the base surface has no second control row to redirect', net: null };
  if (order === 1 && t.depth < 2) return { ok: false, reason: 'the target surface has no second control row to take a tangent direction from', net: null };

  const orientation = opts.orientation || edgeOrientation(baseNet, baseEdge, targetNet, targetEdge);
  const rev = orientation === 'reversed';
  const idx = (k) => (rev ? b.boundary.length - 1 - k : k);

  const net = cloneNet(baseNet);

  // --- G0: the boundary row BECOMES the target's, in the base's own weights.
  for (let k = 0; k < b.boundary.length; k++) {
    const [bi, bj] = b.boundary[k];
    const [ti, tj] = t.boundary[idx(k)];
    const from = euclid(baseNet[bi][bj]);
    const to = euclid(targetNet[ti][tj]);
    const wFrom = weightOf(baseNet[bi][bj]), wTo = weightOf(targetNet[ti][tj]);
    net[bi][bj] = homog(lerp(from, to, blend), wFrom + (wTo - wFrom) * blend);
  }
  if (order === 0) return { ok: true, net, order: 0, blend, orientation };

  // --- G1: REDIRECT the second row so the cross-boundary derivative continues
  // the target's through the seam. The target's own second row points INTO the
  // target, so for the two surfaces to read as one the base's must point the
  // opposite way — anti-parallel, not parallel. Getting that sign wrong folds
  // the base back on itself and still reports a perfect tangent ANGLE, which is
  // why the test asserts the surfaces lie on opposite sides of the seam.
  for (let k = 0; k < b.second.length; k++) {
    const [bi, bj] = b.second[k];
    const [b0i, b0j] = b.boundary[k];
    const [ti, tj] = t.second[idx(k)];
    const [t0i, t0j] = t.boundary[idx(k)];

    const P0 = euclid(net[b0i][b0j]);          // the edge point, already matched above
    const P1 = euclid(baseNet[bi][bj]);        // where the base's second row is now
    const magnitude = len(sub(P1, euclid(baseNet[b0i][b0j])));
    const targetDir = unit(sub(euclid(targetNet[ti][tj]), euclid(targetNet[t0i][t0j])));
    if (len(targetDir) < 0.5) continue;        // a degenerate target row says nothing; leave this column alone

    const wanted = [P0[0] - targetDir[0] * magnitude, P0[1] - targetDir[1] * magnitude, P0[2] - targetDir[2] * magnitude];
    net[bi][bj] = homog(lerp(P1, wanted, blend), weightOf(baseNet[bi][bj]));
  }
  return { ok: true, net, order: 1, blend, orientation };
}

// WHAT WAS ACTUALLY ACHIEVED, measured on the surfaces themselves rather than
// inferred from the control net. The angle between the two cross-boundary
// derivatives at a shared parameter: 180 degrees is tangent-continuous, because
// the two derivatives point away from the seam in opposite directions.
// Reported as the DEVIATION from that, so 0 means matched.
export function tangentDeviationAcross(base, baseEdge, target, targetEdge, samples = 9) {
  const dom = (knots, deg) => [knots[deg], knots[knots.length - 1 - deg]];
  const [buMin, buMax] = dom(base.knotsU, base.degU);
  const [bvMin, bvMax] = dom(base.knotsV, base.degV);
  const [tuMin, tuMax] = dom(target.knotsU, target.degU);
  const [tvMin, tvMax] = dom(target.knotsV, target.degV);

  // The parameter that runs ALONG each edge, and the cross-derivative to read.
  const paramsFor = (edge, uMin, uMax, vMin, vMax, s) => {
    if (edge === 'u0') return { u: uMin, v: vMin + (vMax - vMin) * s, cross: 'u', sign: +1 };
    if (edge === 'u1') return { u: uMax, v: vMin + (vMax - vMin) * s, cross: 'u', sign: -1 };
    if (edge === 'v0') return { u: uMin + (uMax - uMin) * s, v: vMin, cross: 'v', sign: +1 };
    return { u: uMin + (uMax - uMin) * s, v: vMax, cross: 'v', sign: -1 };
  };
  const orientation = edgeOrientation(base.ctrlNet, baseEdge, target.ctrlNet, targetEdge);
  let worst = 0;
  for (let i = 0; i < samples; i++) {
    const s = samples === 1 ? 0.5 : i / (samples - 1);
    const sT = orientation === 'reversed' ? 1 - s : s;
    const pb = paramsFor(baseEdge, buMin, buMax, bvMin, bvMax, s);
    const pt = paramsFor(targetEdge, tuMin, tuMax, tvMin, tvMax, sT);
    const B = surfacePointAndPartials(base, pb.u, pb.v);
    const T = surfacePointAndPartials(target, pt.u, pt.v);
    const db = unit((pb.cross === "u" ? B.su : B.sv).map((c) => c * pb.sign));
    const dt = unit((pt.cross === "u" ? T.su : T.sv).map((c) => c * pt.sign));
    if (len(db) < 0.5 || len(dt) < 0.5) continue;
    // Both point AWAY from the seam into their own surface, so a smooth join is
    // 180 degrees apart.
    const dot = Math.max(-1, Math.min(1, db[0] * dt[0] + db[1] * dt[1] + db[2] * dt[2]));
    worst = Math.max(worst, Math.abs(180 - (Math.acos(dot) * 180) / Math.PI));
  }
  return worst;
}

// MERGE — two patches becoming ONE surface, exactly.
//
// This is not an approximation and does not need to be. A B-spline with an
// interior knot of multiplicity equal to its degree is C0 there, and the two
// halves either side of it reproduce the original patches identically. So the
// merged surface IS both originals, joined, with the seam surviving as a knot
// rather than as a join between two objects. Whatever continuity the pair had
// before the merge — creased, tangent — it still has after; merging changes the
// bookkeeping, not the shape.
//
// The prerequisites are the ones Match Edge already needs: a shared description
// ALONG the seam (harmonizeDirections), and equal degree ACROSS it, which the
// caller supplies by elevating the lower one.
//
// ⚠ ORIENTATION IS HANDLED IN BOTH DIRECTIONS, and both matter. ACROSS the
// seam, each net is turned so the base's shared row is last and the target's is
// first. ALONG it, the target is reversed when the two edges run opposite ways.
// Getting the second wrong builds a surface that passes every count check and
// is folded through itself.

// A net as [cross][along], with the shared row at the requested end.
function orientForMerge(srf, edge, seamAt) {
  const crossIsU = edge === 'u0' || edge === 'u1';
  let net = srf.ctrlNet.map((r) => r.map((p) => [...p]));
  let crossDeg = srf.degU, crossKnots = srf.knotsU.slice();
  let alongDeg = srf.degV, alongKnots = srf.knotsV.slice();
  if (!crossIsU) {
    const nu = net.length, nv = net[0].length;
    const t = [];
    for (let j = 0; j < nv; j++) { const row = []; for (let i = 0; i < nu; i++) row.push(net[i][j]); t.push(row); }
    net = t;
    crossDeg = srf.degV; crossKnots = srf.knotsV.slice();
    alongDeg = srf.degU; alongKnots = srf.knotsU.slice();
  }
  const seamAtStart = edge === 'u0' || edge === 'v0';
  const wantStart = seamAt === 'start';
  if (seamAtStart !== wantStart) {
    net = net.slice().reverse();
    const lo = crossKnots[0], hi = crossKnots[crossKnots.length - 1];
    crossKnots = crossKnots.map((k) => lo + hi - k).reverse();
  }
  return { net, crossDeg, crossKnots, alongDeg, alongKnots };
}

const rescale = (knots, lo, hi) => {
  const a = knots[0], b = knots[knots.length - 1];
  const span = b - a;
  return span > 1e-12 ? knots.map((k) => lo + ((k - a) / span) * (hi - lo)) : knots.map(() => lo);
};

export function mergeAcrossSeam(base, baseEdge, target, targetEdge, opts = {}) {
  const B = orientForMerge(base, baseEdge, 'end');
  const T = orientForMerge(target, targetEdge, 'start');
  if (B.net[0].length !== T.net[0].length) {
    return { ok: false, reason: `the two edges carry different control counts along the seam (${B.net[0].length} and ${T.net[0].length}) — harmonise them first`, srf: null };
  }
  if (B.crossDeg !== T.crossDeg) {
    return { ok: false, reason: `the two surfaces differ in degree across the seam (${B.crossDeg} and ${T.crossDeg}) — elevate the lower one first`, srf: null };
  }
  const orientation = opts.orientation || 'aligned';
  if (orientation === 'reversed') T.net = T.net.map((row) => row.slice().reverse());

  const p = B.crossDeg;
  const bK = rescale(B.crossKnots, 0, 1);
  const tK = rescale(T.crossKnots, 1, 2);
  const interior = (knots, deg) => knots.slice(deg + 1, knots.length - deg - 1);
  const knots = [
    ...new Array(p + 1).fill(0),
    ...interior(bK, p),
    ...new Array(p).fill(1), // FULL MULTIPLICITY: C0 at the seam, which is what makes the join exact
    ...interior(tK, p),
    ...new Array(p + 1).fill(2),
  ];
  // The shared row appears once.
  const net = [...B.net, ...T.net.slice(1)];
  if (knots.length !== net.length + p + 1) {
    return { ok: false, reason: `merged knot vector does not match the merged control count (${knots.length} vs ${net.length + p + 1})`, srf: null };
  }
  return {
    ok: true,
    srf: { degU: p, degV: B.alongDeg, knotsU: knots, knotsV: B.alongKnots.slice(), ctrlNet: net },
  };
}

// G2 — CURVATURE CONTINUITY.
//
// G1 makes the two surfaces share a tangent plane along the seam. G2 makes them
// share the CURVATURE there as well, so a reflection running across the join
// does not merely stay connected but stops bending at it.
//
// THE CONSTRAINT, derived rather than transcribed. Along the seam both surfaces
// already agree (G0), so their derivatives ALONG it agree to every order. What
// is left is the cross-boundary direction. Writing the base's cross-derivative
// as D1 and its second as D2, and the target's as T1 and T2, the two patches
// meet with curvature continuity when the second fundamental forms agree at
// every shared parameter — and with G1 already imposed (D1 = -T1, the
// anti-parallel continuation), that reduces to matching the normal component of
// the second cross-derivative:
//
//     D2 . N  =  T2 . N        at every station along the seam
//
// which is a condition on the base's THIRD control row, because for a clamped
// B-spline the second cross-derivative at the boundary depends on exactly the
// first three rows. Solving it for that row is a linear step, and that is what
// this does.
//
// ⚠ ONLY THE NORMAL COMPONENT IS CONSTRAINED, and that is not a simplification
// — it is the difference between GEOMETRIC continuity and parametric. Forcing
// the whole vector would additionally demand the two surfaces share a
// parametrisation, which they have no reason to and which would drag the base's
// shape around for no visible gain. The tangential part is left where the base
// already had it, exactly as G1 preserves the base's own row spacing.
//
// ⚠ IT NEEDS THREE ROWS. A surface only two control points deep across the seam
// has no third row to solve for, and is refused by name rather than silently
// given G1.

// The row two steps in from an edge — the one G2 solves for.
function thirdRowIndices(ctrlNet, edge) {
  const nu = ctrlNet.length, nv = ctrlNet[0].length;
  if (edge === 'u0') return nu >= 3 ? ctrlNet[2].map((_, j) => [2, j]) : null;
  if (edge === 'u1') return nu >= 3 ? ctrlNet[nu - 3].map((_, j) => [nu - 3, j]) : null;
  if (edge === 'v0') return nv >= 3 ? ctrlNet.map((_, i) => [i, 2]) : null;
  return nv >= 3 ? ctrlNet.map((_, i) => [i, nv - 3]) : null;
}

// THE PARAMETER A CONTROL POINT ACTUALLY SITS AT — its Greville abscissa, the
// average of the q knots that span it. Registering row k to k/(count-1) instead
// is a uniform guess that mis-aims every solve away from the ends, because a
// clamped knot vector is not uniform near its boundary.
function grevilleAbscissae(knots, degree, count) {
  const out = [];
  for (let k = 0; k < count; k++) {
    let sum = 0;
    for (let i = 1; i <= degree; i++) sum += knots[k + i];
    out.push(sum / degree);
  }
  return out;
}

// The parameters of a station on an edge, and which second derivative crosses
// it.
function edgeStation(srf, edge, s) {
  const dom = (k, d) => [k[d], k[k.length - 1 - d]];
  const [uMin, uMax] = dom(srf.knotsU, srf.degU);
  const [vMin, vMax] = dom(srf.knotsV, srf.degV);
  if (edge === 'u0') return { u: uMin, v: vMin + (vMax - vMin) * s, cross: 'u' };
  if (edge === 'u1') return { u: uMax, v: vMin + (vMax - vMin) * s, cross: 'u' };
  if (edge === 'v0') return { u: uMin + (uMax - uMin) * s, v: vMin, cross: 'v' };
  return { u: uMin + (uMax - uMin) * s, v: vMax, cross: 'v' };
}

// NORMAL CURVATURE ACROSS THE BOUNDARY — the quantity G2 has to equalise.
//
// Not the raw second derivative: that depends on how fast each surface is
// parametrised, and two patches can be perfectly curvature-continuous while
// their second derivatives differ by a scale factor. The normal curvature
// (second fundamental form over the first) is the geometric quantity, and it is
// what a reflection line actually reads.
function normalCurvatureAcross(srf, edge, s, derivs2) {
  const st = edgeStation(srf, edge, s);
  const d = derivs2(srf, st.u, st.v);
  const cross = st.cross === 'u' ? d.Su : d.Sv;
  const second = st.cross === 'u' ? d.Suu : d.Svv;
  const n = [
    d.Su[1] * d.Sv[2] - d.Su[2] * d.Sv[1],
    d.Su[2] * d.Sv[0] - d.Su[0] * d.Sv[2],
    d.Su[0] * d.Sv[1] - d.Su[1] * d.Sv[0],
  ];
  const nl = len(n);
  const cl2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
  if (nl < 1e-14 || cl2 < 1e-18) return null;
  const N = [n[0] / nl, n[1] / nl, n[2] / nl];
  return { kn: (second[0] * N[0] + second[1] * N[1] + second[2] * N[2]) / cl2, N };
}

// Apply G2 on top of a net that has already been matched to G1.
//
// SOLVED NUMERICALLY, AND DELIBERATELY SO. The normal curvature at a boundary
// station is LINEAR in the third control row's position — moving that row along
// the normal by d changes the second derivative by a fixed multiple of d and
// changes nothing else, because the first derivative depends only on the two
// rows nearer the edge. So two evaluations give the exact coefficient, and the
// step is a division. Deriving the closed-form B-spline boundary coefficient
// instead would mean transcribing a knot-spacing expression that is easy to get
// subtly wrong and impossible to check by eye — the first attempt here did
// exactly that, squared the first-derivative factor, and moved the curvature
// jump by 3%.
export function applyG2(baseSrf, baseEdge, targetSrf, targetEdge, g1Net, opts = {}) {
  const derivs2 = opts.derivs2;
  if (typeof derivs2 !== 'function') return { ok: false, reason: 'G2 needs a second-derivative evaluator', net: null };
  const blend = Math.max(0, Math.min(1, opts.blend ?? 1));
  const orientation = opts.orientation || 'aligned';
  const bThird = thirdRowIndices(g1Net, baseEdge);
  const tThird = thirdRowIndices(targetSrf.ctrlNet, targetEdge);
  if (!bThird) return { ok: false, reason: 'this surface is only two control points deep across the seam, so it has no third row to solve for — rebuild it deeper to reach G2', net: null };
  if (!tThird) return { ok: false, reason: 'the neighbour is only two control points deep across the seam, so it carries no curvature there to match', net: null };
  const b = edgeRows(g1Net, baseEdge);
  const t = edgeRows(targetSrf.ctrlNet, targetEdge);
  if (b.along !== t.along) return { ok: false, reason: 'the two edges carry different control counts along the seam — harmonise them first', net: null };

  const net = g1Net.map((row) => row.map((cp) => [...cp]));
  const rev = orientation === 'reversed';
  const count = bThird.length;
  // ⚠ THE STATIONS ARE COUPLED, so one pass overshoots. Moving row k changes the
  // curvature at its neighbours too — the along-seam basis functions overlap —
  // so solving each station against a target its neighbours are still moving
  // lands past it. Measured on a five-deep patch: a single pass took the base
  // from 0.0415 to -0.012 against a target of 0.0020, overshooting worst at
  // mid-span, which is exactly where the overlap is greatest.
  //
  // The coupling is LINEAR, so sweeping repeatedly converges (this is
  // Gauss-Seidel on a small, diagonally dominant system). Bounded, and it stops
  // as soon as a sweep stops moving anything.
  const MAX_SWEEPS = 12;
  const SETTLED = 1e-9;
  // The along-seam direction and its knots, so each row is solved at the
  // parameter it actually governs.
  const alongIsV = baseEdge === 'u0' || baseEdge === 'u1';
  const alongKnots = alongIsV ? baseSrf.knotsV : baseSrf.knotsU;
  const alongDeg = alongIsV ? baseSrf.degV : baseSrf.degU;
  const gre = grevilleAbscissae(alongKnots, alongDeg, count);
  const lo = alongKnots[alongDeg], hi = alongKnots[alongKnots.length - 1 - alongDeg];
  for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
  let worstMove = 0;
  for (let k = 0; k < count; k++) {
    const s = hi > lo ? Math.max(0, Math.min(1, (gre[k] - lo) / (hi - lo))) : 0.5;
    const sT = rev ? 1 - s : s;
    const want = normalCurvatureAcross(targetSrf, targetEdge, sT, derivs2);
    const have = normalCurvatureAcross({ ...baseSrf, ctrlNet: net }, baseEdge, s, derivs2);
    if (!want || !have) continue;
    // ⚠ THE TWO NORMALS NEED NOT POINT THE SAME WAY. Each is Su x Sv in its own
    // surface's parametrisation, and two patches that meet perfectly can still
    // be built with opposite orientation — which flips the SIGN of the normal
    // curvature and turns a match into a doubling. Compared through the base's
    // normal, so the quantity means the same thing on both sides.
    const wantKn = (want.N[0] * have.N[0] + want.N[1] * have.N[1] + want.N[2] * have.N[2]) < 0 ? -want.kn : want.kn;

    // One unit probe along the normal gives the exact linear coefficient.
    const [i2, j2] = bThird[k];
    const keep = [...net[i2][j2]];
    const P = euclid(keep), w = weightOf(keep);
    net[i2][j2] = homog([P[0] + have.N[0], P[1] + have.N[1], P[2] + have.N[2]], w);
    const probed = normalCurvatureAcross({ ...baseSrf, ctrlNet: net }, baseEdge, s, derivs2);
    net[i2][j2] = keep;
    if (!probed) continue;
    const slope = probed.kn - have.kn;
    if (Math.abs(slope) < 1e-14) continue; // this row cannot influence the curvature here
    const delta = ((wantKn - have.kn) / slope) * blend;
    if (!Number.isFinite(delta)) continue;
    net[i2][j2] = homog([P[0] + have.N[0] * delta, P[1] + have.N[1] * delta, P[2] + have.N[2] * delta], w);
    worstMove = Math.max(worstMove, Math.abs(delta));
  }
  if (worstMove < SETTLED) break;
  }
  return { ok: true, net };
}

// HOW MUCH THE CURVATURE JUMPS ACROSS THE SEAM.
//
// Measured as the NORMAL CURVATURE in the cross-boundary direction, which is
// exactly the quantity curvature continuity is about and exactly what G2
// solves. Mean curvature is the wrong instrument here: it averages both
// principal directions, so it moves when the along-seam curvature moves — and
// that one is already shared by G0 and has nothing to do with the join.
//
// Relative, because an absolute difference means nothing without a scale:
// 0.001 per mm is enormous on a 5mm fillet and invisible on a 2m panel.
//
// ⚠⚠ BUT THE SCALE CANNOT BE THE LARGER OF THE TWO CURVATURES ALONE. Against a
// PLANAR target the target's normal curvature across the seam is exactly zero,
// and a successful G2 drives the base's to zero with it — so a ratio floored at
// 1e-9 divides a residual of 1e-9 by a scale of 1e-9 and returns 1.0, the worst
// reading the measure can produce, at the exact moment the match is perfect. A
// zebra across that join is smooth while the number calls it a total break.
//
// So the floor is a curvature the MODEL can recognise: one over the base's own
// size. Where both sides are genuinely curved this is the same relative measure
// it always was; where both approach flat it degrades to an absolute difference
// expressed in units of the surface, which is the only thing left that means
// anything.
function netSpan(ctrlNet) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const row of ctrlNet) for (const cp of row) for (let k = 0; k < 3; k++) {
    if (cp[k] < lo[k]) lo[k] = cp[k];
    if (cp[k] > hi[k]) hi[k] = cp[k];
  }
  const d = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return Number.isFinite(d) && d > 0 ? d : 1;
}
// ⚠⚠ RETURNS null WHEN IT CANNOT MEASURE, never NaN. A refined or force-
// harmonised seam can carry zero-width knot spans, and a derivative taken there
// is not a number — which propagated all the way to the Properties panel and
// printed "curvature NaN" beside a match the app was otherwise calling
// achieved. A readout that prints NaN is worse than one that says nothing: it
// looks like a measurement. Stations that cannot be evaluated are skipped, and
// if none survives the answer is "not measurable" rather than a number.
export function curvatureDeviationAcross(base, baseEdge, target, targetEdge, derivs2, samples = 7) {
  const orientation = edgeOrientation(base.ctrlNet, baseEdge, target.ctrlNet, targetEdge);
  const kRef = 1 / netSpan(base.ctrlNet);
  let worst = 0, measured = 0;
  for (let i = 1; i < samples - 1; i++) { // skip the corners, where a neighbouring edge's own behaviour intrudes
    const s = i / (samples - 1);
    const sT = orientation === 'reversed' ? 1 - s : s;
    const cb = normalCurvatureAcross(base, baseEdge, s, derivs2);
    const ct = normalCurvatureAcross(target, targetEdge, sT, derivs2);
    if (!cb || !ct || !Number.isFinite(cb.kn) || !Number.isFinite(ct.kn)) continue;
    // Same orientation caveat as the solve: compare through ONE normal.
    const ctKn = (ct.N[0] * cb.N[0] + ct.N[1] * cb.N[1] + ct.N[2] * cb.N[2]) < 0 ? -ct.kn : ct.kn;
    const scale = Math.max(Math.abs(cb.kn), Math.abs(ctKn), kRef);
    const d = Math.abs(cb.kn - ctKn) / scale;
    if (!Number.isFinite(d)) continue;
    worst = Math.max(worst, d);
    measured++;
  }
  return measured ? worst : null;
}
