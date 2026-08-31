// THE EXACT FORM OF A SEWN SOLID'S EDGES
// ================================================================
// Turns the over-decomposed edge set a boolean sews into the edges a B-rep
// actually has, each carrying a fitted NURBS curve.
//
// WHY THERE IS ANYTHING TO DO. `sewFragments` builds topology from the
// MARCHED POLYLINE, so every sample point along an intersection becomes a
// topological vertex and every step between two samples becomes an edge.
// Measured on the banked torus pair's union: 18 faces, 526 edges, 506
// vertices — and 476 of those vertices have edge-degree TWO. A degree-2
// vertex is not a corner; it is a point in the middle of a smooth run that
// happens to be where the marcher took a sample. Only ~30 vertices are
// genuine branch points.
//
// So the merge is not an optimisation, it is the difference between a B-rep
// and a polyline wearing one. A face boundary here is 76 straight segments
// where the geometry is one smooth curve.
//
// ⚠ THIS DOES NOT MUTATE THE SOLID, and it deliberately does NOT write
// `curve3d` on the original edges. An edge spanning a single sample step has
// no curve worth storing — assigning it the whole chain's curve would make
// every one of 76 edges claim to be the same curve, which is worse than the
// null it replaces. The fitted curve belongs to the MERGED edge, so that is
// what this returns: a derived record beside the solid, leaving every
// existing invariant (naked-edge count included) untouched by construction.
//
// The (u,v) side lives at the bottom of this file — `fitHalfEdgePcurves`. It
// is a separate pass on purpose: the 3-D chains are a property of the SOLID,
// while a pcurve is a property of a half-edge's own FACE, and one chain yields
// a different one on each of the two faces it separates.
import { fitCurveToPoints } from './fitcurve.mjs';

// Which edges touch each vertex. An edge's two vertices are the origins of
// its two half-edges — the pair is created together and never exists singly,
// so this is total.
function vertexIncidence(solid) {
  const inc = new Map();
  const add = (v, e) => {
    if (!v) return;
    let list = inc.get(v);
    if (!list) { list = []; inc.set(v, list); }
    list.push(e);
  };
  for (const e of solid.edges) {
    add(e.halfEdges[0] && e.halfEdges[0].vertex, e);
    add(e.halfEdges[1] && e.halfEdges[1].vertex, e);
  }
  return inc;
}

function otherVertex(edge, v) {
  const a = edge.halfEdges[0] && edge.halfEdges[0].vertex;
  const b = edge.halfEdges[1] && edge.halfEdges[1].vertex;
  return a === v ? b : a;
}

// Walk from a corner along degree-2 vertices until the next corner. The chain
// that comes back is one B-rep edge's worth of geometry.
function walkChain(startV, startE, inc, used) {
  const points = [startV.point];
  const edges = [];
  let v = startV, e = startE;
  for (;;) {
    if (used.has(e)) return null; // another walk already claimed this run
    used.add(e);
    edges.push(e);
    const nv = otherVertex(e, v);
    if (!nv) return null;
    points.push(nv.point);
    const list = inc.get(nv) || [];
    if (list.length !== 2) return { points, edges, endVertex: nv, closed: false };
    const next = list[0] === e ? list[1] : list[0];
    if (used.has(next)) return { points, edges, endVertex: nv, closed: nv === startV };
    v = nv; e = next;
  }
}

// A run with no corner anywhere on it — an intersection loop that closes on
// itself without ever branching. It still has to become one edge; the start
// is arbitrary and the chain is marked closed so the fit wraps.
function walkClosed(startE, inc, used) {
  const v0 = startE.halfEdges[0].vertex;
  const points = [v0.point];
  const edges = [];
  let v = v0, e = startE;
  for (;;) {
    if (used.has(e)) break;
    used.add(e);
    edges.push(e);
    const nv = otherVertex(e, v);
    if (!nv) break;
    if (nv === v0) break; // closed; the repeated first point is not carried
    points.push(nv.point);
    const list = inc.get(nv) || [];
    if (list.length !== 2) break;
    const next = list[0] === e ? list[1] : list[0];
    v = nv; e = next;
  }
  return { points, edges, endVertex: v0, closed: true };
}

// opts: { tolerance } — the bound every fitted curve is held to, and the same
// number the file's own model tolerance should supply.
// Returns { ok, edges: [...], stats } where each entry is
//   { points, sourceEdges, closed, curve, kind, maxDeviation, ctrlPtCount }
// or, where no curve met the bound, { ..., curve: null, reason }.
export function fitSolidEdgeCurves(solid, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-3;
  if (!solid || !Array.isArray(solid.edges)) return { ok: false, reason: 'fitSolidEdgeCurves needs a sewn solid' };
  const inc = vertexIncidence(solid);
  const corners = [];
  for (const [v, list] of inc) if (list.length !== 2) corners.push(v);

  const used = new Set();
  const chains = [];
  for (const c of corners) {
    for (const e of inc.get(c) || []) {
      if (used.has(e)) continue;
      const chain = walkChain(c, e, inc, used);
      if (chain && chain.points.length >= 2) chains.push({ ...chain, startVertex: c });
    }
  }
  // Whatever is left touches no corner at all.
  for (const e of solid.edges) {
    if (used.has(e)) continue;
    const chain = walkClosed(e, inc, used);
    if (chain && chain.points.length >= 3) chains.push({ ...chain, startVertex: chain.endVertex });
  }

  const out = [];
  let worst = 0, refused = 0, exact = 0;
  let relaxedFits = 0, worstRelaxed = 0;
  for (const chain of chains) {
    // A two-point chain is a straight segment between two genuine corners —
    // already exact, and running it through a least-squares fit would be
    // slower and no better.
    let res = chain.points.length === 2
      ? { ok: true, kind: 'line', curve: { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...chain.points[0], 1], [...chain.points[1], 1]] }, maxDeviation: 0, ctrlPtCount: 2 }
      : fitCurveToPoints(chain.points, { tolerance, closed: chain.closed });
    /* ⚠⚠ AN EDGE THAT MISSES THE BOUND BY A HAIR MUST NOT COST THE WHOLE SOLID.
       A refused fit leaves that edge with no curve, which takes every trim
       running along it out of its face's loop; the loop then has a hole the
       width of the edge, the record refuses, and a capped solid drawn with one
       curved side exports as loose surfaces instead of a solid. Measured on the
       plainest such shape: a wavy rim fitted to 1.158e-3 against a 1.0e-3 bound
       — sixteen per cent over, on sampled points whose own spacing is coarser
       than either number, which is the case the fitter's own refusal text warns
       about ("if the bound is below the accuracy of whatever produced these
       points, no curve can meet it").
       A B-rep carries a PER-EDGE tolerance and the writer already hands it to
       ON_Brep::NewEdge, so an edge fitted to its own achievable accuracy is
       writable exactly as it stands. Only the all-or-nothing refusal stood in
       the way, so it is retried once at the accuracy the fitter itself reported.
       ⚠ BOUNDED, AND NOT BY A ROUND NUMBER. The relaxation is capped at a ten
       thousandth of the chain's own extent — the same fraction the boundary
       sampler uses to space these points in the first place — or ten times the
       requested tolerance, whichever is larger. An edge missing by more than the
       spacing of its own samples is not a tolerance question, it is a bad edge,
       and it still refuses. Counted, so the relaxation is never silent. */
    if (!res.ok && Number.isFinite(res.bestDeviation) && res.bestDeviation > 0) {
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const q of chain.points) for (let k = 0; k < 3; k++) { if (q[k] < lo[k]) lo[k] = q[k]; if (q[k] > hi[k]) hi[k] = q[k]; }
      const extent = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
      const relaxCap = Math.max(tolerance * 10, extent * 1e-4);
      if (res.bestDeviation <= relaxCap) {
        const retry = fitCurveToPoints(chain.points, { tolerance: res.bestDeviation * (1 + 1e-6), closed: chain.closed });
        if (retry.ok) {
          relaxedFits++;
          if (retry.maxDeviation > worstRelaxed) worstRelaxed = retry.maxDeviation;
          res = retry;
        }
      }
    }
    if (res.ok) {
      if (res.kind !== 'nurbs') exact++;
      if (res.maxDeviation > worst) worst = res.maxDeviation;
      out.push({ points: chain.points, sourceEdges: chain.edges, closed: chain.closed, curve: res.curve, kind: res.kind, maxDeviation: res.maxDeviation, ctrlPtCount: res.ctrlPtCount });
    } else {
      refused++;
      out.push({ points: chain.points, sourceEdges: chain.edges, closed: chain.closed, curve: null, kind: null, maxDeviation: null, ctrlPtCount: 0, reason: res.reason });
    }
  }

  // Every original edge must land in exactly one chain. A missed edge is a
  // hole in the exported boundary and a doubled one is a duplicate edge, and
  // neither shows up as anything but a wrong picture later.
  const covered = new Set();
  let doubled = 0;
  for (const c of out) for (const e of c.sourceEdges) { if (covered.has(e)) doubled++; covered.add(e); }

  return {
    ok: refused === 0 && doubled === 0 && covered.size === solid.edges.length,
    edges: out,
    stats: {
      originalEdges: solid.edges.length,
      originalVertices: solid.vertices.length,
      cornerVertices: corners.length,
      fittedEdges: out.length,
      exactPrimitives: exact,
      relaxedFits,      // edges fitted to their own achievable accuracy rather than refused
      worstRelaxed,     // the loosest of those, which is what the file will declare as that edge's tolerance
      refusedChains: refused,
      coveredEdges: covered.size,
      doubledEdges: doubled,
      worstDeviation: worst,
      totalControlPoints: out.reduce((s, c) => s + c.ctrlPtCount, 0),
    },
  };
}

// ---------------------------------------------------------------------------
// THE (u,v) SIDE — a pcurve per half-edge.
//
// A B-rep trim is a curve in its OWN face's parameters, and every fitted edge
// above is shared by two half-edges sitting on two different faces. So one
// 3-D chain yields TWO pcurves, generally of different shape, and neither is
// derivable from the other: the same space curve has a different (u,v) image
// on each surface it lies on. This is the last thing standing between a fitted
// solid and a .3dm the authoring bindings can write.
//
// ⚠ THE PROJECTION IS THE RISK, not the fitting. Inversion goes through
// closestPointOnSurface, whose seed grid is sized by the control net, so a
// busy surface on a small net can settle Newton in the wrong basin and return
// a plausible (u,v) in the wrong place. `projectPointsToSurfaceUV` already
// carries the oracle for that — it re-evaluates the surface at every recovered
// parameter and refuses the whole chain when any sample misses the 3-D point
// it came from. Nothing here second-guesses that refusal; it is reported.
//
// ⚠ AND A CHAIN CROSSING A SEAM IS NOT ONE PCURVE. On a closed direction the
// chain leaves the parametric rectangle and re-enters at the far edge, and a
// single curve through those samples would carry a phantom chord straight
// across the face. `seamOpenChains` already splits exactly there, giving
// pieces that each reach the rectangle's edge where the curve genuinely leaves
// it — so a seam crossing produces SEVERAL pcurves for one 3-D chain, which is
// what the topology actually needs, and is reported as such rather than being
// flattened back into one.
import { projectPointsToSurfaceUV, seamOpenChains } from './trim.mjs';
import { surfaceClosure } from './surface.mjs';

// The face a half-edge sits on, and the surface under it.
function faceSurfaceOf(halfEdge) {
  const loop = halfEdge && halfEdge.loop;
  const face = loop && loop.face;
  return face && face.surface ? { face, srf: face.surface } : null;
}

// opts: { tolerance, projectTolerance }
// Returns { ok, pcurves: [...], stats }. Each entry:
//   { edgeIndex, side (0|1), faceId, pieces: [{ uv, curve, kind, maxDeviation }] }
// or { ..., pieces: null, reason } where the projection or a fit refused.
export function fitHalfEdgePcurves(solid, fitted, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-3;
  // The projection bound is about "is this chain ON this surface", which is a
  // different question from "how closely does a curve follow it", so it gets
  // its own number and defaults to the same value rather than borrowing it
  // silently.
  const projectTolerance = opts.projectTolerance ?? tolerance;
  if (!solid || !fitted || !Array.isArray(fitted.edges)) {
    return { ok: false, reason: 'fitHalfEdgePcurves needs a solid and the result of fitSolidEdgeCurves' };
  }
  const out = [];
  let refused = 0, seamSplit = 0, pieces = 0, worst = 0;

  for (const [edgeIndex, chain] of fitted.edges.entries()) {
    // The chain's own source edges name the two half-edges it runs between.
    // Taking the sides from the FIRST source edge is enough: every edge in a
    // chain separates the same two faces, which is what made it one chain.
    const first = chain.sourceEdges && chain.sourceEdges[0];
    if (!first) continue;
    for (const side of [0, 1]) {
      const he = first.halfEdges[side];
      const fs = faceSurfaceOf(he);
      if (!fs) { out.push({ edgeIndex, side, faceId: null, pieces: null, reason: 'the half-edge has no face surface to project onto' }); refused++; continue; }

      const proj = projectPointsToSurfaceUV(chain.points, fs.srf, { tolerance: projectTolerance });
      if (!proj.ok) { out.push({ edgeIndex, side, faceId: fs.face.id, pieces: null, reason: proj.reason }); refused++; continue; }

      // Split at seam crossings if this surface is closed anywhere.
      const closure = surfaceClosure(fs.srf);
      let uvChains = [proj.uv];
      if (closure.closedU || closure.closedV) {
        const split = seamOpenChains(proj.uv, fs.srf);
        if (split.ok && split.chains.length) {
          uvChains = split.chains;
          if (split.chains.length > 1) seamSplit++;
        }
        // A refusal here means "no crossing to split at" for most codes, which
        // leaves the single chain correct as it stands. It is not an error.
      }

      const built = [];
      let bad = null;
      for (const uv of uvChains) {
        if (uv.length < 2) continue;
        // (u,v) fitted as a planar 3-D curve at z = 0 — which is exactly what
        // a pcurve is, and lets the same fitter and the same conservative
        // deviation measure apply without a second implementation.
        const pts = uv.map((p) => [p[0], p[1], 0]);
        const res = fitCurveToPoints(pts, { tolerance, closed: false });
        if (!res.ok) { bad = res.reason; break; }
        if (res.maxDeviation > worst) worst = res.maxDeviation;
        built.push({ uv, curve: res.curve, kind: res.kind, maxDeviation: res.maxDeviation });
      }
      if (bad || !built.length) { out.push({ edgeIndex, side, faceId: fs.face.id, pieces: null, reason: bad || 'every (u,v) piece was degenerate' }); refused++; continue; }
      pieces += built.length;
      out.push({ edgeIndex, side, faceId: fs.face.id, pieces: built });
    }
  }

  return {
    ok: refused === 0,
    pcurves: out,
    stats: {
      fittedEdges: fitted.edges.length,
      sides: out.length,
      withPcurves: out.filter((p) => p.pieces).length,
      refusedSides: refused,
      seamSplitSides: seamSplit,
      totalPieces: pieces,
      worstDeviation: worst,
    },
  };
}

// ---------------------------------------------------------------------------
// FACE LOOPS, IN TRAVERSAL ORDER — what a B-rep loop actually is.
//
// `fitHalfEdgePcurves` above answers "what is this edge, in that face's
// parameters", which is the right question per edge and the wrong shape for a
// loop. A B-rep loop is an ORDERED, ORIENTED cycle: consecutive trims must
// join end-to-start in (u,v), and each one carries a flag saying whether it
// runs with or against its edge's own 3-D curve. Assembling from edge order
// instead produced exactly the complaint OpenNURBS makes:
//
//     brep.m_L[0] loop is not valid.
//       end of brep.m_T[loop.m_ti[0]=0] and start of brep.m_T[loop.m_ti[1]=1]
//       do not match.
//
// So this walks the half-edge cycle FIRST and derives everything from the walk.
// The points handed to the projector are already in traversal order, which
// means each pcurve is correctly oriented BY CONSTRUCTION rather than fitted
// and then flipped — there is no second representation to keep in step.
import { curvePoint } from './curve.mjs';
function samePoint3(a, b, tol) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tol;
}

// The ordered half-edges of one loop. A loop is a cycle through `next`, so the
// walk is bounded by a guard rather than trusted: a malformed cycle is a hang,
// and a hang inside an exporter reads as the file being enormous.
function loopHalfEdges(loop, limit = 100000) {
  const out = [];
  let he = loop.halfEdge;
  if (!he) return out;
  do { out.push(he); he = he.next; } while (he && he !== loop.halfEdge && out.length < limit);
  return out;
}

// opts: { tolerance, projectTolerance, joinTolerance }
// Returns { ok, faces: [{ faceId, srf, loops: [{ loopType, trims: [...] }] }], stats }
// Each trim: { chainIndex, reversed, points, uv, curve, kind, maxDeviation }
// How far a curve wanders from the polyline it was fitted through — the
// direction `maxDeviationFromCurve` does not measure. Sampled on the curve,
// each sample taken to the nearest segment of the run.
function pointSegDistSq2(q, a, b) {
  const ax = b[0] - a[0], ay = b[1] - a[1];
  const qx = q[0] - a[0], qy = q[1] - a[1];
  const len2 = ax * ax + ay * ay;
  let t = len2 > 0 ? (qx * ax + qy * ay) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const dx = qx - ax * t, dy = qy - ay * t;
  return dx * dx + dy * dy;
}
function curveExcursionFromPolyline(crv, pts, samples = 128) {
  if (!crv || !crv.knots || pts.length < 2) return 0;
  const p = crv.degree, U = crv.knots;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  if (!(t1 > t0)) return 0;
  let worstSq = 0;
  for (let i = 0; i <= samples; i++) {
    const q = curvePoint(crv, t0 + (t1 - t0) * (i / samples));
    if (!q) continue;
    let best = Infinity;
    for (let k = 0; k + 1 < pts.length; k++) {
      const d = pointSegDistSq2(q, pts[k], pts[k + 1]);
      if (d < best) best = d;
    }
    if (best > worstSq) worstSq = best;
  }
  return Math.sqrt(worstSq);
}
// Degree 1 through every point: clamped, uniform, and exactly the run it was
// given. #knots = #ctrlPts + degree + 1 by construction.
function polylineCurveThrough(rawPts) {
  /* ⚠ CONSECUTIVE DUPLICATES FIRST. A boundary walk can hand back the same
     (u,v) twice, and a degree-1 curve through them has a ZERO-LENGTH span —
     which OpenNURBS turns into a NaN while computing the trim's own tolerance
     and then refuses the whole brep by name ("cv[0] = -nan is not valid"),
     degrading a joined solid into loose faces. The duplicate carries no
     direction and describes no part of the boundary, so dropping it is exact. */
  const pts = [];
  for (const q of rawPts) {
    const last = pts[pts.length - 1];
    if (last && Math.hypot(q[0] - last[0], q[1] - last[1]) <= 1e-12) continue;
    pts.push(q);
  }
  if (pts.length < 2) return null;
  const m = pts.length - 1;
  const knots = [0, 0];
  for (let i = 1; i < m; i++) knots.push(i);
  knots.push(m, m);
  // ⚠ FOUR COMPONENTS, WEIGHT INCLUDED. Every control point in this kernel is
  // euclidean-plus-weight, and the file writer destructures all four and
  // multiplies through — a three-component point makes `w` undefined and every
  // coordinate NaN, which OpenNURBS refuses by name ("cv[0] = -nan is not
  // valid") and which then degrades the whole joined solid into loose faces.
  return { degree: 1, knots, ctrlPts: pts.map((q) => [q[0], q[1], q.length > 2 ? q[2] : 0, 1]) };
}
export function fitFaceLoops(solid, fitted, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-3;
  const projectTolerance = opts.projectTolerance ?? tolerance;
  // How close two 3-D points must be to count as the same corner when deciding
  // which way round a group runs. Deliberately looser than the fit tolerance:
  // this is an identity question about welded vertices, not an accuracy one.
  const joinTolerance = opts.joinTolerance ?? Math.max(tolerance, 1e-6) * 10;
  if (!solid || !fitted || !Array.isArray(fitted.edges)) {
    return { ok: false, reason: 'fitFaceLoops needs a solid and the result of fitSolidEdgeCurves' };
  }

  const edgeToChain = new Map();
  fitted.edges.forEach((c, i) => { for (const e of c.sourceEdges) edgeToChain.set(e, i); });

  const faces = solid.shells.flatMap((sh) => sh.faces || []);
  const out = [];
  let trimCount = 0, refused = 0, reversedCount = 0, worst = 0, mergedWraps = 0, degenerateTrims = 0, collapsedInUV = 0;
  let ringingTrims = 0, worstExcursion = 0;
  // Below this, a (u,v) run is a point rather than a curve. Scaled off the
  // fit tolerance rather than fixed, since parameter space has no units.
  const degenerateUV = (opts.degenerateUV ?? 1e-12);
  // In model units, so it is the same kind of number the file's tolerance is.
  const degenerate3d = (opts.degenerate3d ?? Math.max(tolerance, 1e-9));

  for (const face of faces) {
    if (!face.surface) { refused++; out.push({ faceId: face.id, srf: null, loops: null, reason: 'the face carries no surface' }); continue; }
    const srf = face.surface;
    const loopsOut = [];
    let faceBad = null;

    for (const loop of face.loops || []) {
      const hes = loopHalfEdges(loop);
      if (!hes.length) continue;

      // Group CONSECUTIVE half-edges that belong to the same fitted chain —
      // one B-rep trim per group, because the merge already decided that a
      // run of degree-2 vertices is one edge.
      const groups = [];
      for (const he of hes) {
        const ci = edgeToChain.get(he.edge);
        const last = groups[groups.length - 1];
        if (last && last.chainIndex === ci) last.halfEdges.push(he);
        else groups.push({ chainIndex: ci, halfEdges: [he] });
      }
      // ⚠ THE WALK CAN START MID-CHAIN. `loop.halfEdge` is whichever half-edge
      // the builder happened to store, so a single chain can appear as the
      // first group AND the last one. Left alone that emits the same edge
      // twice and neither piece joins its neighbour.
      if (groups.length > 1 && groups[0].chainIndex === groups[groups.length - 1].chainIndex) {
        const tail = groups.pop();
        groups[0].halfEdges = [...tail.halfEdges, ...groups[0].halfEdges];
        mergedWraps++;
      }

      // ⚠⚠ PROJECT THE WHOLE LOOP AS ONE WALK, THEN UNWRAP. Projecting each
      // group independently is what MANUFACTURED the seam problem: the
      // projector wraps parameters into the domain, so a point sitting near a
      // seam comes back at either end depending on where its own walk started,
      // and two adjacent trims end up on opposite edges of the rectangle. That
      // reads exactly like a face that wraps the closed direction, and none of
      // these do — measured on all three booleans, 36 / 12 / 24 loops, every
      // one fitting inside a single period once unwrapped and not one
      // genuinely wrapping. So the cure is a consistent walk, not the seam
      // trims the symptom asks for.
      const flat = [];
      const spans = [];
      for (const g of groups) {
        const pts = g.halfEdges.map((h) => h.vertex.point);
        const lastHe = g.halfEdges[g.halfEdges.length - 1];
        const end = lastHe.twin && lastHe.twin.vertex ? lastHe.twin.vertex.point : null;
        if (end) pts.push(end);
        spans.push([flat.length, pts.length]);
        for (const q of pts) flat.push(q);
      }
      const loopProj = projectPointsToSurfaceUV(flat, srf, { tolerance: projectTolerance });
      if (!loopProj.ok) { faceBad = loopProj.reason; break; }
      const loopUV = loopProj.uv.map((q) => q.slice());
      const closureL = surfaceClosure(srf);
      for (const [ai, isClosed] of [[0, closureL.closedU], [1, closureL.closedV]]) {
        if (!isClosed) continue;
        const knots = ai === 0 ? srf.knotsU : srf.knotsV;
        const lo = knots[0], hi = knots[knots.length - 1], P = hi - lo;
        if (!(P > 0)) continue;
        let prev = loopUV[0][ai], min = prev, max = prev;
        for (let i = 1; i < loopUV.length; i++) {
          let d = loopUV[i][ai] - prev;
          d -= P * Math.round(d / P);
          prev += d;
          loopUV[i][ai] = prev;
          if (prev < min) min = prev;
          if (prev > max) max = prev;
        }
        // ⚠⚠ FLOOR, NOT ROUND. The shift must land the range INSIDE the
        // domain, and rounding sends it out: a run at min = 0.666667 with
        // period 1 rounds to a shift of 1, putting the range at [-0.333, 0]
        // — entirely below the rectangle — after which the clamp below
        // squashes all 40 points onto a single parameter. That is a real 3-D
        // run 391 units long arriving as a point in (u,v), and it read
        // convincingly like a pole. Floor puts min - shift in [lo, lo + P) by
        // construction. The epsilon keeps a min a hair below lo from wrapping
        // a whole period to the far edge.
        const shift = P * Math.floor((min - lo) / P + 1e-12);
        for (const q of loopUV) q[ai] = Math.min(hi, Math.max(lo, q[ai] - shift));
      }

      const trims = [];
      for (const [gi, g] of groups.entries()) {
        // Traversal-order points: each half-edge's ORIGIN, then the final
        // half-edge's destination (its twin's origin). This is the loop's own
        // direction, which is the direction a trim must run in.
        const [off, len] = spans[gi];
        const pts = flat.slice(off, off + len);
        const uvSlice = loopUV.slice(off, off + len);
        if (pts.length < 2) continue;

        const chain = fitted.edges[g.chainIndex];
        // Which way does this group run along its chain? Compared at the
        // START, and disambiguated by the SECOND point when the chain is
        // closed and both ends are the same corner.
        let reversed = false;
        if (chain && chain.points.length >= 2) {
          const cFirst = chain.points[0], cLast = chain.points[chain.points.length - 1];
          if (samePoint3(pts[0], cFirst, joinTolerance) && !samePoint3(pts[0], cLast, joinTolerance)) reversed = false;
          else if (samePoint3(pts[0], cLast, joinTolerance) && !samePoint3(pts[0], cFirst, joinTolerance)) reversed = true;
          else {
            // Ambiguous ends (a closed chain): decide on the next step.
            const fwd = samePoint3(pts[1], chain.points[1], joinTolerance);
            const rev = samePoint3(pts[1], chain.points[chain.points.length - 2], joinTolerance);
            reversed = rev && !fwd;
          }
        }
        if (reversed) reversedCount++;

        // No seam split here, and that is the point: the loop was unwrapped as
        // a whole above, so a group's (u,v) is already continuous and already
        // inside the rectangle. Splitting it would reintroduce the very break
        // the unwrap removed.
        const uvChains = [uvSlice];
        for (const uv of uvChains) {
          if (uv.length < 2) continue;
          // ⚠ A TRIM THAT GOES NOWHERE IS NOT A TRIM. A group whose whole
          // (u,v) run collapses to a point produces a zero-length pcurve, and
          // OpenNURBS rejects it by name — "ON_NurbsCurve is a line with no
          // length ... trim curve proxy settings are not valid". It happens
          // where the surface itself is degenerate (a pole) or where two
          // welded vertices land on the same parameter. Dropping it is
          // correct rather than lossy: it contributed no boundary, so the
          // trims either side of it still join each other.
          let uvSpan = 0;
          for (let i = 1; i < uv.length; i++) uvSpan = Math.max(uvSpan, Math.hypot(uv[i][0] - uv[0][0], uv[i][1] - uv[0][1]));
          if (uvSpan <= degenerateUV) {
            // ⚠⚠ DEGENERATE IN (u,v) IS NOT THE SAME AS DEGENERATE, and the
            // difference is measured rather than assumed: a run with real 3-D
            // LENGTH that collapses to a point in parameters is not nothing,
            // and dropping it silently removes a boundary the face has.
            //
            // The split is counted rather than collapsed into one number
            // because the two mean different things to a caller: a run with no
            // 3-D length was never a boundary, and one that has length but no
            // (u,v) extent is a parameterisation failure the face still needs.
            // On the banked torus pair both counts are zero for all three
            // operations at either tolerance, so a non-zero `collapsedInUV`
            // here is a signal, not a background rate.
            let span3d = 0;
            for (let i = 1; i < pts.length; i++) span3d = Math.max(span3d, Math.hypot(pts[i][0] - pts[0][0], pts[i][1] - pts[0][1], pts[i][2] - pts[0][2]));
            if (span3d <= degenerate3d) { degenerateTrims++; continue; }
            collapsedInUV++;
            continue;
          }
          const uvPts = uv.map((p) => [p[0], p[1], 0]);
          const res = fitCurveToPoints(uvPts, { tolerance, closed: false, exactEndpoints: true });
          if (!res.ok) { faceBad = res.reason; break; }
          if (res.maxDeviation > worst) worst = res.maxDeviation;
          /* ⚠⚠ A FIT IS JUDGED ONLY WHERE IT WAS SAMPLED, AND A TRIM'S WHOLE JOB
             IS WHAT HAPPENS BETWEEN THE SAMPLES. `maxDeviationFromCurve` asks of
             each POINT how far it is from the curve, so a smooth fit that passes
             exactly through every corner of a cap's outline and rings far outside
             it in between measures as perfect. Measured on the plainest capped
             extrude this app can make from a drawn profile: a cap whose surface
             domain is [0,1] carried a trim running to u = -0.3775 — 45mm outside
             its own rectangle on a 120mm shape. OpenNURBS then evaluates that
             plane EXTRAPOLATED, and the cap arrives in the file as a plate the
             size of the bounding box with the wall's rim drawn across its middle,
             which is a different solid from the one on screen.
             So the excursion is measured the other way round as well — every
             point ON the curve against the polyline it is meant to follow, which
             is the half a one-directional deviation cannot see. */
          let curve = res.curve, kind = res.kind;
          const excursion = curveExcursionFromPolyline(curve, uvPts);
          if (excursion > tolerance) {
            /* The polyline itself, degree 1 through every sample. Exact where the
               fit was only close, and inside the surface's own rectangle BY
               CONSTRUCTION — a degree-1 curve never leaves the convex hull of the
               points it runs through, and those came from projecting onto this
               surface. A heavier trim than a fitted one and the right one. */
            const poly = polylineCurveThrough(uvPts);
            if (poly) {
              curve = poly;
              kind = 'polyline';
              ringingTrims++;
              if (excursion > worstExcursion) worstExcursion = excursion;
            }
          }
          trims.push({ chainIndex: g.chainIndex, reversed, points: pts, uv, curve, kind, maxDeviation: res.maxDeviation });
          trimCount++;
        }
        if (faceBad) break;
      }
      if (faceBad) break;
      loopsOut.push({ loopType: loop === (face.loops || [])[0] ? 'outer' : 'inner', trims });
    }

    if (faceBad) { refused++; out.push({ faceId: face.id, srf, loops: null, reason: faceBad }); }
    else out.push({ faceId: face.id, srf, loops: loopsOut });
  }

  // THE PROPERTY OPENNURBS ACTUALLY CHECKS, measured here so a break shows up
  // as a number rather than as a validator complaint three layers away: does
  // each trim's (u,v) END where the next one STARTS?
  let worstJoin = 0, joinBreaks = 0;
  for (const f of out) {
    for (const loop of f.loops || []) {
      const ts = loop.trims;
      for (let i = 0; i < ts.length; i++) {
        const a = ts[i].uv[ts[i].uv.length - 1];
        const b = ts[(i + 1) % ts.length].uv[0];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (d > worstJoin) worstJoin = d;
        if (d > joinTolerance) joinBreaks++;
      }
    }
  }

  return {
    ok: refused === 0,
    faces: out,
    stats: {
      faces: faces.length,
      refusedFaces: refused,
      trims: trimCount,
      reversedTrims: reversedCount,
      mergedWrapGroups: mergedWraps,
      degenerateTrimsDropped: degenerateTrims,
      collapsedInUV,
      ringingTrims,      // a fitted pcurve that left the run it was fitted through, replaced by the run itself
      worstExcursion,    // how far the worst of them strayed, in (u,v)
      worstDeviation: worst,
      worstLoopJoin: worstJoin,
      loopJoinBreaks: joinBreaks,
    },
  };
}
