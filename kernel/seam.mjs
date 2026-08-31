// SURFACE CUTTING / SEAM STAGE — turn a non-disk triangulation into a disk
// so that kernel/flatten.mjs's LSCM can accept it.
//
// ================================================================
// WHY THIS FILE EXISTS, AND WHY IT IS NOT INSIDE flatten.mjs
// ================================================================
// flatten.mjs is a PARAMETRIZATION module: every function in it is about
// mapping a disk into the plane and measuring how much that map lied. Its
// honest-refusal gate `validateFlattenMesh` correctly rejects anything that
// is not a topological disk, via an exact Euler-characteristic test — and
// the finding that motivates it is that this app's Revolve routinely produces
// exactly those refusals: a full 360-degree revolve of an open profile
// welds shut at its seam into a topological ANNULUS, and a full revolve of
// a CLOSED profile into a TORUS. So the flattener was correct and nearly
// unusable at the same time.
//
// What closes that gap is a purely TOPOLOGICAL operation with no geometry
// in it at all: duplicate the vertices along a chosen seam so the two sides
// of the seam become distinct boundary vertices, and re-index the faces.
// Not one coordinate moves. That is a different kind of work from solving a
// sparse conformal system, it has its own citations and its own failure
// modes, and it is useful on its own (any consumer that needs a disk, not
// just LSCM). It gets its own module. The dependency runs ONE WAY —
// seam.mjs imports flatten.mjs's already-proven `buildMeshTopology` and
// `weldTriangulation` rather than re-deriving them; flatten.mjs is
// untouched by this file.
//
// ================================================================
// WHAT IS SUPPORTED, STATED PLAINLY, SCOPED TO WHAT THIS APP MAKES
// ================================================================
// For a connected, edge-manifold triangulation with b boundary components
// and (if orientable) genus g, the Euler characteristic is
//   chi = V - E + F = 2 - 2g - b.
// A disk is g=0, b=1, chi=1. This module supports exactly:
//
//   * ALREADY A DISK (chi=1, b=1) — returned unchanged, zero cuts. Not an
//     error; an untrimmed partial revolve is already flattenable.
//
//   * GENUS 0 WITH b >= 2 BOUNDARIES — the common case, and the priority.
//     A full-revolve cylinder or cone frustum is exactly this (b=2, an
//     annulus). Each cut is a shortest interior-edge path between two
//     DIFFERENT boundary components; b-1 cuts reduce it to a disk. On a
//     revolve's own grid tessellation that shortest path IS a parametric
//     ruling — the natural, correct seam — and it falls out of the geometry
//     rather than needing the surface's parameter domain to be threaded
//     through the mesh (see `shortestInteriorPathBetween`).
//
//   * GENUS 1 CLOSED (chi=0, b=0) — a torus. Two cuts: first a
//     non-separating cycle (which opens the handle, leaving an annulus),
//     then the annulus cut above. Real, reachable here (revolve a closed
//     profile through 360 degrees), and materially harder than the case
//     above — see the citations below for exactly which piece is borrowed
//     and which guarantee is deliberately NOT claimed.
//
// REFUSED BY NAME, not attempted:
//   * A CLOSED GENUS-0 surface (a sphere, chi=2). This is not a scope cut
//     for convenience — no single cut turns a sphere into ONE disk. Cutting
//     a closed genus-0 surface along any closed curve separates it into TWO
//     disks; getting one disk requires deleting area, which is a splitting
//     operation, not a seam.
//   * GENUS >= 2, and any genus >= 1 that still has boundary. Both need a
//     general cut-graph, which this module does not build.
//   * Non-manifold, disconnected, or inconsistently-oriented input.
//
// ================================================================
// CITATIONS — WHAT IS ACTUALLY USED, AND WHAT IS NOT
// ================================================================
// USED, for the genus-1 handle cut: the TREE-COTREE decomposition —
// Eppstein, "Dynamic generators of topologically embedded graphs," SODA
// 2003. Build a spanning tree T of the vertex graph and a spanning tree C
// of the DUAL graph using only edges outside T; the edges in neither are
// exactly 2g in number, and each one closes a non-separating cycle with the
// tree path between its endpoints (each is a nonzero class in H_1, and a
// cycle separates iff it is null-homologous).
//
// NOT CLAIMED: Erickson & Whittlesey, "Greedy optimal homotopy and homology
// generators," SODA 2005, computes the genuinely SHORTEST non-trivial cycle
// on a surface. This module does not. It takes the shortest of the 2g
// tree-cotree generators, which is a real cut but a weaker guarantee — the
// resulting seam is valid and exact, just not provably the shortest one
// available. Stated rather than implied, because on a torus the difference
// is visible: a poor generator gives a longer seam than necessary.
//
// NOT USED AT ALL: Sheffer & Hart, "Seamster: Inconspicuous Low-Distortion
// Texture Seam Layout," IEEE Visualization 2002, generates a cut graph by
// seeding at high-Gaussian-curvature vertices and growing a minimum
// spanning structure, specifically to hide seams where they will be least
// visible. That is the right reference for a general arbitrary-genus cut,
// and it is exactly the machinery this module deliberately does not build.
// Named here so a future reader knows where to start, not to decorate this
// one.
//
// ================================================================
// WHAT "GEOMETRY-PRESERVING" MEANS HERE, EXACTLY
// ================================================================
// A cut duplicates vertex indices. Every duplicate is a fresh copy of the
// SAME three numbers (`positions[v].slice()`), and every face keeps its own
// three corner positions in its own original order — only the integers
// naming them change. So the triangle-area sum over the cut mesh is
// BIT-FOR-BIT identical to the sum over the input, not merely close, and
// test/seam.test.mjs asserts exact equality rather than a tolerance. A cut
// that moves geometry is a bug, and that is the check that would catch it.

import { buildMeshTopology, weldTriangulation, flattenLSCM, dropDegenerateFaces } from './flatten.mjs';
import { tessellateTrimmedSurface } from './trimtess.mjs';

// ---------------------------------------------------------------- helpers

function edgeKey(a, b) { return a < b ? `${a}|${b}` : `${b}|${a}`; }
function keyEnds(key) { const i = key.indexOf('|'); return [Number(key.slice(0, i)), Number(key.slice(i + 1))]; }
function dist3(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }

// A minimal binary min-heap. Written here rather than pulled in because
// this kernel has no dependencies; Dijkstra below is the only consumer.
class MinHeap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(key, val) {
    const a = this.a;
    a.push([key, val]);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
}

// ------------------------------------------------------- mesh description

// The BOUNDARY COMPONENTS of a mesh: each is the set of vertices belonging
// to one connected piece of the boundary. Deliberately computed as
// CONNECTED COMPONENTS of the boundary-edge graph rather than by walking
// ordered loops — a walk has to make an arbitrary choice at a vertex with
// more than two incident boundary edges (an edge-manifold mesh can still be
// pinched at a vertex), and nothing downstream here needs the cyclic order,
// only "which boundary is this vertex on."
export function boundaryComponents(topo) {
  const parent = new Map();
  const find = (x) => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  for (const v of topo.boundaryVertices) parent.set(v, v);
  for (const [a, b] of topo.boundaryEdges) {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  }
  const groups = new Map();
  for (const v of topo.boundaryVertices) {
    const r = find(v);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(v);
  }
  return [...groups.values()];
}

// Is every interior edge traversed in OPPOSITE directions by its own two
// faces? For a closed surface this is what distinguishes an orientable
// torus from a Klein bottle — and, honestly, it also fails for a mesh that
// IS orientable but was handed to us wound inconsistently. The refusal
// message below names both readings rather than asserting the first.
function isConsistentlyOriented(faces, edgeFaces) {
  const seen = new Set();
  for (const [a, b, c] of faces) {
    for (const [x, y] of [[a, b], [b, c], [c, a]]) {
      const dir = `${x}>${y}`;
      if (seen.has(dir)) return false; // the same half-edge used twice
      seen.add(dir);
    }
  }
  for (const [key, fs] of edgeFaces) {
    if (fs.length !== 2) continue;
    const [a, b] = keyEnds(key);
    if (!(seen.has(`${a}>${b}`) && seen.has(`${b}>${a}`))) return false;
  }
  return true;
}

// The full topological picture a cut decision needs. Deliberately does NOT
// check triangle AREA — a zero-area triangle is a real problem, but it is
// flatten.mjs's own problem (its `validateFlattenMesh` already refuses it by
// name), and cutting neither cares about it nor makes it worse. Keeping the
// two gates separate means neither has to be relaxed for the other.
export function describeMeshTopology(positions, faces) {
  if (!Array.isArray(positions) || !Array.isArray(faces)) {
    throw new Error('seam: expected {positions, faces} arrays');
  }
  if (faces.length < 1) throw new Error('seam: refusing — the mesh has no triangles at all');
  for (let f = 0; f < faces.length; f++) {
    const tri = faces[f];
    if (!Array.isArray(tri) || tri.length !== 3) throw new Error(`seam: face ${f} is not a triangle`);
    for (const i of tri) {
      if (!Number.isInteger(i) || i < 0 || i >= positions.length) {
        throw new Error(`seam: face ${f} references vertex index ${i}, which does not exist`);
      }
    }
    if (tri[0] === tri[1] || tri[1] === tri[2] || tri[2] === tri[0]) {
      throw new Error(`seam: face ${f} uses the same vertex twice ([${tri}])`);
    }
  }
  const topo = buildMeshTopology(positions, faces);
  if (topo.nonManifoldEdges.length) {
    const [a, b, n] = topo.nonManifoldEdges[0];
    throw new Error(`seam: refusing — the mesh is NON-MANIFOLD (edge ${a}-${b} is shared by ${n} faces; a surface edge can touch at most 2)`);
  }
  if (topo.componentCount > 1) {
    throw new Error(`seam: refusing — the mesh is DISCONNECTED (${topo.componentCount} separate pieces). Cut one connected piece at a time.`);
  }
  const bComps = boundaryComponents(topo);
  const chi = topo.usedVertices.size - topo.edges.length + topo.faceCount;
  const oriented = isConsistentlyOriented(faces, topo.edgeFaces);
  // chi = 2 - 2g - b, so g = (2 - b - chi) / 2. Only meaningful for an
  // orientable surface; reported as null otherwise rather than guessed.
  const genusTwice = 2 - bComps.length - chi;
  const genus = oriented && genusTwice % 2 === 0 ? genusTwice / 2 : null;
  return {
    topo, chi, boundaryCount: bComps.length, boundaryComponents: bComps,
    closed: bComps.length === 0, orientable: oriented, genus,
    isDisk: chi === 1 && bComps.length === 1,
  };
}

// -------------------------------------------------- the cut itself

// CUT the mesh open along a set of interior edges. This is the whole
// operation the rest of the file exists to feed: everything else only ever
// DECIDES which edges, this is what actually opens them.
//
// The rule, stated generally so it is correct for an open path, a closed
// cycle, several cuts at once, and a vertex where two cuts meet, without
// special-casing any of them: for each vertex touched by a cut edge,
// partition its incident faces into groups that are still connected to each
// other through its remaining (UNCUT) incident edges. Each group beyond the
// first gets its own copy of the vertex. A vertex whose faces remain in one
// group is left alone — which is exactly right for a cut that dead-ends at
// an interior vertex (a slit that does not actually open the surface there).
//
// Every cut edge must be INTERIOR (shared by exactly 2 faces). Cutting an
// edge that is already on the boundary is a no-op that would silently
// return an unchanged mesh, so it is refused by name instead.
export function cutMeshAlongEdges(positions, faces, cutEdgeKeys) {
  const cut = cutEdgeKeys instanceof Set ? cutEdgeKeys : new Set(cutEdgeKeys);
  if (!cut.size) throw new Error('cutMeshAlongEdges: no edges given to cut');

  const edgeFaces = new Map();
  for (let f = 0; f < faces.length; f++) {
    const [a, b, c] = faces[f];
    for (const [x, y] of [[a, b], [b, c], [c, a]]) {
      const key = edgeKey(x, y);
      if (!edgeFaces.has(key)) edgeFaces.set(key, []);
      edgeFaces.get(key).push(f);
    }
  }
  for (const key of cut) {
    const fs = edgeFaces.get(key);
    if (!fs) throw new Error(`cutMeshAlongEdges: edge ${key} is not an edge of this mesh`);
    if (fs.length !== 2) {
      throw new Error(`cutMeshAlongEdges: refusing to cut edge ${key} — it is shared by ${fs.length} face(s), and only a genuinely INTERIOR edge (exactly 2) can be opened`);
    }
  }

  // Vertices touched by a cut, and each vertex's incident faces.
  const touched = new Set();
  for (const key of cut) { const [a, b] = keyEnds(key); touched.add(a); touched.add(b); }
  const facesAt = new Map();
  for (const v of touched) facesAt.set(v, []);
  for (let f = 0; f < faces.length; f++) {
    for (const v of faces[f]) if (facesAt.has(v)) facesAt.get(v).push(f);
  }

  const outPositions = positions.map((p) => [p[0], p[1], p[2]]);
  const outFaces = faces.map((t) => [t[0], t[1], t[2]]);
  const vertexOrigin = positions.map((_, i) => i);
  const duplicatedGroups = [];

  for (const v of touched) {
    const inc = facesAt.get(v);
    // Union-find over this vertex's incident faces, joined by any incident
    // edge at v that was NOT cut.
    const idxOf = new Map(inc.map((f, i) => [f, i]));
    const parent = inc.map((_, i) => i);
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (const f of inc) {
      // Read adjacency from the ORIGINAL faces, never the partly-rewritten
      // `outFaces` — an earlier vertex's split may already have renamed a
      // corner of this same face, and `edgeFaces` is keyed by the input's
      // own indices. Splitting one vertex must never change how the NEXT
      // vertex's own neighborhood reads.
      const tri = faces[f];
      for (let k = 0; k < 3; k++) {
        const a = tri[k], b = tri[(k + 1) % 3];
        if (a !== v && b !== v) continue;            // not an edge at v
        const key = edgeKey(a, b);
        if (cut.has(key)) continue;                   // the cut blocks travel
        const fs = edgeFaces.get(key);
        if (fs.length !== 2) continue;                // a boundary edge joins nothing
        const other = fs[0] === f ? fs[1] : fs[0];
        if (!idxOf.has(other)) continue;
        const ra = find(idxOf.get(f)), rb = find(idxOf.get(other));
        if (ra !== rb) parent[ra] = rb;
      }
    }
    const groups = new Map();
    for (const f of inc) {
      const r = find(idxOf.get(f));
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(f);
    }
    if (groups.size <= 1) continue;                   // the cut did not open this vertex

    const groupList = [...groups.values()];
    const copies = [v];
    for (let g = 1; g < groupList.length; g++) {
      const nv = outPositions.length;
      // The duplicate is the SAME three numbers, not a recomputed point.
      outPositions.push([positions[v][0], positions[v][1], positions[v][2]]);
      vertexOrigin.push(v);
      copies.push(nv);
      for (const f of groupList[g]) {
        const tri = outFaces[f];
        for (let k = 0; k < 3; k++) if (tri[k] === v) tri[k] = nv;
      }
    }
    duplicatedGroups.push(copies);
  }

  return { positions: outPositions, faces: outFaces, vertexOrigin, duplicatedGroups };
}

// ------------------------------------------------ choosing where to cut

// SHORTEST INTERIOR-EDGE PATH from one vertex set to another, weighted by
// true 3D edge length (Dijkstra, multi-source). Only INTERIOR edges are
// traversable, for two reasons that both matter: a boundary edge cannot be
// cut at all (`cutMeshAlongEdges` refuses it), and a path allowed to run
// ALONG a boundary would produce a seam that opens nothing.
//
// On a revolve's own grid tessellation this returns a parametric ruling —
// the natural seam — without any knowledge of the surface's parameter
// domain, because a ruling genuinely IS the shortest way across. On a
// full-revolve cylinder every ruling is the same length, so which one comes
// back is arbitrary and, by the surface's own rotational symmetry, produces
// an identical unrolled result either way.
export function shortestInteriorPathBetween(positions, faces, topo, sources, targets) {
  const src = sources instanceof Set ? sources : new Set(sources);
  const dst = targets instanceof Set ? targets : new Set(targets);
  const adj = new Map();
  for (const [key, fs] of topo.edgeFaces) {
    if (fs.length !== 2) continue;
    const [a, b] = keyEnds(key);
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    const w = dist3(positions[a], positions[b]);
    adj.get(a).push([b, w]);
    adj.get(b).push([a, w]);
  }
  const dist = new Map(), prev = new Map();
  const heap = new MinHeap();
  for (const s of src) { dist.set(s, 0); heap.push(0, s); }
  const done = new Set();
  let hit = -1;
  while (heap.size) {
    const [d, v] = heap.pop();
    if (done.has(v)) continue;
    done.add(v);
    if (dst.has(v) && !src.has(v)) { hit = v; break; }
    for (const [w, len] of (adj.get(v) || [])) {
      const nd = d + len;
      if (!dist.has(w) || nd < dist.get(w) - 1e-15) { dist.set(w, nd); prev.set(w, v); heap.push(nd, w); }
    }
  }
  if (hit < 0) {
    throw new Error('cutToDisk: refusing to cut — no path of interior edges connects the two boundaries, so there is no seam to open');
  }
  const keys = [];
  const verts = [hit];
  let cur = hit;
  while (prev.has(cur)) {
    const p = prev.get(cur);
    keys.push(edgeKey(p, cur));
    verts.push(p);
    cur = p;
  }
  verts.reverse();
  return { edgeKeys: keys, vertices: verts, length: dist.get(hit) };
}

// A NON-SEPARATING CYCLE on a CLOSED mesh, via the tree-cotree
// decomposition (Eppstein 2003 — see the header for exactly what is and is
// not claimed here). Returns the shortest of the 2g generators.
export function handleCycle(positions, faces, topo) {
  // Exported, so it has to defend its own precondition rather than trust
  // that `cutToDisk` is the only caller: the tree-cotree argument counts
  // 2g leftover edges only on a CLOSED surface.
  if (topo.boundaryEdges.length) {
    throw new Error(`handleCycle: requires a CLOSED mesh — this one has ${topo.boundaryEdges.length} boundary edge(s), so the tree-cotree edge count would not be 2g`);
  }
  // Primal spanning tree, BFS from vertex 0 of face 0.
  const adj = new Map();
  for (const key of topo.edgeFaces.keys()) {
    const [a, b] = keyEnds(key);
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a).push([b, key]);
    adj.get(b).push([a, key]);
  }
  const root = faces[0][0];
  const parent = new Map([[root, -1]]);
  const parentEdge = new Map();
  const depth = new Map([[root, 0]]);
  const treeEdges = new Set();
  const queue = [root];
  for (let qi = 0; qi < queue.length; qi++) {
    const v = queue[qi];
    for (const [w, key] of (adj.get(v) || [])) {
      if (parent.has(w)) continue;
      parent.set(w, v); parentEdge.set(w, key); depth.set(w, depth.get(v) + 1);
      treeEdges.add(key);
      queue.push(w);
    }
  }

  // Dual spanning tree over FACES, using only edges outside the primal tree.
  const dualAdj = new Map();
  for (let f = 0; f < faces.length; f++) dualAdj.set(f, []);
  for (const [key, fs] of topo.edgeFaces) {
    if (fs.length !== 2 || treeEdges.has(key)) continue;
    dualAdj.get(fs[0]).push([fs[1], key]);
    dualAdj.get(fs[1]).push([fs[0], key]);
  }
  const seenFace = new Set([0]);
  const cotreeEdges = new Set();
  const fq = [0];
  for (let qi = 0; qi < fq.length; qi++) {
    for (const [g, key] of dualAdj.get(fq[qi])) {
      if (seenFace.has(g)) continue;
      seenFace.add(g); cotreeEdges.add(key); fq.push(g);
    }
  }

  // Whatever is in neither tree closes a non-separating cycle.
  const generators = [];
  for (const [key, fs] of topo.edgeFaces) {
    if (fs.length !== 2) continue;
    if (treeEdges.has(key) || cotreeEdges.has(key)) continue;
    generators.push(key);
  }
  if (!generators.length) {
    throw new Error('handleCycle: no non-separating cycle exists — this closed mesh has genus 0 (a sphere), which cannot be cut into a single disk');
  }

  const cycleFor = (key) => {
    const [a, b] = keyEnds(key);
    // Walk both endpoints up to their common ancestor in the primal tree.
    let x = a, y = b;
    const upX = [], upY = [];
    while (depth.get(x) > depth.get(y)) { upX.push(parentEdge.get(x)); x = parent.get(x); }
    while (depth.get(y) > depth.get(x)) { upY.push(parentEdge.get(y)); y = parent.get(y); }
    while (x !== y) {
      upX.push(parentEdge.get(x)); x = parent.get(x);
      upY.push(parentEdge.get(y)); y = parent.get(y);
    }
    const keys = [key, ...upX, ...upY];
    let len = 0;
    for (const k of keys) { const [p, q] = keyEnds(k); len += dist3(positions[p], positions[q]); }
    return { edgeKeys: keys, length: len };
  };

  let best = null;
  for (const g of generators) {
    const c = cycleFor(g);
    if (!best || c.length < best.length) best = c;
  }
  return best;
}

// ------------------------------------------------------------- the driver

// CUT a triangulation open until it is a topological disk, then hand it
// back ready for `flattenLSCM`. Refuses by name for everything outside the
// scope stated in this file's header, rather than returning a plausible
// mesh that is not actually a disk.
//
// Returns { positions, faces, vertexOrigin, duplicatedGroups, cuts,
//           chiBefore, chiAfter, note } — `vertexOrigin[i]` is the index in
// the ORIGINAL mesh that vertex i came from (identity for anything the cut
// did not touch), which is what lets a caller relate a flattened layout
// back to the surface it came from, and lets a test check that every twin
// sits at exactly the position of the vertex it was split from.
export function cutToDisk(positions, faces, opts = {}) {
  const maxCuts = opts.maxCuts != null ? opts.maxCuts : 32;
  const info0 = describeMeshTopology(positions, faces);
  const chiBefore = info0.chi;

  if (info0.isDisk) {
    return {
      positions: positions.map((p) => [p[0], p[1], p[2]]),
      faces: faces.map((t) => [t[0], t[1], t[2]]),
      vertexOrigin: positions.map((_, i) => i),
      duplicatedGroups: [], cuts: [], chiBefore, chiAfter: info0.chi,
      note: 'Input was already a topological disk (V-E+F = 1, one boundary). No cut was needed and none was made.',
    };
  }

  // Refuse everything out of scope BEFORE touching the mesh, by name.
  if (!info0.orientable) {
    throw new Error('cutToDisk: refusing to cut — the mesh is not consistently oriented. Either it is genuinely NON-ORIENTABLE (a Mobius strip or Klein bottle, which this stage does not support), or it is an orientable mesh whose triangles were handed over wound inconsistently; either way the handle/seam analysis below would be meaningless on it.');
  }
  if (info0.closed && info0.chi === 2) {
    throw new Error('cutToDisk: refusing to cut — this is a CLOSED GENUS-0 surface (a sphere, V-E+F = 2). No single cut turns a sphere into one disk: cutting a closed genus-0 surface along any curve separates it into TWO disks. It has to be SPLIT into pieces, which is a different operation from opening a seam.');
  }
  if (info0.genus === null || info0.genus < 0) {
    throw new Error(`cutToDisk: refusing to cut — could not read a consistent genus from this mesh (V-E+F = ${info0.chi}, ${info0.boundaryCount} boundary component(s)).`);
  }
  if (info0.genus > 1) {
    throw new Error(`cutToDisk: refusing to cut — this surface has GENUS ${info0.genus} (${info0.genus} handles, V-E+F = ${info0.chi}). Only genus 0 (any number of boundaries) and genus 1 with no boundary (a torus) are supported. Cutting arbitrary genus needs a general cut-graph (Sheffer & Hart's Seamster, 2002), which this stage deliberately does not implement.`);
  }
  if (info0.genus === 1 && !info0.closed) {
    throw new Error(`cutToDisk: refusing to cut — this surface has a HANDLE AND a boundary (genus 1 with ${info0.boundaryCount} boundary component(s)). Only a closed torus is supported at genus 1; a handled surface that is also open needs a general cut-graph, which this stage deliberately does not implement.`);
  }

  let pos = positions, fac = faces;
  let vertexOrigin = positions.map((_, i) => i);
  const duplicatedGroups = [];
  const cuts = [];

  const applyCut = (edgeKeys, kind, length) => {
    const r = cutMeshAlongEdges(pos, fac, edgeKeys);
    pos = r.positions;
    fac = r.faces;
    // Compose the origin map so it always points back at the ORIGINAL mesh.
    vertexOrigin = r.vertexOrigin.map((i) => vertexOrigin[i]);
    for (const g of r.duplicatedGroups) duplicatedGroups.push(g);
    cuts.push({
      kind, edgeCount: edgeKeys.length, length,
      verticesDuplicated: r.duplicatedGroups.reduce((n, g) => n + g.length - 1, 0),
    });
  };

  // CUT 1 (genus 1 only): open the handle with a non-separating cycle. A
  // torus becomes an annulus — chi is unchanged (a closed cycle of n
  // vertices duplicates n vertices AND splits n edges), but the surface now
  // has two boundaries for the boundary cuts below to work with.
  if (info0.genus === 1) {
    const info = describeMeshTopology(pos, fac);
    const cyc = handleCycle(pos, fac, info.topo);
    applyCut(cyc.edgeKeys, 'handle-cycle', cyc.length);
  }

  // CUT 2..N: connect boundary components until only one remains. Each cut
  // is a shortest interior path between two DIFFERENT boundaries, which
  // raises chi by exactly 1 per cut on the ordinary case (k edges split,
  // k+1 vertices duplicated).
  for (let iter = 0; iter < maxCuts; iter++) {
    const info = describeMeshTopology(pos, fac);
    if (info.boundaryCount <= 1) break;
    const comps = info.boundaryComponents;
    const path = shortestInteriorPathBetween(pos, fac, info.topo, new Set(comps[0]), new Set(comps.slice(1).flat()));
    applyCut(path.edgeKeys, 'boundary-path', path.length);
  }

  // The gate. Nothing above is trusted: the result is re-derived from
  // scratch and must be a genuine disk, or this throws rather than handing
  // a caller a mesh LSCM would fold over.
  const after = describeMeshTopology(pos, fac);
  if (!after.isDisk) {
    throw new Error(`cutToDisk: the cut did not produce a topological disk (V-E+F = ${after.chi}, ${after.boundaryCount} boundary component(s) after ${cuts.length} cut(s)) — refusing to return it.`);
  }

  return {
    positions: pos, faces: fac, vertexOrigin, duplicatedGroups, cuts,
    chiBefore, chiAfter: after.chi,
    note: `Cut open with ${cuts.length} seam(s) (${cuts.map((c) => c.kind).join(', ')}); Euler characteristic V-E+F went ${chiBefore} -> ${after.chi}. No vertex moved: every duplicate is an exact copy of the vertex it was split from.`,
  };
}

// ---------------------------------------------------------- NURBS entry

// The end-to-end payoff, and the reason this stage exists: tessellate a
// NURBS surface with this kernel's OWN already-proven tessellator, weld it,
// CUT it open if it is not already a disk, and flatten it with the
// already-proven LSCM. A full 360-degree Revolve now flattens.
//
// Deliberately takes no trim loop. Measured while building this: the
// trimmed output of `tessellateTrimmedSurface` does NOT weld into an
// edge-manifold mesh in general (independently clipped neighboring cells
// leave T-junctions along their shared edge, and the ear-clipper's own
// residual warnings fire), so a trimmed patch is refused upstream by
// `describeMeshTopology` as non-manifold rather than silently mis-cut. That
// is a real, separate gap in the tessellator, named here rather than worked
// around with a weld tolerance that would paper over it.
export function cutAndFlattenNurbsSurface(srf, opts = {}) {
  if (!srf || !srf.ctrlNet) throw new Error('cutAndFlattenNurbsSurface: expected a NURBS surface with a control net');
  const uRes = opts.uRes != null ? opts.uRes : 24;
  const vRes = opts.vRes != null ? opts.vRes : 24;
  const tris = tessellateTrimmedSurface(srf, null, uRes, vRes);
  if (!tris.length) throw new Error('cutAndFlattenNurbsSurface: the surface tessellated to zero triangles — nothing to flatten');
  // dropDegenerateFaces: a POLE (a revolve profile touching the axis) welds
  // the whole pole row to one point, so the tessellator's own unconditional
  // two-triangles-per-cell emission leaves a repeated-vertex face there —
  // zero area, no topology, and refused by name downstream if kept. See
  // that function's own comment in kernel/flatten.mjs.
  const mesh = dropDegenerateFaces(weldTriangulation(tris, opts.weldTolerance));
  const cut = cutToDisk(mesh.positions, mesh.faces, opts);
  const result = flattenLSCM({ positions: cut.positions, faces: cut.faces }, opts);
  result.seam = {
    cuts: cut.cuts, chiBefore: cut.chiBefore, chiAfter: cut.chiAfter,
    vertexOrigin: cut.vertexOrigin, duplicatedGroups: cut.duplicatedGroups,
    note: cut.note,
  };
  result.tessellation = {
    uRes, vRes, triangleCount: tris.length,
    weldedVertexCount: mesh.positions.length, cutVertexCount: cut.positions.length, droppedPoleFaces: mesh.droppedFaces,
    weldTolerance: mesh.weldTolerance,
  };
  return result;
}
