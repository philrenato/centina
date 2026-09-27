// Surface cutting — turn a non-disk triangulation into a disk so that
// kernel/flatten.mjs's LSCM can accept it.
//
// flatten.mjs's `validateFlattenMesh` rejects anything that is not a
// topological disk, via an exact Euler-characteristic test. A full 360-degree
// revolve of an open profile welds shut at its seam into a topological
// annulus, and a full revolve of a closed profile into a torus; both are
// refused there.
//
// The cut is a purely topological operation: duplicate the vertices along a
// chosen seam so the two sides of the seam become distinct boundary vertices,
// and re-index the faces. No coordinate moves. It is useful to any consumer
// that needs a disk, not just LSCM. The dependency runs one way: seam.mjs
// imports flatten.mjs's `buildMeshTopology` and `weldTriangulation`;
// flatten.mjs does not import this file.
//
// Scope. For a connected, edge-manifold triangulation with b boundary
// components and (if orientable) genus g, the Euler characteristic is
//   chi = V - E + F = 2 - 2g - b.
// A disk is g=0, b=1, chi=1. This module supports exactly:
//
//   * Already a disk (chi=1, b=1) — returned unchanged, zero cuts. Not an
//     error; an untrimmed partial revolve is already flattenable.
//
//   * Genus 0 with b >= 2 boundaries. A full-revolve cylinder or cone
//     frustum is exactly this (b=2, an annulus). Each cut is a shortest
//     interior-edge path between two different boundary components; b-1 cuts
//     reduce it to a disk. On a revolve's own grid tessellation that shortest
//     path is a parametric ruling, found from the geometry without threading
//     the surface's parameter domain through the mesh (see
//     `shortestInteriorPathBetween`).
//
//   * Genus 1 closed (chi=0, b=0) — a torus. Two cuts: first a
//     non-separating cycle (which opens the handle, leaving an annulus),
//     then the annulus cut above. Reachable by revolving a closed profile
//     through 360 degrees. See the citations below for which piece is
//     borrowed and which guarantee is not claimed.
//
// Refused with a reason:
//   * A closed genus-0 surface (a sphere, chi=2). No single cut turns a
//     sphere into one disk: cutting a closed genus-0 surface along any closed
//     curve separates it into two disks; getting one disk requires deleting
//     area, which is a splitting operation, not a seam.
//   * Genus >= 2, and any genus >= 1 that still has boundary. Both need a
//     general cut-graph, which this module does not build.
//   * Non-manifold, disconnected, or inconsistently-oriented input.
//
// Citations. Used, for the genus-1 handle cut: the tree-cotree decomposition —
// Eppstein, "Dynamic generators of topologically embedded graphs," SODA
// 2003. Build a spanning tree T of the vertex graph and a spanning tree C
// of the dual graph using only edges outside T; the edges in neither are
// exactly 2g in number, and each one closes a non-separating cycle with the
// tree path between its endpoints (each is a nonzero class in H_1, and a
// cycle separates iff it is null-homologous).
//
// Not claimed: Erickson & Whittlesey, "Greedy optimal homotopy and homology
// generators," SODA 2005, computes the shortest non-trivial cycle on a
// surface. This module does not. It takes the shortest of the 2g
// tree-cotree generators: a valid, exact cut, but not provably the shortest
// available. On a torus a poor generator gives a longer seam than necessary.
//
// Not used: Sheffer & Hart, "Seamster: Inconspicuous Low-Distortion Texture
// Seam Layout," IEEE Visualization 2002, generates a cut graph by seeding at
// high-Gaussian-curvature vertices and growing a minimum spanning structure,
// to place seams where they are least visible. It is the reference for a
// general arbitrary-genus cut, which this module does not build.
//
// Geometry preservation. A cut duplicates vertex indices. Every duplicate is
// a fresh copy of the same three numbers (`positions[v].slice()`), and every
// face keeps its own three corner positions in its own original order — only
// the integers naming them change. So the triangle-area sum over the cut mesh
// is bit-for-bit identical to the sum over the input, and test/seam.test.mjs
// asserts exact equality rather than a tolerance.

import { buildMeshTopology, weldTriangulation, flattenLSCM, dropDegenerateFaces } from './flatten.mjs';
import { tessellateTrimmedSurface } from './trimtess.mjs';

// Helpers

function edgeKey(a, b) { return a < b ? `${a}|${b}` : `${b}|${a}`; }
function keyEnds(key) { const i = key.indexOf('|'); return [Number(key.slice(0, i)), Number(key.slice(i + 1))]; }
function dist3(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }

// A minimal binary min-heap (the kernel has no dependencies); Dijkstra below
// is the only consumer.
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

// Mesh description

// The boundary components of a mesh: each is the set of vertices belonging
// to one connected piece of the boundary. Computed as connected components
// of the boundary-edge graph rather than by walking
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

// Is every interior edge traversed in opposite directions by its own two
// faces? For a closed surface this distinguishes an orientable torus from a
// Klein bottle; it also fails for an orientable mesh wound inconsistently.
// The refusal message in cutToDisk names both readings.
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

// The full topological picture a cut decision needs. Does not check triangle
// area: a zero-area triangle is refused by flatten.mjs's
// `validateFlattenMesh`, and cutting neither depends on it nor makes it
// worse.
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

// The cut

// Cut the mesh open along a set of interior edges. The rest of the file
// decides which edges; this opens them.
//
// The rule, stated generally so it is correct for an open path, a closed
// cycle, several cuts at once, and a vertex where two cuts meet, without
// special-casing any of them: for each vertex touched by a cut edge,
// partition its incident faces into groups that are still connected to each
// other through its remaining (uncut) incident edges. Each group beyond the
// first gets its own copy of the vertex. A vertex whose faces remain in one
// group is left alone, which is correct for a cut that dead-ends at an
// interior vertex (a slit that does not open the surface there).
//
// Every cut edge must be interior (shared by exactly 2 faces). Cutting an
// edge that is already on the boundary would return an unchanged mesh, so it
// is refused.
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
    // edge at v that was not cut.
    const idxOf = new Map(inc.map((f, i) => [f, i]));
    const parent = inc.map((_, i) => i);
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (const f of inc) {
      // Read adjacency from the original faces, never the partly-rewritten
      // `outFaces`: an earlier vertex's split may already have renamed a
      // corner of this same face, and `edgeFaces` is keyed by the input's
      // own indices. Splitting one vertex must not change how the next
      // vertex's neighborhood reads.
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
      // The duplicate is the same three numbers, not a recomputed point.
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

// Choosing where to cut

// Shortest interior-edge path from one vertex set to another, weighted by
// true 3D edge length (Dijkstra, multi-source). Only interior edges are
// traversable: a boundary edge cannot be cut (`cutMeshAlongEdges` refuses
// it), and a path running along a boundary would produce a seam that opens
// nothing.
//
// On a revolve's own grid tessellation this returns a parametric ruling —
// the natural seam — without any knowledge of the surface's parameter
// domain, because a ruling is the shortest way across. On a
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

// A non-separating cycle on a closed mesh, via the tree-cotree
// decomposition (Eppstein 2003 — see the header for what is and is not
// claimed). Returns the shortest of the 2g generators.
export function handleCycle(positions, faces, topo) {
  // The tree-cotree argument counts 2g leftover edges only on a closed
  // surface.
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

  // Dual spanning tree over faces, using only edges outside the primal tree.
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

// The driver

// Cut a triangulation open until it is a topological disk, ready for
// `flattenLSCM`. Throws for everything outside the scope stated in this
// file's header rather than returning a mesh that is not a disk.
//
// Returns { positions, faces, vertexOrigin, duplicatedGroups, cuts,
//           chiBefore, chiAfter, note } — `vertexOrigin[i]` is the index in
// the original mesh that vertex i came from (identity for anything the cut
// did not touch), which relates a flattened layout back to the surface it
// came from.
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

  // Refuse everything out of scope before touching the mesh.
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
    // Compose the origin map so it always points back at the original mesh.
    vertexOrigin = r.vertexOrigin.map((i) => vertexOrigin[i]);
    for (const g of r.duplicatedGroups) duplicatedGroups.push(g);
    cuts.push({
      kind, edgeCount: edgeKeys.length, length,
      verticesDuplicated: r.duplicatedGroups.reduce((n, g) => n + g.length - 1, 0),
    });
  };

  // Cut 1 (genus 1 only): open the handle with a non-separating cycle. A
  // torus becomes an annulus — chi is unchanged (a closed cycle of n
  // vertices duplicates n vertices and splits n edges), but the surface then
  // has two boundaries for the boundary cuts below.
  if (info0.genus === 1) {
    const info = describeMeshTopology(pos, fac);
    const cyc = handleCycle(pos, fac, info.topo);
    applyCut(cyc.edgeKeys, 'handle-cycle', cyc.length);
  }

  // Cuts 2..N: connect boundary components until only one remains. Each cut
  // is a shortest interior path between two different boundaries, which
  // raises chi by exactly 1 per cut on the ordinary case (k edges split,
  // k+1 vertices duplicated).
  for (let iter = 0; iter < maxCuts; iter++) {
    const info = describeMeshTopology(pos, fac);
    if (info.boundaryCount <= 1) break;
    const comps = info.boundaryComponents;
    const path = shortestInteriorPathBetween(pos, fac, info.topo, new Set(comps[0]), new Set(comps.slice(1).flat()));
    applyCut(path.edgeKeys, 'boundary-path', path.length);
  }

  // The result is re-derived from scratch and must be a disk, or this throws
  // rather than handing a caller a mesh LSCM would fold over.
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

// NURBS entry

// Tessellate a NURBS surface with this kernel's tessellator, weld it, cut it
// open if it is not already a disk, and flatten it with LSCM.
//
// Takes no trim loop. Known limitation: the trimmed output of
// `tessellateTrimmedSurface` does not in general weld into an edge-manifold
// mesh (independently clipped neighboring cells leave T-junctions along
// their shared edge), so a trimmed patch would be refused by
// `describeMeshTopology` as non-manifold. A weld tolerance would hide the
// T-junctions rather than remove them.
export function cutAndFlattenNurbsSurface(srf, opts = {}) {
  if (!srf || !srf.ctrlNet) throw new Error('cutAndFlattenNurbsSurface: expected a NURBS surface with a control net');
  const uRes = opts.uRes != null ? opts.uRes : 24;
  const vRes = opts.vRes != null ? opts.vRes : 24;
  const tris = tessellateTrimmedSurface(srf, null, uRes, vRes);
  if (!tris.length) throw new Error('cutAndFlattenNurbsSurface: the surface tessellated to zero triangles — nothing to flatten');
  // dropDegenerateFaces: a pole (a revolve profile touching the axis) welds
  // the whole pole row to one point, so the tessellator's two-triangles-per-
  // cell emission leaves a repeated-vertex face there — zero area, no
  // topology, and refused downstream if kept. See kernel/flatten.mjs.
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
