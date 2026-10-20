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
//   * Genus 0 closed (chi=2, b=0) — a sphere, a capsule, a closed box. One
//     OPEN cut: the shortest interior path between the two vertices farthest
//     apart. Its two ends keep one copy each and every vertex between is
//     split in two, so a path of k edges adds k-1 vertices and k edges and
//     chi falls from 2 to 1 with one boundary loop — a disk. (A CLOSED curve
//     would separate the surface into two disks; an open one never does.)
//
// Refused with a reason:
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

import { buildMeshTopology, weldTriangulation, flattenLSCM, dropDegenerateFaces, flatteningDistortion, arapRefine, balancedLayoutScale, principalStrains } from './flatten.mjs';
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
      throw new Error(`cutMeshAlongEdges: refusing to cut edge ${key} — it is shared by ${fs.length} face(s), and only an INTERIOR edge (exactly 2) can be opened`);
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
    throw new Error('handleCycle: no non-separating cycle exists — this closed mesh has genus 0 (a sphere), which has no handle to open');
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

// Per-vertex angle sum, one-third triangle area and one-ring, the inputs of
// a discrete Gaussian curvature estimate (defect = 2*pi - angle sum).
export function vertexAngleDefects(positions, faces) {
  const n = positions.length;
  const angleSum = new Float64Array(n), area = new Float64Array(n);
  const ring = new Map();
  const link = (a, b) => { if (!ring.has(a)) ring.set(a, new Set()); ring.get(a).add(b); };
  for (const tri of faces) {
    const P = [positions[tri[0]], positions[tri[1]], positions[tri[2]]];
    const e1 = [P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]];
    const e2 = [P[2][0] - P[0][0], P[2][1] - P[0][1], P[2][2] - P[0][2]];
    const third = 0.5 * Math.hypot(e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]) / 3;
    for (let i = 0; i < 3; i++) {
      const A = P[i], B = P[(i + 1) % 3], C = P[(i + 2) % 3];
      const u = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], v = [C[0] - A[0], C[1] - A[1], C[2] - A[2]];
      const cr = Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
      angleSum[tri[i]] += Math.atan2(cr, u[0] * v[0] + u[1] * v[1] + u[2] * v[2]);
      area[tri[i]] += third;
      link(tri[i], tri[(i + 1) % 3]); link(tri[i], tri[(i + 2) % 3]);
    }
  }
  return { angleSum, area, ring };
}

// Interior vertices that carry a cone point: an angle defect concentrated at
// the vertex rather than spread over the surface. A full revolve that reaches
// its axis (a cone to its apex, a pointed dome) welds into a disk whose apex
// holds the defect 2*pi*(1 - sin a) for a half-angle a; no layout of an
// interior vertex with that defect is isometric, so the apex is cut to the
// rim instead (`cutToDisk`). A smooth pole is told apart by curvature
// density: defect / vertex area estimates the Gaussian curvature K, which at
// a smooth pole matches its one-ring (within a factor of about 4 on a
// revolve grid), while at a cone point the defect stays finite as the
// vertex area shrinks, so K there exceeds every neighbor's by orders of
// magnitude. The comparison needs a regular grid: on a trimmed
// tessellation a clipped cell beside exactly flat grid cells reads as a
// cone, so a trimmed region takes its cone points from its untrimmed
// surface (`cutAndFlattenNurbsSurface`). Read on the uncut input, since a
// cut makes boundary vertices of the neighbors the comparison needs. A
// vertex with fewer than three interior neighbors, or with most of its ring
// on the boundary, is never a cone point. `coneMinDefect` (radians) ignores
// rounding noise on a developable mesh; `coneCurvatureRatio` is the K
// multiple over the one-ring.
export function interiorConePoints(positions, faces, topo, opts = {}) {
  const minDefect = opts.coneMinDefect != null ? opts.coneMinDefect : 1e-3;
  const ratio = opts.coneCurvatureRatio != null ? opts.coneCurvatureRatio : 25;
  const { angleSum, area, ring } = vertexAngleDefects(positions, faces);
  const interior = (v) => ring.has(v) && !topo.boundaryVertices.has(v);
  const K = (v) => (area[v] > 0 ? Math.abs(2 * Math.PI - angleSum[v]) / area[v] : 0);
  const out = [];
  for (const v of ring.keys()) {
    if (!interior(v)) continue;
    const defect = 2 * Math.PI - angleSum[v];
    if (!(Math.abs(defect) > minDefect)) continue;
    let nk = 0, inner = 0;
    for (const w of ring.get(v)) if (interior(w)) { inner++; nk = Math.max(nk, K(w)); }
    if (inner < 3 || 2 * inner < ring.get(v).size) continue;
    if (K(v) > ratio * nk) out.push({ vertex: v, defect });
  }
  out.sort((a, b) => Math.abs(b.defect) - Math.abs(a.defect));
  return out;
}

// `opts.conePoints` ([x, y, z] positions) replaces the detection: each is
// matched to the input vertex at that position (within 1e-9 of the mesh's
// extent), and a position no vertex sits on is outside the region.
function conePointsOf(positions, faces, topo, opts) {
  if (!opts.conePoints) return interiorConePoints(positions, faces, topo, opts);
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of positions) for (let k = 0; k < 3; k++) { if (p[k] < lo[k]) lo[k] = p[k]; if (p[k] > hi[k]) hi[k] = p[k]; }
  const tol = 1e-9 * Math.max(1e-12, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
  const out = [];
  for (const q of opts.conePoints) {
    let best = -1, bd = Infinity;
    for (const v of topo.usedVertices) { const d = dist3(positions[v], q); if (d < bd) { bd = d; best = v; } }
    if (best >= 0 && bd <= tol && !topo.boundaryVertices.has(best)) out.push({ vertex: best, defect: null });
  }
  return out;
}

// The two used vertices farthest apart in space, by a double sweep: the
// vertex farthest from the first used vertex, then the vertex farthest from
// that one. The second distance is at least the first, so the pair spans at
// least half the diameter, and on a sphere it is near-antipodal; the seam it
// gives runs the length of the form.
function farthestVertexPair(positions, topo) {
  const used = [...topo.usedVertices];
  const farthestFrom = (a) => {
    let best = a, bd = -1;
    for (const v of used) { const d = dist3(positions[a], positions[v]); if (d > bd) { bd = d; best = v; } }
    return best;
  };
  const a = farthestFrom(used[0]);
  return [a, farthestFrom(a)];
}

// The open seam on a closed genus-0 mesh: the shortest interior path from
// one end to the other. A path of one edge has no vertex between its ends, so
// cutting it opens nothing; it is then carried one edge further, to the far
// end's neighbor farthest from the start, so the seam splits at least one
// vertex.
function openSeamPath(positions, faces, topo, [a, b]) {
  const path = shortestInteriorPathBetween(positions, faces, topo, new Set([a]), new Set([b]));
  if (path.edgeKeys.length >= 2) return path;
  const end = path.vertices[path.vertices.length - 1], start = path.vertices[0];
  let next = -1, nd = -1;
  for (const [key, fs] of topo.edgeFaces) {
    if (fs.length !== 2) continue;
    const [x, y] = keyEnds(key);
    const w = x === end ? y : y === end ? x : -1;
    if (w < 0 || w === start) continue;
    const d = dist3(positions[start], positions[w]);
    if (d > nd) { nd = d; next = w; }
  }
  return {
    edgeKeys: [...path.edgeKeys, edgeKey(end, next)],
    vertices: [...path.vertices, next],
    length: path.length + dist3(positions[end], positions[next]),
  };
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

  const cones0 = conePointsOf(positions, faces, info0.topo, opts);
  if (info0.isDisk && !cones0.length) {
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
    throw new Error('cutToDisk: refusing to cut — the mesh is not consistently oriented. Either it is NON-ORIENTABLE (a Mobius strip or Klein bottle, which this stage does not support), or it is an orientable mesh whose triangles were handed over wound inconsistently; either way the handle/seam analysis below would be meaningless on it.');
  }
  if (info0.genus === null || info0.genus < 0) {
    throw new Error(`cutToDisk: refusing to cut — could not read a consistent genus from this mesh (V-E+F = ${info0.chi}, ${info0.boundaryCount} boundary component(s)).`);
  }
  if (info0.genus > 1) {
    throw new Error(`cutToDisk: refusing to cut — this surface has GENUS ${info0.genus} (${info0.genus} handles, V-E+F = ${info0.chi}). Only genus 0 (any number of boundaries) and genus 1 with no boundary (a torus) are supported. Cutting arbitrary genus needs a general cut-graph (Sheffer & Hart's Seamster, 2002), which this stage does not implement.`);
  }
  if (info0.genus === 1 && !info0.closed) {
    throw new Error(`cutToDisk: refusing to cut — this surface has a HANDLE AND a boundary (genus 1 with ${info0.boundaryCount} boundary component(s)). Only a closed torus is supported at genus 1; a handled surface that is also open needs a general cut-graph, which this stage does not implement.`);
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

  // Cut 1 (closed genus 0): one open seam between the farthest pair, which
  // leaves a disk with one boundary loop (see the header).
  if (info0.genus === 0 && info0.closed) {
    const ends = farthestVertexPair(pos, info0.topo);
    const path = openSeamPath(pos, fac, info0.topo, ends);
    applyCut(path.edgeKeys, 'open-seam', path.length);
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

  // Cone points: a slit from each to the boundary. The slit's end at the
  // cone point is not duplicated (one fan, see `cutMeshAlongEdges`), its
  // other end and every vertex between are, so chi and the boundary count
  // are unchanged and the cone point becomes a boundary vertex whose
  // defect the layout opens as a wedge. A cone point an earlier slit passed
  // through is already open and is skipped.
  for (const cone of cones0) {
    if (cuts.length >= maxCuts) break;
    const at = [];
    for (let i = 0; i < vertexOrigin.length; i++) if (vertexOrigin[i] === cone.vertex) at.push(i);
    const info = describeMeshTopology(pos, fac);
    if (at.length !== 1 || info.topo.boundaryVertices.has(at[0])) continue;
    const path = shortestInteriorPathBetween(pos, fac, info.topo, new Set(at), info.topo.boundaryVertices);
    applyCut(path.edgeKeys, 'cone-point', path.length);
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
// opts.trimLoop / opts.holes restrict it to a trimmed region (UV loops in
// the surface's own domain, as `tessellateTrimmedSurface` takes them). The
// trimmed tessellator shares every cell-boundary vertex between neighboring
// cells (its T-junction repair), so the region welds edge-manifold and is
// cut like any other mesh: a hole is one more boundary to connect.
//
// A trimmed region's clipped cells are triangulated across the surface's
// rulings, so its own mesh carries an angle defect of order h^2 even on a
// developable surface. Whether the SURFACE is developable, and where its
// cone points are, is therefore read from the untrimmed grid at the same
// resolution (cone points excluded from the defect), and the region's
// residual distortion is reported as the tessellation's.
export function cutAndFlattenNurbsSurface(srf, opts = {}) {
  const d = nurbsDiskMesh(srf, opts);
  const result = flattenLSCM({ positions: d.cut.positions, faces: d.cut.faces }, opts);
  return finishNurbsFlatten(result, d, opts);
}

// Tessellate, weld and cut to a disk: the half of cutAndFlattenNurbsSurface
// before any layout, shared with `cutAndFlattenGores`.
function nurbsDiskMesh(srf, opts) {
  if (!srf || !srf.ctrlNet) throw new Error('cutAndFlattenNurbsSurface: expected a NURBS surface with a control net');
  const uRes = opts.uRes != null ? opts.uRes : 24;
  const vRes = opts.vRes != null ? opts.vRes : 24;
  const trimmed = !!(opts.trimLoop || (opts.holes && opts.holes.length));
  let surfaceDefect = null, cutOpts = opts;
  if (trimmed) {
    const full = dropDegenerateFaces(weldTriangulation(tessellateTrimmedSurface(srf, null, uRes, vRes), opts.weldTolerance));
    const fullTopo = describeMeshTopology(full.positions, full.faces).topo;
    const cones = interiorConePoints(full.positions, full.faces, fullTopo, opts);
    const coneSet = new Set(cones.map((c) => c.vertex));
    const { angleSum, ring } = vertexAngleDefects(full.positions, full.faces);
    surfaceDefect = 0;
    for (const v of ring.keys()) {
      if (fullTopo.boundaryVertices.has(v) || coneSet.has(v)) continue;
      surfaceDefect = Math.max(surfaceDefect, Math.abs(2 * Math.PI - angleSum[v]));
    }
    cutOpts = { ...opts, conePoints: cones.map((c) => full.positions[c.vertex]) };
  }
  const tris = tessellateTrimmedSurface(srf, opts.trimLoop || null, uRes, vRes, opts.holes || []);
  if (!tris.length) throw new Error('cutAndFlattenNurbsSurface: the surface tessellated to zero triangles — nothing to flatten');
  // dropDegenerateFaces: a pole (a revolve profile touching the axis) welds
  // the whole pole row to one point, so the tessellator's two-triangles-per-
  // cell emission leaves a repeated-vertex face there — zero area, no
  // topology, and refused downstream if kept. See kernel/flatten.mjs.
  const welded = weldTriangulation(tris, opts.weldTolerance);
  const mesh = dropDegenerateFaces(welded);
  const cut = cutToDisk(mesh.positions, mesh.faces, cutOpts);
  return { uRes, vRes, tris, welded, mesh, cut, surfaceDefect };
}

// The report fields cutAndFlattenNurbsSurface adds to a layout of the disk.
function finishNurbsFlatten(result, d, opts) {
  const { uRes, vRes, tris, mesh, cut, surfaceDefect } = d;
  if (surfaceDefect != null) {
    const developableTol = opts.developableTolerance != null ? opts.developableTolerance : 1e-9;
    result.surfaceDefect = surfaceDefect;
    if (!result.developable && surfaceDefect <= developableTol) {
      result.developable = true;
      result.note = `The surface is DEVELOPABLE (its untrimmed grid has no interior angle defect); the trimmed region's own triangulation carries a defect up to ${result.distortion.intrinsic.maxAbsInteriorAngleDefect.toExponential(3)} rad, which is the residual distortion below.`;
    }
  }
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

// Gores

// Dijkstra over interior edges by 3D length from every vertex of `sources`.
// A boundary vertex that is not a source is reached but never passed
// through, so a path ends at the first boundary vertex it meets and never
// runs along or across an earlier cut.
function goreDistances(positions, topo, sources) {
  const adj = new Map();
  for (const [key, fs] of topo.edgeFaces) {
    if (fs.length !== 2) continue;
    const [a, b] = keyEnds(key);
    const w = dist3(positions[a], positions[b]);
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a).push([b, w]);
    adj.get(b).push([a, w]);
  }
  const dist = new Map(), prev = new Map(), done = new Set();
  const heap = new MinHeap();
  for (const s of sources) { dist.set(s, 0); heap.push(0, s); }
  while (heap.size) {
    const [d, v] = heap.pop();
    if (done.has(v)) continue;
    done.add(v);
    if (topo.boundaryVertices.has(v) && !sources.has(v)) continue;
    for (const [w, len] of (adj.get(v) || [])) {
      const nd = d + len;
      if (!dist.has(w) || nd < dist.get(w) - 1e-15) { dist.set(w, nd); prev.set(w, v); heap.push(nd, w); }
    }
  }
  return { dist, prev };
}

// The boundary of a disk as one ordered loop of vertices, or null when the
// boundary edges do not close into a single loop through every boundary
// vertex (a pinched boundary).
function goreBoundaryLoop(topo) {
  const nbr = new Map();
  for (const [a, b] of topo.boundaryEdges) {
    if (!nbr.has(a)) nbr.set(a, []);
    if (!nbr.has(b)) nbr.set(b, []);
    nbr.get(a).push(b);
    nbr.get(b).push(a);
  }
  if (!nbr.size) return null;
  for (const ns of nbr.values()) if (ns.length !== 2) return null;
  const start = nbr.keys().next().value;
  const loop = [start];
  let prev = -1, cur = start;
  for (;;) {
    const ns = nbr.get(cur);
    const next = ns[0] === prev ? ns[1] : ns[0];
    if (next === start) break;
    if (loop.length > nbr.size) return null;
    loop.push(next);
    prev = cur;
    cur = next;
  }
  return loop.length === nbr.size ? loop : null;
}

// The gore cut tree of a disk: `count` shortest interior paths from one
// vertex to `count` boundary vertices spaced evenly by 3D arc length around
// the boundary. All paths come from one shortest-path tree, so two paths
// share a prefix or part for good and never cross. The paths meet at the
// interior vertex farthest from the boundary (on a cap, its pole); the
// spacing starts at the boundary vertex nearest that point.
//
// Returns { center, paths: [{ vertices, edgeKeys }] } in the disk's own
// vertex indices; a boundary vertex no interior path reaches is skipped.
export function goreCutPaths(positions, faces, count) {
  const info = describeMeshTopology(positions, faces);
  if (!info.isDisk) throw new Error('goreCutPaths: the mesh is not a topological disk');
  const topo = info.topo;
  const loop = goreBoundaryLoop(topo);
  if (!loop) throw new Error('goreCutPaths: the boundary does not close into one loop, so gores cannot be spaced around it');
  const fromRim = goreDistances(positions, topo, new Set(topo.boundaryVertices)).dist;
  let center = -1, far = -1;
  for (const v of topo.usedVertices) {
    if (topo.boundaryVertices.has(v)) continue;
    const d = fromRim.get(v);
    if (d != null && d > far) { far = d; center = v; }
  }
  if (center < 0) return { center: null, paths: [] };
  const { dist, prev } = goreDistances(positions, topo, new Set([center]));
  const arc = [0];
  for (let i = 1; i <= loop.length; i++) arc.push(arc[i - 1] + dist3(positions[loop[i - 1]], positions[loop[i % loop.length]]));
  const total = arc[loop.length];
  let i0 = 0;
  for (let i = 1; i < loop.length; i++) if ((dist.get(loop[i]) ?? Infinity) < (dist.get(loop[i0]) ?? Infinity)) i0 = i;
  const picked = new Set();
  const paths = [];
  for (let k = 0; k < count; k++) {
    const want = (arc[i0] + k * total / count) % total;
    let best = -1, bd = Infinity;
    for (let i = 0; i < loop.length; i++) {
      const d = Math.min(Math.abs(arc[i] - want), total - Math.abs(arc[i] - want));
      if (d < bd) { bd = d; best = i; }
    }
    const t = loop[best];
    if (picked.has(t) || !prev.has(t)) continue;
    picked.add(t);
    const vertices = [t], edgeKeys = [];
    let cur = t;
    while (prev.has(cur)) { const p = prev.get(cur); edgeKeys.push(edgeKey(p, cur)); vertices.push(p); cur = p; }
    vertices.reverse();
    paths.push({ vertices, edgeKeys });
  }
  return { center, paths };
}

// Faces of a mesh grouped into edge-connected pieces, in face order.
function gorePieces(faces) {
  const parent = faces.map((_, i) => i);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const byEdge = new Map();
  for (let f = 0; f < faces.length; f++) {
    const [a, b, c] = faces[f];
    for (const [x, y] of [[a, b], [b, c], [c, a]]) {
      const k = edgeKey(x, y);
      if (byEdge.has(k)) { const r = find(byEdge.get(k)), s = find(f); if (r !== s) parent[r] = s; } else byEdge.set(k, f);
    }
  }
  const groups = new Map();
  for (let f = 0; f < faces.length; f++) {
    const r = find(f);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(f);
  }
  return [...groups.values()];
}

// Lay pieces out in one row without overlap. Each piece turns so its
// principal axis (the vertex covariance's major eigenvector) is vertical,
// then sits `gap` to the right of the one before, its bottom on v = 0. A
// rotation and a translation change no strain.
function gorePackPieces(uvs, gap) {
  const placed = uvs.map((uv) => {
    let mx = 0, my = 0;
    for (const p of uv) { mx += p[0]; my += p[1]; }
    mx /= uv.length; my /= uv.length;
    let sxx = 0, syy = 0, sxy = 0;
    for (const p of uv) { const x = p[0] - mx, y = p[1] - my; sxx += x * x; syy += y * y; sxy += x * y; }
    const t = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const ca = Math.cos(Math.PI / 2 - t), sa = Math.sin(Math.PI / 2 - t);
    const r = uv.map(([u, v]) => [(u - mx) * ca - (v - my) * sa, (u - mx) * sa + (v - my) * ca]);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of r) { if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0]; if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1]; }
    return { r, x0, x1, y0, y1 };
  });
  let tallest = 0;
  for (const p of placed) tallest = Math.max(tallest, p.y1 - p.y0);
  const g = gap != null ? gap : 0.02 * tallest;
  let x = 0;
  return placed.map((p) => {
    const dx = x - p.x0, dy = -p.y0;
    x += p.x1 - p.x0 + g;
    return p.r.map(([u, v]) => [u + dx, v + dy]);
  });
}

// Each gore path as strokes in the surface's own (u, v), read from the
// tessellation's triangles (welding keeps one position per vertex but a seam
// or pole vertex has several parameter values). Each edge takes the triangle
// whose (u, v) at the shared vertex continues the stroke; where none does (the
// path crosses a parameter seam) a new stroke starts.
function goreUVStrokes(paths, vertexOrigin, welded, tris, srf) {
  const edgeUV = new Map();
  welded.faces.forEach((f, t) => {
    for (let k = 0; k < 3; k++) {
      const a = f[k], b = f[(k + 1) % 3];
      if (a === b) continue;
      const key = edgeKey(a, b);
      if (!edgeUV.has(key)) edgeUV.set(key, []);
      edgeUV.get(key).push([a, tris[t][k].uv, b, tris[t][(k + 1) % 3].uv]);
    }
  });
  const span = Math.max(srf.knotsU[srf.knotsU.length - 1] - srf.knotsU[0], srf.knotsV[srf.knotsV.length - 1] - srf.knotsV[0]);
  return paths.map((p) => {
    const strokes = [];
    let cur = null, prevUV = null;
    for (let i = 0; i + 1 < p.vertices.length; i++) {
      const wa = vertexOrigin[p.vertices[i]], wb = vertexOrigin[p.vertices[i + 1]];
      let best = null, bd = Infinity;
      for (const e of edgeUV.get(edgeKey(wa, wb)) || []) {
        const ua = e[0] === wa ? e[1] : e[3], ub = e[0] === wa ? e[3] : e[1];
        const d = prevUV ? Math.hypot(ua[0] - prevUV[0], ua[1] - prevUV[1]) : 0;
        if (d < bd) { bd = d; best = [ua, ub]; }
      }
      if (!best) { cur = null; prevUV = null; continue; }
      if (!cur || bd > 1e-9 * span) { cur = [[best[0][0], best[0][1]]]; strokes.push(cur); }
      cur.push([best[1][0], best[1][1]]);
      prevUV = best[1];
    }
    return strokes.filter((st) => st.length >= 2);
  });
}

// Cut a surface into `opts.cuts` gores and flatten each on its own.
//
// The disk from nurbsDiskMesh is cut along goreCutPaths. With two or more
// paths every path ends at the meeting point, so the cut splits the disk
// into one piece per path (two paths that share a prefix still bound a
// piece). Each piece is laid out on its own: LSCM, then arapRefine from it
// (`opts.arapIterations` rounds), then the uniform scale that balances its
// largest stretch against its largest compression (balancedLayoutScale),
// kept when its worst principal strain is below LSCM's; the pieces are laid
// out in a row (gorePackPieces). A conformal layout of gores that stay joined
// cannot fan open at the joint (a slit tip keeps its angle), so the pieces are
// separated rather than slit. A gore of longitude width w (radians) on a
// sphere carries a worst strain of about w^2 / 16 whatever the solver: 0.44%
// at 24 gores, 0.25% at 32.
//
// opts.cuts 0 is cutAndFlattenNurbsSurface. A developable surface needs no
// gore and gets that layout unchanged, at the grid asked for, with gores.made
// 0; `tessellation.uRes / vRes` are the grid the gores were cut on. Otherwise the
// result has the shape of cutAndFlattenNurbsSurface's over the gored mesh:
// seam.vertexOrigin points each vertex back at the welded mesh, `pieces`
// is [{ start, count }] ranges of `faces`, and `gores` is { requested, made,
// center (a point), paths (point lists on the surface), uvPaths (per path, its
// strokes in the surface's (u, v), goreUVStrokes) }.
export function cutAndFlattenGores(srf, opts = {}) {
  const requested = Math.max(0, Math.floor(opts.cuts || 0));
  if (!requested) return cutAndFlattenNurbsSurface(srf, opts);
  // Gores are spaced on boundary vertices, so the grid is raised to a
  // multiple of the count: on a revolve the rim then divides evenly and every
  // gore has the same width (a gore one grid step wider carries the larger
  // strain of the wider gore).
  const even = (n) => (requested >= 2 ? Math.ceil(n / requested) * requested : n);
  const gridOpts = { ...opts, uRes: even(opts.uRes != null ? opts.uRes : 24), vRes: even(opts.vRes != null ? opts.vRes : 24) };
  const d = nurbsDiskMesh(srf, gridOpts);
  const { cut } = d;
  const developableTol = opts.developableTolerance != null ? opts.developableTolerance : 1e-9;
  const topo0 = describeMeshTopology(cut.positions, cut.faces).topo;
  const { angleSum, ring } = vertexAngleDefects(cut.positions, cut.faces);
  let defect = 0;
  for (const v of ring.keys()) if (!topo0.boundaryVertices.has(v)) defect = Math.max(defect, Math.abs(2 * Math.PI - angleSum[v]));
  const layoutOpts = { ...opts, pinA: null, pinB: null };
  const unchanged = () => {
    const r = cutAndFlattenNurbsSurface(srf, opts);
    r.pieces = [{ start: 0, count: r.faces.length }];
    r.gores = { requested, made: 0, center: null, paths: [], uvPaths: [] };
    return r;
  };
  if (defect <= developableTol || (d.surfaceDefect != null && d.surfaceDefect <= developableTol)) return unchanged();
  const tree = goreCutPaths(cut.positions, cut.faces, requested);
  if (!tree.paths.length) return unchanged();
  const keys = new Set();
  for (const p of tree.paths) for (const k of p.edgeKeys) keys.add(k);
  const g = cutMeshAlongEdges(cut.positions, cut.faces, keys);
  const groups = gorePieces(g.faces);
  const solved = groups.map((fs) => {
    const local = new Map(), pos = [];
    const fac = fs.map((f) => g.faces[f].map((v) => {
      if (!local.has(v)) { local.set(v, pos.length); pos.push(g.positions[v]); }
      return local.get(v);
    }));
    const lay = flattenLSCM({ positions: pos, faces: fac }, layoutOpts);
    const refined = arapRefine(pos, fac, lay.uv, { iterations: opts.arapIterations });
    const k = balancedLayoutScale(pos, fac, refined.uv);
    const uv = refined.uv.map(([u, v]) => [u * k, v * k]);
    // ARAP lowers the summed energy, not the worst triangle: on a piece far
    // from developable (one slit in a hemisphere) it can raise the worst
    // strain, and then the LSCM layout is kept.
    const worstOf = (q) => { const m = principalStrains(pos, fac, q).summary; return m.max === null ? Infinity : Math.max(m.maxStretch, m.maxCompression); };
    return { fs, local, lay, uv: worstOf(uv) < worstOf(lay.uv) ? uv : lay.uv };
  });
  const packed = gorePackPieces(solved.map((s) => s.uv), opts.pieceGap);
  const uv = g.positions.map(() => [0, 0]);
  const faces = [], pieces = [];
  let iterations = 0, residual = 0, unknowns = 0, equations = 0;
  solved.forEach((s, i) => {
    for (const [v, j] of s.local) uv[v] = packed[i][j];
    pieces.push({ start: faces.length, count: s.fs.length });
    for (const f of s.fs) faces.push(g.faces[f]);
    iterations += s.lay.solver.iterations;
    residual = Math.max(residual, s.lay.solver.residual);
    unknowns += s.lay.solver.unknowns;
    equations += s.lay.solver.equations;
  });
  const result = {
    uv, positions: g.positions, faces,
    pins: null,
    solver: { iterations, residual, converged: true, unknowns, equations },
    scaleApplied: null,
    distortion: flatteningDistortion(g.positions, faces, uv),
    developable: false,
    note: `Cut into ${pieces.length} gore${pieces.length === 1 ? '' : 's'} along ${tree.paths.length} path${tree.paths.length === 1 ? '' : 's'} from one point, each flattened on its own. The surface is doubly curved (max interior angle defect ${defect.toExponential(3)} rad), so every gore carries some distortion; the numbers below are measured over all of them.`,
  };
  const out = finishNurbsFlatten(result, {
    ...d,
    cut: { ...cut, positions: g.positions, vertexOrigin: g.vertexOrigin.map((v) => cut.vertexOrigin[v]) },
  }, opts);
  out.pieces = pieces;
  out.gores = {
    requested, made: tree.paths.length,
    center: cut.positions[tree.center].slice(),
    paths: tree.paths.map((p) => p.vertices.map((v) => cut.positions[v].slice())),
    uvPaths: goreUVStrokes(tree.paths, cut.vertexOrigin, d.welded, d.tris, srf),
  };
  return out;
}
