// Catmull-Clark subdivision-surface math: the kernel under the SuperB
// object type (Rhino calls the analogous feature "SubD"). Kernel only: this
// module has no notion of an app-layer object, a display/render
// tessellation policy, or UI — it is pure cage-in, finer-cage-out math.
//
// A cage (the control net — not the geometry; the limit surface this
// defines is the object, the same relationship a NURBS control polygon has
// to its curve) is plain data:
//   { vertices: [[x,y,z], ...],
//     faces:    [[i0,i1,...], ...],   // >=3 indices per face, quads preferred, ngons legal
//     creases:  { "i_j": weight, ... } }  // sparse — an absent key means weight 0 (smooth)
// A crease weight is a soft-crease value (Rhino 8's SubDCrease scale, 0-100)
// stored on an edge, keyed by `edgeKey(i,j)` (order-independent). Boundary
// edges (used by exactly one face) are always fully sharp regardless of any
// stored weight — there is no second adjacent face to build a smooth
// alternative from.
//
// The refinement step (subdivideCatmullClark) implements the Catmull-Clark
// rules (Catmull & Clark 1978; the face/edge/vertex point formulas as
// commonly restated, e.g. the Wikipedia "Catmull–Clark subdivision surface"
// summary, cross-checked against the DeRose 1998 boundary rule and a worked
// cube example — see the function comments below and test/subd.test.mjs
// for the arithmetic):
//   - Face point: the centroid of a face's own vertices (any n-gon).
//   - Edge point: (v0 + v1 + f0 + f1) / 4 for a smooth interior edge (f0/f1
//     its two adjacent face points); the plain midpoint (v0+v1)/2 for a
//     boundary edge or a fully-sharp crease.
//   - Vertex point: (F + 2R + (n-3)P) / n for a smooth interior vertex of
//     valence n (F = average of adjacent face points, R = average of
//     adjacent edge midpoints — the original edges, not the edge points
//     above); the boundary/crease curve rule (A + 6P + B) / 8 for a vertex
//     on exactly one crease/boundary line (A, B its two crease-neighbor
//     vertices); the vertex's own unmoved position for a corner (3+ sharp
//     edges meeting there, per the standard Catmull-Clark corner
//     treatment). A dart (1 sharp edge) takes the smooth rule; see
//     computeVertexPoint.
//
// Semi-sharp creases (DeRose, Kass & Truong 1998, the mechanism behind
// Rhino 8's SoftCrease): a crease weight above 0 linearly blends the smooth
// and fully-sharp edge/vertex rules above (blend fraction = the weight
// clamped to [0,1]), and decrements by exactly 1.0 every time
// subdivideCatmullClark is called — the returned cage's creases map
// already carries the decremented weight on each edge's two children, so a
// caller calls this function repeatedly with no separate "current level"
// bookkeeping. A weight that has decayed to 0 is dropped from the map (an
// absent key is weight 0), so the edge is an ordinary smooth edge after.
//
// Scope: no Stam eigen-basis limit-surface evaluation (this module only
// produces a finer cage, never the exact limit position — display is a
// fixed 2-3 refinement levels); no non-manifold-edge repair — an edge used
// by 3+ faces is forced fully sharp, like a boundary edge, since there is
// no well-defined "average of N face points" smooth alternative for it.
// That is a stated fallback, not a claim of correctness for non-manifold
// input. No ToNURBS, no object type, no UI. The display-mesh functions at
// the bottom (chooseSuperBRefinementLevel/triangulateFace/superbDisplayMesh)
// are kernel-only as well: no THREE.js here.

import { add, scale } from './vec3.mjs';

function midpoint(a, b) { return scale(add(a, b), 0.5); }
function lerp(a, b, t) { return add(scale(a, 1 - t), scale(b, t)); }
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

// Marked-corner weight floor — see computeVertexPoint's marked-corner
// comment for why a boundary edge's stored weight is reused as a corner
// marker. The threshold must not fire on an ordinary stored weight: the
// app's SUPERB_CREASE_LEVEL_SCALE (the highest weight Crease/SoftCrease can
// store, a "full harden" edge) is 3, so hardening both boundary edges at a
// SuperBPlane corner would otherwise pin that corner at P instead of
// smoothing it along the boundary curve. The floor sits above the closed
// [0, 3] those commands can reach; only a caller that stores a weight this
// high on purpose (TOSUBD's DEFAULT_CORNER_CREASE_WEIGHT,
// kernel/subdconvert.mjs) triggers the marker. A kernel-local constant, not
// an import of the app's SUPERB_CREASE_LEVEL_SCALE: kernel modules import no
// app-layer constants.
export const MARKED_CORNER_WEIGHT_FLOOR = 100;

function validateCage(cage) {
  if (!cage || !Array.isArray(cage.vertices) || cage.vertices.length === 0) {
    throw new Error('subdivideCatmullClark: cage.vertices must be a non-empty array of [x,y,z] points');
  }
  if (!Array.isArray(cage.faces) || cage.faces.length === 0) {
    throw new Error('subdivideCatmullClark: cage.faces must be a non-empty array of vertex-index loops');
  }
  cage.faces.forEach((face, fi) => {
    if (!Array.isArray(face) || face.length < 3) {
      throw new Error(`subdivideCatmullClark: face ${fi} needs at least 3 vertex indices (got ${JSON.stringify(face)})`);
    }
    for (const vi of face) {
      if (!Number.isInteger(vi) || vi < 0 || vi >= cage.vertices.length) {
        throw new Error(`subdivideCatmullClark: face ${fi} references out-of-range vertex index ${vi}`);
      }
    }
  });
}

// Order-independent key for the (undirected) edge between vertex indices a
// and b — every edge/crease lookup in this module goes through this one
// function, so a differently-ordered pair always resolves to the same slot.
export function edgeKey(a, b) { return a < b ? `${a}_${b}` : `${b}_${a}`; }

// The raw, unclamped stored crease weight for an edge (0 if absent — the
// sparse-map default). The current step's effective blend fraction
// (boundary-forcing and the [0,1] clamp applied) is edgeSharpness() below.
export function creaseWeight(cage, v0, v1) {
  const w = cage.creases && cage.creases[edgeKey(v0, v1)];
  return typeof w === 'number' ? w : 0;
}

// A boundary edge (used by exactly one face) is always fully sharp,
// regardless of any stored crease weight — there is no second adjacent
// face to build a smooth alternative from. A non-manifold edge (used by
// 3+ faces) is also forced fully sharp: the smooth rule's "average of
// exactly 2 adjacent face points" has no well-defined generalization to
// 3+ (see the header). An interior edge (used by exactly 2 faces) is the
// only case that reads its stored crease weight — its effective sharpness
// this step is that weight clamped to [0,1] (the semi-sharp blend
// fraction).
function edgeSharpness(cage, edge) {
  if (edge.faces.length !== 2) return 1;
  return clamp01(creaseWeight(cage, edge.v0, edge.v1));
}

// Centroid of a face's own vertices — works for any n-gon (n>=3), quad or
// otherwise; the face-point rule is identical regardless of face size.
export function computeFacePoint(cage, faceIdx) {
  const face = cage.faces[faceIdx];
  let c = [0, 0, 0];
  for (const vi of face) c = add(c, cage.vertices[vi]);
  return scale(c, 1 / face.length);
}

// Builds the adjacency structures a single subdivision pass needs, derived
// from the cage's faces every time (never carried as persistent state):
// boundary-ness is always structural, never hand-tracked, so topology
// re-derived from a freshly subdivided cage has the correct
// boundary/interior classification.
export function buildTopology(cage) {
  const edgeMap = new Map(); // edgeKey -> { v0, v1, faces: [faceIdx, ...] }
  const vertexFacesRaw = cage.vertices.map(() => []);
  const faceCentroids = cage.faces.map((_, fi) => computeFacePoint(cage, fi));

  cage.faces.forEach((face, faceIdx) => {
    const n = face.length;
    for (let c = 0; c < n; c++) {
      const vi = face[c];
      vertexFacesRaw[vi].push(faceIdx);
      const vNext = face[(c + 1) % n];
      const key = edgeKey(vi, vNext);
      let e = edgeMap.get(key);
      if (!e) { e = { v0: vi, v1: vNext, faces: [] }; edgeMap.set(key, e); }
      e.faces.push(faceIdx);
    }
  });

  const vertexFaces = vertexFacesRaw.map((list) => [...new Set(list)]);
  const vertexEdges = cage.vertices.map(() => []);
  for (const e of edgeMap.values()) {
    vertexEdges[e.v0].push(e);
    vertexEdges[e.v1].push(e);
  }

  return { edgeMap, vertexFaces, vertexEdges, faceCentroids };
}

// Edge point — smooth interior rule (v0+v1+f0+f1)/4, generalized to
// however many faces touch the edge (2 for a manifold interior edge; the
// boundary/fully-sharp case returns the midpoint first), blended toward
// the plain midpoint by the edge's current sharpness.
export function computeEdgePoint(cage, ctx, edge) {
  const v0 = cage.vertices[edge.v0];
  const v1 = cage.vertices[edge.v1];
  const mid = midpoint(v0, v1);
  const sharpness = edgeSharpness(cage, edge);
  if (sharpness >= 1) return mid; // boundary, or fully (semi-sharp-decayed-to-sharp) creased
  let sum = add(v0, v1);
  for (const fi of edge.faces) sum = add(sum, ctx.faceCentroids[fi]);
  const smooth = scale(sum, 1 / (2 + edge.faces.length));
  if (sharpness <= 0) return smooth;
  return lerp(smooth, mid, sharpness);
}

// The ordinary smooth Catmull-Clark vertex rule, (F + 2R + (n-3)P) / n —
// F the average of adjacent face points, R the average of adjacent edge
// midpoints (the original edges — not the edge points above). Never
// called for a vertex whose blend has resolved to fully sharp
// (computeVertexPoint short-circuits before this), so the valence-1/2
// degenerate cases below are defensive.
function smoothVertexRule(cage, ctx, vIdx) {
  const P = cage.vertices[vIdx];
  const faces = ctx.vertexFaces[vIdx];
  const edges = ctx.vertexEdges[vIdx];
  const n = edges.length;
  if (n === 0) return P.slice();
  let F = [0, 0, 0];
  for (const fi of faces) F = add(F, ctx.faceCentroids[fi]);
  F = scale(F, 1 / (faces.length || 1));
  let R = [0, 0, 0];
  for (const e of edges) {
    const other = e.v0 === vIdx ? e.v1 : e.v0;
    R = add(R, midpoint(P, cage.vertices[other]));
  }
  R = scale(R, 1 / n);
  const num = add(add(F, scale(R, 2)), scale(P, n - 3));
  return scale(num, 1 / n);
}

// Vertex point — classifies the vertex by how many of its incident edges
// are currently sharp (boundary edges always count; interior creased
// edges count whenever their clamped weight is > 0), per the
// DeRose/Kass/Truong semi-sharp-crease vertex taxonomy:
//   0 sharp edges -> the ordinary smooth interior rule.
//   exactly 1     -> a dart (the open end of a crease line, dead-ending
//                    inside the surface rather than closing a loop or
//                    reaching a boundary/corner) — the smooth rule,
//                    regardless of that edge's sharpness. This is what
//                    makes a semi-sharp crease fade out at its open end
//                    instead of leaving a pinch: there is no crease line
//                    at a dart (that needs exactly 2 sharp edges) for a
//                    curve rule to run along, and blending toward the
//                    vertex's unmoved position would leave a dimple.
//   exactly 2     -> a crease/boundary line running through this vertex —
//                    the standard (A + 6P + B) / 8 curve-subdivision rule,
//                    A/B the vertex's two crease-neighbor points, blended
//                    toward the smooth rule by the (averaged) sharpness of
//                    those two edges.
//   3+            -> a corner (3+ creases meeting) — the vertex stays at
//                    its own position P, blended toward the smooth rule
//                    by the sharpest incident edge.
export function computeVertexPoint(cage, ctx, vIdx) {
  const P = cage.vertices[vIdx];
  const edges = ctx.vertexEdges[vIdx];
  if (edges.length === 0) return P.slice();

  const classified = edges.map((e) => ({
    edge: e,
    other: e.v0 === vIdx ? e.v1 : e.v0,
    sharpness: edgeSharpness(cage, e),
  }));
  const sharp = classified.filter((c) => c.sharpness > 0);
  if (sharp.length === 0 || sharp.length === 1) return smoothVertexRule(cage, ctx, vIdx);

  let creasePoint, blend;
  if (sharp.length === 2) {
    const [a, b] = sharp;
    const A = cage.vertices[a.other];
    const B = cage.vertices[b.other];
    // Marked corner (TOSUBD's Corners=Yes option). A vertex with exactly
    // two incident edges, both boundary edges (a single-face grid corner —
    // e.g. one of a NURBS surface's 4 untrimmed corners), can never reach
    // the 3+ corner branch below: sharp.length is structurally 2, so it
    // takes the boundary-curve rule, which smooths through the vertex and
    // rounds off a corner that may be sharp in the source geometry (the
    // standard Catmull-Clark open-corner behavior, the same reason an
    // uncreased SuperBBox converges toward a sphere).
    //
    // A boundary edge's stored crease weight is otherwise inert
    // (edgeSharpness forces a boundary edge's sharpness to 1), so it is
    // reused as a marker: both boundary edges carrying a stored weight
    // above MARKED_CORNER_WEIGHT_FLOOR means the corner is held at P (the
    // treatment the 3+ branch gives an interior corner) rather than
    // smoothed along the boundary curve. No primitive, edit command or
    // Crease/SoftCrease action stores a weight that high (see the
    // constant), so an unmarked boundary vertex is unaffected.
    const isMarkedCorner = edges.length === 2
      && a.edge.faces.length === 1 && b.edge.faces.length === 1
      && creaseWeight(cage, vIdx, a.other) > MARKED_CORNER_WEIGHT_FLOOR
      && creaseWeight(cage, vIdx, b.other) > MARKED_CORNER_WEIGHT_FLOOR;
    creasePoint = isMarkedCorner ? P.slice() : scale(add(add(A, B), scale(P, 6)), 1 / 8);
    blend = (a.sharpness + b.sharpness) / 2;
  } else {
    creasePoint = P.slice();
    blend = Math.max(...sharp.map((c) => c.sharpness));
  }

  if (blend >= 1) return creasePoint;
  const smoothPoint = smoothVertexRule(cage, ctx, vIdx);
  return lerp(smoothPoint, creasePoint, blend);
}

// One Catmull-Clark subdivision pass. Returns a new cage (the input is
// never mutated) whose faces are all quads, with crease weights
// decremented by exactly 1.0 on every child of a creased edge (dropped
// once decayed to 0).
//
// Vertex points keep the same index as their originating vertex (indices
// 0..N-1 of the new cage are the moved originals, in order); the tests
// rely on this — an extraordinary vertex's index is stable across
// repeated subdivision, so its position can be tracked call over call.
// Edge points are appended next (one per unique edge, in buildTopology's
// edgeMap iteration order), then face points last.
export function subdivideCatmullClark(cage) {
  validateCage(cage);
  const ctx = buildTopology(cage);
  const nVerts = cage.vertices.length;

  const newVertices = new Array(nVerts);
  for (let i = 0; i < nVerts; i++) newVertices[i] = computeVertexPoint(cage, ctx, i);

  const edgePointIdx = new Map();
  for (const [key, edge] of ctx.edgeMap) {
    edgePointIdx.set(key, newVertices.length);
    newVertices.push(computeEdgePoint(cage, ctx, edge));
  }

  const facePointIdx = new Array(cage.faces.length);
  cage.faces.forEach((_, fi) => {
    facePointIdx[fi] = newVertices.length;
    newVertices.push(ctx.faceCentroids[fi].slice());
  });

  const newFaces = [];
  cage.faces.forEach((face, fi) => {
    const n = face.length;
    for (let c = 0; c < n; c++) {
      const vCurr = face[c];
      const vNext = face[(c + 1) % n];
      const vPrev = face[(c - 1 + n) % n];
      const eNext = edgePointIdx.get(edgeKey(vCurr, vNext));
      const ePrev = edgePointIdx.get(edgeKey(vPrev, vCurr));
      newFaces.push([vCurr, eNext, facePointIdx[fi], ePrev]);
    }
  });

  // Semi-sharp decrement: every creased edge (weight > 0, boundary or
  // interior — a boundary edge's weight does not affect its own sharpness,
  // which is forced to 1, but its children carry it forward like any
  // other edge; the marked-corner rule in computeVertexPoint reads it)
  // hands its two children max(0, weight-1). New spoke edges (an edge
  // point or a vertex point to its face point) get no entry — always
  // smooth: Catmull-Clark refinement never originates a crease, only
  // carries an existing one along the subdivided original edge.
  const newCreases = {};
  for (const [key, edge] of ctx.edgeMap) {
    const w = creaseWeight(cage, edge.v0, edge.v1);
    if (w <= 0) continue;
    const nw = Math.max(0, w - 1);
    if (nw <= 0) continue;
    const ePt = edgePointIdx.get(key);
    newCreases[edgeKey(edge.v0, ePt)] = nw;
    newCreases[edgeKey(edge.v1, ePt)] = nw;
  }

  return { vertices: newVertices, faces: newFaces, creases: newCreases };
}

// Display mesh: a cage/mode-in, triangle-soup-out function built on the
// refinement math above, plus the heuristic that picks how many
// refinement levels to run. Kernel-only: no THREE.js, no app-layer object.
//
// Refinement level, chosen per whole cage. A per-face adaptive scheme
// (refining some faces of one cage more than others) needs T-junction
// handling between refined and unrefined neighbors, which this does not
// do. A coarse cage (few faces — display would look faceted at 2 levels)
// or a creased cage (a crease line needs one more level to read as a
// tangent-continuous feature rather than a kink) gets 3 levels; a dense,
// uncreased cage gets 2.
export function chooseSuperBRefinementLevel(cage, opts = {}) {
  const coarseFaceThreshold = opts.coarseFaceThreshold ?? 24;
  const hasCreases = !!(cage.creases && Object.values(cage.creases).some((w) => w > 0));
  const coarse = cage.faces.length <= coarseFaceThreshold;
  return (hasCreases || coarse) ? 3 : 2;
}

// Fan triangulation of one face loop (n>=3 — a cage face can be an ngon
// before the first refinement pass; after one pass every face is a quad),
// for a THREE.js BufferGeometry, which takes triangles only.
export function triangulateFace(face) {
  const tris = [];
  for (let i = 1; i < face.length - 1; i++) tris.push([face[0], face[i], face[i + 1]]);
  return tris;
}

// Mode 'box' returns the raw cage, triangulated as-is (the faceted
// control-net display); mode 'smooth' (default) runs `level` (or the
// heuristic above) subdivideCatmullClark passes and triangulates the
// result — an approximation of the limit surface by a finer cage, never
// eigen-basis limit evaluation. Returns the triangle soup (for THREE.js)
// and the quad `faces` array (for a wireframe overlay or per-face
// selection); never mutates the input cage.
export function superbDisplayMesh(cage, mode = 'smooth', opts = {}) {
  if (mode === 'box') {
    const triangles = cage.faces.flatMap(triangulateFace);
    return { vertices: cage.vertices.map((v) => v.slice()), triangles, level: 0, faces: cage.faces };
  }
  const level = opts.level ?? chooseSuperBRefinementLevel(cage, opts);
  let cur = cage;
  for (let i = 0; i < level; i++) cur = subdivideCatmullClark(cur);
  const triangles = cur.faces.flatMap(triangulateFace);
  return { vertices: cur.vertices.map((v) => v.slice()), triangles, level, faces: cur.faces };
}
