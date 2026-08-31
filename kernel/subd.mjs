// CATMULL-CLARK SUBDIVISION-SURFACE MATH — the shared kernel underlying our
// own SuperB object type (Rhino calls the analogous feature
// "SubD"). KERNEL ONLY: this module has no notion of an app-layer object,
// a display/render tessellation policy, or UI — it is pure cage-in,
// finer-cage-out math, exactly like every other kernel/*.mjs module in this
// project.
//
// A CAGE (the control net — NOT the geometry; the limit surface this
// defines is the real object, same relationship a NURBS control polygon has
// to its curve) is plain data:
//   { vertices: [[x,y,z], ...],
//     faces:    [[i0,i1,...], ...],   // >=3 indices per face, quads preferred, ngons legal
//     creases:  { "i_j": weight, ... } }  // sparse — an absent key means weight 0 (smooth)
// A crease weight is a soft-crease value (Rhino 8's SubDCrease scale, 0-100)
// stored on an EDGE, keyed by `edgeKey(i,j)` (order-independent). Boundary
// edges (used by exactly one face) are ALWAYS fully sharp regardless of any
// stored weight — there is no "smooth" alternative to blend a boundary edge
// against, since there's no second adjacent face to build one from.
//
// THE REFINEMENT STEP (subdivideCatmullClark) implements the classic
// Catmull-Clark rules (Catmull & Clark 1978; the face/edge/vertex point
// formulas as commonly restated, e.g. the Wikipedia "Catmull–Clark
// subdivision surface" summary, cross-checked here against the DeRose 1998
// boundary rule and hand-verified against a known worked cube example — see
// this file's own function comments and test/subd.test.mjs's own derivations
// for the actual arithmetic, not just an appeal to authority):
//   - FACE POINT: the centroid of a face's own vertices (any n-gon).
//   - EDGE POINT: (v0 + v1 + f0 + f1) / 4 for a smooth interior edge (f0/f1
//     its two adjacent face points); the plain midpoint (v0+v1)/2 for a
//     boundary edge or a fully-sharp crease.
//   - VERTEX POINT: (F + 2R + (n-3)P) / n for a smooth interior vertex of
//     valence n (F = average of adjacent face points, R = average of
//     adjacent EDGE MIDPOINTS — the original edges, not the edge points
//     above); the standard boundary/crease curve rule (A + 6P + B) / 8 for a
//     vertex sitting on exactly one crease/boundary LINE (A, B its two
//     crease-neighbor vertices); the vertex's own unmoved position for a
//     "corner" (0, 1, or 3+ sharp edges meeting there, per the standard
//     Catmull-Clark corner treatment).
//
// SEMI-SHARP CREASES (DeRose/Kobbelt-style, the actual mechanism behind
// Rhino 8's SoftCrease): a crease weight above 0 linearly BLENDS the smooth
// and fully-sharp edge/vertex rules above (blend fraction = the weight
// clamped to [0,1]), and DECREMENTS BY EXACTLY 1.0 every time
// subdivideCatmullClark is called — the returned cage's own creases map
// already carries the decremented weight on each edge's two children, so a
// caller just calls this function repeatedly with no separate "current
// level" bookkeeping of its own. A weight that has decayed to 0 is dropped
// from the map entirely (an absent key IS weight 0), so the edge behaves as
// an ordinary smooth edge forever after, with no further special-casing.
//
// V1 SCOPE, stated honestly, matching the milestone's own explicit
// boundaries: no Stam eigen-basis limit-surface evaluation (this module
// only ever produces a FINER CAGE, never the true smooth limit position —
// display is a fixed 2-3 refinement levels, adaptive, not
// exact evaluation); no non-manifold-edge REPAIR (this module makes no
// attempt to fix up genuinely non-manifold input — an edge used by 3+
// faces is simply FORCED fully sharp, same as a real boundary edge, since
// there's no well-defined "average of N face points" smooth alternative
// for it either — a defensible, stated fallback, not a claim of
// correctness for genuinely non-manifold input);
// no ToNURBS, no display tessellation, no object type, no UI — all
// explicitly out of scope for THIS (the refinement-math) milestone.
//
// STATUS RECONCILED: the NEXT milestone (SuperB object type +
// display + primitive creation commands, hand-pasted into the app)
// extends this file with the display-mesh/refinement-level functions near
// the bottom (chooseSuperBRefinementLevel/triangulateFace/
// superbDisplayMesh) — still kernel-only, no THREE.js/app-layer object
// here either; ToNURBS remains genuinely untouched and deferred.

import { add, scale } from './vec3.mjs';

function midpoint(a, b) { return scale(add(a, b), 0.5); }
function lerp(a, b, t) { return add(scale(a, 1 - t), scale(b, t)); }
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

// MARKED-CORNER WEIGHT FLOOR — see computeVertexPoint's own MARKED-CORNER
// EXTENSION comment below for the full derivation of WHY a boundary edge's
// stored weight is reused as a corner marker at all. This threshold must
// NOT fire on any
// nonzero stored weight — that would collide with this app's own real,
// shipped Crease/SoftCrease commands — the app's SUPERB_CREASE_
// LEVEL_SCALE (the highest weight either command can ever store, a plain
// "full harden" edge) is exactly 3, so an entirely ordinary "harden both
// boundary edges at a plain SuperBPlane corner" gesture through the
// existing, shipped UI would otherwise silently pin that corner at P instead
// of the correct boundary-curve smoothing. The stored
// weight must clear a floor no ordinary Crease/SoftCrease action can ever
// reach (their own combined reachable range is a closed [0, 3]) — only a
// caller that deliberately stores a weight this high on purpose (TOSUBD's
// own DEFAULT_CORNER_CREASE_WEIGHT, kernel/subdconvert.mjs, set well above
// this floor) can ever trigger the marker. Kept as a plain kernel-local
// constant (not an import of the app's own SUPERB_CREASE_LEVEL_SCALE) —
// this module stays app-independent, matching every other kernel module's
// own "no app-layer constant imports" discipline; the floor is simply
// chosen generously above ANY plausible real-world semi-sharp weight.
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

// The raw, UNCLAMPED stored crease weight for an edge (0 if absent — the
// sparse-map default). Callers wanting the CURRENT step's effective
// blend fraction (accounting for boundary-forcing and the [0,1] clamp)
// should go through the internal edgeSharpness() below instead; this
// function is the plain "what did the cage's own creases map say" read.
export function creaseWeight(cage, v0, v1) {
  const w = cage.creases && cage.creases[edgeKey(v0, v1)];
  return typeof w === 'number' ? w : 0;
}

// A boundary edge (used by exactly one face) is ALWAYS fully sharp,
// regardless of any stored crease weight — there is no second adjacent
// face to build a "smooth" alternative from. A NON-MANIFOLD edge (used by
// 3+ faces) is ALSO forced fully sharp, for the identical reason — the
// ordinary smooth rule's "average of exactly 2 adjacent face points" has
// no well-defined generalization to 3+, so treating it as boundary-like
// is the same honest fallback, not a claim of correctness for genuinely
// non-manifold input (see this file's own header comment). An ORDINARY
// interior edge (used by exactly 2 faces) is the only case that ever
// reads its own stored crease weight — its effective sharpness THIS
// subdivision step is that weight clamped to [0,1] (the semi-sharp blend
// fraction).
function edgeSharpness(cage, edge) {
  if (edge.faces.length !== 2) return 1;
  return clamp01(creaseWeight(cage, edge.v0, edge.v1));
}

// Centroid of a face's own vertices — works for any n-gon (n>=3), quad or
// otherwise; the FACE POINT rule is identical regardless of face size.
export function computeFacePoint(cage, faceIdx) {
  const face = cage.faces[faceIdx];
  let c = [0, 0, 0];
  for (const vi of face) c = add(c, cage.vertices[vi]);
  return scale(c, 1 / face.length);
}

// Builds the adjacency structures a single subdivision pass needs, derived
// FRESH from the cage's own faces every time (never carried as separate
// persistent state) — the same "boundary-ness is always structural, never
// hand-tracked" discipline this module relies on throughout, so a caller
// re-deriving topology from a freshly-subdivided cage always gets the
// correct new boundary/interior classification for free.
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

// EDGE POINT — smooth interior rule (v0+v1+f0+f1)/4, generalized to
// however many faces genuinely touch the edge (2 for an ordinary manifold
// interior edge; the boundary/fully-sharp case never reaches this branch,
// see below), blended toward the plain midpoint by the edge's own current
// sharpness.
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
// F the average of adjacent face points, R the average of adjacent EDGE
// MIDPOINTS (the original edges — not the edge points above, a real and
// easy point of confusion this project's own header comment calls out
// directly). Never called for a vertex whose own blend has already
// resolved to fully-sharp (computeVertexPoint short-circuits before this),
// so the valence-1/2 degenerate cases below are purely defensive.
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

// VERTEX POINT — classifies the vertex by how many of its incident edges
// are currently sharp (boundary edges always count; interior creased
// edges count whenever their own clamped weight is > 0), matching the
// standard DeRose/Kass/Truong semi-sharp-crease vertex taxonomy exactly:
//   0 sharp edges -> the ordinary smooth interior rule.
//   exactly 1     -> a DART (the OPEN END of a crease line, dead-ending
//                    inside the surface rather than closing a loop or
//                    reaching a boundary/corner) — the standard rule is
//                    the ORDINARY SMOOTH rule, unconditionally, REGARDLESS
//                    of that one edge's own sharpness value. This is
//                    exactly what makes a semi-sharp crease fade OUT
//                    smoothly at its open end instead of leaving a
//                    permanent pinch there — a dart is deliberately never
//                    blended toward any "corner" position the way a true
//                    3+-crease corner is (see below): there is no crease
//                    LINE at a dart (that needs exactly 2 sharp edges),
//                    so there is nothing for a curve-subdivision rule to
//                    run along, and blending toward the vertex's own
//                    unmoved position here would just leave a visible
//                    dimple at the crease's own open end, not a fade-out.
//   exactly 2     -> a crease/boundary LINE running through this vertex —
//                    the standard (A + 6P + B) / 8 curve-subdivision rule,
//                    A/B the vertex's two crease-neighbor points, blended
//                    toward the smooth rule by the (averaged) sharpness of
//                    those two edges.
//   3+            -> a CORNER (3+ creases meeting) — the vertex stays at
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
    // MARKED-CORNER EXTENSION (built for TOSUBD's own
    // Corners=Yes option) — a vertex with EXACTLY two
    // incident edges, BOTH of them boundary edges (the genuine geometric
    // signature of a single-face grid corner — e.g. one of a NURBS
    // surface's own 4 untrimmed corners; there is no third edge for such a
    // vertex to ever reach the ordinary 3+ CORNER branch below), has no
    // way to ask for the "hold fixed at P" corner treatment under the
    // plain sharp-edge-COUNT taxonomy this function otherwise uses —
    // sharp.length is structurally stuck at 2 for it forever, so it always
    // falls into the ordinary boundary-CURVE rule just below, which
    // SMOOTHS through the vertex like any other point along the boundary,
    // rounding off what may have been a genuinely sharp corner in the
    // source geometry (the standard, well-known Catmull-Clark "open
    // corner rounds off" behavior, the same reason an uncreased SuperBBox
    // converges toward a sphere).
    //
    // A boundary edge's own STORED crease weight is otherwise completely
    // INERT (edgeSharpness above forces a boundary edge's sharpness to
    // exactly 1 regardless of what the creases map says — see this file's
    // own header/subdivideCatmullClark comment, which already names this
    // exact possible extension: "this only matters if a boundary edge was
    // ALSO given an explicit weight for some future non-boundary
    // continuation"). This is that continuation: reusing that otherwise-
    // dead value as a deliberate MARKER — both of this vertex's two
    // boundary edges carrying a stored weight ABOVE MARKED_CORNER_WEIGHT_
    // FLOOR (never the case for any cage that has never stored a boundary
    // crease this high before this, since every primitive/edit command,
    // and every ordinary Crease/SoftCrease action a person can actually
    // take through the shipped UI, ships with far less than this floor by
    // default — see that constant's own header for the exact collision
    // this floor closes) means the corner was intentionally marked, and it
    // is held at its own position P (the exact "stays put" treatment the
    // ordinary 3+ branch below already gives an interior corner) rather
    // than smoothed along the boundary curve. An ordinary, unmarked
    // boundary vertex (weight 0, or any ordinary Crease/SoftCrease weight
    // up to a full harden, on either edge — the ONLY case any existing
    // cage/test in this app has ever produced, and the ONLY case its own
    // shipped UI can ever produce) is completely unaffected.
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

// THE REFINEMENT STEP — one Catmull-Clark subdivision pass. Returns a
// brand-new cage (the input is never mutated) whose faces are ALWAYS quads
// (Catmull-Clark's own defining "converges to quads regardless of input
// face size" property), with crease weights decremented by exactly 1.0
// on every child of a creased edge (dropped entirely once decayed to 0).
//
// Vertex points keep the SAME index as their originating vertex (indices
// 0..N-1 of the new cage are the moved originals, in order) — a real,
// load-bearing convention this module's own tests rely on directly (an
// extraordinary vertex's index is stable across repeated subdivision,
// so its position can be tracked call over call with no re-derivation).
// Edge points are appended next (one per unique edge, in the same order
// buildTopology's own edgeMap iterates them), then face points last.
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

  // Semi-sharp decrement: every creased edge (weight > 0, whether boundary
  // or interior — a boundary edge's own weight is irrelevant to ITS
  // sharpness this step, since that's always forced to 1, but its
  // children still carry forward whatever the stored map said, same as
  // any other edge; this only matters if a boundary edge was ALSO given
  // an explicit weight for some future non-boundary continuation) hands
  // its two children max(0, weight-1); brand-new "spoke" edges (an edge
  // point or a vertex point to its face point) are never given an entry
  // at all — always smooth, matching real Catmull-Clark refinement, which
  // never originates a new crease, only ever carries an existing one
  // forward along the subdivided original edge.
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

// DISPLAY-MESH MILESTONE — the SuperB object-type/display
// half of the SubD work, built directly on top of the refinement math above
// (unchanged by this addition). Still kernel-only: no THREE.js, no app-
// layer object — a pure cage/mode-in, triangle-soup-out function, plus the
// one small heuristic that picks HOW MANY refinement levels to run.
//
// ADAPTIVE REFINEMENT LEVEL — the "2-3 refinement levels,
// adaptive" rule, and its explicit permission to
// build "a real, simple, honestly-scoped adaptive heuristic, not a
// research-grade error-driven refinement scheme." A genuinely PER-FACE
// adaptive scheme (refining only some faces of one cage more than others)
// needs T-junction handling between refined and unrefined neighbors — real,
// separate, harder engineering, correctly out of scope per the milestone's
// own explicit ruling-out of research-grade adaptive refinement. This
// heuristic instead adapts at the WHOLE-CAGE granularity: a coarse cage (few
// faces — display would look faceted at just 2 levels) or a creased cage
// (a crease line needs one more level to read as a real tangent-continuous
// feature rather than a visible kink) gets 3 levels; an already-dense,
// uncreased cage gets 2. This is the real, stated, whole-object heuristic
// committed to here — not a per-face one.
export function chooseSuperBRefinementLevel(cage, opts = {}) {
  const coarseFaceThreshold = opts.coarseFaceThreshold ?? 24;
  const hasCreases = !!(cage.creases && Object.values(cage.creases).some((w) => w > 0));
  const coarse = cage.faces.length <= coarseFaceThreshold;
  return (hasCreases || coarse) ? 3 : 2;
}

// Fan triangulation of one face loop (n>=3, any n — a Catmull-Clark cage
// face can be an ngon before the first refinement pass; after >=1 pass
// every face is a quad, see subdivideCatmullClark's own "converges to
// quads regardless of input face size" property) — the plain, standard
// way to hand any polygon face to a THREE.js BufferGeometry, which only
// ever wants triangles.
export function triangulateFace(face) {
  const tris = [];
  for (let i = 1; i < face.length - 1; i++) tris.push([face[0], face[i], face[i + 1]]);
  return tris;
}

// THE DISPLAY MESH ITSELF — mode 'box' returns the RAW cage, triangulated
// as-is (the flat/faceted "control net as real geometry" display);
// mode 'smooth' (default) runs `level` (or the heuristic above)
// real subdivideCatmullClark passes and triangulates the RESULT — the true
// limit-surface APPROXIMATION shipped here (display is a finer
// cage at a fixed few levels, never true eigen-basis limit evaluation, per
// that having been ruled out explicitly as unneeded engineering).
// Returns both the triangle soup (for direct THREE.js consumption) and the
// still-quad `faces` array (useful for a wireframe overlay, or a future
// milestone's own per-face selection) — never mutates the input cage.
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
