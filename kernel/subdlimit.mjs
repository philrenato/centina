// Exact Catmull-Clark limit-surface evaluation. A sibling of subd.mjs:
// subd.mjs produces a finer cage (one discrete refinement step); this module
// evaluates the infinite-refinement limit exactly, at an arbitrary (u,v),
// with no loop.
//
// Regular-patch fast path. Away from any extraordinary vertex (valence != 4)
// or crease, the Catmull-Clark limit surface is exactly an ordinary uniform
// bicubic B-spline patch over the 16 control points surrounding a face (the
// standard Catmull-Clark convergence identity, not an approximation). It is
// evaluated with kernel/surface.mjs's surfacePointAndPartials (the rational
// tensor-product evaluator used for ordinary NURBS surfaces) on a synthetic,
// non-rational, uniform-knot patch built from the 16 points.
//
// The knot vector [-3,-2,-1,0,1,2,3,4] (unclamped, evenly spaced — 4 control
// points, degree 3, so knots.length == n+p+1 == 4+3+1 == 8) is not the
// clamped [0,0,0,0,1,1,1,1] a 4-point Bezier patch would use: the 16 stencil
// points are the control points of a uniform B-spline (the shape repeated
// regular subdivision converges to), not Bezier control points, and a clamped
// knot vector would evaluate a different patch through the same 16 points.
// With 4 control points and degree 3 there is one valid span regardless of
// clamping (n - p == 1); the shift to [-3..4] puts that span, [knots[p],
// knots[n]] == [knots[3], knots[4]], at [0,1], while the surrounding knot
// values (which basisFuns/dersBasisFuns read from a window centered on the
// span) keep the basis functions uniform rather than clamped.
//
// Stencil layout: grid[i][j] for i,j in 0..3 (U-direction first index,
// matching kernel/surface.mjs's ctrlNet[i][j] convention). The evaluated face
// sits at the center 2x2 of the grid ((1,1), (1,2), (2,1), (2,2) are its 4
// corners); the remaining 12 points are its one-ring. regularFaceStencil
// below reads that layout out of a cage.

import { surfacePointAndPartials } from './surface.mjs';
import { add, scale } from './vec3.mjs';
import { buildTopology, subdivideCatmullClark, edgeKey, creaseWeight } from './subd.mjs';

const REGULAR_PATCH_KNOTS = [-3, -2, -1, 0, 1, 2, 3, 4];

// grid4x4[i][j] = [x, y, z] or [x, y, z, w] (weight defaults to 1 — the
// regular fast path is always non-rational, a plain average of real
// points, never weighted).
export function bicubicRegularPatchSurface(grid4x4) {
  if (grid4x4.length !== 4 || grid4x4.some((row) => row.length !== 4)) {
    throw new Error('bicubicRegularPatchSurface: expected a 4x4 grid of control points');
  }
  const ctrlNet = grid4x4.map((row) => row.map((p) => [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]));
  return { degU: 3, degV: 3, knotsU: REGULAR_PATCH_KNOTS, knotsV: REGULAR_PATCH_KNOTS, ctrlNet };
}

// Clamped form of the same patch: the identical surface, with a knot array
// whose ends bound the valid domain.
//
// An unclamped patch evaluates correctly through surfacePoint for a (u,v)
// inside its span. The problem is that its knot array spans [-3,4] while its
// valid domain is [0,1], and roughly ten consumers across this kernel and the
// app tier derive a surface's domain as `knots[0] .. knots[last]` — right for
// a clamped surface, wrong by a factor of seven here. extractBorderCurves is
// the plainest case: it asks for the border at knots[0], four knots outside
// the span, and returns a curve 131 units from the real patch edge. Clamping
// fixes all of those consumers at once.
//
// Why not knot insertion: insertKnot/extractSubCurve are exact for clamped
// curves and not here. Isolating a sub-range raises the boundary knot to
// multiplicity degree+1, and that last step duplicates a control point
// without blending. On a clamped curve that is correct, because at
// multiplicity `degree` the duplicated point already lies on the curve. On an
// unclamped one it does not, and the result is a different curve — off by
// 0.37 units on a plain uniform cubic spanning about 3. decomposeToBezier
// assumes clamped input too. See kernel/knots.mjs's own note.
//
// What is used instead is the standard closed-form conversion of a uniform
// cubic B-spline segment to Bezier form; REGULAR_PATCH_KNOTS is uniform by
// construction. For the span governed by P0..P3:
//     B0 = (P0 + 4*P1 + P2) / 6      B1 = (2*P1 + P2) / 3
//     B2 = (P1 + 2*P2) / 3           B3 = (P1 + 4*P2 + P3) / 6
// It matches the curve's own sampled points to 4.5e-16 (machine precision).
// Being linear, it applies to a tensor grid one direction at a time in either
// order.
//
// Scope: correct for this knot vector and degree only, not for an arbitrary
// unclamped surface; nothing else in this kernel produces an unclamped
// surface. The patches are always non-rational (bicubicRegularPatchSurface's
// stated scope), so the weight rides through as 1 rather than being blended
// in homogeneous space.
function uniformCubicSpanToBezier(p0, p1, p2, p3) {
  const mix = (...terms) => [0, 1, 2].map((d) => terms.reduce((acc, [pt, k]) => acc + pt[d] * k, 0));
  return [
    mix([p0, 1 / 6], [p1, 4 / 6], [p2, 1 / 6]),
    mix([p1, 2 / 3], [p2, 1 / 3]),
    mix([p1, 1 / 3], [p2, 2 / 3]),
    mix([p1, 1 / 6], [p2, 4 / 6], [p3, 1 / 6]),
  ];
}
const CLAMPED_PATCH_KNOTS = [0, 0, 0, 0, 1, 1, 1, 1];
export function clampedBicubicPatchSurface(grid4x4) {
  const src = bicubicRegularPatchSurface(grid4x4).ctrlNet; // reuses its own shape/validation
  // U pass — each control column (fixed j, varying i) is a uniform cubic span.
  const cols = [];
  for (let j = 0; j < 4; j++) cols.push(uniformCubicSpanToBezier(src[0][j], src[1][j], src[2][j], src[3][j]));
  const mid = [];
  for (let i = 0; i < 4; i++) mid.push(cols.map((c) => c[i]));
  // V pass — each row of that result is a uniform cubic span in the other direction.
  const ctrlNet = mid.map((row) => uniformCubicSpanToBezier(row[0], row[1], row[2], row[3]).map((p) => [p[0], p[1], p[2], 1]));
  return { degU: 3, degV: 3, knotsU: CLAMPED_PATCH_KNOTS, knotsV: CLAMPED_PATCH_KNOTS, ctrlNet };
}

// u, v in [0,1]. (0,0)/(1,0)/(0,1)/(1,1) are the patch's parametric corners,
// each converging to the true Catmull-Clark limit position of one of the
// face's 4 vertices (not any single control point: a regular vertex's limit
// position is a weighted average over its one-ring, which is what evaluating
// this patch at a corner computes).
export function regularPatchPointAndPartials(grid4x4, u, v) {
  return surfacePointAndPartials(bicubicRegularPatchSurface(grid4x4), u, v);
}

// Halstead, Kass & DeRose (1993) vertex limit-position mask. Independent
// ground truth for the eigenbasis cross-check below: it reuses only subd.mjs's
// buildTopology/computeFacePoint, nothing from the regular-patch path or the
// eigendecomposition.
//
// Derivation. Consider a rotationally symmetric neighborhood of an
// extraordinary vertex P0 of valence n — n spoke neighbor vertices Qi and, for
// each of the n incident quad faces, one far (diagonal) corner vertex Ri.
// Define the three rotationally invariant scalars p = P0, q = avg(Qi),
// r = avg(Ri). One Catmull-Clark step (subd.mjs's face/edge/vertex-point
// rules, restricted to this symmetric case) gives:
//   new face-centroid average  = (p + 2q + r) / 4
//   new edge-point average     = (3p + 4q + r) / 8
//   new vertex point (smoothVertexRule, valid for any n)
//                               = [(4n-7)p + 6q + r] / (4n)
// — a 3x3 linear map (p,q,r) -> (p',q',r') that depends on n only through its
// coefficients. Every row sums to 1 (each new value is a weighted average of
// old ones), so the matrix is row-stochastic: repeated application converges
// p, q and r to the same value, the dominant (eigenvalue-1) fixed point, which
// is the vertex's limit position (as the neighborhood shrinks to a point, its
// center, spoke average and corner average coincide). Solving (M^T - I) v = 0
// gives v = (n, 4, 1) (the test file re-derives each step); normalized
// (sum = n+5), the limit position is:
//   L = (n*p + 4*q + r) / (n+5)
// Converting q, r into subd.mjs's R (average adjacent edge midpoint, not
// neighbor vertex) and F (average adjacent face centroid) — R = (p+q)/2 so
// q = 2R-p; F = (p+2q+r)/4 so r = p+4F-4R — gives the form used below:
//   L = [(n-3)*P + 4*R + 4*F] / (n+5)
// Cross-check against the published regular-vertex (n=4) mask in terms of the
// vertex V, its 4 edge-neighbor vertices Qi and its 4 face-diagonal corners Ri,
// "(16*V + 4*sum(Qi) + sum(Ri)) / 36": substituting sum(Qi)=4q, sum(Ri)=4r
// gives (16p+16q+4r)/36, identical to (n*p+4*q+r)/(n+5) at n=4 ->
// (4p+4q+r)/9 -> (16p+16q+4r)/36.
export function vertexLimitMaskFromRF(P, R, F, n) {
  const num = add(add(scale(P, n - 3), scale(R, 4)), scale(F, 4));
  return scale(num, 1 / (n + 5));
}

// Cage-facing wrapper. R and F are computed the same way computeVertexPoint's
// smoothVertexRule computes them, so this agrees with what a subdivision step
// takes as this vertex's F/R inputs.
export function vertexLimitPosition(cage, vIdx, ctx = buildTopology(cage)) {
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
    R = add(R, scale(add(P, cage.vertices[other]), 0.5));
  }
  R = scale(R, 1 / n);
  return vertexLimitMaskFromRF(P, R, F, n);
}

// The subdivision matrix A(n) for Stam's eigenbasis method (the general,
// unreduced case). A(n) is a (2n+1)x(2n+1) linear map over the standard local
// neighborhood template of an extraordinary vertex of valence n — index 0 the
// vertex, indices 1..n its n spoke edge-neighbors, indices n+1..2n the far
// diagonal corner of each incident quad face (face i = [0, spoke_i, far_i,
// spoke_{i+1}]) — mapping this neighborhood to the new one around the same
// vertex one subdivision step later (still valence n, same local topology).
//
// Built by running subd.mjs's subdivideCatmullClark on unit basis vectors over
// this template (`standardNeighborhoodCage`), one column of A(n) at a time,
// rather than re-deriving the stencil coefficients symbolically. Every one of
// computeVertexPoint's/computeEdgePoint's/computeFacePoint's rules is a linear
// (weighted-average) function of neighboring positions, so this is exactly
// A(n) for each coordinate axis independently (a "unit basis vector" here is a
// single scalar placed at one neighborhood point and 0 elsewhere; the 3D case
// is the same scalar operation applied once per axis).
//
// Self-containment (checked in the test file): every new center/spoke/corner
// value reads only points inside the (2n+1)-point neighborhood. An outer edge
// of the template, e.g. spoke_i to far_i, is a boundary edge from its own
// perspective (only 1 face uses it here), but this builder never reads a new
// vertex point at a spoke/far position — only the new center vertex point,
// the new edge points on the vertex's n original edges, and the new face
// points of its n original faces, none of which reads outside the template.
export function standardNeighborhoodCage(n) {
  const vertices = [[0, 0, 0]];
  for (let i = 0; i < n; i++) vertices.push([1, i, 0]); // placeholder positions — A(n) is topology-only; geometry here is never read as geometry, only overwritten with unit basis scalars during the build
  for (let i = 0; i < n; i++) vertices.push([2, i + 0.5, 0]);
  const faces = [];
  for (let i = 0; i < n; i++) {
    const spokeI = 1 + i, farI = 1 + n + i, spokeNext = 1 + ((i + 1) % n);
    faces.push([0, spokeI, farI, spokeNext]);
  }
  return { vertices, faces, creases: {} };
}

// Where an original (undirected) edge's new edge-point vertex lands in a
// subdivideCatmullClark output, by subd.mjs's documented ordering (vertex
// points keep their original index; edge points next, in buildTopology's
// edgeMap iteration order; face points last). `template` must be structurally
// identical to the cage subdivideCatmullClark was called on: edgeMap order
// depends only on the faces, not on the vertex positions, which the matrix
// builder overwrites for each unit-basis pass.
export function edgePointIndexMap(template) {
  const ctx = buildTopology(template);
  const map = new Map();
  let i = template.vertices.length;
  for (const key of ctx.edgeMap.keys()) map.set(key, i++);
  return map;
}
export function facePointIndexBase(template) {
  const ctx = buildTopology(template);
  return template.vertices.length + ctx.edgeMap.size;
}

// Reads the (2n+1)-length neighborhood-template vector (center, n spokes, n
// corners, in standardNeighborhoodCage's index order) out of a refined cage,
// full [x,y,z] per point. The test file uses it to cross-check A(n) against a
// non-unit-basis subdivideCatmullClark call on all 3 axes at once; the matrix
// builder reads one channel per pass directly.
export function readNeighborhoodVectors(template, refined, n, edgeIdx, faceBase) {
  const out = [refined.vertices[0].slice()];
  for (let i = 1; i <= n; i++) out.push(refined.vertices[edgeIdx.get(edgeKey(0, i))].slice());
  for (let fi = 0; fi < n; fi++) out.push(refined.vertices[faceBase + fi].slice());
  return out;
}
function readNeighborhoodX(template, refined, n, edgeIdx, faceBase) {
  const out = [refined.vertices[0][0]];
  for (let i = 1; i <= n; i++) out.push(refined.vertices[edgeIdx.get(edgeKey(0, i))][0]);
  for (let fi = 0; fi < n; fi++) out.push(refined.vertices[faceBase + fi][0]);
  return out;
}

// A(n) as a (2n+1)x(2n+1) array of arrays, A[row][col]. Applying it to a
// (2n+1)-vector v (applyMatrix) reproduces one subdivideCatmullClark step on
// that neighborhood, for x, y or z independently.
export function buildSubdivisionMatrix(n) {
  const template = standardNeighborhoodCage(n);
  const edgeIdx = edgePointIndexMap(template);
  const faceBase = facePointIndexBase(template);
  const dim = 2 * n + 1;
  const columns = [];
  for (let col = 0; col < dim; col++) {
    const testCage = {
      vertices: template.vertices.map((_, i) => [i === col ? 1 : 0, 0, 0]),
      faces: template.faces,
      creases: {},
    };
    const refined = subdivideCatmullClark(testCage);
    columns.push(readNeighborhoodX(template, refined, n, edgeIdx, faceBase));
  }
  // columns[col][row] -> A[row][col]
  const A = Array.from({ length: dim }, () => new Array(dim).fill(0));
  for (let col = 0; col < dim; col++) {
    for (let row = 0; row < dim; row++) A[row][col] = columns[col][row];
  }
  return A;
}

// Plain (2n+1)-vector matrix-vector product — A(n) applied to a real
// scalar (or per-axis) neighborhood vector.
export function applyMatrix(A, v) {
  return A.map((row) => row.reduce((sum, a, j) => sum + a * v[j], 0));
}

// Vertex-limit weights per individual neighbor (the P/R/F mask above,
// expanded to all 2n+1 points), plus a numerical eigendecomposition
// cross-check by power iteration.
//
// Scope: Stam's full construction evaluates the limit surface at an arbitrary
// (u,v) near an extraordinary vertex by applying A(n)^k through a cached
// eigendecomposition and handing a regular sub-patch of the refined
// neighborhood to the regular fast path. That needs a larger template than
// the (2n+1) neighborhood here (a full 16-point stencil one ring further out)
// plus the picking matrices of Stam's paper, and is not implemented. What is
// provided is an exact closed-form limit-position weight per individual
// neighbor, derived below, and cross-checked numerically against A(n)'s
// dominant eigenvector by power iteration, which never reads the closed form.
// Every cage vertex position is exact through this; an arbitrary interior
// (u,v) between a vertex and a regular region falls back to bounded discrete
// refinement.
//
// Derivation — direct expansion of R/F in terms of the individual spoke
// values Q_1..Q_n and corner values C_1..C_n, with no symmetry assumption
// (valid for an irregular neighborhood, not just a symmetric fan):
//   R = (1/n) * sum_i (P+Q_i)/2  =  P/2 + (1/(2n)) * sum_i Q_i
//   F = (1/n) * sum_i (P+Q_i+C_i+Q_{i+1})/4
//     = P/4 + (1/(2n)) * sum_i Q_i + (1/(4n)) * sum_i C_i
//     [sum_i Q_{i+1 mod n} == sum_i Q_i, a cyclic relabeling of the same
//      n terms — the "+1" shift never drops or duplicates one]
// Substituting into L = [(n-3)P + 4R + 4F] / (n+5):
//   4R = 2P + (2/n) * sum_i Q_i
//   4F = P + (2/n) * sum_i Q_i + (1/n) * sum_i C_i
//   (n-3)P + 4R + 4F = (n-3+2+1)*P + (4/n) sum_i Q_i + (1/n) sum_i C_i
//                    = n*P + (4/n) sum_i Q_i + (1/n) sum_i C_i
// so, per individual point (dividing through by (n+5)):
//   w(P)   = n / (n+5)
//   w(Q_i) = 4 / (n*(n+5))   for each of the n spokes
//   w(C_i) = 1 / (n*(n+5))   for each of the n corners
// (sums to exactly 1: n/(n+5) + n*[4/(n(n+5))] + n*[1/(n(n+5))] == 1.)
export function vertexLimitWeightsGeneral(n) {
  const w = new Array(2 * n + 1);
  w[0] = n / (n + 5);
  for (let i = 1; i <= n; i++) w[i] = 4 / (n * (n + 5));
  for (let i = n + 1; i <= 2 * n; i++) w[i] = 1 / (n * (n + 5));
  return w;
}

// Numerical cross-check of the closed form above. A(n) is a non-negative,
// row-stochastic matrix (every row sums to 1 — subd.mjs's weighted-average
// rules) with the constant vector as its right eigenvector of eigenvalue 1.
// By Perron-Frobenius that eigenvalue is real, simple and dominant, with a
// strictly positive left eigenvector, so power iteration on A(n)^T from a
// generic seed converges to it. This never reads vertexLimitWeightsGeneral;
// the two are compared in the test file.
export function powerIterationLeftDominant(A, opts = {}) {
  const dim = A.length;
  const maxIter = opts.maxIter ?? 4000;
  const tol = opts.tol ?? 1e-15;
  let w = Array.from({ length: dim }, (_, i) => 1 + i * 0.137); // deterministic, non-symmetric seed
  let lambda = 0;
  for (let iter = 0; iter < maxIter; iter++) {
    const next = new Array(dim).fill(0);
    for (let i = 0; i < dim; i++) {
      const wi = w[i];
      if (wi === 0) continue;
      const row = A[i];
      for (let j = 0; j < dim; j++) next[j] += wi * row[j];
    }
    const norm = Math.sqrt(next.reduce((s, x) => s + x * x, 0));
    if (norm < 1e-300) throw new Error('powerIterationLeftDominant: vector collapsed to zero');
    const normalized = next.map((x) => x / norm);
    let diff = 0;
    for (let j = 0; j < dim; j++) diff = Math.max(diff, Math.abs(normalized[j] - w[j]));
    lambda = norm;
    w = normalized;
    if (iter > 3 && diff < tol) break;
  }
  const sum = w.reduce((s, x) => s + x, 0);
  return { eigenvalue: lambda, eigenvector: w.map((x) => x / sum) };
}

// Semi-sharp hybrid: an exact vertex limit position at a vertex whose
// neighborhood carries an active semi-sharp crease weight (strictly between 0
// and SUPERB_CREASE_LEVEL_SCALE=3, subd.mjs's documented range for an ordinary
// Crease/SoftCrease).
//
// Excluded: a stored crease weight is not always a decaying semi-sharp value.
// subd.mjs's MARKED_CORNER_WEIGHT_FLOOR mechanism stores a weight far above
// the ordinary [0,3] range (TOSUBD's DEFAULT_CORNER_CREASE_WEIGHT, above 100)
// as a permanent "hold this corner at P" marker. Running `ceil(weight)`
// discrete levels for a weight in the hundreds would subdivide the whole cage
// hundreds of times, each level multiplying the face count by 4.
// MAX_SEMISHARP_DECAY_LEVELS (8, a margin above the ordinary [0,3] range)
// refuses instead, and the fallback is already exact: computeVertexPoint's
// marked-corner branch returns the vertex's current position unconditionally
// once sharpness clamps to 1, so the discrete-refinement value equals every
// later level's value, the limit included.
export const MAX_SEMISHARP_DECAY_LEVELS = 8;
//
// The eigenbasis machinery above assumes a smooth vertex. A crease changes
// computeVertexPoint's rule entirely (the crease branch: dart/crease-line/
// corner), which the (2n+1)-point template never models. Rather than a
// crease-aware eigenbasis (DeRose, Kass & Truong's semi-sharp creases and
// their eigenanalysis), this applies subd.mjs's per-level decay
// (`max(0, weight-1)`, dropped once non-positive — see subdivideCatmullClark)
// for as many discrete levels as it takes the crease to decay away from this
// vertex's neighborhood, then hands off to the exact mask once no sharp edge
// is incident to it.
export function semiSharpHybridLimitPosition(cage, vIdx) {
  const ctx0 = buildTopology(cage);
  const edges = ctx0.vertexEdges[vIdx] || [];
  let maxWeight = 0;
  for (const e of edges) {
    const other = e.v0 === vIdx ? e.v1 : e.v0;
    maxWeight = Math.max(maxWeight, creaseWeight(cage, vIdx, other));
  }
  const k = Math.max(0, Math.ceil(maxWeight));
  if (k > MAX_SEMISHARP_DECAY_LEVELS) {
    throw new Error(`semiSharpHybridLimitPosition: vertex ${vIdx}'s own crease weight (${maxWeight}) would need ${k} discrete levels to decay, past MAX_SEMISHARP_DECAY_LEVELS (${MAX_SEMISHARP_DECAY_LEVELS}) — this reads as a PERMANENT marker (e.g. a TOSUBD marked corner), not a decaying semi-sharp crease; a caller should fall back to that vertex's own already-exact discrete-refinement position instead of calling this function on it`);
  }
  let refined = cage;
  for (let i = 0; i < k; i++) refined = subdivideCatmullClark(refined);
  return vertexLimitPosition(refined, vIdx);
}

// Regular-face classification and stencil extraction.
//
// One function says whether a face qualifies; the other reads its 16 points
// out in the order bicubicRegularPatchSurface expects.
//
// What "regular" has to mean: the limit surface over a face is a uniform
// bicubic B-spline patch on its 16-point neighborhood only when nothing in
// that neighborhood perturbs the smooth rules. So each of the face's four
// corners must be interior (a boundary edge is pinned to full sharpness by
// edgeSharpness, so the smooth rules never apply there), of valence exactly 4,
// and its one-ring must be all quads (a triangle or n-gon in the ring leaves
// no 4x4 grid to read). Creases are checked across the whole 3x3 face block,
// not just the center face's edges: a crease on the block's outer boundary
// still changes the subdivision the block's points converge under.
//
// The classifier is conservative on purpose. A face wrongly called irregular
// costs one extra isolation step, which shrinks the region geometrically
// anyway; a face wrongly called regular emits a patch that is not the surface.
function faceAcrossEdge(ctx, faceIdx, v0, v1) {
  const e = ctx.edgeMap.get(edgeKey(v0, v1));
  if (!e || e.faces.length !== 2) return -1; // boundary, or non-manifold
  return e.faces[0] === faceIdx ? e.faces[1] : e.faces[0];
}

export function isRegularFace(cage, faceIdx, ctx = buildTopology(cage)) {
  const face = cage.faces[faceIdx];
  if (!face || face.length !== 4) return false;
  const block = new Set();
  for (const v of face) {
    const edges = ctx.vertexEdges[v];
    const faces = ctx.vertexFaces[v];
    if (edges.length !== 4 || faces.length !== 4) return false; // valence, and (with the interior test below) a complete quad ring
    for (const e of edges) if (e.faces.length !== 2) return false; // boundary or non-manifold vertex
    for (const fi of faces) block.add(fi);
  }
  if (block.size !== 9) return false; // a complete 3x3 block; anything else means the ring folds back on itself
  for (const fi of block) {
    const f = cage.faces[fi];
    if (f.length !== 4) return false;
    for (let c = 0; c < f.length; c++) {
      if (creaseWeight(cage, f[c], f[(c + 1) % f.length]) > 0) return false;
    }
  }
  return true;
}

// The 16 points, in bicubicRegularPatchSurface's own grid[i][j] order
// (i = U, j = V). The face's own four corners land at the center 2x2:
// face[0] -> (1,1), face[1] -> (2,1), face[2] -> (2,2), face[3] -> (1,2),
// so U runs face[0]->face[1] and V runs face[0]->face[3], and the patch's
// parametric corners (0,0)/(1,0)/(1,1)/(0,1) correspond to the face's own
// vertices in cyclic order. Everything else is read off the surrounding
// ring: an edge-neighbor face contributes the two points just outside
// that edge, and the one remaining face at each corner (the diagonal one,
// sharing only that single vertex with the center face) contributes the
// single far corner point.
export function regularFaceStencil(cage, faceIdx, ctx = buildTopology(cage)) {
  if (!isRegularFace(cage, faceIdx, ctx)) {
    throw new Error(`regularFaceStencil: face ${faceIdx} is not regular — classify with isRegularFace before calling`);
  }
  const face = cage.faces[faceIdx];
  const grid = [[null, null, null, null], [null, null, null, null], [null, null, null, null], [null, null, null, null]];
  const P = (i, j, vIdx) => { grid[i][j] = cage.vertices[vIdx]; };
  const [a, b, c, d] = face;
  P(1, 1, a); P(2, 1, b); P(2, 2, c); P(1, 2, d);

  // The neighbor across an edge, read so that each of its two off-edge
  // vertices lands beside the corner it actually adjoins — never assumed
  // from winding, always found by walking that face's own loop.
  const acrossEdge = (p, q, slotP, slotQ) => {
    const nf = faceAcrossEdge(ctx, faceIdx, p, q);
    const f = cage.faces[nf];
    const kp = f.indexOf(p);
    const nextP = f[(kp + 1) % 4], prevP = f[(kp + 3) % 4];
    const outerP = nextP === q ? prevP : nextP; // the neighbor of p in that face that isn't q
    const kq = f.indexOf(q);
    const nextQ = f[(kq + 1) % 4], prevQ = f[(kq + 3) % 4];
    const outerQ = nextQ === p ? prevQ : nextQ;
    grid[slotP[0]][slotP[1]] = cage.vertices[outerP];
    grid[slotQ[0]][slotQ[1]] = cage.vertices[outerQ];
    return nf;
  };
  const nAB = acrossEdge(a, b, [1, 0], [2, 0]); // the low-V edge
  const nBC = acrossEdge(b, c, [3, 1], [3, 2]); // the high-U edge
  const nCD = acrossEdge(c, d, [2, 3], [1, 3]); // the high-V edge
  const nDA = acrossEdge(d, a, [0, 2], [0, 1]); // the low-U edge

  // The diagonal face at a corner is the one incident face that is
  // neither the center face nor either of that corner's two edge
  // neighbors — guaranteed to exist and be unique by the valence-4,
  // 9-face-block conditions isRegularFace already enforced. Its far
  // corner is the vertex opposite the shared one, i.e. two steps around
  // that quad.
  const diagonalAt = (v, nA, nB, slot) => {
    const others = ctx.vertexFaces[v].filter((fi) => fi !== faceIdx && fi !== nA && fi !== nB);
    if (others.length !== 1) throw new Error(`regularFaceStencil: corner ${v} of face ${faceIdx} has no unique diagonal face`);
    const f = cage.faces[others[0]];
    grid[slot[0]][slot[1]] = cage.vertices[f[(f.indexOf(v) + 2) % 4]];
  };
  diagonalAt(a, nAB, nDA, [0, 0]);
  diagonalAt(b, nAB, nBC, [3, 0]);
  diagonalAt(c, nBC, nCD, [3, 3]);
  diagonalAt(d, nCD, nDA, [0, 3]);

  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    if (!grid[i][j]) throw new Error(`regularFaceStencil: stencil slot (${i},${j}) never filled for face ${faceIdx}`);
  }
  return grid;
}

// One regular face becomes one exact bicubic patch, with no fitting,
// tolerance or sampling. Emitted clamped: the same surface, with a knot array
// whose ends bound the valid domain, which the ~ten consumers that derive a
// surface's domain as knots[0]..knots[last] assume. See
// clampedBicubicPatchSurface for the conversion and why it is not knot
// insertion.
export function regularFaceToPatch(cage, faceIdx, ctx = buildTopology(cage)) {
  return clampedBicubicPatchSurface(regularFaceStencil(cage, faceIdx, ctx));
}

// Isolate and emit: turns a whole cage into exact bicubic patches.
//
// A face touching an extraordinary vertex subdivides into four sub-faces, only
// one of which still touches that vertex. Each isolation level emits the three
// that became regular and carries the fourth forward, so the unconverted
// region shrinks by a factor of 4 per level while the patch ring around each
// extraordinary point grows. Every patch returned is the exact limit surface
// over its own face.
//
// Without capping, the shrinking hole around an extraordinary vertex stays
// open. A cap pinned only at that vertex's exact limit position meets its
// neighbors at a point, not along their shared edges, which is a gap a stitch
// cannot close; the cap below instead takes its boundary control rows from
// the adjacent patches (shared exactly, by construction) and pins only the
// remaining interior freedom. Leftover faces are returned, named and measured,
// never covered by a patch that is not the surface.
//
// Coverage is reported in domain terms: an uncovered face at isolation level L
// occupies 4^-L of one original face, so uncoveredFraction is a number a
// caller can act on.
//
// The input cage is never mutated; every level works on a fresh subdivided
// copy.
//
// Capping is opt-in, `{ cap: true }`. Without it the leftovers are reported
// rather than covered. With it, the cap builder emits one patch per leftover
// region and `uncovered` shrinks to the regions that could not be capped,
// each carrying the reason.
//
// isolateRegularFaces is the isolation loop, with what to do about a face
// that came out regular left to the caller: the conversion builds a patch,
// and the pre-flight estimate counts one. The estimate shares the loop so it
// cannot drift from the conversion it predicts.
//
// `onRegular(cage, faceIdx, ctx, level)` is called once per face that
// converged, and whatever it returns is stored in the level's own emitted map
// for the cap builder. Returns the final refined cage, the regions still live
// on it, that last level's emitted map, and the level reached.
function isolateRegularFaces(cage, maxIsolation, onRegular, localShare = ISOLATION_LOCAL_SHARE) {
  let current = { vertices: cage.vertices.map((v) => v.slice()), faces: cage.faces.map((f) => f.slice()), creases: { ...(cage.creases || {}) } };
  let live = current.faces.map((_, fi) => fi);
  let level = 0;
  // The final level's own emitted patches, by face index — the cap builder
  // reads its boundary rows straight out of these, so a cap and its
  // neighbor carry the identical numbers rather than two computations of
  // the same number.
  let emittedAtLevel = new Map();

  for (;;) {
    const ctx = buildTopology(current);
    const stillLive = [];
    emittedAtLevel = new Map();
    for (const fi of live) {
      if (isRegularFace(current, fi, ctx)) emittedAtLevel.set(fi, onRegular(current, fi, ctx, level));
      else stillLive.push(fi);
    }
    live = stillLive;
    if (!live.length || level >= maxIsolation) { current = { ...current, __ctx: undefined }; break; }

    // Isolation is local, so the refinement is too. Every face still live
    // after the first pass touches an extraordinary vertex or a crease, and
    // on any cage with an interior the count stops falling almost at once:
    // a 24x24x6 box cage is 3456 faces, of which 24 are still live after
    // level 0 and 96 at every level after. Refining the whole cage to serve
    // them takes it to 221,184 faces by level 3, 99.96% of it pad, and that
    // is 93% of this function's cost. Cutting the working cage down to the
    // live faces plus a margin of rings does the identical arithmetic on the
    // part that is read.
    //
    // The pad is the one localNeighbourhoodCage describes, spent one ring per
    // remaining refinement: the sub-cage's cut edge is a naked boundary from
    // its own perspective, so the boundary rules fire there and that wrong
    // value spreads one further ring inward per level. Arriving at the final
    // level with LOCAL_PROBE_RINGS rings still correct is what the cap's
    // dyadic probe needs, and ISOLATION_PAD_MARGIN is margin on top of that.
    // The test file compares every emitted patch against the whole-cage
    // answer bit for bit.
    {
      const rings = LOCAL_PROBE_RINGS + ISOLATION_PAD_MARGIN + (maxIsolation - level);
      const local = localNeighbourhoodCage(current, ctx, live, rings, current.faces.length * localShare);
      if (local) {
        current = local.sub;
        live = live.map((fi) => local.faceMap.get(fi));
      }
    }

    // subdivideCatmullClark pushes each face's children in corner order,
    // so a face's own children occupy a contiguous run starting at the
    // running sum of every earlier face's corner count. That is the only
    // thing needed to carry "which regions are still unconverted"
    // forward across a refinement — no separate bookkeeping structure.
    const childBase = new Array(current.faces.length);
    let running = 0;
    current.faces.forEach((f, fi) => { childBase[fi] = running; running += f.length; });
    const nextLive = [];
    for (const fi of live) {
      const n = current.faces[fi].length;
      for (let c = 0; c < n; c++) nextLive.push(childBase[fi] + c);
    }
    current = subdivideCatmullClark(current);
    live = nextLive;
    level++;
  }
  return { current, live, emittedAtLevel, level };
}

export function subdToPatches(cage, opts = {}) {
  const maxIsolation = opts.maxIsolation ?? 3;
  const patches = [];
  const originalFaceCount = cage.faces.length;
  const iso = isolateRegularFaces(cage, maxIsolation, (c, fi, ctx, level) => {
    const srf = regularFaceToPatch(c, fi, ctx);
    patches.push({ srf, level, faceIndex: fi, kind: 'regular' });
    return srf;
  }, opts.localShare ?? ISOLATION_LOCAL_SHARE);
  let current = iso.current;
  let live = iso.live;
  const emittedAtLevel = iso.emittedAtLevel;
  const level = iso.level;

  const ctxFinal = buildTopology(current);
  const describe = (fi) => ({
    level,
    faceIndex: fi,
    // The extraordinary vertices responsible, each with its exact limit
    // position — what a cap builder needs first, and the one point known
    // exactly when everything around it is not.
    extraordinary: current.faces[fi]
      .filter((v) => ctxFinal.vertexEdges[v].length !== 4 || ctxFinal.vertexFaces[v].length !== 4)
      .map((v) => ({ vertex: v, limitPosition: vertexLimitPosition(current, v, ctxFinal) })),
  });

  let caps = [];
  let uncovered;
  if (opts.cap) {
    const built = capStarRegions(current, ctxFinal, live, emittedAtLevel);
    caps = built.caps;
    for (const c of caps) patches.push({ srf: c.srf, level, faceIndex: c.faceIndex, kind: 'cap', star: c.star });
    // A region that could not be capped stays reported, and says why —
    // never silently dropped and never covered by a patch that is not the
    // surface.
    uncovered = built.refused.map((r) => ({ ...describe(r.faceIndex), reason: r.reason }));
  } else {
    uncovered = live.map((fi) => describe(fi));
  }

  return {
    patches,
    caps,
    uncovered,
    levelsUsed: level,
    refinedCage: current,
    // Share of the original parameter domain left unconverted.
    uncoveredFraction: uncovered.reduce((sum, u) => sum + Math.pow(4, -u.level), 0) / originalFaceCount,
  };
}

// What the conversion will cost, without paying it. The number a caller
// decides on is how many surfaces come out, and that is not a property of the
// cage's face count: a torus cage of 576 faces is regular everywhere and
// converts to 576 patches at level 0, while a 6-face box cage reaches 168
// because every one of its corners is a star point and the region around
// each has to be refined three times to isolate it. Nothing readable off the
// cage up front separates those two cases, so this runs the same isolation
// walk subdToPatches runs and skips only the part that builds geometry.
//
// Cost: the walk without patch construction and without the cap's two
// refinements — roughly half of the conversion on a small cage and much less
// on a large one, where building and meshing the patches dominates.
//
// The count is exact, caps included: whether a leftover region can be capped
// is decided by topology alone, so the cap planner runs too.
export function estimateSubdToPatches(cage, opts = {}) {
  const maxIsolation = opts.maxIsolation ?? 3;
  let regular = 0;
  const iso = isolateRegularFaces(cage, maxIsolation, () => { regular++; return null; }, opts.localShare ?? ISOLATION_LOCAL_SHARE);
  let capped = 0, open = iso.live.length;
  if (opts.cap && iso.live.length) {
    const plan = planStarCaps(iso.current, buildTopology(iso.current), iso.live, iso.emittedAtLevel);
    capped = plan.plans.length;
    open = plan.refused.length;
  }
  return {
    regular,
    caps: capped,
    patches: regular + capped,
    uncovered: open,
    levelsUsed: iso.level,
    uncoveredFraction: open * Math.pow(4, -iso.level) / cage.faces.length,
  };
}

// The cap.
//
// Isolation leaves, around every extraordinary vertex E of valence n, a ring
// of n quad faces that never became regular. Each has E at one corner and
// three ordinary valence-4 corners, and exactly two of its four edges border
// a face already emitted as an exact patch; the other two run from E out to a
// neighboring corner and are shared with the next face of the same ring. So a
// cap's boundary is almost entirely decided by surfaces that are exactly the
// limit surface, and the only freedom is near E.
//
// A cap pinned only at E's exact limit position would meet its neighbors at
// a point and nowhere else, which a B-rep stitch cannot close.
//
// The construction is Loop & Schaefer's ACC (Approximating Catmull-Clark
// Subdivision Surfaces with Bicubic Patches, ACM TOG 27(1), 2008), the
// standard answer to this problem and the one used in hardware tessellation
// pipelines. In Bezier form, per quad face:
//
//   corner  b00   the vertex's own exact limit position — the Halstead/
//                 Kass/DeRose mask, which is vertexLimitPosition above.
//   interior b11  (n*v + 2*(ePrev + eNext) + diag) / (n + 5), where v is
//                 the corner, ePrev/eNext its two neighbors in that face,
//                 diag the face's fourth vertex, and n = valence(v).
//   edge    b10   the midpoint of the two adjacent faces' own interior
//                 points at the shared corner.
//
// Those three masks are re-derived here from the paper's stated geometric
// relationship — a corner point is the centroid of the interior points around
// it, and an edge point the midpoint of the two beside it. Writing the
// interior mask as (a*v + b*(ePrev+eNext) + c*diag)/(a+2b+c) and demanding
// that the centroid over the n faces at v reproduce
// (n^2*v + 4*sum(e) + sum(diag))/(n(n+5)) forces a:b:c = n^2 : 2n : n, i.e.
// the form above, with the denominator falling out as n(n+5) — the limit
// mask's own. At n = 4 all three reduce to the uniform B-spline-to-Bezier
// knot-insertion masks: the same numbers clampedBicubicPatchSurface produces
// for a regular face, reached a different way.
//
// ACC's continuity claim is that patches meet smoothly except along an edge
// containing an extraordinary vertex, where they are only C0. A cap's two
// outer edges do not contain E, so the join between a cap and the exact
// regular region is tangent-continuous: the normal deviation across it is
// 0.0000 degrees on every fixture tried, including a valence-5 extruded cage.
// The tangent break is confined to the star edges, cap against cap.
//
// One refinement on top of ACC. Of the sixteen control points, thirteen are
// pinned: the two outer rows are the neighbors' own rows copied element for
// element, and the two rows immediately inside them carry the cross-boundary
// derivative that makes that join exact. Three are touched by no continuity
// condition — the two star-edge control points adjacent to E, and the
// interior point nearest E. Those three are solved to interpolate the true
// limit surface at three points near the star: one quarter of the way along
// each star edge, and the (1/4,1/4) corner of the face. The true limit at a
// dyadic parameter is a twice-refined vertex's limit position under the same
// mask used above. This cuts the worst deviation from the true limit surface
// by about 3.9x and the tangent break across the star edges by about 1.9x,
// and leaves every exactness property of the ACC base untouched.
//
// What is exact and what is not:
//   Exact  the star corner is vertexLimitPosition(E), bit for bit.
//   Exact  the two outer boundary rows are the neighboring patches' own
//          arrays, so the shared edge curve is the same curve.
//   Exact  the two star-edge rows are computed once per edge and read by
//          both caps that share it, so cap meets cap bit for bit.
//   Exact  tangent continuity across a cap's two outer edges.
//   Not    the interior. A single bicubic cannot be the Catmull-Clark limit
//          over a face touching an extraordinary vertex — that surface is
//          an infinite nest of patches, not one. The deviation is measured
//          in the test file, and it does not vanish with isolation level:
//          the region is self-similar under refinement, so the absolute
//          error shrinks (about 2.4x per level) while the error relative to
//          the cap's own size holds near 0.4%.
//   Not    tangent continuity across the star edges; it shrinks with
//          isolation level, which is the knob.
//
// Limit on bit-exactness at the junction corners. A corner where a cap meets
// two regular patches can carry only one value, and the two regular patches
// agree on it to about 1e-15, not to the last bit, because each computes the
// same limit position through a different order of the same arithmetic. No
// two adjacent regular patches in this output are bit-identical along their
// shared row either. So each cap is bit-identical with one of its two regular
// neighbors along the whole shared row, and with the other on the two
// interior control points, differing only at the shared corner and only by
// the amount those two neighbors already differ. Welding the whole emitted
// set to per-edge and per-vertex canonical values would close that gap; it
// is not done because it would perturb patches that are exactly the limit
// surface to fix a disagreement below any consumer's tolerance.
const CAP_QUARTER_BASIS = [27 / 64, 27 / 64, 9 / 64, 1 / 64]; // cubic Bernstein at t = 1/4

function linComb(terms) {
  let out = [0, 0, 0];
  for (const [p, w] of terms) out = add(out, scale(p, w));
  return out;
}

// ACC's interior Bezier point of quad `faceIdx` at its corner `vIdx`.
// Depends only on that face's own four vertices and the corner's valence,
// which is what makes two faces agree about a shared edge point.
export function accInteriorPoint(cage, ctx, faceIdx, vIdx) {
  const f = cage.faces[faceIdx];
  const k = f.indexOf(vIdx);
  if (f.length !== 4 || k < 0) throw new Error(`accInteriorPoint: face ${faceIdx} is not a quad containing vertex ${vIdx}`);
  const n = ctx.vertexEdges[vIdx].length;
  return scale(linComb([
    [cage.vertices[vIdx], n],
    [cage.vertices[f[(k + 1) % 4]], 2],
    [cage.vertices[f[(k + 3) % 4]], 2],
    [cage.vertices[f[(k + 2) % 4]], 1],
  ]), 1 / (n + 5));
}

// The four control points of a clamped bicubic patch along the face edge
// v0 -> v1, in that direction. `faceVerts` must be the vertex loop the
// patch was built from, since that is what fixes which net row is which
// edge: face[0] is at (0,0), face[1] at (1,0), face[2] at (1,1), face[3]
// at (0,1), exactly regularFaceStencil's own convention.
export function patchBoundaryRow(srf, faceVerts, v0, v1) {
  const n = faceVerts.length;
  const B = srf.ctrlNet;
  const rows = [
    [B[0][0], B[1][0], B[2][0], B[3][0]],
    [B[3][0], B[3][1], B[3][2], B[3][3]],
    [B[3][3], B[2][3], B[1][3], B[0][3]],
    [B[0][3], B[0][2], B[0][1], B[0][0]],
  ];
  for (let e = 0; e < n; e++) {
    if (faceVerts[e] === v0 && faceVerts[(e + 1) % n] === v1) return rows[e].slice();
    if (faceVerts[e] === v1 && faceVerts[(e + 1) % n] === v0) return rows[e].slice().reverse();
  }
  throw new Error(`patchBoundaryRow: ${v0}-${v1} is not an edge of the given face loop`);
}

// The faces within `rings` vertex-steps of `seedFaces`, lifted out as a cage
// in their own numbering. Keeps the star correction's two refinements local:
// refining a whole cage twice to read a few dozen points near its
// extraordinary vertices costs an order of magnitude more than everything
// else in this file put together.
//
// The pad. Two things reach outward: subdivision treats the sub-cage's cut
// edge as a naked boundary and applies the boundary rules there, and that
// wrong value spreads one further ring inward per level; and a limit position
// two levels down reads through roughly three rings of the cage it started
// from. A pad of 2 reproduces the whole-cage answer bit-identically on every
// cage tried, so 5 is margin over that floor. The test file checks the
// bit-identity; if it fails, this number is the thing to raise.
const LOCAL_PROBE_RINGS = 5;

// Extra rings the isolation loop carries on top of LOCAL_PROBE_RINGS, so the
// cage it hands the cap probe has the probe's own margin intact rather than
// exactly spent. One ring is one refinement's worth of inward contamination.
const ISOLATION_PAD_MARGIN = 1;

// The share of the cage a localized neighborhood may reach before localizing
// stops being worth its own ring walk. On an open plane cage, whose whole
// naked border stays live at every level, the padded neighborhood is the
// cage, and building it costs `rings` passes over every face to save nothing.
const ISOLATION_LOCAL_SHARE = 0.6;
// `{ localShare: 0 }` turns localizing off entirely (no neighborhood fits a
// budget of zero faces), which keeps the whole-cage computation reachable as
// the reference this shortcut has to agree with, bit for bit.

// `maxFaces` abandons the walk the moment the neighborhood stops being a
// neighborhood. A caller that wants a sub-cage because it is smaller (the
// isolation loop) gains nothing from one that covers the whole cage; it gets
// null instead and carries on with the cage it has. The probe passes no
// budget and always gets its cage.
function localNeighbourhoodCage(cage, ctx, seedFaces, rings, maxFaces = Infinity) {
  let faceSet = new Set(seedFaces);
  if (faceSet.size > maxFaces) return null;
  for (let r = 0; r < rings; r++) {
    const next = new Set(faceSet);
    for (const fi of faceSet) for (const v of cage.faces[fi]) for (const nf of ctx.vertexFaces[v]) next.add(nf);
    if (next.size > maxFaces) return null;
    if (next.size === faceSet.size) break;
    faceSet = next;
  }
  const vertexMap = new Map();
  const faceMap = new Map();
  const vertices = [];
  const faces = [];
  for (const fi of [...faceSet].sort((a, b) => a - b)) {
    faceMap.set(fi, faces.length);
    faces.push(cage.faces[fi].map((v) => {
      if (!vertexMap.has(v)) { vertexMap.set(v, vertices.length); vertices.push(cage.vertices[v].slice()); }
      return vertexMap.get(v);
    }));
  }
  const creases = {};
  for (const [key, w] of Object.entries(cage.creases || {})) {
    const [a, b] = key.split('_').map(Number);
    if (vertexMap.has(a) && vertexMap.has(b)) creases[edgeKey(vertexMap.get(a), vertexMap.get(b))] = w;
  }
  return { sub: { vertices, faces, creases }, vertexMap, faceMap };
}

// The exact limit surface at the dyadic parameters a cap needs, read out of
// two real refinements rather than evaluated: subdivideCatmullClark's own
// output indexing (vertex points keep their index, edge points next in
// edgeMap order, face points last) says exactly which refined vertex sits
// at which parameter, and vertexLimitPosition is exact at any vertex.
function dyadicLimitProbe(outerCage, outerCtx, seedFaces) {
  const { sub, vertexMap, faceMap } = localNeighbourhoodCage(outerCage, outerCtx, seedFaces, LOCAL_PROBE_RINGS);
  const cage = sub;
  const ctx = buildTopology(cage);
  const edgeOrdinal = new Map();
  { let i = cage.vertices.length; for (const key of ctx.edgeMap.keys()) edgeOrdinal.set(key, i++); }
  const childBase = new Array(cage.faces.length);
  { let running = 0; cage.faces.forEach((f, fi) => { childBase[fi] = running; running += f.length; }); }

  const r1 = subdivideCatmullClark(cage);
  const ctx1 = buildTopology(r1);
  const edgeOrdinal1 = new Map();
  { let i = r1.vertices.length; for (const key of ctx1.edgeMap.keys()) edgeOrdinal1.set(key, i++); }
  const faceBase1 = r1.vertices.length + ctx1.edgeMap.size;
  const r2 = subdivideCatmullClark(r1);
  const ctx2 = buildTopology(r2);

  return {
    // The limit surface one quarter of the way along edge v0 -> v1, from v0.
    // Refining once splits that edge at its midpoint, so the quarter point is
    // the midpoint of the first half — an edge point one level further down.
    alongEdge: (outerV0, outerV1) => {
      const v0 = vertexMap.get(outerV0), v1 = vertexMap.get(outerV1);
      const mid = edgeOrdinal.get(edgeKey(v0, v1));
      return vertexLimitPosition(r2, edgeOrdinal1.get(edgeKey(v0, mid)), ctx2);
    },
    // The limit surface at the center of the quarter of face `faceIdx` that
    // sits at corner `cornerSlot` — the face point of that child face.
    quarterCentre: (outerFace, cornerSlot) => vertexLimitPosition(r2, faceBase1 + childBase[faceMap.get(outerFace)] + cornerSlot, ctx2),
  };
}

// Which leftover regions can be capped, decided before any geometry is
// built. `emitted` maps a face index at this level to the patch already
// emitted for it. Separate from the build because the pre-flight estimate
// needs this answer without paying for the caps: on an open cage every
// naked-boundary region refuses (508 of 1564 regions on a 16x16 plane), so an
// estimate that assumed every region cappable would be half again too large.
export function planStarCaps(cage, ctx, live, emitted) {
  const liveSet = new Set(live);
  const refused = [];
  const plans = [];
  for (const fi of live) {
    const f = cage.faces[fi];
    const reject = (reason) => { refused.push({ faceIndex: fi, reason }); };
    if (f.length !== 4) { reject(`face has ${f.length} corners, not 4`); continue; }
    const ex = f.filter((v) => ctx.vertexEdges[v].length !== 4 || ctx.vertexFaces[v].length !== 4);
    if (ex.length !== 1) { reject(`face touches ${ex.length} extraordinary vertices, not exactly 1 — raise maxIsolation`); continue; }
    const rot = f.indexOf(ex[0]);
    const [E, a, b, c] = [0, 1, 2, 3].map((k) => f[(rot + k) % 4]);
    const across = (p, q) => {
      const e = ctx.edgeMap.get(edgeKey(p, q));
      if (!e || e.faces.length !== 2) return -1;
      return e.faces[0] === fi ? e.faces[1] : e.faces[0];
    };
    const nAB = across(a, b), nBC = across(b, c), nEA = across(E, a), nCE = across(c, E);
    if (!emitted.has(nAB) || !emitted.has(nBC)) { reject('the two edges away from the extraordinary vertex are not both bordered by an emitted patch'); continue; }
    if (!liveSet.has(nEA) || !liveSet.has(nCE)) { reject('the two edges at the extraordinary vertex are not both shared with another uncovered region'); continue; }
    plans.push({ fi, E, a, b, c, nAB, nBC, nEA, nCE, rot });
  }
  return { plans, refused };
}

// One cap per cappable leftover region; the rest are returned as refusals
// naming what the region failed.
export function capStarRegions(cage, ctx, live, emitted) {
  const { plans, refused } = planStarCaps(cage, ctx, live, emitted);

  // Boundary rows first, straight out of the neighbors' own nets.
  for (const p of plans) {
    p.rowAB = patchBoundaryRow(emitted.get(p.nAB), cage.faces[p.nAB], p.a, p.b);
    p.rowBC = patchBoundaryRow(emitted.get(p.nBC), cage.faces[p.nBC], p.b, p.c);
  }

  // Corners. E is the exact limit position, shared by every cap of its ring.
  // The rest are claimed from a neighbor's row: a and b from the a-b row,
  // because that pairing is the one that can make a whole row bit-identical
  // — b belongs to this cap alone, and a is claimed by no other cap (in the
  // next cap round the ring, a is the far corner c, which claims nothing).
  const corner = new Map();
  const claim = (v, val) => { if (!corner.has(v)) corner.set(v, val); };
  for (const p of plans) claim(p.E, vertexLimitPosition(cage, p.E, ctx));
  for (const p of plans) { claim(p.a, p.rowAB[0]); claim(p.b, p.rowAB[3]); }
  for (const p of plans) claim(p.c, p.rowBC[3]); // only reached when a ring is partly refused

  const interior = new Map();
  const accInt = (fi, v) => {
    const k = `${fi}:${v}`;
    if (!interior.has(k)) interior.set(k, accInteriorPoint(cage, ctx, fi, v));
    return interior.get(k);
  };
  // ACC's edge point at the FAR end of a star edge, once per edge so both
  // caps that share it read the identical numbers rather than two
  // computations of the same number. The other end, the one next to the star,
  // is ACC's too until the refinement below replaces it, so it is never
  // computed here at all — the far end is pinned by the tangency condition on
  // the neighboring outer edge and the near end is not.
  const starFar = new Map();
  const starEdgeFar = (E, far, f0, f1) => {
    const key = edgeKey(E, far);
    if (!starFar.has(key)) {
      const lo = Math.min(f0, f1), hi = Math.max(f0, f1);
      starFar.set(key, linComb([[accInt(lo, far), 0.5], [accInt(hi, far), 0.5]]));
    }
    return starFar.get(key);
  };
  for (const p of plans) { starEdgeFar(p.E, p.a, p.fi, p.nEA); starEdgeFar(p.E, p.c, p.fi, p.nCE); }

  // The three unconstrained control points, pulled onto the true limit
  // surface. Only built when there is something to cap, since it costs two
  // real refinements of a neighborhood.
  const probe = plans.length ? dyadicLimitProbe(cage, ctx, plans.map((p) => p.fi)) : null;
  const solvedStar = new Map();
  const solveStarEdge = (E, far) => {
    const key = edgeKey(E, far);
    if (!solvedStar.has(key)) {
      const [w0, w1, w2, w3] = CAP_QUARTER_BASIS;
      const target = probe.alongEdge(E, far);
      solvedStar.set(key, scale(linComb([
        [target, 1], [corner.get(E), -w0], [starFar.get(key), -w2], [corner.get(far), -w3],
      ]), 1 / w1));
    }
    return solvedStar.get(key);
  };

  const caps = [];
  for (const p of plans) {
    const { fi, E, a, b, c } = p;
    // Grid convention matches regularFaceStencil: index 0 is U, index 1 is
    // V, and the face's own loop [E, a, b, c] lands at (0,0) (1,0) (1,1)
    // (0,1). So the star sits at the patch's own (0,0) corner.
    const g = [[null, null, null, null], [null, null, null, null], [null, null, null, null], [null, null, null, null]];
    g[0][0] = corner.get(E); g[3][0] = corner.get(a); g[3][3] = corner.get(b); g[0][3] = corner.get(c);
    g[3][1] = p.rowAB[1]; g[3][2] = p.rowAB[2];
    g[2][3] = p.rowBC[1]; g[1][3] = p.rowBC[2];
    g[2][0] = starEdgeFar(E, a, fi, p.nEA);
    g[0][2] = starEdgeFar(E, c, fi, p.nCE);
    // Three of the four interior points are ACC's own; the fourth, b11, is
    // the one the refinement below replaces, so it is never given ACC's
    // value at all.
    g[2][1] = accInt(fi, a); g[2][2] = accInt(fi, b); g[1][2] = accInt(fi, c);
    g[1][0] = solveStarEdge(E, a);
    g[0][1] = solveStarEdge(E, c);
    // The interior point nearest the star, pulled onto the limit surface at
    // the (1/4,1/4) corner of the face. Every other control point is already
    // final, so this is one unknown against one condition.
    {
      const w = CAP_QUARTER_BASIS;
      const terms = [[probe.quarterCentre(fi, p.rot), 1]];
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        if (i === 1 && j === 1) continue;
        terms.push([g[i][j], -w[i] * w[j]]);
      }
      g[1][1] = scale(linComb(terms), 1 / (w[1] * w[1]));
    }
    caps.push({
      faceIndex: fi,
      star: E,
      starLimitPosition: corner.get(E),
      vertexLoop: [E, a, b, c],
      srf: {
        degU: 3, degV: 3, knotsU: CLAMPED_PATCH_KNOTS, knotsV: CLAMPED_PATCH_KNOTS,
        ctrlNet: g.map((row) => row.map((q) => [q[0], q[1], q[2], 1])),
      },
    });
  }
  return { caps, refused };
}
