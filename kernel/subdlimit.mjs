// EXACT Catmull-Clark limit-surface evaluation
// — a sibling of subd.mjs, not a rewrite of it: subd.mjs
// produces a FINER CAGE (one discrete refinement step); this module
// evaluates the true infinite-refinement LIMIT exactly, at an arbitrary
// (u,v), with no loop. Two genuinely different concerns kept in two files
// on purpose, per this project's own module-boundary discipline.
//
// STEP 1 — THE REGULAR-PATCH FAST PATH. Away from any extraordinary
// vertex (valence != 4) or crease, a Catmull-Clark limit surface converges
// EXACTLY to an ordinary uniform bicubic B-spline patch over the 16
// control points surrounding a face (the standard, textbook Catmull-Clark
// convergence identity — not an approximation). This needs zero new
// evaluation math: it reuses kernel/surface.mjs's own already-proven
// surfacePointAndPartials (the same rational tensor-product evaluator
// already shipped for ordinary NURBS surfaces) against a SYNTHETIC,
// non-rational, uniform-knot patch built from the 16 points.
//
// The knot vector [-3,-2,-1,0,1,2,3,4] (a plain, unclamped, evenly-spaced
// integer sequence — 4 control points, degree 3, so knots.length ==
// n+p+1 == 4+3+1 == 8) is deliberately NOT the clamped [0,0,0,0,1,1,1,1]
// a 4-point Bezier patch would use: the 16 stencil points are the
// CONTROL POINTS OF A UNIFORM B-SPLINE (the shape that repeated regular
// subdivision converges to), not Bezier control points, and a clamped
// knot vector would evaluate a genuinely different (wrong) patch through
// the same 16 points. With exactly 4 control points and degree 3 there is
// only ONE valid parametric span regardless of clamping (n - p == 1
// span); the shift to [-3..4] puts that one valid span, [knots[p],
// knots[n]] == [knots[3], knots[4]], at exactly [0,1] — the natural,
// convenient domain for a caller — while the SURROUNDING knot values
// (which basisFuns/dersBasisFuns read from a window centered on the
// span, not just its own two endpoints) still shape the basis functions
// as genuinely uniform, not clamped.
//
// STENCIL LAYOUT: grid[i][j] for i,j in 0..3 (U-direction first index,
// matching kernel/surface.mjs's own ctrlNet[i][j] convention). The face
// being evaluated sits at the CENTER 2x2 of the grid (indices (1,1),
// (1,2), (2,1), (2,2) are the face's own 4 corners); the remaining 12
// points are that face's own one-ring of surrounding vertices. This
// module does not yet care how a caller derives that layout from a real
// SuperB cage — that mapping is later app-layer wiring (STEP 6); this
// step only needs the grid-to-patch evaluator to be correct and reused,
// not re-derived, since every later step depends on it.

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

// CLAMPED FORM OF THE SAME PATCH — identical surface, representation a
// downstream consumer can actually trust.
//
// THE PROBLEM, stated more precisely than it first was. An unclamped patch
// evaluates correctly through surfacePoint when handed a (u,v) inside its own
// true span; that was checked directly, and the earlier claim that surfacePoint
// "returns null for such a patch" does not hold — the null readings that
// produced it came from a malformed cage, not from the knot vector. The real
// defect is narrower and worse: the patch's knot ARRAY spans [-3,4] while its
// valid domain is [0,1], and roughly ten consumers across this kernel and the
// app tier derive a surface's domain as `knots[0] .. knots[last]`. That is
// exactly right for a clamped surface — which, until these patches, was every
// surface this app had — and silently wrong by a factor of seven here.
// extractBorderCurves is the plainest case: it asks for the border at
// knots[0], four knots outside the span, and hands back a curve measured 131
// units from the real patch edge. Clamping fixes all of those consumers at
// once, which is a better trade than auditing a dozen call sites.
//
// WHY NOT KNOT INSERTION, which is the obvious route and the one the plan
// named. insertKnot/extractSubCurve are exact for the CLAMPED curves they were
// written for, and are NOT exact here: isolating a sub-range needs the boundary
// knot raised to multiplicity degree+1, and that last step duplicates a control
// point without blending. On a clamped curve that is correct, because at
// multiplicity `degree` the duplicated point already lies ON the curve. On an
// unclamped one it does not, and the result is a different curve — measured at
// 0.37 units on a plain uniform cubic spanning about 3. decomposeToBezier
// assumes clamped input too. See kernel/knots.mjs's own note.
//
// WHAT IS USED INSTEAD is not a workaround but the exact, standard conversion
// for precisely the shape this module emits: a UNIFORM cubic B-spline segment
// has a closed-form Bezier equivalent, and REGULAR_PATCH_KNOTS is uniform by
// construction. For the span governed by P0..P3:
//     B0 = (P0 + 4*P1 + P2) / 6      B1 = (2*P1 + P2) / 3
//     B2 = (P1 + 2*P2) / 3           B3 = (P1 + 4*P2 + P3) / 6
// Verified against the curve's own sampled points at 4.5e-16 — machine
// precision, not a tolerance. Being linear, it applies to a tensor grid one
// direction at a time in either order.
//
// SCOPED HONESTLY: this is correct for THIS knot vector and degree, not for an
// arbitrary unclamped surface, and it is deliberately not named as though it
// were. Nothing else in this kernel produces an unclamped surface. The patches
// are also always NON-RATIONAL (bicubicRegularPatchSurface's own stated scope —
// a plain average of real points), so the weight rides through as 1 rather than
// being blended in homogeneous space.
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
  // U pass — each control COLUMN (fixed j, varying i) is a uniform cubic span.
  const cols = [];
  for (let j = 0; j < 4; j++) cols.push(uniformCubicSpanToBezier(src[0][j], src[1][j], src[2][j], src[3][j]));
  const mid = [];
  for (let i = 0; i < 4; i++) mid.push(cols.map((c) => c[i]));
  // V pass — each ROW of that result is a uniform cubic span in the other direction.
  const ctrlNet = mid.map((row) => uniformCubicSpanToBezier(row[0], row[1], row[2], row[3]).map((p) => [p[0], p[1], p[2], 1]));
  return { degU: 3, degV: 3, knotsU: CLAMPED_PATCH_KNOTS, knotsV: CLAMPED_PATCH_KNOTS, ctrlNet };
}

// u, v in [0,1] — (0,0)/(1,0)/(0,1)/(1,1) are the patch's own four
// parametric corners, each converging to one of the face's own 4
// vertices' true Catmull-Clark limit positions (NOT any single control
// point directly — a regular vertex's limit position is itself a
// weighted average over its own one-ring, exactly what evaluating this
// patch at a corner computes).
export function regularPatchPointAndPartials(grid4x4, u, v) {
  return surfacePointAndPartials(bicubicRegularPatchSurface(grid4x4), u, v);
}

// STEP 2 — THE HALSTEAD/KASS/DEROSE (1993) VERTEX-LIMIT-POSITION MASK,
// standalone independent ground truth for step 4's eigenbasis evaluator
// (built BEFORE that evaluator exists, so it can never be checked
// tautologically against itself) — reuses ONLY subd.mjs's own already-
// proven buildTopology/computeFacePoint (real, shipped, tested code),
// never anything from this module's own step-1 regular-patch path or a
// future eigen-decomposition.
//
// DERIVATION (worked out here, not transcribed from memory — the exact
// concern this project's own standing discipline names for Stam's own
// closed-form eigenbasis polynomials applies equally to trusting an
// uncertain recollection of ANY published subdivision-limit formula):
// consider a fully rotationally-symmetric neighborhood of an
// extraordinary vertex P0 of valence n — n "spoke" neighbor vertices Qi
// and, for each of the n incident quad faces, one "far" (diagonal)
// corner vertex Ri. Define the three rotationally-invariant (DC-mode)
// scalars p = P0, q = avg(Qi), r = avg(Ri). One Catmull-Clark
// subdivision step (subd.mjs's own already-proven face/edge/vertex-point
// rules, restricted to this fully symmetric case) gives:
//   new face-centroid average  = (p + 2q + r) / 4
//   new edge-point average     = (3p + 4q + r) / 8
//   new vertex point (smoothVertexRule, n valid for any n)
//                               = [(4n-7)p + 6q + r] / (4n)
// — a genuine 3x3 LINEAR map (p,q,r) -> (p',q',r'), independent of n
// only through its own coefficients. Because every row of this matrix
// sums to 1 (a real, checked invariant — each new value really is a
// weighted AVERAGE of old ones), it is row-stochastic: repeated
// application converges every one of p, q, r to the SAME fixed value,
// the matrix's own dominant (eigenvalue-1) fixed point — this IS the
// vertex's true limit position, by construction (as the neighborhood
// shrinks to a point under infinite subdivision, its center, spoke-
// average, and corner-average must all coincide there). Solving
// (M^T - I) v = 0 for this 3x3 system by hand gives the eigenvector
// v = (n, 4, 1) (verified directly in this module's own test file,
// which re-derives and checks every step of this system, not just its
// final answer) — normalized (sum = n+5), the limit position is:
//   L = (n*p + 4*q + r) / (n+5)
// Converting q, r into subd.mjs's OWN R (average adjacent EDGE MIDPOINT,
// not neighbor vertex) and F (average adjacent FACE CENTROID) —
// R = (p+q)/2 so q = 2R-p; F = (p+2q+r)/4 so r = p+4F-4R — and
// substituting gives the form actually used below:
//   L = [(n-3)*P + 4*R + 4*F] / (n+5)
// CROSS-VALIDATED (not just algebraically self-consistent) against the
// widely-published REGULAR-vertex (n=4) mask, stated in terms of the
// vertex V, its 4 edge-neighbor vertices Qi, and its 4 face-diagonal
// corners Ri: "(16*V + 4*sum(Qi) + sum(Ri)) / 36" — substituting
// sum(Qi)=4q, sum(Ri)=4r into that published formula gives
// (16p+16q+4r)/36, algebraically IDENTICAL to this module's own
// (n*p+4*q+r)/(n+5) at n=4 -> (4p+4q+r)/9 -> (16p+16q+4r)/36. Same
// answer, reached two independent ways.
export function vertexLimitMaskFromRF(P, R, F, n) {
  const num = add(add(scale(P, n - 3), scale(R, 4)), scale(F, 4));
  return scale(num, 1 / (n + 5));
}

// Cage-facing wrapper — R/F computed the IDENTICAL way
// computeVertexPoint's own smoothVertexRule already does (reused
// directly, not re-derived a second, possibly-inconsistent way), so this
// can never silently disagree with what a real subdivision step itself
// would compute as this vertex's own F/R inputs.
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

// STEP 3 — THE SUBDIVISION-MATRIX BUILDER, A(n), for Stam's eigenbasis
// method (the general, non-symmetric-reduced case step 2 deliberately
// didn't need). A(n) is a (2n+1)x(2n+1) linear map over the STANDARD
// local neighborhood template of an extraordinary vertex of valence n —
// index 0 the vertex itself, indices 1..n its n "spoke" edge-neighbors,
// indices n+1..2n the n "far" diagonal corner of each incident quad face
// (face i = [0, spoke_i, far_i, spoke_{i+1}]) — mapping this OLD (2n+1)
// neighborhood to the NEW one, one subdivision step later, around the
// SAME vertex (still valence n, still the same local topology — a real,
// checked structural fact of Catmull-Clark refinement, not assumed).
//
// Built the way this doc's own build order specifies: run subd.mjs's
// own already-proven, already-tested subdivideCatmullClark on REAL UNIT
// BASIS VECTORS over this template (`standardNeighborhoodCage`), one
// column of A(n) at a time — never a symbolic re-derivation of the
// stencil coefficients a second time (that risks exactly the kind of
// transcription error this whole project's own standing discipline
// exists to avoid). Because every one of computeVertexPoint's/
// computeEdgePoint's/computeFacePoint's rules is a genuine LINEAR
// (weighted-average) function of neighboring positions, this is exactly
// A(n) — not an approximation of it — for any of the 3 coordinate axes
// independently (a "unit basis vector" here is a single SCALAR value
// placed at one neighborhood point and 0 elsewhere; the real 3D case is
// just this same scalar operation applied three times, once per axis,
// exactly equivalent to applying A(n) directly to real 3-vectors, since
// every underlying rule is a plain per-component weighted average).
//
// SELF-CONTAINMENT, checked directly in this module's own test file, not
// assumed: every one of the new center/new-spoke/new-corner formulas
// above reads ONLY template points already inside this same (2n+1)-point
// neighborhood (an outer "boundary-looking" edge of the template, e.g.
// spoke_i to far_i, genuinely IS a boundary edge from ITS OWN
// perspective — only 1 face uses it here — but this builder never reads
// a NEW VERTEX POINT at a spoke/far position, only the new CENTER vertex
// point, the new EDGE points on the vertex's own n original edges, and
// the new FACE points of the vertex's own n original faces — none of
// which ever needs anything outside the template to compute).
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

// Where a given original (undirected) edge's own NEW edge-point vertex
// lands in a subdivideCatmullClark output — subd.mjs's own documented
// ordering (vertex points keep their original index; edge points
// appended next, in buildTopology's own edgeMap iteration order; face
// points last) reused directly, not re-derived. `template` must be the
// EXACT cage subdivideCatmullClark itself was called against (edgeMap
// order is a pure function of the cage's own faces, so any structurally
// identical cage — same vertex count, same face index/vertex-order
// shape — produces the same order regardless of the vertex POSITIONS,
// which this builder deliberately overwrites for each unit-basis pass).
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

// Reads the (2n+1)-length neighborhood-template vector (center, n
// spokes, n corners, matching standardNeighborhoodCage's own index
// order) out of a REFINED cage — full [x,y,z] per point, generally
// useful (this module's own test file uses it to cross-check A(n)
// against a real, non-unit-basis subdivideCatmullClark call on all 3
// axes at once); the matrix builder below only ever needs one channel
// per pass, read directly rather than through this helper.
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

// A(n) as a plain (2n+1)x(2n+1) array of arrays, A[row][col] — applying
// it to a (2n+1)-vector v (real matrix-vector product, A.map(row =>
// row.reduce sum) is the caller's own job, not built into this function)
// reproduces exactly what one real subdivideCatmullClark step would do
// to that same neighborhood, for any of x/y/z independently.
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

// STEP 4 — VERTEX-LIMIT WEIGHTS PER INDIVIDUAL NEIGHBOR (generalizing
// step 2's reduced P/R/F triple to all 2n+1 points directly), plus a
// numerical eigendecomposition cross-check via power iteration.
//
// SCOPING DECISION, stated honestly, not silently narrowed: Stam's own
// full construction lets a caller evaluate the limit surface at an
// ARBITRARY (u,v) anywhere near an extraordinary vertex, by applying
// A(n)^k (via a cached eigendecomposition, no loop) and then handing a
// "regular sub-patch" of the resulting refined neighborhood to step 1's
// fast path. Building that requires a LARGER template than the (2n+1)
// neighborhood here captures (a full 16-point regular-patch stencil one
// ring further out than this module's own template reaches), plus the
// "picking matrices" Stam's paper uses to extract it at each recursion
// level — real, additional transcription risk this project's own
// standing "don't rush unverified math you can't check against the
// source" rule flags directly. It is NOT attempted here. What step 4
// DOES deliver, safely: an EXACT closed-form limit-POSITION weight per
// INDIVIDUAL neighbor (not just the reduced P/R/F triple step 2 used),
// independently re-derived below (not assumed), numerically cross-
// checked against A(n)'s own dominant eigenvector via power iteration
// (a method that never reads the closed form), and checked for genuine
// geometric convergence under repeated A(n) application. Every cage
// VERTEX position is exact via this; arbitrary interior (u,v) strictly
// between a vertex and a regular region remains the existing bounded
// discrete-refinement approximation SuperB already uses today for that
// one sub-case — a real, named, unchanged limitation, not a regression.
//
// THE DERIVATION — direct algebraic expansion of step 2's own R/F
// definitions in terms of the individual spoke values Q_1..Q_n and
// corner values C_1..C_n, no symmetry assumption needed (valid for a
// genuinely irregular/asymmetric neighborhood, not just a symmetric fan):
//   R = (1/n) * sum_i (P+Q_i)/2  =  P/2 + (1/(2n)) * sum_i Q_i
//   F = (1/n) * sum_i (P+Q_i+C_i+Q_{i+1})/4
//     = P/4 + (1/(2n)) * sum_i Q_i + (1/(4n)) * sum_i C_i
//     [sum_i Q_{i+1 mod n} == sum_i Q_i, just a cyclic relabeling of the
//      exact same n terms — the "+1" shift never drops or duplicates one]
// Substituting into step 2's own L = [(n-3)P + 4R + 4F] / (n+5):
//   4R = 2P + (2/n) * sum_i Q_i
//   4F = P + (2/n) * sum_i Q_i + (1/n) * sum_i C_i
//   (n-3)P + 4R + 4F = (n-3+2+1)*P + (4/n) sum_i Q_i + (1/n) sum_i C_i
//                    = n*P + (4/n) sum_i Q_i + (1/n) sum_i C_i
// so, per individual point (dividing through by (n+5)):
//   w(P)   = n / (n+5)
//   w(Q_i) = 4 / (n*(n+5))   for EACH of the n spokes individually
//   w(C_i) = 1 / (n*(n+5))   for EACH of the n corners individually
// (sums to exactly 1: n/(n+5) + n*[4/(n(n+5))] + n*[1/(n(n+5))] == 1.)
export function vertexLimitWeightsGeneral(n) {
  const w = new Array(2 * n + 1);
  w[0] = n / (n + 5);
  for (let i = 1; i <= n; i++) w[i] = 4 / (n * (n + 5));
  for (let i = n + 1; i <= 2 * n; i++) w[i] = 1 / (n * (n + 5));
  return w;
}

// Numerical, genuinely independent cross-check of the closed form above:
// A(n) is a real, non-negative, row-STOCHASTIC matrix (every row already
// proven above to sum to exactly 1 — kernel/subd.mjs's own weighted-
// average construction) with the constant vector as its own right
// eigenvector of eigenvalue 1 (also already proven above). By Perron-
// Frobenius, that eigenvalue is real, simple, and dominant, with a
// strictly positive LEFT eigenvector — power iteration on A(n)^T from
// any generic seed converges to it. This function never reads
// vertexLimitWeightsGeneral at all; the two are compared only in the
// test file, as two independently-arrived-at answers.
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

// STEP 5 — SEMI-SHARP HYBRID: an EXACT vertex-limit position at a vertex
// whose neighborhood currently carries a real, ACTIVE semi-sharp crease
// weight (a genuine value strictly between 0 and the app's own real cap
// — SUPERB_CREASE_LEVEL_SCALE=3, kernel/subd.mjs's own documented
// reachable range for any ordinary shipped Crease/SoftCrease gesture).
//
// A REAL, DELIBERATE EXCLUSION: a stored crease weight is NOT always a
// decaying semi-sharp
// value in this app — kernel/subd.mjs's own MARKED_CORNER_WEIGHT_FLOOR
// mechanism deliberately stores a weight far above the ordinary [0,3]
// range (TOSUBD's own DEFAULT_CORNER_CREASE_WEIGHT, well above 100) as a
// PERMANENT "hold this corner at P forever" marker, specifically chosen
// so it can NEVER be confused with an ordinary decaying crease. Naively
// running `ceil(weight)` real discrete levels for a weight in the
// hundreds would try to whole-cage-subdivide the cage hundreds of times
// (each level multiplying total face count by 4) — a catastrophic
// blowup, not a slow-but-correct answer. `MAX_SEMISHARP_DECAY_LEVELS`
// (8, a generous margin above the real reachable [0,3] ordinary range)
// refuses honestly rather than attempting this — and the refusal is
// harmless to fall back on: a marked-corner vertex is PROVABLY already
// exact at any discrete level once marked (computeVertexPoint's own
// marked-corner branch returns its OWN CURRENT position unconditionally,
// forever, the instant sharpness clamps to 1 — so "the position never
// moves again" makes the current discrete-refinement value already
// equal to every future level's value, including the true limit,
// trivially).
export const MAX_SEMISHARP_DECAY_LEVELS = 8;
//
// The eigenbasis machinery above (steps 2-4) all assumes an ORDINARY
// smooth vertex — a crease changes computeVertexPoint's own rule
// entirely (the CREASE branch, dart/crease-line/corner), which the
// (2n+1)-point template and every mask/matrix built from it never
// modeled. Rather than re-derive a second, crease-aware eigenbasis (real
// literature exists for this — Biermann/Levin/Zorin's semi-sharp-crease
// eigenanalysis — but transcribing it without the source in hand is
// exactly the risk this project's standing discipline avoids), this
// reuses subd.mjs's own ALREADY-PROVEN, already-tested per-level decay
// (`max(0, weight-1)`, dropped entirely once non-positive — see
// subd.mjs's own subdivideCatmullClark) for EXACTLY as many real
// discrete levels as it takes the crease to fully decay away from this
// one vertex's own immediate neighborhood, then hands off to the exact
// machinery once that neighborhood is genuinely, structurally smooth
// again (0 sharp edges incident to it) — never approximating past that
// point with yet more discrete refinement.
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

// ===================================================================
// STEP 6 — REGULAR-FACE CLASSIFICATION AND STENCIL EXTRACTION
// (ToNURBS, step 1)
//
// bicubicRegularPatchSurface above takes a 4x4 grid and says nothing
// about where that grid comes from; its own header deliberately left
// "how a caller derives that layout from a real cage" for later. This is
// later. Two functions: one that says whether a face QUALIFIES, and one
// that reads the 16 points out in the order the patch evaluator expects.
//
// WHAT "REGULAR" HAS TO MEAN, and why each clause is load-bearing rather
// than defensive: the limit surface over a face is exactly a uniform
// bicubic B-spline patch on its 16-point neighborhood ONLY when nothing
// in that neighborhood perturbs the ordinary smooth rules. So every one
// of the face's four corners must be INTERIOR (a boundary edge is pinned
// to full sharpness forever by edgeSharpness, so the smooth rules never
// apply there), of VALENCE EXACTLY 4 (an extraordinary vertex is the
// whole reason the general case is hard), and its whole one-ring must be
// QUADS (a triangle or n-gon in the ring changes what the stencil even
// is — there is no 4x4 grid to read). Creases are checked across the
// entire 3x3 face block, not just the center face's own four edges: a
// crease on the OUTER boundary of the block still changes the subdivision
// that the block's own points converge under, so a stencil that ignored
// it would return a patch that is simply not the limit surface.
//
// A conservative classifier is the correct kind here. A face wrongly
// called irregular costs one extra isolation step (73's own step 2, which
// shrinks the region geometrically anyway); a face wrongly called regular
// emits a patch that is silently, unrecoverably not the surface.
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
// (i = U, j = V). The face's own four corners land at the CENTER 2x2:
// face[0] -> (1,1), face[1] -> (2,1), face[2] -> (2,2), face[3] -> (1,2),
// so U runs face[0]->face[1] and V runs face[0]->face[3], and the patch's
// parametric corners (0,0)/(1,0)/(1,1)/(0,1) correspond to the face's own
// vertices in cyclic order. Everything else is read off the surrounding
// ring: an EDGE-neighbor face contributes the two points just outside
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
    const outerP = nextP === q ? prevP : nextP; // the neighbor of p in that face that ISN'T q
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

// The whole point of steps 1+6 together: one regular face becomes one
// EXACT bicubic patch, with no fitting, no tolerance and no sampling.
//
// EMITTED CLAMPED. The patch is the same surface either way; what the
// clamped form buys is a knot array whose ends actually bound the valid
// domain, which is what the ~ten consumers that derive a surface's domain
// as knots[0]..knots[last] silently assume. See clampedBicubicPatchSurface
// for the conversion and why it is not knot insertion.
export function regularFaceToPatch(cage, faceIdx, ctx = buildTopology(cage)) {
  return clampedBicubicPatchSurface(regularFaceStencil(cage, faceIdx, ctx));
}

// ===================================================================
// STEP 7 — ISOLATE AND EMIT (ToNURBS,
// steps 0-2). Turns a whole cage into exact bicubic patches.
//
// THE SHAPE OF THE ALGORITHM, and why it terminates usefully: a face
// touching an extraordinary vertex subdivides into four sub-faces, only
// ONE of which still touches that vertex. So each isolation level emits
// the three that became regular and carries the fourth forward, and the
// unconverted region shrinks by a factor of 4 per level while the patch
// ring around each extraordinary point grows. Nothing here approximates:
// every patch this returns is the exact limit surface over its own face.
//
// WHAT THIS DELIBERATELY DOES NOT DO — and this is the honest line, not
// a missing feature discovered later. It does NOT cap the shrinking hole
// around an extraordinary vertex. A cap pinned only at that vertex's own
// exact limit position meets its neighbors at a POINT, not along their
// shared EDGES, which is precisely the gap a stitch cannot close; the
// tractable cap has to take its boundary control rows FROM the adjacent
// patches (they are shared exactly, by construction) and pin only the
// remaining interior freedom. That is its own build. Until it exists,
// the leftover faces are RETURNED, named, measured — never quietly
// papered over with a patch that is not the surface.
//
// COVERAGE IS REPORTED IN DOMAIN TERMS, not by eye: an uncovered face at
// isolation level L occupies 4^-L of one original face, so the returned
// uncoveredFraction is a real number a caller can act on (and a real
// number a test can watch shrink as levels rise).
//
// The input cage is never mutated — every level works on a fresh
// subdivided copy, matching this module's own discipline throughout.
//
// CAPPING IS OPT-IN, `{ cap: true }`. Without it the behavior above is
// unchanged and the leftovers are reported rather than covered, which is
// what a caller that wants to know about them relies on. With it, step 8
// below emits one patch per leftover region and `uncovered` shrinks to
// only the regions that could not be capped, each carrying the reason.
export function subdToPatches(cage, opts = {}) {
  const maxIsolation = opts.maxIsolation ?? 3;
  const patches = [];
  let current = { vertices: cage.vertices.map((v) => v.slice()), faces: cage.faces.map((f) => f.slice()), creases: { ...(cage.creases || {}) } };
  let live = current.faces.map((_, fi) => fi);
  const originalFaceCount = cage.faces.length;
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
      if (isRegularFace(current, fi, ctx)) {
        const srf = regularFaceToPatch(current, fi, ctx);
        emittedAtLevel.set(fi, srf);
        patches.push({ srf, level, faceIndex: fi, kind: 'regular' });
      } else {
        stillLive.push(fi);
      }
    }
    live = stillLive;
    if (!live.length || level >= maxIsolation) { current = { ...current, __ctx: undefined }; break; }

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

  const ctxFinal = buildTopology(current);
  const describe = (fi) => ({
    level,
    faceIndex: fi,
    // The extraordinary vertices actually responsible, each with its own
    // EXACT limit position already in hand — this is what a cap builder
    // needs first, and it is exactly the point 73 notes is already known
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

// ===================================================================
// STEP 8 — THE CAP (ToNURBS, step 3).
//
// WHAT A CAP HAS TO BE. Step 7 leaves, around every extraordinary vertex E
// of valence n, a ring of n quad faces that never became regular. Each one
// has E at one corner and three ordinary valence-4 corners, and — MEASURED,
// not assumed — exactly TWO of its four edges border a face that step 7
// already emitted as an exact patch; the other two run from E out to a
// neighboring corner and are shared with the next face of the same ring.
// So a cap is not a lid over a hole with a free rim. It is a patch whose
// boundary is almost entirely already decided by surfaces that are exactly
// the limit surface, and the only real freedom is near E.
//
// WHY NOT PIN ONLY THE VERTEX. A cap pinned only at E's own exact limit
// position meets its neighbors at a POINT and nowhere else — a hole with
// extra steps, and precisely the gap a B-rep stitch cannot close.
//
// THE CONSTRUCTION IS LOOP & SCHAEFER'S ACC (Approximating Catmull-Clark
// Subdivision Surfaces with Bicubic Patches, ACM TOG 27(1), 2008), the
// standard answer to this problem and the one shipped in hardware
// tessellation pipelines. In Bezier form, per quad face:
//
//   corner  b00   the vertex's own exact limit position — the Halstead/
//                 Kass/DeRose mask, which is vertexLimitPosition above.
//   interior b11  (n*v + 2*(ePrev + eNext) + diag) / (n + 5), where v is
//                 the corner, ePrev/eNext its two neighbors IN THAT FACE,
//                 diag the face's fourth vertex, and n = valence(v).
//   edge    b10   the MIDPOINT of the two adjacent faces' own interior
//                 points at the shared corner.
//
// Those three masks are not transcribed; they are re-derived here from the
// paper's own stated geometric relationship — that a corner point is the
// centroid of the interior points around it, and an edge point the midpoint
// of the two beside it. Writing the interior mask as
// (a*v + b*(ePrev+eNext) + c*diag)/(a+2b+c) and demanding that the centroid
// over the n faces at v reproduce (n^2*v + 4*sum(e) + sum(diag))/(n(n+5))
// forces a:b:c = n^2 : 2n : n, i.e. the form above, with the denominator
// falling out as n(n+5) — the limit mask's own. At n = 4 all three reduce
// to the uniform B-spline-to-Bezier knot-insertion masks, which is the
// correctness check that matters: the same numbers clampedBicubicPatchSurface
// produces for a regular face, reached a completely different way.
//
// WHAT THAT BUYS, and it is more than the plan expected. ACC's continuity
// claim is that patches meet smoothly EXCEPT along an edge containing an
// extraordinary vertex, where they are only C0. A cap's two outer edges do
// not contain E. So the join between a cap and the exact regular region is
// tangent-continuous, and measurably so: the normal deviation across it is
// 0.0000 degrees on every fixture tried, including a valence-5 extruded
// cage. The tangent break is confined to the star edges, cap against cap,
// where it is real and named rather than approximated away.
//
// THE ONE REFINEMENT ON TOP OF ACC. Of the sixteen control points, thirteen
// are pinned: the two outer rows are the neighbors' own rows copied
// element for element, and the two rows immediately inside them carry the
// cross-boundary derivative that makes that join exact. Three are touched
// by no continuity condition at all — the two star-edge control points
// adjacent to E, and the interior point nearest E. Those three are re-
// solved to INTERPOLATE the true limit surface at three points near the
// star: one quarter of the way along each star edge, and the (1/4,1/4)
// corner of the face. The true limit at a dyadic parameter needs no new
// evaluation math — it is a twice-refined vertex's own limit position under
// the same mask this module already uses as ground truth. Measured, this
// cuts the worst deviation from the true limit surface by about 3.9x and
// the tangent break across the star edges by about 1.9x, and leaves every
// exactness property of the ACC base untouched.
//
// WHAT IS EXACT AND WHAT IS NOT, stated plainly:
//   EXACT  the star corner IS vertexLimitPosition(E), bit for bit.
//   EXACT  the two outer boundary rows are the neighboring patches' own
//          arrays, so the shared edge curve is literally the same curve.
//   EXACT  the two star-edge rows are computed once per edge and read by
//          both caps that share it, so cap meets cap bit for bit.
//   EXACT  tangent continuity across a cap's two outer edges.
//   NOT    the interior. A single bicubic cannot be the Catmull-Clark limit
//          over a face touching an extraordinary vertex — that surface is
//          an infinite nest of patches, not one. The deviation is measured
//          in this module's own test file rather than claimed, and it does
//          not vanish with isolation level: the region is self-similar
//          under refinement, so the ABSOLUTE error shrinks (about 2.4x per
//          level) while the error relative to the cap's own size holds
//          steady near 0.4%.
//   NOT    tangent continuity across the star edges. Real, measured, and
//          shrinking with isolation level; this is the knob.
//
// A NAMED LIMIT ON BIT-EXACTNESS AT THE JUNCTION CORNERS. A corner where a
// cap meets two regular patches can only carry one value, and the two
// regular patches do not agree on it to the last bit — they agree to about
// 1e-15, because each computes the same limit position through a different
// order of the same arithmetic. That disagreement is a pre-existing
// property of the emitted set, not something capping introduces: NO two
// adjacent regular patches in this module's output are bit-identical along
// their shared row either. So each cap is bit-identical with one of its two
// regular neighbors along the whole shared row, and with the other on the
// two interior control points, differing only at the shared corner and only
// by the amount those two neighbors already differ. Welding the whole
// emitted set to per-edge and per-vertex canonical values would close that
// last gap; it is deliberately not done here, because it would perturb
// patches that are currently exactly the limit surface in order to fix a
// disagreement no consumer's tolerance can see.
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
// in their own numbering. Used only to keep the star correction's two
// refinements local — refining a whole cage twice to read a few dozen points
// near its extraordinary vertices costs an order of magnitude more than
// everything else in this file put together.
//
// THE PAD IS DELIBERATE MARGIN, not a guess that happened to work. Two
// separate things reach outward: subdivision treats the sub-cage's cut edge
// as a real naked boundary and applies the boundary rules there, and that
// wrong value spreads one further ring inward per level; and a limit position
// two levels down reads through roughly three rings of the cage it started
// from. A pad of 2 already reproduces the whole-cage answer BIT-IDENTICALLY
// on every cage tried, so 5 is margin over a measured floor rather than the
// floor itself. The bit-identity is measured in the test file; if it ever
// fails, this number is the thing to raise.
const LOCAL_PROBE_RINGS = 5;

function localNeighbourhoodCage(cage, ctx, seedFaces, rings) {
  let faceSet = new Set(seedFaces);
  for (let r = 0; r < rings; r++) {
    const next = new Set(faceSet);
    for (const fi of faceSet) for (const v of cage.faces[fi]) for (const nf of ctx.vertexFaces[v]) next.add(nf);
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
    // The limit surface one QUARTER of the way along edge v0 -> v1, from v0.
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

// One cap per leftover region, or an honest refusal naming what the region
// failed. `emitted` maps a face index at this level to the patch already
// emitted for it.
export function capStarRegions(cage, ctx, live, emitted) {
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
