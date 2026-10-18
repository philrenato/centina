// Surface flattening / unroll — LSCM.
// Given a triangulation of a surface (the app's own tessellation, never a
// second tessellator), produce a flat 2D layout of it plus a measured
// report of the flattening's distortion.
//
// Bound on every function below: Gaussian curvature is an intrinsic
// invariant of a surface under any isometric (length-preserving) map. This
// is Gauss's Theorema Egregium. It follows that a surface with nonzero
// Gaussian curvature anywhere cannot be flattened into the plane without
// local stretch or tearing — by any algorithm, at any resolution.
//
// So every function here returns a measured distortion metric in
// usable units alongside its result, and never a claim of exactness for a
// curved input — the same convention kernel/offset.mjs and
// kernel/offsetcurve.mjs follow for their "not exact for a curved input"
// limits.
//
// The one case where an exact answer is mathematically possible is a
// developable input (a cylinder, a cone, a plane — zero Gaussian curvature
// everywhere). That is this module's exactness anchor, checked against an
// analytically known unrolled shape rather than against the flattener's
// own output.
//
// Algorithm: LSCM, and what it does not minimize
// Levy, Petitjean, Ray, Maillot, "Least Squares Conformal Maps for
// Automatic Texture Atlas Generation," SIGGRAPH 2002 (ACM Transactions on
// Graphics 21(3), pp. 362-371).
//
// LSCM minimizes the discrete conformal energy: per triangle, the squared
// deviation of the piecewise-linear map from satisfying the Cauchy-Riemann
// equations. It is a single sparse linear least-squares solve — no outer
// iteration, no convergence risk, no local minima.
//
// LSCM minimizes angle distortion. It does not minimize stretch. A
// flattened result can be locally correct in shape while being
// wrong in scale, which is the property that matters when cutting
// material. So `flatteningDistortion` below reports area and edge-length
// distortion (what LSCM was not optimizing) beside the angle distortion
// (what it was).
//
// ARAP (Liu, Zhang, Xu, Gotsman, Gortler, "A Local/Global Approach to Mesh
// Parameterization," SGP 2008 / Computer Graphics Forum 27(5)) is the
// stretch-minimizing method. `arapRefine` below runs it from an LSCM layout
// for a fixed number of rounds; cutAndFlattenGores (seam.mjs) uses it on
// each gore.
//
// Units
// Positions are in the document's own length unit (mm throughout this app).
// The returned `uv` is in the same unit — the raw LSCM solution is defined
// only up to a global similarity, so it is rescaled by the single uniform
// factor that best fits the true 3D edge lengths in a least-squares sense,
// and translated so its bounding box starts at (0,0). Angles are reported
// in degrees; the intrinsic angle defect in radians; areas in the square of
// the length unit; every ratio-shaped error is dimensionless.

import { surfacePointAndPartials } from './surface.mjs';
import { tessellateTrimmedSurface } from './trimtess.mjs';

// Helpers

function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function len3(a) { return Math.hypot(a[0], a[1], a[2]); }

function bboxDiagonal(positions) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const p of positions) {
    if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
    if (p[2] < minZ) minZ = p[2]; if (p[2] > maxZ) maxZ = p[2];
  }
  if (!Number.isFinite(minX)) return 0;
  return Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
}

function edgeKey(a, b) { return a < b ? `${a}|${b}` : `${b}|${a}`; }

// Welding

// Weld a non-indexed triangle soup (what kernel/trimtess.mjs's
// tessellateTrimmedSurface returns — [[vA,vB,vC], ...] with each v carrying
// {position, normal, uv}) into an indexed mesh, which is what any
// parametrization algorithm needs: LSCM solves for one (u,v) per shared
// vertex, so two triangles meeting along an edge must reference
// the same two vertex indices, not two coincident copies.
//
// This is welding, not tessellation — it never invents or moves a point, it
// only decides which of the points already produced are the same point.
//
// `tolerance` defaults to a relative value (1e-9 of the bounding-box
// diagonal) rather than an absolute one, so the same call behaves the same
// way on a 5mm part and a 5000mm one. In practice a grid tessellation's
// shared corners are already bit-identical (adjacent cells compute the same
// domain fraction from the same integer index), so this is a safety net for
// trimmed/clipped output rather than the common path.
export function weldTriangulation(triangles, tolerance = null) {
  if (!Array.isArray(triangles)) throw new Error('weldTriangulation: expected an array of triangles');
  const allPts = [];
  for (const tri of triangles) {
    if (!Array.isArray(tri) || tri.length !== 3) throw new Error('weldTriangulation: every triangle must be exactly 3 vertices');
    for (const v of tri) {
      const p = Array.isArray(v) ? v : v && v.position;
      if (!p || p.length < 3) throw new Error('weldTriangulation: every vertex needs a [x,y,z] position (or a .position field)');
      allPts.push(p);
    }
  }
  const diag = bboxDiagonal(allPts);
  const tol = tolerance != null ? tolerance : Math.max(1e-12, 1e-9 * Math.max(diag, 1));
  const cell = Math.max(tol * 4, 1e-12);

  const buckets = new Map();
  const positions = [];
  const indexOf = (p) => {
    const bx = Math.floor(p[0] / cell), by = Math.floor(p[1] / cell), bz = Math.floor(p[2] / cell);
    // Search the 27 neighboring cells, not just the containing one — a pair
    // of coincident points can straddle a cell boundary.
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const list = buckets.get(`${bx + dx},${by + dy},${bz + dz}`);
          if (!list) continue;
          for (const idx of list) {
            const q = positions[idx];
            if (Math.abs(q[0] - p[0]) <= tol && Math.abs(q[1] - p[1]) <= tol && Math.abs(q[2] - p[2]) <= tol) return idx;
          }
        }
      }
    }
    const idx = positions.length;
    positions.push([p[0], p[1], p[2]]);
    const key = `${bx},${by},${bz}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(idx);
    return idx;
  };

  const faces = [];
  let k = 0;
  for (let t = 0; t < triangles.length; t++) {
    const a = indexOf(allPts[k++]), b = indexOf(allPts[k++]), c = indexOf(allPts[k++]);
    faces.push([a, b, c]);
  }
  return { positions, faces, weldTolerance: tol };
}

// Drop the faces that welding itself collapsed. A revolve whose profile
// touches the axis has a pole — the whole pole row of the parameter grid is
// one physical point — and `tessellateTrimmedSurface` emits two triangles
// per grid cell unconditionally, so after welding, the pole cell's second
// triangle names the same vertex twice. `validateFlattenMesh` would refuse
// that face, making every pole-bearing surface unflattenable: a cone
// with its apex, a sphere, an ellipsoid — including a partial cone sector
// that is exactly developable and should unroll with zero distortion.
//
// Dropping is lossless, not a tolerance: a repeated-vertex triangle has
// identically zero area and contributes no topology (the "quad" at a pole
// is a triangle, and what remains around the pole is a proper
// triangle fan). Positions are returned untouched and the summed triangle
// area is bit-for-bit unchanged.
//
// Not folded into `weldTriangulation` (a general welder must not silently
// discard faces a caller handed it), and `validateFlattenMesh` is not
// relaxed (a degenerate face in a mesh a caller supplied is still caught).
// Applied only where the problem is created: the NURBS entry points
// (flattenNurbsSurface below, and seam.mjs).
export function dropDegenerateFaces(mesh) {
  const kept = [];
  let dropped = 0;
  for (const [a, b, c] of mesh.faces) {
    if (a === b || b === c || c === a) { dropped++; continue; }
    kept.push([a, b, c]);
  }
  return { ...mesh, faces: kept, droppedFaces: dropped };
}

// Topology

// Edge/boundary/component structure of an indexed triangle mesh. Shared by
// the validity check, the distortion metrics, and a future ARAP — one
// definition, never re-derived per caller.
export function buildMeshTopology(positions, faces) {
  const edgeFaces = new Map(); // "i|j" -> [faceIndex, ...]
  for (let f = 0; f < faces.length; f++) {
    const [a, b, c] = faces[f];
    for (const [x, y] of [[a, b], [b, c], [c, a]]) {
      const key = edgeKey(x, y);
      if (!edgeFaces.has(key)) edgeFaces.set(key, []);
      edgeFaces.get(key).push(f);
    }
  }
  const nonManifoldEdges = [];
  const boundaryEdges = [];
  const edges = [];
  for (const [key, fs] of edgeFaces) {
    const [a, b] = key.split('|').map(Number);
    edges.push([a, b]);
    if (fs.length > 2) nonManifoldEdges.push([a, b, fs.length]);
    else if (fs.length === 1) boundaryEdges.push([a, b]);
  }
  const boundaryVertices = new Set();
  for (const [a, b] of boundaryEdges) { boundaryVertices.add(a); boundaryVertices.add(b); }

  // Connected components over faces (adjacency through shared edges) — a
  // parametrization is only well posed on one connected piece at a time.
  const seen = new Uint8Array(faces.length);
  let componentCount = 0;
  const componentOfFace = new Int32Array(faces.length).fill(-1);
  for (let start = 0; start < faces.length; start++) {
    if (seen[start]) continue;
    componentCount++;
    const id = componentCount - 1;
    const stack = [start];
    seen[start] = 1;
    while (stack.length) {
      const f = stack.pop();
      componentOfFace[f] = id;
      const [a, b, c] = faces[f];
      for (const [x, y] of [[a, b], [b, c], [c, a]]) {
        for (const g of edgeFaces.get(edgeKey(x, y))) {
          if (!seen[g]) { seen[g] = 1; stack.push(g); }
        }
      }
    }
  }
  // Vertices referenced by a face (a welded soup can't produce an
  // orphan, but an externally-supplied mesh can).
  const usedVertices = new Set();
  for (const [a, b, c] of faces) { usedVertices.add(a); usedVertices.add(b); usedVertices.add(c); }

  return {
    edges, edgeFaces, boundaryEdges, boundaryVertices,
    nonManifoldEdges, componentCount, componentOfFace,
    usedVertices, vertexCount: positions.length, faceCount: faces.length,
  };
}

// Refuses by name rather than returning a plausible-looking wrong layout —
// a degenerate input gets a specific message, never a silent result.
// Algorithm-agnostic, so a future ARAP can share it with LSCM.
export function validateFlattenMesh(positions, faces, opts = {}) {
  if (!Array.isArray(positions) || !Array.isArray(faces)) {
    throw new Error('validateFlattenMesh: expected {positions, faces} arrays');
  }
  if (faces.length < 1) {
    throw new Error('validateFlattenMesh: refusing to flatten — the mesh has no triangles at all');
  }
  if (positions.length < 3) {
    throw new Error(`validateFlattenMesh: refusing to flatten — only ${positions.length} vertices, a triangulation needs at least 3`);
  }
  for (const p of positions) {
    if (!p || p.length < 3 || !Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2])) {
      throw new Error('validateFlattenMesh: refusing to flatten — a vertex position is missing or non-finite');
    }
  }
  const diag = bboxDiagonal(positions);
  const areaTol = opts.degenerateAreaTolerance != null
    ? opts.degenerateAreaTolerance
    : 1e-12 * Math.max(diag * diag, 1e-12);

  for (let f = 0; f < faces.length; f++) {
    const tri = faces[f];
    if (!Array.isArray(tri) || tri.length !== 3) throw new Error(`validateFlattenMesh: face ${f} is not a triangle`);
    const [a, b, c] = tri;
    for (const i of tri) {
      if (!Number.isInteger(i) || i < 0 || i >= positions.length) {
        throw new Error(`validateFlattenMesh: face ${f} references vertex index ${i}, which does not exist`);
      }
    }
    if (a === b || b === c || c === a) {
      throw new Error(`validateFlattenMesh: refusing to flatten — face ${f} is a DEGENERATE TRIANGLE (it uses the same vertex twice: [${a}, ${b}, ${c}])`);
    }
    const twoA = len3(cross3(sub3(positions[b], positions[a]), sub3(positions[c], positions[a])));
    if (!(twoA > areaTol)) {
      throw new Error(`validateFlattenMesh: refusing to flatten — face ${f} is a DEGENERATE TRIANGLE (zero area: its three vertices are collinear or coincident)`);
    }
  }

  const topo = buildMeshTopology(positions, faces);
  if (topo.nonManifoldEdges.length) {
    const [a, b, n] = topo.nonManifoldEdges[0];
    throw new Error(`validateFlattenMesh: refusing to flatten — the mesh is NON-MANIFOLD (edge ${a}-${b} is shared by ${n} faces; a surface edge can touch at most 2)`);
  }
  if (topo.componentCount > 1) {
    throw new Error(`validateFlattenMesh: refusing to flatten — the mesh is DISCONNECTED (${topo.componentCount} separate pieces). Flatten one connected piece at a time.`);
  }
  if (topo.boundaryEdges.length === 0) {
    throw new Error('validateFlattenMesh: refusing to flatten — the mesh is CLOSED (it has no boundary). A closed surface cannot be laid flat without first being cut open, whatever its curvature.');
  }
  // A flattening is a map into the plane. Only a topological disk can be
  // mapped into the plane without self-overlap, whatever its curvature —
  // an annulus (a full revolve welded shut at its seam) or a handle has
  // nowhere for the extra topology to go. Euler characteristic V - E + F is
  // exactly 1 for a disk, so this is a cheap, exact test rather than a
  // heuristic. Caught here by name, because LSCM will otherwise
  // return a folded-over layout that looks plausible and is not usable.
  const chi = topo.usedVertices.size - topo.edges.length + topo.faceCount;
  if (chi !== 1) {
    throw new Error(`validateFlattenMesh: refusing to flatten — the mesh is not a TOPOLOGICAL DISK (Euler characteristic V-E+F = ${chi}, a disk is 1). It has a hole or a handle; cut it open along a seam first, then flatten each piece.`);
  }
  // "Too few faces to constrain the solve": LSCM pins 2 vertices to remove
  // the 4-dof similarity ambiguity, so there must be at least one free
  // vertex left for the solve to have anything to determine.
  if (topo.usedVertices.size < 3) {
    throw new Error(`validateFlattenMesh: refusing to flatten — only ${topo.usedVertices.size} vertices are actually used by a face; after pinning 2 to fix the layout's own position/rotation/scale there is nothing left to solve for.`);
  }
  return topo;
}

// Local frames

// The isometric 2D coordinates of one triangle: lay it flat in its own
// plane with vertex 0 at the origin and vertex 1 on the +x axis. This is
// exact (a single triangle is trivially developable) and is the per-triangle
// reference frame LSCM (below) and a future ARAP measure against.
// Returns { x1,y1, x2,y2, x3,y3, twoArea }.
export function triangleLocalFrame(p1, p2, p3) {
  const e1 = sub3(p2, p1);
  const e2 = sub3(p3, p1);
  const x2 = len3(e1);
  if (!(x2 > 0)) return null;
  const ex = [e1[0] / x2, e1[1] / x2, e1[2] / x2];
  const x3 = dot3(e2, ex);
  const y3 = len3(cross3(e1, e2)) / x2; // always >= 0: the frame is chosen so the triangle is CCW
  return { x1: 0, y1: 0, x2, y2: 0, x3, y3, twoArea: x2 * y3 };
}

// Solver

// Sparse linear least squares, matrix-free, via conjugate gradient on the
// normal equations (CGLS/CGNR) with Jacobi (diagonal) preconditioning.
//
// Why this method suits LSCM (this kernel has no dependencies, so the
// solver is written here):
//
//  * The LSCM matrix is very sparse and structured: each triangle
//    contributes exactly 2 rows with at most 6 nonzeros each. A matrix-free
//    Krylov method touches only those nonzeros, so one iteration is O(nnz)
//    with no fill-in at all. A direct sparse factorization (Cholesky/QR)
//    would be faster per solve for a fixed mesh but needs ordering and
//    fill-reducing machinery to beat that — a large amount of code for a
//    solve that is not the bottleneck.
//  * A^T A is symmetric positive definite here, which is what CG
//    requires. It is positive definite because two vertices are pinned: the
//    conformal energy's own nullspace is precisely the 4-parameter
//    similarity group (translate 2, rotate 1, scale 1) of the whole layout,
//    and pinning two vertices removes exactly those 4 degrees of freedom.
//    So the pinning is not a convenience, it is what makes the solve
//    well posed.
//  * Numerical caveat: forming the normal
//    equations squares the condition number, cond(A^T A) = cond(A)^2. Three
//    things keep that acceptable here: (1) each triangle's rows are scaled
//    by 1/sqrt(2*area), which makes every matrix entry O(1) regardless of
//    triangle size, so a mesh with wildly varying triangle areas does not
//    have its conditioning wrecked by the largest ones; (2) Jacobi
//    preconditioning divides out the remaining per-column scale spread;
//    (3) the achieved residual is measured and returned, and a
//    failure to converge throws by name instead of returning a
//    plausible-looking wrong answer.
//
// `triplets` is a flat array of {row, col, val}. Solves min ||A x - b||.
export function solveSparseLeastSquares(triplets, rhs, nCols, opts = {}) {
  const nRows = rhs.length;
  const maxIter = opts.maxIterations != null ? opts.maxIterations : Math.min(200000, 40 * nCols + 1000);
  const relTol = opts.tolerance != null ? opts.tolerance : 1e-11;

  // The triplets packed by row (for A x) and by column (for A^T y), each
  // keeping the triplets' own order inside a row or column, so every sum is
  // accumulated in the same order as a pass over the triplet list and the
  // result is bit-for-bit the same; typed arrays make each pass several
  // times faster than reading the triplet objects.
  const nnz = triplets.length;
  const pack = (major, minor, nMajor) => {
    const ptr = new Int32Array(nMajor + 1);
    for (let k = 0; k < nnz; k++) ptr[triplets[k][major] + 1]++;
    for (let i = 0; i < nMajor; i++) ptr[i + 1] += ptr[i];
    const idx = new Int32Array(nnz), val = new Float64Array(nnz), at = ptr.slice(0, nMajor);
    for (let k = 0; k < nnz; k++) { const t = triplets[k], j = at[t[major]]++; idx[j] = t[minor]; val[j] = t.val; }
    return { ptr, idx, val };
  };
  const byRow = pack('row', 'col', nRows), byCol = pack('col', 'row', nCols);
  const sweep = ({ ptr, idx, val }, n, src, out) => {
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = ptr[i]; k < ptr[i + 1]; k++) s += val[k] * src[idx[k]];
      out[i] = s;
    }
  };
  const applyA = (x, out) => sweep(byRow, nRows, x, out);
  const applyAT = (y, out) => sweep(byCol, nCols, y, out);

  // Jacobi preconditioner: the diagonal of A^T A.
  const diag = new Float64Array(nCols);
  for (let i = 0; i < nCols; i++) {
    let s = 0;
    for (let k = byCol.ptr[i]; k < byCol.ptr[i + 1]; k++) s += byCol.val[k] * byCol.val[k];
    diag[i] = s;
  }
  for (let i = 0; i < nCols; i++) if (!(diag[i] > 0)) diag[i] = 1;

  const x = new Float64Array(nCols);
  const r = new Float64Array(nRows);
  for (let i = 0; i < nRows; i++) r[i] = rhs[i];
  const s = new Float64Array(nCols);
  const z = new Float64Array(nCols);
  const p = new Float64Array(nCols);
  const q = new Float64Array(nRows);

  applyAT(r, s);
  let s0 = 0;
  for (let i = 0; i < nCols; i++) s0 += s[i] * s[i];
  s0 = Math.sqrt(s0);
  if (!(s0 > 0)) {
    // A^T b is exactly zero: x = 0 is already the least-squares solution.
    return { x: Array.from(x), iterations: 0, residual: 0, converged: true };
  }
  for (let i = 0; i < nCols; i++) z[i] = s[i] / diag[i];
  for (let i = 0; i < nCols; i++) p[i] = z[i];
  let gamma = 0;
  for (let i = 0; i < nCols; i++) gamma += s[i] * z[i];

  let iter = 0;
  let resid = 1;
  for (; iter < maxIter; iter++) {
    applyA(p, q);
    let qq = 0;
    for (let i = 0; i < nRows; i++) qq += q[i] * q[i];
    if (!(qq > 0)) break; // A p == 0 with p != 0 would mean a rank deficiency the pins should have removed
    const alpha = gamma / qq;
    for (let i = 0; i < nCols; i++) x[i] += alpha * p[i];
    for (let i = 0; i < nRows; i++) r[i] -= alpha * q[i];
    applyAT(r, s);
    let sn = 0;
    for (let i = 0; i < nCols; i++) sn += s[i] * s[i];
    resid = Math.sqrt(sn) / s0;
    if (resid <= relTol) { iter++; break; }
    for (let i = 0; i < nCols; i++) z[i] = s[i] / diag[i];
    let gammaNew = 0;
    for (let i = 0; i < nCols; i++) gammaNew += s[i] * z[i];
    const beta = gammaNew / gamma;
    for (let i = 0; i < nCols; i++) p[i] = z[i] + beta * p[i];
    gamma = gammaNew;
  }

  for (let i = 0; i < nCols; i++) {
    if (!Number.isFinite(x[i])) {
      throw new Error('solveSparseLeastSquares: the solve produced a non-finite value — the system is degenerate, refusing to return it');
    }
  }
  return { x: Array.from(x), iterations: iter, residual: resid, converged: resid <= Math.max(relTol, 1e-6) };
}

// Metrics

// Extremes of a per-triangle array by a loop: spreading it into Math.min
// overflows the call stack once the mesh has tens of thousands of triangles.
function arrayMin(a) { let m = Infinity; for (const x of a) { if (x !== x) return NaN; if (x < m) m = x; } return m; }
function arrayMax(a) { let m = -Infinity; for (const x of a) { if (x !== x) return NaN; if (x > m) m = x; } return m; }

// Measure the flattening. Computed from the 3D mesh and the 2D layout
// independently of how that layout was produced, so it is a measurement
// rather than a self-consistency check on the solver.
//
// Returned units:
//   angle.maxDeg / rmsDeg / meanDeg    degrees        (per-corner |2D - 3D|)
//   angle.maxRel / rmsRel              dimensionless  (that error / the 3D angle)
//   edge.maxAbsErr / rmsAbsErr         length units   (mm here)
//   edge.maxRelErr / rmsRelErr         dimensionless
//   area.total3d / total2d             length^2       (mm^2 here)
//   area.totalRatio, min/maxRatio      dimensionless  (2D area / 3D area)
//   area.maxRelErr / rmsRelErr         dimensionless  (|ratio - 1|)
//   intrinsic.*AngleDefect             radians        (discrete Gaussian curvature)
//   flippedTriangles                   count          (2D orientation reversals)
//
// intrinsic.maxAbsInteriorAngleDefect is the discrete Gaussian curvature
// (2*pi minus the incident angle sum at an interior vertex). By Theorema
// Egregium it is the quantity that decides, before any flattening
// is attempted, whether zero distortion is possible: it is zero
// everywhere on a developable and nonzero on anything doubly-curved. It is reported so a caller can tell "this flattening is
// good" apart from "this surface was flattenable in the first place."
export function flatteningDistortion(positions, faces, uv) {
  const angleErrs = [];
  const angleRelErrs = [];
  let area3dTotal = 0, area2dTotal = 0;
  const areaRatios = [];
  let flipped = 0;
  const angleSumAt = new Float64Array(positions.length);

  for (const [a, b, c] of faces) {
    const P = [positions[a], positions[b], positions[c]];
    const Q = [uv[a], uv[b], uv[c]];
    const twoA3 = len3(cross3(sub3(P[1], P[0]), sub3(P[2], P[0])));
    const sa2 = 0.5 * ((Q[1][0] - Q[0][0]) * (Q[2][1] - Q[0][1]) - (Q[2][0] - Q[0][0]) * (Q[1][1] - Q[0][1]));
    // Orientation, measured against a canonical reference rather than a
    // majority vote among the triangles themselves: triangleLocalFrame
    // always lays a triangle out counter-clockwise in its own isometric
    // frame, so an orientation-preserving (holomorphic) map must give every
    // triangle a positive signed 2D area. A negative one is a
    // reversal — either a local fold, or (if every triangle is negative) a
    // globally mirrored layout, which for cut material is the difference
    // between a left part and a right one. Both are counted.
    if (!(sa2 > 0)) flipped++;
    area3dTotal += 0.5 * twoA3;
    area2dTotal += Math.abs(sa2);
    const A3 = 0.5 * twoA3;
    if (A3 > 0) areaRatios.push(Math.abs(sa2) / A3);

    // Per-corner angles, 3D vs 2D.
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3, k = (i + 2) % 3;
      const u3 = sub3(P[j], P[i]), v3 = sub3(P[k], P[i]);
      const l3u = len3(u3), l3v = len3(v3);
      const a3 = Math.acos(Math.max(-1, Math.min(1, dot3(u3, v3) / (l3u * l3v))));
      const u2 = [Q[j][0] - Q[i][0], Q[j][1] - Q[i][1]];
      const v2 = [Q[k][0] - Q[i][0], Q[k][1] - Q[i][1]];
      const l2u = Math.hypot(u2[0], u2[1]), l2v = Math.hypot(v2[0], v2[1]);
      const a2 = (l2u > 0 && l2v > 0)
        ? Math.acos(Math.max(-1, Math.min(1, (u2[0] * v2[0] + u2[1] * v2[1]) / (l2u * l2v))))
        : 0;
      const d = Math.abs(a2 - a3);
      angleErrs.push(d * 180 / Math.PI);
      if (a3 > 1e-9) angleRelErrs.push(d / a3);
      angleSumAt[[a, b, c][i]] += a3;
    }
  }
  // Edge lengths, over unique mesh edges.
  const topo = buildMeshTopology(positions, faces);
  const edgeAbs = [], edgeRel = [];
  for (const [a, b] of topo.edges) {
    const l3 = len3(sub3(positions[b], positions[a]));
    const l2 = Math.hypot(uv[b][0] - uv[a][0], uv[b][1] - uv[a][1]);
    edgeAbs.push(Math.abs(l2 - l3));
    if (l3 > 0) edgeRel.push(Math.abs(l2 - l3) / l3);
  }

  // Discrete Gaussian curvature at interior vertices (angle defect).
  let maxDefect = 0, totalAbsDefect = 0, interiorCount = 0;
  for (let v = 0; v < positions.length; v++) {
    if (!topo.usedVertices.has(v) || topo.boundaryVertices.has(v)) continue;
    interiorCount++;
    const defect = 2 * Math.PI - angleSumAt[v];
    if (Math.abs(defect) > Math.abs(maxDefect)) maxDefect = defect;
    totalAbsDefect += Math.abs(defect);
  }

  const stat = (arr) => {
    if (!arr.length) return { max: 0, rms: 0, mean: 0 };
    let max = 0, sum = 0, sq = 0;
    for (const v of arr) { if (v > max) max = v; sum += v; sq += v * v; }
    return { max, rms: Math.sqrt(sq / arr.length), mean: sum / arr.length };
  };
  const aStat = stat(angleErrs), arStat = stat(angleRelErrs);
  const eaStat = stat(edgeAbs), erStat = stat(edgeRel);
  const ratioErrs = areaRatios.map((r) => Math.abs(r - 1));
  const raStat = stat(ratioErrs);

  return {
    angle: {
      maxDeg: aStat.max, rmsDeg: aStat.rms, meanDeg: aStat.mean,
      maxRel: arStat.max, rmsRel: arStat.rms,
      cornerCount: angleErrs.length,
    },
    edge: {
      count: edgeAbs.length,
      maxAbsErr: eaStat.max, rmsAbsErr: eaStat.rms,
      maxRelErr: erStat.max, rmsRelErr: erStat.rms,
    },
    area: {
      total3d: area3dTotal, total2d: area2dTotal,
      totalRatio: area3dTotal > 0 ? area2dTotal / area3dTotal : 0,
      minRatio: areaRatios.length ? arrayMin(areaRatios) : 0,
      maxRatio: areaRatios.length ? arrayMax(areaRatios) : 0,
      maxRelErr: raStat.max, rmsRelErr: raStat.rms,
    },
    intrinsic: {
      interiorVertexCount: interiorCount,
      maxAbsInteriorAngleDefect: Math.abs(maxDefect),
      signedMaxInteriorAngleDefect: maxDefect,
      totalAbsInteriorAngleDefect: totalAbsDefect,
    },
    flippedTriangles: flipped,
  };
}

// Length strain

// The 2x2 linear map J that takes a triangle from its own isometric frame
// (triangleLocalFrame) to its place in the layout: J = U X^-1, where the
// columns of X are the triangle's two edges from vertex 0 in the local frame
// and the columns of U are the same edges in uv. A unit length on the surface
// along a local direction t has length |J t| in the layout. X is upper
// triangular, X = [[x2, x3], [0, y3]], so its inverse is written out.
// Returns null for a triangle with no 3D area: it has no frame to map from.
function strainLayoutJacobian(positions, uv, a, b, c) {
  const fr = triangleLocalFrame(positions[a], positions[b], positions[c]);
  if (!fr || !(fr.y3 > 0)) return null;
  const u1 = uv[b][0] - uv[a][0], v1 = uv[b][1] - uv[a][1];
  const u2 = uv[c][0] - uv[a][0], v2 = uv[c][1] - uv[a][1];
  const i00 = 1 / fr.x2, i01 = -fr.x3 / (fr.x2 * fr.y3), i11 = 1 / fr.y3;
  return {
    j00: u1 * i00, j01: u1 * i01 + u2 * i11,
    j10: v1 * i00, j11: v1 * i01 + v2 * i11,
    area: 0.5 * fr.twoArea,
  };
}

// Area-weighted statistics of a per-triangle error (NaN = not measured, and
// left out). `percentAreaOver` is the share of the measured 3D area whose
// error exceeds `threshold`, in percent. Every field is null when no triangle
// was measured.
function strainAreaStats(err, area, threshold) {
  let max = -Infinity, sum = 0, total = 0, over = 0;
  for (let f = 0; f < err.length; f++) {
    const e = err[f];
    if (!(e === e)) continue;
    if (e > max) max = e;
    sum += e * area[f];
    total += area[f];
    if (e > threshold) over += area[f];
  }
  if (!(total > 0)) return { max: null, mean: null, percentAreaOver: null, threshold };
  return { max, mean: sum / total, percentAreaOver: 100 * over / total, threshold };
}

// Principal length strains of a layout, per triangle: the two singular values
// s1 >= s2 of J. A length on the surface becomes between s2 and s1 times as
// long in the layout, so s1 = s2 = 1 is no strain, s1 > 1 is stretch and
// s2 < 1 is compression. Unlike flatteningDistortion's edge-length errors this
// sees strain in every direction across a triangle, not only along its edges.
//
// Closed form: with E = (j00 + j11)/2, F = (j00 - j11)/2, G = (j10 + j01)/2,
// H = (j10 - j01)/2, J splits into a similarity of scale Q = |(E, H)| and an
// anti-similarity of scale R = |(F, G)|; s1 = Q + R, s2 = |Q - R| and
// det J = Q^2 - R^2. A non-positive det is a reversed triangle (a fold, or a
// mirrored layout); its singular values are the same as its mirror's, and it
// is counted in `flipped`.
//
// Per triangle: s1, s2 (NaN for a triangle with no 3D area), area (3D),
// flipped (1 = reversed). `summary` is over the measured triangles, with a
// triangle's error the larger of |s1 - 1| and |s2 - 1|:
//   maxStretch      max(s1) - 1
//   maxCompression  1 - min(s2)
//   max / mean / percentAreaOver   that error (mean area-weighted),
//                                  over `opts.threshold` (default 0.01)
export function principalStrains(positions, faces, uv, opts = {}) {
  const threshold = opts.threshold ?? 0.01;
  const n = faces.length;
  const s1 = new Float64Array(n), s2 = new Float64Array(n), area = new Float64Array(n);
  const flipped = new Uint8Array(n);
  const err = new Float64Array(n);
  let flippedCount = 0, maxS1 = -Infinity, minS2 = Infinity;
  for (let f = 0; f < n; f++) {
    const [a, b, c] = faces[f];
    const J = strainLayoutJacobian(positions, uv, a, b, c);
    if (!J) { s1[f] = s2[f] = err[f] = NaN; continue; }
    const E = (J.j00 + J.j11) / 2, F = (J.j00 - J.j11) / 2;
    const G = (J.j10 + J.j01) / 2, H = (J.j10 - J.j01) / 2;
    const Q = Math.hypot(E, H), R = Math.hypot(F, G);
    s1[f] = Q + R;
    s2[f] = Math.abs(Q - R);
    area[f] = J.area;
    if (!(J.j00 * J.j11 - J.j01 * J.j10 > 0)) { flipped[f] = 1; flippedCount++; }
    err[f] = Math.max(Math.abs(s1[f] - 1), Math.abs(s2[f] - 1));
    if (s1[f] > maxS1) maxS1 = s1[f];
    if (s2[f] < minS2) minS2 = s2[f];
  }
  const stats = strainAreaStats(err, area, threshold);
  return {
    s1, s2, area, flipped, flippedCount,
    summary: {
      maxStretch: stats.max === null ? null : maxS1 - 1,
      maxCompression: stats.max === null ? null : 1 - minS2,
      ...stats,
    },
  };
}

// Length strain along a grain and across it, per triangle, plus the change
// of the angle between the two thread directions.
//
// `grainDir` is one direction for every triangle, or an array with one per
// face. A 2-vector is a direction in the layout (the grain line drawn on the
// flat pattern); its cross grain is it turned +90 degrees in the layout, and
// both are carried onto the surface through J^-1. A 3-vector is a direction
// on the surface (e.g. a surface parameter's tangent); it is projected into
// the triangle's plane, its cross grain is its in-surface perpendicular, and
// both are carried into the layout through J. Either way the two threads are
// at 90 degrees on the side they were given and are compared on the other.
//
// Per triangle (NaN where the triangle has no 3D area, a 3D direction lies
// along the normal, or a layout direction is given and the triangle has no
// layout area, so J has no inverse):
//   sGrain, sCross  layout length / surface length along each thread; 1 is
//                   no strain, as in principalStrains
//   shearDeg        |angle between the threads on one side - on the other|
// Summary (area-weighted over the measured triangles):
//   grain, cross    stats of |s - 1| over `opts.threshold` (default 0.01),
//                   plus maxStretch = max(s) - 1, maxCompression = 1 - min(s)
//   shear           stats of shearDeg over `opts.shearThresholdDeg` (default 1)
// On a conformal layout (LSCM) J is close to a similarity, so the threads
// keep their angle and shearDeg stays near zero whatever the curvature:
// strain shows up as stretch along both threads, never as shear.
export function grainStrains(positions, faces, uv, grainDir, opts = {}) {
  const threshold = opts.threshold ?? 0.01;
  const shearThreshold = opts.shearThresholdDeg ?? 1;
  const perFace = Array.isArray(grainDir[0]);
  const n = faces.length;
  const sGrain = new Float64Array(n), sCross = new Float64Array(n);
  const shearDeg = new Float64Array(n), area = new Float64Array(n);
  const angle = (x0, y0, x1, y1) => {
    const c = (x0 * x1 + y0 * y1) / (Math.hypot(x0, y0) * Math.hypot(x1, y1));
    return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
  };
  for (let f = 0; f < n; f++) {
    const [a, b, c] = faces[f];
    const d = perFace ? grainDir[f] : grainDir;
    const J = strainLayoutJacobian(positions, uv, a, b, c);
    sGrain[f] = sCross[f] = shearDeg[f] = NaN;
    if (!J || !d) continue;
    area[f] = J.area;
    if (d.length === 2) {
      const det = J.j00 * J.j11 - J.j01 * J.j10;
      const L = Math.hypot(d[0], d[1]);
      if (!(Math.abs(det) > 0) || !(L > 0)) continue;
      const gx = d[0] / L, gy = d[1] / L, cx = -gy, cy = gx;
      // J^-1 = [[j11, -j01], [-j10, j00]] / det
      const ax = (J.j11 * gx - J.j01 * gy) / det, ay = (-J.j10 * gx + J.j00 * gy) / det;
      const bx = (J.j11 * cx - J.j01 * cy) / det, by = (-J.j10 * cx + J.j00 * cy) / det;
      sGrain[f] = 1 / Math.hypot(ax, ay);
      sCross[f] = 1 / Math.hypot(bx, by);
      shearDeg[f] = Math.abs(angle(ax, ay, bx, by) - 90);
    } else {
      const P0 = positions[a];
      const e1 = sub3(positions[b], P0), e2 = sub3(positions[c], P0);
      const l1 = len3(e1);
      const ex = [e1[0] / l1, e1[1] / l1, e1[2] / l1];
      const nrm = cross3(e1, e2);
      const ln = len3(nrm);
      const nz = [nrm[0] / ln, nrm[1] / ln, nrm[2] / ln];
      const ey = cross3(nz, ex);
      const tx = dot3(d, ex), ty = dot3(d, ey);
      const L = Math.hypot(tx, ty);
      if (!(L > 0)) continue;
      const gx = tx / L, gy = ty / L, cx = -gy, cy = gx;
      const ax = J.j00 * gx + J.j01 * gy, ay = J.j10 * gx + J.j11 * gy;
      const bx = J.j00 * cx + J.j01 * cy, by = J.j10 * cx + J.j11 * cy;
      sGrain[f] = Math.hypot(ax, ay);
      sCross[f] = Math.hypot(bx, by);
      shearDeg[f] = Math.abs(angle(ax, ay, bx, by) - 90);
    }
  }
  const threadStats = (s) => {
    const err = new Float64Array(n);
    let maxS = -Infinity, minS = Infinity;
    for (let f = 0; f < n; f++) {
      err[f] = Math.abs(s[f] - 1);
      if (s[f] > maxS) maxS = s[f];
      if (s[f] < minS) minS = s[f];
    }
    const st = strainAreaStats(err, area, threshold);
    return {
      maxStretch: st.max === null ? null : maxS - 1,
      maxCompression: st.max === null ? null : 1 - minS,
      ...st,
    };
  };
  return {
    sGrain, sCross, shearDeg, area,
    summary: {
      grain: threadStats(sGrain),
      cross: threadStats(sCross),
      shear: strainAreaStats(shearDeg, area, shearThreshold),
    },
  };
}

// Formability

// How a ForMatter formability record is judged against a layout, or null when
// it cannot be. `table` is ForMatter's band table: { bands: { name: { stretch } },
// marginal_from }. A record whose stretch band is 'rigid' is INEXTENSIBLE: the
// pattern is used at the layout's own scale and every triangle must keep its
// lengths within the band, stretched or compressed. Any other record is
// STRETCH-ONLY: the pattern may be cut smaller and pulled onto the shape, so it
// is judged on the stretch it needs, never on compression. A stretch given as
// { course, wale } is a knit with a band per direction.
//
// A record with `shear` is a WOVEN cloth: its threads keep their length and
// the cloth follows a shape by shearing the angle between warp and weft, up
// to the locking angle of its band in `table.shear_bands` ({ name:
// { locking_deg } }). It is judged on a Chebyshev drape (chebyshevDrape), never
// on the layout, and carries no stretch or compression band.
export function formabilityMode(record, table) {
  if (!record || !table || !table.bands) return null;
  if (record.shear !== undefined) {
    const b = typeof record.shear === 'string' && table.shear_bands ? table.shear_bands[record.shear] : null;
    return b && b.locking_deg > 0 && b.locking_deg < 90 && record.stretch === undefined && record.compression === undefined ? 'shear' : null;
  }
  const ok = (name) => typeof name === 'string' && table.bands[name] && table.bands[name].stretch > 0;
  const s = record.stretch;
  if (typeof s === 'string') {
    if (!ok(s)) return null;
    if (s !== 'rigid') return 'stretch';
    return record.compression == null || ok(record.compression) ? 'inextensible' : null;
  }
  if (s && typeof s === 'object' && ok(s.course) && ok(s.wale)) return 'stretch';
  return null;
}

// The radius a mesh bends at across each triangle, from the turn between it and
// each edge neighbor: across an interior edge the normals turn by theta over a
// strip as wide as the mean of the two triangles' heights off that edge, so the
// curvature across it is theta / ((h1 + h2) / 2). A triangle's radius is the
// smallest over its edges; Infinity where no neighbor turns (or none exists).
// On a cylinder of radius r tessellated along its rulings this gives r.
// `vertexOrigin` (cutToDisk's) maps each vertex to the one it was split from,
// so two triangles that meet across a seam are still neighbors.
export function faceBendRadii(positions, faces, vertexOrigin = null) {
  const n = faces.length;
  const normal = new Array(n), twoArea = new Float64Array(n);
  for (let f = 0; f < n; f++) {
    const [a, b, c] = faces[f];
    const nr = cross3(sub3(positions[b], positions[a]), sub3(positions[c], positions[a]));
    const l = len3(nr);
    twoArea[f] = l;
    normal[f] = l > 0 ? [nr[0] / l, nr[1] / l, nr[2] / l] : null;
  }
  const byEdge = new Map();
  for (let f = 0; f < n; f++) {
    const fc = faces[f];
    for (let k = 0; k < 3; k++) {
      const p = fc[k], q = fc[(k + 1) % 3];
      const key = vertexOrigin ? edgeKey(vertexOrigin[p], vertexOrigin[q]) : edgeKey(p, q);
      const list = byEdge.get(key);
      if (list) list.push(f); else byEdge.set(key, [f]);
    }
  }
  const radius = new Float64Array(n).fill(Infinity);
  for (const [key, list] of byEdge) {
    if (list.length !== 2) continue;
    const [f, g] = list;
    if (!normal[f] || !normal[g]) continue;
    const fg = faces[f];
    let el = 0;
    for (let k = 0; k < 3; k++) {
      const p = fg[k], q = fg[(k + 1) % 3];
      if ((vertexOrigin ? edgeKey(vertexOrigin[p], vertexOrigin[q]) : edgeKey(p, q)) === key) el = len3(sub3(positions[p], positions[q]));
    }
    if (!(el > 0)) continue;
    const theta = Math.acos(Math.max(-1, Math.min(1, dot3(normal[f], normal[g]))));
    if (!(theta > 1e-12)) continue;
    const r = ((twoArea[f] / el + twoArea[g] / el) / 2) / theta;
    if (r < radius[f]) radius[f] = r;
    if (r < radius[g]) radius[g] = r;
  }
  return radius;
}

// Woven drape

// The point of triangle (a, b, c) closest to p (Ericson, Real-Time Collision
// Detection, 5.1.5).
function drapeClosestOnTriangle(p, a, b, c) {
  const ab = sub3(b, a), ac = sub3(c, a), ap = sub3(p, a);
  const d1 = dot3(ab, ap), d2 = dot3(ac, ap);
  if (d1 <= 0 && d2 <= 0) return [a[0], a[1], a[2]];
  const bp = sub3(p, b), d3 = dot3(ab, bp), d4 = dot3(ac, bp);
  if (d3 >= 0 && d4 <= d3) return [b[0], b[1], b[2]];
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const t = d1 / (d1 - d3); return [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]]; }
  const cp = sub3(p, c), d5 = dot3(ab, cp), d6 = dot3(ac, cp);
  if (d6 >= 0 && d5 <= d6) return [c[0], c[1], c[2]];
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const t = d2 / (d2 - d6); return [a[0] + t * ac[0], a[1] + t * ac[1], a[2] + t * ac[2]]; }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return [b[0] + t * (c[0] - b[0]), b[1] + t * (c[1] - b[1]), b[2] + t * (c[2] - b[2])];
  }
  const den = 1 / (va + vb + vc), v = vb * den, w = vc * den;
  return [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w];
}

// A woven cloth laid on a triangle mesh as a Chebyshev net (the fishnet, or
// kinematic, drape: Mack and Taylor, "The fitting of woven cloth to surfaces",
// J. Textile Inst. 47, 1956). The threads keep their length and only the angle
// between warp and weft changes, which is how a woven cloth follows a curve.
//
// The net starts at the mesh's deepest vertex (farthest from its boundary
// along the edges; on a closed mesh, the vertex nearest the area centroid).
// Two generator threads leave it at 90 degrees in the tangent plane, turned
// `opts.orientationDeg` from its shortest edge (on a grid tessellation, a U or
// V grid line), and run straight across the mesh in steps of `step`: each next
// point is the one at chord length `step` nearest the continuation of the last
// chord. Each quadrant between two half-generators is then filled cell by
// cell: given the corner C and its two neighbors A (along the first thread)
// and B (along the second), the fourth corner is the mesh point at chord
// length `step` from both A and B on the far side of AB; it is found by
// alternating between the circle of points at that distance from A and B and
// the mesh. Every cell is a rhombus of side `step`, so the flat cloth is a
// square grid and the net keeps both threads' lengths everywhere.
//
// A cell's shear is |90 - its corner angle| in degrees, the larger of the
// angles at C and at the new corner. A cell whose corner cannot be placed (A
// and B 2 steps apart or coincident, the corner folding back over AB, or a
// shear of 90) is LOCKED: it reads 90 and nothing grows past it. A corner
// that lands beyond the mesh's boundary is cloth hanging past the piece; the
// cell reads its angle at C and the net stops there. Cut seams are not
// boundaries when `opts.vertexOrigin` (cutToDisk's) is given: the cloth is one
// piece over the uncut surface.
//
// The shear of each triangle is the largest of the cells with a corner on it;
// a triangle no corner landed on takes the value of the nearest one that has
// one (by face adjacency).
//
// opts: vertexOrigin, orientationDeg (default 0), step (default the mean of
// each triangle's two shorter edges), maxNodes (default 60000; the step grows
// until the mesh area holds no more cells than that).
// Returns { shearDeg (per face, NaN where unreached), area (per face), step,
// orientationDeg, center ([x,y,z]), nodeCount, lockedCells, summary: { maxDeg,
// meanDeg (area-weighted) } } or null when the mesh has no faces.
export function chebyshevDrape(positions, faces, opts = {}) {
  const n = faces.length;
  if (!n) return null;
  const origin = opts.vertexOrigin || null;
  const key = (v) => (origin ? origin[v] : v);
  const nbr = new Int32Array(3 * n).fill(-1);
  const byEdge = new Map();
  for (let f = 0; f < n; f++) {
    for (let k = 0; k < 3; k++) {
      const ek = edgeKey(key(faces[f][(k + 1) % 3]), key(faces[f][(k + 2) % 3]));
      const list = byEdge.get(ek);
      if (list) list.push(3 * f + k); else byEdge.set(ek, [3 * f + k]);
    }
  }
  const boundaryKeys = new Set();
  for (const [ek, list] of byEdge) {
    if (list.length === 2) { nbr[list[0]] = (list[1] / 3) | 0; nbr[list[1]] = (list[0] / 3) | 0; }
    else if (list.length === 1) { const f = (list[0] / 3) | 0, k = list[0] % 3; boundaryKeys.add(key(faces[f][(k + 1) % 3])); boundaryKeys.add(key(faces[f][(k + 2) % 3])); }
  }
  const area = new Float64Array(n), normal = new Array(n);
  let areaSum = 0, shortSum = 0, shortCount = 0;
  const centroid = [0, 0, 0];
  for (let f = 0; f < n; f++) {
    const [a, b, c] = faces[f].map((i) => positions[i]);
    const nr = cross3(sub3(b, a), sub3(c, a));
    const l = len3(nr);
    area[f] = l / 2;
    normal[f] = l > 0 ? [nr[0] / l, nr[1] / l, nr[2] / l] : [0, 0, 0];
    areaSum += area[f];
    for (let k = 0; k < 3; k++) centroid[k] += area[f] * (a[k] + b[k] + c[k]) / 3;
    const e = [len3(sub3(b, a)), len3(sub3(c, b)), len3(sub3(a, c))].sort((x, y) => x - y);
    if (l > 0) { shortSum += (e[0] + e[1]) / 2; shortCount++; }
  }
  if (!(areaSum > 0)) return null;
  for (let k = 0; k < 3; k++) centroid[k] /= areaSum;
  const maxNodes = opts.maxNodes ?? 60000;
  let h = opts.step > 0 ? opts.step : shortSum / shortCount;
  if (areaSum / (h * h) > maxNodes) h = Math.sqrt(areaSum / maxNodes);

  // Vertex graph over origin keys: rings, a representative vertex and face.
  const adj = new Map(), rep = new Map(), faceOf = new Map();
  for (let f = 0; f < n; f++) {
    for (let k = 0; k < 3; k++) {
      const v = faces[f][k], kv = key(v);
      if (!rep.has(kv)) { rep.set(kv, v); faceOf.set(kv, f); adj.set(kv, new Set()); }
      adj.get(kv).add(key(faces[f][(k + 1) % 3]));
      adj.get(kv).add(key(faces[f][(k + 2) % 3]));
    }
  }
  const at = (kv) => positions[rep.get(kv)];
  // Dijkstra along edges from a set of sources; returns Map key -> distance.
  const distancesFrom = (sources) => {
    const dist = new Map();
    const heap = [];
    const push = (d, kv) => {
      heap.push([d, kv]);
      let i = heap.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; }
    };
    const pop = () => {
      const top = heap[0], last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
          if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
          if (m === i) break;
          [heap[m], heap[i]] = [heap[i], heap[m]]; i = m;
        }
      }
      return top;
    };
    for (const s of sources) { dist.set(s, 0); push(0, s); }
    while (heap.length) {
      const [d, kv] = pop();
      if (d > dist.get(kv)) continue;
      for (const w of adj.get(kv)) {
        const nd = d + len3(sub3(at(kv), at(w)));
        if (!(dist.has(w) && dist.get(w) <= nd)) { dist.set(w, nd); push(nd, w); }
      }
    }
    return dist;
  };
  let centerKey = null;
  if (boundaryKeys.size) {
    let best = -1;
    for (const [kv, d] of distancesFrom(boundaryKeys)) if (d > best) { best = d; centerKey = kv; }
  } else {
    let best = Infinity;
    for (const kv of rep.keys()) { const d = len3(sub3(at(kv), centroid)); if (d < best) { best = d; centerKey = kv; } }
  }
  let reach = 0;
  for (const d of distancesFrom([centerKey]).values()) if (d > reach) reach = d;
  const maxSteps = Math.ceil(reach / h) + 2;

  // The point of the mesh nearest p, found by walking from face f0 toward p
  // across the edge p lies beyond. `off` is set when that edge is a boundary.
  const locate = (p, f0) => {
    let f = f0, prev = -1;
    const seen = new Set();
    for (;;) {
      const [A, B, C] = faces[f].map((i) => positions[i]);
      const e1 = sub3(B, A), e2 = sub3(C, A), w = sub3(p, A);
      const d00 = dot3(e1, e1), d01 = dot3(e1, e2), d11 = dot3(e2, e2), d20 = dot3(w, e1), d21 = dot3(w, e2);
      const den = d00 * d11 - d01 * d01;
      const bv = den > 0 ? (d11 * d20 - d01 * d21) / den : 0, bw = den > 0 ? (d00 * d21 - d01 * d20) / den : 0;
      const bary = [1 - bv - bw, bv, bw];
      let k = 0;
      if (bary[1] < bary[k]) k = 1;
      if (bary[2] < bary[k]) k = 2;
      if (bary[k] >= -1e-12) return { f, q: [A[0] + bv * e1[0] + bw * e2[0], A[1] + bv * e1[1] + bw * e2[1], A[2] + bv * e1[2] + bw * e2[2]], off: false };
      const g = nbr[3 * f + k];
      seen.add(f);
      if (g < 0) return { f, q: drapeClosestOnTriangle(p, A, B, C), off: true };
      if (g === prev || seen.has(g) || seen.size > 4096) return { f, q: drapeClosestOnTriangle(p, A, B, C), off: false };
      prev = f; f = g;
    }
  };

  // The center, its tangent plane and the two thread directions.
  const X0 = at(centerKey);
  let f0 = faceOf.get(centerKey);
  const nc = [0, 0, 0];
  for (let f = 0; f < n; f++) {
    if (!faces[f].some((v) => key(v) === centerKey)) continue;
    for (let k = 0; k < 3; k++) nc[k] += area[f] * normal[f][k];
  }
  const ncl = len3(nc);
  const N0 = ncl > 0 ? [nc[0] / ncl, nc[1] / ncl, nc[2] / ncl] : normal[f0];
  let ref = null, refLen = Infinity;
  for (const w of adj.get(centerKey)) {
    const e = sub3(at(w), X0);
    const t = sub3(e, [N0[0] * dot3(e, N0), N0[1] * dot3(e, N0), N0[2] * dot3(e, N0)]);
    const l = len3(e), lt = len3(t);
    if (lt > 0 && l < refLen) { refLen = l; ref = [t[0] / lt, t[1] / lt, t[2] / lt]; }
  }
  if (!ref) return null;
  const th = (opts.orientationDeg ?? 0) * Math.PI / 180;
  const side = cross3(N0, ref);
  const d1 = [0, 1, 2].map((k) => Math.cos(th) * ref[k] + Math.sin(th) * side[k]);
  const d2 = cross3(N0, d1);

  const scaleAdd = (P, d, s) => [P[0] + d[0] * s, P[1] + d[1] * s, P[2] + d[2] * s];
  const trace = (dir) => {
    const pts = [X0], fs = [f0];
    let P = X0, f = f0, d = dir;
    for (let s = 0; s < maxSteps; s++) {
      let G = scaleAdd(P, d, h), r = null;
      for (let it = 0; it < 30; it++) {
        r = locate(G, f);
        const t = sub3(r.q, P), l = len3(t);
        if (!(l > 0)) break;
        const G2 = scaleAdd(P, t, h / l);
        const moved = len3(sub3(G2, G));
        G = G2;
        if (moved < 1e-10 * h) break;
      }
      r = locate(G, r ? r.f : f);
      if (r.off) break;
      const t = sub3(r.q, P), l = len3(t);
      if (!(l > 0.5 * h)) break;
      d = [t[0] / l, t[1] / l, t[2] / l];
      P = r.q; f = r.f;
      pts.push(P); fs.push(f);
    }
    return { pts, fs };
  };
  const minus = (v) => [-v[0], -v[1], -v[2]];
  const g1 = [trace(d1), trace(minus(d1))], g2 = [trace(d2), trace(minus(d2))];

  const shearDeg = new Float64Array(n).fill(NaN);
  const splat = (f, s) => { if (!(shearDeg[f] >= s)) shearDeg[f] = s; };
  const angleDeg = (u, v) => {
    const c = dot3(u, v) / (len3(u) * len3(v));
    return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
  };
  let nodeCount = 1, lockedCells = 0;
  for (const ga of g1) {
    for (const gb of g2) {
      const rows = ga.pts.length, cols = gb.pts.length;
      nodeCount += rows - 1 + cols - 1;
      const X = [], F = [];
      for (let i = 0; i < rows; i++) { X.push(new Array(cols).fill(null)); F.push(new Int32Array(cols).fill(-1)); }
      for (let i = 0; i < rows; i++) { X[i][0] = ga.pts[i]; F[i][0] = ga.fs[i]; }
      for (let j = 0; j < cols; j++) { X[0][j] = gb.pts[j]; F[0][j] = gb.fs[j]; }
      for (let i = 1; i < rows; i++) {
        for (let j = 1; j < cols; j++) {
          const C = X[i - 1][j - 1], A = X[i - 1][j], B = X[i][j - 1];
          if (!C || !A || !B) continue;
          const shearC = Math.abs(90 - angleDeg(sub3(A, C), sub3(B, C)));
          const lock = () => { lockedCells++; for (const g of [F[i - 1][j - 1], F[i - 1][j], F[i][j - 1]]) splat(g, 90); };
          const ab = sub3(B, A), dab = len3(ab);
          if (!(dab > 1e-9 * h) || dab >= 2 * h * (1 - 1e-9)) { lock(); continue; }
          const M = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2];
          const e = [ab[0] / dab, ab[1] / dab, ab[2] / dab];
          const rho = Math.sqrt(h * h - dab * dab / 4);
          let G = [A[0] + B[0] - C[0], A[1] + B[1] - C[1], A[2] + B[2] - C[2]];
          let f = F[i - 1][j], r = null, lost = false;
          for (let it = 0; it < 40; it++) {
            r = locate(G, f); f = r.f;
            const v = sub3(r.q, M), ve = dot3(v, e);
            const vp = [v[0] - ve * e[0], v[1] - ve * e[1], v[2] - ve * e[2]], lv = len3(vp);
            if (!(lv > 1e-12 * h)) { lost = true; break; }
            const G2 = scaleAdd(M, vp, rho / lv);
            const moved = len3(sub3(G2, G));
            G = G2;
            if (moved < 1e-10 * h) break;
          }
          if (lost) { lock(); continue; }
          r = locate(G, f);
          if (r.off) { for (const g of [F[i - 1][j - 1], F[i - 1][j], F[i][j - 1]]) splat(g, shearC); continue; }
          const Xn = r.q;
          if (dot3(sub3(Xn, M), sub3(C, M)) > 0) { lock(); continue; }
          const s = Math.max(shearC, Math.abs(90 - angleDeg(sub3(A, Xn), sub3(B, Xn))));
          if (!(s < 90)) { lock(); continue; }
          X[i][j] = Xn; F[i][j] = r.f; nodeCount++;
          for (const g of [F[i - 1][j - 1], F[i - 1][j], F[i][j - 1], r.f]) splat(g, s);
        }
      }
    }
  }

  // Triangles no corner landed on take the nearest reached triangle's value.
  let frontier = [];
  for (let f = 0; f < n; f++) if (shearDeg[f] === shearDeg[f]) frontier.push(f);
  while (frontier.length) {
    const next = [];
    for (const f of frontier) {
      for (let k = 0; k < 3; k++) {
        const g = nbr[3 * f + k];
        if (g >= 0 && shearDeg[g] !== shearDeg[g]) { shearDeg[g] = shearDeg[f]; next.push(g); }
      }
    }
    frontier = next;
  }
  let maxDeg = -Infinity, wsum = 0, asum = 0;
  for (let f = 0; f < n; f++) {
    if (shearDeg[f] !== shearDeg[f] || !(area[f] > 0)) continue;
    if (shearDeg[f] > maxDeg) maxDeg = shearDeg[f];
    wsum += area[f] * shearDeg[f]; asum += area[f];
  }
  return {
    shearDeg, area, step: h, orientationDeg: opts.orientationDeg ?? 0, center: [X0[0], X0[1], X0[2]],
    nodeCount, lockedCells,
    summary: { maxDeg: asum > 0 ? maxDeg : null, meanDeg: asum > 0 ? wsum / asum : null },
  };
}

// The verdict of one formability record on one layout.
//   strains   principalStrains(positions, faces, uv)
//   opts.grain          grainStrains(positions, faces, uv, [1, 0]): stretch along
//                       the layout's x (sGrain) and y (sCross); needed for a knit
//   opts.curvatureMin   the surface's smallest radius of curvature, or null
//                       when it is flat everywhere
//   opts.faceRadius     faceBendRadii of the same mesh, for the bend shading
//   opts.drape          chebyshevDrape of the same mesh, or several (one per
//                       orientation of the threads); needed for a woven record
// Per triangle a strain is compared with its band as a ratio (strain / band):
// past 1 the triangle fails, past `marginal_from` it is marginal. A reversed
// (folded) triangle fails.
//   inextensible   stretch s1 - 1 against the stretch band and compression
//                  1 - s2 against the compression band
//   stretch-only   the pattern is scaled by 1 / max(s1), the largest scale that
//                  compresses nothing; a direction t then needs
//                  max(s1) / s(t) - 1 of stretch, worst across s2. A knit is
//                  judged along the layout's two axes, each band on each axis
//                  in turn (grain 'u' = courses along x, 'v' = along y), and
//                  the orientation that fails less is reported.
//   bend           where the record carries min_bend_radius_t and
//                  nominal_thickness_mm, the smallest radius of curvature
//                  against their product (marginal below 1 / marginal_from of
//                  it); triangles bent tighter than it fail.
//   woven (shear)  each triangle's drape shear against the band's locking
//                  angle; of several drapes the one that fails less is
//                  reported (orientationDeg), with needShearDeg its largest
//                  shear and lockingDeg the band's angle. The layout's
//                  strains give only which triangles are measured.
// Returns null when the record has no mode (formabilityMode), or is woven
// with no drape of this mesh, else
// { mode, verdict: 'ok' | 'marginal' | 'fails', fracFail, fracMarginal (shares
// of the measured area), needStretch, band, bandStretch, thread ('course' |
// 'wale' | null), grain ('u' | 'v' | null), bend ('ok' | 'marginal' | 'fails' |
// null), bendRadius, bendRequired, binding ('strain' | 'bend' | 'shear'),
// faceState (0 ok, 1 marginal, 2 fails) }; a woven verdict adds lockingDeg,
// needShearDeg and orientationDeg, with needStretch and bandStretch null.
export function materialVerdict(strains, record, table, opts = {}) {
  const mode = formabilityMode(record, table);
  if (!mode) return null;
  const m = table.marginal_from > 0 && table.marginal_from < 1 ? table.marginal_from : 0.8;
  const bandOf = (name) => table.bands[name].stretch;
  const { s1, s2, area, flipped } = strains;
  const n = s1.length;
  const measured = (f) => s1[f] === s1[f];
  let maxS1 = -Infinity;
  for (let f = 0; f < n; f++) if (measured(f) && s1[f] > maxS1) maxS1 = s1[f];

  // ratio[f] and the strain that set it, for one way of reading the layout
  const judge = (strainAt) => {
    const ratio = new Float64Array(n);
    let worst = -Infinity, need = 0, band = null;
    for (let f = 0; f < n; f++) {
      if (!measured(f)) { ratio[f] = NaN; continue; }
      const [r, e, b] = strainAt(f);
      ratio[f] = flipped && flipped[f] ? Infinity : r;
      if (r > worst) { worst = r; need = e; band = b; }
    }
    return { ratio, need, band };
  };
  const shares = (state) => {
    let total = 0, fail = 0, marg = 0;
    for (let f = 0; f < n; f++) {
      if (!measured(f)) continue;
      total += area[f];
      if (state[f] === 2) fail += area[f]; else if (state[f] === 1) marg += area[f];
    }
    return total > 0 ? { fracFail: fail / total, fracMarginal: marg / total } : { fracFail: null, fracMarginal: null };
  };
  const stateOf = (ratio) => {
    const st = new Uint8Array(n);
    for (let f = 0; f < n; f++) st[f] = ratio[f] > 1 ? 2 : ratio[f] > m ? 1 : 0;
    return st;
  };

  if (mode === 'shear') {
    const lock = table.shear_bands[record.shear].locking_deg;
    const drapes = (Array.isArray(opts.drape) ? opts.drape : [opts.drape]).filter((d) => d && d.shearDeg && d.shearDeg.length === n);
    if (!drapes.length) return null;
    let best = null;
    for (const d of drapes) {
      const ratio = new Float64Array(n);
      let worst = -Infinity;
      for (let f = 0; f < n; f++) {
        ratio[f] = measured(f) ? d.shearDeg[f] / lock : NaN;
        if (ratio[f] > worst) worst = ratio[f];
      }
      const st = stateOf(ratio), sh = shares(st);
      const c = { d, st, sh, worst };
      if (!best || sh.fracFail < best.sh.fracFail
        || (sh.fracFail === best.sh.fracFail && (sh.fracMarginal < best.sh.fracMarginal
          || (sh.fracMarginal === best.sh.fracMarginal && worst < best.worst)))) best = c;
    }
    const v = best.worst > 1 ? 2 : best.worst > m ? 1 : 0;
    return {
      mode,
      verdict: v === 2 ? 'fails' : v === 1 ? 'marginal' : 'ok',
      fracFail: best.sh.fracFail, fracMarginal: best.sh.fracMarginal,
      needStretch: null, band: record.shear, bandStretch: null,
      lockingDeg: lock, needShearDeg: best.worst === -Infinity ? null : best.worst * lock, orientationDeg: best.d.orientationDeg,
      thread: null, grain: null,
      bend: null, bendRadius: null, bendRequired: null,
      binding: 'shear',
      faceState: best.st,
    };
  }

  let read, thread = null, grain = null;
  if (mode === 'inextensible') {
    const bs = bandOf(record.stretch);
    const cName = record.compression ?? record.stretch, bc = bandOf(cName);
    read = judge((f) => {
      const st = s1[f] - 1, co = 1 - s2[f];
      return st / bs >= co / bc ? [st / bs, st, record.stretch] : [co / bc, co, cName];
    });
  } else if (typeof record.stretch === 'string') {
    const b = bandOf(record.stretch);
    read = judge((f) => { const e = maxS1 / s2[f] - 1; return [e / b, e, record.stretch]; });
  } else {
    const g = opts.grain;
    if (!g || !g.sGrain || g.sGrain.length !== n) return null;
    const bc = bandOf(record.stretch.course), bw = bandOf(record.stretch.wale);
    const along = (courseS, waleS) => judge((f) => {
      const ec = maxS1 / courseS[f] - 1, ew = maxS1 / waleS[f] - 1;
      return ec / bc >= ew / bw ? [ec / bc, ec, 'course'] : [ew / bw, ew, 'wale'];
    });
    const u = along(g.sGrain, g.sCross), v = along(g.sCross, g.sGrain);
    const su = shares(stateOf(u.ratio)), sv = shares(stateOf(v.ratio));
    const worstOf = (x) => { let w = -Infinity; for (const r of x.ratio) if (r > w) w = r; return w; };
    const pickV = sv.fracFail < su.fracFail
      || (sv.fracFail === su.fracFail && (sv.fracMarginal < su.fracMarginal
        || (sv.fracMarginal === su.fracMarginal && worstOf(v) < worstOf(u))));
    read = pickV ? v : u;
    grain = pickV ? 'v' : 'u';
    thread = read.band;
    read = { ...read, band: record.stretch[thread] };
  }

  const state = stateOf(read.ratio);
  let bend = null, bendRadius = null, bendRequired = null, bendRatio = 0;
  if (record.min_bend_radius_t > 0 && record.nominal_thickness_mm > 0) {
    bendRequired = record.min_bend_radius_t * record.nominal_thickness_mm;
    bendRadius = opts.curvatureMin != null && opts.curvatureMin > 0 ? opts.curvatureMin : null;
    bendRatio = bendRadius == null ? 0 : bendRequired / bendRadius;
    bend = bendRatio > 1 ? 'fails' : bendRatio > m ? 'marginal' : 'ok';
    const fr = opts.faceRadius;
    if (fr && fr.length === n) {
      for (let f = 0; f < n; f++) {
        if (!measured(f)) continue;
        const r = bendRequired / fr[f];
        const s = r > 1 ? 2 : r > m ? 1 : 0;
        if (s > state[f]) state[f] = s;
      }
    }
  }
  const { fracFail, fracMarginal } = shares(state);
  let strainWorst = -Infinity;
  for (const r of read.ratio) if (r > strainWorst) strainWorst = r;
  const strainVerdict = strainWorst > 1 ? 2 : strainWorst > m ? 1 : 0;
  const bendVerdict = bend === 'fails' ? 2 : bend === 'marginal' ? 1 : 0;
  const worst = Math.max(strainVerdict, bendVerdict, fracFail > 0 ? 2 : fracMarginal > 0 ? 1 : 0);
  return {
    mode,
    verdict: worst === 2 ? 'fails' : worst === 1 ? 'marginal' : 'ok',
    fracFail, fracMarginal,
    needStretch: read.need,
    band: read.band, bandStretch: bandOf(read.band),
    thread, grain,
    bend, bendRadius, bendRequired,
    binding: bendRatio > strainWorst ? 'bend' : 'strain',
    faceState: state,
  };
}

// LSCM

// Pick the two pinned vertices: the (approximate) diameter pair of the
// mesh, found by two farthest-point passes. Pinning two vertices that are
// far apart keeps the fixed sub-block well conditioned; pinning two
// neighbors would fix the layout's scale off a single short edge and
// amplify that edge's own error across the whole result.
function choosePins(positions, used) {
  const list = [...used];
  let a = list[0];
  // pass 1: farthest from an arbitrary seed
  let best = -1;
  let far = a;
  for (const i of list) {
    const d = len3(sub3(positions[i], positions[a]));
    if (d > best) { best = d; far = i; }
  }
  a = far;
  // pass 2: farthest from that
  best = -1;
  let b = list[0] === a ? list[1] : list[0];
  for (const i of list) {
    if (i === a) continue;
    const d = len3(sub3(positions[i], positions[a]));
    if (d > best) { best = d; b = i; }
  }
  return { pinA: a, pinB: b, pinDistance: best };
}

// Flatten an indexed triangle mesh into the plane via LSCM.
//
// mesh: { positions: [[x,y,z], ...], faces: [[i,j,k], ...] }
// opts: { tolerance, maxIterations, pinA, pinB }
//
// Returns:
//   { uv, positions, faces, pins, solver, distortion, developable }
// where `uv` is one [u,v] per input vertex in the same length unit as the
// input positions, `distortion` is the full measured report from
// flatteningDistortion, and `developable` is a plain-language summary of
// whether an exact answer was possible for this input.
//
// Never claims exactness for a curved input: `distortion` is always
// populated and always measured.
export function flattenLSCM(mesh, opts = {}) {
  const { positions, faces } = mesh || {};
  const topo = validateFlattenMesh(positions, faces, opts);

  const { pinA, pinB, pinDistance } = (opts.pinA != null && opts.pinB != null)
    ? { pinA: opts.pinA, pinB: opts.pinB, pinDistance: len3(sub3(positions[opts.pinB], positions[opts.pinA])) }
    : choosePins(positions, topo.usedVertices);
  if (pinA === pinB || !(pinDistance > 0)) {
    throw new Error('flattenLSCM: refusing to flatten — could not find two distinct vertices to pin (the mesh collapses to a single point)');
  }

  // The two pinned vertices fix the layout's translation (2), rotation (1)
  // and scale (1) — the 4-dimensional nullspace of the conformal
  // energy. Their pinned distance is arbitrary (the whole layout is
  // rescaled below to best fit true edge lengths); the true 3D chord
  // distance is used simply so the intermediate numbers stay near unit
  // scale.
  const pinned = new Map([[pinA, [0, 0]], [pinB, [pinDistance, 0]]]);

  // Map every free vertex to a pair of unknowns (real, imaginary).
  const freeIndex = new Int32Array(positions.length).fill(-1);
  let nFree = 0;
  for (let v = 0; v < positions.length; v++) {
    if (!topo.usedVertices.has(v)) continue;
    if (pinned.has(v)) continue;
    freeIndex[v] = nFree++;
  }
  if (nFree < 1) {
    throw new Error('flattenLSCM: refusing to flatten — every vertex is pinned, there is nothing left to solve for');
  }

  // Assemble the least-squares system. Per triangle, Levy et al. 2002's
  // conformality condition in complex form is  sum_j W_j * U_j = 0, with
  //   W_1 = (x3 - x2) + i*(y3 - y2)
  //   W_2 = (x1 - x3) + i*(y1 - y3)
  //   W_3 = (x2 - x1) + i*(y2 - y1)
  // read off the triangle's own isometric local frame, and U_j the unknown
  // complex 2D position of vertex j. Splitting into real and imaginary
  // parts gives 2 real equations per triangle. Every row is scaled by
  // 1/sqrt(2*area) so a triangle's contribution to the energy is area
  // weighted (and, as a side effect, so every matrix entry is O(1) — see
  // the solver's own conditioning note).
  const triplets = [];
  const rhs = new Float64Array(2 * faces.length);
  for (let f = 0; f < faces.length; f++) {
    const [i0, i1, i2] = faces[f];
    const fr = triangleLocalFrame(positions[i0], positions[i1], positions[i2]);
    if (!fr || !(fr.twoArea > 0)) {
      throw new Error(`flattenLSCM: refusing to flatten — face ${f} is a DEGENERATE TRIANGLE (zero area in its own plane)`);
    }
    const scale = 1 / Math.sqrt(fr.twoArea);
    const W = [
      [fr.x3 - fr.x2, fr.y3 - fr.y2],
      [fr.x1 - fr.x3, fr.y1 - fr.y3],
      [fr.x2 - fr.x1, fr.y2 - fr.y1],
    ];
    const rowRe = 2 * f, rowIm = 2 * f + 1;
    const verts = [i0, i1, i2];
    for (let j = 0; j < 3; j++) {
      const wre = W[j][0] * scale, wim = W[j][1] * scale;
      const v = verts[j];
      if (pinned.has(v)) {
        const [pu, pv] = pinned.get(v);
        // Move the known contribution to the right-hand side.
        rhs[rowRe] -= wre * pu - wim * pv;
        rhs[rowIm] -= wim * pu + wre * pv;
      } else {
        const c = 2 * freeIndex[v];
        triplets.push({ row: rowRe, col: c, val: wre });
        triplets.push({ row: rowRe, col: c + 1, val: -wim });
        triplets.push({ row: rowIm, col: c, val: wim });
        triplets.push({ row: rowIm, col: c + 1, val: wre });
      }
    }
  }

  const solved = solveSparseLeastSquares(triplets, rhs, 2 * nFree, opts);
  if (!solved.converged) {
    throw new Error(`flattenLSCM: the sparse solve did not converge (relative residual ${solved.residual.toExponential(3)} after ${solved.iterations} iterations) — refusing to return a layout that was never actually solved`);
  }

  const uv = new Array(positions.length);
  for (let v = 0; v < positions.length; v++) {
    if (pinned.has(v)) { uv[v] = pinned.get(v).slice(); continue; }
    if (freeIndex[v] < 0) { uv[v] = [0, 0]; continue; } // unused by any face
    uv[v] = [solved.x[2 * freeIndex[v]], solved.x[2 * freeIndex[v] + 1]];
  }

  // A conformal map is only defined up to a global similarity, so the raw
  // solution has an arbitrary overall scale. Fix it with the single uniform
  // factor s minimizing sum over mesh edges of (s*L2d - L3d)^2, i.e.
  //   s = sum(L2d*L3d) / sum(L2d^2).
  // This is the normalization to measure length/area distortion
  // against: it never hides distortion (a curved input still
  // cannot match every edge), it only removes the arbitrary global scale
  // the algorithm itself does not determine.
  let num = 0, den = 0;
  for (const [a, b] of topo.edges) {
    const l3 = len3(sub3(positions[b], positions[a]));
    const l2 = Math.hypot(uv[b][0] - uv[a][0], uv[b][1] - uv[a][1]);
    num += l2 * l3; den += l2 * l2;
  }
  const fitScale = den > 0 ? num / den : 1;
  if (!Number.isFinite(fitScale) || fitScale <= 0) {
    throw new Error('flattenLSCM: the solved layout collapsed to zero extent — refusing to return it');
  }
  let minU = Infinity, minV = Infinity;
  for (let v = 0; v < uv.length; v++) {
    uv[v] = [uv[v][0] * fitScale, uv[v][1] * fitScale];
    if (topo.usedVertices.has(v)) {
      if (uv[v][0] < minU) minU = uv[v][0];
      if (uv[v][1] < minV) minV = uv[v][1];
    }
  }
  // Translate so the used layout's bounding box starts at (0,0) — a stable,
  // predictable frame for a downstream consumer (a cut file, a nesting
  // pass) to build on. Translation is a similarity, so it changes no metric.
  for (let v = 0; v < uv.length; v++) uv[v] = [uv[v][0] - minU, uv[v][1] - minV];

  for (const p of uv) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) {
      throw new Error('flattenLSCM: a flattened coordinate came out non-finite — refusing to return it');
    }
  }

  const distortion = flatteningDistortion(positions, faces, uv);

  // A plain-language verdict — never "exact" unless the input's own
  // discrete Gaussian curvature says an exact answer was possible.
  const defect = distortion.intrinsic.maxAbsInteriorAngleDefect;
  const developableTol = opts.developableTolerance != null ? opts.developableTolerance : 1e-9;
  const developable = defect <= developableTol;
  const note = developable
    ? 'Input is DEVELOPABLE to within tolerance (its discrete Gaussian curvature is zero everywhere), so an essentially exact unrolling exists; the residual distortion below is numerical noise.'
    : `Input is genuinely DOUBLY CURVED (max interior angle defect ${defect.toExponential(3)} rad). By Gauss's Theorema Egregium NO flattening of it can be distortion-free; the numbers below are the real, measured cost. LSCM minimized ANGLE distortion, not stretch — read the area and edge figures before cutting material.`;

  return {
    uv,
    positions,
    faces,
    pins: { a: pinA, b: pinB, distance: pinDistance },
    solver: { iterations: solved.iterations, residual: solved.residual, converged: solved.converged, unknowns: 2 * nFree, equations: 2 * faces.length },
    scaleApplied: fitScale,
    distortion,
    developable,
    note,
  };
}

// ARAP

// As-rigid-as-possible refinement of a layout (Liu, Zhang, Xu, Gotsman,
// Gortler 2008), warm-started from `uv` (an LSCM layout). It minimizes
//   sum over triangles of area * |J - R|^2
// where J is the triangle's layout Jacobian (strainLayoutJacobian) and R the
// rotation nearest J. Local step: R per triangle from the similarity part of
// J (the same split principalStrains uses). Global step: the normal
// equations of that sum, the cotangent Laplacian, which is positive
// semidefinite whatever the triangle shapes because the energy is a sum of
// squares; one vertex is held, so it is definite. The Laplacian does not
// change between rounds, so it is factored once (arapBandFactor) when its
// band is narrow, as a gore's is, and every global step is exact; a wider
// piece (n * band^2 past `opts.bandWork`, default 4e8) is solved by Jacobi-
// preconditioned CG from the current layout, at most `opts.cgIterations`
// steps (default 100) or until the residual falls by `opts.cgTolerance`
// (default 1e-10). A thin piece needs hundreds of CG steps per solve, which
// a 100-step cap leaves unconverged.
//
// The cost is fixed: `opts.iterations` local/global rounds (default 10). A
// round never makes the energy larger: the local step is optimal for the
// layout, the global step for the rotations.
//
// Returns { uv, iterations, energyStart, energyEnd } with energy in area
// units (0 is an isometry). `uv` is a new array; the input is not touched.
export function arapRefine(positions, faces, uv, opts = {}) {
  const rounds = opts.iterations ?? 10;
  const cgMax = opts.cgIterations ?? 100;
  const cgTol = opts.cgTolerance ?? 1e-10;
  const nV = positions.length, nF = faces.length;
  // Per triangle: the isometric frame (x1 = y1 = y2 = 0), its area and the
  // gradient coefficients G (3 corners x 2), so J = sum_c u_c G_c^T.
  const area = new Float64Array(nF), G = new Float64Array(6 * nF), ok = new Uint8Array(nF);
  for (let f = 0; f < nF; f++) {
    const [a, b, c] = faces[f];
    const fr = triangleLocalFrame(positions[a], positions[b], positions[c]);
    if (!fr || !(fr.y3 > 0)) continue;
    ok[f] = 1;
    area[f] = 0.5 * fr.twoArea;
    const i00 = 1 / fr.x2, i01 = -fr.x3 / (fr.x2 * fr.y3), i11 = 1 / fr.y3;
    G[6 * f] = -i00; G[6 * f + 1] = -(i01 + i11);
    G[6 * f + 2] = i00; G[6 * f + 3] = i01;
    G[6 * f + 4] = 0; G[6 * f + 5] = i11;
  }
  // The Laplacian L_ij = sum_t area * G_i . G_j, assembled per row.
  const rows = Array.from({ length: nV }, () => new Map());
  for (let f = 0; f < nF; f++) {
    if (!ok[f]) continue;
    const vs = faces[f];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const w = area[f] * (G[6 * f + 2 * i] * G[6 * f + 2 * j] + G[6 * f + 2 * i + 1] * G[6 * f + 2 * j + 1]);
      const r = rows[vs[i]];
      r.set(vs[j], (r.get(vs[j]) || 0) + w);
    }
  }
  const used = [];
  for (let v = 0; v < nV; v++) if (rows[v].size) used.push(v);
  const out = uv.map((p) => [p[0], p[1]]);
  if (used.length < 3) return { uv: out, iterations: 0, energyStart: 0, energyEnd: 0 };
  const held = used[0];
  // CSR over the free vertices, the held column dropped (moved to the rhs).
  const diag = new Float64Array(nV), heldCol = new Float64Array(nV);
  const rowStart = new Int32Array(nV + 1);
  let nnz = 0;
  for (let v = 0; v < nV; v++) {
    rowStart[v] = nnz;
    if (v !== held) for (const w of rows[v].keys()) if (w !== held) nnz++;
  }
  rowStart[nV] = nnz;
  const col = new Int32Array(nnz), val = new Float64Array(nnz);
  for (let v = 0, k = 0; v < nV; v++) {
    if (v === held) continue;
    for (const [w, x] of rows[v]) {
      if (w === held) { heldCol[v] = x; continue; }
      col[k] = w; val[k] = x; k++;
      if (w === v) diag[v] = x;
    }
  }
  for (const v of used) if (!(diag[v] > 0)) diag[v] = 1;
  const free = used.filter((v) => v !== held);
  const rot = new Float64Array(2 * nF);
  const energy = () => {
    let e = 0;
    for (let f = 0; f < nF; f++) {
      if (!ok[f]) continue;
      const vs = faces[f];
      let j00 = 0, j01 = 0, j10 = 0, j11 = 0;
      for (let c = 0; c < 3; c++) {
        const g0 = G[6 * f + 2 * c], g1 = G[6 * f + 2 * c + 1], p = out[vs[c]];
        j00 += p[0] * g0; j01 += p[0] * g1; j10 += p[1] * g0; j11 += p[1] * g1;
      }
      const cs = rot[2 * f], sn = rot[2 * f + 1];
      e += area[f] * ((j00 - cs) ** 2 + (j01 + sn) ** 2 + (j10 - sn) ** 2 + (j11 - cs) ** 2);
    }
    return e;
  };
  const local = () => {
    for (let f = 0; f < nF; f++) {
      if (!ok[f]) continue;
      const vs = faces[f];
      let j00 = 0, j01 = 0, j10 = 0, j11 = 0;
      for (let c = 0; c < 3; c++) {
        const g0 = G[6 * f + 2 * c], g1 = G[6 * f + 2 * c + 1], p = out[vs[c]];
        j00 += p[0] * g0; j01 += p[0] * g1; j10 += p[1] * g0; j11 += p[1] * g1;
      }
      const E = (j00 + j11) / 2, H = (j10 - j01) / 2, q = Math.hypot(E, H);
      rot[2 * f] = q > 0 ? E / q : 1;
      rot[2 * f + 1] = q > 0 ? H / q : 0;
    }
  };
  const x = new Float64Array(nV), rhs = new Float64Array(nV), r = new Float64Array(nV);
  const z = new Float64Array(nV), p = new Float64Array(nV), Ap = new Float64Array(nV);
  const apply = (src, dst) => {
    for (const v of free) {
      let t = 0;
      for (let k = rowStart[v]; k < rowStart[v + 1]; k++) t += val[k] * src[col[k]];
      dst[v] = t;
    }
  };
  const solve = (k) => {
    // rhs_v = sum_t area * G_v . R_t[k row] - L_v,held * u_held[k]
    rhs.fill(0);
    for (let f = 0; f < nF; f++) {
      if (!ok[f]) continue;
      const cs = rot[2 * f], sn = rot[2 * f + 1];
      const r0 = k === 0 ? cs : sn, r1 = k === 0 ? -sn : cs;
      const vs = faces[f];
      for (let c = 0; c < 3; c++) rhs[vs[c]] += area[f] * (G[6 * f + 2 * c] * r0 + G[6 * f + 2 * c + 1] * r1);
    }
    const hk = out[held][k];
    for (const v of free) { x[v] = out[v][k]; rhs[v] -= heldCol[v] * hk; }
    if (band) {
      band.solve(rhs, x);
      for (const v of free) out[v][k] = x[v];
      return;
    }
    apply(x, Ap);
    let rz = 0, r0n = 0;
    for (const v of free) { r[v] = rhs[v] - Ap[v]; z[v] = r[v] / diag[v]; p[v] = z[v]; rz += r[v] * z[v]; r0n += rhs[v] * rhs[v]; }
    const stop = cgTol * cgTol * Math.max(r0n, 1e-300);
    for (let it = 0; it < cgMax; it++) {
      let rr = 0;
      for (const v of free) rr += r[v] * r[v];
      if (rr <= stop) break;
      apply(p, Ap);
      let pAp = 0;
      for (const v of free) pAp += p[v] * Ap[v];
      if (!(pAp > 0)) break;
      const alpha = rz / pAp;
      let rzNew = 0;
      for (const v of free) { x[v] += alpha * p[v]; r[v] -= alpha * Ap[v]; z[v] = r[v] / diag[v]; rzNew += r[v] * z[v]; }
      const beta = rzNew / rz;
      rz = rzNew;
      for (const v of free) p[v] = z[v] + beta * p[v];
    }
    for (const v of free) out[v][k] = x[v];
  };
  const band = arapBandFactor(free, rowStart, col, val, nV, opts.bandWork ?? 4e8);
  local();
  const energyStart = energy();
  let it = 0;
  for (; it < rounds; it++) {
    solve(0);
    solve(1);
    local();
  }
  for (const q of out) if (!Number.isFinite(q[0]) || !Number.isFinite(q[1])) throw new Error('arapRefine: a refined coordinate came out non-finite');
  return { uv: out, iterations: it, energyStart, energyEnd: energy() };
}

// A banded Cholesky factor of the ARAP Laplacian over the free vertices,
// ordered by reverse Cuthill-McKee so a strip (a gore is a few vertices
// across) has a narrow band. Returns { solve(rhs, x) } on arrays indexed by
// vertex, or null when n * band^2 exceeds `maxWork` (a wide piece, solved by
// CG instead) or a pivot is not positive.
function arapBandFactor(free, rowStart, col, val, nV, maxWork) {
  const n = free.length;
  const deg = new Int32Array(nV);
  for (const v of free) deg[v] = rowStart[v + 1] - rowStart[v];
  const isFree = new Uint8Array(nV);
  for (const v of free) isFree[v] = 1;
  const seen = new Uint8Array(nV), order = [];
  const starts = free.slice().sort((a, b) => deg[a] - deg[b]);
  for (const s0 of starts) {
    if (seen[s0]) continue;
    seen[s0] = 1;
    const queue = [s0];
    for (let h = 0; h < queue.length; h++) {
      const v = queue[h];
      order.push(v);
      const nb = [];
      for (let k = rowStart[v]; k < rowStart[v + 1]; k++) { const w = col[k]; if (isFree[w] && !seen[w]) { seen[w] = 1; nb.push(w); } }
      nb.sort((a, b) => deg[a] - deg[b]);
      for (const w of nb) queue.push(w);
    }
  }
  order.reverse();
  const pos = new Int32Array(nV).fill(-1);
  order.forEach((v, i) => { pos[v] = i; });
  let b = 0;
  for (const v of free) for (let k = rowStart[v]; k < rowStart[v + 1]; k++) b = Math.max(b, Math.abs(pos[v] - pos[col[k]]));
  if (n * b * b > maxWork) return null;
  const w = b + 1, L = new Float64Array(n * w);
  for (const v of free) for (let k = rowStart[v]; k < rowStart[v + 1]; k++) {
    const i = pos[v], j = pos[col[k]];
    if (j <= i) L[i * w + (i - j)] = val[k];
  }
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(0, i - b); j <= i; j++) {
      let t = L[i * w + (i - j)];
      for (let k = Math.max(0, i - b); k < j; k++) t -= L[i * w + (i - k)] * L[j * w + (j - k)];
      if (i === j) {
        if (!(t > 0)) return null;
        L[i * w] = Math.sqrt(t);
      } else {
        L[i * w + (i - j)] = t / L[j * w];
      }
    }
  }
  const y = new Float64Array(n);
  return {
    solve(rhs, x) {
      for (let i = 0; i < n; i++) {
        let t = rhs[order[i]];
        for (let k = Math.max(0, i - b); k < i; k++) t -= L[i * w + (i - k)] * y[k];
        y[i] = t / L[i * w];
      }
      for (let i = n - 1; i >= 0; i--) {
        let t = y[i];
        for (let k = i + 1; k <= Math.min(n - 1, i + b); k++) t -= L[k * w + (k - i)] * y[k];
        y[i] = t / L[i * w];
        x[order[i]] = y[i];
      }
    },
  };
}

// The uniform scale of a layout that makes its largest stretch equal its
// largest compression, k = 2 / (max s1 + min s2) over the measured
// triangles: of all uniform scales it gives the smallest worst principal
// strain max(k s1 - 1, 1 - k s2), which is what an inextensible material is
// judged on. Returns 1 when no triangle is measured.
export function balancedLayoutScale(positions, faces, uv) {
  let hi = -Infinity, lo = Infinity;
  for (const [a, b, c] of faces) {
    const J = strainLayoutJacobian(positions, uv, a, b, c);
    if (!J) continue;
    const E = (J.j00 + J.j11) / 2, F = (J.j00 - J.j11) / 2;
    const G = (J.j10 + J.j01) / 2, H = (J.j10 - J.j01) / 2;
    const Q = Math.hypot(E, H), R = Math.hypot(F, G);
    if (Q + R > hi) hi = Q + R;
    if (Math.abs(Q - R) < lo) lo = Math.abs(Q - R);
  }
  return hi > 0 && Number.isFinite(lo) ? 2 / (hi + lo) : 1;
}

// NURBS entry

// Convenience entry point: flatten a NURBS surface by reusing this kernel's
// trimmed-surface tessellator (kernel/trimtess.mjs), welding its output,
// and running LSCM. Contains no tessellation of its own.
//
// opts: { uRes, vRes, trimLoop, holes, ...flattenLSCM opts }
export function flattenNurbsSurface(srf, opts = {}) {
  if (!srf || !srf.ctrlNet) throw new Error('flattenNurbsSurface: expected a NURBS surface with a control net');
  const uRes = opts.uRes != null ? opts.uRes : 24;
  const vRes = opts.vRes != null ? opts.vRes : 24;
  const tris = tessellateTrimmedSurface(srf, opts.trimLoop || null, uRes, vRes, opts.holes || []);
  if (!tris.length) {
    throw new Error('flattenNurbsSurface: the surface tessellated to zero triangles (an empty trim region?) — nothing to flatten');
  }
  const mesh = dropDegenerateFaces(weldTriangulation(tris, opts.weldTolerance));
  const result = flattenLSCM(mesh, opts);
  result.tessellation = { uRes, vRes, triangleCount: tris.length, vertexCount: mesh.positions.length, weldTolerance: mesh.weldTolerance, droppedPoleFaces: mesh.droppedFaces };
  // The other loss: how much area the tessellation itself gave up before
  // the flattening ran. Only meaningful for an untrimmed patch (the exact
  // integral below covers the whole parametric rectangle, not a trimmed
  // sub-region), so it is reported only then.
  if (!opts.trimLoop && !(opts.holes && opts.holes.length)) {
    const trueArea = nurbsSurfaceArea(srf, opts.areaCellsPerSpan || 8, opts.areaCellsPerSpan || 8);
    result.tessellation.trueSurfaceArea = trueArea;
    result.tessellation.tessellatedArea = result.distortion.area.total3d;
    result.tessellation.areaLostToTessellation = trueArea - result.distortion.area.total3d;
    result.tessellation.areaRatioVsTrue = trueArea > 0 ? result.distortion.area.total3d / trueArea : 0;
  }
  return result;
}

// For a caller that wants to know whether an exact unrolling is possible
// before paying for a solve. Uses the same
// discrete Gaussian curvature (angle defect) the distortion report exposes.
export function isDevelopable(positions, faces, tolerance = 1e-9) {
  const topo = buildMeshTopology(positions, faces);
  const angleSumAt = new Float64Array(positions.length);
  for (const [a, b, c] of faces) {
    const P = [positions[a], positions[b], positions[c]];
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3, k = (i + 2) % 3;
      const u = sub3(P[j], P[i]), v = sub3(P[k], P[i]);
      const lu = len3(u), lv = len3(v);
      if (!(lu > 0 && lv > 0)) continue;
      angleSumAt[[a, b, c][i]] += Math.acos(Math.max(-1, Math.min(1, dot3(u, v) / (lu * lv))));
    }
  }
  let maxDefect = 0;
  for (let v = 0; v < positions.length; v++) {
    if (!topo.usedVertices.has(v) || topo.boundaryVertices.has(v)) continue;
    maxDefect = Math.max(maxDefect, Math.abs(2 * Math.PI - angleSumAt[v]));
  }
  return { developable: maxDefect <= tolerance, maxAbsInteriorAngleDefect: maxDefect };
}

// The true area of an untrimmed NURBS surface patch, from its own first
// fundamental form: dA = |Su x Sv| du dv, integrated with 3-point
// Gauss-Legendre using surfacePointAndPartials.
//
// The integration grid is subdivided at every distinct knot value first,
// then `cellsPerSpan` cells inside each span. A NURBS surface is only
// piecewise smooth, and its derivative is discontinuous across a
// full-multiplicity interior knot (a revolve's own arc-span joints, for
// instance). A uniform grid whose cells straddle such a joint integrates a
// kinked function inside one cell and loses orders of magnitude of
// accuracy: on a 200-degree revolve of a line, a straddling grid is 1.0e-4
// off the analytic area while a knot-aligned grid of the same cell count is
// 1.9e-6 off.
//
// A flattening pipeline loses area twice, and the two losses are reported
// separately. First the tessellation loses area (a chord always undercuts
// its arc, so a triangulated curved patch is always slightly smaller than
// the true surface). Then the flattening loses area. Reporting only the
// flattened-vs-tessellated ratio would hide the first loss.
// `flattenNurbsSurface` above reports both.
export function nurbsSurfaceArea(srf, cellsPerSpanU = 8, cellsPerSpanV = 8) {
  if (!srf || !srf.ctrlNet) throw new Error('nurbsSurfaceArea: expected a NURBS surface with a control net');
  const cellEdges = (knots, cellsPerSpan) => {
    const lo = knots[0], hi = knots[knots.length - 1];
    const distinct = [];
    for (const k of knots) {
      if (k < lo || k > hi) continue;
      if (!distinct.length || Math.abs(k - distinct[distinct.length - 1]) > 1e-12) distinct.push(k);
    }
    const out = [];
    for (let s = 0; s + 1 < distinct.length; s++) {
      const a = distinct[s], b = distinct[s + 1];
      for (let c = 0; c < cellsPerSpan; c++) out.push([a + (b - a) * (c / cellsPerSpan), a + (b - a) * ((c + 1) / cellsPerSpan)]);
    }
    return out;
  };
  // 3-point Gauss-Legendre on [-1,1].
  const g = Math.sqrt(3 / 5);
  const nodes = [[-g, 5 / 9], [0, 8 / 9], [g, 5 / 9]];
  const uCells = cellEdges(srf.knotsU, Math.max(1, cellsPerSpanU));
  const vCells = cellEdges(srf.knotsV, Math.max(1, cellsPerSpanV));
  let area = 0;
  for (const [u0, u1] of uCells) {
    const hu = (u1 - u0) / 2, mu = (u1 + u0) / 2;
    for (const [v0, v1] of vCells) {
      const hv = (v1 - v0) / 2, mv = (v1 + v0) / 2;
      for (const [a, wa] of nodes) {
        for (const [b, wb] of nodes) {
          const { su, sv } = surfacePointAndPartials(srf, mu + hu * a, mv + hv * b);
          area += len3(cross3(su, sv)) * wa * wb * hu * hv;
        }
      }
    }
  }
  return area;
}
