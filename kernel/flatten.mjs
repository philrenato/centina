// SURFACE FLATTENING / UNROLL — TIER 1, LSCM.
// Given a triangulation of a surface (this app's own already-proven
// tessellation, never a second tessellator), produce a flat 2D layout of it
// plus a REAL, MEASURED report of how much the flattening lied.
//
// ================================================================
// THE MATHEMATICAL FACT THAT BOUNDS THIS FOREVER — STATED UP FRONT,
// NOT BURIED, BECAUSE EVERY FUNCTION BELOW IS SUBORDINATE TO IT.
// ================================================================
// Gaussian curvature is an intrinsic invariant of a surface under any
// isometric (length-preserving) map. This is Gauss's Theorema Egregium.
// It follows immediately that a surface with NONZERO Gaussian curvature
// ANYWHERE CANNOT be flattened into the plane without local stretch or
// tearing — not by LSCM, not by ARAP, not by any algorithm, ever, at any
// resolution, with any amount of compute. A flattener that reports "exact"
// for a genuinely curved input is lying.
//
// So: every function here returns a real, measured distortion metric in
// usable units alongside its result, and NEVER a claim of exactness for a
// genuinely curved input. This is the same honest-approximation posture
// kernel/offset.mjs and kernel/offsetcurve.mjs already established for
// THEIR own "not exact for a curved input" limits — reused as a convention,
// not invented fresh here.
//
// The one case where an EXACT answer is mathematically possible is a
// DEVELOPABLE input (a cylinder, a cone, a plane — zero Gaussian curvature
// everywhere). That is this module's exactness anchor, proven directly in
// test/flatten.test.mjs against an ANALYTICALLY known unrolled shape rather
// than against the flattener's own output.
//
// ================================================================
// THE ALGORITHM — LSCM, AND WHAT IT DOES *NOT* MINIMIZE
// ================================================================
// Levy, Petitjean, Ray, Maillot, "Least Squares Conformal Maps for
// Automatic Texture Atlas Generation," SIGGRAPH 2002 (ACM Transactions on
// Graphics 21(3), pp. 362-371).
//
// LSCM minimizes the discrete CONFORMAL energy: per triangle, the squared
// deviation of the piecewise-linear map from satisfying the Cauchy-Riemann
// equations. It is a SINGLE sparse linear least-squares solve — no outer
// iteration, no convergence risk, no local minima. That boundedness is
// exactly why it is tier 1.
//
// LSCM minimizes ANGLE distortion. It does NOT minimize STRETCH. A
// flattened result can be locally correct in SHAPE while being genuinely
// wrong in SCALE — which is the property that actually matters to anyone
// cutting real material. Because of that, `flatteningDistortion` below
// reports AREA and EDGE-LENGTH distortion (what LSCM was not optimizing)
// as first-class numbers beside the angle distortion (what it was), so a
// caller is never handed only the flattering metric.
//
// ARAP (Liu, Zhang, Hu, Zhang, "A Local/Global Approach to Mesh
// Parameterization," SGP 2008 / Computer Graphics Forum 27(5)) is the real
// stretch-minimizing target and is deliberately NOT implemented here. See
// the "SEAM LEFT FOR ARAP" note at the bottom of this header for exactly
// which pieces below it reuses unchanged.
//
// ================================================================
// UNITS
// ================================================================
// Positions are in the document's own length unit (mm throughout this app).
// The returned `uv` is in the SAME unit — the raw LSCM solution is defined
// only up to a global similarity, so it is rescaled by the single uniform
// factor that best fits the true 3D edge lengths in a least-squares sense,
// and translated so its bounding box starts at (0,0). Angles are reported
// in degrees; the intrinsic angle defect in radians; areas in the square of
// the length unit; every ratio-shaped error is dimensionless.
//
// ================================================================
// SEAM LEFT FOR ARAP (tier 2, not built)
// ================================================================
// ARAP is local/global: a LOCAL step fitting a best rigid 2x2 rotation per
// triangle, and a GLOBAL step solving one sparse linear system. Every piece
// it needs on the way in and out already exists here and is exported:
//   * weldTriangulation      — non-indexed triangle soup -> indexed mesh
//   * buildMeshTopology      — edges/boundary/components/manifoldness
//   * validateFlattenMesh    — the honest-refusal gate, algorithm-agnostic
//   * triangleLocalFrame     — the per-triangle isometric 2D coordinates
//                              ARAP's local step differences against
//   * solveSparseLeastSquares — the same matrix-free CG solver ARAP's
//                              global step needs
//   * flatteningDistortion   — the shared measurement, unchanged
//   * flattenLSCM            — ARAP's own required initial guess
// An `flattenARAP(mesh, opts)` would therefore be: validate, weld (both
// already done), call flattenLSCM for the initialization, then alternate
// (local 2x2 SVD per triangle) / (global solve via solveSparseLeastSquares)
// until the change in stretch energy falls below a tolerance, and finish by
// calling flatteningDistortion on the result. Nothing in this file needs to
// change for that; it is additive.

import { surfacePointAndPartials } from './surface.mjs';
import { tessellateTrimmedSurface } from './trimtess.mjs';

// ---------------------------------------------------------------- helpers

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

// ---------------------------------------------------------------- welding

// WELD a non-indexed triangle soup (exactly what kernel/trimtess.mjs's
// tessellateTrimmedSurface returns — [[vA,vB,vC], ...] with each v carrying
// {position, normal, uv}) into an INDEXED mesh, which is what any
// parametrization algorithm needs: LSCM solves for one (u,v) per shared
// VERTEX, so two triangles meeting along an edge must genuinely reference
// the same two vertex indices, not two coincident copies.
//
// This is welding, not tessellation — it never invents or moves a point, it
// only decides which of the points already produced are the same point.
//
// `tolerance` defaults to a RELATIVE value (1e-9 of the bounding-box
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
    // of genuinely coincident points can straddle a cell boundary.
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

// DROP the faces that WELDING itself collapsed. A revolve whose profile
// touches the axis has a POLE — the whole pole row of the parameter grid is
// one physical point — and `tessellateTrimmedSurface` emits two triangles
// per grid cell unconditionally, so after welding, the pole cell's second
// triangle names the same vertex twice. That face was manufactured by this
// module's own NURBS entry points, and `validateFlattenMesh` then (rightly)
// refused it, which made EVERY pole-bearing surface unflattenable: a cone
// with its apex, a sphere, an ellipsoid — including a partial cone SECTOR
// that is exactly developable and should unroll with zero distortion.
//
// Dropping is lossless, not a tolerance: a repeated-vertex triangle has
// identically zero area and contributes no topology (the "quad" at a pole
// is genuinely a triangle, and what remains around the pole is a proper
// triangle fan). Positions are returned untouched — not one coordinate
// moves — and the summed triangle area is bit-for-bit unchanged.
//
// Deliberately NOT folded into `weldTriangulation` (a general welder must
// not silently discard faces a caller handed it) and deliberately NOT
// relaxing `validateFlattenMesh` (a degenerate face in a mesh a CALLER
// supplied is a real thing to catch, and it still is). This is applied
// only where the problem is created: the two NURBS entry points below.
export function dropDegenerateFaces(mesh) {
  const kept = [];
  let dropped = 0;
  for (const [a, b, c] of mesh.faces) {
    if (a === b || b === c || c === a) { dropped++; continue; }
    kept.push([a, b, c]);
  }
  return { ...mesh, faces: kept, droppedFaces: dropped };
}

// ---------------------------------------------------------------- topology

// Edge/boundary/component structure of an indexed triangle mesh. Shared by
// the validity gate, the distortion metrics, and (later) ARAP — one
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

  // Connected components over FACES (adjacency through shared edges) — a
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
  // Vertices genuinely referenced by a face (a welded soup can't produce an
  // orphan, but an externally-supplied mesh can).
  const usedVertices = new Set();
  for (const [a, b, c] of faces) { usedVertices.add(a); usedVertices.add(b); usedVertices.add(c); }

  return {
    edges, edgeFaces, boundaryEdges, boundaryVertices,
    nonManifoldEdges, componentCount, componentOfFace,
    usedVertices, vertexCount: positions.length, faceCount: faces.length,
  };
}

// The honest-refusal gate. REFUSES BY NAME rather than returning a
// plausible-looking wrong layout — matching this kernel's standing rule
// that a degenerate input gets a specific message, never a silent result.
// Deliberately algorithm-agnostic: LSCM and (later) ARAP share it.
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
  // A flattening is a map into the plane. Only a TOPOLOGICAL DISK can be
  // mapped into the plane without self-overlap, whatever its curvature —
  // an annulus (a full revolve welded shut at its seam) or a handle has
  // nowhere for the extra topology to go. Euler characteristic V - E + F is
  // exactly 1 for a disk, so this is a cheap, exact test rather than a
  // heuristic. Caught here by name, because LSCM will otherwise happily
  // return a folded-over layout that looks plausible and is not usable.
  const chi = topo.usedVertices.size - topo.edges.length + topo.faceCount;
  if (chi !== 1) {
    throw new Error(`validateFlattenMesh: refusing to flatten — the mesh is not a TOPOLOGICAL DISK (Euler characteristic V-E+F = ${chi}, a disk is 1). It has a hole or a handle; cut it open along a seam first, then flatten each piece.`);
  }
  // "Too few faces to constrain the solve": LSCM pins 2 vertices to remove
  // the 4-dof similarity ambiguity, so there must be at least one FREE
  // vertex left for the solve to have anything to determine.
  if (topo.usedVertices.size < 3) {
    throw new Error(`validateFlattenMesh: refusing to flatten — only ${topo.usedVertices.size} vertices are actually used by a face; after pinning 2 to fix the layout's own position/rotation/scale there is nothing left to solve for.`);
  }
  return topo;
}

// ---------------------------------------------------------- local frames

// The ISOMETRIC 2D coordinates of one triangle: lay it flat in its own
// plane with vertex 0 at the origin and vertex 1 on the +x axis. This is
// exact (a single triangle is trivially developable) and is the per-triangle
// reference frame BOTH LSCM (below) and ARAP (later) measure against.
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

// ---------------------------------------------------------------- solver

// SPARSE LINEAR LEAST SQUARES, matrix-free, via CONJUGATE GRADIENT ON THE
// NORMAL EQUATIONS (CGLS/CGNR) with Jacobi (diagonal) preconditioning.
//
// WHY THIS METHOD, and why it is numerically appropriate for LSCM
// specifically (this kernel has no dependencies, so the solver is written
// here rather than pulled in — the choice therefore has to be justified,
// not defaulted to):
//
//  * The LSCM matrix is EXTREMELY sparse and structured: each triangle
//    contributes exactly 2 rows with at most 6 nonzeros each. A matrix-free
//    Krylov method touches only those nonzeros, so one iteration is O(nnz)
//    with no fill-in at all. A direct sparse factorization (Cholesky/QR)
//    would be faster per solve for a fixed mesh but needs real ordering and
//    fill-reducing machinery to beat that — a large amount of code, and a
//    large amount of new risk, for a solve that is not the bottleneck.
//  * A^T A is symmetric POSITIVE DEFINITE here, which is exactly what CG
//    requires. It is positive definite BECAUSE two vertices are pinned: the
//    conformal energy's own nullspace is precisely the 4-parameter
//    similarity group (translate 2, rotate 1, scale 1) of the whole layout,
//    and pinning two vertices removes exactly those 4 degrees of freedom.
//    So the pinning is not a convenience, it is what makes the solve
//    well posed.
//  * HONEST NUMERICAL CAVEAT, stated rather than hidden: forming the normal
//    equations squares the condition number, cond(A^T A) = cond(A)^2. Three
//    things keep that acceptable here: (1) each triangle's rows are scaled
//    by 1/sqrt(2*area), which makes every matrix entry O(1) regardless of
//    triangle size, so a mesh with wildly varying triangle areas does not
//    have its conditioning wrecked by the largest ones; (2) Jacobi
//    preconditioning divides out the remaining per-column scale spread;
//    (3) the achieved residual is MEASURED and RETURNED, and a genuine
//    failure to converge throws by name instead of returning a
//    plausible-looking wrong answer.
//
// `triplets` is a flat array of {row, col, val}. Solves min ||A x - b||.
export function solveSparseLeastSquares(triplets, rhs, nCols, opts = {}) {
  const nRows = rhs.length;
  const maxIter = opts.maxIterations != null ? opts.maxIterations : Math.min(200000, 40 * nCols + 1000);
  const relTol = opts.tolerance != null ? opts.tolerance : 1e-11;

  const applyA = (x, out) => {
    out.fill(0);
    for (let k = 0; k < triplets.length; k++) {
      const t = triplets[k];
      out[t.row] += t.val * x[t.col];
    }
  };
  const applyAT = (y, out) => {
    out.fill(0);
    for (let k = 0; k < triplets.length; k++) {
      const t = triplets[k];
      out[t.col] += t.val * y[t.row];
    }
  };

  // Jacobi preconditioner: the diagonal of A^T A.
  const diag = new Float64Array(nCols);
  for (let k = 0; k < triplets.length; k++) {
    const t = triplets[k];
    diag[t.col] += t.val * t.val;
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

// ---------------------------------------------------------------- metrics

// MEASURE the flattening. This is the honest half of the module: it is
// computed from the 3D mesh and the 2D layout INDEPENDENTLY of how that
// layout was produced, so it is a real measurement rather than a
// self-consistency check on the solver.
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
// intrinsic.maxAbsInteriorAngleDefect is the DISCRETE GAUSSIAN CURVATURE
// (2*pi minus the incident angle sum at an interior vertex). By Theorema
// Egregium it is exactly the quantity that decides, BEFORE any flattening
// is attempted, whether zero distortion is even possible: it is zero
// everywhere on a developable and nonzero on anything genuinely
// doubly-curved. It is reported so a caller can tell "this flattening is
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
    // ORIENTATION, measured against a canonical reference rather than a
    // majority vote among the triangles themselves: triangleLocalFrame
    // always lays a triangle out counter-clockwise in its own isometric
    // frame, so an orientation-PRESERVING (holomorphic) map must give every
    // triangle a positive signed 2D area. A negative one is a genuine
    // reversal — either a local fold, or (if every triangle is negative) a
    // globally mirrored layout, which for anyone cutting real material is
    // the difference between a left part and a right one. Both are real
    // failures worth counting, so neither is voted away.
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
      minRatio: areaRatios.length ? Math.min(...areaRatios) : 0,
      maxRatio: areaRatios.length ? Math.max(...areaRatios) : 0,
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

// ---------------------------------------------------------------- LSCM

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

// FLATTEN an indexed triangle mesh into the plane via LSCM.
//
// mesh: { positions: [[x,y,z], ...], faces: [[i,j,k], ...] }
// opts: { tolerance, maxIterations, pinA, pinB }
//
// Returns:
//   { uv, positions, faces, pins, solver, distortion, developable }
// where `uv` is one [u,v] per input vertex in the SAME length unit as the
// input positions, `distortion` is the full measured report from
// flatteningDistortion, and `developable` is a plain-language summary of
// whether an exact answer was even possible for this input.
//
// NEVER claims exactness for a curved input: `distortion` is always
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
  // and scale (1) — exactly the 4-dimensional nullspace of the conformal
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

  // A conformal map is only defined up to a global SIMILARITY, so the raw
  // solution has an arbitrary overall scale. Fix it with the single uniform
  // factor s minimizing sum over mesh edges of (s*L2d - L3d)^2, i.e.
  //   s = sum(L2d*L3d) / sum(L2d^2).
  // This is the honest normalization to measure length/area distortion
  // against: it never hides distortion (a genuinely curved input still
  // cannot match every edge), it only removes the meaningless global scale
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

  // A plain-language, honest verdict — never "exact" unless the input's own
  // discrete Gaussian curvature says an exact answer was possible at all.
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

// ---------------------------------------------------------- NURBS entry

// Convenience entry point: flatten a NURBS surface by reusing this kernel's
// OWN already-proven trimmed-surface tessellator (kernel/trimtess.mjs),
// welding its output, and running LSCM. Deliberately does NOT contain any
// tessellation of its own.
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
  // Honest accounting of the OTHER loss: how much area the tessellation
  // itself already gave up before the flattening ever ran. Only meaningful
  // for an untrimmed patch (the exact integral below covers the whole
  // parametric rectangle, not a trimmed sub-region), so it is reported only
  // then rather than quietly reported wrong.
  if (!opts.trimLoop && !(opts.holes && opts.holes.length)) {
    const trueArea = nurbsSurfaceArea(srf, opts.areaCellsPerSpan || 8, opts.areaCellsPerSpan || 8);
    result.tessellation.trueSurfaceArea = trueArea;
    result.tessellation.tessellatedArea = result.distortion.area.total3d;
    result.tessellation.areaLostToTessellation = trueArea - result.distortion.area.total3d;
    result.tessellation.areaRatioVsTrue = trueArea > 0 ? result.distortion.area.total3d / trueArea : 0;
  }
  return result;
}

// A small, honest convenience for a caller that wants to know whether an
// exact unrolling is even possible BEFORE paying for a solve. Uses the same
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

// The TRUE area of an untrimmed NURBS surface patch, from its own first
// fundamental form: dA = |Su x Sv| du dv, integrated with 3-point
// Gauss-Legendre using the already-proven surfacePointAndPartials.
//
// The integration grid is subdivided AT EVERY DISTINCT KNOT VALUE first,
// then `cellsPerSpan` cells inside each span. This matters and is not
// cosmetic: a NURBS surface is only piecewise smooth, and its derivative is
// genuinely discontinuous across a full-multiplicity interior knot (a
// revolve's own arc-span joints, for instance). A naive uniform grid whose
// cells straddle such a joint integrates a kinked function inside one cell
// and loses several orders of magnitude of accuracy — measured directly:
// on a 200-degree revolve of a line, a straddling grid was 1.0e-4 off the
// analytic area while a knot-aligned grid of the SAME cell count was
// 1.9e-6 off.
//
// WHY THIS IS HERE, and not decoration: a flattening pipeline loses area
// TWICE, and a caller cutting real material deserves both numbers
// separately. First the TESSELLATION loses area (a chord always undercuts
// its arc, so a triangulated curved patch is always slightly SMALLER than
// the true surface). Only then does the FLATTENING lose area. Reporting
// only the flattened-vs-tessellated ratio would quietly hide the first
// loss inside an otherwise flattering number. `flattenNurbsSurface` below
// reports both.
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
