// ToSubD: convert a mesh or an untrimmed NURBS surface to a SubD cage, with
// a Corners=Yes/No option (keep sharp corners creased), after Rhino's
// ToSubD. Existing geometry in, cage (or a refusal) out, like
// kernel/subdprimitives.mjs but for a conversion rather than a primitive.
//
// Scope: Rhino's ToSubD does not quad-remesh an arbitrary triangle mesh — it
// takes an existing quad structure (a surface's own U/V grid, or a mesh
// whose faces are already quads) and uses that as the cage directly. General
// triangle-to-quad remeshing is a separate, much harder problem and is not
// attempted here. Two source shapes, two functions:
//
//   nurbsSurfaceToSuperBCage  — an untrimmed single NurbsSrf. Samples a
//     Greville-abscissae grid (the density convention of kernel/isocurve.mjs's
//     extractWireframeCurves: one isocurve per control-point row) via
//     surfacePoint, welds any closed-direction seam with the rounded-key
//     vertex welder kernel/subdprimitives.mjs's SuperBBox/Sphere use, and
//     builds a quad cage from the resulting grid. Refuses a trimmed surface,
//     or a surface whose profile touches its own sweep axis (a pole — the
//     grid degenerates to triangles there, not quads).
//
//   referenceMeshToSuperBCage — an already-quad mesh (faces as the original,
//     pre-triangulation polygon loops — the app's `refFaces`, captured at OBJ
//     import because the app's triangulated `faces`/display geometry has
//     discarded that structure). Welds coincident vertices at JOIN_TOLERANCE,
//     and refuses if any face is not a quad or if the welded result fails a
//     manifold check (any edge shared by more than 2 faces).
//
// Corners=Yes/No (both functions): the two source kinds need different
// corner-detection methods; each function's header says why.

import { surfacePoint } from './surface.mjs';
import { grevilleAbscissae } from './curve.mjs';
import { trivialTrimLoop } from './trim.mjs';
import { makeVertexWelder } from './subdprimitives.mjs';
import { edgeKey, buildTopology } from './subd.mjs';
import { sub, cross, length as vlen } from './vec3.mjs';
// Bounding-box diagonal (the same "tolerance scales with the object's own
// size" idiom as kernel/subdreflect.mjs's superbBboxDiagonal). The
// pole-detection tolerance is taken from it so it scales across a tiny or
// huge surface rather than being a fixed absolute number.
// surface, not a fixed absolute number.
function bboxDiagonal(points) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) for (let k = 0; k < 3; k++) { if (p[k] < min[k]) min[k] = p[k]; if (p[k] > max[k]) max[k] = p[k]; }
  return Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
}

// The marked-corner crease weight must clear a floor well above
// SUPERB_CREASE_LEVEL_SCALE (3, in the app), which is also the weight the
// app's Crease command stores on an ordinary "harden" gesture (and
// SoftCrease's 0-100 slider reaches every value up to it). Matching it would
// let an ordinary "harden both boundary edges at a plain corner" collide with
// this marker (see kernel/subd.mjs's MARKED_CORNER_WEIGHT_FLOOR for the
// derivation). 1000 clears that floor (100) even after many subdivision
// passes, since a stored weight decrements by exactly 1.0 per pass
// (subdivideCatmullClark) and superbDisplayMesh's adaptive heuristic never
// runs more than 3. It is a duplicated numeric literal rather than an import
// of MARKED_CORNER_WEIGHT_FLOOR so this module stays app-independent; a node
// test checks that this literal clears kernel/subd.mjs's floor.
export const DEFAULT_CORNER_CREASE_WEIGHT = 1000;

// Dihedral-angle sharp-edge threshold for referenceMeshToSuperBCage's
// Corners=Yes heuristic (see that function's header): 30 degrees, the same
// as Blender's "Auto Smooth" default for the same judgment of which edges of
// an already-built coarse polygon cage are hard and which are a smooth
// continuation. Overridable via opts.cornerAngleDeg for a caller that knows
// its source mesh is coarser or finer than typical.
const DEFAULT_CORNER_ANGLE_DEG = 30;

// Source 1 — untrimmed NURBS surface
//
// `srf` is an ordinary NurbsSrf (surface.mjs's shape). `opts`:
//   trimLoop        — the surface's own obj.trimLoop, or null/undefined
//                      for an ordinary untrimmed surface. Passed through
//                      raw (not pre-classified by the caller) so this
//                      function checks it directly via trivialTrimLoop
//                      (kernel/trim.mjs) rather than trusting an app-layer
//                      boolean.
//   corners         — boolean, Rhino's ToSubD "Corners=Yes/No".
//   cornerCreaseWeight — override for DEFAULT_CORNER_CREASE_WEIGHT (tests
//                      only; the app always passes its own constant).
//
// Returns { ok:true, cage, nu, nv } or { ok:false, reason }.
export function nurbsSurfaceToSuperBCage(srf, opts = {}) {
  const corners = !!opts.corners;
  const cornerCreaseWeight = opts.cornerCreaseWeight ?? DEFAULT_CORNER_CREASE_WEIGHT;
  const trimLoop = opts.trimLoop || null;

  // Trimmed-surface refusal. trivialTrimLoop is the untrimmed parametric
  // rectangle this surface would produce; a trimLoop that does not match it
  // point for point is a real trim, not a stored-but-trivial one. The same
  // isFullRect comparison kernel/trim.mjs's trimmedNakedEdgeCount uses for
  // the same question — never assumed from trimLoop's mere presence.
  if (trimLoop) {
    const rect = trivialTrimLoop(srf);
    const isFullRect = trimLoop.length === rect.length
      && trimLoop.every((p, i) => Math.hypot(p[0] - rect[i][0], p[1] - rect[i][1]) < 1e-9);
    if (!isFullRect) {
      return { ok: false, reason: 'this surface is trimmed (a real trim loop cuts away part of its untrimmed domain) — TOSUBD v1 only converts an UNTRIMMED surface; a trimmed piece has no honest, well-formed U/V quad grid to build a cage from' };
    }
  }

  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  if (nu < 2 || nv < 2) return { ok: false, reason: 'this surface has fewer than 2 control points in some direction — no real quad grid to build a cage from' };

  // Greville grid — the density convention of kernel/isocurve.mjs's
  // extractWireframeCurves (one isocurve per control-point row/column in
  // each direction), sampled at every (u,v) pair of those abscissae directly
  // via surfacePoint rather than by extracting whole isocurves first.
  const uVals = grevilleAbscissae({ degree: srf.degU, knots: srf.knotsU, ctrlPts: new Array(nu) });
  const vVals = grevilleAbscissae({ degree: srf.degV, knots: srf.knotsV, ctrlPts: new Array(nv) });
  const grid = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) row.push(surfacePoint(srf, uVals[i], vVals[j]));
    grid.push(row);
  }

  // Pole refusal. A pole (the profile touches its own revolve/sweep axis)
  // shows up as one whole line of the grid collapsing to a single point:
  // every U value at a fixed V that sits on the axis maps to the same world
  // point, regardless of sweep angle (and symmetrically for the other
  // direction). This is checked on the sampled grid itself, with no
  // assumption about which direction is "the sweep" or where an axis is.
  //
  // Every row and every column is tested, not just the grid's 4 outer
  // boundary lines (i=0, i=nu-1, j=0, j=nv-1): a profile that touches the
  // axis at a middle control point (an hourglass or goblet silhouette)
  // collapses an interior row, which a boundary-only check would miss. A
  // 5-point hourglass profile (interior point on the axis) revolved 360
  // degrees would otherwise return `ok:true` with 16 degenerate
  // (repeated-vertex) faces and a non-manifold edge.
  const flatPts = grid.flat();
  const diag = bboxDiagonal(flatPts) || 1;
  const poleTol = Math.max(diag * 1e-4, 1e-6);
  const mutuallyCoincident = (pts) => {
    for (let k = 1; k < pts.length; k++) {
      if (Math.hypot(pts[k][0] - pts[0][0], pts[k][1] - pts[0][1], pts[k][2] - pts[0][2]) > poleTol) return false;
    }
    return true;
  };
  const rowAt = (i) => grid[i];
  const colAt = (j) => grid.map((r) => r[j]);
  let hasPole = false;
  if (nv > 1) { for (let i = 0; i < nu && !hasPole; i++) if (mutuallyCoincident(rowAt(i))) hasPole = true; }
  if (!hasPole && nu > 1) { for (let j = 0; j < nv && !hasPole; j++) if (mutuallyCoincident(colAt(j))) hasPole = true; }
  if (hasPole) {
    return { ok: false, reason: 'this surface\'s profile touches its own sweep/revolve axis (a genuine POLE, at the boundary or an interior control point) — the U/V grid degenerates to triangles there, not quads; TOSUBD v1 refuses rather than attempt a triangle-fan pole collapse' };
  }

  // Weld + build — the rounded-key vertex welder kernel/subdprimitives.mjs's
  // SuperBBox/Sphere use: any closed direction's seam (first and last
  // Greville row/column landing on coincident points, per surfaceClosure's
  // definition of a seam) welds into one shared vertex by construction, so no
  // separate closedU/closedV branch is needed here.
  const { vid, vertices } = makeVertexWelder();
  const gridIdx = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) row.push(vid(grid[i][j][0], grid[i][j][1], grid[i][j][2]));
    gridIdx.push(row);
  }
  const faces = [];
  for (let i = 0; i < nu - 1; i++) {
    for (let j = 0; j < nv - 1; j++) {
      faces.push([gridIdx[i][j], gridIdx[i + 1][j], gridIdx[i + 1][j + 1], gridIdx[i][j + 1]]);
    }
  }
  const cage = { vertices, faces, creases: {} };

  // Manifold/degenerate check. The pole refusal above is the primary
  // defense; this is the same last-resort structural check
  // referenceMeshToSuperBCage applies below (refusing outright beats silently
  // forcing a bad edge fully sharp). Catches any degenerate quad (a welded
  // face that lost a corner to another vertex, collapsing to fewer than 4
  // distinct indices) or non-manifold edge (shared by more than 2 faces) that
  // could still slip past the pole test, e.g. a pole tolerance too tight for
  // some surface shape.
  for (const f of faces) {
    if (new Set(f).size !== 4) {
      return { ok: false, reason: 'this surface\'s sampled U/V grid produced a degenerate (repeated-vertex) quad — likely a pole this refusal check did not anticipate; TOSUBD v1 refuses rather than build a corrupt cage' };
    }
  }
  const manifoldCheckTopo = buildTopology(cage);
  for (const e of manifoldCheckTopo.edgeMap.values()) {
    if (e.faces.length > 2) {
      return { ok: false, reason: `this surface's sampled U/V grid produced a non-manifold edge (between vertices ${e.v0} and ${e.v1}, shared by ${e.faces.length} faces) — likely a pole this refusal check did not anticipate; TOSUBD v1 refuses rather than build a corrupt cage` };
    }
  }

  // Corners=Yes — the 4 nominal grid corners (i,j) each map to a welded
  // vertex; whether that vertex is a corner (rather than an interior seam
  // point from a closed direction, which has more than 2 incident edges
  // once welded) is checked against the cage's own topology, never assumed
  // from (i,j) alone (boundary-ness is always structural; see
  // kernel/subd.mjs's buildTopology header). A qualifying corner's two
  // incident boundary edges each get the marked-crease weight that
  // kernel/subd.mjs's computeVertexPoint reads as "this corner was marked,
  // hold it at P" (see that function's header). Storing an ordinary weight
  // on a boundary edge would have no effect.
  if (corners) {
    const topo = manifoldCheckTopo; // reuse — the cage is untouched between the safety net above and here
    const nominal = [[0, 0], [0, nv - 1], [nu - 1, 0], [nu - 1, nv - 1]];
    const seen = new Set();
    for (const [i, j] of nominal) {
      const vi = gridIdx[i][j];
      if (seen.has(vi)) continue;
      seen.add(vi);
      const incident = topo.vertexEdges[vi];
      if (incident.length === 2 && incident.every((e) => e.faces.length === 1)) {
        for (const e of incident) cage.creases[edgeKey(e.v0, e.v1)] = cornerCreaseWeight;
      }
    }
  }

  return { ok: true, cage, nu, nv };
}

// Source 2 — an already-quad mesh (a ReferenceMesh's original,
// pre-triangulation face loops — the app's `refFaces`)
//
// `positions` — [[x,y,z], ...], the mesh's raw (undeduped) vertex list.
// `faces` — [[vi0,vi1,...], ...] index loops into `positions`, exactly as
//   read from the source file, before any triangulation.
// `opts`:
//   tolerance          — weld tolerance in mm (default: JOIN_TOLERANCE,
//                         0.001).
//   corners            — boolean.
//   cornerAngleDeg      — override for DEFAULT_CORNER_ANGLE_DEG.
//   cornerCreaseWeight  — override for DEFAULT_CORNER_CREASE_WEIGHT.
//
// Returns { ok:true, cage } or { ok:false, reason }.
export function referenceMeshToSuperBCage(positions, faces, opts = {}) {
  const tolerance = opts.tolerance ?? 0.001;
  const corners = !!opts.corners;
  const angleDeg = opts.cornerAngleDeg ?? DEFAULT_CORNER_ANGLE_DEG;
  const cornerCreaseWeight = opts.cornerCreaseWeight ?? DEFAULT_CORNER_CREASE_WEIGHT;

  if (!Array.isArray(faces) || faces.length === 0) return { ok: false, reason: 'this mesh has no faces to build a cage from' };
  for (let fi = 0; fi < faces.length; fi++) {
    if (faces[fi].length !== 4) {
      return { ok: false, reason: `face ${fi} has ${faces[fi].length} vertices, not a quad — TOSUBD v1 only reuses an already-quad mesh's own faces directly as the cage (real Rhino ToSubD does the same, never a general triangle-to-quad remesh); re-export this mesh with quad faces first` };
    }
  }

  // Weld at JOIN_TOLERANCE — a grid-snap rounded key (each coordinate
  // rounded to the nearest `tolerance`-sized step), the same rounded-key idiom
  // as makeVertexWelder above, parametrized by tolerance instead of a fixed
  // 6-decimal round (that fixed precision suits a single surface's
  // near-machine-precision seam; an imported mesh's coincident vertices can
  // differ by up to JOIN_TOLERANCE, not 1e-6mm).
  const map = new Map();
  const remap = new Array(positions.length);
  const welded = [];
  for (let i = 0; i < positions.length; i++) {
    const [x, y, z] = positions[i];
    const key = `${Math.round(x / tolerance)}_${Math.round(y / tolerance)}_${Math.round(z / tolerance)}`;
    let idx = map.get(key);
    if (idx === undefined) { idx = welded.length; welded.push([x, y, z]); map.set(key, idx); }
    remap[i] = idx;
  }
  const cageFaces = faces.map((f) => f.map((vi) => remap[vi]));

  for (let fi = 0; fi < cageFaces.length; fi++) {
    if (new Set(cageFaces[fi]).size !== 4) {
      return { ok: false, reason: `face ${fi} collapsed to a degenerate shape once its coincident vertices were welded within ${tolerance}mm — not a valid quad cage face` };
    }
  }

  const cage = { vertices: welded, faces: cageFaces, creases: {} };

  // Manifold check. kernel/subd.mjs's subdivideCatmullClark does not refuse
  // non-manifold input; it forces a 3+-face edge fully sharp as a fallback
  // (see that file's header). ToSubD refuses outright instead: a converted
  // cage silently forced sharp along an edge the source mesh never meant as
  // a hard edge is a worse surprise than a refusal naming the edge.
  const topo = buildTopology(cage);
  for (const e of topo.edgeMap.values()) {
    if (e.faces.length > 2) {
      return { ok: false, reason: `an edge between vertices ${e.v0} and ${e.v1} is shared by ${e.faces.length} faces — not a manifold cage (a real edge must be shared by at most 2 faces)` };
    }
  }

  // Corners=Yes, mesh source — a different and simpler detection method than
  // the NURBS-surface case: rather than forcing an "exactly 4 named corners"
  // concept onto an arbitrary mesh, this detects sharp edges directly by
  // dihedral angle (the angle between the two face normals meeting at an
  // edge; see DEFAULT_CORNER_ANGLE_DEG for the threshold). A sharp corner
  // vertex follows from how many sharp edges converge there — 3+ at one
  // vertex reads as a held-at-P corner through kernel/subd.mjs's 3+ branch,
  // and exactly 2 reads as a sharp crease line through the same code. Neither
  // needs the NURBS path's marked-corner weight, since these are interior
  // edges (2 adjacent faces), not boundary edges whose stored weight the
  // ordinary rules ignore.
  if (corners) {
    const faceNormal = (f) => {
      const a = cage.vertices[f[0]], b = cage.vertices[f[1]], c = cage.vertices[f[2]];
      const n = cross(sub(b, a), sub(c, a));
      const len = vlen(n) || 1;
      return [n[0] / len, n[1] / len, n[2] / len];
    };
    const normals = cageFaces.map(faceNormal);
    const thresholdCos = Math.cos((angleDeg * Math.PI) / 180);
    for (const [key, e] of topo.edgeMap) {
      if (e.faces.length !== 2) continue; // boundary edge of an open mesh — no second face to compare against
      const n0 = normals[e.faces[0]], n1 = normals[e.faces[1]];
      const cosAngle = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
      if (cosAngle < thresholdCos) cage.creases[key] = cornerCreaseWeight;
    }
  }

  return { ok: true, cage };
}
