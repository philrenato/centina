// TOSUBD ("convert a mesh or
// (untrimmed) NURBS surface to SubD. Corners=Yes/No option (keep sharp
// corners creased), tracked from Rhino's ToSubD.") — KERNEL ONLY: pure
// existing-geometry-in, cage-out (or an honest refusal) math, matching
// kernel/subdprimitives.mjs's own discipline exactly, just for a CONVERSION
// rather than a from-scratch primitive.
//
// THE REAL SCOPE DECISION (derived directly against this app's own code,
// not assumed): real Rhino ToSubD does NOT quad-remesh an arbitrary
// triangle mesh — it takes an ALREADY well-formed quad structure (a
// surface's own real U/V grid, or a mesh whose faces already are quads)
// and uses that AS the cage directly. General triangle-mesh-to-quad
// remeshing is a genuinely separate, much harder problem, not attempted
// here. Two source shapes, two functions:
//
//   nurbsSurfaceToSuperBCage  — an UNTRIMMED single NurbsSrf. Samples a
//     Greville-abscissae grid (the SAME density convention
//     kernel/isocurve.mjs's own extractWireframeCurves already
//     establishes for "one isocurve per control-point row" — reused
//     verbatim, not a new density rule) via surfacePoint, welds any
//     closed-direction seam with the SAME rounded-key vertex welder
//     kernel/subdprimitives.mjs's own SuperBBox/Sphere already prove
//     watertight, and builds a real quad cage from the resulting grid.
//     Refuses honestly for a genuinely TRIMMED surface, or a surface
//     whose profile touches its own sweep axis (a genuine POLE — the
//     grid degenerates to triangles there, not quads).
//
//   referenceMeshToSuperBCage — an already-quad mesh (faces as ORIGINAL,
//     pre-triangulation polygon loops — see the app's own
//     `refFaces`, captured at OBJ-import time specifically so this
//     function has real quad structure to work from, since the app's
//     OWN triangulated `faces`/display geometry has already discarded
//     it). Welds coincident vertices at this app's own JOIN_TOLERANCE,
//     refuses honestly if any face isn't a quad or if the welded result
//     fails a real manifold check (any edge shared by more than 2 faces).
//
// CORNERS=YES/NO (both functions): a real, reasoned choice, stated plainly
// per source kind — see each function's own header comment for why the two
// kinds need genuinely different corner-detection methods.

import { surfacePoint } from './surface.mjs';
import { grevilleAbscissae } from './curve.mjs';
import { trivialTrimLoop } from './trim.mjs';
import { makeVertexWelder } from './subdprimitives.mjs';
import { edgeKey, buildTopology } from './subd.mjs';
import { sub, cross, length as vlen } from './vec3.mjs';

// A cage-agnostic bbox-diagonal helper (the SAME "tolerance scales with the
// object's own size" idiom kernel/subdreflect.mjs's own superbBboxDiagonal
// already establishes for mirror-partner tolerance) — used here purely to
// pick a POLE-detection tolerance that scales sanely across a tiny or huge
// surface, not a fixed absolute number.
function bboxDiagonal(points) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) for (let k = 0; k < 3; k++) { if (p[k] < min[k]) min[k] = p[k]; if (p[k] > max[k]) max[k] = p[k]; }
  return Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
}

// MARKED-CORNER CREASE WEIGHT must clear a floor well above
// SUPERB_CREASE_LEVEL_SCALE (3, in the app) — that is ALSO the exact
// weight the app's own shipped Crease command stores on an ordinary
// "harden" gesture (and SoftCrease's own 0-100 slider
// reaches every value up to it), so matching it exactly would let an
// entirely ordinary "harden both
// boundary edges at a plain corner" through the EXISTING UI collide with
// this marker (see kernel/subd.mjs's own MARKED_CORNER_WEIGHT_FLOOR for the
// full derivation). Raised well above that floor (100) — still comfortably
// clears it even after many subdivision passes, since a stored weight only
// ever decrements by exactly 1.0 per pass (subdivideCatmullClark) and
// superbDisplayMesh's own adaptive heuristic never runs more than 3. Kept
// as a plain duplicated numeric literal (not an import of MARKED_CORNER_
// WEIGHT_FLOOR) — this module stays app-independent, matching every other
// kernel module's own "no app-layer constant imports" discipline; the two
// values are independently chosen but both documented, and a node test
// cross-checks this literal genuinely clears kernel/subd.mjs's own floor.
export const DEFAULT_CORNER_CREASE_WEIGHT = 1000;

// DIHEDRAL-ANGLE SHARP-EDGE THRESHOLD (referenceMeshToSuperBCage's own
// Corners=Yes heuristic, see that function's header for the full
// derivation) — 30 degrees, a real, reasoned choice: it is the same order
// of magnitude as Blender's own long-standing "Auto Smooth" default (30
// degrees) for the identical "which edges of an already-built mesh are
// genuinely hard vs. an ordinary smooth-continuation facet" judgment call
// on a coarse polygon cage. Overridable via opts.cornerAngleDeg for a
// caller that knows its own source mesh is coarser/finer than typical.
const DEFAULT_CORNER_ANGLE_DEG = 30;

// ---------------------------------------------------------------------
// SOURCE 1 — UNTRIMMED NURBS SURFACE
// ---------------------------------------------------------------------
//
// `srf` is an ordinary NurbsSrf (surface.mjs's own shape). `opts`:
//   trimLoop        — the surface's own obj.trimLoop, or null/undefined
//                      for an ordinary untrimmed surface. Passed through
//                      RAW (not pre-classified by the caller) so this
//                      function does its own honest, direct check via
//                      trivialTrimLoop (kernel/trim.mjs) rather than
//                      trusting an app-layer boolean.
//   corners         — boolean, Rhino's own ToSubD "Corners=Yes/No".
//   cornerCreaseWeight — override for DEFAULT_CORNER_CREASE_WEIGHT (tests
//                      only; the app always passes its own real constant).
//
// Returns { ok:true, cage, nu, nv } or { ok:false, reason }.
export function nurbsSurfaceToSuperBCage(srf, opts = {}) {
  const corners = !!opts.corners;
  const cornerCreaseWeight = opts.cornerCreaseWeight ?? DEFAULT_CORNER_CREASE_WEIGHT;
  const trimLoop = opts.trimLoop || null;

  // TRIMMED-SURFACE REFUSAL — checked directly (trivialTrimLoop is the
  // EXACT untrimmed parametric rectangle this surface would produce; a
  // trimLoop that doesn't match it point-for-point is a REAL trim, not
  // just a stored-but-trivial one), the identical isFullRect comparison
  // kernel/trim.mjs's own trimmedNakedEdgeCount already uses for the
  // identical "is this trimLoop actually trimming anything" question —
  // never assumed from trimLoop's mere presence.
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

  // GREVILLE GRID — the SAME density convention
  // kernel/isocurve.mjs's own extractWireframeCurves already establishes
  // (one isocurve per control-point row/column in each direction); here
  // sampled at every (u,v) PAIR of those same abscissae, directly via
  // surfacePoint, rather than extracting whole isocurves first.
  const uVals = grevilleAbscissae({ degree: srf.degU, knots: srf.knotsU, ctrlPts: new Array(nu) });
  const vVals = grevilleAbscissae({ degree: srf.degV, knots: srf.knotsV, ctrlPts: new Array(nv) });
  const grid = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) row.push(surfacePoint(srf, uVals[i], vVals[j]));
    grid.push(row);
  }

  // POLE REFUSAL — a genuine pole (the profile touches its own revolve/
  // sweep axis) shows up, structurally, as one whole LINE of the grid
  // collapsing to a single physical point: every U value at a fixed V
  // that sits on the axis maps to the identical world point, regardless
  // of sweep angle (and symmetrically for the orthogonal direction). This
  // is checked GENERICALLY on the sampled grid itself — no assumption
  // about which direction is "the sweep" or where an axis is.
  //
  // Every row AND every column must be tested here, not just the grid's 4
  // OUTER boundary lines (i=0, i=nu-1, j=0, j=nv-1) — a profile that
  // touches the axis
  // at a MIDDLE control point (an hourglass/goblet silhouette, a real,
  // reachable organic shape, not a contrived edge case) collapses an
  // INTERIOR row instead, which a boundary-only check would never see: a
  // 5-point hourglass profile
  // (interior point ON the axis) revolved 360 degrees would otherwise
  // return
  // `ok:true` with 16 degenerate (repeated-vertex) faces and a genuine
  // non-manifold edge. Testing every row and column stays fully generic
  // (no assumption about
  // which direction is "the sweep"), and doesn't stop at the
  // boundary.
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

  // WELD + BUILD — the SAME rounded-key vertex welder
  // kernel/subdprimitives.mjs's own SuperBBox/Sphere already prove
  // watertight, reused verbatim: any closed direction's own seam (first
  // and last Greville row/column landing on physically-coincident
  // points, per surfaceClosure's own definition of a seam) welds into one
  // shared vertex automatically, by construction — no separate closedU/
  // closedV branch needed here at all.
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

  // MANIFOLD/DEGENERATE SAFETY NET.
  // The pole refusal above is this function's PRIMARY defense, but a last-
  // resort structural check belongs here too, matching
  // referenceMeshToSuperBCage's own identical gate just below in this file
  // (that function's header already names why: refusing outright beats
  // silently forcing a bad edge fully sharp). Catches any degenerate quad
  // (a welded face that lost a corner to another vertex, collapsing to
  // fewer than 4 distinct indices) or non-manifold edge (shared by more
  // than 2 faces) that could in principle still slip past the generic
  // pole test above — e.g. a pole test tolerance genuinely too tight for
  // some future surface shape neither test fixture anticipated.
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

  // CORNERS=YES — the 4 nominal grid corners (i,j) each map to a real
  // welded vertex; whether that vertex IS actually a genuine corner (vs.
  // an interior seam point from a closed direction, which naturally has
  // MORE than 2 incident edges once welded) is checked structurally
  // against the cage's own real topology, never assumed from (i,j) alone
  // — matching this app's "boundary-ness is always structural, never
  // hand-tracked" discipline (kernel/subd.mjs's own buildTopology header).
  // A qualifying corner's own two incident (boundary) edges each get the
  // marked-crease weight kernel/subd.mjs's own computeVertexPoint now
  // reads as "this corner was deliberately marked, hold it at P" (see
  // that function's own header comment for the full derivation) — a real
  // effect, not the inert no-op storing weight on an ordinary boundary
  // edge would otherwise be.
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

// ---------------------------------------------------------------------
// SOURCE 2 — AN ALREADY-QUAD MESH (a ReferenceMesh's own ORIGINAL,
// pre-triangulation face loops — see the app's own `refFaces`)
// ---------------------------------------------------------------------
//
// `positions` — [[x,y,z], ...], the mesh's raw (undeduped) vertex list.
// `faces` — [[vi0,vi1,...], ...] index loops into `positions`, EXACTLY as
//   read from the source file, before any triangulation.
// `opts`:
//   tolerance          — weld tolerance in mm (default matches this app's
//                         own JOIN_TOLERANCE, 0.001).
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

  // WELD BY THIS APP'S OWN JOIN_TOLERANCE — a plain grid-snap rounded key
  // (round each coordinate to the nearest `tolerance`-sized step), the
  // SAME "rounded-key" idiom makeVertexWelder above uses, just tolerance-
  // parametrized instead of a fixed 6-decimal round (that fixed precision
  // is right for a single surface's own near-machine-precision seam; an
  // imported mesh's own coincident vertices can differ by up to a real,
  // named JOIN_TOLERANCE, not 1e-6mm).
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

  // MANIFOLD CHECK — a real, explicit gate this module adds (kernel/
  // subd.mjs's own subdivideCatmullClark deliberately does NOT refuse
  // non-manifold input; it just forces a 3+-face edge fully sharp as an
  // honest fallback, see that file's own header comment). TOSUBD refuses
  // OUTRIGHT instead — a converted cage silently forced-sharp along an
  // edge the source mesh never intended as a hard edge would be a much
  // worse surprise than an honest refusal naming the exact edge.
  const topo = buildTopology(cage);
  for (const e of topo.edgeMap.values()) {
    if (e.faces.length > 2) {
      return { ok: false, reason: `an edge between vertices ${e.v0} and ${e.v1} is shared by ${e.faces.length} faces — not a manifold cage (a real edge must be shared by at most 2 faces)` };
    }
  }

  // CORNERS=YES, MESH SOURCE — a genuinely different, and genuinely
  // simpler, detection method than the NURBS-surface case above: rather
  // than trying to force a NURBS-style "exactly 4 named corners" concept
  // onto an arbitrary mesh, this detects SHARP EDGES directly (the
  // standard, well-known "auto-crease by dihedral angle" technique — the
  // angle between the two face normals meeting at an edge; DEFAULT_CORNER_
  // ANGLE_DEG's own header names the reasoning for the chosen threshold).
  // A "sharp CORNER vertex" then emerges purely as a natural CONSEQUENCE
  // of how many sharp edges converge there — 3+ meeting at one vertex
  // already reads as a true held-at-P corner via kernel/subd.mjs's own
  // EXISTING, unmodified 3+ branch, and exactly 2 meeting already reads
  // as a genuine sharp crease LINE via that same existing code — neither
  // needs the NURBS path's own new marked-corner extension at all, since
  // these are ordinary INTERIOR edges (2 real adjacent faces), not
  // boundary edges whose own stored weight the ordinary rules ignore.
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
