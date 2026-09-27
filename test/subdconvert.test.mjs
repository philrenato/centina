import test from 'node:test';
import assert from 'node:assert/strict';
import { revolve, makeArc } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { loft } from '../kernel/loft.mjs';
import { surfacePoint, closestPointOnSurface } from '../kernel/surface.mjs';
import { buildTopology, subdivideCatmullClark, MARKED_CORNER_WEIGHT_FLOOR } from '../kernel/subd.mjs';
import { nurbsSurfaceToSuperBCage, referenceMeshToSuperBCage, DEFAULT_CORNER_CREASE_WEIGHT } from '../kernel/subdconvert.mjs';

// Fixtures: curved organic shapes rather than a plain cylinder or box, so the
// conversion is exercised on non-trivial curvature. Both sources below are
// hand-built.

// An organic vase profile — bulge/neck/bulge/neck, built by global
// curve interpolation through 5 hand-picked (radius, height) points, offset
// well clear of the revolve axis (min radius 6, axis at x=0) so the primary
// case has no pole at all. Revolved a full 360 degrees, so
// this fixture also exercises seam-welding (closed in the sweep direction).
function makeVaseRevolve() {
  const profile = globalCurveInterp([
    [6, 0, 0], [10, 0, 6], [7, 0, 11], [11, 0, 16], [8, 0, 20],
  ], 3);
  return { profile, srf: revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI) };
}

// A pole-touching profile — same organic kind, but its first point sits
// exactly on the revolve axis (x=0,y=0), a pole once revolved.
function makePoleTouchingRevolve() {
  const profile = globalCurveInterp([
    [0, 0, 0], [6, 0, 5], [9, 0, 10], [6, 0, 15],
  ], 3);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}

// An interior-pole profile — an hourglass/goblet silhouette (an organic
// shape, not a contrived edge case): its middle control point sits
// exactly on the revolve axis, not the first/last one. A pole check that
// tests only the grid's 4 outer boundary lines misses it: global curve
// interpolation guarantees the profile passes exactly through this named
// point at the same Greville abscissa the cage sampler itself uses, so an
// interior row of the sampled grid deterministically collapses to one
// physical point — without the interior check the result is `ok:true` with
// 16 degenerate (repeated-vertex) faces and a non-manifold edge.
function makeHourglassInteriorPoleRevolve() {
  const profile = globalCurveInterp([
    [6, 0, 0], [9, 0, 5], [0, 0, 10], [9, 0, 15], [6, 0, 20],
  ], 3);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}

// A Loft through 4 differently-sized, curved (arc, not straight)
// open sections at increasing height — an organic bulge-neck-bulge
// silhouette like the vase above, but exercising the
// loft path (different kernel construction) and, since each section is an
// open half-arc (sweep=PI, not a closed circle), a patch with 4 true
// corners — open in both directions, exactly what Corners=Yes/No needs to
// have anything to act on.
function makeBulgeLoft() {
  const sections = [
    makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 6, 0, Math.PI),
    makeArc([0, 0, 6], [1, 0, 0], [0, 1, 0], 10, 0, Math.PI),
    makeArc([0, 0, 12], [1, 0, 0], [0, 1, 0], 7, 0, Math.PI),
    makeArc([0, 0, 18], [1, 0, 0], [0, 1, 0], 11, 0, Math.PI),
  ];
  return loft(sections, 24, 3, 3);
}

function maxDeviationFromSurface(srf, points) {
  let worst = 0;
  for (const p of points) {
    const { distance } = closestPointOnSurface(srf, p);
    if (distance > worst) worst = distance;
  }
  return worst;
}

function bboxDiag(points) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) for (let k = 0; k < 3; k++) { if (p[k] < min[k]) min[k] = p[k]; if (p[k] > max[k]) max[k] = p[k]; }
  return Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
}

// Source 1 — NURBS surface

test('TOSUBD organic Revolve (vase): converts to a valid, watertight, correctly-sized cage', () => {
  const { profile, srf } = makeVaseRevolve();
  const result = nurbsSurfaceToSuperBCage(srf, { corners: false });
  assert.equal(result.ok, true, result.reason);
  const { cage, nu, nv } = result;
  assert.equal(nu, profile.ctrlPts.length);
  // Full 360 sweep welds the seam column into the first one — nv-1 distinct
  // circumferential columns survive, nu*(nv-1) vertices, (nu-1)*(nv-1) faces.
  assert.equal(cage.vertices.length, nu * (nv - 1));
  assert.equal(cage.faces.length, (nu - 1) * (nv - 1));
  for (const f of cage.faces) assert.equal(f.length, 4, 'every TOSUBD face must be a quad');
  // Watertight in the closed (sweep) direction: every edge in that direction
  // is interior. The only boundary is the vase's own top and bottom rim
  // (the profile is open) — exactly 2*(nv-1) boundary edges, no more.
  const topo = buildTopology(cage);
  const boundaryEdges = [...topo.edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(boundaryEdges.length, 2 * (nv - 1));
  for (const e of topo.edgeMap.values()) assert.ok(e.faces.length === 1 || e.faces.length === 2, 'no non-manifold edge from a clean revolve conversion');
});

test('TOSUBD organic Revolve (vase): the converted cage tracks the original surface, before and after Catmull-Clark refinement', () => {
  const { srf } = makeVaseRevolve();
  const { cage } = nurbsSurfaceToSuperBCage(srf, { corners: false });
  // Before any subdivision: every cage vertex was sampled directly via
  // surfacePoint, so it must sit on the true surface.
  const rawDeviation = maxDeviationFromSurface(srf, cage.vertices);
  assert.ok(rawDeviation < 1e-6, `raw cage vertices should sit on the true surface, worst deviation was ${rawDeviation}`);
  // After 2 subdivision passes (superbDisplayMesh's own "dense cage"
  // level, chooseSuperBRefinementLevel) the cage is a Catmull-Clark
  // approximation, never exact (no Stam eigen-basis limit evaluation). This
  // cage is deliberately coarse (only 5 profile samples, 8 distinct
  // circumferential columns, not over-refined to make this check pass
  // trivially), so it shrinks in a bit from the true circle radius at each
  // pass — the Catmull-Clark "coarse polygon approximating a circle"
  // behavior. The deviation must stay bounded and small relative to the
  // object's own size (well under 10% of its bounding-box diagonal), not
  // merely finite.
  let refined = cage;
  for (let i = 0; i < 2; i++) refined = subdivideCatmullClark(refined);
  const diag = bboxDiag(cage.vertices);
  const refinedDeviation = maxDeviationFromSurface(srf, refined.vertices);
  assert.ok(refinedDeviation < diag * 0.10, `refined cage should stay close to the true surface (diag=${diag.toFixed(2)}), worst deviation was ${refinedDeviation.toFixed(4)}`);
});

test('TOSUBD organic Loft (4-section bulge): converts to a valid cage with 4 true corners, and Corners=No adds zero creases', () => {
  const srf = makeBulgeLoft();
  const result = nurbsSurfaceToSuperBCage(srf, { corners: false });
  assert.equal(result.ok, true, result.reason);
  const { cage, nu, nv } = result;
  assert.equal(cage.vertices.length, nu * nv, 'open in both directions — no seam to weld away');
  assert.equal(cage.faces.length, (nu - 1) * (nv - 1));
  assert.equal(Object.keys(cage.creases).length, 0, 'Corners=No must add zero creases');
  const rawDeviation = maxDeviationFromSurface(srf, cage.vertices);
  assert.ok(rawDeviation < 1e-6, `raw cage vertices should sit on the true surface, worst deviation was ${rawDeviation}`);
});

test('TOSUBD Corners=Yes (Loft, 4 true corners): creases exactly the 4 corners\' own 8 boundary edges, at the app-matching weight', () => {
  const srf = makeBulgeLoft();
  const result = nurbsSurfaceToSuperBCage(srf, { corners: true, cornerCreaseWeight: 3 });
  assert.equal(result.ok, true, result.reason);
  const { cage, nu, nv } = result;
  const creaseKeys = Object.keys(cage.creases);
  assert.equal(creaseKeys.length, 8, '4 corners x 2 incident boundary edges each');
  for (const w of Object.values(cage.creases)) assert.equal(w, 3);
  // Every creased edge is incident to one of the 4 nominal grid
  // corners, and is a boundary edge (used by exactly 1 face).
  const topo = buildTopology(cage);
  const cornerIdxSet = new Set([0, nu - 1].flatMap((i) => [0, nv - 1].map((j) => i * nv + j)));
  for (const key of creaseKeys) {
    const [a, b] = key.split('_').map(Number);
    assert.ok(cornerIdxSet.has(a) || cornerIdxSet.has(b), `creased edge ${key} should touch a true corner vertex`);
    assert.equal(topo.edgeMap.get(key).faces.length, 1, `creased edge ${key} should be a boundary edge`);
  }
});

test('TOSUBD Corners=Yes holds a marked corner sharp through subdivision (not an inert stored weight)', () => {
  const srf = makeBulgeLoft();
  const withCorners = nurbsSurfaceToSuperBCage(srf, { corners: true }).cage;
  const withoutCorners = nurbsSurfaceToSuperBCage(srf, { corners: false }).cage;
  // Same corner vertex (index 0) in both cages (identical construction
  // otherwise) — subdivide both once and compare how far vertex 0 moved
  // from its own original position.
  const P0 = withCorners.vertices[0].slice();
  const refinedWith = subdivideCatmullClark(withCorners);
  const refinedWithout = subdivideCatmullClark(withoutCorners);
  const distWith = Math.hypot(...refinedWith.vertices[0].map((c, i) => c - P0[i]));
  const distWithout = Math.hypot(...refinedWithout.vertices[0].map((c, i) => c - P0[i]));
  assert.ok(distWith < 1e-9, `a marked corner must stay exactly at its own position, moved ${distWith}`);
  assert.ok(distWithout > 1e-6, `an unmarked corner should move (round off) under ordinary Catmull-Clark, moved only ${distWithout}`);
});

// DEFAULT_CORNER_CREASE_WEIGHT must not equal SUPERB_CREASE_LEVEL_SCALE (3),
// the weight the ordinary Crease command's "harden" gesture stores —
// otherwise TOSUBD's own corner marker and an unrelated Crease action on a
// plain SuperBPlane corner collide on the identical stored value. The
// default clears kernel/subd.mjs's own MARKED_CORNER_WEIGHT_FLOOR with
// margin, not by coincidence.
test('TOSUBD\'s own DEFAULT_CORNER_CREASE_WEIGHT clears kernel/subd.mjs\'s MARKED_CORNER_WEIGHT_FLOOR — never collides with an ordinary Crease/SoftCrease weight', () => {
  assert.ok(DEFAULT_CORNER_CREASE_WEIGHT > MARKED_CORNER_WEIGHT_FLOOR,
    `DEFAULT_CORNER_CREASE_WEIGHT (${DEFAULT_CORNER_CREASE_WEIGHT}) must clear MARKED_CORNER_WEIGHT_FLOOR (${MARKED_CORNER_WEIGHT_FLOOR}) with margin`);
});

test('TOSUBD refuses: a trimmed surface', () => {
  const { srf } = makeVaseRevolve();
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  // An interior trim loop, not the full untrimmed rectangle.
  const uMid = (uMin + uMax) / 2, vMid = (vMin + vMax) / 2;
  const trimLoop = [[uMin, vMin], [uMid, vMin], [uMid, vMid], [uMin, vMid]];
  const result = nurbsSurfaceToSuperBCage(srf, { trimLoop });
  assert.equal(result.ok, false);
  assert.match(result.reason, /trim/i);
});

test('TOSUBD does not refuse a trimLoop that is just the trivial full rectangle (not trimmed)', () => {
  const { srf } = makeVaseRevolve();
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const fullRect = [[uMin, vMin], [uMax, vMin], [uMax, vMax], [uMin, vMax]];
  const result = nurbsSurfaceToSuperBCage(srf, { trimLoop: fullRect });
  assert.equal(result.ok, true, result.reason);
});

test('TOSUBD refuses: a profile that touches its own revolve axis (a pole)', () => {
  const srf = makePoleTouchingRevolve();
  const result = nurbsSurfaceToSuperBCage(srf, { corners: false });
  assert.equal(result.ok, false);
  assert.match(result.reason, /pole|axis/i);
});

// See makeHourglassInteriorPoleRevolve's own comment for why this fixture,
// not the boundary-touching one above, exercises the interior pole check.
test('TOSUBD refuses: a profile that touches its own revolve axis at an interior control point (an hourglass/goblet silhouette), not just at the boundary', () => {
  const srf = makeHourglassInteriorPoleRevolve();
  const result = nurbsSurfaceToSuperBCage(srf, { corners: false });
  assert.equal(result.ok, false, 'must refuse rather than build a cage with degenerate faces');
  assert.match(result.reason, /pole|axis/i);
});

test('TOSUBD (surface): a degenerate 1x1-control-point-direction surface refuses rather than building a garbage cage', () => {
  const srf = { degU: 1, knotsU: [0, 0, 1, 1], degV: 1, knotsV: [0, 0, 1, 1], ctrlNet: [[[0, 0, 0, 1], [0, 1, 0, 1]]] };
  const result = nurbsSurfaceToSuperBCage(srf);
  assert.equal(result.ok, false);
});

// Source 2 — already-quad mesh (a ReferenceMesh's own refFaces)

function makeQuadCube() {
  const positions = [
    [-5, -5, -5], [5, -5, -5], [5, 5, -5], [-5, 5, -5],
    [-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5],
  ];
  const faces = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4],
  ];
  return { positions, faces };
}

test('TOSUBD (mesh): an already-quad cube mesh converts cleanly, manifold, zero creases at Corners=No', () => {
  const { positions, faces } = makeQuadCube();
  const result = referenceMeshToSuperBCage(positions, faces, { corners: false });
  assert.equal(result.ok, true, result.reason);
  assert.equal(result.cage.vertices.length, 8);
  assert.equal(result.cage.faces.length, 6);
  assert.equal(Object.keys(result.cage.creases).length, 0);
  const topo = buildTopology(result.cage);
  for (const e of topo.edgeMap.values()) assert.equal(e.faces.length, 2, 'a closed cube has zero boundary edges');
});

test('TOSUBD (mesh): Corners=Yes on a cube creases every one of its 12 edges (every dihedral angle is a hard 90 degrees)', () => {
  const { positions, faces } = makeQuadCube();
  const result = referenceMeshToSuperBCage(positions, faces, { corners: true, cornerCreaseWeight: 3 });
  assert.equal(result.ok, true, result.reason);
  const topo = buildTopology(result.cage);
  assert.equal(topo.edgeMap.size, 12);
  assert.equal(Object.keys(result.cage.creases).length, 12);
  for (const w of Object.values(result.cage.creases)) assert.equal(w, 3);
});

test('TOSUBD (mesh): welds coincident vertices at the app\'s own JOIN_TOLERANCE-scale duplication (each quad face brings its own copy of every corner)', () => {
  const positions = [];
  const faces = [];
  // The same cube, but every face gets its own 4 fresh vertex records at
  // (numerically) the identical coordinates — what an OBJ
  // export with no shared-vertex welding of its own produces.
  const cubeFaces = [
    [[-5, -5, -5], [5, -5, -5], [5, 5, -5], [-5, 5, -5]],
    [[-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5]],
    [[-5, -5, -5], [5, -5, -5], [5, -5, 5], [-5, -5, 5]],
    [[5, -5, -5], [5, 5, -5], [5, 5, 5], [5, -5, 5]],
    [[5, 5, -5], [-5, 5, -5], [-5, 5, 5], [5, 5, 5]],
    [[-5, -5, -5], [-5, 5, -5], [-5, 5, 5], [-5, -5, 5]],
  ];
  for (const f of cubeFaces) {
    const idxs = f.map((p) => { positions.push(p); return positions.length - 1; });
    faces.push(idxs);
  }
  assert.equal(positions.length, 24, 'no sharing yet — an unwelded import');
  const result = referenceMeshToSuperBCage(positions, faces, { corners: false });
  assert.equal(result.ok, true, result.reason);
  assert.equal(result.cage.vertices.length, 8, 'welding must reduce 24 duplicated records to the true 8 corners');
  const topo = buildTopology(result.cage);
  for (const e of topo.edgeMap.values()) assert.equal(e.faces.length, 2);
});

test('TOSUBD refuses: a non-quad (triangulated) mesh, naming the offending face', () => {
  // A plain tetrahedron — every face a triangle, the "already-
  // triangulated import" case the app's OBJ importer produces
  // (parseOBJText fan-triangulates).
  const positions = [[0, 0, 0], [10, 0, 0], [5, 10, 0], [5, 5, 8]];
  const faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]];
  const result = referenceMeshToSuperBCage(positions, faces);
  assert.equal(result.ok, false);
  assert.match(result.reason, /quad/i);
  assert.match(result.reason, /face 0/);
});

test('TOSUBD refuses: a non-manifold mesh (an edge shared by 3 faces)', () => {
  // Three quad "pages of a book" all hinged on the same shared edge
  // (0,1) — a deliberately non-manifold construction. Each page
  // needs 2 more distinct points forming a quad loop through 0/1.
  const p = [[0, 0, 0], [0, 0, 10], [10, 0, 5], [10, 5, 5], [-10, 0, 5], [-10, 5, 5], [0, 10, 5], [0, 15, 5]];
  const realFaces = [
    [0, 2, 3, 1],
    [0, 4, 5, 1],
    [0, 6, 7, 1],
  ];
  const result = referenceMeshToSuperBCage(p, realFaces);
  assert.equal(result.ok, false);
  assert.match(result.reason, /manifold|shared by 3/i);
});

test('TOSUBD (mesh): refuses an empty face list', () => {
  const result = referenceMeshToSuperBCage([[0, 0, 0]], []);
  assert.equal(result.ok, false);
});

test('TOSUBD (mesh): cornerAngleDeg threshold discriminates creased edges by dihedral angle', () => {
  const { positions, faces } = makeQuadCube();
  // Cube dihedral angles are all 90 degrees.
  // At threshold 45 deg, all 12 edges are creased.
  const at45 = referenceMeshToSuperBCage(positions, faces, { corners: true, cornerAngleDeg: 45 });
  assert.equal(at45.ok, true);
  assert.equal(Object.keys(at45.cage.creases).length, 12);

  // At threshold 95 deg, no edges exceed 95 deg, so 0 creases.
  const at95 = referenceMeshToSuperBCage(positions, faces, { corners: true, cornerAngleDeg: 95 });
  assert.equal(at95.ok, true);
  assert.equal(Object.keys(at95.cage.creases).length, 0);
});

test('TOSUBD (mesh): coplanar quads produce zero creases; L-shaped quads crease shared interior edge', () => {
  // Two coplanar quads sharing edge between (0,0,0) and (0,1,0)
  const coplanarPos = [
    [-1, 0, 0], [0, 0, 0], [0, 1, 0], [-1, 1, 0],
    [1, 0, 0], [1, 1, 0]
  ];
  const coplanarFaces = [
    [0, 1, 2, 3],
    [1, 4, 5, 2]
  ];
  const flatResult = referenceMeshToSuperBCage(coplanarPos, coplanarFaces, { corners: true, cornerAngleDeg: 30 });
  assert.equal(flatResult.ok, true);
  assert.equal(Object.keys(flatResult.cage.creases).length, 0);

  // Two quads meeting at 90 degrees (L-shape): face 1 on XY plane, face 2 on YZ plane
  const lPos = [
    [-1, 0, 0], [0, 0, 0], [0, 1, 0], [-1, 1, 0],
    [0, 0, 1], [0, 1, 1]
  ];
  const lFaces = [
    [0, 1, 2, 3],
    [1, 4, 5, 2]
  ];
  const lResultAt30 = referenceMeshToSuperBCage(lPos, lFaces, { corners: true, cornerAngleDeg: 30 });
  assert.equal(lResultAt30.ok, true);
  assert.equal(Object.keys(lResultAt30.cage.creases).length, 1);

  const lResultAt95 = referenceMeshToSuperBCage(lPos, lFaces, { corners: true, cornerAngleDeg: 95 });
  assert.equal(lResultAt95.ok, true);
  assert.equal(Object.keys(lResultAt95.cage.creases).length, 0);
});
