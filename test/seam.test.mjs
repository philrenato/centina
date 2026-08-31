// SURFACE CUTTING / SEAM STAGE — kernel/seam.mjs
//
// The discipline of this file, matching test/flatten.test.mjs's own: nothing
// here checks the cutter against itself. Topology is proven by computing the
// Euler characteristic V-E+F exactly, from scratch, on the returned mesh —
// never by trusting what `cutToDisk` reported. Geometry preservation is
// proven with STRICT EQUALITY (===), not a tolerance, because a cut has no
// business changing a coordinate at all. And the end-to-end case is checked
// against the cylinder's own analytically known unrolled rectangle, not
// against the flattener's own opinion of itself.

import test from 'node:test';
import assert from 'node:assert';
import { makeLine, makeCircle, revolve } from '../kernel/primitives.mjs';
import { weldTriangulation, flattenLSCM } from '../kernel/flatten.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import {
  cutToDisk, cutMeshAlongEdges, describeMeshTopology, boundaryComponents,
  shortestInteriorPathBetween, handleCycle, cutAndFlattenNurbsSurface,
} from '../kernel/seam.mjs';

// ---- exact, independent measurements -----------------------------------

// Euler characteristic computed here, in this file, from the returned mesh.
function eulerCharacteristic(positions, faces) {
  const used = new Set();
  const edges = new Set();
  for (const [a, b, c] of faces) {
    used.add(a); used.add(b); used.add(c);
    for (const [x, y] of [[a, b], [b, c], [c, a]]) edges.add(x < y ? `${x}|${y}` : `${y}|${x}`);
  }
  return used.size - edges.size + faces.length;
}

// Total triangle area, summed in face order so two meshes with the same
// faces in the same order give a BIT-IDENTICAL result.
function totalArea(positions, faces) {
  let s = 0;
  for (const [a, b, c] of faces) {
    const P = positions[a], Q = positions[b], R = positions[c];
    const u = [Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]];
    const v = [R[0] - P[0], R[1] - P[1], R[2] - P[2]];
    s += 0.5 * Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
  }
  return s;
}

function boundaryEdgeCount(faces) {
  const count = new Map();
  for (const [a, b, c] of faces) {
    for (const [x, y] of [[a, b], [b, c], [c, a]]) {
      const k = x < y ? `${x}|${y}` : `${y}|${x}`;
      count.set(k, (count.get(k) || 0) + 1);
    }
  }
  return [...count.values()].filter((n) => n === 1).length;
}

const uvDist = (uv, a, b) => Math.hypot(uv[a][0] - uv[b][0], uv[a][1] - uv[b][1]);

// ---- fixtures ----------------------------------------------------------
// These are the shapes this app's own Revolve actually makes. Deliberately
// non-square grids and an odd tube/major radius ratio, so a coincidence of
// equal counts cannot pass for correctness.

const CYL_R = 40, CYL_H = 55, CYL_U = 6, CYL_V = 24;
const cylinder360 = () => revolve(makeLine([CYL_R, 0, 0], [CYL_R, 0, CYL_H]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);

const CONE_R1 = 40, CONE_R2 = 16, CONE_H = 50;
const cone360 = () => revolve(makeLine([CONE_R1, 0, 0], [CONE_R2, 0, CONE_H]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);

const TORUS_MAJOR = 60, TORUS_MINOR = 18;
const torus = () => revolve(makeCircle([TORUS_MAJOR, 0, 0], [1, 0, 0], [0, 0, 1], TORUS_MINOR),
  [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);

function weldedMesh(srf, uRes, vRes) {
  return weldTriangulation(tessellateTrimmedSurface(srf, null, uRes, vRes));
}

// AN ASYMMETRIC ANNULUS, built by hand rather than revolved — the point of
// this fixture is that its seam CANNOT have equal-sized fans on both sides.
// Seven outer vertices against five inner ones, at jittered angles and
// jittered radii, strip-triangulated by advancing whichever ring's next
// vertex comes first in angle. No two vertices have the same valence, so an
// off-by-one in the vertex-splitting fan partition has nowhere to hide
// behind symmetry. Kept planar so that, once cut, an EXACT isometric
// flattening exists and any re-indexing error shows up as real edge error.
function asymmetricAnnulus() {
  const outerA = [0.05, 0.71, 1.55, 2.40, 3.35, 4.30, 5.40];
  const outerR = [30, 34, 28, 33, 29, 35, 31];
  const innerA = [0.30, 1.60, 2.90, 4.05, 5.30];
  const innerR = [12, 15, 11, 14, 13];
  const m = outerA.length, n = innerA.length;
  const positions = [];
  for (let i = 0; i < m; i++) positions.push([outerR[i] * Math.cos(outerA[i]), outerR[i] * Math.sin(outerA[i]), 0]);
  for (let j = 0; j < n; j++) positions.push([innerR[j] * Math.cos(innerA[j]), innerR[j] * Math.sin(innerA[j]), 0]);
  const O = (i) => i % m;
  const I = (j) => m + (j % n);
  const faces = [];
  let i = 0, j = 0;
  const angO = (k) => outerA[k % m] + 2 * Math.PI * Math.floor(k / m);
  const angI = (k) => innerA[k % n] + 2 * Math.PI * Math.floor(k / n);
  for (let s = 0; s < m + n; s++) {
    const takeOuter = i < m && (j >= n || angO(i + 1) <= angI(j + 1));
    if (takeOuter) { faces.push([O(i), O(i + 1), I(j)]); i++; }
    else { faces.push([O(i), I(j + 1), I(j)]); j++; }
  }
  return { positions, faces };
}

// A closed OCTAHEDRON — a genuine closed genus-0 (sphere-like) surface, for
// the refusal that says a sphere cannot be cut into one disk at all.
function octahedron() {
  const positions = [[7, 0, 0], [-5, 0, 0], [0, 9, 0], [0, -6, 0], [0, 0, 11], [0, 0, -4]];
  const faces = [
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
  ];
  return { positions, faces };
}

// A GENUS-2 closed surface: two tori, one triangle deleted from each, glued
// along the resulting triangular hole with reversed orientation (the
// connected sum). Built for the refusal test only — nothing tries to cut it.
function genus2() {
  const t = weldedMesh(torus(), 8, 10);
  const n = t.positions.length;
  const positions = [
    ...t.positions.map((p) => [p[0], p[1], p[2]]),
    ...t.positions.map((p) => [p[0] + 400, p[1], p[2]]),
  ];
  const faces1 = t.faces.map((f) => [f[0], f[1], f[2]]);
  const faces2 = t.faces.map((f) => [f[0] + n, f[1] + n, f[2] + n]);
  const [a1, b1, c1] = faces1.pop();
  const [a2, b2, c2] = faces2.pop();
  // Identify the second torus's hole with the first's, orientation reversed
  // (a2 -> a1, b2 -> c1, c2 -> b1) so the two remaining surfaces agree.
  const remap = new Map([[a2, a1], [b2, c1], [c2, b1]]);
  for (const f of faces2) for (let k = 0; k < 3; k++) if (remap.has(f[k])) f[k] = remap.get(f[k]);
  return { positions, faces: [...faces1, ...faces2] };
}

// ================================================================
// EULER CHARACTERISTIC — the strongest structural check there is
// ================================================================

test('EULER CHARACTERISTIC — a full-revolve CYLINDER is an annulus (chi 0) and cuts to a disk (chi exactly 1)', () => {
  const m = weldedMesh(cylinder360(), CYL_U, CYL_V);
  // Never trust the fixture's label: confirm it really is closed at the seam.
  const before = describeMeshTopology(m.positions, m.faces);
  assert.strictEqual(eulerCharacteristic(m.positions, m.faces), 0, 'a welded full revolve must be an annulus');
  assert.strictEqual(before.boundaryCount, 2, 'an annulus has exactly two boundary components');
  assert.strictEqual(before.genus, 0);

  const r = cutToDisk(m.positions, m.faces);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1, 'the cut mesh must be a topological disk');
  assert.strictEqual(describeMeshTopology(r.positions, r.faces).boundaryCount, 1);
  assert.strictEqual(r.cuts.length, 1, 'an annulus needs exactly one cut');
  assert.strictEqual(r.cuts[0].kind, 'boundary-path');
  assert.strictEqual(r.faces.length, m.faces.length, 'a cut never adds or removes a triangle');
});

test('EULER CHARACTERISTIC — a full-revolve CONE FRUSTUM cuts to a disk, and the seam it picks is a true ruling', () => {
  const m = weldedMesh(cone360(), 5, 18);
  assert.strictEqual(eulerCharacteristic(m.positions, m.faces), 0);
  const r = cutToDisk(m.positions, m.faces);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);
  assert.strictEqual(r.cuts.length, 1);

  // The chosen seam's own 3D length must be the frustum's SLANT length —
  // i.e. the shortest interior path really is the parametric ruling, not a
  // staircase across the grid. Computed from the cone's own geometry.
  const slant = Math.hypot(CONE_R1 - CONE_R2, CONE_H);
  assert.ok(Math.abs(r.cuts[0].length - slant) < 1e-9,
    `seam length ${r.cuts[0].length} should be the cone's own slant ${slant}`);
});

test('EULER CHARACTERISTIC — a TORUS (genus 1, no boundary) needs TWO cuts and reaches chi exactly 1', () => {
  const m = weldedMesh(torus(), 12, 16);
  const before = describeMeshTopology(m.positions, m.faces);
  assert.strictEqual(eulerCharacteristic(m.positions, m.faces), 0);
  assert.strictEqual(before.boundaryCount, 0, 'a torus is closed — no boundary at all');
  assert.strictEqual(before.genus, 1);
  assert.strictEqual(before.orientable, true);

  const r = cutToDisk(m.positions, m.faces);
  assert.strictEqual(r.cuts.length, 2, 'a torus needs two cuts: open the handle, then open the annulus');
  assert.strictEqual(r.cuts[0].kind, 'handle-cycle');
  assert.strictEqual(r.cuts[1].kind, 'boundary-path');
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);
  assert.strictEqual(describeMeshTopology(r.positions, r.faces).boundaryCount, 1);
  assert.strictEqual(r.faces.length, m.faces.length);
});

test('EULER CHARACTERISTIC — the ASYMMETRIC hand-built annulus (7 outer vs 5 inner vertices) cuts to a disk', () => {
  const { positions, faces } = asymmetricAnnulus();
  const before = describeMeshTopology(positions, faces);
  assert.strictEqual(eulerCharacteristic(positions, faces), 0, 'the fixture must genuinely be an annulus');
  assert.strictEqual(before.boundaryCount, 2);
  assert.strictEqual(before.orientable, true, 'the fixture must be consistently wound');

  // The point of this fixture: the two rings have DIFFERENT vertex counts,
  // so the strip has genuinely uneven valences and no symmetry to hide an
  // off-by-one behind.
  const valence = new Map();
  for (const [a, b, c] of faces) for (const v of [a, b, c]) valence.set(v, (valence.get(v) || 0) + 1);
  assert.ok(new Set(valence.values()).size > 1, 'fixture should have genuinely uneven valences');

  const r = cutToDisk(positions, faces);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);
  assert.strictEqual(r.cuts.length, 1);
});

// ================================================================
// GEOMETRY PRESERVATION — strict equality, not a tolerance
// ================================================================

test('GEOMETRY PRESERVING — every duplicated vertex sits at EXACTLY its twin\'s position, and total area is bit-for-bit unchanged', () => {
  for (const [label, m] of [
    ['cylinder', weldedMesh(cylinder360(), CYL_U, CYL_V)],
    ['cone', weldedMesh(cone360(), 5, 18)],
    ['torus', weldedMesh(torus(), 12, 16)],
    ['asymmetric annulus', asymmetricAnnulus()],
  ]) {
    const r = cutToDisk(m.positions, m.faces);

    // Every vertex of the cut mesh — twin or not — is bit-identical to the
    // original vertex it came from. Strict ===, on every component.
    for (let i = 0; i < r.positions.length; i++) {
      const src = m.positions[r.vertexOrigin[i]];
      assert.strictEqual(r.positions[i][0], src[0], `${label}: vertex ${i} x moved`);
      assert.strictEqual(r.positions[i][1], src[1], `${label}: vertex ${i} y moved`);
      assert.strictEqual(r.positions[i][2], src[2], `${label}: vertex ${i} z moved`);
    }

    // ...and the twins genuinely exist and genuinely coincide.
    assert.ok(r.duplicatedGroups.length > 0, `${label}: a cut must have duplicated something`);
    for (const group of r.duplicatedGroups) {
      assert.ok(group.length >= 2, `${label}: a duplicate group must have at least two members`);
      for (const v of group) {
        assert.strictEqual(r.positions[v][0], r.positions[group[0]][0]);
        assert.strictEqual(r.positions[v][1], r.positions[group[0]][1]);
        assert.strictEqual(r.positions[v][2], r.positions[group[0]][2]);
      }
    }

    // The total surface area is the same NUMBER, not a nearby one. The
    // faces are the same triangles in the same order carrying the same
    // coordinates, so any difference at all would mean geometry moved.
    assert.strictEqual(totalArea(r.positions, r.faces), totalArea(m.positions, m.faces),
      `${label}: total area must be bit-for-bit identical`);
  }
});

test('GEOMETRY PRESERVING — every face keeps its own three corner POSITIONS, in its own original order', () => {
  const m = weldedMesh(cylinder360(), CYL_U, CYL_V);
  const r = cutToDisk(m.positions, m.faces);
  assert.strictEqual(r.faces.length, m.faces.length);
  for (let f = 0; f < m.faces.length; f++) {
    for (let k = 0; k < 3; k++) {
      const before = m.positions[m.faces[f][k]];
      const after = r.positions[r.faces[f][k]];
      assert.deepStrictEqual(after, before, `face ${f} corner ${k} changed position`);
    }
  }
});

test('the cut is a real vertex SPLIT — the extra boundary is exactly two copies of the seam', () => {
  const m = weldedMesh(cylinder360(), CYL_U, CYL_V);
  const r = cutToDisk(m.positions, m.faces);
  const k = r.cuts[0].edgeCount;
  // Opening a k-edge path duplicates k+1 vertices and turns k interior
  // edges into 2k boundary edges. Both are checked here against the mesh
  // itself, not against what cutToDisk claimed.
  assert.strictEqual(r.positions.length - m.positions.length, k + 1,
    'a k-edge seam must duplicate exactly k+1 vertices');
  assert.strictEqual(boundaryEdgeCount(r.faces) - boundaryEdgeCount(m.faces), 2 * k,
    'each cut edge must become two boundary edges');
});

// ================================================================
// THE PAYOFF — cut, then flatten, against ANALYTIC ground truth
// ================================================================

test('END TO END — a genuine closed Revolve CYLINDER cuts and flattens to the analytically predicted rectangle', () => {
  const r = cutAndFlattenNurbsSurface(cylinder360(), { uRes: CYL_U, vRes: CYL_V });

  // The mesh reaching LSCM was a genuine disk, reached by exactly one cut.
  assert.strictEqual(r.seam.chiBefore, 0);
  assert.strictEqual(r.seam.chiAfter, 1);
  assert.strictEqual(r.seam.cuts.length, 1);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);

  // Confirm the fixture really is a cylinder before using cylinder maths.
  for (const p of r.positions) {
    assert.ok(Math.abs(Math.hypot(p[0], p[1]) - CYL_R) < 1e-9, 'every vertex must lie on the cylinder');
  }

  // Cutting is EXACT, so the cut-then-flattened result must reach the same
  // numerical-noise level flatten.mjs's own open-patch developable anchor
  // does (~2e-11 mm there). Tolerances stated: 1e-8 relative / 1e-6 mm
  // absolute on a patch whose edges run 10-11 mm.
  const d = r.distortion;
  assert.strictEqual(r.developable, true);
  assert.ok(d.intrinsic.maxAbsInteriorAngleDefect < 1e-12,
    `a cylinder's tessellation is exactly developable; defect ${d.intrinsic.maxAbsInteriorAngleDefect}`);
  assert.ok(d.edge.maxRelErr < 1e-8, `edge maxRelErr ${d.edge.maxRelErr}`);
  assert.ok(d.edge.maxAbsErr < 1e-6, `edge maxAbsErr ${d.edge.maxAbsErr} mm`);
  assert.ok(d.angle.maxDeg < 1e-6, `angle maxDeg ${d.angle.maxDeg}`);
  assert.ok(Math.abs(d.area.totalRatio - 1) < 1e-9, `area ratio ${d.area.totalRatio}`);
  assert.strictEqual(d.flippedTriangles, 0);

  // GROUND TRUTH from the cylinder's own equation, not from the flattener:
  // unrolling the FULL closed prism strip lays its whole polygonal
  // cross-section out straight, so the unrolled width is the sum of the
  // cross-section's chords, 2R*sin(dTheta/2) per step, all the way around;
  // the unrolled height is exactly the cylinder's own height. The seam's
  // two sides are the two ends of that rectangle, which is what makes this
  // the cut's own proof and not just the flattener's.
  //
  // The chords are summed from each vertex's OWN measured angle rather than
  // from an assumed uniform 2*pi/CYL_V step, and that distinction is real,
  // not pedantry: a full revolve is four rational quadratic arc spans, and
  // sampling those uniformly in the KNOT parameter does not sample uniformly
  // in ANGLE. Assuming equal steps here predicts 250.6103 mm against a true
  // 250.6061 mm — a 4e-3 mm error that would have been charged to the cut.
  const ang = (p) => { let a = Math.atan2(p[1], p[0]); if (a < -1e-12) a += 2 * Math.PI; return a; };
  const bottomAngles = [...new Set(
    r.positions.map((p, i) => (Math.abs(p[2]) < 1e-9 ? r.seam.vertexOrigin[i] : -1)).filter((i) => i >= 0),
  )].map((i) => ang(r.positions[r.seam.vertexOrigin.indexOf(i)])).sort((a, b) => a - b);
  assert.strictEqual(bottomAngles.length, CYL_V, 'the bottom rim should have one vertex per sweep step');
  let analyticWidth = 0;
  for (let i = 0; i < bottomAngles.length; i++) {
    const d = (bottomAngles[(i + 1) % bottomAngles.length] - bottomAngles[i] + 2 * Math.PI) % (2 * Math.PI);
    analyticWidth += 2 * CYL_R * Math.sin(d / 2);
  }

  const seamAt = (z) => {
    const g = r.seam.duplicatedGroups.filter((grp) => Math.abs(r.positions[grp[0]][2] - z) < 1e-9);
    assert.strictEqual(g.length, 1, `expected exactly one seam vertex at z=${z}`);
    assert.strictEqual(g[0].length, 2, 'a simple seam splits each of its vertices in two');
    return g[0];
  };
  const bot = seamAt(0), top = seamAt(CYL_H);

  // The two sides of the seam, at either end of the cylinder, must be the
  // full unrolled width apart.
  assert.ok(Math.abs(uvDist(r.uv, bot[0], bot[1]) - analyticWidth) < 1e-7,
    `unrolled width at the bottom: analytic ${analyticWidth}, measured ${uvDist(r.uv, bot[0], bot[1])}`);
  assert.ok(Math.abs(uvDist(r.uv, top[0], top[1]) - analyticWidth) < 1e-7,
    `unrolled width at the top: analytic ${analyticWidth}, measured ${uvDist(r.uv, top[0], top[1])}`);

  // ...and the four seam corners form a genuine RECTANGLE: two sides of
  // exactly CYL_H, and two diagonals of exactly hypot(CYL_H, width).
  const diag = Math.hypot(CYL_H, analyticWidth);
  const ds = [];
  for (const b of bot) for (const t of top) ds.push(uvDist(r.uv, b, t));
  ds.sort((x, y) => x - y);
  assert.ok(Math.abs(ds[0] - CYL_H) < 1e-7 && Math.abs(ds[1] - CYL_H) < 1e-7,
    `unrolled height should be ${CYL_H}; got ${ds[0]} and ${ds[1]}`);
  assert.ok(Math.abs(ds[2] - diag) < 1e-7 && Math.abs(ds[3] - diag) < 1e-7,
    `unrolled diagonal should be ${diag}; got ${ds[2]} and ${ds[3]}`);

  // Directly, not only by side lengths: the corner is a right angle.
  const o = r.uv[bot[0]];
  const far = uvDist(r.uv, bot[0], top[0]) < uvDist(r.uv, bot[0], top[1]) ? top[0] : top[1];
  const v1 = [r.uv[bot[1]][0] - o[0], r.uv[bot[1]][1] - o[1]];
  const v2 = [r.uv[far][0] - o[0], r.uv[far][1] - o[1]];
  const cosA = (v1[0] * v2[0] + v1[1] * v2[1]) / (Math.hypot(...v1) * Math.hypot(...v2));
  assert.ok(Math.abs(Math.acos(cosA) * 180 / Math.PI - 90) < 1e-6,
    `corner angle ${Math.acos(cosA) * 180 / Math.PI} deg`);
});

test('END TO END — the ASYMMETRIC planar annulus cuts and flattens ISOMETRICALLY (the re-indexing proof)', () => {
  // This fixture is flat, so an EXACT isometric flattening exists once it
  // is a disk. That makes LSCM's own residual a direct test of the cut's
  // re-indexing: hand any face the WRONG twin of a seam vertex and the
  // mesh's intrinsic lengths change, so the edge error blows up. Symmetry
  // cannot rescue it here — the two rings have different vertex counts.
  const { positions, faces } = asymmetricAnnulus();
  const cut = cutToDisk(positions, faces);
  const r = flattenLSCM({ positions: cut.positions, faces: cut.faces });

  assert.strictEqual(r.developable, true);
  assert.ok(r.distortion.edge.maxAbsErr < 1e-9, `edge maxAbsErr ${r.distortion.edge.maxAbsErr} mm`);
  assert.ok(r.distortion.angle.maxDeg < 1e-7, `angle maxDeg ${r.distortion.angle.maxDeg}`);
  assert.strictEqual(r.distortion.flippedTriangles, 0);

  // ...and a second, sharper statement that only holds if the cut moved
  // nothing: this annulus was ALREADY flat, so its exact unrolling is
  // itself, and the two sides of the seam must land back on top of each
  // other. (On the cylinder above they are a full circumference apart —
  // the difference between the two cases is the surface, not the cutter.)
  for (const g of cut.duplicatedGroups) {
    assert.ok(uvDist(r.uv, g[0], g[1]) < 1e-9,
      `a flat annulus unrolls to itself, so seam twins ${g} must land back together; got ${uvDist(r.uv, g[0], g[1])}`);
  }
});

test('END TO END — a TORUS flattens after two cuts, and reports its real (nonzero) distortion honestly', () => {
  const m = weldedMesh(torus(), 12, 16);
  const cut = cutToDisk(m.positions, m.faces);
  const r = flattenLSCM({ positions: cut.positions, faces: cut.faces });

  assert.strictEqual(eulerCharacteristic(cut.positions, cut.faces), 1);
  assert.strictEqual(cut.cuts.length, 2);
  assert.strictEqual(totalArea(cut.positions, cut.faces), totalArea(m.positions, m.faces));

  // A torus is genuinely doubly curved, so no cut makes it flattenable
  // without cost. The result must say so rather than claim exactness.
  assert.strictEqual(r.developable, false);
  assert.ok(r.distortion.intrinsic.maxAbsInteriorAngleDefect > 1e-3,
    `a torus has real Gaussian curvature; defect ${r.distortion.intrinsic.maxAbsInteriorAngleDefect}`);
  assert.match(r.note, /Theorema Egregium/);
  assert.ok(r.distortion.area.maxRelErr > 0.01, `area maxRelErr ${r.distortion.area.maxRelErr}`);
  for (const [u, v] of r.uv) assert.ok(Number.isFinite(u) && Number.isFinite(v));
});

// ================================================================
// SUPPORTING MACHINERY
// ================================================================

test('an input that is ALREADY a disk is returned untouched, with zero cuts — not an error', () => {
  const patch = revolve(makeLine([CYL_R, 0, 0], [CYL_R, 0, CYL_H]), [0, 0, 0], [0, 0, 1], 0, 200 * Math.PI / 180);
  const m = weldedMesh(patch, 6, 10);
  assert.strictEqual(eulerCharacteristic(m.positions, m.faces), 1);
  const r = cutToDisk(m.positions, m.faces);
  assert.strictEqual(r.cuts.length, 0);
  assert.strictEqual(r.duplicatedGroups.length, 0);
  assert.strictEqual(r.positions.length, m.positions.length);
  assert.deepStrictEqual(r.faces, m.faces);
  assert.match(r.note, /already a topological disk/);
});

test('boundaryComponents separates the two rims of an annulus and finds none on a closed torus', () => {
  const cyl = weldedMesh(cylinder360(), CYL_U, CYL_V);
  const comps = boundaryComponents(describeMeshTopology(cyl.positions, cyl.faces).topo);
  assert.strictEqual(comps.length, 2);
  assert.strictEqual(comps[0].length, CYL_V);
  assert.strictEqual(comps[1].length, CYL_V);
  // ...and they are genuinely the two different ends, not the same ring twice.
  const zOf = (c) => cyl.positions[c[0]][2];
  assert.ok(Math.abs(zOf(comps[0]) - zOf(comps[1])) > 1, 'the two rims must be at different heights');

  const tor = weldedMesh(torus(), 10, 12);
  assert.strictEqual(boundaryComponents(describeMeshTopology(tor.positions, tor.faces).topo).length, 0);
});

test('shortestInteriorPathBetween picks a genuine parametric ruling on a cylinder, weighted by real 3D length', () => {
  const m = weldedMesh(cylinder360(), CYL_U, CYL_V);
  const info = describeMeshTopology(m.positions, m.faces);
  const [c0, c1] = info.boundaryComponents;
  const path = shortestInteriorPathBetween(m.positions, m.faces, info.topo, new Set(c0), new Set(c1));
  assert.strictEqual(path.edgeKeys.length, CYL_U, 'a ruling crosses exactly uRes edges');
  assert.ok(Math.abs(path.length - CYL_H) < 1e-9, `ruling length ${path.length} should be the cylinder height ${CYL_H}`);
  // Every step is purely vertical — this really is a ruling, not a staircase.
  for (let k = 1; k < path.vertices.length; k++) {
    const a = m.positions[path.vertices[k - 1]], b = m.positions[path.vertices[k]];
    assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9, 'a ruling never moves around the cylinder');
  }
});

test('handleCycle returns a genuine simple cycle on a torus, and refuses on a sphere', () => {
  const m = weldedMesh(torus(), 10, 12);
  const info = describeMeshTopology(m.positions, m.faces);
  const cyc = handleCycle(m.positions, m.faces, info.topo);
  // A cycle: every vertex it touches is used by exactly two of its edges.
  const deg = new Map();
  for (const k of cyc.edgeKeys) {
    for (const v of k.split('|').map(Number)) deg.set(v, (deg.get(v) || 0) + 1);
  }
  assert.strictEqual(deg.size, cyc.edgeKeys.length, 'a simple cycle has as many vertices as edges');
  for (const [v, d] of deg) assert.strictEqual(d, 2, `vertex ${v} has degree ${d} in the cycle, expected 2`);
  assert.ok(cyc.length > 0 && Number.isFinite(cyc.length));

  const oct = octahedron();
  const oinfo = describeMeshTopology(oct.positions, oct.faces);
  assert.throws(() => handleCycle(oct.positions, oct.faces, oinfo.topo), /genus 0 \(a sphere\)/);

  // ...and defends its own precondition when called directly: the
  // tree-cotree edge count is only 2g on a CLOSED surface.
  const cyl = weldedMesh(cylinder360(), 4, 8);
  const cinfo = describeMeshTopology(cyl.positions, cyl.faces);
  assert.throws(() => handleCycle(cyl.positions, cyl.faces, cinfo.topo), /requires a CLOSED mesh/);
});

test('cutMeshAlongEdges leaves an INTERIOR endpoint alone — a slit that dead-ends inside the surface opens nothing there', () => {
  // A six-triangle wheel: one genuinely INTERIOR hub (vertex 0, with a
  // closed fan of faces all around it) and six boundary rim vertices.
  // Cutting the single spoke 0-1 must split the RIM vertex (its fan is an
  // arc, and the cut severs it) but must NOT split the hub (its fan is a
  // closed loop, and removing one edge from a loop leaves it connected).
  // This is the case that makes the general fan-partition rule worth
  // having instead of "duplicate every vertex on the cut."
  const positions = [[0, 0, 0]];
  const rim = 6;
  for (let k = 0; k < rim; k++) {
    const a = 2 * Math.PI * k / rim;
    positions.push([10 * Math.cos(a), 10 * Math.sin(a), 0]);
  }
  const faces = [];
  for (let k = 0; k < rim; k++) faces.push([0, 1 + k, 1 + ((k + 1) % rim)]);
  assert.strictEqual(eulerCharacteristic(positions, faces), 1);

  const r = cutMeshAlongEdges(positions, faces, ['0|1']);
  assert.strictEqual(r.positions.length, positions.length + 1, 'exactly one vertex — the rim one — should split');
  assert.strictEqual(r.duplicatedGroups.length, 1);
  assert.strictEqual(r.vertexOrigin[r.positions.length - 1], 1, 'the duplicate must be a copy of the RIM vertex');
  assert.ok(!r.duplicatedGroups.some((g) => g.includes(0)), 'the interior hub must NOT be duplicated');
  // A slit that dead-ends inside the surface leaves it a disk.
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);
  assert.strictEqual(totalArea(r.positions, r.faces), totalArea(positions, faces));
});

test('cutMeshAlongEdges refuses to "cut" an edge that is already on the boundary', () => {
  const positions = [[0, 0, 0], [10, 0, 0], [5, 8, 0]];
  const faces = [[0, 1, 2]];
  assert.throws(() => cutMeshAlongEdges(positions, faces, ['0|1']), /genuinely INTERIOR edge/);
  assert.throws(() => cutMeshAlongEdges(positions, faces, ['0|2']), /shared by 1 face/);
});

test('cutMeshAlongEdges refuses an edge that is not in the mesh at all, and an empty cut', () => {
  const positions = [[0, 0, 0], [10, 0, 0], [5, 8, 0]];
  const faces = [[0, 1, 2]];
  assert.throws(() => cutMeshAlongEdges(positions, faces, ['0|9']), /not an edge of this mesh/);
  assert.throws(() => cutMeshAlongEdges(positions, faces, []), /no edges given to cut/);
});

// ================================================================
// HONEST REFUSALS, BY NAME
// ================================================================

test('refuses a CLOSED GENUS-0 surface (a sphere) by name — no single cut makes it one disk', () => {
  const oct = octahedron();
  assert.strictEqual(eulerCharacteristic(oct.positions, oct.faces), 2);
  assert.throws(() => cutToDisk(oct.positions, oct.faces), /CLOSED GENUS-0 surface \(a sphere/);
  assert.throws(() => cutToDisk(oct.positions, oct.faces), /SPLIT into pieces/);
});

test('refuses GENUS 2 by name, and names the general cut-graph it deliberately does not build', () => {
  const g2 = genus2();
  const info = describeMeshTopology(g2.positions, g2.faces);
  assert.strictEqual(eulerCharacteristic(g2.positions, g2.faces), -2, 'the fixture must genuinely be genus 2');
  assert.strictEqual(info.boundaryCount, 0);
  assert.strictEqual(info.orientable, true, 'the connected sum must come out consistently wound');
  assert.strictEqual(info.genus, 2);
  assert.throws(() => cutToDisk(g2.positions, g2.faces), /GENUS 2/);
  assert.throws(() => cutToDisk(g2.positions, g2.faces), /Seamster/);
});

test('refuses a HANDLE WITH A BOUNDARY by name (genus 1 that is not closed)', () => {
  const m = weldedMesh(torus(), 10, 12);
  const faces = m.faces.slice(0, m.faces.length - 1); // punch one triangle out
  assert.strictEqual(eulerCharacteristic(m.positions, faces), -1);
  assert.throws(() => cutToDisk(m.positions, faces), /HANDLE AND a boundary/);
});

test('refuses a NON-ORIENTABLE / inconsistently-wound mesh by name, naming both readings', () => {
  const m = weldedMesh(cylinder360(), 4, 8);
  const faces = m.faces.map((f, i) => (i === 3 ? [f[0], f[2], f[1]] : [f[0], f[1], f[2]]));
  assert.throws(() => cutToDisk(m.positions, faces), /not consistently oriented/);
  assert.throws(() => cutToDisk(m.positions, faces), /NON-ORIENTABLE/);
});

test('refuses a NON-MANIFOLD and a DISCONNECTED mesh by name, before attempting any cut', () => {
  assert.throws(() => cutToDisk(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
    [[0, 1, 2], [0, 1, 3], [0, 1, 4]],
  ), /NON-MANIFOLD/);
  assert.throws(() => cutToDisk(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
    [[0, 1, 2], [3, 4, 5]],
  ), /DISCONNECTED/);
});

test('refuses structurally broken input by name (no triangles, a bad index, a repeated vertex)', () => {
  assert.throws(() => cutToDisk([[0, 0, 0], [1, 0, 0], [0, 1, 0]], []), /no triangles at all/);
  assert.throws(() => cutToDisk([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 7]]), /does not exist/);
  assert.throws(() => cutToDisk([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 1]]), /same vertex twice/);
});

test('cutAndFlattenNurbsSurface refuses a non-surface, and reports its seam and tessellation honestly', () => {
  assert.throws(() => cutAndFlattenNurbsSurface(null), /expected a NURBS surface/);
  const r = cutAndFlattenNurbsSurface(cone360(), { uRes: 5, vRes: 18 });
  assert.strictEqual(r.seam.cuts.length, 1);
  assert.strictEqual(r.tessellation.cutVertexCount - r.tessellation.weldedVertexCount, r.seam.cuts[0].edgeCount + 1);
  assert.match(r.seam.note, /No vertex moved/);
  assert.ok(r.distortion.edge.maxRelErr < 1e-8, `cone edge maxRelErr ${r.distortion.edge.maxRelErr}`);
});
