// Surface cutting and seams — kernel/seam.mjs
//
// The discipline of this file, matching test/flatten.test.mjs's own: nothing
// here checks the cutter against itself. Topology is proven by computing the
// Euler characteristic V-E+F exactly, from scratch, on the returned mesh —
// never by trusting what `cutToDisk` reported. Geometry preservation is
// proven with strict equality (===), not a tolerance, because a cut has no
// business changing a coordinate at all. And the end-to-end case is checked
// against the cylinder's own analytically known unrolled rectangle, not
// against the flattener's own opinion of itself.

import test from 'node:test';
import assert from 'node:assert';
import { makeLine, makeCircle, makeArc, revolve } from '../kernel/primitives.mjs';
import { weldTriangulation, flattenLSCM, dropDegenerateFaces } from '../kernel/flatten.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import {
  cutToDisk, cutMeshAlongEdges, describeMeshTopology, boundaryComponents,
  shortestInteriorPathBetween, handleCycle, cutAndFlattenNurbsSurface,
} from '../kernel/seam.mjs';

// Exact, independent measurements

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
// faces in the same order give a bit-identical result.
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

// Fixtures
// These are the shapes this app's own Revolve makes. Deliberately
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

// An asymmetric annulus, built by hand rather than revolved — the point of
// this fixture is that its seam cannot have equal-sized fans on both sides.
// Seven outer vertices against five inner ones, at jittered angles and
// jittered radii, strip-triangulated by advancing whichever ring's next
// vertex comes first in angle. No two vertices have the same valence, so an
// off-by-one in the vertex-splitting fan partition has nowhere to hide
// behind symmetry. Kept planar so that, once cut, an exact isometric
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

// A closed octahedron — a closed genus-0 (sphere-like) surface with uneven
// half-axes, so its farthest pair is not a coincidence of symmetry.
function octahedron() {
  const positions = [[7, 0, 0], [-5, 0, 0], [0, 9, 0], [0, -6, 0], [0, 0, 11], [0, 0, -4]];
  const faces = [
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
  ];
  return { positions, faces };
}

// A genus-2 closed surface: two tori, one triangle deleted from each, glued
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

// Euler characteristic, the strongest structural check there is

test('Euler characteristic — a full-revolve cylinder is an annulus (chi 0) and cuts to a disk (chi exactly 1)', () => {
  const m = weldedMesh(cylinder360(), CYL_U, CYL_V);
  // Never trust the fixture's label: confirm it is closed at the seam.
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

test('Euler characteristic — a full-revolve cone frustum cuts to a disk, and the seam it picks is a true ruling', () => {
  const m = weldedMesh(cone360(), 5, 18);
  assert.strictEqual(eulerCharacteristic(m.positions, m.faces), 0);
  const r = cutToDisk(m.positions, m.faces);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);
  assert.strictEqual(r.cuts.length, 1);

  // The chosen seam's own 3D length must be the frustum's slant length —
  // i.e. the shortest interior path is the parametric ruling, not a
  // staircase across the grid. Computed from the cone's own geometry.
  const slant = Math.hypot(CONE_R1 - CONE_R2, CONE_H);
  assert.ok(Math.abs(r.cuts[0].length - slant) < 1e-9,
    `seam length ${r.cuts[0].length} should be the cone's own slant ${slant}`);
});

test('Euler characteristic — a torus (genus 1, no boundary) needs two cuts and reaches chi exactly 1', () => {
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

test('Euler characteristic — the asymmetric hand-built annulus (7 outer vs 5 inner vertices) cuts to a disk', () => {
  const { positions, faces } = asymmetricAnnulus();
  const before = describeMeshTopology(positions, faces);
  assert.strictEqual(eulerCharacteristic(positions, faces), 0, 'the fixture must be an annulus');
  assert.strictEqual(before.boundaryCount, 2);
  assert.strictEqual(before.orientable, true, 'the fixture must be consistently wound');

  // The two rings have different vertex counts,
  // so the strip has uneven valences and no symmetry to hide an
  // off-by-one behind.
  const valence = new Map();
  for (const [a, b, c] of faces) for (const v of [a, b, c]) valence.set(v, (valence.get(v) || 0) + 1);
  assert.ok(new Set(valence.values()).size > 1, 'fixture should have uneven valences');

  const r = cutToDisk(positions, faces);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);
  assert.strictEqual(r.cuts.length, 1);
});

// Geometry preservation: strict equality, not a tolerance

test('geometry preserving — every duplicated vertex sits at exactly its twin\'s position, and total area is bit-for-bit unchanged', () => {
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

    // ...and the twins exist and coincide.
    assert.ok(r.duplicatedGroups.length > 0, `${label}: a cut must have duplicated something`);
    for (const group of r.duplicatedGroups) {
      assert.ok(group.length >= 2, `${label}: a duplicate group must have at least two members`);
      for (const v of group) {
        assert.strictEqual(r.positions[v][0], r.positions[group[0]][0]);
        assert.strictEqual(r.positions[v][1], r.positions[group[0]][1]);
        assert.strictEqual(r.positions[v][2], r.positions[group[0]][2]);
      }
    }

    // The total surface area is the same number, not a nearby one. The
    // faces are the same triangles in the same order carrying the same
    // coordinates, so any difference at all would mean geometry moved.
    assert.strictEqual(totalArea(r.positions, r.faces), totalArea(m.positions, m.faces),
      `${label}: total area must be bit-for-bit identical`);
  }
});

test('geometry preserving — every face keeps its own three corner positions, in its own original order', () => {
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

test('the cut is a real vertex split — the extra boundary is exactly two copies of the seam', () => {
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

// Cut, then flatten, against analytic ground truth

test('end to end — a closed Revolve cylinder cuts and flattens to the analytically predicted rectangle', () => {
  const r = cutAndFlattenNurbsSurface(cylinder360(), { uRes: CYL_U, vRes: CYL_V });

  // The mesh reaching LSCM was a disk, reached by exactly one cut.
  assert.strictEqual(r.seam.chiBefore, 0);
  assert.strictEqual(r.seam.chiAfter, 1);
  assert.strictEqual(r.seam.cuts.length, 1);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);

  // Confirm the fixture is a cylinder before using cylinder math.
  for (const p of r.positions) {
    assert.ok(Math.abs(Math.hypot(p[0], p[1]) - CYL_R) < 1e-9, 'every vertex must lie on the cylinder');
  }

  // Cutting is exact, so the cut-then-flattened result must reach the same
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

  // Ground truth from the cylinder's own equation, not from the flattener:
  // unrolling the full closed prism strip lays its whole polygonal
  // cross-section out straight, so the unrolled width is the sum of the
  // cross-section's chords, 2R*sin(dTheta/2) per step, all the way around;
  // the unrolled height is exactly the cylinder's own height. The seam's
  // two sides are the two ends of that rectangle, which is what makes this
  // the cut's own proof and not just the flattener's.
  //
  // The chords are summed from each vertex's own measured angle rather than
  // from an assumed uniform 2*pi/CYL_V step: a full revolve is four rational quadratic arc spans, and
  // sampling those uniformly in the knot parameter does not sample uniformly
  // in angle. Assuming equal steps here predicts 250.6103 mm against a true
  // 250.6061 mm, a 4e-3 mm error that would be charged to the cut.
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

  // ...and the four seam corners form a rectangle: two sides of
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

test('end to end — the asymmetric planar annulus cuts and flattens isometrically (the re-indexing proof)', () => {
  // This fixture is flat, so an exact isometric flattening exists once it
  // is a disk. That makes LSCM's own residual a direct test of the cut's
  // re-indexing: hand any face the wrong twin of a seam vertex and the
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
  // nothing: this annulus was already flat, so its exact unrolling is
  // itself, and the two sides of the seam must land back on top of each
  // other. (On the cylinder above they are a full circumference apart —
  // the difference between the two cases is the surface, not the cutter.)
  for (const g of cut.duplicatedGroups) {
    assert.ok(uvDist(r.uv, g[0], g[1]) < 1e-9,
      `a flat annulus unrolls to itself, so seam twins ${g} must land back together; got ${uvDist(r.uv, g[0], g[1])}`);
  }
});

test('end to end — a torus flattens after two cuts, and reports its real (nonzero) distortion', () => {
  const m = weldedMesh(torus(), 12, 16);
  const cut = cutToDisk(m.positions, m.faces);
  const r = flattenLSCM({ positions: cut.positions, faces: cut.faces });

  assert.strictEqual(eulerCharacteristic(cut.positions, cut.faces), 1);
  assert.strictEqual(cut.cuts.length, 2);
  assert.strictEqual(totalArea(cut.positions, cut.faces), totalArea(m.positions, m.faces));

  // A torus is doubly curved, so no cut makes it flattenable
  // without cost. The result must say so rather than claim exactness.
  assert.strictEqual(r.developable, false);
  assert.ok(r.distortion.intrinsic.maxAbsInteriorAngleDefect > 1e-3,
    `a torus has real Gaussian curvature; defect ${r.distortion.intrinsic.maxAbsInteriorAngleDefect}`);
  assert.match(r.note, /Theorema Egregium/);
  assert.ok(r.distortion.area.maxRelErr > 0.01, `area maxRelErr ${r.distortion.area.maxRelErr}`);
  for (const [u, v] of r.uv) assert.ok(Number.isFinite(u) && Number.isFinite(v));
});

// Supporting machinery

test('an input that is already a disk is returned untouched, with zero cuts — not an error', () => {
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
  // ...and they are the two different ends, not the same ring twice.
  const zOf = (c) => cyl.positions[c[0]][2];
  assert.ok(Math.abs(zOf(comps[0]) - zOf(comps[1])) > 1, 'the two rims must be at different heights');

  const tor = weldedMesh(torus(), 10, 12);
  assert.strictEqual(boundaryComponents(describeMeshTopology(tor.positions, tor.faces).topo).length, 0);
});

test('shortestInteriorPathBetween picks a parametric ruling on a cylinder, weighted by real 3D length', () => {
  const m = weldedMesh(cylinder360(), CYL_U, CYL_V);
  const info = describeMeshTopology(m.positions, m.faces);
  const [c0, c1] = info.boundaryComponents;
  const path = shortestInteriorPathBetween(m.positions, m.faces, info.topo, new Set(c0), new Set(c1));
  assert.strictEqual(path.edgeKeys.length, CYL_U, 'a ruling crosses exactly uRes edges');
  assert.ok(Math.abs(path.length - CYL_H) < 1e-9, `ruling length ${path.length} should be the cylinder height ${CYL_H}`);
  // Every step is purely vertical — this is a ruling, not a staircase.
  for (let k = 1; k < path.vertices.length; k++) {
    const a = m.positions[path.vertices[k - 1]], b = m.positions[path.vertices[k]];
    assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9, 'a ruling never moves around the cylinder');
  }
});

test('handleCycle returns a simple cycle on a torus, and refuses on a sphere', () => {
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
  // tree-cotree edge count is only 2g on a closed surface.
  const cyl = weldedMesh(cylinder360(), 4, 8);
  const cinfo = describeMeshTopology(cyl.positions, cyl.faces);
  assert.throws(() => handleCycle(cyl.positions, cyl.faces, cinfo.topo), /requires a CLOSED mesh/);
});

test('cutMeshAlongEdges leaves an interior endpoint alone — a slit that dead-ends inside the surface opens nothing there', () => {
  // A six-triangle wheel: one interior hub (vertex 0, with a
  // closed fan of faces all around it) and six boundary rim vertices.
  // Cutting the single spoke 0-1 must split the rim vertex (its fan is an
  // arc, and the cut severs it) but must not split the hub (its fan is a
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
  assert.strictEqual(r.vertexOrigin[r.positions.length - 1], 1, 'the duplicate must be a copy of the rim vertex');
  assert.ok(!r.duplicatedGroups.some((g) => g.includes(0)), 'the interior hub must not be duplicated');
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

// Refusals, by name

// Closed genus 0: one open seam

// A capsule: a cylinder of radius CAP_R and length CAP_L between two
// hemispherical caps, as latitude rings around the z axis with a vertex at
// each pole. Built by hand so it is a closed genus-0 surface that is not a
// sphere and carries no revolve grid's weld.
const CAP_R = 15, CAP_L = 50;
function capsule(around, capRings, bodyRings) {
  const prof = [];
  for (let i = 1; i < capRings; i++) { const a = -Math.PI / 2 + (i / capRings) * (Math.PI / 2); prof.push([CAP_R * Math.cos(a), CAP_R * Math.sin(a)]); }
  for (let i = 0; i <= bodyRings; i++) prof.push([CAP_R, (i / bodyRings) * CAP_L]);
  for (let i = 1; i < capRings; i++) { const a = (i / capRings) * (Math.PI / 2); prof.push([CAP_R * Math.cos(a), CAP_L + CAP_R * Math.sin(a)]); }
  const positions = [[0, 0, -CAP_R]];
  for (const [r, z] of prof) for (let k = 0; k < around; k++) { const t = (2 * Math.PI * k) / around; positions.push([r * Math.cos(t), r * Math.sin(t), z]); }
  positions.push([0, 0, CAP_L + CAP_R]);
  const top = positions.length - 1, ring = (j, k) => 1 + j * around + (k % around);
  const faces = [];
  for (let k = 0; k < around; k++) faces.push([0, ring(0, k + 1), ring(0, k)]);
  for (let j = 0; j + 1 < prof.length; j++) {
    for (let k = 0; k < around; k++) {
      faces.push([ring(j, k), ring(j, k + 1), ring(j + 1, k + 1)]);
      faces.push([ring(j, k), ring(j + 1, k + 1), ring(j + 1, k)]);
    }
  }
  const last = prof.length - 1;
  for (let k = 0; k < around; k++) faces.push([top, ring(last, k), ring(last, k + 1)]);
  return { positions, faces };
}

// A closed box, each face an n x n grid of split squares, vertices shared
// across the edges and corners.
function gridBox(n, half) {
  const positions = [], faces = [], at = new Map();
  const id = (p) => { const k = p.map((x) => x.toFixed(9)).join(','); if (!at.has(k)) { at.set(k, positions.length); positions.push(p); } return at.get(k); };
  const sides = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]];
  for (const [nn, a, b] of sides) {
    const q = (s, t) => id([0, 1, 2].map((k) => half * (nn[k] + (2 * s / n - 1) * a[k] + (2 * t / n - 1) * b[k])));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      faces.push([q(i, j), q(i + 1, j), q(i + 1, j + 1)], [q(i, j), q(i + 1, j + 1), q(i, j + 1)]);
    }
  }
  return { positions, faces };
}

// The cut mesh is one disk: chi 1, one boundary loop, no face lost, the
// area bit-identical, and LSCM lays it out with no triangle reversed.
function assertOpensToOneDisk(label, positions, faces) {
  const before = describeMeshTopology(positions, faces);
  assert.strictEqual(eulerCharacteristic(positions, faces), 2, `${label}: the fixture must be a closed genus-0 surface`);
  assert.strictEqual(before.boundaryCount, 0, `${label}: the fixture must be closed`);
  assert.strictEqual(before.orientable, true, `${label}: the fixture must be consistently wound`);
  const r = cutToDisk(positions, faces);
  assert.strictEqual(r.chiBefore, 2);
  assert.strictEqual(r.chiAfter, 1);
  assert.strictEqual(r.cuts[0].kind, 'open-seam', `${label}: the first cut is the open seam`);
  assert.ok(r.cuts[0].edgeCount >= 2, `${label}: an open seam of ${r.cuts[0].edgeCount} edge(s) splits no vertex`);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1, `${label}: chi after the cut`);
  const after = describeMeshTopology(r.positions, r.faces);
  assert.strictEqual(after.boundaryCount, 1, `${label}: one boundary loop`);
  assert.strictEqual(r.faces.length, faces.length);
  assert.strictEqual(totalArea(r.positions, r.faces), totalArea(positions, faces));
  for (let i = 0; i < r.positions.length; i++) assert.deepStrictEqual(r.positions[i], positions[r.vertexOrigin[i]]);
  const flat = flattenLSCM({ positions: r.positions, faces: r.faces });
  assert.strictEqual(flat.uv.length, r.positions.length);
  for (const [u, v] of flat.uv) assert.ok(Number.isFinite(u) && Number.isFinite(v), `${label}: a non-finite layout coordinate`);
  assert.strictEqual(flat.distortion.flippedTriangles, 0, `${label}: LSCM reversed ${flat.distortion.flippedTriangles} triangle(s)`);
  return { r, flat };
}

test('closed genus 0 — an octahedron opens along one seam to a disk and flattens', () => {
  const oct = octahedron();
  const { r } = assertOpensToOneDisk('octahedron', oct.positions, oct.faces);
  assert.strictEqual(r.cuts.length, 1);
  // The double sweep: the farthest from vertex 0 is 4 (13.04 apart), the
  // farthest from 4 is 5 (15 apart). The seam's two edges run through one
  // vertex, the one split.
  assert.strictEqual(r.cuts[0].edgeCount, 2);
  assert.strictEqual(r.cuts[0].verticesDuplicated, 1);
});

test('closed genus 0 — a tetrahedron, whose farthest pair shares an edge, still opens to a disk', () => {
  const tet = { positions: [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], faces: [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] };
  const { r } = assertOpensToOneDisk('tetrahedron', tet.positions, tet.faces);
  assert.strictEqual(r.cuts[0].edgeCount, 2);
});

test('closed genus 0 — a capsule and a closed grid box open to a disk and flatten', () => {
  const cap = capsule(16, 6, 5);
  const { r: rc } = assertOpensToOneDisk('capsule', cap.positions, cap.faces);
  // The farthest pair of a capsule is its two poles, CAP_L + 2 CAP_R apart,
  // and the seam between them runs down one side.
  assert.ok(rc.cuts[0].length >= CAP_L + 2 * CAP_R - 1e-9, `the capsule's seam is ${rc.cuts[0].length}, shorter than pole to pole`);
  assert.ok(rc.cuts[0].length <= CAP_L + Math.PI * CAP_R + 1e-9, `the capsule's seam is ${rc.cuts[0].length}, longer than a meridian`);
  const box = gridBox(4, 20);
  assertOpensToOneDisk('box', box.positions, box.faces);
});

test('closed genus 0 — a revolved NURBS sphere cuts along a diameter and flattens, reported far from developable', () => {
  const R = 40;
  const sphere = revolve(makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], R, -Math.PI / 2, Math.PI, 2), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  for (const [uRes, vRes] of [[12, 12], [8, 16], [24, 24]]) {
    const r = cutAndFlattenNurbsSurface(sphere, { uRes, vRes });
    assert.strictEqual(r.seam.chiBefore, 2, `${uRes}x${vRes}: the welded sphere is closed genus 0`);
    assert.strictEqual(r.seam.chiAfter, 1);
    assert.deepStrictEqual(r.seam.cuts.map((c) => c.kind), ['open-seam']);
    assert.strictEqual(describeMeshTopology(r.positions, r.faces).boundaryCount, 1);
    assert.strictEqual(boundaryEdgeCount(r.faces), 2 * r.seam.cuts[0].edgeCount, 'the seam is the whole boundary, two sides of it');
    // The seam's ends are an antipodal pair (every vertex of a sphere is
    // equally far from its antipode), and it is at most a meridian long.
    const ends = r.seam.vertexOrigin.length - new Set(r.seam.vertexOrigin).size + 2;
    assert.strictEqual(ends, r.seam.cuts[0].edgeCount + 1, 'every seam vertex but the two ends is split once');
    const len = r.seam.cuts[0].length;
    assert.ok(len >= 2 * R - 1e-9 && len <= Math.PI * R + 1e-9, `${uRes}x${vRes}: seam ${len} for a sphere of radius ${R}`);
    const rim = [...describeMeshTopology(r.positions, r.faces).topo.boundaryVertices].map((v) => r.positions[v]);
    let span = 0;
    for (const p of rim) for (const q of rim) span = Math.max(span, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    assert.ok(span > 2 * R * (1 - 1e-9), `${uRes}x${vRes}: the seam's ends are ${span} apart, not a diameter`);
    assert.strictEqual(r.distortion.flippedTriangles, 0);
    assert.strictEqual(r.developable, false);
    assert.ok(r.distortion.edge.maxRelErr > 0.5, `${uRes}x${vRes}: a whole sphere laid flat stretches some edge by more than half (${r.distortion.edge.maxRelErr})`);
  }
});

test('refuses genus 2 by name, and names the general cut-graph it deliberately does not build', () => {
  const g2 = genus2();
  const info = describeMeshTopology(g2.positions, g2.faces);
  assert.strictEqual(eulerCharacteristic(g2.positions, g2.faces), -2, 'the fixture must be genus 2');
  assert.strictEqual(info.boundaryCount, 0);
  assert.strictEqual(info.orientable, true, 'the connected sum must come out consistently wound');
  assert.strictEqual(info.genus, 2);
  assert.throws(() => cutToDisk(g2.positions, g2.faces), /GENUS 2/);
  assert.throws(() => cutToDisk(g2.positions, g2.faces), /Seamster/);
});

test('refuses a handle with a boundary by name (genus 1 that is not closed)', () => {
  const m = weldedMesh(torus(), 10, 12);
  const faces = m.faces.slice(0, m.faces.length - 1); // punch one triangle out
  assert.strictEqual(eulerCharacteristic(m.positions, faces), -1);
  assert.throws(() => cutToDisk(m.positions, faces), /HANDLE AND a boundary/);
});

test('refuses a non-orientable / inconsistently-wound mesh by name, naming both readings', () => {
  const m = weldedMesh(cylinder360(), 4, 8);
  const faces = m.faces.map((f, i) => (i === 3 ? [f[0], f[2], f[1]] : [f[0], f[1], f[2]]));
  assert.throws(() => cutToDisk(m.positions, faces), /not consistently oriented/);
  assert.throws(() => cutToDisk(m.positions, faces), /NON-ORIENTABLE/);
});

test('refuses a non-manifold and a disconnected mesh by name, before attempting any cut', () => {
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

test('cutAndFlattenNurbsSurface refuses a non-surface, and reports its seam and tessellation', () => {
  assert.throws(() => cutAndFlattenNurbsSurface(null), /expected a NURBS surface/);
  const r = cutAndFlattenNurbsSurface(cone360(), { uRes: 5, vRes: 18 });
  assert.strictEqual(r.seam.cuts.length, 1);
  assert.strictEqual(r.tessellation.cutVertexCount - r.tessellation.weldedVertexCount, r.seam.cuts[0].edgeCount + 1);
  assert.match(r.seam.note, /No vertex moved/);
  assert.ok(r.distortion.edge.maxRelErr < 1e-8, `cone edge maxRelErr ${r.distortion.edge.maxRelErr}`);
});

// Cone points and trimmed regions

// A full revolve that reaches the axis welds into a disk whose apex is an
// interior vertex carrying the whole angle defect 2*pi*(1 - R/L). No interior
// flattening of that vertex is isometric, so the apex is cut to the rim.
const APEX_R = 30, APEX_H = 40;
const cone360ToApex = () => revolve(makeLine([APEX_R, 0, 0], [0, 0, APEX_H]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);

test('a full cone to its apex is cut from the apex to the rim and unrolls exactly', () => {
  const m = dropDegenerateFaces(weldedMesh(cone360ToApex(), 12, 6));
  assert.strictEqual(eulerCharacteristic(m.positions, m.faces), 1, 'a full cone to its apex welds into a disk');
  const r = cutAndFlattenNurbsSurface(cone360ToApex(), { uRes: 12, vRes: 6 });
  assert.ok(r.seam.cuts.some((c) => c.kind === 'cone-point'), `expected a cone-point cut, got ${JSON.stringify(r.seam.cuts.map((c) => c.kind))}`);
  assert.strictEqual(eulerCharacteristic(r.positions, r.faces), 1);
  assert.strictEqual(r.developable, true);
  assert.ok(r.distortion.edge.maxRelErr < 1e-8, `cone edge maxRelErr ${r.distortion.edge.maxRelErr}`);
  // The sector's radius is the slant length: every rim vertex lands at
  // distance L from the apex.
  const L = Math.hypot(APEX_R, APEX_H);
  const apex = r.positions.findIndex((p) => Math.hypot(p[0], p[1]) < 1e-9);
  for (let i = 0; i < r.positions.length; i++) {
    if (Math.abs(r.positions[i][2]) > 1e-9) continue;
    assert.ok(Math.abs(uvDist(r.uv, apex, i) - L) < 1e-6 * L, `rim vertex ${i} at ${uvDist(r.uv, apex, i)} from the apex, slant ${L}`);
  }
});

test('a smooth pole is not a cone point — a hemisphere is not cut', () => {
  const hemi = revolve(makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 25, 0, Math.PI / 2, 2), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  for (const [u, v] of [[8, 8], [16, 12], [24, 24]]) {
    const m = dropDegenerateFaces(weldedMesh(hemi, u, v));
    const r = cutToDisk(m.positions, m.faces);
    assert.strictEqual(r.cuts.length, 0, `hemisphere at ${u}x${v} was cut ${r.cuts.length} time(s)`);
  }
});

test('a trimmed patch flattens its trimmed region, not the patch it sits on', () => {
  // A 187.2 x 107.2 plane trimmed to a 180 x 100 rectangle.
  const srf = { degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[[-3.6, -3.6, 0, 1], [-3.6, 103.6, 0, 1]], [[183.6, -3.6, 0, 1], [183.6, 103.6, 0, 1]]] };
  const s = (x) => (x + 3.6) / 187.2, t = (y) => (y + 3.6) / 107.2;
  const trimLoop = [[s(0), t(0)], [s(180), t(0)], [s(180), t(100)], [s(0), t(100)]];
  const r = cutAndFlattenNurbsSurface(srf, { uRes: 8, vRes: 8, trimLoop });
  const area = totalArea(r.positions, r.faces);
  assert.ok(Math.abs(area - 18000) < 1e-6, `trimmed area ${area}, expected 18000`);
  // The layout's own rotation is free, so its size is read rotation-free:
  // flat area and boundary perimeter of the 180 x 100 rectangle.
  const flat = totalArea(r.uv.map(([u, v]) => [u, v, 0]), r.faces);
  assert.ok(Math.abs(flat - 18000) < 1e-6, `flat area ${flat}, expected 18000`);
  const count = new Map();
  for (const [a, b, c] of r.faces) for (const [x, y] of [[a, b], [b, c], [c, a]]) { const k = x < y ? `${x}|${y}` : `${y}|${x}`; count.set(k, (count.get(k) || 0) + 1); }
  let perim = 0;
  for (const [k, n] of count) if (n === 1) { const [a, b] = k.split('|').map(Number); perim += uvDist(r.uv, a, b); }
  assert.ok(Math.abs(perim - 560) < 1e-6, `flat perimeter ${perim}, expected 560`);
});

test('a trimmed curved patch with a hole is cut to a disk; developable is the surface\'s verdict', () => {
  const half = revolve(makeLine([20, 0, 0], [20, 0, 30]), [0, 0, 0], [0, 0, 1], 0, Math.PI);
  const star = [];
  for (let i = 0; i < 40; i++) { const a = 2 * Math.PI * i / 40, rr = i % 2 ? 0.25 : 0.4; star.push([0.5 + 0.9 * rr * Math.cos(a), 0.5 + rr * Math.sin(a)]); }
  const hole = [];
  for (let i = 0; i < 12; i++) { const a = -2 * Math.PI * i / 12; hole.push([0.37 + 0.05 * Math.cos(a), 0.43 + 0.06 * Math.sin(a)]); }
  const err = [];
  for (const res of [24, 96]) {
    const r = cutAndFlattenNurbsSurface(half, { uRes: res, vRes: res, trimLoop: star, holes: [hole] });
    assert.strictEqual(r.seam.chiBefore, 0, 'a region with one hole is an annulus');
    assert.deepStrictEqual(r.seam.cuts.map((c) => c.kind), ['boundary-path'], `at ${res}: one boundary cut, no cone point`);
    // Developable is the surface's verdict; the residual is the clipped
    // cells' own triangulation and falls as the grid is refined.
    assert.strictEqual(r.developable, true, `at ${res}`);
    assert.ok(r.surfaceDefect < 1e-9, `surface defect ${r.surfaceDefect}`);
    err.push(r.distortion.edge.maxRelErr);
  }
  assert.ok(err[1] < err[0] / 8, `edge error does not fall with the grid: ${err}`);
});
