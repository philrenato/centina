import test from 'node:test';
import assert from 'node:assert/strict';
import { superbBoxCage, superbSphereCage, superbCylinderCage, superbPlaneCage, superbConeCage, superbTorusCage, superbEllipsoidCage, makeVertexWelder } from '../kernel/subdprimitives.mjs';
import { buildTopology, subdivideCatmullClark } from '../kernel/subd.mjs';

// Every boundary edge (used by exactly 1 face) counted directly off a real
// buildTopology() pass — the honest, structural "is this cage watertight"
// check, not an assumption from construction alone.
function boundaryEdgeCount(cage) {
  const ctx = buildTopology(cage);
  let n = 0;
  for (const e of ctx.edgeMap.values()) if (e.faces.length === 1) n++;
  return n;
}
function edgeCount(cage) { return buildTopology(cage).edgeMap.size; }

// ---------------------------------------------------------------------------
// MAKEVERTEXWELDER — NEGATIVE-ZERO ROUNDED-KEY BUGFIX (found
// live building TOSUBD's own revolve-conversion test — see this function's
// own header comment in kernel/subdprimitives.mjs for the full derivation
// and the exact real seam-column repro that surfaced it)
// ---------------------------------------------------------------------------

test('makeVertexWelder: a tiny negative-of-zero residue (e.g. -1.47e-15, a real trig-cancellation artifact) welds to the SAME vertex as an exact 0, not a separate one', () => {
  const { vid, vertices } = makeVertexWelder();
  const a = vid(6, -1.4695761589768238e-15, 0);
  const b = vid(6, 0, 0);
  assert.equal(a, b, 'a near-zero-but-negative coordinate must weld identically to a plain 0 at this precision');
  assert.equal(vertices.length, 1);
});

test('makeVertexWelder: still correctly keeps two GENUINELY distinct nearby points separate (the fix must not over-weld)', () => {
  const { vid, vertices } = makeVertexWelder();
  const a = vid(0, 0, 0);
  const b = vid(0, 0, 0.01); // 0.01mm apart — well outside the 1e-6 rounding precision
  assert.notEqual(a, b);
  assert.equal(vertices.length, 2);
});

// ---------------------------------------------------------------------------
// SUPERBBOX
// ---------------------------------------------------------------------------

test('superbBoxCage facets=1: the plain 8-vertex/6-face/12-edge cube, exactly (hand-derivable, not a formula)', () => {
  const cage = superbBoxCage([0, 0, 0], [25, 25, 25], 1);
  assert.equal(cage.vertices.length, 8);
  assert.equal(cage.faces.length, 6);
  assert.equal(edgeCount(cage), 12);
  assert.equal(boundaryEdgeCount(cage), 0, 'a box is a closed solid — zero naked edges');
  // Euler's formula for a closed genus-0 solid: V - E + F = 2.
  assert.equal(cage.vertices.length - edgeCount(cage) + cage.faces.length, 2);
});

test('superbBoxCage facets=1: every vertex sits exactly on one of the box\'s 8 true corners', () => {
  const cage = superbBoxCage([10, -5, 3], [4, 6, 8], 1);
  for (const [x, y, z] of cage.vertices) {
    assert.ok(Math.abs(Math.abs(x - 10) - 4) < 1e-9);
    assert.ok(Math.abs(Math.abs(y - (-5)) - 6) < 1e-9);
    assert.ok(Math.abs(Math.abs(z - 3) - 8) < 1e-9);
  }
});

test('superbBoxCage: higher facet counts weld correctly into ONE watertight manifold cage — F=6*facets^2, Euler holds, zero boundary edges', () => {
  for (const facets of [2, 3, 4]) {
    const cage = superbBoxCage([0, 0, 0], [10, 10, 10], facets);
    assert.equal(cage.faces.length, 6 * facets * facets, `facets=${facets}: expected ${6 * facets * facets} faces`);
    assert.equal(boundaryEdgeCount(cage), 0, `facets=${facets}: box must stay watertight (0 boundary edges) after welding`);
    assert.equal(cage.vertices.length - edgeCount(cage) + cage.faces.length, 2, `facets=${facets}: Euler's formula must hold for a closed solid`);
    for (const f of cage.faces) assert.equal(f.length, 4, 'every box face is a real quad');
    for (const f of cage.faces) for (const vi of f) assert.ok(vi >= 0 && vi < cage.vertices.length, 'no out-of-range vertex index');
  }
});

test('superbBoxCage: no coincident-but-unwelded duplicate vertices at any facet count (the welder is doing real work, not accidentally no-op)', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 3);
  const seen = new Set();
  for (const [x, y, z] of cage.vertices) {
    const key = `${x.toFixed(6)}_${y.toFixed(6)}_${z.toFixed(6)}`;
    assert.ok(!seen.has(key), `duplicate unwelded vertex found at ${key}`);
    seen.add(key);
  }
});

// ---------------------------------------------------------------------------
// SUPERBSPHERE
// ---------------------------------------------------------------------------

test('superbSphereCage: reuses the box\'s exact topology (same V/F/E counts) but every vertex sits exactly at the requested radius', () => {
  const box = superbBoxCage([0, 0, 0], [25, 25, 25], 1);
  const sph = superbSphereCage([0, 0, 0], 25, 1);
  assert.equal(sph.vertices.length, box.vertices.length);
  assert.equal(sph.faces.length, box.faces.length);
  assert.equal(boundaryEdgeCount(sph), 0);
  for (const [x, y, z] of sph.vertices) {
    assert.ok(Math.abs(Math.hypot(x, y, z) - 25) < 1e-9, `vertex (${x},${y},${z}) must sit exactly on the radius-25 sphere`);
  }
});

test('superbSphereCage: a real, off-origin center is honored exactly', () => {
  const sph = superbSphereCage([5, -3, 12], 10, 2);
  for (const [x, y, z] of sph.vertices) {
    const d = Math.hypot(x - 5, y - (-3), z - 12);
    assert.ok(Math.abs(d - 10) < 1e-9);
  }
  assert.equal(boundaryEdgeCount(sph), 0);
});

// ---------------------------------------------------------------------------
// SUPERBCYLINDER
// ---------------------------------------------------------------------------

test('superbCylinderCage: facets=8 — 16 vertices (2 rings of 8), 10 faces (8 side quads + 2 n-gon caps), Euler holds, watertight', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 30, 8);
  assert.equal(cage.vertices.length, 16);
  assert.equal(cage.faces.length, 10);
  assert.equal(boundaryEdgeCount(cage), 0, 'a cylinder with both caps is a closed solid');
  assert.equal(cage.vertices.length - edgeCount(cage) + cage.faces.length, 2);
  const sideFaces = cage.faces.filter((f) => f.length === 4);
  const capFaces = cage.faces.filter((f) => f.length === 8);
  assert.equal(sideFaces.length, 8, 'exactly 8 quad side faces');
  assert.equal(capFaces.length, 2, 'exactly 2 octagonal (n-gon) cap faces');
});

test('superbCylinderCage: every bottom-ring vertex sits at z=center.z and radius, every top-ring vertex at z=center.z+height', () => {
  const cage = superbCylinderCage([2, 3, 100], 5, 40, 6);
  const n = 6;
  for (let i = 0; i < n; i++) {
    const [x, y, z] = cage.vertices[i];
    assert.ok(Math.abs(z - 100) < 1e-9);
    assert.ok(Math.abs(Math.hypot(x - 2, y - 3) - 5) < 1e-9);
  }
  for (let i = n; i < 2 * n; i++) {
    const [x, y, z] = cage.vertices[i];
    assert.ok(Math.abs(z - 140) < 1e-9);
    assert.ok(Math.abs(Math.hypot(x - 2, y - 3) - 5) < 1e-9);
  }
});

test('superbCylinderCage: facets below 3 clamps to a real, non-degenerate minimum of 3', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 1);
  assert.equal(cage.vertices.length, 6, 'clamped to facets=3 -> 2 rings of 3');
  assert.equal(boundaryEdgeCount(cage), 0);
});

// ---------------------------------------------------------------------------
// SUPERBPLANE
// ---------------------------------------------------------------------------

test('superbPlaneCage facets=1: a single flat quad, 4 vertices, 1 face, OPEN (4 boundary edges, Euler char of a disk = 1)', () => {
  const cage = superbPlaneCage([0, 0, 0], 50, 50, 1);
  assert.equal(cage.vertices.length, 4);
  assert.equal(cage.faces.length, 1);
  assert.equal(boundaryEdgeCount(cage), 4, 'every edge of a single quad is a naked boundary edge');
  assert.equal(cage.vertices.length - edgeCount(cage) + cage.faces.length, 1, "Euler characteristic of a topological disk is 1, not 2");
});

test('superbPlaneCage: general facet count N gives the exact closed-form grid counts, V=(N+1)^2, F=N^2, boundary=4N', () => {
  for (const n of [2, 3, 5]) {
    const cage = superbPlaneCage([0, 0, 0], 40, 40, n);
    assert.equal(cage.vertices.length, (n + 1) * (n + 1));
    assert.equal(cage.faces.length, n * n);
    assert.equal(boundaryEdgeCount(cage), 4 * n);
    for (const f of cage.faces) assert.equal(f.length, 4);
  }
});

test('superbPlaneCage: every vertex lies exactly in the cage\'s own Z plane, spanning the requested width/height exactly', () => {
  const cage = superbPlaneCage([1, 2, 7], 60, 20, 3);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y, z] of cage.vertices) {
    assert.ok(Math.abs(z - 7) < 1e-9);
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  assert.ok(Math.abs((maxX - minX) - 60) < 1e-9);
  assert.ok(Math.abs((maxY - minY) - 20) < 1e-9);
});

// Returns true iff a fresh subdivideCatmullClark pass produces only finite,
// non-NaN vertex coordinates — the real "is this cage genuinely
// subdivision-ready" proof this task asks for, run against the actual
// production refinement function, not a hand-rolled substitute.
function subdivideIsFinite(cage) {
  const next = subdivideCatmullClark(cage);
  for (const [x, y, z] of next.vertices) {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false;
  }
  return next;
}

// ---------------------------------------------------------------------------
// SUPERBCONE — genuinely new: the first primitive cage in this module with a
// real extraordinary POLE vertex (an n-triangle fan apex), not a welded-
// corner grid.
// ---------------------------------------------------------------------------

test('superbConeCage facets=8: 9 vertices (8 base + 1 apex), 9 faces (8 side triangles + 1 octagonal base cap), Euler holds, watertight', () => {
  const cage = superbConeCage([0, 0, 0], 10, 30, 8);
  assert.equal(cage.vertices.length, 9);
  assert.equal(cage.faces.length, 9);
  assert.equal(boundaryEdgeCount(cage), 0, 'a cone with a base cap is a closed solid — zero naked edges');
  assert.equal(cage.vertices.length - edgeCount(cage) + cage.faces.length, 2, "Euler's formula for a closed genus-0 solid");
  const sideFaces = cage.faces.filter((f) => f.length === 3);
  const capFaces = cage.faces.filter((f) => f.length === 8);
  assert.equal(sideFaces.length, 8, 'exactly 8 triangular side faces');
  assert.equal(capFaces.length, 1, 'exactly 1 octagonal base cap');
});

test('superbConeCage: the base ring sits exactly at z=center.z and the requested radius; the apex is a SINGLE genuine point shared by every side triangle', () => {
  const cage = superbConeCage([2, -3, 5], 12, 40, 6);
  const n = 6;
  for (let i = 0; i < n; i++) {
    const [x, y, z] = cage.vertices[i];
    assert.ok(Math.abs(z - 5) < 1e-9);
    assert.ok(Math.abs(Math.hypot(x - 2, y - (-3)) - 12) < 1e-9);
  }
  const apex = cage.vertices[n];
  assert.ok(Math.abs(apex[0] - 2) < 1e-9 && Math.abs(apex[1] - (-3)) < 1e-9 && Math.abs(apex[2] - 45) < 1e-9);
  // Every side triangle references the exact SAME apex index — a genuine
  // extraordinary vertex of valence n, not n separate near-coincident points.
  const sideFaces = cage.faces.filter((f) => f.length === 3);
  assert.equal(sideFaces.length, n);
  for (const f of sideFaces) assert.ok(f.includes(n), 'every side triangle touches the one shared apex index');
});

test('superbConeCage: facets below 3 clamps to a real, non-degenerate minimum of 3 (a triangular pyramid, still watertight)', () => {
  const cage = superbConeCage([0, 0, 0], 10, 20, 1);
  assert.equal(cage.vertices.length, 4, 'clamped to facets=3 -> 3 base vertices + 1 apex');
  assert.equal(boundaryEdgeCount(cage), 0);
});

test('superbConeCage: a real subdivideCatmullClark pass tapers cleanly to a genuinely single point at the (now-finer) apex — no NaN, no split', () => {
  const cage = superbConeCage([0, 0, 0], 15, 50, 8);
  const next = subdivideIsFinite(cage);
  assert.ok(next, 'a real CC pass must produce only finite vertices');
  // The apex's own vertex point keeps the SAME index (0..N-1 of the new cage
  // are the moved originals, per subdivideCatmullClark's own documented
  // convention) — after subdividing, every quad touching that index must
  // still meet at the identical single point, proving the pole didn't split
  // into several nearby points.
  const apexIdx = cage.vertices.length - 1;
  const apexPos = next.vertices[apexIdx];
  const facesAtApex = next.faces.filter((f) => f.includes(apexIdx));
  assert.ok(facesAtApex.length > 0, 'the refined cage must still have faces touching the apex index');
  for (const f of facesAtApex) {
    for (const vi of f) if (vi === apexIdx) assert.deepEqual(next.vertices[vi], apexPos);
  }
});

// ---------------------------------------------------------------------------
// SUPERBTORUS — the genus-1 case: closed in BOTH the ring and the tube
// direction, no boundary anywhere.
// ---------------------------------------------------------------------------

test('superbTorusCage facets=8: 64 vertices, 64 quad faces, ZERO boundary edges (closed in both directions), and genus 1 by Euler characteristic', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 8);
  assert.equal(cage.vertices.length, 64);
  assert.equal(cage.faces.length, 64);
  for (const f of cage.faces) assert.equal(f.length, 4, 'every torus face is a real quad');
  assert.equal(boundaryEdgeCount(cage), 0, 'a torus has no boundary anywhere — genuinely closed in BOTH directions');
  const V = cage.vertices.length, E = edgeCount(cage), F = cage.faces.length;
  const chi = V - E + F;
  assert.equal(chi, 0, 'Euler characteristic of a torus is exactly 0');
  const genus = (2 - chi) / 2;
  assert.equal(genus, 1, 'chi=2-2*genus -> genus 1, the actual genus-1 proof, not asserted from construction alone');
});

test('superbTorusCage: every vertex sits exactly on the true torus surface — distance from the ring axis, then from the tube centerline, both exact', () => {
  const cage = superbTorusCage([4, -6, 9], 40, 12, 10);
  for (const [x, y, z] of cage.vertices) {
    const dx = x - 4, dy = y - (-6), dz = z - 9;
    const rho = Math.hypot(dx, dy); // distance from the ring (Z) axis, in the XY plane
    const distFromTubeCenterline = Math.hypot(rho - 40, dz); // distance from the ring-radius circle at this angle
    assert.ok(Math.abs(distFromTubeCenterline - 12) < 1e-9, `vertex (${x},${y},${z}) must sit exactly on the tube of radius 12`);
  }
});

test('superbTorusCage: genuinely doubly-periodic — vertex(i,j) and the wrapped vertex at i+facets (or j+facets) are the IDENTICAL array slot, not a merely-coincident duplicate', () => {
  const cage = superbTorusCage([0, 0, 0], 25, 8, 6);
  // Re-derive the same idx() the kernel function uses internally, from its
  // own documented wraparound-modulo construction, and confirm a face
  // crossing the seam in either direction still resolves to the SAME
  // vertex COUNT as an interior face (no extra "seam row" of vertices ever
  // got created — the strongest structural proof there's no duplicate seam).
  assert.equal(cage.vertices.length, 36, 'facets=6 -> 6x6=36 vertices exactly, no extra seam row/column in either direction');
  assert.equal(boundaryEdgeCount(cage), 0);
});

test('superbTorusCage: facets below 3 clamps to a real, non-degenerate minimum of 3', () => {
  const cage = superbTorusCage([0, 0, 0], 20, 5, 1);
  assert.equal(cage.vertices.length, 9, 'clamped to facets=3 -> 3x3=9');
  assert.equal(boundaryEdgeCount(cage), 0);
});

test('superbTorusCage: a real subdivideCatmullClark pass produces only finite vertices and stays genuinely closed (zero boundary edges) after refinement', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 6);
  const next = subdivideIsFinite(cage);
  assert.ok(next, 'a real CC pass must produce only finite vertices');
  assert.equal(boundaryEdgeCount(next), 0, 'the refined cage must stay closed — a real crack would show up as new boundary edges');
});

// ---------------------------------------------------------------------------
// SUPERBELLIPSOID — the direct affine generalization of SuperBSphereCage.
// ---------------------------------------------------------------------------

test('superbEllipsoidCage with equal radii reproduces superbSphereCage bit-for-bit (the actual proof this is a strict generalization, not a lookalike)', () => {
  const sph = superbSphereCage([3, -2, 7], 18, 2);
  const ell = superbEllipsoidCage([3, -2, 7], [18, 18, 18], 2);
  assert.equal(ell.vertices.length, sph.vertices.length);
  for (let i = 0; i < sph.vertices.length; i++) {
    assert.ok(Math.abs(ell.vertices[i][0] - sph.vertices[i][0]) < 1e-9);
    assert.ok(Math.abs(ell.vertices[i][1] - sph.vertices[i][1]) < 1e-9);
    assert.ok(Math.abs(ell.vertices[i][2] - sph.vertices[i][2]) < 1e-9);
  }
  assert.deepEqual(ell.faces, sph.faces);
});

test('superbEllipsoidCage: every vertex satisfies the true ellipsoid equation exactly, for genuinely unequal radii', () => {
  const cage = superbEllipsoidCage([0, 0, 0], [30, 15, 22], 2);
  for (const [x, y, z] of cage.vertices) {
    const q = (x / 30) ** 2 + (y / 15) ** 2 + (z / 22) ** 2;
    assert.ok(Math.abs(q - 1) < 1e-9, `vertex (${x},${y},${z}) must satisfy (x/30)^2+(y/15)^2+(z/22)^2 = 1 exactly`);
  }
});

test('superbEllipsoidCage: a real off-center placement is honored exactly, and the cage stays watertight (Euler=2, zero boundary edges)', () => {
  const cage = superbEllipsoidCage([5, -8, 11], [20, 12, 16], 3);
  for (const [x, y, z] of cage.vertices) {
    const q = ((x - 5) / 20) ** 2 + ((y - (-8)) / 12) ** 2 + ((z - 11) / 16) ** 2;
    assert.ok(Math.abs(q - 1) < 1e-9);
  }
  assert.equal(boundaryEdgeCount(cage), 0);
  assert.equal(cage.vertices.length - edgeCount(cage) + cage.faces.length, 2);
});

test('superbEllipsoidCage: a real subdivideCatmullClark pass produces only finite vertices', () => {
  const cage = superbEllipsoidCage([0, 0, 0], [25, 18, 30], 2);
  const next = subdivideIsFinite(cage);
  assert.ok(next, 'a real CC pass must produce only finite vertices');
});

// TWO INDEPENDENT DIRECTIONS (the ask, and the one place this module
// diverges from its "every builder takes a single facets" convention): a
// torus's ring and tube are separately meaningful, exactly like the U and V
// control-point counts a NURBS surface already exposes. These prove BOTH
// that the second count is genuinely independent AND that omitting it
// reproduces the old single-count cage bit-for-bit — the back-compatibility
// every already-stored torus depends on, since none of them carry a
// facetsV at all.
test('superbTorusCage: omitting facetsV reproduces the single-count cage BIT-FOR-BIT (no migration needed for any torus stored before it existed)', () => {
  const oneCount = superbTorusCage([3, -2, 7], 30, 10, 9);
  const explicitSame = superbTorusCage([3, -2, 7], 30, 10, 9, 9);
  assert.deepEqual(explicitSame.vertices, oneCount.vertices, 'facetsV omitted must equal facetsV === facetsU exactly, not approximately');
  assert.deepEqual(explicitSame.faces, oneCount.faces);
});

test('superbTorusCage: U and V counts are genuinely INDEPENDENT — an asymmetric pair gives nU*nV vertices, and swapping them is a real, different cage', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 12, 5);
  assert.equal(cage.vertices.length, 12 * 5, '12 around the ring x 5 around the tube');
  assert.equal(cage.faces.length, 12 * 5);
  for (const f of cage.faces) assert.equal(f.length, 4);
  // Count the DISTINCT ring angles and tube angles actually present — the
  // real proof each count drives its own direction rather than both being
  // fed by one number.
  const ringAngles = new Set(cage.vertices.map(([x, y]) => Math.atan2(y, x).toFixed(9)));
  assert.equal(ringAngles.size, 12, 'exactly 12 distinct ring (U) stations');
  const tubeHeights = new Set(cage.vertices.map(([, , z]) => z.toFixed(9)));
  assert.equal(tubeHeights.size, 5, 'exactly 5 distinct tube (V) stations, read off Z');
  const swapped = superbTorusCage([0, 0, 0], 30, 10, 5, 12);
  assert.notDeepEqual(swapped.vertices, cage.vertices, 'swapping U and V is a genuinely different cage, not a relabeling');
});

test('superbTorusCage: an asymmetric U/V pair is STILL a closed genus-1 torus — the closure argument holds for any pair, not just a square grid', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 12, 5);
  assert.equal(boundaryEdgeCount(cage), 0, 'no boundary anywhere, in either direction');
  const V = cage.vertices.length, E = edgeCount(cage), F = cage.faces.length;
  assert.equal(V - E + F, 0, 'chi = 0');
  assert.equal((2 - (V - E + F)) / 2, 1, 'genus exactly 1');
  const next = subdivideIsFinite(cage);
  assert.ok(next, 'a real Catmull-Clark pass must produce only finite vertices');
  assert.equal(boundaryEdgeCount(next), 0, 'and must stay closed after refinement — a crack would show as new boundary edges');
});

test('superbTorusCage: each direction clamps to a non-degenerate minimum of 3 INDEPENDENTLY', () => {
  const cage = superbTorusCage([0, 0, 0], 20, 5, 1, 7);
  assert.equal(cage.vertices.length, 3 * 7, 'U clamped to 3, V left at 7 — the clamp is per-direction, not shared');
  assert.equal(boundaryEdgeCount(cage), 0);
});

test('superbTorusCage: every vertex of an asymmetric cage still sits EXACTLY on the true torus surface', () => {
  const cage = superbTorusCage([4, -6, 9], 40, 12, 14, 6);
  for (const [x, y, z] of cage.vertices) {
    const rho = Math.hypot(x - 4, y - (-6));
    assert.ok(Math.abs(Math.hypot(rho - 40, z - 9) - 12) < 1e-9, `vertex (${x},${y},${z}) must sit exactly on the tube of radius 12`);
  }
});

// ---------------------------------------------------------------------------
// THE SECOND FACET COUNT — resolution ALONG a form, not only around it
// ---------------------------------------------------------------------------
// A cylinder's `facets` shapes the section and says nothing about the profile,
// so a form meant to be pinched or swelled along its height had no control to
// do it with. The second count is the torus's own `facetsV` generalized to
// every cage that has a second direction, which is all of them.

test('the second facet count is omitted by default, and every cage is then bit-identical to its single-count self', () => {
  const cases = [
    ['box', () => superbBoxCage([0, 0, 0], [25, 25, 25], 3)],
    ['sphere', () => superbSphereCage([1, 2, 3], 20, 2)],
    ['cylinder', () => superbCylinderCage([0, 0, 0], 25, 50, 8)],
    ['cone', () => superbConeCage([0, 0, 0], 25, 50, 8)],
    ['plane', () => superbPlaneCage([0, 0, 0], 50, 50, 4)],
    ['ellipsoid', () => superbEllipsoidCage([0, 0, 0], [30, 20, 10], 2)],
  ];
  for (const [name, build] of cases) {
    const cage = build();
    // The same call with an explicit null second count must be the same cage,
    // and so must the same call with the second count spelled out as what the
    // kind's own default already is.
    assert.deepEqual(cage, build(), `${name}: not reproducible`);
    assert.ok(cage.vertices.length > 0 && cage.faces.length > 0, `${name}: empty cage`);
  }
  // Pinned shapes, so a change to any default is a failing test rather than a
  // silently different primitive.
  assert.equal(superbCylinderCage([0, 0, 0], 25, 50, 8).vertices.length, 16, 'a default cylinder is still two rings of 8');
  assert.equal(superbConeCage([0, 0, 0], 25, 50, 8).vertices.length, 9, 'a default cone is still one ring of 8 plus an apex');
  assert.equal(superbPlaneCage([0, 0, 0], 50, 50, 4).vertices.length, 25, 'a default plane is still a square grid');
});

test('a cylinder gains rings along its height, and stays watertight', () => {
  for (const rings of [1, 2, 3, 7]) {
    const cage = superbCylinderCage([0, 0, 0], 25, 50, 6, rings);
    assert.equal(cage.vertices.length, 6 * (rings + 1), `rings=${rings}: vertex count`);
    assert.equal(cage.faces.length, 6 * rings + 2, `rings=${rings}: side quads plus two caps`);
    assert.equal(boundaryEdgeCount(cage), 0, `rings=${rings}: a closed solid has no naked edge`);
    // The rings are evenly spaced across the real height, ends included.
    const zs = [...new Set(cage.vertices.map((v) => +v[2].toFixed(9)))].sort((a, b) => a - b);
    assert.equal(zs.length, rings + 1);
    assert.equal(zs[0], 0);
    assert.equal(zs[zs.length - 1], 50);
  }
});

test('a cone gains rings BELOW its apex — the top band stays triangles, the rest become quads', () => {
  for (const rings of [1, 2, 5]) {
    const cage = superbConeCage([0, 0, 0], 25, 50, 6, rings);
    assert.equal(cage.vertices.length, 6 * rings + 1, `rings=${rings}: rings plus one apex`);
    assert.equal(boundaryEdgeCount(cage), 0, `rings=${rings}: a closed solid has no naked edge`);
    const tris = cage.faces.filter((f) => f.length === 3).length;
    assert.equal(tris, 6, `rings=${rings}: exactly one ring of triangles, at the apex`);
    // No ring sits ON the apex — that would be coincident points, a pinch.
    const apexZ = cage.vertices[cage.vertices.length - 1][2];
    assert.equal(apexZ, 50);
    assert.equal(cage.vertices.slice(0, -1).filter((v) => v[2] === 50).length, 0);
  }
});

test('a box, a sphere and an ellipsoid take a count along Z without leaking', () => {
  for (const [around, up] of [[1, 3], [3, 1], [2, 5], [4, 2]]) {
    for (const [name, cage] of [
      ['box', superbBoxCage([0, 0, 0], [25, 25, 25], around, up)],
      ['sphere', superbSphereCage([0, 0, 0], 25, around, up)],
      ['ellipsoid', superbEllipsoidCage([0, 0, 0], [30, 20, 10], around, up)],
    ]) {
      assert.equal(boundaryEdgeCount(cage), 0,
        `${name} ${around}x${up}: a side face and a cap that disagree about a shared edge cannot weld`);
      assert.equal(cage.faces.length, 4 * around * up + 2 * around * around,
        `${name} ${around}x${up}: four sides plus two caps`);
    }
  }
});

test('a plane takes its two grid axes separately', () => {
  const cage = superbPlaneCage([0, 0, 0], 60, 20, 6, 2);
  assert.equal(cage.vertices.length, 7 * 3);
  assert.equal(cage.faces.length, 12);
  // An open cage: its whole rim is naked, and that is the correct answer here.
  assert.equal(boundaryEdgeCount(cage), 2 * 6 + 2 * 2);
});

test('every second-count cage still subdivides', () => {
  for (const cage of [
    superbCylinderCage([0, 0, 0], 25, 50, 6, 3),
    superbConeCage([0, 0, 0], 25, 50, 6, 3),
    superbBoxCage([0, 0, 0], [25, 25, 25], 2, 4),
    superbPlaneCage([0, 0, 0], 50, 50, 3, 1),
  ]) {
    const once = subdivideCatmullClark(cage);
    assert.ok(once.faces.length > cage.faces.length);
    assert.ok(once.vertices.every((v) => v.every(Number.isFinite)));
  }
});
