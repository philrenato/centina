import test from 'node:test';
import assert from 'node:assert/strict';
import { tessellateTrimmedSurface, tessellationArea, clipPolygonToRect, triangulatePolygon2D, mergeLoopsKeyhole } from '../kernel/trimtess.mjs';
import { makeLine, makeArc, extrude } from '../kernel/primitives.mjs';
import { trivialTrimLoop, signedArea2D, pointInUVPolygon } from '../kernel/trim.mjs';

// A flat, fully bilinear 10x10 plane — extrude() of a degree-1 Line is a
// ruled (degV=1) surface over a degree-1 (degU=1) profile, i.e. a true
// bilinear patch: S(u,v) = (10u, 10v, 0) exactly, domain [0,1]x[0,1] on
// both axes (makeLine's own knots). Deliberately chosen because its
// UV->XYZ Jacobian is an exact, constant 100 (10 * 10) everywhere — any
// UV-space area, multiplied by 100, is the exact world-space triangle area,
// with zero curvature-driven approximation error to account for. This is
// what makes an exact (not merely convergent) area check possible below.
function makeFlatPlane() {
  return extrude(makeLine([0, 0, 0], [10, 0, 0]), [0, 1, 0], 10);
}

// A regular N-gon inscribed in a circle of radius r centered at (cx,cy) in
// UV space — the trim data model's own shape (kernel/trim.mjs: trim curves
// are polylines, not interpolated smooth NURBS curves), not an idealized
// circle. Its own analytic area (shoelace, computed via signedArea2D — the
// same function trimLoopsValid uses) is the ground truth here, not the
// circle's pi*r^2 (which the polygon only approaches as N grows).
function makeUVPolygon(cx, cy, r, n) {
  const loop = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * 2 * Math.PI;
    loop.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return loop;
}

test('tessellateTrimmedSurface with loop=null reproduces the exact flat-plane area', () => {
  const srf = makeFlatPlane();
  const tris = tessellateTrimmedSurface(srf, null, 6, 6);
  const area = tessellationArea(tris);
  assert.ok(Math.abs(area - 100) < 1e-9, `expected exactly 100, got ${area}`);
});

test('loop=null and loop=trivialTrimLoop(srf) (the full parametric rectangle) give the identical exact area', () => {
  const srf = makeFlatPlane();
  const untrimmed = tessellationArea(tessellateTrimmedSurface(srf, null, 5, 5));
  const trivial = tessellationArea(tessellateTrimmedSurface(srf, trivialTrimLoop(srf), 5, 5));
  assert.ok(Math.abs(untrimmed - trivial) < 1e-9, `untrimmed=${untrimmed} trivial=${trivial} should match exactly`);
  assert.ok(Math.abs(untrimmed - 100) < 1e-9);
});

test('a trim polygon cuts the flat plane to exactly its own analytic area, independent of grid resolution', () => {
  const srf = makeFlatPlane();
  const loop = makeUVPolygon(0.5, 0.5, 0.3, 64); // safely interior to the [0,1]x[0,1] domain
  const expectedArea = 100 * Math.abs(signedArea2D(loop)); // exact Jacobian (100) times the polygon's own true UV area
  const coarse = tessellationArea(tessellateTrimmedSurface(srf, loop, 8, 8));
  const fine = tessellationArea(tessellateTrimmedSurface(srf, loop, 40, 40));
  assert.ok(Math.abs(coarse - expectedArea) < 1e-6, `coarse grid: expected ${expectedArea}, got ${coarse}`);
  assert.ok(Math.abs(fine - expectedArea) < 1e-6, `fine grid: expected ${expectedArea}, got ${fine}`);
  // This shows exact clipping, not a coarse "count whole cells"
  // approximation converging toward the truth: an approximate
  // method would disagree between 8x8 and 40x40 resolution; an exact
  // per-cell clip does not, since the union of (loop ∩ cell) over every
  // cell in a partition of the domain is exactly (loop ∩ domain) = loop.
  assert.ok(Math.abs(coarse - fine) < 1e-6, `coarse (${coarse}) and fine (${fine}) grids should agree exactly on a flat surface`);
});

test('every emitted triangle of a trimmed tessellation lies inside (or on the boundary of) the trim loop, never outside', () => {
  const srf = makeFlatPlane();
  const loop = makeUVPolygon(0.5, 0.5, 0.3, 24);
  const tris = tessellateTrimmedSurface(srf, loop, 20, 20);
  assert.ok(tris.length > 0);
  for (const [a, b, c] of tris) {
    const cu = (a.uv[0] + b.uv[0] + c.uv[0]) / 3;
    const cv = (a.uv[1] + b.uv[1] + c.uv[1]) / 3;
    const state = pointInUVPolygon(loop, cu, cv);
    assert.notEqual(state, 'outside', `triangle centroid (${cu},${cv}) classified ${state}, should never be outside the trim loop`);
  }
});

test('cells fully outside the trim loop are skipped, cells fully inside use the fast path, both are correct', () => {
  const srf = makeFlatPlane();
  // A small loop tucked in one corner — most of the 10x10 grid is outside it.
  const loop = makeUVPolygon(0.15, 0.15, 0.08, 32);
  const tris = tessellateTrimmedSurface(srf, loop, 10, 10);
  const area = tessellationArea(tris);
  const expected = 100 * Math.abs(signedArea2D(loop));
  assert.ok(Math.abs(area - expected) < 1e-6, `expected ${expected}, got ${area}`);
  // Far corner of the domain, nowhere near the loop, should contribute zero triangles.
  const farTris = tris.filter(([a, b, c]) => [a, b, c].every((v) => v.uv[0] > 0.5 && v.uv[1] > 0.5));
  assert.equal(farTris.length, 0, 'the far corner of the domain should contribute no geometry at all');
});

test('analytic normals are exact for a flat plane — every vertex normal is the true constant plane normal', () => {
  const srf = makeFlatPlane();
  const loop = makeUVPolygon(0.5, 0.5, 0.3, 16);
  const tris = tessellateTrimmedSurface(srf, loop, 10, 10);
  for (const [a, b, c] of tris) {
    for (const v of [a, b, c]) {
      // S(u,v) = (10u, 10v, 0) -> Su=(10,0,0), Sv=(0,10,0) -> normal = +-Z
      assert.ok(Math.abs(Math.abs(v.normal[2]) - 1) < 1e-9, `expected a unit Z normal, got ${v.normal}`);
      assert.ok(Math.abs(v.normal[0]) < 1e-9 && Math.abs(v.normal[1]) < 1e-9);
    }
  }
});

test('clipPolygonToRect: two adjacent cells clipping the same loop agree exactly along their shared edge (no crack)', () => {
  // A triangle straddling the vertical seam u=1 between cell A [0,1]x[0,1]
  // and cell B [1,2]x[0,1] — a boundary-crossing case for both cells,
  // sharing one edge of the trim loop along the way.
  const loop = [[0.2, 0.2], [1.8, 0.2], [1.0, 1.8]];
  const clippedA = clipPolygonToRect(loop, 0, 1, 0, 1);
  const clippedB = clipPolygonToRect(loop, 1, 2, 0, 1);
  const onSeam = (poly, expectedU) => poly.filter((p) => Math.abs(p[0] - expectedU) < 1e-9).map((p) => p[1]).sort((x, y) => x - y);
  const seamA = onSeam(clippedA, 1);
  const seamB = onSeam(clippedB, 1);
  assert.ok(seamA.length >= 2, 'cell A should have points on the shared u=1 seam');
  assert.equal(seamA.length, seamB.length, 'both cells should report the same number of seam points');
  for (let i = 0; i < seamA.length; i++) {
    assert.ok(Math.abs(seamA[i] - seamB[i]) < 1e-12, `seam point ${i} disagrees: A=${seamA[i]} B=${seamB[i]}`);
  }
});

test('triangulatePolygon2D handles a non-convex (reflex-corner) polygon correctly, no triangle escapes the polygon', () => {
  // An "L" shape — non-convex, the exact case a naive centroid
  // fan would get wrong.
  const poly = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]];
  const tris = triangulatePolygon2D(poly);
  assert.equal(tris.length, poly.length - 2, 'a simple polygon triangulates into exactly n-2 triangles');
  let totalArea = 0;
  for (const [i, j, k] of tris) {
    const [ax, ay] = poly[i], [bx, by] = poly[j], [cx, cy] = poly[k];
    totalArea += Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2;
  }
  const expectedArea = Math.abs(signedArea2D(poly));
  assert.ok(Math.abs(totalArea - expectedArea) < 1e-9, `triangulated area ${totalArea} should equal the polygon's true area ${expectedArea}`);
});

/* Trim holes — the keyhole/bridge merge. Holes always wind
   opposite the outer loop, matching trimLoopsValid's own convention
   — makeUVPolygon's own +CCW loop needs its point order reversed to wind
   CW for a hole (a plain .reverse(), not a different construction). */
function makeUVHole(cx, cy, r, n) { return makeUVPolygon(cx, cy, r, n).reverse(); }

test('mergeLoopsKeyhole: a single interior hole reduces the merged polygon\'s own signed area by exactly the hole\'s true area (the bridge itself contributes zero)', () => {
  const outer = makeUVPolygon(0, 0, 1, 4); // a unit-radius square-ish quad, CCW
  const hole = makeUVHole(0.3, 0.3, 0.1, 24);
  const merged = mergeLoopsKeyhole(outer, [hole]);
  const expected = signedArea2D(outer) + signedArea2D(hole); // hole's own signed area is already negative (opposite winding)
  assert.ok(Math.abs(signedArea2D(merged) - expected) < 1e-9, `merged area ${signedArea2D(merged)} should equal outer+hole=${expected}`);
});

test('mergeLoopsKeyhole: two well-separated holes both merge in, area is outer minus both holes exactly', () => {
  const outer = makeUVPolygon(0, 0, 1, 4);
  const holeA = makeUVHole(0.4, 0.4, 0.1, 16);
  const holeB = makeUVHole(-0.4, -0.4, 0.12, 16);
  const merged = mergeLoopsKeyhole(outer, [holeA, holeB]);
  const expected = signedArea2D(outer) + signedArea2D(holeA) + signedArea2D(holeB);
  assert.ok(Math.abs(signedArea2D(merged) - expected) < 1e-9, `merged area ${signedArea2D(merged)} should equal ${expected}`);
});

test('mergeLoopsKeyhole: refuses (throws) rather than silently mis-merging when a hole has no valid bridge', () => {
  // A hole whose own boundary entirely surrounds the outer loop's only
  // reachable region from every angle is contrived to construct directly;
  // instead, check the refusal the cheap way — a hole placed
  // outside the outer loop's own bbox has every candidate bridge crossing
  // the outer loop's boundary itself (the bridge must cross out of the
  // outer loop to reach a hole that isn't inside it at all).
  const outer = makeUVPolygon(0, 0, 0.2, 4);
  const farHole = makeUVHole(5, 5, 0.1, 8);
  assert.throws(() => mergeLoopsKeyhole(outer, [farHole]), /no valid non-crossing bridge/);
});

test('tessellateTrimmedSurface with a hole cuts the flat plane to exactly outer-minus-hole area, independent of grid resolution', () => {
  const srf = makeFlatPlane();
  // An off-center hole, not concentric with the outer loop; the exactly
  // concentric pair is a narrow residual case with its own tests below.
  const outer = makeUVPolygon(0.5, 0.5, 0.4, 37);
  const hole = makeUVHole(0.35, 0.65, 0.1, 23);
  const expectedArea = 100 * (Math.abs(signedArea2D(outer)) - Math.abs(signedArea2D(hole)));
  const coarse = tessellationArea(tessellateTrimmedSurface(srf, outer, 8, 8, [hole]));
  const fine = tessellationArea(tessellateTrimmedSurface(srf, outer, 40, 40, [hole]));
  assert.ok(Math.abs(coarse - expectedArea) < 1e-8, `coarse: expected ${expectedArea}, got ${coarse}`);
  assert.ok(Math.abs(fine - expectedArea) < 1e-8, `fine: expected ${expectedArea}, got ${fine}`);
  assert.ok(Math.abs(coarse - fine) < 1e-8, `coarse (${coarse}) and fine (${fine}) should agree exactly on a flat surface`);
});

test('tessellateTrimmedSurface: a hole small enough to sit entirely inside one grid cell is still excluded (per-cell subtraction cannot do this)', () => {
  const srf = makeFlatPlane();
  const outer = makeUVPolygon(0.5, 0.5, 0.45, 40); // nearly the whole [0,1]x[0,1] domain
  // A coarse 4x4 grid means each cell spans 0.25 UV units — this hole's
  // own diameter (0.1) sits comfortably inside a single cell, never
  // touching that cell's own boundary.
  const hole = makeUVHole(0.625, 0.625, 0.05, 24);
  const holeBox = { uMin: 0.575, uMax: 0.675, vMin: 0.575, vMax: 0.675 };
  assert.ok(holeBox.uMin > 0.5 && holeBox.uMax < 0.75, 'setup: the hole sits within a single 4x4 grid cell [0.5,0.75]x[0.5,0.75], never touching its edges');
  const expectedArea = 100 * (Math.abs(signedArea2D(outer)) - Math.abs(signedArea2D(hole)));
  const tris = tessellateTrimmedSurface(srf, outer, 4, 4, [hole]);
  const area = tessellationArea(tris);
  assert.ok(Math.abs(area - expectedArea) < 1e-4, `expected ${expectedArea}, got ${area} (a naive per-cell approach would get the full cell's own area here, not excluding the hole at all)`);
  for (const [a, b, c] of tris) {
    const cu = (a.uv[0] + b.uv[0] + c.uv[0]) / 3, cv = (a.uv[1] + b.uv[1] + c.uv[1]) / 3;
    assert.notEqual(pointInUVPolygon(hole, cu, cv), 'inside', `triangle centroid (${cu},${cv}) must never be classified inside the hole`);
  }
});

test('tessellateTrimmedSurface with a hole: the total area of any triangle whose centroid reads as misclassified is vanishingly small — no unbounded leak', () => {
  const srf = makeFlatPlane();
  const outer = makeUVPolygon(0.5, 0.5, 0.4, 37);
  const hole = makeUVHole(0.35, 0.65, 0.1, 23);
  const tris = tessellateTrimmedSurface(srf, outer, 20, 20, [hole]);
  assert.ok(tris.length > 0);
  // A known residual class: a rare boundary-adjacent cell (the same one the
  // ear-clip safety net logs) can leave a sliver-thin triangle whose
  // centroid reads a hair on the wrong side of a curve (the same "boundary"
  // ambiguity the single-loop suite's own equivalent test allows for, just
  // occasionally tipping into 'inside'/'outside' instead of a clean
  // 'boundary' read). The invariant checked here is that any such
  // triangle's own area is negligible, never an unbounded leak; the
  // whole-tessellation area test above is the primary,
  // resolution-independent correctness check, as elsewhere in this file.
  let misclassifiedArea = 0;
  const totalArea = tessellationArea(tris);
  for (const tri of tris) {
    const [a, b, c] = tri;
    const cu = (a.uv[0] + b.uv[0] + c.uv[0]) / 3, cv = (a.uv[1] + b.uv[1] + c.uv[1]) / 3;
    if (pointInUVPolygon(outer, cu, cv) === 'outside' || pointInUVPolygon(hole, cu, cv) === 'inside') {
      misclassifiedArea += tessellationArea([tri]);
    }
  }
  assert.ok(misclassifiedArea / totalArea < 1e-4, `misclassified area fraction ${misclassifiedArea / totalArea} should be negligible, got absolute ${misclassifiedArea} of ${totalArea}`);
});

test('tessellateTrimmedSurface: an exactly-concentric hole/outer pair (a narrow residual case) still produces a bounded, near-exact area — the safety net catches the degeneracy instead of corrupting the whole result', () => {
  const srf = makeFlatPlane();
  const outer = makeUVPolygon(0.5, 0.5, 0.35, 48);
  const hole = makeUVHole(0.5, 0.5, 0.12, 32);
  const expectedArea = 100 * (Math.abs(signedArea2D(outer)) - Math.abs(signedArea2D(hole)));
  const origError = console.error;
  let residualCount = 0;
  console.error = () => { residualCount++; };
  const area = tessellationArea(tessellateTrimmedSurface(srf, outer, 40, 40, [hole]));
  console.error = origError;
  const relErr = Math.abs(area - expectedArea) / expectedArea;
  assert.ok(relErr < 0.001, `an exactly-concentric hole/outer pair should still land within 0.1% of the true area, got relErr ${relErr}`);
});

test('tessellateTrimmedSurface: calling with no holes argument at all is byte-identical to passing an empty holes array', () => {
  const srf = makeFlatPlane();
  const loop = makeUVPolygon(0.5, 0.5, 0.3, 24);
  const withoutHolesArg = tessellationArea(tessellateTrimmedSurface(srf, loop, 10, 10));
  const withEmptyHoles = tessellationArea(tessellateTrimmedSurface(srf, loop, 10, 10, []));
  assert.equal(withoutHolesArg, withEmptyHoles, 'an empty/omitted holes array must produce the exact same result');
});

test('a curved (revolve/extrude-adjacent) surface: trivial-loop area matches the untrimmed area closely', () => {
  // A curved surface (an extruded arc, not planar) — the diagonal
  // choice for a non-planar quad can differ (very slightly) between the
  // fast emitQuad path and a clipped-then-ear-clipped quad, since a
  // non-planar quad's two possible diagonals don't split it into exactly
  // equal-area triangle pairs the way a planar quad's do — a second-
  // order geometric effect, not a bug, so this check uses a relative
  // tolerance rather than exact equality (unlike the flat-plane tests above).
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 5, 0.2, 1.0);
  const srf = extrude(profile, [0, 1, 0], 8);
  const untrimmed = tessellationArea(tessellateTrimmedSurface(srf, null, 12, 12));
  const trivial = tessellationArea(tessellateTrimmedSurface(srf, trivialTrimLoop(srf), 12, 12));
  assert.ok(untrimmed > 0 && Number.isFinite(untrimmed));
  assert.ok(trivial > 0 && Number.isFinite(trivial));
  const relErr = Math.abs(untrimmed - trivial) / untrimmed;
  assert.ok(relErr < 0.02, `expected close agreement, untrimmed=${untrimmed} trivial=${trivial} relErr=${relErr}`);
});

// Manifoldness. Every test above measures area, and area is exactly what
// a zero-width collinear needle spliced in by Sutherland-Hodgman does not
// disturb: the needle contributes nothing to the shoelace sum, so the
// trimmed area stays right while the welded mesh is non-manifold. These
// tests measure the topology of the welded result directly, which is what
// a renderer, an STL export and a watertightness check consume.

// Weld the non-indexed triangle soup by exact-ish 3D position. Deliberately
// a tight tolerance: loosening the weld is how a structural defect becomes
// an invisible one, so anything that only welds at a sloppy tolerance must
// be treated as a crack, not smoothed over here.
function weldMesh(tris, tol = 1e-7) {
  const verts = [];
  const map = new Map();
  const faces = [];
  const key = (p) => p.map((c) => Math.round(c / tol)).join(',');
  for (const tri of tris) {
    const idx = [];
    for (const v of tri) {
      const k = key(v.position);
      let i = map.get(k);
      if (i === undefined) { i = verts.length; verts.push(v.position.slice()); map.set(k, i); }
      idx.push(i);
    }
    if (idx[0] === idx[1] || idx[1] === idx[2] || idx[0] === idx[2]) continue; // degenerate, carries no topology
    faces.push(idx);
  }
  const edgeUse = new Map();
  for (const f of faces) {
    for (let i = 0; i < 3; i++) {
      const a = f[i], b = f[(i + 1) % 3];
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      edgeUse.set(k, (edgeUse.get(k) || 0) + 1);
    }
  }
  const V = verts.length, E = edgeUse.size, F = faces.length;
  const nonManifold = [...edgeUse.values()].filter((c) => c > 2).length;
  return { verts, faces, edgeUse, V, E, F, chi: V - E + F, nonManifold };
}

// A vertex lying strictly in the interior of some other face's edge. This is
// the crack a renderer shows as a hairline and an exporter shows as a leak.
function countTJunctions(welded, tol = 1e-6) {
  const { verts, edgeUse } = welded;
  const edges = [...edgeUse.keys()].map((k) => k.split(',').map(Number));
  let n = 0;
  for (let vi = 0; vi < verts.length; vi++) {
    const p = verts[vi];
    for (const [a, b] of edges) {
      if (a === vi || b === vi) continue;
      const A = verts[a], B = verts[b];
      const x = B[0] - A[0], y = B[1] - A[1], z = B[2] - A[2];
      const L2 = x * x + y * y + z * z;
      if (L2 < 1e-18) continue;
      const t = ((p[0] - A[0]) * x + (p[1] - A[1]) * y + (p[2] - A[2]) * z) / L2;
      if (t <= 1e-9 || t >= 1 - 1e-9) continue;
      if (Math.hypot(p[0] - (A[0] + t * x), p[1] - (A[1] + t * y), p[2] - (A[2] + t * z)) < tol) { n++; break; }
    }
  }
  return n;
}

test('manifold: an ordinary off-center hole welds into an annulus — chi == 0, zero non-manifold edges, zero T-junctions, exact area', () => {
  const srf = makeFlatPlane();
  const outer = trivialTrimLoop(srf);          // the full parametric rectangle: a plane
  const hole = makeUVHole(0.38, 0.44, 0.22, 32); // off-center
  const expectedArea = 100 * (Math.abs(signedArea2D(outer)) - Math.abs(signedArea2D(hole)));
  const tris = tessellateTrimmedSurface(srf, outer, 10, 10, [hole]);
  const w = weldMesh(tris);
  assert.equal(w.nonManifold, 0, `every edge must be used by at most two faces, got ${w.nonManifold} used by three or more`);
  assert.equal(countTJunctions(w), 0, 'no vertex may lie in the interior of another face\'s edge');
  // A disk with exactly one hole is an annulus: V - E + F == 0, exactly.
  assert.equal(w.chi, 0, `a disk-with-one-hole must have Euler characteristic 0, got ${w.chi} (V=${w.V} E=${w.E} F=${w.F})`);
  // A manifold mesh with the wrong area is not a pass — the flat plane's
  // Jacobian is an exact constant 100, so this is an exact claim.
  const area = tessellationArea(tris);
  assert.ok(Math.abs(area - expectedArea) < 1e-8, `expected ${expectedArea}, got ${area}`);
});

test('manifold: a hole crossing grid lines at a shallow angle (where seam artifacts are worst) is still manifold and exact', () => {
  const srf = makeFlatPlane();
  const outer = trivialTrimLoop(srf);
  // Deliberately irrational-looking center and a radius that puts long,
  // near-tangent arc runs along several cell edges — the configuration a
  // crossing-point mismatch would show up in first.
  const hole = makeUVHole(0.413, 0.527, 0.25, 48);
  const expectedArea = 100 * (Math.abs(signedArea2D(outer)) - Math.abs(signedArea2D(hole)));
  for (const res of [7, 10, 13]) {
    const tris = tessellateTrimmedSurface(srf, outer, res, res, [hole]);
    const w = weldMesh(tris);
    assert.equal(w.nonManifold, 0, `res ${res}: ${w.nonManifold} non-manifold edges`);
    assert.equal(countTJunctions(w), 0, `res ${res}: T-junctions present`);
    assert.equal(w.chi, 0, `res ${res}: expected chi 0, got ${w.chi}`);
    const area = tessellationArea(tris);
    assert.ok(Math.abs(area - expectedArea) < 1e-8, `res ${res}: expected ${expectedArea}, got ${area}`);
  }
});

test('manifold: an untrimmed-rectangle trim (no holes) welds into a plain disk — chi == 1, and the repair passes leave it alone', () => {
  const srf = makeFlatPlane();
  const w = weldMesh(tessellateTrimmedSurface(srf, trivialTrimLoop(srf), 10, 10));
  assert.equal(w.nonManifold, 0);
  assert.equal(countTJunctions(w), 0);
  assert.equal(w.chi, 1, `a plain disk must have Euler characteristic 1, got ${w.chi}`);
  /* The claim is the shape, not the grid. A planar face is triangulated from
     its trim loop rather than sampled on a 10x10 grid it has no curvature to
     justify, so the triangle count is not the grid's 200; two triangles are
     the same surface, exactly. What protects the repair passes is that the
     result stays one manifold disk of the right area with nothing welded or
     split. */
  assert.ok(w.F >= 2, `a rectangle must survive as at least two triangles, got ${w.F}`);
  const area = tessellationArea(tessellateTrimmedSurface(srf, trivialTrimLoop(srf), 10, 10));
  assert.ok(Math.abs(area - 100) < 1e-9, `and it must still be the whole 10x10 plane, got ${area}`);
});

test('manifold: the ear-clip residual warning does not fire for an ordinary single-loop trim', () => {
  const srf = makeFlatPlane();
  const loop = makeUVPolygon(0.5, 0.5, 0.3, 37);
  const origError = console.error;
  let residuals = 0;
  console.error = () => { residuals++; };
  try { tessellateTrimmedSurface(srf, loop, 10, 10); } finally { console.error = origError; }
  assert.equal(residuals, 0, 'a plain trim with no holes must not report an ear-clip residual');
});

test('manifold: removeCollinearSpikes\' own case — the area-neutral needle S-H splices in breaks the weld, so removing it must not move the area at all', () => {
  const srf = makeFlatPlane();
  const outer = trivialTrimLoop(srf);
  const hole = makeUVHole(0.38, 0.44, 0.22, 32);
  const expectedArea = 100 * (Math.abs(signedArea2D(outer)) - Math.abs(signedArea2D(hole)));
  // Resolution-independence shows this is exact clipping and not
  // a repair that happens to cancel out at one grid size.
  const areas = [6, 10, 16, 24].map((r) => tessellationArea(tessellateTrimmedSurface(srf, outer, r, r, [hole])));
  for (let i = 0; i < areas.length; i++) {
    assert.ok(Math.abs(areas[i] - expectedArea) < 1e-8, `resolution index ${i}: expected ${expectedArea}, got ${areas[i]}`);
  }
});

test('manifold: an exactly concentric hole (the residual case above) welds manifold, with exact area', () => {
  const srf = makeFlatPlane();
  const outer = makeUVPolygon(0.5, 0.5, 0.35, 48);
  const hole = makeUVHole(0.5, 0.5, 0.12, 32);
  const expectedArea = 100 * (Math.abs(signedArea2D(outer)) - Math.abs(signedArea2D(hole)));
  const origError = console.error;
  console.error = () => {};
  let tris;
  try { tris = tessellateTrimmedSurface(srf, outer, 40, 40, [hole]); } finally { console.error = origError; }
  const w = weldMesh(tris);
  // Topology is clean for this case too.
  assert.equal(w.nonManifold, 0, `concentric pair: ${w.nonManifold} non-manifold edges`);
  assert.equal(countTJunctions(w), 0, 'concentric pair: T-junctions present');
  // Area for this fixture is exact, tighter than the 0.1% bound asserted
  // above. That bound is a property of this fixture and grid resolution,
  // not a general guarantee — see the module header.
  const area = tessellationArea(tris);
  assert.ok(Math.abs(area - expectedArea) < 1e-6, `concentric pair: expected ${expectedArea}, got ${area}`);
});
