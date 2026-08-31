import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPlanarArrangement, mergeFaces } from '../kernel/arrangement.mjs';
import { signedArea2D } from '../kernel/trim.mjs';

// Sorts a face's outer/hole loops so comparisons don't depend on which
// vertex the half-edge walk happened to start at, or which rotation
// direction landed first in the faces array.
function loopsMatch(a, b, tol = 1e-6) {
  if (a.length !== b.length) return false;
  const n = a.length;
  for (let offset = 0; offset < n; offset++) {
    let ok = true;
    for (let i = 0; i < n; i++) {
      const pa = a[i], pb = b[(i + offset) % n];
      if (Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) > tol) { ok = false; break; }
    }
    if (ok) return true;
    // also try reversed direction
    ok = true;
    for (let i = 0; i < n; i++) {
      const pa = a[i], pb = b[(offset - i + n) % n];
      if (Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) > tol) { ok = false; break; }
    }
    if (ok) return true;
  }
  return false;
}

test('a single closed square produces exactly one bounded face with no holes', () => {
  const square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const arr = buildPlanarArrangement([square]);
  assert.equal(arr.faces.length, 1);
  assert.equal(arr.faces[0].holes.length, 0);
  assert.ok(loopsMatch(arr.faces[0].outer, [[0, 0], [10, 0], [10, 10], [0, 10]]));
  assert.ok(Math.abs(Math.abs(signedArea2D(arr.faces[0].outer)) - 100) < 1e-6);
});

test('two overlapping squares produce THREE regions: left-only, overlap, right-only, all mutually contiguous through the overlap', () => {
  // Square A: (0,0)-(10,0)-(10,10)-(0,10). Square B: (5,0)-(15,0)-(15,10)-(5,10).
  // Overlap region: x in [5,10], y in [0,10].
  const a = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const b = [[5, 0], [15, 0], [15, 10], [5, 10], [5, 0]];
  const arr = buildPlanarArrangement([a, b]);
  assert.equal(arr.faces.length, 3, `expected 3 regions, got ${arr.faces.length}`);
  const areas = arr.faces.map((f) => Math.abs(signedArea2D(f.outer))).sort((x, y) => x - y);
  // left-only (0-5 x 0-10 = 50), overlap (5-10 x 0-10 = 50), right-only (10-15 x 0-10 = 50)
  // all three are actually equal-area in this fixture (50 each) — real
  // adjacency is the thing that actually distinguishes them, checked below.
  for (const area of areas) assert.ok(Math.abs(area - 50) < 1e-6, `expected each region to be area 50, got ${area}`);
  // The overlap region must be adjacent to BOTH the other two (it borders
  // both x=5 internal edges); left-only and right-only must NOT be
  // directly adjacent to each other (they don't share a boundary at all).
  const degrees = arr.adjacency.map((n) => n.length).sort((x, y) => x - y);
  assert.deepEqual(degrees, [1, 1, 2], `expected two regions with 1 neighbor and one with 2 (the overlap), got ${JSON.stringify(degrees)}`);
});

test('THE ANNULUS CASE: a square fully inside another, not touching, produces an annulus WITH a real hole plus a separate disk — not two overlapping full squares', () => {
  const outer = [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]];
  const inner = [[5, 5], [15, 5], [15, 15], [5, 15], [5, 5]];
  const arr = buildPlanarArrangement([outer, inner]);
  assert.equal(arr.faces.length, 2, `expected 2 faces (annulus + disk), got ${arr.faces.length}`);
  const annulus = arr.faces.find((f) => f.holes.length === 1);
  const disk = arr.faces.find((f) => f.holes.length === 0);
  assert.ok(annulus, 'one face must have exactly one hole (the annulus)');
  assert.ok(disk, 'one face must have zero holes (the disk)');
  assert.ok(loopsMatch(annulus.outer, [[0, 0], [20, 0], [20, 20], [0, 20]]), 'annulus outer boundary is the big square');
  assert.ok(loopsMatch(annulus.holes[0], [[5, 5], [15, 5], [15, 15], [5, 15]]), 'annulus hole is the small square');
  assert.ok(loopsMatch(disk.outer, [[5, 5], [15, 5], [15, 15], [5, 15]]), 'disk outer boundary is the small square');
  // The annulus's true area (big minus small) — not just "some positive
  // number" — 400 - 100 = 300 by construction; not directly stored as a
  // field, but the outer/holes decomposition itself IS that area
  // implicitly (outer 400, hole 100), confirmed via each loop's own area.
  assert.ok(Math.abs(Math.abs(signedArea2D(annulus.outer)) - 400) < 1e-6);
  assert.ok(Math.abs(Math.abs(signedArea2D(annulus.holes[0])) - 100) < 1e-6);
  // Annulus and disk ARE contiguous (they share the small square's own boundary).
  assert.equal(arr.adjacency[arr.faces.indexOf(annulus)].length, 1);
  assert.equal(arr.adjacency[arr.faces.indexOf(disk)].length, 1);
});

test('a dangling open curve with no closing partner contributes NO new region — the square is unaffected', () => {
  const square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const withoutDangler = buildPlanarArrangement([square]);
  const dangler = [[3, 3], [7, 8]]; // a free line segment inside the square, touching nothing
  const withDangler = buildPlanarArrangement([square, dangler]);
  assert.equal(withDangler.faces.length, withoutDangler.faces.length, 'the dangler must not create or destroy any region');
  assert.ok(loopsMatch(withDangler.faces[0].outer, withoutDangler.faces[0].outer), 'the square face itself must be byte-identical, not corrupted by a spur');
});

test('a T-junction (a curve endpoint landing on another curve\'s body) genuinely splits the region — not ignored', () => {
  const square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  // A line from the midpoint of the bottom edge (5,0) straight up to the midpoint of the top edge (5,10) — both ends are T-junctions, not proper crossings at a corner.
  const splitter = [[5, 0], [5, 10]];
  const arr = buildPlanarArrangement([square, splitter]);
  assert.equal(arr.faces.length, 2, `expected the square split into 2 halves by the T-junction line, got ${arr.faces.length} faces`);
  for (const f of arr.faces) assert.ok(Math.abs(Math.abs(signedArea2D(f.outer)) - 50) < 1e-6, 'each half must be exactly half the square\'s own area (50)');
  assert.equal(arr.adjacency[0].length, 1);
  assert.equal(arr.adjacency[1].length, 1);
});

test('an EXACTLY coincident duplicate curve (e.g. a zero-offset Copy/Paste) produces ONE clean face, not a duplicate or degenerate result', () => {
  const square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const duplicate = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const arr = buildPlanarArrangement([square, duplicate]);
  assert.equal(arr.faces.length, 1, `expected exactly 1 face from two identical overlapping squares, got ${arr.faces.length}`);
  assert.ok(loopsMatch(arr.faces[0].outer, [[0, 0], [10, 0], [10, 10], [0, 10]]));
});

test('two curves that overlap PARTIALLY along a collinear span (not a full duplicate) still produce a valid, correctly-split arrangement', () => {
  // Two squares sharing the ENTIRE right edge of A / left edge of B, plus
  // an extra collinear segment along part of that shared edge from a
  // THIRD curve — a real collinear-overlap stress case, not just a clean
  // shared-edge touch.
  const a = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const b = [[10, 0], [20, 0], [20, 10], [10, 10], [10, 0]];
  const extraCollinear = [[10, 2], [10, 8]]; // lies exactly along the shared edge, a sub-span of it
  const arr = buildPlanarArrangement([a, b, extraCollinear]);
  assert.equal(arr.faces.length, 2, `expected the two squares to stay 2 clean regions despite the collinear overlap stress, got ${arr.faces.length}`);
  const areas = arr.faces.map((f) => Math.abs(signedArea2D(f.outer))).sort((x, y) => x - y);
  assert.ok(Math.abs(areas[0] - 100) < 1e-6 && Math.abs(areas[1] - 100) < 1e-6, `expected two 100-area squares, got ${JSON.stringify(areas)}`);
});

test('two curves that do not touch at all produce zero faces (no bounded region exists yet)', () => {
  const a = [[0, 0], [5, 5]];
  const b = [[10, 10], [15, 15]];
  const arr = buildPlanarArrangement([a, b]);
  assert.equal(arr.faces.length, 0);
});

test('an OPEN curve with no closure of its own still seeds a real region once combined with other curves (open squiggles overlapping a circle/square)', () => {
  // A square, plus an OPEN "squiggle" that dips into the square and back
  // out through a DIFFERENT edge — never closed on its own, but genuinely
  // splits the square into two regions together with the square's own
  // boundary, matching the open-curve participation THE ALGORITHM section
  // names as the real target behavior (not just a closed-polygon demo).
  const square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const squiggle = [[5, 0], [4, 3], [6, 6], [5, 10]]; // open: starts on the bottom edge, ends on the top edge, wanders inside
  const arr = buildPlanarArrangement([square, squiggle]);
  assert.equal(arr.faces.length, 2, `expected the open squiggle to split the square into 2 regions (T-junction at both its own ends), got ${arr.faces.length}`);
  const totalArea = arr.faces.reduce((sum, f) => sum + Math.abs(signedArea2D(f.outer)), 0);
  assert.ok(Math.abs(totalArea - 100) < 1e-6, `the two regions must together account for the WHOLE square's own area (100), got ${totalArea}`);
});

test('three squares in a row (A-B-C, each touching only its immediate neighbor) produce correct 3-way adjacency — B is contiguous with both A and C, but A and C are not directly contiguous', () => {
  const a = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const b = [[10, 0], [20, 0], [20, 10], [10, 10], [10, 0]];
  const c = [[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]];
  const arr = buildPlanarArrangement([a, b, c]);
  assert.equal(arr.faces.length, 3);
  const degrees = arr.adjacency.map((n) => n.length).sort((x, y) => x - y);
  assert.deepEqual(degrees, [1, 1, 2], `expected A and C with 1 neighbor each, B with 2, got ${JSON.stringify(degrees)}`);
});

test('mergeFaces: shift-selecting two contiguous overlapping-square regions dissolves their shared internal edge into ONE combined region', () => {
  const a = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const b = [[5, 0], [15, 0], [15, 10], [5, 10], [5, 0]];
  const arr = buildPlanarArrangement([a, b]);
  assert.equal(arr.faces.length, 3);
  // Find the "left-only" (x 0-5) and "overlap" (x 5-10) faces specifically
  // by their own centroid-ish bbox, then merge just those two.
  function bboxOf(loop) {
    const xs = loop.map((p) => p[0]), ys = loop.map((p) => p[1]);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }
  const leftIdx = arr.faces.findIndex((f) => Math.abs(bboxOf(f.outer).minX - 0) < 1e-6);
  const overlapIdx = arr.faces.findIndex((f) => Math.abs(bboxOf(f.outer).minX - 5) < 1e-6 && Math.abs(bboxOf(f.outer).maxX - 10) < 1e-6);
  assert.ok(leftIdx >= 0 && overlapIdx >= 0 && leftIdx !== overlapIdx);
  assert.ok(arr.adjacency[leftIdx].includes(overlapIdx), 'left and overlap must be genuinely adjacent for this test to be meaningful');
  const merged = mergeFaces(arr, [leftIdx, overlapIdx]);
  assert.equal(merged.length, 1, `expected exactly 1 merged region, got ${merged.length}`);
  assert.equal(merged[0].holes.length, 0);
  // The merged region should be x in [0,10], y in [0,10] — area 100 — the
  // two 50-area halves combined, with the internal x=5 seam dissolved.
  assert.ok(Math.abs(Math.abs(signedArea2D(merged[0].outer)) - 100) < 1e-6, `expected merged area 100, got ${Math.abs(signedArea2D(merged[0].outer))}`);
});

test('mergeFaces: merging ALL THREE regions of the two-overlapping-squares case reconstructs the full union (both squares combined, no internal seams left)', () => {
  const a = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const b = [[5, 0], [15, 0], [15, 10], [5, 10], [5, 0]];
  const arr = buildPlanarArrangement([a, b]);
  const merged = mergeFaces(arr, [0, 1, 2]);
  assert.equal(merged.length, 1);
  // Full union area: two 10x10 squares overlapping on a 5-wide strip = 100 + 100 - 50 = 150.
  assert.ok(Math.abs(Math.abs(signedArea2D(merged[0].outer)) - 150) < 1e-6, `expected full-union area 150, got ${Math.abs(signedArea2D(merged[0].outer))}`);
});

test('mergeFaces: merging the annulus with its own disk reconstructs the FULL outer square with no hole (the hole is exactly what the disk fills in)', () => {
  const outer = [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]];
  const inner = [[5, 5], [15, 5], [15, 15], [5, 15], [5, 5]];
  const arr = buildPlanarArrangement([outer, inner]);
  const annulusIdx = arr.faces.findIndex((f) => f.holes.length === 1);
  const diskIdx = arr.faces.findIndex((f) => f.holes.length === 0);
  const merged = mergeFaces(arr, [annulusIdx, diskIdx]);
  assert.equal(merged.length, 1);
  assert.equal(merged[0].holes.length, 0, 'merging the annulus with its own disk must fill the hole completely');
  assert.ok(Math.abs(Math.abs(signedArea2D(merged[0].outer)) - 400) < 1e-6, 'the merged region is exactly the full outer square, area 400');
});
