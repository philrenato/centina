import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  hash01, gridVertices, quadCells, triangularCells, hexGenerators, paintKeepPredicate,
  delaunayTriangulate, circumcenter, voronoiCells, voronoiTessellate, polygonArea2D,
  tessellateCells, normalizeTessParams,
} from '../kernel/tessellate.mjs';

// --------------------------------------------------------------------------
// QUAD / TRIANGULAR — exact cell counts, the irregularity-0 regular-grid
// identity, and a measurable jitter at irregularity>0.
// --------------------------------------------------------------------------
test('quad: MxN grid produces exactly M*N cells, each a 4-corner loop', () => {
  const cells = quadCells(6, 4, 0, 1);
  assert.equal(cells.length, 24);
  for (const c of cells) assert.equal(c.length, 4);
});

test('triangular: MxN grid produces exactly 2*M*N triangles', () => {
  const cells = triangularCells(6, 4, 0, 1);
  assert.equal(cells.length, 48);
  for (const c of cells) assert.equal(c.length, 3);
});

test('irregularity=0 is a PERFECTLY regular grid (checkable identity)', () => {
  const V = gridVertices(5, 5, 0, 7);
  for (let i = 0; i <= 5; i++) for (let j = 0; j <= 5; j++) {
    assert.ok(Math.abs(V[i][j][0] - i / 5) < 1e-15, `vertex ${i},${j} u`);
    assert.ok(Math.abs(V[i][j][1] - j / 5) < 1e-15, `vertex ${i},${j} v`);
  }
});

test('irregularity>0 measurably jitters interior grid intersections (and never crosses a cell)', () => {
  const V0 = gridVertices(6, 6, 0, 3);
  const V1 = gridVertices(6, 6, 0.8, 3);
  let maxDev = 0;
  for (let i = 1; i < 6; i++) for (let j = 1; j < 6; j++) {
    maxDev = Math.max(maxDev, Math.hypot(V1[i][j][0] - V0[i][j][0], V1[i][j][1] - V0[i][j][1]));
  }
  assert.ok(maxDev > 0.01, `interior jitter should be measurable, got ${maxDev}`);
  // Boundary vertices are NEVER jittered (clean domain edge + seam).
  for (let i = 0; i <= 6; i++) {
    assert.ok(Math.abs(V1[i][0][1]) < 1e-15 && Math.abs(V1[i][6][1] - 1) < 1e-15);
    assert.ok(Math.abs(V1[0][i][0]) < 1e-15 && Math.abs(V1[6][i][0] - 1) < 1e-15);
  }
  // Bounded jitter (<= 0.45*du) means an interior vertex never leaves its cell.
  const du = 1 / 6;
  for (let i = 1; i < 6; i++) for (let j = 1; j < 6; j++) {
    assert.ok(Math.abs(V1[i][j][0] - i * du) <= 0.45 * du + 1e-12);
  }
});

// --------------------------------------------------------------------------
// DETERMINISM — same seed = same tessellation, different seed = different.
// --------------------------------------------------------------------------
test('determinism: same seed reproduces bit-identical cells; different seed differs', () => {
  const a = JSON.stringify(quadCells(6, 6, 0.5, 42));
  const b = JSON.stringify(quadCells(6, 6, 0.5, 42));
  const c = JSON.stringify(quadCells(6, 6, 0.5, 43));
  assert.equal(a, b);
  assert.notEqual(a, c);
  // Voronoi too.
  const va = JSON.stringify(voronoiTessellate({ count: 40, irregularity: 0.5, seed: 5 }).generators);
  const vb = JSON.stringify(voronoiTessellate({ count: 40, irregularity: 0.5, seed: 5 }).generators);
  const vc = JSON.stringify(voronoiTessellate({ count: 40, irregularity: 0.5, seed: 6 }).generators);
  assert.equal(va, vb);
  assert.notEqual(va, vc);
});

// --------------------------------------------------------------------------
// VORONOI — the real correctness proof: EQUIDISTANCE on interior edges.
// --------------------------------------------------------------------------
test('circumcenter is equidistant from all three triangle vertices', () => {
  const a = [0, 0], b = [4, 0], c = [1, 3];
  const cc = circumcenter(a, b, c);
  const ra = Math.hypot(cc[0] - a[0], cc[1] - a[1]);
  const rb = Math.hypot(cc[0] - b[0], cc[1] - b[1]);
  const rc = Math.hypot(cc[0] - c[0], cc[1] - c[1]);
  assert.ok(Math.abs(ra - rb) < 1e-12 && Math.abs(ra - rc) < 1e-12);
});

test('Delaunay: every triangle circumcircle contains no other point (empty-circle property)', () => {
  const pts = hexGenerators(50, 0.5, 11, null);
  const tris = delaunayTriangulate(pts);
  assert.ok(tris.length > 0);
  for (const [i, j, k] of tris) {
    const cc = circumcenter(pts[i], pts[j], pts[k]);
    const r2 = (pts[i][0] - cc[0]) ** 2 + (pts[i][1] - cc[1]) ** 2;
    for (let p = 0; p < pts.length; p++) {
      if (p === i || p === j || p === k) continue;
      const d2 = (pts[p][0] - cc[0]) ** 2 + (pts[p][1] - cc[1]) ** 2;
      assert.ok(d2 >= r2 * (1 - 1e-7), `point ${p} inside circumcircle of ${i},${j},${k}`);
    }
  }
});

// The DEFINING Voronoi property, proven numerically on several non-trivial
// point sets: every point on an INTERIOR cell-boundary edge is genuinely
// EQUIDISTANT from its own generator and its nearest OTHER generator.
function checkEquidistance(generators) {
  const cells = voronoiCells(generators);
  let interiorEdgesChecked = 0;
  for (const cell of cells) {
    if (cell.polygon.length < 3) continue;
    const g = cell.generator;
    const poly = cell.polygon;
    for (let e = 0; e < poly.length; e++) {
      const a = poly[e], b = poly[(e + 1) % poly.length];
      // Skip edges lying on the domain boundary (those are clipped edges).
      const onBoundary = (p) => p[0] < 1e-6 || p[0] > 1 - 1e-6 || p[1] < 1e-6 || p[1] > 1 - 1e-6;
      if (onBoundary(a) && onBoundary(b)) continue;
      // Sample interior points of the edge.
      for (const t of [0.25, 0.5, 0.75]) {
        const px = a[0] + t * (b[0] - a[0]), py = a[1] + t * (b[1] - a[1]);
        if (onBoundary([px, py])) continue;
        const dOwn = Math.hypot(px - g[0], py - g[1]);
        let dNearestOther = Infinity;
        for (const cell2 of cells) {
          if (cell2.site === cell.site) continue;
          const g2 = cell2.generator;
          dNearestOther = Math.min(dNearestOther, Math.hypot(px - g2[0], py - g2[1]));
        }
        assert.ok(Math.abs(dOwn - dNearestOther) < 1e-6,
          `equidistance: point on interior edge should be equidistant from own gen (${dOwn}) and nearest other (${dNearestOther})`);
        interiorEdgesChecked++;
      }
    }
  }
  return interiorEdgesChecked;
}

test('Voronoi EQUIDISTANCE proof — hex lattice, irregularity 0', () => {
  const gens = hexGenerators(60, 0, 1, null);
  const n = checkEquidistance(gens);
  assert.ok(n > 20, `should have checked many interior edge points, got ${n}`);
});

test('Voronoi EQUIDISTANCE proof — jittered lattice, irregularity 0.6', () => {
  const gens = hexGenerators(60, 0.6, 9, null);
  const n = checkEquidistance(gens);
  assert.ok(n > 20, `should have checked many interior edge points, got ${n}`);
});

test('Voronoi EQUIDISTANCE proof — a small hand-built point set', () => {
  const gens = [[0.2, 0.2], [0.8, 0.2], [0.5, 0.7], [0.5, 0.35], [0.25, 0.75], [0.75, 0.75]];
  const n = checkEquidistance(gens);
  assert.ok(n > 5, `should have checked interior edge points, got ${n}`);
});

// --------------------------------------------------------------------------
// VORONOI — irregularity increases the cell-size distribution variance
// (a real statistical property, not eyeballed), and cells are valid
// (non-self-intersecting, positive area) at the domain edge.
// --------------------------------------------------------------------------
test('Voronoi irregularity increases cell-SIZE variance (regular -> irregular)', () => {
  const areasCV = (irr) => {
    const cells = voronoiTessellate({ count: 80, irregularity: irr, seed: 4 }).cells;
    const areas = cells.map((c) => Math.abs(polygonArea2D(c.polygon))).filter((a) => a > 1e-9);
    const mean = areas.reduce((s, a) => s + a, 0) / areas.length;
    const varr = areas.reduce((s, a) => s + (a - mean) ** 2, 0) / areas.length;
    return Math.sqrt(varr) / mean; // coefficient of variation
  };
  const cvRegular = areasCV(0);
  const cvIrregular = areasCV(1);
  assert.ok(cvIrregular > cvRegular + 0.02,
    `cell-size CV should rise with irregularity: regular ${cvRegular} vs irregular ${cvIrregular}`);
});

test('Voronoi boundary clipping produces valid, non-degenerate cells at the domain edge', () => {
  const cells = voronoiTessellate({ count: 60, irregularity: 0.5, seed: 8 }).cells;
  let edgeCells = 0;
  for (const c of cells) {
    const poly = c.polygon;
    assert.ok(poly.length >= 3, 'a cell must be a real polygon');
    // Positive area (non-degenerate, consistent winding after clip).
    assert.ok(Math.abs(polygonArea2D(poly)) > 1e-9, 'cell must have real area');
    // Every vertex is inside the unit square (clipped).
    for (const [u, v] of poly) {
      assert.ok(u >= -1e-9 && u <= 1 + 1e-9 && v >= -1e-9 && v <= 1 + 1e-9, `vertex ${u},${v} in domain`);
    }
    if (poly.some(([u, v]) => u < 1e-6 || u > 1 - 1e-6 || v < 1e-6 || v > 1 - 1e-6)) edgeCells++;
  }
  assert.ok(edgeCells > 0, 'some cells should genuinely touch the clipped domain boundary');
});

// --------------------------------------------------------------------------
// PAINT-DRIVEN DENSITY — denser Voronoi where the painted field is higher.
// --------------------------------------------------------------------------
test('paint-driven density: MORE generators where the field is higher', () => {
  // Field = 1 on the left half (u<0.5), 0 on the right.
  const field = (u, v) => (u < 0.5 ? 1 : 0);
  const gens = hexGenerators(120, 0, 1, paintKeepPredicate(field, 0.2));
  const left = gens.filter((g) => g[0] < 0.5).length;
  const right = gens.filter((g) => g[0] >= 0.5).length;
  assert.ok(left > right * 1.5, `left (high paint) should be denser: left ${left} vs right ${right}`);
});

// --------------------------------------------------------------------------
// normalizeTessParams + tessellateCells dispatch.
// --------------------------------------------------------------------------
test('normalizeTessParams clamps and defaults sanely; irregularity always present', () => {
  const p = normalizeTessParams({ type: 'bogus', nu: 1000, irregularity: 5 });
  assert.equal(p.type, 'voronoi');
  assert.equal(p.nu, 64);
  assert.equal(p.irregularity, 1);
  const q = normalizeTessParams({});
  assert.ok(Number.isFinite(q.irregularity)); // mandatory, always exposed
});

test('tessellateCells dispatches by type and returns closed loops', () => {
  assert.equal(tessellateCells({ type: 'quad', nu: 4, nv: 4 }).length, 16);
  assert.equal(tessellateCells({ type: 'triangular', nu: 4, nv: 4 }).length, 32);
  const vor = tessellateCells({ type: 'voronoi', count: 40, irregularity: 0.4, seed: 2 });
  assert.ok(vor.length > 10 && vor.every((c) => c.length >= 3));
});
