// TESSELLATE — a "surface -> curves" generator
// ("TESSELLATION — BUILD, pulled OUT of the Modifiers stack entirely"). It is
// a category sibling of PaintCurves (surface -> N real new curve objects),
// NOT a surface->surface modifier-chain member, exactly as the reconciled
// note rules. Three cell patterns over the surface's OWN UV DOMAIN (this app
// works trims/paint/tessellation in UV space, matching Trim/Paint):
//   1. QUAD        — a regular MxN grid subdivision, each cell -> its 4-edge
//                    boundary loop. An "irregularity" jitter (0 = perfectly
//                    regular grid) displaces interior grid intersections.
//   2. TRIANGULAR  — the same MxN grid, each quad split into 2 triangles.
//   3. VORONOI     — the flagship. Scatter N generator points across the UV
//                    domain (a REGULAR near-hexagonal lattice at
//                    irregularity 0, so it never defaults to the "soccer
//                    ball" over-regular look the doc warns against), then the
//                    real 2D Voronoi diagram of those points via the
//                    Delaunay dual (Bowyer-Watson triangulation -> the
//                    Voronoi vertices are the Delaunay triangle circumcenters,
//                    Voronoi edges connect circumcenters of adjacent
//                    triangles). Cells touching the UV rectangle edge are
//                    CLIPPED to it (reusing trimtess.mjs's proven
//                    Sutherland-Hodgman clipPolygonToRect, not a new one).
//
// EVERYTHING here works in the normalized fraction square [0,1]x[0,1]. The
// caller (the app) maps a fraction (fu,fv) -> the surface's real
// (u,v) domain -> surfacePoint -> a real 3D curve object, exactly as Paint's
// own node map does. A CLOSED UV direction needs no special handling for
// Quad/Triangular: the grid always covers the full [0,1] domain, and the
// seam cell's outer edge maps to fraction 1.0 == fraction 0.0 physically
// (surfacePoint(uMax) == surfacePoint(uMin) on a closed surface), so it wraps
// continuously by construction (the same "seam falls out for free" property
// Paint's 3D-distance brush already relies on). Voronoi is OPEN-DOMAIN-ONLY
// in v1 (its point-scatter/dual construction isn't naturally seam-periodic);
// the app refuses it honestly on a closed-direction surface.
//
// Determinism: the same (seed, indices) always hashes to the same jitter, so
// a given seed reproduces the exact same tessellation — the SAME integer-hash
// PRNG (hashU32/hash01) the Noise modifier and the Curve Generator already
// prove, reused verbatim, NEVER Math.random.

import { clipPolygonToRect } from './trimtess.mjs';

// ---------------------------------------------------------------------------
// Shared deterministic PRNG — BYTE-IDENTICAL to kernel/noise.mjs and
// kernel/curvegen.mjs (Murmur3 finalizer + FNV combine). Reused, not
// reinvented, per the standing "one seeded hash, never Math.random" rule.
// ---------------------------------------------------------------------------
function hashU32(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return (x ^ (x >>> 16)) >>> 0;
}
export function hash01(...vals) {
  let h = 0x811c9dc5 >>> 0; // FNV offset basis
  for (const v of vals) h = hashU32(h ^ (Math.imul(v | 0, 0x9e3779b1) >>> 0));
  return (h >>> 0) / 4294967296;
}

// A symmetric jitter in [-1,1) from a set of integer keys.
function jitter(...keys) { return 2 * hash01(...keys) - 1; }

export function normalizeTessParams(p = {}) {
  const type = ['voronoi', 'triangular', 'quad'].includes(p.type) ? p.type : 'voronoi';
  return {
    type,
    // Quad/Triangular grid divisions (segments per direction).
    nu: clampInt(p.nu, 2, 64, 6),
    nv: clampInt(p.nv, 2, 64, 6),
    // Voronoi generator target count.
    count: clampInt(p.count, 4, 400, 60),
    // Mandatory irregularity 0..1 (0 = perfectly regular).
    irregularity: Number.isFinite(p.irregularity) ? Math.max(0, Math.min(1, p.irregularity)) : 0.4,
    seed: Number.isFinite(p.seed) ? Math.round(p.seed) : 1,
    paintDriven: !!p.paintDriven,
  };
}
function clampInt(v, lo, hi, dflt) {
  if (!Number.isFinite(v)) return dflt;
  return Math.max(lo, Math.min(hi, Math.round(v)));
}

// ===========================================================================
// QUAD + TRIANGULAR
// ===========================================================================
//
// The (nu+1)x(nv+1) grid vertices in [0,1]^2. An interior intersection (not
// on any of the four outer grid lines) is jittered by up to
// irregularity * 0.45 * cellSize, so even at irregularity 1 a vertex never
// crosses into a neighboring cell (no inverted/self-overlapping cell). The
// outer grid lines (i=0,i=nu,j=0,j=nv) are NEVER jittered, keeping both the
// UV domain boundary and a closed direction's seam clean.
export function gridVertices(nu, nv, irregularity, seed) {
  const du = 1 / nu, dv = 1 / nv;
  const verts = [];
  for (let i = 0; i <= nu; i++) {
    const row = [];
    for (let j = 0; j <= nv; j++) {
      let u = i * du, v = j * dv;
      const interior = i > 0 && i < nu && j > 0 && j < nv;
      if (interior && irregularity > 0) {
        u += irregularity * 0.45 * du * jitter(seed, i, j, 1);
        v += irregularity * 0.45 * dv * jitter(seed, i, j, 2);
      }
      row.push([u, v]);
    }
    verts.push(row);
  }
  return verts;
}

// nu*nv quad cells, each a 4-corner CCW loop of [u,v] points in [0,1]^2.
export function quadCells(nu, nv, irregularity, seed) {
  const V = gridVertices(nu, nv, irregularity, seed);
  const cells = [];
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      cells.push([V[i][j], V[i + 1][j], V[i + 1][j + 1], V[i][j + 1]]);
    }
  }
  return cells;
}

// 2*nu*nv triangles, splitting each quad on its (i,j)->(i+1,j+1) diagonal.
export function triangularCells(nu, nv, irregularity, seed) {
  const V = gridVertices(nu, nv, irregularity, seed);
  const cells = [];
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const a = V[i][j], b = V[i + 1][j], c = V[i + 1][j + 1], d = V[i][j + 1];
      cells.push([a, b, c]);
      cells.push([a, c, d]);
    }
  }
  return cells;
}

// ===========================================================================
// VORONOI
// ===========================================================================
//
// A near-hexagonal generator lattice inset slightly from the domain edge (so
// its reflections, below, are genuinely outside). irregularity>0 jitters each
// point by up to irregularity * 0.5 * spacing. At irregularity 0 this is a
// clean hex packing — deliberately NOT a square lattice, which would produce
// 4-cocircular Delaunay degeneracies AND is exactly the over-regular
// "soccer ball" look the doc warns against defaulting to.
export function hexGenerators(targetCount, irregularity, seed, keepFn = null) {
  const n = Math.max(4, targetCount);
  // Hex packing: dy = dx*sqrt(3)/2, area 1 => n*dx*dy ~= 1.
  const dx = Math.sqrt(2 / (n * Math.sqrt(3)));
  const dy = dx * Math.sqrt(3) / 2;
  const cols = Math.max(2, Math.round(1 / dx));
  const rows = Math.max(2, Math.round(1 / dy));
  const pts = [];
  for (let r = 0; r < rows; r++) {
    const v = (r + 0.5) / rows;
    const offset = (r % 2) * 0.5;
    for (let c = 0; c < cols; c++) {
      let u = (c + 0.5 + offset) / cols;
      if (u >= 1) continue; // odd-row offset can push the last column past the edge
      let uu = u, vv = v;
      if (irregularity > 0) {
        uu += irregularity * 0.5 / cols * jitter(seed, r, c, 1);
        vv += irregularity * 0.5 / rows * jitter(seed, r, c, 2);
      }
      // Keep strictly inside the open domain so reflections are outside it.
      uu = Math.max(1e-4, Math.min(1 - 1e-4, uu));
      vv = Math.max(1e-4, Math.min(1 - 1e-4, vv));
      if (keepFn && !keepFn(uu, vv, r, c, seed)) continue;
      pts.push([uu, vv]);
    }
  }
  return pts;
}

// PAINT-DRIVEN DENSITY (optional). A keep-predicate for hexGenerators that
// deterministically THINS generators in LOW-paint regions, so the Voronoi is
// DENSER where the painted scalar is HIGHER. Direction chosen deliberately:
// a student paints to MARK where they want more detail/cells, so "more paint
// -> more cells" is the intuitive reading. `field(u,v)` returns a value in
// [0,1]; a generator at a node with field value f is kept with probability
// minKeep + (1-minKeep)*f (deterministic, hash01-driven), so the sparsest a
// region ever gets is minKeep of the base lattice.
export function paintKeepPredicate(field, minKeep = 0.25) {
  return (u, v, r, c, seed) => {
    const f = Math.max(0, Math.min(1, field(u, v)));
    const keepProb = minKeep + (1 - minKeep) * f;
    return hash01(seed, r, c, 99) < keepProb;
  };
}

export function circumcenter(a, b, c) {
  const ax = a[0], ay = a[1], bx = b[0], by = b[1], cx = c[0], cy = c[1];
  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(d) < 1e-18) return null; // collinear
  const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
  const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
  const uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
  return [ux, uy];
}

// Standard incremental Bowyer-Watson Delaunay triangulation of a set of 2D
// points. Returns triangles as index triples into `points`. A super-triangle
// far outside the bbox seeds it and its vertices are stripped at the end.
export function delaunayTriangulate(points) {
  const n = points.length;
  if (n < 3) return [];
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const [x, y] of points) {
    if (x < minx) minx = x; if (x > maxx) maxx = x;
    if (y < miny) miny = y; if (y > maxy) maxy = y;
  }
  const dmax = Math.max(maxx - minx, maxy - miny) || 1;
  const midx = (minx + maxx) / 2, midy = (miny + maxy) / 2;
  const M = 20 * dmax;
  // Super-triangle vertices appended at indices n, n+1, n+2.
  const pts = points.slice();
  pts.push([midx - M, midy - M]);
  pts.push([midx + M, midy - M]);
  pts.push([midx, midy + M]);
  const superIdx = [n, n + 1, n + 2];

  const makeTri = (i, j, k) => {
    const cc = circumcenter(pts[i], pts[j], pts[k]);
    let r2 = Infinity;
    if (cc) r2 = (pts[i][0] - cc[0]) ** 2 + (pts[i][1] - cc[1]) ** 2;
    return { v: [i, j, k], cc, r2 };
  };
  let tris = [makeTri(superIdx[0], superIdx[1], superIdx[2])];

  for (let p = 0; p < n; p++) {
    const px = pts[p][0], py = pts[p][1];
    const bad = [];
    const good = [];
    for (const t of tris) {
      if (t.cc && ((px - t.cc[0]) ** 2 + (py - t.cc[1]) ** 2) <= t.r2 * (1 + 1e-9)) bad.push(t);
      else good.push(t);
    }
    // Boundary of the bad-triangle cavity: edges used by exactly one bad tri.
    const edgeCount = new Map();
    const key = (a, b) => (a < b ? a + ',' + b : b + ',' + a);
    for (const t of bad) {
      const [i, j, k] = t.v;
      for (const [a, b] of [[i, j], [j, k], [k, i]]) {
        const kk = key(a, b);
        edgeCount.set(kk, (edgeCount.get(kk) || 0) + 1);
      }
    }
    tris = good;
    for (const [kk, cnt] of edgeCount) {
      if (cnt === 1) {
        const [a, b] = kk.split(',').map(Number);
        tris.push(makeTri(a, b, p));
      }
    }
  }
  return tris.filter((t) => !t.v.some((idx) => superIdx.includes(idx))).map((t) => t.v);
}

// The Voronoi diagram of `generators` (points in [0,1]^2), as one clipped cell
// polygon per generator. Uses the REFLECTION-PADDING technique: every
// generator is mirrored across each of the four domain edges, and the
// Delaunay of the augmented set is taken. This bounds every ORIGINAL
// generator's cell cleanly at (or inside) the domain edge — the bisector
// between a near-edge site and its own reflection across that edge IS the
// domain edge — so the interior Voronoi edges between two real generators are
// exact perpendicular bisectors (the equidistance property), while boundary
// cells terminate at the domain edge. Each polygon is a final Sutherland-
// Hodgman clip to [0,1]^2 (trimtess.mjs's proven clipPolygonToRect, reused).
export function voronoiCells(generators) {
  const nG = generators.length;
  if (nG < 3) return generators.map((g, i) => ({ site: i, generator: g, polygon: [] }));
  // Reflected ghost generators (index >= nG are ghosts).
  const all = generators.slice();
  for (const [u, v] of generators) {
    all.push([-u, v]);       // across u=0
    all.push([2 - u, v]);    // across u=1
    all.push([u, -v]);       // across v=0
    all.push([u, 2 - v]);    // across v=1
  }
  const tris = delaunayTriangulate(all);
  // Per original site: incident triangles' circumcenters.
  const incident = Array.from({ length: nG }, () => []);
  for (const [i, j, k] of tris) {
    const cc = circumcenter(all[i], all[j], all[k]);
    if (!cc) continue;
    for (const idx of [i, j, k]) if (idx < nG) incident[idx].push(cc);
  }
  const cells = [];
  for (let s = 0; s < nG; s++) {
    const g = generators[s];
    const ccs = incident[s];
    if (ccs.length < 3) { cells.push({ site: s, generator: g, polygon: [] }); continue; }
    // Dedupe near-identical circumcenters, then order CCW around the site
    // (a Voronoi cell is always convex, so angle-sort is exact).
    const uniq = [];
    for (const c of ccs) {
      if (!uniq.some((q) => Math.hypot(q[0] - c[0], q[1] - c[1]) < 1e-9)) uniq.push(c);
    }
    uniq.sort((p, q) => Math.atan2(p[1] - g[1], p[0] - g[0]) - Math.atan2(q[1] - g[1], q[0] - g[0]));
    const clipped = clipPolygonToRect(uniq, 0, 1, 0, 1);
    cells.push({ site: s, generator: g, polygon: clipped });
  }
  return cells;
}

// Convenience: full Voronoi tessellation from params, returning the
// generators (post-jitter) and the clipped cell polygons.
export function voronoiTessellate(params) {
  const p = normalizeTessParams({ ...params, type: 'voronoi' });
  const generators = hexGenerators(p.count, p.irregularity, p.seed);
  return { generators, cells: voronoiCells(generators).filter((c) => c.polygon.length >= 3) };
}

// The signed area of a [u,v] polygon (shoelace) — used by callers/tests to
// measure cell-size distribution and confirm valid (non-degenerate) cells.
export function polygonArea2D(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    a += x0 * y1 - x1 * y0;
  }
  return a / 2;
}

// The full cell set in [0,1]^2 fraction space for any tessellation type. Each
// cell is an array of [u,v] boundary points (a closed loop). For Voronoi the
// caller must gate closed-direction surfaces out first (open-domain-only v1).
export function tessellateCells(params) {
  const p = normalizeTessParams(params);
  if (p.type === 'quad') return quadCells(p.nu, p.nv, p.irregularity, p.seed);
  if (p.type === 'triangular') return triangularCells(p.nu, p.nv, p.irregularity, p.seed);
  return voronoiTessellate(p).cells.map((c) => c.polygon);
}
