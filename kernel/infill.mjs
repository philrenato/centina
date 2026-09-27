// Infill — a lattice graph inside a closed body, as plain data.
//
// Everything here is a graph of nodes and straight struts, then a list of
// curves; nothing here builds a surface. The app pipes the curves into a
// SuperB; another app may turn the same graph into a field. The module has
// no imports on purpose: it is copied whole into other apps, and one file
// with no dependencies is the unit that survives a copy.
//
// API (all inputs and outputs are plain JSON-able data)
//
//   INFILL_CELLS                      the cell types and their facts
//   infillFrame(bbox, opts)           the lattice frame: size, orientation, anchor
//   latticeGraph(bbox, opts)          the unclipped graph over a box
//   infillBodyCheck(mesh)             closed / manifold / volume, or a reason
//   windingNumber(mesh, p)            generalized winding number at p
//   clipGraphToBody(graph, mesh, o)   trim to the body, boundary modes, prune
//   skinNet(graph, mesh, opts)        the surface net woven from the hits
//   infillCurves(graph, opts)         one degree-3 NURBS per strut
//   applyInfillOverrides(graph, ov)   hand edits by stable identity
//   enforceArmAngle(graph, deg)       no two struts at a node closer than deg
//   infillPipeRadius(graph, opts)     the widest tube the network carries
//   buildInfill(mesh, opts)           all of the above in order
//   infillEstimate(volume, opts)      the strut count, before running
//   infillGraphArrays(graph)          typed arrays for another app
//   infillGraphHash(graph, size)      a canonical hash for parity checks
//
// A mesh is { positions: [x,y,z, ...], indices (or triIdx): [a,b,c, ...] },
// closed, triangles wound consistently (either way: the sign is measured).
//
// A graph is { nodes: [{ id, p, kind }], struts: [{ id, a, b, kind }] }:
// flat lists, `a`/`b` index into `nodes`. `kind` is 'core' or 'skin'. Ids
// are stable identities, never list positions:
//   regular cells  node 'n:i,j,k:b'      strut 's:i,j,k:e'
//                  (cell i,j,k of the conventional cubic cell, basis or
//                  local edge index)
//   Voronoi        node 'v:' + the four seed ids meeting there, sorted
//                  strut 'e:' + the three seed ids sharing it, sorted
//                  (in 3D a seed pair names a Voronoi face; an edge is
//                  shared by three cells and a vertex by four)
//   trimmed ends   the strut id + '@a' or '@b', the end kept inside
//   skin struts    'k:' + the two skin-node ids, sorted
//
// Units are the caller's model units throughout.
//
// Cell geometry is standard crystallography, each net on its conventional
// cubic cell of edge `size`:
//   kelvin   the edges of the bitruncated cubic honeycomb (the Voronoi of
//            the body-centered cubic points; Kelvin 1887). Valence 4.
//   diamond  dia: FCC plus FCC + (1/4,1/4,1/4). Valence 4.
//   cubic    pcu: one node per cell. Valence 6.
//   gyroid   srs: Wyckoff 8a of I4(1)32 (No. 214), x = 1/8. Valence 3.
//            The skeleton of Schoen's gyroid. RCSR (O'Keeffe et al.), srs.
//   octet    fcu: FCC nearest neighbors. Valence 12.
//   voronoi  3D Voronoi edges of BCC seeds plus jitter. Valence 4.
// Every regular net's edges are its nearest-neighbor pairs, derived below
// from the basis rather than typed in, so a wrong coordinate shows up as a
// wrong valence in the tests rather than as a silently missing strut.

// Vectors

const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vlen = (a) => Math.sqrt(vdot(a, a));
const vdist = (a, b) => vlen(vsub(a, b));
const vnorm = (a) => { const l = vlen(a); return l > 0 ? vscale(a, 1 / l) : [0, 0, 0]; };
const vlerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// The suite's seeded hash (the 0x45d9f3b xor-shift-multiply hash), identical to
// kernel/noise.mjs and kernel/tessellate.mjs, so a seed means one thing.
function infillHashU32(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return (x ^ (x >>> 16)) >>> 0;
}
function infillHash01(...vals) {
  let h = 0x811c9dc5 >>> 0;
  for (const v of vals) h = infillHashU32(h ^ (Math.imul(v | 0, 0x9e3779b1) >>> 0));
  return (h >>> 0) / 4294967296;
}

// Cells

// Integer basis coordinates on a period-P grid; the conventional cell is
// [0,P)^3 and maps to [0,size)^3.
function permutationsWithSigns(v) {
  const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  const out = [];
  for (const p of perms) for (let s = 0; s < 8; s++) {
    out.push([0, 1, 2].map((k) => v[p[k]] * ((s >> k) & 1 ? -1 : 1)));
  }
  return out;
}
function modBasis(points, P) {
  const seen = new Map();
  for (const q of points) {
    const r = q.map((x) => ((x % P) + P) % P);
    seen.set(r.join(','), r);
  }
  return [...seen.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
}
const FCC4 = [[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0]];

const CELL_DEFS = {
  kelvin: {
    label: 'Kelvin', period: 4,
    // The truncated octahedron about each BCC point: the permutations of
    // (0, ±1, ±2) in quarter-cells, about the corner and the body center.
    basis: modBasis([
      ...permutationsWithSigns([0, 1, 2]),
      ...permutationsWithSigns([0, 1, 2]).map((q) => vadd(q, [2, 2, 2])),
    ], 4),
  },
  diamond: { label: 'Diamond', period: 4, basis: modBasis([...FCC4, ...FCC4.map((q) => vadd(q, [1, 1, 1]))], 4) },
  cubic: { label: 'Cubic', period: 1, basis: [[0, 0, 0]] },
  gyroid: {
    label: 'Gyroid', period: 8,
    // I4(1)32 8a, x = 1/8, in eighths: (x,x,x) (-x+1/2,-x,x+1/2)
    // (-x,x+1/2,-x+1/2) (x+1/2,-x+1/2,-x), plus the body centering.
    basis: modBasis([
      [1, 1, 1], [3, -1, 5], [-1, 5, 3], [5, 3, -1],
      [5, 5, 5], [7, 3, 9], [3, 9, 7], [9, 7, 3],
    ], 8),
  },
  octet: { label: 'Octet', period: 4, basis: FCC4.map((q) => q.slice()) },
  voronoi: { label: 'Voronoi', period: 4, basis: null },
};

// Nearest-neighbor edges of a periodic basis. Each undirected edge is kept
// once, owned by the cell of its lexicographically lower (offset, basis)
// end, and gets a local index by sorted order: that index is the strut's
// identity inside its cell.
function deriveEdges(basis, P) {
  let best = Infinity;
  const cand = [];
  for (let i = 0; i < basis.length; i++) {
    for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) for (let oz = -1; oz <= 1; oz++) {
      for (let j = 0; j < basis.length; j++) {
        if (ox === 0 && oy === 0 && oz === 0 && i === j) continue;
        const q = [basis[j][0] + ox * P, basis[j][1] + oy * P, basis[j][2] + oz * P];
        const d = vsub(q, basis[i]);
        const d2 = vdot(d, d);
        if (d2 < best) best = d2;
        cand.push({ i, j, o: [ox, oy, oz], d2 });
      }
    }
  }
  const lexGreater = (o, j, i) => {
    if (o[0] !== 0) return o[0] > 0;
    if (o[1] !== 0) return o[1] > 0;
    if (o[2] !== 0) return o[2] > 0;
    return j > i;
  };
  const edges = cand.filter((c) => c.d2 === best && lexGreater(c.o, c.j, c.i))
    .map((c) => ({ i: c.i, j: c.j, o: c.o }))
    .sort((a, b) => a.i - b.i || a.o[0] - b.o[0] || a.o[1] - b.o[1] || a.o[2] - b.o[2] || a.j - b.j);
  return { edges, edgeSq: best };
}

for (const def of Object.values(CELL_DEFS)) {
  if (!def.basis) continue;
  const { edges, edgeSq } = deriveEdges(def.basis, def.period);
  def.edges = edges;
  def.strutLength = Math.sqrt(edgeSq) / def.period; // in units of size
  def.valence = (2 * edges.length) / def.basis.length;
}

// Every Kelvin strut is an edge of three truncated octahedra, so it borders
// three BCC cells: the three BCC points nearest its midpoint, all at one
// distance. Kept per local edge as offsets (di, dj, dk, b) from the owning
// cell, b = 0 the corner and 1 the body center — the same naming the
// Voronoi seeds use, so a face (a pair of cells) has one key in both.
{
  const def = CELL_DEFS.kelvin, P = def.period;
  def.edgeSeeds = def.edges.map((e) => {
    const qa = def.basis[e.i].map((v) => v / P);
    const qb = def.basis[e.j].map((v, k) => (v + e.o[k] * P) / P);
    const m = vscale(vadd(qa, qb), 0.5);
    const cand = [];
    const f = m.map(Math.floor);
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) for (let b = 0; b < 2; b++) {
      const c = [f[0] + di + b / 2, f[1] + dj + b / 2, f[2] + dk + b / 2];
      cand.push({ s: [f[0] + di, f[1] + dj, f[2] + dk, b], d: vdist(c, m) });
    }
    cand.sort((x, y) => x.d - y.d);
    return cand.filter((c) => c.d < cand[0].d + 1e-9).map((c) => c.s);
  });
}

export const INFILL_CELL_TYPES = ['kelvin', 'voronoi', 'diamond', 'cubic', 'gyroid', 'octet'];

// Facts per cell type, read-only: nodes and struts per conventional cell,
// valence, strut length as a fraction of Size.
export const INFILL_CELLS = Object.freeze(Object.fromEntries(INFILL_CELL_TYPES.map((k) => {
  const d = CELL_DEFS[k];
  if (k === 'voronoi') {
    const kel = CELL_DEFS.kelvin;
    return [k, Object.freeze({ label: d.label, nodesPerCell: kel.basis.length, strutsPerCell: kel.edges.length, valence: 4, strutLength: kel.strutLength })];
  }
  return [k, Object.freeze({ label: d.label, nodesPerCell: d.basis.length, strutsPerCell: d.edges.length, valence: d.valence, strutLength: d.strutLength })];
})));

// Frame

export const INFILL_DEFAULTS = Object.freeze({
  cell: 'kelvin',
  cellsAcross: 4,        // default Size = smallest body extent / this
  orientation: [0, 0, 0], // XYZ Euler degrees, applied X then Y then Z
  origin: [0, 0, 0],     // model-unit offset of the lattice from the body center
  jitter: 0,             // Voronoi only, 0..1
  seed: 1,
  boundary: 'trim',      // 'trim' | 'snap' | 'cells'
  skin: true,
  clearance: 0,          // model units; 0 = off
  prune: true,
  mergeFraction: 0.25,   // a clipped strut shorter than this x strut length merges
  relax: 0,              // 0..1, straight struts -> arcs through each node
  detail: 5,             // control points per strut, 4..7
  minArmAngle: 25,       // degrees; two struts leaving a node closer than this lose one
});

function rotationMatrix(deg) {
  const [rx, ry, rz] = (deg || [0, 0, 0]).map((d) => (Number(d) || 0) * Math.PI / 180);
  const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry), cz = Math.cos(rz), sz = Math.sin(rz);
  // R = Rz * Ry * Rx, rows
  return [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx],
  ];
}
const matVec = (R, v) => [vdot(R[0], v), vdot(R[1], v), vdot(R[2], v)];
const matTVec = (R, v) => [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2], R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2], R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]];

// bbox: { min: [x,y,z], max: [x,y,z] }. The lattice's cell (0,0,0) sits at
// the box center plus `origin`, so a symmetric body gets a symmetric lattice
// and Origin slides the phase.
export function infillFrame(bbox, opts = {}) {
  const ext = vsub(bbox.max, bbox.min);
  const smallest = Math.min(...ext.filter((e) => e > 0));
  const cellsAcross = opts.cellsAcross ?? INFILL_DEFAULTS.cellsAcross;
  const size = opts.size > 0 ? opts.size : (smallest > 0 ? smallest / cellsAcross : 1);
  const R = rotationMatrix(opts.orientation ?? INFILL_DEFAULTS.orientation);
  const anchor = vadd(vscale(vadd(bbox.min, bbox.max), 0.5), opts.origin ?? INFILL_DEFAULTS.origin);
  return { size, orientation: (opts.orientation ?? INFILL_DEFAULTS.orientation).slice(), anchor, R };
}
// Lattice coordinates (in units of size, unrotated) <-> model space.
function toWorld(frame, q) { return vadd(frame.anchor, matVec(frame.R, vscale(q, frame.size))); }
function toLattice(frame, p) { return vscale(matTVec(frame.R, vsub(p, frame.anchor)), 1 / frame.size); }

// The integer cell range covering the box, in lattice coordinates, with a
// margin of `pad` cells.
function cellRange(frame, bbox, pad) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < 8; c++) {
    const p = [c & 1 ? bbox.max[0] : bbox.min[0], c & 2 ? bbox.max[1] : bbox.min[1], c & 4 ? bbox.max[2] : bbox.min[2]];
    const q = toLattice(frame, p);
    for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], q[k]); hi[k] = Math.max(hi[k], q[k]); }
  }
  return { lo: lo.map((v) => Math.floor(v) - pad), hi: hi.map((v) => Math.floor(v) + pad) };
}

// Lattice graph (unclipped)

// The graph over every cell whose box meets `bbox` (one cell of margin), in
// model space. opts: cell, size | cellsAcross, orientation, origin, jitter,
// seed. Returns { ok, graph, frame, cell } or { ok: false, reason }.
export function latticeGraph(bbox, opts = {}) {
  const cell = opts.cell ?? INFILL_DEFAULTS.cell;
  if (!CELL_DEFS[cell]) return { ok: false, reason: `unknown cell type "${cell}" (one of ${INFILL_CELL_TYPES.join(', ')})` };
  const frame = infillFrame(bbox, opts);
  if (!(frame.size > 0) || !Number.isFinite(frame.size)) return { ok: false, reason: 'the cell size is not a positive number' };
  const range = cellRange(frame, bbox, 1);
  const cells = (range.hi[0] - range.lo[0] + 1) * (range.hi[1] - range.lo[1] + 1) * (range.hi[2] - range.lo[2] + 1);
  const limit = opts.maxCells ?? 200000;
  if (cells > limit) return { ok: false, reason: `${cells} cells at this Size; raise Size (the limit is ${limit})`, cells };
  const graph = cell === 'voronoi' ? voronoiGraph(frame, range, opts) : regularGraph(CELL_DEFS[cell], frame, range);
  return { ok: true, graph, frame: { size: frame.size, orientation: frame.orientation, anchor: frame.anchor }, cell, cells };
}

function regularGraph(def, frame, range) {
  const P = def.period;
  const nodes = [];
  const index = new Map();
  const nodeAt = (i, j, k, b) => {
    const id = `n:${i},${j},${k}:${b}`;
    let n = index.get(id);
    if (n === undefined) {
      const q = [i + def.basis[b][0] / P, j + def.basis[b][1] / P, k + def.basis[b][2] / P];
      n = nodes.length;
      nodes.push({ id, p: toWorld(frame, q), kind: 'core' });
      index.set(id, n);
    }
    return n;
  };
  const struts = [];
  const { lo, hi } = range;
  for (let i = lo[0]; i <= hi[0]; i++) for (let j = lo[1]; j <= hi[1]; j++) for (let k = lo[2]; k <= hi[2]; k++) {
    def.edges.forEach((e, ei) => {
      const ti = i + e.o[0], tj = j + e.o[1], tk = k + e.o[2];
      if (ti < lo[0] || ti > hi[0] || tj < lo[1] || tj > hi[1] || tk < lo[2] || tk > hi[2]) return;
      const s = { id: `s:${i},${j},${k}:${ei}`, a: nodeAt(i, j, k, e.i), b: nodeAt(ti, tj, tk, e.j), kind: 'core' };
      if (def.edgeSeeds) s.seeds = def.edgeSeeds[ei].map(([di, dj, dk, b]) => `${i + di},${j + dj},${k + dk},${b}`);
      struts.push(s);
    });
  }
  return { nodes, struts };
}

// Voronoi — per-seed cell clipping.
//
// Each seed's cell starts as a box and is cut by the bisector plane of every
// nearby seed. A polyhedron is kept as faces, each a cyclic list of vertex
// keys; a vertex key is the sorted triple of the face labels meeting there
// (a seed index, or a negative number for a box face). A new vertex made
// where a cut crosses an edge is named by the two faces sharing that edge
// and the cutting face, so vertex identity comes from combinatorics and two
// cells meeting at one Voronoi vertex give it the same four-seed name
// without any distance tolerance.

function clipCell(seedIdx, seeds, neighbours, half) {
  const s = seeds[seedIdx];
  const verts = new Map(); // key -> { p, labels }
  const mk = (labels, p) => { const l = [...labels].sort((a, b) => a - b); const key = l.join('|'); verts.set(key, { p, labels: l }); return key; };
  // Box: faces -1 (-x) -2 (+x) -3 (-y) -4 (+y) -5 (-z) -6 (+z)
  const corner = (sx, sy, sz) => mk([sx ? -2 : -1, sy ? -4 : -3, sz ? -6 : -5], [s[0] + (sx ? half : -half), s[1] + (sy ? half : -half), s[2] + (sz ? half : -half)]);
  const c = [];
  for (let n = 0; n < 8; n++) c.push(corner(n & 1, (n >> 1) & 1, (n >> 2) & 1));
  // Faces wound outward.
  let faces = [
    { label: -1, v: [c[0], c[4], c[6], c[2]] },
    { label: -2, v: [c[1], c[3], c[7], c[5]] },
    { label: -3, v: [c[0], c[1], c[5], c[4]] },
    { label: -4, v: [c[2], c[6], c[7], c[3]] },
    { label: -5, v: [c[0], c[2], c[3], c[1]] },
    { label: -6, v: [c[4], c[5], c[7], c[6]] },
  ];
  for (const ni of neighbours) {
    const t = seeds[ni];
    const nrm = vsub(t, s);
    const mid = vscale(vadd(s, t), 0.5);
    const dn = vlen(nrm);
    const eps = 1e-10 * Math.max(1, dn);
    const sd = (key) => vdot(vsub(verts.get(key).p, mid), nrm) / dn;
    const dist = new Map();
    let anyOut = false, anyIn = false;
    for (const f of faces) for (const k of f.v) if (!dist.has(k)) { const d = sd(k); dist.set(k, d); if (d > eps) anyOut = true; if (d < -eps) anyIn = true; }
    if (!anyOut) continue;
    if (!anyIn) return null;
    const capKeys = new Set();
    const newFaces = [];
    const cutKey = new Map();
    for (const f of faces) {
      const out = [];
      const m = f.v.length;
      for (let q = 0; q < m; q++) {
        const ka = f.v[q], kb = f.v[(q + 1) % m];
        const da = dist.get(ka), db = dist.get(kb);
        if (da <= eps) out.push(ka);
        if (Math.abs(da) <= eps) capKeys.add(ka);
        if ((da < -eps && db > eps) || (da > eps && db < -eps)) {
          const ek = ka < kb ? ka + '#' + kb : kb + '#' + ka;
          let nk = cutKey.get(ek);
          if (!nk) {
            const A = verts.get(ka), B = verts.get(kb);
            const common = A.labels.filter((l) => B.labels.includes(l));
            const tt = da / (da - db);
            nk = mk([...common.slice(0, 2), ni], vlerp(A.p, B.p, tt));
            cutKey.set(ek, nk);
          }
          out.push(nk);
          capKeys.add(nk);
        }
      }
      const dedup = out.filter((k, q) => k !== out[(q + 1) % out.length]);
      if (dedup.length >= 3) newFaces.push({ label: f.label, v: dedup });
    }
    if (capKeys.size >= 3) {
      // Order the cap about its centroid, wound outward (along +nrm).
      const keys = [...capKeys];
      const cen = vscale(keys.reduce((acc, k) => vadd(acc, verts.get(k).p), [0, 0, 0]), 1 / keys.length);
      const u = vnorm(vsub(verts.get(keys[0]).p, cen));
      const w = vcross(vnorm(nrm), u);
      keys.sort((a, b) => {
        const pa = vsub(verts.get(a).p, cen), pb = vsub(verts.get(b).p, cen);
        return Math.atan2(vdot(pa, w), vdot(pa, u)) - Math.atan2(vdot(pb, w), vdot(pb, u));
      });
      newFaces.push({ label: ni, v: keys });
    }
    faces = newFaces;
    for (const k of [...verts.keys()]) if (dist.has(k) && dist.get(k) > eps) verts.delete(k);
  }
  return { faces, verts };
}

const VORONOI_TIE_NUDGE = 1e-7;

function voronoiGraph(frame, range, opts) {
  const jitter = Math.max(0, Math.min(1, Number(opts.jitter ?? 0)));
  const seed = (opts.seed ?? INFILL_DEFAULTS.seed) | 0;
  // Seeds: the BCC points of cells lo-2..hi+2 (lattice coords, size 1). The
  // outer two rings only bound the inner cells and are not themselves built.
  const seeds = [], seedIds = [], seedCell = [];
  const pad = 2;
  const lo = range.lo.map((v) => v - pad), hi = range.hi.map((v) => v + pad);
  const grid = new Map();
  for (let i = lo[0]; i <= hi[0]; i++) for (let j = lo[1]; j <= hi[1]; j++) for (let k = lo[2]; k <= hi[2]; k++) {
    for (let b = 0; b < 2; b++) {
      const base = b ? [i + 0.5, j + 0.5, k + 0.5] : [i, j, k];
      // Exact BCC points put five seeds on one sphere at intermediate cut
      // vertices, and a cut through a vertex has no combinatorial name. A
      // 1e-7-cell nudge breaks every such tie; the BCC diagram itself is
      // simple (four cells at every vertex), so the nudge moves each vertex
      // by the same order and changes no edge.
      const amp = jitter * 0.25 + VORONOI_TIE_NUDGE;
      const q = base.map((v, ax) => v + amp * (2 * infillHash01(i, j, k, b, seed, ax) - 1));
      const idx = seeds.length;
      seeds.push(q);
      seedIds.push(`${i},${j},${k},${b}`);
      seedCell.push([i, j, k]);
      const key = `${i},${j},${k}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(idx);
    }
  }
  const inner = (c) => c[0] >= range.lo[0] && c[0] <= range.hi[0] && c[1] >= range.lo[1] && c[1] <= range.hi[1] && c[2] >= range.lo[2] && c[2] <= range.hi[2];
  const nodes = [], nodeIndex = new Map(), struts = [], strutIndex = new Set();
  const nodeFor = (labels, p) => {
    const id = 'v:' + labels.map((l) => seedIds[l]).sort().join('|');
    let n = nodeIndex.get(id);
    if (n === undefined) { n = nodes.length; nodes.push({ id, p: toWorld(frame, p), kind: 'core' }); nodeIndex.set(id, n); }
    return n;
  };
  for (let si = 0; si < seeds.length; si++) {
    if (!inner(seedCell[si])) continue;
    const [ci, cj, ck] = seedCell[si];
    const nb = [];
    for (let di = -2; di <= 2; di++) for (let dj = -2; dj <= 2; dj++) for (let dk = -2; dk <= 2; dk++) {
      const g = grid.get(`${ci + di},${cj + dj},${ck + dk}`);
      if (g) for (const t of g) if (t !== si) nb.push(t);
    }
    const s = seeds[si];
    nb.sort((a, b) => vdist(seeds[a], s) - vdist(seeds[b], s));
    const cellPoly = clipCell(si, seeds, nb, 1.5);
    if (!cellPoly) continue;
    const { faces, verts } = cellPoly;
    for (const f of faces) {
      if (f.label < 0) continue;
      const m = f.v.length;
      for (let q = 0; q < m; q++) {
        const A = verts.get(f.v[q]), B = verts.get(f.v[(q + 1) % m]);
        if (A.labels.some((l) => l < 0) || B.labels.some((l) => l < 0)) continue;
        const shared = A.labels.filter((l) => B.labels.includes(l));
        if (shared.length < 2) continue;
        const tri = [si, ...shared.slice(0, 2)];
        const id = 'e:' + tri.map((l) => seedIds[l]).sort().join('|');
        if (strutIndex.has(id)) continue;
        strutIndex.add(id);
        const a = nodeFor([si, ...A.labels], A.p), b = nodeFor([si, ...B.labels], B.p);
        if (a !== b) struts.push({ id, a, b, kind: 'core', seeds: tri.map((l) => seedIds[l]).sort() });
      }
    }
  }
  return { nodes, struts };
}

// The body — a closed triangle mesh.

// Welds coincident vertices (a tessellated B-rep repeats its seam vertices
// per face), drops zero-area triangles, and measures what the clip needs:
// naked and non-manifold edges, signed volume (its sign gives the winding
// convention), area and box.
export function infillBodyCheck(mesh) {
  const P = mesh && mesh.positions, I = mesh && (mesh.indices || mesh.triIdx);
  if (!P || !I || P.length < 12 || I.length < 12) return { ok: false, reason: 'no body mesh (fewer than four triangles)' };
  const nv = P.length / 3;
  let bmin = [Infinity, Infinity, Infinity], bmax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) { const v = P[3 * i + k]; if (v < bmin[k]) bmin[k] = v; if (v > bmax[k]) bmax[k] = v; }
  const diag = vdist(bmin, bmax);
  if (!(diag > 0) || !Number.isFinite(diag)) return { ok: false, reason: 'the body has no extent' };
  const tol = diag * 1e-6; // float32 display meshes repeat a point to about 1e-7 of its magnitude
  const remap = new Int32Array(nv), pos = [], cellMap = new Map();
  for (let i = 0; i < nv; i++) {
    const p = [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
    const key = p.map((v) => Math.round(v / tol)).join(',');
    let idx = cellMap.get(key);
    if (idx === undefined) {
      // Neighboring quantization cells, so two points a hair apart across
      // a cell boundary still weld.
      const q = p.map((v) => Math.round(v / tol));
      outer: for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const j = cellMap.get(`${q[0] + dx},${q[1] + dy},${q[2] + dz}`);
        if (j !== undefined) { idx = j; break outer; }
      }
      if (idx === undefined) { idx = pos.length / 3; pos.push(p[0], p[1], p[2]); }
      cellMap.set(key, idx);
    }
    remap[i] = idx;
  }
  const tris = [];
  let area = 0, vol6 = 0;
  for (let t = 0; t + 2 < I.length; t += 3) {
    const a = remap[I[t]], b = remap[I[t + 1]], c = remap[I[t + 2]];
    if (a === b || b === c || a === c) continue;
    const A = [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]], B = [pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]], C = [pos[3 * c], pos[3 * c + 1], pos[3 * c + 2]];
    const ar = vlen(vcross(vsub(B, A), vsub(C, A))) / 2;
    if (!(ar > tol * tol)) continue;
    area += ar;
    vol6 += vdot(A, vcross(B, C));
    tris.push(a, b, c);
  }
  const edgeUse = new Map();
  for (let t = 0; t < tris.length; t += 3) for (let k = 0; k < 3; k++) {
    const a = tris[t + k], b = tris[t + (k + 1) % 3];
    const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
    edgeUse.set(key, (edgeUse.get(key) || 0) + 1);
  }
  let naked = 0, nonManifold = 0;
  for (const n of edgeUse.values()) { if (n === 1) naked++; else if (n > 2) nonManifold++; }
  const volume = vol6 / 6;
  const out = { positions: pos, indices: tris, naked, nonManifold, volume: Math.abs(volume), windingSign: volume < 0 ? -1 : 1, area, bbox: { min: bmin, max: bmax }, triangles: tris.length / 3 };
  if (naked > 0) return { ok: false, reason: `the body is open: ${naked} naked edge${naked === 1 ? '' : 's'}; close it first`, ...out };
  if (nonManifold > 0) return { ok: false, reason: `the body is non-manifold: ${nonManifold} edge${nonManifold === 1 ? '' : 's'} shared by 3+ faces`, ...out };
  if (!(Math.abs(volume) > diag * diag * diag * 1e-9)) return { ok: false, reason: 'the body encloses no volume', ...out };
  return { ok: true, ...out };
}

// Generalized winding number (Jacobson, Kavan, Sorkine-Hornung 2013): the
// summed solid angle of every triangle seen from p, over 4 pi. 1 inside a
// closed outward-wound mesh, 0 outside, and near those on a mesh with small
// defects, which is why it is thresholded at 1/2 rather than trusted as a
// parity. Per triangle, the solid angle is Van Oosterom and Strackee's
// (1983) closed form.
export function windingNumber(mesh, p) {
  const P = mesh.positions, I = mesh.indices;
  let sum = 0;
  for (let t = 0; t < I.length; t += 3) sum += triSolidAngle(P, I[t], I[t + 1], I[t + 2], p[0], p[1], p[2]);
  return sum / (4 * Math.PI) * (mesh.windingSign || 1);
}
function triSolidAngle(P, ia, ib, ic, px, py, pz) {
  const ax = P[3 * ia] - px, ay = P[3 * ia + 1] - py, az = P[3 * ia + 2] - pz;
  const bx = P[3 * ib] - px, by = P[3 * ib + 1] - py, bz = P[3 * ib + 2] - pz;
  const cx = P[3 * ic] - px, cy = P[3 * ic + 1] - py, cz = P[3 * ic + 2] - pz;
  const la = Math.sqrt(ax * ax + ay * ay + az * az), lb = Math.sqrt(bx * bx + by * by + bz * bz), lc = Math.sqrt(cx * cx + cy * cy + cz * cz);
  const det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  const den = la * lb * lc + (ax * bx + ay * by + az * bz) * lc + (ax * cx + ay * cy + az * cz) * lb + (bx * cx + by * cy + bz * cz) * la;
  return 2 * Math.atan2(det, den);
}

// A uniform grid of triangle boxes, for segment hits and nearby closest
// points. Cells of about one strut length: a strut's box meets a handful.
function buildTriGrid(body, cellSize) {
  const P = body.positions, I = body.indices;
  const lo = body.bbox.min.map((v) => v - cellSize), h = cellSize;
  const dims = [0, 1, 2].map((k) => Math.max(1, Math.ceil((body.bbox.max[k] + cellSize - lo[k]) / h)));
  const cells = new Map();
  const nt = I.length / 3;
  for (let t = 0; t < nt; t++) {
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < 3; k++) { const v = I[3 * t + k]; for (let a = 0; a < 3; a++) { const x = P[3 * v + a]; if (x < mn[a]) mn[a] = x; if (x > mx[a]) mx[a] = x; } }
    const c0 = mn.map((v, a) => Math.max(0, Math.floor((v - lo[a]) / h))), c1 = mx.map((v, a) => Math.min(dims[a] - 1, Math.floor((v - lo[a]) / h)));
    for (let x = c0[0]; x <= c1[0]; x++) for (let y = c0[1]; y <= c1[1]; y++) for (let z = c0[2]; z <= c1[2]; z++) {
      const key = (x * dims[1] + y) * dims[2] + z;
      let list = cells.get(key);
      if (!list) cells.set(key, list = []);
      list.push(t);
    }
  }
  const stamp = new Int32Array(nt);
  let tick = 0;
  // Triangles whose cells meet the box [mn, mx].
  const query = (mn, mx, fn) => {
    tick++;
    const c0 = mn.map((v, a) => Math.max(0, Math.floor((v - lo[a]) / h))), c1 = mx.map((v, a) => Math.min(dims[a] - 1, Math.floor((v - lo[a]) / h)));
    for (let x = c0[0]; x <= c1[0]; x++) for (let y = c0[1]; y <= c1[1]; y++) for (let z = c0[2]; z <= c1[2]; z++) {
      const list = cells.get((x * dims[1] + y) * dims[2] + z);
      if (!list) continue;
      for (const t of list) { if (stamp[t] === tick) continue; stamp[t] = tick; fn(t); }
    }
  };
  return { query, cellSize };
}

// Every parameter t in (0,1) where segment a->b crosses the body, sorted,
// hits closer than 1e-9 (a shared edge or vertex) counted once, and hits
// within `endTol` (a length) of either end ignored: an end placed on the
// surface is not a crossing.
function segmentHits(body, grid, a, b, endTol = 0) {
  const P = body.positions, I = body.indices;
  const d = vsub(b, a);
  const L = vlen(d);
  const tLo = Math.max(1e-9, L > 0 ? endTol / L : 0), tHi = 1 - tLo;
  const mn = [0, 1, 2].map((k) => Math.min(a[k], b[k])), mx = [0, 1, 2].map((k) => Math.max(a[k], b[k]));
  const ts = [];
  grid.query(mn, mx, (t) => {
    const i0 = 3 * I[3 * t], i1 = 3 * I[3 * t + 1], i2 = 3 * I[3 * t + 2];
    const v0 = [P[i0], P[i0 + 1], P[i0 + 2]];
    const e1 = [P[i1] - v0[0], P[i1 + 1] - v0[1], P[i1 + 2] - v0[2]];
    const e2 = [P[i2] - v0[0], P[i2 + 1] - v0[1], P[i2 + 2] - v0[2]];
    const h = vcross(d, e2);
    const det = vdot(e1, h);
    if (Math.abs(det) < 1e-18) return;
    const inv = 1 / det;
    const s = vsub(a, v0);
    const u = inv * vdot(s, h);
    if (u < -1e-12 || u > 1 + 1e-12) return;
    const q = vcross(s, e1);
    const v = inv * vdot(d, q);
    if (v < -1e-12 || u + v > 1 + 1e-12) return;
    const tt = inv * vdot(e2, q);
    if (tt > tLo && tt < tHi) ts.push(tt);
  });
  ts.sort((x, y) => x - y);
  const out = [];
  for (const t of ts) if (!out.length || t - out[out.length - 1] > 1e-9) out.push(t);
  return out;
}

// A strut from A to B lies inside the body or on its surface: between each
// pair of crossings the middle is inside, or no farther than a hair
// (1e-6 of the body's diagonal) outside. The hair is what lets a chord
// between two skin nodes lie along a facet, or across the diagonal of a
// planar quad split into two triangles, whose fold is concave by rounding.
// The test every node move must pass.
function strutStaysInside(body, grid, A, B) {
  const hair = vdist(body.bbox.min, body.bbox.max) * 1e-6;
  const ts = [0, ...segmentHits(body, grid, A, B, hair), 1];
  for (let q = 0; q + 1 < ts.length; q++) {
    const m = vlerp(A, B, (ts[q] + ts[q + 1]) / 2);
    if (windingNumber(body, m) > 0.5) continue;
    const c = closestOnBody(body, grid, m, 2 * hair);
    if (!c || c.d > hair) return false;
  }
  return true;
}

function closestOnTriangle(p, a, b, c) {
  // Ericson, Real-Time Collision Detection 5.1.5.
  const ab = vsub(b, a), ac = vsub(c, a), ap = vsub(p, a);
  const d1 = vdot(ab, ap), d2 = vdot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const bp = vsub(p, b), d3 = vdot(ab, bp), d4 = vdot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return vadd(a, vscale(ab, d1 / (d1 - d3)));
  const cp = vsub(p, c), d5 = vdot(ab, cp), d6 = vdot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return vadd(a, vscale(ac, d2 / (d2 - d6)));
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return vadd(b, vscale(vsub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
  const den = 1 / (va + vb + vc);
  return vadd(a, vadd(vscale(ab, vb * den), vscale(ac, vc * den)));
}

// The closest body point within `reach` of p, or null. Grows the search box
// once to the whole body when nothing is found nearby and reach is Infinity.
function closestOnBody(body, grid, p, reach = Infinity) {
  const P = body.positions, I = body.indices;
  let best = null, bestD = Infinity;
  const tryBox = (r) => {
    grid.query(p.map((v) => v - r), p.map((v) => v + r), (t) => {
      const i0 = 3 * I[3 * t], i1 = 3 * I[3 * t + 1], i2 = 3 * I[3 * t + 2];
      const q = closestOnTriangle(p, [P[i0], P[i0 + 1], P[i0 + 2]], [P[i1], P[i1 + 1], P[i1 + 2]], [P[i2], P[i2 + 1], P[i2 + 2]]);
      const d = vdist(p, q);
      if (d < bestD) { bestD = d; best = { p: q, d, tri: t }; }
    });
  };
  let r = Math.min(reach, grid.cellSize);
  tryBox(r);
  while ((!best || bestD > r) && r < reach) {
    r = Math.min(reach, r * 2);
    tryBox(r);
    if (r > vdist(body.bbox.min, body.bbox.max) * 2) break;
  }
  return best && bestD <= reach ? best : null;
}

// Clipping

// Keeps the part of every strut inside the body.
//   A strut wholly inside is kept whole; wholly outside is dropped. One that
//   crosses is cut at each crossing, and each inside piece is kept with a
//   skin node at its surface end. A strut that leaves and re-enters (a wall
//   thinner than a strut) keeps both inside pieces, so a thin wall is never
//   bridged through the air.
//   boundary 'cells' keeps only whole struts, no skin nodes.
//   clearance d > 0: a core node closer than d to the surface moves onto it
//   and becomes a skin node, so no strut runs grazing under the skin.
//   Then a piece shorter than mergeFraction x strut length merges into its
//   neighbor (a skin end wins, so the surface keeps its node), and prune
//   removes core nodes left with one strut, repeatedly.
// Returns { graph, report }.
export function clipGraphToBody(graph, body, opts = {}) {
  const t0 = nowMs();
  const boundary = opts.boundary ?? INFILL_DEFAULTS.boundary;
  const strutLength = opts.strutLength ?? meanStrutLength(graph);
  const grid = opts.grid ?? buildTriGrid(body, Math.max(strutLength, vdist(body.bbox.min, body.bbox.max) / 200));
  // The winding number of a closed mesh is 1 at any inside point however
  // near the surface, 0 outside, and fractional only on the surface (1/2 on
  // a face). So a node is inside, outside or on it. Two nodes joined by a
  // strut that crosses nothing share their state, so the winding number is
  // taken once per such connected piece of the lattice (and once per node
  // lying on the surface) rather than once per node: a 3,000-node lattice
  // in a 50,000-triangle body is a handful of evaluations, not 150 million
  // solid angles. Pieces of a cut strut alternate at each crossing, counted
  // from an end that is inside or outside; where that count disagrees with
  // the far end (a grazing touch counted as a crossing), each piece is
  // judged by its own midpoint instead.
  const hair = vdist(body.bbox.min, body.bbox.max) * 1e-6;
  const hitsOf = graph.struts.map((s) => segmentHits(body, grid, graph.nodes[s.a].p, graph.nodes[s.b].p));
  const onSurface = graph.nodes.map((n) => !!closestOnBody(body, grid, n.p, hair));
  const parent = graph.nodes.map((_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  graph.struts.forEach((s, si) => { if (!hitsOf[si].length && !onSurface[s.a] && !onSurface[s.b]) parent[find(s.a)] = find(s.b); });
  const rootState = new Map();
  let windings = 0;
  const state = graph.nodes.map((n, i) => {
    if (onSurface[i]) { windings++; const v = windingNumber(body, n.p); return v > 0.99 ? 1 : v < 0.01 ? 0 : 0.5; }
    const r = find(i);
    if (!rootState.has(r)) { windings++; rootState.set(r, windingNumber(body, graph.nodes[r].p) > 0.5 ? 1 : 0); }
    return rootState.get(r);
  });
  const tWinding = nowMs() - t0;
  const nodes = [], nodeIndex = new Map();
  const keepNode = (n, on = false) => {
    let k = nodeIndex.get(n.id);
    if (k === undefined) { k = nodes.length; nodes.push({ id: n.id, p: n.p.slice(), kind: on && boundary !== 'cells' ? 'skin' : n.kind }); nodeIndex.set(n.id, k); }
    return k;
  };
  const midInside = (A, B, ta, tb) => { windings++; return windingNumber(body, vlerp(A, B, (ta + tb) / 2)) > 0.5; };
  const struts = [];
  const cuts = new Array(graph.struts.length);
  let cut = 0, dropped = 0, reentrant = 0;
  for (let si = 0; si < graph.struts.length; si++) {
    const s = graph.struts[si];
    const A = graph.nodes[s.a], B = graph.nodes[s.b];
    const hits = hitsOf[si];
    if (!hits.length) {
      const ins = state[s.a] === 1 && state[s.b] === 1 ? true : state[s.a] === 0 || state[s.b] === 0 ? false : midInside(A.p, B.p, 0, 1);
      cuts[si] = [ins];
      if (ins) struts.push({ id: s.id, a: keepNode(A, state[s.a] === 0.5), b: keepNode(B, state[s.b] === 0.5), kind: s.kind });
      else dropped++;
      continue;
    }
    if (boundary === 'cells') { dropped++; cuts[si] = [false]; continue; }
    cut++;
    const ts = [0, ...hits, 1];
    let pieces = 0;
    let flags = null;
    if (state[s.a] !== 0.5 && state[s.b] !== 0.5) {
      flags = [];
      for (let q = 0; q + 1 < ts.length; q++) flags.push((state[s.a] === 1) !== (q % 2 === 1));
      if (flags[flags.length - 1] !== (state[s.b] === 1)) flags = null;
    }
    if (!flags) flags = ts.slice(0, -1).map((ta, q) => midInside(A.p, B.p, ta, ts[q + 1]));
    cuts[si] = [];
    for (let q = 0; q + 1 < ts.length; q++) {
      const ta = ts[q], tb = ts[q + 1];
      const ins = flags[q];
      cuts[si].push(ins);
      if (!ins) continue;
      pieces++;
      const endA = ta === 0 ? keepNode(A, state[s.a] === 0.5) : keepNode({ id: `h:${s.id}:${q - 1}`, p: vlerp(A.p, B.p, ta), kind: 'skin' });
      const endB = tb === 1 ? keepNode(B, state[s.b] === 0.5) : keepNode({ id: `h:${s.id}:${q}`, p: vlerp(A.p, B.p, tb), kind: 'skin' });
      const id = ta === 0 && tb === 1 ? s.id : ta === 0 ? `${s.id}@a` : tb === 1 ? `${s.id}@b` : `${s.id}@m${q}`;
      struts.push({ id, a: endA, b: endB, kind: s.kind });
    }
    if (pieces > 1 || (pieces === 1 && hits.length > 1)) reentrant++;
    if (!pieces) dropped++;
  }
  let faceSkinStruts = 0;
  if (opts.faceSkin && boundary !== 'cells') faceSkinStruts = weaveFaceSkin(graph, cuts, state, nodeIndex, nodes, struts, body, grid);
  let out = { nodes, struts };
  const report = { crossing: cut, dropped, reentrant, faceSkinStruts, cleared: 0, merged: 0, pruned: 0 };
  const clearance = opts.clearance ?? INFILL_DEFAULTS.clearance;
  if (clearance > 0 && boundary !== 'cells') {
    const inc = out.nodes.map(() => []);
    out.struts.forEach((s) => { inc[s.a].push(s.b); inc[s.b].push(s.a); });
    out.nodes.forEach((n, i) => {
      if (n.kind !== 'core') return;
      const c = closestOnBody(body, grid, n.p, clearance);
      if (!c) return;
      if (inc[i].some((o) => !strutStaysInside(body, grid, c.p, out.nodes[o].p))) { report.clearanceDeclined = (report.clearanceDeclined || 0) + 1; return; }
      n.p = c.p; n.kind = 'skin'; report.cleared++;
    });
  }
  const mergeFraction = opts.mergeFraction ?? INFILL_DEFAULTS.mergeFraction;
  const m = mergeShortStruts(out, mergeFraction * strutLength, { body, grid });
  out = m.graph; report.merged = m.merged; report.mergeDeclined = m.declined; report.stubsDropped = m.stubs;
  if (opts.prune ?? INFILL_DEFAULTS.prune) { const p = pruneDangling(out); out = p.graph; report.pruned = p.removed; }
  report.windingMs = Math.round(tWinding);
  report.windings = windings;
  report.ms = Math.round(nowMs() - t0);
  return { graph: out, report, grid };
}

// The face skin, for lattices whose struts know their three cells (Kelvin,
// Voronoi). A face is a pair of cells; its struts form a closed polygon.
// The body surface cuts that polygon along curves, and every place a curve
// meets the polygon's boundary is a strut hit — so walking the polygon, each
// run of boundary inside the body starts at one hit and ends at another,
// and that pair is joined by a skin strut, the face's own trace on the skin.
// The skin net is then exactly (lattice faces) cut by (surface): its nodes
// are the hits, each hit lies on the three faces of its strut, and a skin
// node carries three skin struts and its core strut. A node lying on the
// surface is a hit of its own.
function weaveFaceSkin(graph, cuts, state, nodeIndex, nodes, struts, body, grid) {
  const faces = new Map();
  graph.struts.forEach((s, si) => {
    if (!s.seeds || s.seeds.length !== 3) return;
    const [x, y, z] = s.seeds;
    for (const pr of [[x, y], [x, z], [y, z]]) {
      const key = pr.slice().sort().join('|');
      if (!faces.has(key)) faces.set(key, []);
      faces.get(key).push(si);
    }
  });
  const seen = new Set(struts.map((s) => (s.a < s.b ? `${s.a},${s.b}` : `${s.b},${s.a}`)));
  let added = 0;
  for (const [key, list] of faces) {
    const touches = list.some((si) => cuts[si] && (cuts[si].length > 1 || state[graph.struts[si].a] === 0.5 || state[graph.struts[si].b] === 0.5));
    if (!touches) continue;
    const inc = new Map();
    for (const si of list) for (const v of [graph.struts[si].a, graph.struts[si].b]) { if (!inc.has(v)) inc.set(v, []); inc.get(v).push(si); }
    if ([...inc.values()].some((l) => l.length !== 2)) continue; // an open face, cut off by the lattice range
    // Walk the polygon once.
    const walk = [];
    let si = list[0], v = graph.struts[si].a;
    for (let guard = 0; guard <= list.length; guard++) {
      const s = graph.struts[si];
      const fwd = s.a === v;
      walk.push({ si, fwd, from: v });
      v = fwd ? s.b : s.a;
      si = inc.get(v)[0] === si ? inc.get(v)[1] : inc.get(v)[0];
      if (si === list[0]) break;
    }
    if (walk.length !== list.length) continue;
    // Events: a hit between two pieces, or a polygon corner on the surface.
    const events = [];
    walk.forEach((e, k) => {
      const s = graph.struts[e.si];
      const flags = cuts[e.si];
      const ordered = e.fwd ? flags : [...flags].reverse();
      const hitName = (q) => `h:${s.id}:${e.fwd ? q : flags.length - 2 - q}`;
      for (let q = 0; q + 1 < ordered.length; q++) events.push({ id: hitName(q), before: ordered[q], after: ordered[q + 1] });
      const nextE = walk[(k + 1) % walk.length];
      const corner = e.fwd ? s.b : s.a;
      if (state[corner] === 0.5) {
        const nf = cuts[nextE.si];
        events.push({ id: graph.nodes[corner].id, before: ordered[ordered.length - 1], after: nextE.fwd ? nf[0] : nf[nf.length - 1] });
      }
    });
    const turns = events.filter((e) => e.before !== e.after && nodeIndex.has(e.id));
    if (turns.length < 2) continue;
    // Two hits: one curve. Four or more: the surface may cross the face as
    // separate caps (pair each entry with the exit after it) or as a band
    // (pair each exit with the entry after it) — the marching-squares
    // ambiguity. The pairing whose chords stay nearer the surface is the
    // one that follows it.
    const pairsFrom = (first) => {
      const out = [];
      for (let k = 0; k < turns.length; k++) {
        const e = turns[(first + k) % turns.length], x = turns[(first + k + 1) % turns.length];
        if (e.after === turns[first].after && x.after !== e.after) out.push([e, x]);
      }
      return out;
    };
    const entry = turns.findIndex((e) => e.after);
    if (entry < 0) continue;
    let pairs = pairsFrom(entry);
    if (turns.length >= 4) {
      const exitFirst = turns.findIndex((e) => !e.after);
      const alt = pairsFrom(exitFirst);
      const cost = (ps) => ps.reduce((acc, [e, x]) => {
        const m = vlerp(nodes[nodeIndex.get(e.id)].p, nodes[nodeIndex.get(x.id)].p, 0.5);
        const c = closestOnBody(body, grid, m);
        return acc + (c ? c.d : Infinity);
      }, 0);
      if (cost(alt) < cost(pairs)) pairs = alt;
    }
    let run = 0;
    for (const [e, x] of pairs) {
      const a = nodeIndex.get(e.id), b = nodeIndex.get(x.id);
      if (a === undefined || b === undefined || a === b) continue;
      const pk = a < b ? `${a},${b}` : `${b},${a}`;
      if (seen.has(pk)) continue;
      seen.add(pk);
      struts.push({ id: `f:${key}${run ? '#' + run : ''}`, a, b, kind: 'skin' });
      run++;
      added++;
    }
  }
  return added;
}

function nowMs() { return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now(); }

function meanStrutLength(graph) {
  let sum = 0;
  for (const s of graph.struts) sum += vdist(graph.nodes[s.a].p, graph.nodes[s.b].p);
  return graph.struts.length ? sum / graph.struts.length : 1;
}

// Collapses every strut shorter than minLen, shortest first. A skin node
// absorbs a core node (the surface keeps its point); two nodes of one kind
// meet at their weighted mean, put back on the surface when they are skin.
// With `guard` ({ body, grid }) a merge that would carry any strut out of
// the body is declined and the short strut stays: a sliver is a cosmetic
// fault, a strut through the air is a wrong answer. Self-loops and doubled
// struts left behind are removed, the lower id kept.
export function mergeShortStruts(graph, minLen, guard = null) {
  const n = graph.nodes.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const pos = graph.nodes.map((q) => q.p.slice()), kind = graph.nodes.map((q) => q.kind), weight = new Array(n).fill(1);
  const members = graph.nodes.map((_, i) => [i]);
  const inc = graph.nodes.map(() => []);
  graph.struts.forEach((s, i) => { inc[s.a].push(i); inc[s.b].push(i); });
  const order = graph.struts.map((s, i) => ({ i, L: vdist(graph.nodes[s.a].p, graph.nodes[s.b].p) })).filter((e) => e.L < minLen).sort((x, y) => x.L - y.L);
  let merged = 0, declined = 0, stubs = 0;
  const dropStrut = new Set();
  for (const { i } of order) {
    const s = graph.struts[i];
    const ra = find(s.a), rb = find(s.b);
    if (ra === rb) continue;
    if (vdist(pos[ra], pos[rb]) >= minLen) continue;
    let keep = ra, gone = rb, at;
    if (kind[ra] !== kind[rb]) { keep = kind[ra] === 'skin' ? ra : rb; gone = keep === ra ? rb : ra; at = pos[keep]; }
    else {
      at = vscale(vadd(vscale(pos[ra], weight[ra]), vscale(pos[rb], weight[rb])), 1 / (weight[ra] + weight[rb]));
      if (kind[ra] === 'skin' && guard) { const c = closestOnBody(guard.body, guard.grid, at); if (c) at = c.p; }
    }
    if (guard) {
      let ok = true;
      for (const m of [...members[ra], ...members[rb]]) {
        for (const si of inc[m]) {
          const t = graph.struts[si];
          if (t.kind !== 'core') continue;
          const ea = find(t.a), eb = find(t.b);
          const onA = ea === ra || ea === rb, onB = eb === ra || eb === rb;
          if (onA && onB) continue;
          const other = onA ? pos[eb] : pos[ea];
          if (!strutStaysInside(guard.body, guard.grid, at, other)) { ok = false; break; }
        }
        if (!ok) break;
      }
      if (!ok) {
        // A core stub to the skin that cannot merge is dropped rather than
        // kept as a sliver no joint can be built on; the skin node stays in
        // the skin net.
        if (s.kind === 'core' && kind[ra] !== kind[rb]) { dropStrut.add(i); stubs++; } else declined++;
        continue;
      }
    }
    pos[keep] = at;
    weight[keep] = weight[ra] + weight[rb];
    members[keep] = [...members[ra], ...members[rb]];
    parent[gone] = keep;
    merged++;
  }
  if (!merged && !stubs) return { graph, merged: 0, declined, stubs };
  const nodes = [], map = new Map();
  graph.nodes.forEach((q, i) => {
    const r = find(i);
    if (!map.has(r)) { map.set(r, nodes.length); nodes.push({ id: graph.nodes[r].id, p: pos[r], kind: kind[r] }); }
  });
  const struts = [], seen = new Map();
  for (const s of [...graph.struts].filter((_, i) => !dropStrut.has(i)).sort((x, y) => (x.id < y.id ? -1 : x.id > y.id ? 1 : 0))) {
    const a = map.get(find(s.a)), b = map.get(find(s.b));
    if (a === b) continue;
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    if (seen.has(key)) continue;
    seen.set(key, true);
    struts.push({ ...s, a, b });
  }
  return { graph: { nodes, struts }, merged, declined, stubs };
}

// Removes core nodes of valence <= 1 and their struts until none is left,
// then any node with no strut. Skin nodes stay: a strut ending on the skin
// is attached, not dangling.
export function pruneDangling(graph) {
  const n = graph.nodes.length;
  const alive = new Array(graph.struts.length).fill(true);
  const deg = new Array(n).fill(0);
  const inc = Array.from({ length: n }, () => []);
  graph.struts.forEach((s, i) => { deg[s.a]++; deg[s.b]++; inc[s.a].push(i); inc[s.b].push(i); });
  const stack = [];
  for (let i = 0; i < n; i++) if (graph.nodes[i].kind === 'core' && deg[i] <= 1) stack.push(i);
  let removed = 0;
  while (stack.length) {
    const i = stack.pop();
    for (const si of inc[i]) {
      if (!alive[si]) continue;
      alive[si] = false; removed++;
      const s = graph.struts[si], o = s.a === i ? s.b : s.a;
      deg[i]--; deg[o]--;
      if (graph.nodes[o].kind === 'core' && deg[o] === 1) stack.push(o);
    }
  }
  const map = new Map(), nodes = [];
  graph.nodes.forEach((q, i) => { if (deg[i] > 0) { map.set(i, nodes.length); nodes.push(q); } });
  const struts = graph.struts.filter((_, i) => alive[i]).map((s) => ({ ...s, a: map.get(s.a), b: map.get(s.b) }));
  return { graph: { nodes, struts }, removed };
}

// Joint angles

// Two struts leaving one node at a shallow angle cannot both be piped: the
// tubes run alongside each other, and the pull-back a clean joint needs is
// r cot(theta/2), which runs away as theta closes (5.7 radii at 20 degrees).
// Measured on the fixtures at defaults, every such pair but two is a core
// strut grazing the skin at a skin node. So a pair under `minAngle` loses
// its core strut (the skin carries the surface; a grazing strut adds
// nothing to it), or the longer strut of two skin struts, or the shorter
// of two core struts; then prune runs again. The angle is measured the way
// the pipe measures it: along each strut's own end tangent when `curves`
// (one per strut, as infillCurves makes them) are given — a skin strut lies
// along the surface and leaves its node closer to the tangent plane than its
// chord does — else along the chords.
export function enforceArmAngle(graph, minAngleDeg = INFILL_DEFAULTS.minArmAngle, curves = null) {
  const cosMax = Math.cos(minAngleDeg * Math.PI / 180);
  const alive = graph.struts.map(() => true);
  const len = graph.struts.map((s) => vdist(graph.nodes[s.a].p, graph.nodes[s.b].p));
  const inc = graph.nodes.map(() => []);
  graph.struts.forEach((s, i) => { inc[s.a].push(i); inc[s.b].push(i); });
  const dirFrom = (v, i) => {
    const s = graph.struts[i];
    if (curves && curves[i]) {
      const P = curves[i].curve.ctrlPts, n = P.length;
      const d = s.a === v ? vsub(P[1], P[0]) : vsub(P[n - 2], P[n - 1]);
      if (vlen(d) > 0) return vnorm(d);
    }
    const o = s.a === v ? s.b : s.a;
    return vnorm(vsub(graph.nodes[o].p, graph.nodes[v].p));
  };
  let removed = 0;
  for (let v = 0; v < graph.nodes.length; v++) {
    for (;;) {
      const live = inc[v].filter((i) => alive[i]);
      let worst = null;
      for (let x = 0; x < live.length; x++) for (let y = x + 1; y < live.length; y++) {
        const c = vdot(dirFrom(v, live[x]), dirFrom(v, live[y]));
        if (c > cosMax && (!worst || c > worst.c)) worst = { c, i: live[x], j: live[y] };
      }
      if (!worst) break;
      const A = graph.struts[worst.i], B = graph.struts[worst.j];
      let drop;
      if (A.kind !== B.kind) drop = A.kind === 'core' ? worst.i : worst.j;
      else if (A.kind === 'skin') drop = len[worst.i] >= len[worst.j] ? worst.i : worst.j;
      else drop = len[worst.i] <= len[worst.j] ? worst.i : worst.j;
      alive[drop] = false;
      removed++;
    }
  }
  if (!removed) return { graph, removed: 0 };
  const pruned = pruneDangling({ nodes: graph.nodes, struts: graph.struts.filter((_, i) => alive[i]) });
  return { graph: pruned.graph, removed, pruned: pruned.removed };
}

// The widest tube radius the whole network can carry with hull joints. At
// a node whose tightest pair of struts is theta apart, every tube stops
// margin x max(floor, cot(theta/2)) radii short of the node (the pipe
// network's own inset, with its hull margin); a strut must be longer than
// the pull-backs at its two ends. The ceiling is that inequality solved for
// one shared radius: the least length / (factor A + factor B) over the
// struts. `radius` is the default a new pipe starts at: `fraction` of the
// strut length, held under the ceiling with 4% to spare.
export const INFILL_PIPE_HULL_MARGIN = 1.3;
export const INFILL_PIPE_INSET_FLOOR = 0.62;
export function infillPipeRadius(graph, opts = {}) {
  const margin = opts.margin ?? INFILL_PIPE_HULL_MARGIN;
  const floor = opts.floor ?? INFILL_PIPE_INSET_FLOOR;
  const inc = graph.nodes.map(() => []);
  graph.struts.forEach((s, i) => { inc[s.a].push(i); inc[s.b].push(i); });
  const factor = graph.nodes.map((n, v) => {
    const dirs = inc[v].map((i) => { const s = graph.struts[i]; const o = s.a === v ? s.b : s.a; return vnorm(vsub(graph.nodes[o].p, n.p)); });
    if (dirs.length < 2) return 0;
    let theta = Math.PI;
    for (let x = 0; x < dirs.length; x++) for (let y = x + 1; y < dirs.length; y++) theta = Math.min(theta, Math.acos(Math.max(-1, Math.min(1, vdot(dirs[x], dirs[y])))));
    return margin * Math.max(floor, 1 / Math.tan(Math.max(theta, 1e-9) / 2));
  });
  let ceiling = Infinity, binding = null, total = 0;
  graph.struts.forEach((s, i) => {
    const L = vdist(graph.nodes[s.a].p, graph.nodes[s.b].p);
    total += L;
    const f = factor[s.a] + factor[s.b];
    if (f > 0 && L / f < ceiling) { ceiling = L / f; binding = s.id; }
  });
  const strutLength = opts.strutLength ?? (graph.struts.length ? total / graph.struts.length : 0);
  const want = (opts.fraction ?? 0.12) * strutLength;
  const roof = Number.isFinite(ceiling) ? 0.96 * ceiling : Infinity;
  return { ceiling: Number.isFinite(ceiling) ? ceiling : null, binding, radius: Math.min(want, roof), clamped: want > roof };
}

// Skin net

// Splits every triangle 1-to-4, level by level, until the longest edge is
// under h (at most `maxTris` triangles). The split is flat: the surface
// graph needs a fine sampling of the body, not a smoother one.
function refineBodySurface(body, h, maxTris = 400000) {
  let pos = body.positions.slice(), idx = body.indices.slice();
  const longest = () => {
    let m = 0;
    for (let t = 0; t < idx.length; t += 3) for (let k = 0; k < 3; k++) {
      const a = idx[t + k], b = idx[t + (k + 1) % 3];
      const d = Math.hypot(pos[3 * a] - pos[3 * b], pos[3 * a + 1] - pos[3 * b + 1], pos[3 * a + 2] - pos[3 * b + 2]);
      if (d > m) m = d;
    }
    return m;
  };
  while (longest() > h && idx.length / 3 * 4 <= maxTris) {
    const mid = new Map(), next = [];
    const midOf = (a, b) => {
      const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
      let m = mid.get(key);
      if (m === undefined) { m = pos.length / 3; pos.push((pos[3 * a] + pos[3 * b]) / 2, (pos[3 * a + 1] + pos[3 * b + 1]) / 2, (pos[3 * a + 2] + pos[3 * b + 2]) / 2); mid.set(key, m); }
      return m;
    };
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t], b = idx[t + 1], c = idx[t + 2];
      const ab = midOf(a, b), bc = midOf(b, c), ca = midOf(c, a);
      next.push(a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca);
    }
    idx = next;
  }
  return { positions: pos, indices: idx };
}

// A binary min-heap of (key, value).
function makeHeap() {
  const k = [], v = [];
  return {
    get size() { return k.length; },
    push(key, val) {
      k.push(key); v.push(val);
      let i = k.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (k[p] <= k[i]) break; [k[p], k[i]] = [k[i], k[p]]; [v[p], v[i]] = [v[i], v[p]]; i = p; }
    },
    pop() {
      const top = [k[0], v[0]];
      const lk = k.pop(), lv = v.pop();
      if (k.length) {
        k[0] = lk; v[0] = lv;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < k.length && k[l] < k[m]) m = l;
          if (r < k.length && k[r] < k[m]) m = r;
          if (m === i) break;
          [k[m], k[i]] = [k[i], k[m]]; [v[m], v[i]] = [v[i], v[m]]; i = m;
        }
      }
      return top;
    },
  };
}

// Groups points closer than `r` (transitively), by a hash of cell r.
function clusterPoints(points, r) {
  const n = points.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const cells = new Map();
  const key = (c) => c.join(',');
  points.forEach((p, i) => {
    const c = p.map((v) => Math.floor(v / r));
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const list = cells.get(key([c[0] + dx, c[1] + dy, c[2] + dz]));
      if (list) for (const j of list) if (vdist(points[j], p) < r) parent[find(i)] = find(j);
    }
    const k = key(c);
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(i);
  });
  return points.map((_, i) => find(i));
}

// The skin net: struts that lie on the body surface between skin nodes.
//
// 'woven' (default): the skin nodes are where core struts reach the skin.
// Skin nodes closer than `mergeLength` join first (at their mean, put back
// on the surface), so two struts landing side by side share one node. Then
// every skin node grows a region over a refined copy of the surface by
// graph distance (multi-source Dijkstra; a geodesic Voronoi, never a chart
// to a plane), two nodes whose regions touch are candidates, and a
// candidate is kept when no neighboring node sits inside the sphere on
// its chord as diameter (the Gabriel graph, Gabriel and Sokal 1969). The
// net's rhythm follows the lattice because its nodes are the lattice's.
//
// 'given': the caller supplies a net on the surface ({ nodes: [p],
// edges: [[i, j]] }: a SuperB's cage edges, isocurves, a mesh's own
// edges). A core strut's skin end snaps to the nearest net node within
// `snapTolerance`, else splits the nearest net edge at its foot.
export function skinNet(graph, body, opts = {}) {
  const t0 = nowMs();
  const strutLength = opts.strutLength ?? meanStrutLength(graph);
  const grid = opts.grid ?? buildTriGrid(body, strutLength);
  const source = opts.source ?? 'woven';
  if (source === 'given') return givenSkin(graph, body, grid, opts, t0);
  const mergeLength = opts.mergeLength ?? INFILL_DEFAULTS.mergeFraction * strutLength;

  // 1. Join skin nodes that sit closer than mergeLength.
  let nodes = graph.nodes.map((n) => ({ ...n, p: n.p.slice() }));
  let struts = graph.struts.map((s) => ({ ...s }));
  const skinIdx = [];
  nodes.forEach((n, i) => { if (n.kind === 'skin') skinIdx.push(i); });
  const cl = clusterPoints(skinIdx.map((i) => nodes[i].p), mergeLength);
  const rep = new Map();
  let joined = 0;
  skinIdx.forEach((ni, k) => {
    const root = skinIdx[cl[k]];
    if (root === ni) return;
    joined++;
    if (!rep.has(root)) rep.set(root, [root]);
    rep.get(root).push(ni);
  });
  let joinDeclined = 0;
  if (joined) {
    const redirect = new Map();
    const inc = nodes.map(() => []);
    struts.forEach((s) => { if (s.kind === 'core') { inc[s.a].push(s.b); inc[s.b].push(s.a); } });
    for (const [root, members] of rep) {
      const mean = vscale(members.reduce((acc, i) => vadd(acc, nodes[i].p), [0, 0, 0]), 1 / members.length);
      const c = closestOnBody(body, grid, mean);
      const at = c ? c.p : mean;
      // A join that would carry a core strut out of the body is declined.
      const set = new Set(members);
      if (members.some((i) => inc[i].some((o) => !set.has(o) && !strutStaysInside(body, grid, at, nodes[o].p)))) { joined -= members.length - 1; joinDeclined++; continue; }
      nodes[root].p = at;
      nodes[root].id = members.map((i) => nodes[i].id).sort()[0];
      for (const i of members) if (i !== root) redirect.set(i, root);
    }
    const keepMap = new Map(), kept = [];
    nodes.forEach((n, i) => { if (!redirect.has(i)) { keepMap.set(i, kept.length); kept.push(n); } });
    const seen = new Set(), next = [];
    for (const s of struts) {
      const a = keepMap.get(redirect.get(s.a) ?? s.a), b = keepMap.get(redirect.get(s.b) ?? s.b);
      if (a === b) continue;
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      if (seen.has(key)) continue;
      seen.add(key);
      next.push({ ...s, a, b });
    }
    nodes = kept; struts = next;
  }
  const sites = [];
  nodes.forEach((n, i) => { if (n.kind === 'skin') sites.push(i); });
  if (sites.length < 2) return { graph: { nodes, struts }, report: { skinNodes: sites.length, skinStruts: 0, joined, ms: Math.round(nowMs() - t0) } };

  // 2. Regions on the refined surface.
  const h = opts.surfaceStep ?? strutLength / 4;
  const S = refineBodySurface(body, h);
  const nv = S.positions.length / 3;
  const adj = Array.from({ length: nv }, () => []);
  const edges = new Set();
  for (let t = 0; t < S.indices.length; t += 3) for (let k = 0; k < 3; k++) {
    const a = S.indices[t + k], b = S.indices[t + (k + 1) % 3];
    const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
    if (edges.has(key)) continue;
    edges.add(key);
    const w = Math.hypot(S.positions[3 * a] - S.positions[3 * b], S.positions[3 * a + 1] - S.positions[3 * b + 1], S.positions[3 * a + 2] - S.positions[3 * b + 2]);
    adj[a].push(b, w); adj[b].push(a, w);
  }
  const vgrid = new Map();
  const vkey = (p) => p.map((v) => Math.floor(v / h)).join(',');
  for (let v = 0; v < nv; v++) {
    const k = vkey([S.positions[3 * v], S.positions[3 * v + 1], S.positions[3 * v + 2]]);
    if (!vgrid.has(k)) vgrid.set(k, []);
    vgrid.get(k).push(v);
  }
  const dist = new Float64Array(nv).fill(Infinity), label = new Int32Array(nv).fill(-1);
  const heap = makeHeap();
  sites.forEach((ni, si) => {
    const p = nodes[ni].p, c = p.map((v) => Math.floor(v / h));
    let found = false;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const list = vgrid.get(`${c[0] + dx},${c[1] + dy},${c[2] + dz}`);
      if (!list) continue;
      for (const v of list) {
        const d = Math.hypot(S.positions[3 * v] - p[0], S.positions[3 * v + 1] - p[1], S.positions[3 * v + 2] - p[2]);
        if (d <= 1.5 * h && d < dist[v]) { dist[v] = d; label[v] = si; heap.push(d, v); found = true; }
      }
    }
    if (!found) {
      let best = -1, bd = Infinity;
      for (let v = 0; v < nv; v++) { const d = Math.hypot(S.positions[3 * v] - p[0], S.positions[3 * v + 1] - p[1], S.positions[3 * v + 2] - p[2]); if (d < bd) { bd = d; best = v; } }
      if (best >= 0 && bd < dist[best]) { dist[best] = bd; label[best] = si; heap.push(bd, best); }
    }
  });
  while (heap.size) {
    const [d, v] = heap.pop();
    if (d > dist[v]) continue;
    const a = adj[v];
    for (let q = 0; q < a.length; q += 2) {
      const u = a[q], nd = d + a[q + 1];
      if (nd < dist[u]) { dist[u] = nd; label[u] = label[v]; heap.push(nd, u); }
    }
  }
  const cand = new Map();
  const nbrs = sites.map(() => new Set());
  for (const key of edges) {
    const a = Math.floor(key / 4294967296), b = key - a * 4294967296;
    const la = label[a], lb = label[b];
    if (la < 0 || lb < 0 || la === lb) continue;
    const pk = la < lb ? `${la},${lb}` : `${lb},${la}`;
    cand.set(pk, la < lb ? [la, lb] : [lb, la]);
    nbrs[la].add(lb); nbrs[lb].add(la);
  }

  // 3. Gabriel filter among region neighbors.
  const existing = new Set(struts.map((s) => (s.a < s.b ? `${s.a},${s.b}` : `${s.b},${s.a}`)));
  let skinStruts = 0;
  for (const [i, j] of cand.values()) {
    const pi = nodes[sites[i]].p, pj = nodes[sites[j]].p;
    const m = vscale(vadd(pi, pj), 0.5), r = vdist(pi, pj) / 2;
    let blocked = false;
    for (const k of new Set([...nbrs[i], ...nbrs[j]])) {
      if (k === i || k === j) continue;
      if (vdist(nodes[sites[k]].p, m) < r * (1 - 1e-9)) { blocked = true; break; }
    }
    if (blocked) continue;
    const a = sites[i], b = sites[j];
    const ek = a < b ? `${a},${b}` : `${b},${a}`;
    if (existing.has(ek)) continue;
    existing.add(ek);
    const ids = [nodes[a].id, nodes[b].id].sort();
    struts.push({ id: `k:${ids[0]}|${ids[1]}`, a, b, kind: 'skin' });
    skinStruts++;
  }
  // A node whose region is empty (a nearer node took every surface vertex
  // round it) joins the nodes owning the vertices nearest to it, so every
  // skin node is woven in.
  const skinDeg = new Map(sites.map((ni) => [ni, 0]));
  for (const s of struts) if (s.kind === 'skin') { skinDeg.set(s.a, skinDeg.get(s.a) + 1); skinDeg.set(s.b, skinDeg.get(s.b) + 1); }
  let rescued = 0;
  sites.forEach((ni, si) => {
    if (skinDeg.get(ni) > 0) return;
    const p = nodes[ni].p;
    const owners = [];
    const near = [];
    for (let v = 0; v < nv; v++) { if (label[v] < 0 || label[v] === si) continue; near.push([Math.hypot(S.positions[3 * v] - p[0], S.positions[3 * v + 1] - p[1], S.positions[3 * v + 2] - p[2]), label[v]]); }
    near.sort((x, y) => x[0] - y[0]);
    for (const [, l] of near) { if (!owners.includes(l)) owners.push(l); if (owners.length === 2) break; }
    for (const l of owners) {
      const a = ni, b = sites[l];
      const ek = a < b ? `${a},${b}` : `${b},${a}`;
      if (existing.has(ek)) continue;
      existing.add(ek);
      const ids = [nodes[a].id, nodes[b].id].sort();
      struts.push({ id: `k:${ids[0]}|${ids[1]}`, a, b, kind: 'skin' });
      skinStruts++;
    }
    rescued++;
  });
  return { graph: { nodes, struts }, report: { skinNodes: sites.length, skinStruts, joined, joinDeclined, rescued, surfaceTriangles: S.indices.length / 3, ms: Math.round(nowMs() - t0) } };
}

function givenSkin(graph, body, grid, opts, t0) {
  const net = opts.net;
  if (!net || !Array.isArray(net.nodes) || !Array.isArray(net.edges)) return { graph, report: { skinNodes: 0, skinStruts: 0, reason: 'no net given' } };
  const strutLength = opts.strutLength ?? meanStrutLength(graph);
  const tol = opts.snapTolerance ?? 0.5 * strutLength;
  const nodes = graph.nodes.map((n) => ({ ...n, p: n.p.slice() }));
  const netIdx = net.nodes.map((p, i) => { nodes.push({ id: `g:${i}`, p: p.slice(), kind: 'skin' }); return nodes.length - 1; });
  let netEdges = net.edges.map(([i, j]) => [netIdx[i], netIdx[j]]);
  const redirect = new Map();
  let snapped = 0, split = 0;
  nodes.forEach((n, ni) => {
    if (n.kind !== 'skin' || n.id.startsWith('g:')) return;
    let best = -1, bd = Infinity;
    for (const gi of netIdx) { const d = vdist(nodes[gi].p, n.p); if (d < bd) { bd = d; best = gi; } }
    if (best >= 0 && bd <= tol) { redirect.set(ni, best); snapped++; return; }
    // Split the nearest net edge at the foot of this node.
    let be = -1, bt = 0, bdist = Infinity;
    netEdges.forEach(([a, b], ei) => {
      const A = nodes[a].p, B = nodes[b].p, d = vsub(B, A), L2 = vdot(d, d);
      const t = L2 > 0 ? Math.max(0, Math.min(1, vdot(vsub(n.p, A), d) / L2)) : 0;
      const q = vadd(A, vscale(d, t)), dd = vdist(q, n.p);
      if (dd < bdist) { bdist = dd; be = ei; bt = t; }
    });
    if (be < 0) return;
    const [a, b] = netEdges[be];
    const foot = vlerp(nodes[a].p, nodes[b].p, bt);
    const c = closestOnBody(body, grid, foot);
    n.p = c ? c.p : foot;
    netEdges.splice(be, 1, [a, ni], [ni, b]);
    split++;
  });
  const keepMap = new Map(), kept = [];
  nodes.forEach((n, i) => { if (!redirect.has(i)) { keepMap.set(i, kept.length); kept.push(n); } });
  const struts = [], seen = new Set();
  const add = (s) => {
    const a = keepMap.get(redirect.get(s.a) ?? s.a), b = keepMap.get(redirect.get(s.b) ?? s.b);
    if (a === undefined || b === undefined || a === b) return;
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    if (seen.has(key)) return;
    seen.add(key);
    struts.push({ ...s, a, b });
  };
  for (const s of graph.struts) add(s);
  for (const [a, b] of netEdges) {
    const ids = [nodes[a].id, nodes[b].id].sort();
    add({ id: `k:${ids[0]}|${ids[1]}`, a, b, kind: 'skin' });
  }
  const used = new Set(struts.flatMap((s) => [s.a, s.b]));
  const map2 = new Map(), finalNodes = [];
  kept.forEach((n, i) => { if (used.has(i)) { map2.set(i, finalNodes.length); finalNodes.push(n); } });
  const finalStruts = struts.map((s) => ({ ...s, a: map2.get(s.a), b: map2.get(s.b) }));
  return { graph: { nodes: finalNodes, struts: finalStruts }, report: { skinNodes: finalNodes.filter((n) => n.kind === 'skin').length, skinStruts: finalStruts.filter((s) => s.kind === 'skin').length, snapped, split, ms: Math.round(nowMs() - t0) } };
}

// Curves
//
// Each strut is an open, clamped, uniform, degree-3 NURBS with `detail`
// control points (4..7, default 5: two spans, C2 at the inner knot). Five
// rather than four gives a middle point that bends the strut while its end
// tangents barely turn. Across a node, curve-to-curve continuity has no
// meaning (three or more curves meet there); smoothness through a node is
// the joint's job, and the joint follows each strut's end tangent.

export function infillStrutKnots(n, degree = 3) {
  const spans = n - degree;
  const k = [];
  for (let i = 0; i <= degree; i++) k.push(0);
  for (let i = 1; i < spans; i++) k.push(i / spans);
  for (let i = 0; i <= degree; i++) k.push(1);
  return k;
}
// de Boor evaluation of a non-rational B-spline.
function bsplinePoint(knots, degree, ctrl, u) {
  const n = ctrl.length;
  let span = degree;
  while (span < n - 1 && u >= knots[span + 1]) span++;
  const d = [];
  for (let j = 0; j <= degree; j++) d.push(ctrl[span - degree + j].slice());
  for (let r = 1; r <= degree; r++) {
    for (let j = degree; j >= r; j--) {
      const i = span - degree + j;
      const den = knots[i + degree - r + 1] - knots[i];
      const a = den > 0 ? (u - knots[i]) / den : 0;
      d[j] = vlerp(d[j - 1], d[j], a);
    }
  }
  return d[degree];
}
function greville(knots, degree, n) {
  const g = [];
  for (let i = 0; i < n; i++) { let s = 0; for (let k = 1; k <= degree; k++) s += knots[i + k]; g.push(s / degree); }
  return g;
}

// The frame a strut's hand edits are stored in: x along the chord, y and z
// across it, from an up vector fixed by rule (world Z, or world Y when the
// chord is within 18 degrees of Z). Offsets are fractions of the chord
// length, so a strut that moves, turns a little or stretches with its body
// carries its bend with it.
export function strutChordFrame(A, B) {
  const d = vsub(B, A), L = vlen(d);
  const x = L > 0 ? vscale(d, 1 / L) : [1, 0, 0];
  const up = Math.abs(x[2]) > 0.95 ? [0, 1, 0] : [0, 0, 1];
  const y = vnorm(vsub(up, vscale(x, vdot(up, x))));
  const z = vcross(x, y);
  return { origin: A.slice(), x, y, z, length: L };
}

// The chord-frame offsets that turn `birth` control points into `edited`
// ones, for a strut from A to B. What the app stores when a point is
// dragged.
export function infillCvOffsets(A, B, birth, edited) {
  const f = strutChordFrame(A, B);
  const L = f.length || 1;
  return birth.map((p, i) => {
    const d = vsub(edited[i], p);
    return [vdot(d, f.x) / L, vdot(d, f.y) / L, vdot(d, f.z) / L];
  });
}

// Offsets stored at one detail, read at another: the offset B-spline
// evaluated at the new control points' Greville abscissae. Exact for any
// offset field linear along the strut.
function resampleOffsets(off, n) {
  if (off.length === n) return off;
  if (off.length < 4) return new Array(n).fill(0).map(() => [0, 0, 0]);
  const k0 = infillStrutKnots(off.length), k1 = infillStrutKnots(n);
  return greville(k1, 3, n).map((u) => bsplinePoint(k0, 3, off, u));
}

// At each node, struts are paired most-opposite first; a pair's two ends
// take the direction that runs one into the other (d_s - d_partner), so a
// relaxed lattice is a set of smooth fibers crossing at the nodes. An
// unpaired strut (odd valence, or no partner within 90 degrees of
// opposite) keeps its chord. Relax blends chord -> that direction.
function relaxedEndDirections(graph, relax) {
  const n = graph.nodes.length;
  const inc = Array.from({ length: n }, () => []);
  graph.struts.forEach((s, i) => { inc[s.a].push([i, 'a']); inc[s.b].push([i, 'b']); });
  const out = graph.struts.map((s) => {
    const d = vnorm(vsub(graph.nodes[s.b].p, graph.nodes[s.a].p));
    return { a: d, b: vscale(d, -1) };
  });
  if (!(relax > 0)) return out;
  for (let v = 0; v < n; v++) {
    const ends = inc[v].filter(([i]) => graph.struts[i].kind === graph.nodes[v].kind || graph.nodes[v].kind === 'core');
    const dirs = ends.map(([i, e]) => out[i][e]);
    const pairs = [];
    for (let p = 0; p < ends.length; p++) for (let q = p + 1; q < ends.length; q++) pairs.push([vdot(dirs[p], dirs[q]), p, q]);
    pairs.sort((x, y) => x[0] - y[0]);
    const used = new Set();
    const through = new Map();
    for (const [c, p, q] of pairs) {
      if (c >= 0) break;
      if (used.has(p) || used.has(q)) continue;
      used.add(p); used.add(q);
      const t = vnorm(vsub(dirs[p], dirs[q]));
      through.set(p, t); through.set(q, vscale(t, -1));
    }
    for (const [p, t] of through) {
      const [i, e] = ends[p];
      out[i] = { ...out[i], [e]: vnorm(vlerp(dirs[p], t, relax)) };
    }
  }
  return out;
}

// One curve per strut. opts: detail, relax, and body + grid to put a skin
// strut's inner control points on the surface.
export function infillCurves(graph, opts = {}) {
  const n = Math.max(4, Math.min(7, Math.round(opts.detail ?? INFILL_DEFAULTS.detail)));
  const knots = infillStrutKnots(n);
  const ends = relaxedEndDirections(graph, opts.relax ?? INFILL_DEFAULTS.relax);
  const body = opts.body, grid = opts.grid;
  return graph.struts.map((s, i) => {
    const A = graph.nodes[s.a].p, B = graph.nodes[s.b].p;
    const L = vdist(A, B);
    const h = L / (n - 1);
    const P1 = vadd(A, vscale(ends[i].a, h)), Pm = vadd(B, vscale(ends[i].b, h));
    const birth = [A.slice(), P1];
    for (let k = 2; k <= n - 3; k++) birth.push(vlerp(P1, Pm, (k - 1) / (n - 3)));
    birth.push(Pm, B.slice());
    if (s.kind === 'skin' && body && grid) {
      for (let k = 1; k < n - 1; k++) { const c = closestOnBody(body, grid, birth[k]); if (c) birth[k] = c.p; }
    }
    let ctrlPts = birth;
    if (s.cvOffsets) {
      const off = resampleOffsets(s.cvOffsets, n);
      const f = strutChordFrame(A, B);
      ctrlPts = birth.map((p, k) => (k === 0 || k === n - 1 ? p : vadd(p, vscale(vadd(vadd(vscale(f.x, off[k][0]), vscale(f.y, off[k][1])), vscale(f.z, off[k][2])), f.length))));
    }
    const out = { id: s.id, kind: s.kind, a: s.a, b: s.b, curve: { degree: 3, knots: knots.slice(), ctrlPts }, birth };
    if (s.radius > 0) out.radius = s.radius;
    return out;
  });
}

// Overrides — hand edits, keyed by identity.
//
//   { nodes:  { [nodeId]:  { offset: [x,y,z] } },
//     struts: { [strutId]: { cv: [[fx,fy,fz], ...], deleted: true, radius } },
//     added:  [{ id, a: nodeId, b: nodeId }],
//     merged: [[keepNodeId, goneNodeId]] }
//
// Applied in that order: node moves, merges, deletions, additions, then
// shapes and radii ride on the surviving struts. An override whose identity
// is not in this graph is an orphan: listed, never applied, never dropped —
// the caller keeps the overrides object whole, so an undo or a slider
// return that brings the identity back re-attaches the edit.
export function applyInfillOverrides(graph, ov = {}) {
  const orphans = [];
  const byId = new Map(graph.nodes.map((n, i) => [n.id, i]));
  let nodes = graph.nodes.map((n) => ({ ...n, p: n.p.slice() }));
  let struts = graph.struts.map((s) => ({ ...s }));
  let applied = 0;
  for (const [id, o] of Object.entries(ov.nodes || {})) {
    const i = byId.get(id);
    if (i === undefined) { orphans.push({ kind: 'node', id }); continue; }
    if (o && Array.isArray(o.offset)) { nodes[i].p = vadd(nodes[i].p, o.offset); applied++; }
  }
  const redirect = new Map();
  for (const pair of ov.merged || []) {
    const [keep, gone] = pair || [];
    const ik = byId.get(keep), ig = byId.get(gone);
    if (ik === undefined || ig === undefined || ik === ig) { orphans.push({ kind: 'merged', id: `${keep}+${gone}` }); continue; }
    redirect.set(ig, ik); applied++;
  }
  if (redirect.size) {
    const root = (i) => { while (redirect.has(i)) i = redirect.get(i); return i; };
    const seen = new Set();
    struts = struts.map((s) => ({ ...s, a: root(s.a), b: root(s.b) })).filter((s) => {
      if (s.a === s.b) return false;
      const k = s.a < s.b ? `${s.a},${s.b}` : `${s.b},${s.a}`;
      if (seen.has(k)) return false;
      seen.add(k); return true;
    });
  }
  const sOv = ov.struts || {};
  const present = new Set(struts.map((s) => s.id));
  for (const id of Object.keys(sOv)) if (!present.has(id) && !(ov.added || []).some((a) => a.id === id)) orphans.push({ kind: 'strut', id });
  struts = struts.filter((s) => { if (sOv[s.id] && sOv[s.id].deleted) { applied++; return false; } return true; });
  for (const add of ov.added || []) {
    const a = byId.get(add.a), b = byId.get(add.b);
    if (a === undefined || b === undefined || redirect.has(a) || redirect.has(b)) { orphans.push({ kind: 'added', id: add.id }); continue; }
    const kind = nodes[a].kind === 'skin' && nodes[b].kind === 'skin' ? 'skin' : 'core';
    struts.push({ id: add.id, a, b, kind, added: true });
    applied++;
  }
  for (const s of struts) {
    const o = sOv[s.id];
    if (!o || o.deleted) continue;
    if (Array.isArray(o.cv)) { s.cvOffsets = o.cv.map((v) => v.slice()); applied++; }
    if (o.radius > 0) { s.radius = o.radius; applied++; }
  }
  if (redirect.size) {
    const used = new Set(struts.flatMap((s) => [s.a, s.b]));
    const map = new Map(), kept = [];
    nodes.forEach((n, i) => { if (!redirect.has(i) || used.has(i)) { map.set(i, kept.length); kept.push(n); } });
    struts = struts.map((s) => ({ ...s, a: map.get(s.a), b: map.get(s.b) }));
    nodes = kept;
  }
  return { graph: { nodes, struts }, orphans, applied };
}

// The whole operation

// Size from the body's volume, so a default run holds about `targetStruts`
// core struts whatever the cell type: low density is the design target and
// each cell type packs a different number of struts per cell.
export const INFILL_TARGET_STRUTS = 300;
export function infillDefaultSize(volume, cell = INFILL_DEFAULTS.cell, targetStruts = INFILL_TARGET_STRUTS) {
  const c = INFILL_CELLS[cell] || INFILL_CELLS.kelvin;
  return Math.cbrt(Math.max(volume, 0) * c.strutsPerCell / Math.max(1, targetStruts));
}
// The core strut count a run will make, before running it: volume over
// cell volume times struts per cell, so a caller can size a run before starting it.
export function infillEstimate(volume, opts = {}) {
  const cell = opts.cell ?? INFILL_DEFAULTS.cell;
  const c = INFILL_CELLS[cell] || INFILL_CELLS.kelvin;
  const size = opts.size > 0 ? opts.size : infillDefaultSize(volume, cell);
  return { size, struts: Math.round(volume / (size * size * size) * c.strutsPerCell), cell };
}

// Exchange — the graph as flat typed arrays, and a hash two apps can
// compare for one fixture and one set of settings.

// { nodes: Float64Array(3n), edges: Uint32Array(2m), nodeSkin: Uint8Array(n),
//   edgeSkin: Uint8Array(m), strutR: Float32Array(m) (0 = the default) }
export function infillGraphArrays(graph) {
  const n = graph.nodes.length, m = graph.struts.length;
  const nodes = new Float64Array(3 * n), nodeSkin = new Uint8Array(n);
  graph.nodes.forEach((q, i) => { nodes[3 * i] = q.p[0]; nodes[3 * i + 1] = q.p[1]; nodes[3 * i + 2] = q.p[2]; nodeSkin[i] = q.kind === 'skin' ? 1 : 0; });
  const edges = new Uint32Array(2 * m), edgeSkin = new Uint8Array(m), strutR = new Float32Array(m);
  graph.struts.forEach((s, i) => { edges[2 * i] = s.a; edges[2 * i + 1] = s.b; edgeSkin[i] = s.kind === 'skin' ? 1 : 0; strutR[i] = s.radius > 0 ? s.radius : 0; });
  return { nodes, edges, nodeSkin, edgeSkin, strutR };
}

// Canonical hash: each coordinate quantized to round(v / (1e-6 x size));
// nodes sorted lexicographically by their three quantized ints; each edge
// written (lo, hi) in that order; edges sorted by (lo, hi). FNV-1a 32 over
// every quantized node int, then every edge int, each fed as the four
// little-endian bytes of its int32. Eight lowercase hex digits.
export function infillGraphHash(graph, size) {
  const quant = 1e-6 * size;
  const q = graph.nodes.map((n, i) => ({ i, k: n.p.map((v) => Math.round(v / quant)) }));
  q.sort((a, b) => a.k[0] - b.k[0] || a.k[1] - b.k[1] || a.k[2] - b.k[2]);
  const rank = new Int32Array(graph.nodes.length);
  q.forEach((e, r) => { rank[e.i] = r; });
  const edges = graph.struts.map((s) => { const a = rank[s.a], b = rank[s.b]; return a < b ? [a, b] : [b, a]; });
  edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let h = 0x811c9dc5;
  const feed = (v) => { for (let k = 0; k < 4; k++) { h ^= (v >>> (8 * k)) & 0xff; h = Math.imul(h, 0x01000193) >>> 0; } };
  for (const e of q) for (const v of e.k) feed(v | 0);
  for (const [a, b] of edges) { feed(a); feed(b); }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function valenceHistogram(graph, kind) {
  const deg = new Array(graph.nodes.length).fill(0);
  for (const s of graph.struts) { deg[s.a]++; deg[s.b]++; }
  const h = {};
  graph.nodes.forEach((n, i) => { if (!kind || n.kind === kind) h[deg[i]] = (h[deg[i]] || 0) + 1; });
  return h;
}
function componentCount(graph) {
  const parent = graph.nodes.map((_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (const s of graph.struts) parent[find(s.a)] = find(s.b);
  return new Set(graph.nodes.map((_, i) => find(i))).size;
}

// The skin source a run uses: 'given' for Snap (the caller's net), 'faces'
// for a lattice whose struts know their cells, 'woven' (Gabriel) otherwise.
export function infillSkinSource(cell, boundary, asked) {
  if (boundary === 'snap') return 'given';
  if (asked === 'woven' || asked === 'given') return asked;
  return cell === 'kelvin' || cell === 'voronoi' ? 'faces' : 'woven';
}

// buildInfill(mesh, opts) -> { ok, graph, curves, report } | { ok: false, reason }
// opts: INFILL_DEFAULTS keys, size (else from volume), skinSource ('woven'
// | 'given'), net (for 'given'), snapTolerance, overrides, targetStruts.
// `mesh` may also be what infillBodyCheck returned for it (ok, welded), which
// skips the weld: a caller rebuilding against one body many times checks once.
export function buildInfill(mesh, opts = {}) {
  const t0 = nowMs();
  const body = mesh && mesh.ok === true && mesh.windingSign && mesh.bbox ? mesh : infillBodyCheck(mesh);
  if (!body.ok) return { ok: false, reason: body.reason, body: { naked: body.naked, nonManifold: body.nonManifold, volume: body.volume } };
  const cell = opts.cell ?? INFILL_DEFAULTS.cell;
  if (!CELL_DEFS[cell]) return { ok: false, reason: `unknown cell type "${cell}"` };
  const size = opts.size > 0 ? opts.size : infillDefaultSize(body.volume, cell, opts.targetStruts);
  const lat = latticeGraph(body.bbox, { ...opts, cell, size });
  if (!lat.ok) return lat;
  const strutLength = INFILL_CELLS[cell].strutLength * size;
  const boundary = opts.boundary ?? INFILL_DEFAULTS.boundary;
  const skinOn = (opts.skin ?? INFILL_DEFAULTS.skin) && boundary !== 'cells';
  const skinSource = infillSkinSource(cell, boundary, opts.skinSource);
  const clip = clipGraphToBody(lat.graph, body, { ...opts, boundary, strutLength, faceSkin: skinOn && skinSource === 'faces' });
  let graph = clip.graph;
  let skinReport = null;
  if (skinOn && skinSource === 'faces') {
    skinReport = { source: 'faces', skinNodes: graph.nodes.filter((n) => n.kind === 'skin').length, skinStruts: graph.struts.filter((s) => s.kind === 'skin').length };
  } else if (skinOn) {
    const sk = skinNet(graph, body, { source: skinSource, net: opts.net, snapTolerance: opts.snapTolerance, strutLength, grid: clip.grid, mergeLength: (opts.mergeFraction ?? INFILL_DEFAULTS.mergeFraction) * strutLength });
    graph = sk.graph; skinReport = { source: skinSource, ...sk.report };
  }
  const trial = infillCurves(graph, { detail: opts.detail, relax: opts.relax, body, grid: clip.grid });
  const angle = enforceArmAngle(graph, opts.minArmAngle ?? INFILL_DEFAULTS.minArmAngle, trial);
  graph = angle.graph;
  const ov = applyInfillOverrides(graph, opts.overrides || {});
  graph = ov.graph;
  const curves = infillCurves(graph, { detail: opts.detail, relax: opts.relax, body, grid: clip.grid });
  let minLen = Infinity, maxLen = 0;
  for (const s of graph.struts) { const L = vdist(graph.nodes[s.a].p, graph.nodes[s.b].p); if (L < minLen) minLen = L; if (L > maxLen) maxLen = L; }
  const report = {
    cell, size, strutLength,
    nodes: graph.nodes.length,
    struts: graph.struts.length,
    coreStruts: graph.struts.filter((s) => s.kind === 'core').length,
    skinStruts: graph.struts.filter((s) => s.kind === 'skin').length,
    skinNodes: graph.nodes.filter((n) => n.kind === 'skin').length,
    valence: valenceHistogram(graph),
    shortest: graph.struts.length ? minLen : null,
    longest: graph.struts.length ? maxLen : null,
    components: componentCount(graph),
    clip: clip.report,
    skin: skinReport,
    shallowRemoved: angle.removed,
    pipe: infillPipeRadius(graph, { strutLength }),
    orphans: ov.orphans,
    body: { triangles: body.triangles, volume: body.volume, area: body.area, bbox: body.bbox },
    ms: Math.round(nowMs() - t0),
  };
  if (!graph.struts.length) return { ok: false, reason: 'no strut fits inside this body at this Size; lower Size', report };
  return { ok: true, graph, curves, frame: lat.frame, report };
}
