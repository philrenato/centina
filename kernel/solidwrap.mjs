// WRAPPING N SOLIDS INTO ONE SuperB CAGE — the field, and the cage that is
// projected onto it.
//
// WHAT THIS IS. A box cage, moved onto the signed distance field of the input
// solids. No isosurface is ever extracted, so no triangle is ever produced and
// there is no quad remesh to write — which is exactly why this is affordable
// where a marching-cubes or dual-contouring route is not. `superbBoxCage`
// builds an all-quad cube-sphere; `superbSphereCage` and `superbEllipsoidCage`
// are that cage projected onto an implicit surface. This is the same cage
// projected onto an arbitrary one, so the topology is fixed before the solve
// begins and cannot degrade: all quads, interior valence 4, exactly eight
// valence-3 corners, Euler characteristic 2.
//
// ⚠ WHAT IT COSTS, PERMANENTLY, AND THESE ARE NOT DEFECTS. The result is genus
// 0: a through-hole in the input fills. Deep narrow pockets skin over. Sharp
// edges arrive rounded unless creased afterwards, and a joined box comes back
// as a soft box. Fidelity here is a slider, not a promise; if the ask is "give
// me my polysurface back", this is not that tool and no amount of iteration
// makes it that tool.
//
// ⚠ THE INPUT IS A TRIANGLE SOUP, DELIBERATELY. Every host — SuperB, NURBS,
// polysurface — reaches this through its display mesh, so all three take one
// identical path and nothing here touches the patch conversion. The cost is
// that a Skin of 0 means "touches the tessellation", off by the document's own
// chord tolerance.
//
// The four stages: field, start box, wrap, refit. Only the field is new; the
// wrap is Kobbelt et al., "A Shrink Wrapping Approach to Remeshing Polygonal
// Surfaces" (Eurographics 1999), alternating Taubin smoothing with a snap along
// the field gradient, and the refit is `refitCageToLimitTargets`, which already
// ships.
//
// ⚠⚠ `wrapSolidsToSuperbCage` IS A PURE FUNCTION OF ITS INPUTS AND PARAMETERS,
// and that is a requirement rather than a happy accident: the controls above it
// are live, so every drag of a slider re-runs the whole solve from the stored
// inputs. Nothing here is random, nothing is cached across calls, nothing reads
// a clock, and the one place a solver could drift — the eigen decomposition of
// a degenerate covariance — has a deterministic tie-break of its own. Same
// inputs and params in, byte-identical cage out. Anything added here that does
// not hold that turns the controls into a one-shot dialog.

import { buildBVH, buildMeshPseudonormals, bvhClosestPoint, bvhIntersect } from './bvh.mjs';
import { superbBoxCage } from './subdprimitives.mjs';
import { buildTopology, subdivideCatmullClark, triangulateFace } from './subd.mjs';
import { refitCageToLimitTargets } from './subdsimplify.mjs';
import { jacobiEigenSym3 } from './refit.mjs';

// Refusals are named, and the names are the contract a caller matches on — the
// message is for a person and may be reworded, the reason may not.
export const SOLID_WRAP_REFUSAL = Object.freeze({
  NO_VOLUME: 'no-volume',
  MEMBERS_DO_NOT_FUSE: 'members-do-not-fuse',
});

// ---------------------------------------------------------------------------
// THE BLEND
// ---------------------------------------------------------------------------

/**
 * The published quadratic polynomial smooth minimum. At k = 0 it is exactly
 * `Math.min`, which is the exact union — the only thing the wrap needs outside
 * the solids — and above 0 it rounds the union's crease by about k/4.
 *
 * ⚠ IT IS NOT ASSOCIATIVE, so folding it over N members depends on the fold
 * order by up to k/4 inside the blend region. The fold below runs in the
 * caller's own member order, which makes the answer deterministic rather than
 * order-independent; that is a named limitation, not a bug to be found later.
 */
export function smoothMinPoly(a, b, k) {
  if (!(k > 0)) return Math.min(a, b);
  let h = 0.5 + 0.5 * (b - a) / k;
  if (h < 0) h = 0; else if (h > 1) h = 1;
  return (b + h * (a - b)) - k * h * (1 - h);
}

/**
 * The blend radius that makes the Fuse control read as *gaps up to this wide
 * are bridged*.
 *
 * Two surfaces a gap g apart both read g/2 at the midpoint between them, and
 * `smoothMinPoly(g/2, g/2, k)` is g/2 - k/4. That reaches zero — the moment a
 * bridge exists at all — at k = 2g. So the control a person types is halved
 * into the formula's own k, and the label is then literally true instead of
 * being a number whose meaning has to be learned by dragging it.
 */
export function fuseBlendRadius(fuse) {
  return Math.max(0, fuse) * 2;
}

// ---------------------------------------------------------------------------
// THE FIELD
// ---------------------------------------------------------------------------

function boundsOfPositions(positions) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < positions.length; i += 3) {
    for (let d = 0; d < 3; d += 1) {
      const v = positions[i + d];
      if (v < lo[d]) lo[d] = v;
      if (v > hi[d]) hi[d] = v;
    }
  }
  return { lo, hi };
}

/**
 * The scalar field N solids present to the wrap.
 *
 * `members` is an array of `{ positions }`, each a flat unindexed triangle
 * array — the shape `buildBVH` and a renderer already have.
 *
 *   opts.fuse   mm. Gaps up to this wide are bridged, and creases between
 *               members round by about this much. 0 is the sharp union.
 *   opts.skin   mm, signed. The isolevel: 0 lands on the input's surfaces,
 *               positive stands the bag off them, negative eats in.
 *
 * Returns a record whose `distanceAt(p)` is the blended signed distance with
 * the skin already subtracted, so the surface the wrap chases is always the
 * zero set. A member whose mesh is not closed contributes an UNSIGNED distance
 * and is listed in `openMembers`: there is no inside to report for it, and
 * inventing one is how a wrap ends up inside-out.
 */
/* THE SIGNED VOLUME OF A CLOSED TRIANGLE SOUP. Negative when the mesh faces
   inward, and its magnitude is the enclosed volume either way. */
export function soupSignedVolume(p) {
  let vol = 0;
  for (let i = 0; i + 8 < p.length; i += 9) {
    vol += (p[i] * (p[i + 4] * p[i + 8] - p[i + 5] * p[i + 7])
          - p[i + 1] * (p[i + 3] * p[i + 8] - p[i + 5] * p[i + 6])
          + p[i + 2] * (p[i + 3] * p[i + 7] - p[i + 4] * p[i + 6])) / 6;
  }
  return vol;
}

/* ⚠ SOLIDITY IS IMPUTED, NOT DEMANDED. A mesh that plainly bounds a region is
   treated as one whatever its rim does: every curved primitive in this kernel's
   host is a wall plus caps whose rims sample the same circle at different
   angles, so no weld tolerance closes them and none is combinatorially closed.
   A caller that knows better still wins — `solid: false` refuses the imputation.
   The 1% floor is against a flat sheet, whose signed volume is a rounding
   error against its own bounding box. */
export function soupEnclosesVolume(p) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < p.length; i += 3) for (let d = 0; d < 3; d += 1) {
    if (p[i + d] < lo[d]) lo[d] = p[i + d];
    if (p[i + d] > hi[d]) hi[d] = p[i + d];
  }
  const box = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
  return box > 0 && Math.abs(soupSignedVolume(p)) / box > 0.01;
}

/* ⚠⚠ AN INWARD-WOUND MEMBER POISONS THE WHOLE FIELD, so every soup is turned to
   face outward before the field is built from it. It costs nothing for a member
   signed by ray parity, which counts crossings and cannot tell winding from
   anything. It matters completely for one whose mesh IS combinatorially closed,
   because that one is signed by its pseudonormals and they follow the winding:
   five spheres wound inward read -276 at a point 70mm OUTSIDE all of them, the
   blended minimum went negative across the whole scene, and the wrap collapsed
   into the one lobe the flooding left — a chain of five came back as a lump
   18mm across. Whole classes of mesh arrive this way; three of the four curved
   primitives in this kernel's own host do. */
export function orientSoupOutward(p) {
  if (soupSignedVolume(p) >= 0) return p;
  const out = p.slice();
  for (let i = 0; i + 8 < out.length; i += 9) {
    const t0 = out[i + 3], t1 = out[i + 4], t2 = out[i + 5];
    out[i + 3] = out[i + 6]; out[i + 4] = out[i + 7]; out[i + 5] = out[i + 8];
    out[i + 6] = t0; out[i + 7] = t1; out[i + 8] = t2;
  }
  return out;
}

export function makeSolidsField(members, opts = {}) {
  const fuse = Math.max(0, opts.fuse ?? 0);
  const skin = opts.skin ?? 0;
  const k = fuseBlendRadius(fuse);
  const parts = [];
  const openMembers = [];
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];

  members.forEach((m, idx) => {
    if (!m.positions || m.positions.length < 9) { openMembers.push(idx); return; }
    const positions = orientSoupOutward(m.positions);
    const bvh = buildBVH(positions);
    const pn = buildMeshPseudonormals(positions, opts);
    /* ⚠ A PARITY-SIGNED MEMBER IS NOT OPEN. The flag was added above and this
       line was left as it was, so a member the field signs perfectly well still
       reported as having no inside — which is the list the caller reads to tell
       a reader what it could not wrap, and the count the refusal is built on. */
    const solid = m.solid === true || (m.solid !== false && soupEnclosesVolume(positions));
    if (!pn.closed && !solid) openMembers.push(idx);
    const b = boundsOfPositions(positions);
    for (let d = 0; d < 3; d += 1) {
      if (b.lo[d] < lo[d]) lo[d] = b.lo[d];
      if (b.hi[d] > hi[d]) hi[d] = b.hi[d];
    }
    parts.push({ index: idx, positions, bvh, pn, bounds: b, parity: !pn.closed && solid });
  });

  const finite = parts.length > 0;
  const bounds = finite ? { lo, hi } : { lo: [0, 0, 0], hi: [0, 0, 0] };
  const diagonal = finite ? Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) : 0;
  // The field bulges outward by at most the skin plus the blend's own rounding,
  // so anything that has to contain the zero set — the start box, the counting
  // grid — is padded by this and not by the members' own bounds.
  const padding = Math.max(0, skin) + fuse + (diagonal > 0 ? diagonal * 0.02 : 0);
  const gradientStep = opts.gradientStep ?? Math.max(diagonal * 5e-4, 1e-9);

  /* ⚠⚠ THE SIGN OF A MESH THAT IS NOT COMBINATORIALLY CLOSED, and this is the
     ordinary case rather than the exotic one. A host draws a solid as a side
     surface plus caps whose rims sample the same curve at different angles, so
     nothing welds, nothing is a T-junction, and no repair could zip them: a
     cone's own naked boundary came back as one loop at z = 0 spanning radii 47
     to 179, wandering across the cap rather than round a rim.
     The pseudonormal is meaningless there. Asked anyway — by asserting the mesh
     is closed when the B-rep says the OBJECT is — it signed 84% of that cone's
     bounding box as SOLID; capping the boundary instead signed 0% of it. Two
     ways of being wrong, neither a near miss, both invisible in the result
     except as a shape nobody can explain.
     RAY PARITY DOES NOT CARE. A surface that bounds a region bounds it whether
     or not its triangles agree about who owns an edge, so crossings still count.
     That is what makes this the answer for anything CLOSE to solid rather than
     only for what happens to be watertight. Three directions and a majority: one
     ray that slips through a seam is wrong, three that all do is not worth
     guarding against, and the directions are oblique so none runs along an
     axis-aligned face. */
  const PARITY_DIRS = [
    [0.5773502692, 0.5773502692, 0.5773502692],
    [-0.3128931094, 0.8146393089, 0.4884561897],
    [0.7071067812, -0.2357022604, 0.6666666667],
  ];
  const parityInside = (part, p) => {
    let votes = 0;
    for (const d of PARITY_DIRS) {
      let crossings = 0;
      const o = [p[0], p[1], p[2]];
      /* Advanced past each hit rather than gathered in one traversal: the BVH
         reports the NEAREST hit, and a bounded re-cast is the whole of counting
         them. The bound guards a degenerate mesh; a solid a ray crosses more
         times than this is not one this wrap was going to help with. */
      for (let step = 0; step < 64; step += 1) {
        const hit = bvhIntersect(part.bvh, part.positions, o, d, Infinity);
        const t = hit && (typeof hit === 'number' ? hit : hit.t);
        if (!Number.isFinite(t) || t <= 0) break;
        crossings += 1;
        const eps = Math.max(1e-7, Math.abs(t) * 1e-6);
        o[0] += d[0] * (t + eps); o[1] += d[1] * (t + eps); o[2] += d[2] * (t + eps);
      }
      if (crossings % 2 === 1) votes += 1;
    }
    return votes >= 2;
  };
  const signedFor = (part, p) => {
    const hit = bvhClosestPoint(part.bvh, part.positions, p, { pseudonormals: part.pn });
    if (!hit) return Infinity;
    if (!part.parity) return hit.signedDistance;
    return parityInside(part, p) ? -hit.distance : hit.distance;
  };
  const memberSignedDistances = (p) => parts.map((part) => signedFor(part, p));

  const distanceAt = (p) => {
    let acc = Infinity;
    for (let i = 0; i < parts.length; i += 1) {
      const d = signedFor(parts[i], p);
      acc = i === 0 ? d : smoothMinPoly(acc, d, k);
    }
    return acc - skin;
  };

  /* CENTRAL DIFFERENCES, NOT THE ANALYTIC GRADIENT. Each member's own gradient
     is its outward pseudonormal and is available for free, but the blend's
     chain rule across a fold of N of them is a second formula that has to agree
     with the first one forever. Six evaluations buy one definition of the
     field, and the snap is not the hot loop — the closest-point query is. */
  const gradientAt = (p) => {
    const h = gradientStep;
    const g = [0, 0, 0];
    for (let d = 0; d < 3; d += 1) {
      const a = p.slice(); a[d] += h;
      const b = p.slice(); b[d] -= h;
      g[d] = (distanceAt(a) - distanceAt(b)) / (2 * h);
    }
    return g;
  };

  return {
    memberCount: parts.length,
    /* ⚠ A PARITY-SIGNED MEMBER HAS A VOLUME TOO. This counts the members the
       field can report an INSIDE for, which is the question the refusal asks —
       and reading it as "combinatorially closed" refused every solid whose mesh
       is a side plus caps, which is most of them. */
    closedCount: parts.filter((part) => part.pn.closed || part.parity).length,
    openMembers,
    fuse,
    skin,
    blendRadius: k,
    bounds,
    diagonal,
    padding,
    gradientStep,
    parts,
    distanceAt,
    gradientAt,
    memberSignedDistances,
  };
}

// ---------------------------------------------------------------------------
// DOES IT FUSE
// ---------------------------------------------------------------------------

/**
 * How many connected components the field's interior falls into, by flood fill
 * on a coarse occupancy grid.
 *
 * ⚠ A COUNTING GRID, NEVER A MESHING GRID. Nothing about this lattice reaches
 * the output — it answers one integer question — so the voxel signature that
 * makes grid-based surface extraction unusable here does not arise. What the
 * resolution does decide is the PRECISION of the answer: two members whose
 * occupied cells touch across a gap narrower than one cell read as fused. The
 * resolution is therefore returned with the count rather than hidden.
 */
export function solidsFieldComponents(field, opts = {}) {
  /* ⚠⚠ AN EMPTY GRID IS A RESOLUTION FAILURE BEFORE IT IS AN ANSWER. A member
     thinner than one cell is sampled by nothing at all: a 2x2x160mm rod scored
     ZERO occupied cells in a 160mm box and was refused as enclosing no space,
     while the wrap itself handles it perfectly well (mass 0.99 once past the
     refusal). So a grid that finds nothing is refined and asked again rather
     than believed. It costs nothing in the ordinary case, because the ordinary
     case is not empty; the ceiling is the same 96 the clamp already names. */
  const first = componentsAtResolution(field, opts, Math.max(4, Math.min(96, Math.round(opts.resolution ?? 32))));
  if (first.occupied > 0 || !field.memberCount) return first;
  for (const res of [64, 96]) {
    if (res <= first.resolution) continue;
    const finer = componentsAtResolution(field, opts, res);
    if (finer.occupied > 0) return finer;
  }
  return first;
}

function componentsAtResolution(field, opts, resolution) {
  if (!field.memberCount) return { components: 0, occupied: 0, resolution, cell: 0, dims: [0, 0, 0] };

  const lo = field.bounds.lo.map((v) => v - field.padding);
  const hi = field.bounds.hi.map((v) => v + field.padding);
  const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const longest = Math.max(ext[0], ext[1], ext[2]);
  if (!(longest > 0)) return { components: 0, occupied: 0, resolution, cell: 0, dims: [0, 0, 0] };
  const cell = longest / resolution;
  const dims = ext.map((e) => Math.max(2, Math.ceil(e / cell) + 1));

  const total = dims[0] * dims[1] * dims[2];
  const solid = new Uint8Array(total);
  let occupied = 0;
  const p = [0, 0, 0];
  const lipschitz = field.parts.every((part) => !part.parity);
  /* ⚠ THE SCAN LINE SKIPS WHAT THE DISTANCE ALREADY DECIDED. This grid is the
     single largest cost in a drag — 125ms of 232 at density 8, more than the
     wrap itself — and it was evaluating the field at every one of 20,460 cells.
     The field is 1-Lipschitz (the smooth minimum's two partials sum to 1 and
     both lie in [0,1], and every member's own term is a true distance), so a
     reading of d at one cell fixes the SIGN for floor(|d|/cell - 0.5) cells
     either side of it along any line. Skipping those is then exact rather than
     approximate — the occupancy is identical cell for cell, which the gate
     asserts against the unskipped loop.
     ⚠⚠ AND ONLY WHERE EVERY MEMBER IS PROPERLY SIGNED. A member whose solidity
     is imputed is signed by RAY PARITY, which puts a boolean sign on an unsigned
     distance: near a thin feature the sign flips faster than the magnitude
     changes and the field is not Lipschitz at all. Measured on a helix, which is
     the one fixture here signed that way — the skipped grid and the full grid
     disagreed by a cell at every margin tried, and a cell is enough to move the
     handle count. So the skip is switched off for those and they pay the full
     grid, which is the honest trade: the optimisation is worth having only
     while it is exact. */
  for (let i = 0; i < dims[0]; i += 1) {
    p[0] = lo[0] + (i + 0.5) * cell;
    for (let j = 0; j < dims[1]; j += 1) {
      p[1] = lo[1] + (j + 0.5) * cell;
      const row = (i * dims[1] + j) * dims[2];
      for (let l = 0; l < dims[2];) {
        p[2] = lo[2] + (l + 0.5) * cell;
        const d = field.distanceAt(p);
        if (!Number.isFinite(d)) { l += 1; continue; }
        const run = lipschitz ? Math.max(0, Math.floor(Math.abs(d) / cell - 0.5)) : 0;
        const end = Math.min(dims[2], l + 1 + run);
        if (d <= 0) for (let k = l; k < end; k += 1) { solid[row + k] = 1; occupied += 1; }
        l = end;
      }
    }
  }

  // Six-connected flood fill. An eighteen- or twenty-six-connected fill would
  // join two blobs that meet only at a corner, which is a bridge no surface can
  // actually be drawn through.
  const seen = new Uint8Array(total);
  const sizes = [];
  const stack = [];
  for (let start = 0; start < total; start += 1) {
    if (!solid[start] || seen[start]) continue;
    let size = 0;
    seen[start] = 1;
    stack.length = 0;
    stack.push(start);
    while (stack.length) {
      const idx = stack.pop();
      size += 1;
      const l = idx % dims[2];
      const j = ((idx - l) / dims[2]) % dims[1];
      const i = (idx - l - j * dims[2]) / (dims[1] * dims[2]);
      for (let d = 0; d < 6; d += 1) {
        const ni = i + (d === 0 ? -1 : d === 1 ? 1 : 0);
        const nj = j + (d === 2 ? -1 : d === 3 ? 1 : 0);
        const nl = l + (d === 4 ? -1 : d === 5 ? 1 : 0);
        if (ni < 0 || nj < 0 || nl < 0 || ni >= dims[0] || nj >= dims[1] || nl >= dims[2]) continue;
        const n = (ni * dims[1] + nj) * dims[2] + nl;
        if (solid[n] && !seen[n]) { seen[n] = 1; stack.push(n); }
      }
    }
    sizes.push(size);
  }
  sizes.sort((a, b) => b - a);
  /* ⚠⚠ A SPECK IS NOT A PIECE, and counting one as a piece refuses the whole
     operation. A sharp corner or an open rim lands a handful of isolated cells
     the sampling cannot join to the body they belong to: three overlapping
     tetrahedra scored 4,201 cells plus EIGHT single-cell specks and were refused
     as nine separate solids. The floor is relative with a small absolute base,
     so it scales with the scene and still separates what genuinely is apart —
     two spheres 900mm apart score 10 cells each and both survive it. */
  const minCells = Math.max(4, (sizes[0] ?? 0) * 0.005);
  const components = sizes.filter((n) => n >= minCells).length;

  /* ⚠⚠ HOW MANY HANDLES, PER COMPONENT, WITH CAVITIES COUNTED — because the wrap
     cannot keep them and a reader deserves to be told which of their holes is
     about to close. The first version summed the Euler characteristic over EVERY
     occupied cell and subtracted it from the component count, so the
     sub-resolution specks the count deliberately ignores still moved χ, and an
     interior cavity moved it again: a torus beside two small cubes reported ZERO
     handles while plainly having one. χ is a per-component quantity and has to
     be computed that way. For one solid component with c interior cavities,
     handles = 1 + c - χ. It is a report, never a refusal — filling a hole is
     often exactly what somebody wants from a blob. */
  const labels = new Int32Array(total).fill(-1);
  {
    let lab = 0;
    const st = [];
    const mark = new Uint8Array(total);
    for (let start = 0; start < total; start += 1) {
      if (!solid[start] || mark[start]) continue;
      st.length = 0; st.push(start); mark[start] = 1;
      while (st.length) {
        const idx = st.pop();
        labels[idx] = lab;
        const l = idx % dims[2];
        const j2 = ((idx - l) / dims[2]) % dims[1];
        const i2 = (idx - l - j2 * dims[2]) / (dims[1] * dims[2]);
        for (let d = 0; d < 6; d += 1) {
          const ni = i2 + (d === 0 ? -1 : d === 1 ? 1 : 0);
          const nj = j2 + (d === 2 ? -1 : d === 3 ? 1 : 0);
          const nl = l + (d === 4 ? -1 : d === 5 ? 1 : 0);
          if (ni < 0 || nj < 0 || nl < 0 || ni >= dims[0] || nj >= dims[1] || nl >= dims[2]) continue;
          const n = (ni * dims[1] + nj) * dims[2] + nl;
          if (solid[n] && !mark[n]) { mark[n] = 1; st.push(n); }
        }
      }
      lab += 1;
    }
  }
  const compCount = sizes.length;
  const compSize = new Int32Array(Math.max(1, compCount));
  for (let idx = 0; idx < total; idx += 1) if (labels[idx] >= 0) compSize[labels[idx]] += 1;

  // AIR THE BORDER CAN REACH. What it cannot reach is a cavity, and a cavity
  // adds one to χ exactly as a handle subtracts one — so they cancel unless
  // both are counted.
  const air = new Uint8Array(total);
  {
    const st = [];
    const push = (n) => { if (!solid[n] && !air[n]) { air[n] = 1; st.push(n); } };
    for (let i2 = 0; i2 < dims[0]; i2 += 1) for (let j2 = 0; j2 < dims[1]; j2 += 1) {
      push((i2 * dims[1] + j2) * dims[2]); push((i2 * dims[1] + j2) * dims[2] + dims[2] - 1);
    }
    for (let i2 = 0; i2 < dims[0]; i2 += 1) for (let l = 0; l < dims[2]; l += 1) {
      push((i2 * dims[1]) * dims[2] + l); push((i2 * dims[1] + dims[1] - 1) * dims[2] + l);
    }
    for (let j2 = 0; j2 < dims[1]; j2 += 1) for (let l = 0; l < dims[2]; l += 1) {
      push(j2 * dims[2] + l); push(((dims[0] - 1) * dims[1] + j2) * dims[2] + l);
    }
    while (st.length) {
      const idx = st.pop();
      const l = idx % dims[2];
      const j2 = ((idx - l) / dims[2]) % dims[1];
      const i2 = (idx - l - j2 * dims[2]) / (dims[1] * dims[2]);
      for (let d = 0; d < 6; d += 1) {
        const ni = i2 + (d === 0 ? -1 : d === 1 ? 1 : 0);
        const nj = j2 + (d === 2 ? -1 : d === 3 ? 1 : 0);
        const nl = l + (d === 4 ? -1 : d === 5 ? 1 : 0);
        if (ni < 0 || nj < 0 || nl < 0 || ni >= dims[0] || nj >= dims[1] || nl >= dims[2]) continue;
        push((ni * dims[1] + nj) * dims[2] + nl);
      }
    }
  }
  const cavitiesOf = new Int32Array(Math.max(1, compCount));
  {
    const seenAir = new Uint8Array(total);
    const st = [];
    for (let start = 0; start < total; start += 1) {
      if (solid[start] || air[start] || seenAir[start]) continue;
      st.length = 0; st.push(start); seenAir[start] = 1;
      let owner = -1;
      while (st.length) {
        const idx = st.pop();
        const l = idx % dims[2];
        const j2 = ((idx - l) / dims[2]) % dims[1];
        const i2 = (idx - l - j2 * dims[2]) / (dims[1] * dims[2]);
        for (let d = 0; d < 6; d += 1) {
          const ni = i2 + (d === 0 ? -1 : d === 1 ? 1 : 0);
          const nj = j2 + (d === 2 ? -1 : d === 3 ? 1 : 0);
          const nl = l + (d === 4 ? -1 : d === 5 ? 1 : 0);
          if (ni < 0 || nj < 0 || nl < 0 || ni >= dims[0] || nj >= dims[1] || nl >= dims[2]) continue;
          const n = (ni * dims[1] + nj) * dims[2] + nl;
          if (solid[n]) { if (owner < 0) owner = labels[n]; continue; }
          if (!seenAir[n] && !air[n]) { seenAir[n] = 1; st.push(n); }
        }
      }
      if (owner >= 0) cavitiesOf[owner] += 1;
    }
  }

  // χ per component, over the cubical complex its own cells form.
  let handles = 0;
  const perComponent = [];
  for (let c = 0; c < compCount; c += 1) {
    if (compSize[c] < minCells) continue;
    const V = new Set(), E = new Set(), F = new Set();
    let cubes = 0;
    const key = (x, y, z) => ((x * (dims[1] + 2)) + y) * (dims[2] + 2) + z;
    for (let i2 = 0; i2 < dims[0]; i2 += 1) for (let j2 = 0; j2 < dims[1]; j2 += 1) for (let l = 0; l < dims[2]; l += 1) {
      if (labels[(i2 * dims[1] + j2) * dims[2] + l] !== c) continue;
      cubes += 1;
      for (let a = 0; a < 2; a += 1) for (let b = 0; b < 2; b += 1) for (let g = 0; g < 2; g += 1) V.add(key(i2 + a, j2 + b, l + g));
      for (let a = 0; a < 2; a += 1) for (let b = 0; b < 2; b += 1) {
        E.add(`x${key(i2, j2 + a, l + b)}`); E.add(`y${key(i2 + a, j2, l + b)}`); E.add(`z${key(i2 + a, j2 + b, l)}`);
        F.add(`X${key(i2 + a, j2, l)}`); F.add(`Y${key(i2, j2 + a, l)}`); F.add(`Z${key(i2, j2, l + a)}`);
      }
    }
    const chi = V.size - E.size + F.size - cubes;
    const h = Math.max(0, 1 + cavitiesOf[c] - chi);
    perComponent.push({ size: compSize[c], chi, cavities: cavitiesOf[c], handles: h });
    handles += h;
  }
  /* The occupancy rides out with the count because the hole search needs the
     same lattice, and re-sampling it cost 81ms on a default drag. */
  return { components, occupied, resolution, cell, dims, componentSizes: sizes, minCells,
    handles, perComponent, occupancy: solid };
}

/* ⚠⚠⚠ A REFUSAL IS THE WORST OUTCOME, AND ONE WAS ADDED HERE ON ONE FIXTURE.
   Eroding a solid whose inside is INFERRED is genuinely ill-defined — parity
   says whether a point is in, never where the missing surface is, so an uncapped
   cylinder of radius 40 came back at 45.6 at Skin -15 and 47.6 at -30, growing
   as it was eaten. That measurement was right and the conclusion drawn from it
   was not: it refused every negative Skin on every imputed solid, and a
   primitive's display mesh is imputed — a wall plus caps whose rims sample the
   same circle — so a saved document that had been fusing a torus, a box and an
   ellipsoid at Skin -15 stopped opening. It was one synthetic fixture, and a
   torus's mesh is watertight AS A POINT SET where that cylinder genuinely is
   not.
   What survives is the check that measures rather than assumes: the component
   count already refuses the settings that fragment the solid, which is where
   erosion of an imputed inside actually fails (-5, -8 and -10 on that same
   cylinder). Past that the cost is a blob larger than it should be, which a
   reader can see and undo. A volume-agreement threshold was tried as a
   replacement and rejected on measurement too — it would refuse a helix at 2.19
   and an open cone at 0.48, both of which are results somebody may want.
 *
 * The refusals, by name, before anything is built. Returns null when the field
 * can be wrapped.
 *
 * Members further apart than Fuse can bridge cannot share one cage: a single
 * cage stretched over both is a web between them, which is a confident wrong
 * answer rather than a loose one. That is refused rather than fudged.
 */
export function wrapSolidsRefusal(field, opts = {}) {
  if (!field.memberCount || !field.closedCount) {
    return {
      reason: SOLID_WRAP_REFUSAL.NO_VOLUME,
      message: 'nothing in this selection encloses space.',
    };
  }
  /* ⚠⚠ A NEGATIVE SKIN CANNOT EAT INTO A SOLID THAT WAS INFERRED. A member whose
     mesh is not closed is signed by ray parity, and parity answers "inside" from
     crossing counts without ever describing where the missing surface is — so the
     distance reported deep in an open tube is the distance to its WALL, never to
     the cap it does not have. Eroding that field does not shrink the solid, it
     carves along a ridge that is not a boundary: an uncapped cylinder of radius
     40 came back at radius 45.2 at Skin -15 and 47.1 at -30, GROWING as it was
     eaten, and fragmented into three pieces at -5, -8 and -10 in between. The
     positive direction is unaffected, because offsetting outward only ever reads
     the field near the real surface, where parity and distance are both sound. */
  const comp = opts.components || solidsFieldComponents(field, opts);
  /* ⚠ THE SAME EMPTY GRID HAS TWO CAUSES AND THEY WANT OPPOSITE ANSWERS. Nothing
     is occupied either because no member encloses space — which the check above
     already covers — or because a NEGATIVE Skin ate everything that did. Telling
     a reader "nothing in this selection encloses space" while they are looking at
     two solids is a false statement about their selection, and it names nothing
     they can act on; the number they need to move is the one they just moved. */
  if (!comp.occupied) {
    const eaten = (opts.skin ?? 0) < 0;
    return {
      reason: SOLID_WRAP_REFUSAL.NO_VOLUME,
      message: eaten
        ? `Skin is eating more than there is to eat — at ${opts.skin}mm nothing is left to wrap. Raise it toward 0.`
        : 'nothing in this selection encloses space.',
      components: comp,
    };
  }
  if (comp.components > 1) {
    const n = field.memberCount;
    /* ⚠ ONE SOLID CAN SPLIT TOO, and the plural message was written as though it
       could not: a single member pinched in two by a waist or by a negative Skin
       came back as "these 1 solids are further apart than Fuse can bridge",
       which names the wrong cause and asks for something that cannot help. */
    return {
      reason: SOLID_WRAP_REFUSAL.MEMBERS_DO_NOT_FUSE,
      message: n === 1
        ? `this solid falls into ${comp.components} separate pieces at these settings, and one SuperB cannot wrap them as one. ${(opts.skin ?? 0) < 0 ? 'Raise Skin toward 0' : 'Raise Fuse'} until they meet.`
        : `these ${n} solids are further apart than Fuse can bridge; one SuperB cannot wrap them without a web stretched between them. Raise Fuse, or convert them one at a time.`,
      components: comp,
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// THE START BOX
// ---------------------------------------------------------------------------

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function unit(v, fallback) {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 1e-12 ? [v[0] / l, v[1] / l, v[2] / l] : fallback.slice();
}
function anyPerp(t) {
  const ref = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return unit(cross3(t, ref), [0, 1, 0]);
}

// The principal frame of the members' own mesh vertices. The cube-sphere's
// eight valence-3 corners and twelve seam edges have to go SOMEWHERE, and their
// placement is otherwise arbitrary; putting the box on the shape's own axes at
// least makes it a property of the shape rather than of the world grid.
//
// ⚠ THE VERTICES ARE WEIGHTED AS THEY COME, so a finely tessellated region
// pulls the frame toward itself. Area weighting would fix that and is not done
// here; on a display mesh of roughly uniform density it does not bite.
function principalFrame(parts) {
  let n = 0;
  let cx = 0, cy = 0, cz = 0;
  for (const part of parts) {
    const pos = part.positions;
    for (let i = 0; i + 2 < pos.length; i += 3) { cx += pos[i]; cy += pos[i + 1]; cz += pos[i + 2]; n += 1; }
  }
  if (!n) return { origin: [0, 0, 0], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] };
  cx /= n; cy /= n; cz /= n;
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (const part of parts) {
    const pos = part.positions;
    for (let i = 0; i + 2 < pos.length; i += 3) {
      const dx = pos[i] - cx, dy = pos[i + 1] - cy, dz = pos[i + 2] - cz;
      xx += dx * dx; xy += dx * dy; xz += dx * dz;
      yy += dy * dy; yz += dy * dz; zz += dz * dz;
    }
  }
  const eig = jacobiEigenSym3([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]);
  // jacobiEigenSym3 sorts ascending; the widest spread is last.
  const e0 = unit(eig[2].vector, [1, 0, 0]);
  let e1 = unit(eig[1].vector, [0, 1, 0]);
  // Re-orthogonalize rather than trust two eigenvectors of a degenerate
  // spectrum to be perpendicular: a sphere-like covariance has no preferred
  // pair, and a skew frame silently shears the start box.
  e1 = unit(sub3(e1, scale3(e0, dot3(e0, e1))), anyPerp(e0));
  const e2 = cross3(e0, e1);
  return { origin: [cx, cy, cz], axes: [e0, e1, e2] };
}

// ---------------------------------------------------------------------------
// THE WRAP
// ---------------------------------------------------------------------------

// Taubin's lambda|mu pair — "A Signal Processing Approach to Fair Surface
// Design" (SIGGRAPH 1995). One shrinking pass at lambda followed by one
// expanding pass at mu with mu < -lambda leaves the low frequencies where they
// were, so the cage fairs without collapsing toward its own centroid. A plain
// Laplacian here would shrink the wrap away from the field on every pass and
// the snap would spend itself undoing that.
const TAUBIN_LAMBDA = 0.5;
const TAUBIN_MU = -0.53;

function neighborsOf(cage) {
  const topo = buildTopology(cage);
  return cage.vertices.map((_, v) => topo.vertexEdges[v].map((e) => (e.v0 === v ? e.v1 : e.v0)));
}

function laplacianStep(V, neighbors, w) {
  if (!(w !== 0)) return V;
  const out = new Array(V.length);
  for (let i = 0; i < V.length; i += 1) {
    const nb = neighbors[i];
    const p = V[i];
    if (!nb.length) { out[i] = p.slice(); continue; }
    let ax = 0, ay = 0, az = 0;
    for (const j of nb) { ax += V[j][0]; ay += V[j][1]; az += V[j][2]; }
    ax = ax / nb.length - p[0]; ay = ay / nb.length - p[1]; az = az / nb.length - p[2];
    out[i] = [p[0] + w * ax, p[1] + w * ay, p[2] + w * az];
  }
  return out;
}


/* ⚠⚠ THE START CAGE DECIDES THE TOPOLOGY, AND NOTHING AFTER IT CAN CHANGE THAT.
   The wrap moves vertices; it never cuts or joins, so the result has exactly the
   genus its start cage had. A cube-sphere start therefore fills every through
   hole, which is why a torus fused with a box came back solid through the middle
   and read as a defect rather than as the documented cost it was.
   A TORUS START CAGE COSTS THE SOLVER NOTHING. Measured on a torus of major 100
   and minor 30: hole radius 70.0mm against a true 70, silhouette 0.97, against a
   box cage's pinched 14.0mm and 0.76 — the same passes, the same clamp, the same
   refit. It is also a topologically NICER cage than the box: every vertex is
   valence 4, where the box cage carries eight valence-3 corners.
   ⚠ IT IS ALSO WRONG FOR SOME GENUS-1 SOLIDS, which is why the choice is
   measured rather than inferred. A drilled box is genus 1 and a round tube
   cannot represent its slab, so the torus cage scored 0.53 IoU against the box
   cage's 0.85 on one. Both cages are built and the better is kept. */
function torusStartCage(center, u, v, d, R, r, nu, nv) {
  const vertices = [], faces = [];
  for (let i = 0; i < nu; i += 1) for (let j = 0; j < nv; j += 1) {
    const a = i / nu * 2 * Math.PI, b = j / nv * 2 * Math.PI;
    const rad = R + r * Math.cos(b), h = r * Math.sin(b);
    vertices.push([0, 1, 2].map((k) => center[k] + rad * (Math.cos(a) * u[k] + Math.sin(a) * v[k]) + h * d[k]));
  }
  const at = (i, j) => (((i % nu) + nu) % nu) * nv + (((j % nv) + nv) % nv);
  for (let i = 0; i < nu; i += 1) for (let j = 0; j < nv; j += 1) {
    faces.push([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
  }
  return { vertices, faces };
}

/**
 * Where a through-hole runs, from the occupancy the component count already
 * builds. A hole is a direction you can SEE THROUGH: project the occupied cells
 * down it and the shadow encloses empty space the border cannot flood into.
 *
 * ⚠ THE THREE WORLD AXES ONLY, DELIBERATELY. Searching 67 directions over a
 * hemisphere and taking the largest enclosed shadow is WORSE, not better:
 * an oblique direction can enclose more area by coincidental alignment, and it
 * cost this exact selection — a torus with a box and an ellipsoid — 0.86 IoU
 * against 0.56. A tilted hole is simply not found, its torus cage loses the
 * volume comparison below, and the result is the plain genus-0 wrap. Finding one
 * wants a better objective than area, not more directions.
 */
export function findHoleAxis(field, comp) {
  if (!comp || !comp.handles || !comp.dims || !(comp.cell > 0)) return null;
  const lo = field.bounds.lo.map((x) => x - field.padding);
  const { dims, cell } = comp;
  const solid = comp.occupancy || (() => {
    const g = new Uint8Array(dims[0] * dims[1] * dims[2]);
    const p = [0, 0, 0];
    for (let i = 0; i < dims[0]; i += 1) { p[0] = lo[0] + (i + 0.5) * cell;
      for (let j = 0; j < dims[1]; j += 1) { p[1] = lo[1] + (j + 0.5) * cell;
        for (let l = 0; l < dims[2]; l += 1) { p[2] = lo[2] + (l + 0.5) * cell;
          if (field.distanceAt(p) <= 0) g[(i * dims[1] + j) * dims[2] + l] = 1;
        } } }
    return g;
  })();
  let best = null;
  for (let axis = 0; axis < 3; axis += 1) {
    const [ua, va] = [[1, 2], [0, 2], [0, 1]][axis];
    const nu = dims[ua], nv = dims[va], nw = dims[axis];
    const shadow = new Uint8Array(nu * nv);
    for (let a = 0; a < nu; a += 1) for (let b = 0; b < nv; b += 1) {
      for (let c = 0; c < nw; c += 1) {
        const idx = [0, 0, 0]; idx[ua] = a; idx[va] = b; idx[axis] = c;
        if (solid[(idx[0] * dims[1] + idx[1]) * dims[2] + idx[2]]) { shadow[a * nv + b] = 1; break; }
      }
    }
    const seen = new Uint8Array(nu * nv), stack = [];
    const push = (q) => { if (!shadow[q] && !seen[q]) { seen[q] = 1; stack.push(q); } };
    for (let a = 0; a < nu; a += 1) { push(a * nv); push(a * nv + nv - 1); }
    for (let b = 0; b < nv; b += 1) { push(b); push((nu - 1) * nv + b); }
    while (stack.length) {
      const q = stack.pop(), a = (q / nv) | 0, b = q % nv;
      if (a > 0) push((a - 1) * nv + b);
      if (a < nu - 1) push((a + 1) * nv + b);
      if (b > 0) push(a * nv + b - 1);
      if (b < nv - 1) push(a * nv + b + 1);
    }
    let area = 0, sa = 0, sb = 0;
    for (let q = 0; q < nu * nv; q += 1) if (!shadow[q] && !seen[q]) { area += 1; sa += (q / nv) | 0; sb += q % nv; }
    if (!area || (best && area <= best.areaCells)) continue;
    const cu = lo[ua] + (sa / area + 0.5) * cell, cv = lo[va] + (sb / area + 0.5) * cell;
    let far = 0;
    for (let a = 0; a < nu; a += 1) for (let b = 0; b < nv; b += 1) {
      if (!shadow[a * nv + b]) continue;
      far = Math.max(far, Math.hypot(lo[ua] + (a + 0.5) * cell - cu, lo[va] + (b + 0.5) * cell - cv));
    }
    const inner = Math.sqrt(area / Math.PI) * cell;
    const center = [0, 0, 0];
    center[ua] = cu; center[va] = cv;
    center[axis] = (field.bounds.lo[axis] + field.bounds.hi[axis]) / 2;
    const u = [0, 0, 0], v = [0, 0, 0], d = [0, 0, 0];
    u[ua] = 1; v[va] = 1; d[axis] = 1;
    best = { axis, u, v, dir: d, center, areaCells: area, innerRadius: inner,
      R: (far + inner) / 2, r: (far - inner) / 2 };
  }
  return best;
}

/* The volume a cage's LIMIT surface encloses. Two Catmull-Clark steps is what
   every other measurement here reads, so it is what this reads. */
function limitVolume(cage) {
  let c = { vertices: cage.vertices, faces: cage.faces, creases: {} };
  c = subdivideCatmullClark(subdivideCatmullClark(c));
  let vol = 0;
  for (const f of c.faces) for (const t of triangulateFace(f)) {
    const a = c.vertices[t[0]], b = c.vertices[t[1]], q = c.vertices[t[2]];
    vol += (a[0] * (b[1] * q[2] - b[2] * q[1]) + a[1] * (b[2] * q[0] - b[0] * q[2])
          + a[2] * (b[0] * q[1] - b[1] * q[0])) / 6;
  }
  return Math.abs(vol);
}

/* ⚠ ONE LOOP, TWO CANDIDATE CAGES. The wrap runs unchanged on each start cage
   and the better one is kept, so a start cage that turns out to be the wrong
   shape costs quality and never correctness. Factored out for exactly that
   reason: a second copy of this loop would drift from the first. */
function runWrap(cage, field, opts, fitFrac, facets, featureScale) {
  const neighbors = neighborsOf(cage);

  let meanEdge = 0, edgeCount = 0;
  for (let i = 0; i < cage.vertices.length; i += 1) {
    for (const j of neighbors[i]) {
      if (j <= i) continue;
      const a = cage.vertices[i], b = cage.vertices[j];
      meanEdge += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      edgeCount += 1;
    }
  }
  meanEdge = edgeCount ? meanEdge / edgeCount : field.diagonal || 1;

  /* ⚠⚠ FIT SPENDS PASSES ON SNAPPING OR ON SMOOTHING, and that split is the
     whole control. Conformity is not monotonic in snap passes: on a torus, a box
     and an ellipsoid at density 14 the mean field error over the limit surface
     falls 197, 76, 37, 7.6, 1.7mm across 3, 6, 8, 12, 18 passes and then CLIMBS
     — 1.9, 2.5, 3.3 at 26, 40, 56. Past the optimum the annealed clamp is
     smaller than the distance Taubin's pair moves a vertex, so further passes
     can only smooth. A schedule that simply bought passes with Fit therefore ran
     past the optimum at its own top end: 1.7mm at Fit 25 against 2.5mm at Fit
     100, the control delivering less at maximum than at a quarter. Snapping
     stops at the optimum instead, and the passes Fit does not spend on it are
     spent smoothing — so Fit 0 is a SMOOTH melted lump rather than an
     under-converged lumpy one. Measured on a torus, a box and an ellipsoid at
     density 8, the mean field error over the limit surface runs 31.4, 26.3,
     21.2, 15.0, 5.9mm across Fit 0, 25, 50, 75, 100, and monotonically on a lone
     sphere and a chain of five as well.
     ⚠ THE NEWTON STEP IS NOT DAMPED, which was tried and is worse. At 0.6 the
     sphere refit missed the field by 0.0708 against 0.0004 undamped — the
     instability was never in the step length. And the clamp does not grow with
     Fit: the step is `-s*g/|g|^2`, exact only where |g| = 1, and a smooth
     minimum has |g| < 1 through the blend, so a clamp that rose with Fit made
     the control that means "hold this closer" the one that let it run away
     (furthest cage vertex 21.2mm at Fit 50 to 39.0mm at Fit 100).
     ⚠ THE SMOOTHING IS NOT A TRADE-OFF AGAINST CLOSENESS. Measured the other
     way round: holding it at full strength beat every reduced setting at every
     Fit, and 0.1 at Fit 100 was the worst result in the sweep — 26.6mm against
     5.4mm. Taubin's lambda|mu pair is volume-preserving, so it regularizes the
     cage without pulling it off the field; what it prevents is one overshooting
     vertex staying overshot and dragging its neighbors into a sail. */
  const snapPasses = Math.round(8 + 10 * fitFrac);
  /* ⚠ THE MELT COUNT SCALES WITH THE FACET COUNT, or Fit means something
     different at every Density. Smoothing spreads a vertex over roughly the
     square root of the pass count in EDGES, and an edge is the form divided by
     the facets — so a fixed count melts a coarse cage far more than a fine one.
     Unscaled, Fit 0 gave a mean field error of 57mm at density 4 and 6.4mm at
     density 20 on the same three members, and a chain of five lost a third of
     its mass at the coarse end while the fine end barely moved. Squared, because
     holding the spread at a fixed fraction of the form needs passes to rise with
     the facets squared: the same Fit then lands within a third across the range
     — 35, 30, 26, 24mm at densities 4, 8, 14, 20.
     THE MELT IS ALSO THE CHEAP HALF. A snap pass evaluates the field at every
     cage vertex; a melt pass is two weighted averages over the ring and touches
     no field at all. */
  const meltScale = Math.min(16, (facets / 8) ** 2);
  let melt = Math.round(60 * (1 - fitFrac) * meltScale);
  /* ⚠⚠ AND THE MELT MAY NOT BE ALLOWED TO EAT A THIN FEATURE. Smoothing spreads
     a vertex about sqrt(passes) EDGES, so a melt sized for a solid lump closes a
     tube whose wall is thinner than that reach — a torus cage of major 100 and
     minor 30 lost 45% of its mass at Fit 50 and came back at 0.55 IoU, worse
     than the genus-0 cage it was chosen over. The cap is the cage's own thinnest
     dimension: half of it is the furthest a vertex may travel, which leaves the
     feature standing. It binds only where a feature is thin, so the box cage's
     numbers are unchanged. */
  if (featureScale > 0) {
    let e = 0, n = 0;
    for (let i = 0; i < cage.vertices.length; i += 1) for (const j of neighbors[i]) {
      if (j <= i) continue;
      const a = cage.vertices[i], b = cage.vertices[j];
      e += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); n += 1;
    }
    const edge = n ? e / n : 0;
    if (edge > 0) melt = Math.min(melt, Math.max(0, Math.round((0.5 * featureScale / edge) ** 2)));
  }
  const passes = snapPasses + melt;
  const smoothWeight = 1;
  /* ⚠⚠ THE CLAMP IS A DISTANCE TO TRAVEL, NOT A CAGE SPACING, and tying it to
     the spacing made Density run backwards. The annealed schedule lets a vertex
     cover about eight times the clamp in total, so a clamp of half the mean edge
     gives a dense cage a fraction of the reach a coarse one has — and the start
     box stands off the members by the padding whatever the density is. Measured
     on a lone sphere the mass ratio went 1.01, 1.03, 1.18, 1.60 across densities
     4, 12, 16, 24: raising the resolution INFLATED the wrap, the control that
     means "resolve this better" making it worse. A cone and a cylinder went to
     1.82 the same way and a helix to 5.31. Floored at a tenth of the field's
     diagonal every density converges instead — 1.00, 1.04, 3.01 for those three
     — and the coarse end is untouched, because there the spacing is the larger
     of the two. */
  const stepClamp = Math.max(meanEdge * 0.5, field.diagonal * 0.1);

  let V = cage.vertices.map((v) => v.slice());
  for (let pass = 0; pass < passes; pass += 1) {
    V = laplacianStep(V, neighbors, TAUBIN_LAMBDA * smoothWeight);
    V = laplacianStep(V, neighbors, TAUBIN_MU * smoothWeight);
    // Newton along the field gradient: one step of -(s)/|grad s| in the
    // gradient direction, clamped so a vertex sitting where the field is nearly
    // flat cannot be flung across the model.
    if (pass >= snapPasses) continue;
    for (let i = 0; i < V.length; i += 1) {
      const s = field.distanceAt(V[i]);
      if (!Number.isFinite(s)) continue;
      const g = field.gradientAt(V[i]);
      const g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
      if (!(g2 > 1e-18)) continue;
      let t = -s / g2;
      /* A FIXED DECAY, NOT A FRACTION OF THE PASS COUNT. Annealed as
         `1 - 0.65 * pass/(passes-1)` the schedule stretches when Fit buys more
         passes, so a higher Fit spends longer at a large clamp and comes out
         looser — the control degrading the thing it names. */
      const clampNow = stepClamp * Math.pow(0.92, pass);
      const stepLen = Math.abs(t) * Math.sqrt(g2);
      if (stepLen > clampNow) t *= clampNow / stepLen;
      V[i] = [V[i][0] + g[0] * t, V[i][1] + g[1] * t, V[i][2] + g[2] * t];
    }
  }

  let worstField = 0;
  for (const v of V) {
    const s = Math.abs(field.distanceAt(v));
    if (s > worstField) worstField = s;
  }

  /* THE CAGE IS NOT THE SURFACE. Control points sitting on the field leave the
     LIMIT surface hovering inside them by a fraction of the cage spacing, and
     every number reported about the result would be off by that. The refit
     moves the control points until their limit positions land on the wrapped
     targets instead. */
  const targets = V.map((v) => v.slice());
  let limitError = null;
  let vertices = V;
  if (opts.refit !== false) {
    const refit = refitCageToLimitTargets({ vertices: V, faces: cage.faces, creases: {} }, targets, { passes: opts.refitPasses });
    vertices = refit.vertices.map((v) => v.slice());
    limitError = refit.maxError;
  }

  return { vertices, targets, limitError, worstField, passes, snapPasses, stepClamp, smoothWeight, faces: cage.faces };
}

/**
 * Wrap a selection of solids into one all-quad SuperB cage.
 *
 *   opts.density  2-10. Facet count along the widest axis, so roughly 24 to
 *                 600 quads. Stepped: each value re-wraps from scratch and the
 *                 shape changes with it, not only the resolution.
 *   opts.skin     mm, signed. The isolevel.
 *   opts.fuse     mm. Gaps up to this wide are bridged.
 *   opts.fit      0-100. How hard the wrap chases the field into concavities.
 *                 0 is a rounded bounding box; 100 is the tightest the cage can
 *                 reach.
 *   opts.refit    Correct the cage against its own LIMIT surface. Default true.
 *
 * ⚠ FIT DRIVES BOTH THE PASS COUNT AND THE STEP CLAMP, and which of the two it
 * should really be is not settled by anything measured. Both are named in the
 * result so a caller can see what it got.
 *
 * Returns `{ ok: true, cage, targets, ... }` or `{ ok: false, reason, message }`.
 */
export function wrapSolidsToSuperbCage(members, opts = {}) {
  /* ⚠ THE FIELD MAY BE HANDED IN. Everything below this line reads the field
     RECORD and never the meshes it came from — the two places that touch
     `part.positions` (principalFrame and the local-bounds loop) consume it as a
     point cloud, not as triangles. So a field built from analytic balls, or from
     anything else that can answer a signed distance, drives this whole solver
     unchanged. Without the hook a second field kind would have to copy the wrap. */
  const field = opts.field || makeSolidsField(members, opts);
  const components = solidsFieldComponents(field, opts);
  const refusal = wrapSolidsRefusal(field, { ...opts, components });
  if (refusal) return { ok: false, ...refusal, field, components };

  /* ⚠ THE CEILING IS WHERE THE SOLVE STOPS BEING WORTH WAITING FOR, not a round
     number. Measured on a box and a cone: density 10 gives 378 quads in 289ms,
     14 gives 672 in 354ms, 20 gives 1,326 in 550ms, 28 gives 2,688 in 1.1s and
     40 gives 5,670 in 2.5s. Ten was well inside what the machine can do; 32 is
     roughly 3,500 quads and comfortably under two seconds, which is the point at
     which a control the caller re-runs on every drag has stopped being one.
     The CONTROL's own range is a separate decision and lives with the control —
     see the fuse paramSpec, where the slider stays inside the live band and the
     typed field reaches the rest. */
  const density = Math.max(2, Math.min(32, Math.round(opts.density ?? 4)));
  const fit = Math.max(0, Math.min(100, opts.fit ?? 50));
  const fitFrac = fit / 100;

  const frame = principalFrame(field.parts);
  const [e0, e1, e2] = frame.axes;
  const toLocal = (p) => {
    const r = sub3(p, frame.origin);
    return [dot3(r, e0), dot3(r, e1), dot3(r, e2)];
  };
  const toWorld = (l) => [
    frame.origin[0] + e0[0] * l[0] + e1[0] * l[1] + e2[0] * l[2],
    frame.origin[1] + e0[1] * l[0] + e1[1] * l[1] + e2[1] * l[2],
    frame.origin[2] + e0[2] * l[0] + e1[2] * l[1] + e2[2] * l[2],
  ];

  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  const p = [0, 0, 0];
  for (const part of field.parts) {
    const pos = part.positions;
    for (let i = 0; i + 2 < pos.length; i += 3) {
      p[0] = pos[i]; p[1] = pos[i + 1]; p[2] = pos[i + 2];
      const l = toLocal(p);
      for (let d = 0; d < 3; d += 1) { if (l[d] < lo[d]) lo[d] = l[d]; if (l[d] > hi[d]) hi[d] = l[d]; }
    }
  }
  const pad = field.padding;
  const half = [0, 1, 2].map((d) => Math.max((hi[d] - lo[d]) / 2 + pad, 1e-6));
  const centerLocal = [0, 1, 2].map((d) => (hi[d] + lo[d]) / 2);

  /* ⚠ THE BOX CAGE CARRIES TWO FACET COUNTS, NOT ONE PER AXIS. Its caps are a square
     grid in the first two axes, so x and y must share a count or the caps do
     not weld to the sides. The around-count is therefore taken from the MEAN of
     the two, which keeps faces near-square on a box that is wide in one of them
     and not in the other, and is exactly square when they agree. */
  const cell = Math.max(half[0], half[1], half[2]) * 2 / density;
  const facetFor = (extent) => Math.max(1, Math.min(64, Math.round(extent / cell)));
  const facets = facetFor(half[0] + half[1]);
  const facetsH = facetFor(half[2] * 2);

  const boxCage0 = superbBoxCage(centerLocal, half, facets, facetsH);
  const candidates = [{ topology: 'sphere', featureScale: Math.min(half[0], half[1], half[2]) * 2,
    cage: { vertices: boxCage0.vertices.map(toWorld), faces: boxCage0.faces, creases: {} } }];

  /* ⚠⚠ A THROUGH-HOLE NEEDS A START CAGE THAT HAS ONE. See torusStartCage's own
     header for why this is the only place the genus can be decided. The choice
     between the two is MEASURED, not inferred from the handle count: the cage
     whose limit volume agrees with the occupancy the field already counted wins.
     That comparison is right on every fixture tried, including the two a field
     error comparison gets WRONG — a tilted torus, whose axis this detector does
     not find, produces a badly placed torus cage that hugs the field locally
     while enclosing the wrong volume, and loses. So a missed or mistaken hole
     degrades to the genus-0 wrap that shipped before, never to a worse one. */
  const hole = opts.topology === 'sphere' ? null : findHoleAxis(field, components);
  if (hole && hole.r > 0 && hole.R > hole.r) {
    const around = Math.max(8, Math.min(96, Math.round(facets * 2.4)));
    const through = Math.max(6, Math.min(64, Math.round(facetsH * 1.6)));
    candidates.push({ topology: 'torus', hole, featureScale: hole.r * 2,
      cage: { ...torusStartCage(hole.center, hole.u, hole.v, hole.dir, hole.R, hole.r, around, through), creases: {} } });
  }

  const fieldVolume = components.occupied * components.cell ** 3;
  let chosen = null;
  for (const c of candidates) {
    const out = runWrap(c.cage, field, opts, fitFrac, facets, c.featureScale);
    const vol = limitVolume({ vertices: out.vertices, faces: out.faces });
    const err = fieldVolume > 0 ? Math.abs(vol - fieldVolume) / fieldVolume : Infinity;
    if (opts.topology === c.topology) { chosen = { ...c, out, volumeError: err }; break; }
    if (!chosen || err < chosen.volumeError) chosen = { ...c, out, volumeError: err };
  }
  const cage = chosen.cage;
  const { vertices, targets, limitError, worstField, passes, snapPasses, stepClamp, smoothWeight } = chosen.out;

  const quadCount = cage.faces.length;
  return {
    ok: true,
    cage: { vertices, faces: cage.faces, creases: {} },
    targets,
    field,
    frame,
    facets,
    facetsH,
    quadCount,
    topology: chosen.topology,
    hole: chosen.hole || null,
    volumeError: chosen.volumeError,
    components,
    handles: components.handles,
    allQuads: cage.faces.every((f) => f.length === 4),
    density,
    fit,
    passes,
    snapPasses,
    stepClamp,
    smoothWeight,
    worstFieldError: worstField,
    limitRefitError: limitError,
    openMembers: field.openMembers,
  };
}

/* A FIELD MADE OF BALLS, for the same solver the meshes use.
   A metablob and a fused set of solids are the same question asked of different
   inputs: what surface encloses this field? So this returns the identical record
   `makeSolidsField` does and changes nothing downstream.
   ⚠ `positions` IS A POINT CLOUD HERE, and that is not a shortcut. The only two
   consumers that read it — the principal frame and the start box's local bounds
   — walk it three numbers at a time and never ask which three form a triangle.
   A modest sample of each sphere therefore carries exactly the information they
   need, and the distance function is analytic rather than a BVH lookup: measured
   at 0.21us against 19.0us, which is what makes a ball field live on a drag.
   Balls are [x, y, z, r] with an optional 5th number, that ball's own melt —
   the 3D reading of the same record the flat outline uses. */
const BALL_SAMPLES = 42;
export function makeBallsField(balls, opts = {}) {
  const fuse = Math.max(0, opts.fuse ?? 0);
  const skin = opts.skin ?? 0;
  const parts = [];
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  const good = [];
  balls.forEach((b, idx) => {
    if (!Array.isArray(b) || b.length < 4 || !Number.isFinite(b[0]) || !Number.isFinite(b[1])
      || !Number.isFinite(b[2]) || !(b[3] > 0)) return;
    const r = b[3] + skin;
    if (!(r > 0)) return;
    good.push({ c: [b[0], b[1], b[2]], r, melt: b.length > 4 && b[4] != null && Number.isFinite(b[4]) ? Math.max(0, b[4]) : null, index: idx });
    for (let d = 0; d < 3; d += 1) {
      if (b[d] - r < lo[d]) lo[d] = b[d] - r;
      if (b[d] + r > hi[d]) hi[d] = b[d] + r;
    }
  });
  // A Fibonacci sphere: evenly spread without a pole, and deterministic.
  for (const g of good) {
    const pos = new Float64Array(BALL_SAMPLES * 3);
    for (let i = 0; i < BALL_SAMPLES; i += 1) {
      const y = 1 - (2 * i + 1) / BALL_SAMPLES;
      const rad = Math.sqrt(Math.max(0, 1 - y * y));
      const th = Math.PI * (1 + Math.sqrt(5)) * i;
      pos[i * 3] = g.c[0] + g.r * Math.cos(th) * rad;
      pos[i * 3 + 1] = g.c[1] + g.r * y;
      pos[i * 3 + 2] = g.c[2] + g.r * Math.sin(th) * rad;
    }
    parts.push({
      index: g.index, positions: pos,
      bounds: { lo: [g.c[0] - g.r, g.c[1] - g.r, g.c[2] - g.r], hi: [g.c[0] + g.r, g.c[1] + g.r, g.c[2] + g.r] },
      // A sphere is closed and encloses a volume, so nothing here is signed by
      // ray parity — which is also what lets the scan-line skipping stay on.
      pn: { closed: true }, parity: false, ball: g,
    });
  }
  const finite = parts.length > 0;
  const bounds = finite ? { lo, hi } : { lo: [0, 0, 0], hi: [0, 0, 0] };
  const diagonal = finite ? Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) : 0;
  const padding = Math.max(0, skin) + fuse + (diagonal > 0 ? diagonal * 0.02 : 0);
  const gradientStep = opts.gradientStep ?? Math.max(diagonal * 5e-4, 1e-9);

  /* The same decreasing-melt fold the flat outline uses, for the same reason:
     what two balls do where they meet is governed by the softer of the two, and
     folding softest-first makes each step's k exactly min(b_i, b_j). */
  const order = good.map((g, i) => i).sort((a, b) => {
    const ma = good[a].melt == null ? fuse : Math.min(good[a].melt, fuse);
    const mb = good[b].melt == null ? fuse : Math.min(good[b].melt, fuse);
    return mb - ma || a - b;
  });
  const ks = order.map((i) => fuseBlendRadius(good[i].melt == null ? fuse : Math.min(good[i].melt, fuse)));

  const distanceAt = (p) => {
    let acc = Infinity;
    for (let n = 0; n < order.length; n += 1) {
      const g = good[order[n]];
      const d = Math.hypot(p[0] - g.c[0], p[1] - g.c[1], p[2] - g.c[2]) - g.r;
      acc = n === 0 ? d : smoothMinPoly(acc, d, ks[n]);
    }
    return acc === Infinity ? 1 : acc;
  };
  const gradientAt = (p) => {
    const h = gradientStep;
    return [
      (distanceAt([p[0] + h, p[1], p[2]]) - distanceAt([p[0] - h, p[1], p[2]])) / (2 * h),
      (distanceAt([p[0], p[1] + h, p[2]]) - distanceAt([p[0], p[1] - h, p[2]])) / (2 * h),
      (distanceAt([p[0], p[1], p[2] + h]) - distanceAt([p[0], p[1], p[2] - h])) / (2 * h),
    ];
  };
  const memberSignedDistances = (p) => good.map((g) => Math.hypot(p[0] - g.c[0], p[1] - g.c[1], p[2] - g.c[2]) - g.r);

  return {
    memberCount: parts.length,
    closedCount: parts.length,
    openMembers: [],
    fuse, skin,
    blendRadius: fuseBlendRadius(fuse),
    bounds, diagonal, padding, gradientStep, parts,
    distanceAt, gradientAt, memberSignedDistances,
    kind: 'balls',
  };
}
