// Light deformer — a directional "light" that carves a quad cage the way a
// hard raking light carves a surface: the side facing the light barely moves,
// the side in shadow is displaced. Pure math over a plain quad cage
// (flat positions + flat Uint32Array quads, stride 4), the same operand shape
// puff.mjs and the subd primitives use. No rendering, no scene graph — the
// "light" is only a direction and a shade curve turned into a displacement.
//
// Pipeline per vertex:
//   n_i  smoothed unit vertex normal (area-weighted face normals, then one
//        neighbor-averaging pass — raw normals collapse creases toward the
//        facet planes, so the smoothing pass is what makes the shade read as
//        a rounded terminator rather than a faceted one)
//   d_i  = dot(n_i, L)                       cosine of the light angle
//   w    = 1 - 0.9*(hardness/100)            terminator half-width (penumbra)
//   s_i  = smoothstep(0, 1, 0.5 - d_i/(2w))  0 fully lit .. 1 full shadow
//   push:  disp_i = magnitude*(ambient + (1-ambient)*s_i)   outward-only bulge
//   erode: disp_i = magnitude*(s_i - 0.5)*2                 lit in, shadow out
// The displacement runs along n_i ('normal') or along -L ('light'). A fold
// guard caps any inward move so it cannot cross more than SAFE_FRACTION of the
// distance to the vertex's nearest neighbor, which is what stops a strong
// deform from folding a facet through its neighbors.

export const SAFE_FRACTION = 0.5;

// Non-throwing vector helper: a zero normal must survive.
function vlen(x, y, z) { return Math.sqrt(x * x + y * y + z * z); }

// GLSL-style smoothstep: clamp to [edge0, edge1], then the Hermite 3t^2-2t^3.
// Zero derivative at both ends, so the terminator has no visible seam.
export function smoothstep(edge0, edge1, x) {
  const span = edge1 - edge0;
  let t = span === 0 ? (x < edge0 ? 0 : 1) : (x - edge0) / span;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}

// Resolve a light direction into a unit object-frame vector.
//  · a 3-vector [x,y,z] is simply normalized.
//  · a 2D disc dot {x,y} with |(x,y)|<=1 maps to a hemisphere: the depth
//    lz = sqrt(1 - x^2 - y^2) is floored at 0.15 so a rim dot still has a
//    forward component, then the whole vector is normalized — the floor can
//    push |(x,y,lz)| past 1 at the rim, and a light direction must stay unit.
export function resolveLightDir(lightDir) {
  if (Array.isArray(lightDir) && lightDir.length >= 3) {
    const [x, y, z] = lightDir;
    if (![x, y, z].every(Number.isFinite)) throw new Error('resolveLightDir: 3-vector has a non-finite component');
    const l = vlen(x, y, z);
    if (l < 1e-12) throw new Error('resolveLightDir: zero-length light direction');
    return [x / l, y / l, z / l];
  }
  if (lightDir && Number.isFinite(lightDir.x) && Number.isFinite(lightDir.y)) {
    const x = lightDir.x, y = lightDir.y;
    const r2 = Math.min(1, x * x + y * y);
    const lz = Math.max(Math.sqrt(1 - r2), 0.15);
    const l = vlen(x, y, lz);
    return [x / l, y / l, lz / l];
  }
  throw new Error('resolveLightDir: lightDir must be a 3-vector [x,y,z] or a disc dot {x,y}');
}

// Edge-neighbor sets from the quad list. Every quad edge (0-1,1-2,2-3,3-0)
// links its two endpoints; the result drives both normal smoothing and the
// nearest-neighbor distance the fold guard is measured against.
function buildNeighbors(quadCount, quads) {
  const nbr = [];
  const link = (a, b) => {
    (nbr[a] || (nbr[a] = new Set())).add(b);
    (nbr[b] || (nbr[b] = new Set())).add(a);
  };
  for (let q = 0; q < quads.length; q += 4) {
    const a = quads[q], b = quads[q + 1], c = quads[q + 2], d = quads[q + 3];
    link(a, b); link(b, c); link(c, d); link(d, a);
  }
  return nbr;
}

// Area-weighted smoothed unit vertex normals. Face normals come from Newell's
// method, whose magnitude is proportional to face area, so accumulating the
// un-normalized face normals onto their vertices is the area weighting. The
// optional smoothing pass then averages each unit normal with its edge
// neighbors' and re-normalizes. Returns one [x,y,z] per vertex; a vertex with
// no defined normal (degenerate fan) stays [0,0,0] and so is never displaced.
export function vertexNormals(positions, quads, smooth = true) {
  const V = positions.length / 3;
  const raw = new Array(V);
  for (let i = 0; i < V; i++) raw[i] = [0, 0, 0];

  for (let q = 0; q < quads.length; q += 4) {
    const idx = [quads[q], quads[q + 1], quads[q + 2], quads[q + 3]];
    let nx = 0, ny = 0, nz = 0;
    for (let e = 0; e < 4; e++) {
      const a = idx[e], b = idx[(e + 1) % 4];
      const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
      const bx = positions[b * 3], by = positions[b * 3 + 1], bz = positions[b * 3 + 2];
      nx += (ay - by) * (az + bz);
      ny += (az - bz) * (ax + bx);
      nz += (ax - bx) * (ay + by);
    }
    for (let e = 0; e < 4; e++) {
      const r = raw[idx[e]];
      r[0] += nx; r[1] += ny; r[2] += nz;
    }
  }

  const unit = new Array(V);
  for (let i = 0; i < V; i++) {
    const r = raw[i], l = vlen(r[0], r[1], r[2]);
    unit[i] = l < 1e-12 ? [0, 0, 0] : [r[0] / l, r[1] / l, r[2] / l];
  }
  if (!smooth) return unit;

  const nbr = buildNeighbors(V, quads);
  const out = new Array(V);
  for (let i = 0; i < V; i++) {
    let sx = unit[i][0], sy = unit[i][1], sz = unit[i][2];
    const ns = nbr[i];
    if (ns) for (const j of ns) { sx += unit[j][0]; sy += unit[j][1]; sz += unit[j][2]; }
    const l = vlen(sx, sy, sz);
    out[i] = l < 1e-12 ? [0, 0, 0] : [sx / l, sy / l, sz / l];
  }
  return out;
}

// Distance to each vertex's nearest edge neighbor. The fold guard caps inward
// motion at SAFE_FRACTION of this, so a compressed facet cannot reach a
// neighbor vertex. A vertex with no neighbor gets Infinity (never capped).
function nearestNeighborDist(positions, quads) {
  const V = positions.length / 3;
  const nbr = buildNeighbors(V, quads);
  const out = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    let best = Infinity;
    const ns = nbr[i];
    if (ns) for (const j of ns) {
      const dx = positions[i * 3] - positions[j * 3];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      const d = vlen(dx, dy, dz);
      if (d < best) best = d;
    }
    out[i] = best;
  }
  return out;
}

// The largest raw magnitude this cage can carry before the fold guard would
// have to bite anywhere. The worst inward move any single light produces is one
// whole `magnitude` along a vertex normal (erode fully lit, s=0), and the
// guard caps an inward move at safeFraction * nearest-neighbor distance, so a
// magnitude at or below safeFraction * min(nearest-neighbor) can never fold a
// facet regardless of light direction, hardness or mode. That bound is a
// property of the cage alone, so it is the ceiling for the magnitude
// control: a UI magnitude in [0,1] scaled by this reaches the fold guard
// exactly at 1 and never crosses it. Isolated vertices (no neighbor) do not
// constrain it; a cage with no measurable edge at all returns Infinity.
export function foldGuardCeiling(positions, quads, safeFraction = SAFE_FRACTION) {
  const nn = nearestNeighborDist(positions, quads);
  let min = Infinity;
  for (let i = 0; i < nn.length; i++) if (Number.isFinite(nn[i]) && nn[i] < min) min = nn[i];
  return Number.isFinite(min) ? safeFraction * min : Infinity;
}

function normOpts(opts = {}) {
  const hardness = Number.isFinite(opts.hardness) ? Math.max(0, Math.min(100, opts.hardness)) : 30;
  return {
    magnitude: Number.isFinite(opts.magnitude) ? opts.magnitude : 1,
    hardness,
    mode: opts.mode === 'erode' ? 'erode' : 'push',
    ambient: Number.isFinite(opts.ambient) ? Math.max(0, Math.min(1, opts.ambient)) : 0,
  };
}

// Per-vertex shade s_i in [0,1] (0 lit .. 1 shadow) for a given normal set and
// unit light L. Exposed so the terminator width can be measured directly.
function shadeScalars(normals, L, hardness) {
  const w = 1 - 0.9 * (Math.max(0, Math.min(100, hardness)) / 100); // in [0.1, 1]
  const inv2w = 1 / (2 * w);
  const s = new Float64Array(normals.length);
  for (let i = 0; i < normals.length; i++) {
    const n = normals[i];
    const d = n[0] * L[0] + n[1] * L[1] + n[2] * L[2];
    s[i] = smoothstep(0, 1, 0.5 - d * inv2w);
  }
  return s;
}

// Signed per-vertex displacement scalar (before the fold guard and before the
// direction choice) for one light. 'push' is outward-only; 'erode' is centered.
function dispScalars(normals, L, p) {
  const s = shadeScalars(normals, L, p.hardness);
  const out = new Float64Array(s.length);
  for (let i = 0; i < s.length; i++) {
    out[i] = p.mode === 'erode'
      ? p.magnitude * (s[i] - 0.5) * 2
      : p.magnitude * (p.ambient + (1 - p.ambient) * s[i]);
  }
  return out;
}

// Public shade field: s_i in [0,1] per vertex for opts.lightDir / opts.hardness.
export function shadeField(positions, quads, opts = {}) {
  const normals = vertexNormals(positions, quads, opts.smooth !== false);
  const L = resolveLightDir(opts.lightDir);
  const hardness = Number.isFinite(opts.hardness) ? opts.hardness : 30;
  return shadeScalars(normals, L, hardness);
}

// Public displacement field: the signed scalar disp_i per vertex (pre-guard),
// so a caller can sum or compare fields without applying them.
export function displacementField(positions, quads, opts = {}) {
  const normals = vertexNormals(positions, quads, opts.smooth !== false);
  const L = resolveLightDir(opts.lightDir);
  return dispScalars(normals, L, normOpts(opts));
}

// Apply an already-built per-vertex displacement vector field with the fold
// guard, returning a new positions array of the same kind as the input. The
// guard acts on the inward part only: if the net move points into the surface
// (against n_i) and is longer than SAFE_FRACTION * nearest-neighbor distance,
// the whole vector is scaled back to that cap. Outward moves are never capped.
function applyVectorField(positions, quads, normals, disp, safeFrac) {
  const nn = nearestNeighborDist(positions, quads);
  const out = positions.slice(); // copy preserves Float32Array vs number[]
  for (let i = 0; i < normals.length; i++) {
    let vx = disp[i][0], vy = disp[i][1], vz = disp[i][2];
    const n = normals[i];
    const proj = vx * n[0] + vy * n[1] + vz * n[2];
    if (proj < 0) {
      const len = vlen(vx, vy, vz);
      const cap = safeFrac * nn[i];
      if (len > cap && Number.isFinite(cap)) {
        const k = cap / len;
        vx *= k; vy *= k; vz *= k;
      }
    }
    out[i * 3] += vx; out[i * 3 + 1] += vy; out[i * 3 + 2] += vz;
  }
  return out;
}

// applyLightDeform(positions, quads, opts) — the single-light deformer.
// opts: { lightDir (required), magnitude=1, hardness=30, mode='push',
//         along='normal'|'light', ambient=0, safeFraction=SAFE_FRACTION }
export function applyLightDeform(positions, quads, opts = {}) {
  const p = normOpts(opts);
  const normals = vertexNormals(positions, quads, opts.smooth !== false);
  const L = resolveLightDir(opts.lightDir);
  const along = opts.along === 'light' ? 'light' : 'normal';
  const safeFrac = Number.isFinite(opts.safeFraction) ? opts.safeFraction : SAFE_FRACTION;
  const scalar = dispScalars(normals, L, p);

  // Direction is n_i (bulge along the surface) or -L (drive away from the
  // light). Both are unit, so the scalar is the true move distance.
  const negL = [-L[0], -L[1], -L[2]];
  const disp = new Array(normals.length);
  for (let i = 0; i < normals.length; i++) {
    const dir = along === 'light' ? negL : normals[i];
    const d = scalar[i];
    disp[i] = [dir[0] * d, dir[1] * d, dir[2] * d];
  }
  return applyVectorField(positions, quads, normals, disp, safeFrac);
}

// compositeLightRigs(positions, quads, rigs, opts) — several lights at once.
// The shade of each rig contributes its own signed displacement; the vectors
// are summed per vertex (additive shade) and the sum is applied once through
// the same fold guard. rigs: [{ lightDir, magnitude, hardness, mode, ambient }].
// opts.along ('normal' default | 'light') is shared; under 'light' each rig
// drives along its own -L, so the summed vector can point anywhere.
export function compositeLightRigs(positions, quads, rigs, opts = {}) {
  const normals = vertexNormals(positions, quads, opts.smooth !== false);
  const along = opts.along === 'light' ? 'light' : 'normal';
  const safeFrac = Number.isFinite(opts.safeFraction) ? opts.safeFraction : SAFE_FRACTION;

  const disp = new Array(normals.length);
  for (let i = 0; i < normals.length; i++) disp[i] = [0, 0, 0];

  for (const rig of (rigs || [])) {
    const L = resolveLightDir(rig.lightDir);
    const scalar = dispScalars(normals, L, normOpts(rig));
    const negL = [-L[0], -L[1], -L[2]];
    for (let i = 0; i < normals.length; i++) {
      const dir = along === 'light' ? negL : normals[i];
      const d = scalar[i];
      disp[i][0] += dir[0] * d; disp[i][1] += dir[1] * d; disp[i][2] += dir[2] * d;
    }
  }
  return applyVectorField(positions, quads, normals, disp, safeFrac);
}
