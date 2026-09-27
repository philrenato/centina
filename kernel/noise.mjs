// Surface noise: the opposite of Fair. kernel/fair.mjs's fairControlNet is
// Laplacian relaxation: every interior control point moves toward its own
// 4-neighbor average, removing information. noiseControlNet moves those same
// interior control points away from where they are, injecting information.
// Same operand (the control net), opposite sign, so this mirrors fair.mjs's
// structure and API: a pure function taking a control net + params and
// returning a new control net, displacing interior points only (boundary
// rows/columns pinned exactly, which also keeps a Revolve pole's coincident
// boundary row from tearing apart), the rational weight (index 3) never
// touched — position-only, like Fair, Cage and Point Edits.
//
// Design:
//   - Displace control points, not evaluated points: the output stays a
//     NURBS surface that downstream Loft/Sweep/etc. consume natively, with no
//     refitting. Noise frequency is therefore bounded by control-point
//     density, so `refine` (an optional int) knot-inserts the surface before
//     displacement (via kernel/knots.mjs's insertKnot) for higher-frequency
//     detail.
//   - Amplitude 0 returns the input unchanged, by early return (matching
//     fair.mjs's amount=0 passthrough), so amplitude works as a tween between
//     the surface as built and its noisy version.
//   - Seeded and fully deterministic: the registry recomputes on every
//     upstream edit, and unseeded randomness would reroll the model on every
//     touch. A per-control-point integer hash keyed by (seed, i, j) is the
//     whole source of randomness — never Math.random.
//   - Chaining is composition: this is pure and derives its normal frames
//     from its actual input on every call (never caching the original
//     geometry's frames), so a second Noise stage riding the already-displaced
//     net composes (fBm/octave behavior by hand).

import { insertKnot } from './knots.mjs';
import { surfacePointAndPartials, surfaceClosure } from './surface.mjs';

export const NOISE_STYLES = ['value', 'sine', 'randomWalk'];
export const NOISE_DIRECTIONS = ['normal', 'world-x', 'world-y', 'world-z'];

// Fill defaults + clamp — analogous to fairParamsFromAmount, but Noise
// carries a small param bag rather than one knob, so this normalizes the
// whole set (an unknown style falls back to 'value', a non-positive
// frequency to 1, seed/refine rounded to integers).
export function normalizeNoiseParams(params) {
  const p = params || {};
  return {
    style: NOISE_STYLES.includes(p.style) ? p.style : 'value',
    amplitude: Number.isFinite(p.amplitude) ? Math.max(0, p.amplitude) : 0,
    frequency: Number.isFinite(p.frequency) && p.frequency > 0 ? p.frequency : 1,
    direction: NOISE_DIRECTIONS.includes(p.direction) ? p.direction : 'normal',
    seed: Number.isFinite(p.seed) ? Math.round(p.seed) : 1,
    refine: Number.isFinite(p.refine) ? Math.max(0, Math.round(p.refine)) : 0,
    // 0 keeps the displacement at full strength to the boundary.
    falloff: Number.isFinite(p.falloff) ? Math.max(0, Math.min(1, p.falloff)) : 0,
  };
}

// A standard integer avalanche (the 0x45d9f3b xor-shift-multiply hash),
// seeded from the FNV offset basis: a deterministic per-index PRNG.
// hash01(...) returns a value in [0,1) that depends only on its integer
// arguments, so (seed, i, j) always hashes to the same displacement.
function hashU32(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return (x ^ (x >>> 16)) >>> 0;
}
function hash01(...vals) {
  let h = 0x811c9dc5 >>> 0; // FNV offset basis
  for (const v of vals) h = hashU32(h ^ (Math.imul(v | 0, 0x9e3779b1) >>> 0));
  return (h >>> 0) / 4294967296;
}

// A single lattice hash value in [-1, 1].
function latticeVal(seed, ix, iy) { return 2 * hash01(seed, ix, iy) - 1; }
// Bilinear-with-smoothstep value noise sampled at continuous (x, y): smooth
// between lattice points, deterministic per seed. Frequency (applied by the
// caller as x=i*freq) controls how fast it oscillates across the control grid.
//
// wrapX/wrapY (both optional, default none) remove the seam crease on a
// closed axis. A plain lattice hash has no concept of "this direction
// wraps", so two lattice cells that are the same physical neighborhood (one
// step either side of a closed seam) hash to two unrelated values, a visible
// discontinuity in the noise texture even though the control net never
// tears. An integer period here makes the lattice periodic (it wraps the
// integer cell index, not just the two physical endpoints). The caller
// (noiseScalarGrid) must choose a frequency whose cell count divides evenly
// into that period, or the wrap has no effect. Omitted (undefined) on an
// open axis.
export function valueNoise2D(x, y, seed, wrapX, wrapY) {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = x - x0, fy = y - y0;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const wrap = (v, period) => (period ? ((v % period) + period) % period : v);
  const x0w = wrap(x0, wrapX), x1w = wrap(x0 + 1, wrapX);
  const y0w = wrap(y0, wrapY), y1w = wrap(y0 + 1, wrapY);
  const v00 = latticeVal(seed, x0w, y0w);
  const v10 = latticeVal(seed, x1w, y0w);
  const v01 = latticeVal(seed, x0w, y1w);
  const v11 = latticeVal(seed, x1w, y1w);
  const a = v00 + (v10 - v00) * sx;
  const b = v01 + (v11 - v01) * sx;
  return a + (b - a) * sy;
}
// The requested frequency, snapped so the lattice closes exactly after
// `period` steps (period = a closed axis's control-point count minus 1 —
// row/col 0 and row/col period are the same physical point).
// `cells` (a whole number of lattice cells spanning the loop, never 0)
// is what valueNoise2D's own wrap argument needs; `freq` is what the
// caller multiplies the raw index by so x reaches exactly `cells` at
// the seam.
export function quantizeClosedFrequencyLinear(frequency, period) {
  const cells = Math.max(1, Math.round(frequency * period));
  return { freq: cells / period, cells };
}
// Same idea, for `sine`'s different (radians-per-step, not
// lattice-cells-per-step) frequency meaning: `i*frequency` must advance by a
// whole multiple of 2*PI over `period` steps to close smoothly, as for Wave
// (`waveFrequencyClosedAxisClamp`).
export function quantizeClosedFrequencySine(frequency, period) {
  const cycles = Math.max(1, Math.round((frequency * period) / (2 * Math.PI)));
  return (cycles * 2 * Math.PI) / period;
}
// A single 1D cumulative hash walk of length n (the salt keeps the U and V
// walks independent at the same seed). When `closed`, a Brownian bridge
// subtracts a linear ramp so the walk's last value returns exactly to its
// first, which removes randomWalk's seam crease. A joint 2D raster
// accumulator has no periodicity to exploit — every step depends on
// everything before it in both directions — so it cannot be re-quantized
// the way a smooth function like sine can. Left as a plain, unbridged walk
// when `closed` is false.
export function bridgedWalk1D(seed, salt, n, closed) {
  const raw = new Array(n);
  let acc = 0;
  for (let k = 0; k < n; k++) { acc += 2 * hash01(seed, salt, k) - 1; raw[k] = acc; }
  if (!closed || n < 2) return raw;
  const drift = raw[n - 1] - raw[0];
  return raw.map((v, k) => v - (k / (n - 1)) * drift);
}

// The per-control-point signed scalar (~[-1, 1]) each style produces,
// precomputed for the whole grid so randomWalk's raster accumulator is
// coherent (and cheap). Three distinct styles:
//   value      — smooth pseudo-noise (valueNoise2D).
//   sine       — a periodic tensor sinusoid, seed-phased (deterministic).
//   randomWalk — a raster-order cumulative hash walk, tanh-bounded so it
//                stays in [-1, 1] and never spikes a control point to a
//                degenerate extreme; frequency drives its saturation.
// closedU/closedV (both default false) make the pattern continuous across a
// closed surface's seam (a Torus's U or V wrap, a Cylinder's single closed
// sweep direction). Keyed to the raw control-point index with no concept of
// wraparound, the two interior points flanking a coincident seam (index 1
// and index n-2) would sample two unrelated points of an open pattern and
// crease there. Each style has its own fix (as for Wave, there is no one
// shared mechanism), applied per axis independently, so a Cylinder (closed
// in one direction) and a Torus (closed in both) fall out of the same two
// booleans, never a shape-specific branch. With neither set, every style
// produces the open construction.
function noiseScalarGrid(p, nu, nv, closedU, closedV) {
  const g = Array.from({ length: nu }, () => new Array(nv).fill(0));
  if (p.style === 'sine') {
    const phaseA = hash01(p.seed, 11) * Math.PI * 2;
    const phaseB = hash01(p.seed, 22) * Math.PI * 2;
    const freqI = closedU ? quantizeClosedFrequencySine(p.frequency, nu - 1) : p.frequency;
    const freqJ = closedV ? quantizeClosedFrequencySine(p.frequency, nv - 1) : p.frequency;
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) g[i][j] = Math.sin(i * freqI + phaseA) * Math.cos(j * freqJ + phaseB);
  } else if (p.style === 'randomWalk') {
    if (!closedU && !closedV) {
      // A single joint raster accumulator: an open surface needs no
      // correction.
      let acc = 0;
      for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) { acc += 2 * hash01(p.seed, i, j, 3) - 1; g[i][j] = Math.tanh(acc * p.frequency * 0.25); }
    } else {
      // A single joint accumulator cannot be closed after the fact (see
      // bridgedWalk1D), so this is the sum of two independent 1D walks, each
      // bridged shut on whichever axis is closed. The texture differs from
      // the open branch above; that is the cost of making a raster-style
      // walk wrap.
      const walkU = bridgedWalk1D(p.seed, 101, nu, closedU);
      const walkV = bridgedWalk1D(p.seed, 202, nv, closedV);
      for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) g[i][j] = Math.tanh((walkU[i] + walkV[j]) * p.frequency * 0.25);
    }
  } else { // 'value'
    const wu = closedU ? quantizeClosedFrequencyLinear(p.frequency, nu - 1) : null;
    const wv = closedV ? quantizeClosedFrequencyLinear(p.frequency, nv - 1) : null;
    const freqI = wu ? wu.freq : p.frequency, freqJ = wv ? wv.freq : p.frequency;
    const wrapX = wu ? wu.cells : undefined, wrapY = wv ? wv.cells : undefined;
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) g[i][j] = valueNoise2D(i * freqI, j * freqJ, p.seed, wrapX, wrapY);
  }
  return g;
}

// The Greville abscissa of each control point in one direction (its
// parameter of maximum influence) — a copy of pointedit.mjs's
// grevilleFromKnots, used only for the normal frame, so a control point's
// displacement normal is read at the parameter that control point governs.
export function grevilleFromKnots(knots, p, count) {
  const g = [];
  for (let i = 0; i < count; i++) {
    if (p <= 0) { g.push(knots[i] ?? 0); continue; }
    let s = 0;
    for (let k = i + 1; k <= i + p; k++) s += knots[k];
    g.push(s / p);
  }
  return g;
}
// The surface's local unit normal at (u,v) = normalize(Su × Sv), with the
// same pole nudge as pointedit.mjs's surfaceNormalAtParam (a Greville
// abscissa can land on a degenerate pole where the cross product collapses;
// return null there and the caller displaces nothing rather than a NaN).
export function surfaceNormalAtParam(srf, u, v) {
  const uMin = srf.knotsU[srf.degU], uMax = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const tries = [
    [u, v],
    [u + (u < (uMin + uMax) / 2 ? 1 : -1) * (uMax - uMin) * 1e-4, v + (v < (vMin + vMax) / 2 ? 1 : -1) * (vMax - vMin) * 1e-4],
  ];
  for (const [uu, vv] of tries) {
    const cu = Math.max(uMin, Math.min(uMax, uu)), cv = Math.max(vMin, Math.min(vMax, vv));
    const { su, sv } = surfacePointAndPartials(srf, cu, cv);
    const nx = su[1] * sv[2] - su[2] * sv[1];
    const ny = su[2] * sv[0] - su[0] * sv[2];
    const nz = su[0] * sv[1] - su[1] * sv[0];
    const len = Math.hypot(nx, ny, nz);
    if (len > 1e-9) return [nx / len, ny / len, nz / len];
  }
  return null;
}

// Knot-insertion refine, via kernel/knots.mjs's insertKnot per row/column.
// Inserting a knot in U treats each column as a curve of degU over knotsU;
// in V, each row as a curve of degV over knotsV. `refine` levels each insert
// the midpoint of every current span, roughly doubling the control-point
// count per direction per level, so a higher `refine` raises the density
// available to carry noise. Knot insertion never changes geometry, so
// refine alone (amplitude 0) would leave the surface unchanged; amplitude 0
// therefore returns early, before refine.
function spanMidpoints(knots, degree) {
  const uMin = knots[degree], uMax = knots[knots.length - 1 - degree];
  const distinct = [];
  for (const k of knots) {
    if (k < uMin - 1e-12 || k > uMax + 1e-12) continue;
    if (!distinct.some((v) => Math.abs(v - k) < 1e-9)) distinct.push(k);
  }
  distinct.sort((a, b) => a - b);
  const mids = [];
  for (let i = 0; i < distinct.length - 1; i++) mids.push((distinct[i] + distinct[i + 1]) / 2);
  return mids;
}
function insertKnotU(srf, u) {
  const nv = srf.ctrlNet[0].length;
  let newKnotsU = srf.knotsU;
  const cols = [];
  for (let j = 0; j < nv; j++) {
    const col = srf.ctrlNet.map((row) => row[j].slice());
    const crv = insertKnot({ degree: srf.degU, knots: srf.knotsU, ctrlPts: col }, u, 1);
    newKnotsU = crv.knots;
    cols.push(crv.ctrlPts);
  }
  const newNu = cols[0].length;
  const ctrlNet = [];
  for (let i = 0; i < newNu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) row.push(cols[j][i]);
    ctrlNet.push(row);
  }
  return { ...srf, knotsU: newKnotsU, ctrlNet };
}
function insertKnotV(srf, v) {
  const nu = srf.ctrlNet.length;
  let newKnotsV = srf.knotsV;
  const rows = [];
  for (let i = 0; i < nu; i++) {
    const rowPts = srf.ctrlNet[i].map((cp) => cp.slice());
    const crv = insertKnot({ degree: srf.degV, knots: srf.knotsV, ctrlPts: rowPts }, v, 1);
    newKnotsV = crv.knots;
    rows.push(crv.ctrlPts);
  }
  return { ...srf, knotsV: newKnotsV, ctrlNet: rows };
}
export function refineSurface(srf, levels) {
  let s = srf;
  for (let l = 0; l < levels; l++) {
    for (const u of spanMidpoints(s.knotsU, s.degU)) s = insertKnotU(s, u);
    for (const v of spanMidpoints(s.knotsV, s.degV)) s = insertKnotV(s, v);
  }
  return s;
}

// Self-intersection-safe amplitude clamp. Generalizes kernel/offset.mjs's
// safeOffsetMagnitude (the adjacent-control-point-crossing check behind
// Offset/Thicken/Shell's auto-clamp) from a uniform scalar distance along a
// per-point normal to an arbitrary per-point displacement vector — Wave's and
// Noise's amplitude*dir at every control point, which bakes in both
// magnitude and direction at amplitude=1. The math has the same form
// (offset.mjs's dn = nb-na generalizes to dv = vb-va, a full
// displacement-vector difference) but is simpler: Wave's and Noise's
// amplitude is clamped non-negative (normalizeWaveParams/normalizeNoiseParams
// floor it at 0), so scaling a fixed unit displacement field by a growing
// amplitude only ever moves in one direction, and there is no signDir to
// track.
const SELF_INTERSECT_SAFETY = 0.98; // same margin as offset.mjs's SELF_INTERSECT_SAFETY: stay just off the degenerate boundary, never on the zero-area edge itself

// Fade toward the edges. A displacement that runs at full strength straight
// into the boundary is unusable wherever the surface has to meet something: it
// pushes the very control rows a Join, a Match Edge or a Bridge depends on, so
// the seam moves off its neighbor. With a falloff the detail
// lives in the middle of the panel and the edges stay where they were put.
//
// Measured in the control net, not in model space. The net is what is being
// displaced, and its own index distance to the boundary is exactly "how many
// control points from the edge" — which is the thing that decides whether an
// edge row moves. A model-space distance would need the surface's own metric
// and would behave differently on a stretched net for no gain.
//
// falloff 0 is the identity; falloff 1 fades to nothing across the two rows
// that carry continuity. Smoothstepped rather than linear, so the fade has no
// visible crease of its own where it reaches full strength.
export function boundaryFalloffGrid(nu, nv, falloff) {
  const f = Math.max(0, Math.min(1, falloff || 0));
  const grid = Array.from({ length: nu }, () => new Array(nv).fill(1));
  if (f === 0 || nu < 3 || nv < 3) return grid;
  // The weight reaches zero across the first two rows, not just the outermost
  // one. Continuity lives in two rows: the boundary row is the edge curve
  // (G0), and the row behind it sets the cross-boundary tangent (G1). A fade
  // that only spared the outermost row would still tilt every seam it
  // touched, so a matched or joined edge would come apart under noise.
  const reach = (n) => Math.max(1, (n - 1) / 2 - 1);
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const du = (Math.min(i, nu - 1 - i) - 1) / reach(nu);
      const dv = (Math.min(j, nv - 1 - j) - 1) / reach(nv);
      const t = Math.max(0, Math.min(1, Math.min(du, dv)));
      const smooth = t * t * (3 - 2 * t);
      grid[i][j] = 1 - f + f * smooth;
    }
  }
  return grid;
}

// maxSafeDisplacementScale(ctrlNet, unitDisplacement) -> the largest
// amplitude for which no adjacent control-net edge (U or V direction)
// crosses/reverses, or Infinity if nothing constrains it (e.g. every
// unitDisplacement entry is the same vector, or all zero — a flat
// translate or a no-op field can never fold an edge). `unitDisplacement`
// is a same-shape grid of [dx,dy,dz] vectors (an untouched/boundary/pole
// point contributes [0,0,0], which imposes no constraint).
export function maxSafeDisplacementScale(ctrlNet, unitDisplacement) {
  const nu = ctrlNet.length, nv = ctrlNet[0].length;
  let maxScale = Infinity;
  const consider = (ai, aj, bi, bj) => {
    const Pa = ctrlNet[ai][aj], Pb = ctrlNet[bi][bj];
    const va = unitDisplacement[ai][aj], vb = unitDisplacement[bi][bj];
    const ex = Pb[0] - Pa[0], ey = Pb[1] - Pa[1], ez = Pb[2] - Pa[2];
    const dvx = vb[0] - va[0], dvy = vb[1] - va[1], dvz = vb[2] - va[2];
    const e2 = ex * ex + ey * ey + ez * ez;
    if (e2 < 1e-18) return; // coincident control points (a pole edge) — nothing to cross
    const a = dvx * ex + dvy * ey + dvz * ez;
    // e'.e = e2 + s*a, s>=0 always (amplitude never goes negative). Reaches
    // 0 only when a < 0 (this edge shrinks as s grows), at
    // s = e2/|a|; a >= 0 means this edge never shrinks, no constraint.
    if (a < -1e-15) {
      const crit = e2 / Math.abs(a);
      if (crit < maxScale) maxScale = crit;
    }
  };
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    if (i + 1 < nu) consider(i, j, i + 1, j);
    if (j + 1 < nv) consider(i, j, i, j + 1);
  }
  return maxScale === Infinity ? Infinity : maxScale * SELF_INTERSECT_SAFETY;
}

// noiseControlNet(srf, params) — the public entry, mirroring fairControlNet's
// (srf, params) → new srf shape. Displaces every interior control point away
// from its position by amplitude·scalar(i,j) along the chosen direction (a
// world axis, or the surface's own normal at that control point's Greville
// (u,v)). Boundary rows/columns and every rational weight are left exactly
// untouched.
// opts.weightAt — an optional (fu, fv) -> multiplier, where fu/fv are the
// control point's Greville position as a fraction of the surface's domain.
// Omitted, it has no effect.
//
// Each consumer of a field declares its own sampling: Noise reads a field
// per control point, at Greville fractions, because control points are what
// it displaces; Tessellate reads the same field per cell, at cell fractions,
// because cells are what it emits.
//
// The weight multiplies the unit displacement field before the
// self-intersection clamp. The clamp is a property of whatever field it is
// handed, so weighting first keeps it exact without its own awareness of the
// weighting: an unweighted region contributes a zero displacement, which the
// clamp already treats as no constraint.
export function noiseControlNet(srf, params, opts = {}) {
  const weightAt = typeof opts.weightAt === 'function' ? opts.weightAt : null;
  const p = normalizeNoiseParams(params);
  if (p.amplitude === 0) return srf; // exact identity (matches fair's amount=0 passthrough)
  let s = p.refine > 0 ? refineSurface(srf, p.refine) : srf;
  const nu = s.ctrlNet.length, nv = s.ctrlNet[0].length;
  if (nu < 3 || nv < 3) return srf; // no interior control point to displace: return the original, unrefined surface
  // Closure is a structural fact about this surface's net (knot insertion
  // never changes which rows/columns coincide), not shape-specific. See
  // noiseScalarGrid for the per-style handling.
  const { closedU, closedV } = surfaceClosure(s);
  const grid = noiseScalarGrid(p, nu, nv, closedU, closedV);
  const useNormal = p.direction === 'normal';
  const axis = p.direction === 'world-x' ? [1, 0, 0] : p.direction === 'world-y' ? [0, 1, 0] : p.direction === 'world-z' ? [0, 0, 1] : null;
  // Greville abscissae are needed by the normal direction and by a weight
  // function (which asks where a control point sits in the domain), so
  // compute them when either wants them.
  const needGreville = useNormal || !!weightAt;
  const gU = needGreville ? grevilleFromKnots(s.knotsU, s.degU, nu) : null;
  const gV = needGreville ? grevilleFromKnots(s.knotsV, s.degV, nv) : null;
  // The surface's parameter domain. Knot insertion (refine) does
  // not move it, so a fraction means the same thing before and after a
  // refine — which is what lets a field painted on the original surface
  // still line up with a refined net.
  const uMin = s.knotsU[s.degU], uMax = s.knotsU[s.knotsU.length - 1 - s.degU];
  const vMin = s.knotsV[s.degV], vMax = s.knotsV[s.knotsV.length - 1 - s.degV];
  const uSpan = uMax - uMin, vSpan = vMax - vMin;

  // Phase 1: the unit (amplitude=1) displacement field, every interior
  // point's dir*grid vector, boundary/pole points left at [0,0,0] (which
  // maxSafeDisplacementScale treats as no constraint). Cached so phase 3
  // does not re-derive the normals.
  const fade = boundaryFalloffGrid(nu, nv, p.falloff);
  const unitDisp = Array.from({ length: nu }, () => new Array(nv).fill(null).map(() => [0, 0, 0]));
  for (let i = 1; i < nu - 1; i++) {
    for (let j = 1; j < nv - 1; j++) {
      let dir = axis;
      if (useNormal) { dir = surfaceNormalAtParam(s, gU[i], gV[j]); if (!dir) continue; } // a pole: no defined normal, displace nothing rather than a NaN
      let g = grid[i][j];
      if (weightAt) {
        const fu = uSpan > 0 ? (gU[i] - uMin) / uSpan : 0;
        const fv = vSpan > 0 ? (gV[j] - vMin) / vSpan : 0;
        const w = weightAt(fu, fv);
        g *= Number.isFinite(w) ? w : 0;
      }
      g *= fade[i][j];
      unitDisp[i][j] = [dir[0] * g, dir[1] * g, dir[2] * g];
    }
  }

  // Phase 2: the self-intersection-safe clamp (Offset/Thicken/Shell's
  // auto-clamp-with-status convention, for a per-point displacement field).
  // safeMax is the largest amplitude for which no adjacent control-net edge
  // crosses; a larger request is clamped to it and reported in `ampClamp`.
  const safeMax = maxSafeDisplacementScale(s.ctrlNet, unitDisp);
  const appliedAmplitude = Math.min(p.amplitude, safeMax);

  // Phase 3: apply the (possibly clamped) amplitude to the cached unit field.
  const net = s.ctrlNet.map((row) => row.map((cp) => [...cp]));
  for (let i = 1; i < nu - 1; i++) {
    for (let j = 1; j < nv - 1; j++) {
      const d = unitDisp[i][j];
      net[i][j][0] += d[0] * appliedAmplitude;
      net[i][j][1] += d[1] * appliedAmplitude;
      net[i][j][2] += d[2] * appliedAmplitude;
    }
  }
  return {
    ...s,
    ctrlNet: net,
    ampClamp: { requested: p.amplitude, applied: appliedAmplitude, safeMax, clamped: appliedAmplitude < p.amplitude },
  };
}
