// SURFACE NOISE — the
// surface modifier chain's newest member, and FAIR'S CONCEPTUAL OPPOSITE,
// LITERALLY. kernel/fair.mjs's fairControlNet is Laplacian relaxation:
// every INTERIOR control point moves TOWARD its own 4-neighbor average,
// converging (removing information). noiseControlNet moves those SAME
// interior control points AWAY from where they are (injecting
// information). Same operand (the control net), same chain, opposite
// sign — so this deliberately mirrors fair.mjs's own structure/API shape:
// a pure function taking a control net + params and returning a NEW
// control net, displacing INTERIOR points only (boundary rows/columns
// pinned exactly, the same operand Fair relaxes — which also keeps a
// Revolve pole's own coincident boundary row from tearing apart), the
// rational WEIGHT (index 3) never touched — position-only, exactly like
// Fair/Cage/Point-Edits.
//
// KEY DECISIONS (all settled by the doc, implemented here, not re-derived):
//   - DISPLACE CONTROL POINTS, not evaluated points — the output stays a
//     genuine NURBS surface; downstream Loft/Sweep/etc. consume it
//     natively, no refitting. Noise frequency is therefore bounded by
//     control-point density, so `refine` (an optional int) knot-inserts
//     the surface BEFORE displacement (reusing kernel/knots.mjs's own
//     proven insertKnot — no new knot math hand-rolled here) for
//     higher-frequency detail.
//   - AMPLITUDE 0 = INPUT COMPLETELY UNCHANGED, EXACTLY — an early return
//     (matching fair.mjs's own amount=0 → 0-iterations byte-identical
//     passthrough), so amplitude is natively a tween slider between the
//     surface as built and its noisy version.
//   - SEEDED AND FULLY DETERMINISTIC — non-negotiable (the registry
//     recomputes on every upstream edit; unseeded randomness would reroll
//     the model on every touch). A per-control-point integer hash keyed by
//     (seed, i, j) is the whole source of randomness — never Math.random.
//   - CHAINING IS COMPOSITION — because this is pure and derives its
//     normal frames from its ACTUAL input every call (never caching the
//     original geometry's frames), a second Noise stage riding the
//     already-displaced net composes (fBm/octave behavior by hand).

import { insertKnot } from './knots.mjs';
import { surfacePointAndPartials, surfaceClosure } from './surface.mjs';

export const NOISE_STYLES = ['value', 'sine', 'randomWalk'];
export const NOISE_DIRECTIONS = ['normal', 'world-x', 'world-y', 'world-z'];
const NOISE_DEFAULTS = { style: 'value', amplitude: 0, frequency: 1, direction: 'normal', seed: 1, refine: 0 };

// Fill defaults + clamp — analogous to fairParamsFromAmount, but Noise
// carries a small param bag rather than one knob, so this normalizes the
// whole set (an unknown style falls back to 'value', a non-positive
// frequency to 1, seed/refine rounded to real integers).
export function normalizeNoiseParams(params) {
  const p = params || {};
  return {
    style: NOISE_STYLES.includes(p.style) ? p.style : 'value',
    amplitude: Number.isFinite(p.amplitude) ? Math.max(0, p.amplitude) : 0,
    frequency: Number.isFinite(p.frequency) && p.frequency > 0 ? p.frequency : 1,
    direction: NOISE_DIRECTIONS.includes(p.direction) ? p.direction : 'normal',
    seed: Number.isFinite(p.seed) ? Math.round(p.seed) : 1,
    refine: Number.isFinite(p.refine) ? Math.max(0, Math.round(p.refine)) : 0,
    // 0 keeps the displacement at full strength to the boundary, which is what
    // this did before falloff existed — so the default is byte-identical.
    falloff: Number.isFinite(p.falloff) ? Math.max(0, Math.min(1, p.falloff)) : 0,
  };
}

// A standard integer avalanche (Murmur3 finalizer) + FNV-style combine —
// a real, deterministic per-index PRNG. hash01(...) returns a value in
// [0,1) that depends ONLY on its integer arguments, so (seed, i, j) always
// hashes to the same displacement, satisfying the doc's "seeded and fully
// deterministic" non-negotiable. This is the RIGHT tool here (a simple
// multiplicative/xorshift hash), never Math.random.
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
// Bilinear-with-smoothstep VALUE noise sampled at continuous (x, y) — the
// classic "value noise" the doc names: smooth between lattice points,
// deterministic per seed. Frequency (applied by the caller as x=i*freq)
// controls how fast it oscillates across the control grid.
//
// wrapX/wrapY (both optional, default none): the real fix for a CLOSED
// axis's own visible seam crease — a plain lattice hash has no concept
// of "this direction wraps," so two lattice cells that are actually the
// SAME physical neighborhood (one step either side of a closed seam)
// hash to two unrelated values, a visible discontinuity in the noise
// TEXTURE itself even though the surface's own control net never tears.
// Passing a real integer period here makes the underlying lattice
// genuinely PERIODIC (wraps the integer cell index, not just the two
// physical endpoints) — the caller (noiseScalarGrid) is responsible for
// choosing a frequency whose own cell count divides evenly into that
// period, or this wrap is a no-op protecting nothing. Omitted entirely
// (undefined) on an OPEN axis — behaves byte-for-byte as before.
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
// The requested frequency, snapped so the LATTICE closes exactly after
// `period` steps (period = a closed axis's own real control-point count
// minus 1 — row/col 0 and row/col period ARE the same physical point).
// `cells` (a whole number of lattice cells spanning the loop, never 0)
// is what valueNoise2D's own wrap argument needs; `freq` is what the
// caller multiplies the raw index by so x reaches exactly `cells` at
// the seam.
export function quantizeClosedFrequencyLinear(frequency, period) {
  const cells = Math.max(1, Math.round(frequency * period));
  return { freq: cells / period, cells };
}
// Same idea, for `sine`'s own genuinely different (radians-per-step, not
// lattice-cells-per-step) frequency meaning — `i*frequency` must advance
// by a whole multiple of 2*PI over `period` steps to close smoothly,
// the exact same "a periodic ripple needs a whole number of wraps" fact
// already proven and shipped for Wave (`waveFrequencyClosedAxisClamp`).
export function quantizeClosedFrequencySine(frequency, period) {
  const cycles = Math.max(1, Math.round((frequency * period) / (2 * Math.PI)));
  return (cycles * 2 * Math.PI) / period;
}
// A single 1D cumulative hash walk of length n (own salt keeps the U and
// V walks independent even at the same seed) — CLOSED-LOOP-SAFE via the
// standard Brownian-bridge trick when `closed`: subtract a linear ramp so
// the walk's own last value returns exactly to its first, which is what
// actually fixes randomWalk's own seam crease (a joint 2D raster
// accumulator has NO periodicity to exploit — every step depends on
// everything before it in BOTH directions, so it can't be patched after
// the fact the way a smooth function like sine can be re-quantized).
// Left as a plain, unbridged walk when `closed` is false — an open axis
// needs no correction at all.
export function bridgedWalk1D(seed, salt, n, closed) {
  const raw = new Array(n);
  let acc = 0;
  for (let k = 0; k < n; k++) { acc += 2 * hash01(seed, salt, k) - 1; raw[k] = acc; }
  if (!closed || n < 2) return raw;
  const drift = raw[n - 1] - raw[0];
  return raw.map((v, k) => v - (k / (n - 1)) * drift);
}

// The per-control-point signed scalar (~[-1, 1]) each style produces,
// precomputed for the WHOLE grid so randomWalk's raster accumulator is
// coherent (and cheap). Three genuinely distinct styles:
//   value      — smooth pseudo-noise (valueNoise2D).
//   sine       — a periodic tensor sinusoid, seed-phased (deterministic).
//   randomWalk — a raster-order cumulative hash walk, tanh-bounded so it
//                stays in [-1, 1] and never spikes a control point to a
//                degenerate extreme; frequency drives its saturation.
// closedU/closedV (both default false): a real bug — Noise
// (and Wave, fixed separately) visibly creases exactly at a closed
// surface's own seam (a Torus's U or V wrap, a Cylinder's single closed
// sweep direction) — confirmed structurally, not per-shape: every style
// here keys its randomness/phase to the RAW control-point index with no
// concept of wraparound, so the two interior points flanking a physically
// coincident seam (index 1 and index n-2) sample two unrelated points of
// an otherwise-open pattern. Each style gets its own honest fix (there is
// no one shared mechanism, matching Wave's own established precedent of
// "the right tool for THIS function's own shape"), applied per axis
// independently — a Cylinder (closed in exactly one direction) and a
// Torus (closed in both) both fall out of the same two booleans, never a
// shape-specific branch. Neither boolean set: byte-for-byte identical to
// the pre-fix construction, on every axis, in every style.
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
      // byte-identical to the original single joint raster accumulator —
      // the overwhelmingly common open-surface case needs no correction.
      let acc = 0;
      for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) { acc += 2 * hash01(p.seed, i, j, 3) - 1; g[i][j] = Math.tanh(acc * p.frequency * 0.25); }
    } else {
      // a single joint accumulator can't be patched into a closed loop
      // after the fact (see bridgedWalk1D's own comment) — rebuilt as the
      // sum of two INDEPENDENT 1D walks, each bridged shut on whichever
      // axis is actually closed. A genuinely different texture from the
      // fully-open branch above, by real mathematical necessity, not a
      // cosmetic difference — the honest cost of making a raster-style
      // walk safe to wrap at all.
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
// parameter of maximum influence) — a copy of pointedit.mjs's own
// grevilleFromKnots, used only for normal-frame direction so a control
// point's own displacement normal is read at the parameter that control
// point actually governs.
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
// same pole-nudge honest-degrade as pointedit.mjs's own
// surfaceNormalAtParam (a Greville abscissa can land on a degenerate pole
// where the cross product collapses; return null there and the caller
// displaces nothing rather than a NaN).
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

// ---- knot-insertion REFINE (reuses kernel/knots.mjs's insertKnot per
// row/column — no new knot math). Inserting a knot in U treats each column
// as a curve of degU over knotsU; in V, each row as a curve of degV over
// knotsV. `refine` levels each insert the midpoint of every current span,
// roughly doubling the control-point count per direction per level, so a
// higher `refine` genuinely raises the density available to carry noise
// before any displacement runs. Shape-preserving by construction (knot
// insertion never changes geometry), so refine alone (amplitude 0) would
// leave the surface unchanged — which is exactly why amplitude 0 returns
// early ABOVE refine (the whole op is a true no-op then). ----
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

// SELF-INTERSECTION-SAFE AMPLITUDE CLAMP — generalizes kernel/offset.mjs's
// own safeOffsetMagnitude (the adjacent-control-point-crossing check behind
// Offset/Thicken/Shell's own auto-clamp) from a UNIFORM scalar distance +
// per-point-varying NORMAL to an arbitrary per-point-varying DISPLACEMENT
// VECTOR — Wave/Noise's own amplitude*dir at every control point, which
// already bakes in both magnitude and direction at "amplitude=1". The
// underlying math is identical in form (offset.mjs's own dn = nb-na,
// direction-only, generalizes to dv = vb-va, a full displacement-vector
// difference) but genuinely SIMPLER here: Wave/Noise's own amplitude is
// already clamped non-negative (normalizeWaveParams/normalizeNoiseParams
// both floor it at 0), so scaling a FIXED unit displacement field by a
// growing amplitude only ever moves in ONE direction — unlike Offset's own
// signed distance, there is no signDir to track here.
const SELF_INTERSECT_SAFETY = 0.98; // same margin as offset.mjs's own SELF_INTERSECT_SAFETY — stay just off the exact degenerate boundary, never sitting on the zero-area edge itself

// maxSafeDisplacementScale(ctrlNet, unitDisplacement) -> the largest
// amplitude for which no adjacent control-net edge (U or V direction)
// crosses/reverses, or Infinity if nothing constrains it (e.g. every
// unitDisplacement entry is the same vector, or all zero — a flat
// translate or a no-op field can never fold an edge). `unitDisplacement`
// is a same-shape grid of [dx,dy,dz] vectors (an untouched/boundary/pole
// point contributes [0,0,0], the honest no-constraint case).
// FADE TOWARD THE EDGES. A displacement that runs at full strength straight
// into the boundary is unusable wherever the surface has to meet something: it
// pushes the very control rows a Join, a Match Edge or a Bridge depends on, so
// the seam moves and the neighbour no longer fits. With a falloff the detail
// lives in the middle of the panel and the edges stay where they were put.
//
// MEASURED IN THE CONTROL NET, not in model space. The net is what is being
// displaced, and its own index distance to the boundary is exactly "how many
// control points from the edge" — which is the thing that decides whether an
// edge row moves. A model-space distance would need the surface's own metric
// and would behave differently on a stretched net for no gain.
//
// falloff 0 is the IDENTITY, so the default stays byte-identical to before this
// existed; falloff 1 fades to nothing across the two rows that carry
// continuity. Smoothstepped rather than linear, so the fade has no visible
// crease of its own where it reaches full strength.
export function boundaryFalloffGrid(nu, nv, falloff) {
  const f = Math.max(0, Math.min(1, falloff || 0));
  const grid = Array.from({ length: nu }, () => new Array(nv).fill(1));
  if (f === 0 || nu < 3 || nv < 3) return grid;
  // ⚠ IT REACHES ZERO ACROSS THE FIRST TWO ROWS, not just the outermost one,
  // and that is the whole point rather than a detail. Continuity lives in two
  // rows: the boundary row IS the edge curve (G0), and the row behind it sets
  // the cross-boundary tangent (G1). A fade that only spared the outermost row
  // would still tilt every seam it touched, so a matched or joined edge would
  // come apart the moment anyone added noise — which is exactly the failure
  // this exists to prevent.
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
    // 0 only when a < 0 (this edge genuinely shrinks as s grows), at
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

// noiseControlNet(srf, params) — the public surface, mirroring
// fairControlNet's own (srf, params) → new srf shape. Displaces every
// INTERIOR control point away from its position by amplitude·scalar(i,j)
// along the chosen direction (a world axis, or the surface's OWN normal at
// that control point's Greville (u,v)). Boundary rows/columns and every
// rational weight are left exactly untouched.
// opts.weightAt — an OPTIONAL (fu, fv) -> multiplier, where fu/fv are the
// control point's own Greville position as a fraction of the surface's own
// domain. Omitted (the default) this function is BYTE-IDENTICAL to what it
// was, proven by test rather than asserted.
//
// This is "the consumer declares its own sampling" landing on a
// second consumer: Noise reads a field per CONTROL POINT, at Greville
// fractions, because control points are what it displaces — Tessellate
// reads the same field per CELL, at cell fractions, because cells are what
// it emits. Neither convention is baked into the field; each consumer picks
// the one that matches what it operates on.
//
// The weight multiplies the UNIT displacement field BEFORE the
// self-intersection clamp, deliberately. The clamp is a property of
// whatever field it is handed, so weighting first keeps it exactly correct
// rather than needing its own awareness of the weighting: an unweighted
// region contributes a zero displacement, which is already the honest
// "not displaced at all, so no constraint" case it handles.
export function noiseControlNet(srf, params, opts = {}) {
  const weightAt = typeof opts.weightAt === 'function' ? opts.weightAt : null;
  const p = normalizeNoiseParams(params);
  if (p.amplitude === 0) return srf; // EXACT identity — the amplitude=0 tween baseline, byte-for-byte (matches fair's amount=0 passthrough)
  let s = p.refine > 0 ? refineSurface(srf, p.refine) : srf;
  const nu = s.ctrlNet.length, nv = s.ctrlNet[0].length;
  if (nu < 3 || nv < 3) return srf; // no genuine interior control point to displace — honest no-op (return the ORIGINAL, un-refined surface, since a refine with nothing to noise is pointless)
  // CLOSED-AXIS SEAM FIX — a real, structural fact about THIS surface's own
  // net (knot-insertion/refine never changes which rows/columns coincide),
  // not guessed at or shape-specific. See noiseScalarGrid's own header
  // comment for the actual per-style fix.
  const { closedU, closedV } = surfaceClosure(s);
  const grid = noiseScalarGrid(p, nu, nv, closedU, closedV);
  const useNormal = p.direction === 'normal';
  const axis = p.direction === 'world-x' ? [1, 0, 0] : p.direction === 'world-y' ? [0, 1, 0] : p.direction === 'world-z' ? [0, 0, 1] : null;
  // Greville abscissae are needed by the normal direction AND by a weight
  // function (which asks where a control point sits in the domain), so
  // compute them when either wants them.
  const needGreville = useNormal || !!weightAt;
  const gU = needGreville ? grevilleFromKnots(s.knotsU, s.degU, nu) : null;
  const gV = needGreville ? grevilleFromKnots(s.knotsV, s.degV, nv) : null;
  // The surface's own real parameter domain. Knot insertion (refine) does
  // not move it, so a fraction means the same thing before and after a
  // refine — which is what lets a field painted on the original surface
  // still line up with a refined net.
  const uMin = s.knotsU[s.degU], uMax = s.knotsU[s.knotsU.length - 1 - s.degU];
  const vMin = s.knotsV[s.degV], vMax = s.knotsV[s.knotsV.length - 1 - s.degV];
  const uSpan = uMax - uMin, vSpan = vMax - vMin;

  // PHASE 1 — the UNIT (amplitude=1) displacement field, every interior
  // point's own dir*grid vector, boundary/pole points left at [0,0,0] (the
  // honest "not displaced at all" case maxSafeDisplacementScale already
  // treats as no constraint). Cached here so phase 3 doesn't re-derive a
  // pole's own normal a second time.
  const fade = boundaryFalloffGrid(nu, nv, p.falloff);
  const unitDisp = Array.from({ length: nu }, () => new Array(nv).fill(null).map(() => [0, 0, 0]));
  for (let i = 1; i < nu - 1; i++) {
    for (let j = 1; j < nv - 1; j++) {
      let dir = axis;
      if (useNormal) { dir = surfaceNormalAtParam(s, gU[i], gV[j]); if (!dir) continue; } // a true pole — no defined normal, displace nothing rather than a NaN
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

  // PHASE 2 — the SELF-INTERSECTION-SAFE CLAMP (generalizes Offset/Thicken/
  // Shell's own auto-clamp-with-honest-status convention to this per-point-
  // varying displacement field). safeMax is the largest amplitude for which
  // no adjacent control-net edge crosses; requesting more auto-clamps down
  // to it rather than silently folding the surface.
  const safeMax = maxSafeDisplacementScale(s.ctrlNet, unitDisp);
  const appliedAmplitude = Math.min(p.amplitude, safeMax);

  // PHASE 3 — apply the (possibly clamped) amplitude to the cached unit field.
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
