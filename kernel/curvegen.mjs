// CURVE GENERATORS — a family of "no geometric input, build a brand-new
// standalone curve from scratch" routines, kept DELIBERATELY in one module
// because they are the three MODES of a single de-stacked Curve Generator
// node (one node with a TYPE dropdown, not one node per family):
//   1. L-System   — string-rewriting turtle graphics (Koch/Dragon/Plant).
//   2. Lorenz      — the classic Lorenz chaotic attractor (RK4-integrated).
//   3. Random Walk — a seeded, deterministic random-walk path through space.
//
// NOTHING here takes an existing surface/curve as input. In particular the
// "Random Walk / Noise Curve" mode is a COMPLETELY DIFFERENT operation from
// the already-shipped Noise MODIFIER (kernel/noise.mjs, noiseControlNet),
// which DISPLACES the control net of an EXISTING surface. This one builds a
// new curve out of nothing. The user-facing label is deliberately "Random
// Walk Curve" / "Noise Curve", never bare "Noise", to keep that distinction
// unmistakable (documented naming-collision guard).
//
// Each generator produces a plain [x,y,z][] point chain. The two smooth
// families (Lorenz, Random Walk) are meant to be decimated + fed through
// interpolate.mjs's globalCurveInterp into a real NURBS curve; the L-System
// stays a polyline (degree-1 interpolation) so its sharp fractal corners are
// preserved exactly rather than rounded away.

import { normalize, sub, length } from './vec3.mjs';

// ---------------------------------------------------------------------------
// Shared deterministic PRNG. BYTE-IDENTICAL to kernel/noise.mjs's own proven
// hashU32/hash01 (a Murmur3 finalizer + FNV combine) — reused verbatim, NOT
// reinvented, per the standing "one seeded hash, never Math.random" rule.
// A per-index integer hash keyed by (seed, i, component) is the whole source
// of randomness, so a given seed always reproduces the exact same curve.
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

// ===========================================================================
// 1. L-SYSTEM
// ===========================================================================
//
// Standard, well-documented, textbook L-system presets — NOT invented rule
// grammars. Each is verifiable by hand against a known reference shape:
//   - koch   : the Koch-curve-family generator F->F+F--F+F, angle 60. Each
//              iteration replaces every F with 4 F's, so segment count is
//              EXACTLY 4^iterations (an exact, checkable growth law).
//   - dragon : the Heighway dragon curve (Wikipedia's canonical rules),
//              axiom FX, X->X+YF+, Y->-FX-Y, angle 90. F count doubles each
//              iteration -> EXACTLY 2^iterations segments.
//   - plant  : the classic bracketed branching plant (Wikipedia "L-system"
//              example 7), axiom X, X->F+[[X]-X]-F[-FX]+X, F->FF, angle 25.
//              Exercises the push/pop [ ] branch stack.
export const LSYSTEM_PRESETS = {
  koch:   { axiom: 'F',  rules: { F: 'F+F--F+F' },                       angle: 60, label: 'Koch curve' },
  dragon: { axiom: 'FX', rules: { X: 'X+YF+', Y: '-FX-Y' },              angle: 90, label: 'Dragon curve' },
  plant:  { axiom: 'X',  rules: { X: 'F+[[X]-X]-F[-FX]+X', F: 'FF' },    angle: 25, label: 'Branching plant' },
};

// Count how many symbols (total) and how many drawing symbols ('F') an
// expansion WOULD produce, WITHOUT building the (potentially enormous)
// string. Tracks a per-symbol population and advances it one generation at a
// time — cheap and exact — so the caller can refuse an over-cap request
// before allocating anything. This is the "sanity cap, refuse rather than
// hang the browser" convention ArrayLinear/ArrayPolar's own 500-copy cap
// already established, applied to L-system's exponential growth.
export function countLSystemGrowth(axiom, rules, iterations) {
  let counts = new Map();
  for (const ch of axiom) counts.set(ch, (counts.get(ch) || 0) + 1);
  for (let it = 0; it < iterations; it++) {
    const next = new Map();
    for (const [sym, c] of counts) {
      const succ = rules[sym] !== undefined ? rules[sym] : sym;
      for (const ch of succ) next.set(ch, (next.get(ch) || 0) + c);
    }
    counts = next;
  }
  let total = 0;
  for (const [, c] of counts) total += c;
  return { fCount: counts.get('F') || 0, total };
}

// A generous ceiling on generated segments (F count). Beyond this the browser
// would choke building/rendering the polyline, so we refuse honestly rather
// than hang — matching the ArrayLinear/ArrayPolar refusal posture exactly.
export const MAX_LSYSTEM_SEGMENTS = 8000;
export const MAX_LSYSTEM_SYMBOLS = 400000;

// Expand an L-system string `iterations` times, refusing (throwing an honest,
// specific message) if the result would exceed the sanity caps.
export function expandLSystem(axiom, rules, iterations) {
  if (!Number.isInteger(iterations) || iterations < 0) throw new Error('L-system iterations must be a non-negative integer');
  const { fCount, total } = countLSystemGrowth(axiom, rules, iterations);
  if (fCount > MAX_LSYSTEM_SEGMENTS) {
    throw new Error(`L-system would generate ${fCount} segments, over the ${MAX_LSYSTEM_SEGMENTS} sanity cap — reduce iterations`);
  }
  if (total > MAX_LSYSTEM_SYMBOLS) {
    throw new Error(`L-system would generate ${total} symbols, over the ${MAX_LSYSTEM_SYMBOLS} sanity cap — reduce iterations`);
  }
  let s = axiom;
  for (let it = 0; it < iterations; it++) {
    let out = '';
    for (const ch of s) out += (rules[ch] !== undefined ? rules[ch] : ch);
    s = out;
  }
  return s;
}

// A real 3D turtle. State = { position, heading, up }; '+'/'-' rotate the
// heading around the UP axis by `angle` (standard planar turtle-graphics
// interpretation — a genuine 3D state carried, just with a fixed up so the
// classic 2D fractals read correctly); 'F' moves forward stepLength and draws
// a segment; '[' pushes the full state, ']' pops it (branching). Returns:
//   - segments : one [start,end] pair per drawn 'F' (segments.length === the
//                exact F count, the checkable growth-law property).
//   - polyline : a single continuous [x,y,z][] path over the whole figure. A
//                non-branching system chains naturally (every segment starts
//                where the last ended). A branching system inserts a "travel"
//                vertex whenever a pop jumps the pen elsewhere, so the output
//                is still one connected curve (an honest v1 for a node whose
//                output is a single curve).
export function lSystemTurtle(str, { angle = 90, stepLength = 1 } = {}) {
  const a = (angle * Math.PI) / 180;
  const rotZ = (v, ang) => {
    const c = Math.cos(ang), s = Math.sin(ang);
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
  };
  let pos = [0, 0, 0];
  let heading = [1, 0, 0];
  const stack = [];
  const segments = [];
  for (const ch of str) {
    if (ch === 'F') {
      const nextPos = [pos[0] + heading[0] * stepLength, pos[1] + heading[1] * stepLength, pos[2] + heading[2] * stepLength];
      segments.push([pos.slice(), nextPos.slice()]);
      pos = nextPos;
    } else if (ch === '+') {
      heading = rotZ(heading, a);
    } else if (ch === '-') {
      heading = rotZ(heading, -a);
    } else if (ch === '[') {
      stack.push({ pos: pos.slice(), heading: heading.slice() });
    } else if (ch === ']') {
      const st = stack.pop();
      if (st) { pos = st.pos; heading = st.heading; }
    }
    // any other symbol (X, Y, ...) is a no-op for the turtle (variables only)
  }
  // Build a single continuous polyline from the segment set.
  const same = (p, q) => Math.abs(p[0] - q[0]) < 1e-9 && Math.abs(p[1] - q[1]) < 1e-9 && Math.abs(p[2] - q[2]) < 1e-9;
  const polyline = [];
  if (segments.length) {
    polyline.push(segments[0][0].slice());
    for (const [s0, s1] of segments) {
      const cur = polyline[polyline.length - 1];
      if (!same(cur, s0)) polyline.push(s0.slice()); // travel move after a branch pop
      polyline.push(s1.slice());
    }
  }
  return { segments, polyline };
}

// ===========================================================================
// 2. LORENZ ATTRACTOR
// ===========================================================================
//
// The canonical Lorenz system:  dx/dt = sigma(y-x),  dy/dt = x(rho-z)-y,
// dz/dt = xy - beta*z, with the classic sigma=10, rho=28, beta=8/3 that give
// the well-known butterfly attractor. Integrated with RK4 (a real 4th-order
// scheme) — NOT naive forward-Euler, which visibly drifts/diverges from the
// true attractor at any reasonable step size. The shared derivative is
// exported so a test can build an Euler run from the SAME right-hand side and
// prove the integrator choice actually matters.
export function lorenzDeriv([x, y, z], sigma, rho, beta) {
  return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z];
}

export const LORENZ_DEFAULTS = { sigma: 10, rho: 28, beta: 8 / 3, dt: 0.01, steps: 2000, start: [0.1, 0, 0] };

export function lorenzTrajectory(params = {}) {
  const { sigma, rho, beta, dt, steps, start } = { ...LORENZ_DEFAULTS, ...params };
  let s = start.slice();
  const pts = [s.slice()];
  for (let i = 0; i < steps; i++) {
    const k1 = lorenzDeriv(s, sigma, rho, beta);
    const s2 = [s[0] + k1[0] * dt / 2, s[1] + k1[1] * dt / 2, s[2] + k1[2] * dt / 2];
    const k2 = lorenzDeriv(s2, sigma, rho, beta);
    const s3 = [s[0] + k2[0] * dt / 2, s[1] + k2[1] * dt / 2, s[2] + k2[2] * dt / 2];
    const k3 = lorenzDeriv(s3, sigma, rho, beta);
    const s4 = [s[0] + k3[0] * dt, s[1] + k3[1] * dt, s[2] + k3[2] * dt];
    const k4 = lorenzDeriv(s4, sigma, rho, beta);
    s = [
      s[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
      s[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
      s[2] + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    ];
    pts.push(s.slice());
  }
  return pts;
}

// ===========================================================================
// 3. RANDOM WALK ("Noise Curve" / "Random Walk Curve")
// ===========================================================================
//
// A seeded, fully deterministic random walk through 3D space. Each step picks
// a uniformly-random unit direction from the shared integer hash (seed, i, k)
// — NEVER Math.random — then blends it toward the PREVIOUS heading by
// `roughness` (0 = keep heading exactly, i.e. a straight line; 1 = fully
// random each step, i.e. maximally jagged), so the knob genuinely controls
// how correlated consecutive steps are. Same seed => bit-identical curve;
// different seed => a genuinely different curve.
export const RANDOM_WALK_DEFAULTS = { seed: 1, stepCount: 200, stepLength: 1, roughness: 0.5, start: [0, 0, 0] };

export function randomWalkCurve(params = {}) {
  const { seed, stepCount, stepLength, roughness, start } = { ...RANDOM_WALK_DEFAULTS, ...params };
  const r = Math.min(1, Math.max(0, roughness));
  let pos = start.slice();
  let dir = [1, 0, 0];
  const pts = [pos.slice()];
  for (let i = 1; i <= stepCount; i++) {
    const u = hash01(seed, i, 1);
    const v = hash01(seed, i, 2);
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1); // uniform on the sphere
    const rnd = [Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi)];
    const blended = [
      dir[0] + (rnd[0] - dir[0]) * r,
      dir[1] + (rnd[1] - dir[1]) * r,
      dir[2] + (rnd[2] - dir[2]) * r,
    ];
    // Guard the rare degenerate case (blend cancels to ~zero) by falling back
    // to the raw random direction rather than throwing.
    dir = length(blended) < 1e-9 ? rnd : normalize(blended);
    pos = [pos[0] + dir[0] * stepLength, pos[1] + dir[1] * stepLength, pos[2] + dir[2] * stepLength];
    pts.push(pos.slice());
  }
  return pts;
}

// ===========================================================================
// DISTANCE (single scalar) — a trivial but real kernel helper, node-tested
// alongside the generators. The straight-line Euclidean distance between two
// points; the Distance node's numeric output.
// ===========================================================================
export function pointDistance(a, b) {
  return length(sub(a, b));
}

// ===========================================================================
// 4. WAVE — sine, square, triangle, sawtooth, from ONE parameter block
// ===========================================================================
//
// Asked for directly ("things like sine waves"), and the family whose own
// stated purpose is to be made three times at different phases and lofted.
// The four waveforms share every parameter because they ARE the same curve
// with a different unit shape, so adding the other three to `sine` costs one
// switch rather than three generators.
//
// ⚠ SQUARE AND SAWTOOTH ARE DISCONTINUOUS AND MUST NOT BE FITTED SMOOTH.
// `smoothFit` is false for them by default, exactly as the L-System above
// stays degree-1 to keep its fractal corners: a degree-3 interpolation through
// a step rings badly on both sides of every edge, which reads as a modelling
// error rather than as the square wave the reader asked for. Exposed rather
// than hidden so a reader who WANTS the ringing (it is a real, useful shape)
// can have it.
//
// `skew` pulls the waveform's peak away from the centre of its period: 0.5 is
// symmetric, and at the extremes a triangle becomes a sawtooth. It is applied
// to the phase WITHIN the cycle, so it leaves period and amplitude untouched.
export const WAVE_FORMS = ['sine', 'square', 'triangle', 'sawtooth'];
export const WAVE_CURVE_DEFAULTS = {
  form: 'sine', amplitude: 10, cycles: 3, length: 100, phase: 0,
  offset: 0, damping: 0, skew: 0.5, samples: 200, start: [0, 0, 0],
};
// One period of each waveform on u in [0,1), returning -1..1. Kept separate
// and exported so a test can pin each shape at its own known stations without
// going through the placement, phase and damping machinery around it.
export function waveUnit(form, u) {
  const t = ((u % 1) + 1) % 1;
  switch (form) {
    case 'square': return t < 0.5 ? 1 : -1;
    // ⚠ ALL FOUR CROSS ZERO RISING AT t = 0, which is what makes the claim
    // above ("the same curve with a different unit shape") true rather than
    // merely tidy. A triangle written as 4t-1 starts at -1 and is a quarter
    // period out of phase with the sine beside it, so lofting a sine section
    // to a triangle section would twist for a reason nothing on screen
    // explains. Peaks land on the quarter points, exactly like the sine.
    case 'triangle': return t < 0.25 ? 4 * t : (t < 0.75 ? 2 - 4 * t : 4 * t - 4);
    case 'sawtooth': return t < 0.5 ? 2 * t : 2 * t - 2;
    case 'sine':
    default: return Math.sin(2 * Math.PI * t);
  }
}
export function waveCurve(params = {}) {
  const p = { ...WAVE_CURVE_DEFAULTS, ...params };
  const form = WAVE_FORMS.includes(p.form) ? p.form : 'sine';
  const n = Math.max(2, Math.round(p.samples));
  const skew = Math.min(0.999, Math.max(0.001, p.skew));
  const pts = [];
  for (let i = 0; i < n; i++) {
    const s = i / (n - 1);                       // 0..1 along the run
    const x = p.start[0] + s * p.length;
    // Phase in cycles, then skewed within the cycle. Splitting the cycle at
    // `skew` and rescaling each half keeps every period exactly one period
    // long, so skew changes the SHAPE without changing the frequency.
    let u = s * p.cycles + p.phase / 360;
    const cyc = ((u % 1) + 1) % 1;
    const whole = Math.floor(u);
    const skewed = cyc < skew ? (cyc / skew) * 0.5 : 0.5 + ((cyc - skew) / (1 - skew)) * 0.5;
    u = whole + skewed;
    // Damping is per UNIT LENGTH of the run, so the same value means the same
    // decay whatever `length` is set to.
    const decay = p.damping ? Math.exp(-p.damping * s * p.cycles) : 1;
    const y = p.start[1] + p.offset + p.amplitude * decay * waveUnit(form, u);
    pts.push([x, y, p.start[2]]);
  }
  return pts;
}
// Whether this waveform should be interpolated smooth. See waveCurve's own
// comment: the discontinuous forms are polylines on purpose.
export function waveWantsSmoothFit(form) { return form === 'sine' || form === 'triangle'; }

// ===========================================================================
// 5. HARMONIC — a Fourier sum, which is the wave family generalised
// ===========================================================================
//
// The teaching curve of the set: one term is a sine, and adding odd harmonics
// at 1/n amplitude walks visibly toward a square wave, which is the fact a
// student is meant to SEE rather than be told. `falloff` is the exponent on
// 1/n, so 1 is the square/sawtooth law and 2 is the triangle law.
export const HARMONIC_DEFAULTS = {
  terms: 5, oddOnly: true, falloff: 1, amplitude: 10, cycles: 2,
  length: 100, phase: 0, samples: 300, start: [0, 0, 0],
};
export function harmonicCurve(params = {}) {
  const p = { ...HARMONIC_DEFAULTS, ...params };
  const n = Math.max(2, Math.round(p.samples));
  const terms = Math.max(1, Math.round(p.terms));
  const pts = [];
  for (let i = 0; i < n; i++) {
    const s = i / (n - 1);
    const base = 2 * Math.PI * (s * p.cycles + p.phase / 360);
    let y = 0;
    for (let k = 1; k <= terms; k++) {
      const h = p.oddOnly ? (2 * k - 1) : k;
      y += Math.sin(base * h) / Math.pow(h, p.falloff);
    }
    pts.push([p.start[0] + s * p.length, p.start[1] + p.amplitude * y, p.start[2]]);
  }
  return pts;
}

// ===========================================================================
// 6. FBM NOISE CURVE — fractal Brownian motion along a run
// ===========================================================================
//
// Asked for directly ("perlin noise curves"). Summed octaves of value noise:
// each octave `lacunarity` times the frequency of the last and `persistence`
// times the amplitude, which is the standard fBm parameter set.
//
// ⚠ DISTINCT FROM THE NOISE MODIFIER, and the header of this module explains
// why that distinction is guarded: kernel/noise.mjs's noiseControlNet
// DISPLACES an existing surface. This builds a new curve out of nothing.
//
// ⚠ CLOSED IS NOT A CLAMP. A closed noise curve samples the noise field
// around a CIRCLE in noise space rather than along a line, so the value at
// t=1 is the value at t=0 by construction and the seam cannot show. Clamping
// or mirroring the ends instead leaves a visible discontinuity exactly where
// a reader is most likely to look.
export const NOISE_CURVE_DEFAULTS = {
  seed: 1, octaves: 4, frequency: 1, lacunarity: 2, persistence: 0.5,
  amplitude: 10, length: 100, samples: 240, closed: false, radius: 40, start: [0, 0, 0],
};
// The fBm scalar itself, exported so its octave behaviour can be pinned
// directly: with persistence 0.5 the octave amplitudes are 1, 1/2, 1/4 ...
// and the normalisation below keeps the result in -1..1 whatever `octaves` is.
export function fbm1D(x, y, seed, octaves, frequency, lacunarity, persistence, noise2D) {
  let sum = 0, amp = 1, freq = frequency, norm = 0;
  const oct = Math.max(1, Math.round(octaves));
  for (let o = 0; o < oct; o++) {
    /* ⚠ NO -1..1 REMAP HERE. kernel/noise.mjs's own latticeVal is
       `2 * hash01(...) - 1`, so valueNoise2D ALREADY returns -1..1 — the
       usual `* 2 - 1` applied on top pushes the sum to -3..1 and biases every
       curve downward, which reads as "that is just what noise looks like"
       rather than as a defect. Measured before the fix: -1.33 out of a
       normaliser that guarantees -1..1. Check the convention of the noise you
       are handed; do not assume the [0,1] one. */
    sum += amp * noise2D(x * freq, y * freq, seed + o * 1013);
    norm += amp;
    amp *= persistence;
    freq *= lacunarity;
  }
  return norm > 0 ? sum / norm : 0;
}
export function noiseCurve(params = {}, noise2D) {
  const p = { ...NOISE_CURVE_DEFAULTS, ...params };
  if (typeof noise2D !== 'function') throw new Error('noiseCurve needs a 2D noise function (kernel/noise.mjs valueNoise2D)');
  const n = Math.max(2, Math.round(p.samples));
  const pts = [];
  if (p.closed) {
    // A ring in space, displaced radially by noise sampled on a circle in the
    // noise field — so both the geometry and the noise close exactly.
    for (let i = 0; i < n; i++) {
      const s = i / n;                              // NOT n-1: the last point is not a repeat of the first
      const a = 2 * Math.PI * s;
      const d = fbm1D(Math.cos(a), Math.sin(a), p.seed, p.octaves, p.frequency, p.lacunarity, p.persistence, noise2D);
      const r = p.radius + p.amplitude * d;
      pts.push([p.start[0] + r * Math.cos(a), p.start[1] + r * Math.sin(a), p.start[2]]);
    }
    pts.push(pts[0].slice()); // the wrap segment, explicitly — a closed loop that omits it looks right and is not
    return pts;
  }
  for (let i = 0; i < n; i++) {
    const s = i / (n - 1);
    const d = fbm1D(s * 4, 0, p.seed, p.octaves, p.frequency, p.lacunarity, p.persistence, noise2D);
    pts.push([p.start[0] + s * p.length, p.start[1] + p.amplitude * d, p.start[2]]);
  }
  return pts;
}

// ===========================================================================
// 7. ROULETTE — hypotrochoid / epitrochoid, the spirograph family
// ===========================================================================
//
// One formula, an enormous range, and several named classical curves fall out
// as parameter presets rather than as separate generators:
//   d = r          -> hypocycloid / epicycloid
//   R = 4r, d = r  -> astroid          R = r (epi) -> cardioid
//   R = 3r, d = r  -> deltoid          R = 2r (hypo) -> a straight line (real, and worth seeing)
//
// The curve only CLOSES when R/r is rational; `turns` is therefore a real
// parameter and not a detail, and the honest default walks enough turns to
// close the common ratios.
export const ROULETTE_MODES = ['hypotrochoid', 'epitrochoid'];
export const ROULETTE_DEFAULTS = { mode: 'hypotrochoid', R: 50, r: 15, d: 22, turns: 0, samples: 720, start: [0, 0, 0] };
// How many turns of the driving circle are needed for the tracing point to
// return to where it began: r/gcd(R,r) turns, on the integer part of the
// ratio. Returns 0 when the ratio is not usefully rational, which the caller
// reads as "use the requested turns and do not claim it closes".
export function rouletteClosingTurns(R, r) {
  const scale = 1000;
  const a = Math.round(Math.abs(R) * scale), b = Math.round(Math.abs(r) * scale);
  if (!a || !b) return 0;
  const gcd = (x, y) => (y ? gcd(y, x % y) : x);
  const t = b / gcd(a, b);
  return t > 0 && t <= 200 ? t : 0;
}
export function rouletteCurve(params = {}) {
  const p = { ...ROULETTE_DEFAULTS, ...params };
  const mode = ROULETTE_MODES.includes(p.mode) ? p.mode : 'hypotrochoid';
  const R = p.R, r = p.r, d = p.d;
  const turns = p.turns > 0 ? p.turns : (rouletteClosingTurns(R, r) || 1);
  const n = Math.max(8, Math.round(p.samples));
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const t = (i / n) * turns * 2 * Math.PI;
    let x, y;
    if (mode === 'epitrochoid') {
      const k = (R + r) / r;
      x = (R + r) * Math.cos(t) - d * Math.cos(k * t);
      y = (R + r) * Math.sin(t) - d * Math.sin(k * t);
    } else {
      const k = (R - r) / r;
      x = (R - r) * Math.cos(t) + d * Math.cos(k * t);
      y = (R - r) * Math.sin(t) - d * Math.sin(k * t);
    }
    pts.push([p.start[0] + x, p.start[1] + y, p.start[2]]);
  }
  return pts;
}

// ===========================================================================
// 8. SUPERFORMULA (Gielis) — the widest shape range per parameter here
// ===========================================================================
//
// r(a) = ( |cos(m*a/4)/A|^n2 + |sin(m*a/4)/B|^n3 ) ^ (-1/n1)
//
// Proposed by Johan Gielis as a description of forms found in nature; it
// covers circles, ellipses, superellipses, rounded polygons, stars and flower
// forms continuously as the six numbers move. n1 = n2 = n3 = 2 with m = 4 is
// an ellipse — an exact closed form, which is what the test pins it against.
export const SUPERFORMULA_DEFAULTS = { a: 1, b: 1, m: 6, n1: 1, n2: 1, n3: 1, scale: 40, samples: 360, start: [0, 0, 0] };
export function superformulaRadius(theta, { a, b, m, n1, n2, n3 }) {
  const t1 = Math.pow(Math.abs(Math.cos(m * theta / 4) / a), n2);
  const t2 = Math.pow(Math.abs(Math.sin(m * theta / 4) / b), n3);
  const s = t1 + t2;
  if (!(s > 0) || !Number.isFinite(s)) return 0;
  return Math.pow(s, -1 / n1);
}
export function superformulaCurve(params = {}) {
  const p = { ...SUPERFORMULA_DEFAULTS, ...params };
  const n = Math.max(8, Math.round(p.samples));
  const pts = [];
  for (let i = 0; i < n; i++) {
    const th = (i / n) * 2 * Math.PI;
    const r = superformulaRadius(th, p) * p.scale;
    pts.push([p.start[0] + r * Math.cos(th), p.start[1] + r * Math.sin(th), p.start[2]]);
  }
  pts.push(pts[0].slice()); // closed by construction — the wrap segment is explicit
  return pts;
}

// ---------------------------------------------------------------------------
// Shared integer GCD for the families below whose closure depends on a
// rational ratio (rose, torus knot). rouletteClosingTurns above carries its
// own recursive copy from before this was needed; it is left alone rather
// than re-pointed, because its inputs are SCALED FLOATS (Math.round(R*1000))
// and this one takes plain integers — merging them would silently widen the
// contract of a function four tests already pin.
// ---------------------------------------------------------------------------
function intGcd(a, b) {
  a = Math.abs(a | 0); b = Math.abs(b | 0);
  while (b) { const t = a % b; a = b; b = t; }
  return a;
}

// ===========================================================================
// 9. SPIRAL — archimedean / logarithmic / fermat, planar or conical
// ===========================================================================
//
// Three classical spirals behind ONE `kind` switch, because they differ only
// in r(theta) and share every other parameter — the same reasoning that put
// the four waveforms into waveCurve rather than into four generators.
//
//   archimedean : r = a + b*theta   — turns are EQUALLY SPACED. Successive
//                 turns differ by exactly 2*pi*b, whatever theta is. This is
//                 the spiral of a coiled rope or a clock spring.
//   logarithmic : r = a * e^(b*theta) — turns grow by a constant RATIO
//                 e^(2*pi*b) instead. Bernoulli's spira mirabilis: the shape
//                 is SELF-SIMILAR, so it looks the same at every zoom, which
//                 the equal-spacing spiral above emphatically does not.
//   fermat      : r = a * sqrt(theta) — the parabolic spiral; equal AREA per
//                 turn rather than equal spacing or equal ratio.
//
// ⚠ theta IS IN RADIANS and `growth` is therefore PER RADIAN, not per turn.
// That is the form the three formulas are quoted in everywhere, so quoting
// them any other way would make every reference the reader checks disagree
// with the code. The per-turn consequences (spacing 2*pi*b, ratio e^(2*pi*b))
// are stated above and pinned in the tests.
//
// ⚠ `startRadius` IS NOT THE STARTING RADIUS FOR FERMAT. It is the
// coefficient a in every kind, and for fermat r(0) = a*sqrt(0) = 0 — a Fermat
// spiral always begins at the origin, and `a` sets how fast it leaves. Naming
// it `coefficient` would be accurate and unreadable; naming it `startRadius`
// is readable and true for two kinds of three, so the trap is documented here
// rather than papered over by special-casing fermat into starting somewhere
// it does not.
//
// `height` lifts the curve linearly along z over the whole run, turning the
// planar spiral into a conical one (radius still growing, now climbing too).
// height = 0 is the planar default.
export const SPIRAL_KINDS = ['archimedean', 'logarithmic', 'fermat'];
export const SPIRAL_DEFAULTS = {
  kind: 'archimedean', turns: 4, startRadius: 5, growth: 2, height: 0,
  samples: 360, start: [0, 0, 0],
};
export function spiralRadius(kind, theta, a, b) {
  switch (kind) {
    case 'logarithmic': return a * Math.exp(b * theta);
    case 'fermat': return a * Math.sqrt(Math.max(0, theta));
    case 'archimedean':
    default: return a + b * theta;
  }
}
export function spiralCurve(params = {}) {
  const p = { ...SPIRAL_DEFAULTS, ...params };
  const kind = SPIRAL_KINDS.includes(p.kind) ? p.kind : 'archimedean';
  const n = Math.max(2, Math.round(p.samples));
  const thetaMax = p.turns * 2 * Math.PI;
  // ⚠ REFUSE THE OVERFLOW RATHER THAN EMIT Infinity/NaN. A logarithmic spiral
  // is an EXPONENTIAL in a number the user types: growth 1 with 40 turns is
  // e^251, which is representable, but growth 3 with 40 turns is e^754 and is
  // not. Left unguarded the chain fills with Infinity, every downstream fit
  // produces NaN control points, and the reported error names interpolate.mjs
  // — a place with nothing wrong with it.
  if (kind === 'logarithmic') {
    const rEnd = Math.abs(p.startRadius) * Math.exp(p.growth * thetaMax);
    if (!Number.isFinite(rEnd)) {
      throw new Error(`logarithmic spiral overflows: growth ${p.growth} over ${p.turns} turns is e^${(p.growth * thetaMax).toFixed(0)} — reduce growth or turns`);
    }
  }
  const pts = [];
  for (let i = 0; i < n; i++) {
    const s = i / (n - 1);
    const th = s * thetaMax;
    const r = spiralRadius(kind, th, p.startRadius, p.growth);
    pts.push([
      p.start[0] + r * Math.cos(th),
      p.start[1] + r * Math.sin(th),
      p.start[2] + s * p.height,
    ]);
  }
  return pts;
}

// ===========================================================================
// 10. LISSAJOUS — x = A sin(a t + delta), y = B sin(b t), z = C sin(c t)
// ===========================================================================
//
// The oscilloscope figure: two (here three) perpendicular sinusoids at
// different frequencies. The whole interest of the family is that the
// frequency RATIO decides the topology — 1:1 is an ellipse (a line at phase
// 0), 1:2 is a figure-eight, 3:2 is the classic three-lobed knot-looking
// figure — and the phase decides how that figure is presented.
//
// ⚠ IT ONLY CLOSES WHEN THE RATIO IS RATIONAL, and "rational" is not the same
// thing as "whole". Integer frequencies close at t = 2pi, but so do half-integer
// ones — at 4pi. sin(2.5 * 2pi) = sin(5pi) = 0 = sin(0), so a chain sampled over
// [0, 2pi] at freq 2.5 comes back to the same POINT with the opposite tangent:
// it is a half traversal that looks closed and is not, and a wrap segment added
// there welds the figure to the middle of itself. lissajousPeriodTurns solves
// for the real period instead, and an irrationally-related set (freq = pi) is
// emitted OPEN with no closing repeat.
//
// ⚠ freqZ IS INERT WHEN ampZ IS 0, which is the default. That is not a bug
// (a zero-amplitude axis has no frequency to see) but it is a real trap for
// anything trying to test freqZ: a fixture with ampZ = 0 cannot tell a
// correct freqZ from a discarded one.
export const LISSAJOUS_DEFAULTS = {
  freqX: 3, freqY: 2, freqZ: 0, phase: 90,
  ampX: 40, ampY: 40, ampZ: 0, samples: 720, start: [0, 0, 0],
};
// Best rational p/q for x with q <= maxDen, or null if none is exact enough.
// The tolerance is deliberately far tighter than the search is wide: 355/113
// is within 2.7e-7 of pi and would otherwise make an irrational frequency
// "close" after 113 turns, which is a lie that only shows at the seam.
function ratApprox(x, maxDen = 512) {
  for (let q = 1; q <= maxDen; q++) {
    const num = Math.round(x * q);
    if (Math.abs(x - num / q) < 1e-12) return [Math.abs(num), q];
  }
  return null;
}
// How many turns of the base 2*pi parameter the figure needs to close, or 0
// for "it does not". Every frequency must complete a whole number of its own
// cycles at the same t, so with f_i = p_i / q_i in lowest terms the answer is
// lcm(q_i) / gcd(p_i). A zero frequency is dropped first: that axis is a
// constant and constrains nothing.
export function lissajousPeriodTurns(freqX, freqY, freqZ) {
  const fs = [freqX, freqY, freqZ].filter((f) => f !== 0);
  if (!fs.length) return 1;
  const rats = fs.map((f) => ratApprox(Math.abs(f)));
  if (rats.some((r) => r === null)) return 0;
  let num = 0, den = 1;
  for (const [pi, qi] of rats) { num = intGcd(num, pi); den = (den / intGcd(den, qi)) * qi; }
  if (!num) return 0;
  const g = intGcd(den, num);
  const k = (den / g) / (num / g);
  return Number.isFinite(k) && k > 0 && k <= 1000 ? k : 0;
}
export function lissajousCloses(freqX, freqY, freqZ) {
  return lissajousPeriodTurns(freqX, freqY, freqZ) > 0;
}
export function lissajousCurve(params = {}) {
  const p = { ...LISSAJOUS_DEFAULTS, ...params };
  const n = Math.max(8, Math.round(p.samples));
  const delta = (p.phase * Math.PI) / 180;
  const at = (t) => [
    p.start[0] + p.ampX * Math.sin(p.freqX * t + delta),
    p.start[1] + p.ampY * Math.sin(p.freqY * t),
    p.start[2] + p.ampZ * Math.sin(p.freqZ * t),
  ];
  const pts = [];
  const turns = lissajousPeriodTurns(p.freqX, p.freqY, p.freqZ);
  if (turns > 0) {
    const span = turns * 2 * Math.PI;
    for (let i = 0; i < n; i++) pts.push(at((i / n) * span)); // NOT n-1: the last sample is not a repeat of the first
    pts.push(pts[0].slice()); // the wrap segment, explicitly
    return pts;
  }
  // Never returns, so one base turn is shown and no closure is claimed.
  for (let i = 0; i <= n; i++) pts.push(at((i / n) * 2 * Math.PI));
  return pts;
}

// ===========================================================================
// 11. ROSE (rhodonea) — r = a * cos(n*theta/d)
// ===========================================================================
//
// The petal count is the whole point and it is NOT n. For d = 1:
//   n odd  -> exactly n petals, traced over theta in [0, pi)
//   n even -> exactly 2n petals, traced over theta in [0, 2pi)
// The reason is that r goes NEGATIVE for half the range, and a negative r in
// polar coordinates plots at theta + pi — so an odd rose retraces its own
// petals on the second half turn instead of drawing new ones, while an even
// rose draws a fresh petal in each gap. Rendering |r| instead of the signed r
// (a very natural-looking "fix") destroys exactly this: it turns every rose
// into a 2n-petal one and the classic 3-petal trefoil becomes a 6-petal
// flower.
//
// The rational generalisation r = a*cos(n*theta/d) with n/d in lowest terms
// closes after d*pi when n*d is odd and 2*d*pi otherwise (the same signed-r
// argument, one period of cos(n*theta/d) later). n/d = 7/2 and 2/7 are both
// real, very different, and both closed.
//
// ⚠ n AND d MUST BE POSITIVE INTEGERS. An irrational ratio never closes — it
// fills an annulus densely — so it is refused by name rather than emitted as
// a chain with a chord across it.
export const ROSE_DEFAULTS = { n: 5, d: 1, amplitude: 40, samples: 720, start: [0, 0, 0] };
// The angular span of one complete traversal, after reducing n/d to lowest
// terms. Exported because the sampling and any consumer wanting to subdivide
// the curve need the same number.
export function roseThetaMax(n, d) {
  const g = intGcd(n, d) || 1;
  const nn = n / g, dd = d / g;
  return ((nn * dd) % 2 === 1) ? dd * Math.PI : 2 * dd * Math.PI;
}
export function roseCurve(params = {}) {
  const p = { ...ROSE_DEFAULTS, ...params };
  if (!Number.isInteger(p.n) || !Number.isInteger(p.d) || p.n < 1 || p.d < 1) {
    throw new Error(`roseCurve needs positive INTEGER n and d (got n=${p.n}, d=${p.d}) — r = a*cos(n*theta/d) only closes for a rational ratio`);
  }
  const n = Math.max(8, Math.round(p.samples));
  const thetaMax = roseThetaMax(p.n, p.d);
  const k = p.n / p.d;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const th = (i / n) * thetaMax;              // NOT n-1: the last sample is not a repeat of the first
    const r = p.amplitude * Math.cos(k * th);   // ⚠ SIGNED — see the header
    pts.push([p.start[0] + r * Math.cos(th), p.start[1] + r * Math.sin(th), p.start[2]]);
  }
  pts.push(pts[0].slice()); // the wrap segment, explicitly
  return pts;
}

// ===========================================================================
// 12. HELIX — the genuinely 3D staple: cylinder, cone, spring
// ===========================================================================
//
// ⚠ pitch, turns AND height ARE ONE PARAMETER TOO MANY. height = pitch *
// turns identically, so a UI offering all three can be put into a state that
// contradicts itself. Rather than silently preferring one and letting the
// other read as broken, the rule is stated and exported: PITCH DRIVES, and
// `height` is an optional override — set height > 0 and pitch is DERIVED as
// height / turns instead. helixResolve returns the triple actually used, so
// the relationship can be read (and asserted) rather than inferred.
//
// `taper` is the radius at the top as a fraction removed: 0 leaves a cylinder
// of constant radius, 1 closes the radius to exactly zero at the last point
// (a cone / conical spring). Clamped to [0,1]: beyond 1 the radius goes
// negative and the curve passes through the axis and unwinds on the far side,
// which is a different shape from the one this parameter names.
export const HELIX_DEFAULTS = {
  radius: 20, pitch: 10, turns: 5, height: 0, taper: 0, phase: 0,
  samples: 400, start: [0, 0, 0],
};
export function helixResolve(params = {}) {
  const p = { ...HELIX_DEFAULTS, ...params };
  const turns = p.turns;
  const pitch = p.height > 0 ? p.height / turns : p.pitch;
  return { turns, pitch, height: pitch * turns };
}
export function helixCurve(params = {}) {
  const p = { ...HELIX_DEFAULTS, ...params };
  if (!(p.turns > 0)) throw new Error(`helixCurve needs turns > 0 (got ${p.turns}) — zero turns is a point, not a helix`);
  const { pitch, height } = helixResolve(p);
  const taper = Math.min(1, Math.max(0, p.taper));
  const ph = (p.phase * Math.PI) / 180;
  const n = Math.max(2, Math.round(p.samples));
  const pts = [];
  for (let i = 0; i < n; i++) {
    const s = i / (n - 1);
    const th = ph + s * p.turns * 2 * Math.PI;
    const r = p.radius * (1 - taper * s);
    pts.push([p.start[0] + r * Math.cos(th), p.start[1] + r * Math.sin(th), p.start[2] + s * height]);
  }
  return pts;
}
// The exact arc length of the UNTAPERED helix: each turn is the hypotenuse of
// a right triangle whose legs are the circumference and the pitch, so
// L = turns * sqrt((2*pi*R)^2 + pitch^2). A closed form, so the polyline can
// be measured against it rather than against another polyline.
export function helixArcLength(radius, pitch, turns) {
  return turns * Math.hypot(2 * Math.PI * radius, pitch);
}

// ===========================================================================
// 13. CATENARY — y = a*cosh(x/a), the hanging chain
// ===========================================================================
//
// The curve a uniform chain takes under its own weight, and famously NOT the
// parabola it is mistaken for. The single shape parameter is `a` (the ratio
// of horizontal tension to weight per unit length): small a is a deep sag,
// large a is nearly flat.
//
// ⚠ THIS FAMILY TAKES span AND sag, AND SOLVES FOR a. That is the useful
// input pair (a reader knows how wide the gap is and how far the chain should
// hang) but it is a TRANSCENDENTAL relation:
//     sag = a * (cosh(span / (2a)) - 1)
// which has no closed-form inverse in elementary functions. It is inverted
// numerically, by bisection, and that is stated here rather than hidden
// behind a formula-looking helper. The parabolic approximation a ~ span^2 /
// (8*sag) is used only to SEED the bracket, never as the answer: it is the
// leading term of the series and is exactly the "a catenary is a parabola"
// error. At span 100 with sag 25 it gives a = 50 where the true a is 53.716,
// so a curve built on it hangs 8.6% too deep.
//
// Bisection rather than Newton because sag(a) is strictly monotonic
// decreasing on a > 0, so a bracket cannot be lost; Newton on the same
// function is faster and can walk off toward a = 0, where cosh overflows.
export const CATENARY_DEFAULTS = { span: 100, sag: 25, samples: 121, start: [0, 0, 0] };
// Solve sag = a*(cosh(span/(2a)) - 1) for a. Converges to `tol` RELATIVE,
// which is what matters here: a scales with span, so an absolute tolerance
// would mean something different for a 1mm chain and a 1km one.
export function catenaryParameter(span, sag, tol = 1e-14) {
  if (!(span > 0)) throw new Error(`catenaryCurve needs span > 0 (got ${span})`);
  if (!(sag > 0)) throw new Error(`catenaryCurve needs sag > 0 (got ${sag}) — a chain with no sag is a straight line under infinite tension, not a catenary`);
  const sagOf = (a) => a * (Math.cosh(span / (2 * a)) - 1);
  const seed = (span * span) / (8 * sag);   // parabolic approximation, a SEED only
  let lo = seed, hi = seed;                 // sagOf is DECREASING: lo is small-a/deep-sag
  for (let g = 0; g < 200 && sagOf(hi) > sag; g++) hi *= 2;
  for (let g = 0; g < 200 && sagOf(lo) < sag; g++) lo /= 2;
  for (let i = 0; i < 300; i++) {
    const mid = 0.5 * (lo + hi);
    if (sagOf(mid) > sag) lo = mid; else hi = mid;
    if (hi - lo <= tol * hi) break;
  }
  return 0.5 * (lo + hi);
}
// ⚠ THE DEFAULT SAMPLE COUNT IS ODD ON PURPOSE. With an odd count one sample
// lands exactly on the span midpoint, which is where the vertex is — so the
// lowest point of the chain is a point on the curve rather than something the
// polyline cuts a chord across.
export function catenaryCurve(params = {}) {
  const p = { ...CATENARY_DEFAULTS, ...params };
  const n = Math.max(2, Math.round(p.samples));
  const a = catenaryParameter(p.span, p.sag);
  const half = p.span / 2;
  const top = a * Math.cosh(half / a);   // the common height of the two ends
  const pts = [];
  for (let i = 0; i < n; i++) {
    const s = i / (n - 1);
    const u = -half + s * p.span;        // u = 0 at the span midpoint, where the vertex is
    // The textbook y = a*cosh(u/a), shifted so the two suspension points sit
    // at start[1] and the chain hangs DOWN from them to start[1] - sag.
    pts.push([p.start[0] + half + u, p.start[1] + a * Math.cosh(u / a) - top, p.start[2]]);
  }
  return pts;
}

// ===========================================================================
// 14. TORUS KNOT — the (p,q) knot, wound on a torus
// ===========================================================================
//
//   x = (R + r*cos(q*t)) * cos(p*t)
//   y = (R + r*cos(q*t)) * sin(p*t)
//   z =      r*sin(q*t)                       t in [0, 2*pi)
//
// Over one period the curve goes round the main axis exactly p times and
// round the tube exactly q times, and EVERY point satisfies the implicit
// equation of the torus it is wound on:
//     (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2
// identically, since sqrt(x^2+y^2) - R = r*cos(q*t) and z = r*sin(q*t). That
// identity is the family's oracle and it holds to machine precision.
//
// ⚠ gcd(p,q) MUST BE 1 OR IT IS NOT A KNOT. With gcd(p,q) = g > 1 the
// parametrisation above returns to its start after 2*pi/g and then retraces
// the same points g times over — the object is a LINK of g separate
// components, and a single point chain cannot represent it. That is refused
// by name: emitting the g-fold retrace would look correct on screen and give
// a curve with g coincident copies of itself, which every downstream fit and
// offset would then choke on for reasons naming the wrong module.
//
// (p,q) and (q,p) are the SAME knot type (a torus is symmetric in its two
// circles) but they are different CURVES in space, so both are offered.
export const TORUS_KNOT_DEFAULTS = { p: 2, q: 3, R: 40, r: 12, samples: 720, start: [0, 0, 0] };
export function torusKnotCurve(params = {}) {
  const cfg = { ...TORUS_KNOT_DEFAULTS, ...params };
  if (!Number.isInteger(cfg.p) || !Number.isInteger(cfg.q)) {
    throw new Error(`torusKnotCurve needs INTEGER p and q (got p=${cfg.p}, q=${cfg.q}) — a non-integer winding never closes`);
  }
  const g = intGcd(cfg.p, cfg.q);
  if (g !== 1) {
    throw new Error(`torusKnotCurve: (${cfg.p},${cfg.q}) is not a knot — gcd is ${g}, so it is a LINK of ${g} components and cannot be one point chain`);
  }
  const n = Math.max(8, Math.round(cfg.samples));
  const pts = [];
  for (let i = 0; i < n; i++) {
    const t = (i / n) * 2 * Math.PI;     // NOT n-1: the last sample is not a repeat of the first
    const ring = cfg.R + cfg.r * Math.cos(cfg.q * t);
    pts.push([
      cfg.start[0] + ring * Math.cos(cfg.p * t),
      cfg.start[1] + ring * Math.sin(cfg.p * t),
      cfg.start[2] + cfg.r * Math.sin(cfg.q * t),
    ]);
  }
  pts.push(pts[0].slice()); // the wrap segment, explicitly
  return pts;
}
