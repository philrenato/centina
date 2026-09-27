// Curve generators: routines that build a standalone curve from parameters
// alone, with no geometric input. They share one module because they are the
// modes of a single Curve Generator node (one node with a type dropdown):
//   1. L-System   — string-rewriting turtle graphics (Koch/Dragon/Plant).
//   2. Lorenz      — the Lorenz attractor (RK4-integrated).
//   3. Random Walk — a seeded, deterministic random-walk path through space.
//   ...and the wave, harmonic, fBm, roulette, superformula, spiral, Lissajous,
//   rose, helix, catenary and torus-knot families below.
//
// Nothing here takes an existing surface or curve as input. The Random Walk /
// Noise Curve mode is a different operation from the Noise modifier
// (kernel/noise.mjs, noiseControlNet), which displaces the control net of an
// existing surface; the user-facing label is "Random Walk Curve" / "Noise
// Curve", never bare "Noise", to keep the two apart.
//
// Each generator produces a plain [x,y,z][] point chain. The smooth families
// are meant to be decimated and fed through interpolate.mjs's
// globalCurveInterp into a NURBS curve; the L-System stays a polyline
// (degree-1 interpolation) so its fractal corners are kept exactly.

import { normalize, sub, length } from './vec3.mjs';

// Shared deterministic PRNG, identical to kernel/noise.mjs's hashU32/hash01
// (the 0x45d9f3b xor-shift-multiply hash); never Math.random. A per-index
// integer hash keyed by (seed, i, component) is the only source of randomness,
// so a given seed always reproduces the same curve.
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

// 1. L-system
//
// Standard textbook L-system presets, each checkable by hand against a known
// reference shape:
//   - koch   : the Koch-curve-family generator F->F+F--F+F, angle 60. Each
//              iteration replaces every F with 4 F's, so the segment count is
//              exactly 4^iterations.
//   - dragon : the Heighway dragon curve, axiom FX, X->X+YF+, Y->-FX-Y,
//              angle 90. The F count doubles each iteration -> 2^iterations
//              segments.
//   - plant  : the bracketed branching plant (Wikipedia "L-system" example 7),
//              axiom X, X->F+[[X]-X]-F[-FX]+X, F->FF, angle 25. Exercises the
//              push/pop [ ] branch stack.
export const LSYSTEM_PRESETS = {
  koch:   { axiom: 'F',  rules: { F: 'F+F--F+F' },                       angle: 60, label: 'Koch curve' },
  dragon: { axiom: 'FX', rules: { X: 'X+YF+', Y: '-FX-Y' },              angle: 90, label: 'Dragon curve' },
  plant:  { axiom: 'X',  rules: { X: 'F+[[X]-X]-F[-FX]+X', F: 'FF' },    angle: 25, label: 'Branching plant' },
};

// Count the symbols (total) and drawing symbols ('F') an expansion would
// produce, without building the string. Tracks a per-symbol population one
// generation at a time — cheap and exact — so the caller can refuse an
// over-cap request before allocating anything, as ArrayLinear/ArrayPolar do
// with their 500-copy cap.
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

// Ceiling on generated segments (F count). Beyond this, building and rendering
// the polyline stalls the browser, so the request is refused.
export const MAX_LSYSTEM_SEGMENTS = 8000;
export const MAX_LSYSTEM_SYMBOLS = 400000;

// Expand an L-system string `iterations` times, throwing a specific message if
// the result would exceed the caps.
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

// A 3D turtle. State = { position, heading, up }; '+'/'-' rotate the heading
// around the up axis by `angle` (planar turtle-graphics interpretation, with a
// fixed up so the 2D fractals read correctly); 'F' moves forward stepLength
// and draws a segment; '[' pushes the full state, ']' pops it (branching).
// Returns:
//   - segments : one [start,end] pair per drawn 'F' (segments.length === the
//                F count).
//   - polyline : a single continuous [x,y,z][] path over the whole figure. A
//                non-branching system chains naturally. A branching system
//                inserts a travel vertex whenever a pop moves the pen, so the
//                output is still one connected curve.
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

// 2. Lorenz attractor
//
// The Lorenz system:  dx/dt = sigma(y-x),  dy/dt = x(rho-z)-y,
// dz/dt = xy - beta*z, with sigma=10, rho=28, beta=8/3 (the butterfly
// attractor). Integrated with RK4 rather than forward Euler, which drifts from
// the attractor at any reasonable step size. The derivative is exported so a
// test can build an Euler run from the same right-hand side and compare.
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

// 3. Random walk ("Noise Curve" / "Random Walk Curve")
//
// A seeded, deterministic random walk through 3D space. Each step picks a
// uniformly random unit direction from the shared integer hash (seed, i, k),
// then blends it toward the previous heading by `roughness` (0 = keep the
// heading, a straight line; 1 = fully random each step), so the parameter
// controls how correlated consecutive steps are. Same seed => bit-identical
// curve.
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

// Distance: the straight-line Euclidean distance between two points; the
// Distance node's numeric output.
export function pointDistance(a, b) {
  return length(sub(a, b));
}

// 4. Wave — sine, square, triangle, sawtooth, from one parameter block
//
// The family meant to be made several times at different phases and lofted.
// The four waveforms share every parameter because they are the same curve
// with a different unit shape.
//
// Square and sawtooth are discontinuous and are not fitted smooth by default:
// `smoothFit` is false for them, as the L-System stays degree-1 to keep its
// corners, because a degree-3 interpolation through a step rings on both sides
// of every edge. It is exposed so the ringing can be had on purpose.
//
// `skew` pulls the waveform's peak away from the center of its period: 0.5 is
// symmetric, and at the extremes a triangle becomes a sawtooth. It is applied
// to the phase within the cycle, so it leaves period and amplitude untouched.
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
    // All four cross zero rising at t = 0, so they are the same curve with a
    // different unit shape. A triangle written as 4t-1 would start at -1, a
    // quarter period out of phase with the sine, and a loft from a sine section
    // to a triangle section would twist. Peaks land on the quarter points,
    // as for the sine.
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
    // long, so skew changes the shape without changing the frequency.
    let u = s * p.cycles + p.phase / 360;
    const cyc = ((u % 1) + 1) % 1;
    const whole = Math.floor(u);
    const skewed = cyc < skew ? (cyc / skew) * 0.5 : 0.5 + ((cyc - skew) / (1 - skew)) * 0.5;
    u = whole + skewed;
    // Damping is per unit length of the run, so the same value means the same
    // decay whatever `length` is set to.
    const decay = p.damping ? Math.exp(-p.damping * s * p.cycles) : 1;
    const y = p.start[1] + p.offset + p.amplitude * decay * waveUnit(form, u);
    pts.push([x, y, p.start[2]]);
  }
  return pts;
}
// Whether this waveform should be interpolated smooth. The discontinuous forms
// are polylines on purpose (see waveCurve).
export function waveWantsSmoothFit(form) { return form === 'sine' || form === 'triangle'; }

// 5. Harmonic — a Fourier sum, the wave family generalized
//
// One term is a sine; adding odd harmonics at 1/n amplitude converges toward a
// square wave. `falloff` is the exponent on 1/n, so 1 is the square/sawtooth
// law and 2 is the triangle law.
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

// 6. fBm noise curve — fractal Brownian motion along a run
//
// Summed octaves of value noise: each octave `lacunarity` times the frequency
// of the last and `persistence` times the amplitude, the standard fBm
// parameter set. Distinct from the Noise modifier (kernel/noise.mjs,
// noiseControlNet), which displaces an existing surface.
//
// A closed noise curve samples the noise field around a circle in noise space
// rather than along a line, so the value at t=1 is the value at t=0 by
// construction and there is no seam. Clamping or mirroring the ends instead
// leaves a visible discontinuity.
export const NOISE_CURVE_DEFAULTS = {
  seed: 1, octaves: 4, frequency: 1, lacunarity: 2, persistence: 0.5,
  amplitude: 10, length: 100, samples: 240, closed: false, radius: 40, start: [0, 0, 0],
};
// The fBm scalar itself, exported so its octave behavior can be pinned
// directly: with persistence 0.5 the octave amplitudes are 1, 1/2, 1/4 ...
// and the normalization below keeps the result in -1..1 whatever `octaves` is.
export function fbm1D(x, y, seed, octaves, frequency, lacunarity, persistence, noise2D) {
  let sum = 0, amp = 1, freq = frequency, norm = 0;
  const oct = Math.max(1, Math.round(octaves));
  for (let o = 0; o < oct; o++) {
    /* No -1..1 remap here. kernel/noise.mjs's latticeVal is
       `2 * hash01(...) - 1`, so valueNoise2D already returns -1..1; applying
       `* 2 - 1` on top would push the sum to -3..1 and bias every curve
       downward. */
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
      const s = i / n;                              // not n-1: the last point is not a repeat of the first
      const a = 2 * Math.PI * s;
      const d = fbm1D(Math.cos(a), Math.sin(a), p.seed, p.octaves, p.frequency, p.lacunarity, p.persistence, noise2D);
      const r = p.radius + p.amplitude * d;
      pts.push([p.start[0] + r * Math.cos(a), p.start[1] + r * Math.sin(a), p.start[2]]);
    }
    pts.push(pts[0].slice()); // the wrap segment, explicitly — a closed loop needs it
    return pts;
  }
  for (let i = 0; i < n; i++) {
    const s = i / (n - 1);
    const d = fbm1D(s * 4, 0, p.seed, p.octaves, p.frequency, p.lacunarity, p.persistence, noise2D);
    pts.push([p.start[0] + s * p.length, p.start[1] + p.amplitude * d, p.start[2]]);
  }
  return pts;
}

// 7. Roulette — hypotrochoid / epitrochoid, the spirograph family
//
// One formula; several named classical curves are parameter presets rather
// than separate generators:
//   d = r          -> hypocycloid / epicycloid
//   R = 4r, d = r  -> astroid          R = r (epi) -> cardioid
//   R = 3r, d = r  -> deltoid          R = 2r (hypo) -> a straight line
//
// The curve closes only when R/r is rational, so `turns` is a real parameter,
// and the default walks enough turns to close the common ratios.
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

// 8. Superformula (Gielis)
//
// r(a) = ( |cos(m*a/4)/A|^n2 + |sin(m*a/4)/B|^n3 ) ^ (-1/n1)
//
// Proposed by Johan Gielis as a description of natural forms; it covers
// circles, ellipses, superellipses, rounded polygons, stars and flower forms
// continuously as the six numbers move. n1 = n2 = n3 = 2 with m = 4 is an
// ellipse — an exact closed form, which the test pins it against.
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

// Integer GCD for the families whose closure depends on a rational ratio
// (rose, torus knot). rouletteClosingTurns keeps its own recursive copy because
// its inputs are scaled floats (Math.round(R*1000)) while this one takes plain
// integers.
function intGcd(a, b) {
  a = Math.abs(a | 0); b = Math.abs(b | 0);
  while (b) { const t = a % b; a = b; b = t; }
  return a;
}

// 9. Spiral — archimedean / logarithmic / fermat, planar or conical
//
// Three classical spirals behind one `kind` switch, because they differ only
// in r(theta) and share every other parameter.
//
//   archimedean : r = a + b*theta   — turns are equally spaced. Successive
//                 turns differ by exactly 2*pi*b, whatever theta is (a coiled
//                 rope, a clock spring).
//   logarithmic : r = a * e^(b*theta) — turns grow by a constant ratio
//                 e^(2*pi*b) instead. Bernoulli's spira mirabilis: the shape is
//                 self-similar, the same at every zoom.
//   fermat      : r = a * sqrt(theta) — the parabolic spiral; equal area per
//                 turn rather than equal spacing or equal ratio.
//
// theta is in radians and `growth` is therefore per radian, not per turn —
// the form the three formulas are quoted in everywhere. The per-turn
// consequences (spacing 2*pi*b, ratio e^(2*pi*b)) are pinned in the tests.
//
// `startRadius` is not the starting radius for fermat. It is the coefficient a
// in every kind, and for fermat r(0) = a*sqrt(0) = 0: a Fermat spiral always
// begins at the origin, and `a` sets how fast it leaves.
//
// `height` lifts the curve linearly along z over the whole run, turning the
// planar spiral into a conical one. height = 0 is the planar default.
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
  // Refuse overflow rather than emit Infinity/NaN. A logarithmic spiral is an
  // exponential in a user-typed number: growth 1 with 40 turns is e^251, which
  // is representable, but growth 3 with 40 turns is e^754 and is not.
  // Unguarded, the chain fills with Infinity, downstream fits produce NaN
  // control points, and the error surfaces in interpolate.mjs.
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

// 10. Lissajous — x = A sin(a t + delta), y = B sin(b t), z = C sin(c t)
//
// Two (here three) perpendicular sinusoids at different frequencies. The
// frequency ratio decides the topology — 1:1 is an ellipse (a line at phase
// 0), 1:2 is a figure-eight, 3:2 is the three-lobed figure — and the phase
// decides how that figure is presented.
//
// It closes only when the ratio is rational, which is not the same as whole.
// Integer frequencies close at t = 2pi, half-integer ones at 4pi.
// sin(2.5 * 2pi) = sin(5pi) = 0 = sin(0), so a chain sampled over [0, 2pi] at
// freq 2.5 returns to the same point with the opposite tangent: a half
// traversal, and a wrap segment added there would weld the figure to its own
// middle. lissajousPeriodTurns solves for the true period, and an
// irrationally related set (freq = pi) is emitted open with no closing repeat.
//
// freqZ has no effect when ampZ is 0, the default, so a test of freqZ needs a
// nonzero ampZ.
export const LISSAJOUS_DEFAULTS = {
  freqX: 3, freqY: 2, freqZ: 0, phase: 90,
  ampX: 40, ampY: 40, ampZ: 0, samples: 720, start: [0, 0, 0],
};
// Best rational p/q for x with q <= maxDen, or null if none is exact enough.
// The tolerance is far tighter than the search is wide: 355/113 is within
// 2.7e-7 of pi and would otherwise make an irrational frequency close after
// 113 turns, with a mismatch at the seam.
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
    for (let i = 0; i < n; i++) pts.push(at((i / n) * span)); // not n-1: the last sample is not a repeat of the first
    pts.push(pts[0].slice()); // the wrap segment, explicitly
    return pts;
  }
  // Never returns, so one base turn is shown and no closure is claimed.
  for (let i = 0; i <= n; i++) pts.push(at((i / n) * 2 * Math.PI));
  return pts;
}

// 11. Rose (rhodonea) — r = a * cos(n*theta/d)
//
// The petal count is not n. For d = 1:
//   n odd  -> exactly n petals, traced over theta in [0, pi)
//   n even -> exactly 2n petals, traced over theta in [0, 2pi)
// r goes negative for half the range, and a negative r in polar coordinates
// plots at theta + pi — so an odd rose retraces its petals on the second half
// turn, while an even rose draws a new petal in each gap. Plotting |r| instead
// of the signed r turns every rose into a 2n-petal one (the 3-petal trefoil
// becomes a 6-petal flower).
//
// The rational generalization r = a*cos(n*theta/d) with n/d in lowest terms
// closes after d*pi when n*d is odd and 2*d*pi otherwise (the same signed-r
// argument, one period of cos(n*theta/d) later). n/d = 7/2 and 2/7 are
// different and both closed.
//
// n and d must be positive integers. An irrational ratio never closes — it
// fills an annulus densely — so it is refused rather than emitted as a chain
// with a chord across it.
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
    const th = (i / n) * thetaMax;              // not n-1: the last sample is not a repeat of the first
    const r = p.amplitude * Math.cos(k * th);   // signed — see the header
    pts.push([p.start[0] + r * Math.cos(th), p.start[1] + r * Math.sin(th), p.start[2]]);
  }
  pts.push(pts[0].slice()); // the wrap segment, explicitly
  return pts;
}

// 12. Helix — cylinder, cone, spring
//
// pitch, turns and height are one parameter too many: height = pitch * turns.
// The rule: pitch drives, and `height` is an optional override — set
// height > 0 and pitch is derived as height / turns. helixResolve returns the
// triple actually used, so the relationship can be read and asserted.
//
// `taper` is the fraction of the radius removed at the top: 0 leaves a cylinder
// of constant radius, 1 closes the radius to exactly zero at the last point
// (a cone / conical spring). Clamped to [0,1]: beyond 1 the radius goes
// negative and the curve passes through the axis and unwinds on the far side.
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
  const { height } = helixResolve(p);
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
// The exact arc length of the untapered helix: each turn is the hypotenuse of
// a right triangle whose legs are the circumference and the pitch, so
// L = turns * sqrt((2*pi*R)^2 + pitch^2).
export function helixArcLength(radius, pitch, turns) {
  return turns * Math.hypot(2 * Math.PI * radius, pitch);
}

// 13. Catenary — y = a*cosh(x/a), the hanging chain
//
// The curve of a uniform chain under its own weight, not a parabola. The shape
// parameter `a` is the ratio of horizontal tension to weight per unit length:
// small a is a deep sag, large a is nearly flat.
//
// This family takes span and sag and solves for a. The relation is
// transcendental:
//     sag = a * (cosh(span / (2a)) - 1)
// with no closed-form inverse in elementary functions, so it is inverted
// numerically by bisection. The parabolic approximation a ~ span^2 / (8*sag)
// only seeds the bracket: it is the leading term of the series, and at span
// 100 with sag 25 it gives a = 50 where the true a is 53.716 (a curve built on
// it hangs 8.6% too deep).
//
// Bisection rather than Newton because sag(a) is strictly decreasing on a > 0,
// so a bracket cannot be lost; Newton on the same function can walk off toward
// a = 0, where cosh overflows.
export const CATENARY_DEFAULTS = { span: 100, sag: 25, samples: 121, start: [0, 0, 0] };
// Solve sag = a*(cosh(span/(2a)) - 1) for a. Converges to a relative `tol`,
// because a scales with span.
export function catenaryParameter(span, sag, tol = 1e-14) {
  if (!(span > 0)) throw new Error(`catenaryCurve needs span > 0 (got ${span})`);
  if (!(sag > 0)) throw new Error(`catenaryCurve needs sag > 0 (got ${sag}) — a chain with no sag is a straight line under infinite tension, not a catenary`);
  const sagOf = (a) => a * (Math.cosh(span / (2 * a)) - 1);
  const seed = (span * span) / (8 * sag);   // parabolic approximation, a seed only
  let lo = seed, hi = seed;                 // sagOf is decreasing: lo is small-a/deep-sag
  for (let g = 0; g < 200 && sagOf(hi) > sag; g++) hi *= 2;
  for (let g = 0; g < 200 && sagOf(lo) < sag; g++) lo /= 2;
  for (let i = 0; i < 300; i++) {
    const mid = 0.5 * (lo + hi);
    if (sagOf(mid) > sag) lo = mid; else hi = mid;
    if (hi - lo <= tol * hi) break;
  }
  return 0.5 * (lo + hi);
}
// The default sample count is odd so that one sample lands exactly on the span
// midpoint, where the vertex is — the lowest point of the chain is then a
// point on the curve rather than on a chord.
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
    // y = a*cosh(u/a), shifted so the two suspension points sit at start[1]
    // and the chain hangs down from them to start[1] - sag.
    pts.push([p.start[0] + half + u, p.start[1] + a * Math.cosh(u / a) - top, p.start[2]]);
  }
  return pts;
}

// 14. Torus knot — the (p,q) knot, wound on a torus
//
//   x = (R + r*cos(q*t)) * cos(p*t)
//   y = (R + r*cos(q*t)) * sin(p*t)
//   z =      r*sin(q*t)                       t in [0, 2*pi)
//
// Over one period the curve goes round the main axis exactly p times and round
// the tube exactly q times, and every point satisfies the implicit equation of
// its torus:
//     (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2
// identically, since sqrt(x^2+y^2) - R = r*cos(q*t) and z = r*sin(q*t). The
// tests use that identity as the oracle; it holds to machine precision.
//
// gcd(p,q) must be 1. With gcd(p,q) = g > 1 the parametrization returns to its
// start after 2*pi/g and retraces the same points g times — the object is a
// link of g components, which a single point chain cannot represent. It is
// refused rather than emitted as g coincident copies, which downstream fits
// and offsets would fail on.
//
// (p,q) and (q,p) are the same knot type (a torus is symmetric in its two
// circles) but different curves in space, so both are offered.
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
    const t = (i / n) * 2 * Math.PI;     // not n-1: the last sample is not a repeat of the first
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
