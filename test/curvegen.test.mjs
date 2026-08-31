import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  LSYSTEM_PRESETS, countLSystemGrowth, expandLSystem, lSystemTurtle,
  MAX_LSYSTEM_SEGMENTS,
  lorenzDeriv, lorenzTrajectory, LORENZ_DEFAULTS,
  randomWalkCurve, pointDistance,
  WAVE_FORMS, WAVE_CURVE_DEFAULTS, waveUnit, waveCurve, waveWantsSmoothFit,
  NOISE_CURVE_DEFAULTS, fbm1D, noiseCurve,
  HARMONIC_DEFAULTS, harmonicCurve,
  ROULETTE_MODES, ROULETTE_DEFAULTS, rouletteClosingTurns, rouletteCurve,
  SUPERFORMULA_DEFAULTS, superformulaRadius, superformulaCurve,
  SPIRAL_KINDS, SPIRAL_DEFAULTS, spiralRadius, spiralCurve,
  LISSAJOUS_DEFAULTS, lissajousCloses, lissajousPeriodTurns, lissajousCurve,
  ROSE_DEFAULTS, roseThetaMax, roseCurve,
  HELIX_DEFAULTS, helixResolve, helixCurve, helixArcLength,
  CATENARY_DEFAULTS, catenaryParameter, catenaryCurve,
  TORUS_KNOT_DEFAULTS, torusKnotCurve,
} from '../kernel/curvegen.mjs';
import { valueNoise2D } from '../kernel/noise.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { decimateOpenToCount } from '../kernel/simplify.mjs';

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// ===========================================================================
// L-SYSTEM
// ===========================================================================

test('L-system Koch: segment count is EXACTLY 4^iterations (checkable growth law)', () => {
  const { axiom, rules } = LSYSTEM_PRESETS.koch;
  for (let n = 0; n <= 5; n++) {
    const expected = Math.pow(4, n); // each F -> F+F--F+F has exactly 4 F's
    const { fCount } = countLSystemGrowth(axiom, rules, n);
    assert.equal(fCount, expected, `Koch F count at n=${n}`);
    const str = expandLSystem(axiom, rules, n);
    const { segments } = lSystemTurtle(str, { angle: LSYSTEM_PRESETS.koch.angle, stepLength: 1 });
    assert.equal(segments.length, expected, `Koch turtle segments at n=${n}`);
  }
});

test('L-system Dragon: segment count is EXACTLY 2^iterations', () => {
  const { axiom, rules, angle } = LSYSTEM_PRESETS.dragon;
  for (let n = 0; n <= 8; n++) {
    const expected = Math.pow(2, n);
    const { fCount } = countLSystemGrowth(axiom, rules, n);
    assert.equal(fCount, expected, `Dragon F count at n=${n}`);
    const str = expandLSystem(axiom, rules, n);
    const { segments } = lSystemTurtle(str, { angle, stepLength: 1 });
    assert.equal(segments.length, expected, `Dragon turtle segments at n=${n}`);
  }
});

test('L-system Koch shape sanity: axiom+1 iteration reproduces the classic F+F--F+F bump', () => {
  // One iteration of Koch on axiom "F" is literally "F+F--F+F" — 4 unit
  // segments, start at origin, and (with the classic 60-degree turn) the
  // curve ends exactly at (3,0) — the standard Koch generator spans 3 units.
  const str = expandLSystem('F', { F: 'F+F--F+F' }, 1);
  assert.equal(str, 'F+F--F+F');
  const { segments, polyline } = lSystemTurtle(str, { angle: 60, stepLength: 1 });
  assert.equal(segments.length, 4);
  assert.ok(dist(polyline[0], [0, 0, 0]) < 1e-9, 'starts at origin');
  const end = polyline[polyline.length - 1];
  assert.ok(dist(end, [3, 0, 0]) < 1e-9, `Koch generator ends at (3,0), got ${end}`);
  // The middle bump genuinely rises above the baseline (the peak has y>0).
  const maxY = Math.max(...polyline.map((p) => p[1]));
  assert.ok(maxY > 0.5, `Koch bump rises above baseline, peak y=${maxY}`);
});

test('L-system branching stack: push-then-pop returns the turtle to the EXACT pushed state (position AND heading)', () => {
  // "F[+F]F": draw a unit F (0,0)->(1,0); push at (1,0) heading +X; +F draws
  // a turned branch; pop back to (1,0) heading +X; final F must then run
  // (1,0)->(2,0). If the pop failed to restore POSITION, segment 3 would not
  // start at (1,0). If it failed to restore HEADING, segment 3 would not run
  // along +X. Both are checked.
  const { segments } = lSystemTurtle('F[+F]F', { angle: 90, stepLength: 1 });
  assert.equal(segments.length, 3, 'three F draws');
  const third = segments[2];
  assert.ok(dist(third[0], [1, 0, 0]) < 1e-9, `3rd segment starts at pushed position, got ${third[0]}`);
  assert.ok(dist(third[1], [2, 0, 0]) < 1e-9, `3rd segment runs along restored heading, got ${third[1]}`);
  // The branch segment (2nd F, after +90) genuinely went in a different (turned) direction.
  const branch = segments[1];
  const bdir = [branch[1][0] - branch[0][0], branch[1][1] - branch[0][1], branch[1][2] - branch[0][2]];
  assert.ok(Math.abs(bdir[1]) > 0.5, `branch turned off the baseline, dir=${bdir}`);
});

test('L-system plant preset uses the branch stack and produces a real figure', () => {
  const { axiom, rules, angle } = LSYSTEM_PRESETS.plant;
  const str = expandLSystem(axiom, rules, 3);
  assert.ok(str.includes('['), 'plant expansion contains branch pushes');
  const { segments, polyline } = lSystemTurtle(str, { angle, stepLength: 1 });
  const { fCount } = countLSystemGrowth(axiom, rules, 3);
  assert.equal(segments.length, fCount, 'segment count matches F count exactly');
  assert.ok(polyline.length >= 2, 'a real polyline came out');
  assert.ok(polyline.every((p) => p.every(Number.isFinite)), 'no NaN/Inf');
});

test('L-system sanity cap: an over-cap iteration count refuses honestly (throws), does not build', () => {
  const { axiom, rules } = LSYSTEM_PRESETS.koch; // 4^n grows fast
  // 4^8 = 65536 > MAX_LSYSTEM_SEGMENTS (8000) -> must refuse.
  assert.ok(Math.pow(4, 8) > MAX_LSYSTEM_SEGMENTS);
  assert.throws(() => expandLSystem(axiom, rules, 8), /sanity cap/);
  // A within-cap iteration count still works.
  assert.doesNotThrow(() => expandLSystem(axiom, rules, 5));
});

test('L-system polyline feeds cleanly into globalCurveInterp (degree 1, exact polyline)', () => {
  const str = expandLSystem('F', { F: 'F+F--F+F' }, 2);
  const { polyline } = lSystemTurtle(str, { angle: 60, stepLength: 1 });
  const crv = globalCurveInterp(polyline, 1);
  assert.equal(crv.degree, 1);
  assert.ok(crv.ctrlPts.every((p) => p.every(Number.isFinite)), 'finite control net');
  // Degree-1 interpolation IS the polyline: control points equal the input points.
  assert.equal(crv.ctrlPts.length, polyline.length);
});

// ===========================================================================
// LORENZ
// ===========================================================================

test('Lorenz RK4: trajectory stays BOUNDED within the classic attractor region', () => {
  const pts = lorenzTrajectory({ ...LORENZ_DEFAULTS, steps: 5000, dt: 0.01 });
  // Skip the initial transient (first 200 steps) as the point spirals onto
  // the attractor; then it stays inside the well-known butterfly bounds.
  let maxX = 0, maxY = 0, minZ = Infinity, maxZ = 0;
  for (let i = 200; i < pts.length; i++) {
    maxX = Math.max(maxX, Math.abs(pts[i][0]));
    maxY = Math.max(maxY, Math.abs(pts[i][1]));
    minZ = Math.min(minZ, pts[i][2]);
    maxZ = Math.max(maxZ, pts[i][2]);
  }
  assert.ok(pts.every((p) => p.every(Number.isFinite)), 'no NaN/Inf anywhere');
  assert.ok(maxX < 30, `|x| stays bounded (<30), got ${maxX}`);
  assert.ok(maxY < 40, `|y| stays bounded (<40), got ${maxY}`);
  assert.ok(minZ > -1 && maxZ < 60, `z in ~[0,50] region, got [${minZ},${maxZ}]`);
  // It genuinely explores the attractor (not a fixed point): real spread.
  assert.ok(maxX > 10 && maxZ > 25, 'trajectory genuinely explores the butterfly');
});

test('Lorenz: RK4 stays bounded where naive forward-Euler visibly blows up (the integrator choice matters)', () => {
  // At a coarse dt the naive Euler scheme (built from the SAME shared
  // derivative) genuinely diverges to infinity while RK4 stays bounded on
  // the true attractor.
  const { sigma, rho, beta } = LORENZ_DEFAULTS;
  const dt = 0.03, steps = 4000, start = [0.1, 0, 0];
  const rk4 = lorenzTrajectory({ sigma, rho, beta, dt, steps, start });
  let rk4Max = 0;
  for (const p of rk4) rk4Max = Math.max(rk4Max, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));

  // Naive forward Euler from the identical right-hand side.
  let s = start.slice();
  let eulerMax = 0;
  for (let i = 0; i < steps; i++) {
    const d = lorenzDeriv(s, sigma, rho, beta);
    s = [s[0] + d[0] * dt, s[1] + d[1] * dt, s[2] + d[2] * dt];
    eulerMax = Math.max(eulerMax, Math.abs(s[0]), Math.abs(s[1]), Math.abs(s[2]));
  }
  assert.ok(Number.isFinite(rk4Max) && rk4Max < 60, `RK4 stays bounded, max coord ${rk4Max}`);
  assert.ok(!Number.isFinite(eulerMax), `naive Euler blows up (diverges to infinity), max coord ${eulerMax}`);
});

test('Lorenz: two different start points (seeds) produce genuinely different trajectories (chaos)', () => {
  const a = lorenzTrajectory({ ...LORENZ_DEFAULTS, start: [0.1, 0, 0], steps: 3000 });
  const b = lorenzTrajectory({ ...LORENZ_DEFAULTS, start: [5, 5, 5], steps: 3000 });
  assert.equal(a.length, b.length);
  assert.notEqual(JSON.stringify(a), JSON.stringify(b), 'different starts give different curves');
  const endGap = dist(a[a.length - 1], b[b.length - 1]);
  assert.ok(endGap > 1, `trajectories are genuinely different (end gap ${endGap})`);

  // Sensitive dependence: even a TINY (1e-5) perturbation grows to attractor
  // scale given enough time (~50 time units) — the defining chaos property.
  const base = lorenzTrajectory({ ...LORENZ_DEFAULTS, start: [0.1, 0, 0], steps: 6000 });
  const pert = lorenzTrajectory({ ...LORENZ_DEFAULTS, start: [0.10001, 0, 0], steps: 6000 });
  let maxGap = 0;
  for (let i = 0; i < base.length; i++) maxGap = Math.max(maxGap, dist(base[i], pert[i]));
  assert.ok(maxGap > 10, `a 1e-5 perturbation grows to attractor scale (max gap ${maxGap})`);
});

test('Lorenz: dense trajectory decimates + feeds cleanly into globalCurveInterp', () => {
  const pts = lorenzTrajectory({ ...LORENZ_DEFAULTS, steps: 3000 });
  const decimated = decimateOpenToCount(pts, 120);
  assert.equal(decimated.length, 120);
  const crv = globalCurveInterp(decimated, 3);
  assert.equal(crv.degree, 3);
  assert.ok(crv.ctrlPts.every((p) => p.every(Number.isFinite)), 'finite control net from Lorenz curve');
});

// ===========================================================================
// RANDOM WALK
// ===========================================================================

test('Random Walk: same seed twice produces a BIT-IDENTICAL curve (determinism)', () => {
  const a = randomWalkCurve({ seed: 7, stepCount: 150, stepLength: 2, roughness: 0.6 });
  const b = randomWalkCurve({ seed: 7, stepCount: 150, stepLength: 2, roughness: 0.6 });
  assert.equal(JSON.stringify(a), JSON.stringify(b), 'bit-identical for the same seed');
});

test('Random Walk: different seeds produce genuinely different curves', () => {
  const a = randomWalkCurve({ seed: 1, stepCount: 150, stepLength: 2, roughness: 0.6 });
  const b = randomWalkCurve({ seed: 2, stepCount: 150, stepLength: 2, roughness: 0.6 });
  assert.notEqual(JSON.stringify(a), JSON.stringify(b), 'different seeds differ');
  const endGap = dist(a[a.length - 1], b[b.length - 1]);
  assert.ok(endGap > 1e-6, `end points genuinely differ (${endGap})`);
});

test('Random Walk: roughness 0 is a straight line; roughness 1 is genuinely jagged', () => {
  const straight = randomWalkCurve({ seed: 3, stepCount: 100, stepLength: 1, roughness: 0 });
  // roughness 0 keeps the initial heading exactly -> a dead-straight line.
  const total = dist(straight[0], straight[straight.length - 1]);
  assert.ok(Math.abs(total - 100) < 1e-6, `roughness 0 gives a straight line of length 100, got ${total}`);

  const jagged = randomWalkCurve({ seed: 3, stepCount: 100, stepLength: 1, roughness: 1 });
  const jaggedEndToEnd = dist(jagged[0], jagged[jagged.length - 1]);
  // A fully-random walk's net displacement is far less than its 100-unit path.
  assert.ok(jaggedEndToEnd < 60, `roughness 1 wanders (net displacement ${jaggedEndToEnd} << 100 path)`);
});

test('Random Walk: feeds cleanly into globalCurveInterp', () => {
  const pts = randomWalkCurve({ seed: 42, stepCount: 60, stepLength: 3, roughness: 0.5 });
  const crv = globalCurveInterp(pts, 3);
  assert.equal(crv.degree, 3);
  assert.ok(crv.ctrlPts.every((p) => p.every(Number.isFinite)), 'finite control net from random walk');
});

// ===========================================================================
// DISTANCE
// ===========================================================================

test('pointDistance: correct Euclidean distance (3-4-5)', () => {
  assert.ok(Math.abs(pointDistance([0, 0, 0], [3, 4, 0]) - 5) < 1e-12);
  assert.ok(Math.abs(pointDistance([1, 2, 3], [1, 2, 3])) < 1e-12, 'zero distance for identical points');
  assert.ok(Math.abs(pointDistance([0, 0, 0], [0, 0, 10]) - 10) < 1e-12);
});

// ===========================================================================
// WAVE CURVE
// ===========================================================================

// A sampled family is only as trustworthy as the one period it repeats, so the
// unit shape is pinned at its own known stations BEFORE any placement, phase or
// damping machinery is allowed near it.
test('waveUnit: every waveform stays within -1..1 across a dense sweep', () => {
  for (const form of WAVE_FORMS) {
    for (let i = 0; i <= 2000; i++) {
      const v = waveUnit(form, i / 2000);
      assert.ok(v >= -1 - 1e-12 && v <= 1 + 1e-12, `${form} out of range at u=${i / 2000}: ${v}`);
    }
  }
});

/* ⭐ THE PHASE ALIGNMENT IS THE LOFT-CRITICAL PROPERTY, not a tidiness one.
   The whole point of four waveforms sharing one parameter block is that a
   reader can loft a sine section to a triangle section and get the surface the
   two profiles imply. If one form starts a quarter period out of step with the
   others, that surface twists and nothing on screen explains why. So every form
   is pinned to the SAME phase convention: rising through zero at u=0, peak at
   u=0.25, falling through zero at u=0.5. */
test('waveUnit: all four forms rise through zero at u=0 (the loft-critical alignment)', () => {
  // The three continuous forms genuinely pass through zero at u=0.
  for (const form of ['sine', 'triangle', 'sawtooth']) {
    assert.ok(Math.abs(waveUnit(form, 0)) < 1e-12, `${form} should start at zero, got ${waveUnit(form, 0)}`);
  }
  // Square is discontinuous, so "rising through zero at 0" is its rising EDGE:
  // high immediately after u=0, low immediately before it.
  assert.equal(waveUnit('square', 0), 1);
  assert.equal(waveUnit('square', 0.999), -1);
  // Each form rises INTO its first half rather than falling out of it — the
  // property that actually stops a loft between two forms from twisting.
  for (const form of WAVE_FORMS) {
    assert.ok(waveUnit(form, 0.01) > 0, `${form} should be rising just after u=0`);
    assert.ok(waveUnit(form, 0.99) < 0, `${form} should be below the baseline just before it wraps`);
  }
});

/* WHERE EACH FORM PEAKS IS A PROPERTY OF THE FORM, and flattening all four into
   one rule would be a false claim rather than a tidy one. A sine and a triangle
   turn over at the quarter point. A SAWTOOTH ramps all the way to its
   discontinuity, so its peak is approached at u -> 0.5 from below and is never
   attained at 0.25 (it sits at exactly half height there). A SQUARE is at full
   height across its whole first half. Each is pinned as what it is. */
test('waveUnit: each form reaches full height where its own shape says it does', () => {
  for (const form of ['sine', 'triangle']) {
    assert.ok(Math.abs(waveUnit(form, 0.25) - 1) < 1e-9, `${form} should peak at u=0.25, got ${waveUnit(form, 0.25)}`);
    assert.ok(Math.abs(waveUnit(form, 0.5)) < 1e-12, `${form} should cross zero at u=0.5`);
  }
  assert.equal(waveUnit('square', 0.25), 1, 'square is at full height across its first half');
  // Sawtooth: exactly half height at the quarter point, and arbitrarily close
  // to full height just before the step.
  assert.ok(Math.abs(waveUnit('sawtooth', 0.25) - 0.5) < 1e-12, 'sawtooth is mid-ramp at u=0.25');
  assert.ok(waveUnit('sawtooth', 0.4999) > 0.999, 'sawtooth approaches full height at its discontinuity');
  assert.equal(waveUnit('sawtooth', 0.5), -1, 'and steps to full negative height at it');
  // No form ever exceeds unit height, which is what makes `amplitude` mean
  // what the label says.
  for (const form of WAVE_FORMS) {
    for (let i = 0; i <= 4000; i++) {
      assert.ok(Math.abs(waveUnit(form, i / 4000)) <= 1 + 1e-12, `${form} exceeds unit height`);
    }
  }
});

test('waveUnit: one period exactly — u and u+1 agree for every form', () => {
  for (const form of WAVE_FORMS) {
    for (const u of [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9]) {
      assert.ok(Math.abs(waveUnit(form, u) - waveUnit(form, u + 1)) < 1e-12, `${form} not periodic at ${u}`);
      assert.ok(Math.abs(waveUnit(form, u) - waveUnit(form, u + 5)) < 1e-12, `${form} not periodic at ${u}+5`);
    }
  }
});

/* THE ORACLE FOR THE PLACED CURVE: with one cycle across the run and a sample
   landing exactly on the quarter point, the extreme y is EXACTLY the amplitude
   — not approximately. `samples` is chosen so (n-1)/4 is a whole number, which
   is what puts a sample on the peak; at any other count the curve is still
   correct and the measured peak is merely the nearest sample to it. */
test('waveCurve: cycles=1 reaches EXACTLY +/- amplitude where the form allows it', () => {
  // `samples: 201` puts a sample exactly on the quarter point ((n-1)/4 whole),
  // which is what makes these exact rather than nearly-exact.
  for (const form of ['sine', 'triangle', 'square']) {
    const ys = waveCurve({ form, cycles: 1, amplitude: 25, samples: 201, offset: 0 }).map((q) => q[1]);
    assert.ok(Math.abs(Math.max(...ys) - 25) < 1e-9, `${form} max ${Math.max(...ys)}`);
    assert.ok(Math.abs(Math.min(...ys) + 25) < 1e-9, `${form} min ${Math.min(...ys)}`);
  }
  /* SAWTOOTH IS THE HONEST EXCEPTION, asserted rather than excused: its peak
     lives at a discontinuity it never lands on, so the highest SAMPLE sits one
     step below full amplitude and gets closer as the sampling gets denser. The
     trough IS attained, because the step lands on it exactly. */
  const coarse = waveCurve({ form: 'sawtooth', cycles: 1, amplitude: 25, samples: 201 }).map((q) => q[1]);
  const fine = waveCurve({ form: 'sawtooth', cycles: 1, amplitude: 25, samples: 2001 }).map((q) => q[1]);
  assert.ok(Math.max(...coarse) < 25, 'sawtooth never overshoots its amplitude');
  assert.ok(Math.max(...fine) > Math.max(...coarse), 'denser sampling gets closer to the true peak');
  assert.ok(Math.max(...fine) > 24.9, `dense sawtooth should approach 25, got ${Math.max(...fine)}`);
  assert.ok(Math.abs(Math.min(...coarse) + 25) < 1e-9, 'sawtooth attains its trough exactly');
});

test('waveCurve: the run spans exactly `length`, and `offset` moves the baseline', () => {
  const pts = waveCurve({ length: 100, samples: 51, start: [7, 3, -2] });
  assert.ok(Math.abs(pts[0][0] - 7) < 1e-12, 'starts at start.x');
  assert.ok(Math.abs(pts[pts.length - 1][0] - 107) < 1e-12, 'ends exactly one `length` along');
  assert.ok(pts.every((q) => Math.abs(q[2] + 2) < 1e-12), 'stays in its own plane');
  const lifted = waveCurve({ length: 100, samples: 51, start: [7, 3, -2], offset: 40 });
  for (let i = 0; i < pts.length; i++) {
    assert.ok(Math.abs((lifted[i][1] - pts[i][1]) - 40) < 1e-12, 'offset is a pure baseline shift');
  }
});

test('waveCurve: phase 360 is one whole period — identical to phase 0', () => {
  const a = waveCurve({ phase: 0, samples: 101 });
  const b = waveCurve({ phase: 360, samples: 101 });
  for (let i = 0; i < a.length; i++) assert.ok(dist(a[i], b[i]) < 1e-9, `phase wrap differs at ${i}`);
  // ...and a quarter-turn of phase genuinely moves the curve, so the param is
  // not silently inert.
  const q = waveCurve({ phase: 90, samples: 101 });
  assert.ok(a.some((pt, i) => dist(pt, q[i]) > 1e-6), 'phase 90 should change the curve');
});

test('waveCurve: damping decays later peaks, and damping 0 does nothing', () => {
  const flat = waveCurve({ damping: 0, cycles: 4, amplitude: 10, samples: 401 });
  assert.ok(Math.abs(Math.max(...flat.map((q) => q[1])) - 10) < 1e-9, 'undamped keeps full amplitude');
  const damped = waveCurve({ damping: 1, cycles: 4, amplitude: 10, samples: 401 });
  const firstHalf = Math.max(...damped.slice(0, 200).map((q) => q[1]));
  const lastHalf = Math.max(...damped.slice(200).map((q) => q[1]));
  assert.ok(lastHalf < firstHalf, `damped wave should shrink along the run (${firstHalf} -> ${lastHalf})`);
  assert.ok(firstHalf <= 10 + 1e-9, 'damping never AMPLIFIES');
});

/* SKEW CHANGES SHAPE WITHOUT CHANGING FREQUENCY. That is the claim waveCurve's
   own comment makes, and it is the one worth pinning: a skew that quietly
   altered the period would read as a broken frequency slider. Counting zero
   crossings is the frequency measurement that does not depend on the shape. */
test('waveCurve: skew reshapes the wave but preserves its period', () => {
  const zeroCrossings = (pts) => {
    let n = 0;
    for (let i = 1; i < pts.length; i++) if ((pts[i - 1][1] < 0) !== (pts[i][1] < 0)) n++;
    return n;
  };
  const even = waveCurve({ form: 'triangle', skew: 0.5, cycles: 3, samples: 601 });
  const lean = waveCurve({ form: 'triangle', skew: 0.15, cycles: 3, samples: 601 });
  assert.equal(zeroCrossings(lean), zeroCrossings(even), 'skew must not change the period');
  assert.ok(even.some((pt, i) => dist(pt, lean[i]) > 1e-6), 'skew should genuinely reshape the wave');
});

test('waveWantsSmoothFit: only the continuous forms are fitted smooth', () => {
  assert.equal(waveWantsSmoothFit('sine'), true);
  assert.equal(waveWantsSmoothFit('triangle'), true);
  assert.equal(waveWantsSmoothFit('square'), false, 'a degree-3 fit of a step rings at every edge');
  assert.equal(waveWantsSmoothFit('sawtooth'), false);
});

test('waveCurve: an unknown waveform falls back to sine rather than throwing', () => {
  const bogus = waveCurve({ form: 'not-a-waveform', samples: 51 });
  const sine = waveCurve({ form: 'sine', samples: 51 });
  for (let i = 0; i < sine.length; i++) assert.ok(dist(bogus[i], sine[i]) < 1e-12);
});

test('waveCurve: defaults are sane and it feeds cleanly into globalCurveInterp', () => {
  assert.equal(WAVE_CURVE_DEFAULTS.form, 'sine');
  const pts = waveCurve();
  assert.equal(pts.length, WAVE_CURVE_DEFAULTS.samples);
  assert.ok(pts.every((q) => q.every(Number.isFinite)), 'finite wave chain');
  const crv = globalCurveInterp(decimateOpenToCount(pts, 120), 3);
  assert.equal(crv.degree, 3);
  assert.ok(crv.ctrlPts.every((q) => q.every(Number.isFinite)), 'finite control net from wave');
});

// ===========================================================================
// NOISE CURVE (fBm)
// ===========================================================================

/* ⭐ THE CONVENTION OF THE NOISE WE ARE HANDED, ASSERTED RATHER THAN ASSUMED.
   kernel/noise.mjs's latticeVal is `2 * hash01 - 1`, so valueNoise2D already
   returns -1..1. The habitual `* 2 - 1` on top of a [0,1] noise would push the
   fBm sum to -3..1 and bias every curve downward -- which reads as "that is
   just what noise looks like" rather than as a defect, and is exactly why this
   is pinned at the source instead of being taken on trust. */
test('valueNoise2D returns -1..1, which is the convention fbm1D is built on', () => {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < 4000; i++) {
    const v = valueNoise2D((i % 97) * 0.31, Math.floor(i / 97) * 0.27, 7);
    lo = Math.min(lo, v); hi = Math.max(hi, v);
  }
  assert.ok(lo >= -1 - 1e-12 && hi <= 1 + 1e-12, `valueNoise2D out of -1..1: ${lo}..${hi}`);
  // ...and it genuinely USES the negative half. A [0,1] noise would pass the
  // bound above and fail this, which is the discrimination that matters.
  assert.ok(lo < -0.2, `valueNoise2D never goes meaningfully negative (min ${lo}) -- it is not a -1..1 field`);
  assert.ok(hi > 0.2, `valueNoise2D never goes meaningfully positive (max ${hi})`);
});

test('fbm1D stays within -1..1 at EVERY octave count, and stays centered', () => {
  for (let oct = 1; oct <= 8; oct++) {
    let lo = Infinity, hi = -Infinity, sum = 0, n = 0;
    for (let i = 0; i < 2000; i++) {
      const v = fbm1D(i * 0.017, 0, 3, oct, 1, 2, 0.5, valueNoise2D);
      lo = Math.min(lo, v); hi = Math.max(hi, v); sum += v; n++;
    }
    assert.ok(lo >= -1 - 1e-12 && hi <= 1 + 1e-12, `fbm1D octaves=${oct} out of range: ${lo}..${hi}`);
    // THE DOWNWARD BIAS IS THE ACTUAL DEFECT THIS GUARDS. A stray `* 2 - 1`
    // leaves the mean near -0.5 while the range check above can still pass.
    assert.ok(Math.abs(sum / n) < 0.25, `fbm1D octaves=${oct} is biased off-center (mean ${(sum / n).toFixed(3)})`);
  }
});

test('fbm1D: persistence 0 is one octave, and more octaves add detail not amplitude', () => {
  const one = fbm1D(0.37, 0, 5, 1, 1, 2, 0.5, valueNoise2D);
  const quiet = fbm1D(0.37, 0, 5, 6, 1, 2, 0, valueNoise2D);
  assert.ok(Math.abs(one - quiet) < 1e-12, 'persistence 0 silences every octave after the first');
  // Six octaves must differ from one -- otherwise the octave knob does nothing.
  const many = fbm1D(0.37, 0, 5, 6, 1, 2, 0.5, valueNoise2D);
  assert.ok(Math.abs(many - one) > 1e-9, 'more octaves should change the value');
});

test('noiseCurve: deterministic from its seed, and seed-sensitive', () => {
  const a = noiseCurve({ seed: 4, samples: 60 }, valueNoise2D);
  const b = noiseCurve({ seed: 4, samples: 60 }, valueNoise2D);
  assert.deepEqual(a, b, 'same seed must give a bit-identical chain');
  const c = noiseCurve({ seed: 5, samples: 60 }, valueNoise2D);
  assert.ok(a.some((p, i) => dist(p, c[i]) > 1e-9), 'a different seed must give a different curve');
});

test('noiseCurve: refuses honestly when handed no noise function', () => {
  assert.throws(() => noiseCurve({ samples: 10 }), /needs a 2D noise function/);
});

test('noiseCurve OPEN: spans `length`, and amplitude bounds the displacement', () => {
  const amp = 12;
  const pts = noiseCurve({ samples: 200, length: 100, amplitude: amp, start: [5, 2, -1] }, valueNoise2D);
  assert.ok(Math.abs(pts[0][0] - 5) < 1e-12, 'starts at start.x');
  assert.ok(Math.abs(pts[pts.length - 1][0] - 105) < 1e-12, 'ends exactly one length along');
  for (const q of pts) {
    assert.ok(Math.abs(q[1] - 2) <= amp + 1e-9, `displacement ${q[1] - 2} exceeds amplitude ${amp}`);
    assert.ok(Math.abs(q[2] + 1) < 1e-12, 'stays in its own plane');
  }
  // ...and genuinely USES the amplitude, in BOTH directions -- a downward bias
  // would keep it in range while never rising above the baseline.
  const ys = pts.map((q) => q[1] - 2);
  assert.ok(Math.max(...ys) > 0.2 * amp, `noise never rises (max ${Math.max(...ys).toFixed(2)})`);
  assert.ok(Math.min(...ys) < -0.2 * amp, `noise never falls (min ${Math.min(...ys).toFixed(2)}) -- the -1..1 convention is not holding`);
});

/* ⭐ THE WRAP SEGMENT IS PRESENT AND THE SEAM IS INVISIBLE. Two different
   claims, and a closed curve can fail either one alone: dropping the closing
   point leaves a gap that looks like a rendering artifact, and sampling the
   noise along a LINE instead of around a CIRCLE closes the geometry while
   leaving a visible crease in the shape exactly at the join. */
test('noiseCurve CLOSED: the ring closes exactly, and the noise closes with it', () => {
  const pts = noiseCurve({ samples: 240, closed: true, radius: 40, amplitude: 8, seed: 11 }, valueNoise2D);
  const first = pts[0], last = pts[pts.length - 1];
  assert.ok(dist(first, last) < 1e-12, `the wrap segment is missing (gap ${dist(first, last)})`);
  assert.equal(pts.length, 241, 'n samples plus the explicit closing repeat');
  /* THE SEAM IS NOT SPECIAL. If the noise were sampled along a line, the step
     across the join would be far larger than a typical step. Comparing the
     seam step against the median step is what makes this sensitive to a crease
     rather than merely to a gap. */
  const steps = [];
  for (let i = 1; i < pts.length; i++) steps.push(dist(pts[i - 1], pts[i]));
  const sorted = [...steps].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const seamStep = dist(pts[pts.length - 2], pts[0]);
  assert.ok(seamStep < median * 3, `the seam step ${seamStep.toFixed(4)} is far larger than the median ${median.toFixed(4)} -- the noise does not close`);
  // ...and the radius genuinely varies, or this is just a circle.
  const radii = pts.map((q) => Math.hypot(q[0], q[1]));
  assert.ok(Math.max(...radii) - Math.min(...radii) > 1, 'a closed noise ring must actually be noisy');
  assert.ok(Math.max(...radii) <= 48 + 1e-9 && Math.min(...radii) >= 32 - 1e-9, 'and stay within radius +/- amplitude');
});

test('noiseCurve: defaults are sane and it feeds cleanly into globalCurveInterp', () => {
  assert.equal(NOISE_CURVE_DEFAULTS.closed, false);
  const pts = noiseCurve({}, valueNoise2D);
  assert.equal(pts.length, NOISE_CURVE_DEFAULTS.samples);
  assert.ok(pts.every((q) => q.every(Number.isFinite)), 'finite noise chain');
  const crv = globalCurveInterp(decimateOpenToCount(pts, 120), 3);
  assert.equal(crv.degree, 3);
  assert.ok(crv.ctrlPts.every((q) => q.every(Number.isFinite)), 'finite control net from noise');
});

// ===========================================================================
// HARMONIC SERIES
// ===========================================================================

test('harmonicCurve: ONE term is exactly a sine of the requested amplitude', () => {
  const pts = harmonicCurve({ terms: 1, cycles: 1, amplitude: 25, samples: 201, phase: 0 });
  const ys = pts.map((q) => q[1]);
  assert.ok(Math.abs(Math.max(...ys) - 25) < 1e-9, `one-term max ${Math.max(...ys)}`);
  assert.ok(Math.abs(Math.min(...ys) + 25) < 1e-9, `one-term min ${Math.min(...ys)}`);
  // ...and it IS the sine, sample for sample, not merely the same height.
  const sine = waveCurve({ form: 'sine', cycles: 1, amplitude: 25, samples: 201, phase: 0 });
  for (let i = 0; i < pts.length; i++) assert.ok(dist(pts[i], sine[i]) < 1e-9, `one-term harmonic differs from a sine at ${i}`);
});

/* ⭐ THE TEACHING CLAIM, MEASURED. "Adding odd harmonics at 1/n walks visibly
   toward a square wave" is the reason this family exists, so it is asserted
   rather than described: the mean distance from a unit square must DECREASE
   monotonically as terms are added. A curve that merely got taller, or noisier,
   would fail this. */
test('harmonicCurve: odd harmonics at 1/n converge toward a square wave', () => {
  const squareness = (terms) => {
    const pts = harmonicCurve({ terms, oddOnly: true, falloff: 1, cycles: 1, amplitude: 1, samples: 801 });
    const ys = pts.map((q) => q[1]);
    const peak = Math.max(...ys.map(Math.abs));
    let err = 0;
    for (let i = 0; i < ys.length; i++) {
      const s = i / (ys.length - 1);
      const target = (s % 1) < 0.5 ? 1 : -1;   // the square this series converges to
      err += Math.abs(ys[i] / peak - target);
    }
    return err / ys.length;
  };
  const errs = [1, 3, 7, 15].map(squareness);
  for (let i = 1; i < errs.length; i++) {
    assert.ok(errs[i] < errs[i - 1], `adding terms should approach a square wave, got ${errs.map((e) => e.toFixed(3)).join(' -> ')}`);
  }
  assert.ok(errs[errs.length - 1] < errs[0] * 0.6, `15 terms should be markedly squarer than 1 (${errs[0].toFixed(3)} -> ${errs[errs.length - 1].toFixed(3)})`);
});

test('harmonicCurve: oddOnly and falloff both genuinely change the shape', () => {
  const base = harmonicCurve({ terms: 6, samples: 101 });
  const even = harmonicCurve({ terms: 6, oddOnly: false, samples: 101 });
  const steep = harmonicCurve({ terms: 6, falloff: 2, samples: 101 });
  assert.ok(base.some((p, i) => dist(p, even[i]) > 1e-9), 'oddOnly must change the series');
  assert.ok(base.some((p, i) => dist(p, steep[i]) > 1e-9), 'falloff must change the series');
  assert.equal(HARMONIC_DEFAULTS.oddOnly, true);
});

// ===========================================================================
// ROULETTE
// ===========================================================================

test('rouletteClosingTurns: the classical ratios close in the turns they should', () => {
  assert.equal(rouletteClosingTurns(40, 10), 1, 'R = 4r closes in one turn');
  assert.equal(rouletteClosingTurns(30, 10), 1, 'R = 3r closes in one turn');
  assert.equal(rouletteClosingTurns(50, 15), 3, 'R/r = 10/3 needs three turns');
  assert.equal(rouletteClosingTurns(0, 10), 0, 'a degenerate ratio reports "cannot say"');
});

/* ⭐ AN EXACT CLOSED FORM, NOT A RESEMBLANCE. A hypotrochoid with R = 4r and
   d = r IS the astroid: expanding cos3t/sin3t collapses the two terms to
   x = R cos^3 t, y = R sin^3 t, whose points satisfy
   |x|^(2/3) + |y|^(2/3) = R^(2/3) identically. Checking that identity at every
   sample is a real oracle; checking "it has four cusps" would not be. */
test('rouletteCurve: R = 4r with d = r is EXACTLY an astroid', () => {
  const R = 40;
  const pts = rouletteCurve({ mode: 'hypotrochoid', R, r: 10, d: 10, turns: 1, samples: 720 });
  const p23 = Math.pow(R, 2 / 3);
  for (const q of pts) {
    const v = Math.pow(Math.abs(q[0]), 2 / 3) + Math.pow(Math.abs(q[1]), 2 / 3);
    assert.ok(Math.abs(v - p23) < 1e-6, `off the astroid: ${v} vs ${p23}`);
  }
});

/* AND THE DEGENERATE CASE IS REAL GEOMETRY, not a failure to guard: a
   hypotrochoid with R = 2r and d = r collapses to a STRAIGHT LINE through the
   center (the Tusi couple). Worth pinning because a "that looks broken" guard
   added later would destroy a correct result. */
test('rouletteCurve: R = 2r with d = r is exactly a straight line (the Tusi couple)', () => {
  const pts = rouletteCurve({ mode: 'hypotrochoid', R: 20, r: 10, d: 10, turns: 1, samples: 400 });
  for (const q of pts) assert.ok(Math.abs(q[1]) < 1e-9, `should be flat in y, got ${q[1]}`);
  const xs = pts.map((q) => q[0]);
  assert.ok(Math.abs(Math.max(...xs) - 20) < 1e-9 && Math.abs(Math.min(...xs) + 20) < 1e-9, 'and span exactly +/-R');
});

/* THE CARDIOID, FROM ITS OWN CLOSED FORM RATHER THAN FROM A GUESS. For an
   epitrochoid the distance from the center satisfies
     rho^2 = (R+r)^2 + d^2 - 2(R+r)d*cos((R/r)*t)
   so with R = r = d the radius runs between |(R+r)-d| and (R+r)+d -- that is
   20..60 here, NOT 0..80. A first draft of this test asserted the cusp sat at
   the ORIGIN; it does not, it sits on the fixed circle at distance R, and the
   test was wrong rather than the curve. The cusp is pinned by what actually
   makes it a cusp: the tracing point comes momentarily to rest, so the step
   between samples collapses there. */
test('rouletteCurve: an epitrochoid with R = r = d is a cardioid, with a real cusp', () => {
  const R = 20, r = 20, d = 20;
  const pts = rouletteCurve({ mode: 'epitrochoid', R, r, d, turns: 1, samples: 720 });
  const radii = pts.map((q) => Math.hypot(q[0], q[1]));
  assert.ok(Math.abs(Math.min(...radii) - Math.abs((R + r) - d)) < 1e-6, `min radius ${Math.min(...radii)}, expected ${Math.abs((R + r) - d)}`);
  assert.ok(Math.abs(Math.max(...radii) - ((R + r) + d)) < 1e-6, `max radius ${Math.max(...radii)}, expected ${(R + r) + d}`);
  // The cusp itself: a vanishing step where a smooth curve would have none.
  const steps = [];
  for (let i = 1; i < pts.length; i++) steps.push(dist(pts[i - 1], pts[i]));
  const median = [...steps].sort((a, b) => a - b)[Math.floor(steps.length / 2)];
  assert.ok(Math.min(...steps) < median * 0.05, `no cusp found (min step ${Math.min(...steps).toFixed(5)} vs median ${median.toFixed(5)})`);
});

test('rouletteCurve: a closing ratio returns to its start; both modes are honoured', () => {
  const closed = rouletteCurve({ R: 40, r: 10, d: 7, turns: 0, samples: 360 });
  assert.ok(dist(closed[0], closed[closed.length - 1]) < 1e-9, 'turns:0 walks exactly as far as closure needs');
  const hypo = rouletteCurve({ mode: 'hypotrochoid', R: 40, r: 10, d: 7, turns: 1, samples: 90 });
  const epi = rouletteCurve({ mode: 'epitrochoid', R: 40, r: 10, d: 7, turns: 1, samples: 90 });
  assert.ok(hypo.some((p, i) => dist(p, epi[i]) > 1), 'the two modes must trace genuinely different curves');
  assert.deepEqual(ROULETTE_MODES, ['hypotrochoid', 'epitrochoid']);
  assert.equal(ROULETTE_DEFAULTS.turns, 0);
});

// ===========================================================================
// SUPERFORMULA (Gielis)
// ===========================================================================

/* ⭐ THE CLOSED FORM AGAIN: n1 = n2 = n3 = 2 with a = b = 1 reduces the radius
   to (cos^2 + sin^2)^(-1/2) = 1 for EVERY theta and every m, so the curve is an
   exact circle of `scale`. This is the strongest oracle in the family and it
   does not depend on m at all, which is itself worth asserting. */
test('superformulaRadius: n1=n2=n3=2 with a=b=1 is exactly 1 at every angle, for every m', () => {
  for (const m of [0, 3, 4, 6, 7, 12]) {
    for (let i = 0; i < 200; i++) {
      const th = (i / 200) * 2 * Math.PI;
      const r = superformulaRadius(th, { a: 1, b: 1, m, n1: 2, n2: 2, n3: 2 });
      assert.ok(Math.abs(r - 1) < 1e-12, `m=${m} theta=${th}: r=${r}, expected 1`);
    }
  }
});

test('superformulaCurve: the same settings give an exact circle of `scale`', () => {
  const pts = superformulaCurve({ a: 1, b: 1, m: 4, n1: 2, n2: 2, n3: 2, scale: 40, samples: 360 });
  for (const q of pts) assert.ok(Math.abs(Math.hypot(q[0], q[1]) - 40) < 1e-9, `radius ${Math.hypot(q[0], q[1])}`);
  assert.ok(dist(pts[0], pts[pts.length - 1]) < 1e-12, 'and it carries its wrap segment');
  assert.equal(pts.length, 361, 'n samples plus the explicit closing repeat');
});

/* ⚠⚠ THE CIRCLE ORACLE ABOVE CANNOT SEE `n1` AT ALL, and that is not a
   quibble: with n1=n2=n3=2 and a=b=1 the bracket is cos^2+sin^2 = 1, and 1
   raised to ANY power is 1 -- so the exponent could be `-1/n1`, `+1/n1` or
   anything else and every assertion above would still pass. Flipping that sign
   in the kernel was tried directly and reddened NOTHING. The fixture was
   sitting on the one symmetry that cancels the defect.
   So the exponent is pinned somewhere the bracket is NOT 1. With m=4 and
   n2=n3=4 at theta=pi/4 both terms are (sqrt(2)/2)^4 = 1/4, so the bracket is
   exactly 1/2 and the radius is exactly 2^(1/(2*... )) -- written below as the
   closed form 0.5^(-1/n1), which is >1 for the correct sign and <1 for the
   flipped one. */
test('superformulaRadius: the n1 exponent is pinned where the bracket is not 1', () => {
  const th = Math.PI / 4;
  for (const n1 of [0.5, 1, 2, 4]) {
    const r = superformulaRadius(th, { a: 1, b: 1, m: 4, n1, n2: 4, n3: 4 });
    const expected = Math.pow(0.5, -1 / n1);   // the bracket is exactly 1/2 here
    assert.ok(Math.abs(r - expected) < 1e-12, `n1=${n1}: r=${r}, expected ${expected}`);
    assert.ok(r > 1, `n1=${n1}: the correct exponent bulges OUTWARD here (r=${r}); a flipped sign gives ${Math.pow(0.5, 1 / n1)}`);
  }
  // ...and n1 genuinely moves the shape, so the parameter is not inert.
  const a = superformulaRadius(th, { a: 1, b: 1, m: 4, n1: 1, n2: 4, n3: 4 });
  const b = superformulaRadius(th, { a: 1, b: 1, m: 4, n1: 4, n2: 4, n3: 4 });
  assert.ok(Math.abs(a - b) > 0.1, `n1 must change the radius (${a} vs ${b})`);
});

test('superformulaRadius: a and b scale their own axes, and n2/n3 are not interchangeable', () => {
  // a stretches the cos term only -> the two half-axes stop agreeing.
  const wide = superformulaCurve({ a: 2, b: 1, m: 4, n1: 2, n2: 2, n3: 2, scale: 40, samples: 360 });
  const xs = wide.map((q) => Math.abs(q[0])), ys = wide.map((q) => Math.abs(q[1]));
  assert.ok(Math.max(...xs) > Math.max(...ys) + 1e-6, `a>b should widen x (${Math.max(...xs)} vs ${Math.max(...ys)})`);
  // n2 and n3 act on different terms, so swapping them is a real change.
  const p1 = superformulaRadius(Math.PI / 6, { a: 1, b: 1, m: 5, n1: 1, n2: 3, n3: 0.5 });
  const p2 = superformulaRadius(Math.PI / 6, { a: 1, b: 1, m: 5, n1: 1, n2: 0.5, n3: 3 });
  assert.ok(Math.abs(p1 - p2) > 1e-6, 'n2 and n3 must not be interchangeable');
});

test('superformulaCurve: m controls the symmetry — the shape repeats m times around', () => {
  // A 5-lobed form must be invariant under a 1/5 turn, and NOT under 1/4.
  const m = 5;
  const rAt = (th) => superformulaRadius(th, { a: 1, b: 1, m, n1: 0.3, n2: 1.7, n3: 1.7 });
  for (let i = 0; i < 50; i++) {
    const th = (i / 50) * 2 * Math.PI;
    assert.ok(Math.abs(rAt(th) - rAt(th + 2 * Math.PI / m)) < 1e-9, `not ${m}-fold symmetric at ${th}`);
  }
  let differs = false;
  for (let i = 0; i < 50; i++) {
    const th = (i / 50) * 2 * Math.PI;
    if (Math.abs(rAt(th) - rAt(th + 2 * Math.PI / 4)) > 1e-6) { differs = true; break; }
  }
  assert.ok(differs, 'a 5-lobed form must NOT also be 4-fold symmetric');
});

test('superformulaCurve: a degenerate radius is refused as 0 rather than NaN', () => {
  // n2/n3 large with a tiny a/b drives the sum to Infinity; the guard returns 0.
  const r = superformulaRadius(0, { a: 1, b: 1, m: 4, n1: 1, n2: 1, n3: 1 });
  assert.ok(Number.isFinite(r), 'a normal radius is finite');
  const pts = superformulaCurve({ a: 1e-9, b: 1e-9, m: 4, n1: 1, n2: 20, n3: 20, scale: 40, samples: 60 });
  assert.ok(pts.every((q) => q.every(Number.isFinite)), 'no NaN reaches the point chain');
  assert.equal(SUPERFORMULA_DEFAULTS.m, 6);
});

// ===========================================================================
// SPIRAL — archimedean / logarithmic / fermat
// ===========================================================================

/* Sample the radius at the exact turn boundaries. `turns * perTurn + 1`
   samples put a point on theta = 2*pi*k for every whole k, so the per-turn
   quantities below are read off the curve rather than recomputed from the
   formula the curve used. */
const turnRadii = (params, turns, perTurn = 720) => {
  const pts = spiralCurve({ ...params, turns, samples: turns * perTurn + 1 });
  const out = [];
  for (let k = 0; k <= turns; k++) {
    const q = pts[k * perTurn];
    out.push(Math.hypot(q[0], q[1]));
  }
  return out;
};
const spreadRatio = (xs) => Math.max(...xs) / Math.min(...xs);

/* ⭐ THE EXACT LAW, AND THE ONE THAT SEPARATES THE KINDS. An Archimedean
   spiral gains EXACTLY 2*pi*b of radius per turn — r(theta+2pi) - r(theta) =
   b*2*pi identically, independent of theta. Nothing else in the family does
   that, which is why the same fixture is then run against the ratio law and
   required to FAIL it. */
test('spiralCurve archimedean: successive turns differ by EXACTLY 2*pi*growth', () => {
  const a = 5, b = 2;
  const r = turnRadii({ kind: 'archimedean', startRadius: a, growth: b }, 5);
  assert.equal(r.length, 6);
  assert.ok(Math.abs(r[0] - a) < 1e-9, `starts at startRadius, got ${r[0]}`);
  const expected = 2 * Math.PI * b;
  for (let k = 1; k < r.length; k++) {
    assert.ok(Math.abs((r[k] - r[k - 1]) - expected) < 1e-9, `turn ${k}: gained ${r[k] - r[k - 1]}, expected ${expected}`);
  }
  // ...and it is NOT the logarithmic law: the RATIOS are all over the place.
  const ratios = r.slice(1).map((v, i) => v / r[i]);
  assert.ok(spreadRatio(ratios) > 2, `an Archimedean spiral must NOT have a constant ratio (ratios ${ratios.map((x) => x.toFixed(2)).join(', ')})`);
});

/* ⭐ AND THE MIRROR CLAIM. A logarithmic spiral is self-similar, so each turn
   multiplies the radius by exactly e^(2*pi*b). Run against the spacing law
   above it must fail — the two tests together are what make either of them
   evidence about WHICH kind was built, rather than merely that a spiral was. */
test('spiralCurve logarithmic: successive turns differ by a constant RATIO e^(2*pi*growth)', () => {
  const a = 5, b = 0.15;
  const r = turnRadii({ kind: 'logarithmic', startRadius: a, growth: b }, 5);
  assert.ok(Math.abs(r[0] - a) < 1e-9, `starts at startRadius, got ${r[0]}`);
  const expected = Math.exp(2 * Math.PI * b);
  for (let k = 1; k < r.length; k++) {
    assert.ok(Math.abs(r[k] / r[k - 1] - expected) < 1e-9, `turn ${k}: ratio ${r[k] / r[k - 1]}, expected ${expected}`);
  }
  // ...and it is NOT the Archimedean law: the per-turn GAINS grow without bound.
  const gains = r.slice(1).map((v, i) => v - r[i]);
  assert.ok(spreadRatio(gains) > 2, `a logarithmic spiral must NOT have constant spacing (gains ${gains.map((x) => x.toFixed(2)).join(', ')})`);
});

/* Fermat's own closed form, which is neither of the other two: r^2 = a^2 *
   theta, so r^2 is LINEAR in theta. Checked at the turn boundaries where
   theta is known exactly, and required to fail both other laws. */
test('spiralCurve fermat: r^2 is exactly linear in theta, and it starts at the origin', () => {
  const a = 8;
  const r = turnRadii({ kind: 'fermat', startRadius: a, growth: 0 }, 5);
  assert.ok(Math.abs(r[0]) < 1e-12, `a Fermat spiral begins at r=0, got ${r[0]}`);
  for (let k = 1; k < r.length; k++) {
    const theta = 2 * Math.PI * k;
    assert.ok(Math.abs(r[k] * r[k] - a * a * theta) < 1e-9, `turn ${k}: r^2 ${r[k] * r[k]}, expected ${a * a * theta}`);
  }
  /* Neither of the other two laws holds, and the exact reason is available:
     r_k = a*sqrt(2*pi*k), so the per-turn RATIO is sqrt((k+1)/k) — a real
     closed form, decreasing, and therefore not the constant the logarithmic
     spiral has. The gains sqrt(k+1) - sqrt(k) shrink for the same reason. */
  const gains = r.slice(1).map((v, i) => v - r[i]);
  const ratios = r.slice(2).map((v, i) => v / r[i + 1]);   // r[0] is 0, so start the ratios at turn 1
  for (let i = 0; i < ratios.length; i++) {
    assert.ok(Math.abs(ratios[i] - Math.sqrt((i + 2) / (i + 1))) < 1e-9, `turn ratio ${i}: ${ratios[i]}, expected ${Math.sqrt((i + 2) / (i + 1))}`);
  }
  assert.ok(spreadRatio(gains) > 1.5, 'a Fermat spiral has neither constant spacing...');
  assert.ok(spreadRatio(ratios) > 1.2, '...nor a constant ratio');
  // `growth` is genuinely unread by this kind, which the formula says: r = a*sqrt(theta).
  const other = turnRadii({ kind: 'fermat', startRadius: a, growth: 99 }, 2);
  assert.ok(Math.abs(other[1] - r[1]) < 1e-9, 'r = a*sqrt(theta) has no b in it');
});

test('spiralCurve: `height` makes it conical, and z is linear over the run', () => {
  const flat = spiralCurve({ kind: 'archimedean', turns: 3, height: 0, samples: 100, start: [0, 0, 7] });
  assert.ok(flat.every((q) => q[2] === 7), 'height 0 is planar at start.z');
  const cone = spiralCurve({ kind: 'archimedean', turns: 3, height: 30, samples: 101, start: [0, 0, 7] });
  assert.ok(Math.abs(cone[0][2] - 7) < 1e-12, 'starts at start.z');
  assert.ok(Math.abs(cone[100][2] - 37) < 1e-12, `ends exactly height above it, got ${cone[100][2]}`);
  for (let i = 0; i < cone.length; i++) {
    assert.ok(Math.abs(cone[i][2] - (7 + 30 * i / 100)) < 1e-12, `z is linear in the parameter at ${i}`);
  }
  // The radius still grows, so this really is a cone rather than a cylinder.
  assert.ok(Math.hypot(cone[100][0], cone[100][1]) > Math.hypot(cone[0][0], cone[0][1]) + 1, 'a conical spiral still opens out');
});

test('spiralCurve: an overflowing logarithmic spiral is refused by name, not emitted as Infinity', () => {
  assert.throws(() => spiralCurve({ kind: 'logarithmic', turns: 40, growth: 3, startRadius: 5 }), /overflows/);
  // The neighboring case that DOES fit is not refused, so the guard is not just "big numbers are scary".
  const ok = spiralCurve({ kind: 'logarithmic', turns: 40, growth: 1, startRadius: 5, samples: 200 });
  assert.ok(ok.every((q) => q.every(Number.isFinite)), 'e^251 is representable and must be produced');
  assert.deepEqual(SPIRAL_KINDS, ['archimedean', 'logarithmic', 'fermat']);
  assert.equal(SPIRAL_DEFAULTS.height, 0, 'planar by default');
  // An unknown kind falls back to archimedean rather than emitting NaN.
  const fallback = spiralCurve({ kind: 'nonsense', turns: 2, samples: 50 });
  const arch = spiralCurve({ kind: 'archimedean', turns: 2, samples: 50 });
  assert.ok(fallback.every((q, i) => dist(q, arch[i]) < 1e-12), 'unknown kind falls back to archimedean');
  assert.ok(Number.isFinite(spiralRadius('archimedean', 0, 5, 2)));
});

// ===========================================================================
// LISSAJOUS
// ===========================================================================

/* ⭐ THE DEGENERATE 1:1 CASE, WHICH IS A REAL RESULT AND NOT A FAILURE. With
   equal frequencies and zero phase both coordinates are the SAME sinusoid, so
   the figure collapses to the straight diagonal y/B = x/A traced back and
   forth. Worth pinning precisely because a later "the curve is degenerate,
   guard it" instinct would destroy a correct answer. */
test('lissajousCurve: equal frequencies at phase 0 are EXACTLY the straight diagonal', () => {
  const A = 40, B = 25;
  const pts = lissajousCurve({ freqX: 1, freqY: 1, phase: 0, ampX: A, ampY: B, ampZ: 0, samples: 360 });
  for (const q of pts) {
    assert.ok(Math.abs(q[1] * A - q[0] * B) < 1e-9, `off the diagonal at ${q}: y*A - x*B = ${q[1] * A - q[0] * B}`);
    assert.ok(q[2] === 0, 'planar');
  }
  const xs = pts.map((q) => q[0]);
  assert.ok(Math.abs(Math.max(...xs) - A) < 1e-6 && Math.abs(Math.min(...xs) + A) < 1e-6, 'and spans exactly +/-ampX');
  // THE ORACLE IS NOT VACUOUS: a non-zero phase opens the line into an ellipse
  // and the same identity fails hard.
  const opened = lissajousCurve({ freqX: 1, freqY: 1, phase: 10, ampX: A, ampY: B, ampZ: 0, samples: 360 });
  assert.ok(opened.some((q) => Math.abs(q[1] * A - q[0] * B) > 100), 'phase must open the diagonal');
});

/* The same 1:1 pair at a quarter-turn of phase is an exact ellipse:
   x = A*sin(t + pi/2) = A*cos t, y = B*sin t, so (x/A)^2 + (y/B)^2 = 1 for
   every t. An exact closed form that reads BOTH amplitudes and the phase. */
test('lissajousCurve: 1:1 at phase 90 is exactly the ellipse (x/A)^2 + (y/B)^2 = 1', () => {
  const A = 40, B = 25;
  const pts = lissajousCurve({ freqX: 1, freqY: 1, phase: 90, ampX: A, ampY: B, ampZ: 0, samples: 512 });
  for (const q of pts) {
    const v = (q[0] / A) ** 2 + (q[1] / B) ** 2;
    assert.ok(Math.abs(v - 1) < 1e-12, `off the ellipse: ${v}`);
  }
  // ...and A != B, so this is genuinely an ellipse and not a circle in disguise.
  assert.ok(Math.abs(Math.max(...pts.map((q) => q[0])) - A) < 1e-9);
  assert.ok(Math.abs(Math.max(...pts.map((q) => q[1])) - B) < 1e-9);
});

/* ⭐ THE FIGURE-EIGHT, FROM ITS IMPLICIT EQUATION. With a 1:2 ratio at phase 0,
   y = B*sin(2t) = 2B*sin(t)*cos(t) = 2B*(x/A)*cos(t), and cos^2 = 1 - sin^2,
   so every point satisfies (y/B)^2 = 4*(x/A)^2*(1 - (x/A)^2) identically. This
   is a real oracle for the SHAPE; "it looks like an eight" would not be. */
test('lissajousCurve: a 1:2 ratio satisfies the figure-eight identity exactly, and crosses the origin exactly twice', () => {
  const A = 30, B = 20;
  const pts = lissajousCurve({ freqX: 1, freqY: 2, phase: 0, ampX: A, ampY: B, ampZ: 0, samples: 720 });
  for (const q of pts) {
    const u = q[0] / A, v = q[1] / B;
    assert.ok(Math.abs(v * v - 4 * u * u * (1 - u * u)) < 1e-12, `off the lemniscate-of-Gerono identity at ${q}`);
  }
  /* The self-crossing is at the origin and there are exactly two parameter
     values reaching it (t = 0 and t = pi), both of which are sampled here. The
     nearest other sample is |sin(2pi/720)|*A = 0.26 away, so counting is safe. */
  const body = pts.slice(0, -1);   // drop the wrap repeat, or t = 0 counts twice
  const atOrigin = body.filter((q) => Math.hypot(q[0], q[1]) < 1e-9);
  assert.equal(atOrigin.length, 2, `a figure-eight touches the origin exactly twice, found ${atOrigin.length}`);
  // NOT VACUOUS: a 1:3 curve is a different figure and fails the same identity.
  const three = lissajousCurve({ freqX: 1, freqY: 3, phase: 0, ampX: A, ampY: B, ampZ: 0, samples: 720 });
  assert.ok(three.some((q) => Math.abs((q[1] / B) ** 2 - 4 * (q[0] / A) ** 2 * (1 - (q[0] / A) ** 2)) > 0.1), '1:3 is not a figure-eight');
});

test('lissajousCurve: a rational ratio closes over its OWN period; an irrational one is emitted OPEN', () => {
  assert.equal(lissajousPeriodTurns(3, 2, 0), 1, 'whole frequencies close in one turn');
  assert.equal(lissajousPeriodTurns(3, 2.5, 0), 2, '3 : 5/2 is 6 : 5, so it needs two turns');
  assert.equal(lissajousPeriodTurns(1.5, 2.5, 0), 2, '3/2 : 5/2 needs two turns');
  /* (2,4,6) is (1,2,3) traced at double speed, so it closes in HALF a base
     turn — a fractional period is a real answer, not a rounding failure. */
  assert.equal(lissajousPeriodTurns(2, 4, 6), 0.5);
  const fast = lissajousCurve({ freqX: 2, freqY: 4, freqZ: 6, ampZ: 10, phase: 40, samples: 300 });
  assert.ok(dist(fast[0], fast[300]) < 1e-12, 'and half a turn really does close it');
  assert.equal(lissajousPeriodTurns(0.5, 0.5, 0), 2, 'half frequencies take two turns, not one');
  assert.equal(lissajousPeriodTurns(3, Math.PI, 0), 0, 'an irrational ratio never returns');
  assert.equal(lissajousCloses(3, 2, 0), true);
  assert.equal(lissajousCloses(3, Math.PI, 0), false);
  /* ⚠ THE HALF-INTEGER TRAP, PINNED. sin(2.5 * 2pi) = sin(5pi) = 0 = sin(0),
     so at freq 2.5 the point at t = 2pi COINCIDES with the start while the
     tangent is reversed — a half traversal that any first-point-equals-last
     test would call closed. It is the period, not the point, that decides. */
  const half = lissajousCurve({ freqX: 3, freqY: 2.5, ampZ: 0, samples: 400 });
  const mid = half[200];   // t = 2pi, where the naive reading would have stopped
  assert.ok(dist(mid, half[0]) < 1e-9, 'the point at one turn really does coincide with the start');
  assert.ok(half.length === 401 && dist(half[0], half[400]) < 1e-12, 'and the chain runs the full two turns and closes');
  // The two halves are genuinely different curves, which is why one turn is not enough.
  assert.ok(half.slice(0, 200).some((q, i) => dist(q, half[200 + i]) > 1), 'the second turn is not a retrace of the first');
  const closed = lissajousCurve({ freqX: 3, freqY: 2, freqZ: 0, samples: 360 });
  assert.equal(closed.length, 361, 'n samples plus the explicit closing repeat');
  assert.ok(dist(closed[0], closed[closed.length - 1]) < 1e-12, 'the wrap segment is missing');
  /* AND THE SEAM IS NOT SPECIAL — a chain that merely repeats its first point
     onto an unrelated last point would pass the gap test above. */
  const steps = [];
  for (let i = 1; i < closed.length; i++) steps.push(dist(closed[i - 1], closed[i]));
  const median = [...steps].sort((a, b) => a - b)[Math.floor(steps.length / 2)];
  assert.ok(steps[steps.length - 1] < median * 3, 'the closing step is an ordinary step, not a chord');
  // The irrational case never returns, so no closure is claimed.
  const open = lissajousCurve({ freqX: 3, freqY: Math.PI, samples: 360 });
  assert.equal(open.length, 361, 'the open form still returns samples+1 points');
  assert.ok(dist(open[0], open[open.length - 1]) > 1, 'and does NOT pretend to close');
});

test('lissajousCurve: the third axis is a real sinusoid, and freqZ is inert when ampZ is 0', () => {
  const pts = lissajousCurve({ freqX: 3, freqY: 2, freqZ: 5, ampX: 40, ampY: 40, ampZ: 12, samples: 360, phase: 0 });
  for (let i = 0; i < 360; i++) {
    const t = (i / 360) * 2 * Math.PI;
    assert.ok(Math.abs(pts[i][2] - 12 * Math.sin(5 * t)) < 1e-12, `z is not the requested sinusoid at ${i}`);
  }
  assert.ok(Math.max(...pts.map((q) => Math.abs(q[2]))) > 11.9, 'and it genuinely leaves the plane');
  /* ⚠ THE DOCUMENTED BLIND SPOT, ASSERTED SO NOBODY "FIXES" IT: with ampZ = 0
     the curve cannot depend on freqZ, so any fixture at the default ampZ is
     unable to test freqZ at all. */
  const a = lissajousCurve({ freqZ: 4, ampZ: 0, samples: 90 });
  const b = lissajousCurve({ freqZ: 9, ampZ: 0, samples: 90 });
  assert.ok(a.every((q, i) => dist(q, b[i]) < 1e-15), 'freqZ has nothing to act on at ampZ 0');
  assert.equal(LISSAJOUS_DEFAULTS.ampZ, 0, 'which is the default, so freqZ needs its own fixture');
});

// ===========================================================================
// ROSE (rhodonea)
// ===========================================================================

/* ⭐ THE PETAL COUNT, COUNTED. r = a*cos(k*theta) sweeps out one petal between
   consecutive zeros of r, so the number of sign changes of r over one full
   traversal IS the petal count. It is read off the emitted geometry (radius
   from x,y with the sign recovered from which side of the origin the point
   lies) rather than from the formula, and compared against the textbook rule
   written out independently here. */
const countPetals = (params) => {
  const pts = roseCurve({ ...params, samples: 36000 });
  const body = pts.slice(0, -1);
  // Signed radius: the point is at angle theta, so the sign is whether the
  // point lies along +theta or along theta+pi.
  const thetaMax = roseThetaMax(params.n, params.d);
  let changes = 0, prev = 0;
  for (let i = 0; i < body.length; i++) {
    const th = (i / body.length) * thetaMax;
    const signed = body[i][0] * Math.cos(th) + body[i][1] * Math.sin(th);  // projection onto the ray
    const s = Math.sign(signed);
    if (i > 0 && s !== 0 && prev !== 0 && s !== prev) changes++;
    if (s !== 0) prev = s;
  }
  return changes;
};

test('roseCurve: the petal count is exactly n for odd n and exactly 2n for even n', () => {
  // The textbook rule, written here rather than imported, so the two can disagree.
  const rule = (n) => (n % 2 === 1 ? n : 2 * n);
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) {
    assert.equal(countPetals({ n, d: 1, amplitude: 40 }), rule(n), `n=${n} should have ${rule(n)} petals`);
  }
});

test('roseCurve: the rational generalization n/d gives the petals the same rule predicts', () => {
  /* For n/d in lowest terms the traversal is d*pi when n*d is odd and 2*d*pi
     otherwise, and r = cos(n*theta/d) has one zero every pi/(n/d) of theta —
     so the petal count is n*d / gcd, i.e. n if n*d is odd and 2n otherwise
     once reduced. Spot-checked on ratios that are NOT integers. */
  assert.equal(countPetals({ n: 7, d: 2, amplitude: 40 }), 14, '7/2 rose');
  assert.equal(countPetals({ n: 3, d: 2, amplitude: 40 }), 6, '3/2 rose');
  assert.equal(countPetals({ n: 2, d: 3, amplitude: 40 }), 4, '2/3 rose');
  assert.equal(countPetals({ n: 5, d: 3, amplitude: 40 }), 5, '5/3 rose (n*d odd -> the odd rule)');
  // roseThetaMax reduces n/d first: 4/2 IS 2/1 and must not be given twice the span.
  assert.ok(Math.abs(roseThetaMax(4, 2) - roseThetaMax(2, 1)) < 1e-12, '4/2 reduces to 2/1');
  assert.ok(Math.abs(roseThetaMax(3, 1) - Math.PI) < 1e-12, 'an odd rose closes in pi');
  assert.ok(Math.abs(roseThetaMax(4, 1) - 2 * Math.PI) < 1e-12, 'an even rose needs 2pi');
});

/* ⚠ THE SIGNED-RADIUS TRAP, PINNED. Plotting |r| instead of r turns the
   3-petal trefoil into a 6-petal flower — a change that looks like a nicer
   result and is a different curve. A 3-petal rose is invariant under a third
   of a turn and NOT under a sixth; the 6-petal one made by |r| is invariant
   under both, so this separates them. */
test('roseCurve: a 3-petal rose has 3-fold symmetry and NOT 6-fold (signed r, not |r|)', () => {
  const pts = roseCurve({ n: 3, d: 1, amplitude: 40, samples: 36000 }).slice(0, -1);
  const inSet = (p, ang) => {
    const c = Math.cos(ang), s = Math.sin(ang);
    const rx = p[0] * c - p[1] * s, ry = p[0] * s + p[1] * c;
    let best = Infinity;
    for (const q of pts) best = Math.min(best, Math.hypot(q[0] - rx, q[1] - ry));
    return best;
  };
  const probes = [pts[500], pts[3000], pts[9000], pts[21000]];
  for (const p of probes) assert.ok(inSet(p, 2 * Math.PI / 3) < 0.01, `not invariant under a third turn (${inSet(p, 2 * Math.PI / 3)})`);
  assert.ok(probes.some((p) => inSet(p, Math.PI / 3) > 0.5), 'a 3-petal rose must NOT be 6-fold symmetric');
  // Amplitude is the petal length exactly.
  assert.ok(Math.abs(Math.max(...pts.map((q) => Math.hypot(q[0], q[1]))) - 40) < 1e-6, 'petal tips reach exactly `amplitude`');
});

test('roseCurve: it closes with an explicit wrap, and a non-integer ratio is refused by name', () => {
  const pts = roseCurve({ n: 5, d: 1, samples: 720 });
  assert.equal(pts.length, 721, 'n samples plus the explicit closing repeat');
  assert.ok(dist(pts[0], pts[pts.length - 1]) < 1e-12, 'the wrap segment is missing');
  assert.throws(() => roseCurve({ n: 2.5, d: 1 }), /positive INTEGER/);
  assert.throws(() => roseCurve({ n: 3, d: 0 }), /positive INTEGER/);
  assert.throws(() => roseCurve({ n: -3, d: 1 }), /positive INTEGER/);
  assert.equal(ROSE_DEFAULTS.d, 1);
});

// ===========================================================================
// HELIX
// ===========================================================================

/* ⭐ THREE EXACT PROPERTIES AT ONCE, none of which a wrong helix satisfies:
   a constant radius, an arc length equal to turns*sqrt((2*pi*R)^2 + pitch^2),
   and a height equal to pitch*turns. The arc length is the discriminating one
   — it reads radius, pitch AND turns together, so no single one of them can
   be dropped without moving it. */
test('helixCurve: taper 0 holds its radius to machine precision, and pitch*turns is the height', () => {
  const radius = 20, pitch = 10, turns = 5;
  const pts = helixCurve({ radius, pitch, turns, taper: 0, samples: 4001, start: [3, -4, 11] });
  for (const q of pts) {
    const r = Math.hypot(q[0] - 3, q[1] + 4);
    assert.ok(Math.abs(r - radius) < 1e-12, `radius drifted to ${r}`);
  }
  assert.ok(Math.abs(pts[0][2] - 11) < 1e-12, 'starts at start.z');
  assert.ok(Math.abs(pts[pts.length - 1][2] - (11 + pitch * turns)) < 1e-12, 'and rises exactly pitch*turns');
  assert.equal(helixResolve({ radius, pitch, turns }).height, pitch * turns);
});

test('helixCurve: the polyline length converges on the closed-form helix arc length', () => {
  const radius = 20, pitch = 10, turns = 5;
  const exact = helixArcLength(radius, pitch, turns);
  const measure = (samples) => {
    const pts = helixCurve({ radius, pitch, turns, taper: 0, samples });
    let L = 0;
    for (let i = 1; i < pts.length; i++) L += dist(pts[i - 1], pts[i]);
    return L;
  };
  assert.ok(Math.abs(measure(40001) / exact - 1) < 1e-6, `measured ${measure(40001)} vs exact ${exact}`);
  // A chord polyline is always SHORT of the true arc, and it converges: the
  // deficit must shrink as the sampling refines, or the "exact" number is not
  // the thing being approached.
  const coarse = Math.abs(measure(401) / exact - 1);
  const fine = Math.abs(measure(4001) / exact - 1);
  assert.ok(measure(401) < exact && measure(4001) < exact, 'chords undercut the arc');
  assert.ok(fine < coarse / 50, `deficit must fall as O(1/n^2) (${coarse} -> ${fine})`);
  // NOT VACUOUS: the closed form reads all three parameters.
  assert.ok(helixArcLength(21, pitch, turns) > exact + 1, 'radius moves the arc length');
  assert.ok(helixArcLength(radius, 30, turns) > exact + 10, 'pitch moves the arc length');
  assert.ok(helixArcLength(radius, pitch, 6) > exact + 100, 'turns moves the arc length');
});

test('helixCurve: taper 1 closes the radius to exactly zero, and taper is monotone in between', () => {
  const cone = helixCurve({ radius: 20, pitch: 8, turns: 4, taper: 1, samples: 401 });
  const rOf = (q) => Math.hypot(q[0], q[1]);
  assert.ok(Math.abs(rOf(cone[0]) - 20) < 1e-12, 'starts at full radius');
  assert.ok(rOf(cone[400]) < 1e-12, `a full taper ends exactly on the axis, got ${rOf(cone[400])}`);
  for (let i = 1; i < cone.length; i++) assert.ok(rOf(cone[i]) < rOf(cone[i - 1]) + 1e-12, 'radius never grows back');
  // Half taper leaves exactly half the radius at the top.
  const half = helixCurve({ radius: 20, pitch: 8, turns: 4, taper: 0.5, samples: 401 });
  assert.ok(Math.abs(rOf(half[400]) - 10) < 1e-12, `taper 0.5 ends at half radius, got ${rOf(half[400])}`);
  /* ⚠ AND THE TAPER IS LINEAR IN BETWEEN, asserted at every station rather
     than only at the two ends. Both endpoints and monotonicity hold just as
     well for a taper that runs as s^2 — a cone with curved sides — so an
     endpoint-only test cannot tell a straight-sided cone from a bell. */
  for (let i = 0; i < half.length; i++) {
    const s = i / (half.length - 1);
    assert.ok(Math.abs(rOf(half[i]) - 20 * (1 - 0.5 * s)) < 1e-12, `taper is not linear at s=${s}: r=${rOf(half[i])}`);
  }
  // Beyond 1 the radius would go negative and unwind on the far side, so it is
  // clamped — an intentional inertness, pinned so it is not read as a defect.
  const over = helixCurve({ radius: 20, pitch: 8, turns: 4, taper: 3, samples: 401 });
  assert.ok(over.every((q, i) => dist(q, cone[i]) < 1e-12), 'taper is clamped to [0,1]');
});

/* ⚠ pitch/turns/height ARE ONE PARAMETER TOO MANY, and the module's stated
   rule is that pitch drives unless height > 0. Both halves are asserted:
   height genuinely overrides, and when it does, pitch is genuinely ignored. */
test('helixCurve: `height` overrides pitch, and the resolved triple stays consistent', () => {
  const r = helixResolve({ pitch: 10, turns: 5, height: 0 });
  assert.equal(r.pitch, 10);
  assert.equal(r.height, 50);
  const o = helixResolve({ pitch: 10, turns: 5, height: 80 });
  assert.equal(o.pitch, 16, 'height / turns');
  assert.equal(o.height, 80);
  const a = helixCurve({ radius: 20, pitch: 10, turns: 5, height: 80, samples: 201 });
  const b = helixCurve({ radius: 20, pitch: 99, turns: 5, height: 80, samples: 201 });
  assert.ok(a.every((q, i) => dist(q, b[i]) < 1e-12), 'with height set, pitch is ignored');
  assert.ok(Math.abs(a[200][2] - 80) < 1e-12, 'and the height is the one asked for');
  // Without height, pitch is very much not ignored.
  const c = helixCurve({ radius: 20, pitch: 99, turns: 5, height: 0, samples: 201 });
  assert.ok(Math.abs(c[200][2] - 495) < 1e-9, 'pitch drives when height is 0');
});

test('helixCurve: phase rotates the start exactly, and zero turns is refused', () => {
  const p0 = helixCurve({ radius: 20, phase: 0, turns: 2, samples: 100 })[0];
  assert.ok(Math.abs(p0[0] - 20) < 1e-12 && Math.abs(p0[1]) < 1e-12, `phase 0 starts on +x, got ${p0}`);
  const p90 = helixCurve({ radius: 20, phase: 90, turns: 2, samples: 100 })[0];
  assert.ok(Math.abs(p90[0]) < 1e-12 && Math.abs(p90[1] - 20) < 1e-12, `phase 90 starts on +y, got ${p90}`);
  assert.throws(() => helixCurve({ turns: 0 }), /turns > 0/);
  assert.throws(() => helixCurve({ turns: -2 }), /turns > 0/);
  assert.equal(HELIX_DEFAULTS.taper, 0);
  assert.equal(HELIX_DEFAULTS.height, 0);
});

// ===========================================================================
// CATENARY
// ===========================================================================

/* The span/sag -> a inversion is TRANSCENDENTAL, so the solver is checked
   against its own defining equation rather than against a formula: whatever a
   it returns must reproduce the requested sag. Checked across four decades of
   sag/span ratio, because a bracket that only ever works near the seed is a
   solver that works on the fixture. */
test('catenaryParameter: the returned a reproduces the requested sag exactly', () => {
  for (const span of [1, 100, 5000]) {
    for (const ratio of [0.001, 0.01, 0.1, 0.25, 1, 4, 40]) {
      const sag = span * ratio;
      const a = catenaryParameter(span, sag);
      const got = a * (Math.cosh(span / (2 * a)) - 1);
      assert.ok(Number.isFinite(a) && a > 0, `a must be a positive finite number, got ${a}`);
      assert.ok(Math.abs(got / sag - 1) < 1e-10, `span ${span} sag ${sag}: a=${a} gives sag ${got}`);
    }
  }
});

/* ⚠ AND IT IS NOT THE PARABOLA. The parabolic approximation a = span^2/(8*sag)
   is the seed, so a solver that simply returned its seed would pass every
   "is it finite / does it scale" check. It is separated here by the number
   that distinguishes them. */
test('catenaryParameter: the answer is the catenary one, not the parabolic seed', () => {
  const a = catenaryParameter(100, 25);
  const seed = (100 * 100) / (8 * 25);
  assert.equal(seed, 50);
  assert.ok(Math.abs(a - 53.71600941298045) < 1e-9, `a = ${a}`);
  assert.ok(a - seed > 3, 'the true a is materially larger than the parabolic estimate');
  // A chain built on the seed would hang 8.6% too deep, which is the whole reason.
  const seedSag = seed * (Math.cosh(100 / (2 * seed)) - 1);
  assert.ok(Math.abs(seedSag / 25 - 1) > 0.08, `the parabola is off by ${(seedSag / 25 - 1) * 100}%`);
});

/* ⭐ y = a*cosh(x/a) AT EVERY SAMPLED STATION, with x measured from the vertex
   and the whole curve shifted so the two suspension points sit at start.y. */
test('catenaryCurve: every point satisfies y = a*cosh(x/a), with the vertex exactly at the span midpoint', () => {
  const span = 100, sag = 25, sx = 7, sy = -3, sz = 2;
  const a = catenaryParameter(span, sag);
  const top = a * Math.cosh(span / (2 * a));
  const pts = catenaryCurve({ span, sag, samples: 121, start: [sx, sy, sz] });
  assert.equal(pts.length, 121);
  for (const q of pts) {
    const u = q[0] - sx - span / 2;                 // horizontal distance from the vertex
    const expected = sy + a * Math.cosh(u / a) - top;
    assert.ok(Math.abs(q[1] - expected) < 1e-9, `y ${q[1]} != a*cosh(u/a) shifted = ${expected} at u=${u}`);
    assert.ok(q[2] === sz, 'planar');
  }
  // The two suspension points sit exactly at start.y, and span exactly `span`.
  assert.ok(Math.abs(pts[0][0] - sx) < 1e-12 && Math.abs(pts[120][0] - (sx + span)) < 1e-12, 'spans exactly `span`');
  assert.ok(Math.abs(pts[0][1] - sy) < 1e-9 && Math.abs(pts[120][1] - sy) < 1e-9, 'ends sit at start.y');
  /* THE VERTEX IS A SAMPLE, because the default sample count is odd. Index 60
     of 121 is the exact midpoint. */
  assert.ok(Math.abs(pts[60][0] - (sx + span / 2)) < 1e-12, 'index 60 is the span midpoint');
  assert.ok(Math.abs(pts[60][1] - (sy - sag)) < 1e-9, `the vertex hangs exactly sag below, got ${pts[60][1] - sy}`);
  const ys = pts.map((q) => q[1]);
  assert.equal(ys.indexOf(Math.min(...ys)), 60, 'and it is the lowest point of the chain');
  // NOT VACUOUS: the same identity with a WRONG a misses by a mile.
  const bad = a * 1.1;
  const badTop = bad * Math.cosh(span / (2 * bad));
  assert.ok(pts.some((q) => Math.abs(q[1] - (sy + bad * Math.cosh((q[0] - sx - span / 2) / bad) - badTop)) > 0.1), 'a 10% wrong a must show');
});

test('catenaryCurve: it is symmetric, convex, and NOT a parabola', () => {
  const span = 100, sag = 25;
  const pts = catenaryCurve({ span, sag, samples: 201 });
  for (let i = 0; i < pts.length; i++) {
    const j = pts.length - 1 - i;
    assert.ok(Math.abs(pts[i][1] - pts[j][1]) < 1e-9, `not symmetric about the midpoint at ${i}`);
  }
  // A hanging chain is convex: the second difference is positive everywhere.
  for (let i = 1; i < pts.length - 1; i++) {
    const d2 = pts[i - 1][1] - 2 * pts[i][1] + pts[i + 1][1];
    assert.ok(d2 > 0, `not convex at ${i} (second difference ${d2})`);
  }
  /* The parabola through the same three points (both ends and the vertex) is a
     DIFFERENT curve, and by a visible amount — 0.35% of the span here. This is
     the claim that makes the family worth having at all rather than being a
     quadratic with a fancy name. */
  let worst = 0;
  for (const q of pts) {
    const u = (q[0] - span / 2) / (span / 2);
    worst = Math.max(worst, Math.abs(q[1] - (-sag + sag * u * u)));
  }
  assert.ok(worst > 0.3, `a catenary must measurably differ from the parabola through its own three points, got ${worst}`);
});

test('catenaryCurve: a degenerate span or sag is refused by name', () => {
  assert.throws(() => catenaryCurve({ span: 100, sag: 0 }), /sag > 0/);
  assert.throws(() => catenaryCurve({ span: 100, sag: -5 }), /sag > 0/);
  assert.throws(() => catenaryCurve({ span: 0, sag: 5 }), /span > 0/);
  assert.throws(() => catenaryCurve({ span: -100, sag: 5 }), /span > 0/);
  assert.equal(CATENARY_DEFAULTS.samples % 2, 1, 'an odd default puts a sample on the vertex');
});

// ===========================================================================
// TORUS KNOT
// ===========================================================================

/* ⭐ THE IMPLICIT EQUATION OF THE TORUS, at every point:
     (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2
   This is the strongest oracle in the six because it reads R and r together
   and holds to machine precision for every (p,q). */
test('torusKnotCurve: every point lies exactly on the torus it is wound on', () => {
  for (const [p, q] of [[2, 3], [3, 2], [3, 4], [5, 2], [1, 0], [0, 1], [7, 3]]) {
    const R = 40, r = 12;
    const pts = torusKnotCurve({ p, q, R, r, samples: 900, start: [5, -6, 7] });
    for (const s of pts) {
      const v = (Math.hypot(s[0] - 5, s[1] + 6) - R) ** 2 + (s[2] - 7) ** 2;
      assert.ok(Math.abs(v - r * r) < 1e-9, `(${p},${q}) off the torus: ${v} vs ${r * r}`);
    }
  }
  // NOT VACUOUS: the same test against the WRONG tube radius fails hard.
  const pts = torusKnotCurve({ p: 2, q: 3, R: 40, r: 12, samples: 200 });
  assert.ok(pts.some((s) => Math.abs((Math.hypot(s[0], s[1]) - 40) ** 2 + s[2] * s[2] - 169) > 1), 'r must be readable from the geometry');
  assert.ok(pts.some((s) => Math.abs((Math.hypot(s[0], s[1]) - 44) ** 2 + s[2] * s[2] - 144) > 1), 'R must be readable from the geometry');
});

/* ⭐ THE WINDING NUMBERS, WHICH ARE WHAT MAKE IT THE (p,q) KNOT AND NOT SOME
   OTHER CURVE ON THE SAME TORUS. Unwrapping the two angles around the closed
   loop must give exactly p turns about the main axis and exactly q turns about
   the tube — and the curve must NOT return to its start before then, which is
   what "closes after exactly q turns" means. */
const windings = (pts, R, start = [0, 0, 0]) => {
  const unwrap = (angs) => {
    let total = 0;
    for (let i = 1; i < angs.length; i++) {
      let d = angs[i] - angs[i - 1];
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      total += d;
    }
    return total / (2 * Math.PI);
  };
  const ring = pts.map((s) => Math.atan2(s[1] - start[1], s[0] - start[0]));
  const tube = pts.map((s) => Math.atan2(s[2] - start[2], Math.hypot(s[0] - start[0], s[1] - start[1]) - R));
  return { ring: unwrap(ring), tube: unwrap(tube) };
};

test('torusKnotCurve: the loop winds exactly p times around the ring and q times around the tube', () => {
  for (const [p, q] of [[2, 3], [3, 2], [3, 5], [5, 3], [1, 4], [4, 1], [7, 2]]) {
    const pts = torusKnotCurve({ p, q, R: 40, r: 12, samples: 4000 });
    const w = windings(pts, 40);
    assert.ok(Math.abs(w.ring - p) < 1e-9, `(${p},${q}): ring winding ${w.ring}, expected ${p}`);
    assert.ok(Math.abs(w.tube - q) < 1e-9, `(${p},${q}): tube winding ${w.tube}, expected ${q}`);
  }
});

test('torusKnotCurve: it closes with an explicit wrap and does NOT close early', () => {
  const p = 2, q = 3, n = 900;
  const pts = torusKnotCurve({ p, q, R: 40, r: 12, samples: n });
  assert.equal(pts.length, n + 1, 'n samples plus the explicit closing repeat');
  assert.ok(dist(pts[0], pts[pts.length - 1]) < 1e-12, 'the wrap segment is missing');
  /* Closing EARLY is the failure mode a gcd > 1 would cause, so it is checked
     directly: after each of the q-1 intermediate whole turns of the tube the
     point must be nowhere near the start. */
  for (let k = 1; k < q; k++) {
    const idx = Math.round((n * k) / q);
    assert.ok(dist(pts[idx], pts[0]) > 1, `(${p},${q}) returned to its start after ${k}/${q} of the run`);
  }
  // And every interior point stays away from the start, so the loop is traced once.
  const body = pts.slice(1, -1);
  assert.ok(Math.min(...body.map((s) => dist(s, pts[0]))) > 0.3, 'the curve is traced exactly once');
});

test('torusKnotCurve: a non-coprime (p,q) is refused as a LINK, and non-integers are refused too', () => {
  assert.throws(() => torusKnotCurve({ p: 2, q: 4 }), /LINK of 2 components/);
  assert.throws(() => torusKnotCurve({ p: 6, q: 9 }), /LINK of 3 components/);
  assert.throws(() => torusKnotCurve({ p: 0, q: 0 }), /LINK/);
  assert.throws(() => torusKnotCurve({ p: 2.5, q: 3 }), /INTEGER/);
  // The neighboring coprime cases are NOT refused, so the guard is discriminating.
  assert.doesNotThrow(() => torusKnotCurve({ p: 2, q: 5, samples: 50 }));
  assert.doesNotThrow(() => torusKnotCurve({ p: 6, q: 7, samples: 50 }));
  assert.doesNotThrow(() => torusKnotCurve({ p: 1, q: 0, samples: 50 }));
  assert.equal(TORUS_KNOT_DEFAULTS.p, 2);
  assert.equal(TORUS_KNOT_DEFAULTS.q, 3);
});

// ===========================================================================
// CROSS-CUTTING: no inert parameter, real closure, finite everywhere
// ===========================================================================

/* ⚠⚠ AN EXACT ORACLE IS NOT AUTOMATICALLY A DISCRIMINATING ONE. The
   superformula tests above carry the scar: a fixture that agreed to nine
   decimals was completely blind to n1, because cos^2 + sin^2 = 1 and 1 raised
   to any power is 1. The defense is mechanical rather than clever — every
   parameter of every family is perturbed from a fixture chosen so nothing
   masks it, and the emitted chain must MOVE.

   The fixtures below are therefore deliberately off every symmetry: no zero
   amplitudes (a zero-amplitude axis cannot see its own frequency), no zero
   taper, no zero phase, no zero height, and a non-origin `start`. */
const chainDiffers = (a, b) => a.length !== b.length || a.some((p, i) => dist(p, b[i]) > 1e-9);

const PARAM_SWEEP = [
  ['spiralCurve archimedean', spiralCurve,
    { kind: 'archimedean', turns: 3, startRadius: 5, growth: 2, height: 9, samples: 120, start: [1, 2, 3] },
    { kind: 'logarithmic', turns: 4.5, startRadius: 9, growth: 3.5, height: 21, samples: 121, start: [4, 2, 3] }],
  ['spiralCurve logarithmic', spiralCurve,
    { kind: 'logarithmic', turns: 3, startRadius: 5, growth: 0.2, height: 9, samples: 120, start: [1, 2, 3] },
    { kind: 'fermat', turns: 4.5, startRadius: 9, growth: 0.35, height: 21, samples: 121, start: [4, 2, 3] }],
  ['spiralCurve fermat', spiralCurve,
    { kind: 'fermat', turns: 3, startRadius: 5, height: 9, samples: 120, start: [1, 2, 3] },
    { kind: 'archimedean', turns: 4.5, startRadius: 9, height: 21, samples: 121, start: [4, 2, 3] }],
  ['lissajousCurve', lissajousCurve,
    { freqX: 3, freqY: 2, freqZ: 4, phase: 30, ampX: 40, ampY: 25, ampZ: 15, samples: 180, start: [1, 2, 3] },
    { freqX: 5, freqY: 7, freqZ: 9, phase: 55, ampX: 44, ampY: 31, ampZ: 21, samples: 181, start: [4, 2, 3] }],
  ['roseCurve', roseCurve,
    { n: 5, d: 1, amplitude: 40, samples: 180, start: [1, 2, 3] },
    { n: 4, d: 3, amplitude: 47, samples: 181, start: [4, 2, 3] }],
  ['helixCurve', helixCurve,
    { radius: 20, pitch: 10, turns: 5, height: 0, taper: 0.3, phase: 20, samples: 120, start: [1, 2, 3] },
    { radius: 26, pitch: 13, turns: 6, height: 80, taper: 0.7, phase: 55, samples: 121, start: [4, 2, 3] }],
  ['catenaryCurve', catenaryCurve,
    { span: 100, sag: 25, samples: 51, start: [1, 2, 3] },
    { span: 140, sag: 41, samples: 61, start: [4, 2, 3] }],
  ['torusKnotCurve', torusKnotCurve,
    { p: 2, q: 3, R: 40, r: 12, samples: 200, start: [1, 2, 3] },
    { p: 5, q: 7, R: 55, r: 7, samples: 201, start: [4, 2, 3] }],
];

test('NO INERT PARAMETER: perturbing any single parameter of any family moves the curve', () => {
  let checked = 0;
  for (const [name, fn, base, perturbed] of PARAM_SWEEP) {
    const baseline = fn(base);
    for (const key of Object.keys(base)) {
      assert.ok(key in perturbed, `${name}: no perturbation supplied for ${key}`);
      assert.notDeepEqual(base[key], perturbed[key], `${name}: the perturbation of ${key} is not a change`);
      const moved = fn({ ...base, [key]: perturbed[key] });
      assert.ok(chainDiffers(baseline, moved), `${name}: '${key}' is INERT at this fixture — the test cannot see it`);
      checked++;
    }
  }
  // A floor, so a table that silently loses a row cannot pass quietly.
  assert.ok(checked >= 45, `expected at least 45 parameter perturbations, ran ${checked}`);
});

test('closure: the closed families carry a real wrap segment, the open ones do not fake one', () => {
  const closed = [
    ['lissajousCurve', lissajousCurve({ freqX: 3, freqY: 2, samples: 240 })],
    ['roseCurve', roseCurve({ n: 5, d: 1, samples: 240 })],
    ['torusKnotCurve', torusKnotCurve({ p: 2, q: 3, samples: 240 })],
  ];
  for (const [name, pts] of closed) {
    assert.equal(pts.length, 241, `${name}: samples + 1 for the explicit repeat`);
    assert.ok(dist(pts[0], pts[pts.length - 1]) < 1e-12, `${name}: the wrap segment is missing`);
    /* AND THE SEAM IS AN ORDINARY STEP. A chain that merely appends a copy of
       its first point to an unrelated last point passes the gap test above and
       draws a chord across the picture. */
    const steps = [];
    for (let i = 1; i < pts.length; i++) steps.push(dist(pts[i - 1], pts[i]));
    const median = [...steps].sort((a, b) => a - b)[Math.floor(steps.length / 2)];
    assert.ok(steps[steps.length - 1] < median * 4, `${name}: the closing step ${steps[steps.length - 1]} dwarfs the median ${median}`);
  }
  /* ⚠ `samples` MEANS TWO DIFFERENT THINGS, exactly as it already does in the
     shipped families: for a CLOSED curve it is the number of DISTINCT samples
     and the chain is samples+1 long (the last entry is the explicit repeat),
     while for an OPEN curve it is the point count outright. noiseCurve and
     waveCurve already split this way; the new families follow them rather than
     inventing a third convention. */
  const open = [
    ['spiralCurve', spiralCurve({ turns: 3, samples: 241 })],
    ['helixCurve', helixCurve({ turns: 3, samples: 241 })],
    ['catenaryCurve', catenaryCurve({ samples: 241 })],
  ];
  for (const [name, pts] of open) {
    assert.equal(pts.length, 241, `${name}: samples is the point count for an open chain`);
    assert.ok(dist(pts[0], pts[pts.length - 1]) > 1, `${name}: an open family must not return to its start`);
  }
});

test('every new family is finite for its defaults and for extreme parameters', () => {
  const finite = (name, pts) => {
    assert.ok(Array.isArray(pts) && pts.length >= 2, `${name}: a real chain came out`);
    assert.ok(pts.every((q) => Array.isArray(q) && q.length === 3 && q.every(Number.isFinite)), `${name}: NaN/Infinity in the chain`);
  };
  finite('spiral defaults', spiralCurve());
  finite('lissajous defaults', lissajousCurve());
  finite('rose defaults', roseCurve());
  finite('helix defaults', helixCurve());
  finite('catenary defaults', catenaryCurve());
  finite('torusKnot defaults', torusKnotCurve());

  finite('spiral zero turns', spiralCurve({ turns: 0, samples: 8 }));
  finite('spiral negative growth', spiralCurve({ kind: 'archimedean', startRadius: 1, growth: -50, turns: 6, samples: 200 }));
  finite('spiral tiny logarithmic', spiralCurve({ kind: 'logarithmic', startRadius: 1e-9, growth: -3, turns: 20, samples: 200 }));
  finite('spiral 2 samples', spiralCurve({ samples: 1 }));   // clamped up to 2
  finite('lissajous huge freq', lissajousCurve({ freqX: 500, freqY: 499, ampZ: 5, freqZ: 250, samples: 64 }));
  finite('lissajous zero amps', lissajousCurve({ ampX: 0, ampY: 0, ampZ: 0, samples: 32 }));
  finite('rose n=1 d=1', roseCurve({ n: 1, d: 1, samples: 16 }));
  finite('rose huge n', roseCurve({ n: 199, d: 200, samples: 64 }));
  finite('rose zero amplitude', roseCurve({ amplitude: 0, samples: 16 }));
  finite('helix zero radius', helixCurve({ radius: 0, turns: 3, samples: 32 }));
  finite('helix fractional turns', helixCurve({ turns: 0.25, samples: 32 }));
  finite('helix huge turns', helixCurve({ turns: 500, samples: 64 }));
  finite('catenary hair-thin sag', catenaryCurve({ span: 100, sag: 1e-6, samples: 21 }));
  finite('catenary sag 100x span', catenaryCurve({ span: 100, sag: 10000, samples: 21 }));
  finite('catenary tiny span', catenaryCurve({ span: 1e-6, sag: 1e-7, samples: 21 }));
  finite('torusKnot r > R', torusKnotCurve({ p: 2, q: 3, R: 5, r: 40, samples: 64 }));
  finite('torusKnot large p,q', torusKnotCurve({ p: 101, q: 100, samples: 2048 }));
  finite('torusKnot 8 samples', torusKnotCurve({ samples: 1 }));   // clamped up to 8
});

test('the new families feed cleanly into globalCurveInterp, like the shipped ones', () => {
  for (const [name, pts] of [
    ['spiral', spiralCurve({ turns: 4, height: 20, samples: 240 })],
    ['helix', helixCurve({ turns: 4, taper: 0.4, samples: 240 })],
    ['catenary', catenaryCurve({ samples: 241 })],
  ]) {
    const crv = globalCurveInterp(decimateOpenToCount(pts, 60), 3);
    assert.equal(crv.degree, 3, `${name}: degree 3`);
    assert.ok(crv.ctrlPts.every((q) => q.every(Number.isFinite)), `${name}: finite control net`);
  }
});
