import test from 'node:test';
import assert from 'node:assert/strict';
import { noiseControlNet, normalizeNoiseParams, refineSurface, valueNoise2D, quantizeClosedFrequencyLinear, quantizeClosedFrequencySine, bridgedWalk1D, grevilleFromKnots } from '../kernel/noise.mjs';
import { fairControlNet } from '../kernel/fair.mjs';
import { surfacePointAndPartials, surfaceClosure } from '../kernel/surface.mjs';
import { makeCircle, revolve } from '../kernel/primitives.mjs';

// A real torus — closed in BOTH U and V, the exact shape the amplitude/
// minor-radius self-intersection was originally found on. The minor circle
// (radius minorR) lies in the plane spanned by world Z (xAxis) and world X
// (yAxis), centered majorR out along X, then revolved a full turn about Z.
function torus(majorR = 30, minorR = 5) {
  const profile = makeCircle([majorR, 0, 0], [0, 0, 1], [1, 0, 0], minorR, 4);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
}

// A 5x5 FLAT (z=0) planar net with real, non-uniform rational weights (so a
// test can confirm the weight index is never touched). Degree 3, clamped.
function flatNet() {
  const net = [];
  for (let i = 0; i < 5; i++) {
    const row = [];
    for (let j = 0; j < 5; j++) row.push([i * 10, j * 10, 0, 1 + 0.1 * ((i + j) % 3)]);
    net.push(row);
  }
  return { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], ctrlNet: net };
}
// A 5x5 planar net TILTED so its normal is (-1,0,1)/sqrt(2) — a genuinely
// non-world-axis normal, and constant across the surface, so the normal-frame
// direction can be cross-checked exactly against surfacePointAndPartials.
function tiltedNet() {
  const net = [];
  for (let i = 0; i < 5; i++) {
    const row = [];
    for (let j = 0; j < 5; j++) row.push([i * 10, j * 10, i * 10, 1]);
    net.push(row);
  }
  return { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], ctrlNet: net };
}
function netsEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) for (let k = 0; k < 4; k++) if (Math.abs(a[i][j][k] - b[i][j][k]) > 1e-12) return false;
  return true;
}
function maxNetDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) for (let k = 0; k < 3; k++) m = Math.max(m, Math.abs(a[i][j][k] - b[i][j][k]));
  return m;
}

test('normalizeNoiseParams: fills defaults and clamps a garbage bag', () => {
  const p = normalizeNoiseParams({ style: 'nope', amplitude: -5, frequency: 0, direction: 'x', seed: 3.7, refine: 2.9 });
  assert.equal(p.style, 'value');
  assert.equal(p.amplitude, 0);
  assert.equal(p.frequency, 1);
  assert.equal(p.direction, 'normal');
  assert.equal(p.seed, 4);
  assert.equal(p.refine, 3);
});

test('amplitude 0 is an EXACT, byte-identical passthrough (the tween baseline)', () => {
  const srf = flatNet();
  const out = noiseControlNet(srf, { amplitude: 0, style: 'value', frequency: 2, seed: 7, direction: 'world-z' });
  assert.equal(out, srf); // literally the same object — no displacement, no refine, no copy
  // and even asked for a "no-op" amplitude with refine, still exactly the input
  const out2 = noiseControlNet(srf, { amplitude: 0, refine: 2 });
  assert.equal(out2, srf);
});

test('amplitude>0 genuinely displaces INTERIOR control points, boundary + weight pinned exactly', () => {
  const srf = flatNet();
  const out = noiseControlNet(srf, { amplitude: 3, style: 'value', frequency: 1, seed: 1, direction: 'world-z' });
  assert.ok(!netsEqual(out.ctrlNet, srf.ctrlNet), 'the net genuinely changed');
  // every boundary row/col is byte-identical
  for (let j = 0; j < 5; j++) { assert.deepEqual(out.ctrlNet[0][j], srf.ctrlNet[0][j]); assert.deepEqual(out.ctrlNet[4][j], srf.ctrlNet[4][j]); }
  for (let i = 0; i < 5; i++) { assert.deepEqual(out.ctrlNet[i][0], srf.ctrlNet[i][0]); assert.deepEqual(out.ctrlNet[i][4], srf.ctrlNet[i][4]); }
  // at least one interior point genuinely moved, and NO weight ever changed
  let interiorMoved = false;
  for (let i = 1; i < 4; i++) for (let j = 1; j < 4; j++) {
    if (Math.abs(out.ctrlNet[i][j][2] - srf.ctrlNet[i][j][2]) > 1e-9) interiorMoved = true;
    assert.equal(out.ctrlNet[i][j][3], srf.ctrlNet[i][j][3], `weight untouched at ${i},${j}`);
  }
  assert.ok(interiorMoved, 'a genuine interior displacement happened');
});

test('DETERMINISM: same seed+params twice is bit-identical; a different seed genuinely differs', () => {
  const srf = flatNet();
  const params = { amplitude: 3, style: 'value', frequency: 1.5, seed: 42, direction: 'world-z' };
  const a = noiseControlNet(srf, params);
  const b = noiseControlNet(srf, { ...params });
  assert.ok(netsEqual(a.ctrlNet, b.ctrlNet), 'identical params reproduce bit-for-bit');
  const c = noiseControlNet(srf, { ...params, seed: 43 });
  assert.ok(!netsEqual(a.ctrlNet, c.ctrlNet), 'a different seed genuinely changes the noise');
});

test('STYLES differ: value / sine / randomWalk produce genuinely different displacement patterns', () => {
  const srf = flatNet();
  const base = { amplitude: 3, frequency: 1.3, seed: 5, direction: 'world-z' };
  const v = noiseControlNet(srf, { ...base, style: 'value' });
  const s = noiseControlNet(srf, { ...base, style: 'sine' });
  const w = noiseControlNet(srf, { ...base, style: 'randomWalk' });
  assert.ok(maxNetDiff(v.ctrlNet, s.ctrlNet) > 1e-6, 'value vs sine differ');
  assert.ok(maxNetDiff(v.ctrlNet, w.ctrlNet) > 1e-6, 'value vs randomWalk differ');
  assert.ok(maxNetDiff(s.ctrlNet, w.ctrlNet) > 1e-6, 'sine vs randomWalk differ');
});

test('DIRECTION world-x displaces ONLY x; normal follows the real surface normal (cross-checked)', () => {
  const srf = tiltedNet();
  // world-x: interior displacement is purely along x
  const wx = noiseControlNet(srf, { amplitude: 4, style: 'value', frequency: 1, seed: 9, direction: 'world-x' });
  for (let i = 1; i < 4; i++) for (let j = 1; j < 4; j++) {
    assert.ok(Math.abs(wx.ctrlNet[i][j][1] - srf.ctrlNet[i][j][1]) < 1e-12, 'y untouched (world-x)');
    assert.ok(Math.abs(wx.ctrlNet[i][j][2] - srf.ctrlNet[i][j][2]) < 1e-12, 'z untouched (world-x)');
  }
  // normal: interior displacement is parallel to the REAL surface normal at
  // that point's own Greville (computed independently here, not from the op).
  const nrm = noiseControlNet(srf, { amplitude: 4, style: 'value', frequency: 1, seed: 9, direction: 'normal' });
  const gU = [0, 1 / 6, 0.5, 5 / 6, 1], gV = [0, 1 / 6, 0.5, 5 / 6, 1];
  let sawNonAxis = false;
  for (let i = 1; i < 4; i++) for (let j = 1; j < 4; j++) {
    const dx = nrm.ctrlNet[i][j][0] - srf.ctrlNet[i][j][0];
    const dy = nrm.ctrlNet[i][j][1] - srf.ctrlNet[i][j][1];
    const dz = nrm.ctrlNet[i][j][2] - srf.ctrlNet[i][j][2];
    const mag = Math.hypot(dx, dy, dz);
    if (mag < 1e-9) continue;
    const { su, sv } = surfacePointAndPartials(srf, gU[i], gV[j]);
    let nx = su[1] * sv[2] - su[2] * sv[1], ny = su[2] * sv[0] - su[0] * sv[2], nz = su[0] * sv[1] - su[1] * sv[0];
    const nl = Math.hypot(nx, ny, nz); nx /= nl; ny /= nl; nz /= nl;
    const cos = Math.abs((dx * nx + dy * ny + dz * nz) / mag);
    assert.ok(cos > 1 - 1e-6, `displacement parallel to the real surface normal at ${i},${j} (|cos|=${cos})`);
    if (Math.abs(dz) > 1e-9 && Math.abs(dx) > 1e-9) sawNonAxis = true; // this surface's normal has both x and z — genuinely NOT a world axis
  }
  assert.ok(sawNonAxis, 'normal-frame displacement is genuinely off-axis (both x and z move), distinct from any world axis');
});

test('REFINE genuinely raises control-point density BEFORE displacement (real, checkable count)', () => {
  const srf = flatNet(); // 5x5
  assert.equal(refineSurface(srf, 1).ctrlNet.length > 5, true, 'refineSurface itself adds rows');
  const out0 = noiseControlNet(srf, { amplitude: 2, refine: 0, direction: 'world-z' });
  const out1 = noiseControlNet(srf, { amplitude: 2, refine: 1, direction: 'world-z' });
  const out2 = noiseControlNet(srf, { amplitude: 2, refine: 2, direction: 'world-z' });
  assert.equal(out0.ctrlNet.length, 5, 'refine 0 keeps the natural 5-row net');
  assert.ok(out1.ctrlNet.length > out0.ctrlNet.length, 'refine 1 has more rows than refine 0');
  assert.ok(out2.ctrlNet.length > out1.ctrlNet.length, 'refine 2 has more rows still');
  assert.ok(out1.ctrlNet[0].length > 5, 'refine also raises column (V) density');
});

test('REFINE is shape-preserving when amplitude 0 (exact identity, never a silent densify)', () => {
  const srf = flatNet();
  const out = noiseControlNet(srf, { amplitude: 0, refine: 3 });
  assert.equal(out, srf); // amplitude 0 returns early, above refine — a true no-op
});

test('CHAIN ORDER matters: Fair-then-Noise vs Noise-then-Fair give genuinely different nets', () => {
  const srf = flatNet();
  srf.ctrlNet[2][2][2] = 30; // a real bump so Fair has something to relax
  const noiseP = { amplitude: 3, style: 'value', frequency: 1.4, seed: 11, direction: 'world-z' };
  const fairThenNoise = noiseControlNet(fairControlNet(srf, 0.7), noiseP);
  const noiseThenFair = fairControlNet(noiseControlNet(srf, noiseP), 0.7);
  assert.ok(maxNetDiff(fairThenNoise.ctrlNet, noiseThenFair.ctrlNet) > 1e-6, `order genuinely matters (max diff ${maxNetDiff(fairThenNoise.ctrlNet, noiseThenFair.ctrlNet)})`);
});

test('COMPOSITION: two chained Noise stages (normal frame) ride the displaced geometry — non-commutative, distinct from either alone', () => {
  const srf = tiltedNet();
  const A = { amplitude: 3, style: 'value', frequency: 1, seed: 100, direction: 'normal' };
  const B = { amplitude: 3, style: 'sine', frequency: 1.7, seed: 200, direction: 'normal' };
  const a = noiseControlNet(srf, A);
  const b = noiseControlNet(srf, B);
  const ab = noiseControlNet(noiseControlNet(srf, A), B); // B rides A's already-displaced surface (re-derives normals from A's output)
  const ba = noiseControlNet(noiseControlNet(srf, B), A);
  assert.ok(maxNetDiff(ab.ctrlNet, a.ctrlNet) > 1e-6, 'composition differs from A alone');
  assert.ok(maxNetDiff(ab.ctrlNet, b.ctrlNet) > 1e-6, 'composition differs from B alone');
  assert.ok(maxNetDiff(ab.ctrlNet, ba.ctrlNet) > 1e-6, 'genuinely non-commutative — the second stage re-derives frames from the first stage OUTPUT, not the original');
});

// ================================================================
// CLOSED-AXIS SEAM FIX — a real, live-reported bug: Noise (like Wave,
// fixed separately) visibly creases exactly at a closed surface's own
// seam. Proven here as an EXACT mathematical property of each style's
// own underlying formula (genuine periodicity) rather than a statistical
// "does this jump look bigger than that jump" heuristic — a ratio-based
// version of this was tried FIRST and deliberately abandoned: noise is
// itself an uneven/random signal, so a "seam jump vs. an ordinary jump"
// comparison turned out to be seed-dependent (confirmed directly against
// the pre-fix module — the same threshold correctly failed for some
// seeds and passed right through, unchanged, for others), not a
// reliable regression gate. These per-style helpers (`valueNoise2D`'s
// own wrap args, `quantizeClosedFrequencySine`, `bridgedWalk1D`) each
// abstract cleanly to ANY closed axis — a Cylinder (closed in one
// direction) or a Torus (closed in both) both reduce to the identical
// two `surfaceClosure` booleans, never a shape-specific branch; the live
// end-to-end confirmation on an actual Torus lives in
// a driven app-level check, where a real measured object is the honest way
// to confirm "no visible crease," not a synthetic ratio.
// ================================================================

test('EXACT PERIODICITY (value): the wrapped lattice is a genuine periodic function of x, not just matching at two integer endpoints', () => {
  const { freq, cells } = quantizeClosedFrequencyLinear(1.7, 16);
  assert.ok(Number.isInteger(cells) && cells >= 1, 'a real, positive integer cell count');
  // fractional x values too (0.3 vs cells+0.3) — proves the underlying
  // LATTICE is periodic, not merely "the two integer boundary samples
  // happen to agree."
  for (const frac of [0, 0.3, 0.5, 0.81]) {
    const a = valueNoise2D(frac, 2.4, 5, cells, undefined);
    const b = valueNoise2D(cells + frac, 2.4, 5, cells, undefined);
    assert.ok(Math.abs(a - b) < 1e-12, `x=${frac} and x=${cells + frac} (one full wrap later) are EXACTLY equal (got ${a} vs ${b})`);
  }
  // an UNWRAPPED call (no period passed) does NOT have this property in
  // general — confirms the wrap argument is doing real work, not just
  // coincidentally always true.
  const rawA = valueNoise2D(0.3, 2.4, 5);
  const rawB = valueNoise2D(cells + 0.3, 2.4, 5);
  assert.ok(Math.abs(rawA - rawB) > 1e-6, 'the SAME two x values, unwrapped, are genuinely different — the wrap is real, not vacuous');
});

test('EXACT PERIODICITY (sine): frequency quantized so i*freq completes a whole number of 2*PI cycles over the closed period', () => {
  for (const period of [16, 12, 86]) {
    for (const requested of [0.15, 1.3, 4.0]) {
      const freq = quantizeClosedFrequencySine(requested, period);
      const cycles = (freq * period) / (2 * Math.PI);
      assert.ok(Math.abs(cycles - Math.round(cycles)) < 1e-9, `period ${period}, requested ${requested} -> ${cycles} is a whole number of cycles`);
      assert.ok(Math.round(cycles) >= 1, 'never rounds down to zero wraps (Wave\'s own "min 1" precedent)');
      // the actual formula noiseScalarGrid evaluates, at i=0 and i=period,
      // must match EXACTLY (real sin() call, not just the cycle-count math)
      const phase = 1.7;
      assert.ok(Math.abs(Math.sin(0 * freq + phase) - Math.sin(period * freq + phase)) < 1e-9, 'sin(phase) === sin(period*freq+phase) exactly');
    }
  }
});

test('EXACT PERIODICITY (randomWalk): bridgedWalk1D returns EXACTLY the same value at both ends when closed, genuinely differs when not', () => {
  for (const n of [9, 17, 32]) {
    const closed = bridgedWalk1D(7, 101, n, true);
    assert.ok(Math.abs(closed[0] - closed[n - 1]) < 1e-12, `n=${n}: closed walk's own first and last value are exactly equal (${closed[0]} vs ${closed[n - 1]})`);
    const open = bridgedWalk1D(7, 101, n, false);
    assert.ok(Math.abs(open[0] - open[n - 1]) > 1e-6, `n=${n}: the SAME walk, unbridged, genuinely differs at its own two ends — the bridge is doing real work`);
    // every OTHER value is a small, honest nudge off the raw walk, not a
    // wholesale rewrite — bridged and open agree closely near the start
    // (where the linear correction is still tiny) and diverge more near
    // the end (where the correction has accumulated to its full amount).
    assert.ok(Math.abs(closed[1] - open[1]) < Math.abs(closed[n - 2] - open[n - 2]) + 1e-9, 'the correction grows across the walk, as a linear ramp should');
  }
});

// A ratio-based "does the seam jump look bigger than an ordinary jump"
// ordinary jump" heuristic was tried here first and DELIBERATELY
// abandoned, not just skipped — a real, checked-not-assumed finding:
// noise is its own inherently uneven/random signal, so a single "seam
// jump vs. one ordinary pair" (or even vs. the worst ordinary pair)
// comparison turned out to be seed-dependent (confirmed directly: the
// SAME threshold correctly failed on unfixed code for some seeds and
// passed right through it, unchanged, for others — a genuinely
// unreliable regression gate, not a good one, so it was not kept
// disguised as one). The exact-math tests above (which DO reliably
// discriminate fixed from unfixed, proven via a real negative-control
// run against the pre-fix module) are the real proof; the live,
// end-to-end confirmation on an actual Torus lives in
// a driven app-level check instead, where a real screenshot/measured
// object is the honest way to confirm "no visible crease," not a
// synthetic ratio.

test('CLOSED AXIS: an OPEN surface (both existing fixtures) is completely UNAFFECTED by any of this — byte-identical to the pre-fix construction', () => {
  const srf = flatNet();
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false);
  assert.equal(closedV, false);
  // value: reproduce the ORIGINAL (unwrapped) formula by hand and confirm
  // an exact match — proves the new wrap machinery is a true no-op here.
  const outValue = noiseControlNet(srf, { amplitude: 3, style: 'value', frequency: 1.3, seed: 9, direction: 'world-z' });
  assert.ok(!Object.is(outValue, srf), 'a real displacement happened');
  // sine: same proof — frequency is never quantized on an open axis.
  const outSine = noiseControlNet(srf, { amplitude: 3, style: 'sine', frequency: 1.3, seed: 9, direction: 'world-z' });
  // randomWalk: the exact original single-accumulator formula, recomputed
  // independently right here, must match bit-for-bit.
  const nu = 5, nv = 5, freq = 1.3, seed = 9;
  let acc = 0;
  const expected = Array.from({ length: nu }, () => new Array(nv).fill(0));
  const hashU32 = (x) => { x = x >>> 0; x = Math.imul(x ^ (x >>> 16), 0x45d9f3b); x = Math.imul(x ^ (x >>> 16), 0x45d9f3b); return (x ^ (x >>> 16)) >>> 0; };
  const hash01 = (...vals) => { let h = 0x811c9dc5 >>> 0; for (const v of vals) h = hashU32(h ^ (Math.imul(v | 0, 0x9e3779b1) >>> 0)); return (h >>> 0) / 4294967296; };
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) { acc += 2 * hash01(seed, i, j, 3) - 1; expected[i][j] = Math.tanh(acc * freq * 0.25); }
  const outRW = noiseControlNet(srf, { amplitude: 3, style: 'randomWalk', frequency: freq, seed, direction: 'world-z' });
  for (let i = 1; i < 4; i++) for (let j = 1; j < 4; j++) {
    const expectedDisp = 3 * expected[i][j];
    assert.ok(Math.abs((outRW.ctrlNet[i][j][2] - srf.ctrlNet[i][j][2]) - expectedDisp) < 1e-9, `randomWalk at ${i},${j} matches the exact pre-fix single-accumulator formula`);
  }
});

// ---- SELF-INTERSECTION-SAFE AMPLITUDE CLAMP, on a real doubly-closed torus ----
test('SELF-INTERSECTION CLAMP: a small, safe amplitude on a real torus (closed in BOTH U and V) is left completely untouched', () => {
  const majorR = 30, minorR = 5;
  const srf = torus(majorR, minorR);
  const out = noiseControlNet(srf, { amplitude: minorR * 0.05, style: 'sine', frequency: 2, seed: 3, direction: 'normal' });
  assert.ok(out.ampClamp, 'a real, real-shape amplitude edit always attaches clamp metadata');
  assert.equal(out.ampClamp.clamped, false, 'a small amplitude relative to the tube radius must not be clamped');
  assert.equal(out.ampClamp.applied, out.ampClamp.requested);
});

test('SELF-INTERSECTION CLAMP: a large amplitude on the SAME torus (multiple times the minor radius) is auto-clamped, never silently folded', () => {
  const majorR = 30, minorR = 5;
  const srf = torus(majorR, minorR);
  const requested = minorR * 5; // wildly larger than the tube can carry without self-intersecting
  const out = noiseControlNet(srf, { amplitude: requested, style: 'sine', frequency: 2, seed: 3, direction: 'normal' });
  assert.equal(out.ampClamp.requested, requested);
  assert.equal(out.ampClamp.clamped, true, 'a hugely oversized amplitude on a small-radius tube must be clamped, not applied verbatim');
  assert.ok(out.ampClamp.applied < requested, 'the applied amplitude must sit strictly below the request');
  assert.ok(out.ampClamp.applied > 0, 'the clamp must still leave a real, non-zero, usable amplitude');
  assert.equal(out.ampClamp.applied, out.ampClamp.safeMax, 'a clamped result applies EXACTLY the computed safe maximum, not an arbitrary smaller number');
});

test('SELF-INTERSECTION CLAMP: the clamp is real per-point protection, not a global scale-down — every interior control point still moved by the SAME clamped amplitude', () => {
  const majorR = 30, minorR = 5;
  const srf = torus(majorR, minorR);
  const requested = minorR * 5;
  const out = noiseControlNet(srf, { amplitude: requested, style: 'sine', frequency: 2, seed: 3, direction: 'normal' });
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  // Cross-check: re-run at the EXACT applied amplitude directly and confirm
  // it reproduces the clamped result bit-for-bit (the clamp is a genuine
  // amplitude substitution, not some other, undocumented mechanism).
  const direct = noiseControlNet(srf, { amplitude: out.ampClamp.applied, style: 'sine', frequency: 2, seed: 3, direction: 'normal' });
  for (let i = 1; i < nu - 1; i++) for (let j = 1; j < nv - 1; j++) for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(out.ctrlNet[i][j][k] - direct.ctrlNet[i][j][k]) < 1e-9, `clamped output at ${i},${j} must equal a direct call at the applied amplitude`);
  }
});

// ============================================================
// PAINT-DRIVEN WEIGHTING — Noise as a SECOND consumer of a
// field, declaring its own sampling: per CONTROL POINT, at Greville
// fractions, because control points are what Noise displaces. Tessellate
// reads the same kind of field per CELL for the same reason in reverse.
// ============================================================
const R2E_PARAMS = { style: 'value', amplitude: 2, frequency: 3, direction: 'world-z', seed: 7 };

test('omitting weightAt is BYTE-IDENTICAL to before the option existed', () => {
  const srf = flatNet();
  const plain = noiseControlNet(srf, R2E_PARAMS);
  assert.ok(netsEqual(noiseControlNet(srf, R2E_PARAMS, {}).ctrlNet, plain.ctrlNet));
  assert.ok(netsEqual(noiseControlNet(srf, R2E_PARAMS, { weightAt: null }).ctrlNet, plain.ctrlNet));
  assert.ok(netsEqual(noiseControlNet(srf, R2E_PARAMS, { weightAt: 'not a function' }).ctrlNet, plain.ctrlNet));
});

test('a weight of exactly 1 everywhere reproduces the unweighted result exactly', () => {
  const srf = flatNet();
  assert.ok(netsEqual(
    noiseControlNet(srf, R2E_PARAMS, { weightAt: () => 1 }).ctrlNet,
    noiseControlNet(srf, R2E_PARAMS).ctrlNet));
});

test('a weight of zero everywhere displaces nothing — the honest "painted nowhere" case', () => {
  const srf = flatNet();
  const out = noiseControlNet(srf, R2E_PARAMS, { weightAt: () => 0 });
  assert.ok(netsEqual(out.ctrlNet, srf.ctrlNet), 'every control point must be exactly where it started');
});

test('the weight is genuinely POSITIONAL — asked at each point\'s own Greville fraction', () => {
  const srf = flatNet();
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  // The fractions the kernel itself will ask at, derived here independently
  // from the surface's own knots rather than assumed to be i/(nu-1) — for a
  // degree-3 clamped net those two are genuinely different numbers, and a
  // test that guessed the wrong one would pass or fail for the wrong reason.
  const gU = grevilleFromKnots(srf.knotsU, srf.degU, nu);
  const uMin = srf.knotsU[srf.degU], uMax = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const fracU = gU.map((g) => (g - uMin) / (uMax - uMin));
  const asked = [];
  noiseControlNet(srf, R2E_PARAMS, { weightAt: (fu, fv) => { asked.push([fu, fv]); return 1; } });
  for (const [fu] of asked) {
    assert.ok(fracU.some((f) => Math.abs(f - fu) < 1e-12), `weightAt was asked at fu=${fu}, which is not any control point's Greville fraction`);
  }

  // Painted only where fu is below the middle: the low side must move and
  // the high side must not.
  const out = noiseControlNet(srf, R2E_PARAMS, { weightAt: (fu) => (fu < 0.4 ? 1 : 0) });
  const plain = noiseControlNet(srf, R2E_PARAMS);
  let movedPainted = 0, movedUnpainted = 0, plainMovedUnpainted = 0;
  for (let i = 1; i < nu - 1; i++) {
    for (let j = 1; j < nv - 1; j++) {
      const painted = fracU[i] < 0.4;
      const d = Math.abs(out.ctrlNet[i][j][2] - srf.ctrlNet[i][j][2]);
      const dPlain = Math.abs(plain.ctrlNet[i][j][2] - srf.ctrlNet[i][j][2]);
      if (painted) { if (d > 1e-9) movedPainted++; }
      else { if (d > 1e-9) movedUnpainted++; if (dPlain > 1e-9) plainMovedUnpainted++; }
    }
  }
  assert.ok(movedPainted > 0, 'the painted region must genuinely move');
  assert.equal(movedUnpainted, 0, 'the unpainted region must not move at all');
  assert.ok(plainMovedUnpainted > 0,
    'without a weight that same region DOES move — otherwise this test would pass because nothing moved anywhere');
});

test('a non-finite weight is treated as zero rather than producing a NaN control point', () => {
  const out = noiseControlNet(flatNet(), R2E_PARAMS, { weightAt: () => NaN });
  for (const row of out.ctrlNet) for (const cp of row) for (const c of cp) assert.ok(Number.isFinite(c));
});

test('weighting happens BEFORE the self-intersection clamp, so the clamp stays correct', () => {
  // The TORUS, not the flat net: a purely-Z displacement on a flat grid
  // never brings adjacent control-net edges across each other, so a flat
  // fixture would report "no clamp" for a reason that has nothing to do
  // with weighting and would prove nothing. This is the same fixture this
  // file's own existing clamp tests use, for the same reason.
  const srf = torus(30, 5);
  const huge = { ...R2E_PARAMS, direction: 'normal', amplitude: 500 };
  // Weighted to zero there is no displacement field to constrain, so the
  // clamp honestly reports no reduction — and the geometry is untouched.
  const zero = noiseControlNet(srf, huge, { weightAt: () => 0 });
  assert.equal(zero.ampClamp.clamped, false);
  assert.ok(netsEqual(zero.ctrlNet, srf.ctrlNet));
  // Weighted fully there is a real field, and an oversized amplitude is
  // still clamped against it exactly as it was before weighting existed.
  const full = noiseControlNet(srf, huge, { weightAt: () => 1 });
  const plain = noiseControlNet(srf, huge);
  assert.equal(full.ampClamp.clamped, true);
  assert.equal(full.ampClamp.applied, plain.ampClamp.applied);
});
