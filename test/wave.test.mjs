import test from 'node:test';
import assert from 'node:assert/strict';
import { waveControlNet, normalizeWaveParams } from '../kernel/wave.mjs';
import { makeCircle, revolve } from '../kernel/primitives.mjs';

// A real torus — closed in BOTH U and V, matching noise.test.mjs's own
// identical fixture (the exact shape this self-intersection was originally
// found on, both closed directions checked at once, not just one).
function torus(majorR = 30, minorR = 5) {
  const profile = makeCircle([majorR, 0, 0], [0, 0, 1], [1, 0, 0], minorR, 4);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
}

// Same fixture shapes as noise.test.mjs — a flat 5x5 net (weights varied,
// never touched) and a tilted 5x5 net whose true normal is a known,
// constant, non-world-axis vector.
function flatNet() {
  const net = [];
  for (let i = 0; i < 5; i++) {
    const row = [];
    for (let j = 0; j < 5; j++) row.push([i * 10, j * 10, 0, 1 + 0.1 * ((i + j) % 3)]);
    net.push(row);
  }
  return { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], ctrlNet: net };
}
function tiltedNet() {
  const net = [];
  for (let i = 0; i < 5; i++) {
    const row = [];
    for (let j = 0; j < 5; j++) row.push([i * 10, j * 10, i * 10, 1]);
    net.push(row);
  }
  return { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], ctrlNet: net };
}
function maxNetDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) for (let k = 0; k < 3; k++) m = Math.max(m, Math.abs(a[i][j][k] - b[i][j][k]));
  return m;
}

test('normalizeWaveParams: fills defaults and clamps a garbage bag', () => {
  const p = normalizeWaveParams({ axis: 'nope', amplitude: -5, frequency: 0, phase: 'x', direction: 'y', refine: 2.9 });
  assert.equal(p.axis, 'u');
  assert.equal(p.amplitude, 0);
  assert.equal(p.frequency, 1);
  assert.equal(p.phase, 0);
  assert.equal(p.direction, 'normal');
  assert.equal(p.refine, 3);
});

test('amplitude 0 is an EXACT, byte-identical passthrough', () => {
  const srf = flatNet();
  const out = waveControlNet(srf, { amplitude: 0, frequency: 2, phase: 1, direction: 'world-z' });
  assert.equal(out, srf);
  const out2 = waveControlNet(srf, { amplitude: 0, refine: 2 });
  assert.equal(out2, srf);
});

test('amplitude>0 displaces only INTERIOR control points; boundary rows/cols and every weight untouched', () => {
  const srf = flatNet();
  const out = waveControlNet(srf, { amplitude: 5, frequency: 1, axis: 'u', direction: 'world-z' });
  const nu = out.ctrlNet.length, nv = out.ctrlNet[0].length;
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const isBoundary = i === 0 || i === nu - 1 || j === 0 || j === nv - 1;
      assert.equal(out.ctrlNet[i][j][3], srf.ctrlNet[i][j][3]); // weight NEVER touched
      if (isBoundary) {
        assert.deepEqual(out.ctrlNet[i][j].slice(0, 3), srf.ctrlNet[i][j].slice(0, 3));
      }
    }
  }
  // at least one real interior point genuinely moved
  assert.ok(maxNetDiff(out.ctrlNet, srf.ctrlNet) > 0.01);
});

test('world-axis direction displaces EXACTLY along that axis, matching the closed-form sine at every interior point', () => {
  const srf = flatNet();
  const p = { amplitude: 3, frequency: 2, phase: 0.7, axis: 'v', direction: 'world-x' };
  const out = waveControlNet(srf, p);
  const nu = out.ctrlNet.length, nv = out.ctrlNet[0].length;
  for (let i = 1; i < nu - 1; i++) {
    for (let j = 1; j < nv - 1; j++) {
      const fv = j / (nv - 1);
      const expected = p.amplitude * Math.sin(2 * Math.PI * p.frequency * fv + p.phase);
      assert.ok(Math.abs(out.ctrlNet[i][j][0] - (srf.ctrlNet[i][j][0] + expected)) < 1e-9);
      assert.equal(out.ctrlNet[i][j][1], srf.ctrlNet[i][j][1]); // Y untouched — world-x only
      assert.equal(out.ctrlNet[i][j][2], srf.ctrlNet[i][j][2]); // Z untouched
    }
  }
});

test('direction:normal displaces along the surface\'s own TRUE (constant, tilted) normal', () => {
  const srf = tiltedNet(); // true normal is (-1,0,1)/sqrt(2) everywhere
  const out = waveControlNet(srf, { amplitude: 4, frequency: 1, axis: 'u', direction: 'normal' });
  const n = 1 / Math.sqrt(2);
  let sawReal = false;
  for (let i = 1; i < 4; i++) {
    for (let j = 1; j < 4; j++) {
      const dx = out.ctrlNet[i][j][0] - srf.ctrlNet[i][j][0];
      const dz = out.ctrlNet[i][j][2] - srf.ctrlNet[i][j][2];
      const dy = out.ctrlNet[i][j][1] - srf.ctrlNet[i][j][1];
      assert.ok(Math.abs(dy) < 1e-9); // the tilt has no Y component, so Y must never move
      if (Math.abs(dx) > 1e-6) { sawReal = true; assert.ok(Math.abs(dx / dz - (-n) / n) < 1e-6 || (Math.abs(dx) < 1e-9 && Math.abs(dz) < 1e-9)); }
    }
  }
  assert.ok(sawReal);
});

test('axis choice genuinely changes the displacement pattern (u vs v vs diagonal all differ)', () => {
  const srf = flatNet();
  const p = { amplitude: 5, frequency: 3, direction: 'world-z' };
  const outU = waveControlNet(srf, { ...p, axis: 'u' });
  const outV = waveControlNet(srf, { ...p, axis: 'v' });
  const outD = waveControlNet(srf, { ...p, axis: 'diagonal' });
  assert.ok(maxNetDiff(outU.ctrlNet, outV.ctrlNet) > 0.01);
  assert.ok(maxNetDiff(outU.ctrlNet, outD.ctrlNet) > 0.01);
  assert.ok(maxNetDiff(outV.ctrlNet, outD.ctrlNet) > 0.01);
});

test('phase genuinely shifts the wave (a real animatable knob, per the doc\'s own "no special Animator coupling needed" instruction)', () => {
  const srf = flatNet();
  const outA = waveControlNet(srf, { amplitude: 5, frequency: 1, axis: 'u', direction: 'world-z', phase: 0 });
  const outB = waveControlNet(srf, { amplitude: 5, frequency: 1, axis: 'u', direction: 'world-z', phase: Math.PI / 2 });
  assert.ok(maxNetDiff(outA.ctrlNet, outB.ctrlNet) > 0.01);
});

test('refine raises control-point density before displacement (shape-preserving knot insertion, reused from noise.mjs)', () => {
  const srf = flatNet();
  const out = waveControlNet(srf, { amplitude: 2, frequency: 1, refine: 1 });
  assert.ok(out.ctrlNet.length > srf.ctrlNet.length);
  assert.ok(out.ctrlNet[0].length > srf.ctrlNet[0].length);
});

test('a pole (no defined normal) is skipped honestly under direction:normal — no NaN/Infinity', () => {
  // collapse one interior row to a single point (pole-like degeneracy)
  const srf = flatNet();
  srf.ctrlNet[2] = srf.ctrlNet[2].map(() => [20, 20, 0, 1]);
  const out = waveControlNet(srf, { amplitude: 5, frequency: 2, direction: 'normal' });
  for (const row of out.ctrlNet) for (const cp of row) for (const v of cp) assert.ok(Number.isFinite(v));
});

// ---- SELF-INTERSECTION-SAFE AMPLITUDE CLAMP, on a real doubly-closed torus ----
test('SELF-INTERSECTION CLAMP: a small, safe amplitude on a real torus (closed in BOTH U and V) is left completely untouched', () => {
  const majorR = 30, minorR = 5;
  const srf = torus(majorR, minorR);
  const out = waveControlNet(srf, { amplitude: minorR * 0.05, axis: 'u', frequency: 2, direction: 'normal' });
  assert.ok(out.ampClamp, 'a real, real-shape amplitude edit always attaches clamp metadata');
  assert.equal(out.ampClamp.clamped, false, 'a small amplitude relative to the tube radius must not be clamped');
  assert.equal(out.ampClamp.applied, out.ampClamp.requested);
});

test('SELF-INTERSECTION CLAMP: a large amplitude on the SAME torus (multiple times the minor radius) is auto-clamped, never silently folded', () => {
  const majorR = 30, minorR = 5;
  const srf = torus(majorR, minorR);
  const requested = minorR * 5; // wildly larger than the tube can carry without self-intersecting
  const out = waveControlNet(srf, { amplitude: requested, axis: 'u', frequency: 2, direction: 'normal' });
  assert.equal(out.ampClamp.requested, requested);
  assert.equal(out.ampClamp.clamped, true, 'a hugely oversized amplitude on a small-radius tube must be clamped, not applied verbatim');
  assert.ok(out.ampClamp.applied < requested, 'the applied amplitude must sit strictly below the request');
  assert.ok(out.ampClamp.applied > 0, 'the clamp must still leave a real, non-zero, usable amplitude');
  assert.equal(out.ampClamp.applied, out.ampClamp.safeMax, 'a clamped result applies EXACTLY the computed safe maximum, not an arbitrary smaller number');
});

test('SELF-INTERSECTION CLAMP: the SAME torus, checked in the OTHER closed direction (axis:v) too — not a lopsided per-axis fix', () => {
  const majorR = 30, minorR = 5;
  const srf = torus(majorR, minorR);
  const requested = minorR * 5;
  const outU = waveControlNet(srf, { amplitude: requested, axis: 'u', frequency: 2, direction: 'normal' });
  const outV = waveControlNet(srf, { amplitude: requested, axis: 'v', frequency: 2, direction: 'normal' });
  assert.equal(outU.ampClamp.clamped, true, 'axis:u direction is clamped');
  assert.equal(outV.ampClamp.clamped, true, 'axis:v direction is ALSO clamped — the protection is not lopsided per axis');
});
