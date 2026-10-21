// The self-collision hold: a displacement that passes the surface through
// itself is held back until no new pair crosses, only where it crosses; a
// displacement that crosses nothing is returned untouched; strength 0 is none;
// a crossing the base already had is not the deformer's to hold.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { holdSelfCollision, holdCageSelfCollision, collisionMesh, collisionPositions, collidingPairs } from '../kernel/selfcollision.mjs';
import { makeLine, makeCircle, revolve } from '../kernel/primitives.mjs';
import { noiseControlNet, refineSurface } from '../kernel/noise.mjs';

const tube = refineSurface(revolve(makeLine([10, 0, 0], [10, 0, 40]), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2), 3);
const mid = Math.floor(tube.ctrlNet.length / 2);
const pushRing = (srf, f) => ({ ...srf, ctrlNet: srf.ctrlNet.map((row, i) => row.map((cp) => (i === mid ? [cp[0] * f, cp[1] * f, cp[2], cp[3]] : [...cp]))) });
const crossings = (base, srf) => {
  const m = collisionMesh(base, srf);
  const own = collidingPairs(m.basePos, m.tris);
  let n = 0;
  for (const k of collidingPairs(collisionPositions(m, new Float64Array(m.nu * m.nv).fill(1)), m.tris)) if (!own.has(k)) n++;
  return n;
};

test('a ring pushed through the axis is held until nothing crosses', () => {
  const pushed = pushRing(tube, -0.6);
  assert.ok(crossings(tube, pushed) > 0);
  const r = holdSelfCollision(tube, pushed, 1);
  assert.ok(r.collided > 0 && r.held);
  assert.equal(crossings(tube, r.srf), 0);
});

test('the hold is local: rings far from the crossing keep their whole move', () => {
  const long = refineSurface(revolve(makeLine([10, 0, 0], [10, 0, 80]), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2), 4);
  const m = Math.floor(long.ctrlNet.length / 2);
  const pushed = { ...long, ctrlNet: long.ctrlNet.map((row, i) => row.map((cp) => (i === m ? [cp[0] * -0.6, cp[1] * -0.6, cp[2], cp[3]] : i === 2 ? [cp[0] * 1.3, cp[1] * 1.3, cp[2], cp[3]] : [...cp]))) };
  const r = holdSelfCollision(long, pushed, 1);
  assert.ok(r.held && r.hold[m].every((h) => h > 0));
  assert.ok(r.hold[2].every((h) => h === 0));
  assert.deepEqual(r.srf.ctrlNet[2], pushed.ctrlNet[2]);
});

test('a move that crosses nothing is returned as it was', () => {
  const pushed = pushRing(tube, 0.5);
  const r = holdSelfCollision(tube, pushed, 1);
  assert.equal(r.collided, 0);
  assert.equal(r.srf, pushed);
});

test('strength 0 is none; half strength is half the strict hold', () => {
  const pushed = pushRing(tube, -0.6);
  assert.equal(holdSelfCollision(tube, pushed, 0).srf, pushed);
  const strict = holdSelfCollision(tube, pushed, 1), half = holdSelfCollision(tube, pushed, 0.5);
  assert.ok(Math.abs(half.maxHold - strict.maxHold / 2) < 1e-12);
});

test('Noise on a torus that crosses itself is held clear', () => {
  const tor = revolve(makeCircle([30, 0, 0], [1, 0, 0], [0, 0, 1], 10), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const out = noiseControlNet(tor, { style: 'sine', direction: 'world-z', frequency: 4, seed: 2, amplitude: 1000 });
  assert.ok(crossings(tor, out) > 0);
  const r = holdSelfCollision(tor, out, 1);
  assert.ok(r.held);
  assert.equal(crossings(tor, r.srf), 0);
});

test('a crossing the base already has is not held', () => {
  const spindle = revolve(makeCircle([8, 0, 0], [1, 0, 0], [0, 0, 1], 12), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const m = collisionMesh(spindle, spindle);
  assert.ok(collidingPairs(m.basePos, m.tris).size > 0);
  const nudged = { ...spindle, ctrlNet: spindle.ctrlNet.map((row) => row.map((cp) => [cp[0] * 1.01, cp[1] * 1.01, cp[2], cp[3]])) };
  const r = holdSelfCollision(spindle, nudged, 1);
  assert.ok(r.maxHold < 1);
});

test('a cage vertex pushed through the opposite face is held', () => {
  const vertices = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10], [5, 5, 10]];
  const faces = [[0, 3, 2, 1], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8]];
  const base = { vertices, faces };
  const deformed = { vertices: vertices.map((v, i) => (i === 8 ? [5, 5, -6] : v)), faces };
  const r = holdCageSelfCollision(base, deformed, 1);
  assert.ok(r.collided > 0 && r.held);
  assert.ok(r.cage.vertices[8][2] > 0, `apex held at z ${r.cage.vertices[8][2]}`);
  assert.equal(holdCageSelfCollision(base, deformed, 0).cage, deformed);
});
