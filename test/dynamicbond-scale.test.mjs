import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dynamicBondFrame, createDynamicBondSimulation, createHostContact, dynamicBondContactSpan, dynamicBondOptions,
  dynamicBondSettleStep, settleDynamicBond } from '../kernel/dynamicbond.mjs';

// A cantilever 5 segments long with a square section a fifth of its length,
// pinned by its end face ring the way a bind pins the seat's faces.
function beam(length) {
  const n = 5, s = length / n, w = length / 5, vertices = [], faces = [];
  for (let i = 0; i <= n; i++) for (const [y, z] of [[0, 0], [w, 0], [w, w], [0, w]]) vertices.push([i * s, y, z]);
  const id = (i, c) => 4 * i + c;
  faces.push([0, 3, 2, 1], [id(n, 0), id(n, 1), id(n, 2), id(n, 3)]);
  for (let i = 0; i < n; i++) for (let c = 0; c < 4; c++) { const d = (c + 1) % 4; faces.push([id(i, c), id(i, d), id(i + 1, d), id(i + 1, c)]); }
  return { vertices, faces };
}
const PARAMS = { gravity: 9.81, softness: 0.3, damping: 2, volume: true };
function settle(length, floor = null) {
  const cage = beam(length), pins = [0, 1, 2, 3];
  const frame = dynamicBondFrame([0, 0, 0], [1, 0, 0], [0, 1, 0]);
  const sim = createDynamicBondSimulation({ cage, frame, pinned: pins, step: dynamicBondSettleStep(pins) });
  let contact = null;
  if (floor != null) {
    const z = floor * length;
    const closest = (p, max) => (Math.abs(p[2] - z) <= max ? { point: [p[0], p[1], z], normal: [0, 0, 1] } : null);
    contact = createHostContact({ closest, cage, ...dynamicBondContactSpan(sim) });
  }
  settleDynamicBond(sim, frame, dynamicBondOptions(PARAMS, contact, sim));
  return { x: Array.from(sim.solver.x, (v) => v / length), time: sim.solver.time };
}
const tipSag = (x) => -Math.min(x[62], x[65], x[68], x[71]);

// The settle is the same shape at every size: compliance and gravity read the
// body's own length, so a body scaled by k settles to the shape scaled by k.
test('a cantilever settles to the same shape scaled by k for k in 0.1, 1, 10, 100', () => {
  const ref = settle(40);
  assert.ok(tipSag(ref.x) > 0.3 && tipSag(ref.x) < 0.9, `the 40 mm cantilever sags ${tipSag(ref.x)} of its length`);
  for (const k of [0.1, 10, 100]) {
    const r = settle(40 * k);
    let worst = 0;
    for (let i = 0; i < ref.x.length; i++) worst = Math.max(worst, Math.abs(r.x[i] - ref.x[i]));
    assert.ok(worst < 1e-6, `k=${k}: sag ${tipSag(r.x).toFixed(4)} vs ${tipSag(ref.x).toFixed(4)}, worst ${worst} of the length`);
    assert.ok(Math.abs(r.time - ref.time) < 1e-9, `k=${k}: settled in ${r.time} s vs ${ref.time} s`);
  }
});
test('resting on a host is scale-invariant too', () => {
  const ref = settle(40, -0.3);
  assert.ok(Math.min(...ref.x.filter((_, i) => i % 3 === 2)) > -0.3, 'the floor holds the 40 mm beam');
  for (const k of [0.1, 100]) {
    const r = settle(40 * k, -0.3);
    let worst = 0;
    for (let i = 0; i < ref.x.length; i++) worst = Math.max(worst, Math.abs(r.x[i] - ref.x[i]));
    assert.ok(worst < 1e-6, `k=${k}: worst ${worst} of the length`);
  }
});
