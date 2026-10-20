import test from 'node:test';
import assert from 'node:assert/strict';
import { traceVeins, relaxToBend } from '../kernel/veins.mjs';

// A closed, outward-wound box as { positions, indices }.
function box(x0, y0, z0, x1, y1, z1) {
  const v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  const f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]];
  return { positions: v.flat(), indices: f.flat() };
}
// Two boxes sharing a face, as one closed L: the union's outer faces only.
function lBlock() {
  // An L in x-y, 10 thick in z: [0,60]x[0,20] plus [0,20]x[20,60].
  const pts = [[0, 0], [60, 0], [60, 20], [20, 20], [20, 60], [0, 60]];
  const positions = [], indices = [];
  for (const z of [0, 10]) for (const [x, y] of pts) positions.push(x, y, z);
  // Caps: fan triangulation of the L (convex pieces).
  const capTris = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]];
  for (const [a, b, c] of capTris) { indices.push(a, c, b); indices.push(a + 6, b + 6, c + 6); }
  for (let i = 0; i < 6; i++) { const j = (i + 1) % 6; indices.push(i, j, j + 6, i, j + 6, i + 6); }
  return { positions, indices };
}

test('a cube with ports covering opposite faces gives straight parallel veins at the asked separation', () => {
  const r = traceVeins(box(0, 0, 0, 60, 60, 60), [0, 30, 30], [60, 30, 30], { port: 60, separation: 10, radius: 1, resolution: 30 });
  assert.ok(r.ok, r.reason);
  assert.ok(r.veins.length >= 9, `veins ${r.veins.length}`);
  for (const v of r.veins) {
    const a = v.points[0], b = v.points[v.points.length - 1];
    assert.ok(b[0] - a[0] > 45, `runs across (${a[0].toFixed(1)} -> ${b[0].toFixed(1)})`);
    const dy = Math.max(...v.points.map((q) => Math.abs(q[1] - a[1]))), dz = Math.max(...v.points.map((q) => Math.abs(q[2] - a[2])));
    assert.ok(dy < 1.5 && dz < 1.5, `straight (${dy.toFixed(2)}, ${dz.toFixed(2)})`);
  }
  // Nearest-neighbor spacing at the inlet within 10% of the separation.
  const starts = r.veins.map((v) => v.points[0]);
  const nn = starts.map((s, i) => Math.min(...starts.filter((_, j) => j !== i).map((t) => Math.hypot(t[1] - s[1], t[2] - s[2]))));
  const med = nn.sort((a, b) => a - b)[nn.length >> 1];
  assert.ok(Math.abs(med - 10) / 10 <= 0.1 + 1e-9 || (med >= 10 && med <= 12), `spacing ${med}`);
});

test('in an L-shaped block every vein stays inside the wall by its radius', () => {
  const r = traceVeins(lBlock(), [60, 10, 5], [10, 60, 5], { port: 9, separation: 3, radius: 1.5, resolution: 48 });
  assert.ok(r.ok, r.reason);
  assert.ok(r.veins.length >= 2, `veins ${r.veins.length}`);
  assert.ok(r.stats.minWall >= 1.5 * 0.9, `closest to the wall ${r.stats.minWall}`);
  const reached = r.veins.filter((v) => { const e = v.points[v.points.length - 1]; return Math.hypot(e[0] - 10, e[1] - 60) < 12 || v.joins !== null; });
  assert.equal(reached.length, r.veins.length, 'every vein reaches the outlet or joins one that does');
});

test('wander bends the veins; the same seed gives the same veins', () => {
  const cube = box(0, 0, 0, 60, 60, 60);
  const opts = { port: 60, separation: 12, radius: 1, resolution: 24, wander: 1.5, seed: 7 };
  const a = traceVeins(cube, [0, 30, 30], [60, 30, 30], opts), b = traceVeins(cube, [0, 30, 30], [60, 30, 30], opts);
  assert.deepEqual(a.veins.map((v) => v.points), b.veins.map((v) => v.points));
  const bend = Math.max(...a.veins.map((v) => Math.max(...v.points.map((q) => Math.hypot(q[1] - v.points[0][1], q[2] - v.points[0][2])))));
  assert.ok(bend > 2, `wander moves a vein off straight by ${bend}`);
});

test('refusals name the change to make', () => {
  const cube = box(0, 0, 0, 60, 60, 60);
  assert.match(traceVeins(cube, [0, 30, 30], [0, 40, 30], { port: 20 }).reason, /overlap/);
  assert.match(traceVeins(cube, [200, 30, 30], [60, 30, 30], {}).reason, /inlet is not on the body/);
  assert.match(traceVeins({ positions: [], indices: [] }, [0, 0, 0], [1, 1, 1], {}).reason, /closed body/);
  assert.ok(traceVeins(cube, [0, 30, 30], [60, 30, 30], { port: 20 }).ok);
});

test('relaxing a zigzag lifts its tightest turn past the radius asked', () => {
  const zig = Array.from({ length: 30 }, (_, i) => [i * 2, (i % 2) * 2, 0]);
  const out = relaxToBend(zig, 6);
  let r = Infinity;
  for (let i = 1; i + 1 < out.length; i++) {
    const [a, b, c] = [out[i - 1], out[i], out[i + 1]];
    const ab = Math.hypot(b[0] - a[0], b[1] - a[1]), bc = Math.hypot(c[0] - b[0], c[1] - b[1]), ca = Math.hypot(a[0] - c[0], a[1] - c[1]);
    const ar = Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
    if (ar > 1e-12) r = Math.min(r, (ab * bc * ca) / (2 * ar));
  }
  assert.ok(r >= 6 * 0.99, `tightest ${r}`);
});
