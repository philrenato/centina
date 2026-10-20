// Hub planarity: the residual both hubs test against their 25% tolerance is
// the largest distance of the arm centers from their least-squares plane, as
// a fraction of the centers' mean distance from their mean. Each case below
// has that number in closed form.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bridgeEdgeRunsHub } from '../kernel/subdedit.mjs';
import { subdPipeNetwork } from '../kernel/subdnetwork.mjs';
import { makeLine } from '../kernel/primitives.mjs';

// N one-quad flaps around the origin; flap k's outer row (the run) sits at
// radius 40 and height +h or -h, alternating, then the whole cage is turned
// by `rot`. For N = 4 the run centers are (±40, 0, h) and (0, ±40, -h): their
// best plane is the flaps' own mid plane, so the residual is h / sqrt(40² + h²).
function flaps(N, h, rot = (p) => p) {
  const vertices = [], faces = [], runs = [];
  for (let k = 0; k < N; k++) {
    const a = (k / N) * Math.PI * 2, z = (k % 2 === 0 ? 1 : -1) * h;
    const ca = Math.cos(a), sa = Math.sin(a);
    const inner = [], outer = [];
    for (let j = 0; j < 3; j++) {
      const s = (j - 1) * 10;
      inner.push(vertices.push(rot([ca * 20 - sa * s, sa * 20 + ca * s, 0])) - 1);
      outer.push(vertices.push(rot([ca * 40 - sa * s, sa * 40 + ca * s, z])) - 1);
    }
    for (let j = 0; j < 2; j++) faces.push([inner[j], inner[j + 1], outer[j + 1], outer[j]]);
    runs.push(outer);
  }
  return { cage: { vertices, faces, creases: {} }, runs };
}
// A turn about an axis that is aligned with nothing, so no coordinate plane
// is the hub's plane.
function turn(p) {
  const k = [1, 2, 3].map((v) => v / Math.sqrt(14)), t = 0.83, c = Math.cos(t), s = Math.sin(t);
  const kd = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
  const kx = [k[1] * p[2] - k[2] * p[1], k[2] * p[0] - k[0] * p[2], k[0] * p[1] - k[1] * p[0]];
  return [0, 1, 2].map((i) => p[i] * c + kx[i] * s + k[i] * kd * (1 - c));
}
const hubResidual = (N, h, rot) => {
  const { cage, runs } = flaps(N, h, rot);
  try { bridgeEdgeRunsHub(cage, runs, 1, 1, 0); return null; } catch (e) {
    const m = /sit (\d+)% of their spread off one plane/.exec(e.message);
    if (!m) throw e;
    return Number(m[1]) / 100;
  }
};

test('bridgeEdgeRunsHub: a planar hub in a turned plane is not refused', () => {
  for (const N of [4, 5, 6]) assert.equal(hubResidual(N, 0, turn), null, `N=${N}`);
});

test('bridgeEdgeRunsHub: a saddle hub reads its true residual, either side of the tolerance', () => {
  // h = 9.5 -> 23.1%, accepted; the widest-pair normal read this hub near twice that.
  assert.equal(hubResidual(4, 9.5), null);
  assert.equal(hubResidual(4, 9.5, turn), null);
  // h = 11 -> 26.5%, refused and reported as 27%.
  const h = 11, want = h / Math.hypot(40, h);
  assert.ok(want > 0.25);
  assert.equal(hubResidual(4, h), Math.round(want * 100) / 100);
  assert.equal(hubResidual(4, h, turn), Math.round(want * 100) / 100);
});

// Four straight rails from the origin, unit directions (cos a cos t,
// sin a cos t, ±sin t) at a = 0, 90, 180, 270 degrees: their mean is the
// origin, each lies 1 from it, and their best plane is z = 0, so the residual
// is exactly sin t.
function saddleRails(sinT, rot = (p) => p) {
  const ct = Math.sqrt(1 - sinT * sinT);
  return [0, 1, 2, 3].map((k) => {
    const a = (k * Math.PI) / 2, z = (k % 2 === 0 ? 1 : -1) * sinT;
    const d = rot([Math.cos(a) * ct, Math.sin(a) * ct, z]);
    return makeLine([0, 0, 0], d.map((v) => v * 60));
  });
}

test('subdPipeNetwork: a planar four-arm junction in a turned plane builds on the ring', () => {
  const net = subdPipeNetwork(saddleRails(0, turn), { radius: 4, facets: 8, segments: 2, junction: 'ring' });
  assert.equal(net.ok, true, net.reason);
});

test('subdPipeNetwork: a saddle junction reports its true residual', () => {
  const inside = subdPipeNetwork(saddleRails(0.24), { radius: 4, facets: 8, segments: 2, junction: 'ring' });
  assert.equal(inside.ok, true, inside.reason);
  for (const rot of [undefined, turn]) {
    const out = subdPipeNetwork(saddleRails(0.3, rot), { radius: 4, facets: 8, segments: 2, junction: 'ring' });
    assert.equal(out.ok, false);
    assert.ok(Math.abs(out.planarityResidual - 0.3) < 1e-9, `residual ${out.planarityResidual}`);
  }
});
