// Revolve along rails (skinned through 5-degree sections, so a circle is held to
// SKIN mm): a rail matching the profile reproduces the revolve, a rail
// off the profile pulls the gripped point onto it at every angle, two rails hold
// their own ends, a part turn is open and a full turn closes without a seam.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { railRevolve, railTable, railAt, railRevolveFrame } from '../kernel/railrevolve.mjs';
import { makeLine, makeCircle, makeEllipse } from '../kernel/primitives.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

const prof = makeLine([10, 0, 0], [10, 0, 20]);
const ring = (r, z) => makeCircle([0, 0, z], [1, 0, 0], [0, 1, 0], r);
const at = (srf, u, v) => {
  const ku = srf.knotsU, kv = srf.knotsV;
  return surfacePoint(srf, ku[0] + (ku[ku.length - 1] - ku[0]) * u, kv[0] + (kv[kv.length - 1] - kv[0]) * v);
};
const rOf = (p) => Math.hypot(p[0], p[1]);
const SKIN = 1e-4;
const FRACS = [0, 0.1, 0.25, 0.4, 0.5, 0.66, 0.9];

test('a rail on the profile reproduces the plain revolve', () => {
  const r = railRevolve(prof, [ring(10, 20)], [0, 0, 0], [0, 0, 1], Math.PI * 2);
  assert.equal(r.ok, true);
  for (const v of FRACS) for (const u of [0, 0.5, 1]) assert.ok(Math.abs(rOf(at(r.srf, u, v)) - 10) < SKIN);
});

test('a rail off the profile pulls the gripped point onto it at every angle', () => {
  const r = railRevolve(prof, [ring(15, 20)], [0, 0, 0], [0, 0, 1], Math.PI * 2);
  for (const v of FRACS) {
    const top = at(r.srf, 1, v);
    assert.ok(Math.abs(rOf(top) - 15) < SKIN, `top r ${rOf(top)}`);
    assert.ok(Math.abs(top[2] - 20) < 1e-9);
  }
});

test('two rails hold their own ends; between them the profile blends', () => {
  const r = railRevolve(prof, [ring(5, 0), ring(15, 20)], [0, 0, 0], [0, 0, 1], Math.PI * 2);
  for (const v of FRACS) {
    assert.ok(Math.abs(rOf(at(r.srf, 0, v)) - 5) < SKIN);
    assert.ok(Math.abs(rOf(at(r.srf, 1, v)) - 15) < SKIN);
    assert.ok(Math.abs(rOf(at(r.srf, 0.5, v)) - 10) < SKIN);
  }
});

test('an elliptical rail is followed to within a section step', () => {
  const r = railRevolve(prof, [makeEllipse([0, 0, 20], [1, 0, 0], [0, 1, 0], 10, 20)], [0, 0, 0], [0, 0, 1], Math.PI * 2);
  for (const deg of [30, 45, 60, 90, 135]) {
    const t = (deg * Math.PI) / 180, want = 1 / Math.sqrt(Math.cos(t) ** 2 / 100 + Math.sin(t) ** 2 / 400);
    assert.ok(Math.abs(rOf(at(r.srf, 1, deg / 360)) - want) < 0.05, `${deg}: ${rOf(at(r.srf, 1, deg / 360))} vs ${want}`);
  }
});

test('a full turn closes with matching tangents; a part turn is open', () => {
  const full = railRevolve(prof, [makeEllipse([0, 0, 20], [1, 0, 0], [0, 1, 0], 10, 20)], [0, 0, 0], [0, 0, 1], Math.PI * 2);
  assert.equal(full.closed, true);
  const a = at(full.srf, 1, 0), b = at(full.srf, 1, 1);
  assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-9);
  const e = 1e-4, a1 = at(full.srf, 1, e), b1 = at(full.srf, 1, 1 - e);
  const da = [a1[0] - a[0], a1[1] - a[1]], db = [b[0] - b1[0], b[1] - b1[1]];
  const cos = (da[0] * db[0] + da[1] * db[1]) / (Math.hypot(...da) * Math.hypot(...db));
  assert.ok(cos > 0.9999, `seam tangent cos ${cos}`);
  const half = railRevolve(prof, [ring(15, 20)], [0, 0, 0], [0, 0, 1], Math.PI);
  assert.equal(half.closed, false);
  const end = at(half.srf, 1, 1);
  assert.ok(Math.abs(Math.atan2(end[1], end[0]) - Math.PI) < 1e-6 || Math.abs(Math.atan2(end[1], end[0]) + Math.PI) < 1e-6);
});

test('a rail that does not reach an angle holds its nearest end', () => {
  const arc = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[12, 0, 20, 1], [0, 12, 20, 1]] };
  const frame = railRevolveFrame(prof, [0, 0, 0], [0, 0, 1]);
  const tb = railTable(arc, frame);
  assert.ok(Math.abs(railAt(tb, Math.PI).r - 12) < 1e-9);
  const r = railRevolve(prof, [arc], [0, 0, 0], [0, 0, 1], Math.PI * 2);
  assert.equal(r.ok, true);
  for (const v of FRACS) assert.ok(Number.isFinite(rOf(at(r.srf, 1, v))));
});

test('no axis direction is refused by name', () => {
  assert.equal(railRevolve(prof, [ring(15, 20)], [0, 0, 0], [0, 0, 0], Math.PI).ok, false);
});
