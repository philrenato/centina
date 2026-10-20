// A size edit on a rigidly deformed exact form: the form is rebuilt at the new
// size from the SOURCE's fields and the deform's own transform is run over the
// rebuilt points. The child's stored centre has already been carried by the
// turn, so rebuilding from it and turning again moves the centre twice.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle } from '../kernel/primitives.mjs';
import { transformPoint } from '../kernel/transform.mjs';

const frame = { center: [5, -3, 2], axisDir: [1, 2, 2].map((v) => v / 3) };
const angle = 0.9;
const turn = (p) => transformPoint('rotate', frame, angle, p);
const sourceCentre = [40, 10, 0], xAxis = [1, 0, 0], yAxis = [0, 1, 0];
// Control points as the app holds them: [center, ...ctrlPts], weights kept.
const turnNet = (centre, radius) => {
  const crv = makeCircle(centre, xAxis, yAxis, radius);
  return { centre: turn(centre), pts: crv.ctrlPts.map((cp) => [...turn(cp.slice(0, 3)), cp[3]]) };
};
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

test('the rebuilt form keeps the turn: a true circle of the new radius about the turned centre', () => {
  const before = turnNet(sourceCentre, 30);
  const after = turnNet(sourceCentre, 5);
  assert.ok(dist(after.centre, before.centre) < 1e-12, 'the centre stays where the turn put it');
  for (const p of after.pts) {
    const r = dist(p, after.centre);
    assert.ok(Math.abs(r - 5) < 1e-9 || Math.abs(r - 5 * Math.SQRT2) < 1e-9, `control point at ${r}`);
  }
  // The circle's plane turned with it: its normal is the turned z axis.
  const n = [0, 1, 2].map((k) => turn([0, 0, 1])[k] - turn([0, 0, 0])[k]);
  for (const p of after.pts) {
    const d = [0, 1, 2].map((k) => p[k] - after.centre[k]);
    assert.ok(Math.abs(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) < 1e-9);
  }
  // Each point is the turned image of the natural circle's point, weight kept.
  const nat = makeCircle(sourceCentre, xAxis, yAxis, 5).ctrlPts;
  after.pts.forEach((p, i) => { assert.ok(dist(p, turn(nat[i].slice(0, 3))) < 1e-12); assert.equal(p[3], nat[i][3]); });
});

test('rebuilding from the child\'s turned centre and turning again moves the centre twice', () => {
  const child = turnNet(sourceCentre, 30);
  const wrong = turnNet(child.centre, 5);
  assert.ok(dist(wrong.centre, child.centre) > 1, `off by ${dist(wrong.centre, child.centre)}`);
});

test('a point-count rebuild keeps the turn too: 6 segments, 13 points about the turned centre', () => {
  const crv = makeCircle(sourceCentre, xAxis, yAxis, 30, 6);
  const c = turn(sourceCentre);
  const pts = crv.ctrlPts.map((cp) => turn(cp.slice(0, 3)));
  assert.equal(pts.length, 13);
  for (const p of pts) {
    const r = dist(p, c);
    assert.ok(Math.abs(r - 30) < 1e-9 || Math.abs(r - 30 / Math.cos(Math.PI / 6)) < 1e-9, `control point at ${r}`);
  }
});
