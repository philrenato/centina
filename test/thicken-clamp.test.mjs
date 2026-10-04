// Thicken builds a wall that does not pass through itself, as thick as asked at a corner, and
// facing out wherever the sheet sits.
//
// A wall grown inward past the form's own middle turns the offset sheet inside out: the cage
// still reads clean, the limit crosses itself thousands of times. Thicken holds such a wall at
// the largest thickness that does not cross and says what it used. At a corner the offset vertex
// moved along the averaged normal stands 0.577 of the thickness off each face; on the miter it
// stands the whole thickness off. And the shell's facing is read about its own centroid, so a
// sheet far from the origin comes out facing out too.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { thickenCage } from '../kernel/subdedit.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';
import { selfCrossings } from './fixtures/mesh-crossings.mjs';

const boxV = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const boxF = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
const openBox = (s = 10, t = [0, 0, 0]) => ({ vertices: boxV.map((p) => p.map((c, k) => c * s + t[k])), faces: boxF.filter((_, i) => i !== 1) });
const tube = (n, R, H) => {
  const V = [], F = [];
  for (let k = 0; k < 2; k++) for (let i = 0; i < n; i++) { const a = (i / n) * 2 * Math.PI; V.push([R * Math.cos(a), R * Math.sin(a), k * H]); }
  for (let i = 0; i < n; i++) { const j = (i + 1) % n; F.push([i, j, n + j, n + i]); }
  return { vertices: V, faces: F };
};
const limit = (c) => subdivideCatmullClark(subdivideCatmullClark({ vertices: c.vertices, faces: c.faces, creases: c.creases || {} }));
const volumeAbout = (c, o) => {
  let v = 0;
  for (const f of c.faces) for (let i = 1; i + 1 < f.length; i++) {
    const [a, b, d] = [f[0], f[i], f[i + 1]].map((k) => c.vertices[k].map((x, j) => x - o[j]));
    v += a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0]) + a[2] * (b[0] * d[1] - b[1] * d[0]);
  }
  return v / 6;
};

for (const [name, cage, d] of [['an open tube of radius 5', tube(8, 5, 10), -6], ['an open 10 mm box', openBox(), -12]]) {
  test(`${name} thickened ${d} is held where it stops crossing, and says so`, () => {
    const r = thickenCage(cage, d);
    assert.equal(r.clamped, true, 'reported as held');
    assert.equal(Math.sign(r.distance), Math.sign(d), 'on the side asked');
    assert.ok(Math.abs(r.distance) < Math.abs(d), `held at ${r.distance}`);
    assert.ok(Math.abs(r.distance) > 0.5 * Math.abs(d) * 0.5, `not held far short (${r.distance})`);
    assert.equal(selfCrossings(r.cage.vertices, r.cage.faces), 0, 'cage crossings');
    const L = limit(r.cage);
    assert.equal(selfCrossings(L.vertices, L.faces), 0, 'limit crossings');
    // The hold is the largest: a tenth more crosses, so it is held again.
    const more = thickenCage(cage, r.distance * 1.1);
    assert.equal(more.clamped, true, 'a tenth more is held too');
  });
}

test('a thickness that fits is built as asked', () => {
  for (const [cage, d] of [[openBox(), -1], [openBox(), 1], [tube(8, 5, 10), -2]]) {
    const r = thickenCage(cage, d);
    assert.equal(r.clamped, false);
    assert.equal(r.distance, d);
  }
});

test('a corner wall is as thick as asked', () => {
  const cage = openBox();
  for (const d of [1, -1]) {
    const r = thickenCage(cage, d);
    const n = cage.vertices.length;
    // Every offset vertex stands |d| off each plane of the faces it came from.
    let worst = Infinity;
    cage.faces.forEach((f) => {
      const [A, B, C] = f.map((k) => cage.vertices[k]);
      const u = B.map((x, i) => x - A[i]), w = C.map((x, i) => x - A[i]);
      const N = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]], l = Math.hypot(...N);
      for (const k of f) { const p = r.cage.vertices[k + n]; worst = Math.min(worst, Math.abs(N.reduce((s, x, i) => s + x * (p[i] - A[i]), 0)) / l); }
    });
    assert.ok(Math.abs(worst - Math.abs(d)) < 1e-6 * Math.abs(d), `thinnest wall ${worst.toFixed(3)} for ${d}`);
  }
});

test('a sheet far from the origin comes out facing out', () => {
  const t = [1e6, 1e6, 1e6], centre = [1e6 + 5, 1e6 + 5, 1e6 + 5];
  for (const d of [0.01, -0.01, 1]) {
    const r = thickenCage(openBox(10, t), d);
    assert.ok(volumeAbout(r.cage, centre) > 0, `d ${d}: inside out`);
  }
});
