// pointSetDeviation: per-point distance from a sample set to a triangle mesh.
//
// The identical and offset-sphere cases are symmetric, so a measure that
// swapped its roles would pass them; the half plate inside a plate is not, and
// pins which side is sampled.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { pointSetDeviation } from '../kernel/bvh.mjs';

// A UV sphere as an unindexed soup and its vertex list.
function sphere(radius, nu, nv) {
  const at = (i, j) => {
    const phi = (2 * Math.PI * (i % nu)) / nu, theta = (Math.PI * j) / nv;
    return [radius * Math.sin(theta) * Math.cos(phi), radius * Math.sin(theta) * Math.sin(phi), radius * Math.cos(theta)];
  };
  const tris = [], verts = [];
  for (let j = 0; j < nv; j += 1) for (let i = 0; i < nu; i += 1) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    if (j > 0) tris.push(...a, ...b, ...c);
    if (j < nv - 1) tris.push(...a, ...c, ...d);
  }
  for (let j = 0; j <= nv; j += 1) for (let i = 0; i < (j === 0 || j === nv ? 1 : nu); i += 1) verts.push(...at(i, j));
  return { positions: new Float64Array(tris), points: new Float64Array(verts) };
}

// A flat grid plate in z = 0, x in [0, w], y in [0, h], n cells a side.
function plate(w, h, n) {
  const at = (i, j) => [(w * i) / n, (h * j) / n, 0];
  const tris = [], verts = [];
  for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    tris.push(...a, ...b, ...c, ...a, ...c, ...d);
  }
  for (let i = 0; i <= n; i += 1) for (let j = 0; j <= n; j += 1) verts.push(...at(i, j));
  return { positions: new Float64Array(tris), points: new Float64Array(verts) };
}

test('identical meshes deviate by no more than 1e-12', () => {
  for (const m of [sphere(10, 48, 24), plate(30, 20, 12)]) {
    const r = pointSetDeviation(m.points, m.positions);
    assert.ok(r, 'measured');
    assert.equal(r.count, m.points.length / 3);
    assert.ok(r.max <= 1e-12, `max ${r.max}`);
    assert.ok(r.rms <= 1e-12, `rms ${r.rms}`);
  }
});

test('a sphere offset by d deviates by d within 1%, both ways', () => {
  const R = 10, d = 1;
  const inner = sphere(R, 128, 64), outer = sphere(R + d, 112, 56);
  for (const [from, to, name] of [[outer, inner, 'outer to inner'], [inner, outer, 'inner to outer']]) {
    const r = pointSetDeviation(from.points, to.positions);
    assert.ok(Math.abs(r.max - d) <= 0.01 * d, `${name} max ${r.max}`);
    assert.ok(Math.abs(r.rms - d) <= 0.01 * d, `${name} rms ${r.rms}`);
    let lo = Infinity; for (const x of r.distances) lo = Math.min(lo, x);
    assert.ok(Math.abs(lo - d) <= 0.01 * d, `${name} min ${lo}`);
  }
});

test('a half plate inside a plate: A to B is 0, B to A is half the width, and the swap swaps', () => {
  const W = 40, H = 20;
  const B = plate(W, H, 16), A = plate(W / 2, H, 8);
  const ab = pointSetDeviation(A.points, B.positions);
  const ba = pointSetDeviation(B.points, A.positions);
  assert.ok(ab.max <= 1e-12, `A to B max ${ab.max}`);
  assert.ok(Math.abs(ba.max - W / 2) <= 1e-9, `B to A max ${ba.max}`);
  assert.ok(ba.rms > 0 && ba.rms < ba.max, `B to A rms ${ba.rms}`);
  // Every B point over A reads 0; every one past x = W/2 reads its distance to that edge.
  for (let i = 0; i < ba.count; i += 1) {
    const x = B.points[i * 3];
    assert.ok(Math.abs(ba.distances[i] - Math.max(0, x - W / 2)) <= 1e-9, `B point ${i} at x=${x}: ${ba.distances[i]}`);
  }
});

test('nothing measured is null, never 0', () => {
  const m = plate(10, 10, 2);
  assert.equal(pointSetDeviation(new Float64Array(0), m.positions), null);
  assert.equal(pointSetDeviation(m.points, new Float64Array(0)), null);
  assert.equal(pointSetDeviation(null, m.positions), null);
});
