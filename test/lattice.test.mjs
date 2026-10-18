import test from 'node:test';
import assert from 'node:assert/strict';
import {
  makeLattice, latticeFrame, latticeDeformPoint, latticeDeformPoints, latticeEval,
  latticeSetCount, latticeIsRest, latticeDrift, latticeIndex, cloneLattice,
  bsplineBasis, uniformClampedKnots, latticeMapPoints, latticeLocal,
} from '../kernel/lattice.mjs';

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function cloud(n = 200, seed = 7) {
  let s = seed;
  const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  const pts = [];
  for (let i = 0; i < n; i++) pts.push([r() * 80 - 40, r() * 30 - 5, r() * 50 + 10]);
  return pts;
}

// A lattice with a few points moved off rest, deterministically.
function moved(lattice) {
  const out = cloneLattice(lattice);
  out.pts.forEach((p, i) => {
    p[0] += 3 * Math.sin(i * 1.7); p[1] += 2 * Math.cos(i * 0.9); p[2] += 4 * Math.sin(i * 0.4 + 1);
  });
  return out;
}

const gridSamples = () => {
  const out = [];
  for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) for (let k = 0; k <= 6; k++) out.push([i / 6, j / 6, k / 6]);
  return out;
};

test('basis partitions unity at every count the lattice allows', () => {
  for (let n = 2; n <= 12; n++) {
    const p = Math.min(3, n - 1);
    const kn = uniformClampedKnots(n, p);
    for (const x of [0, 0.13, 0.5, 0.77, 1]) {
      const b = bsplineBasis(kn, p, n, x);
      assert.ok(Math.abs(b.reduce((a, c) => a + c, 0) - 1) < 1e-12, `n=${n} x=${x}`);
    }
  }
});

test('an unmoved lattice maps every point inside it to itself, at every count, world and part frames', () => {
  const pts = cloud();
  for (const mode of ['world', 'part']) {
    const frame = latticeFrame(pts, mode);
    for (const counts of [[2, 2, 2], [3, 2, 4], [4, 4, 4], [5, 3, 2], [7, 6, 9], [12, 2, 5]]) {
      const L = makeLattice(frame, counts);
      assert.ok(latticeIsRest(L));
      for (const p of pts) assert.ok(dist(latticeDeformPoint(L, p), p) < 1e-9, `${mode} ${counts} ${p}`);
    }
  }
});

test('a point outside the box past the margin does not move; one inside the margin moves less than inside', () => {
  const L = moved(makeLattice({ origin: [0, 0, 0], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], size: [10, 10, 10] }, [3, 3, 3]));
  assert.deepEqual(latticeDeformPoint(L, [50, 5, 5]), [50, 5, 5]);
  const edge = latticeDeformPoint(L, [10, 5, 5]);
  const near = latticeDeformPoint(L, [10.5, 5, 5]);
  assert.ok(dist(near, [10.5, 5, 5]) < dist(edge, [10, 5, 5]) + 1e-9);
});

test('raising a count keeps a moved lattice\'s shape exactly (elevation, then insertion), each direction', () => {
  const frame = latticeFrame(cloud(), 'world');
  for (const axis of [0, 1, 2]) {
    let L = moved(makeLattice(frame, [2, 3, 2]));
    const ref = cloneLattice(L);
    for (let n = L.counts[axis] + 1; n <= 9; n++) {
      const r = latticeSetCount(L, axis, n);
      assert.equal(r.exact, true);
      assert.equal(r.lattice.counts[axis], n);
      let worst = 0;
      for (const [s, t, u] of gridSamples()) worst = Math.max(worst, dist(latticeEval(r.lattice, s, t, u), latticeEval(ref, s, t, u)));
      assert.ok(worst < 1e-9, `axis ${axis} n ${n} worst ${worst}`);
      L = r.lattice;
    }
  }
});

test('a jump of several counts at once is exact and deforms points the same', () => {
  const pts = cloud(120);
  const L = moved(makeLattice(latticeFrame(pts, 'part'), [2, 2, 2]));
  const r = latticeSetCount(L, 0, 5);
  const a = latticeDeformPoints(L, pts), b = latticeDeformPoints(r.lattice, pts);
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, dist(a[i], b[i]));
  assert.ok(worst < 1e-9, `${worst}`);
});

test('a raised then lowered count comes back with no drift; lowering a busier lattice reports a positive drift', () => {
  const frame = latticeFrame(cloud(), 'world');
  const L = moved(makeLattice(frame, [3, 3, 3]));
  const up = latticeSetCount(L, 1, 6).lattice;
  const down = latticeSetCount(up, 1, 3);
  assert.equal(down.lattice.counts[1], 3);
  assert.ok(down.drift < 1e-7, `round trip drift ${down.drift}`);
  const busy = moved(makeLattice(frame, [6, 6, 6]));
  const fewer = latticeSetCount(busy, 2, 3);
  assert.ok(fewer.drift > 1e-3, `drift ${fewer.drift}`);
  assert.equal(fewer.exact, false);
  assert.ok(Math.abs(fewer.drift - latticeDrift(busy, fewer.lattice)) < 1e-12);
});

test('lowering an unmoved lattice keeps it the identity', () => {
  const pts = cloud(80);
  const L = makeLattice(latticeFrame(pts, 'world'), [7, 5, 4]);
  const r = latticeSetCount(L, 0, 3);
  assert.ok(r.drift < 1e-9);
  for (const p of pts) assert.ok(dist(latticeDeformPoint(r.lattice, p), p) < 1e-9);
});

test('moving one control point moves the volume only near it (local support past 4 points)', () => {
  const L = makeLattice({ origin: [0, 0, 0], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], size: [100, 100, 100] }, [8, 8, 8]);
  const M = cloneLattice(L);
  const idx = latticeIndex(M, 0, 0, 0);
  M.pts[idx] = [M.pts[idx][0] - 10, M.pts[idx][1], M.pts[idx][2]];
  assert.ok(dist(latticeDeformPoint(M, [95, 95, 95]), [95, 95, 95]) < 1e-12);
  assert.ok(dist(latticeDeformPoint(M, [1, 1, 1]), [1, 1, 1]) > 1);
});

test('an affine map of the points maps the deformed shape by the same map', () => {
  const pts = cloud(60);
  const L = moved(makeLattice(latticeFrame(pts, 'world'), [4, 3, 5]));
  const shift = [5, -2, 7];
  const T = latticeMapPoints(L, (p) => [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]);
  for (const p of pts) {
    const a = latticeDeformPoint(L, p), b = latticeDeformPoint(T, p);
    assert.ok(dist([a[0] + shift[0], a[1] + shift[1], a[2] + shift[2]], b) < 1e-9);
  }
});

test('the part frame is right-handed and orthonormal and holds every point', () => {
  const pts = cloud(150).map(([x, y, z]) => [x + 0.5 * y, y - 0.3 * x, z]);
  const f = latticeFrame(pts, 'part');
  const [a, b, c] = f.axes;
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  assert.ok(Math.abs(dot(a, b)) < 1e-9 && Math.abs(dot(a, c)) < 1e-9 && Math.abs(dot(b, c)) < 1e-9);
  const cr = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  assert.ok(dot(cr, c) > 0.999);
  const L = makeLattice(f, [2, 2, 2]);
  for (const p of pts) assert.ok(latticeLocal(L, p).every((x) => x > 0 && x < 1));
});

// ---- Earlier per-surface bands open as lattices with the same shape.
import { makeRestLattice, deformWithBands } from '../kernel/cage.mjs';
import { latticeFromLegacyBand, latticeDeformPointsMany, latticeTransformed } from '../kernel/lattice.mjs';

function legacyBand(min, max, d, seed) {
  const b = makeRestLattice(min, max, d[0], d[1], d[2]);
  let s = seed;
  for (const plane of b.ctrlPts) for (const row of plane) for (const cp of row) {
    s = (s * 48271) % 2147483647; cp[0] += (s / 2147483647 - 0.5) * 9;
    s = (s * 48271) % 2147483647; cp[1] += (s / 2147483647 - 0.5) * 9;
    s = (s * 48271) % 2147483647; cp[2] += (s / 2147483647 - 0.5) * 9;
  }
  return b;
}
// Points inside the box, inside the fade margin and past it.
function probePoints(min, max) {
  const out = [];
  for (const f of [-0.3, -0.08, -0.02, 0, 0.2, 0.5, 0.77, 1, 1.03, 1.09, 1.4]) {
    for (const g of [0.1, 0.5, 1.05]) {
      out.push([min[0] + f * (max[0] - min[0]), min[1] + g * (max[1] - min[1]), min[2] + (1 - f) * (max[2] - min[2])]);
    }
  }
  return out;
}

test('an earlier band opens as a lattice that deforms every point the same, inside, in the fade and past it', () => {
  const min = [-20, -5, 3], max = [40, 25, 33];
  for (const d of [[2, 2, 2], [3, 3, 3], [4, 2, 5], [5, 5, 5], [6, 3, 2]]) {
    const band = legacyBand(min, max, d, 11 + d[0] * 7 + d[2]);
    const L = latticeFromLegacyBand(band);
    for (const p of probePoints(min, max)) {
      const a = deformWithBands([band], p), b = latticeDeformPointsMany([L], [p])[0];
      assert.ok(dist(a, b) < 1e-9, `${d} ${p} ${dist(a, b)}`);
    }
  }
});

test('several earlier bands on different boxes add the same way', () => {
  const b1 = legacyBand([-20, -5, 3], [40, 25, 33], [2, 2, 2], 5);
  const b2 = legacyBand([-25, -8, 0], [45, 28, 36], [3, 3, 3], 9);
  const b3 = legacyBand([0, 0, 10], [30, 20, 30], [4, 4, 4], 13);
  const Ls = [b1, b2, b3].map(latticeFromLegacyBand);
  for (const p of probePoints([-25, -8, 0], [45, 28, 36])) {
    const a = deformWithBands([b1, b2, b3], p), b = latticeDeformPointsMany(Ls, [p])[0];
    assert.ok(dist(a, b) < 1e-9, `${p} ${dist(a, b)}`);
  }
});

test('a converted band takes more points without moving the shape, inside and in the fade', () => {
  const min = [-20, -5, 3], max = [40, 25, 33];
  const L = latticeFromLegacyBand(legacyBand(min, max, [5, 3, 2], 21));
  const up = latticeSetCount(latticeSetCount(L, 0, 8).lattice, 2, 4).lattice;
  for (const p of probePoints(min, max)) {
    assert.ok(dist(latticeDeformPoint(L, p), latticeDeformPoint(up, p)) < 1e-9, `${p}`);
  }
  const down = latticeSetCount(L, 0, 4);
  assert.ok(down.lattice && down.lattice.counts[0] === 4 && Number.isFinite(down.drift));
});

test('a converted band carried through an object transform deforms the transformed points the same', () => {
  const band = legacyBand([-20, -5, 3], [40, 25, 33], [3, 2, 4], 31);
  const L = latticeFromLegacyBand(band);
  const c = Math.cos(0.6), s = Math.sin(0.6), sx = 1.5, sy = 0.8, sz = 2;
  const e = [c * sx, s * sx, 0, 0, -s * sy, c * sy, 0, 0, 0, 0, sz, 0, 7, -3, 11, 1];
  const M = (q) => [e[0] * q[0] + e[4] * q[1] + e[8] * q[2] + e[12], e[1] * q[0] + e[5] * q[1] + e[9] * q[2] + e[13], e[2] * q[0] + e[6] * q[1] + e[10] * q[2] + e[14]];
  const T = latticeTransformed(L, e);
  for (const p of probePoints([-20, -5, 3], [40, 25, 33])) {
    assert.ok(dist(M(latticeDeformPoint(L, p)), latticeDeformPoint(T, M(p))) < 1e-9, `${p}`);
  }
});
