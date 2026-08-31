import test from 'node:test';
import assert from 'node:assert';
import { massProperties, volumeIdentityResidual } from '../kernel/massprops.mjs';

// Every check below compares against a CLOSED-FORM answer for the same solid
// — a box's own w*h*d, a tetrahedron's own det/6, a sphere's 4/3 pi r^3 — not
// against a second run of the code under test. That is the whole point: the
// oracle these feed is only worth anything if the measurement itself was
// proven against something it cannot have derived from.

// An axis-aligned box as six OUTWARD-wound quad loops.
function boxLoops(lo, hi) {
  const [x0, y0, z0] = lo, [x1, y1, z1] = hi;
  return [
    [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]], // z = z0, outward is -z
    [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], // z = z1, outward is +z
    [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], // y = y0
    [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]], // y = y1
    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], // x = x0
    [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], // x = x1
  ];
}

test('massProperties: an axis-aligned box, well away from the origin, matches its own closed form exactly', () => {
  // Deliberately off-origin and non-cubic: an origin-centred cube would let a
  // sign error, a centroid error and a missing translation term all cancel.
  const lo = [7, -13, 3], hi = [7 + 4, -13 + 6, 3 + 9];
  const w = 4, h = 6, d = 9;
  const m = massProperties(boxLoops(lo, hi));

  assert.ok(Math.abs(m.volume - w * h * d) < 1e-9, `volume ${m.volume} != ${w * h * d}`);
  const expectedArea = 2 * (w * h + h * d + w * d);
  assert.ok(Math.abs(m.area - expectedArea) < 1e-9, `area ${m.area} != ${expectedArea}`);
  for (let k = 0; k < 3; k++) {
    const expected = (lo[k] + hi[k]) / 2;
    assert.ok(Math.abs(m.centroid[k] - expected) < 1e-9, `centroid[${k}] ${m.centroid[k]} != ${expected}`);
  }
  assert.deepStrictEqual(m.bbox.lo, lo);
  assert.deepStrictEqual(m.bbox.hi, hi);
  assert.ok(m.closureResidual < 1e-14, `a closed box must have ~zero closure residual, got ${m.closureResidual}`);
});

test("massProperties: the box's inertia tensor about its centroid matches m/12*(b^2+c^2), off-diagonals zero", () => {
  const w = 4, h = 6, d = 9;
  const m = massProperties(boxLoops([7, -13, 3], [7 + w, -13 + h, 3 + d]));
  const M = m.volume; // unit density
  const I = m.inertiaCentroid;
  const exp = [M * (h * h + d * d) / 12, M * (w * w + d * d) / 12, M * (w * w + h * h) / 12];
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(I[k * 3 + k] - exp[k]) < 1e-6, `I[${k}][${k}] ${I[k * 3 + k]} != ${exp[k]}`);
  }
  // A box aligned with the axes has no products of inertia at all — the
  // strongest single check that the parallel-axis translation is right, since
  // the tensor about the ORIGIN (7,-13,3 away) has large off-diagonals.
  for (const [r, q] of [[0, 1], [0, 2], [1, 2]]) {
    assert.ok(Math.abs(I[r * 3 + q]) < 1e-6, `off-diagonal I[${r}][${q}] should be 0, got ${I[r * 3 + q]}`);
    assert.ok(Math.abs(I[r * 3 + q] - I[q * 3 + r]) < 1e-9, 'the tensor must be symmetric');
  }
  assert.ok(Math.abs(m.inertiaOrigin[1]) > 1, 'the ORIGIN tensor should have real products of inertia here');
});

test('massProperties: an arbitrary tetrahedron matches det/6, exactly', () => {
  const a = [1, 0, 0], b = [0, 2, 0], c = [0, 0, 5], o = [3, -2, 1];
  const p = [o, [o[0] + a[0], o[1] + a[1], o[2] + a[2]],
    [o[0] + b[0], o[1] + b[1], o[2] + b[2]], [o[0] + c[0], o[1] + c[1], o[2] + c[2]]];
  // Outward winding for a tet whose (a,b,c) form a right-handed frame.
  const loops = [[p[0], p[2], p[1]], [p[0], p[1], p[3]], [p[0], p[3], p[2]], [p[1], p[2], p[3]]];
  const m = massProperties(loops);
  const expected = (1 * 2 * 5) / 6;
  assert.ok(Math.abs(m.volume - expected) < 1e-12, `tetrahedron volume ${m.volume} != ${expected}`);
  // A tetrahedron's centroid is the plain average of its four vertices.
  for (let k = 0; k < 3; k++) {
    const avg = (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4;
    assert.ok(Math.abs(m.centroid[k] - avg) < 1e-12, `tet centroid[${k}] ${m.centroid[k]} != ${avg}`);
  }
  assert.ok(m.closureResidual < 1e-14, 'a closed tetrahedron must have ~zero closure residual');
});

test('massProperties: a NON-CONVEX planar face is measured correctly by the fan — convexity is not assumed', () => {
  // An L-shaped prism. If the fan from vertex 0 were only valid for a convex
  // loop, the escaping triangles would not cancel and this volume would be
  // wrong by exactly the notch.
  const L = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]];
  const zLo = 0, zHi = 2;
  const bot = L.map(([x, y]) => [x, y, zLo]);
  const top = L.map(([x, y]) => [x, y, zHi]);
  const loops = [bot.slice().reverse(), top];
  for (let i = 0; i < L.length; i++) {
    const j = (i + 1) % L.length;
    loops.push([bot[i], bot[j], top[j], top[i]]);
  }
  const m = massProperties(loops);
  const footprint = 3 * 1 + 1 * 2; // the L, as two rectangles
  assert.ok(Math.abs(m.volume - footprint * (zHi - zLo)) < 1e-9,
    `non-convex prism volume ${m.volume} != ${footprint * (zHi - zLo)}`);
  assert.ok(m.closureResidual < 1e-14, 'the L-prism must still read as closed');
});

test('massProperties: a sphere tessellation CONVERGES on 4/3 pi r^3 from below, as an inscribed shell must', () => {
  const R = 5;
  const build = (n) => {
    const P = [];
    for (let i = 0; i <= n; i++) {
      const th = Math.PI * (i / n);
      P.push([]);
      for (let j = 0; j < 2 * n; j++) {
        const ph = 2 * Math.PI * (j / (2 * n));
        P[i].push([R * Math.sin(th) * Math.cos(ph), R * Math.sin(th) * Math.sin(ph), R * Math.cos(th)]);
      }
    }
    const loops = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 2 * n; j++) {
        const j2 = (j + 1) % (2 * n);
        loops.push([P[i][j], P[i + 1][j], P[i + 1][j2], P[i][j2]]);
      }
    }
    return massProperties(loops);
  };
  const truth = (4 / 3) * Math.PI * R ** 3;
  const coarse = build(16), mid = build(32), fine = build(64);
  for (const m of [coarse, mid, fine]) {
    assert.ok(m.volume < truth, 'an inscribed polyhedron is always SMALLER than the sphere it is inscribed in');
  }
  // The load-bearing claim is the convergence ORDER, not a magic percentage:
  // a second-order scheme divides its error by 4 on every doubling. A number
  // picked to fit one resolution would pass a first-order bug too.
  const e = [coarse, mid, fine].map((m) => (truth - m.volume) / truth);
  for (const k of [0, 1]) {
    const ratio = e[k] / e[k + 1];
    assert.ok(ratio > 3.5 && ratio < 4.5,
      `doubling the bands must quarter the error (second order); got a ratio of ${ratio.toFixed(3)}`);
  }
  assert.ok(e[2] < 2e-3, `a 64-band sphere should already be within 0.2%, got ${(e[2] * 100).toFixed(4)}%`);
  // The centroid is exact at ANY resolution — it is fixed by symmetry, not by
  // how finely the surface is cut, so this checks the weighting rather than
  // the discretisation.
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(coarse.centroid[k]) < 1e-9);
  // A sphere's inertia is isotropic: 2/5 m r^2, approached from below with it.
  const iso = (2 / 5) * fine.volume * R * R;
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(fine.inertiaCentroid[k * 3 + k] - iso) / iso < 5e-3,
      `sphere inertia I[${k}][${k}] should be ~2/5 m r^2 (${iso}), got ${fine.inertiaCentroid[k * 3 + k]}`);
  }
});

test('massProperties: an INSIDE-OUT shell reports a negative volume rather than hiding it', () => {
  const flipped = boxLoops([0, 0, 0], [2, 3, 4]).map((l) => l.slice().reverse());
  const m = massProperties(flipped);
  assert.ok(m.volume < 0, `an inside-out box must report a negative volume, got ${m.volume}`);
  assert.ok(Math.abs(Math.abs(m.volume) - 24) < 1e-9, 'its magnitude is still the true volume');
  assert.ok(m.closureResidual < 1e-14, 'consistently reversed is still consistently wound, so still closed');
});

test('massProperties: closureResidual genuinely detects an OPEN shell and a SINGLE flipped face', () => {
  const closed = boxLoops([0, 0, 0], [2, 3, 4]);
  assert.ok(massProperties(closed).closureResidual < 1e-14);

  const open = closed.slice(0, 5); // lid removed
  assert.ok(massProperties(open).closureResidual > 0.1,
    'a box missing a face must fail the closed-surface theorem, loudly');

  const oneFlipped = closed.map((l, i) => (i === 2 ? l.slice().reverse() : l));
  assert.ok(massProperties(oneFlipped).closureResidual > 0.1,
    'one face wound the wrong way must fail it too — this is the case chi and the naked-edge count both pass');
});

test('massProperties: refuses malformed input by name instead of measuring nonsense', () => {
  assert.throws(() => massProperties([]), /at least one face loop/);
  assert.throws(() => massProperties([[[0, 0, 0], [1, 0, 0]]]), /fewer than 3 points/);
  assert.throws(() => massProperties([[[0, 0, 0], [1, 0, 0], [0, NaN, 0]]]), /non-finite/);
});

test('volumeIdentityResidual: passes a consistent set and CATCHES a plausible-but-wrong one', () => {
  // Two boxes overlapping at a corner: every volume is known in closed form.
  const A = 10 ** 3, B = 10 ** 3, I = 5 ** 3;
  const U = A + B - I, D = A - I;
  const good = volumeIdentityResidual({ a: A, b: B, union: U, intersect: I, difference: D });
  assert.equal(good.ok, true, `a consistent set must pass, residual ${good.worst}`);
  assert.ok(good.worst < 1e-12);

  // A union that kept one fragment too many. It is still a closed solid with
  // an ordinary Euler characteristic — nothing about chi or naked edges knows
  // anything is wrong, which is exactly why this oracle exists.
  const bad = volumeIdentityResidual({ a: A, b: B, union: U + 125, intersect: I, difference: D });
  assert.equal(bad.ok, false, 'an over-counted union must be caught');
  assert.ok(bad.unionResidual > 0.1, `the residual should be large and obvious, got ${bad.unionResidual}`);

  // The difference identity is independent: a union can be right while the
  // difference is wrong, and only checking one would miss it.
  const badDiff = volumeIdentityResidual({ a: A, b: B, union: U, intersect: I, difference: D + 125 });
  assert.equal(badDiff.ok, false);
  assert.ok(badDiff.unionResidual < 1e-12, 'the union half is still clean here');
  assert.ok(badDiff.differenceResidual > 0.1, 'the difference half is what catches it');
});
