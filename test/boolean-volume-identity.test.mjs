import test from 'node:test';
import assert from 'node:assert';
import { booleanSolids } from '../kernel/boolean.mjs';
import { massPropertiesOfBrep, volumeIdentityResidual } from '../kernel/massprops.mjs';

// THE BOOLEAN VALIDITY ORACLE, RUN FOR REAL.
//
//     volume(A u B) + volume(A n B) == volume(A) + volume(B)
//     volume(A - B) + volume(A n B) == volume(A)
//
// The point is what these catch that nothing else here does. A boolean that
// keeps a fragment it should have dropped, or drops one it should have kept,
// still sews into a closed shell with an ordinary Euler characteristic and
// zero naked edges — chi passes it, the naked-edge count passes it, and it
// looks entirely plausible in a viewport. Its volume does not add up. That is
// the whole reason to measure rather than to look.
//
// WHY THIS IS A FUZZ RUN AND NOT ONE FIXTURE. The already-shipped boolean
// test pins ONE hand-derived box pair, and a single fixture is exactly where
// a fragment-keeping bug hides: it either happens to be right there or the
// test was written around it. Sweeping a family of corner overlaps at varying
// extents and offsets makes the keep-rule answer differently case by case.
//
// THE FAMILY IS BOX-LIKE ON PURPOSE, for now: box-like solids are
// where fuzzing starts, because every volume here also has a CLOSED FORM, so
// each case is checked twice — against the analytic answer AND against the
// identity. The analytic half proves the identity is not passing vacuously;
// the identity half is what survives when the corpus grows to wrapped-face
// and revolve-x-revolve cases that have no closed form at all.
//
// EVERY CASE IS DETERMINISTIC. The generator is a seeded integer hash, never
// Math.random — the same convention kernel/noise.mjs already holds to, so a
// failure names a case number that reproduces exactly.

function hash01(seed, i) {
  let h = (seed * 374761393 + i * 668265263) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function quadSurface(p00, p10, p11, p01) {
  const w = (p) => [p[0], p[1], p[2], 1];
  return { degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1], ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]] };
}

// Face order is fixed and referenced by index from the curve table below:
// 0: z=lo  1: z=hi  2: y=lo  3: y=hi  4: x=lo  5: x=hi
function boxFaces([x0, y0, z0], [x1, y1, z1]) {
  return [
    { srf: quadSurface([x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]) },
    { srf: quadSurface([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]) },
    { srf: quadSurface([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]) },
    { srf: quadSurface([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]) },
    { srf: quadSurface([x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]) },
    { srf: quadSurface([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]) },
  ];
}

function boxTriangles([x0, y0, z0], [x1, y1, z1]) {
  const v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  const q = (a, b, c, d) => [[v[a], v[b], v[c]], [v[a], v[c], v[d]]];
  return [...q(0, 3, 2, 1), ...q(4, 5, 6, 7), ...q(0, 1, 5, 4),
    ...q(3, 7, 6, 2), ...q(0, 4, 7, 3), ...q(1, 2, 6, 5)];
}

// The marcher would hand over many samples and the projection warm-starts
// from one to the next, so a two-point chain would be an unrealistically
// easy input.
function seg(from, to, n = 9) {
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    out.push([from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, from[2] + (to[2] - from[2]) * t]);
  }
  return out;
}

// The six shared edges of a corner overlap: each is where one MAX face of A
// meets one MIN face of B, running the full width of the overlap box in the
// third axis. Derived from the two boxes' own extents, not read back out of
// anything under test.
function cornerCurves(a1, b0) {
  const O = { lo: b0, hi: a1 };
  return [
    { samples: seg([a1[0], b0[1], O.lo[2]], [a1[0], b0[1], O.hi[2]]), faceA: 5, faceB: 2 },
    { samples: seg([a1[0], O.lo[1], b0[2]], [a1[0], O.hi[1], b0[2]]), faceA: 5, faceB: 0 },
    { samples: seg([b0[0], a1[1], O.lo[2]], [b0[0], a1[1], O.hi[2]]), faceA: 3, faceB: 4 },
    { samples: seg([O.lo[0], a1[1], b0[2]], [O.hi[0], a1[1], b0[2]]), faceA: 3, faceB: 0 },
    { samples: seg([b0[0], O.lo[1], a1[2]], [b0[0], O.hi[1], a1[2]]), faceA: 1, faceB: 4 },
    { samples: seg([O.lo[0], b0[1], a1[2]], [O.hi[0], b0[1], a1[2]]), faceA: 1, faceB: 2 },
  ];
}

// A corner overlap with NO coplanar faces anywhere: b0 lands strictly inside
// A on every axis, and B extends strictly past a1 on every axis. Coincident
// faces are a named research-grade problem and deliberately not
// what this sweeps.
function makeCase(seed, i) {
  const r = (k) => hash01(seed, i * 16 + k);
  const a0 = [-20 + 40 * r(0), -20 + 40 * r(1), -20 + 40 * r(2)];
  const ea = [4 + 10 * r(3), 4 + 10 * r(4), 4 + 10 * r(5)];
  const a1 = a0.map((v, k) => v + ea[k]);
  const b0 = a0.map((v, k) => v + ea[k] * (0.25 + 0.5 * r(6 + k)));
  const b1 = a1.map((v, k) => v + ea[k] * (0.2 + 0.8 * r(9 + k)));
  return { a0, a1, b0, b1 };
}

const boxVol = (lo, hi) => (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);

test('VOLUME IDENTITY: a swept family of corner overlaps satisfies both identities AND their own closed forms', () => {
  const CASES = 24;
  let worstIdentity = 0, worstAnalytic = 0, worstClosure = 0;
  const insideOut = { union: 0, intersect: 0, difference: 0 };

  for (let i = 0; i < CASES; i++) {
    const { a0, a1, b0, b1 } = makeCase(20260803, i);
    const A = { faces: boxFaces(a0, a1), triangles: boxTriangles(a0, a1) };
    const B = { faces: boxFaces(b0, b1), triangles: boxTriangles(b0, b1) };
    const curves = cornerCurves(a1, b0);

    const VA = boxVol(a0, a1), VB = boxVol(b0, b1), VI = boxVol(b0, a1);
    const truth = { union: VA + VB - VI, intersect: VI, difference: VA - VI };

    const measured = {};
    for (const op of ['union', 'intersect', 'difference']) {
      const res = booleanSolids(A, B, curves, op);
      assert.equal(res.ok, true, `case ${i} ${op} did not close: ${res.reason ?? res.verdict}`);
      assert.equal(res.stats.nakedEdgeCount, 0, `case ${i} ${op} has naked edges`);
      assert.equal(res.stats.shellCount, 1, `case ${i} ${op} is not one shell`);

      const m = massPropertiesOfBrep(res.solid);
      // The closed-surface theorem, measured rather than assumed: a genuinely
      // closed, consistently wound shell sums its face area vectors to zero.
      assert.ok(m.closureResidual < 1e-12,
        `case ${i} ${op} fails the closed-surface theorem, residual ${m.closureResidual}`);
      worstClosure = Math.max(worstClosure, m.closureResidual);
      if (m.volume < 0) insideOut[op]++;

      // Magnitudes throughout: orientation is a SEPARATE property with its own
      // separate report below, and folding the two together would let one mask
      // the other.
      const rel = Math.abs(Math.abs(m.volume) - truth[op]) / truth[op];
      assert.ok(rel < 1e-9,
        `case ${i} ${op}: measured |V| ${Math.abs(m.volume)} != closed form ${truth[op]} (rel ${rel})`);
      worstAnalytic = Math.max(worstAnalytic, rel);
      measured[op] = m.volume;
    }

    const id = volumeIdentityResidual({
      a: VA, b: VB, union: measured.union, intersect: measured.intersect, difference: measured.difference,
    });
    assert.equal(id.ok, true, `case ${i} fails the volume identity: ${JSON.stringify(id)}`);
    worstIdentity = Math.max(worstIdentity, id.worst);
  }

  assert.ok(worstIdentity < 1e-12, `worst identity residual across ${CASES} cases: ${worstIdentity}`);
  assert.ok(worstAnalytic < 1e-12, `worst analytic residual: ${worstAnalytic}`);
  assert.ok(worstClosure < 1e-12, `worst closure residual: ${worstClosure}`);
});

test('VOLUME IDENTITY: every boolean result comes out OUTWARD-oriented, not merely consistent', () => {
  // `orientLoops` guarantees consistency; outwardness is a separate fact and
  // is enforced separately. This pins it, because a negative-volume result is
  // a real shell that every downstream consumer — a renderer, an exporter, a
  // Thicken — reads inside-out while chi, the naked-edge count and even the
  // volume MAGNITUDE all still look perfect.
  for (let i = 0; i < 8; i++) {
    const { a0, a1, b0, b1 } = makeCase(777, i);
    const A = { faces: boxFaces(a0, a1), triangles: boxTriangles(a0, a1) };
    const B = { faces: boxFaces(b0, b1), triangles: boxTriangles(b0, b1) };
    const curves = cornerCurves(a1, b0);
    for (const op of ['union', 'intersect', 'difference']) {
      const res = booleanSolids(A, B, curves, op);
      assert.equal(res.ok, true, `case ${i} ${op}: ${res.reason ?? res.verdict}`);
      const m = massPropertiesOfBrep(res.solid);
      assert.ok(m.volume > 0,
        `case ${i} ${op} came out inside-out (signed volume ${m.volume})`);
    }
  }
});

test('VOLUME IDENTITY: the oracle genuinely FAILS a boolean given a deliberately wrong keep set', () => {
  // The negative control this whole file rests on. Nothing above proves the
  // oracle can fail until something wrong is handed to it — and the wrong
  // thing has to be wrong in the way a real bug is wrong: a closed, ordinary,
  // entirely plausible solid whose volume simply is not the answer.
  const { a0, a1, b0, b1 } = makeCase(20260803, 0);
  const VA = boxVol(a0, a1), VB = boxVol(b0, b1), VI = boxVol(b0, a1);

  const A = { faces: boxFaces(a0, a1), triangles: boxTriangles(a0, a1) };
  const B = { faces: boxFaces(b0, b1), triangles: boxTriangles(b0, b1) };
  const curves = cornerCurves(a1, b0);

  const u = booleanSolids(A, B, curves, 'union');
  const n = booleanSolids(A, B, curves, 'intersect');
  const d = booleanSolids(A, B, curves, 'difference');
  assert.ok(u.ok && n.ok && d.ok);

  const U = massPropertiesOfBrep(u.solid).volume;
  const I = massPropertiesOfBrep(n.solid).volume;
  const D = massPropertiesOfBrep(d.solid).volume;
  assert.equal(volumeIdentityResidual({ a: VA, b: VB, union: U, intersect: I, difference: D }).ok, true);

  // Substitute the DIFFERENCE result where the union belongs — a real solid,
  // genuinely closed, genuinely produced by this same machine, simply the
  // wrong one. chi and the naked-edge count are identical either way.
  const swapped = volumeIdentityResidual({ a: VA, b: VB, union: D, intersect: I, difference: D });
  assert.equal(swapped.ok, false, 'the oracle must catch a plausible, closed, WRONG union');
  assert.ok(swapped.unionResidual > 0.1,
    `and catch it obviously, not marginally — residual ${swapped.unionResidual}`);
});
