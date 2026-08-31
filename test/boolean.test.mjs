import test from 'node:test';
import assert from 'node:assert';
import { booleanSolids } from '../kernel/boolean.mjs';

// ---------------------------------------------------------------------------
// THE FIXTURE: two axis-aligned boxes overlapping at ONE CORNER, offset so
// that no face of either is coplanar with any face of the other. Every face
// pair that meets, meets transversally — which is exactly the case the
// pipeline scopes to, and deliberately not the aligned box-on-box named as
// a trap (its coincident faces are the research-grade problem, not the
// ordinary one). A is 0..10, B is 5..15, so they share the 5..10 cube.
//
// Every curve below is a hand-derived line segment, not something read back
// out of a marcher — so the expected results are independent of any code
// under test.
// ---------------------------------------------------------------------------

function quadSurface(p00, p10, p11, p01) {
  const w = (p) => [p[0], p[1], p[2], 1];
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]],
  };
}

// Face order is fixed and referenced by index from the curve table below:
// 0: z=lo   1: z=hi   2: y=lo   3: y=hi   4: x=lo   5: x=hi
function boxFaces(lo, hi) {
  const c = (x, y, z) => [x, y, z];
  return [
    { srf: quadSurface(c(lo, lo, lo), c(hi, lo, lo), c(hi, hi, lo), c(lo, hi, lo)) },
    { srf: quadSurface(c(lo, lo, hi), c(hi, lo, hi), c(hi, hi, hi), c(lo, hi, hi)) },
    { srf: quadSurface(c(lo, lo, lo), c(hi, lo, lo), c(hi, lo, hi), c(lo, lo, hi)) },
    { srf: quadSurface(c(lo, hi, lo), c(hi, hi, lo), c(hi, hi, hi), c(lo, hi, hi)) },
    { srf: quadSurface(c(lo, lo, lo), c(lo, hi, lo), c(lo, hi, hi), c(lo, lo, hi)) },
    { srf: quadSurface(c(hi, lo, lo), c(hi, hi, lo), c(hi, hi, hi), c(hi, lo, hi)) },
  ];
}

function boxTriangles(lo, hi) {
  const v = [
    [lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
    [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi],
  ];
  const quad = (a, b, c, d) => [[v[a], v[b], v[c]], [v[a], v[c], v[d]]];
  return [
    ...quad(0, 3, 2, 1), ...quad(4, 5, 6, 7), ...quad(0, 1, 5, 4),
    ...quad(3, 7, 6, 2), ...quad(0, 4, 7, 3), ...quad(1, 2, 6, 5),
  ];
}

const solidA = { faces: boxFaces(0, 10), triangles: boxTriangles(0, 10) };
const solidB = { faces: boxFaces(5, 15), triangles: boxTriangles(5, 15) };

// Densify a straight segment — the marcher would hand over many samples, and
// the projection warm-starts from one to the next, so a two-point chain would
// be an unrealistically easy input.
function seg(from, to, n = 9) {
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    out.push([from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, from[2] + (to[2] - from[2]) * t]);
  }
  return out;
}

// The six shared edges of the overlap region that lie on a face of each solid.
// Each runs from the triple-corner (10,5,5)-style junction out to the edge of
// the A face it sits on.
const CURVES = [
  { samples: seg([10, 5, 5], [10, 5, 10]), faceA: 5, faceB: 2 }, // A x=10  n  B y=5
  { samples: seg([10, 5, 5], [10, 10, 5]), faceA: 5, faceB: 0 }, // A x=10  n  B z=5
  { samples: seg([5, 10, 5], [5, 10, 10]), faceA: 3, faceB: 4 }, // A y=10  n  B x=5
  { samples: seg([5, 10, 5], [10, 10, 5]), faceA: 3, faceB: 0 }, // A y=10  n  B z=5
  { samples: seg([5, 5, 10], [5, 10, 10]), faceA: 1, faceB: 4 }, // A z=10  n  B x=5
  { samples: seg([5, 5, 10], [10, 5, 10]), faceA: 1, faceB: 2 }, // A z=10  n  B y=5
];

function bboxOf(res) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const v of res.solid.vertices) {
    for (let k = 0; k < 3; k++) {
      if (v.point[k] < lo[k]) lo[k] = v.point[k];
      if (v.point[k] > hi[k]) hi[k] = v.point[k];
    }
  }
  return { lo, hi };
}

test('INTERSECT of two corner-overlapping boxes is exactly the shared cube', () => {
  const res = booleanSolids(solidA, solidB, CURVES, 'intersect');
  assert.ok(res.ok, res.reason || res.verdict);

  // The overlap of 0..10 and 5..15 is the 5..10 cube: six faces, and the
  // Euler characteristic of a sphere.
  assert.equal(res.stats.chi, 2, res.verdict);
  assert.equal(res.stats.F, 6, 'one surviving fragment from each of the six cut faces');
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.shellCount, 1);

  // NOT 8 vertices. A cut edge carries its own curve's SAMPLE points into the
  // topology, so the shared cube arrives with a vertex at every sample along
  // every cut rather than only at its true corners. That is correct and is
  // what real kernels do too — the intersection curve's discretization is
  // part of the result — but it means a vertex count here is sample-driven,
  // so the invariant worth pinning is Euler, not a hand-counted V and E.
  assert.ok(res.stats.V > 8, 'cut edges genuinely carry their samples');
  assert.equal(res.stats.V - res.stats.E + res.stats.F, 2, 'Euler-Poincare holds whatever the sampling');

  // Independently derived from the two boxes' own extents, not from the split.
  const { lo, hi } = bboxOf(res);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(lo[k] - 5) < 1e-9, `low bound on axis ${k}: ${lo[k]}`);
    assert.ok(Math.abs(hi[k] - 10) < 1e-9, `high bound on axis ${k}: ${hi[k]}`);
  }
  // Every vertex sits ON the shared cube's own surface — the check that the
  // extra sample vertices are real points of the answer and not stray ones.
  for (const v of res.solid.vertices) {
    const onFace = v.point.some((c) => Math.abs(c - 5) < 1e-9 || Math.abs(c - 10) < 1e-9);
    assert.ok(onFace, `vertex ${v.point} is not on the shared cube`);
  }

  // The three faces of A that B never reaches are dropped whole, without
  // being split — the no-curve path doing its job rather than being skipped.
  const untouchedA = res.faceReports.filter((r) => r.label.startsWith('A') && !r.split);
  assert.equal(untouchedA.length, 3);
  for (const r of untouchedA) assert.equal(r.kept, 0, `${r.label} is outside B and must not survive an intersect`);
});

test('UNION keeps the outside of both and is still one closed solid', () => {
  const res = booleanSolids(solidA, solidB, CURVES, 'union');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.chi, 2, res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.shellCount, 1);
  // Three whole faces from each solid, plus the outside piece of each of the
  // six cut faces.
  assert.equal(res.stats.F, 12);

  // The union spans both boxes end to end.
  const { lo, hi } = bboxOf(res);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(lo[k] - 0) < 1e-9, `low bound on axis ${k}: ${lo[k]}`);
    assert.ok(Math.abs(hi[k] - 15) < 1e-9, `high bound on axis ${k}: ${hi[k]}`);
  }
});

test('DIFFERENCE is A with B\'s corner notched out — and is NOT the union', () => {
  const res = booleanSolids(solidA, solidB, CURVES, 'difference');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.chi, 2, res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0);

  // A's three far faces whole, A's three cut faces' outside pieces, and B's
  // three cut faces' INSIDE pieces lining the notch.
  assert.equal(res.stats.F, 9);

  // The result stays within A — nothing of B's own far side survives, which
  // is the check that the second operand's rule really was reversed.
  const { lo, hi } = bboxOf(res);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(lo[k] - 0) < 1e-9, `low bound on axis ${k}: ${lo[k]}`);
    assert.ok(Math.abs(hi[k] - 10) < 1e-9, `high bound on axis ${k}: ${hi[k]}`);
  }
});

test('the three operators genuinely differ on the same inputs', () => {
  const u = booleanSolids(solidA, solidB, CURVES, 'union');
  const i = booleanSolids(solidA, solidB, CURVES, 'intersect');
  const d = booleanSolids(solidA, solidB, CURVES, 'difference');
  assert.ok(u.ok && i.ok && d.ok);
  const faces = [u.stats.F, i.stats.F, d.stats.F];
  assert.equal(new Set(faces).size, 3, `all three should differ, got ${faces}`);
});

test('a curve tagged onto a face it does not lie on refuses, naming that face', () => {
  const wrong = CURVES.map((c, k) => (k === 0 ? { ...c, faceA: 4 } : c)); // x=lo, nowhere near it
  const res = booleanSolids(solidA, solidB, wrong, 'intersect');
  assert.equal(res.ok, false);
  assert.match(res.reason, /A face 4/);
  assert.match(res.reason, /does not lie on it|not on the surface/i);
});

test('an unknown operator and a missing tessellation both refuse by name', () => {
  const bad = booleanSolids(solidA, solidB, CURVES, 'merge');
  assert.equal(bad.ok, false);
  assert.match(bad.reason, /unknown boolean operation/i);

  const noTris = booleanSolids({ faces: solidA.faces }, solidB, CURVES, 'union');
  assert.equal(noTris.ok, false);
  assert.match(noTris.reason, /tessellation/i);
});
