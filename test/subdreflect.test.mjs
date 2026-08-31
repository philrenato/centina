import test from 'node:test';
import assert from 'node:assert/strict';
import { edgeKey } from '../kernel/subd.mjs';
import { superbBoxCage } from '../kernel/subdprimitives.mjs';
import {
  reflectPoint, reflectVector, findCageMirrorPartners, mirrorFaceIndex, mirrorEdgeKey,
} from '../kernel/subdreflect.mjs';

// A facets=1 box cage centered at the origin (8 verts, 6 faces, 12 edges) —
// genuinely symmetric about EVERY one of its 3 axis planes, the same real
// primitive the app layer's own SuperBBox command builds.
const box = () => superbBoxCage([0, 0, 0], [25, 25, 25], 1);

test('reflectPoint mirrors a point across an arbitrary plane, exactly', () => {
  // Plane x=0 (origin at [0,0,0], normal +X): a point at [5,3,7] reflects to
  // [-5,3,7] — the textbook axis-plane case, hand-verifiable directly.
  const r = reflectPoint([5, 3, 7], [0, 0, 0], [1, 0, 0]);
  assert.ok(Math.abs(r[0] - -5) < 1e-9 && Math.abs(r[1] - 3) < 1e-9 && Math.abs(r[2] - 7) < 1e-9);
  // An off-origin plane (x=10): [15,0,0] -> [5,0,0] (5 units on each side of x=10).
  const r2 = reflectPoint([15, 0, 0], [10, 0, 0], [1, 0, 0]);
  assert.ok(Math.abs(r2[0] - 5) < 1e-9);
  // A point ALREADY on the plane maps to itself.
  const r3 = reflectPoint([10, 4, -2], [10, 0, 0], [1, 0, 0]);
  assert.ok(Math.abs(r3[0] - 10) < 1e-9 && Math.abs(r3[1] - 4) < 1e-9 && Math.abs(r3[2] - -2) < 1e-9);
  // A non-unit normal is normalized internally, not assumed pre-normalized.
  const r4 = reflectPoint([5, 0, 0], [0, 0, 0], [3, 0, 0]);
  assert.ok(Math.abs(r4[0] - -5) < 1e-9);
});

test('reflectVector mirrors a direction/delta by orientation only, ignoring position', () => {
  // A pure in-plane delta (y-component, plane normal +X) is UNCHANGED.
  const v1 = reflectVector([0, 5, 0], [1, 0, 0]);
  assert.ok(Math.abs(v1[0]) < 1e-9 && Math.abs(v1[1] - 5) < 1e-9 && Math.abs(v1[2]) < 1e-9);
  // A pure out-of-plane delta (x-component) flips sign.
  const v2 = reflectVector([7, 0, 0], [1, 0, 0]);
  assert.ok(Math.abs(v2[0] - -7) < 1e-9);
  // A mixed delta: only the normal-component flips.
  const v3 = reflectVector([3, 4, -2], [1, 0, 0]);
  assert.ok(Math.abs(v3[0] - -3) < 1e-9 && Math.abs(v3[1] - 4) < 1e-9 && Math.abs(v3[2] - -2) < 1e-9);
});

test('findCageMirrorPartners pairs every vertex of a symmetric box cage about its own center plane, none on-plane', () => {
  const cage = box();
  const { partner, onPlaneCount } = findCageMirrorPartners(cage, [0, 0, 0], [1, 0, 0]);
  assert.equal(partner.length, 8);
  assert.equal(onPlaneCount, 0); // a facets=1 box has no vertex sitting exactly at x=0
  // Every vertex's OWN partner, reflected back, lands within tolerance of it —
  // checked directly against the real cage positions, not just trusting the
  // index bookkeeping.
  for (let i = 0; i < cage.vertices.length; i++) {
    const j = partner[i];
    assert.notEqual(j, i, `vertex ${i} should have a DISTINCT partner (no box vertex sits on x=0)`);
    const back = reflectPoint(cage.vertices[i], [0, 0, 0], [1, 0, 0]);
    const d = Math.hypot(back[0] - cage.vertices[j][0], back[1] - cage.vertices[j][1], back[2] - cage.vertices[j][2]);
    assert.ok(d < 1e-6, `vertex ${i}'s partner ${j} is not actually its mirror image (dist ${d})`);
  }
  // A genuine involution: applying partner twice returns the original vertex.
  for (let i = 0; i < cage.vertices.length; i++) assert.equal(partner[partner[i]], i);
});

test('findCageMirrorPartners classifies an on-plane vertex correctly (partner === self)', () => {
  // A plane cage (kernel/subdprimitives.mjs's own superbPlaneCage would work
  // too, but a hand-built 2x1 grid keeps this test self-contained and
  // explicit): a flat XY grid at z=0, 3 columns x 1 row, mirrored about the
  // plane x=1 (its own vertical centerline) — the middle column of vertices
  // (x===1) sits exactly ON that plane.
  const vertices = [
    [0, 0, 0], [1, 0, 0], [2, 0, 0],
    [0, 1, 0], [1, 1, 0], [2, 1, 0],
  ];
  const faces = [[0, 1, 4, 3], [1, 2, 5, 4]];
  const cage = { vertices, faces, creases: {} };
  const { partner, onPlaneCount } = findCageMirrorPartners(cage, [1, 0, 0], [1, 0, 0]);
  assert.equal(onPlaneCount, 2); // vertices 1 and 4 (x===1) both self-pair
  assert.equal(partner[1], 1);
  assert.equal(partner[4], 4);
  assert.equal(partner[0], 2); // x=0 <-> x=2, same row
  assert.equal(partner[3], 5); // x=0 <-> x=2, other row
});

test('findCageMirrorPartners refuses honestly when the cage is not actually symmetric about the given plane', () => {
  const cage = box();
  // Nudge ONE vertex off its own symmetric position — now nothing on the
  // other side reflects back to it within tolerance.
  cage.vertices[0] = [cage.vertices[0][0] + 5, cage.vertices[0][1], cage.vertices[0][2]];
  assert.throws(() => findCageMirrorPartners(cage, [0, 0, 0], [1, 0, 0]), /no mirror partner within tolerance/);
});

test('findCageMirrorPartners refuses honestly on an ambiguous (too-coarse-relative-to-tolerance) pairing', () => {
  // A small, purpose-built fixture (not the box — a box's own minimum
  // vertex separation equals its own self-reflection distance, so no
  // tolerance can ever force an ambiguity on it without ALSO making every
  // vertex trivially self-pair; see this file's own on-plane test above
  // for that distinct case): vertex 0 at x=-5 reflects EXACTLY to [5,0,0],
  // which sits almost exactly between two REAL candidate vertices at
  // [5,0.05,0] and [5,-0.05,0] — both genuinely within a 0.1 tolerance of
  // the reflected point, a real, deliberately-constructed ambiguity, not a
  // contrived crash.
  const cage = { vertices: [[-5, 0, 0], [5, 0.05, 0], [5, -0.05, 0]], faces: [], creases: {} };
  assert.throws(() => findCageMirrorPartners(cage, [0, 0, 0], [1, 0, 0], { tolerance: 0.1 }), /ambiguous candidate mirror partners/);
});

test('findCageMirrorPartners pairs a genuine, DISTINCT close-to-plane mirror pair instead of silently self-pairing both (review-caught regression)', () => {
  // Two real, distinct vertices straddling the plane by only 0.006 units
  // total (0.003 each side) alongside a box whose own vertices are 50
  // units apart — a real reachable case (fine detail near the seam), not
  // a contrived one. The OLD algorithm checked "is my own reflection
  // within tolerance of MYSELF" first and got both of these self-paired
  // (silently wrong: reported as on-plane when neither vertex is actually
  // on the plane, and they simply stopped mirroring each other).
  const vertices = [
    [25, 25, 25], [-25, 25, 25], [25, -25, 25], [-25, -25, 25],
    [25, 25, -25], [-25, 25, -25], [25, -25, -25], [-25, -25, -25],
    [0.003, 10, 10], [-0.003, 10, 10],
  ];
  const cage = { vertices, faces: [], creases: {} };
  const { partner, onPlaneCount } = findCageMirrorPartners(cage, [0, 0, 0], [1, 0, 0]);
  assert.equal(partner[8], 9, 'vertex 8 should pair with its real, exact mirror partner 9, not itself');
  assert.equal(partner[9], 8, 'vertex 9 should pair with its real, exact mirror partner 8, not itself');
  assert.equal(onPlaneCount, 0, 'neither vertex 8 nor 9 is actually on the plane');
  // Reproduced with an OFF-CENTER plane too, per the review's own second
  // repro — not an origin-only artifact.
  const vertices2 = [
    [35, 25, 25], [-15, 25, 25], [35, -25, 25], [-15, -25, 25],
    [35, 25, -25], [-15, 25, -25], [35, -25, -25], [-15, -25, -25],
    [10.002, 10, 10], [9.998, 10, 10],
  ];
  const cage2 = { vertices: vertices2, faces: [], creases: {} };
  const { partner: partner2 } = findCageMirrorPartners(cage2, [10, 0, 0], [1, 0, 0]);
  assert.equal(partner2[8], 9);
  assert.equal(partner2[9], 8);
});

test('mirrorFaceIndex finds the real corresponding face across a symmetric box cage', () => {
  const cage = box();
  const { partner } = findCageMirrorPartners(cage, [0, 0, 0], [1, 0, 0]);
  // The +X face and -X face of the box are each other's mirror image about x=0.
  const plusXIdx = cage.faces.findIndex((f) => f.every((vi) => cage.vertices[vi][0] > 0));
  const minusXIdx = cage.faces.findIndex((f) => f.every((vi) => cage.vertices[vi][0] < 0));
  assert.ok(plusXIdx >= 0 && minusXIdx >= 0);
  assert.equal(mirrorFaceIndex(cage, partner, plusXIdx), minusXIdx);
  assert.equal(mirrorFaceIndex(cage, partner, minusXIdx), plusXIdx);
  // A face lying IN the y/z-ish direction (not purely +X or -X, e.g. the top
  // +Z face) mirrors onto ITSELF (the "straddles the plane"
  // case — the top face's own 4 vertices split 2-and-2 across x=0, and its
  // own mapped vertex set is exactly its own original vertex set).
  const topIdx = cage.faces.findIndex((f) => f.every((vi) => cage.vertices[vi][2] > 0));
  assert.equal(mirrorFaceIndex(cage, partner, topIdx), topIdx);
});

test('mirrorEdgeKey finds the real corresponding edge, and returns null for a foreign/nonexistent key', () => {
  const cage = box();
  const { partner } = findCageMirrorPartners(cage, [0, 0, 0], [1, 0, 0]);
  // A vertical edge of the +X face and its mirror on the -X face.
  const plusXVerts = cage.vertices.map((v, i) => i).filter((i) => cage.vertices[i][0] > 0);
  // Find a real edge between two +X vertices sharing the same y (a vertical Z edge).
  let srcKey = null;
  outer:
  for (const a of plusXVerts) {
    for (const b of plusXVerts) {
      if (a >= b) continue;
      if (Math.abs(cage.vertices[a][1] - cage.vertices[b][1]) < 1e-9 && Math.abs(cage.vertices[a][0] - cage.vertices[b][0]) < 1e-9) {
        srcKey = edgeKey(a, b);
        break outer;
      }
    }
  }
  assert.ok(srcKey, 'found a real +X vertical edge to test with');
  const mirrored = mirrorEdgeKey(cage, partner, srcKey);
  assert.ok(mirrored, 'a real mirrored edge was found');
  const [ma, mb] = mirrored.split('_').map(Number);
  assert.ok(cage.vertices[ma][0] < 0 && cage.vertices[mb][0] < 0, 'the mirrored edge sits on the -X side');
  // A synthesized key between two vertices that are NOT actually connected
  // by any real cage edge returns null, not a fabricated key — found
  // directly from the real topology (the two vertices farthest apart, a
  // box's own space-diagonal corners, are never a real edge).
  let farA = 0, farB = 0, farDist = -1;
  for (let a = 0; a < cage.vertices.length; a++) {
    for (let b = 0; b < cage.vertices.length; b++) {
      if (a === b) continue;
      const d = Math.hypot(...cage.vertices[a].map((c, k) => c - cage.vertices[b][k]));
      if (d > farDist) { farDist = d; farA = a; farB = b; }
    }
  }
  assert.equal(mirrorEdgeKey(cage, partner, edgeKey(farA, farB)), null);
});
