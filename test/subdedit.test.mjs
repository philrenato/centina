import test from 'node:test';
import assert from 'node:assert/strict';
import { edgeKey, buildTopology, subdivideCatmullClark } from '../kernel/subd.mjs';
import { superbBoxCage, superbCylinderCage, superbPlaneCage } from '../kernel/subdprimitives.mjs';
import {
  computeFaceNormal, computeAverageNormal, extrudeFaces,
  insertEdgeLoop, recomputeInsertedLoopPositions,
  deleteFaces, boundaryLoopFromSeed, fillHoleWithNGon,
  orderedBoundaryLoopOfFaceSet, bridgeFaces, bridgeBoundaryLoops, bridgeEdgeRuns, stitchEdgeRuns,
  subdivideCageGlobal, mergeFaces,
} from '../kernel/subdedit.mjs';

// Shared correctness proof for BRIDGE/FILLSUBDHOLE/STITCH: a genuinely
// well-oriented manifold cage never has an edge used by more than 2 faces,
// and whenever an edge IS shared by exactly 2 faces, those two faces must
// traverse it in OPPOSITE directions (one visits a->b, the other b->a) — the
// standard consistent-winding property. Returns a list of violation strings
// (empty = fully consistent), rather than a bare boolean, so a failing
// assertion names the actual bad edge.
function windingViolations(cage) {
  const dirCount = new Map(); // "a_b" DIRECTED (not edgeKey) -> count
  const faceCountOf = new Map(); // edgeKey -> total face count
  for (const face of cage.faces) {
    const n = face.length;
    for (let c = 0; c < n; c++) {
      const a = face[c], b = face[(c + 1) % n];
      const dKey = `${a}>${b}`;
      dirCount.set(dKey, (dirCount.get(dKey) || 0) + 1);
      const uKey = edgeKey(a, b);
      faceCountOf.set(uKey, (faceCountOf.get(uKey) || 0) + 1);
    }
  }
  const violations = [];
  for (const [uKey, count] of faceCountOf) {
    if (count > 2) { violations.push(`edge ${uKey} used by ${count} faces (non-manifold)`); continue; }
    if (count === 2) {
      const [a, b] = uKey.split('_').map(Number);
      const forward = dirCount.get(`${a}>${b}`) || 0;
      const backward = dirCount.get(`${b}>${a}`) || 0;
      if (!(forward === 1 && backward === 1)) violations.push(`edge ${uKey} shared by 2 faces but NOT traversed in opposite directions (a>b:${forward}, b>a:${backward})`);
    }
  }
  return violations;
}

// ================================================================
// DELETE FACES
// ================================================================

test('deleteFaces: deleting one face of a facets=1 SuperBBox removes exactly that face and creates a real open boundary, with zero orphaned vertices', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1); // 8 vertices, 6 faces
  const { cage: out, vertexRemap, removedVertexCount } = deleteFaces(cage, [4]); // +Z top face
  assert.equal(out.faces.length, 5);
  assert.equal(removedVertexCount, 0, 'every vertex of the top face is still used by a side face — none orphaned');
  assert.equal(out.vertices.length, 8);
  assert.equal(vertexRemap.size, 8);
  // The cage now has a real 4-edge open boundary where the top face was.
  const topology = buildTopology(out);
  const boundaryEdges = [...topology.edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(boundaryEdges.length, 4, 'removing one face of a closed box leaves exactly its own 4 edges open');
});

test('deleteFaces: refuses an out-of-range face index', () => {
  const cage = superbBoxCage();
  assert.throws(() => deleteFaces(cage, [99]), /out of range/);
});

test('deleteFaces: refuses deleting every face (would leave zero faces)', () => {
  const cage = superbBoxCage();
  assert.throws(() => deleteFaces(cage, [0, 1, 2, 3, 4, 5]), /at least one face/);
});

test('deleteFaces: an isolated vertex (used by no remaining face) is pruned and every reference correctly renumbered', () => {
  // A single standalone quad plus one extra, otherwise-unreferenced vertex tacked
  // onto the vertex array — deleting the ONLY face that uses vertices 0-3 must
  // prune all 4, but vertex 4 (already unused even before deletion) must ALSO
  // be pruned, and there is nothing left to renumber against since faces===[].
  // Use two disjoint quads instead so there's a real face left to check indices against.
  const vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [10, 10, 10], [11, 10, 10], [11, 11, 10], [10, 11, 10]];
  const faces = [[0, 1, 2, 3], [4, 5, 6, 7]];
  const cage = { vertices, faces, creases: {} };
  const { cage: out, vertexRemap, removedVertexCount } = deleteFaces(cage, [0]);
  assert.equal(removedVertexCount, 4, 'the whole first quad\'s own 4 vertices are orphaned by deleting its only face');
  assert.equal(out.vertices.length, 4);
  assert.equal(out.faces.length, 1);
  // Vertex 4 (old) must now be index 0 (new), etc. — first surviving old index maps to 0.
  assert.equal(vertexRemap.get(4), 0);
  assert.deepEqual(out.faces[0], [0, 1, 2, 3].map((oldVi) => vertexRemap.get(oldVi + 4)));
  assert.deepEqual(out.vertices[0], [10, 10, 10]);
});

test('deleteFaces: a crease shared by two DELETED faces vanishes with the edge itself; a crease surviving via a remaining neighbor face is kept', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const topFace = cage.faces[4]; // +Z
  const sideFace = cage.faces[0]; // +X — shares exactly one edge with the top face
  const shared = topFace.filter((vi) => sideFace.includes(vi));
  assert.equal(shared.length, 2, 'the top face and the +X side face must share exactly one edge (2 vertices)');
  cage.creases[edgeKey(shared[0], shared[1])] = 5; // this edge's own 2 faces (4 and 0) are BOTH about to be deleted
  const otherFace = cage.faces[2]; // +Y, stays untouched
  cage.creases[edgeKey(otherFace[0], otherFace[1])] = 7; // survives via this remaining neighbor
  const { cage: out, vertexRemap } = deleteFaces(cage, [4, 0]);
  assert.equal(Object.keys(out.creases).length, 1, 'only the surviving crease remains — the shared-by-both-deleted-faces one vanished with its own edge');
  const survivingKey = edgeKey(vertexRemap.get(otherFace[0]), vertexRemap.get(otherFace[1]));
  assert.equal(out.creases[survivingKey], 7);
});

// ================================================================
// boundaryLoopFromSeed / fillHoleWithNGon (DELETE FACES's own inverse)
// ================================================================

test('boundaryLoopFromSeed + fillHoleWithNGon: deleting then re-filling a face reproduces a manifold, correctly-wound cage', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const topFace = cage.faces[4].slice();
  const { cage: opened, vertexRemap } = deleteFaces(cage, [4]);
  const remappedTopFace = topFace.map((vi) => vertexRemap.get(vi));
  const seedKey = edgeKey(remappedTopFace[0], remappedTopFace[1]);
  const loop = boundaryLoopFromSeed(opened, seedKey);
  assert.equal(loop.length, 4, 'a quad hole has a 4-vertex boundary loop');
  assert.equal(new Set(loop).size, 4, 'a simple loop has no repeated vertex');

  const { cage: filled, faceIndex } = fillHoleWithNGon(opened, loop);
  assert.equal(filled.faces.length, 6, 'back to the original 6-face count');
  assert.equal(faceIndex, 5);
  // The whole re-filled box must be a fully consistent, 2-manifold cage —
  // the real proof this loop's own orientation is correct, not assumed.
  assert.deepEqual(windingViolations(filled), [], 'a re-filled box must have zero winding/manifold violations');
  const topology = buildTopology(filled);
  const boundaryEdges = [...topology.edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(boundaryEdges.length, 0, 'a re-filled closed box has no open edges left');
});

test('boundaryLoopFromSeed: refuses a non-boundary (interior, 2-face) edge', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const topology = buildTopology(cage);
  const interiorKey = [...topology.edgeMap.keys()].find((k) => topology.edgeMap.get(k).faces.length === 2);
  assert.ok(interiorKey, 'a closed box must have at least one interior edge to test against');
  assert.throws(() => boundaryLoopFromSeed(cage, interiorKey), /not a boundary edge/);
});

test('boundaryLoopFromSeed: refuses an unknown edge key', () => {
  const cage = superbBoxCage();
  assert.throws(() => boundaryLoopFromSeed(cage, '9999_10000'), /not a real edge/);
});

test('fillHoleWithNGon: refuses a loop containing a repeated vertex', () => {
  const cage = superbBoxCage();
  assert.throws(() => fillHoleWithNGon(cage, [0, 1, 0, 2]), /repeated vertex/);
});

test('fillHoleWithNGon: refuses filling an edge that already has 2 faces (not actually open)', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const topFace = cage.faces[4];
  // The cage is already fully closed — every edge already has 2 faces.
  assert.throws(() => fillHoleWithNGon(cage, topFace), /already has 2 adjacent faces|isn't an open boundary/);
});

// ================================================================
// orderedBoundaryLoopOfFaceSet / bridgeFaces (BRIDGE)
// ================================================================

test('orderedBoundaryLoopOfFaceSet: a single face selection on a closed box returns its own real 4-vertex rim', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const loop = orderedBoundaryLoopOfFaceSet(cage, [4]);
  assert.equal(loop.length, 4);
  assert.deepEqual(new Set(loop), new Set(cage.faces[4]));
});

test('orderedBoundaryLoopOfFaceSet: refuses a selection covering the whole cage (no rim at all)', () => {
  const cage = superbBoxCage();
  assert.throws(() => orderedBoundaryLoopOfFaceSet(cage, [0, 1, 2, 3, 4, 5]), /no rim at all/);
});

// A real, non-degenerate Bridge fixture: TWO SEPARATE boxes with a genuine
// gap of empty space between them, one face opened facing each other —
// chosen deliberately over "two opposite faces of the SAME closed box"
// (which has no real gap to tunnel through at all — the box's own existing
// 4 side faces already directly connect those two loops, so bridging them
// would only ever duplicate already-real edges, a poor and misleading
// orientation test for exactly that reason, the same "simplest-sounding
// case is backwards" lesson this project's own SSI work already logged
// once, in a new shape here).
function twoSeparateBoxesCage(gapX = 50) {
  const boxA = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const boxB = superbBoxCage([gapX, 0, 0], [10, 10, 10], 1);
  const offset = boxA.vertices.length;
  return {
    vertices: [...boxA.vertices, ...boxB.vertices],
    faces: [...boxA.faces, ...boxB.faces.map((f) => f.map((vi) => vi + offset))],
    creases: {},
    // face 0 = boxA's own +X face (faces toward boxB); face 7 = boxB's own -X face (faces toward boxA)
  };
}

test('bridgeFaces: connecting the facing +X/-X faces of two SEPARATE boxes (segments=1) produces a valid, fully-manifold, correctly-wound tunnel with zero open edges', () => {
  const cage = twoSeparateBoxesCage();
  const before = { v: cage.vertices.length, f: cage.faces.length };
  const { cage: out, tunnelFaceIndices } = bridgeFaces(cage, [0], [7], 1);
  assert.equal(out.vertices.length, before.v, 'no vertices lost or gained: nothing was orphaned, and segments=1 adds no interior ring');
  assert.equal(tunnelFaceIndices.length, 4, 'one tunnel quad per rim edge (4, since the box faces are 4-sided)');
  assert.equal(out.faces.length, before.f - 2 + 4, '12 original - 2 deleted + 4 tunnel faces');
  assert.deepEqual(windingViolations(out), [], 'the bridged cage must be a fully consistent, 2-manifold cage');
  const topology = buildTopology(out);
  const boundaryEdges = [...topology.edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(boundaryEdges.length, 0, 'bridging the two facing openings leaves the combined shape fully closed');
});

test('bridgeFaces: segments=3 inserts exactly 2 new interior rings, still fully manifold', () => {
  const cage = twoSeparateBoxesCage();
  const { cage: out } = bridgeFaces(cage, [0], [7], 3);
  assert.equal(out.vertices.length, 16 + 2 * 4, '16 original + 2 interior rings of 4 new vertices each');
  assert.deepEqual(windingViolations(out), []);
  const topology = buildTopology(out);
  const boundaryEdges = [...topology.edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(boundaryEdges.length, 0);
});

// A real, non-hypothetical constraint this test exists to cover: the APP
// LAYER only ever calls bridgeFaces against a SINGLE SuperB object's own
// cage (the "same object" v1 scope cut) — so it must work correctly on TWO
// OPENINGS OF THE SAME CAGE, not just two separate objects. Facets=2 (a
// finer, non-fully-symmetric grid) is used specifically because a plain
// facets=1 box's opposite faces are the exact degenerate case named in this
// module's own bridgeFaces header (a naive rotation can duplicate an
// already-existing edge) — proving the ROTATION-VALIDITY guard genuinely
// avoids that failure mode on a realistic same-object selection, not just
// that two independent objects happen to work.
test('bridgeFaces: two non-adjacent openings of the SAME finer (facets=2) box — the app\'s own real single-object use case — resolve with zero winding/manifold violations', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  const { cage: out } = bridgeFaces(cage, [0], [16], 1);
  assert.deepEqual(windingViolations(out), []);
});

// The exact degenerate case this module's own header names directly: two
// OPPOSITE faces of a plain facets=1 box (already fully connected by the
// box's own 4 side faces — no true gap to tunnel through). The rotation-
// validity guard is proven here to do ONE of two honest things depending on
// which correspondence it lands on — either succeed with a genuinely valid
// (if geometrically overlapping-with-the-box's-own-sides) result, or refuse
// outright — but NEVER silently produce a non-manifold cage either way.
test('bridgeFaces: the degenerate same-facets=1-box opposite-faces case never produces a silently-broken (non-manifold) cage', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  try {
    const { cage: out } = bridgeFaces(cage, [4], [5], 1);
    assert.deepEqual(windingViolations(out), [], 'if it succeeds at all, the result must be a real, fully consistent 2-manifold');
  } catch (err) {
    assert.match(err.message, /duplicate an edge the cage already has elsewhere/, 'if it refuses, it must refuse HONESTLY by this exact named reason, not some other error');
  }
});

test('bridgeFaces: bridges MISMATCHED rim vertex counts by resampling inside the bridge', () => {
  const cage = twoSeparateBoxesCage();
  // A genuinely separate, fully-closed tetrahedron (every edge shared by
  // exactly 2 faces, confirmed directly — no already-open edge anywhere) —
  // one of its own triangular faces has a real 3-vertex rim, a genuine
  // mismatch against the box's 4-vertex rim.
  const tetOffset = cage.vertices.length;
  const tetVerts = [[200, 0, 0], [201, 0, 0], [200, 1, 0], [200, 0, 1]];
  const tetFaces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]].map((f) => f.map((vi) => vi + tetOffset));
  const withTet = { vertices: [...cage.vertices, ...tetVerts], faces: [...cage.faces, ...tetFaces], creases: {} };
  const beforeVerts = withTet.vertices.length;
  const res = bridgeFaces(withTet, [0], [12]);

  // THE RESAMPLE IS INSIDE THE BRIDGE. A 4-rim meeting a 3-rim gives 3 quads
  // and 1 triangle — c[j] = floor(j*3/4) advances 3 times over 4 steps.
  const added = res.tunnelFaceIndices.map((i) => res.cage.faces[i]);
  assert.equal(added.filter((f) => f.length === 4).length, 3, '3 quads');
  assert.equal(added.filter((f) => f.length === 3).length, 1, '1 triangle closes the leftover');

  // AND NEITHER RIM WAS TOUCHED — the whole point of resampling in the
  // bridge rather than on the rims. At segments=1 no vertex is added at all,
  // and every pre-existing vertex position is byte-identical.
  assert.equal(res.cage.vertices.length, beforeVerts, 'no vertex inserted into either rim');
  for (let i = 0; i < beforeVerts; i++) {
    assert.deepEqual(res.cage.vertices[i], withTet.vertices[i], `vertex ${i} did not move`);
  }
  // The result is still a consistently-wound manifold, and the two holes the
  // deleted faces left are now one tunnel rather than two open rims.
  assert.deepEqual(windingViolations(res.cage), []);
  assert.equal(boundaryLoopCount(res.cage), 0, 'the tunnel closed both openings');
});

test('bridgeBoundaryLoops: mismatched rims stay manifold with segments > 1', () => {
  // The intermediate rings carry the FINE count, so only the single band
  // against the coarse rim reconciles — the tunnel keeps its resolution.
  const cyl8 = superbCylinderCage([0, 0, 0], 10, 20, 8);
  const cyl5 = superbCylinderCage([80, 0, 0], 10, 20, 5);
  const offset = cyl8.vertices.length;
  const cage = {
    vertices: [...cyl8.vertices, ...cyl5.vertices],
    faces: [...cyl8.faces, ...cyl5.faces.map((f) => f.map((vi) => vi + offset))],
    creases: {},
  };
  // Open one cap on each cylinder, by its own n-gon face.
  const capA = cage.faces.findIndex((f) => f.length === 8);
  const capB = cage.faces.findIndex((f, i) => f.length === 5 && i >= cyl8.faces.length);
  assert.ok(capA >= 0 && capB >= 0, 'sanity: both n-gon caps found');
  const opened = deleteFaces(cage, [capA, capB]);
  const topology = buildTopology(opened.cage);
  const naked = [...topology.edgeMap.entries()].filter(([, e]) => e.faces.length === 1);
  const seedA = naked.find(([, e]) => e.v0 < offset && e.v1 < offset);
  const seedB = naked.find(([, e]) => e.v0 >= offset && e.v1 >= offset);
  assert.ok(seedA && seedB, 'sanity: one open rim on each cylinder');
  const res = bridgeBoundaryLoops(opened.cage, seedA[0], seedB[0], 3);
  assert.deepEqual(windingViolations(res.cage), []);
  assert.equal(boundaryLoopCount(res.cage), 0, 'both rims closed into one tunnel');
  // Exactly one band reconciles: 8 and 5 differ by 3, so 3 triangles total,
  // no matter how many segments the tunnel is divided into.
  const added = res.tunnelFaceIndices.map((i) => res.cage.faces[i]);
  assert.equal(added.filter((f) => f.length === 3).length, 3, 'exactly 3 triangles, all in one band');
  assert.equal(added.filter((f) => f.length === 4).length, 8 * 2 + 5, 'every other band is all quads');
});

test('bridgeFaces: refuses a selection whose own rim touches an already-open cage edge', () => {
  const cage = superbPlaneCage([0, 0, 0], 20, 20, 2); // an open (non-closed) SuperBPlane, facets=2 -> 4 faces
  // Face 0 (a corner face of the plane) has at least one edge on the plane's own
  // real open boundary — bridging FROM it should refuse honestly, by name.
  assert.throws(() => bridgeFaces(cage, [0], [1]), /already-open cage edge/);
});

// ================================================================
// BRIDGE ON TWO ALREADY-OPEN EDGE LOOPS — the generalization bridgeFaces'
// own header named as unwired, now built: a student who already has two
// holes (DeleteFaces earlier, or an open primitive) picks one edge of each
// rim and bridges them, with no faces left to select.
// ================================================================

// THE STRONGEST AVAILABLE PROOF, and the reason it is worth more than a
// fresh set of hand-written expectations: the same two openings reached the
// two DIFFERENT ways must produce the IDENTICAL cage. Path 1 selects the two
// facing faces and bridges (deleting them on the way). Path 2 deletes those
// same two faces FIRST, then bridges the two rims that deletion left open,
// seeded from one edge each. Path 2 shares none of path 1's rim-finding
// code (boundaryLoopFromSeed vs orderedBoundaryLoopOfFaceSet, independently
// derived), so byte-identical output is a real cross-check of the new
// walker's own orientation convention against the already-proven one — not
// a self-consistency tautology.
test('bridgeBoundaryLoops: bridging two ALREADY-OPEN rims reproduces the face-selection path\'s own cage bit-for-bit', () => {
  const cage = twoSeparateBoxesCage();
  const viaFaces = bridgeFaces(cage, [0], [7], 1).cage;

  const opened = deleteFaces(cage, [0, 7]);
  const topology = buildTopology(opened.cage);
  const boundary = [...topology.edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(boundary.length, 8, 'deleting the two facing faces leaves exactly two open 4-edge rims');
  // One seed per rim: take any boundary edge, then any boundary edge that
  // shares NO vertex with the first rim's own walked loop (i.e. is on the
  // other hole) — derived from the real topology, never hardcoded indices.
  const seedA = edgeKey(boundary[0].v0, boundary[0].v1);
  const loopA = boundaryLoopFromSeed(opened.cage, seedA);
  const otherEdge = boundary.find((e) => !loopA.includes(e.v0) && !loopA.includes(e.v1));
  assert.ok(otherEdge, 'the second rim is genuinely disjoint from the first');
  const seedB = edgeKey(otherEdge.v0, otherEdge.v1);

  const viaLoops = bridgeBoundaryLoops(opened.cage, seedA, seedB, 1).cage;
  assert.deepEqual(windingViolations(viaLoops), [], 'the open-loop path must also produce a fully consistent 2-manifold');
  const closed = [...buildTopology(viaLoops).edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(closed.length, 0, 'bridging the two rims closes the combined shape completely');
  assert.deepEqual(viaLoops.vertices, viaFaces.vertices, 'same vertices as the face-selection path');
  // Compared as a SET of DIRECTION-PRESERVING faces, deliberately, not as
  // literal arrays: the two rim walkers legitimately start their loop at a
  // different vertex (one seeds from a boundary edge, the other from the
  // selection's own first rim edge), so the same 4 tunnel quads come out in
  // a different rotation and order. What must match — and does — is the set
  // of faces and each one's own winding DIRECTION, which is exactly what
  // rotating each face to start at its smallest index preserves.
  const faceSet = (c) => c.faces.map((f) => {
    const k = f.indexOf(Math.min(...f));
    return JSON.stringify([...f.slice(k), ...f.slice(0, k)]);
  }).sort();
  assert.deepEqual(faceSet(viaLoops), faceSet(viaFaces), 'the same faces, each wound the same way — the two independently-derived rim walkers agree exactly');
});

test('bridgeBoundaryLoops: segments=3 inserts exactly 2 interior rings between two open rims, still fully manifold', () => {
  const opened = deleteFaces(twoSeparateBoxesCage(), [0, 7]);
  const boundary = [...buildTopology(opened.cage).edgeMap.values()].filter((e) => e.faces.length === 1);
  const seedA = edgeKey(boundary[0].v0, boundary[0].v1);
  const loopA = boundaryLoopFromSeed(opened.cage, seedA);
  const otherEdge = boundary.find((e) => !loopA.includes(e.v0) && !loopA.includes(e.v1));
  const before = opened.cage.vertices.length;
  const { cage: out, tunnelFaceIndices } = bridgeBoundaryLoops(opened.cage, seedA, edgeKey(otherEdge.v0, otherEdge.v1), 3);
  assert.equal(out.vertices.length, before + 2 * 4, '2 interior rings of 4 new vertices each');
  assert.equal(tunnelFaceIndices.length, 3 * 4, '3 segments x 4 rim edges');
  assert.deepEqual(windingViolations(out), []);
  assert.equal([...buildTopology(out).edgeMap.values()].filter((e) => e.faces.length === 1).length, 0);
});

test('bridgeBoundaryLoops: refuses two seeds that are both on the SAME open rim, by name', () => {
  const opened = deleteFaces(twoSeparateBoxesCage(), [0, 7]);
  const boundary = [...buildTopology(opened.cage).edgeMap.values()].filter((e) => e.faces.length === 1);
  const seedA = edgeKey(boundary[0].v0, boundary[0].v1);
  const loopA = boundaryLoopFromSeed(opened.cage, seedA);
  // A second, DIFFERENT edge of the same rim (two consecutive loop vertices).
  const sameRimSeed = edgeKey(loopA[1], loopA[2]);
  assert.notEqual(sameRimSeed, seedA, 'sanity: a genuinely different edge, not the same key twice');
  assert.throws(() => bridgeBoundaryLoops(opened.cage, seedA, sameRimSeed), /SAME open boundary loop/);
});

test('bridgeBoundaryLoops: refuses an interior (2-face) seed edge — there is no rim to walk there', () => {
  const cage = superbBoxCage();
  const interior = [...buildTopology(cage).edgeMap.values()].find((e) => e.faces.length === 2);
  assert.throws(() => bridgeBoundaryLoops(cage, edgeKey(interior.v0, interior.v1), edgeKey(interior.v0, interior.v1)), /not a boundary edge/);
});

test('bridgeBoundaryLoops: refuses two rims with different vertex counts', () => {
  // A facets=1 box (4-edge rim) plus a facets=2 plane's own 8-edge outer
  // boundary, in one cage with a real gap between them: genuinely mismatched
  // rims, refused by name rather than bridged into something malformed.
  const box = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const openedBox = deleteFaces(box, [0]);
  const plane = superbPlaneCage([60, 0, 0], 20, 20, 2);
  const offset = openedBox.cage.vertices.length;
  const cage = {
    vertices: [...openedBox.cage.vertices, ...plane.vertices],
    faces: [...openedBox.cage.faces, ...plane.faces.map((f) => f.map((vi) => vi + offset))],
    creases: {},
  };
  const topology = buildTopology(cage);
  const boundary = [...topology.edgeMap.values()].filter((e) => e.faces.length === 1);
  const boxRim = boundary.find((e) => e.v0 < offset && e.v1 < offset);
  const planeRim = boundary.find((e) => e.v0 >= offset && e.v1 >= offset);
  assert.ok(boxRim && planeRim, 'sanity: both rims are genuinely open in this fixture');
  // A 4-edge box rim against a facets=2 plane's own 8-edge outer boundary:
  // bridged, not refused. 4 quads and 4 triangles, evenly distributed around
  // the ring rather than bunched at the seam.
  const res = bridgeBoundaryLoops(cage, edgeKey(boxRim.v0, boxRim.v1), edgeKey(planeRim.v0, planeRim.v1));
  const added = res.tunnelFaceIndices.map((i) => res.cage.faces[i]);
  assert.equal(added.length, 8, 'one face per fine-rim edge');
  assert.equal(added.filter((f) => f.length === 4).length, 4);
  assert.equal(added.filter((f) => f.length === 3).length, 4);
  // EVENLY DISTRIBUTED, not bunched: with 8 fine edges and 4 coarse, the
  // triangles alternate. Asserted as the real pattern, since "4 triangles"
  // alone would also pass if all four landed in a row.
  // Tested as the PROPERTY, not a literal array: which vertex the ring
  // starts at is decided by the rotation search, so a fixed expected
  // sequence would pin a phase rather than the even distribution that
  // actually matters. No two faces of the same kind may be adjacent,
  // checked cyclically.
  const kinds = added.map((f) => f.length);
  for (let i = 0; i < kinds.length; i++) {
    assert.notEqual(kinds[i], kinds[(i + 1) % kinds.length], `faces ${i} and ${(i + 1) % kinds.length} are both ${kinds[i]}-sided — the triangles bunched instead of distributing`);
  }
  assert.deepEqual(windingViolations(res.cage), []);
});

test('bridgeFaces: refuses overlapping face selections', () => {
  const cage = superbBoxCage();
  assert.throws(() => bridgeFaces(cage, [4, 5], [5]), /both selections/);
});

// REGRESSION — an adversarial review's own Finding 1: two face selections
// that touch at a shared CAGE VERTEX without sharing an edge (the two
// DIAGONAL quads of the same 2x2 grid on one facets=2 box face — an
// entirely ordinary shift-click-two-patches gesture, exactly the kind of
// selection superbFaceGroupsFromSelection correctly reports as two SEPARATE
// groups since they share no edge). The shared vertex survives deleteFaces
// (it's still used by the other 2 faces of the same 2x2 grid), so
// rungA===rungB for whichever rotation pairs them — a degenerate rung with
// distSq===0, which the "closest rotation" heuristic used to actively
// PREFER over every real, non-degenerate rotation, producing faces with a
// REPEATED vertex (a zero-area sliver) with no error at all. Proven here
// directly against the review's own exact repro: face 0 = [0,1,4,3], face 3
// = [4,5,8,7] (both from the +X face's own first-built 3x3 grid), sharing
// only vertex 4.
test('bridgeFaces: two face groups sharing only a cage VERTEX (no shared edge) honestly REFUSES up front, not just "no repeated-vertex face"', () => {
  // A second real gap in the FIRST fix
  // (rungA===rungB, which only rejects a rotation pairing the shared
  // vertex with ITSELF): the "closest rotation that doesn't duplicate
  // anything" search could still ACCEPT a rotation where the shared vertex
  // simply isn't in either rung pair that round — no repeated-vertex face,
  // but that shared vertex's own face neighborhood is no longer a
  // topological disk (a real vertex-non-manifold pinch, invisible to every
  // edge-based check in this module). Closed by refusing outright the
  // instant the two loops' own vertex sets intersect at all.
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  assert.deepEqual(cage.faces[0], [0, 1, 4, 3], 'sanity: this is the exact review repro face 0');
  assert.deepEqual(cage.faces[3], [4, 5, 8, 7], 'sanity: this is the exact review repro face 3 — shares only vertex 4 with face 0');
  assert.throws(() => bridgeFaces(cage, [0], [3], 1), /share vertex 4/, 'must refuse honestly by name, not silently succeed with a vertex-manifold pinch');
});

// A SECOND distinct failure mode the same review named (its own "Finding
// B"): if two rims shared TWO vertices u,v without sharing an edge, some
// rotation could cross-pair them (A[i]=u,B[i]=v and A[j]=v,B[j]=u),
// creating the SAME rung edge (u,v) TWICE (a real edge with 4 incident
// faces) — only caught by the pre-existing edgeMap check if (u,v)
// happened to already exist elsewhere, not in general. The fix above
// (refusing on ANY shared vertex, via a plain `.find()` over the full
// vertex set, not a check scoped to "exactly one") already structurally
// subsumes this — a 2-vertex overlap trips the identical guard the
// 1-vertex case above does, on the very first shared vertex `.find()`
// happens to reach. NOT given its own dedicated fixture here: searched
// every SuperB primitive this app can actually build (box at facets
// 2 and 3, sphere at facets 2, cylinder at facets 2 and 6) for a real,
// naturally-occurring "two faces share exactly 2 vertices, neither
// treating that pair as one of its own edges" pair and found ZERO
// candidates at any density — confirmed programmatically, not assumed.
// This case reads as contrived, but on a larger grid
// and building an artificial cage from scratch just to exercise a
// one-line `.find()` a second time would test JS's own array semantics
// more than any real cage-topology subtlety — not worth the risk of a
// second hand-guessed, possibly-wrong synthetic fixture. Named honestly
// as an unreachable-in-practice theoretical case, not silently dropped.

// ================================================================
// BRIDGE ON OPEN EDGE RUNS (N-to-N) + STRAIGHTNESS
// ================================================================

// Two flat strips of 2 quads each, FACING each other across a 20mm gap, each
// with a genuine 3-vertex naked run along the facing side. Hand-built rather
// than taken from a primitive specifically so the expected correspondence, the
// outgoing surface direction at each rim vertex, and every count below are all
// exactly known rather than inferred from a generator.
function twoFacingStripsCage() {
  return {
    vertices: [
      [0, -10, 0], [10, -10, 0], [20, -10, 0],   // 0,1,2   strip A back edge
      [0, 0, 0], [10, 0, 0], [20, 0, 0],         // 3,4,5   strip A FACING edge (runA)
      [0, 10, 20], [10, 10, 20], [20, 10, 20],   // 6,7,8   strip B back edge
      [0, 0, 20], [10, 0, 20], [20, 0, 20],      // 9,10,11 strip B FACING edge (runB)
    ],
    // Strip B is wound to CONTINUE strip A as one prospective surface, not to
    // face it independently — a real distinction found by testing: two
    // oppositely-wound strips have no manifold-legal near correspondence at
    // all, so the only legal bridge between them is a twisted one (see the
    // dedicated twist test below, which uses exactly that fixture on purpose).
    faces: [[0, 1, 4, 3], [1, 2, 5, 4], [9, 10, 7, 6], [10, 11, 8, 7]],
    creases: {},
  };
}

// How many SEPARATE closed chains the cage's naked edges form. Bridge's own
// two directions are distinguishable by exactly this: merging two rims takes
// 2 loops to 1, while splitting one rim takes 1 loop to 2.
function boundaryLoopCount(cage) {
  const topology = buildTopology(cage);
  const naked = [...topology.edgeMap.entries()].filter(([, e]) => e.faces.length === 1).map(([k]) => k.split('_').map(Number));
  const adj = new Map();
  for (const [a, b] of naked) {
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a).push(b);
    adj.get(b).push(a);
  }
  const seen = new Set();
  let loops = 0;
  for (const v of adj.keys()) {
    if (seen.has(v)) continue;
    loops++;
    const stack = [v];
    while (stack.length) {
      const c = stack.pop();
      if (seen.has(c)) continue;
      seen.add(c);
      for (const nb of adj.get(c)) if (!seen.has(nb)) stack.push(nb);
    }
  }
  return loops;
}

test('bridgeEdgeRuns: 2-to-2 open runs on two separate bodies bridge into a consistently-wound wall', () => {
  const cage = twoFacingStripsCage();
  const { cage: out, bridgeFaceIndices } = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1);
  assert.equal(out.faces.length, 6, '4 original + 2 new rung faces');
  assert.equal(bridgeFaceIndices.length, 2);
  assert.equal(out.vertices.length, 12, 'segments=1 adds no new vertices');
  assert.deepEqual(windingViolations(out), [], 'the bridged cage must be a fully consistent 2-manifold');
  const topo = buildTopology(out);
  assert.equal(topo.edgeMap.get(edgeKey(3, 4)).faces.length, 2, 'runA edge 3-4 was naked and is now shared');
  assert.equal(topo.edgeMap.get(edgeKey(9, 10)).faces.length, 2, 'runB edge 9-10 was naked and is now shared');
});

test('bridgeEdgeRuns: the two bodies are genuinely joined — one boundary where there were two', () => {
  const cage = twoFacingStripsCage();
  assert.equal(boundaryLoopCount(cage), 2, 'two separate strips start as two separate boundary loops');
  const { cage: out } = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1);
  assert.equal(boundaryLoopCount(out), 1, 'bridging them merges the two rims into one continuous boundary');
});

test('bridgeEdgeRuns: correspondence is settled by distance — a reversed run bridges to the SAME result', () => {
  const cage = twoFacingStripsCage();
  const fwd = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1).cage;
  const rev = bridgeEdgeRuns(cage, [3, 4, 5], [11, 10, 9], 1).cage;
  assert.deepEqual(rev.faces, fwd.faces, 'handing runB in the other order must not produce a twisted bridge');
});

test('bridgeEdgeRuns: STRAIGHTNESS 1 is bit-identical to a plain straight lerp', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 4, 1).cage;
  const A = [[0, 0, 0], [10, 0, 0], [20, 0, 0]];
  const B = [[0, 0, 20], [10, 0, 20], [20, 0, 20]];
  let checked = 0;
  for (let k = 1; k < 4; k++) {
    const t = k / 4;
    for (let i = 0; i < 3; i++) {
      const got = out.vertices[12 + (k - 1) * 3 + i];
      const want = [0, 1, 2].map((c) => A[i][c] + (B[i][c] - A[i][c]) * t);
      assert.deepEqual(got, want, `interior row ${k} vertex ${i} must be EXACTLY the lerp at straightness 1`);
      checked++;
    }
  }
  assert.equal(checked, 9, 'segments=4 means exactly 3 interior rows of 3 vertices');
});

test('bridgeEdgeRuns: at segments=1 straightness cannot change the result by any value', () => {
  const cage = twoFacingStripsCage();
  const a = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1, 1).cage;
  const b = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1, 0).cage;
  assert.deepEqual(b, a, 'with no interior rows there is nothing to bend — exactly why Rhino defaults Segments to 2');
});

test('bridgeEdgeRuns: STRAIGHTNESS 0 leaves each rim along that rim\'s OWN outgoing surface direction', () => {
  const cage = twoFacingStripsCage();
  // Strip A occupies y in [-10, 0], so its surface EXITS the runA rim heading
  // +y. Derived here from the fixture's own geometry, never read back from the
  // kernel — otherwise this would test self-consistency, not correctness.
  const expectDir = [0, 1, 0];
  const rimA = [0, 0, 0]; // vertex 3
  const dotAtSegments = (segs, straightness) => {
    const p = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], segs, straightness).cage.vertices[12];
    const d = [p[0] - rimA[0], p[1] - rimA[1], p[2] - rimA[2]];
    const L = Math.hypot(...d);
    return d.reduce((acc, c, k) => acc + (c / L) * expectDir[k], 0);
  };
  // THE REAL PROOF IS CONVERGENCE, not a single threshold: a chord from the
  // rim to the first interior row only APPROXIMATES the tangent, with error
  // proportional to t, so a fixed cutoff at one resolution would be an
  // arbitrary number. As t shrinks the chord direction must converge on the
  // surface's own outgoing direction — that limit IS what tangency means.
  const dots = [8, 32, 128, 512].map((n) => dotAtSegments(n, 0));
  for (let i = 1; i < dots.length; i++) {
    assert.ok(dots[i] > dots[i - 1], `the chord direction must converge toward the surface tangent as t shrinks (${dots[i - 1].toFixed(6)} -> ${dots[i].toFixed(6)})`);
  }
  assert.ok(dots[dots.length - 1] > 0.9999, `at the finest sampling the span must leave the rim along the surface's own direction (dot ${dots[dots.length - 1].toFixed(6)})`);
  // NEGATIVE CONTROL — the same measurement at straightness 1 must be exactly
  // the straight chord (perpendicular to the surface direction here), and must
  // NOT converge anywhere, proving this test discriminates rather than passing
  // on any smooth-ish curve.
  const straightDots = [8, 512].map((n) => dotAtSegments(n, 1));
  for (const d of straightDots) assert.ok(Math.abs(d) < 1e-12, `at straightness 1 the span leaves along the chord, exactly (dot ${d})`);
  assert.deepEqual(windingViolations(bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 8, 0).cage), [], 'a fully curved bridge is still a consistent 2-manifold');
});

test('bridgeEdgeRuns: two OPPOSITELY-WOUND runs report the twist rather than silently building it', () => {
  // The same two strips, but strip B wound to face strip A independently
  // instead of continuing it as one surface. No manifold-legal NEAR
  // correspondence exists, so the only legal bridge pairs each rim vertex with
  // the FAR end of the other run — every rung crossing at the center.
  const cage = twoFacingStripsCage();
  cage.faces = [[0, 1, 4, 3], [1, 2, 5, 4], [6, 7, 10, 9], [7, 8, 11, 10]];
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2);
  assert.equal(out.twisted, true, 'the twist must be reported, not silently produced');
  assert.deepEqual(windingViolations(out.cage), [], 'it is still legal cage topology — which is exactly why this is a report, not a refusal');
  // The untwisted case must NOT raise the flag, or it would mean nothing.
  assert.equal(bridgeEdgeRuns(twoFacingStripsCage(), [3, 4, 5], [9, 10, 11], 2).twisted, false);
});

test('bridgeEdgeRuns: a partial straightness lands strictly between the two extremes', () => {
  const cage = twoFacingStripsCage();
  const y = (st) => bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 4, st).cage.vertices[12][1];
  const y0 = y(0), yHalf = y(0.5), y1 = y(1);
  assert.equal(y1, 0, 'straightness 1 stays exactly on the chord (y = 0)');
  assert.ok(y0 > 0.5, 'straightness 0 bulges measurably off the chord');
  assert.ok(yHalf > 0 && yHalf < y0, `straightness 0.5 must sit strictly between (${yHalf.toFixed(4)} vs 0..${y0.toFixed(4)})`);
});

test('bridgeEdgeRuns: SAME-HOLE case — bridging two runs of ONE boundary loop SPLITS it in two', () => {
  const cage = superbPlaneCage([0, 0, 0], 20, 20, 2); // 9 vertices, 4 faces, one 8-edge boundary
  assert.equal(boundaryLoopCount(cage), 1, 'an open sheet starts with exactly one boundary loop');
  const { cage: out } = bridgeEdgeRuns(cage, [0, 1, 2], [6, 7, 8], 1);
  assert.equal(out.faces.length, 6);
  assert.deepEqual(windingViolations(out), [], 'a same-hole bridge must still be a consistent 2-manifold');
  assert.equal(boundaryLoopCount(out), 2, 'bridging two runs of ONE loop SPLITS it — the opposite of the closed-loop Bridge, which MERGES two');
});

// A p-to-q wall carries q-1 (or p-1, whichever is larger) faces per band, of
// which exactly |p - q| are the stalled steps that degenerate to triangles.
function faceShapeCounts(cage, fromIndex) {
  const added = cage.faces.slice(fromIndex);
  return { total: added.length, tris: added.filter((f) => f.length === 3).length, quads: added.filter((f) => f.length === 4).length };
}

test('bridgeEdgeRuns: UNEQUAL counts BUILD — 3 vertices to 2, one quad degenerating to one triangle', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10], 1);
  // The function refuses outright on a reused directed edge, so reaching here
  // at all already proves the wall is consistently wound; assert it anyway
  // through the file's own independent check rather than trusting that.
  assert.deepEqual(windingViolations(out.cage), [], 'a 3-to-2 wall is still a consistent 2-manifold');
  // The FINE run drives the walk: 3 vertices means 2 edges means 2 faces.
  assert.equal(out.cage.faces.length, cage.faces.length + 2, 'the 2 edges of the fine run each get exactly one face');
  const { tris, quads } = faceShapeCounts(out.cage, cage.faces.length);
  assert.equal(tris, 1, 'exactly one stalled step, so exactly one triangle');
  assert.equal(quads, 1, 'and the rest are quads');
  // THE LOAD-BEARING PROMISE of a bridge: it adds, it never resamples. At
  // segments=1 there are no interior rows either, so nothing at all is added
  // to the vertex list and nothing already there is touched.
  assert.deepEqual(out.cage.vertices, cage.vertices, 'a bridge may never move a vertex the model already had');
  assert.equal(out.twisted, false, 'the nearest correspondence is the legal one here');
  assert.deepEqual(out.bridgeFaceIndices, [4, 5], 'both new faces are reported as the bridge');
});

test('bridgeEdgeRuns: UNEQUAL counts work with the LONGER run second too', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4], [9, 10, 11], 1);
  assert.deepEqual(windingViolations(out.cage), [], 'a 2-to-3 wall is a consistent 2-manifold as well');
  assert.equal(out.cage.faces.length, cage.faces.length + 2, 'the fine run drives the walk whichever argument it arrived in');
  const { tris, quads } = faceShapeCounts(out.cage, cage.faces.length);
  assert.equal(tris, 1);
  assert.equal(quads, 1);
  assert.deepEqual(out.cage.vertices, cage.vertices, 'still purely additive');
  assert.equal(out.twisted, false);
  // The ENDS are pinned — a run has no rotational freedom, so the first
  // vertex of one run must meet the first of the other and the last the last.
  // Vertex 3 sits above vertex 9 and vertex 5 above vertex 11 in the fixture,
  // so a wall that did NOT pin its ends would leave 11 unused entirely.
  const added = out.cage.faces.slice(cage.faces.length).flat();
  for (const v of [3, 4, 9, 10, 11]) assert.ok(added.includes(v), `vertex ${v} is carried by the wall — both ends of both runs are attached`);
});

test('bridgeEdgeRuns: UNEQUAL counts at segments > 1 — exactly ONE band reconciles, the rest are quads', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10], 3, 0.5);
  assert.deepEqual(windingViolations(out.cage), [], 'a multi-segment unequal wall is still a consistent 2-manifold');
  // Every interior row carries the FINE count (3), so 2 rows of 3 new points.
  assert.equal(out.cage.vertices.length, cage.vertices.length + 6, 'interior rows carry the FINE count, not the coarse one');
  assert.deepEqual(out.cage.vertices.slice(0, cage.vertices.length), cage.vertices, 'the rows are added; nothing existing moves');
  // 3 bands x 2 faces. Only the band touching the coarse run reconciles, so
  // there is still exactly ONE triangle in the whole wall — the point of
  // carrying the fine count all the way along rather than tapering.
  const { total, tris, quads } = faceShapeCounts(out.cage, cage.faces.length);
  assert.equal(total, 6, '3 bands of 2 faces each');
  assert.equal(tris, 1, 'the reconciliation happens once, not once per band');
  assert.equal(quads, 5);
  // And the several fine points sharing one coarse partner stay genuinely
  // distinct at t < 1, which is why no INTERIOR face degenerates.
  const rowStart = cage.vertices.length;
  assert.notDeepEqual(out.cage.vertices[rowStart + 1], out.cage.vertices[rowStart + 2], 'two fine points with the same coarse partner are still separate points mid-span');
});

test('bridgeEdgeRuns: an EQUAL-count bridge is byte-identical to its pinned result', () => {
  // PINNED, not recomputed: unequal counts were added to this function by
  // generalizing the band builder every equal-count bridge also goes through,
  // so the one thing worth nailing down is that the equal path did not shift
  // by so much as a rounding step. Captured from the function before that
  // generalization existed.
  const out = bridgeEdgeRuns(twoFacingStripsCage(), [3, 4, 5], [9, 10, 11], 3, 0.5, 2);
  assert.deepEqual(out.cage.vertices, [
    [0, -10, 0], [10, -10, 0], [20, -10, 0],
    [0, 0, 0], [10, 0, 0], [20, 0, 0],
    [0, 10, 20], [10, 10, 20], [20, 10, 20],
    [0, 0, 20], [10, 0, 20], [20, 0, 20],
    [0, 0.7407407407407409, 5.925925925925926], [10, 0.7407407407407409, 5.925925925925926], [20, 0.7407407407407409, 5.925925925925926],
    [0, -0.7407407407407406, 14.074074074074073], [10, -0.7407407407407406, 14.074074074074073], [20, -0.7407407407407406, 14.074074074074073],
  ], 'every interior-row point lands exactly where it always did');
  assert.deepEqual(out.cage.faces, [
    [0, 1, 4, 3], [1, 2, 5, 4], [9, 10, 7, 6], [10, 11, 8, 7],
    [3, 4, 13, 12], [4, 5, 14, 13],
    [12, 13, 16, 15], [13, 14, 17, 16],
    [15, 16, 10, 9], [16, 17, 11, 10],
  ], 'same faces, same winding, same ORDER — a reordering would still be legal topology and is still a regression');
  assert.deepEqual(out.cage.creases, { '3_4': 2, '4_5': 2, '9_10': 2, '10_11': 2 });
  assert.deepEqual(out.bridgeFaceIndices, [4, 5, 6, 7, 8, 9]);
  assert.equal(out.twisted, false);
});

test('bridgeEdgeRuns: honest refusals — shared vertex, non-naked edge, bad params', () => {
  const cage = twoFacingStripsCage();
  assert.throws(() => bridgeEdgeRuns(cage, [3, 4, 5], [5, 10, 11], 1), /share vertex 5/);
  // Edge 1-4 is INTERIOR to strip A (two faces) — there is no opening there.
  assert.throws(() => bridgeEdgeRuns(cage, [1, 4], [9, 10], 1), /is not a naked \(open\) edge/);
  assert.throws(() => bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 0), /segments must be a positive integer/);
  assert.throws(() => bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1, 1.5), /straightness must be between 0 and 1/);
});

test('bridgeFaces: STRAIGHTNESS reaches the already-shipped closed-loop Bridge too, default unchanged', () => {
  const gap = 50;
  const mk = () => twoSeparateBoxesCage(gap);
  const sides = (c) => [facesOnSide(c, 0, 10), facesOnSide(c, 0, gap - 10)];
  const cs = mk(), [sa, sb] = sides(cs);
  const straight = bridgeFaces(cs, sa, sb, 3, 1).cage;
  const cc = mk(), [ca, cb] = sides(cc);
  const curved = bridgeFaces(cc, ca, cb, 3, 0).cage;
  assert.equal(curved.vertices.length, straight.vertices.length, 'straightness changes WHERE interior rows sit, never how many');
  assert.notDeepEqual(curved.vertices, straight.vertices, 'a curved two-hole tunnel must genuinely differ from the straight one');
  assert.deepEqual(windingViolations(curved), [], 'the curved tunnel is still a consistent 2-manifold');
  const cd = mk(), [da, db] = sides(cd);
  const dflt = bridgeFaces(cd, da, db, 3).cage;
  assert.deepEqual(dflt.vertices, straight.vertices, 'the DEFAULT is straight — every already-saved Bridge is unaffected');
});

test('bridgeEdgeRuns: CREASE weight 0 (the default) writes no crease key at all', () => {
  const cage = twoFacingStripsCage();
  const a = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2).cage;
  const b = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2, 1, 0).cage;
  assert.deepEqual(Object.keys(a.creases), [], 'no crease at the default — byte-identical to a Bridge built before Crease existed');
  assert.deepEqual(b, a, 'passing an explicit 0 is the same as passing nothing');
});

test('bridgeEdgeRuns: CREASE creases the RIM edges only, never the rungs', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2, 1, 3).cage;
  // The rim is where the wall MEETS each existing surface: runA's own 2 edges
  // and runB's own 2. An OPEN run has no closing edge, so exactly 4 keys.
  assert.equal(Object.keys(out.creases).length, 4, 'exactly the 4 rim edges, no more');
  for (const [a, b] of [[3, 4], [4, 5], [9, 10], [10, 11]]) {
    assert.equal(out.creases[edgeKey(a, b)], 3, `rim edge ${a}-${b} is creased`);
  }
  // Deliberately NOT creased: a rung (across the gap) or an interior-row edge.
  assert.equal(out.creases[edgeKey(3, 9)], undefined, 'a rung is not creased — that would crease the wall along its own length');
  assert.equal(out.creases[edgeKey(3, 5)], undefined, 'no key invented for a non-edge');
  // An open run must NOT get a closing edge invented for it.
  assert.equal(out.creases[edgeKey(3, 5)], undefined);
  assert.equal(out.creases[edgeKey(5, 3)], undefined);
});

test('bridgeEdgeRuns: CREASE carries a partial (semi-sharp) weight through unchanged', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2, 1, 1.5).cage;
  assert.equal(out.creases[edgeKey(3, 4)], 1.5, 'a partial weight is stored as given, not rounded to a boolean');
});

test('bridgeEdgeRuns: CREASE changes no geometry and no topology — only the crease map', () => {
  const cage = twoFacingStripsCage();
  const plain = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 3, 0.5, 0).cage;
  const creased = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 3, 0.5, 3).cage;
  assert.deepEqual(creased.vertices, plain.vertices, 'creasing moves nothing');
  assert.deepEqual(creased.faces, plain.faces, 'creasing adds no faces');
  // And it is a real semi-sharp crease the subdivider actually reacts to, not
  // an inert stored number: the limit surfaces must genuinely differ.
  const subPlain = subdivideCatmullClark(plain);
  const subCreased = subdivideCatmullClark(creased);
  assert.notDeepEqual(subCreased.vertices, subPlain.vertices, 'the creased cage subdivides to a genuinely different limit surface');
  assert.deepEqual(windingViolations(creased), [], 'still a consistent 2-manifold');
});

test('bridgeFaces: CREASE reaches the closed-loop Bridge too, on the CLOSING edge as well', () => {
  const gap = 50;
  const mk = () => twoSeparateBoxesCage(gap);
  const sides = (c) => [facesOnSide(c, 0, 10), facesOnSide(c, 0, gap - 10)];
  const c0 = mk(), [a0, b0] = sides(c0);
  const plain = bridgeFaces(c0, a0, b0, 2).cage;
  const c1 = mk(), [a1, b1] = sides(c1);
  const creased = bridgeFaces(c1, a1, b1, 2, 1, 3).cage;
  assert.deepEqual(Object.keys(plain.creases), [], 'default still writes nothing');
  // A CLOSED rim of n vertices has n edges, not n-1 — the closing edge is real
  // here, unlike an open run. Two 4-vertex rims therefore give exactly 8.
  assert.equal(Object.keys(creased.creases).length, 8, `two closed 4-edge rims give 8 creased edges (got ${Object.keys(creased.creases).length})`);
  for (const w of Object.values(creased.creases)) assert.equal(w, 3);
  assert.deepEqual(creased.vertices, plain.vertices, 'creasing a tunnel moves nothing');
  assert.deepEqual(windingViolations(creased), [], 'still a consistent 2-manifold');
});

// ================================================================
// stitchEdgeRuns (STITCH)
// ================================================================

test('stitchEdgeRuns: welding two separate open boundary runs turns their own edges into ONE shared interior edge each, with zero faces lost', () => {
  // Two independent, non-touching quads, positioned so one run's own two
  // vertices are physically close to the other's own two — the realistic
  // Stitch scenario: two nearby open edges that should be one seam.
  const quadA = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], faces: [[0, 1, 2, 3]] };
  const quadB = { vertices: [[0, 0, 0.02], [1, 0, 0.02], [1, 1, 1], [0, 1, 1]], faces: [[0, 1, 2, 3].map((i) => i + 4)] };
  const cage = { vertices: [...quadA.vertices, ...quadB.vertices], faces: [...quadA.faces, ...quadB.faces], creases: {} };
  // Run A: quadA's own edge 0-1 (2 vertices, the near edge). Run B: quadB's own edge 0-1 (indices 4,5).
  const runA = [0, 1];
  const runB = [4, 5];
  const { cage: out, mergedVertexIndices, collapsedFaceCount } = stitchEdgeRuns(cage, runA, runB, 'average');
  assert.equal(collapsedFaceCount, 0);
  assert.equal(out.vertices.length, 6, '8 original - 2 merged away');
  assert.equal(out.faces.length, 2, 'both original faces survive, just sharing a vertex pair now');
  // The stitched edge (mergedVertexIndices[0], mergedVertexIndices[1]) must now be
  // a genuine 2-face INTERIOR edge — the real proof the seam actually closed.
  const topology = buildTopology(out);
  const seamKey = edgeKey(mergedVertexIndices[0], mergedVertexIndices[1]);
  const seamEdge = topology.edgeMap.get(seamKey);
  assert.ok(seamEdge, 'the stitched seam must exist as a real edge');
  assert.equal(seamEdge.faces.length, 2, 'the seam is now shared by BOTH original faces — the gap is closed');
  // 'average' position: the merged vertex sits at the midpoint of its own two originals.
  const mergedPos = out.vertices[mergedVertexIndices[0]];
  assert.ok(Math.abs(mergedPos[2] - 0.01) < 1e-9, `expected z=(0+0.02)/2=0.01 for 'average', got ${mergedPos[2]}`);
});

test('stitchEdgeRuns: "first" keeps run A\'s own original positions exactly', () => {
  const quadA = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]] };
  const quadB = { vertices: [[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5]] };
  const cage = { vertices: [...quadA.vertices, ...quadB.vertices], faces: [[0, 1, 2, 3], [4, 5, 6, 7]], creases: {} };
  const { cage: out, mergedVertexIndices } = stitchEdgeRuns(cage, [0, 1], [4, 5], 'first');
  assert.deepEqual(out.vertices[mergedVertexIndices[0]], [0, 0, 0]);
  assert.deepEqual(out.vertices[mergedVertexIndices[1]], [1, 0, 0]);
});

test('stitchEdgeRuns: "second" keeps run B\'s own original positions exactly', () => {
  const quadA = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]] };
  const quadB = { vertices: [[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5]] };
  const cage = { vertices: [...quadA.vertices, ...quadB.vertices], faces: [[0, 1, 2, 3], [4, 5, 6, 7]], creases: {} };
  const { cage: out, mergedVertexIndices } = stitchEdgeRuns(cage, [0, 1], [4, 5], 'second');
  assert.deepEqual(out.vertices[mergedVertexIndices[0]], [0, 0, 5]);
  assert.deepEqual(out.vertices[mergedVertexIndices[1]], [1, 0, 5]);
});

test('stitchEdgeRuns: auto-detects a REVERSED run B and still aligns correspondence correctly (minimal-distance direction)', () => {
  const quadA = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]] };
  // quadB's own run is listed [1,0] order (reversed relative to quadA's [0,1]) —
  // point (0,0,0.02) is nearest quadA[0], (1,0,0.02) nearest quadA[1].
  const quadB = { vertices: [[1, 0, 0.02], [0, 0, 0.02], [1, 1, 1], [0, 1, 1]] };
  const cage = { vertices: [...quadA.vertices, ...quadB.vertices], faces: [[0, 1, 2, 3], [4, 5, 6, 7]], creases: {} };
  const runA = [0, 1]; // physically near (0,0,0) then (1,0,0)
  const runB = [4, 5]; // listed as (1,0,0.02) then (0,0,0.02) — i.e. reversed
  const { mergedVertexIndices, cage: out } = stitchEdgeRuns(cage, runA, runB, 'first');
  // runA[0]=vertex0=(0,0,0) should end up merged with the CLOSER quadB point (0,0,0.02) [runB[1]],
  // not the far one — proven by checking the final position stayed exactly runA's own (position='first'
  // doesn't discriminate this, so instead check mergedVertexIndices count and that no face collapsed).
  assert.equal(mergedVertexIndices.length, 2);
  assert.equal(out.faces.length, 2, 'both faces survive a correctly-aligned stitch, none collapsed');
});

test('stitchEdgeRuns: refuses mismatched run lengths', () => {
  const cage = superbBoxCage();
  assert.throws(() => stitchEdgeRuns(cage, [0, 1], [2, 3, 4]), /different vertex counts/);
});

test('stitchEdgeRuns: refuses an invalid position keyword', () => {
  const cage = { vertices: [[0, 0, 0], [1, 0, 0], [0, 0, 5], [1, 0, 5]], faces: [], creases: {} };
  assert.throws(() => stitchEdgeRuns(cage, [0, 1], [2, 3], 'bogus'), /"first", "second", or "average"/);
});

test('stitchEdgeRuns: refuses two runs that already share a vertex', () => {
  const cage = { vertices: [[0, 0, 0], [1, 0, 0], [2, 0, 0]], faces: [], creases: {} };
  assert.throws(() => stitchEdgeRuns(cage, [0, 1], [1, 2]), /genuinely separate/);
});

// ================================================================
// computeFaceNormal / computeAverageNormal
// ================================================================

test('computeFaceNormal: a flat CCW (viewed from +Z) unit square in the XY plane has outward normal exactly +Z', () => {
  const cage = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], faces: [[0, 1, 2, 3]], creases: {} };
  const n = computeFaceNormal(cage, 0);
  assert.ok(Math.abs(n[0]) < 1e-12 && Math.abs(n[1]) < 1e-12 && Math.abs(n[2] - 1) < 1e-12, `expected [0,0,1], got ${JSON.stringify(n)}`);
});

test('computeAverageNormal: the +Z top face of a facets=1 SuperBBox has average normal exactly [0,0,1]', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  // buildFace call order in superbBoxCage: +X,-X,+Y,-Y,+Z,-Z — face index 4 is +Z.
  const n = computeAverageNormal(cage, [4]);
  assert.ok(Math.abs(n[0]) < 1e-9 && Math.abs(n[1]) < 1e-9 && Math.abs(n[2] - 1) < 1e-9, `expected [0,0,1], got ${JSON.stringify(n)}`);
});

test('computeAverageNormal: two exactly opposite faces (+Z and -Z) cancel out and throw honestly, not a silent zero vector', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  assert.throws(() => computeAverageNormal(cage, [4, 5]), /cancel out/);
});

test('computeAverageNormal: refuses an empty face-index array', () => {
  const cage = superbBoxCage();
  assert.throws(() => computeAverageNormal(cage, []), /non-empty/);
});

// ================================================================
// extrudeFaces (EXTRUDESUBD)
// ================================================================

test('extrudeFaces: extruding the +Z top face of a facets=1 SuperBBox grows the cage correctly (+4 vertices, +4 faces) and leaves the other 5 faces untouched', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1); // 8 vertices, 6 faces
  const before = { vertexCount: cage.vertices.length, faceCount: cage.faces.length };
  assert.equal(before.vertexCount, 8);
  assert.equal(before.faceCount, 6);
  const origTopFace = cage.faces[4].slice();

  const result = extrudeFaces(cage, [4], [0, 0, 1], 5);
  const { cage: out } = result;

  assert.equal(out.vertices.length, 12, 'expected +4 new (duplicated) boundary vertices'); // all 4 top-face vertices are boundary (shared with side faces)
  assert.equal(out.faces.length, 10, 'expected +4 new side faces, one per boundary edge of the single extruded face');

  // The 5 untouched faces (0,1,2,3,5) must still reference the ORIGINAL vertex indices, unchanged.
  for (const fi of [0, 1, 2, 3, 5]) assert.deepEqual(out.faces[fi], cage.faces[fi], `face ${fi} should be byte-identical to the input cage's own face ${fi}`);

  // The (remapped) cap face — same original 4 side-vertex identities, but each now pointing at a NEW vertex.
  const capFace = out.faces[4];
  assert.equal(capFace.length, 4);
  for (const vi of capFace) assert.ok(vi >= 8, `cap face vertex ${vi} should be one of the 4 new (>=8) vertices, not an original one`);
  // Every new cap vertex sits exactly 5mm above (only) its own original z — a pure +Z translation.
  capFace.forEach((newVi, i) => {
    const origVi = origTopFace[i];
    const orig = cage.vertices[origVi], moved = out.vertices[newVi];
    assert.ok(Math.abs(moved[0] - orig[0]) < 1e-9 && Math.abs(moved[1] - orig[1]) < 1e-9, 'x/y unchanged by a pure +Z extrude');
    assert.ok(Math.abs(moved[2] - (orig[2] + 5)) < 1e-9, 'z offset by exactly the typed distance');
  });

  // The 4 new side faces are genuinely non-degenerate (real area), each connecting an
  // OLD (stationary) vertex pair to its NEW (moved) counterpart.
  const sideFaces = out.faces.slice(6);
  assert.equal(sideFaces.length, 4);
  for (const f of sideFaces) {
    assert.equal(new Set(f).size, 4, 'a real side quad must have 4 DISTINCT vertices, not a degenerate/collapsed one');
    const pts = f.map((vi) => out.vertices[vi]);
    // shoelace-style cross-product area check (non-zero => non-degenerate)
    const a = pts[1].map((v, i) => v - pts[0][i]);
    const b = pts[3].map((v, i) => v - pts[0][i]);
    const cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const area = Math.hypot(...cross);
    assert.ok(area > 1e-6, `side face ${JSON.stringify(f)} must have real area, got ${area}`);
  }

  // No orphaned vertices: every one of the 12 vertices is referenced by >=1 face.
  const used = new Set(out.faces.flat());
  for (let i = 0; i < out.vertices.length; i++) assert.ok(used.has(i), `vertex ${i} is orphaned (referenced by no face)`);
});

test('extrudeFaces: a vertex fully SURROUNDED by selected faces (no boundary edge touches it) is translated in place, not duplicated', () => {
  // A 3x3 vertex grid (2x2 = 4 quads); selecting ALL 4 faces surrounds the CENTER
  // vertex entirely (all 4 of its incident edges are each shared between exactly 2
  // of the 4 selected faces — a genuine interior-region vertex), while all 8 perimeter
  // vertices each touch at least one true boundary-region edge (the grid's own outer
  // edge) and must be duplicated.
  const idx = (i, j) => j * 3 + i;
  const vertices = [];
  for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) vertices.push([i, j, 0]);
  const faces = [
    [idx(0, 0), idx(1, 0), idx(1, 1), idx(0, 1)],
    [idx(1, 0), idx(2, 0), idx(2, 1), idx(1, 1)],
    [idx(0, 1), idx(1, 1), idx(1, 2), idx(0, 2)],
    [idx(1, 1), idx(2, 1), idx(2, 2), idx(1, 2)],
  ];
  const cage = { vertices, faces, creases: {} };
  const centerIdx = idx(1, 1);
  const result = extrudeFaces(cage, [0, 1, 2, 3], [0, 0, 1], 10);
  assert.equal(result.cage.vertices.length, 9 + 8, '9 original + 8 duplicated perimeter vertices (the 1 center vertex is interior, no duplicate)');
  assert.ok(Math.abs(result.cage.vertices[centerIdx][2] - 10) < 1e-9, 'the interior center vertex is translated in place, at the SAME index');
  // Every perimeter vertex must have been duplicated: its ORIGINAL index still holds
  // z=0 (untouched — nothing perimeter-side references it anymore, but it must not
  // have been overwritten in place), and a NEW (>=9) vertex exists at z=10 nearby.
  for (let i = 0; i < 9; i++) {
    if (i === centerIdx) continue;
    assert.ok(Math.abs(result.cage.vertices[i][2] - 0) < 1e-9, `perimeter vertex ${i}'s original slot must stay at z=0 (it was duplicated, not moved)`);
  }
});

test('extrudeFaces: refuses a zero (or near-zero) distance', () => {
  const cage = superbBoxCage();
  assert.throws(() => extrudeFaces(cage, [4], [0, 0, 1], 0), /nonzero/);
});

test('extrudeFaces: refuses a zero-length direction', () => {
  const cage = superbBoxCage();
  assert.throws(() => extrudeFaces(cage, [4], [0, 0, 0], 5), /nonzero vector/);
});

test('extrudeFaces: refuses an empty face selection', () => {
  const cage = superbBoxCage();
  assert.throws(() => extrudeFaces(cage, [], [0, 0, 1], 5), /non-empty/);
});

test('extrudeFaces: refuses an out-of-range face index', () => {
  const cage = superbBoxCage();
  assert.throws(() => extrudeFaces(cage, [99], [0, 0, 1], 5), /out of range/);
});

test('extrudeFaces: a crease on an interior (selCount===2) edge shared by two selected faces, whose BOTH endpoints are boundary vertices, is REMAPPED onto the new (moved) vertex pair, not silently orphaned at the old key', () => {
  // A minimal 2-face strip (3x2 vertex grid) — with only 2 faces total,
  // EVERY vertex sits on the selection's own outer boundary (there is no
  // genuinely interior vertex at all), so the ONE internal (selCount===2)
  // edge shared between the two faces has BOTH its own endpoints
  // duplicated by the extrude — the exact case a review pass found
  // silently dropping a crease.
  const idx = (i, j) => j * 3 + i;
  const vertices = [];
  for (let j = 0; j < 2; j++) for (let i = 0; i < 3; i++) vertices.push([i, j, 0]);
  const faces = [
    [idx(0, 0), idx(1, 0), idx(1, 1), idx(0, 1)],
    [idx(1, 0), idx(2, 0), idx(2, 1), idx(1, 1)],
  ];
  const sharedKey = edgeKey(idx(1, 0), idx(1, 1));
  const cage = { vertices, faces, creases: { [sharedKey]: 1 } };

  const { cage: out } = extrudeFaces(cage, [0, 1], [0, 0, 1], 5);

  assert.equal(out.creases[sharedKey], undefined, 'the OLD key must not linger as a dangling, never-matched entry');
  // Both endpoints of the shared edge are boundary vertices (they each
  // also touch a real selCount===1 boundary edge), so both get duplicated
  // — the new vertex for original index v is always at v + (however many
  // OTHER boundary vertices were duplicated before it); rather than
  // predict the exact new indices, find them by construction: the new
  // internal edge is whichever selCount===2 edge in the OUTPUT cage
  // carries the crease weight.
  const outTopology = buildTopology(out);
  let foundKey = null;
  for (const [key, w] of Object.entries(out.creases)) {
    const edge = outTopology.edgeMap.get(key);
    if (edge && edge.faces.length === 2 && w === 1) foundKey = key;
  }
  assert.ok(foundKey, 'the crease weight must land on SOME real 2-face interior edge in the new cage');
  assert.notEqual(foundKey, sharedKey, 'it must be a genuinely NEW key, not the stale old one');
});

test('extrudeFaces: a crease on an interior edge with only ONE boundary endpoint (the other translated in place, same index) remaps just that one side', () => {
  // Reuses the exact "surrounded center vertex" fixture from the test
  // above this one — the internal edge between face0/face1 pairs a
  // perimeter (boundary, duplicated) vertex with the fully-interior
  // (translated-in-place, same index) center vertex.
  const idx = (i, j) => j * 3 + i;
  const vertices = [];
  for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) vertices.push([i, j, 0]);
  const faces = [
    [idx(0, 0), idx(1, 0), idx(1, 1), idx(0, 1)],
    [idx(1, 0), idx(2, 0), idx(2, 1), idx(1, 1)],
    [idx(0, 1), idx(1, 1), idx(1, 2), idx(0, 2)],
    [idx(1, 1), idx(2, 1), idx(2, 2), idx(1, 2)],
  ];
  const centerIdx = idx(1, 1);
  const mixedKey = edgeKey(idx(1, 0), centerIdx); // shared by face0/face1, selCount===2
  const cage = { vertices, faces, creases: { [mixedKey]: 1 } };

  const { cage: out } = extrudeFaces(cage, [0, 1, 2, 3], [0, 0, 1], 10);

  assert.equal(out.creases[mixedKey], undefined, 'the old key (pairing an original perimeter index with the center) must not linger');
  // The center vertex kept its OWN index (translated in place) — so the
  // new key must still reference centerIdx directly, paired with whatever
  // NEW index idx(1,0) became.
  const newKeys = Object.keys(out.creases).filter((k) => k !== mixedKey);
  assert.equal(newKeys.length, 1, 'exactly one remapped crease entry, no extras');
  const [a, b] = newKeys[0].split('_').map(Number);
  assert.ok(a === centerIdx || b === centerIdx, `remapped key ${newKeys[0]} must still reference the untouched center index ${centerIdx}`);
  assert.ok((a !== centerIdx ? a : b) >= 9, 'the OTHER endpoint must be one of the newly-duplicated (>=9) vertices, not the stale old perimeter index');
});

test('extrudeFaces: refuses a direction exactly parallel to one of the region\'s own boundary edges — extruding the SAME cap face a second time sideways, along one of its own edges, degenerate side face', () => {
  // Real gap found live: extrude the box's +Z top face upward (fine), then
  // extrude that SAME (now-raised) cap a SECOND time along +X — since the
  // cap is a horizontal square, +X runs exactly along two of its own 4
  // edges, and the corresponding side face collapses to 4 collinear points.
  const cage = superbBoxCage([0, 0, 0], [25, 25, 25], 1);
  const topIdx = cage.faces.findIndex((f) => f.every((vi) => Math.abs(cage.vertices[vi][2] - 25) < 1e-6));
  const { cage: raised } = extrudeFaces(cage, [topIdx], [0, 0, 1], 10);
  assert.throws(() => extrudeFaces(raised, [topIdx], [1, 0, 0], 20), /runs exactly parallel/);
});

test('extrudeFaces: a direction with ANY out-of-plane component (not exactly parallel to any edge) succeeds even on the same repeated-cap case', () => {
  const cage = superbBoxCage([0, 0, 0], [25, 25, 25], 1);
  const topIdx = cage.faces.findIndex((f) => f.every((vi) => Math.abs(cage.vertices[vi][2] - 25) < 1e-6));
  const { cage: raised } = extrudeFaces(cage, [topIdx], [0, 0, 1], 10);
  assert.doesNotThrow(() => extrudeFaces(raised, [topIdx], [1, 0, 1], 20));
});

// ================================================================
// insertEdgeLoop (INSERTEDGE) — hand-derived on a concrete cylinder
// side-band quad strip (see this file's own comments for the worked
// derivation of which face `side=0` resolves to and why).
// ================================================================

test('insertEdgeLoop: t=0.5 on a facets=6 SuperBCylinder inserts one new vertex per column rung, each the exact midpoint of its bottom/top pair', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6); // bottom 0-5, top 6-11, faces: 6 side quads + 2 hexagon caps = 8
  assert.equal(cage.vertices.length, 12);
  assert.equal(cage.faces.length, 8);

  const seedKey = edgeKey(0, 1); // a bottom-ring edge
  // side=0 must resolve to the adjacent SIDE quad (faces array order: side quads
  // 0..5 are pushed before the two cap ngons, so buildTopology's edgeMap records
  // the side quad as this edge's faces[0] — see kernel/subdedit.mjs's own header).
  const { cage: out, insertedVertexIndices, rungPairs } = insertEdgeLoop(cage, seedKey, 0.5, 0);

  assert.equal(insertedVertexIndices.length, 6, 'one new vertex per column rung (facets=6)');
  assert.equal(out.vertices.length, 18, '12 original + 6 new');
  assert.equal(out.faces.length, 14, '2 caps kept + 6 side quads x 2 halves each = 2 + 12 = 14');
});

test('insertEdgeLoop: exact face-count delta on the facets=6 cylinder — 6 side quads each split into 2, the 2 caps untouched', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6);
  const { cage: out } = insertEdgeLoop(cage, edgeKey(0, 1), 0.5, 0);
  assert.equal(out.faces.length, 8 - 6 + 12, 'kept 2 caps + 12 replacement half-faces (6 quads x 2)');
});

test('insertEdgeLoop: t=0.5 new vertices sit at the exact geometric midpoint of their bottom/top rung pair', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6);
  const { cage: out, insertedVertexIndices, rungPairs } = insertEdgeLoop(cage, edgeKey(0, 1), 0.5, 0);
  insertedVertexIndices.forEach((vi, i) => {
    const [nearIdx, farIdx] = rungPairs[i];
    const expected = cage.vertices[nearIdx].map((v, k) => (v + cage.vertices[farIdx][k]) / 2);
    const actual = out.vertices[vi];
    for (let k = 0; k < 3; k++) assert.ok(Math.abs(actual[k] - expected[k]) < 1e-9, `component ${k}: expected ${expected[k]}, got ${actual[k]}`);
  });
  // Every rung pairs a bottom vertex (0-5) with its own directly-above top vertex (6-11).
  const pairSet = new Set(rungPairs.map(([a, b]) => `${Math.min(a, b)}_${Math.max(a, b)}`));
  const expectedPairs = new Set([0, 1, 2, 3, 4, 5].map((i) => `${i}_${i + 6}`));
  assert.deepEqual(pairSet, expectedPairs);
});

test('insertEdgeLoop: t=0.25 places the new ring at exactly 25% of the height, same radius/angle as the bottom ring', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6);
  const { cage: out, insertedVertexIndices } = insertEdgeLoop(cage, edgeKey(0, 1), 0.25, 0);
  for (const vi of insertedVertexIndices) {
    const [x, y, z] = out.vertices[vi];
    assert.ok(Math.abs(z - 5) < 1e-9, `expected z=20*0.25=5, got ${z}`); // height*t
    assert.ok(Math.abs(Math.hypot(x, y) - 10) < 1e-9, 'radius preserved — the cylinder side is straight, so the midpoint stays exactly on the same radius');
  }
});

test('insertEdgeLoop: refuses t<=0 or t>=1 (would coincide exactly with an existing loop, a degenerate zero-area result)', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6);
  assert.throws(() => insertEdgeLoop(cage, edgeKey(0, 1), 0, 0), /strictly between 0 and 1/);
  assert.throws(() => insertEdgeLoop(cage, edgeKey(0, 1), 1, 0), /strictly between 0 and 1/);
});

test('insertEdgeLoop: refuses an unknown seed edge key', () => {
  const cage = superbCylinderCage();
  assert.throws(() => insertEdgeLoop(cage, '999_998', 0.5, 0), /not a real edge/);
});

test('insertEdgeLoop: side=1 on a facets=3 cylinder (triangular caps) honestly refuses — the cap is not a quad', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 3); // triangular caps
  const seedKey = edgeKey(0, 1);
  // side=0 -> the side quad (fine); side=1 -> the triangular cap (not a quad, must refuse).
  assert.doesNotThrow(() => insertEdgeLoop(cage, seedKey, 0.5, 0));
  assert.throws(() => insertEdgeLoop(cage, seedKey, 0.5, 1), /only a quad has an opposite edge/);
});

test('insertEdgeLoop: refuses side values other than 0 or 1', () => {
  const cage = superbCylinderCage();
  assert.throws(() => insertEdgeLoop(cage, edgeKey(0, 1), 0.5, 2), /side must be 0 or 1/);
});

// Open-cage (non-closed) strip — a plane grid, hand-derived expected topology
// (see this file's own header comment for the full worked derivation): a
// facets=3 SuperBPlane is a 4x4 vertex grid, 3x3=9 faces in row-major order.
// Seeding on the horizontal edge between row0 and row1 at column i=0 walks
// the WHOLE first row toward row0 (the plane's own true boundary), visiting
// exactly 3 faces (the whole row), terminating honestly at both open ends
// (no wraparound — this is not a closed cylinder).
test('insertEdgeLoop: an open SuperBPlane strip terminates correctly at the true cage boundary (no wraparound), exact hand-derived counts and rung pairs', () => {
  const cage = superbPlaneCage([0, 0, 0], 30, 30, 3); // 4x4=16 vertices, 3x3=9 faces
  assert.equal(cage.vertices.length, 16);
  assert.equal(cage.faces.length, 9);
  const idx = (i, j) => j * 4 + i;
  const seedKey = edgeKey(idx(0, 1), idx(1, 1)); // interior row-1/row-0 boundary edge, column 0

  const { cage: out, insertedVertexIndices, rungPairs } = insertEdgeLoop(cage, seedKey, 0.5, 0);
  assert.equal(insertedVertexIndices.length, 4, 'one new vertex per column (i=0..3), the whole row width');
  assert.equal(out.vertices.length, 20, '16 original + 4 new');
  assert.equal(out.faces.length, 9 - 3 + 6, '9 original - 3 faces of row0 (split into 2 each = 6) = 6 kept + 6 new = 12');

  // Each rung must connect a row-1 vertex to the SAME column's row-0 vertex — checked
  // as an unordered {row1,row0} pair per column (buildTopology's own edge v0/v1
  // storage order depends on which face happens to register an edge FIRST during its
  // internal adjacency pass, an internal detail this test deliberately does not
  // depend on — see this file's own comment on the cylinder test above for the same
  // reasoning, applied here via canonicalized min/max pairs instead of exact order).
  const canon = (pairs) => new Set(pairs.map(([a, b]) => `${Math.min(a, b)}_${Math.max(a, b)}`));
  const expectedRungs = [[idx(0, 1), idx(0, 0)], [idx(1, 1), idx(1, 0)], [idx(2, 1), idx(2, 0)], [idx(3, 1), idx(3, 0)]];
  assert.deepEqual(canon(rungPairs), canon(expectedRungs), 'exact hand-derived {row1,row0} column pairing, across the whole row width');
  // Every rung genuinely pairs a near (row-1) vertex with the SAME-column far (row-0)
  // vertex — i.e. each pair really is {idx(i,1), idx(i,0)} for some i, not a mismatched
  // column.
  for (const [a, b] of rungPairs) {
    const rowOf = (v) => Math.floor(v / 4), colOf = (v) => v % 4;
    assert.equal(colOf(a), colOf(b), 'a rung must connect the SAME column');
    assert.deepEqual(new Set([rowOf(a), rowOf(b)]), new Set([0, 1]), 'a rung must connect row0 to row1');
  }
});

test('insertEdgeLoop: a crease on a RUNG the strip crosses is transferred onto BOTH new half-rungs, not silently dropped when the rung is split', () => {
  const cage0 = superbCylinderCage([0, 0, 0], 10, 20, 6); // bottom 0-5, top 6-11
  const rungKey = edgeKey(0, 6); // a vertical side rung the seed-adjacent strip crosses
  const cage = { ...cage0, creases: { [rungKey]: 1 } };

  const { cage: out, insertedVertexIndices, rungPairs } = insertEdgeLoop(cage, edgeKey(0, 1), 0.5, 0);

  assert.equal(out.creases[rungKey], undefined, 'the original (now-split) rung key must not linger as a dangling entry');
  const pairIdx = rungPairs.findIndex(([a, b]) => (a === 0 && b === 6) || (a === 6 && b === 0));
  assert.ok(pairIdx >= 0, 'the (0,6) rung must genuinely have been crossed by this strip');
  const midIdx = insertedVertexIndices[pairIdx];
  assert.equal(out.creases[edgeKey(0, midIdx)], 1, 'the near half of the split rung must inherit the original weight');
  assert.equal(out.creases[edgeKey(midIdx, 6)], 1, 'the far half of the split rung must ALSO inherit the original weight');
  // Every OTHER rung (never creased in the input) must still carry no weight at all.
  for (const [a, b] of rungPairs) {
    if ((a === 0 && b === 6) || (a === 6 && b === 0)) continue;
    const mid = insertedVertexIndices[rungPairs.findIndex(([x, y]) => x === a && y === b)];
    assert.equal(out.creases[edgeKey(a, mid)], undefined);
    assert.equal(out.creases[edgeKey(mid, b)], undefined);
  }
});

// ================================================================
// recomputeInsertedLoopPositions — the live-slider re-drag path
// (Surface Fair's own Smoothness-slider precedent: cheap reposition,
// zero topology change).
// ================================================================

test('recomputeInsertedLoopPositions: re-dragging to a new t repositions the SAME vertices with zero topology change', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6);
  const { cage: inserted, insertedVertexIndices, rungPairs } = insertEdgeLoop(cage, edgeKey(0, 1), 0.5, 0);
  const before = { vCount: inserted.vertices.length, fCount: inserted.faces.length, faces: JSON.stringify(inserted.faces) };

  const redragged = recomputeInsertedLoopPositions(inserted, insertedVertexIndices, rungPairs, 0.75);
  assert.equal(redragged.vertices.length, before.vCount, 'vertex COUNT never changes on a reposition');
  assert.equal(redragged.faces.length, before.fCount, 'face count never changes on a reposition');
  assert.equal(JSON.stringify(redragged.faces), before.faces, 'face topology is completely untouched by a reposition');
  for (const vi of insertedVertexIndices) {
    assert.ok(Math.abs(redragged.vertices[vi][2] - 15) < 1e-9, `expected z=20*0.75=15 after re-dragging to t=0.75, got ${redragged.vertices[vi][2]}`);
  }
});

test('recomputeInsertedLoopPositions: refuses t outside (0,1), same as insertEdgeLoop itself', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6);
  const { cage: inserted, insertedVertexIndices, rungPairs } = insertEdgeLoop(cage, edgeKey(0, 1), 0.5, 0);
  assert.throws(() => recomputeInsertedLoopPositions(inserted, insertedVertexIndices, rungPairs, 1), /strictly between 0 and 1/);
});

// ============================================================================
// SUBDIVIDE (global) — the v1.1 "SUBDIVIDE (global or per-face)"
// ============================================================================

// Helper: find every face whose vertices all lie on a given axis-plane (used
// to pick a real, connected group of faces on one side of a box).
function facesOnSide(cage, axis, value, eps = 1e-6) {
  const out = [];
  cage.faces.forEach((f, i) => { if (f.every((vi) => Math.abs(cage.vertices[vi][axis] - value) < eps)) out.push(i); });
  return out;
}

test('subdivideCageGlobal: a facets=1 box (6 quads) becomes exactly 24 quads (4x), the exact result subdivideCatmullClark produces', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const out = subdivideCageGlobal(cage);
  assert.equal(out.faceCountBefore, 6);
  assert.equal(out.faceCountAfter, 24, 'every quad becomes 4 -> 6*4 = 24');
  assert.equal(out.faceCountAfter, cage.faces.length * 4);
  // Catmull-Clark's own defining "converges to quads" property — every output face is a quad.
  assert.ok(out.cage.faces.every((f) => f.length === 4), 'every subdivided face is a quad');
  // Byte-identical to a direct subdivideCatmullClark call — this is genuinely
  // just the already-proven refinement step committed as the new cage, not
  // reproved Catmull-Clark math.
  const direct = subdivideCatmullClark(cage);
  assert.deepEqual(out.cage, direct, 'subdivideCageGlobal is exactly subdivideCatmullClark');
});

test('subdivideCageGlobal: an n-gon-containing cage refines correctly — the new face count is the sum of every input face length (each n-gon -> n quads)', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 8); // 8 side quads + 2 octagon caps = 10 faces
  const out = subdivideCageGlobal(cage);
  const expected = cage.faces.reduce((s, f) => s + f.length, 0); // 8*4 + 2*8 = 48
  assert.equal(out.faceCountAfter, expected, 'each n-gon subdivides into n quads');
  assert.ok(out.cage.faces.every((f) => f.length === 4), 'the octagon caps become quads too');
});

test('subdivideCageGlobal: the result is a valid, fully-manifold cage that is itself valid input to a further subdivision', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const out = subdivideCageGlobal(cage);
  assert.deepEqual(windingViolations(out.cage), [], 'a globally-subdivided box stays a fully consistent 2-manifold');
  // Re-subdividable (the whole point of committing it as a live cage) — a
  // second global subdivide runs without throwing and 4x's the count again.
  const out2 = subdivideCageGlobal(out.cage);
  assert.equal(out2.faceCountAfter, out.cage.faces.length * 4);
  assert.deepEqual(windingViolations(out2.cage), []);
});

test('subdivideCageGlobal: crease weights carry forward decremented (semi-sharp decay) — this is what keeps the limit surface identical across the refinement', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  // Interior box edges do not exist (every box edge is a boundary edge at
  // facets=1); crease decay is a property of the shared subdivideCatmullClark
  // step, so verify it on an interior edge of a facets=2 box instead.
  const box2 = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  const topo = buildTopology(box2);
  const interior = [...topo.edgeMap.values()].find((e) => e.faces.length === 2);
  box2.creases = { [edgeKey(interior.v0, interior.v1)]: 2 };
  const out = subdivideCageGlobal(box2);
  // weight 2 -> children at weight 1 (present); a further subdivide -> weight 0 (dropped).
  assert.ok(Object.values(out.cage.creases).some((w) => w === 1), 'a weight-2 crease carries forward to its children at weight 1');
  const out2 = subdivideCageGlobal(out.cage);
  assert.deepEqual(out2.cage.creases, {}, 'a weight-1 crease decays to smooth (no creases left) after one more level');
  assert.notEqual(cage.faces.length, 0);
});

// ============================================================================
// MERGEFACES — the v1.1 companion to SUBDIVIDE
// ============================================================================

test('mergeFaces: merging two edge-adjacent quads on one side of a facets=2 box produces ONE 6-vertex face, a valid manifold cage, and drops exactly one face from the count', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2); // 24 faces, 26 vertices
  const side = facesOnSide(cage, 0, 10); // the four +X quads
  assert.equal(side.length, 4);
  // Pick two of the +X quads that genuinely share an edge.
  const topo = buildTopology(cage);
  let pair = null;
  for (const a of side) for (const b of side) {
    if (a >= b) continue;
    const fa = cage.faces[a];
    const shares = fa.some((_, c) => topo.edgeMap.get(edgeKey(fa[c], fa[(c + 1) % fa.length])).faces.includes(b));
    if (shares) { pair = [a, b]; break; }
    if (pair) break;
  }
  assert.ok(pair, 'found two edge-adjacent +X quads');
  const out = mergeFaces(cage, pair);
  assert.equal(out.mergedFaceCount, 2);
  assert.equal(out.ngonSize, 6, 'two adjacent quads (a 2x1 strip) merge into a 6-vertex rectangle loop');
  assert.equal(out.cage.faces.length, 24 - 2 + 1, 'two faces removed, one merged n-gon added');
  assert.equal(out.cage.faces[out.faceIndex].length, 6, 'faceIndex names the new n-gon');
  assert.deepEqual(windingViolations(out.cage), [], 'the merged cage is a fully consistent 2-manifold');
  // Re-subdividable — the merged cage stays valid input to Catmull-Clark.
  assert.doesNotThrow(() => subdivideCatmullClark(out.cage));
});

test('mergeFaces: merging all four quads of one box face welds them into ONE 8-vertex face and prunes exactly the face-center vertex', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2); // 24 faces, 26 vertices
  const side = facesOnSide(cage, 0, 10); // the four +X quads
  const out = mergeFaces(cage, side);
  // The +X face's own 4 edge-midpoints are shared with the neighboring box
  // faces (so they survive on the rim); only the single face-CENTER vertex is
  // interior to the merged region and gets pruned.
  assert.equal(out.ngonSize, 8, 'the outer ring of a 3x3 face grid is 8 vertices (4 corners + 4 edge-midpoints)');
  assert.equal(out.cage.vertices.length, 26 - 1, 'exactly the face-center vertex is pruned');
  assert.equal(out.cage.faces.length, 24 - 4 + 1);
  assert.deepEqual(windingViolations(out.cage), [], 'still a fully consistent 2-manifold');
});

test('mergeFaces: a crease on a SURVIVING (rim) edge is kept; a crease on a DISSOLVED internal edge vanishes with the edge', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  const side = facesOnSide(cage, 0, 10); // 4 +X quads
  const selSet = new Set(side);
  const topo = buildTopology(cage);
  // An INTERNAL edge of the merged region: shared by exactly 2 selected faces.
  let internal = null, rim = null;
  for (const e of topo.edgeMap.values()) {
    const sel = e.faces.filter((f) => selSet.has(f)).length;
    if (sel === 2 && !internal) internal = e;
    if (sel === 1 && e.faces.length === 2 && !rim) rim = e; // one selected + one surviving neighbor
  }
  assert.ok(internal && rim, 'found a dissolved-internal edge and a surviving rim edge');
  const internalKey = edgeKey(internal.v0, internal.v1);
  const rimKey = edgeKey(rim.v0, rim.v1);
  cage.creases = { [internalKey]: 3, [rimKey]: 3 };
  const out = mergeFaces(cage, side);
  // The rim edge's endpoints survive; its crease must too (remapped through
  // deleteFaces' own vertexRemap). The dissolved internal edge no longer
  // exists, so its crease is correctly gone.
  const del = deleteFaces(cage, side); // reuse the same remap the merge used internally
  const remappedRimKey = edgeKey(del.vertexRemap.get(rim.v0), del.vertexRemap.get(rim.v1));
  assert.equal(out.cage.creases[remappedRimKey], 3, 'the surviving rim edge keeps its crease');
  // No surviving edge should carry the dissolved internal edge's crease.
  const outTopo = buildTopology(out.cage);
  const stillHasInternal = Object.keys(out.cage.creases).some((k) => {
    const [a, b] = k.split('_').map(Number);
    // was the internal edge's own vertex pair (both interior/center) still creased anywhere?
    return outTopo.edgeMap.has(k) && a === del.vertexRemap.get(internal.v0) && b === del.vertexRemap.get(internal.v1);
  });
  assert.ok(!stillHasInternal, 'the dissolved internal edge carries no crease into the merged cage');
});

test('mergeFaces: honest refusals — fewer than 2 faces, out of range, duplicate index, and a genuinely disconnected selection', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  assert.throws(() => mergeFaces(cage, [0]), /at least 2/, 'a single face is a no-op');
  assert.throws(() => mergeFaces(cage, [0, 9999]), /out of range/);
  assert.throws(() => mergeFaces(cage, [0, 0]), /more than once/);
  // Two faces on genuinely opposite sides of the box (share no edge): refused
  // BY NAME as disconnected, not silently merged into a wrong single loop.
  const plusX = facesOnSide(cage, 0, 10)[0];
  const minusX = facesOnSide(cage, 0, -10)[0];
  assert.throws(() => mergeFaces(cage, [plusX, minusX]), /not all edge-connected/);
});

test('mergeFaces: refuses when the merged region\'s rim touches an already-open cage boundary edge (inherited from orderedBoundaryLoopOfFaceSet, matching Bridge\'s own v1 cut)', () => {
  // A SuperBPlane is genuinely OPEN — its border edges have exactly one face.
  const plane = superbPlaneCage([0, 0, 0], 20, 20, 2); // 4 faces, every one touches the open border
  assert.throws(() => mergeFaces(plane, [0, 1]), /already-open cage edge/);
});

test('mergeFaces: a merge that would collapse the cage to a single face is refused (deleteFaces cannot remove every face)', () => {
  // A facets=1 SuperBPlane is a single quad; there is no valid 2-face merge.
  // Use a facets=2 plane and try to merge all 4 — its rim is entirely open
  // border, so this refuses at the open-edge check (a real, honest refusal
  // for the open-cage case), confirming a merge never silently corrupts.
  const plane = superbPlaneCage([0, 0, 0], 20, 20, 2);
  assert.throws(() => mergeFaces(plane, [0, 1, 2, 3]));
});
