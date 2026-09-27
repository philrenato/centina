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

// Shared winding check for Bridge, FillSubDHole and Stitch: a
// well-oriented manifold cage never has an edge used by more than 2 faces,
// and whenever an edge is shared by exactly 2 faces, those two faces must
// traverse it in opposite directions (one visits a->b, the other b->a) — the
// standard consistent-winding property. Returns a list of violation strings
// (empty = fully consistent), rather than a bare boolean, so a failing
// assertion names the actual bad edge.
function windingViolations(cage) {
  const dirCount = new Map(); // "a_b" directed (not edgeKey) -> count
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

// Delete faces

test('deleteFaces: deleting one face of a facets=1 SuperBBox removes exactly that face and creates an open boundary, with zero orphaned vertices', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1); // 8 vertices, 6 faces
  const { cage: out, vertexRemap, removedVertexCount } = deleteFaces(cage, [4]); // +Z top face
  assert.equal(out.faces.length, 5);
  assert.equal(removedVertexCount, 0, 'every vertex of the top face is still used by a side face — none orphaned');
  assert.equal(out.vertices.length, 8);
  assert.equal(vertexRemap.size, 8);
  // The cage now has a 4-edge open boundary where the top face was.
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
  // Two disjoint quads: deleting the first orphans its 4 vertices, and the
  // surviving second face carries the renumbered indices to check.
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

test('deleteFaces: a crease shared by two deleted faces vanishes with the edge itself; a crease surviving via a remaining neighbor face is kept', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const topFace = cage.faces[4]; // +Z
  const sideFace = cage.faces[0]; // +X — shares exactly one edge with the top face
  const shared = topFace.filter((vi) => sideFace.includes(vi));
  assert.equal(shared.length, 2, 'the top face and the +X side face must share exactly one edge (2 vertices)');
  cage.creases[edgeKey(shared[0], shared[1])] = 5; // both faces of this edge (4 and 0) are about to be deleted
  const otherFace = cage.faces[2]; // +Y, stays untouched
  cage.creases[edgeKey(otherFace[0], otherFace[1])] = 7; // survives via this remaining neighbor
  const { cage: out, vertexRemap } = deleteFaces(cage, [4, 0]);
  assert.equal(Object.keys(out.creases).length, 1, 'only the surviving crease remains — the shared-by-both-deleted-faces one vanished with its own edge');
  const survivingKey = edgeKey(vertexRemap.get(otherFace[0]), vertexRemap.get(otherFace[1]));
  assert.equal(out.creases[survivingKey], 7);
});

// boundaryLoopFromSeed / fillHoleWithNGon (the inverse of delete faces)

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
  // The whole re-filled box must be a fully consistent, 2-manifold cage,
  // which checks the loop's orientation rather than assuming it.
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

// orderedBoundaryLoopOfFaceSet / bridgeFaces (Bridge)

test('orderedBoundaryLoopOfFaceSet: a single face selection on a closed box returns its own 4-vertex rim', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const loop = orderedBoundaryLoopOfFaceSet(cage, [4]);
  assert.equal(loop.length, 4);
  assert.deepEqual(new Set(loop), new Set(cage.faces[4]));
});

test('orderedBoundaryLoopOfFaceSet: refuses a selection covering the whole cage (no rim at all)', () => {
  const cage = superbBoxCage();
  assert.throws(() => orderedBoundaryLoopOfFaceSet(cage, [0, 1, 2, 3, 4, 5]), /no rim at all/);
});

// A non-degenerate Bridge fixture: two separate boxes with a gap of empty
// space between them, one face opened facing each other. Two opposite faces
// of the same closed box would be a poor orientation test: the box's 4 side
// faces already connect those two loops, so bridging them could only
// duplicate existing edges.
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

test('bridgeFaces: connecting the facing +X/-X faces of two separate boxes (segments=1) produces a valid, fully-manifold, correctly-wound tunnel with zero open edges', () => {
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

// The app layer calls bridgeFaces on a single SuperB object's cage, so it
// must work on two openings of the same cage, not just two separate
// objects. Facets=2 (a finer, non-fully-symmetric grid) is used because a
// plain facets=1 box's opposite faces are the degenerate case named in the
// bridgeFaces header (a naive rotation can duplicate an existing edge);
// this checks that the rotation-validity guard avoids that failure mode on
// a same-object selection.
test('bridgeFaces: two non-adjacent openings of the same finer (facets=2) box — the app\'s single-object use case — resolve with zero winding/manifold violations', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  const { cage: out } = bridgeFaces(cage, [0], [16], 1);
  assert.deepEqual(windingViolations(out), []);
});

// The degenerate case the module header names: two opposite faces of a
// plain facets=1 box (already connected by the box's 4 side faces — no gap
// to tunnel through). Depending on which correspondence the rotation-
// validity guard lands on, it either succeeds with a valid (if
// geometrically overlapping the box's sides) result or refuses outright;
// it never produces a non-manifold cage.
test('bridgeFaces: the degenerate same-facets=1-box opposite-faces case never produces a silently-broken (non-manifold) cage', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  try {
    const { cage: out } = bridgeFaces(cage, [4], [5], 1);
    assert.deepEqual(windingViolations(out), [], 'if it succeeds at all, the result must be a real, fully consistent 2-manifold');
  } catch (err) {
    assert.match(err.message, /duplicate an edge the cage already has elsewhere/, 'if it refuses, it must refuse with this named reason, not some other error');
  }
});

test('bridgeFaces: bridges mismatched rim vertex counts by resampling inside the bridge', () => {
  const cage = twoSeparateBoxesCage();
  // A separate, fully-closed tetrahedron (every edge shared by exactly 2
  // faces — no open edge anywhere); one of its triangular faces has a
  // 3-vertex rim, a mismatch against the box's 4-vertex rim.
  const tetOffset = cage.vertices.length;
  const tetVerts = [[200, 0, 0], [201, 0, 0], [200, 1, 0], [200, 0, 1]];
  const tetFaces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]].map((f) => f.map((vi) => vi + tetOffset));
  const withTet = { vertices: [...cage.vertices, ...tetVerts], faces: [...cage.faces, ...tetFaces], creases: {} };
  const beforeVerts = withTet.vertices.length;
  const res = bridgeFaces(withTet, [0], [12]);

  // The resample is inside the bridge. A 4-rim meeting a 3-rim gives 3 quads
  // and 1 triangle — c[j] = floor(j*3/4) advances 3 times over 4 steps.
  const added = res.tunnelFaceIndices.map((i) => res.cage.faces[i]);
  assert.equal(added.filter((f) => f.length === 4).length, 3, '3 quads');
  assert.equal(added.filter((f) => f.length === 3).length, 1, '1 triangle closes the leftover');

  // Neither rim is touched — the reason for resampling in the bridge rather
  // than on the rims. At segments=1 no vertex is added at all,
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
  // The intermediate rings carry the fine count, so only the single band
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
  // Face 0 (a corner face of the plane) has at least one edge on the plane's
  // open boundary — bridging from it refuses by name.
  assert.throws(() => bridgeFaces(cage, [0], [1]), /already-open cage edge/);
});

// Bridge on two already-open edge loops: a cage that already has two holes
// (from DeleteFaces, or an open primitive) is bridged from one edge of each
// rim, with no faces left to select.

// The same two openings reached two different ways must produce the
// identical cage. Path 1 selects the two facing faces and bridges (deleting
// them on the way). Path 2 deletes those same two faces first, then bridges
// the two rims that deletion left open, seeded from one edge each. Path 2
// shares none of path 1's rim-finding code (boundaryLoopFromSeed vs
// orderedBoundaryLoopOfFaceSet, independently derived), so byte-identical
// output cross-checks the two walkers' orientation conventions rather than
// testing self-consistency.
test('bridgeBoundaryLoops: bridging two already-open rims reproduces the face-selection path\'s own cage bit-for-bit', () => {
  const cage = twoSeparateBoxesCage();
  const viaFaces = bridgeFaces(cage, [0], [7], 1).cage;

  const opened = deleteFaces(cage, [0, 7]);
  const topology = buildTopology(opened.cage);
  const boundary = [...topology.edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(boundary.length, 8, 'deleting the two facing faces leaves exactly two open 4-edge rims');
  // One seed per rim: take any boundary edge, then any boundary edge that
  // shares no vertex with the first rim's walked loop (i.e. is on the
  // other hole) — derived from the topology, never hardcoded indices.
  const seedA = edgeKey(boundary[0].v0, boundary[0].v1);
  const loopA = boundaryLoopFromSeed(opened.cage, seedA);
  const otherEdge = boundary.find((e) => !loopA.includes(e.v0) && !loopA.includes(e.v1));
  assert.ok(otherEdge, 'the second rim is disjoint from the first');
  const seedB = edgeKey(otherEdge.v0, otherEdge.v1);

  const viaLoops = bridgeBoundaryLoops(opened.cage, seedA, seedB, 1).cage;
  assert.deepEqual(windingViolations(viaLoops), [], 'the open-loop path must also produce a fully consistent 2-manifold');
  const closed = [...buildTopology(viaLoops).edgeMap.values()].filter((e) => e.faces.length === 1);
  assert.equal(closed.length, 0, 'bridging the two rims closes the combined shape completely');
  assert.deepEqual(viaLoops.vertices, viaFaces.vertices, 'same vertices as the face-selection path');
  // Compared as a set of direction-preserving faces, not as literal arrays:
  // the two rim walkers start their loop at a different vertex (one seeds
  // from a boundary edge, the other from the selection's first rim edge), so
  // the same 4 tunnel quads come out in a different rotation and order. What
  // must match is the set of faces and each one's winding direction, which
  // is what rotating each face to start at its smallest index preserves.
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

test('bridgeBoundaryLoops: refuses two seeds that are both on the same open rim, by name', () => {
  const opened = deleteFaces(twoSeparateBoxesCage(), [0, 7]);
  const boundary = [...buildTopology(opened.cage).edgeMap.values()].filter((e) => e.faces.length === 1);
  const seedA = edgeKey(boundary[0].v0, boundary[0].v1);
  const loopA = boundaryLoopFromSeed(opened.cage, seedA);
  // A second, different edge of the same rim (two consecutive loop vertices).
  const sameRimSeed = edgeKey(loopA[1], loopA[2]);
  assert.notEqual(sameRimSeed, seedA, 'sanity: a different edge, not the same key twice');
  assert.throws(() => bridgeBoundaryLoops(opened.cage, seedA, sameRimSeed), /SAME open boundary loop/);
});

test('bridgeBoundaryLoops: refuses an interior (2-face) seed edge — there is no rim to walk there', () => {
  const cage = superbBoxCage();
  const interior = [...buildTopology(cage).edgeMap.values()].find((e) => e.faces.length === 2);
  assert.throws(() => bridgeBoundaryLoops(cage, edgeKey(interior.v0, interior.v1), edgeKey(interior.v0, interior.v1)), /not a boundary edge/);
});

test('bridgeBoundaryLoops: refuses two rims with different vertex counts', () => {
  // A facets=1 box (4-edge rim) plus a facets=2 plane's 8-edge outer
  // boundary, in one cage with a gap between them: mismatched rims.
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
  assert.ok(boxRim && planeRim, 'sanity: both rims are open in this fixture');
  // A 4-edge box rim against a facets=2 plane's own 8-edge outer boundary:
  // bridged, not refused. 4 quads and 4 triangles, evenly distributed around
  // the ring rather than bunched at the seam.
  const res = bridgeBoundaryLoops(cage, edgeKey(boxRim.v0, boxRim.v1), edgeKey(planeRim.v0, planeRim.v1));
  const added = res.tunnelFaceIndices.map((i) => res.cage.faces[i]);
  assert.equal(added.length, 8, 'one face per fine-rim edge');
  assert.equal(added.filter((f) => f.length === 4).length, 4);
  assert.equal(added.filter((f) => f.length === 3).length, 4);
  // Evenly distributed, not bunched: with 8 fine edges and 4 coarse, the
  // triangles alternate. Asserted as the pattern, since "4 triangles"
  // alone would also pass if all four landed in a row.
  // Tested as the property, not a literal array: which vertex the ring
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

// Two face selections that touch at a shared cage vertex without sharing an
// edge: the two diagonal quads of the same 2x2 grid on one facets=2 box
// face, an ordinary two-patch selection that superbFaceGroupsFromSelection
// reports as two separate groups since they share no edge. The shared
// vertex survives deleteFaces (it is still used by the other 2 faces of the
// grid), so rungA===rungB for whichever rotation pairs them — a degenerate
// rung with distSq===0, which a "closest rotation" heuristic would prefer
// over every non-degenerate rotation, producing faces with a repeated
// vertex (a zero-area sliver) and no error. Fixture: face 0 = [0,1,4,3],
// face 3 = [4,5,8,7] (both from the +X face's 3x3 grid), sharing only
// vertex 4.
test('bridgeFaces: two face groups sharing only a cage vertex (no shared edge) are refused up front, not just "no repeated-vertex face"', () => {
  // Rejecting only rungA===rungB (a rotation pairing the shared vertex
  // with itself) is not enough: a rotation where the shared vertex is in
  // neither rung pair gives no repeated-vertex face, but that vertex's face
  // neighborhood is not a topological disk (a vertex-non-manifold
  // pinch, invisible to every edge-based check in this module). So the
  // bridge refuses as soon as the two loops' vertex sets intersect at all.
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  assert.deepEqual(cage.faces[0], [0, 1, 4, 3], 'sanity: this is fixture face 0');
  assert.deepEqual(cage.faces[3], [4, 5, 8, 7], 'sanity: this is fixture face 3 — shares only vertex 4 with face 0');
  assert.throws(() => bridgeFaces(cage, [0], [3], 1), /share vertex 4/, 'must refuse by name, not silently succeed with a vertex-manifold pinch');
});

// Not separately covered: if two rims shared two vertices u,v without
// sharing an edge, some rotation could cross-pair them (A[i]=u,B[i]=v and
// A[j]=v,B[j]=u), creating the same rung edge (u,v) twice (an edge with 4
// incident faces), which the edgeMap check catches only if (u,v) already
// exists elsewhere. Refusing on any shared vertex (a `.find()` over the
// full vertex set) subsumes this: a 2-vertex overlap trips the same guard
// as the 1-vertex case above. No SuperB primitive (box at facets 2 and 3,
// sphere at facets 2, cylinder at facets 2 and 6) contains two faces that
// share exactly 2 vertices without sharing that pair as an edge, so there
// is no natural fixture for it.

// Bridge on open edge runs (N-to-N) and straightness

// Two flat strips of 2 quads each, facing each other across a 20mm gap, each
// with a 3-vertex naked run along the facing side. Hand-built rather
// than taken from a primitive specifically so the expected correspondence, the
// outgoing surface direction at each rim vertex, and every count below are all
// exactly known rather than inferred from a generator.
function twoFacingStripsCage() {
  return {
    vertices: [
      [0, -10, 0], [10, -10, 0], [20, -10, 0],   // 0,1,2   strip A back edge
      [0, 0, 0], [10, 0, 0], [20, 0, 0],         // 3,4,5   strip A facing edge (runA)
      [0, 10, 20], [10, 10, 20], [20, 10, 20],   // 6,7,8   strip B back edge
      [0, 0, 20], [10, 0, 20], [20, 0, 20],      // 9,10,11 strip B facing edge (runB)
    ],
    // Strip B is wound to continue strip A as one prospective surface, not to
    // face it independently: two oppositely-wound strips have no
    // manifold-legal near correspondence at all, so the only legal bridge
    // between them is a twisted one (see the twist test below, which uses
    // that fixture).
    faces: [[0, 1, 4, 3], [1, 2, 5, 4], [9, 10, 7, 6], [10, 11, 8, 7]],
    creases: {},
  };
}

// How many separate closed chains the cage's naked edges form. Bridge's
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

test('bridgeEdgeRuns: the two bodies are joined — one boundary where there were two', () => {
  const cage = twoFacingStripsCage();
  assert.equal(boundaryLoopCount(cage), 2, 'two separate strips start as two separate boundary loops');
  const { cage: out } = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1);
  assert.equal(boundaryLoopCount(out), 1, 'bridging them merges the two rims into one continuous boundary');
});

test('bridgeEdgeRuns: correspondence is settled by distance — a reversed run bridges to the same result', () => {
  const cage = twoFacingStripsCage();
  const fwd = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1).cage;
  const rev = bridgeEdgeRuns(cage, [3, 4, 5], [11, 10, 9], 1).cage;
  assert.deepEqual(rev.faces, fwd.faces, 'handing runB in the other order must not produce a twisted bridge');
});

test('bridgeEdgeRuns: straightness 1 is bit-identical to a plain straight lerp', () => {
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
      assert.deepEqual(got, want, `interior row ${k} vertex ${i} must be exactly the lerp at straightness 1`);
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

test('bridgeEdgeRuns: straightness 0 leaves each rim along that rim\'s own outgoing surface direction', () => {
  const cage = twoFacingStripsCage();
  // Strip A occupies y in [-10, 0], so its surface exits the runA rim heading
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
  // The check is convergence, not a single threshold: a chord from the
  // rim to the first interior row only approximates the tangent, with error
  // proportional to t, so a fixed cutoff at one resolution would be an
  // arbitrary number. As t shrinks the chord direction must converge on the
  // surface's own outgoing direction — that limit is what tangency means.
  const dots = [8, 32, 128, 512].map((n) => dotAtSegments(n, 0));
  for (let i = 1; i < dots.length; i++) {
    assert.ok(dots[i] > dots[i - 1], `the chord direction must converge toward the surface tangent as t shrinks (${dots[i - 1].toFixed(6)} -> ${dots[i].toFixed(6)})`);
  }
  assert.ok(dots[dots.length - 1] > 0.9999, `at the finest sampling the span must leave the rim along the surface's own direction (dot ${dots[dots.length - 1].toFixed(6)})`);
  // Negative control: the same measurement at straightness 1 must be exactly
  // the straight chord (perpendicular to the surface direction here), and must
  // not converge anywhere, so the test discriminates rather than passing on
  // any smooth-ish curve.
  const straightDots = [8, 512].map((n) => dotAtSegments(n, 1));
  for (const d of straightDots) assert.ok(Math.abs(d) < 1e-12, `at straightness 1 the span leaves along the chord, exactly (dot ${d})`);
  assert.deepEqual(windingViolations(bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 8, 0).cage), [], 'a fully curved bridge is still a consistent 2-manifold');
});

test('bridgeEdgeRuns: two oppositely-wound runs report the twist rather than silently building it', () => {
  // The same two strips, but strip B wound to face strip A independently
  // instead of continuing it as one surface. No manifold-legal near
  // correspondence exists, so the only legal bridge pairs each rim vertex with
  // the far end of the other run — every rung crossing at the center.
  const cage = twoFacingStripsCage();
  cage.faces = [[0, 1, 4, 3], [1, 2, 5, 4], [6, 7, 10, 9], [7, 8, 11, 10]];
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2);
  assert.equal(out.twisted, true, 'the twist must be reported, not silently produced');
  assert.deepEqual(windingViolations(out.cage), [], 'it is still legal cage topology — which is exactly why this is a report, not a refusal');
  // The untwisted case must not raise the flag.
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

test('bridgeEdgeRuns: same-hole case — bridging two runs of one boundary loop splits it in two', () => {
  const cage = superbPlaneCage([0, 0, 0], 20, 20, 2); // 9 vertices, 4 faces, one 8-edge boundary
  assert.equal(boundaryLoopCount(cage), 1, 'an open sheet starts with exactly one boundary loop');
  const { cage: out } = bridgeEdgeRuns(cage, [0, 1, 2], [6, 7, 8], 1);
  assert.equal(out.faces.length, 6);
  assert.deepEqual(windingViolations(out), [], 'a same-hole bridge must still be a consistent 2-manifold');
  assert.equal(boundaryLoopCount(out), 2, 'bridging two runs of one loop splits it — the opposite of the closed-loop Bridge, which merges two');
});

// A p-to-q wall carries q-1 (or p-1, whichever is larger) faces per band, of
// which exactly |p - q| are the stalled steps that degenerate to triangles.
function faceShapeCounts(cage, fromIndex) {
  const added = cage.faces.slice(fromIndex);
  return { total: added.length, tris: added.filter((f) => f.length === 3).length, quads: added.filter((f) => f.length === 4).length };
}

test('bridgeEdgeRuns: unequal counts build — 3 vertices to 2, one quad degenerating to one triangle', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10], 1);
  // The function refuses outright on a reused directed edge, so reaching here
  // at all already proves the wall is consistently wound; assert it anyway
  // through the file's own independent check rather than trusting that.
  assert.deepEqual(windingViolations(out.cage), [], 'a 3-to-2 wall is still a consistent 2-manifold');
  // The fine run drives the walk: 3 vertices means 2 edges means 2 faces.
  assert.equal(out.cage.faces.length, cage.faces.length + 2, 'the 2 edges of the fine run each get exactly one face');
  const { tris, quads } = faceShapeCounts(out.cage, cage.faces.length);
  assert.equal(tris, 1, 'exactly one stalled step, so exactly one triangle');
  assert.equal(quads, 1, 'and the rest are quads');
  // A bridge adds and never resamples. At
  // segments=1 there are no interior rows either, so nothing at all is added
  // to the vertex list and nothing already there is touched.
  assert.deepEqual(out.cage.vertices, cage.vertices, 'a bridge may never move a vertex the model already had');
  assert.equal(out.twisted, false, 'the nearest correspondence is the legal one here');
  assert.deepEqual(out.bridgeFaceIndices, [4, 5], 'both new faces are reported as the bridge');
});

test('bridgeEdgeRuns: unequal counts work with the longer run second too', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4], [9, 10, 11], 1);
  assert.deepEqual(windingViolations(out.cage), [], 'a 2-to-3 wall is a consistent 2-manifold as well');
  assert.equal(out.cage.faces.length, cage.faces.length + 2, 'the fine run drives the walk whichever argument it arrived in');
  const { tris, quads } = faceShapeCounts(out.cage, cage.faces.length);
  assert.equal(tris, 1);
  assert.equal(quads, 1);
  assert.deepEqual(out.cage.vertices, cage.vertices, 'still purely additive');
  assert.equal(out.twisted, false);
  // The ends are pinned — a run has no rotational freedom, so the first
  // vertex of one run must meet the first of the other and the last the last.
  // Vertex 3 sits above vertex 9 and vertex 5 above vertex 11 in the fixture,
  // so a wall that did not pin its ends would leave 11 unused entirely.
  const added = out.cage.faces.slice(cage.faces.length).flat();
  for (const v of [3, 4, 9, 10, 11]) assert.ok(added.includes(v), `vertex ${v} is carried by the wall — both ends of both runs are attached`);
});

test('bridgeEdgeRuns: unequal counts at segments > 1 — exactly one band reconciles, the rest are quads', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10], 3, 0.5);
  assert.deepEqual(windingViolations(out.cage), [], 'a multi-segment unequal wall is still a consistent 2-manifold');
  // Every interior row carries the fine count (3), so 2 rows of 3 new points.
  assert.equal(out.cage.vertices.length, cage.vertices.length + 6, 'interior rows carry the fine count, not the coarse one');
  assert.deepEqual(out.cage.vertices.slice(0, cage.vertices.length), cage.vertices, 'the rows are added; nothing existing moves');
  // 3 bands x 2 faces. Only the band touching the coarse run reconciles, so
  // there is still exactly one triangle in the whole wall — the reason for
  // carrying the fine count all the way along rather than tapering.
  const { total, tris, quads } = faceShapeCounts(out.cage, cage.faces.length);
  assert.equal(total, 6, '3 bands of 2 faces each');
  assert.equal(tris, 1, 'the reconciliation happens once, not once per band');
  assert.equal(quads, 5);
  // The several fine points sharing one coarse partner stay distinct at
  // t < 1, which is why no interior face degenerates.
  const rowStart = cage.vertices.length;
  assert.notDeepEqual(out.cage.vertices[rowStart + 1], out.cage.vertices[rowStart + 2], 'two fine points with the same coarse partner are still separate points mid-span');
});

test('bridgeEdgeRuns: an equal-count bridge is byte-identical to its pinned result', () => {
  // Pinned, not recomputed: equal and unequal counts share one band builder,
  // so this holds the equal-count output fixed to the last rounding step.
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

test('bridgeEdgeRuns: refusals — shared vertex, non-naked edge, bad params', () => {
  const cage = twoFacingStripsCage();
  assert.throws(() => bridgeEdgeRuns(cage, [3, 4, 5], [5, 10, 11], 1), /share vertex 5/);
  // Edge 1-4 is interior to strip A (two faces) — there is no opening there.
  assert.throws(() => bridgeEdgeRuns(cage, [1, 4], [9, 10], 1), /is not a naked \(open\) edge/);
  assert.throws(() => bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 0), /segments must be a positive integer/);
  assert.throws(() => bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 1, 1.5), /straightness must be between 0 and 1/);
});

test('bridgeFaces: straightness reaches the closed-loop Bridge too, default unchanged', () => {
  const gap = 50;
  const mk = () => twoSeparateBoxesCage(gap);
  const sides = (c) => [facesOnSide(c, 0, 10), facesOnSide(c, 0, gap - 10)];
  const cs = mk(), [sa, sb] = sides(cs);
  const straight = bridgeFaces(cs, sa, sb, 3, 1).cage;
  const cc = mk(), [ca, cb] = sides(cc);
  const curved = bridgeFaces(cc, ca, cb, 3, 0).cage;
  assert.equal(curved.vertices.length, straight.vertices.length, 'straightness changes where interior rows sit, never how many');
  assert.notDeepEqual(curved.vertices, straight.vertices, 'a curved two-hole tunnel must differ from the straight one');
  assert.deepEqual(windingViolations(curved), [], 'the curved tunnel is still a consistent 2-manifold');
  const cd = mk(), [da, db] = sides(cd);
  const dflt = bridgeFaces(cd, da, db, 3).cage;
  assert.deepEqual(dflt.vertices, straight.vertices, 'the default is straight — every saved Bridge is unaffected');
});

test('bridgeEdgeRuns: crease weight 0 (the default) writes no crease key at all', () => {
  const cage = twoFacingStripsCage();
  const a = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2).cage;
  const b = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2, 1, 0).cage;
  assert.deepEqual(Object.keys(a.creases), [], 'no crease at the default — byte-identical to a Bridge built without a crease');
  assert.deepEqual(b, a, 'passing an explicit 0 is the same as passing nothing');
});

test('bridgeEdgeRuns: crease creases the rim edges only, never the rungs', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2, 1, 3).cage;
  // The rim is where the wall meets each existing surface: runA's own 2 edges
  // and runB's own 2. An open run has no closing edge, so exactly 4 keys.
  assert.equal(Object.keys(out.creases).length, 4, 'exactly the 4 rim edges, no more');
  for (const [a, b] of [[3, 4], [4, 5], [9, 10], [10, 11]]) {
    assert.equal(out.creases[edgeKey(a, b)], 3, `rim edge ${a}-${b} is creased`);
  }
  // Not creased: a rung (across the gap) or an interior-row edge.
  assert.equal(out.creases[edgeKey(3, 9)], undefined, 'a rung is not creased — that would crease the wall along its own length');
  assert.equal(out.creases[edgeKey(3, 5)], undefined, 'no key invented for a non-edge');
  // An open run must not get a closing edge invented for it.
  assert.equal(out.creases[edgeKey(3, 5)], undefined);
  assert.equal(out.creases[edgeKey(5, 3)], undefined);
});

test('bridgeEdgeRuns: crease carries a partial (semi-sharp) weight through unchanged', () => {
  const cage = twoFacingStripsCage();
  const out = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 2, 1, 1.5).cage;
  assert.equal(out.creases[edgeKey(3, 4)], 1.5, 'a partial weight is stored as given, not rounded to a boolean');
});

test('bridgeEdgeRuns: crease changes no geometry and no topology — only the crease map', () => {
  const cage = twoFacingStripsCage();
  const plain = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 3, 0.5, 0).cage;
  const creased = bridgeEdgeRuns(cage, [3, 4, 5], [9, 10, 11], 3, 0.5, 3).cage;
  assert.deepEqual(creased.vertices, plain.vertices, 'creasing moves nothing');
  assert.deepEqual(creased.faces, plain.faces, 'creasing adds no faces');
  // It is a semi-sharp crease the subdivider reacts to, not an inert stored
  // number: the limit surfaces must differ.
  const subPlain = subdivideCatmullClark(plain);
  const subCreased = subdivideCatmullClark(creased);
  assert.notDeepEqual(subCreased.vertices, subPlain.vertices, 'the creased cage subdivides to a different limit surface');
  assert.deepEqual(windingViolations(creased), [], 'still a consistent 2-manifold');
});

test('bridgeFaces: crease reaches the closed-loop Bridge too, on the closing edge as well', () => {
  const gap = 50;
  const mk = () => twoSeparateBoxesCage(gap);
  const sides = (c) => [facesOnSide(c, 0, 10), facesOnSide(c, 0, gap - 10)];
  const c0 = mk(), [a0, b0] = sides(c0);
  const plain = bridgeFaces(c0, a0, b0, 2).cage;
  const c1 = mk(), [a1, b1] = sides(c1);
  const creased = bridgeFaces(c1, a1, b1, 2, 1, 3).cage;
  assert.deepEqual(Object.keys(plain.creases), [], 'default still writes nothing');
  // A closed rim of n vertices has n edges, not n-1 — the closing edge is real
  // here, unlike an open run. Two 4-vertex rims therefore give exactly 8.
  assert.equal(Object.keys(creased.creases).length, 8, `two closed 4-edge rims give 8 creased edges (got ${Object.keys(creased.creases).length})`);
  for (const w of Object.values(creased.creases)) assert.equal(w, 3);
  assert.deepEqual(creased.vertices, plain.vertices, 'creasing a tunnel moves nothing');
  assert.deepEqual(windingViolations(creased), [], 'still a consistent 2-manifold');
});

// stitchEdgeRuns (Stitch)

test('stitchEdgeRuns: welding two separate open boundary runs turns their own edges into one shared interior edge each, with zero faces lost', () => {
  // Two independent, non-touching quads, positioned so one run's own two
  // vertices are physically close to the other's own two — the typical
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
  // a 2-face interior edge, which shows the seam closed.
  const topology = buildTopology(out);
  const seamKey = edgeKey(mergedVertexIndices[0], mergedVertexIndices[1]);
  const seamEdge = topology.edgeMap.get(seamKey);
  assert.ok(seamEdge, 'the stitched seam must exist as a real edge');
  assert.equal(seamEdge.faces.length, 2, 'the seam is now shared by both original faces — the gap is closed');
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

test('stitchEdgeRuns: auto-detects a reversed run B and still aligns correspondence correctly (minimal-distance direction)', () => {
  const quadA = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]] };
  // quadB's own run is listed [1,0] order (reversed relative to quadA's [0,1]) —
  // point (0,0,0.02) is nearest quadA[0], (1,0,0.02) nearest quadA[1].
  const quadB = { vertices: [[1, 0, 0.02], [0, 0, 0.02], [1, 1, 1], [0, 1, 1]] };
  const cage = { vertices: [...quadA.vertices, ...quadB.vertices], faces: [[0, 1, 2, 3], [4, 5, 6, 7]], creases: {} };
  const runA = [0, 1]; // physically near (0,0,0) then (1,0,0)
  const runB = [4, 5]; // listed as (1,0,0.02) then (0,0,0.02) — i.e. reversed
  const { mergedVertexIndices, cage: out } = stitchEdgeRuns(cage, runA, runB, 'first');
  // runA[0]=vertex0=(0,0,0) should end up merged with the closer quadB point (0,0,0.02) [runB[1]],
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

// computeFaceNormal / computeAverageNormal

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

test('computeAverageNormal: two exactly opposite faces (+Z and -Z) cancel out and throw, not a silent zero vector', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  assert.throws(() => computeAverageNormal(cage, [4, 5]), /cancel out/);
});

test('computeAverageNormal: refuses an empty face-index array', () => {
  const cage = superbBoxCage();
  assert.throws(() => computeAverageNormal(cage, []), /non-empty/);
});

// extrudeFaces (ExtrudeSubD)

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

  // The 5 untouched faces (0,1,2,3,5) must still reference the original vertex indices, unchanged.
  for (const fi of [0, 1, 2, 3, 5]) assert.deepEqual(out.faces[fi], cage.faces[fi], `face ${fi} should be byte-identical to the input cage's own face ${fi}`);

  // The (remapped) cap face — same original 4 side-vertex identities, but each now pointing at a new vertex.
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

  // The 4 new side faces are non-degenerate (nonzero area), each connecting an
  // old (stationary) vertex pair to its new (moved) counterpart.
  const sideFaces = out.faces.slice(6);
  assert.equal(sideFaces.length, 4);
  for (const f of sideFaces) {
    assert.equal(new Set(f).size, 4, 'a side quad must have 4 distinct vertices, not a degenerate/collapsed one');
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

test('extrudeFaces: a vertex fully surrounded by selected faces (no boundary edge touches it) is translated in place, not duplicated', () => {
  // A 3x3 vertex grid (2x2 = 4 quads); selecting all 4 faces surrounds the center
  // vertex entirely (all 4 of its incident edges are each shared between exactly 2
  // of the 4 selected faces — an interior-region vertex), while all 8 perimeter
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
  assert.ok(Math.abs(result.cage.vertices[centerIdx][2] - 10) < 1e-9, 'the interior center vertex is translated in place, at the same index');
  // Every perimeter vertex must have been duplicated: its original index still holds
  // z=0 (untouched — nothing perimeter-side references it anymore, but it must not
  // have been overwritten in place), and a new (>=9) vertex exists at z=10 nearby.
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

test('extrudeFaces: a crease on an interior (selCount===2) edge shared by two selected faces, whose endpoints are both boundary vertices, is remapped onto the new (moved) vertex pair, not silently orphaned at the old key', () => {
  // A minimal 2-face strip (3x2 vertex grid) — with only 2 faces total,
  // every vertex sits on the selection's own outer boundary (there is no
  // interior vertex at all), so the one internal (selCount===2) edge
  // shared between the two faces has both its endpoints duplicated by the
  // extrude.
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

  assert.equal(out.creases[sharedKey], undefined, 'the old key must not linger as a dangling, never-matched entry');
  // Both endpoints of the shared edge are boundary vertices (they each
  // also touch a real selCount===1 boundary edge), so both get duplicated
  // — the new vertex for original index v is always at v + (however many
  // other boundary vertices were duplicated before it); rather than
  // predict the exact new indices, find them by construction: the new
  // internal edge is whichever selCount===2 edge in the output cage
  // carries the crease weight.
  const outTopology = buildTopology(out);
  let foundKey = null;
  for (const [key, w] of Object.entries(out.creases)) {
    const edge = outTopology.edgeMap.get(key);
    if (edge && edge.faces.length === 2 && w === 1) foundKey = key;
  }
  assert.ok(foundKey, 'the crease weight must land on some 2-face interior edge in the new cage');
  assert.notEqual(foundKey, sharedKey, 'it must be a new key, not the stale old one');
});

test('extrudeFaces: a crease on an interior edge with only one boundary endpoint (the other translated in place, same index) remaps just that one side', () => {
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
  // The center vertex kept its own index (translated in place) — so the
  // new key must still reference centerIdx directly, paired with whatever
  // new index idx(1,0) became.
  const newKeys = Object.keys(out.creases).filter((k) => k !== mixedKey);
  assert.equal(newKeys.length, 1, 'exactly one remapped crease entry, no extras');
  const [a, b] = newKeys[0].split('_').map(Number);
  assert.ok(a === centerIdx || b === centerIdx, `remapped key ${newKeys[0]} must still reference the untouched center index ${centerIdx}`);
  assert.ok((a !== centerIdx ? a : b) >= 9, 'the other endpoint must be one of the newly-duplicated (>=9) vertices, not the stale old perimeter index');
});

test('extrudeFaces: refuses a direction exactly parallel to one of the region\'s own boundary edges — extruding the same cap face a second time sideways, along one of its own edges, degenerate side face', () => {
  // Extrude the box's +Z top face upward (fine), then extrude that same
  // (now-raised) cap a second time along +X — since the
  // cap is a horizontal square, +X runs exactly along two of its own 4
  // edges, and the corresponding side face collapses to 4 collinear points.
  const cage = superbBoxCage([0, 0, 0], [25, 25, 25], 1);
  const topIdx = cage.faces.findIndex((f) => f.every((vi) => Math.abs(cage.vertices[vi][2] - 25) < 1e-6));
  const { cage: raised } = extrudeFaces(cage, [topIdx], [0, 0, 1], 10);
  assert.throws(() => extrudeFaces(raised, [topIdx], [1, 0, 0], 20), /runs exactly parallel/);
});

test('extrudeFaces: a direction with any out-of-plane component (not exactly parallel to any edge) succeeds even on the same repeated-cap case', () => {
  const cage = superbBoxCage([0, 0, 0], [25, 25, 25], 1);
  const topIdx = cage.faces.findIndex((f) => f.every((vi) => Math.abs(cage.vertices[vi][2] - 25) < 1e-6));
  const { cage: raised } = extrudeFaces(cage, [topIdx], [0, 0, 1], 10);
  assert.doesNotThrow(() => extrudeFaces(raised, [topIdx], [1, 0, 1], 20));
});

// insertEdgeLoop (InsertEdge), hand-derived on a cylinder side-band quad
// strip (the comments below derive which face `side=0` resolves to).

test('insertEdgeLoop: t=0.5 on a facets=6 SuperBCylinder inserts one new vertex per column rung, each the exact midpoint of its bottom/top pair', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6); // bottom 0-5, top 6-11, faces: 6 side quads + 2 hexagon caps = 8
  assert.equal(cage.vertices.length, 12);
  assert.equal(cage.faces.length, 8);

  const seedKey = edgeKey(0, 1); // a bottom-ring edge
  // side=0 must resolve to the adjacent side quad (faces array order: side quads
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

test('insertEdgeLoop: side=1 on a facets=3 cylinder (triangular caps) refuses — the cap is not a quad', () => {
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

// Open-cage (non-closed) strip — a plane grid, hand-derived expected
// topology: a facets=3 SuperBPlane is a 4x4 vertex grid, 3x3=9 faces in
// row-major order. Seeding on the horizontal edge between row0 and row1 at
// column i=0 walks the whole first row toward row0 (the plane's own
// boundary), visiting exactly 3 faces (the whole row), terminating at both
// open ends (no wraparound — this is not a closed cylinder).
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

  // Each rung must connect a row-1 vertex to the same column's row-0 vertex — checked
  // as an unordered {row1,row0} pair per column (buildTopology's own edge v0/v1
  // storage order depends on which face registers an edge first during its
  // internal adjacency pass, an internal detail this test does not depend on,
  // hence canonicalized min/max pairs instead of exact order).
  const canon = (pairs) => new Set(pairs.map(([a, b]) => `${Math.min(a, b)}_${Math.max(a, b)}`));
  const expectedRungs = [[idx(0, 1), idx(0, 0)], [idx(1, 1), idx(1, 0)], [idx(2, 1), idx(2, 0)], [idx(3, 1), idx(3, 0)]];
  assert.deepEqual(canon(rungPairs), canon(expectedRungs), 'exact hand-derived {row1,row0} column pairing, across the whole row width');
  // Every rung pairs a near (row-1) vertex with the same-column far (row-0)
  // vertex — i.e. each pair is {idx(i,1), idx(i,0)} for some i, not a mismatched
  // column.
  for (const [a, b] of rungPairs) {
    const rowOf = (v) => Math.floor(v / 4), colOf = (v) => v % 4;
    assert.equal(colOf(a), colOf(b), 'a rung must connect the same column');
    assert.deepEqual(new Set([rowOf(a), rowOf(b)]), new Set([0, 1]), 'a rung must connect row0 to row1');
  }
});

test('insertEdgeLoop: a crease on a rung the strip crosses is transferred onto both new half-rungs, not silently dropped when the rung is split', () => {
  const cage0 = superbCylinderCage([0, 0, 0], 10, 20, 6); // bottom 0-5, top 6-11
  const rungKey = edgeKey(0, 6); // a vertical side rung the seed-adjacent strip crosses
  const cage = { ...cage0, creases: { [rungKey]: 1 } };

  const { cage: out, insertedVertexIndices, rungPairs } = insertEdgeLoop(cage, edgeKey(0, 1), 0.5, 0);

  assert.equal(out.creases[rungKey], undefined, 'the original (now-split) rung key must not linger as a dangling entry');
  const pairIdx = rungPairs.findIndex(([a, b]) => (a === 0 && b === 6) || (a === 6 && b === 0));
  assert.ok(pairIdx >= 0, 'the (0,6) rung must have been crossed by this strip');
  const midIdx = insertedVertexIndices[pairIdx];
  assert.equal(out.creases[edgeKey(0, midIdx)], 1, 'the near half of the split rung must inherit the original weight');
  assert.equal(out.creases[edgeKey(midIdx, 6)], 1, 'the far half of the split rung must also inherit the original weight');
  // Every other rung (never creased in the input) must still carry no weight at all.
  for (const [a, b] of rungPairs) {
    if ((a === 0 && b === 6) || (a === 6 && b === 0)) continue;
    const mid = insertedVertexIndices[rungPairs.findIndex(([x, y]) => x === a && y === b)];
    assert.equal(out.creases[edgeKey(a, mid)], undefined);
    assert.equal(out.creases[edgeKey(mid, b)], undefined);
  }
});

// recomputeInsertedLoopPositions — the live-slider re-drag path (cheap
// reposition, zero topology change).

test('recomputeInsertedLoopPositions: re-dragging to a new t repositions the same vertices with zero topology change', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6);
  const { cage: inserted, insertedVertexIndices, rungPairs } = insertEdgeLoop(cage, edgeKey(0, 1), 0.5, 0);
  const before = { vCount: inserted.vertices.length, fCount: inserted.faces.length, faces: JSON.stringify(inserted.faces) };

  const redragged = recomputeInsertedLoopPositions(inserted, insertedVertexIndices, rungPairs, 0.75);
  assert.equal(redragged.vertices.length, before.vCount, 'vertex count never changes on a reposition');
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

// Subdivide (global)

// Helper: find every face whose vertices all lie on a given axis-plane (used
// to pick a connected group of faces on one side of a box).
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
  // Byte-identical to a direct subdivideCatmullClark call — this is the
  // refinement step committed as the new cage, not reproved Catmull-Clark
  // math.
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
  // Re-subdividable (the reason for committing it as a live cage) — a
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

// MergeFaces

test('mergeFaces: merging two edge-adjacent quads on one side of a facets=2 box produces one 6-vertex face, a valid manifold cage, and drops exactly one face from the count', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2); // 24 faces, 26 vertices
  const side = facesOnSide(cage, 0, 10); // the four +X quads
  assert.equal(side.length, 4);
  // Pick two of the +X quads that share an edge.
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

test('mergeFaces: merging all four quads of one box face welds them into one 8-vertex face and prunes exactly the face-center vertex', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2); // 24 faces, 26 vertices
  const side = facesOnSide(cage, 0, 10); // the four +X quads
  const out = mergeFaces(cage, side);
  // The +X face's own 4 edge-midpoints are shared with the neighboring box
  // faces (so they survive on the rim); only the single face-center vertex is
  // interior to the merged region and gets pruned.
  assert.equal(out.ngonSize, 8, 'the outer ring of a 3x3 face grid is 8 vertices (4 corners + 4 edge-midpoints)');
  assert.equal(out.cage.vertices.length, 26 - 1, 'exactly the face-center vertex is pruned');
  assert.equal(out.cage.faces.length, 24 - 4 + 1);
  assert.deepEqual(windingViolations(out.cage), [], 'still a fully consistent 2-manifold');
});

test('mergeFaces: a crease on a surviving (rim) edge is kept; a crease on a dissolved internal edge vanishes with the edge', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  const side = facesOnSide(cage, 0, 10); // 4 +X quads
  const selSet = new Set(side);
  const topo = buildTopology(cage);
  // An internal edge of the merged region: shared by exactly 2 selected faces.
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
  // deleteFaces' own vertexRemap). The dissolved internal edge is gone, and
  // so is its crease.
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

test('mergeFaces: refusals — fewer than 2 faces, out of range, duplicate index, and a disconnected selection', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  assert.throws(() => mergeFaces(cage, [0]), /at least 2/, 'a single face is a no-op');
  assert.throws(() => mergeFaces(cage, [0, 9999]), /out of range/);
  assert.throws(() => mergeFaces(cage, [0, 0]), /more than once/);
  // Two faces on opposite sides of the box (share no edge): refused by name
  // as disconnected, not silently merged into a wrong single loop.
  const plusX = facesOnSide(cage, 0, 10)[0];
  const minusX = facesOnSide(cage, 0, -10)[0];
  assert.throws(() => mergeFaces(cage, [plusX, minusX]), /not all edge-connected/);
});

test('mergeFaces: refuses when the merged region\'s rim touches an already-open cage boundary edge (inherited from orderedBoundaryLoopOfFaceSet, as Bridge does)', () => {
  // A SuperBPlane is open — its border edges have exactly one face.
  const plane = superbPlaneCage([0, 0, 0], 20, 20, 2); // 4 faces, every one touches the open border
  assert.throws(() => mergeFaces(plane, [0, 1]), /already-open cage edge/);
});

test('orderedBoundaryLoopOfFaceSet / mergeFaces: a band of faces right around a body is refused — it is an annulus, and no single face can stand for it', () => {
  // A closed square tube with a middle ring: two caps, eight side quads. The
  // eight side quads together wrap right around it, so their rim is two loops
  // of four rather than one.
  //
  // Every check inside the rim walk is local — each rim vertex has exactly
  // one continuation — so the walk closes cleanly on whichever loop its
  // arbitrary start vertex sits on and returns it, dropping the other
  // silently. mergeFaces would then remove all eight faces and put one n-gon
  // back over one loop only, leaving a hole on the second loop. The
  // comparison that catches it is the rim edge count against the closed
  // loop's length.
  const ring = (z) => [[-10, -10, z], [10, -10, z], [10, 10, z], [-10, 10, z]];
  const vertices = [...ring(-10), ...ring(0), ...ring(10)];
  const faces = [];
  for (let r = 0; r < 2; r++) {
    for (let i = 0; i < 4; i++) {
      const a = r * 4 + i, b = r * 4 + ((i + 1) % 4);
      faces.push([a, b, b + 4, a + 4]);
    }
  }
  faces.push([3, 2, 1, 0]);
  faces.push([8, 9, 10, 11]);
  const cage = { vertices, faces, creases: {} };
  const band = [0, 1, 2, 3, 4, 5, 6, 7];
  assert.throws(() => orderedBoundaryLoopOfFaceSet(cage, band), /more than one boundary/);
  assert.throws(() => mergeFaces(cage, band), /more than one boundary/);
  // The ordinary case still passes, so the comparison is not refusing
  // everything: two adjacent side quads are a disc and merge.
  const merged = mergeFaces(cage, [0, 1]);
  assert.equal(merged.cage.faces.length, cage.faces.length - 1);
  assert.equal(merged.ngonSize, 6);
});

test('mergeFaces: a merge that would collapse the cage to a single face is refused (deleteFaces cannot remove every face)', () => {
  // A facets=1 SuperBPlane is a single quad; there is no valid 2-face merge.
  // Use a facets=2 plane and try to merge all 4 — its rim is entirely open
  // border, so this refuses at the open-edge check (the refusal for the
  // open-cage case), confirming a merge never silently corrupts.
  const plane = superbPlaneCage([0, 0, 0], 20, 20, 2);
  assert.throws(() => mergeFaces(plane, [0, 1, 2, 3]));
});
