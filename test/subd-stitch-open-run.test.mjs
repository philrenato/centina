// STITCH's own open-run preconditions — the same three the bridges check,
// plus the closed-rim refusal, now shared through checkOpenRunChain.
//
// The closed-rim fixtures are REAL subdPipeCage tubes rather than hand-typed
// rings, for the same reason the closed-rim hub's own tests are: a rim built
// by the construction that would actually produce one carries whatever ring
// orientation and vertex ordering that construction genuinely gives, and a
// loop typed to be convenient carries whatever the author assumed. The
// rotational mismatch these tests are about only exists because two real tubes
// do not agree on where vertex 0 sits.
//
// Every structural claim is recomputed here from the raw { vertices, faces },
// never asked of buildTopology, so a shared mistake between this file and the
// code under test cannot agree its way to a pass.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { stitchEdgeRuns } from '../kernel/subdedit.mjs';
import { subdPipeCage } from '../kernel/subdpipe.mjs';
import { makeLine } from '../kernel/primitives.mjs';

function edgeFaceCounts(cage) {
  const counts = new Map();
  for (const f of cage.faces) {
    for (let i = 0; i < f.length; i++) {
      const a = f[i], b = f[(i + 1) % f.length];
      const k = a < b ? `${a}_${b}` : `${b}_${a}`;
      counts.set(k, (counts.get(k) || 0) + 1);
    }
  }
  return counts;
}
const nakedEdgeCount = (cage) => [...edgeFaceCounts(cage)].filter(([, n]) => n === 1).length;
const nonManifoldEdgeCount = (cage) => [...edgeFaceCounts(cage)].filter(([, n]) => n > 2).length;
function directedEdgeReuse(cage) {
  const seen = new Set();
  let reused = 0;
  for (const f of cage.faces) for (let i = 0; i < f.length; i++) {
    const k = `${f[i]}>${f[(i + 1) % f.length]}`;
    if (seen.has(k)) reused++;
    seen.add(k);
  }
  return reused;
}

// Two real tubes pointing at each other along Z, each capped at its far end so
// the only naked edges in the assembled cage are the two facing rims.
function facingTubes({ facets = 6, radius = 5 } = {}) {
  const lower = subdPipeCage(makeLine([0, 0, 0], [0, 0, 20]), { radius, facets, segments: 2, capStart: 'flat', capEnd: 'none' });
  const upper = subdPipeCage(makeLine([0, 0, 60], [0, 0, 40]), { radius, facets, segments: 2, capStart: 'flat', capEnd: 'none' });
  const vertices = [...lower.vertices.map((v) => v.slice()), ...upper.vertices.map((v) => v.slice())];
  const off = lower.vertices.length;
  const faces = [...lower.faces.map((f) => [...f]), ...upper.faces.map((f) => f.map((i) => i + off))];
  return {
    cage: { vertices, faces, creases: {} },
    rimA: [...lower.endRim],
    rimB: upper.endRim.map((i) => i + off),
  };
}

// Two independent flat quads SIDE BY SIDE with a small gap, both wound
// counter-clockwise about +Z. The legitimate Stitch case, kept alongside every
// refusal so a guard that refuses everything cannot pass these tests.
//
// Side by side rather than stacked, deliberately: welding A's right edge
// [1,2] to B's left edge [4,7] is the one pairing that leaves the shared edge
// traversed in OPPOSITE directions by its two faces, so the result is a
// consistently wound sheet. Two same-wound quads stacked face to face and
// welded along the same edge of each would not be — stitchEdgeRuns has no
// winding check, and cannot get one by searching, since a merge builds no new
// faces whose winding it could choose.
function twoQuads() {
  return {
    vertices: [
      [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
      [10.2, 0, 0], [20.2, 0, 0], [20.2, 10, 0], [10.2, 10, 0],
    ],
    faces: [[0, 1, 2, 3], [4, 5, 6, 7]],
    creases: {},
  };
}

test('stitchEdgeRuns: the fixture really is two open tube rims — 12 naked edges, nothing non-manifold', () => {
  const { cage, rimA, rimB } = facingTubes();
  assert.equal(rimA.length, 6);
  assert.equal(rimB.length, 6);
  assert.equal(nakedEdgeCount(cage), 12, 'six naked edges per open rim');
  assert.equal(nonManifoldEdgeCount(cage), 0);
  assert.equal(directedEdgeReuse(cage), 0, 'each tube is consistently wound on its own');
  // The two rims genuinely disagree about where vertex 0 sits: this is the
  // rotational mismatch the refusal exists for, and it is a property of two
  // real tubes, not something arranged for the test.
  const worstPairXY = Math.max(...rimA.map((a, i) => {
    const p = cage.vertices[a], q = cage.vertices[rimB[i]];
    return Math.hypot(p[0] - q[0], p[1] - q[1]);
  }));
  assert.ok(worstPairXY > 1, `index-for-index pairing is genuinely misaligned in plan (worst ${worstPairXY.toFixed(2)}mm)`);
});

test('stitchEdgeRuns: refuses a CLOSED rim by name, naming the offending closing edge and the right function', () => {
  const { cage, rimA, rimB } = facingTubes();
  assert.throws(() => stitchEdgeRuns(cage, rimA, rimB, 'average'), (err) => {
    assert.match(err.message, /^stitchEdgeRuns: runA is a CLOSED rim, not an open run/);
    assert.match(err.message, /already joined by the naked edge "\d+_\d+"/, 'names the real closing edge');
    assert.match(err.message, /bridgeBoundaryLoops/, 'points at the two-rim function');
    assert.match(err.message, /bridgeClosedRimsHub/, 'points at the N-rim function');
    return true;
  });
});

test('stitchEdgeRuns: refuses a closed rim however it is written — reversed, rotated, or wrapped', () => {
  const { cage, rimA, rimB } = facingTubes();
  const rotated = [...rimB.slice(3), ...rimB.slice(0, 3)];
  assert.throws(() => stitchEdgeRuns(cage, rimA, [...rimB].reverse()), /CLOSED rim, not an open run/);
  assert.throws(() => stitchEdgeRuns(cage, rimA, rotated), /CLOSED rim, not an open run/);
  // A wrapped list repeats its first vertex and is caught one check earlier,
  // by the simple-chain rule rather than the closing-edge rule.
  assert.throws(() => stitchEdgeRuns(cage, [...rimA, rimA[0]], [...rimB, rimB[0]]), /repeats a vertex/);
});

test('stitchEdgeRuns: the refused closed-rim merge is what the OLD code produced — a crushed, inconsistently wound weld', () => {
  // Reproduces the unguarded behaviour directly, so the refusal is proven to
  // protect against something real rather than being merely conservative.
  // This is stitchEdgeRuns' own arithmetic — pair by index (forward or
  // reversed, whichever is nearer in total), average each pair — run here
  // instead of trusting a description of it.
  const { cage, rimA, rimB } = facingTubes();
  const totalDistSq = (seq) => rimA.reduce((acc, a, i) => {
    const p = cage.vertices[a], q = cage.vertices[seq[i]];
    return acc + (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  }, 0);
  const reversed = [...rimB].reverse();
  const aligned = totalDistSq(reversed) < totalDistSq(rimB) ? reversed : rimB;

  const merged = rimA.map((a, i) => {
    const p = cage.vertices[a], q = cage.vertices[aligned[i]];
    return [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
  });
  const radii = merged.map((p) => Math.hypot(p[0], p[1]));
  const worstRadiusError = Math.max(...radii.map((r) => Math.abs(r - 5)));
  assert.ok(worstRadiusError > 0.5, `the index-paired weld genuinely crushes the ring off its own 5mm radius (worst error ${worstRadiusError.toFixed(3)}mm)`);
  assert.ok(Math.min(...radii) < 0.001, 'diametrically opposite vertices average onto the tube axis');

  // And the topology it would have produced: every rim edge welded (no slit —
  // the bridges' failure mode is NOT this one), but every welded edge
  // traversed twice in the same direction.
  const remap = new Map(aligned.map((b, i) => [b, rimA[i]]));
  const welded = { vertices: cage.vertices, faces: cage.faces.map((f) => f.map((i) => (remap.has(i) ? remap.get(i) : i))), creases: {} };
  assert.equal(nakedEdgeCount(welded), 0, 'no slit — a weld leaves no closing edge unattached');
  assert.equal(directedEdgeReuse(welded), 6, 'all six welded edges are traversed twice the same way — not consistently wound');
});

test('stitchEdgeRuns: refuses a run whose vertices are not joined by real cage edges', () => {
  const { cage, rimA, rimB } = facingTubes();
  // Every other vertex around each rim: real vertices, no edge between them.
  assert.throws(
    () => stitchEdgeRuns(cage, [rimA[0], rimA[2], rimA[4]], [rimB[0], rimB[2], rimB[4]]),
    /runA is not a connected chain — vertices \d+ and \d+ are not joined by a real cage edge/,
  );
});

test('stitchEdgeRuns: refuses a run of INTERIOR (2-face) edges, which welding turns non-manifold', () => {
  const { cage } = facingTubes();
  const counts = edgeFaceCounts(cage);
  const interiorRun = (() => {
    // A connected chain around one tube's own middle ring, whose edges are all
    // 2-face interior edges.
    const ring = cage.vertices.map((_, i) => i).filter((i) => Math.abs(cage.vertices[i][2] - 10) < 1e-9);
    return ring.slice(0, 3);
  })();
  assert.equal(interiorRun.length, 3);
  for (let i = 0; i + 1 < interiorRun.length; i++) {
    const a = interiorRun[i], b = interiorRun[i + 1];
    assert.equal(counts.get(a < b ? `${a}_${b}` : `${b}_${a}`), 2, 'the fixture run really is interior');
  }
  const other = cage.vertices.map((_, i) => i).filter((i) => Math.abs(cage.vertices[i][2] - 50) < 1e-9).slice(0, 3);
  assert.throws(
    () => stitchEdgeRuns(cage, interiorRun, other),
    /runA's edge "\d+_\d+" is not a naked \(open\) edge — it already has 2 faces/,
  );
});

test('stitchEdgeRuns: a genuinely open, connected, naked run still welds — the guard refuses nothing legitimate', () => {
  const cage = twoQuads();
  assert.equal(nakedEdgeCount(cage), 8);
  const { cage: out, mergedVertexIndices, collapsedFaceCount } = stitchEdgeRuns(cage, [1, 2], [4, 7], 'average');
  assert.equal(collapsedFaceCount, 0);
  assert.equal(out.vertices.length, 6);
  assert.equal(out.faces.length, 2);
  assert.equal(nakedEdgeCount(out), 6, 'the two runs became one shared interior edge');
  assert.equal(nonManifoldEdgeCount(out), 0);
  assert.equal(directedEdgeReuse(out), 0, 'a correctly paired seam is still consistently wound');
  assert.equal(mergedVertexIndices.length, 2);
  assert.ok(Math.abs(out.vertices[mergedVertexIndices[0]][0] - 10.1) < 1e-9, "'average' still lands at the midpoint of the gap");
});

test('stitchEdgeRuns: a 2-vertex run is exempt from the closing-edge rule — its own single edge IS the edge between its ends', () => {
  // Without the length >= 3 exemption this legitimate case would read as a
  // closed rim, since a 2-vertex run's ends are joined by a naked edge.
  const cage = twoQuads();
  assert.doesNotThrow(() => stitchEdgeRuns(cage, [1, 2], [4, 7]));
});

test('stitchEdgeRuns: an OPEN 3-vertex run whose ends happen to be joined is a closed triangle, and is refused', () => {
  // The one place the closing-edge rule is deliberately strict rather than
  // over-strict: a run that already spans its whole rim bar one edge is the
  // same loop under another name.
  const cage = {
    vertices: [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0, 0, 5], [1, 0, 5], [0.5, 1, 5]],
    faces: [[0, 1, 2], [3, 4, 5]],
    creases: {},
  };
  assert.equal(nakedEdgeCount(cage), 6);
  assert.throws(() => stitchEdgeRuns(cage, [0, 1, 2], [3, 4, 5]), /runA is a CLOSED rim, not an open run/);
});
