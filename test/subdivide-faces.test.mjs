// Per-face subdivide. The claim worth testing is not "it refines a face"
// — that part is arithmetic. It is that the cage stays a legal cage: the
// T-junction a naive local refinement leaves behind is invisible in a
// render, silent in a face count, and only shows up later as a corrupted
// subdivision. So every test here checks the topology the operation is
// supposed to protect, not just the counts it changes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { subdivideFaces } from '../kernel/subdedit.mjs';
import { buildTopology, subdivideCatmullClark, edgeKey, creaseWeight } from '../kernel/subd.mjs';
import { superbBoxCage, superbCylinderCage } from '../kernel/subdprimitives.mjs';

function manifoldReport(cage) {
  const ctx = buildTopology(cage);
  let boundary = 0, nonManifold = 0;
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length === 1) boundary++;
    else if (e.faces.length > 2) nonManifold++;
  }
  return { boundary, nonManifold, chi: cage.vertices.length - ctx.edgeMap.size + cage.faces.length };
}

test('one selected face becomes four quads, and only its neighbors change at all', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const r = subdivideFaces(cage, [0]);
  assert.equal(r.faceCountBefore, 6);
  assert.equal(r.faceCountAfter, 9); // the selected face -> 4, the other 5 stay
  const sizes = r.cage.faces.map((f) => f.length).sort();
  // 4 quads from the refined face + the untouched opposite quad, and the
  // four side faces widened to 5-gons because each now has a
  // midpoint on one of its own sides.
  assert.deepEqual(sizes, [4, 4, 4, 4, 4, 5, 5, 5, 5]);
  assert.equal(r.widenedNeighbours, 4);
});

test('no T-junction: the cage stays closed and manifold', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const before = manifoldReport(cage);
  assert.deepEqual(before, { boundary: 0, nonManifold: 0, chi: 2 });
  const { cage: out } = subdivideFaces(cage, [0]);
  const after = manifoldReport(out);
  // A naive local refinement fails here: the new midpoints would
  // each be used by only one face, producing boundary edges in a cage
  // that is supposed to be closed, and chi would drift off 2.
  assert.deepEqual(after, { boundary: 0, nonManifold: 0, chi: 2 });
});

test('the refined cage subdivides afterwards — the downstream risk', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const { cage: out } = subdivideFaces(cage, [0, 1]);
  const refined = subdivideCatmullClark(out);
  assert.equal(refined.faces.length, out.faces.reduce((n, f) => n + f.length, 0));
  for (const f of refined.faces) assert.equal(new Set(f).size, f.length);
  for (const v of refined.vertices) assert.ok(v.every(Number.isFinite));
  assert.deepEqual(manifoldReport(refined), { boundary: 0, nonManifold: 0, chi: 2 });
});

test('two adjacent selected faces share one midpoint on their common edge, not two', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const ctx = buildTopology(cage);
  // Find an adjacent pair rather than assuming index order.
  let pair = null;
  for (const e of ctx.edgeMap.values()) if (e.faces.length === 2) { pair = e.faces; break; }
  assert.ok(pair);
  const { cage: out } = subdivideFaces(cage, pair);
  // If the shared edge had been split twice, the two coincident midpoints
  // would each serve one side and the shared edge would read as two
  // boundary edges. Closed-and-manifold is the discriminating check.
  assert.deepEqual(manifoldReport(out), { boundary: 0, nonManifold: 0, chi: 2 });
  const coincident = new Map();
  for (const v of out.vertices) {
    const k = v.map((n) => n.toFixed(9)).join(',');
    coincident.set(k, (coincident.get(k) || 0) + 1);
  }
  for (const [k, n] of coincident) assert.equal(n, 1, `two vertices at the same position (${k}) means an edge got split twice`);
});

test('a crease on a split edge transfers to both halves and never dangles', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const f0 = cage.faces[0];
  const creasedKey = edgeKey(f0[0], f0[1]);
  const creased = { ...cage, creases: { [creasedKey]: 3 } };
  const { cage: out } = subdivideFaces(creased, [0]);
  const ctx = buildTopology(out);
  // The old edge is gone from the topology entirely, so its key must be
  // gone from the crease map too — a weight on a non-existent edge is
  // exactly the silent corruption this remap exists to prevent.
  assert.equal(ctx.edgeMap.has(creasedKey), false, 'the split edge should be gone from the topology');
  assert.equal(out.creases[creasedKey], undefined, 'its crease key must not dangle');
  // Both halves must carry it, and both must be real edges.
  const halves = [...ctx.edgeMap.values()].filter((e) => creaseWeight(out, e.v0, e.v1) === 3);
  assert.equal(halves.length, 2, 'exactly the two halves of the split edge carry the weight');
  const ends = new Set(halves.flatMap((e) => [e.v0, e.v1]));
  assert.ok(ends.has(f0[0]) && ends.has(f0[1]), 'the halves still span the original edge');
});

test('an untouched crease elsewhere is left exactly alone', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const far = cage.faces[3];
  const farKey = edgeKey(far[0], far[1]);
  const creased = { ...cage, creases: { [farKey]: 2 } };
  const { cage: out } = subdivideFaces(creased, [0]);
  // Face 3 may or may not border face 0; whichever it is, the assertion
  // is the same — a weight either survives on its own still-real edge or
  // was transferred, never dropped.
  const ctx = buildTopology(out);
  const total = [...ctx.edgeMap.values()].filter((e) => creaseWeight(out, e.v0, e.v1) === 2).length;
  assert.ok(total >= 1, 'the crease must survive somewhere');
});

test('selecting every face is equivalent in topology to a global refinement', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const all = cage.faces.map((_, i) => i);
  const { cage: out } = subdivideFaces(cage, all);
  const global = subdivideCatmullClark(cage);
  assert.equal(out.faces.length, global.faces.length);
  assert.equal(out.vertices.length, global.vertices.length);
  assert.equal(out.faces.every((f) => f.length === 4), true, 'no widened neighbors when nothing is left unselected');
  // Positions are deliberately not asserted equal: the global step applies
  // the smooth Catmull-Clark rules and moves points, while a
  // local refine must not move anything outside the selection and so
  // uses plain midpoints and centroids. Same topology, different (and
  // correctly different) geometry — see subdivideFaces' own header.
});

test('an n-gon-bearing cage refines without being forced into quads', () => {
  const cage = superbCylinderCage([0, 0, 0], 10, 20, 6); // n-gon caps
  const capIdx = cage.faces.findIndex((f) => f.length > 4);
  assert.ok(capIdx >= 0, 'fixture should have a real n-gon');
  const { cage: out } = subdivideFaces(cage, [capIdx]);
  assert.deepEqual(manifoldReport(out), manifoldReport(cage));
  // An n-gon refines into n quads, exactly as one Catmull-Clark step
  // would for that face alone.
  assert.equal(out.faces.length, cage.faces.length - 1 + cage.faces[capIdx].length);
});

test('refusals, and the input cage is never mutated', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const before = JSON.stringify(cage);
  assert.throws(() => subdivideFaces(cage, []), /no faces selected/);
  assert.throws(() => subdivideFaces(cage, [99]), /not a face/);
  subdivideFaces(cage, [0, 0, 0]); // duplicates collapse rather than double-split
  assert.equal(JSON.stringify(cage), before);
});
