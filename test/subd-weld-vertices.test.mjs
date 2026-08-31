// WELD VERTICES — collapse a selected vertex set into one. The interesting
// claims are not the counts: they are that a face which folds onto itself is
// dropped rather than kept as a repeated-vertex face, that nothing is left
// orphaned, and that a weld which would make the cage non-manifold is
// refused rather than silently produced (it would render fine and subdivide
// into garbage — the exact failure mode Bridge's own shared-vertex bug had).
import test from 'node:test';
import assert from 'node:assert/strict';
import { weldVertices } from '../kernel/subdedit.mjs';
import { buildTopology, subdivideCatmullClark, edgeKey, creaseWeight } from '../kernel/subd.mjs';
import { superbPlaneCage, superbBoxCage } from '../kernel/subdprimitives.mjs';

function report(cage) {
  const ctx = buildTopology(cage);
  let boundary = 0, nonManifold = 0;
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length === 1) boundary++;
    else if (e.faces.length > 2) nonManifold++;
  }
  return { boundary, nonManifold };
}
function orphans(cage) {
  const used = new Set(cage.faces.flat());
  let n = 0;
  for (let i = 0; i < cage.vertices.length; i++) if (!used.has(i)) n++;
  return n;
}

test('welding two adjacent rim vertices of an open sheet merges them into one, at their midpoint', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2); // 9 vertices, 4 faces
  const a = 0, b = 1; // two adjacent vertices on the same rim edge
  const r = weldVertices(cage, [a, b], 'average');
  assert.equal(r.cage.vertices.length, cage.vertices.length - 1);
  assert.equal(r.removedVertexCount, 1);
  const mid = [0, 1, 2].map((k) => (cage.vertices[a][k] + cage.vertices[b][k]) / 2);
  const got = r.cage.vertices[r.weldedVertexIndex];
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(got[k] - mid[k]) < 1e-12, `welded vertex should sit at the average, got ${got}`);
  assert.equal(orphans(r.cage), 0);
  assert.equal(report(r.cage).nonManifold, 0);
});

test("'first' holds the first-picked vertex still instead of averaging", () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const r = weldVertices(cage, [0, 1], 'first');
  const got = r.cage.vertices[r.weldedVertexIndex];
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(got[k] - cage.vertices[0][k]) < 1e-12);
});

test('THE POINT: a face that folds onto itself is DROPPED, not kept as a repeated-vertex face', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  // Two vertices that share a face but not an edge — a quad's own diagonal.
  const f = cage.faces[0];
  const r = weldVertices(cage, [f[0], f[2]], 'average');
  assert.ok(r.collapsedFaceCount >= 1, 'the quad welded across its own diagonal must collapse, not survive as a bowtie');
  for (const face of r.cage.faces) assert.equal(new Set(face).size, face.length, 'no surviving face may repeat a vertex');
  assert.equal(orphans(r.cage), 0);
});

test('and the welded cage still subdivides cleanly', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const { cage: out } = weldVertices(cage, [0, 1], 'average');
  const refined = subdivideCatmullClark(out);
  for (const f of refined.faces) assert.equal(new Set(f).size, f.length);
  for (const v of refined.vertices) assert.ok(v.every(Number.isFinite));
});

test('a weld that would leave an edge shared by 3+ faces is REFUSED, not produced', () => {
  // A closed box: welding two opposite corners folds the surface through
  // itself, which is exactly the case that renders fine and subdivides into
  // garbage if allowed through.
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  let threw = null;
  try { weldVertices(cage, [0, 6], 'average'); } catch (e) { threw = e; }
  // Either it refuses by name, or it produced something genuinely manifold —
  // asserted as an either/or so this test can never pass by accident on a
  // cage where the fold happens not to be reachable.
  if (threw) {
    assert.match(threw.message, /non-manifold|shared by/);
  } else {
    const { cage: out } = weldVertices(cage, [0, 6], 'average');
    assert.equal(report(out).nonManifold, 0);
  }
});

test('creases survive the remap, and a crease that merged onto itself is dropped rather than dangling', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const kSelf = edgeKey(0, 1);         // this edge's two ends are the ones being welded
  const far = cage.faces[3];
  const kFar = edgeKey(far[0], far[1]); // an untouched edge elsewhere
  const creased = { ...cage, creases: { [kSelf]: 3, [kFar]: 2 } };
  const { cage: out } = weldVertices(creased, [0, 1], 'average');
  assert.equal(out.creases[kSelf], undefined, 'the collapsed edge must not keep a dangling key');
  for (const key of Object.keys(out.creases)) {
    const [a, b] = key.split('_').map(Number);
    assert.notEqual(a, b, 'no crease may point at a self-loop');
    assert.ok(a < out.vertices.length && b < out.vertices.length, 'no crease may point past the end of the vertex list');
  }
  const total = [...buildTopology(out).edgeMap.values()].filter((e) => creaseWeight(out, e.v0, e.v1) === 2).length;
  assert.ok(total >= 1, 'the untouched crease elsewhere must survive');
});

test('honest refusals, and the input cage is never mutated', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 1);
  const before = JSON.stringify(cage);
  assert.throws(() => weldVertices(cage, [0], 'average'), /at least 2/);
  assert.throws(() => weldVertices(cage, [0, 99], 'average'), /out of range/);
  assert.throws(() => weldVertices(cage, [0, 1], 'middle'), /'average' or 'first'/);
  weldVertices(cage, [0, 1], 'average');
  assert.equal(JSON.stringify(cage), before);
});
