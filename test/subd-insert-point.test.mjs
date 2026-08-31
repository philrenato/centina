// INSERT POINT — split one edge with a new vertex. What matters is not that
// a vertex appeared but that BOTH faces using that edge learned about it: a
// cage where only one did still renders and still counts right, and
// subdivides into a crack.
import test from 'node:test';
import assert from 'node:assert/strict';
import { insertPointOnEdge } from '../kernel/subdedit.mjs';
import { buildTopology, subdivideCatmullClark, edgeKey, creaseWeight } from '../kernel/subd.mjs';
import { superbBoxCage, superbPlaneCage } from '../kernel/subdprimitives.mjs';

function report(cage) {
  const ctx = buildTopology(cage);
  let boundary = 0, nonManifold = 0;
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length === 1) boundary++;
    else if (e.faces.length > 2) nonManifold++;
  }
  return { boundary, nonManifold, chi: cage.vertices.length - ctx.edgeMap.size + cage.faces.length };
}

test('an interior edge gains one vertex, and BOTH its faces become 5-gons', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const ctx = buildTopology(cage);
  const interior = [...ctx.edgeMap.entries()].find(([, e]) => e.faces.length === 2);
  const [key, edge] = interior;
  const r = insertPointOnEdge(cage, key, 0.5);
  assert.equal(r.cage.vertices.length, cage.vertices.length + 1);
  assert.equal(r.cage.faces.length, cage.faces.length); // no new faces — this is not a subdivide
  assert.equal(r.widenedFaceIndices.length, 2);
  for (const fi of edge.faces) assert.equal(r.cage.faces[fi].length, 5);
  // Every other face untouched.
  cage.faces.forEach((f, fi) => { if (!edge.faces.includes(fi)) assert.deepEqual(r.cage.faces[fi], f); });
});

test('THE POINT: no T-junction — the new vertex is used by both faces, and the cage stays closed and manifold', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const before = report(cage);
  const ctx = buildTopology(cage);
  const key = [...ctx.edgeMap.keys()][0];
  const { cage: out, newVertexIndex } = insertPointOnEdge(cage, key, 0.5);
  const users = out.faces.filter((f) => f.includes(newVertexIndex)).length;
  assert.equal(users, 2, 'the new vertex must be used by BOTH faces that shared the split edge');
  const after = report(out);
  assert.equal(after.boundary, before.boundary);
  assert.equal(after.nonManifold, 0);
  assert.equal(after.chi, before.chi, 'splitting an edge adds one vertex and one edge — the Euler characteristic must not move');
});

test('and the widened cage really subdivides — the failure a T-junction would only show up in', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const ctx = buildTopology(cage);
  const { cage: out } = insertPointOnEdge(cage, [...ctx.edgeMap.keys()][0], 0.35);
  const refined = subdivideCatmullClark(out);
  for (const f of refined.faces) assert.equal(new Set(f).size, f.length);
  for (const v of refined.vertices) assert.ok(v.every(Number.isFinite));
  assert.equal(report(refined).nonManifold, 0);
  assert.equal(report(refined).boundary, 0);
});

test('t places the point along the edge, exactly', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const ctx = buildTopology(cage);
  const [key, edge] = [...ctx.edgeMap.entries()][0];
  const { cage: out, newVertexIndex } = insertPointOnEdge(cage, key, 0.25);
  const a = cage.vertices[edge.v0], b = cage.vertices[edge.v1];
  const expect = [0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * 0.25);
  const got = out.vertices[newVertexIndex];
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(got[k] - expect[k]) < 1e-12);
});

test('a NAKED edge works too — one face widens, the boundary stays a boundary', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 1);
  const ctx = buildTopology(cage);
  const [key, edge] = [...ctx.edgeMap.entries()].find(([, e]) => e.faces.length === 1);
  const { cage: out, newVertexIndex, widenedFaceIndices } = insertPointOnEdge(cage, key, 0.5);
  assert.equal(widenedFaceIndices.length, 1);
  assert.equal(out.faces[edge.faces[0]].length, 5);
  assert.equal(out.faces.filter((f) => f.includes(newVertexIndex)).length, 1);
  assert.equal(report(out).nonManifold, 0);
  assert.equal(report(out).boundary, report(cage).boundary + 1, 'splitting one naked edge leaves two naked edges in its place');
});

test('a crease on the split edge transfers to BOTH halves and never dangles', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const ctx = buildTopology(cage);
  const [key, edge] = [...ctx.edgeMap.entries()][0];
  const creased = { ...cage, creases: { [key]: 4 } };
  const { cage: out, newVertexIndex } = insertPointOnEdge(creased, key, 0.5);
  const outCtx = buildTopology(out);
  assert.equal(outCtx.edgeMap.has(key), false, 'the split edge no longer exists');
  assert.equal(out.creases[key], undefined, 'so its key must not dangle');
  assert.equal(creaseWeight(out, edge.v0, newVertexIndex), 4);
  assert.equal(creaseWeight(out, newVertexIndex, edge.v1), 4);
});

test('honest refusals, and the input cage is never mutated', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const before = JSON.stringify(cage);
  assert.throws(() => insertPointOnEdge(cage, '999_1000', 0.5), /not a real edge/);
  const key = [...buildTopology(cage).edgeMap.keys()][0];
  assert.throws(() => insertPointOnEdge(cage, key, 0), /strictly between/);
  assert.throws(() => insertPointOnEdge(cage, key, 1), /strictly between/);
  insertPointOnEdge(cage, key, 0.5);
  assert.equal(JSON.stringify(cage), before);
});
