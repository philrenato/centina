// SLIDE EDGE — move an edge (or a loop of them) along the surface. Pure
// geometry: the claims worth checking are that topology does not move at all,
// that each vertex lands on a real rail of its own rather than drifting off
// the surface, that the loop slides COHERENTLY (all one way, not zig-zag),
// and that a slide big enough to collapse an edge is refused.
import test from 'node:test';
import assert from 'node:assert/strict';
import { slideEdges } from '../kernel/subdedit.mjs';
import { buildTopology, edgeKey, subdivideCatmullClark } from '../kernel/subd.mjs';
import { superbPlaneCage, superbBoxCage } from '../kernel/subdprimitives.mjs';

// A 3x3-vertex plane: the middle column is a real interior "loop" of two
// edges with rails running left and right of it.
function midColumnKeys(cage) {
  // vertices are laid out row-major, 3 per row: indices 1, 4, 7 are the
  // middle column.
  return [edgeKey(1, 4), edgeKey(4, 7)];
}

test('topology is completely unchanged — a slide moves positions and nothing else', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const { cage: out } = slideEdges(cage, midColumnKeys(cage), 0.3);
  assert.equal(out.vertices.length, cage.vertices.length);
  assert.deepEqual(out.faces, cage.faces);
  assert.deepEqual(out.creases, cage.creases);
});

test('THE POINT: every slid vertex lands ON one of its own rails — not off the surface', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const t = 0.4;
  const { cage: out, slidVertexCount } = slideEdges(cage, midColumnKeys(cage), t);
  assert.equal(slidVertexCount, 3, 'the middle column has three vertices to slide');
  const ctx = buildTopology(cage);
  for (const vi of [1, 4, 7]) {
    const before = cage.vertices[vi], after = out.vertices[vi];
    // The move must be exactly t of the way along SOME real incident edge.
    const ok = ctx.vertexEdges[vi].some((e) => {
      const other = e.v0 === vi ? e.v1 : e.v0;
      const expect = [0, 1, 2].map((k) => before[k] + (cage.vertices[other][k] - before[k]) * t);
      return [0, 1, 2].every((k) => Math.abs(after[k] - expect[k]) < 1e-9);
    });
    assert.ok(ok, `vertex ${vi} must land exactly ${t} of the way along a real incident edge, got ${after}`);
  }
});

test('and the whole loop slides the SAME way — a per-vertex choice would zig-zag', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const { cage: out } = slideEdges(cage, midColumnKeys(cage), 0.4);
  const dx = [1, 4, 7].map((vi) => out.vertices[vi][0] - cage.vertices[vi][0]);
  assert.ok(dx.every((d) => Math.abs(d) > 1e-9), 'every vertex genuinely moved');
  assert.ok(dx.every((d) => Math.sign(d) === Math.sign(dx[0])), `all three must move the same direction, got ${dx}`);
});

test('a negative t slides the other way', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const fwd = slideEdges(cage, midColumnKeys(cage), 0.4).cage;
  const back = slideEdges(cage, midColumnKeys(cage), -0.4).cage;
  const dFwd = fwd.vertices[4][0] - cage.vertices[4][0];
  const dBack = back.vertices[4][0] - cage.vertices[4][0];
  assert.ok(Math.abs(dFwd) > 1e-9 && Math.abs(dBack) > 1e-9);
  assert.equal(Math.sign(dFwd) === Math.sign(dBack), false, `forward and back must move opposite ways, got ${dFwd} and ${dBack}`);
});

test('t = 0 is an exact no-op', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const { cage: out } = slideEdges(cage, midColumnKeys(cage), 0);
  assert.deepEqual(out.vertices, cage.vertices);
});

test('the result still subdivides cleanly', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  const ctx = buildTopology(cage);
  const someInterior = [...ctx.edgeMap.keys()].slice(0, 2);
  const { cage: out } = slideEdges(cage, someInterior, 0.25);
  const refined = subdivideCatmullClark(out);
  for (const f of refined.faces) assert.equal(new Set(f).size, f.length);
  for (const v of refined.vertices) assert.ok(v.every(Number.isFinite));
});

test('honest refusals, and the input cage is never mutated', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const before = JSON.stringify(cage);
  assert.throws(() => slideEdges(cage, [], 0.3), /non-empty/);
  assert.throws(() => slideEdges(cage, ['999_1000'], 0.3), /not a real edge/);
  assert.throws(() => slideEdges(cage, midColumnKeys(cage), 1), /under 0.95/);
  assert.throws(() => slideEdges(cage, midColumnKeys(cage), Number.NaN), /finite/);
  slideEdges(cage, midColumnKeys(cage), 0.3);
  assert.equal(JSON.stringify(cage), before);
});
