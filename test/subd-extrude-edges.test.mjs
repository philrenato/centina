// EXTRUDING AN OPEN EDGE. The claim worth testing is not "new faces appear"
// — that is bookkeeping. It is that the result is still one manifold
// surface: a chain of selected edges has to grow ONE strip sharing its
// rungs, wound consistently with the faces already there, and an edge with
// no free side has to be refused rather than silently torn.
import test from 'node:test';
import assert from 'node:assert/strict';
import { extrudeEdges } from '../kernel/subdedit.mjs';
import { buildTopology, edgeKey, subdivideCatmullClark } from '../kernel/subd.mjs';
import { superbPlaneCage, superbBoxCage } from '../kernel/subdprimitives.mjs';

function manifoldReport(cage) {
  const ctx = buildTopology(cage);
  let boundary = 0, nonManifold = 0;
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length === 1) boundary++;
    else if (e.faces.length > 2) nonManifold++;
  }
  return { boundary, nonManifold };
}
// Does every pair of faces sharing an edge traverse it in OPPOSITE
// directions? That is the definition of a consistently oriented surface —
// the thing a wrongly-wound new face breaks, invisibly, until the next
// subdivision or normal calculation.
function windingConsistent(cage) {
  const seen = new Map();
  for (const f of cage.faces) {
    for (let i = 0; i < f.length; i++) {
      const a = f[i], b = f[(i + 1) % f.length];
      const k = `${a}>${b}`;
      if (seen.has(k)) return false; // same directed edge used twice => two faces agree, i.e. inconsistent
      seen.set(k, true);
    }
  }
  return true;
}
function nakedEdgesOf(cage) {
  const ctx = buildTopology(cage);
  return [...ctx.edgeMap.values()].filter((e) => e.faces.length === 1).map((e) => edgeKey(e.v0, e.v1));
}

test('one naked edge grows exactly one face and two vertices, wound with its own neighbour', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 1);
  const naked = nakedEdgesOf(cage);
  assert.equal(naked.length, 4);
  const r = extrudeEdges(cage, [naked[0]], [0, 0, 1], 10);
  assert.equal(r.cage.faces.length, cage.faces.length + 1);
  assert.equal(r.cage.vertices.length, cage.vertices.length + 2);
  assert.equal(r.newFaceIndices.length, 1);
  assert.equal(windingConsistent(r.cage), true);
  assert.deepEqual(manifoldReport(r.cage), { boundary: 6, nonManifold: 0 });
});

test('THE POINT: a chain of naked edges grows ONE strip — the shared endpoint is duplicated once, not twice', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2); // 8 naked edges around the rim
  const ctx = buildTopology(cage);
  const naked = [...ctx.edgeMap.values()].filter((e) => e.faces.length === 1);
  // Two naked edges sharing a vertex.
  let pair = null;
  for (const a of naked) for (const b of naked) {
    if (a === b) continue;
    if (a.v0 === b.v0 || a.v0 === b.v1 || a.v1 === b.v0 || a.v1 === b.v1) { pair = [a, b]; break; }
    if (pair) break;
  }
  assert.ok(pair);
  const keys = pair.map((e) => edgeKey(e.v0, e.v1));
  const r = extrudeEdges(cage, keys, [0, 0, 1], 10);
  assert.equal(r.cage.faces.length, cage.faces.length + 2);
  // Three new vertices, not four: the shared endpoint's copy is reused. Two
  // separate quads would leave the strip split down its own rung, which
  // reads as one surface and subdivides as two.
  assert.equal(r.cage.vertices.length, cage.vertices.length + 3);
  assert.equal(r.newVertexIndices.length, 3);
  assert.equal(windingConsistent(r.cage), true);
  assert.equal(manifoldReport(r.cage).nonManifold, 0);
});

test('extruding a whole closed boundary loop makes a collar, still closed and still manifold', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const naked = nakedEdgesOf(cage);
  assert.equal(naked.length, 8);
  const r = extrudeEdges(cage, naked, [0, 0, 1], 10);
  assert.equal(r.cage.faces.length, cage.faces.length + 8);
  assert.equal(r.cage.vertices.length, cage.vertices.length + 8); // one per rim vertex, each shared by two new faces
  assert.equal(windingConsistent(r.cage), true);
  const after = manifoldReport(r.cage);
  assert.equal(after.nonManifold, 0);
  assert.equal(after.boundary, 8); // the collar's own new rim, and only that
});

test('and the grown cage really subdivides afterwards — the actual downstream risk', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const { cage: out } = extrudeEdges(cage, nakedEdgesOf(cage), [0, 0, 1], 10);
  const refined = subdivideCatmullClark(out);
  for (const f of refined.faces) assert.equal(new Set(f).size, f.length);
  for (const v of refined.vertices) assert.ok(v.every(Number.isFinite));
  assert.equal(windingConsistent(refined), true);
});

test('an INTERIOR edge is refused by name — it has no free side to grow into', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1); // closed: every edge interior
  const ctx = buildTopology(cage);
  const anyEdge = edgeKey([...ctx.edgeMap.values()][0].v0, [...ctx.edgeMap.values()][0].v1);
  assert.throws(() => extrudeEdges(cage, [anyEdge], [0, 0, 1], 5), /INTERIOR edge/);
});

test('honest refusals, and the input cage is never mutated', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 1);
  const before = JSON.stringify(cage);
  const naked = nakedEdgesOf(cage);
  assert.throws(() => extrudeEdges(cage, [], [0, 0, 1], 10), /non-empty/);
  assert.throws(() => extrudeEdges(cage, ['999_1000'], [0, 0, 1], 10), /not a real edge/);
  assert.throws(() => extrudeEdges(cage, [naked[0]], [0, 0, 1], 0), /nonzero finite/);
  assert.throws(() => extrudeEdges(cage, [naked[0]], [0, 0, 0], 10), /nonzero vector/);
  // Parallel to the edge itself: the new face would be four collinear points.
  const e = buildTopology(cage).edgeMap.get(naked[0]);
  const along = [cage.vertices[e.v1][0] - cage.vertices[e.v0][0], cage.vertices[e.v1][1] - cage.vertices[e.v0][1], cage.vertices[e.v1][2] - cage.vertices[e.v0][2]];
  assert.throws(() => extrudeEdges(cage, [naked[0]], along, 10), /exactly parallel/);
  assert.equal(JSON.stringify(cage), before);
});

test('THE DEFAULT DIRECTION: a whole boundary loop with no direction given grows OUTWARD in the sheet, each edge its own way — one shared vector could not do this', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const naked = nakedEdgesOf(cage);
  const r = extrudeEdges(cage, naked, null, 10);
  assert.equal(r.cage.faces.length, cage.faces.length + 8);
  assert.equal(r.cage.vertices.length, cage.vertices.length + 8);
  assert.equal(windingConsistent(r.cage), true);
  // Every new vertex is further from the centre than the rim it came from,
  // and still in the sheet's own plane (z unchanged) — that is what "grew
  // the sheet" means, as opposed to folding a wall up out of it.
  for (const vi of r.newVertexIndices) {
    const v = r.cage.vertices[vi];
    assert.ok(Math.hypot(v[0], v[1]) > 20.0001, `new vertex ${vi} should sit outside the original 40mm plane`);
    assert.ok(Math.abs(v[2]) < 1e-9, `new vertex ${vi} should stay in the sheet's own plane`);
  }
});

test('a vertex whose own selected edges point exactly opposite is refused rather than collapsed', () => {
  // Two colinear naked edges meeting at a shared vertex, each owned by a face
  // on the OPPOSITE side — their outward directions cancel exactly there.
  const cage = {
    vertices: [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [2, -1, 0]],
    faces: [[0, 1, 4, 3], [1, 2, 6, 5]],
    creases: {},
  };
  assert.throws(() => extrudeEdges(cage, [edgeKey(0, 1), edgeKey(1, 2)], null, 1), /no outward|opposite/);
});
