// MatchSubD on short runs. An open uniform cubic spans its second vertex to its
// second-from-last, so an open run needs four vertices to carry a segment; a
// closed run wraps and needs three. Every run runStations accepts must evaluate.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildTopology, edgeKey } from '../kernel/subd.mjs';
import { boundaryRunFromEdges, runStations, runPolygons, uniformCubicAt, uniformCubicDomain } from '../kernel/subdmatch.mjs';

// A grid cage `cols` vertices wide and two rows deep; a run along all of row 0
// ends on corners, so every station has one edge across the boundary.
function strip(cols) {
  const vertices = [], faces = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < cols; j++) vertices.push([j * 10, i * 10, 0]);
  for (let j = 0; j + 1 < cols; j++) faces.push([j, j + 1, cols + j + 1, cols + j]);
  return { vertices, faces, creases: {} };
}

function stationsFor(cage, keys) {
  const ctx = buildTopology(cage);
  const r = boundaryRunFromEdges(cage, keys, ctx);
  assert.ok(r.ok, r.reason);
  return { r, st: runStations(cage, r.run, r.closed, ctx) };
}

test('an open run of three vertices is refused by name', () => {
  const cage = strip(3);
  const { st } = stationsFor(cage, [edgeKey(0, 1), edgeKey(1, 2)]);
  assert.equal(st.ok, false);
  assert.match(st.reason, /at least 4 boundary vertices in an open run, and this one has 3/);
});

test('an open run of four vertices is accepted and evaluates across its whole domain', () => {
  const cage = strip(4);
  const { r, st } = stationsFor(cage, [edgeKey(0, 1), edgeKey(1, 2), edgeKey(2, 3)]);
  assert.ok(st.ok, st.reason);
  const { P } = runPolygons(cage, st.stations);
  const [lo, hi] = uniformCubicDomain(P.length, r.closed);
  for (let i = 0; i <= 8; i++) {
    const p = uniformCubicAt(P, r.closed, lo + (hi - lo) * (i / 8), 0);
    assert.ok(p.every(Number.isFinite), `finite at step ${i}`);
  }
});

test('a closed run of three vertices is still accepted', () => {
  // A triangle fan's rim: three boundary vertices, each with one spoke inward.
  const vertices = [[0, 0, 0], [10, 0, 0], [5, 9, 0], [5, 3, 0], [-6, -4, 0], [16, -4, 0], [5, 16, 0]];
  const faces = [[4, 5, 1, 0], [5, 6, 2, 1], [6, 4, 0, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
  const cage = { vertices, faces, creases: {} };
  const ctx = buildTopology(cage);
  const st = runStations(cage, [4, 5, 6], true, ctx);
  assert.ok(st.ok, st.reason);
  const { P } = runPolygons(cage, st.stations);
  const [lo, hi] = uniformCubicDomain(P.length, true);
  for (let i = 0; i <= 6; i++) assert.ok(uniformCubicAt(P, true, lo + (hi - lo) * (i / 6), 0).every(Number.isFinite));
});
