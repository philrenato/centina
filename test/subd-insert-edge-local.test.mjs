import test from 'node:test';
import assert from 'node:assert/strict';
import { superbCylinderCage, superbTorusCage } from '../kernel/subdprimitives.mjs';
import { buildTopology, edgeKey, subdivideCatmullClark, superbDisplayMesh } from '../kernel/subd.mjs';
import { insertEdgeLoop, insertEdgeLocal, extrudeFaces, computeAverageNormal, recomputeInsertedLoopPositions } from '../kernel/subdedit.mjs';

// LOCAL EDGE INSERT — insertEdgeLoop with { local: true } (and its named
// sibling insertEdgeLocal): the same cut as the whole-loop insert, stopped
// after the seed face, leaving a T-junction at each end of the new edge.
//
// THE FIXTURE is a torus cage with one face extruded — genuinely curved,
// closed, and irregular (it carries valence-3 and valence-5 extraordinary
// vertices, so nothing here can pass by accident on a regular grid). Every
// number asserted about it below is MEASURED first, in the fixture test, and
// the rest of the file reads those measurements back out of the same builder.

function buildFixture() {
  const torus = superbTorusCage([0, 0, 0], 30, 10, 8, 6);
  return extrudeFaces(torus, [0], computeAverageNormal(torus, [0]), 12).cage;
}

function cageStats(cage) {
  const topo = buildTopology(cage);
  const faceSizes = {};
  for (const f of cage.faces) faceSizes[f.length] = (faceSizes[f.length] || 0) + 1;
  let naked = 0, nonManifold = 0;
  for (const [, e] of topo.edgeMap) {
    if (e.faces.length === 1) naked++;
    if (e.faces.length > 2) nonManifold++;
  }
  const valences = {};
  for (const list of topo.vertexEdges) valences[list.length] = (valences[list.length] || 0) + 1;
  const V = cage.vertices.length, E = topo.edgeMap.size, F = cage.faces.length;
  return { V, E, F, euler: V - E + F, faceSizes, naked, nonManifold, valences, topo };
}

// Everything "a valid cage" has to mean here, checked on the RESULT rather
// than predicted from the input: in-range indices, no vertex used twice
// within one face, consistent winding (a consistently wound manifold uses
// each DIRECTED edge at most once — a reused one is a face wound backwards
// against its neighbour), no edge left non-manifold, and acceptance by
// subdivideCatmullClark itself, which is what runs kernel/subd.mjs's own
// validateCage over the cage.
function assertValidCage(cage, label) {
  cage.faces.forEach((face, fi) => {
    assert.ok(face.length >= 3, `${label}: face ${fi} has only ${face.length} sides`);
    for (const vi of face) {
      assert.ok(Number.isInteger(vi) && vi >= 0 && vi < cage.vertices.length,
        `${label}: face ${fi} references out-of-range vertex ${vi}`);
    }
    assert.equal(new Set(face).size, face.length,
      `${label}: face ${fi} uses a vertex twice: ${JSON.stringify(face)}`);
  });
  const directed = new Map();
  cage.faces.forEach((face, fi) => {
    for (let c = 0; c < face.length; c++) {
      const key = `${face[c]}>${face[(c + 1) % face.length]}`;
      assert.ok(!directed.has(key),
        `${label}: directed edge ${key} is used by both face ${directed.get(key)} and face ${fi} — inconsistent winding`);
      directed.set(key, fi);
    }
  });
  const topo = buildTopology(cage);
  for (const [key, e] of topo.edgeMap) {
    assert.ok(e.faces.length <= 2, `${label}: edge ${key} is used by ${e.faces.length} faces`);
  }
  assert.doesNotThrow(() => subdivideCatmullClark(cage), `${label}: rejected by subdivideCatmullClark`);
}

// ---------------------------------------------------------------------------
// THE FIXTURE, MEASURED
// ---------------------------------------------------------------------------

test('local insert fixture: the extruded torus cage is curved, closed, all-quad and genuinely irregular — measured, not assumed', () => {
  const cage = buildFixture();
  const s = cageStats(cage);
  assert.deepEqual({ V: s.V, E: s.E, F: s.F }, { V: 52, E: 104, F: 52 });
  assert.equal(s.euler, 0, 'a torus cage has Euler characteristic 0');
  assert.deepEqual(s.faceSizes, { 4: 52 }, 'every face starts as a quad');
  assert.equal(s.naked, 0, 'closed: no naked edge');
  assert.equal(s.nonManifold, 0);
  // The extrusion is what makes this NOT a regular grid: a plain torus cage is
  // valence-4 everywhere, and a regular grid can hide orientation and
  // continuation bugs that only an extraordinary vertex exposes.
  assert.deepEqual(s.valences, { 3: 4, 4: 44, 5: 4 });
  // Curved in the sense that matters: the cage is not planar in any direction.
  for (let axis = 0; axis < 3; axis++) {
    const vals = cage.vertices.map((v) => v[axis]);
    assert.ok(Math.max(...vals) - Math.min(...vals) > 10, `cage is flat along axis ${axis}`);
  }
});

// ---------------------------------------------------------------------------
// THE CONSTRUCTION'S OWN PREDICTED COUNTS
// ---------------------------------------------------------------------------
//
// Derived before measuring. A local insert on a quad seed face whose two
// rungs are both interior:
//   VERTICES  +2 — one per rung, at its own interior
//   FACES     +1 — the seed quad is removed and two quads replace it
//   EDGES     +3 — each rung becomes two edges (+1 each), plus the new edge
//   EULER      0 — (V+2) - (E+3) + (F+1) = V - E + F, unchanged
//   FACE SIZES two quads become pentagons; nothing else changes size

test('local insert: exact vertex/face/edge deltas the construction predicts, and Euler unchanged', () => {
  const cage = buildFixture();
  const before = cageStats(cage);
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  const { cage: out } = insertEdgeLocal(cage, seedKey, 0.5, 0);
  const after = cageStats(out);

  assert.equal(after.V, before.V + 2, 'exactly two new vertices, one per split rung');
  assert.equal(after.F, before.F + 1, 'one quad removed, two added');
  assert.equal(after.E, before.E + 3, 'each rung splits in two (+2) and the new edge is +1');
  assert.equal(after.euler, before.euler, 'a refinement must not change the Euler characteristic');
  assert.deepEqual(after.faceSizes, { 4: before.faceSizes[4] - 1, 5: 2 },
    'the seed quad becomes two quads (net -1+2 = +1 quad, so 4-count goes 52 -> 51) and exactly two neighbours become pentagons');
  assert.equal(after.naked, 0, 'still closed');
  assert.equal(after.nonManifold, 0);
  assertValidCage(out, 'local insert');
});

test('local insert: the T-junction is exactly where the construction puts it — two pentagons, one new vertex each, each new vertex valence 3', () => {
  const cage = buildFixture();
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  const r = insertEdgeLocal(cage, seedKey, 0.5, 0);
  const topo = buildTopology(r.cage);

  assert.equal(r.insertedVertexIndices.length, 2);
  assert.equal(r.tJunctionFaceIndices.length, 2, 'one widened neighbour per split rung');
  assert.equal(r.splitFaceIndices.length, 2, 'the seed face is replaced by exactly two faces');

  const pentagons = r.tJunctionFaceIndices.map((i) => r.cage.faces[i]);
  for (const p of pentagons) assert.equal(p.length, 5, 'a widened quad neighbour is a 5-gon');
  // Every pentagon in the OUTPUT is one of the reported ones — the report is
  // complete, not merely correct about the faces it happens to name.
  const allPentagonIdx = r.cage.faces.map((f, i) => (f.length === 5 ? i : -1)).filter((i) => i >= 0);
  assert.deepEqual(allPentagonIdx.slice().sort((a, b) => a - b), r.tJunctionFaceIndices.slice().sort((a, b) => a - b));

  for (const nv of r.insertedVertexIndices) {
    // A T-vertex sits in the interior of a rung, so it has the rung's two
    // halves plus the one new edge = valence 3, and it is used by the two
    // replacement halves plus the single widened neighbour = 3 faces.
    assert.equal(topo.vertexEdges[nv].length, 3, `new vertex ${nv} should be valence 3`);
    assert.equal(topo.vertexFaces[nv].length, 3, `new vertex ${nv} should touch 3 faces`);
    const owners = pentagons.filter((p) => p.includes(nv));
    assert.equal(owners.length, 1, `new vertex ${nv} should lie on exactly one pentagon`);
  }
  // No pentagon may carry a vertex its own neighbour across that edge lacks:
  // that is the crack this repair exists to prevent, and it is checked
  // structurally by assertValidCage's directed-edge pass on the whole cage.
  assertValidCage(r.cage, 'T-junction cage');
});

test('local insert: the new vertices are the exact lerp of their own rung endpoints at t', () => {
  const cage = buildFixture();
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  for (const t of [0.25, 0.5, 0.8]) {
    const r = insertEdgeLocal(cage, seedKey, t, 0);
    r.insertedVertexIndices.forEach((vi, i) => {
      const [near, far] = r.rungPairs[i];
      const a = cage.vertices[near], b = cage.vertices[far];
      for (let k = 0; k < 3; k++) {
        assert.ok(Math.abs(r.cage.vertices[vi][k] - (a[k] + (b[k] - a[k]) * t)) < 1e-12,
          `t=${t}: new vertex ${vi} is not the exact lerp on its rung`);
      }
    });
  }
});

test('local insert: the position slider still works — recomputeInsertedLoopPositions moves a LOCAL insert with no topology change', () => {
  const cage = buildFixture();
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  const r = insertEdgeLocal(cage, seedKey, 0.5, 0);
  const moved = recomputeInsertedLoopPositions(r.cage, r.insertedVertexIndices, r.rungPairs, 0.2);
  assert.deepEqual(moved.faces, r.cage.faces, 'a reposition must not touch topology');
  assert.equal(moved.vertices.length, r.cage.vertices.length);
  r.insertedVertexIndices.forEach((vi, i) => {
    const [near, far] = r.rungPairs[i];
    const a = r.cage.vertices[near], b = r.cage.vertices[far];
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(moved.vertices[vi][k] - (a[k] + (b[k] - a[k]) * 0.2)) < 1e-12);
    }
  });
});

// ---------------------------------------------------------------------------
// LOCAL IS LOCAL, AND THE SEED EDGE'S OTHER FACE IS DELIBERATELY UNTOUCHED
// ---------------------------------------------------------------------------

test('local insert splits ONE face where the loop insert splits the whole strip, and leaves the seed edge itself intact', () => {
  const cage = buildFixture();
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  const loop = insertEdgeLoop(cage, seedKey, 0.5, 0);
  const local = insertEdgeLocal(cage, seedKey, 0.5, 0);

  assert.equal(local.cage.faces.length, cage.faces.length + 1, 'local adds one face');
  assert.ok(loop.cage.faces.length > local.cage.faces.length + 5,
    `the loop insert should refine the whole chain (it adds ${loop.cage.faces.length - cage.faces.length} faces) while local adds 1`);
  assert.equal(local.insertedVertexIndices.length, 2);
  assert.ok(loop.insertedVertexIndices.length > 2,
    `the loop insert should place a vertex on every rung it crosses (it placed ${loop.insertedVertexIndices.length})`);

  // The seed edge is reused VERBATIM by the near-half replacement quad, so it
  // still exists as an edge of the output.
  assert.ok(buildTopology(local.cage).edgeMap.has(seedKey), 'the seed edge itself is never split');
});

test('local insert: the face on the OTHER side of the seed edge is left byte-identical — nothing to repair, because the seed edge is not split', () => {
  const cage = buildFixture();
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  const topo = buildTopology(cage);
  const [faceA, faceB] = topo.edgeMap.get(seedKey).faces;
  assert.equal(typeof faceB, 'number', 'the fixture seed edge must be interior for this test to mean anything');

  const r = insertEdgeLocal(cage, seedKey, 0.5, 0); // side 0 -> faceA splits
  // faceB is not removed and not widened, so it survives unchanged somewhere
  // in the output; the strongest statement is that its exact loop is still
  // present as a face of the result.
  const wanted = JSON.stringify(cage.faces[faceB]);
  assert.ok(r.cage.faces.some((f) => JSON.stringify(f) === wanted),
    'the seed edge\'s other face must come through untouched');
  assert.ok(!r.cage.faces.some((f) => JSON.stringify(f) === JSON.stringify(cage.faces[faceA])),
    'the chosen side\'s face must be gone, replaced by its two halves');

  // side 1 is the way to reach the OTHER face: a second, independent local
  // cut, not a continuation of the first.
  const r1 = insertEdgeLocal(cage, seedKey, 0.5, 1);
  const wantedA = JSON.stringify(cage.faces[faceA]);
  assert.ok(r1.cage.faces.some((f) => JSON.stringify(f) === wantedA));
  assert.notDeepEqual(r.rungPairs, r1.rungPairs, 'the two sides split different rungs — they are different cuts');
});

// ---------------------------------------------------------------------------
// THE LIMIT SURFACE BARELY MOVES
// ---------------------------------------------------------------------------

function closestPointOnTriangleDistance(p, a, b, c) {
  const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return Math.hypot(ap[0], ap[1], ap[2]);
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return Math.hypot(bp[0], bp[1], bp[2]);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return Math.hypot(ap[0] - v * ab[0], ap[1] - v * ab[1], ap[2] - v * ab[2]);
  }
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return Math.hypot(cp[0], cp[1], cp[2]);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return Math.hypot(ap[0] - w * ac[0], ap[1] - w * ac[1], ap[2] - w * ac[2]);
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return Math.hypot(bp[0] - w * (c[0] - b[0]), bp[1] - w * (c[1] - b[1]), bp[2] - w * (c[2] - b[2]));
  }
  const den = 1 / (va + vb + vc), v = vb * den, w = vc * den;
  return Math.hypot(ap[0] - v * ab[0] - w * ac[0], ap[1] - v * ab[1] - w * ac[1], ap[2] - v * ab[2] - w * ac[2]);
}

function oneSidedHausdorff(meshA, meshB) {
  let worst = 0;
  for (const p of meshA.vertices) {
    let best = Infinity;
    for (const t of meshB.triangles) {
      const d = closestPointOnTriangleDistance(p, meshB.vertices[t[0]], meshB.vertices[t[1]], meshB.vertices[t[2]]);
      if (d < best) best = d;
    }
    if (best > worst) worst = best;
  }
  return worst;
}

function surfaceDeviation(cageA, cageB, level = 2) {
  const A = superbDisplayMesh(cageA, 'smooth', { level });
  const B = superbDisplayMesh(cageB, 'smooth', { level });
  return Math.max(oneSidedHausdorff(A, B), oneSidedHausdorff(B, A));
}

test('local insert is a refinement, not a deformation: the limit surface moves no more than the already-shipped LOOP insert moves it', () => {
  const cage = buildFixture();
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);

  const localDev = surfaceDeviation(cage, insertEdgeLocal(cage, seedKey, 0.5, 0).cage);
  // GROUND TRUTH, not a chosen tolerance: the whole-loop insert is the
  // shipped, already-accepted behaviour for this same seed edge and the same
  // t. A local cut touches strictly less of the cage, so it has no business
  // disturbing the limit surface by more than the loop cut does. Comparing
  // against that number instead of an invented epsilon means the bar moves
  // with the operation rather than with whatever looked small enough.
  const loopDev = surfaceDeviation(cage, insertEdgeLoop(cage, seedKey, 0.5, 0).cage);
  assert.ok(localDev <= loopDev * 1.1,
    `local deviation ${localDev.toFixed(6)} should not exceed the loop insert's own ${loopDev.toFixed(6)} by more than 10%`);

  // And an absolute sanity floor on the same measurement, expressed against
  // the cage's own size so it carries over to a rescaled fixture.
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const v of cage.vertices) for (let i = 0; i < 3; i++) { if (v[i] < lo[i]) lo[i] = v[i]; if (v[i] > hi[i]) hi[i] = v[i]; }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  assert.ok(localDev < 0.01 * diag,
    `local deviation ${localDev.toFixed(6)} should be well under 1% of the cage diagonal ${diag.toFixed(4)}`);
  // Non-vacuous: the operation really does change the surface, so a measured
  // zero here would mean the measurement, not the operation, is broken.
  assert.ok(localDev > 0, 'the cage changed, so some deviation is expected');
});

// ---------------------------------------------------------------------------
// A CAGE THAT ALREADY HAS A T-JUNCTION MUST SURVIVE THE NEXT INSERT
// ---------------------------------------------------------------------------

test('second local insert on a cage that already carries a T-junction: the pentagon widens again to a 6-gon holding BOTH new vertices', () => {
  const cage = buildFixture();
  const first = insertEdgeLocal(cage, edgeKey(cage.faces[20][0], cage.faces[20][1]), 0.5, 0);
  assertValidCage(first.cage, 'after first insert');

  // Choose the second seed by construction rather than by luck: a seed whose
  // own rung is a side of one of the pentagons is exactly the case that
  // re-widens an already-widened face.
  const pentagonFaces = first.tJunctionFaceIndices.map((i) => first.cage.faces[i]);
  const pentagonSides = new Set();
  for (const p of pentagonFaces) for (let c = 0; c < p.length; c++) pentagonSides.add(edgeKey(p[c], p[(c + 1) % p.length]));

  let chosen = null;
  const topo1 = buildTopology(first.cage);
  outer:
  for (const [k] of topo1.edgeMap) {
    for (const side of [0, 1]) {
      let r;
      try { r = insertEdgeLocal(first.cage, k, 0.5, side); } catch { continue; }
      if (r.rungPairs.some(([a, b]) => pentagonSides.has(edgeKey(a, b)))) { chosen = { k, side, r }; break outer; }
    }
  }
  assert.ok(chosen, 'the fixture must offer at least one second insert that cuts across an existing pentagon');

  const before = cageStats(first.cage);
  const after = cageStats(chosen.r.cage);
  assert.equal(after.V, before.V + 2);
  assert.equal(after.F, before.F + 1);
  assert.equal(after.E, before.E + 3);
  assert.equal(after.euler, before.euler, 'Euler still unchanged on the second insert');
  assert.equal(after.faceSizes[6], 1, 'the already-widened pentagon becomes a 6-gon');
  assert.equal(after.naked, 0);
  assert.equal(after.nonManifold, 0);
  assertValidCage(chosen.r.cage, 'after second insert');

  // The 6-gon carries BOTH T-vertices: the one from the first insert and the
  // one from the second. That is the specific thing a naive implementation
  // loses, by rebuilding the neighbour from the original quad and dropping
  // the vertex already spliced into it.
  const hex = chosen.r.cage.faces.find((f) => f.length === 6);
  const firstT = first.insertedVertexIndices.filter((v) => hex.includes(v));
  const secondT = chosen.r.insertedVertexIndices.filter((v) => hex.includes(v));
  assert.equal(firstT.length, 1, 'the 6-gon must still hold the FIRST insert\'s T-vertex');
  assert.equal(secondT.length, 1, 'the 6-gon must hold the SECOND insert\'s T-vertex');
});

test('third local insert on a doubly-T-junctioned cage stays valid for every seed it accepts, and refuses an n-gon seed honestly', () => {
  const cage = buildFixture();
  const c1 = insertEdgeLocal(cage, edgeKey(cage.faces[20][0], cage.faces[20][1]), 0.5, 0).cage;
  const topo1 = buildTopology(c1);
  let c2 = null;
  for (const [k] of topo1.edgeMap) {
    if (c2) break;
    for (const side of [0, 1]) {
      try { const r = insertEdgeLocal(c1, k, 0.5, side); if (r.cage.faces.some((f) => f.length === 6)) { c2 = r.cage; break; } } catch { /* n-gon seed */ }
    }
  }
  assert.ok(c2, 'a doubly-T-junctioned cage must be reachable');

  const topo2 = buildTopology(c2);
  const beforeStats = cageStats(c2);
  let accepted = 0, refusedForNgonSeed = 0;
  for (const [k] of topo2.edgeMap) {
    for (const side of [0, 1]) {
      let r;
      try { r = insertEdgeLocal(c2, k, 0.5, side); } catch (e) {
        assert.match(e.message, /is not a quad/, `unexpected refusal on ${k}/${side}: ${e.message}`);
        refusedForNgonSeed++;
        continue;
      }
      accepted++;
      const s = cageStats(r.cage);
      assert.equal(s.euler, beforeStats.euler, `${k}/${side}: Euler changed`);
      assert.equal(s.V, beforeStats.V + 2, `${k}/${side}: vertex delta`);
      assert.equal(s.F, beforeStats.F + 1, `${k}/${side}: face delta`);
      assert.equal(s.E, beforeStats.E + 3, `${k}/${side}: edge delta`);
      assert.equal(s.naked, 0);
      assert.equal(s.nonManifold, 0);
      assertValidCage(r.cage, `third insert ${k}/${side}`);
    }
  }
  assert.ok(accepted > 100, `expected the third insert to be accepted broadly, got ${accepted}`);
  // Every refusal is an n-gon seed face and nothing else: the pentagons and
  // the hexagon each refuse from every one of their own sides, and no quad
  // seed refuses at all.
  const ngonSideCount = c2.faces.filter((f) => f.length > 4).reduce((a, f) => a + f.length, 0);
  assert.equal(refusedForNgonSeed, ngonSideCount,
    'the only refusals are exactly the (edge, side) pairs whose seed face is an n-gon');
});

// ---------------------------------------------------------------------------
// A FULL SWEEP: EVERY EDGE, BOTH SIDES
// ---------------------------------------------------------------------------

test('local insert: every edge of the fixture, from both sides, yields a valid cage with the same predicted deltas — no seed is a special case', () => {
  const cage = buildFixture();
  const before = cageStats(cage);
  let count = 0;
  for (const [k] of before.topo.edgeMap) {
    for (const side of [0, 1]) {
      const r = insertEdgeLocal(cage, k, 0.5, side);
      const s = cageStats(r.cage);
      assert.equal(s.V, before.V + 2, `${k}/${side}`);
      assert.equal(s.E, before.E + 3, `${k}/${side}`);
      assert.equal(s.F, before.F + 1, `${k}/${side}`);
      assert.equal(s.euler, before.euler, `${k}/${side}`);
      assert.deepEqual(s.faceSizes, { 4: before.faceSizes[4] - 1, 5: 2 }, `${k}/${side}`);
      assert.equal(s.naked, 0, `${k}/${side}`);
      assert.equal(s.nonManifold, 0, `${k}/${side}`);
      assertValidCage(r.cage, `sweep ${k}/${side}`);
      count++;
    }
  }
  assert.equal(count, before.E * 2, 'every edge, both sides');
});

// ---------------------------------------------------------------------------
// THE WHOLE-LOOP PATH IS UNCHANGED
// ---------------------------------------------------------------------------

test('whole-loop insert is unchanged: a frozen hand-checkable result on the facets=6 cylinder at t=0.25', () => {
  const cyl = superbCylinderCage([0, 0, 0], 25, 50, 6);
  const r = insertEdgeLoop(cyl, '0_1', 0.25, 0);
  assert.deepEqual(r.cage.faces, [
    [5, 4, 3, 2, 1, 0], [6, 7, 8, 9, 10, 11],
    [0, 1, 13, 12], [12, 13, 7, 6],
    [5, 0, 12, 14], [14, 12, 6, 11],
    [1, 2, 15, 13], [13, 15, 8, 7],
    [4, 5, 14, 16], [16, 14, 11, 10],
    [2, 3, 17, 15], [15, 17, 9, 8],
    [3, 4, 16, 17], [17, 16, 10, 9],
  ]);
  assert.deepEqual(r.insertedVertexIndices, [12, 13, 14, 15, 16, 17]);
  assert.deepEqual(r.rungPairs, [[0, 6], [1, 7], [5, 11], [2, 8], [4, 10], [3, 9]]);
  // Every inserted vertex sits at exactly a quarter of the 0..50 height, on
  // the same circle as the ring it came from.
  for (const vi of r.insertedVertexIndices) {
    assert.ok(Math.abs(r.cage.vertices[vi][2] - 12.5) < 1e-12);
    assert.ok(Math.abs(Math.hypot(r.cage.vertices[vi][0], r.cage.vertices[vi][1]) - 25) < 1e-9);
  }
});

test('whole-loop insert never reaches the T-junction repair: it leaves no widened face on any seed of the fixture', () => {
  const cage = buildFixture();
  const topo = buildTopology(cage);
  let checked = 0;
  for (const [k] of topo.edgeMap) {
    for (const side of [0, 1]) {
      const r = insertEdgeLoop(cage, k, 0.5, side);
      assert.equal(r.tJunctionFaceIndices.length, 0, `${k}/${side}: a whole-loop insert must leave no T-junction`);
      assert.ok(r.cage.faces.every((f) => f.length === 4), `${k}/${side}: a whole-loop insert on an all-quad cage stays all-quad`);
      checked++;
    }
  }
  assert.equal(checked, topo.edgeMap.size * 2);
});

test('the local option is genuinely opt-in: omitted, {}, and {local:false} all produce the identical whole-loop result', () => {
  const cage = buildFixture();
  const topo = buildTopology(cage);
  for (const [k] of topo.edgeMap) {
    const a = insertEdgeLoop(cage, k, 0.5, 0);
    const b = insertEdgeLoop(cage, k, 0.5, 0, {});
    const c = insertEdgeLoop(cage, k, 0.5, 0, { local: false });
    assert.deepEqual(b.cage, a.cage, `${k}: {} changed the result`);
    assert.deepEqual(c.cage, a.cage, `${k}: {local:false} changed the result`);
    assert.deepEqual(b.rungPairs, a.rungPairs);
    assert.deepEqual(c.insertedVertexIndices, a.insertedVertexIndices);
  }
});

test('insertEdgeLocal is exactly insertEdgeLoop with { local: true } — no second implementation to drift', () => {
  const cage = buildFixture();
  const topo = buildTopology(cage);
  for (const [k] of topo.edgeMap) {
    for (const side of [0, 1]) {
      assert.deepEqual(insertEdgeLocal(cage, k, 0.4, side), insertEdgeLoop(cage, k, 0.4, side, { local: true }), `${k}/${side}`);
    }
  }
});

test('local insert keeps every guard the loop insert has: t range, side range, unknown seed, and a non-quad seed face', () => {
  const cage = buildFixture();
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  assert.throws(() => insertEdgeLocal(cage, seedKey, 0, 0), /strictly between 0 and 1/);
  assert.throws(() => insertEdgeLocal(cage, seedKey, 1, 0), /strictly between 0 and 1/);
  assert.throws(() => insertEdgeLocal(cage, '9999_9998', 0.5, 0), /not a real edge/);
  assert.throws(() => insertEdgeLoop(cage, seedKey, 0.5, 2, { local: true }), /side must be 0 or 1/);

  // A pentagon seed face: honestly refused, with a message that says why a
  // LOCAL insert needs a quad rather than repeating the strip wording.
  const withT = insertEdgeLocal(cage, seedKey, 0.5, 0);
  const pentagon = withT.cage.faces[withT.tJunctionFaceIndices[0]];
  const pentEdge = edgeKey(pentagon[0], pentagon[1]);
  const sides = buildTopology(withT.cage).edgeMap.get(pentEdge).faces;
  const sideAtPentagon = sides.indexOf(withT.tJunctionFaceIndices[0]);
  assert.ok(sideAtPentagon >= 0);
  assert.throws(() => insertEdgeLocal(withT.cage, pentEdge, 0.5, sideAtPentagon),
    /is not a quad .* a local edge insert needs a well-defined opposite edge/);
});

test('local insert transfers a crease on a split rung onto BOTH halves, and leaves the untouched near/far creases alone', () => {
  const cage = buildFixture();
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  const probe = insertEdgeLocal(cage, seedKey, 0.5, 0);
  const [rn, rf] = probe.rungPairs[0];
  const rungKey = edgeKey(rn, rf);

  const creased = { ...cage, creases: { [rungKey]: 3, [seedKey]: 2 } };
  const r = insertEdgeLocal(creased, seedKey, 0.5, 0);
  const mid = r.insertedVertexIndices[0];
  assert.equal(r.cage.creases[rungKey], undefined, 'the split rung no longer exists as an edge, so its key must not dangle');
  assert.equal(r.cage.creases[edgeKey(rn, mid)], 3);
  assert.equal(r.cage.creases[edgeKey(mid, rf)], 3);
  assert.equal(r.cage.creases[seedKey], 2, 'the seed edge is reused verbatim, so its crease survives untouched');
  assert.ok(buildTopology(r.cage).edgeMap.has(edgeKey(rn, mid)));
  assert.ok(buildTopology(r.cage).edgeMap.has(edgeKey(mid, rf)));
});

test('local insert never mutates the input cage', () => {
  const cage = buildFixture();
  const snapshot = JSON.stringify(cage);
  const seedKey = edgeKey(cage.faces[20][0], cage.faces[20][1]);
  const r = insertEdgeLocal(cage, seedKey, 0.5, 0);
  assert.equal(JSON.stringify(cage), snapshot);
  // and the output must not alias the input's own face arrays
  r.cage.faces[0].push(0);
  assert.equal(JSON.stringify(cage), snapshot);
});
