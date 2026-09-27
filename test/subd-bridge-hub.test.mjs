// N-way bridge — a Y, a cross, or any N ≥ 3 runs joined through one hub.
//
// The claim under test is topological, so it is checked topologically: the
// result has to be a consistent 2-manifold, every input run has to
// stop being naked, and the hub has to be one n-gon of the right size —
// not merely "some faces were added".
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bridgeEdgeRunsHub, bridgeEdgeRuns } from '../kernel/subdedit.mjs';
import { buildTopology, edgeKey, subdivideCatmullClark } from '../kernel/subd.mjs';

// A fixture built to be awkward rather than convenient: N arms as
// separate one-quad flaps standing around a center, each contributing one
// naked run, deliberately not coplanar (every arm is tilted out of the hub
// plane) so nothing here can pass by lying flat.
function flapRing(n, m = 2, tilt = 0.1) {
  const vertices = [], faces = [], runs = [];
  for (let i = 0; i < n; i++) {
    const mi = Array.isArray(m) ? m[i] : m;
    const a = (i / n) * Math.PI * 2;
    const ux = Math.cos(a), uy = Math.sin(a);
    // Each flap is a quad: an inner edge (the run, facing the hub) and an
    // outer edge further out. The run is the inner pair.
    const base = vertices.length;
    const run = [];
    for (let j = 0; j < mi; j++) {
      const s = (j - (mi - 1) / 2) * 0.6;
      // inner row — this is the run
      vertices.push([ux * 2 - uy * s, uy * 2 + ux * s, tilt * (i % 2 ? 1 : -1)]);
      run.push(base + j);
    }
    for (let j = 0; j < mi; j++) {
      const s = (j - (mi - 1) / 2) * 0.6;
      vertices.push([ux * 4 - uy * s, uy * 4 + ux * s, tilt * (i % 2 ? 1 : -1)]);
    }
    for (let j = 0; j + 1 < mi; j++) {
      faces.push([base + j, base + j + 1, base + mi + j + 1, base + mi + j]);
    }
    runs.push(run);
  }
  return { cage: { vertices, faces, creases: {} }, runs };
}

function nakedKeys(cage) {
  const t = buildTopology(cage);
  return [...t.edgeMap.entries()].filter(([, e]) => e.faces.length === 1).map(([k]) => k);
}
function directedReuse(faces) {
  const seen = new Set();
  let reuse = 0;
  for (const f of faces) for (let i = 0; i < f.length; i++) {
    const k = `${f[i]}>${f[(i + 1) % f.length]}`;
    if (seen.has(k)) reuse++;
    seen.add(k);
  }
  return reuse;
}

test('bridgeEdgeRunsHub: three runs join through one hub, and the result is a consistent 2-manifold', () => {
  const { cage, runs } = flapRing(3, 2);
  const before = nakedKeys(cage).length;
  const out = bridgeEdgeRunsHub(cage, runs, 1);
  // Every run edge was naked and is now shared by two faces — the arms
  // attached rather than floating alongside.
  const t = buildTopology(out.cage);
  for (const run of runs) for (let j = 0; j + 1 < run.length; j++) {
    const e = t.edgeMap.get(edgeKey(run[j], run[j + 1]));
    assert.ok(e, 'the run edge still exists');
    assert.equal(e.faces.length, 2, 'each bridged run edge is now interior, not naked');
  }
  assert.equal(directedReuse(out.cage.faces), 0, 'no directed edge is traversed twice — consistently wound');
  // A junction is a patch, not a cap: it closes the runs it was given and
  // opens its own free sides between neighboring arms, two per arm per
  // segment row. Asserted as exact accounting rather than as a direction,
  // because "fewer naked edges" is the wrong expectation here — and
  // an approximate one would hide an arm that failed to attach.
  const runEdges = runs.reduce((n2, r) => n2 + r.length - 1, 0);
  assert.equal(nakedKeys(out.cage).length, before - runEdges + 2 * 3 * 1,
    'exactly the run edges closed, and exactly two free sides opened per arm');
  assert.equal(out.armCount, 3);
});

test('bridgeEdgeRunsHub: the hub is one n-gon whose size is the ring, N(m-1)', () => {
  for (const [n, m] of [[3, 2], [4, 2], [5, 2], [3, 3], [4, 3], [6, 4]]) {
    const { cage, runs } = flapRing(n, m);
    const out = bridgeEdgeRunsHub(cage, runs, 1);
    const hub = out.cage.faces[out.hubFaceIndex];
    assert.equal(hub.length, n * (m - 1), `N=${n} m=${m}: the hub n-gon has one vertex per ring step`);
    assert.equal(new Set(hub).size, hub.length, `N=${n} m=${m}: the hub visits each ring vertex once`);
    assert.equal(out.ringLength, n * (m - 1));
    assert.equal(directedReuse(out.cage.faces), 0, `N=${n} m=${m}: consistently wound`);
  }
});

test('bridgeEdgeRunsHub: a cross (4) and a 5-way are as legal as a Y, and each arm gets its own quads', () => {
  for (const n of [3, 4, 5, 6]) {
    const { cage, runs } = flapRing(n, 3);
    const segments = 2;
    const out = bridgeEdgeRunsHub(cage, runs, segments);
    // Per arm: (m-1) quads per row x segments rows. Plus exactly one hub.
    const expectedArmFaces = n * (3 - 1) * segments;
    assert.equal(out.bridgeFaceIndices.length, expectedArmFaces + 1, `N=${n}: ${expectedArmFaces} arm quads + 1 hub`);
    const arms = out.bridgeFaceIndices.slice(0, -1).map((fi) => out.cage.faces[fi]);
    assert.ok(arms.every((f) => f.length === 4), `N=${n}: every arm face is a quad`);
    assert.equal(directedReuse(out.cage.faces), 0);
  }
});

test('bridgeEdgeRunsHub: the result subdivides cleanly through the Catmull-Clark kernel', () => {
  const { cage, runs } = flapRing(3, 3);
  const out = bridgeEdgeRunsHub(cage, runs, 2);
  const refined = subdivideCatmullClark(out.cage);
  assert.ok(refined.vertices.every((p) => p.every(Number.isFinite)), 'no NaN or Infinity anywhere after refinement');
  // Catmull-Clark emits one quad per corner of every face, whatever its size
  // — which is why an n-gon hub needs no special handling.
  const corners = out.cage.faces.reduce((n2, f) => n2 + f.length, 0);
  assert.equal(refined.faces.length, corners, 'one refined quad per cage corner, n-gon hub included');
});

test('bridgeEdgeRunsHub: segments adds interior rings without touching the run or the hub', () => {
  const { cage, runs } = flapRing(4, 2);
  const one = bridgeEdgeRunsHub(cage, runs, 1);
  const three = bridgeEdgeRunsHub(cage, runs, 3);
  // Two extra rows per arm, each of m vertices, and the hub ring is unchanged.
  assert.equal(three.cage.vertices.length - one.cage.vertices.length, 4 * 2 * 2, 'two interior rows of 2 vertices on each of 4 arms');
  assert.equal(three.ringLength, one.ringLength, 'the hub ring itself does not grow with segments');
  assert.equal(three.cage.faces[three.hubFaceIndex].length, one.cage.faces[one.hubFaceIndex].length);
  assert.equal(directedReuse(three.cage.faces), 0);
});

test('bridgeEdgeRunsHub: unequal 1/2/3-edge arms keep their own counts and meet in one six-sided hub', () => {
  const counts = [2, 3, 4];
  const { cage, runs } = flapRing(3, counts);
  const originalVertices = cage.vertices.map((v) => [...v]);
  const beforeNaked = nakedKeys(cage).length;
  const segments = 3;
  const out = bridgeEdgeRunsHub(cage, runs, segments, 0.35, 1.5);
  const ringLength = counts.reduce((n, m) => n + m - 1, 0);
  assert.equal(out.ringLength, ringLength);
  assert.equal(out.cage.faces[out.hubFaceIndex].length, ringLength, 'one central n-gon absorbs the different arm counts');
  assert.equal(new Set(out.cage.faces[out.hubFaceIndex]).size, ringLength, 'the hub ring contains no repeated vertex');
  assert.equal(out.bridgeFaceIndices.length, segments * ringLength + 1, 'one quad per original arm edge per segment, plus one hub');
  assert.ok(out.bridgeFaceIndices.slice(0, -1).every((fi) => out.cage.faces[fi].length === 4), 'every arm face remains a quad');
  assert.deepEqual(out.cage.vertices.slice(0, cage.vertices.length), originalVertices, 'no existing run vertex is moved or resampled');
  assert.equal(directedReuse(out.cage.faces), 0, 'the unequal hub is consistently wound');
  assert.equal(nakedKeys(out.cage).length, beforeNaked - ringLength + 2 * runs.length * segments, 'exact naked-edge accounting');
  const topo = buildTopology(out.cage);
  for (const run of runs) for (let j = 0; j + 1 < run.length; j++) assert.equal(topo.edgeMap.get(edgeKey(run[j], run[j + 1])).faces.length, 2, 'every source run edge becomes interior');
  for (const run of runs) for (let j = 0; j + 1 < run.length; j++) assert.equal(out.cage.creases[edgeKey(run[j], run[j + 1])], 1.5, 'crease is applied only along the original run');
  const refined = subdivideCatmullClark(out.cage);
  assert.ok(refined.vertices.every((p) => p.every(Number.isFinite)), 'the unequal hub subdivides without NaN or Infinity');
});

test('bridgeEdgeRunsHub: straightness 1 is a plain lerp, and 0 bends the arms', () => {
  const { cage, runs } = flapRing(3, 2);
  const straight = bridgeEdgeRunsHub(cage, runs, 4, 1);
  const bent = bridgeEdgeRunsHub(cage, runs, 4, 0);
  let maxDelta = 0;
  for (let i = cage.vertices.length; i < straight.cage.vertices.length; i++) {
    const a = straight.cage.vertices[i], b = bent.cage.vertices[i];
    maxDelta = Math.max(maxDelta, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  }
  assert.ok(maxDelta > 1e-3, `straightness 0 moves the interior rows somewhere else entirely (max delta ${maxDelta})`);
});

test('bridgeEdgeRunsHub: refuses by name for every case it cannot do', () => {
  const { cage, runs } = flapRing(3, 2);
  assert.throws(() => bridgeEdgeRunsHub(cage, runs.slice(0, 2), 1), /at least 3 edge runs/);
  const uneven = [runs[0], runs[1], [runs[2][0]]];
  assert.throws(() => bridgeEdgeRunsHub(cage, uneven, 1), /at least 2 vertices|different vertex counts/);
  const overlapping = [runs[0], runs[1], [runs[0][0], runs[0][1]]];
  assert.throws(() => bridgeEdgeRunsHub(cage, overlapping, 1), /share vertex/);
  assert.throws(() => bridgeEdgeRunsHub(cage, runs, 0), /positive integer/);
  assert.throws(() => bridgeEdgeRunsHub(cage, runs, 1, 5), /straightness must be between 0 and 1/);
  // An interior (already 2-faced) edge has no opening to bridge from.
  const closed = flapRing(3, 2);
  closed.cage.faces.push([closed.runs[0][0], closed.runs[0][1], closed.runs[1][1], closed.runs[1][0]]);
  assert.throws(() => bridgeEdgeRunsHub(closed.cage, closed.runs, 1), /not a naked \(open\) edge/);
  const spatial = flapRing(4, 2, 0.8);
  assert.throws(() => bridgeEdgeRunsHub(spatial.cage, spatial.runs, 1), /off one plane.*bridge the faces/);
});

test('bridgeEdgeRunsHub: two runs is still the ordinary bridge, and says so', () => {
  const { cage, runs } = flapRing(3, 2);
  assert.throws(() => bridgeEdgeRunsHub(cage, [runs[0], runs[1]], 1), /use bridgeEdgeRuns/);
  // And that ordinary path still works on the same fixture, unchanged.
  const two = bridgeEdgeRuns(cage, runs[0], runs[1], 1);
  assert.equal(directedReuse(two.cage.faces), 0);
});

// Order independence. The runs arrive as an unordered set — a selection walk
// hands them over in whatever order it found them, in whatever direction it
// walked each chain. Two derivations inside this function must not depend on
// that order, and both are reachable from an ordinary selection:
//
//  - The junction plane: a sum of signed cross products over array-ordered
//    pairs changes sign when two runs swap; for a symmetric junction (N arcs
//    evenly spaced around one rim) the terms cancel to zero, and a coplanar,
//    well-spread set of runs would be refused as "collinear".
//  - Run direction is chosen by which endpoint lies nearer the next run, which
//    is an exact tie on a symmetric junction, so a strict `<` falls through to
//    "keep" and direction would come from input order.
//
// A square rim is the fixture because it is symmetric; the tilted flaps above
// never tie.
function squareRimRuns(n) {
  // A flat 3x3 quad grid — 16 vertices, 9 faces, a 12-edge rim.
  const vertices = [], faces = [];
  const idx = (i, j) => j * 4 + i;
  for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) vertices.push([(i - 1.5) * 20, (j - 1.5) * 20, 0]);
  for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) faces.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
  const cage = { vertices, faces, creases: {} };
  // Walk the rim so the arcs can be spread evenly around it.
  const topo = buildTopology(cage);
  const nakedOf = (vi) => [...topo.edgeMap.entries()].filter(([, e]) => e.faces.length === 1).map(([k]) => k.split('_').map(Number)).filter((p) => p.includes(vi)).map((p) => (p[0] === vi ? p[1] : p[0]));
  const rim = [idx(0, 0)];
  while (rim.length < 12) {
    const next = nakedOf(rim[rim.length - 1]).find((vi) => !rim.includes(vi));
    if (next === undefined) break;
    rim.push(next);
  }
  const step = Math.floor(rim.length / n);
  return { cage, runs: Array.from({ length: n }, (_, k) => [rim[(k * step) % rim.length], rim[(k * step + 1) % rim.length]]) };
}
function permutations(a) {
  return a.length <= 1 ? [a] : a.flatMap((x, i) => permutations([...a.slice(0, i), ...a.slice(i + 1)]).map((p) => [x, ...p]));
}

test('bridgeEdgeRunsHub: unequal arms succeed for every selection order and direction', () => {
  const { cage, runs } = flapRing(3, [2, 3, 4]);
  let tried = 0;
  for (const perm of permutations([0, 1, 2])) {
    for (let bits = 0; bits < 8; bits++) {
      const variant = perm.map((i) => ((bits >> i) & 1 ? [...runs[i]].reverse() : [...runs[i]]));
      const out = bridgeEdgeRunsHub(cage, variant, 3, 0.35, 1.5);
      tried++;
      assert.equal(out.ringLength, 6);
      assert.equal(out.bridgeFaceIndices.length, 19, '18 arm quads plus the six-sided hub');
      assert.equal(directedReuse(out.cage.faces), 0, 'consistently wound in every order');
      assert.equal(out.cage.faces[out.hubFaceIndex].length, 6);
      assert.equal(new Set(out.cage.faces[out.hubFaceIndex]).size, 6);
      const used = new Set(out.cage.faces.flat());
      assert.equal(used.size, out.cage.vertices.length, 'no rejected trial leaves orphaned vertices');
      const topo = buildTopology(out.cage);
      for (const run of runs) for (let j = 0; j + 1 < run.length; j++) {
        assert.equal(topo.edgeMap.get(edgeKey(run[j], run[j + 1])).faces.length, 2, 'every original run edge closes');
      }
    }
  }
  assert.equal(tried, 48, 'all 3! orders × 2³ directions were exercised');
});

for (const n of [3, 4]) {
  test(`bridgeEdgeRunsHub: a symmetric ${n}-arc junction succeeds for every run order and direction`, () => {
    const { cage, runs } = squareRimRuns(n);
    assert.equal(new Set(runs.flat()).size, 2 * n, 'the fixture arcs are disjoint');
    let tried = 0;
    for (const perm of permutations(runs.map((_, i) => i))) {
      for (let bits = 0; bits < (1 << n); bits++) {
        const variant = perm.map((i) => ((bits >> i) & 1 ? [...runs[i]].reverse() : [...runs[i]]));
        tried++;
        const out = bridgeEdgeRunsHub(cage, variant, 2, 0);
        // Not merely "it did not throw": every variant has to produce the same
        // shape of junction, consistently wound, rather than a silently
        // different result.
        assert.equal(directedReuse(out.cage.faces), 0, 'consistently wound');
        assert.equal(out.armCount, n);
        assert.equal(out.ringLength, n * (2 - 1));
        assert.equal(out.cage.faces[out.hubFaceIndex].length, n * (2 - 1), 'the hub is one n-gon of the right size');
        // No orphaned vertices from a rejected direction or winding attempt.
        const used = new Set(out.cage.faces.flat());
        assert.equal(used.size, out.cage.vertices.length, 'every vertex in the result is used by a face');
      }
    }
    assert.equal(tried, factorialOf(n) * (1 << n), `all ${tried} variants were exercised`);
  });
}
function factorialOf(n) { let f = 1; for (let i = 2; i <= n; i++) f *= i; return f; }
