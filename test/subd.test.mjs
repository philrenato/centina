import test from 'node:test';
import assert from 'node:assert/strict';
import { add, scale } from '../kernel/vec3.mjs';
import {
  edgeKey, creaseWeight, buildTopology,
  computeFacePoint, computeEdgePoint, computeVertexPoint,
  subdivideCatmullClark,
  chooseSuperBRefinementLevel, triangulateFace, superbDisplayMesh,
  MARKED_CORNER_WEIGHT_FLOOR,
} from '../kernel/subd.mjs';

function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function assertFinitePoint(p, label) {
  assert.ok(Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]), `${label}: expected a finite point, got ${JSON.stringify(p)}`);
}

// ---------------------------------------------------------------------------
// FIXTURES
// ---------------------------------------------------------------------------

// A unit cube, 8 vertices, 6 quad faces, every vertex valence 3 — the
// classic worked example this file's own tests hand-derive against.
// Vertex 6 = (1,1,1); its three faces are z=+1 (centroid (0,0,1)), x=+1
// (centroid (1,0,0)), y=+1 (centroid (0,1,0)); its three neighbors are
// vertex 5 (1,-1,1), vertex 7 (-1,1,1), vertex 2 (1,1,-1) — checked
// directly against the face list below, not assumed.
function makeCube() {
  const vertices = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
  ];
  const faces = [
    [0, 1, 2, 3], // z=-1
    [4, 5, 6, 7], // z=+1
    [0, 1, 5, 4], // y=-1
    [1, 2, 6, 5], // x=+1
    [2, 3, 7, 6], // y=+1
    [0, 3, 7, 4], // x=-1
  ];
  return { vertices, faces };
}

// A single planar N-gon, embedded in an arbitrary (non-axis-aligned) plane
// so "stays planar" is a real, non-trivial check rather than a z=0 freebie.
// Every edge is boundary by construction (only ever one face).
function makePlanarNgon(n, planeOrigin, uAxis, vAxis) {
  const vertices = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    const u = Math.cos(a), v = Math.sin(a);
    vertices.push(add(planeOrigin, add(scale(uAxis, u), scale(vAxis, v))));
  }
  const faces = [Array.from({ length: n }, (_, i) => i)];
  return { vertices, faces };
}

function planeNormalFrom3(p0, p1, p2) {
  const e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
  const e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
  const n = [
    e1[1] * e2[2] - e1[2] * e2[1],
    e1[2] * e2[0] - e1[0] * e2[2],
    e1[0] * e2[1] - e1[1] * e2[0],
  ];
  const len = Math.hypot(n[0], n[1], n[2]);
  return [n[0] / len, n[1] / len, n[2] / len];
}
function planeOffset(p, normal) { return p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2]; }

// A 3x3 grid of vertices / 2x2 grid of quad faces, laid out:
//   6 7 8
//   3 4 5
//   0 1 2
// Vertex 4 (the center) is a genuine INTERIOR vertex (valence 4, all 4
// incident edges shared between two faces, no boundary edges at all) —
// the fixture used to isolate interior semi-sharp vertex blending from
// boundary-forced sharpness (which would otherwise dominate at any
// vertex actually sitting on this small grid's own outer edge). Row/col
// coordinate arrays default to a perfectly regular [0,1,2] grid (fine for
// every test that only cares about topology/weight bookkeeping); pass an
// IRREGULAR spacing (rows/cols) for a test that needs the smooth-vs-sharp
// blend to be genuinely, visibly distinct rather than coincide by the
// regular grid's own mirror symmetry across the crease line.
function makeGrid3x3(creases = {}, rows = [0, 1, 2], cols = [0, 1, 2]) {
  const vertices = [];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) vertices.push([cols[col], rows[row], 0]);
  }
  const idx = (col, row) => row * 3 + col;
  const faces = [
    [idx(0, 0), idx(1, 0), idx(1, 1), idx(0, 1)],
    [idx(1, 0), idx(2, 0), idx(2, 1), idx(1, 1)],
    [idx(0, 1), idx(1, 1), idx(1, 2), idx(0, 2)],
    [idx(1, 1), idx(2, 1), idx(2, 2), idx(1, 2)],
  ];
  return { vertices, faces, creases };
}

// A "wheel" of n quad faces around a single, genuinely interior valence-n
// vertex C (index 0): a ring of n mid vertices M_i (each connected to C —
// n spoke edges, each shared between two adjacent faces, so C carries no
// boundary edges regardless of n) plus one unique outer vertex N_i per
// face (so the M_i/N_i edges ARE boundary — an open "cap" mesh — but C
// itself stays a clean, uncreased extraordinary interior vertex of exactly
// valence n, the minimal fixture for testing arbitrary valence).
function makeWheel(n) {
  const vertices = [[0, 0, 0]];
  const mIdx = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    mIdx.push(vertices.length);
    vertices.push([Math.cos(a), Math.sin(a), 0]);
  }
  const faces = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * (i + 0.5)) / n;
    const nIdx = vertices.length;
    vertices.push([2 * Math.cos(a), 2 * Math.sin(a), 0.3]); // lifted out of plane, a real 3D fixture
    faces.push([0, mIdx[i], nIdx, mIdx[(i + 1) % n]]);
  }
  return { vertices, faces };
}

// A genuine, guess-free convergence proof needs no external "the true
// limit is exactly X" oracle at all — Catmull-Clark's own subdivision
// matrix has a single dominant eigenvalue of 1 (the trivial
// affine/translation-invariance direction) and every OTHER eigenvalue
// strictly smaller in magnitude, which is exactly what makes the discrete
// vertex sequence a genuine CONTRACTION toward a single fixed point,
// provably (not just plausibly) via shrinking successive deltas — the
// property these tests actually check, rather than asserting a specific
// numeric limit value recalled from memory and risking exactly the kind
// of unverified-claim mistake this project's own standing rules warn
// against. (An earlier draft of this file DID attempt a specific
// closed-form "exact evaluation" mask as an independent oracle — checked
// directly against the DISCRETE subdivision output at several levels, it
// failed a real self-consistency requirement any genuine exact-evaluation
// mask must satisfy, i.e. it was not, in fact, the correct mask, and was
// removed rather than shipped as unverified "ground truth.")
function successiveDeltas(seq) {
  const out = [];
  for (let i = 1; i < seq.length; i++) out.push(dist(seq[i], seq[i - 1]));
  return out;
}

// ---------------------------------------------------------------------------
// TEST 1 — planarity invariant on flat, uncreased input
// ---------------------------------------------------------------------------

// A tiny "book hinge" fixture — a genuinely NON-MANIFOLD edge (used by 3+
// faces at once): 3 quad "pages" fanned out at deliberately IRREGULAR
// angles/radii (never symmetric — a symmetric fan would make the smooth
// 3-face-centroid average coincide with the plain midpoint BY that
// symmetry alone, the same fixture pitfall TEST 4's own partial-blend
// test already documents and avoids) from a shared vertical spine edge
// (vertex 0 = A, vertex 1 = B).
function makeBookHinge() {
  const A = [0, 0, 0];
  const B = [0, 0, 1];
  const vertices = [A, B];
  const faces = [];
  const pages = [{ a: 0, r: 1 }, { a: 0.7, r: 1.3 }, { a: 2.5, r: 0.6 }];
  for (const { a, r } of pages) {
    const dir = [Math.cos(a) * r, Math.sin(a) * r, 0];
    const C = [A[0] + dir[0], A[1] + dir[1], A[2]];
    const D = [B[0] + dir[0], B[1] + dir[1], B[2]];
    const cIdx = vertices.length; vertices.push(C);
    const dIdx = vertices.length; vertices.push(D);
    faces.push([0, cIdx, dIdx, 1]); // closes with the (1,0) edge every time -> the shared spine
  }
  return { vertices, faces };
}

test('subdividing a planar quad with no creases keeps every new point exactly on the same plane', () => {
  const origin = [3, -2, 5];
  const uAxis = [1, 0.3, -0.2];
  const vAxis = [0.1, 1, 0.4];
  const cage = makePlanarNgon(4, origin, uAxis, vAxis);
  const normal = planeNormalFrom3(cage.vertices[0], cage.vertices[1], cage.vertices[2]);
  const offset = planeOffset(cage.vertices[0], normal);

  const refined = subdivideCatmullClark(cage);
  for (const p of refined.vertices) {
    assert.ok(Math.abs(planeOffset(p, normal) - offset) < 1e-9, `point ${JSON.stringify(p)} left the plane`);
  }
});

test('subdividing a planar pentagon (ngon, not just a quad) with no creases also stays exactly planar', () => {
  const origin = [0, 0, 10];
  const uAxis = [1, 0, 0];
  const vAxis = [0, 1, 0.6];
  const cage = makePlanarNgon(5, origin, uAxis, vAxis);
  const normal = planeNormalFrom3(cage.vertices[0], cage.vertices[1], cage.vertices[2]);
  const offset = planeOffset(cage.vertices[0], normal);

  const refined = subdivideCatmullClark(cage);
  assert.equal(refined.faces.length, 5, 'a pentagon must refine into exactly 5 new quads (one per corner)');
  for (const p of refined.vertices) {
    assert.ok(Math.abs(planeOffset(p, normal) - offset) < 1e-9, `point ${JSON.stringify(p)} left the plane`);
  }

  // A SECOND pass (subdividing the already-refined, all-boundary pentagon
  // again) must stay planar too — proves the invariant isn't a one-shot
  // fluke of the very first refinement.
  const refined2 = subdivideCatmullClark(refined);
  for (const p of refined2.vertices) {
    assert.ok(Math.abs(planeOffset(p, normal) - offset) < 1e-9, `second-pass point ${JSON.stringify(p)} left the plane`);
  }
});

// ---------------------------------------------------------------------------
// TEST 2 — the cube, hand-derived first-level value + analytic-limit convergence
// ---------------------------------------------------------------------------

test('cube corner (valence 3): one subdivision step exactly matches the hand-derived Catmull-Clark formula', () => {
  const cage = makeCube();
  // Hand derivation (also in this file's own header comment above
  // makeCube): F = avg((0,0,1),(1,0,0),(0,1,0)) = (1/3,1/3,1/3); R =
  // avg(midpoint((1,1,1),(1,-1,1))=(1,0,1), midpoint((1,1,1),(-1,1,1))=(0,1,1),
  // midpoint((1,1,1),(1,1,-1))=(1,1,0)) = (2/3,2/3,2/3); n=3, so
  // (F + 2R + 0*P)/3 = ((1/3+4/3)/3, ...) = (5/9,5/9,5/9) in every axis.
  const refined = subdivideCatmullClark(cage);
  const expected = [5 / 9, 5 / 9, 5 / 9];
  assert.ok(dist(refined.vertices[6], expected) < 1e-9,
    `expected (5/9,5/9,5/9), got ${JSON.stringify(refined.vertices[6])}`);
});

test('cube corner (valence 3): repeated subdivision genuinely CONTRACTS toward a single fixed point — each step moves strictly less than the last, and the ratio settles (geometric convergence, not linear/no convergence)', () => {
  const cage = makeCube();
  let cur = cage;
  const positions = [cur.vertices[6]];
  for (let level = 0; level < 7; level++) {
    cur = subdivideCatmullClark(cur);
    positions.push(cur.vertices[6]);
  }
  const deltas = successiveDeltas(positions);

  // Strictly shrinking step sizes — the actual, provable Catmull-Clark
  // contraction property (every eigenvalue of the local subdivision
  // matrix besides the trivial 1 is strictly smaller in magnitude).
  for (let i = 1; i < deltas.length; i++) {
    assert.ok(deltas[i] < deltas[i - 1], `expected step ${i} (${deltas[i]}) strictly smaller than step ${i - 1} (${deltas[i - 1]}) — deltas: ${JSON.stringify(deltas)}`);
  }
  // Genuinely GEOMETRIC (not merely shrinking) convergence: the ratio of
  // consecutive steps should settle to a roughly constant value well
  // under 1, not, say, shrink only additively toward a nonzero floor.
  const lastRatios = [];
  for (let i = deltas.length - 3; i < deltas.length; i++) lastRatios.push(deltas[i] / deltas[i - 1]);
  for (const r of lastRatios) assert.ok(r > 0 && r < 0.9, `expected each late-stage step ratio well under 1 (genuine geometric decay), got ${JSON.stringify(lastRatios)}`);
  const ratioSpread = Math.max(...lastRatios) - Math.min(...lastRatios);
  assert.ok(ratioSpread < 0.05, `expected the step ratio to have settled to a near-constant value by the later iterations, got ${JSON.stringify(lastRatios)}`);

  // A real Cauchy-convergence proof: the LAST step is tiny compared to
  // the very first one — the sequence is genuinely settling down, not
  // wandering or diverging.
  assert.ok(deltas[deltas.length - 1] < deltas[0] * 0.01, `expected the final step to be under 1% of the first, got first=${deltas[0]} last=${deltas[deltas.length - 1]}`);
});

// ---------------------------------------------------------------------------
// TEST 3 — a fully-creased cage behaves exactly like naive linear (faceted) splitting
// ---------------------------------------------------------------------------

test('a fully-creased cube (every edge weight >= 1) subdivides EXACTLY like naive linear quad-splitting — vertices unmoved, edges at midpoints, faces at centroids', () => {
  const cage = makeCube();
  const ctx = buildTopology(cage);
  cage.creases = {};
  for (const key of ctx.edgeMap.keys()) cage.creases[key] = 5; // "at least 1" and then some, per the milestone's own wording

  const refined = subdivideCatmullClark(cage);

  // Independently computed "naive" (pure geometric, no smoothing at all)
  // linear subdivision: vertex points = the ORIGINAL vertices, unmoved;
  // edge points = plain midpoints; face points = plain centroids — built
  // here by hand, not by calling any of subd.mjs's own vertex/edge rule
  // functions, so this is a genuine independent check of the actual
  // formula's OUTPUT, not a tautology against its own internals.
  for (let i = 0; i < cage.vertices.length; i++) {
    assert.ok(dist(refined.vertices[i], cage.vertices[i]) < 1e-9,
      `fully-creased vertex ${i} must stay exactly at its original position, got ${JSON.stringify(refined.vertices[i])} vs original ${JSON.stringify(cage.vertices[i])}`);
  }
  let idx = cage.vertices.length;
  for (const edge of ctx.edgeMap.values()) {
    const expectedMid = [
      (cage.vertices[edge.v0][0] + cage.vertices[edge.v1][0]) / 2,
      (cage.vertices[edge.v0][1] + cage.vertices[edge.v1][1]) / 2,
      (cage.vertices[edge.v0][2] + cage.vertices[edge.v1][2]) / 2,
    ];
    assert.ok(dist(refined.vertices[idx], expectedMid) < 1e-9,
      `fully-creased edge point at index ${idx} must be the plain midpoint, got ${JSON.stringify(refined.vertices[idx])} vs expected ${JSON.stringify(expectedMid)}`);
    idx++;
  }
  cage.faces.forEach((face, fi) => {
    let c = [0, 0, 0];
    for (const vi of face) c = add(c, cage.vertices[vi]);
    c = scale(c, 1 / face.length);
    assert.ok(dist(refined.vertices[idx], c) < 1e-9,
      `fully-creased face point ${fi} must be the plain centroid, got ${JSON.stringify(refined.vertices[idx])} vs expected ${JSON.stringify(c)}`);
    idx++;
  });

  // Structural check: 8 vertex points + 12 edge points + 6 face points =
  // 26, and 6 faces * 4 corners = 24 new quads — the same counts a smooth
  // (uncreased) cube subdivision would also produce, since crease weight
  // never changes topology, only position.
  assert.equal(refined.vertices.length, 26, 'expected 8 vertex + 12 edge + 6 face points = 26');
  assert.equal(refined.faces.length, 24, 'expected 6 faces * 4 corners = 24 new quads');
});

// ---------------------------------------------------------------------------
// TEST 4 — semi-sharp decrement, both the weight bookkeeping and the real
// blended geometry, isolated from boundary-forced sharpness via the grid
// ---------------------------------------------------------------------------

test('semi-sharp crease weight decrements by exactly 1.0 per subdivision call, and drops out of the creases map once decayed to 0', () => {
  // A horizontal crease LINE through the 3x3 grid's own interior vertex 4
  // (edges (3,4) and (4,5), both weight 2.3 — this file's own worked
  // "sharp/sharp/blended-0.3/smooth" example from the spec).
  let cage = makeGrid3x3({ [edgeKey(3, 4)]: 2.3, [edgeKey(4, 5)]: 2.3 });

  // Level 0 -> 1: both edges are still fully sharp THIS step (clamp(2.3)=1)
  // but decrement to max(0,2.3-1)=1.3 for their children.
  cage = subdivideCatmullClark(cage);
  let values = Object.values(cage.creases);
  assert.equal(values.length, 4, `expected 4 child crease entries (2 edges x 2 children), got ${values.length}`);
  for (const v of values) assert.ok(Math.abs(v - 1.3) < 1e-9, `expected 1.3, got ${v}`);

  // Level 1 -> 2: still sharp this step (clamp(1.3)=1), decrement to 0.3.
  cage = subdivideCatmullClark(cage);
  values = Object.values(cage.creases);
  assert.equal(values.length, 8, `expected 8 child crease entries, got ${values.length}`);
  for (const v of values) assert.ok(Math.abs(v - 0.3) < 1e-9, `expected 0.3, got ${v}`);

  // Level 2 -> 3: BLENDED this step (clamp(0.3)=0.3, neither pure smooth
  // nor pure sharp), decrement to max(0,0.3-1)=0 -> dropped entirely.
  cage = subdivideCatmullClark(cage);
  values = Object.values(cage.creases);
  assert.equal(values.length, 0, `expected the crease line to have fully decayed away (0 entries), got ${values.length}`);

  // Level 3 -> 4: now an ordinary smooth cage forever after — subdividing
  // again must not resurrect anything.
  cage = subdivideCatmullClark(cage);
  assert.equal(Object.keys(cage.creases).length, 0, 'a decayed-to-0 crease must never reappear');
});

test('semi-sharp decrement at several other starting weights: <1 decays in one step, exactly 1 decays in one step, 0/absent never creased at all', () => {
  for (const w of [0.5, 1, 2]) {
    let cage = makeGrid3x3({ [edgeKey(3, 4)]: w, [edgeKey(4, 5)]: w });
    cage = subdivideCatmullClark(cage);
    const expectedRemaining = Math.max(0, w - 1);
    const values = Object.values(cage.creases);
    if (expectedRemaining <= 0) {
      assert.equal(values.length, 0, `weight ${w}: expected full decay after 1 step, got ${values.length} entries`);
    } else {
      assert.equal(values.length, 4, `weight ${w}: expected 4 child entries, got ${values.length}`);
      for (const v of values) assert.ok(Math.abs(v - expectedRemaining) < 1e-9, `weight ${w}: expected ${expectedRemaining}, got ${v}`);
    }
  }
  // Genuinely uncreased from the start: never in the map, at any level.
  let plain = makeGrid3x3();
  for (let i = 0; i < 3; i++) {
    plain = subdivideCatmullClark(plain);
    assert.equal(Object.keys(plain.creases).length, 0, `an uncreased cage must never gain crease entries (level ${i + 1})`);
  }
});

test('a partial (0.3) semi-sharp crease produces a REAL blended edge point and vertex point — genuinely between the smooth and sharp values, not equal to either', () => {
  const w = 0.3;
  // Deliberately IRREGULAR row/col spacing — a perfectly regular grid is
  // mirror-symmetric across the horizontal crease line through vertex 4,
  // which (found directly, by running this test against the regular
  // grid first) makes the smooth and sharp formulas coincide EXACTLY by
  // that symmetry alone, a fixture artifact that would make the "not
  // equal to either" assertions below spuriously fail even though the
  // blend computation itself is correct. Irregular spacing breaks the
  // symmetry so smooth and sharp are genuinely, visibly distinct values.
  const cage = makeGrid3x3({ [edgeKey(3, 4)]: w, [edgeKey(4, 5)]: w }, [0, 1, 2.7], [0, 1.4, 2]);
  const ctx = buildTopology(cage);

  // --- edge point on (3,4) ---
  const edge34 = ctx.edgeMap.get(edgeKey(3, 4));
  const v3 = cage.vertices[3], v4 = cage.vertices[4];
  const sharpMid = [(v3[0] + v4[0]) / 2, (v3[1] + v4[1]) / 2, (v3[2] + v4[2]) / 2];
  let sum = add(v3, v4);
  for (const fi of edge34.faces) sum = add(sum, computeFacePoint(cage, fi));
  const smoothEdge = scale(sum, 1 / (2 + edge34.faces.length));
  const expectedEdgePt = [
    smoothEdge[0] * (1 - w) + sharpMid[0] * w,
    smoothEdge[1] * (1 - w) + sharpMid[1] * w,
    smoothEdge[2] * (1 - w) + sharpMid[2] * w,
  ];
  const actualEdgePt = computeEdgePoint(cage, ctx, edge34);
  assert.ok(dist(actualEdgePt, expectedEdgePt) < 1e-9,
    `expected the blended edge point ${JSON.stringify(expectedEdgePt)}, got ${JSON.stringify(actualEdgePt)}`);
  assert.ok(dist(actualEdgePt, sharpMid) > 1e-4, 'a 0.3 blend must NOT equal the pure-sharp midpoint');
  assert.ok(dist(actualEdgePt, smoothEdge) > 1e-4, 'a 0.3 blend must NOT equal the pure-smooth value either');

  // --- vertex point at the shared center vertex 4 ---
  // Smooth rule computed independently: F = avg of the 4 face centroids
  // touching vertex 4; R = avg of the 4 edge midpoints from vertex 4 to
  // its neighbors 1,3,5,7; n=4.
  const faces4 = ctx.vertexFaces[4];
  let F = [0, 0, 0];
  for (const fi of faces4) F = add(F, computeFacePoint(cage, fi));
  F = scale(F, 1 / faces4.length);
  const neighbors = [1, 3, 5, 7];
  let R = [0, 0, 0];
  for (const nb of neighbors) R = add(R, [(v4[0] + cage.vertices[nb][0]) / 2, (v4[1] + cage.vertices[nb][1]) / 2, (v4[2] + cage.vertices[nb][2]) / 2]);
  R = scale(R, 1 / 4);
  const smoothVertex = scale(add(add(F, scale(R, 2)), scale(v4, 4 - 3)), 1 / 4);
  const v3v5Crease = scale(add(add(cage.vertices[3], cage.vertices[5]), scale(v4, 6)), 1 / 8);
  const expectedVertex = [
    smoothVertex[0] * (1 - w) + v3v5Crease[0] * w,
    smoothVertex[1] * (1 - w) + v3v5Crease[1] * w,
    smoothVertex[2] * (1 - w) + v3v5Crease[2] * w,
  ];
  const actualVertex = computeVertexPoint(cage, ctx, 4);
  assert.ok(dist(actualVertex, expectedVertex) < 1e-9,
    `expected the blended vertex point ${JSON.stringify(expectedVertex)}, got ${JSON.stringify(actualVertex)}`);
  assert.ok(dist(actualVertex, v3v5Crease) > 1e-4, 'a 0.3 blend must NOT equal the pure-crease-line value');
  assert.ok(dist(actualVertex, smoothVertex) > 1e-4, 'a 0.3 blend must NOT equal the pure-smooth value either');

  // Full-pipeline consistency: subdivideCatmullClark's own output at
  // vertex index 4 (indices preserved) must match the direct call above.
  const refined = subdivideCatmullClark(cage);
  assert.ok(dist(refined.vertices[4], actualVertex) < 1e-12, 'the full pipeline must wire computeVertexPoint in unchanged');
});

test('fully sharp (weight >= 1) and fully smooth (weight 0) edge/vertex blends resolve to EXACTLY the sharp/smooth value, not a near-miss', () => {
  const sharpCage = makeGrid3x3({ [edgeKey(3, 4)]: 1, [edgeKey(4, 5)]: 1 });
  const smoothCage = makeGrid3x3(); // no creases at all
  const ctxSharp = buildTopology(sharpCage);
  const ctxSmooth = buildTopology(smoothCage);

  const edgeSharp = computeEdgePoint(sharpCage, ctxSharp, ctxSharp.edgeMap.get(edgeKey(3, 4)));
  const midSharp = [
    (sharpCage.vertices[3][0] + sharpCage.vertices[4][0]) / 2,
    (sharpCage.vertices[3][1] + sharpCage.vertices[4][1]) / 2,
    (sharpCage.vertices[3][2] + sharpCage.vertices[4][2]) / 2,
  ];
  assert.ok(dist(edgeSharp, midSharp) < 1e-12, 'weight>=1 must give the EXACT plain midpoint');

  const vertexSharp = computeVertexPoint(sharpCage, ctxSharp, 4);
  const creaseSharp = [
    (sharpCage.vertices[3][0] + sharpCage.vertices[5][0] + 6 * sharpCage.vertices[4][0]) / 8,
    (sharpCage.vertices[3][1] + sharpCage.vertices[5][1] + 6 * sharpCage.vertices[4][1]) / 8,
    (sharpCage.vertices[3][2] + sharpCage.vertices[5][2] + 6 * sharpCage.vertices[4][2]) / 8,
  ];
  assert.ok(dist(vertexSharp, creaseSharp) < 1e-12, 'weight>=1 must give the EXACT crease-line rule value');

  const vertexSmooth = computeVertexPoint(smoothCage, ctxSmooth, 4);
  // The smoothed center of a perfectly flat, regular grid must stay
  // exactly where it started (the same fixed-point property proven above
  // via the analytic-limit formula, now proven for the DISCRETE one-step
  // rule too).
  assert.ok(dist(vertexSmooth, smoothCage.vertices[4]) < 1e-9, 'an uncreased regular flat grid vertex must be unmoved by the smooth rule');
});

// ---------------------------------------------------------------------------
// TEST 5 — extraordinary vertices (valence 3, 5, 7): finite, sane, converging
// ---------------------------------------------------------------------------

for (const n of [3, 5, 6, 7]) {
  test(`valence-${n} extraordinary interior vertex (no creases): repeated subdivision stays finite everywhere and genuinely converges (qualitative Catmull-Clark behavior — slightly different limit tangent behavior than valence-4, but finite and well-behaved)`, () => {
    const cage = makeWheel(n);
    const ctx0 = buildTopology(cage);
    assert.equal(ctx0.vertexEdges[0].length, n, `expected the wheel's own center vertex to have valence ${n}`);

    let cur = cage;
    const positions = [cur.vertices[0]];
    for (let level = 0; level < 6; level++) {
      cur = subdivideCatmullClark(cur);
      for (const p of cur.vertices) assertFinitePoint(p, `valence ${n} level ${level + 1}`);
      positions.push(cur.vertices[0]);
    }
    const deltas = successiveDeltas(positions);
    for (let i = 1; i < deltas.length; i++) {
      assert.ok(deltas[i] < deltas[i - 1] + 1e-12, `valence ${n}: expected step ${i} (${deltas[i]}) no larger than step ${i - 1} (${deltas[i - 1]}) — deltas: ${JSON.stringify(deltas)}`);
    }
    assert.ok(deltas[deltas.length - 1] < deltas[0] * 0.1,
      `valence ${n}: expected the final step to have shrunk well below the first (genuine convergence, not a blowup) — first=${deltas[0]} last=${deltas[deltas.length - 1]}`);
    // No coordinate should have run away in magnitude either — a real,
    // if coarse, "not diverging" sanity bound distinct from the per-point
    // finiteness check above (which would pass even for a wild but
    // finite blowup).
    for (const p of cur.vertices) {
      assert.ok(Math.hypot(p[0], p[1], p[2]) < 10, `valence ${n}: expected bounded coordinates (no blowup), got ${JSON.stringify(p)}`);
    }
  });
}

// ---------------------------------------------------------------------------
// TEST 6 — DART vertices (exactly one incident sharp edge) use the
// ordinary smooth rule, not the corner rule
// ---------------------------------------------------------------------------

test('a DART vertex (exactly one incident sharp/creased edge — the OPEN END of a crease line) uses the ordinary smooth vertex rule, not the corner rule — the crease must fade out smoothly instead of leaving a permanent pinch', () => {
  // Irregular spacing (same reasoning as TEST 4's own partial-blend test):
  // a perfectly regular grid is mirror-symmetric across vertex 4, which
  // would make the smooth and the (buggy) corner-pinned-at-P values
  // coincide by that symmetry alone.
  const cage = makeGrid3x3({ [edgeKey(3, 4)]: 1 }, [0, 1, 2], [0, 1.4, 2.7]);
  const ctx = buildTopology(cage);
  const v4 = cage.vertices[4];

  // Independently hand-derive the smooth rule at vertex 4: F = avg of its
  // 4 face centroids; R = avg of its 4 edge midpoints to neighbors
  // 1, 3, 5, 7; n=4. Same derivation TEST 4's own partial-blend test
  // already uses for this identical vertex.
  const faces4 = ctx.vertexFaces[4];
  let F = [0, 0, 0];
  for (const fi of faces4) F = add(F, computeFacePoint(cage, fi));
  F = scale(F, 1 / faces4.length);
  const neighbors = [1, 3, 5, 7];
  let R = [0, 0, 0];
  for (const nb of neighbors) {
    const nbV = cage.vertices[nb];
    R = add(R, [(v4[0] + nbV[0]) / 2, (v4[1] + nbV[1]) / 2, (v4[2] + nbV[2]) / 2]);
  }
  R = scale(R, 1 / 4);
  const expectedSmooth = scale(add(add(F, scale(R, 2)), scale(v4, 4 - 3)), 1 / 4);

  const actual = computeVertexPoint(cage, ctx, 4);
  assert.ok(dist(actual, expectedSmooth) < 1e-9,
    `a dart vertex (1 sharp incident edge) must equal the ordinary smooth rule exactly, got ${JSON.stringify(actual)} vs expected ${JSON.stringify(expectedSmooth)}`);

  // The OLD, incorrect behavior pinned a dart at its own unmoved position
  // (lumped in with the true 3+-edge corner rule) — confirm the fixed
  // result is genuinely NOT that.
  assert.ok(dist(actual, v4) > 1e-4,
    'a dart must NOT be pinned at its own original position (the old, incorrect corner treatment)');

  // A SECOND, independent check: a dart's own single incident edge's
  // sharpness VALUE must not matter at all (0.5 gives the identical
  // pure-smooth answer as 1) — a dart is never "partially" pinned the way
  // a true 2-edge crease or a 3+-edge corner is; there is no crease LINE
  // or corner for it to blend toward.
  const cageHalf = makeGrid3x3({ [edgeKey(3, 4)]: 0.5 }, [0, 1, 2], [0, 1.4, 2.7]);
  const ctxHalf = buildTopology(cageHalf);
  const actualHalf = computeVertexPoint(cageHalf, ctxHalf, 4);
  assert.ok(dist(actualHalf, expectedSmooth) < 1e-9,
    'a dart at a DIFFERENT (partial, 0.5) sharpness value must give the identical pure-smooth answer as weight 1');
});

// ---------------------------------------------------------------------------
// MARKED-CORNER EXTENSION (built for TOSUBD's own Corners=Yes/
// No option) — a true single-face grid corner (valence 2,
// both edges boundary) has no third edge to ever reach the ordinary 3+
// CORNER branch, so it structurally can't ask for the "hold at P" treatment
// under the plain sharp-edge-count taxonomy. This tests the new marker:
// storing an explicit nonzero weight on BOTH of a corner's own boundary
// edges (otherwise completely inert for the edge's own sharpness, see
// edgeSharpness) now flips that corner to the held-at-P rule.
// ---------------------------------------------------------------------------

// A weight safely ABOVE MARKED_CORNER_WEIGHT_FLOOR — the only kind of
// stored weight that should ever trigger the marked-corner treatment.
const MARK_WEIGHT = MARKED_CORNER_WEIGHT_FLOOR + 50;

test('MARKED CORNER: a true grid corner (valence 2, both edges boundary) with a weight ABOVE MARKED_CORNER_WEIGHT_FLOOR on BOTH its boundary edges is held EXACTLY at its own position P, not the ordinary boundary-curve rule', () => {
  // Irregular spacing (same reasoning as the DART test above) so the
  // curve-rule answer and P don't coincide by symmetry alone.
  const rows = [0, 1, 2.6], cols = [0, 1.3, 2];
  const markedCage = makeGrid3x3({ [edgeKey(0, 1)]: MARK_WEIGHT, [edgeKey(0, 3)]: MARK_WEIGHT }, rows, cols);
  const ctxMarked = buildTopology(markedCage);
  const P0 = markedCage.vertices[0];

  const actual = computeVertexPoint(markedCage, ctxMarked, 0);
  assert.ok(dist(actual, P0) < 1e-12,
    `a marked corner must be held EXACTLY at its own position, got ${JSON.stringify(actual)} vs P=${JSON.stringify(P0)}`);

  // Independent control: the IDENTICAL cage/vertex with NO stored weight
  // on those two boundary edges (the default, and the ONLY case any
  // existing cage in this app has ever produced before this) must fall
  // through to the ordinary (A+6P+B)/8 boundary-curve rule, genuinely
  // different from P — proving the marker (not some incidental geometry)
  // is what made the difference above.
  const unmarkedCage = makeGrid3x3({}, rows, cols);
  const ctxUnmarked = buildTopology(unmarkedCage);
  const actualUnmarked = computeVertexPoint(unmarkedCage, ctxUnmarked, 0);
  const A = unmarkedCage.vertices[1], B = unmarkedCage.vertices[3];
  const expectedCurve = scale(add(add(A, B), scale(P0, 6)), 1 / 8);
  assert.ok(dist(actualUnmarked, expectedCurve) < 1e-9, 'an unmarked corner must match the ordinary boundary-curve rule exactly');
  assert.ok(dist(actualUnmarked, P0) > 1e-4, 'an unmarked corner must genuinely differ from P (otherwise this fixture proves nothing)');
});

test('MARKED CORNER: only ONE of the two boundary edges carrying a weight above the floor is NOT enough — both must be marked', () => {
  const rows = [0, 1, 2.6], cols = [0, 1.3, 2];
  const halfMarkedCage = makeGrid3x3({ [edgeKey(0, 1)]: MARK_WEIGHT }, rows, cols); // only one of vertex 0's two boundary edges
  const ctx = buildTopology(halfMarkedCage);
  const P0 = halfMarkedCage.vertices[0];
  const actual = computeVertexPoint(halfMarkedCage, ctx, 0);
  const A = halfMarkedCage.vertices[1], B = halfMarkedCage.vertices[3];
  const expectedCurve = scale(add(add(A, B), scale(P0, 6)), 1 / 8);
  assert.ok(dist(actual, expectedCurve) < 1e-9, 'half-marked (only one boundary edge weighted) must still use the ordinary curve rule, not be held at P');
});

test('MARKED CORNER: a mid-boundary vertex (valence 3 — 2 boundary + 1 smooth interior edge) is NEVER treated as a marked corner even if both its boundary edges carry a weight above the floor — it is not a true single-face grid corner', () => {
  // Vertex 1 (top row, middle column) of the 3x3 grid: boundary edges to
  // vertices 0 and 2, PLUS an interior (smooth, uncreased) edge down to
  // vertex 4 — valence 3, not the true corner's valence 2.
  const rows = [0, 1, 2.6], cols = [0, 1.3, 2];
  const cage = makeGrid3x3({ [edgeKey(1, 0)]: MARK_WEIGHT, [edgeKey(1, 2)]: MARK_WEIGHT }, rows, cols);
  const ctx = buildTopology(cage);
  assert.equal(ctx.vertexEdges[1].length, 3, 'sanity: vertex 1 must genuinely have 3 incident edges for this test to mean anything');
  const P1 = cage.vertices[1];
  const actual = computeVertexPoint(cage, ctx, 1);
  const A = cage.vertices[0], B = cage.vertices[2];
  const expectedCurve = scale(add(add(A, B), scale(P1, 6)), 1 / 8);
  assert.ok(dist(actual, expectedCurve) < 1e-9, 'a valence-3 mid-boundary vertex must use the ordinary curve rule, never the marked-corner hold-at-P treatment');
});

// REGRESSION — the exact collision the review
// found: this app's own shipped Crease command stores SUPERB_CREASE_LEVEL_
// SCALE (3) on a full "harden", and SoftCrease's 0-100 slider reaches every
// value up to it. Proves a corner creased this way via the ORDINARY,
// already-shipped Crease command (both boundary edges of a plain corner,
// weight 3 — nowhere near MARKED_CORNER_WEIGHT_FLOOR) is NOT reinterpreted
// as a marked TOSUBD corner and keeps the old boundary-curve behavior.
test('MARKED CORNER: an ordinary Crease-command weight (3, this app\'s own real SUPERB_CREASE_LEVEL_SCALE) on both boundary edges of a corner must NOT trigger the marked-corner treatment — only TOSUBD\'s own far-higher weight should', () => {
  const rows = [0, 1, 2.6], cols = [0, 1.3, 2];
  const creaseCommandWeight = 3; // the app's own SUPERB_CREASE_LEVEL_SCALE
  assert.ok(creaseCommandWeight <= MARKED_CORNER_WEIGHT_FLOOR, 'sanity: this test only means something if 3 is comfortably below the floor');
  const cage = makeGrid3x3({ [edgeKey(0, 1)]: creaseCommandWeight, [edgeKey(0, 3)]: creaseCommandWeight }, rows, cols);
  const ctx = buildTopology(cage);
  const P0 = cage.vertices[0];
  const actual = computeVertexPoint(cage, ctx, 0);
  const A = cage.vertices[1], B = cage.vertices[3];
  const expectedCurve = scale(add(add(A, B), scale(P0, 6)), 1 / 8);
  assert.ok(dist(actual, expectedCurve) < 1e-9,
    'an ordinary Crease-command weight on both boundary edges of a corner must still use the boundary-curve rule, not be silently pinned at P');
});

// ---------------------------------------------------------------------------
// TEST 7 — a genuinely NON-MANIFOLD edge (3+ incident faces) is forced
// fully sharp, matching this module's own stated header-comment contract
// ---------------------------------------------------------------------------

test('a non-manifold edge (shared by 3 faces — a "book hinge") is forced fully sharp, matching the plain boundary-edge midpoint rule, not a smooth blend across all 3 face centroids', () => {
  const cage = makeBookHinge();
  const ctx = buildTopology(cage);
  const hingeEdge = ctx.edgeMap.get(edgeKey(0, 1));
  assert.equal(hingeEdge.faces.length, 3, 'expected the spine edge to be used by all 3 pages (a genuine non-manifold edge)');

  const A = cage.vertices[0], B = cage.vertices[1];
  const actual = computeEdgePoint(cage, ctx, hingeEdge);
  const expectedMid = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2];
  assert.ok(dist(actual, expectedMid) < 1e-9,
    `a non-manifold edge must be forced fully sharp (the plain midpoint), got ${JSON.stringify(actual)} vs expected midpoint ${JSON.stringify(expectedMid)}`);

  // Confirm this ISN'T a coincidence of the fixture's own symmetry — the
  // smooth (average-of-3-face-centroids) alternative, computed
  // independently here, must land somewhere genuinely different from the
  // plain midpoint, or this fixture wouldn't actually discriminate the fix
  // from the pre-fix behavior.
  let smoothSum = add(A, B);
  for (const fi of hingeEdge.faces) smoothSum = add(smoothSum, computeFacePoint(cage, fi));
  const smoothAlternative = scale(smoothSum, 1 / (2 + hingeEdge.faces.length));
  assert.ok(dist(smoothAlternative, expectedMid) > 1e-4,
    'the smooth (3-face-average) alternative must be genuinely different from the plain midpoint — otherwise this fixture would not discriminate the fix');
});

test('subdivideCatmullClark never mutates its input cage', () => {
  const cage = makeCube();
  const snapshotVerts = JSON.parse(JSON.stringify(cage.vertices));
  const snapshotFaces = JSON.parse(JSON.stringify(cage.faces));
  subdivideCatmullClark(cage);
  assert.deepEqual(cage.vertices, snapshotVerts, 'input vertices must be untouched');
  assert.deepEqual(cage.faces, snapshotFaces, 'input faces must be untouched');
});

test('subdivideCatmullClark refuses malformed cages honestly rather than producing garbage', () => {
  assert.throws(() => subdivideCatmullClark({ vertices: [], faces: [[0, 1, 2]] }));
  assert.throws(() => subdivideCatmullClark({ vertices: [[0, 0, 0], [1, 0, 0]], faces: [] }));
  assert.throws(() => subdivideCatmullClark({ vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1]] })); // face with only 2 indices
  assert.throws(() => subdivideCatmullClark({ vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1, 9]] })); // out-of-range index
});

// ---------------------------------------------------------------------------
// SuperB DISPLAY MESH — chooseSuperBRefinementLevel / triangulateFace /
// superbDisplayMesh
// ---------------------------------------------------------------------------

test('chooseSuperBRefinementLevel: a coarse cage (the plain cube, 6 faces) gets 3 levels; a dense uncreased cage gets 2', () => {
  const cube = makeCube();
  assert.equal(chooseSuperBRefinementLevel(cube), 3, 'a 6-face cube is coarse (<=24 faces by default) -> 3 levels');
  // Build a synthetic "dense" cage purely by faking a long faces array —
  // chooseSuperBRefinementLevel only ever reads .length and .creases, so a
  // real cage isn't needed to exercise the coarse/dense branch honestly.
  const dense = { vertices: cube.vertices, faces: new Array(30).fill([0, 1, 2, 3]), creases: {} };
  assert.equal(chooseSuperBRefinementLevel(dense), 2, 'a >24-face, uncreased cage is not coarse -> 2 levels');
});

test('chooseSuperBRefinementLevel: any positive crease weight forces 3 levels even on an otherwise-dense cage', () => {
  const dense = { vertices: makeCube().vertices, faces: new Array(30).fill([0, 1, 2, 3]), creases: { '0_1': 0.5 } };
  assert.equal(chooseSuperBRefinementLevel(dense), 3, 'a real crease anywhere forces 3 levels regardless of face count');
});

test('chooseSuperBRefinementLevel: the coarseFaceThreshold option is honored', () => {
  const cage = { vertices: makeCube().vertices, faces: new Array(10).fill([0, 1, 2, 3]), creases: {} };
  assert.equal(chooseSuperBRefinementLevel(cage, { coarseFaceThreshold: 5 }), 2, '10 faces > a lowered threshold of 5 -> NOT coarse by this stricter threshold -> 2');
  assert.equal(chooseSuperBRefinementLevel(cage, { coarseFaceThreshold: 50 }), 3, '10 faces <= a raised threshold of 50 -> coarse -> 3');
});

test('triangulateFace: fan-triangulates a quad and an ngon correctly, by vertex count and shared apex', () => {
  const quadTris = triangulateFace([10, 11, 12, 13]);
  assert.deepEqual(quadTris, [[10, 11, 12], [10, 12, 13]], 'a quad fans into exactly 2 triangles sharing vertex 10');
  const pentTris = triangulateFace([0, 1, 2, 3, 4]);
  assert.equal(pentTris.length, 3, 'an n-gon fans into n-2 triangles');
  for (const t of pentTris) assert.equal(t[0], 0, 'every fan triangle shares the same apex (index 0 of the face loop)');
  const triTris = triangulateFace([5, 6, 7]);
  assert.deepEqual(triTris, [[5, 6, 7]], 'a triangle fans into exactly itself, unchanged');
});

test('superbDisplayMesh box mode: returns the RAW cage triangulated as-is, level 0, never calls subdivideCatmullClark', () => {
  const cube = makeCube();
  const disp = superbDisplayMesh(cube, 'box');
  assert.equal(disp.level, 0);
  assert.equal(disp.vertices.length, 8, 'box mode keeps the raw 8 cage vertices, no refinement');
  assert.equal(disp.faces.length, 6, 'box mode keeps the raw 6 cage faces');
  assert.equal(disp.triangles.length, 12, '6 quads fan into exactly 12 triangles');
  // Must be a real copy, not aliasing the input cage's own arrays.
  disp.vertices[0][0] = 999;
  assert.notEqual(cube.vertices[0][0], 999, 'box-mode display vertices must not alias the input cage');
});

test('superbDisplayMesh smooth mode: runs the heuristic-chosen level count, produces a real, denser, all-quad-refined mesh', () => {
  const cube = makeCube();
  const disp = superbDisplayMesh(cube, 'smooth');
  assert.equal(disp.level, 3, 'a plain 6-face cube is coarse -> the heuristic picks 3 levels');
  // After N>=1 Catmull-Clark passes, every face is a quad, so the exact
  // triangle count is fully determined: 6 * 4^level quads (each pass
  // quadruples face count for an all-quad starting cage), 2 triangles/quad.
  const expectedFaces = 6 * Math.pow(4, disp.level);
  assert.equal(disp.faces.length, expectedFaces);
  assert.equal(disp.triangles.length, expectedFaces * 2);
  assert.ok(disp.vertices.length > cube.vertices.length, 'smooth mode is genuinely denser than the raw cage');
  // Never mutates the input cage — subdivideCatmullClark's own no-mutation
  // guarantee, re-confirmed at this higher level too.
  assert.equal(cube.vertices.length, 8);
  assert.equal(cube.faces.length, 6);
});

test('superbDisplayMesh smooth mode: an explicit level option overrides the heuristic', () => {
  const cube = makeCube();
  const disp = superbDisplayMesh(cube, 'smooth', { level: 2 });
  assert.equal(disp.level, 2);
  assert.equal(disp.faces.length, 6 * 16);
});

test('superbDisplayMesh: box and smooth modes on the SAME cage produce genuinely different vertex/triangle counts — the real, numeric Box/Smooth toggle proof', () => {
  const cube = makeCube();
  const box = superbDisplayMesh(cube, 'box');
  const smooth = superbDisplayMesh(cube, 'smooth');
  assert.notEqual(box.vertices.length, smooth.vertices.length);
  assert.notEqual(box.triangles.length, smooth.triangles.length);
  assert.ok(smooth.vertices.length > box.vertices.length);
  assert.ok(smooth.triangles.length > box.triangles.length);
});
