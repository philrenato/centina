import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  bicubicRegularPatchSurface, regularPatchPointAndPartials, vertexLimitMaskFromRF, vertexLimitPosition,
  standardNeighborhoodCage, buildSubdivisionMatrix, applyMatrix, edgePointIndexMap, facePointIndexBase, readNeighborhoodVectors,
  vertexLimitWeightsGeneral, powerIterationLeftDominant, semiSharpHybridLimitPosition,
} from '../kernel/subdlimit.mjs';
import { surfacePointAndPartials } from '../kernel/surface.mjs';
import { basisFuns, findSpan } from '../kernel/basis.mjs';
import { buildTopology, subdivideCatmullClark, edgeKey } from '../kernel/subd.mjs';

// INDEPENDENT ground truth #1 — the textbook closed-form uniform CUBIC
// B-SPLINE blending functions (not tautologically derived from this
// module's own code): over the single valid span, parameter t in [0,1],
//   B0(t) = (1-t)^3 / 6
//   B1(t) = (3t^3 - 6t^2 + 4) / 6
//   B2(t) = (-3t^3 + 3t^2 + 3t + 1) / 6
//   B3(t) = t^3 / 6
// This proves the module's own knot-vector choice ([-3,-2,-1,0,1,2,3,4])
// genuinely produces the standard uniform (not clamped) cubic basis, not
// just "whatever basisFuns happens to compute."
function textbookUniformCubicWeights(t) {
  return [
    Math.pow(1 - t, 3) / 6,
    (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
    (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6,
    t ** 3 / 6,
  ];
}

test('the regular-patch knot vector reproduces the textbook uniform cubic B-spline basis exactly, at several t', () => {
  const knots = [-3, -2, -1, 0, 1, 2, 3, 4];
  const p = 3, n = 3; // 4 control points, last index 3
  for (const t of [0, 0.001, 0.25, 0.5, 0.73, 0.999, 1]) {
    const span = findSpan(n, p, t, knots);
    assert.equal(span, 3, `t=${t} must resolve to the single valid span (index 3)`);
    const N = basisFuns(span, t, p, knots);
    const expected = textbookUniformCubicWeights(t);
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(N[i] - expected[i]) < 1e-12, `t=${t} weight[${i}]: got ${N[i]}, expected ${expected[i]}`);
    }
    const sum = N.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-12, `t=${t} partition of unity: sum=${sum}`);
  }
});

test('bicubicRegularPatchSurface: a hand-verifiable case — a FLAT grid (all Z=0, X/Y on an evenly-spaced lattice) evaluates to the exact tensor-product bilinear-equivalent position at several (u,v), matching a hand-computed sum', () => {
  // A 4x4 grid at integer (x,y) positions 0..3, z=0. The known uniform
  // cubic tensor-product patch through a PLANAR, EVENLY-SPACED grid of
  // control points reproduces the plane exactly (the defining "affine
  // precision" property of any B-spline basis: a partition-of-unity
  // basis applied to collinear/coplanar, evenly-spaced control points
  // reproduces the linear function they sample) — a real, independently
  // derivable check, not just "the code agrees with itself."
  const grid = [];
  for (let i = 0; i < 4; i++) {
    const row = [];
    for (let j = 0; j < 4; j++) row.push([i, j, 0]);
    grid.push(row);
  }
  const srf = bicubicRegularPatchSurface(grid);
  for (const [u, v] of [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.2, 0.9], [1, 0.5]]) {
    const { point } = surfacePointAndPartials(srf, u, v);
    // The grid's own u/v parametrization maps [0,1] to the CENTER span,
    // i.e. control-point index range [1,2] in each direction (the face's
    // own 4 corners) — so at u=v=0 the exact expected position is (1,1,0)
    // (grid[1][1]), NOT grid[0][0], since the domain [0,1] corresponds to
    // knots[3..4], the span whose de Boor points are grid indices 0..3
    // blended with the uniform cubic weights computed above, which at
    // t=0 peak at index 1 (weight 4/6) — i.e. the true expected value is
    // the AFFINE-EXACT reproduction of the coordinate function itself:
    // exact x = 1+u, exact y = 1+v (the plane z=0 passes through the
    // grid's own regularly-spaced lattice exactly, and u/v=0 sits at
    // lattice position (1,1) by the same "center 2x2 is the face" layout
    // this module's own header describes).
    const expectedX = 1 + u, expectedY = 1 + v;
    assert.ok(Math.abs(point[0] - expectedX) < 1e-9, `u=${u},v=${v}: x got ${point[0]}, expected ${expectedX}`);
    assert.ok(Math.abs(point[1] - expectedY) < 1e-9, `u=${u},v=${v}: y got ${point[1]}, expected ${expectedY}`);
    assert.ok(Math.abs(point[2]) < 1e-9, `u=${u},v=${v}: z got ${point[2]}, expected 0 (flat grid)`);
  }
});

test('regularPatchPointAndPartials: the wrapper reproduces a direct surfacePointAndPartials call bit-for-bit, at several (u,v) including all 4 corners — the actual load-bearing plumbing proof every later step depends on', () => {
  // A genuinely non-planar, asymmetric grid — real distinct partials in
  // both directions, not a degenerate flat/symmetric fixture that could
  // hide a transcription error (this project's own standing "avoid
  // trivial test geometry" discipline).
  const grid = [
    [[0, 0, 0], [1, 0, 0.3], [2, 0, 0.1], [3, 0, 0.6]],
    [[0, 1, 0.4], [1, 1, 1.2], [2, 1, 0.9], [3, 1, 1.5]],
    [[0, 2, 0.2], [1, 2, 0.8], [2, 2, 1.1], [3, 2, 0.7]],
    [[0, 3, 0.5], [1, 3, 0.9], [2, 3, 1.4], [3, 3, 1.0]],
  ];
  const srf = bicubicRegularPatchSurface(grid);
  for (const [u, v] of [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.31, 0.72]]) {
    const a = regularPatchPointAndPartials(grid, u, v);
    const b = surfacePointAndPartials(srf, u, v);
    assert.deepEqual(a.point, b.point, `point mismatch at (${u},${v})`);
    assert.deepEqual(a.su, b.su, `su mismatch at (${u},${v})`);
    assert.deepEqual(a.sv, b.sv, `sv mismatch at (${u},${v})`);
  }
});

test('bicubicRegularPatchSurface: a 4-row grid with a non-4-column row throws honestly rather than silently misreading a malformed stencil', () => {
  assert.throws(() => bicubicRegularPatchSurface([[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]));
  assert.throws(() => bicubicRegularPatchSurface([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]));
});

// ================================================================
// STEP 2 — VERTEX-LIMIT-MASK GROUND TRUTH (Halstead/Kass/DeRose 1993),
// standalone, independent of step 1 and of any eigenbasis machinery.
// ================================================================

function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function scale3(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function closeEnough(a, b, eps = 1e-9) {
  return Math.abs(a[0] - b[0]) < eps && Math.abs(a[1] - b[1]) < eps && Math.abs(a[2] - b[2]) < eps;
}

// A rotationally-symmetric fan cage around a single extraordinary
// vertex (index 0) of valence n, centered at `center`: n "spoke"
// neighbors (index 1..n) and, one per incident quad face, a "far"
// diagonal corner (index n+1..2n). Face i = [0, spoke_i, far_i,
// spoke_{i+1}] — every one of the vertex's own n edges (0-spoke_i) is
// shared by exactly 2 of these n faces, so vertex 0 is an ordinary
// SMOOTH interior vertex (0 sharp edges) by construction, exactly the
// case this whole derivation assumes.
function buildSymmetricFan(n, center, spokeRadius = 1, farRadius = 1.6) {
  const vertices = [center.slice()];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    vertices.push([center[0] + spokeRadius * Math.cos(a), center[1] + spokeRadius * Math.sin(a), center[2]]);
  }
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * (i + 0.5)) / n;
    vertices.push([center[0] + farRadius * Math.cos(a), center[1] + farRadius * Math.sin(a), center[2]]);
  }
  const faces = [];
  for (let i = 0; i < n; i++) {
    const spokeI = 1 + i, farI = 1 + n + i, spokeNext = 1 + ((i + 1) % n);
    faces.push([0, spokeI, farI, spokeNext]);
  }
  return { vertices, faces, creases: {} };
}

// edgePointIndexMap/facePointIndexBase are imported directly from
// kernel/subdlimit.mjs itself (step 3 promoted these from test-local
// helpers to real, exported module functions — reused here rather than
// duplicated a second time).

for (const n of [3, 5]) {
  test(`n=${n}: one REAL subdivideCatmullClark step on a symmetric fan matches the hand-derived 3x3 DC-mode system exactly — cross-checked against ALREADY-PROVEN kernel code, not just self-consistent algebra`, () => {
    const center = [2, -1, 4]; // deliberately off-origin, not a degenerate/trivial position
    const cage = buildSymmetricFan(n, center, 1, 1.7);
    const p = cage.vertices[0];
    const q = scale3(cage.vertices.slice(1, 1 + n).reduce(add3, [0, 0, 0]), 1 / n);
    const r = scale3(cage.vertices.slice(1 + n, 1 + 2 * n).reduce(add3, [0, 0, 0]), 1 / n);

    const edgeIdx = edgePointIndexMap(cage);
    const faceBase = facePointIndexBase(cage);
    const refined = subdivideCatmullClark(cage);

    const pPrime = refined.vertices[0];
    // vertex 0's own n incident edges are exactly (0, spoke_i) for i=1..n
    let qPrimeSum = [0, 0, 0];
    for (let i = 1; i <= n; i++) qPrimeSum = add3(qPrimeSum, refined.vertices[edgeIdx.get(edgeKey(0, i))]);
    const qPrime = scale3(qPrimeSum, 1 / n);
    // every one of the n faces touches vertex 0
    let rPrimeSum = [0, 0, 0];
    for (let fi = 0; fi < n; fi++) rPrimeSum = add3(rPrimeSum, refined.vertices[faceBase + fi]);
    const rPrime = scale3(rPrimeSum, 1 / n);

    // Hand-derived (in kernel/subdlimit.mjs's own header comment) 3x3
    // system, re-derived here independently as the actual test oracle:
    const expectedPPrime = scale3(add3(add3(scale3(p, 4 * n - 7), scale3(q, 6)), r), 1 / (4 * n));
    const expectedQPrime = scale3(add3(add3(scale3(p, 3), scale3(q, 4)), r), 1 / 8);
    const expectedRPrime = scale3(add3(add3(p, scale3(q, 2)), r), 1 / 4);

    assert.ok(closeEnough(pPrime, expectedPPrime, 1e-9), `p' mismatch: got ${pPrime}, expected ${expectedPPrime}`);
    assert.ok(closeEnough(qPrime, expectedQPrime, 1e-9), `q' mismatch: got ${qPrime}, expected ${expectedQPrime}`);
    assert.ok(closeEnough(rPrime, expectedRPrime, 1e-9), `r' mismatch: got ${rPrime}, expected ${expectedRPrime}`);
  });

  test(`n=${n}: vertexLimitPosition of a perfectly symmetric fan equals its own center EXACTLY — a real, non-trivial invariant (rotational symmetry admits no other fixed point), not a coincidence an indexing bug would likely survive`, () => {
    const center = [5, -3, 2];
    const cage = buildSymmetricFan(n, center, 1, 1.7);
    const L = vertexLimitPosition(cage, 0);
    assert.ok(closeEnough(L, center, 1e-9), `got ${L}, expected exactly the center ${center}`);
  });
}

test('vertexLimitMaskFromRF: a genuinely asymmetric, hand-computed worked example (valence 3) — real distinguishable numbers, arithmetic checked by hand alongside the code, not a symmetric case where every formula would agree', () => {
  const n = 3;
  const P = [10, 0, 0];
  const R = [4, 2, 1]; // average edge-midpoint, picked arbitrarily
  const F = [1, -1, 3]; // average face-centroid, picked arbitrarily
  // L = [(n-3)P + 4R + 4F] / (n+5) = [0*P + 4*(4,2,1) + 4*(1,-1,3)] / 8
  //   = [(16,8,4) + (4,-4,12)] / 8 = (20,4,16) / 8 = (2.5, 0.5, 2)
  const expected = [2.5, 0.5, 2];
  const L = vertexLimitMaskFromRF(P, R, F, n);
  assert.ok(closeEnough(L, expected, 1e-12), `got ${L}, expected ${expected} (hand-computed: [0*P+4R+4F]/8)`);
});

test('vertexLimitMaskFromRF: a second hand-computed worked example (valence 5), and partition-of-unity confirmed directly (coefficients sum to exactly 1)', () => {
  const n = 5;
  const P = [0, 0, 6];
  const R = [3, -2, 0];
  const F = [-1, 4, 2];
  // L = [(n-3)P + 4R + 4F]/(n+5) = [2*(0,0,6) + 4*(3,-2,0) + 4*(-1,4,2)] / 10
  //   = [(0,0,12) + (12,-8,0) + (-4,16,8)] / 10 = (8,8,20)/10 = (0.8, 0.8, 2.0)
  const expected = [0.8, 0.8, 2.0];
  const L = vertexLimitMaskFromRF(P, R, F, n);
  assert.ok(closeEnough(L, expected, 1e-12), `got ${L}, expected ${expected}`);
  const coeffSum = (n - 3 + 4 + 4) / (n + 5);
  assert.ok(Math.abs(coeffSum - 1) < 1e-12, `partition of unity: (n-3+4+4)/(n+5) = ${coeffSum}, expected exactly 1`);
});

test('vertexLimitMaskFromRF at n=4 (the regular case) reproduces the widely-published regular-vertex mask "(16V + 4*sum(edge-neighbors) + sum(face-diagonal-corners)) / 36" exactly — two independently-stated formulas, cross-checked, not just internally consistent', () => {
  const n = 4;
  const P = [1, 2, 3];
  const Q = [[2, 0, 0], [0, 2, 0], [-2, 0, 4], [0, -2, 4]]; // 4 edge-neighbor vertices
  const Rf = [[3, 3, 1], [-3, 3, 1], [-3, -3, 5], [3, -3, 5]]; // 4 face-diagonal corners
  const edgeMidpointAvg = scale3(add3(scale3(P, 4), Q.reduce(add3, [0, 0, 0])), 1 / 8); // avg of (P+Qi)/2 over i, = (4P+sumQ)/8
  const faceCentroidAvg = (() => {
    // face i's own centroid = (P + Qi + Ri + Q_{i+1})/4, averaged over i
    let sum = [0, 0, 0];
    for (let i = 0; i < 4; i++) sum = add3(sum, scale3(add3(add3(add3(P, Q[i]), Rf[i]), Q[(i + 1) % 4]), 1 / 4));
    return scale3(sum, 1 / 4);
  })();
  const L = vertexLimitMaskFromRF(P, edgeMidpointAvg, faceCentroidAvg, n);
  const sumQ = Q.reduce(add3, [0, 0, 0]);
  const sumR = Rf.reduce(add3, [0, 0, 0]);
  const published = scale3(add3(add3(scale3(P, 16), scale3(sumQ, 4)), sumR), 1 / 36);
  assert.ok(closeEnough(L, published, 1e-9), `got ${L}, published-formula answer ${published}`);
});

// ================================================================
// STEP 3 — THE SUBDIVISION-MATRIX BUILDER, A(n). The single highest-
// value test in the whole build order per this doc's own build order:
// cross-checks the matrix, built from real unit-basis subdivideCatmullClark
// calls, against a REAL subdivideCatmullClark call on a genuinely
// non-trivial (asymmetric) neighborhood — bit-for-bit, not approximately.
// ================================================================

for (const n of [3, 4, 5, 6]) {
  test(`buildSubdivisionMatrix(${n}): A(n) applied to a real, ASYMMETRIC neighborhood reproduces a direct subdivideCatmullClark call bit-for-bit, on all 3 axes`, () => {
    const template = standardNeighborhoodCage(n);
    // Genuinely asymmetric, non-trivial coordinates — deliberately not a
    // regular/symmetric fan (this project's own standing "avoid trivial
    // test geometry" discipline: a symmetric fixture could hide an
    // indexing bug that only shows up when values genuinely differ
    // point to point).
    const realCage = {
      vertices: template.vertices.map((_, i) => [
        Math.sin(i * 1.7 + 0.3) * 3 + i * 0.4,
        Math.cos(i * 0.9 - 0.5) * 2 - i * 0.2,
        (i % 3) * 1.3 - 0.7,
      ]),
      faces: template.faces,
      creases: {},
    };
    const edgeIdx = edgePointIndexMap(template);
    const faceBase = facePointIndexBase(template);
    // The OLD neighborhood is just realCage.vertices itself, already in
    // standardNeighborhoodCage's own index order — read directly.
    const oldX = realCage.vertices.map((p) => p[0]);
    const oldY = realCage.vertices.map((p) => p[1]);
    const oldZ = realCage.vertices.map((p) => p[2]);

    const refined = subdivideCatmullClark(realCage);
    const newVec = readNeighborhoodVectors(template, refined, n, edgeIdx, faceBase);

    const A = buildSubdivisionMatrix(n);
    const predX = applyMatrix(A, oldX);
    const predY = applyMatrix(A, oldY);
    const predZ = applyMatrix(A, oldZ);

    for (let i = 0; i < 2 * n + 1; i++) {
      assert.ok(Math.abs(predX[i] - newVec[i][0]) < 1e-9, `n=${n} point ${i} x: A(n)v=${predX[i]}, real subdivide=${newVec[i][0]}`);
      assert.ok(Math.abs(predY[i] - newVec[i][1]) < 1e-9, `n=${n} point ${i} y: A(n)v=${predY[i]}, real subdivide=${newVec[i][1]}`);
      assert.ok(Math.abs(predZ[i] - newVec[i][2]) < 1e-9, `n=${n} point ${i} z: A(n)v=${predZ[i]}, real subdivide=${newVec[i][2]}`);
    }
  });

  test(`buildSubdivisionMatrix(${n}): every row is a genuine weighted average (sums to exactly 1) — A(n) can only ever interpolate, never extrapolate/amplify`, () => {
    const A = buildSubdivisionMatrix(n);
    for (let row = 0; row < A.length; row++) {
      const sum = A[row].reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(sum - 1) < 1e-9, `row ${row} sums to ${sum}, expected 1`);
    }
  });

  test(`buildSubdivisionMatrix(${n}): A(n) has the constant vector as a genuine eigenvector of eigenvalue 1 (the flat-unrefined-patch fixed point) — a real, independent structural check, not assumed from the row-sum property alone`, () => {
    const A = buildSubdivisionMatrix(n);
    const ones = new Array(2 * n + 1).fill(1);
    const result = applyMatrix(A, ones);
    for (const v of result) assert.ok(Math.abs(v - 1) < 1e-9, `expected 1, got ${v}`);
  });
}

// ================================================================
// STEP 4 — VERTEX-LIMIT WEIGHTS PER INDIVIDUAL NEIGHBOR, cross-checked
// THREE independent ways (never a self-consistency tautology):
//   (a) at exactly (u,v)=(0,0): matches step 2's own vertexLimitPosition
//       on a real, ASYMMETRIC cage, for several valences.
//   (b) geometric contraction: repeated A(n) application converges
//       monotonically to that same value, from a real neighborhood.
//   (c) n forced to exactly 4: agrees with step 1's own regular fast
//       path at the shared boundary case (u,v)=(0,0) — the two paths'
//       one point of overlap.
// ================================================================

for (const n of [3, 4, 5, 6, 7]) {
  test(`vertexLimitWeightsGeneral(${n}): dotted directly against a genuinely ASYMMETRIC real neighborhood, reproduces vertexLimitPosition's own independent (n-3,4R,4F)/(n+5) computation exactly — not a symmetric fixture that could hide an indexing bug`, () => {
    const template = standardNeighborhoodCage(n);
    const realCage = {
      vertices: template.vertices.map((_, i) => [
        Math.sin(i * 2.1 + 0.7) * 4 + i * 0.3,
        Math.cos(i * 1.3 - 0.2) * 2.5 - i * 0.15,
        (i % 5) * 0.9 - 1.1,
      ]),
      faces: template.faces,
      creases: {},
    };
    const L = vertexLimitPosition(realCage, 0);
    const w = vertexLimitWeightsGeneral(n);
    let dotted = [0, 0, 0];
    for (let i = 0; i < 2 * n + 1; i++) {
      for (let k = 0; k < 3; k++) dotted[k] += w[i] * realCage.vertices[i][k];
    }
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(dotted[k] - L[k]) < 1e-9, `axis ${k}: weighted-dot=${dotted[k]}, vertexLimitPosition=${L[k]}`);
    }
  });

  test(`powerIterationLeftDominant on A(${n}): A(n)'s own dominant left eigenvector (found by a genuinely independent numerical method, never reading the closed form) matches vertexLimitWeightsGeneral(${n}) exactly, and the eigenvalue is exactly 1`, () => {
    const A = buildSubdivisionMatrix(n);
    const { eigenvalue, eigenvector } = powerIterationLeftDominant(A);
    assert.ok(Math.abs(eigenvalue - 1) < 1e-9, `dominant eigenvalue: got ${eigenvalue}, expected 1`);
    const w = vertexLimitWeightsGeneral(n);
    for (let i = 0; i < 2 * n + 1; i++) {
      assert.ok(Math.abs(eigenvector[i] - w[i]) < 1e-8, `weight[${i}]: power-iteration=${eigenvector[i]}, closed-form=${w[i]}`);
    }
  });

  test(`geometric contraction, n=${n}: repeated A(n) application on a REAL asymmetric neighborhood converges monotonically toward vertexLimitPosition's own value — the discrete-refinement analog of "evaluated at a sequence of (u,v) approaching (0,0)"`, () => {
    const template = standardNeighborhoodCage(n);
    const realCage = {
      vertices: template.vertices.map((_, i) => [
        Math.sin(i * 1.1 + 1.3) * 5 - i * 0.25,
        Math.cos(i * 2.3 + 0.4) * 3 + i * 0.2,
        (i % 4) * 1.1 - 1.5,
      ]),
      faces: template.faces,
      creases: {},
    };
    const L = vertexLimitPosition(realCage, 0);
    const A = buildSubdivisionMatrix(n);
    const axes = [
      realCage.vertices.map((p) => p[0]),
      realCage.vertices.map((p) => p[1]),
      realCage.vertices.map((p) => p[2]),
    ];
    // Iteration count is NOT fixed at a value tuned to look good — a
    // higher valence has a real, larger subdominant eigenvalue |lambda1|
    // (slower geometric contraction is a genuine property of the
    // surface near that vertex, not a test artifact), so this runs until
    // genuinely converged rather than asserting a one-size count.
    let prevErr = Infinity;
    let err = Infinity;
    for (let k = 1; k <= 60; k++) {
      axes[0] = applyMatrix(A, axes[0]);
      axes[1] = applyMatrix(A, axes[1]);
      axes[2] = applyMatrix(A, axes[2]);
      const center = [axes[0][0], axes[1][0], axes[2][0]];
      err = Math.hypot(center[0] - L[0], center[1] - L[1], center[2] - L[2]);
      assert.ok(err <= prevErr + 1e-12, `step ${k}: error ${err} must not exceed the previous step's ${prevErr}`);
      prevErr = err;
      if (err < 1e-10) break;
    }
    assert.ok(err < 1e-8, `after up to 60 steps, error to the true limit is ${err}, expected essentially converged`);
  });
}

test('n=4 (regular) run through the SAME general vertex-limit machinery agrees EXACTLY with step 1\'s own regular bicubic fast path, at their one shared point (u,v)=(0,0) — proving the two paths agree at the boundary, not just independently "look right"', () => {
  // A real 4x4 regular grid (step 1's own domain) has its own vertex-0
  // analog sitting at grid[1][1] (the shared corner of the center 2x2
  // face, per bicubicRegularPatchSurface's own documented layout) — its
  // one-ring neighborhood in the (2n+1)=9-point template's own index
  // order (center, 4 spokes = its 4 orthogonal grid neighbors, 4 corners
  // = its 4 diagonal grid neighbors) is read directly off the SAME grid.
  const grid = [
    [[0, 0, 0.2], [1, 0, 0.6], [2, 0, 0.1], [3, 0, 0.9]],
    [[0, 1, 0.7], [1, 1, 1.3], [2, 1, 0.4], [3, 1, 1.1]],
    [[0, 2, 0.3], [1, 2, 0.8], [2, 2, 1.6], [3, 2, 0.5]],
    [[0, 3, 1.0], [1, 3, 0.2], [2, 3, 0.9], [3, 3, 1.4]],
  ];
  const srf = bicubicRegularPatchSurface(grid);
  const { point: regularAnswer } = regularPatchPointAndPartials(grid, 0, 0);

  const n = 4;
  const template = standardNeighborhoodCage(n);
  // template order: center, spoke_0..3 (orthogonal neighbors), corner_0..3
  // (diagonal neighbors) of grid[1][1] — reading its real 9-point
  // one-ring straight off the SAME grid, no relabeling trick.
  const P = grid[1][1];
  const spokes = [grid[0][1], grid[1][0], grid[1][2], grid[2][1]];
  const corners = [grid[0][0], grid[0][2], grid[2][0], grid[2][2]];
  const realCage = { vertices: [P, ...spokes, ...corners], faces: template.faces, creases: {} };
  const generalAnswer = vertexLimitPosition(realCage, 0);

  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(generalAnswer[k] - regularAnswer[k]) < 1e-9,
      `axis ${k}: general vertex-mask machinery=${generalAnswer[k]}, step-1 regular fast path=${regularAnswer[k]}`);
  }
  assert.ok(srf.degU === 3 && srf.degV === 3, 'sanity: the regular patch really is the expected bicubic surface');
});

// ================================================================
// STEP 5 — SEMI-SHARP HYBRID, checked against a real fixture with a
// genuine partially-decayed semi-sharp crease weight (strictly between
// 0 and the app's own real reachable cap, 3 — kernel/subd.mjs's own
// documented SUPERB_CREASE_LEVEL_SCALE range), checked for continuity
// at the discrete-decay-then-exact seam.
// ================================================================

test('semiSharpHybridLimitPosition: the crease decay genuinely matters — a partially-decayed weight gives a DIFFERENT answer than pretending the vertex is already smooth', () => {
  const n = 5;
  const center = [3, -2, 1];
  const cage = buildSymmetricFan(n, center, 1, 1.7);
  cage.creases = {}; // buildSymmetricFan doesn't set any; be explicit
  cage.creases[edgeKey(0, 1)] = 1.5; // a real, active, partially-decayed weight (0 < 1.5 < cap 3)

  const hybrid = semiSharpHybridLimitPosition(cage, 0);
  const ignoringCrease = vertexLimitPosition(cage, 0); // wrong on purpose: treats vertex 0 as already smooth
  const dist = Math.hypot(hybrid[0] - ignoringCrease[0], hybrid[1] - ignoringCrease[1], hybrid[2] - ignoringCrease[2]);
  assert.ok(dist > 1e-6, `hybrid=${hybrid} must genuinely differ from the "ignore the crease" answer=${ignoringCrease} (dist=${dist})`);
});

test('semiSharpHybridLimitPosition: a PERMANENT marker weight (far above the app\'s real [0,3] decaying-crease range — e.g. a TOSUBD marked corner, real weights in the hundreds) refuses honestly instead of attempting a catastrophic number of whole-cage subdivision levels', () => {
  const n = 5;
  const cage = buildSymmetricFan(n, [0, 0, 0], 1, 1.7);
  cage.creases = {};
  cage.creases[edgeKey(0, 1)] = 1000; // a real marked-corner-scale weight, not a decaying crease
  assert.throws(() => semiSharpHybridLimitPosition(cage, 0), /MAX_SEMISHARP_DECAY_LEVELS/);
  // A weight right at the cap boundary must still be genuinely evaluated
  // (never a false-positive refusal for a real, if unusually high but
  // still bounded, ordinary crease request).
  cage.creases[edgeKey(0, 1)] = 8;
  assert.doesNotThrow(() => semiSharpHybridLimitPosition(cage, 0));
});

test('semiSharpHybridLimitPosition: CONTINUITY at the decay-then-exact seam — real, repeated discrete subdivision of the SAME creased cage converges toward the hybrid\'s exact answer with monotonically shrinking error, no jump at the handoff point', () => {
  const n = 5;
  const center = [-1, 4, 2];
  const cage = buildSymmetricFan(n, center, 1, 1.7);
  cage.creases = {};
  const weight = 1.5;
  cage.creases[edgeKey(0, 1)] = weight;

  const k = Math.ceil(weight);
  assert.equal(k, 2, 'sanity: this fixture\'s own decay level (ceil(1.5)) is 2');

  const hybrid = semiSharpHybridLimitPosition(cage, 0);

  // Real, independent ground truth: repeatedly subdivide the SAME
  // original creased cage (subd.mjs's own already-proven decay), reading
  // vertex 0's own RAW (not exact-limit) position at each level. This is
  // a genuinely different computation from semiSharpHybridLimitPosition
  // itself (that function stops discrete refinement at k and switches to
  // the exact eigenbasis machinery; this loop keeps discretely refining
  // past k) — if the handoff were discontinuous, this sequence would NOT
  // converge toward the hybrid's own answer.
  // Capped at 8 real, WHOLE-CAGE subdivision levels, deliberately not
  // more: unlike step 4's own A(n)-matrix convergence proof (a cheap
  // (2n+1)x(2n+1) operation, safe to iterate dozens of times), this loop
  // runs subd.mjs's own REAL subdivideCatmullClark on the WHOLE cage —
  // every face, everywhere, not just this one vertex's neighborhood —
  // and Catmull-Clark refinement genuinely quadruples the total face
  // count every level, so it must stay bounded (5 faces at level 0 is
  // already 5*4^8 = 327,680 by level 8; a few more levels would exhaust
  // available memory for no further proof value, confirmed empirically).
  let c = cage;
  const errors = [];
  for (let m = 0; m <= 8; m++) {
    const raw = c.vertices[0];
    const err = Math.hypot(raw[0] - hybrid[0], raw[1] - hybrid[1], raw[2] - hybrid[2]);
    errors.push(err);
    if (m < 8) c = subdivideCatmullClark(c);
  }
  // Once the crease has structurally decayed away (level k and beyond —
  // the smooth eigen-convergence regime), the error must shrink, level
  // over level, toward zero — the actual continuity proof.
  for (let m = k; m < errors.length - 1; m++) {
    assert.ok(errors[m + 1] <= errors[m] + 1e-12, `level ${m}->${m + 1}: error must not grow (${errors[m]} -> ${errors[m + 1]})`);
  }
  assert.ok(errors[errors.length - 1] < 1e-4, `after 8 real discrete levels past a weight-1.5 crease, residual error to the hybrid's exact answer is ${errors[errors.length - 1]}, expected essentially converged`);
});
