// The gate for ToNURBS step 1: a regular face read out as a bicubic patch
// is the limit surface. Steps 2-4 build on this decomposition, so if it
// fails nothing downstream is valid.
//
// What is being gated. Not "is the theorem true" — a regular
// Catmull-Clark face converging to a uniform bicubic B-spline patch is
// the standard result. What a numerical test catches is the part that
// can silently be wrong: whether the 4x4 stencil is read out in the
// right order and orientation, and whether the patch's own parameter
// domain lines up with the face's own corners the way the caller
// assumes. A transposed or rotated stencil still produces a perfectly
// valid bicubic patch through perfectly real points — it just isn't the
// limit surface, and nothing downstream would notice.
//
// Ground truth, independent of the thing under test. Every expected
// value here comes from vertexLimitPosition — the Halstead/Kass/DeRose
// mask, built in step 2 of this same module so it could serve as ground
// truth, and itself verified three independent ways. It shares no code path with
// bicubicRegularPatchSurface. The correspondences used:
//
//   - the patch's four parametric corners are the four face vertices'
//     own limit positions;
//   - subdividing once, the face's own face point is a vertex of the
//     refined cage whose limit position is the same limit surface at
//     that face's center — i.e. the patch at (0.5, 0.5);
//   - subdividing twice, each of the four sub-face centers gives a
//     off-knot interior sample at a quarter parameter. Those
//     matter most: this patch's only knots are at 0 and 1, so a test
//     that only ever sampled corners and the exact center would use a
//     fixture too clean to discriminate.
import test from 'node:test';
import assert from 'node:assert/strict';
import { subdivideCatmullClark, buildTopology } from '../kernel/subd.mjs';
import { superbBoxCage, superbTorusCage } from '../kernel/subdprimitives.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { curvePoint } from '../kernel/curve.mjs';
import { extractBorderCurves } from '../kernel/isocurve.mjs';
import {
  isRegularFace, regularFaceStencil, regularFaceToPatch, bicubicRegularPatchSurface,
  regularPatchPointAndPartials, vertexLimitPosition, subdToPatches, patchBoundaryRow,
} from '../kernel/subdlimit.mjs';

// A box cage subdivided twice: its eight original corners stay valence 3
// (extraordinary) forever, so the cage carries both regular
// faces and irregular ones — exactly what a classifier needs
// to be tested against, rather than a fixture where every face passes.
function twiceSubdividedBox() {
  let cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  cage = subdivideCatmullClark(cage);
  cage = subdivideCatmullClark(cage);
  return cage;
}

function firstRegularFace(cage, ctx) {
  for (let fi = 0; fi < cage.faces.length; fi++) if (isRegularFace(cage, fi, ctx)) return fi;
  return -1;
}

function near(a, b, tol, what) {
  const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  assert.ok(d < tol, `${what}: expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}, distance ${d}`);
}

test('the fixture discriminates: some faces regular, some not', () => {
  const cage = twiceSubdividedBox();
  const ctx = buildTopology(cage);
  let regular = 0, irregular = 0;
  for (let fi = 0; fi < cage.faces.length; fi++) (isRegularFace(cage, fi, ctx) ? regular++ : irregular++);
  assert.ok(regular > 0, 'expected at least one regular face');
  // Exactly the 8 original box corners stay valence 3, and each is
  // touched by 3 faces — so 24 faces are irregular and the rest are not.
  assert.equal(irregular, 24, `expected exactly 24 corner-touching faces to be irregular, got ${irregular}`);
  assert.equal(regular, cage.faces.length - 24);
});

test('a face touching an extraordinary vertex is refused, and refuses to hand out a stencil', () => {
  const cage = twiceSubdividedBox();
  const ctx = buildTopology(cage);
  const bad = cage.faces.findIndex((_, fi) => !isRegularFace(cage, fi, ctx));
  assert.ok(bad >= 0);
  assert.throws(() => regularFaceStencil(cage, bad, ctx), /not regular/);
});

test('a crease anywhere in the 3x3 block disqualifies the face', () => {
  const cage = twiceSubdividedBox();
  const ctx = buildTopology(cage);
  const fi = firstRegularFace(cage, ctx);
  assert.ok(fi >= 0);
  // Crease an edge of the outer ring — not one of this face's own four
  // edges. It still changes the subdivision the stencil converges under,
  // so it must still disqualify.
  const block = new Set();
  for (const v of cage.faces[fi]) for (const bf of ctx.vertexFaces[v]) block.add(bf);
  const own = new Set(cage.faces[fi].map((v, i, arr) => `${Math.min(v, arr[(i + 1) % 4])}_${Math.max(v, arr[(i + 1) % 4])}`));
  let creasedOuter = null;
  for (const bf of block) {
    const f = cage.faces[bf];
    for (let c = 0; c < f.length && !creasedOuter; c++) {
      const p = f[c], q = f[(c + 1) % f.length];
      const key = `${Math.min(p, q)}_${Math.max(p, q)}`;
      if (!own.has(key)) creasedOuter = key;
    }
    if (creasedOuter) break;
  }
  assert.ok(creasedOuter, 'fixture should contain an outer-ring edge');
  const creased = { ...cage, creases: { [creasedOuter]: 2 } };
  assert.equal(isRegularFace(creased, fi, buildTopology(creased)), false);
});

test('the stencil is 16 real points with the face at its center 2x2', () => {
  const cage = twiceSubdividedBox();
  const ctx = buildTopology(cage);
  const fi = firstRegularFace(cage, ctx);
  const grid = regularFaceStencil(cage, fi, ctx);
  const [a, b, c, d] = cage.faces[fi];
  assert.deepEqual(grid[1][1], cage.vertices[a]);
  assert.deepEqual(grid[2][1], cage.vertices[b]);
  assert.deepEqual(grid[2][2], cage.vertices[c]);
  assert.deepEqual(grid[1][2], cage.vertices[d]);
  const flat = grid.flat();
  assert.equal(flat.length, 16);
  for (const p of flat) assert.ok(p && p.length >= 3 && p.every(Number.isFinite));
  // All 16 distinct: a stencil that accidentally read the same neighbor
  // twice would still evaluate, and would still be wrong.
  const keys = new Set(flat.map((p) => p.slice(0, 3).map((n) => n.toFixed(9)).join(',')));
  assert.equal(keys.size, 16, 'expected 16 distinct stencil points');
});

test('Gate: the patch reproduces the exact limit surface at its four corners', () => {
  const cage = twiceSubdividedBox();
  const ctx = buildTopology(cage);
  const fi = firstRegularFace(cage, ctx);
  const grid = regularFaceStencil(cage, fi, ctx);
  const [a, b, c, d] = cage.faces[fi];
  const corners = [[0, 0, a], [1, 0, b], [1, 1, c], [0, 1, d]];
  for (const [u, v, vIdx] of corners) {
    const got = regularPatchPointAndPartials(grid, u, v).point;
    const want = vertexLimitPosition(cage, vIdx, ctx);
    near(got, want, 1e-12, `patch(${u},${v}) vs exact limit of vertex ${vIdx}`);
  }
});

test('Gate: the patch reproduces the exact limit surface at the face center', () => {
  const cage = twiceSubdividedBox();
  const ctx = buildTopology(cage);
  const fi = firstRegularFace(cage, ctx);
  const grid = regularFaceStencil(cage, fi, ctx);
  // subdivideCatmullClark appends edge points then face points, so the
  // face point of face fi is at nVerts + nEdges + fi in the refined cage.
  const nVerts = cage.vertices.length, nEdges = ctx.edgeMap.size;
  const refined = subdivideCatmullClark(cage);
  const facePointIdx = nVerts + nEdges + fi;
  const want = vertexLimitPosition(refined, facePointIdx);
  const got = regularPatchPointAndPartials(grid, 0.5, 0.5).point;
  near(got, want, 1e-12, 'patch(0.5,0.5) vs the refined face point\'s own exact limit');
});

test('Gate: the patch matches at off-knot interior quarter points', () => {
  const cage = twiceSubdividedBox();
  const ctx = buildTopology(cage);
  const fi = firstRegularFace(cage, ctx);
  const grid = regularFaceStencil(cage, fi, ctx);

  const nVerts1 = cage.vertices.length, nEdges1 = ctx.edgeMap.size;
  const r1 = subdivideCatmullClark(cage);
  const ctx1 = buildTopology(r1);
  const r2 = subdivideCatmullClark(r1);
  const nVerts2 = r1.vertices.length, nEdges2 = ctx1.edgeMap.size;

  // Level-1 sub-faces are pushed in corner order, all-quad cage, so face
  // fi's own four children are 4*fi .. 4*fi+3, child c being the one at
  // the face's corner c. Each child's own level-2 face point therefore
  // sits at the center of that quarter of the original face.
  const quarterOf = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]];
  for (let c = 0; c < 4; c++) {
    const childFace = 4 * fi + c;
    const facePointIdx = nVerts2 + nEdges2 + childFace;
    const want = vertexLimitPosition(r2, facePointIdx);
    const [u, v] = quarterOf[c];
    const got = regularPatchPointAndPartials(grid, u, v).point;
    near(got, want, 1e-12, `patch(${u},${v}) vs the twice-refined sub-face point's own exact limit`);
  }
});

test('the patch is a real surface, not a flat or degenerate one', () => {
  const cage = twiceSubdividedBox();
  const ctx = buildTopology(cage);
  const fi = firstRegularFace(cage, ctx);
  const srf = regularFaceToPatch(cage, fi, ctx);
  assert.equal(srf.degU, 3);
  assert.equal(srf.degV, 3);
  assert.equal(srf.ctrlNet.length, 4);
  const { su, sv } = regularPatchPointAndPartials(regularFaceStencil(cage, fi, ctx), 0.5, 0.5);
  assert.ok(Math.hypot(su[0], su[1], su[2]) > 1e-6, 'the U partial must not vanish');
  assert.ok(Math.hypot(sv[0], sv[1], sv[2]) > 1e-6, 'the V partial must not vanish');
});

// Isolate and emit. The claim being tested is not "it
// runs" but three specific properties: every emitted patch is exactly
// the limit surface over its own face, the unconverted region
// shrinks by 4x per isolation level, and a cage that is regular
// everywhere converts completely with no leftovers at all.
test('a fully regular cage (a torus) converts completely, with zero uncovered faces', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 8);
  const r = subdToPatches(cage, { maxIsolation: 0 });
  assert.equal(r.patches.length, cage.faces.length, 'one exact patch per face');
  assert.equal(r.uncovered.length, 0);
  assert.equal(r.uncoveredFraction, 0);
  assert.equal(r.levelsUsed, 0, 'no isolation needed when nothing is extraordinary');
});

test('a box cage isolates: the unconverted region shrinks by exactly 4x per level', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const fracs = [2, 3, 4].map((lv) => subdToPatches(cage, { maxIsolation: lv }).uncoveredFraction);
  for (let i = 1; i < fracs.length; i++) {
    assert.ok(Math.abs(fracs[i] * 4 - fracs[i - 1]) < 1e-12, `level ${i + 2}: expected exactly a quarter of ${fracs[i - 1]}, got ${fracs[i]}`);
  }
  // Eight valence-3 corners, three faces each, is what stays live — the
  // count must stay fixed as levels rise while the area shrinks. A count
  // that grew would mean isolation was failing to converge.
  for (const lv of [2, 3, 4]) assert.equal(subdToPatches(cage, { maxIsolation: lv }).uncovered.length, 24);
});

test('every emitted patch is exactly the limit surface over its own face', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const r = subdToPatches(cage, { maxIsolation: 2 });
  assert.ok(r.patches.length > 0);
  // Re-derive each patch's own level's cage independently, and check the
  // patch's four corners against the vertex-limit mask there — the same
  // ground truth the step-1 gate uses, applied to the whole emitted set
  // rather than to one hand-picked face.
  const byLevel = new Map();
  let c = cage;
  for (let lv = 0; lv <= 2; lv++) { byLevel.set(lv, c); c = subdivideCatmullClark(c); }
  let checked = 0;
  for (const p of r.patches) {
    const lvCage = byLevel.get(p.level);
    const ctx = buildTopology(lvCage);
    const face = lvCage.faces[p.faceIndex];
    const grid = regularFaceStencil(lvCage, p.faceIndex, ctx);
    const corners = [[0, 0, face[0]], [1, 0, face[1]], [1, 1, face[2]], [0, 1, face[3]]];
    for (const [u, v, vIdx] of corners) {
      near(regularPatchPointAndPartials(grid, u, v).point, vertexLimitPosition(lvCage, vIdx, ctx), 1e-12,
        `emitted patch (level ${p.level}, face ${p.faceIndex}) corner (${u},${v})`);
      checked++;
    }
  }
  assert.ok(checked >= 100, `expected a real sample of patches, checked ${checked} corners`);
});

test('the leftover faces are named with the extraordinary vertex responsible, and its exact limit', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const r = subdToPatches(cage, { maxIsolation: 2 });
  assert.ok(r.uncovered.length > 0);
  for (const u of r.uncovered) {
    assert.ok(u.extraordinary.length >= 1, 'a leftover face must name why it is left over');
    for (const e of u.extraordinary) {
      assert.ok(Number.isInteger(e.vertex));
      assert.ok(e.limitPosition.length === 3 && e.limitPosition.every(Number.isFinite));
    }
  }
  // Isolation's whole point: after enough levels each leftover touches
  // exactly one extraordinary vertex, so a cap has a single point to be
  // built around rather than several.
  assert.ok(r.uncovered.every((u) => u.extraordinary.length === 1), 'each leftover should be down to a single extraordinary corner');
});

test('subdToPatches never mutates the cage it was given', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const before = JSON.stringify(cage);
  subdToPatches(cage, { maxIsolation: 3 });
  assert.equal(JSON.stringify(cage), before);
});

// Do two patches sharing a cage edge share their boundary control rows
// exactly? If they do, ToNURBS emits multi-face solids whose shared edges are
// exactly coincident rather than tolerance-fitted, which is the property a
// Boolean stitch across such an edge relies on.
//
// Asked of the control nets, not of sampled points, and that is the
// stronger question anyway: two patches agreeing along a shared edge when
// sampled only says the curves coincide; agreeing in their control rows
// says they carry the identical representation, which is what a Boolean
// stitch across that edge would rely on. Patches are emitted clamped, so
// each one's boundary row is a real Bezier control row of the shared
// edge curve — and a clamped Bezier representation of a curve is unique,
// so the two patches' rows have no freedom to differ.
//
// How exactly, measured rather than argued, because the argument above is
// about the curve and the question asked is about the numbers. The two are
// not the same claim. Two adjacent patches derive the same shared row by
// running the identical algebra in a different order — the U pass and the V
// pass swap roles when the neighboring face's own loop starts elsewhere —
// so the rows are algebraically identical and can still round differently in
// the last bit. On this fixture 108 of 120 adjacent pairs come out bit-
// identical and 12 do not; subdivide the cage twice first and none of 120 do,
// while the worst disagreement anywhere is 1.8e-15. So the durable invariant
// is agreement to rounding, asserted below alongside the bit count, and the
// stitch conclusion survives it intact: 1.8e-15 is not a tolerance fit, it is
// the same number written twice. The second assertion is checked from both
// sides so neither a regression to a real tolerance gap nor a silent claim of
// exactness can pass. See test/subd-cap-patch.test.mjs for the twice-
// subdivided control.
test('two adjacent regular patches share whole control lines — the same numbers, to rounding, and mostly to the bit', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 4); // dense enough to have regular interior faces
  const ctx = buildTopology(cage);
  const regular = cage.faces.map((_, i) => i).filter((i) => isRegularFace(cage, i, ctx));
  assert.ok(regular.length > 8, `fixture must have plenty of regular faces, got ${regular.length}`);
  const regSet = new Set(regular);
  let pair = null;
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length === 2 && regSet.has(e.faces[0]) && regSet.has(e.faces[1])) { pair = e.faces; break; }
  }
  assert.ok(pair, 'fixture must contain two regular faces sharing an edge');

  const A = regularFaceToPatch(cage, pair[0], ctx);
  const B = regularFaceToPatch(cage, pair[1], ctx);
  const key = (pt) => pt.slice(0, 3).map((x) => x.toFixed(12)).join(',');
  const linesOf = (srf) => {
    const rows = srf.ctrlNet.map((r) => r.map(key));
    const cols = srf.ctrlNet[0].map((_, j) => srf.ctrlNet.map((r) => key(r[j])));
    return [...rows, ...cols];
  };
  const la = linesOf(A), lb = linesOf(B);
  let shared = 0;
  for (const a of la) for (const b of lb) {
    if (a.join('|') === b.join('|') || a.join('|') === [...b].reverse().join('|')) shared++;
  }
  // The answer: they share whole control lines. Two adjacent regular faces'
  // own 4x4 stencils overlap in a 4x2 block, so the shared edge's row is not
  // merely close — it is the same numbers. A boolean stitch across such an
  // edge is exact by construction, which a revolve-built seam is not.
  assert.ok(shared >= 1, `adjacent regular patches must share at least their common boundary row, got ${shared}`);

  // The same question of every adjacent pair, asked of the raw doubles.
  let pairs = 0, bitIdentical = 0, worst = 0;
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length !== 2 || !regSet.has(e.faces[0]) || !regSet.has(e.faces[1])) continue;
    pairs++;
    const rowA = patchBoundaryRow(regularFaceToPatch(cage, e.faces[0], ctx), cage.faces[e.faces[0]], e.v0, e.v1);
    const rowB = patchBoundaryRow(regularFaceToPatch(cage, e.faces[1], ctx), cage.faces[e.faces[1]], e.v0, e.v1);
    let same = true;
    for (let i = 0; i < 4; i++) for (let d = 0; d < 3; d++) {
      if (rowA[i][d] !== rowB[i][d]) same = false;
      worst = Math.max(worst, Math.abs(rowA[i][d] - rowB[i][d]));
    }
    if (same) bitIdentical++;
  }
  assert.equal(pairs, 120);
  assert.ok(bitIdentical >= 100, `most pairs are bit-identical on this fixture, got ${bitIdentical}/${pairs}`);
  assert.ok(worst > 0, 'and not all of them — a zero here would mean the check stopped discriminating');
  assert.ok(worst < 1e-14, `every pair must agree to rounding, worst ${worst}`);
});

// Clamping. ToNURBS step 1 emits a standalone clamped bicubic Bezier
// patch, not the same surface on the unclamped uniform knot vector its own
// evaluator assumes.
//
// surfacePoint evaluates an unclamped patch correctly inside its own true
// span. Clamping matters because an unclamped knot array spans [-3,4] while
// the valid domain is [0,1], and roughly ten consumers derive a surface's
// domain as knots[0]..knots[last].
// extractBorderCurves is the plainest case, and is asserted below
// against a number (131 units off) large enough that no tolerance
// argument applies.

test('clamping is the same surface — every emitted patch samples identically clamped and unclamped, to machine precision', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 8);
  const patches = subdToPatches(cage, { maxIsolation: 0 }).patches;
  assert.ok(patches.length > 0, 'the torus cage must convert completely on the first pass');
  let worst = 0;
  const ctx = buildTopology(cage);
  for (const pt of patches) {
    // The emitted patch is the clamped one; rebuild the unclamped form from
    // the same stencil to compare against. Reading the stencil back off the
    // clamped net would be circular, so it comes from regularFaceStencil.
    const unclamped = bicubicRegularPatchSurface(regularFaceStencil(cage, pt.faceIndex, ctx));
    for (let a = 0; a <= 6; a++) for (let b = 0; b <= 6; b++) {
      const A = surfacePoint(unclamped, a / 6, b / 6);
      const B = surfacePoint(pt.srf, a / 6, b / 6);
      worst = Math.max(worst, Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]));
    }
  }
  assert.ok(worst < 1e-9, `clamped and unclamped forms must be the same surface, worst deviation ${worst}`);
});

test('an emitted patch carries clamped knots — its array ends bound its own valid domain, which is what domain-deriving consumers assume', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 8);
  const { srf } = subdToPatches(cage, { maxIsolation: 0 }).patches[0];
  assert.deepEqual(srf.knotsU, [0, 0, 0, 0, 1, 1, 1, 1]);
  assert.deepEqual(srf.knotsV, [0, 0, 0, 0, 1, 1, 1, 1]);
  assert.equal(srf.degU, 3);
  assert.equal(srf.degV, 3);
  // Non-rational by construction — the regular fast path averages real
  // points and never weights them, so carrying w=1 is the whole story.
  for (const row of srf.ctrlNet) for (const p of row) assert.equal(p[3], 1);
});

test('clamped means interpolating at the corners — each corner control point is the surface there, which the unclamped form is not', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 8);
  const patches = subdToPatches(cage, { maxIsolation: 0 }).patches;
  // (i,j) of a net corner and the (u,v) it must equal on a clamped patch.
  const corners = [[0, 0, 0, 0], [3, 0, 1, 0], [0, 3, 0, 1], [3, 3, 1, 1]];
  const ctx = buildTopology(cage);
  let worstClamped = 0, worstUnclamped = 0;
  for (const pt of patches.slice(0, 12)) {
    const unclamped = bicubicRegularPatchSurface(regularFaceStencil(cage, pt.faceIndex, ctx));
    for (const [i, j, u, v] of corners) {
      const s = surfacePoint(pt.srf, u, v);
      const cp = pt.srf.ctrlNet[i][j];
      worstClamped = Math.max(worstClamped, Math.hypot(cp[0] - s[0], cp[1] - s[1], cp[2] - s[2]));
      const uc = unclamped.ctrlNet[i][j];
      worstUnclamped = Math.max(worstUnclamped, Math.hypot(uc[0] - s[0], uc[1] - s[1], uc[2] - s[2]));
    }
  }
  assert.ok(worstClamped < 1e-12, `a clamped patch's corner control point must BE its corner, worst ${worstClamped}`);
  // The negative control: without this the assertion above would pass for
  // a fixture that happened to be nearly flat everywhere.
  assert.ok(worstUnclamped > 1, `the unclamped net's corners must NOT lie on the surface, worst ${worstUnclamped}`);
});

test('the consumer that depends on it: extractBorderCurves returns the patch\'s real border, not one derived from a knot array that overshoots its domain', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 8);
  const { srf } = subdToPatches(cage, { maxIsolation: 0 }).patches[0];
  const borders = extractBorderCurves(srf);
  assert.equal(borders.length, 4, 'an open patch has four naked borders');
  const check = (edge, uAt, vAt) => {
    const crv = borders.find((b) => b.edge === edge).crv;
    const a = crv.knots[crv.degree], z = crv.knots[crv.knots.length - 1 - crv.degree];
    let worst = 0;
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      const got = curvePoint(crv, a + t * (z - a));
      const want = surfacePoint(srf, uAt === null ? t : uAt, vAt === null ? t : vAt);
      worst = Math.max(worst, Math.hypot(got[0] - want[0], got[1] - want[1], got[2] - want[2]));
    }
    assert.ok(worst < 1e-9, `${edge} border must lie on the surface's own edge, worst ${worst}`);
  };
  check('uMin', 0, null);
  check('uMax', 1, null);
  check('vMin', null, 0);
  check('vMax', null, 1);
});
