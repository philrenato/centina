// A CLOSEST-POINT WALK IS ONLY WORTH HAVING IF IT AGREES WITH BRUTE FORCE, AND
// A SIGN IS ONLY WORTH HAVING IF SOMETHING INDEPENDENT AGREES ABOUT INSIDE.
//
// Two separate oracles, because the query answers two separate questions and a
// single fixture can hide a failure in either:
//
//   * the DISTANCE and the point are checked against testing every triangle, so
//     what is under test is the descent and its pruning, not the point-triangle
//     math both halves share;
//   * the SIGN is checked against RAY PARITY — an odd number of crossings means
//     inside — which shares no code with the pseudonormal at all. That matters,
//     because the whole reason the pseudonormal exists is that the obvious
//     answer (the nearest triangle's own face normal) is wrong over most of
//     space, and a test written from the same idea as the code would agree with
//     it while both were wrong.
import { strict as assert } from 'node:assert';
import { buildBVH, bvhClosestPoint, buildMeshPseudonormals, closestPointOnTriangle } from '../kernel/bvh.mjs';

// A deterministic pseudo-random source: a test that cannot be re-run on the
// scene that failed it is a test that reports rather than explains.
let seed = 0x2545f491;
const rnd = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return ((seed >>> 0) / 4294967296); };

// ---------------------------------------------------------------------------
// FIXTURES — closed triangle soups, built by INDEX WRAP rather than by emitting
// a duplicate seam column, so nothing depends on two trig results agreeing to
// the last bit before the mesh counts as watertight.
// ---------------------------------------------------------------------------

function sphereSoup(center, radius, nu = 32, nv = 16) {
  const at = (i, j) => {
    const phi = (2 * Math.PI * i) / nu;
    const theta = (Math.PI * j) / nv;
    return [
      center[0] + radius * Math.sin(theta) * Math.cos(phi),
      center[1] + radius * Math.sin(theta) * Math.sin(phi),
      center[2] + radius * Math.cos(theta),
    ];
  };
  const tris = [];
  const push = (a, b, c) => tris.push(...a, ...b, ...c);
  for (let j = 0; j < nv; j += 1) {
    for (let i = 0; i < nu; i += 1) {
      const i1 = (i + 1) % nu;
      const a = at(i, j), b = at(i1, j), c = at(i1, j + 1), d = at(i, j + 1);
      // Wound so the face normals point OUT. A sphere is the one fixture where
      // an inverted winding is invisible to every check except the sign.
      if (j === 0) push(a, d, c);            // north cap: one triangle per column
      else if (j === nv - 1) push(a, c, b);  // south cap
      else { push(a, c, b); push(a, d, c); }
    }
  }
  return new Float32Array(tris);
}

function torusSoup(center, R, r, nu = 48, nv = 24) {
  const at = (i, j) => {
    const u = (2 * Math.PI * i) / nu, v = (2 * Math.PI * j) / nv;
    const rad = R + r * Math.cos(v);
    return [center[0] + rad * Math.cos(u), center[1] + rad * Math.sin(u), center[2] + r * Math.sin(v)];
  };
  const tris = [];
  for (let i = 0; i < nu; i += 1) {
    for (let j = 0; j < nv; j += 1) {
      const i1 = (i + 1) % nu, j1 = (j + 1) % nv;
      const a = at(i, j), b = at(i1, j), c = at(i1, j1), d = at(i, j1);
      tris.push(...a, ...b, ...c, ...a, ...c, ...d);
    }
  }
  return new Float32Array(tris);
}

// A cube whose quads are split the ordinary way, so its corners are shared by
// UNEQUAL numbers of triangles — two from one face, one from another. That
// asymmetry is the whole point of the fixture: see the angle-weight test.
function cubeSoup(half) {
  const h = half;
  const v = [
    [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
    [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
  ];
  const quads = [
    [4, 5, 6, 7], [1, 0, 3, 2], [0, 1, 5, 4],
    [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5],
  ];
  const tris = [];
  for (const [a, b, c, d] of quads) tris.push(...v[a], ...v[b], ...v[c], ...v[a], ...v[c], ...v[d]);
  return new Float32Array(tris);
}

// ---------------------------------------------------------------------------
// ORACLES
// ---------------------------------------------------------------------------

function bruteClosest(positions, p) {
  const q = [0, 0, 0];
  let best = null;
  for (let t = 0; t < positions.length / 9; t += 1) {
    const o = t * 9;
    const a = [positions[o], positions[o + 1], positions[o + 2]];
    const b = [positions[o + 3], positions[o + 4], positions[o + 5]];
    const c = [positions[o + 6], positions[o + 7], positions[o + 8]];
    closestPointOnTriangle(p, a, b, c, q);
    const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
    if (!best || d < best.distance) best = { distance: d, point: q.slice(), tri: t };
  }
  return best;
}

// Inside by crossing parity, Möller-Trumbore over every triangle. Shares
// nothing with the pseudonormal. A ray that grazes an edge is counted twice or
// not at all, so three directions vote and a disagreement makes the sample
// undecidable rather than wrong.
function crossings(positions, origin, dir) {
  let hits = 0;
  for (let t = 0; t < positions.length / 9; t += 1) {
    const p = t * 9;
    const e1 = [positions[p + 3] - positions[p], positions[p + 4] - positions[p + 1], positions[p + 5] - positions[p + 2]];
    const e2 = [positions[p + 6] - positions[p], positions[p + 7] - positions[p + 1], positions[p + 8] - positions[p + 2]];
    const pv = [dir[1] * e2[2] - dir[2] * e2[1], dir[2] * e2[0] - dir[0] * e2[2], dir[0] * e2[1] - dir[1] * e2[0]];
    const det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
    if (Math.abs(det) < 1e-14) continue;
    const inv = 1 / det;
    const tv = [origin[0] - positions[p], origin[1] - positions[p + 1], origin[2] - positions[p + 2]];
    const u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
    if (u < 0 || u > 1) continue;
    const qv = [tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]];
    const v = (dir[0] * qv[0] + dir[1] * qv[1] + dir[2] * qv[2]) * inv;
    if (v < 0 || u + v > 1) continue;
    const hit = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
    if (hit > 1e-9) hits += 1;
  }
  return hits;
}

const VOTE_DIRS = [
  [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
  [-0.2672612419124244, 0.5345224838248488, 0.8017837257372732],
  [0.7071067811865476, -0.7071067811865476, 0.0031622776601684],
];
function insideByParity(positions, p) {
  let yes = 0;
  for (const d of VOTE_DIRS) if (crossings(positions, p, d) % 2 === 1) yes += 1;
  if (yes === 3) return true;
  if (yes === 0) return false;
  return null; // the rays disagree; the sample sits on an edge and is undecidable
}

// ---------------------------------------------------------------------------
// TEST 1 — the point-triangle routine itself, on answers computed by hand.
// ---------------------------------------------------------------------------
{
  const a = [0, 0, 0], b = [4, 0, 0], c = [0, 3, 0];
  const q = [0, 0, 0];

  let bary = closestPointOnTriangle([1, 1, 5], a, b, c, q);
  assert.deepEqual(q, [1, 1, 0], 'a point over the interior drops straight onto the face');
  assert.ok(bary.every((w) => w > 0), 'and reads as the face interior, no zero coordinate');

  bary = closestPointOnTriangle([-2, -2, 0], a, b, c, q);
  assert.deepEqual(q, [0, 0, 0], 'a point in vertex A\'s region lands on A');
  assert.deepEqual(bary, [1, 0, 0], 'and reads as a vertex: two exact zeros');

  bary = closestPointOnTriangle([2, -5, 1], a, b, c, q);
  assert.deepEqual(q, [2, 0, 0], 'a point below edge AB lands on AB');
  assert.equal(bary[2], 0, 'and reads as an edge: exactly one zero');
  assert.ok(bary[0] > 0 && bary[1] > 0, 'with both endpoints represented');

  bary = closestPointOnTriangle([9, 0, 0], a, b, c, q);
  assert.deepEqual(q, [4, 0, 0], 'a point beyond B lands on B');
  assert.deepEqual(bary, [0, 1, 0], 'and reads as vertex B');

  // A degenerate triangle must not return NaN: a NaN distance compares false
  // against everything and would silently win no comparison anywhere.
  closestPointOnTriangle([1, 1, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0], q);
  assert.ok(q.every(Number.isFinite), 'a zero-area triangle still answers with a finite point');
  console.log('  triangle:       face, three edge and vertex regions, and a degenerate sliver');
}

// ---------------------------------------------------------------------------
// TEST 2 — the descent agrees with testing every triangle.
// ---------------------------------------------------------------------------
{
  const positions = torusSoup([0, 0, 0], 30, 10, 48, 24);
  const bvh = buildBVH(positions);
  assert.ok(bvh.maxDepth > 5, `tree is only ${bvh.maxDepth} deep — not enough to be pruning anything`);

  let worstDistance = 0, worstPoint = 0, checked = 0;
  for (let n = 0; n < 400; n += 1) {
    const p = [(rnd() - 0.5) * 140, (rnd() - 0.5) * 140, (rnd() - 0.5) * 140];
    const got = bvhClosestPoint(bvh, positions, p);
    const want = bruteClosest(positions, p);
    assert.ok(got, 'the walk found nothing where brute force found a triangle');
    worstDistance = Math.max(worstDistance, Math.abs(got.distance - want.distance));
    worstPoint = Math.max(worstPoint, Math.hypot(
      got.point[0] - want.point[0], got.point[1] - want.point[1], got.point[2] - want.point[2],
    ));
    checked += 1;
  }
  assert.ok(worstDistance <= 1e-6, `distance disagrees with brute force by ${worstDistance}`);
  assert.ok(worstPoint <= 1e-6, `the closest point disagrees with brute force by ${worstPoint}`);
  console.log(`  brute control:  ${checked} queries on ${bvh.triangleCount} triangles, worst distance gap ${worstDistance.toExponential(2)}`);

  // maxDistance is a real bound, not a hint: past it the answer is null, and
  // inside it the answer is unchanged.
  const p = [200, 200, 200];
  assert.equal(bvhClosestPoint(bvh, positions, p, { maxDistance: 10 }), null, 'a bound of 10 rejects a point 300 away');
  const near = bvhClosestPoint(bvh, positions, [35, 0, 0]);
  const bounded = bvhClosestPoint(bvh, positions, [35, 0, 0], { maxDistance: 100 });
  assert.ok(Math.abs(near.distance - bounded.distance) < 1e-9, 'a generous bound changes nothing');
  console.log('  bound:          rejects past maxDistance, unchanged within it');
}

// ---------------------------------------------------------------------------
// TEST 3 — the sign agrees with ray parity, on a shape with real concavity.
// ---------------------------------------------------------------------------
{
  const positions = torusSoup([0, 0, 0], 30, 10, 48, 24);
  const bvh = buildBVH(positions);
  const pn = buildMeshPseudonormals(positions);
  assert.equal(pn.boundaryEdges, 0, 'the torus fixture is not watertight');
  assert.equal(pn.nonManifoldEdges, 0, 'the torus fixture has a non-manifold edge');
  assert.ok(pn.closed, 'the torus fixture does not read as closed');

  const regions = { face: 0, edge: 0, vertex: 0 };
  let agreed = 0, undecidable = 0, insideSeen = 0;
  for (let n = 0; n < 600; n += 1) {
    const p = [(rnd() - 0.5) * 100, (rnd() - 0.5) * 100, (rnd() - 0.5) * 40];
    const got = bvhClosestPoint(bvh, positions, p, { pseudonormals: pn });
    assert.ok(got.signed, 'a closed mesh must give a signed answer');
    if (got.distance < 1e-4) continue; // sitting on the surface: neither answer means anything
    const want = insideByParity(positions, p);
    if (want === null) { undecidable += 1; continue; }
    assert.equal(
      got.inside, want,
      `sign disagrees with ray parity at ${p.map((v) => v.toFixed(3))}: pseudonormal says ${got.inside ? 'inside' : 'outside'}, parity says ${want ? 'inside' : 'outside'} (closest feature: ${got.region})`,
    );
    regions[got.region] += 1;
    if (want) insideSeen += 1;
    agreed += 1;
  }
  assert.ok(agreed > 400, `only ${agreed} decidable samples — the fixture is not being exercised`);
  assert.ok(insideSeen > 10, `only ${insideSeen} samples landed inside the torus — the inside case is untested`);
  /* ⚠ IF NO SAMPLE EVER LANDS ON AN EDGE OR A VERTEX, THIS TEST PASSES WITHOUT
     TESTING THE PSEUDONORMAL AT ALL — a plain face normal is correct in the
     face interior. The region tally is what keeps that from being invisible. */
  assert.ok(regions.edge > 0, 'no query landed on an edge; the edge pseudonormal went untested');
  assert.ok(regions.vertex > 0, 'no query landed on a vertex; the vertex pseudonormal went untested');
  console.log(`  sign vs parity: ${agreed} samples agree (${insideSeen} inside), features ${regions.face} face / ${regions.edge} edge / ${regions.vertex} vertex, ${undecidable} undecidable`);
}

// ---------------------------------------------------------------------------
// TEST 4 — magnitude on a shape whose distance is known in closed form.
// ---------------------------------------------------------------------------
{
  const R = 20;
  const positions = sphereSoup([0, 0, 0], R, 64, 32);
  const bvh = buildBVH(positions);
  const pn = buildMeshPseudonormals(positions);
  assert.ok(pn.closed, 'the sphere fixture is not closed');

  /* The mesh is INSCRIBED in the sphere it samples, so a measured distance is
     never the analytic one — it is short by up to the chord sagitta on the
     inside and long by it on the outside. The tolerance is that sagitta, not a
     number picked until the test went green. */
  const sagitta = R * (1 - Math.cos(Math.PI / 32));
  let worst = 0, signErrors = 0;
  for (let n = 0; n < 300; n += 1) {
    const dir = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
    const l = Math.hypot(...dir) || 1;
    const radius = 2 + rnd() * 38;
    const p = [dir[0] / l * radius, dir[1] / l * radius, dir[2] / l * radius];
    const got = bvhClosestPoint(bvh, positions, p, { pseudonormals: pn });
    worst = Math.max(worst, Math.abs(Math.abs(got.signedDistance) - Math.abs(radius - R)));
    if (Math.abs(radius - R) > sagitta && got.inside !== (radius < R)) signErrors += 1;
  }
  assert.equal(signErrors, 0, `${signErrors} points got the wrong side of a sphere`);
  assert.ok(worst < sagitta * 1.5, `worst magnitude error ${worst} exceeds the chord sagitta ${sagitta}`);
  console.log(`  sphere:         worst |d| error ${worst.toFixed(4)} against a chord sagitta of ${sagitta.toFixed(4)}, 0 sign errors`);
}

// ---------------------------------------------------------------------------
// TEST 5 — the weight in "angle-weighted" is load-bearing.
// ---------------------------------------------------------------------------
{
  /* A cube corner is shared by three faces, and every face contributes exactly
     a right angle there however its quad happens to be split. So the ANGLE-
     weighted normal at every corner is exactly (±1,±1,±1)/sqrt(3) — while a
     plain sum of incident TRIANGLE normals is not, because this cube's corners
     sit in unequal numbers of triangles. The exact diagonal is therefore proof
     the weighting is applied and not merely written down. */
  const positions = cubeSoup(10);
  const pn = buildMeshPseudonormals(positions);
  assert.ok(pn.closed, 'the cube fixture is not closed');
  assert.equal(pn.vertexCount, 8, `the cube welded to ${pn.vertexCount} vertices, not 8`);

  const incident = new Array(8).fill(0);
  for (let i = 0; i < pn.corner.length; i += 1) incident[pn.corner[i]] += 1;
  assert.ok(new Set(incident).size > 1, 'the fixture corners are all in the same number of triangles — it cannot show the bias');

  let worst = 0;
  for (let v = 0; v < 8; v += 1) {
    const n = [pn.vertexNormals[v * 3], pn.vertexNormals[v * 3 + 1], pn.vertexNormals[v * 3 + 2]];
    const l = Math.hypot(...n);
    const u = [n[0] / l, n[1] / l, n[2] / l];
    for (let d = 0; d < 3; d += 1) worst = Math.max(worst, Math.abs(Math.abs(u[d]) - 1 / Math.sqrt(3)));
  }
  assert.ok(worst < 1e-12, `a corner normal is off the cube diagonal by ${worst} — the angle weight is not being applied`);
  console.log(`  angle weight:   8 corners on the exact body diagonal (worst ${worst.toExponential(2)}), incident counts ${[...new Set(incident)].sort().join('/')}`);
}

// ---------------------------------------------------------------------------
// TEST 6 — an open mesh has no inside, and says so.
// ---------------------------------------------------------------------------
{
  const positions = new Float32Array([0, 0, 0, 10, 0, 0, 0, 10, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0]);
  const bvh = buildBVH(positions);
  const pn = buildMeshPseudonormals(positions);
  assert.equal(pn.closed, false, 'a flat quad read as a closed solid');
  assert.equal(pn.boundaryEdges, 4, `expected 4 boundary edges, got ${pn.boundaryEdges}`);
  const got = bvhClosestPoint(bvh, positions, [5, 5, -3], { pseudonormals: pn });
  assert.equal(got.signed, false, 'an open mesh must not claim a sign');
  assert.ok(got.signedDistance > 0, 'and must report the unsigned distance');
  assert.ok(Math.abs(got.distance - 3) < 1e-6, `distance to the quad should be 3, got ${got.distance}`);
  console.log('  open mesh:      no sign claimed, unsigned distance still correct');
}

console.log('bvh closest point: ok');
