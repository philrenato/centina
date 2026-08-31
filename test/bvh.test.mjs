// A BVH IS ONLY WORTH HAVING IF IT AGREES WITH BRUTE FORCE.
//
// Every assertion here compares the tree against testing every triangle. That is
// the only oracle available: a tree can be well-formed, well-balanced, cheap by
// its own cost metric, and still miss geometry — and a renderer built on it
// shows that as holes nobody can trace back to the accelerator.
import { strict as assert } from 'node:assert';
import { buildBVH, bvhIntersect, bvhCost, packBVHForGPU } from '../kernel/bvh.mjs';

// Brute force: the same Möller-Trumbore, every triangle, no tree.
function brute(positions, origin, dir) {
  let best = null;
  for (let t = 0; t < positions.length / 9; t += 1) {
    const p = t * 9;
    const e1 = [positions[p + 3] - positions[p], positions[p + 4] - positions[p + 1], positions[p + 5] - positions[p + 2]];
    const e2 = [positions[p + 6] - positions[p], positions[p + 7] - positions[p + 1], positions[p + 8] - positions[p + 2]];
    const pv = [dir[1] * e2[2] - dir[2] * e2[1], dir[2] * e2[0] - dir[0] * e2[2], dir[0] * e2[1] - dir[1] * e2[0]];
    const det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
    if (Math.abs(det) < 1e-12) continue;
    const inv = 1 / det;
    const tv = [origin[0] - positions[p], origin[1] - positions[p + 1], origin[2] - positions[p + 2]];
    const u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
    if (u < -1e-7 || u > 1 + 1e-7) continue;
    const qv = [tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]];
    const v = (dir[0] * qv[0] + dir[1] * qv[1] + dir[2] * qv[2]) * inv;
    if (v < -1e-7 || u + v > 1 + 1e-7) continue;
    const hit = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
    if (hit > 1e-7 && (!best || hit < best.t)) best = { t: hit, tri: t };
  }
  return best;
}

// A deterministic pseudo-random source: a test that cannot be re-run on the
// scene that failed it is a test that reports rather than explains.
let seed = 0x9e3779b9;
const rnd = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return ((seed >>> 0) / 4294967296); };

function scatterTriangles(n, spread) {
  const out = new Float32Array(n * 9);
  for (let i = 0; i < n; i += 1) {
    const cx = (rnd() - 0.5) * spread, cy = (rnd() - 0.5) * spread, cz = (rnd() - 0.5) * spread;
    for (let k = 0; k < 3; k += 1) {
      out[i * 9 + k * 3] = cx + (rnd() - 0.5);
      out[i * 9 + k * 3 + 1] = cy + (rnd() - 0.5);
      out[i * 9 + k * 3 + 2] = cz + (rnd() - 0.5);
    }
  }
  return out;
}

// STRUCTURE: every triangle appears exactly once, leaves are never empty, and
// interior nodes contain their children. An empty leaf reads as an interior node
// pointing at garbage, which is why the count field can never be zero on a leaf.
function checkStructure(bvh, label) {
  const seen = new Uint8Array(bvh.triangleCount);
  let covered = 0;
  for (let i = 0; i < bvh.nodeCount; i += 1) {
    const b = i * 8;
    const count = bvh.nodes[b + 7];
    assert.ok(bvh.nodes[b] <= bvh.nodes[b + 4], `${label}: node ${i} has inverted x bounds`);
    assert.ok(bvh.nodes[b + 1] <= bvh.nodes[b + 5], `${label}: node ${i} has inverted y bounds`);
    assert.ok(bvh.nodes[b + 2] <= bvh.nodes[b + 6], `${label}: node ${i} has inverted z bounds`);
    if (count > 0) {
      const first = bvh.nodes[b + 3];
      for (let k = first; k < first + count; k += 1) {
        assert.equal(seen[bvh.order[k]], 0, `${label}: triangle ${bvh.order[k]} is in two leaves`);
        seen[bvh.order[k]] = 1; covered += 1;
      }
    } else {
      assert.ok(bvh.nodes[b + 3] > i && bvh.nodes[b + 3] < bvh.nodeCount, `${label}: node ${i} right child out of range`);
    }
  }
  assert.equal(covered, bvh.triangleCount, `${label}: ${covered} of ${bvh.triangleCount} triangles are reachable`);
}

for (const [label, tris, spread] of [['scattered 200', 200, 20], ['scattered 2000', 2000, 60], ['dense 500', 500, 3]]) {
  const pos = scatterTriangles(tris, spread);
  const bvh = buildBVH(pos);
  checkStructure(bvh, label);
  // AGREEMENT WITH BRUTE FORCE, on rays aimed from all around.
  let checked = 0, agreed = 0;
  for (let r = 0; r < 400; r += 1) {
    const o = [(rnd() - 0.5) * spread * 3, (rnd() - 0.5) * spread * 3, (rnd() - 0.5) * spread * 3];
    const d = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
    const L = Math.hypot(d[0], d[1], d[2]); d[0] /= L; d[1] /= L; d[2] /= L;
    const a = bvhIntersect(bvh, pos, o, d);
    const b = brute(pos, o, d);
    checked += 1;
    if (!a && !b) { agreed += 1; continue; }
    assert.ok(a && b, `${label}: tree and brute force disagree about whether ray ${r} hits anything`);
    assert.ok(Math.abs(a.t - b.t) < 1e-4, `${label}: ray ${r} hit at ${a.t} against ${b.t}`);
    agreed += 1;
  }
  assert.equal(agreed, checked);
  console.log(`  ${label.padEnd(15)} ${String(bvh.nodeCount).padStart(5)} nodes  depth ${String(bvh.maxDepth).padStart(2)}  ${String(bvh.leaves).padStart(4)} leaves  SAH cost ${bvhCost(bvh).toFixed(1)}  ${checked} rays agree`);
}

/* ⚠ DEGENERATE INPUT MUST NOT HANG. Triangles whose centroids all coincide give
   a split that separates nothing; recursing on it never terminates, and the
   symptom is a frozen tab rather than an error. */
{
  const same = new Float32Array(64 * 9);
  for (let i = 0; i < 64; i += 1) same.set([0, 0, 0, 1, 0, 0, 0, 1, 0], i * 9);
  const bvh = buildBVH(same);
  checkStructure(bvh, 'coincident');
  assert.ok(bvh.nodeCount > 0);
  console.log(`  ${'coincident 64'.padEnd(15)} ${String(bvh.nodeCount).padStart(5)} nodes  (a split that separates nothing becomes a leaf)`);
}
{
  const flat = new Float32Array(200 * 9);
  for (let i = 0; i < 200; i += 1) {
    const x = i * 0.5;
    flat.set([x, 0, 0, x + 0.4, 0, 0, x, 0, 0.4], i * 9);   // zero thickness in y
  }
  const bvh = buildBVH(flat);
  checkStructure(bvh, 'coplanar');
  const hit = bvhIntersect(bvh, flat, [50, 5, 0.1], [0, -1, 0]);
  const ref = brute(flat, [50, 5, 0.1], [0, -1, 0]);
  assert.equal(!!hit, !!ref, 'coplanar: a flat sheet must still be hit');
  console.log(`  ${'coplanar 200'.padEnd(15)} ${String(bvh.nodeCount).padStart(5)} nodes  (a zero-thickness slab still intersects)`);
}
{
  assert.equal(buildBVH(new Float32Array(0)).nodeCount, 0, 'an empty scene builds an empty tree');
  const one = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const b1 = buildBVH(one);
  assert.equal(b1.nodeCount, 1);
  assert.ok(bvhIntersect(b1, one, [0.2, 0.2, 1], [0, 0, -1]), 'a one-triangle tree still hits');
  console.log('  empty and single-triangle trees behave');
}

/* ⚠⚠ AXIS-ALIGNED RAYS, WHICH ARE THE COMMON CASE AND THE ONE THAT BREAKS.
   `1/0` is Infinity, and where a box face lies exactly on the ray's origin the
   slab test computes `0 * Infinity` = NaN. Comparisons against NaN are all
   false, so the node is not rejected — it is silently MISSED. Every one of these
   rays is aimed straight down an axis and originates on a coordinate a box face
   sits on, which is exactly the alignment a modeling app produces constantly
   and a random ray never does. */
{
  const pos = scatterTriangles(600, 12);
  const bvh = buildBVH(pos);
  let checked = 0;
  for (const axis of [0, 1, 2]) {
    for (const sign of [1, -1]) {
      for (let k = 0; k < 40; k += 1) {
        const o = [0, 0, 0];
        const d = [0, 0, 0];
        d[axis] = sign;
        // origin placed ON a vertex coordinate, which is where the NaN appears
        const src = (k * 7) % (pos.length / 3);
        o[(axis + 1) % 3] = pos[src * 3 + ((axis + 1) % 3)];
        o[(axis + 2) % 3] = pos[src * 3 + ((axis + 2) % 3)];
        o[axis] = -40 * sign;
        const a2 = bvhIntersect(bvh, pos, o, d);
        const b2 = brute(pos, o, d);
        checked += 1;
        assert.equal(!!a2, !!b2, `axis-aligned: tree and brute force disagree on axis ${axis} sign ${sign} ray ${k}`);
        if (a2 && b2) assert.ok(Math.abs(a2.t - b2.t) < 1e-4, `axis-aligned: hit at ${a2.t} against ${b2.t}`);
      }
    }
  }
  console.log(`  axis-aligned:   ${checked} rays down the three axes, all agree with brute force`);
}

/* THE TREE MUST BE WORTH ITS COST. A structurally perfect BVH that puts every
   triangle in one leaf agrees with brute force on every ray and saves nothing —
   so quality is asserted, not assumed. */
{
  const pos = scatterTriangles(4000, 80);
  const bvh = buildBVH(pos);
  const flatCost = 4000;
  assert.ok(bvhCost(bvh) < flatCost * 0.05,
    `SAH cost ${bvhCost(bvh).toFixed(1)} is not meaningfully better than testing all ${flatCost}`);
  assert.ok(bvh.maxDepth < 64, `depth ${bvh.maxDepth} is too deep to walk with a fixed stack`);
  console.log(`  quality: 4000 triangles, SAH cost ${bvhCost(bvh).toFixed(1)} against ${flatCost} flat, depth ${bvh.maxDepth}`);
}
/* ⚠⚠ THE TWO CHILDREN MUST BE ADJACENT. A traversal reads the left index and
   takes the right as `left + 1` without storing it — so building the left
   subtree entirely before allocating the right, which is the obvious recursive
   order, separates them and every interior node then points at whatever the left
   subtree happened to end at. The tree stays structurally valid and every
   brute-force ray still agrees, because the CPU walk reads both indices; only a
   shader sees it. This is the assertion that stands in for that shader. */
{
  const pos = scatterTriangles(1500, 30);
  const bvh = buildBVH(pos);
  let interior = 0;
  for (let i = 0; i < bvh.nodeCount; i += 1) {
    const b = i * 8;
    if (bvh.nodes[b + 7] > 0) continue;
    interior += 1;
    const left = bvh.nodes[b + 3];
    assert.ok(left > i, `node ${i}: left child ${left} is not after it`);
    assert.ok(left + 1 < bvh.nodeCount, `node ${i}: right child ${left + 1} is past the end`);
  }
  assert.ok(interior > 100, `only ${interior} interior nodes — not enough to be checking anything`);

  /* THE SHADER'S STACK IS FORTY DEEP AND HAS NO OVERFLOW PATH: past that it
     silently drops a subtree, which is missing geometry with nothing to blame. */
  assert.ok(bvh.maxDepth < 40, `depth ${bvh.maxDepth} would overrun a shader stack of 40`);

  /* AND THE INDICES SURVIVE THE TRIP AS INTEGERS. A float cannot carry an index
     past 2^24; written as a float it works on every scene small enough to test
     by hand and starts losing consecutive integers at sixteen million. */
  const packed = packBVHForGPU(bvh);
  const asU32 = new Uint32Array(packed.buffer);
  let checked = 0;
  for (let i = 0; i < bvh.nodeCount; i += 1) {
    const b = i * 8;
    assert.equal(asU32[b + 3], bvh.nodes[b + 3], `node ${i}: index field did not survive packing`);
    assert.equal(asU32[b + 7], bvh.nodes[b + 7], `node ${i}: count field did not survive packing`);
    assert.equal(packed[b], bvh.nodes[b], `node ${i}: bounds changed in packing`);
    checked += 1;
  }
  assert.equal(packed.length, bvh.nodeCount * 8, 'the packed buffer is two vec4 per node');
  console.log(`  contract:       ${interior} interior nodes all have adjacent children, depth ${bvh.maxDepth} < 40, ${checked} nodes pack as u32 bits`);
}

console.log('bvh: ok');
