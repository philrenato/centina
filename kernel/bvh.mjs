// A BOUNDING VOLUME HIERARCHY OVER TRIANGLES, BUILT BY BINNED SAH.
//
// Binned surface-area-heuristic construction after Wald, "On fast Construction
// of SAH-based Bounding Volume Hierarchies" (IEEE Symposium on Interactive Ray
// Tracing, 2007): candidate splits are evaluated in a fixed number of bins along
// the widest axis rather than at every centroid, which turns an O(n log^2 n)
// sort-based build into a linear pass per level for a tree of nearly the same
// quality.
//
// ⚠ THE OUTPUT IS FLAT TYPED ARRAYS, DELIBERATELY. A tree of objects is pleasant
// to write and useless to a GPU: the whole point of this structure is that it
// can be uploaded as a storage buffer and walked by a shader without a pointer
// anywhere. Every node is a fixed 8 floats, and the triangle order is a separate
// index array, so a leaf names a contiguous RUN rather than a list.
//
// NODE LAYOUT, 8 floats (32 bytes), which is what a GPU wants and also what
// keeps a node inside one cache line pair on the CPU:
//   [0..2] bounds min
//   [3]    LEFT child index      (interior)  |  first triangle  (leaf)
//   [4..6] bounds max
//   [7]    0                     (interior)  |  triangle count  (leaf, > 0)
// A count of 0 marks an interior node, which is why a leaf may never be empty —
// an empty leaf would read as an interior node pointing at garbage.
//
// ⚠⚠ THE TWO CHILDREN ARE ADJACENT, AND THAT IS PART OF THE CONTRACT rather than
// an accident of allocation: a traversal reads the left index and takes the
// right as `left + 1` without storing it. Building the left subtree entirely
// before allocating the right — the obvious recursive order — separates them,
// and every interior node then points a shader at whatever the left subtree
// happened to end at. Both slots are reserved BEFORE either is filled.
//
// ⚠ AND THE TWO INDEX SLOTS ARE INTEGERS LIVING IN FLOAT FIELDS. A shader reads
// them with `bitcast<u32>`, so they must be written as the BITS of a u32 and not
// as a float that happens to have that value: past 2^24 a float can no longer
// represent consecutive integers, and the failure begins silently at a scene
// size nobody tests at. `packBVHForGPU` does that conversion; the plain `nodes`
// array keeps float indices for CPU use, where they are only ever compared.

const BINS = 12;
const LEAF_MAX = 4;

/** Centroid and bounds of every triangle, in one pass. */
function triangleData(positions) {
  const n = Math.floor(positions.length / 9);
  const cen = new Float32Array(n * 3);
  const lo = new Float32Array(n * 3);
  const hi = new Float32Array(n * 3);
  for (let t = 0; t < n; t += 1) {
    for (let d = 0; d < 3; d += 1) {
      const a = positions[t * 9 + d], b = positions[t * 9 + 3 + d], c = positions[t * 9 + 6 + d];
      lo[t * 3 + d] = Math.min(a, b, c);
      hi[t * 3 + d] = Math.max(a, b, c);
      cen[t * 3 + d] = (a + b + c) / 3;
    }
  }
  return { n, cen, lo, hi };
}

const surfaceArea = (lo, hi) => {
  const x = hi[0] - lo[0], y = hi[1] - lo[1], z = hi[2] - lo[2];
  if (x < 0 || y < 0 || z < 0) return 0;
  return 2 * (x * y + y * z + z * x);
};

/**
 * Build a BVH over unindexed triangles.
 *
 * `positions` is a flat [x,y,z] * 3 per triangle array — the same shape a
 * renderer already has, so nothing has to be re-packed to build the tree.
 *
 * Returns `{ nodes, order, nodeCount, triangleCount, maxDepth, leaves }`.
 * `order[i]` is the original triangle index of the i-th triangle in leaf order.
 */
export function buildBVH(positions, opts = {}) {
  const leafMax = Math.max(1, opts.leafMax || LEAF_MAX);
  const bins = Math.max(2, opts.bins || BINS);
  const { n, cen, lo, hi } = triangleData(positions);
  if (n === 0) return { nodes: new Float32Array(0), order: new Uint32Array(0), nodeCount: 0, triangleCount: 0, maxDepth: 0, leaves: 0 };

  const order = new Uint32Array(n);
  for (let i = 0; i < n; i += 1) order[i] = i;
  /* A binary tree over n leaves of at least one triangle each has at most
     2n-1 nodes, so the array is sized once and never grown. */
  const nodes = new Float32Array((2 * n - 1) * 8);
  let nodeCount = 0, leaves = 0, maxDepth = 0;

  const boundsOf = (start, count, out) => {
    out[0] = out[1] = out[2] = Infinity;
    out[3] = out[4] = out[5] = -Infinity;
    for (let i = start; i < start + count; i += 1) {
      const t = order[i];
      for (let d = 0; d < 3; d += 1) {
        if (lo[t * 3 + d] < out[d]) out[d] = lo[t * 3 + d];
        if (hi[t * 3 + d] > out[3 + d]) out[3 + d] = hi[t * 3 + d];
      }
    }
  };

  const scratch = new Float32Array(6);
  const binLo = new Float32Array(bins * 6);
  const binCount = new Int32Array(bins);
  const leftArea = new Float32Array(bins);
  const leftCount = new Int32Array(bins);

  const build = (self, start, count, depth) => {
    if (depth > maxDepth) maxDepth = depth;
    boundsOf(start, count, scratch);
    const b = self * 8;
    nodes[b] = scratch[0]; nodes[b + 1] = scratch[1]; nodes[b + 2] = scratch[2];
    nodes[b + 4] = scratch[3]; nodes[b + 5] = scratch[4]; nodes[b + 6] = scratch[5];

    const makeLeaf = () => {
      nodes[b + 3] = start;
      nodes[b + 7] = count;
      leaves += 1;
    };
    if (count <= leafMax) { makeLeaf(); return self; }

    /* SPLIT ALONG THE WIDEST SPREAD OF CENTROIDS, not of bounds. Long thin
       triangles make the bounds wide on an axis the centroids barely vary
       along, and splitting there puts everything on one side. */
    let cLo = [Infinity, Infinity, Infinity], cHi = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < start + count; i += 1) {
      const t = order[i];
      for (let d = 0; d < 3; d += 1) {
        if (cen[t * 3 + d] < cLo[d]) cLo[d] = cen[t * 3 + d];
        if (cen[t * 3 + d] > cHi[d]) cHi[d] = cen[t * 3 + d];
      }
    }
    let axis = 0, ext = cHi[0] - cLo[0];
    for (let d = 1; d < 3; d += 1) { const e = cHi[d] - cLo[d]; if (e > ext) { ext = e; axis = d; } }
    // Every centroid at the same place: no split can separate them, and
    // recursing would not terminate.
    if (!(ext > 1e-12)) { makeLeaf(); return self; }

    binLo.fill(0); binCount.fill(0);
    for (let k = 0; k < bins; k += 1) {
      binLo[k * 6] = binLo[k * 6 + 1] = binLo[k * 6 + 2] = Infinity;
      binLo[k * 6 + 3] = binLo[k * 6 + 4] = binLo[k * 6 + 5] = -Infinity;
    }
    const scale = bins / ext;
    for (let i = start; i < start + count; i += 1) {
      const t = order[i];
      let k = Math.floor((cen[t * 3 + axis] - cLo[axis]) * scale);
      if (k < 0) k = 0; if (k >= bins) k = bins - 1;
      binCount[k] += 1;
      for (let d = 0; d < 3; d += 1) {
        if (lo[t * 3 + d] < binLo[k * 6 + d]) binLo[k * 6 + d] = lo[t * 3 + d];
        if (hi[t * 3 + d] > binLo[k * 6 + 3 + d]) binLo[k * 6 + 3 + d] = hi[t * 3 + d];
      }
    }

    // Sweep left, then right, so each candidate split costs O(1) rather than a
    // rescan — the reason binning is fast at all.
    let accLo = [Infinity, Infinity, Infinity], accHi = [-Infinity, -Infinity, -Infinity], acc = 0;
    for (let k = 0; k < bins - 1; k += 1) {
      if (binCount[k]) {
        for (let d = 0; d < 3; d += 1) {
          if (binLo[k * 6 + d] < accLo[d]) accLo[d] = binLo[k * 6 + d];
          if (binLo[k * 6 + 3 + d] > accHi[d]) accHi[d] = binLo[k * 6 + 3 + d];
        }
        acc += binCount[k];
      }
      leftCount[k] = acc;
      leftArea[k] = acc ? surfaceArea(accLo, accHi) : 0;
    }
    let bestCost = Infinity, bestSplit = -1;
    accLo = [Infinity, Infinity, Infinity]; accHi = [-Infinity, -Infinity, -Infinity]; acc = 0;
    for (let k = bins - 1; k > 0; k -= 1) {
      if (binCount[k]) {
        for (let d = 0; d < 3; d += 1) {
          if (binLo[k * 6 + d] < accLo[d]) accLo[d] = binLo[k * 6 + d];
          if (binLo[k * 6 + 3 + d] > accHi[d]) accHi[d] = binLo[k * 6 + 3 + d];
        }
        acc += binCount[k];
      }
      const lC = leftCount[k - 1], rC = acc;
      if (!lC || !rC) continue;
      const cost = leftArea[k - 1] * lC + surfaceArea(accLo, accHi) * rC;
      if (cost < bestCost) { bestCost = cost; bestSplit = k; }
    }

    /* ⚠ AND A SPLIT MUST BEAT NOT SPLITTING. The SAH compares the cost of
       tracing both children against the cost of testing every triangle here;
       where no split wins, a leaf larger than the target is the right answer,
       and forcing one anyway builds a deeper tree that is slower to walk. */
    const parentCost = surfaceArea([scratch[0], scratch[1], scratch[2]], [scratch[3], scratch[4], scratch[5]]) * count;
    if (bestSplit < 0 || bestCost >= parentCost) { makeLeaf(); return self; }

    // Partition in place around the chosen bin.
    let i = start, j = start + count - 1;
    while (i <= j) {
      const t = order[i];
      let k = Math.floor((cen[t * 3 + axis] - cLo[axis]) * scale);
      if (k < 0) k = 0; if (k >= bins) k = bins - 1;
      if (k < bestSplit) { i += 1; } else { const tmp = order[i]; order[i] = order[j]; order[j] = tmp; j -= 1; }
    }
    const leftN = i - start;
    if (leftN === 0 || leftN === count) { makeLeaf(); return self; }

    // Both slots reserved before either is filled, so they are adjacent whatever
    // the subtrees do.
    const kids = nodeCount; nodeCount += 2;
    nodes[b + 3] = kids;
    nodes[b + 7] = 0;
    build(kids, start, leftN, depth + 1);
    build(kids + 1, start + leftN, count - leftN, depth + 1);
    return self;
  };

  nodeCount = 1;
  build(0, 0, n, 0);
  return { nodes: nodes.subarray(0, nodeCount * 8), order, nodeCount, triangleCount: n, maxDepth, leaves };
}

/* THE SLAB TEST, AND THE TWO WAYS IT GOES WRONG ON AN AXIS-ALIGNED RAY.
   A modeling app produces axis-aligned rays constantly and a random ray
   generator never does, so both of these hide from exactly the test most likely
   to be written first.

   ⚠ `1/0` IS INFINITY, and where a box face lies exactly on the ray's origin the
   product is `0 * Infinity` = NaN. Comparisons against NaN are all false, so the
   node is not rejected, it is silently MISSED.

   ⚠⚠ AND CLAMPING THE DIRECTION TO A TINY EPSILON DOES NOT FIX IT. With a huge
   finite reciprocal, a ray lying exactly ON a slab's far face gets the t-range
   [-huge, 0] — a range the ray is geometrically inside, reported as ending
   before it begins — and the box is rejected anyway. That reads as a fixed bug
   and is the same bug with a smaller epsilon.

   A ray PARALLEL to an axis is not a division at all. Either its origin lies
   within that slab, in which case the slab constrains nothing, or it does not,
   in which case the box is missed outright. Asked as a containment test. */
const PARALLEL = 1e-12;
function slabAxis(o, d, lo, hi, state) {
  if (Math.abs(d) < PARALLEL) {
    if (o < lo || o > hi) { state.miss = true; }
    return;
  }
  const inv = 1 / d;
  let t1 = (lo - o) * inv, t2 = (hi - o) * inv;
  if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
  if (t1 > state.tmin) state.tmin = t1;
  if (t2 < state.tmax) state.tmax = t2;
}
function slab(nodes, b, ox, oy, oz, dx, dy, dz, tMax) {
  const state = { tmin: -Infinity, tmax: Infinity, miss: false };
  slabAxis(ox, dx, nodes[b], nodes[b + 4], state);
  slabAxis(oy, dy, nodes[b + 1], nodes[b + 5], state);
  slabAxis(oz, dz, nodes[b + 2], nodes[b + 6], state);
  if (state.miss) return Infinity;
  return (state.tmax >= Math.max(state.tmin, 0) && state.tmin < tMax) ? Math.max(state.tmin, 0) : Infinity;
}

/**
 * Nearest triangle along a ray. Returns `{ t, tri, u, v }` or null.
 *
 * Möller-Trumbore, and the same walk a shader performs — kept here because a
 * tree whose only consumer is a shader can only be tested through a shader,
 * and a CPU query makes the structure checkable against brute force.
 */
export function bvhIntersect(bvh, positions, origin, dir, tMax = Infinity) {
  if (!bvh.nodeCount) return null;
  const { nodes, order } = bvh;

  const stack = [0];
  let best = null, bestT = tMax;
  while (stack.length) {
    const node = stack.pop();
    const b = node * 8;
    if (slab(nodes, b, origin[0], origin[1], origin[2], dir[0], dir[1], dir[2], bestT) === Infinity) continue;
    const count = nodes[b + 7];
    if (count > 0) {
      const first = nodes[b + 3];
      for (let i = first; i < first + count; i += 1) {
        const t = order[i], p = t * 9;
        const e1x = positions[p + 3] - positions[p], e1y = positions[p + 4] - positions[p + 1], e1z = positions[p + 5] - positions[p + 2];
        const e2x = positions[p + 6] - positions[p], e2y = positions[p + 7] - positions[p + 1], e2z = positions[p + 8] - positions[p + 2];
        const px = dir[1] * e2z - dir[2] * e2y, py = dir[2] * e2x - dir[0] * e2z, pz = dir[0] * e2y - dir[1] * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        if (Math.abs(det) < 1e-12) continue;
        const inv = 1 / det;
        const tx = origin[0] - positions[p], ty = origin[1] - positions[p + 1], tz = origin[2] - positions[p + 2];
        const u = (tx * px + ty * py + tz * pz) * inv;
        if (u < -1e-7 || u > 1 + 1e-7) continue;
        const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
        const v = (dir[0] * qx + dir[1] * qy + dir[2] * qz) * inv;
        if (v < -1e-7 || u + v > 1 + 1e-7) continue;
        const hit = (e2x * qx + e2y * qy + e2z * qz) * inv;
        if (hit > 1e-7 && hit < bestT) { bestT = hit; best = { t: hit, tri: t, u, v }; }
      }
    } else {
      const left = nodes[b + 3];
      stack.push(left + 1);
      stack.push(left);
    }
  }
  return best;
}

/** The SAH cost of a built tree, for comparing one build against another. */
export function bvhCost(bvh, traversalCost = 1, triangleCost = 1) {
  if (!bvh.nodeCount) return 0;
  const { nodes } = bvh;
  const rootArea = surfaceArea([nodes[0], nodes[1], nodes[2]], [nodes[4], nodes[5], nodes[6]]);
  if (!(rootArea > 0)) return 0;
  let cost = 0;
  for (let i = 0; i < bvh.nodeCount; i += 1) {
    const b = i * 8;
    const area = surfaceArea([nodes[b], nodes[b + 1], nodes[b + 2]], [nodes[b + 4], nodes[b + 5], nodes[b + 6]]);
    const count = nodes[b + 7];
    cost += (area / rootArea) * (count > 0 ? triangleCost * count : traversalCost);
  }
  return cost;
}

/**
 * The tree as a GPU storage buffer: two `vec4` per node, with the two index
 * fields written as the BITS of a u32 so a shader may `bitcast` them.
 *
 * ⚠ A FLOAT CANNOT CARRY AN INDEX PAST 2^24. Writing the child index or the
 * first-triangle index as a float works perfectly on every scene small enough to
 * test by hand and begins losing consecutive integers at sixteen million — where
 * the symptom is geometry quietly attaching to the wrong node rather than an
 * error. The conversion belongs here, once, rather than at each write.
 */
export function packBVHForGPU(bvh) {
  const out = new Float32Array(bvh.nodeCount * 8);
  const asU32 = new Uint32Array(out.buffer);
  for (let i = 0; i < bvh.nodeCount; i += 1) {
    const b = i * 8;
    out[b] = bvh.nodes[b]; out[b + 1] = bvh.nodes[b + 1]; out[b + 2] = bvh.nodes[b + 2];
    out[b + 4] = bvh.nodes[b + 4]; out[b + 5] = bvh.nodes[b + 5]; out[b + 6] = bvh.nodes[b + 6];
    asU32[b + 3] = bvh.nodes[b + 3] >>> 0;
    asU32[b + 7] = bvh.nodes[b + 7] >>> 0;
  }
  return out;
}
