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

// ---------------------------------------------------------------------------
// CLOSEST POINT ON THE MESH, AND ITS SIGN
// ---------------------------------------------------------------------------
//
// The same tree, a different descent. A ray walk prunes on a t-interval; a
// closest-point walk prunes on the squared distance from the query point to a
// node's box, and descends into the nearer child first so the running best
// shrinks as early as possible and the far subtree is usually rejected without
// being opened.
//
// ⚠ A FACE NORMAL GIVES THE WRONG SIGN WHENEVER THE CLOSEST POINT IS NOT IN A
// FACE'S INTERIOR, which on any tessellated surface is most of the time: the
// closest feature is an edge or a vertex over a large part of space, and there
// the "nearest triangle" is whichever of several tied triangles the loop
// happened to keep. The published fix is the ANGLE-WEIGHTED PSEUDONORMAL
// (Bærentzen & Aanæs, "Signed Distance Computation Using the Angle Weighted
// Pseudonormal", IEEE TVCG 11(3), 2005): a vertex carries the sum of its
// incident face normals weighted by the incident ANGLE, an edge carries the sum
// of its two face normals, and the sign of the dot product with (p - q) is then
// correct everywhere for a closed, consistently-oriented mesh.
//
// ⚠⚠ AND THE THEOREM WANTS A CLOSED, CONSISTENTLY ORIENTED MANIFOLD. On an open
// or non-manifold mesh there is no inside, so `buildMeshPseudonormals` reports
// `closed` and a query against a mesh that is not closed returns `signed: false`
// rather than a confident sign nothing supports.

// Distance from a point to a node's box, squared. Zero inside the box.
function boxDist2(nodes, b, x, y, z) {
  let dx = 0, dy = 0, dz = 0;
  if (x < nodes[b]) dx = nodes[b] - x; else if (x > nodes[b + 4]) dx = x - nodes[b + 4];
  if (y < nodes[b + 1]) dy = nodes[b + 1] - y; else if (y > nodes[b + 5]) dy = y - nodes[b + 5];
  if (z < nodes[b + 2]) dz = nodes[b + 2] - z; else if (z > nodes[b + 6]) dz = z - nodes[b + 6];
  return dx * dx + dy * dy + dz * dz;
}

/**
 * Closest point on one triangle, with the barycentric coordinates that say
 * WHICH FEATURE it landed on — Ericson, *Real-Time Collision Detection* (2005),
 * section 5.1.5, by Voronoi region rather than by projecting and clamping.
 *
 * The barycentric triple is what the pseudonormal lookup needs: the region
 * branches return an exact zero in the coordinates that are off, so "two zeros
 * is a vertex, one zero is an edge, none is the face interior" reads straight
 * off the numbers instead of being tracked separately.
 *
 * `out` is a 3-array filled with the closest point; the return value is the
 * barycentric triple [wa, wb, wc].
 */
export function closestPointOnTriangle(p, a, b, c, out) {
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
  const apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) { out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; return [1, 0, 0]; }

  const bpx = p[0] - b[0], bpy = p[1] - b[1], bpz = p[2] - b[2];
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) { out[0] = b[0]; out[1] = b[1]; out[2] = b[2]; return [0, 1, 0]; }

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    out[0] = a[0] + abx * v; out[1] = a[1] + aby * v; out[2] = a[2] + abz * v;
    return [1 - v, v, 0];
  }

  const cpx = p[0] - c[0], cpy = p[1] - c[1], cpz = p[2] - c[2];
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) { out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; return [0, 0, 1]; }

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    out[0] = a[0] + acx * w; out[1] = a[1] + acy * w; out[2] = a[2] + acz * w;
    return [1 - w, 0, w];
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    out[0] = b[0] + (c[0] - b[0]) * w; out[1] = b[1] + (c[1] - b[1]) * w; out[2] = b[2] + (c[2] - b[2]) * w;
    return [0, 1 - w, w];
  }

  /* A DEGENERATE TRIANGLE MAKES THE INTERIOR DENOMINATOR ZERO. The Voronoi
     branches above cover a zero-area triangle in every direction that matters,
     but a sliver can reach here with va+vb+vc underflowed; returning NaN from a
     distance query poisons every comparison downstream silently, so the corner
     answer is taken instead. */
  const denom = va + vb + vc;
  if (!(denom > 0)) { out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; return [1, 0, 0]; }
  const inv = 1 / denom;
  const v = vb * inv, w = vc * inv;
  out[0] = a[0] + abx * v + acx * w;
  out[1] = a[1] + aby * v + acy * w;
  out[2] = a[2] + abz * v + acz * w;
  return [1 - v - w, v, w];
}

// Weld key for the connectivity a pseudonormal needs. Rounded to a fixed
// decimal count as a NUMBER first and with -0 canonicalized to +0, because
// `(-1e-15).toFixed(6)` is the string "-0.000000" and would split a seam pair
// that agrees to twelve more digits than the key claims to resolve.
function weldKey(x, y, z, mul) {
  let rx = Math.round(x * mul) / mul; if (rx === 0) rx = 0;
  let ry = Math.round(y * mul) / mul; if (ry === 0) ry = 0;
  let rz = Math.round(z * mul) / mul; if (rz === 0) rz = 0;
  return `${rx}_${ry}_${rz}`;
}

/**
 * The connectivity and the three tiers of pseudonormal, for a triangle soup.
 *
 * `positions` is the same flat array `buildBVH` takes, so a mesh does not have
 * to be re-packed to be queried. Triangle soup has no shared vertices, so the
 * corners are WELDED by rounded position first — that weld is what makes an
 * edge or a vertex a thing at all here.
 *
 * Returns `{ triangleCount, vertexCount, corner, faceNormals, vertexNormals,
 * edgeNormals, boundaryEdges, nonManifoldEdges, degenerate, closed }`.
 * `corner[t*3+k]` is the welded id of triangle t's k-th corner; `edgeNormals`
 * is keyed by `min*vertexCount + max` of the two welded ids.
 */
export function buildMeshPseudonormals(positions, opts = {}) {
  const mul = 10 ** (opts.weldDecimals ?? 6);
  const triangleCount = Math.floor(positions.length / 9);
  const ids = new Map();
  const corner = new Uint32Array(triangleCount * 3);
  let vertexCount = 0;
  for (let t = 0; t < triangleCount; t += 1) {
    for (let k = 0; k < 3; k += 1) {
      const p = t * 9 + k * 3;
      const key = weldKey(positions[p], positions[p + 1], positions[p + 2], mul);
      let id = ids.get(key);
      if (id === undefined) { id = vertexCount; vertexCount += 1; ids.set(key, id); }
      corner[t * 3 + k] = id;
    }
  }

  const faceNormals = new Float64Array(triangleCount * 3);
  const vertexNormals = new Float64Array(vertexCount * 3);
  const edgeNormals = new Map();
  const edgeFaces = new Map();
  let degenerate = 0;

  for (let t = 0; t < triangleCount; t += 1) {
    const p = t * 9;
    const ax = positions[p], ay = positions[p + 1], az = positions[p + 2];
    const bx = positions[p + 3], by = positions[p + 4], bz = positions[p + 5];
    const cx = positions[p + 6], cy = positions[p + 7], cz = positions[p + 8];
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) { nx /= len; ny /= len; nz /= len; } else { degenerate += 1; }
    faceNormals[t * 3] = nx; faceNormals[t * 3 + 1] = ny; faceNormals[t * 3 + 2] = nz;

    // THE ANGLE IS THE WEIGHT, and it is the whole content of the theorem: an
    // equal-weight vertex normal is biased by however finely the tessellator
    // happened to fan that corner, and a biased normal is a wrong SIGN, not a
    // slightly wrong shade.
    const vx = [ax, bx, cx], vy = [ay, by, cy], vz = [az, bz, cz];
    for (let k = 0; k < 3; k += 1) {
      const i1 = (k + 1) % 3, i2 = (k + 2) % 3;
      const ux = vx[i1] - vx[k], uy = vy[i1] - vy[k], uz = vz[i1] - vz[k];
      const wx = vx[i2] - vx[k], wy = vy[i2] - vy[k], wz = vz[i2] - vz[k];
      const lu = Math.hypot(ux, uy, uz), lw = Math.hypot(wx, wy, wz);
      if (!(lu > 0) || !(lw > 0)) continue;
      let cosA = (ux * wx + uy * wy + uz * wz) / (lu * lw);
      if (cosA < -1) cosA = -1; else if (cosA > 1) cosA = 1;
      const angle = Math.acos(cosA);
      const v = corner[t * 3 + k];
      vertexNormals[v * 3] += nx * angle;
      vertexNormals[v * 3 + 1] += ny * angle;
      vertexNormals[v * 3 + 2] += nz * angle;
    }

    for (let k = 0; k < 3; k += 1) {
      const a = corner[t * 3 + k], b = corner[t * 3 + (k + 1) % 3];
      if (a === b) continue;
      const key = (a < b ? a : b) * vertexCount + (a < b ? b : a);
      const acc = edgeNormals.get(key);
      if (acc) { acc[0] += nx; acc[1] += ny; acc[2] += nz; } else { edgeNormals.set(key, [nx, ny, nz]); }
      edgeFaces.set(key, (edgeFaces.get(key) || 0) + 1);
    }
  }

  let boundaryEdges = 0, nonManifoldEdges = 0;
  for (const count of edgeFaces.values()) {
    if (count < 2) boundaryEdges += 1;
    else if (count > 2) nonManifoldEdges += 1;
  }

  return {
    triangleCount,
    vertexCount,
    corner,
    faceNormals,
    vertexNormals,
    edgeNormals,
    boundaryEdges,
    nonManifoldEdges,
    degenerate,
    closed: triangleCount > 0 && boundaryEdges === 0 && nonManifoldEdges === 0,
  };
}

// Which feature the barycentric triple landed on, and the normal that speaks
// for it. An exact zero is what the Voronoi branches produce for an edge or a
// vertex; the epsilon only catches an interior answer that rounded onto the
// boundary, where both normals agree in sign anyway because (p - q) is
// perpendicular to the face there.
const BARY_EPS = 1e-10;
function pseudonormalAt(pn, tri, bary) {
  const zeros = (bary[0] <= BARY_EPS ? 1 : 0) + (bary[1] <= BARY_EPS ? 1 : 0) + (bary[2] <= BARY_EPS ? 1 : 0);
  if (zeros >= 2) {
    const k = bary[0] > BARY_EPS ? 0 : (bary[1] > BARY_EPS ? 1 : 2);
    const v = pn.corner[tri * 3 + k];
    return { region: 'vertex', n: [pn.vertexNormals[v * 3], pn.vertexNormals[v * 3 + 1], pn.vertexNormals[v * 3 + 2]] };
  }
  if (zeros === 1) {
    const k = bary[0] <= BARY_EPS ? 0 : (bary[1] <= BARY_EPS ? 1 : 2);
    const a = pn.corner[tri * 3 + (k + 1) % 3], b = pn.corner[tri * 3 + (k + 2) % 3];
    const key = (a < b ? a : b) * pn.vertexCount + (a < b ? b : a);
    const acc = pn.edgeNormals.get(key);
    if (acc) return { region: 'edge', n: acc };
  }
  return { region: 'face', n: [pn.faceNormals[tri * 3], pn.faceNormals[tri * 3 + 1], pn.faceNormals[tri * 3 + 2]] };
}

/**
 * Nearest point on the mesh to `point`, walking the same tree `bvhIntersect`
 * walks.
 *
 * `opts.pseudonormals` — the record from `buildMeshPseudonormals`. Supplying it
 * is what makes the answer SIGNED; without it the result is an unsigned
 * distance and `signed` is false.
 * `opts.maxDistance` — give up beyond this radius and return null. A caller
 * with a reach already knows it does not care past that, and the bound prunes
 * the whole tree in one comparison at the root.
 *
 * Returns `{ distance, signedDistance, signed, point, tri, bary, region,
 * normal, inside }` or null.
 */
export function bvhClosestPoint(bvh, positions, point, opts = {}) {
  if (!bvh.nodeCount) return null;
  const { nodes, order } = bvh;
  const px = point[0], py = point[1], pz = point[2];
  const maxDistance = opts.maxDistance ?? Infinity;
  let bestD2 = Number.isFinite(maxDistance) ? maxDistance * maxDistance : Infinity;
  let bestTri = -1;
  const bestQ = [0, 0, 0];
  let bestBary = null;

  const q = [0, 0, 0];
  const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
  const stack = [0];
  const dstack = [0];
  while (stack.length) {
    const node = stack.pop();
    const enter = dstack.pop();
    if (enter >= bestD2) continue;
    const nb = node * 8;
    const count = nodes[nb + 7];
    if (count > 0) {
      const first = nodes[nb + 3];
      for (let i = first; i < first + count; i += 1) {
        const t = order[i], p = t * 9;
        a[0] = positions[p]; a[1] = positions[p + 1]; a[2] = positions[p + 2];
        b[0] = positions[p + 3]; b[1] = positions[p + 4]; b[2] = positions[p + 5];
        c[0] = positions[p + 6]; c[1] = positions[p + 7]; c[2] = positions[p + 8];
        const bary = closestPointOnTriangle(point, a, b, c, q);
        const dx = q[0] - px, dy = q[1] - py, dz = q[2] - pz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < bestD2) {
          bestD2 = d2; bestTri = t; bestBary = bary;
          bestQ[0] = q[0]; bestQ[1] = q[1]; bestQ[2] = q[2];
        }
      }
    } else {
      /* NEARER CHILD FIRST, which is the whole reason this is cheaper than
         brute force: the far subtree is tested against a best that has already
         shrunk, so it is usually rejected at its own root. A stack pops last
         first, so the FARTHER child is pushed first. */
      const left = nodes[nb + 3];
      const dl = boxDist2(nodes, left * 8, px, py, pz);
      const dr = boxDist2(nodes, (left + 1) * 8, px, py, pz);
      if (dl <= dr) {
        if (dr < bestD2) { stack.push(left + 1); dstack.push(dr); }
        if (dl < bestD2) { stack.push(left); dstack.push(dl); }
      } else {
        if (dl < bestD2) { stack.push(left); dstack.push(dl); }
        if (dr < bestD2) { stack.push(left + 1); dstack.push(dr); }
      }
    }
  }
  if (bestTri < 0) return null;

  const distance = Math.sqrt(bestD2);
  const pn = opts.pseudonormals;
  if (!pn) {
    return {
      distance, signedDistance: distance, signed: false,
      point: bestQ, tri: bestTri, bary: bestBary, region: 'face',
      normal: null, inside: false,
    };
  }
  const { region, n } = pseudonormalAt(pn, bestTri, bestBary);
  const nlen = Math.hypot(n[0], n[1], n[2]);
  const normal = nlen > 0 ? [n[0] / nlen, n[1] / nlen, n[2] / nlen] : [0, 0, 0];
  const s = (px - bestQ[0]) * normal[0] + (py - bestQ[1]) * normal[1] + (pz - bestQ[2]) * normal[2];
  const inside = pn.closed && s < 0;
  return {
    distance,
    signedDistance: pn.closed ? (inside ? -distance : distance) : distance,
    signed: pn.closed,
    point: bestQ,
    tri: bestTri,
    bary: bestBary,
    region,
    normal,
    inside,
  };
}
