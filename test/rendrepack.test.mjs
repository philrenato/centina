// THE THREE FAILURES THAT DRAW A PLAUSIBLE PICTURE.
//
// Nothing in the geometry upload throws. A permutation applied to two of three
// arrays, an index written as a float instead of as its bits, and a node's two
// index fields swapped all produce buffers of exactly the right size and shape,
// and each one renders: wrong materials on right shapes, one material on
// everything, or the right image at a hundredth of the speed. So the assertions
// here are not "did it produce a buffer" — they are the three specific
// questions those failures answer differently.
//
// Each is written so that it FAILS on its own defect: the defect was introduced
// in `kernel/rendrepack.mjs`, the failure watched, and the defect removed.
import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import {
  packSceneForGPU, gatherGeometry, reorderToGPU, packPartMats,
  matPack, packMaterials,
  MAT_STRIDE_FLOATS, TRI_STRIDE_FLOATS, NODE_STRIDE_FLOATS,
  PART_HIDDEN_BIT, PART_ROW_MASK,
} from '../kernel/rendrepack.mjs';

const f32 = Math.fround;
const LEAF_MAX = 4;          // the builder's own leaf target, passed explicitly
const SHADER_STACK = 40;     // the traversal's fixed stack depth

// A deterministic source, so a scene that fails can be re-run.
let seed = 0x2545f491;
const rnd = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return ((seed >>> 0) / 4294967296); };

/* THE FIXTURE'S PARTS ARE IN A DIFFERENT SPATIAL ORDER FROM THEIR PART INDICES.
   A scene whose parts happen to be laid out in index order can come back from a
   spatial sort in its original order, and then a permutation applied to two
   arrays out of three is indistinguishable from one applied to all three. Part
   0 sits at the far end deliberately. */
const BOX_X = [30, 0, 20, 10];
const TRI_COUNTS = [7, 11, 5, 9];

function makeParts() {
  const parts = [];
  for (let p = 0; p < BOX_X.length; p += 1) {
    const n = TRI_COUNTS[p];
    const pos = new Float32Array(n * 9), nrm = new Float32Array(n * 9);
    for (let t = 0; t < n; t += 1) {
      for (let k = 0; k < 3; k += 1) {
        pos[t * 9 + k * 3] = BOX_X[p] + rnd();
        pos[t * 9 + k * 3 + 1] = rnd() * 2 - 1;
        pos[t * 9 + k * 3 + 2] = rnd() * 2 - 1;
        // Per-corner normals, distinct per part so a mixed-up normal is visible.
        nrm[t * 9 + k * 3] = 0; nrm[t * 9 + k * 3 + 1] = 0; nrm[t * 9 + k * 3 + 2] = p + 1;
      }
    }
    parts.push({ pos, nrm, material: p * 3 + 1, visible: p !== 2 });
  }
  return parts;
}

const boxOf = (part) => {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < part.pos.length; i += 3) {
    for (let d = 0; d < 3; d += 1) {
      if (part.pos[i + d] < lo[d]) lo[d] = part.pos[i + d];
      if (part.pos[i + d] > hi[d]) hi[d] = part.pos[i + d];
    }
  }
  return { lo, hi };
};
const boxHolds = (b, x, y, z) => x >= b.lo[0] && x <= b.hi[0] && y >= b.lo[1] && y <= b.hi[1] && z >= b.lo[2] && z <= b.hi[2];
const boxesOverlap = (a, b) => {
  for (let d = 0; d < 3; d += 1) if (a.hi[d] < b.lo[d] || b.hi[d] < a.lo[d]) return false;
  return true;
};

const parts = makeParts();
const boxes = parts.map(boxOf);
const nT = TRI_COUNTS.reduce((a, b) => a + b, 0);

/* VERIFY THE FIXTURE BEFORE BLAMING THE CODE. Every assertion below about a
   triangle naming its owner rests on the boxes being separable; if two boxes
   overlapped, a triangle in the overlap would satisfy the containment test for
   the wrong part and the whole check would pass on a scrambled buffer. */
{
  let pairs = 0;
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      assert.ok(!boxesOverlap(boxes[i], boxes[j]), `fixture: part ${i} and part ${j} share space, so containment cannot name an owner`);
      pairs += 1;
    }
  }
  assert.equal(pairs, 6, 'fixture: all four parts were compared');
  console.log(`  fixture:        ${parts.length} parts, ${nT} triangles, ${pairs} box pairs all disjoint`);
}

const scene = packSceneForGPU(parts, { leafMax: LEAF_MAX });
const trisU = new Uint32Array(scene.tris.buffer);
const nodesU = new Uint32Array(scene.nodes.buffer);

// ---------------------------------------------------------------- shape ----
{
  assert.equal(scene.triangleCount, nT, 'every part\'s triangles reached the buffer');
  assert.equal(scene.tris.length, nT * TRI_STRIDE_FLOATS, 'tris is 3 vec4 per triangle');
  assert.equal(scene.norms.length, nT * TRI_STRIDE_FLOATS, 'norms is 3 vec4 per triangle');
  assert.equal(scene.nodes.length, scene.nodeCount * NODE_STRIDE_FLOATS, 'nodes is 2 vec4 per node');
  assert.equal(scene.partMats.length, parts.length, 'one partMats entry per part');
  assert.ok(scene.maxDepth < SHADER_STACK, `tree is ${scene.maxDepth} deep against a ${SHADER_STACK}-entry shader stack`);

  /* AND THE PERMUTATION IS NOT THE IDENTITY. Every check below would pass
     trivially on an unpermuted scene, which would make them assertions about
     nothing. */
  let moved = 0;
  for (let i = 0; i < nT; i += 1) if (scene.order[i] !== i) moved += 1;
  assert.ok(moved > nT / 4, `the BVH reordered only ${moved} of ${nT} triangles — the permutation checks would be vacuous`);
  console.log(`  layout:         ${scene.nodeCount} nodes, depth ${scene.maxDepth}, ${moved}/${nT} triangles moved by the reorder`);
}

// ------------------------------------- FAILURE 1: the third array not permuted ----
/* The material index carried on triangle i must name the part whose bounding
   box actually contains triangle i's vertices — for every i, not for a sample.
   A sample is what the original guard proposed and it is not enough: a
   permutation that agrees with itself on most of a scene is exactly what a
   partition produces, and a spot check finds the agreeing half. */
{
  let checked = 0;
  for (let i = 0; i < nT; i += 1) {
    const d = i * TRI_STRIDE_FLOATS;
    const claimed = trisU[d + 3];
    assert.ok(claimed < parts.length, `triangle ${i} names part ${claimed}, which does not exist`);
    let owner = -1, owners = 0;
    for (let p = 0; p < boxes.length; p += 1) {
      const b = boxes[p];
      if (boxHolds(b, scene.tris[d], scene.tris[d + 1], scene.tris[d + 2])
        && boxHolds(b, scene.tris[d + 4], scene.tris[d + 5], scene.tris[d + 6])
        && boxHolds(b, scene.tris[d + 8], scene.tris[d + 9], scene.tris[d + 10])) { owner = p; owners += 1; }
    }
    assert.equal(owners, 1, `triangle ${i} sits in ${owners} part boxes, so the fixture cannot name its owner`);
    assert.equal(claimed, owner, `triangle ${i} is inside part ${owner}'s box and claims part ${claimed} — the reorder missed an array`);
    checked += 1;
  }

  /* THE NORMALS ARE THE THIRD ARRAY AND CARRY NO INDEX OF THEIR OWN, so they
     are checked by content: each part's corner normals encode its index. */
  for (let i = 0; i < nT; i += 1) {
    const d = i * TRI_STRIDE_FLOATS;
    assert.equal(scene.norms[d + 2], trisU[d + 3] + 1, `triangle ${i}: the normal came from a different part than the position`);
  }
  console.log(`  failure 1:      all ${checked} triangles name the part whose box holds them, normals agree`);
}

// ------------------------------------------ FAILURE 2: .w written as a float ----
/* Read through a Uint32Array view of the SAME buffer, which is what the shader
   does with `bitcast<u32>`. Written as a float, part 3 arrives as 1077936128;
   the shader clamps that instead of faulting, so every triangle in the scene
   resolves to one material and the symptom is "materials do nothing". */
{
  const { mp } = gatherGeometry(parts);
  let nonZero = 0, wouldDiffer = 0;
  for (let i = 0; i < nT; i += 1) {
    const want = mp[scene.order[i]];
    assert.equal(trisU[i * TRI_STRIDE_FLOATS + 3], want, `triangle ${i}: v0.w is not the u32 bits of part ${want}`);
    if (want !== 0) nonZero += 1;
    // A float write and a bit write agree only at index 0, so the fixture has
    // to contain indices above zero for this assertion to discriminate at all.
    const asFloatBits = new Uint32Array(new Float32Array([want]).buffer)[0];
    if (asFloatBits !== want) wouldDiffer += 1;
  }
  assert.ok(nonZero > 0, 'fixture: every part index is 0, where a float write and a bit write agree');
  assert.equal(wouldDiffer, nonZero, 'fixture: a float write differs from a bit write for every non-zero index');
  console.log(`  failure 2:      ${nT} v0.w fields are u32 bits; ${wouldDiffer} of them would differ if written as floats`);
}

// ------------------------------------------- FAILURE 3: node .w fields swapped ----
/* Swapping first-triangle and count turns every interior node into a leaf with
   a plausible count and every early leaf into an interior node. The traversal
   still terminates and the image is still CORRECT — it just walks the whole
   scene per ray. There is no visual tell, so the only place to catch it is the
   buffer.
   The strongest form is not a bound on the counts but a PARTITION: the leaf
   runs must cover [0, nT) exactly once each. */
{
  const covered = new Uint8Array(nT);
  let leafNodes = 0, innerNodes = 0, sum = 0, maxLeaf = 0;
  for (let i = 0; i < scene.nodeCount; i += 1) {
    const b = i * NODE_STRIDE_FLOATS;
    const first = nodesU[b + 3], count = nodesU[b + 7];
    if (count === 0) {
      innerNodes += 1;
      assert.ok(first + 1 < scene.nodeCount, `node ${i}: children ${first},${first + 1} are outside the ${scene.nodeCount} nodes built`);
      assert.notEqual(first, 0, `node ${i}: an interior node cannot have the root as a child`);
    } else {
      leafNodes += 1;
      assert.ok(count <= LEAF_MAX, `node ${i}: leaf of ${count} triangles against the builder's leaf bound of ${LEAF_MAX}`);
      assert.ok(first + count <= nT, `node ${i}: leaf run [${first}, ${first + count}) runs past the ${nT} triangles`);
      for (let t = first; t < first + count; t += 1) {
        assert.equal(covered[t], 0, `triangle ${t} is claimed by more than one leaf`);
        covered[t] = 1;
      }
      sum += count;
      if (count > maxLeaf) maxLeaf = count;
    }
  }
  assert.equal(sum, nT, `leaf counts sum to ${sum}, not the ${nT} triangles in the scene`);
  assert.equal(leafNodes, scene.leaves, 'the leaf nodes in the buffer are the leaves the builder says it made');
  assert.ok(innerNodes > 0, 'fixture: the tree has no interior node, so a swap would be invisible');
  for (let t = 0; t < nT; t += 1) assert.equal(covered[t], 1, `triangle ${t} belongs to no leaf`);
  console.log(`  failure 3:      ${innerNodes} interior counts are 0, ${leafNodes} leaves (max ${maxLeaf} <= ${LEAF_MAX}) partition all ${nT} triangles`);
}

// ------------------------------------------------- the shader's own walk ----
/* Traversal parity against brute force, walking the GPU buffers exactly as the
   shader does — nodes[n*2].w as the left child or first triangle, nodes[n*2+1].w
   as the count, the right child taken as left+1. This is the check that the
   reorder and the tree agree with each other rather than each being
   self-consistent. */
function shaderWalk(origin, dir) {
  const stack = [0]; let bestT = Infinity, bestPart = -1;
  const slab = (b) => {
    let tmin = 0, tmax = Infinity;
    for (let d = 0; d < 3; d += 1) {
      if (Math.abs(dir[d]) < 1e-12) { if (origin[d] < scene.nodes[b + d] || origin[d] > scene.nodes[b + 4 + d]) return false; continue; }
      const inv = 1 / dir[d];
      let t1 = (scene.nodes[b + d] - origin[d]) * inv, t2 = (scene.nodes[b + 4 + d] - origin[d]) * inv;
      if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
      if (t1 > tmin) tmin = t1;
      if (t2 < tmax) tmax = t2;
    }
    return tmax >= tmin && tmin < bestT;
  };
  while (stack.length) {
    const n = stack.pop(), b = n * NODE_STRIDE_FLOATS;
    if (!slab(b)) continue;
    const count = nodesU[b + 7], first = nodesU[b + 3];
    if (count === 0) { stack.push(first + 1); stack.push(first); continue; }
    for (let i = first; i < first + count; i += 1) {
      const h = triRay(i, origin, dir);
      if (h !== null && h < bestT) { bestT = h; bestPart = trisU[i * TRI_STRIDE_FLOATS + 3]; }
    }
  }
  return bestPart < 0 ? null : { t: bestT, part: bestPart };
}
function triRay(i, o, d) {
  const p = i * TRI_STRIDE_FLOATS;
  const ax = scene.tris[p], ay = scene.tris[p + 1], az = scene.tris[p + 2];
  const e1 = [scene.tris[p + 4] - ax, scene.tris[p + 5] - ay, scene.tris[p + 6] - az];
  const e2 = [scene.tris[p + 8] - ax, scene.tris[p + 9] - ay, scene.tris[p + 10] - az];
  const pv = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
  const det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
  if (Math.abs(det) < 1e-12) return null;
  const inv = 1 / det;
  const tv = [o[0] - ax, o[1] - ay, o[2] - az];
  const u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
  if (u < 0 || u > 1) return null;
  const qv = [tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]];
  const v = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv;
  if (v < 0 || u + v > 1) return null;
  const t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
  return t > 1e-7 ? t : null;
}
{
  let rays = 0, hits = 0;
  /* AIMED RAYS, NOT RANDOM ONES. Scattered triangles occupy a vanishing
     fraction of the scene's bounding box, so random rays miss almost
     everything and a parity check made of misses proves nothing. Each ray here
     is aimed at a point inside a randomly chosen triangle; the nearest hit
     along it is still whatever the scene puts first. */
  for (let r = 0; r < 400; r += 1) {
    const target = Math.floor(rnd() * nT) * TRI_STRIDE_FLOATS;
    let a = rnd(), b = rnd();
    if (a + b > 1) { a = 1 - a; b = 1 - b; }
    const at = [0, 0, 0];
    for (let d = 0; d < 3; d += 1) {
      at[d] = scene.tris[target + d] * (1 - a - b) + scene.tris[target + 4 + d] * a + scene.tris[target + 8 + d] * b;
    }
    const origin = [at[0] + (rnd() - 0.5) * 8, at[1] + (rnd() - 0.5) * 8, -20];
    const dir = [at[0] - origin[0], at[1] - origin[1], at[2] - origin[2]];
    const len = Math.hypot(dir[0], dir[1], dir[2]);
    dir[0] /= len; dir[1] /= len; dir[2] /= len;
    const walked = shaderWalk(origin, dir);
    let bestT = Infinity, bestPart = -1;
    for (let i = 0; i < nT; i += 1) {
      const h = triRay(i, origin, dir);
      if (h !== null && h < bestT) { bestT = h; bestPart = trisU[i * TRI_STRIDE_FLOATS + 3]; }
    }
    if (bestPart < 0) { assert.equal(walked, null, `ray ${r}: the tree found a hit brute force did not`); }
    else {
      assert.ok(walked !== null, `ray ${r}: the tree missed a triangle brute force hit`);
      assert.ok(Math.abs(walked.t - bestT) < 1e-6, `ray ${r}: nearest hit disagrees, ${walked.t} vs ${bestT}`);
      assert.equal(walked.part, bestPart, `ray ${r}: the tree and brute force name different parts`);
      hits += 1;
    }
    rays += 1;
  }
  assert.ok(hits > rays / 2, `only ${hits} of ${rays} rays hit anything — the parity check is nearly vacuous`);
  console.log(`  traversal:      ${rays} rays walk the GPU buffers to the same nearest hit as brute force (${hits} hits)`);
}

// ------------------------------------------------------------ partMats ----
{
  const pm = packPartMats(parts);
  for (let i = 0; i < parts.length; i += 1) {
    assert.equal(pm[i] & PART_ROW_MASK, parts[i].material, `part ${i}: material row did not survive packing`);
    const hidden = (pm[i] & PART_HIDDEN_BIT) !== 0;
    assert.equal(hidden, parts[i].visible === false, `part ${i}: visibility bit is wrong`);
  }
  assert.equal(packPartMats([]).length, 1, 'an empty scene still gets a one-entry partMats — a zero-length storage buffer is a validation error');
  console.log(`  partMats:       ${parts.length} rows, 1 hidden, mask and bit 31 round-trip`);
}

// -------------------------------------------------------- the empty scene ----
{
  const empty = packSceneForGPU([], { leafMax: LEAF_MAX });
  assert.equal(empty.triangleCount, 1, 'an empty scene synthesises one degenerate triangle');
  assert.equal(empty.tris.length, TRI_STRIDE_FLOATS, 'the degenerate triangle is a full 3-vec4 stride');
  assert.ok(empty.nodeCount >= 1, 'the degenerate triangle still builds a root node');
  assert.equal(empty.partMats.length, 1, 'partMats is never zero-length');
  assert.ok(empty.degenerate, 'the empty case reports itself');
  console.log('  empty scene:    one degenerate triangle, one node, one partMats entry — no zero-length buffer');
}

// ---------------------------------------------------- the material stride ----
/* The stride is 52 floats because `getMat` reads mats[i*13u] through
   mats[i*13u+12u]. Taking 48 from the buffer's stale binding comment leaves row
   0 correct and every later row progressively wrong — asserted here as an
   actual collision rather than as a restatement of the number. */
{
  assert.equal(MAT_STRIDE_FLOATS, 52, 'thirteen vec4 per material row');

  const grey = { base: [0.5, 0.5, 0.5] };
  const red = { base: [1, 0, 0], metallic: 1, rough: 0.1 };
  const table = packMaterials([grey, red]);
  assert.equal(table.length, 2 * MAT_STRIDE_FLOATS, 'the table is one stride per material');
  assert.equal(table[MAT_STRIDE_FLOATS], 1, 'row 1 starts at float 52');
  assert.equal(table[MAT_STRIDE_FLOATS + 1], 0, 'row 1 base is where row 1 begins');

  /* Rows written 48 apart, read back at the 52 the shader uses. Row 0 is still
     correct, which is what makes this so hard to see; row 1 comes back shifted
     by four floats, and every later row by four more. */
  const wrong = new Float32Array(2 * MAT_STRIDE_FLOATS);
  matPack(wrong, 0, grey); matPack(wrong, 48, red);
  assert.notEqual(wrong[MAT_STRIDE_FLOATS], f32(red.base[0]), 'a 48-float stride puts row 1 where getMat does not look');
  assert.equal(wrong[MAT_STRIDE_FLOATS], f32(red.rough), 'row 1 comes back shifted by exactly four floats — rough where base.r belongs');
  assert.equal(wrong[0], table[0], 'row 0 is still correct, which is why this reads as "some materials are broken"');
  assert.notEqual(wrong[48], table[48], 'row 1 also overwrote row 0\'s thirteenth vec4');

  /* Defaults are the ones the shader assumes for a material authored before a
     field existed; every one of them is a value, not a zero, and getting one
     wrong changes every material in the library at once. */
  const r0 = table.subarray(0, MAT_STRIDE_FLOATS);
  for (const [slot, want, name] of [
    [3, 0, 'metallic'], [4, 0.4, 'rough'], [5, 1.5, 'ior'], [6, 0, 'trans'], [7, 1, 'alpha'],
    [13, 0.05, 'coatRough'], [14, 0, 'kind'],
    [31, 1, 'absorb — written for EVERY material, not only for glass'],
    [36, 1, 'bumpScale'], [41, 0.08, 'frontOp'], [42, 0.70, 'edgeOp'], [43, 3, 'falloff'],
    [44, 1.4, 'filmIor'], [45, 0, 'thicknessNm'], [48, 0, 'reliefDepth'],
  ]) assert.equal(r0[slot], f32(want), `${name} defaults to ${want} at float ${slot}`);
  for (let k = 49; k < MAT_STRIDE_FLOATS; k += 1) assert.equal(r0[k], 0, `float ${k} is free and must be zero`);

  // Emission is pre-multiplied here because the shader adds it with no second
  // multiply; leaving the multiply to the shader makes every emitter unit-bright.
  const lamp = packMaterials([{ base: [1, 1, 1], emis: [1, 0.5, 0.25], emisStr: 8 }]);
  assert.equal(lamp[8], 8, 'emis.r is pre-multiplied by emisStr');
  assert.equal(lamp[9], 4, 'emis.g is pre-multiplied by emisStr');
  assert.equal(lamp[10], 2, 'emis.b is pre-multiplied by emisStr');
  console.log('  material row:   stride 52 floats (13 vec4), defaults and pre-multiplied emission hold');
}

// ------------------------------------------- the pattern exclusion chain ----
/* marble, woven, grain, grid and mottle share the SAME sixteen floats under
   different names. A material carrying two must resolve to exactly one, in the
   documented order, or the shader reads the sub-fields under the wrong names
   and draws a plausible wrong texture. */
{
  const kindOf = (m) => packMaterials([m])[14];
  const base = [0.5, 0.5, 0.5];
  assert.equal(kindOf({ base }), 0, 'no pattern is kind 0');
  assert.equal(kindOf({ base, marble: { structure: 'breccia' } }), 1, 'marble is kind 1');
  assert.equal(kindOf({ base, woven: { structure: 'twill' } }), 2, 'woven is kind 2');
  assert.equal(kindOf({ base, grain: {} }), 3, 'grain is kind 3');
  assert.equal(kindOf({ base, grid: { shape: 'circular' } }), 4, 'grid is kind 4');
  assert.equal(kindOf({ base, mottle: {} }), 5, 'mottle is kind 5');

  assert.equal(kindOf({ base, marble: {}, woven: {}, grain: {}, grid: {}, mottle: {} }), 1, 'marble outranks every later family');
  assert.equal(kindOf({ base, woven: {}, grain: {}, grid: {}, mottle: {} }), 2, 'woven outranks grain, grid and mottle');
  assert.equal(kindOf({ base, grain: {}, grid: {}, mottle: {} }), 3, 'grain outranks grid and mottle');
  assert.equal(kindOf({ base, grid: {}, mottle: {} }), 4, 'grid outranks mottle');
  // A medium replaces the surface outright, so it suppresses the families below
  // it — but the volume machinery is starved, so kind 6 is never emitted.
  assert.equal(kindOf({ base, medium: { albedo: [1, 1, 1] }, marble: {} }), 0, 'a medium suppresses the pattern chain and never emits kind 6');

  const marbled = packMaterials([{ base, marble: { structure: 'speckled', vein: [0.1, 0.2, 0.3], clast: [0.4, 0.5, 0.6], veinScale: 7 } }]);
  assert.equal(marbled[15], 3, 'speckled is structure 3');
  assert.deepEqual([marbled[16], marbled[17], marbled[18]], [0.1, 0.2, 0.3].map(f32), 'veinCol lands in the fifth vec4');
  assert.equal(marbled[19], 7, 'veinScale rides that vec4\'s w');
  assert.equal(marbled[20], f32(0.4), 'clastCol lands in the sixth vec4');

  const woven = packMaterials([{ base, woven: { structure: 'satin', thread2: [0.9, 0.8, 0.7], pitch: 22 } }]);
  assert.equal(woven[15], 2, 'satin is structure 2');
  assert.equal(woven[19], 22, 'woven pitch reuses marble\'s veinScale slot');
  assert.equal(woven[20], 0, 'woven leaves marble\'s clast slot alone');

  const mottle = packMaterials([{ base, mottle: { tone2: [0.3, 0.3, 0.3], tone3: [0.9, 0.9, 0.9], soft: 0.5 } }]);
  assert.equal(mottle[20], f32(0.9), 'mottle is the one family that uses the clast slot, for a third tone');
  assert.equal(mottle[47], 0.5, 'mottleSoft rides the twelfth vec4\'s w');
  console.log('  pattern chain:  six kinds resolve in order, sub-fields land in the slots getMat reads');
}

// ------------------------------------------ the real library, packed once ----
/* A hundred authored materials through the packer. The value here is not the
   count — it is that a field the packer reaches for and does not find must
   produce a DEFAULT and never a NaN. One NaN in a material row poisons every
   pixel that material touches, and it arrives from an object where a number was
   expected, which is exactly what an authored library is full of. */
{
  const lib = JSON.parse(readFileSync(new URL('../rendre_assets/materials.json', import.meta.url), 'utf8')).materials;
  assert.ok(lib.length >= 100, `the library has ${lib.length} materials`);
  const packed = packMaterials(lib);
  assert.equal(packed.length, lib.length * MAT_STRIDE_FLOATS, 'the packed table is one stride per material');
  let bad = 0, kinds = new Set();
  for (let i = 0; i < packed.length; i += 1) if (!Number.isFinite(packed[i])) bad += 1;
  for (let i = 0; i < lib.length; i += 1) kinds.add(packed[i * MAT_STRIDE_FLOATS + 14]);
  assert.equal(bad, 0, `${bad} non-finite floats in the packed library`);
  assert.ok(!kinds.has(6), 'no kind 6 row was emitted — the volume machinery is starved, not fed');
  assert.ok(kinds.size > 1, 'the library exercises more than one pattern kind');
  console.log(`  library:        ${lib.length} authored materials pack to ${packed.length} finite floats, kinds {${[...kinds].sort().join(',')}}`);
}

// --------------------------------------------------- reorder in isolation ----
{
  const { pos, nrm, mp, nT: n } = gatherGeometry(parts);
  const identity = new Uint32Array(n);
  for (let i = 0; i < n; i += 1) identity[i] = i;
  const { tris, norms } = reorderToGPU(pos, nrm, mp, n, identity);
  const u = new Uint32Array(tris.buffer);
  for (let i = 0; i < n; i += 1) {
    assert.equal(tris[i * TRI_STRIDE_FLOATS], f32(pos[i * 9]), `identity reorder moved triangle ${i}`);
    assert.equal(u[i * TRI_STRIDE_FLOATS + 3], mp[i], `identity reorder lost triangle ${i}'s part`);
    assert.equal(norms[i * TRI_STRIDE_FLOATS + 2], f32(nrm[i * 9 + 2]), `identity reorder moved triangle ${i}'s normal`);
    assert.equal(tris[i * TRI_STRIDE_FLOATS + 7], 0, 'v1.w is unused and must be zero');
    assert.equal(tris[i * TRI_STRIDE_FLOATS + 11], 0, 'v2.w is unused and must be zero');
  }
  console.log(`  reorder:        the identity permutation is a no-op on all three arrays (${n} triangles)`);
}

console.log('rendrepack: ok');
