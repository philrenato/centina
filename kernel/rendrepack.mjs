// THE PATH TRACER'S BUFFERS, PACKED THE WAY ITS SHADER READS THEM.
//
// Everything the trace shader binds as geometry or material is `array<vec4f>`
// or `array<u32>` — never a WGSL struct. That is deliberate on the shader's
// side and it is what makes this module checkable: with no struct in a storage
// buffer there is no shader-side padding for a float-counting packer to
// disagree with, and the only rule left is that every stride is a whole number
// of 16-byte `vec4`.
//
// Three layouts live here, and each one has a failure that produces a PLAUSIBLE
// picture rather than an error:
//
//   tris   3 vec4 / triangle   v0.w carries the owning part index, AS U32 BITS
//   norms  3 vec4 / triangle   per-corner shading normals, .w unused
//   mats  13 vec4 / material   52 floats — see MAT_STRIDE_FLOATS below
//
// plus `partMats`, one `u32` per part, and the BVH nodes from `bvh.mjs` in
// their GPU form (2 vec4 / node).
//
// ⚠⚠ THE PERMUTATION MUST REACH ALL THREE ARRAYS. Triangles are stored in BVH
// order so a leaf is a contiguous run, which means positions, normals and the
// per-triangle part index are all permuted by the same `order`. Permuting two
// of the three is the classic way to get a picture that is right in every
// respect a silhouette can show and wrong in every material: objects wearing
// each other's finishes, no error anywhere.
//
// ⚠⚠ AND THE INDEX FIELDS ARE U32 BITS SITTING IN FLOAT SLOTS. The shader reads
// them with `bitcast<u32>`, so a part index written as the FLOAT 3.0 arrives as
// 1077936128. WGSL clamps that out-of-range read instead of faulting, so every
// triangle in the scene resolves to the same clamped material row — which reads
// as "materials aren't working at all" rather than as an index bug, and sends
// the search to the material table. Every index below is written through a
// `Uint32Array` aliased on the same ArrayBuffer.

import { buildBVH, packBVHForGPU } from './bvh.mjs';

/* MATERIAL STRIDE, MEASURED FROM THE SHADER RATHER THAN FROM PROSE.
   `getMat` loads `mats[i*13u]` through `mats[i*13u+12u]` — thirteen vec4, so
   52 floats / 208 bytes per row. The buffer's own binding comment says 12 vec4
   and is stale; it drifted when the struct grew for thin-film interference and
   nothing forced it to move. Taking 48 from that comment leaves row 0 correct
   and every later row shifted four floats further wrong, which reads as "some
   materials are broken" and never points at the stride. */
export const MAT_STRIDE_FLOATS = 52;

/** Floats per triangle in `tris` and in `norms`: 3 vec4. */
export const TRI_STRIDE_FLOATS = 12;

/** Floats per BVH node in the GPU buffer: 2 vec4. */
export const NODE_STRIDE_FLOATS = 8;

/** Bit 31 of a `partMats` entry means hidden; the low 31 bits are the row. */
export const PART_HIDDEN_BIT = 0x80000000;
export const PART_ROW_MASK = 0x7fffffff;

/* The three pattern enumerations the shader branches on, by the names the
   material library authors them under. A name the shader has no arm for falls
   to 0, which is each family's plain/original case. */
export const MARBLE_STRUCT = { veined: 0, clouded: 1, breccia: 2, speckled: 3, uniform: 4 };
export const WEAVE_STRUCT = { plain: 0, twill: 1, satin: 2, basket: 3, rib: 4 };
export const GRID_SHAPE = { square: 0, circular: 1, diamond: 2 };

/**
 * Write one material into `f` at float offset `o`.
 *
 * ⚠ THE ROW MUST BE ZERO BEFORE THIS RUNS. Unwritten floats are meaningful —
 * every default in the table is "whatever a fresh Float32Array holds" for the
 * fields this does not touch — so a row is packed into fresh storage and never
 * patched in place over an older material.
 *
 * ⚠⚠ THE PATTERN FAMILIES ARE MUTUALLY EXCLUSIVE AND ORDERED. marble, woven,
 * grain, grid and mottle all reuse the SAME sixteen floats with different
 * meanings, selected by `kind`. A material carrying two of them must resolve to
 * exactly one, in this order, or the sub-fields are read under the wrong names
 * and the surface gets a plausible wrong texture.
 */
export function matPack(f, o, m) {
  f[o] = m.base[0]; f[o + 1] = m.base[1]; f[o + 2] = m.base[2]; f[o + 3] = m.metallic || 0;
  f[o + 4] = m.rough != null ? m.rough : 0.4; f[o + 5] = m.ior || 1.5; f[o + 6] = m.trans || 0; f[o + 7] = m.alpha != null ? m.alpha : 1;
  /* emis is PRE-MULTIPLIED by its strength here. The shader adds `T * m.emis`
     with no second multiply, so leaving the multiply to the shader would make
     every emitter unit-bright. */
  const es = m.emisStr || 0;
  f[o + 8] = (m.emis ? m.emis[0] : 0) * es; f[o + 9] = (m.emis ? m.emis[1] : 0) * es; f[o + 10] = (m.emis ? m.emis[2] : 0) * es;
  f[o + 11] = m.sheen || 0;
  f[o + 12] = m.coat || 0; f[o + 13] = m.coatRough != null ? m.coatRough : 0.05;
  /* Beer-Lambert absorption density, written for EVERY material and not only
     for glass: 1 = the base tint at one scene unit, 0 = perfectly clear. A
     transmissive material that never had this field authored still needs the
     default, and zero here would make every tinted glass colorless. */
  f[o + 31] = m.absorb != null ? m.absorb : 1.0;
  f[o + 32] = m.aniso || 0; f[o + 33] = m.sss || 0;
  /* Spectral dispersion on the transmission lobe, 0 = achromatic glass.
     Rendre also accepts an authored ABBE NUMBER here and converts it, but that
     conversion is not part of the extracted source and no material in the
     library carries one — so `dispersion` is taken as authored, and a caller
     that wants to work in Abbe numbers must resolve one to a dispersion
     strength before packing rather than have this guess at the curve. */
  f[o + 34] = m.dispersion || 0;
  f[o + 35] = m.bump || 0;
  f[o + 36] = m.bumpScale != null ? m.bumpScale : 1.0;
  const ga = m.grainAxis; f[o + 37] = ga ? ga[0] : 0; f[o + 38] = ga ? ga[1] : 0; f[o + 39] = ga ? ga[2] : 0;
  f[o + 40] = m.phantom ? 1 : 0; f[o + 41] = m.frontOp != null ? m.frontOp : 0.08; f[o + 42] = m.edgeOp != null ? m.edgeOp : 0.70; f[o + 43] = m.falloff != null ? m.falloff : 3.0;
  f[o + 44] = m.filmIor != null ? m.filmIor : 1.4; f[o + 45] = m.thicknessNm || 0;
  f[o + 46] = m.isolate || 0;
  /* A DISTANCE in world units, not a strength — this is how far the surface is
     carved DOWN. Deliberately not derived from `bump`, which is a dimensionless
     shading amount: deriving one from the other would silently carve every
     bumped material in the library. */
  f[o + 48] = m.relief || 0;

  /* A participating medium REPLACES the surface material outright, so it
     outranks every pattern rather than joining the tail of the chain. The
     volume machinery is not part of this transplant and the shader's medium
     arms are starved rather than removed, so `kind` is clamped to <= 5 here and
     kind 6 is never emitted. The medium still suppresses the pattern families
     below it — dropping it from the chain instead would pack a medium
     material's marble as if the medium were not there. */
  const md = m.medium || null;
  const mb = !md ? m.marble : null;
  const wv = (!md && !mb) ? m.woven : null;
  const gn = (!md && !mb && !wv) ? m.grain : null;
  const gd = (!md && !mb && !wv && !gn) ? m.grid : null;
  const mt = (!md && !mb && !wv && !gn && !gd) ? m.mottle : null;
  f[o + 14] = md ? 0 : (mb ? 1 : (wv ? 2 : (gn ? 3 : (gd ? 4 : (mt ? 5 : 0)))));
  f[o + 15] = mb ? (MARBLE_STRUCT[mb.structure] || 0) : (wv ? (WEAVE_STRUCT[wv.structure] || 0) : (gd ? (GRID_SHAPE[gd.shape] || 0) : 0));

  if (mb) {
    const vc = mb.vein || [0.5, 0.5, 0.5], cc = mb.clast || mb.vein || m.base;
    f[o + 16] = vc[0]; f[o + 17] = vc[1]; f[o + 18] = vc[2]; f[o + 19] = mb.veinScale != null ? mb.veinScale : 2.5;
    f[o + 20] = cc[0]; f[o + 21] = cc[1]; f[o + 22] = cc[2]; f[o + 23] = mb.veinContrast != null ? mb.veinContrast : 0.4;
    f[o + 24] = mb.warp != null ? mb.warp : 0.8; f[o + 25] = mb.redox != null ? mb.redox : 0.5;
    f[o + 26] = mb.speckle != null ? mb.speckle : 0; f[o + 27] = mb.clastScale != null ? mb.clastScale : 3;
    f[o + 28] = mb.clastRand != null ? mb.clastRand : 0.9; f[o + 29] = mb.hue != null ? mb.hue : 0; f[o + 30] = mb.seed != null ? mb.seed : 0;
  } else if (wv) {
    // Woven borrows marble's own slots: second thread color, pitch, contrast,
    // yarn irregularity. Same struct, no growth.
    const tc = wv.thread2 || [0.5, 0.5, 0.5];
    f[o + 16] = tc[0]; f[o + 17] = tc[1]; f[o + 18] = tc[2]; f[o + 19] = wv.pitch != null ? wv.pitch : 18.0;
    f[o + 23] = wv.contrast != null ? wv.contrast : 0.5;
    f[o + 24] = wv.irregular != null ? wv.irregular : 0.3;
    f[o + 29] = wv.hue != null ? wv.hue : 0; f[o + 30] = wv.seed != null ? wv.seed : 0;
  } else if (gn) {
    const rc = gn.ringCol || [0.3, 0.2, 0.1];
    f[o + 16] = rc[0]; f[o + 17] = rc[1]; f[o + 18] = rc[2]; f[o + 19] = gn.scale != null ? gn.scale : 2.5;
    f[o + 23] = gn.contrast != null ? gn.contrast : 0.5;
    f[o + 24] = gn.fiber != null ? gn.fiber : 0.8;
    f[o + 29] = gn.hue != null ? gn.hue : 0; f[o + 30] = gn.seed != null ? gn.seed : 0;
  } else if (gd) {
    // Grid's cell shape rides the same structure slot marble's stone type uses.
    const bc = gd.barCol || [0.1, 0.1, 0.1];
    f[o + 16] = bc[0]; f[o + 17] = bc[1]; f[o + 18] = bc[2]; f[o + 19] = gd.pitch != null ? gd.pitch : 18.0;
    f[o + 23] = gd.thickness != null ? gd.thickness : 0.5;
    f[o + 29] = gd.hue != null ? gd.hue : 0; f[o + 30] = gd.seed != null ? gd.seed : 0;
  } else if (mt) {
    // Mottle is the one family that needed a real THIRD color, and takes
    // marble's clast slot for it — the slot woven, grain and grid leave alone.
    const t2 = mt.tone2 || [0.3, 0.3, 0.2], t3 = mt.tone3 || [0.15, 0.15, 0.1];
    f[o + 16] = t2[0]; f[o + 17] = t2[1]; f[o + 18] = t2[2]; f[o + 19] = mt.scale != null ? mt.scale : 2.5;
    f[o + 20] = t3[0]; f[o + 21] = t3[1]; f[o + 22] = t3[2];
    f[o + 24] = mt.irregular != null ? mt.irregular : 0.8;
    f[o + 30] = mt.seed != null ? mt.seed : 0;
    f[o + 47] = mt.soft != null ? mt.soft : 0;
  }
  return f;
}

/**
 * The whole material table as one `Float32Array(n * 52)`.
 *
 * Built fresh on every call rather than edited in place, because the zero fill
 * is load-bearing: see `matPack`.
 */
export function packMaterials(materials) {
  const n = Math.max(materials.length, 0);
  const f = new Float32Array(n * MAT_STRIDE_FLOATS);
  for (let i = 0; i < n; i += 1) matPack(f, i * MAT_STRIDE_FLOATS, materials[i]);
  return f;
}

/**
 * `partMats[i] = (row & 0x7fffffff) | (visible ? 0 : 0x80000000)`.
 *
 * This is the CHEAP buffer, and keeping it separate from the geometry is what
 * makes hiding a part or reassigning its material a few hundred bytes of
 * upload instead of a BVH rebuild.
 *
 * ⚠ NEVER ZERO-LENGTH. A zero-length storage buffer is a WebGPU validation
 * error, so an empty scene still gets one entry — the same reason
 * `gatherGeometry` synthesizes a degenerate triangle.
 */
export function packPartMats(parts) {
  const out = new Uint32Array(Math.max(parts.length, 1));
  for (let i = 0; i < parts.length; i += 1) {
    const p = parts[i];
    const row = (p.material >>> 0) & PART_ROW_MASK;
    out[i] = p.visible === false ? (row | PART_HIDDEN_BIT) >>> 0 : row;
  }
  return out;
}

/**
 * Concatenate every part's triangles into one unindexed soup, recording which
 * part each triangle came from.
 *
 * `parts[i]` supplies `pos` and `nrm` as `Float32Array(n*9)` — three corners of
 * three components, already in world space — and the ARRAY POSITION of a part
 * is its part index, which is also how `partMats` is indexed. The two orders
 * are the same order by construction; keeping two independent orderings is how
 * the material index and the geometry come apart.
 *
 * ⚠ AN EMPTY SCENE STILL PRODUCES ONE TRIANGLE. A degenerate, never-hit
 * triangle keeps the buffer, the bind group and the traversal on their normal
 * paths; the alternative is a zero-length buffer, which WebGPU rejects, and
 * three special cases downstream to avoid it.
 */
export function gatherGeometry(parts) {
  let nT = 0;
  for (const p of parts) nT += Math.floor(p.pos.length / 9);

  if (nT === 0) {
    return { pos: new Float32Array(9), nrm: new Float32Array(9), mp: new Uint32Array(1), nT: 1, degenerate: true };
  }
  const pos = new Float32Array(nT * 9), nrm = new Float32Array(nT * 9), mp = new Uint32Array(nT);
  let t = 0;
  for (let pi = 0; pi < parts.length; pi += 1) {
    const p = parts[pi], n = Math.floor(p.pos.length / 9);
    if (!n) continue;
    pos.set(p.pos.subarray(0, n * 9), t * 9);
    if (p.nrm) nrm.set(p.nrm.subarray(0, n * 9), t * 9);
    for (let i = 0; i < n; i += 1) mp[t + i] = pi;
    t += n;
  }
  return { pos, nrm, mp, nT, degenerate: false };
}

/**
 * Positions, normals and part indices permuted into BVH order and widened to
 * the shader's 3-vec4 stride.
 *
 * `order[i]` is the ORIGINAL index of the triangle that now lives at slot `i`,
 * so every read is `order[i]` and every write is `i`. Reading and writing
 * through the same index is a permutation that silently does nothing on an
 * already-sorted scene and scrambles a real one.
 */
export function reorderToGPU(pos, nrm, mp, nT, order) {
  const tris = new Float32Array(nT * TRI_STRIDE_FLOATS);
  const norms = new Float32Array(nT * TRI_STRIDE_FLOATS);
  // Aliased on the SAME ArrayBuffer: this is the only way the part index
  // reaches the shader as u32 bits rather than as a float of that value.
  const triU = new Uint32Array(tris.buffer);
  for (let i = 0; i < nT; i += 1) {
    const s = order[i], so = s * 9, d = i * TRI_STRIDE_FLOATS;
    tris[d] = pos[so]; tris[d + 1] = pos[so + 1]; tris[d + 2] = pos[so + 2]; triU[d + 3] = mp[s];
    tris[d + 4] = pos[so + 3]; tris[d + 5] = pos[so + 4]; tris[d + 6] = pos[so + 5]; tris[d + 7] = 0;
    tris[d + 8] = pos[so + 6]; tris[d + 9] = pos[so + 7]; tris[d + 10] = pos[so + 8]; tris[d + 11] = 0;
    norms[d] = nrm[so]; norms[d + 1] = nrm[so + 1]; norms[d + 2] = nrm[so + 2]; norms[d + 3] = 0;
    norms[d + 4] = nrm[so + 3]; norms[d + 5] = nrm[so + 4]; norms[d + 6] = nrm[so + 5]; norms[d + 7] = 0;
    norms[d + 8] = nrm[so + 6]; norms[d + 9] = nrm[so + 7]; norms[d + 10] = nrm[so + 8]; norms[d + 11] = 0;
  }
  return { tris, norms };
}

/**
 * Gather, build, permute and pack — the whole geometry side of a frame's
 * upload, as pure arrays.
 *
 * Returns `{ tris, norms, nodes, partMats, order, nodeCount, triangleCount,
 * maxDepth, leaves }`. `nodes` is the GPU form, 2 vec4 per node with both index
 * fields as u32 bits; `bvh` is the CPU-side tree kept alongside for querying
 * the same structure from JS.
 *
 * ⚠ THE TRAVERSAL STACK IS 40 DEEP on the shader side and a deeper tree loses a
 * subtree with nothing said, so `maxDepth` comes back for the caller to check.
 * With a leaf target of four and a split that always separates, depth is
 * logarithmic and 40 is a very long way off — but it is a fixed number in the
 * shader, so it is a property of the tree, not an assumption about it.
 */
export function packSceneForGPU(parts, opts = {}) {
  const { pos, nrm, mp, nT, degenerate } = gatherGeometry(parts);
  const bvh = buildBVH(pos, opts);
  const { tris, norms } = reorderToGPU(pos, nrm, mp, nT, bvh.order);
  return {
    tris,
    norms,
    nodes: packBVHForGPU(bvh),
    partMats: packPartMats(parts),
    order: bvh.order,
    nodeCount: bvh.nodeCount,
    triangleCount: nT,
    maxDepth: bvh.maxDepth,
    leaves: bvh.leaves,
    degenerate,
    bvh,
  };
}
