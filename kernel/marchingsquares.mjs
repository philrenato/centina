// MARCHING SQUARES (the PAINT tool's contour extractor)
// ================================================================
// Trace iso-contour(s) of a scalar field sampled on a regular grid, at a
// chosen threshold — the standard, textbook 2D marching-squares algorithm
// (the doc's own "genuinely tractable, cheap marching-squares-in-UV math").
//
// The field is `values`, a flat array of length uCount*vCount, indexed
// `values[i * vCount + j]` where i ∈ [0,uCount) runs the U direction and
// j ∈ [0,vCount) runs the V direction. Each entry is the scalar value at a
// grid NODE. A CELL spans four adjacent nodes; the contour crosses a cell
// edge wherever the two endpoint values straddle the threshold, linearly
// interpolated to the exact crossing.
//
// Output: an array of polylines, each `{ closed, pts }`, where `pts` is an
// array of `[ci, cj]` CONTINUOUS grid-index coordinates (ci ∈ [0,uCount] in
// the wrapU case, else [0,uCount-1]; cj likewise). The CALLER maps each
// [ci,cj] to a real surface (u,v) parameter and then to a 3D point — this
// module knows nothing about NURBS, only the scalar grid, exactly so it
// stays a small, node-testable, geometry-agnostic primitive.
//
// SEAM WRAP (a real PAINT build requirement, not an afterthought): a surface
// closed in a direction (a Revolve/Cylinder closed in U or V — an entirely
// ordinary single-face case, not a polysurface edge) has its two domain
// ends be the SAME physical location. `opts.wrapU`/`opts.wrapV` add the ring
// of cells that bridge node (N-1) back to node 0, so a contour crossing the
// seam traces as ONE continuous curve rather than two disconnected halves.
// The wrap cell's far edge is emitted at continuous index uCount (not 0), so
// the caller's index→param map lands it correctly in the seam region; the
// endpoint-chaining fold (folding index uCount ≡ 0) is what actually stitches
// the two halves into one polyline — see `foldKey` below.

// Corner order per cell (i,j): c0 = (i,j), c1 = (i+1,j), c2 = (i+1,j+1),
// c3 = (i,j+1). Case bit k is set when corner k's value >= threshold.
// Edge A joins c0-c1 (bottom), B joins c1-c2 (right), C joins c2-c3 (top),
// D joins c3-c0 (left). Each non-ambiguous case emits ONE segment between
// two edge crossings; the two saddle cases (5, 10) emit two, disambiguated
// by the cell-center average (the standard resolution).
const SEG_TABLE = {
  1: [['D', 'A']],
  2: [['A', 'B']],
  3: [['D', 'B']],
  4: [['B', 'C']],
  6: [['A', 'C']],
  7: [['D', 'C']],
  8: [['C', 'D']],
  9: [['C', 'A']],
  11: [['B', 'C']],
  12: [['B', 'D']],
  13: [['A', 'B']],
  14: [['A', 'D']],
};

export function marchingSquares(values, uCount, vCount, threshold, opts = {}) {
  const wrapU = !!opts.wrapU, wrapV = !!opts.wrapV;
  if (uCount < 2 || vCount < 2) return [];
  const val = (ci, cj) => {
    const i = wrapU ? ((ci % uCount) + uCount) % uCount : ci;
    const j = wrapV ? ((cj % vCount) + vCount) % vCount : cj;
    return values[i * vCount + j];
  };
  // Edge crossing coordinate, in continuous [ci,cj] index space. `lerp`
  // returns the fraction along the edge where the field equals threshold.
  const lerp = (va, vb) => {
    const d = vb - va;
    if (Math.abs(d) < 1e-12) return 0.5; // both endpoints equal the threshold — degenerate; a harmless midpoint
    return (threshold - va) / d;
  };
  const nCellsU = wrapU ? uCount : uCount - 1;
  const nCellsV = wrapV ? vCount : vCount - 1;
  const segments = [];
  for (let i = 0; i < nCellsU; i++) {
    for (let j = 0; j < nCellsV; j++) {
      const v0 = val(i, j), v1 = val(i + 1, j), v2 = val(i + 1, j + 1), v3 = val(i, j + 1);
      const b0 = v0 >= threshold ? 1 : 0, b1 = v1 >= threshold ? 1 : 0, b2 = v2 >= threshold ? 1 : 0, b3 = v3 >= threshold ? 1 : 0;
      let cs = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
      if (cs === 0 || cs === 15) continue;
      const edgePt = (e) => {
        switch (e) {
          case 'A': return [i + lerp(v0, v1), j];
          case 'B': return [i + 1, j + lerp(v1, v2)];
          case 'C': return [i + lerp(v3, v2), j + 1];
          case 'D': return [i, j + lerp(v0, v3)];
        }
      };
      let pairs;
      if (cs === 5 || cs === 10) {
        const centre = (v0 + v1 + v2 + v3) / 4;
        const cAbove = centre >= threshold;
        if (cs === 5) pairs = cAbove ? [['A', 'B'], ['C', 'D']] : [['D', 'A'], ['B', 'C']];
        else pairs = cAbove ? [['D', 'A'], ['B', 'C']] : [['A', 'B'], ['C', 'D']];
      } else {
        pairs = SEG_TABLE[cs];
      }
      for (const [e1, e2] of pairs) segments.push([edgePt(e1), edgePt(e2)]);
    }
  }
  return chainSegments(segments, uCount, vCount, wrapU, wrapV);
}

// Stitch the loose per-cell segments into continuous polylines. Endpoints on
// a shared cell edge carry identical coordinates (same edge, same interpolant
// from the same node values), so a quantized key matches them. The seam fold
// (index uCount ≡ 0 when wrapU) is what lets the two halves of a seam-crossing
// contour join into ONE polyline.
function chainSegments(segments, uCount, vCount, wrapU, wrapV) {
  const Q = 1e6;
  const foldKey = (p) => {
    let ci = p[0], cj = p[1];
    if (wrapU) ci = ((ci % uCount) + uCount) % uCount;
    if (wrapV) cj = ((cj % vCount) + vCount) % vCount;
    return Math.round(ci * Q) + ',' + Math.round(cj * Q);
  };
  // Adjacency: key -> list of { pt, other } (each undirected segment appears twice).
  const adj = new Map();
  const addHalf = (a, b) => {
    const k = foldKey(a);
    if (!adj.has(k)) adj.set(k, []);
    adj.get(k).push({ pt: a, other: b });
  };
  for (const [a, b] of segments) { addHalf(a, b); addHalf(b, a); }
  const usedSeg = new Set();
  const segId = (a, b) => { const ka = foldKey(a), kb = foldKey(b); return ka < kb ? ka + '|' + kb : kb + '|' + ka; };
  const polylines = [];
  // Prefer to START chains at OPEN ends (a key with an odd/single incidence),
  // so an open contour is walked end-to-end before any closed loops are
  // picked up — matters for the honest open-vs-closed classification.
  const startKeys = [...adj.keys()].sort((ka, kb) => adj.get(ka).length - adj.get(kb).length);
  for (const startK of startKeys) {
    const startList = adj.get(startK);
    for (const startHalf of startList) {
      const sid = segId(startHalf.pt, startHalf.other);
      if (usedSeg.has(sid)) continue;
      // Walk forward from this segment as far as it chains.
      const pts = [startHalf.pt];
      let curPt = startHalf.pt, nextPt = startHalf.other;
      let closed = false;
      while (true) {
        const sidStep = segId(curPt, nextPt);
        if (usedSeg.has(sidStep)) break;
        usedSeg.add(sidStep);
        pts.push(nextPt);
        const k = foldKey(nextPt);
        if (k === foldKey(startHalf.pt) && pts.length > 2) { closed = true; break; }
        const cands = adj.get(k) || [];
        let advanced = false;
        for (const c of cands) {
          const csid = segId(c.pt, c.other);
          if (usedSeg.has(csid)) continue;
          curPt = c.pt; nextPt = c.other; advanced = true; break;
        }
        if (!advanced) break;
      }
      if (pts.length >= 2) polylines.push({ closed, pts });
    }
  }
  return polylines;
}
