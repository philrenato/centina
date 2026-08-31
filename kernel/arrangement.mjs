// PLANAR ARRANGEMENT — a five-stage algorithm: given a
// set of 2D polylines (already-flattened curves, in a shared plane's own
// (u,v) coordinates — this module knows nothing about NURBS or 3D at all,
// exactly the same "kernel is curve-agnostic 2D math" posture kernel/
// trim.mjs's own polygon helpers already have), find every CLOSED BOUNDED
// REGION the polylines form together — including regions that exist only
// because two curves happen to cross, never individually drawn closed.
//
// Reference technique: planar straight-line graph (PSLG) -> half-edge
// (DCEL) structure -> leftmost-turn face traversal -> hole-to-face
// assignment via point-in-polygon containment. Standard, present under some
// name in every computational-geometry reference and in OCCT's own 2D
// face-building code (CGAL's Arrangement_2 solves the identical problem) —
// read for TECHNIQUE, written fresh here in this kernel's own idiom. Those
// sources are a REFERENCE for the technique, never transcribed code.
// The five-stage shape below is deliberate. Two of the
// stages are correctness requirements rather than conveniences and are easy to
// leave out: hole assignment (Stage 5), without which a region containing an
// island reports the island's area as its own, and dangling-edge pruning
// (Stage 3), without which an edge with a free end joins no face and the walk
// never closes.

import { signedArea2D, pointInUVPolygon } from './trim.mjs';

// ----------------------------------------------------------------
// STAGE 2 (part 1): robust segment-segment intersection, including the
// cases kernel/trim.mjs's own `segmentsIntersect` explicitly excludes
// (its own comment: "collinear/touching edge cases are a real, separate
// robustness concern — not silently claimed handled here") — that
// review named these as the COMMON case for Shapereason's real inputs
// (T-junctions from ordinary endpoint snapping; exactly-coincident
// segments from a zero-offset Copy/Paste), not a rare edge case worth
// deferring. Returns every real intersection point between segment
// A=(p0,p1) and B=(p2,p3) as {t, s} pairs (parameters along each segment,
// 0..1) — ONE point for a proper crossing or a single T-junction touch,
// TWO points (the overlap interval's own two ends) for a genuine
// collinear overlap, zero for no intersection at all.
function segmentIntersections(p0, p1, p2, p3, tol) {
  const ax = p1[0] - p0[0], ay = p1[1] - p0[1];
  const bx = p3[0] - p2[0], by = p3[1] - p2[1];
  const denom = ax * by - ay * bx;
  const cross = (ux, uy, vx, vy) => ux * vy - uy * vx;
  if (Math.abs(denom) > tol * tol) {
    // Non-parallel — the ordinary, exact-solve case (Cramer's rule).
    const dx = p2[0] - p0[0], dy = p2[1] - p0[1];
    const t = cross(dx, dy, bx, by) / denom;
    const s = cross(dx, dy, ax, ay) / denom;
    const eps = 1e-9;
    if (t >= -eps && t <= 1 + eps && s >= -eps && s <= 1 + eps) {
      return [{ t: Math.max(0, Math.min(1, t)), s: Math.max(0, Math.min(1, s)) }];
    }
    return [];
  }
  // Parallel (denom ~ 0) — either collinear (real overlap possible) or
  // genuinely separate parallel lines (no intersection at all). Test
  // collinearity via the cross product of (p2-p0) against A's own
  // direction — zero (within tolerance, scaled by A's own length) means
  // p2 lies ON the infinite line through p0/p1.
  const dx0 = p2[0] - p0[0], dy0 = p2[1] - p0[1];
  const aLen = Math.hypot(ax, ay) || 1;
  if (Math.abs(cross(dx0, dy0, ax, ay)) > tol * aLen) return []; // parallel, not collinear
  // Collinear — project every one of the 4 endpoints onto A's own
  // direction (a 1D parametrization shared by both segments since they
  // sit on the same line) and intersect the two resulting 1D intervals.
  // `projB` (B's own t-along-B parametrization) is defined up front so
  // BOTH the single-touch and genuine-overlap branches below can convert
  // any point on the shared line back to B's own parameter the same way.
  const aLenSq = ax * ax + ay * ay || 1;
  const bLenSq = bx * bx + by * by || 1;
  const projA = (px, py) => ((px - p0[0]) * ax + (py - p0[1]) * ay) / aLenSq; // t-along-A units
  const projB = (px, py) => ((px - p2[0]) * bx + (py - p2[1]) * by) / bLenSq; // t-along-B units
  const aLo = 0, aHi = 1;
  let bLo = projA(p2[0], p2[1]), bHi = projA(p3[0], p3[1]);
  if (bLo > bHi) [bLo, bHi] = [bHi, bLo];
  const lo = Math.max(aLo, bLo), hi = Math.min(aHi, bHi);
  if (hi < lo - 1e-9) return []; // disjoint collinear segments
  if (hi < lo + 1e-9) { // touching at a single point, not a real overlap span
    const t = Math.max(0, Math.min(1, lo));
    const px = p0[0] + ax * t, py = p0[1] + ay * t;
    return [{ t, s: Math.max(0, Math.min(1, projB(px, py))) }];
  }
  // A genuine overlap SPAN — return both ends as real split points on
  // EACH segment.
  const out = [];
  for (const t of [lo, hi]) {
    const px = p0[0] + ax * t, py = p0[1] + ay * t;
    out.push({ t, s: Math.max(0, Math.min(1, projB(px, py))) });
  }
  return out;
}

// ----------------------------------------------------------------
// STAGES 1-2: flatten (the caller's own job — this function takes already-
// flattened 2D polylines) -> intersect every segment pair -> split into a
// PLANAR STRAIGHT-LINE GRAPH (PSLG): a shared vertex list (welded within
// `weldTolerance`) plus an edge list where no two edges cross except at a
// shared, recorded vertex.
// ----------------------------------------------------------------
// `sources` (optional) is a tag per input polyline. When supplied, the
// returned `edgeSources[i]` is the set of tags that produced `edges[i]` —
// a SET rather than a single tag because two coincident input segments
// dedupe into one edge, and a consumer splitting a trimmed face needs to
// know when an intersection curve runs exactly along a trim boundary
// rather than crossing the interior. Omitting `sources` leaves every
// downstream result byte-identical to before this existed.
function buildPSLG(polylines, weldTolerance, sources) {
  // Vertex welding: a plain spatial-bucket hash keyed by rounded
  // coordinate, matching this kernel's own "simple until proven
  // insufficient" posture (the same call Stage 2 makes about
  // brute-force segment testing being fine until a real scale test says
  // otherwise) — a grid cell wide enough that any two points within
  // weldTolerance of each other land in the SAME or an ADJACENT cell, so
  // only the 3x3 neighborhood around a candidate cell needs checking.
  const cellSize = Math.max(weldTolerance * 4, 1e-6);
  const buckets = new Map(); // "cx,cy" -> [vertexIdx, ...]
  const vertices = [];
  function cellKey(cx, cy) { return `${cx},${cy}`; }
  function weldVertex(pt) {
    const cx = Math.floor(pt[0] / cellSize), cy = Math.floor(pt[1] / cellSize);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const bucket = buckets.get(cellKey(cx + dx, cy + dy));
        if (!bucket) continue;
        for (const idx of bucket) {
          const v = vertices[idx];
          if (Math.hypot(v[0] - pt[0], v[1] - pt[1]) <= weldTolerance) return idx;
        }
      }
    }
    const idx = vertices.length;
    vertices.push([pt[0], pt[1]]);
    const key = cellKey(cx, cy);
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(idx);
    return idx;
  }

  // Raw segments, one per consecutive pair within each input polyline —
  // each carries its own split-parameter list (starts with just {0,1},
  // grows as intersections are found against every OTHER segment).
  const rawSegs = [];
  polylines.forEach((poly, pi) => {
    for (let i = 0; i < poly.length - 1; i++) {
      rawSegs.push({ p0: poly[i], p1: poly[i + 1], splitTs: new Set([0, 1]), src: sources ? sources[pi] : null });
    }
  });

  for (let i = 0; i < rawSegs.length; i++) {
    for (let j = i + 1; j < rawSegs.length; j++) {
      const a = rawSegs[i], b = rawSegs[j];
      const hits = segmentIntersections(a.p0, a.p1, b.p0, b.p1, weldTolerance);
      for (const { t, s } of hits) { a.splitTs.add(t); b.splitTs.add(s); }
    }
  }

  const edgeSet = new Map(); // canonical "min_max" key -> edge index. Dedupes exact-duplicate edges (from coincident input segments).
  const edges = [];
  const edgeSources = sources ? [] : null;
  for (const seg of rawSegs) {
    const ts = [...seg.splitTs].sort((x, y) => x - y);
    for (let k = 0; k < ts.length - 1; k++) {
      const t0 = ts[k], t1 = ts[k + 1];
      if (t1 - t0 < 1e-9) continue; // a split point that landed on another split point — zero-length, drop
      const pA = [seg.p0[0] + (seg.p1[0] - seg.p0[0]) * t0, seg.p0[1] + (seg.p1[1] - seg.p0[1]) * t0];
      const pB = [seg.p0[0] + (seg.p1[0] - seg.p0[0]) * t1, seg.p0[1] + (seg.p1[1] - seg.p0[1]) * t1];
      const vA = weldVertex(pA), vB = weldVertex(pB);
      if (vA === vB) continue; // welded to the same vertex — degenerate, drop
      const key = vA < vB ? `${vA}_${vB}` : `${vB}_${vA}`;
      if (edgeSet.has(key)) {
        // A duplicate edge is dropped, but its SOURCE still counts — this is
        // exactly the "an intersection curve runs along a trim boundary" case.
        if (edgeSources) edgeSources[edgeSet.get(key)].add(seg.src);
        continue;
      }
      edgeSet.set(key, edges.length);
      edges.push([vA, vB]);
      if (edgeSources) edgeSources.push(new Set([seg.src]));
    }
  }
  return { vertices, edges, edgeSources };
}

// ----------------------------------------------------------------
// STAGE 3: prune dangling edges — a real, missing stage in the first
// draft (found by review). An open curve with no closing partner
// leaves a degree-1 vertex; a face walk reaching it would otherwise
// traverse the edge out and immediately back (stitching a zero-width
// spur into a boundary). Iteratively strip every degree-1 vertex (and
// its one edge) until none remain.
// ----------------------------------------------------------------
// Returns the SURVIVING INDICES into the original edge list rather than the
// edges themselves, so a caller carrying anything parallel to that list (per-
// edge source tags) can filter it in lockstep instead of re-deriving it.
function pruneDanglingEdgeIndices(vertexCount, edges) {
  let current = edges.map((_, i) => i);
  for (;;) {
    const degree = new Array(vertexCount).fill(0);
    for (const i of current) { degree[edges[i][0]]++; degree[edges[i][1]]++; }
    const next = current.filter((i) => degree[edges[i][0]] > 1 && degree[edges[i][1]] > 1);
    if (next.length === current.length) return next;
    current = next;
  }
}

// ----------------------------------------------------------------
// STAGE 4: half-edge (DCEL) construction + leftmost-turn face traversal.
// Every undirected PSLG edge becomes two directed half-edges; each
// vertex's own outgoing half-edges are sorted by angle. Walking each
// half-edge's own "next" pointer traces every cycle in the arrangement —
// CCW (positive signed area) cycles are bounded faces, CW (negative
// signed area) cycles occur once per connected component and are handled
// by Stage 5 below, not discarded here.
// ----------------------------------------------------------------
function buildHalfEdgesAndWalkCycles(vertices, edges) {
  // Each half-edge: { origin, target, twin: idx of its own twin }.
  const halfEdges = [];
  const outgoing = Array.from({ length: vertices.length }, () => []); // vertex -> [halfEdge idx]
  for (const [a, b] of edges) {
    const hAB = halfEdges.length; halfEdges.push({ origin: a, target: b, twin: hAB + 1 });
    const hBA = halfEdges.length; halfEdges.push({ origin: b, target: a, twin: hAB });
    outgoing[a].push(hAB);
    outgoing[b].push(hBA);
  }
  // Sort each vertex's own outgoing half-edges by angle, INCREASING
  // (standard math convention — CCW as angle grows).
  for (const v of outgoing) {
    v.sort((h1, h2) => {
      const e1 = halfEdges[h1], e2 = halfEdges[h2];
      const a1 = Math.atan2(vertices[e1.target][1] - vertices[e1.origin][1], vertices[e1.target][0] - vertices[e1.origin][0]);
      const a2 = Math.atan2(vertices[e2.target][1] - vertices[e2.origin][1], vertices[e2.target][0] - vertices[e2.origin][0]);
      return a1 - a2;
    });
  }
  // next(h): arriving at h.target via h, the SAME face's boundary
  // continues via the half-edge at h.target that sits IMMEDIATELY
  // CLOCKWISE (i.e., the entry immediately BEFORE, in the increasing-
  // angle/CCW-sorted list) from twin(h) — the standard "hug the interior,
  // smallest possible turn" DCEL rule. Precomputed once, over the whole
  // structure.
  const next = new Array(halfEdges.length);
  for (let h = 0; h < halfEdges.length; h++) {
    const v = halfEdges[h].target;
    const list = outgoing[v];
    const twinIdx = halfEdges[h].twin;
    const pos = list.indexOf(twinIdx);
    next[h] = list[(pos - 1 + list.length) % list.length];
  }
  // Walk every half-edge exactly once (via next) to enumerate cycles.
  const visited = new Array(halfEdges.length).fill(false);
  const cycles = []; // each: { halfEdgeIdxs: [...], vertexLoop: [...], area }
  for (let h0 = 0; h0 < halfEdges.length; h0++) {
    if (visited[h0]) continue;
    const idxs = [];
    let h = h0;
    do {
      visited[h] = true;
      idxs.push(h);
      h = next[h];
    } while (h !== h0 && !visited[h]);
    const vertexLoop = idxs.map((hi) => vertices[halfEdges[hi].origin]);
    cycles.push({ halfEdgeIdxs: idxs, vertexLoop, area: signedArea2D(vertexLoop) });
  }
  return { halfEdges, cycles };
}

// ----------------------------------------------------------------
// STAGE 5: assign each CW (hole) cycle to its true containing CCW
// (bounded-face) cycle via point-in-polygon containment — the real
// correctness fix that review caught: a CW cycle is NOT "the one
// unbounded face," it occurs once per connected component, and belongs
// to whichever face genuinely contains it (possibly the true unbounded
// outside). A CW cycle contained by no CCW cycle IS the true outside and
// is discarded.
// ----------------------------------------------------------------
function assignHolesToFaces(cycles) {
  const ccw = cycles.filter((c) => c.area > 1e-9);
  const cw = cycles.filter((c) => c.area < -1e-9);
  const faces = ccw.map((c) => ({ outer: c.vertexLoop, holes: [], outerHalfEdges: c.halfEdgeIdxs, holeHalfEdges: [] }));
  for (const hole of cw) {
    const [u, v] = hole.vertexLoop[0];
    let bestFace = null, bestArea = Infinity;
    for (const face of faces) {
      const classification = pointInUVPolygon(face.outer, u, v);
      if (classification !== 'inside') continue;
      const area = Math.abs(signedArea2D(face.outer));
      if (area < bestArea) { bestArea = area; bestFace = face; }
    }
    if (bestFace) { bestFace.holes.push(hole.vertexLoop); bestFace.holeHalfEdges.push(hole.halfEdgeIdxs); }
    // No containing face — this CW cycle is the true unbounded "outside",
    // discarded (not an error; every arrangement has exactly one).
  }
  return faces;
}

// ----------------------------------------------------------------
// CONTIGUITY: two faces are adjacent (for shift-select-union) iff they
// share at least one PSLG edge — i.e. some half-edge's own twin belongs
// to the OTHER face's boundary. Computed directly from the half-edge
// ownership already recorded per face (outer + hole boundaries both
// count — a hole's own boundary is shared with whichever face sits
// inside it, if any).
// ----------------------------------------------------------------
function computeAdjacency(faces, halfEdges) {
  const faceOfHalfEdge = new Map();
  faces.forEach((face, faceIdx) => {
    for (const h of face.outerHalfEdges) faceOfHalfEdge.set(h, faceIdx);
    for (const holeHs of face.holeHalfEdges) for (const h of holeHs) faceOfHalfEdge.set(h, faceIdx);
  });
  const adjacency = faces.map(() => new Set());
  for (const [h, faceIdx] of faceOfHalfEdge) {
    const twinFaceIdx = faceOfHalfEdge.get(halfEdges[h].twin);
    if (twinFaceIdx != null && twinFaceIdx !== faceIdx) { adjacency[faceIdx].add(twinFaceIdx); adjacency[twinFaceIdx].add(faceIdx); }
  }
  return adjacency.map((s) => [...s]);
}

// ----------------------------------------------------------------
// TOP-LEVEL ENTRY POINT
// ----------------------------------------------------------------
// `polylines`: array of arrays of [x,y] pairs — each one an already-
// flattened, NOT-necessarily-closed 2D curve sample chain (the caller's
// own job to flatten real NURBS curves into this shape via
// sampleCurveAdaptive or equivalent — this module is deliberately
// curve-type-agnostic). `opts.weldTolerance` — the "gap tolerance /
// near-miss closure" knob, deliberately left open; defaults to a
// small but non-zero value so ordinary snapped-endpoint T-junctions
// (floating-point-adjacent, not genuinely far apart) always weld.
// `opts.sources` (optional) is a tag per input polyline. Supplying it adds a
// `sources` field to every returned face — per boundary edge, the set of tags
// that produced it, in the same order as that loop's own points. Omitting it
// leaves the result byte-identical to before this option existed.
export function buildPlanarArrangement(polylines, opts = {}) {
  const weldTolerance = opts.weldTolerance ?? 1e-4;
  const { vertices, edges: rawEdges, edgeSources: rawEdgeSources } = buildPSLG(polylines, weldTolerance, opts.sources);
  const keep = pruneDanglingEdgeIndices(vertices.length, rawEdges);
  const edges = keep.map((i) => rawEdges[i]);
  const edgeSources = rawEdgeSources ? keep.map((i) => rawEdgeSources[i]) : null;
  const { halfEdges, cycles } = buildHalfEdgesAndWalkCycles(vertices, edges);
  const faces = assignHolesToFaces(cycles);
  const adjacency = computeAdjacency(faces, halfEdges);
  // Half-edges are emitted two per edge, in edge order, so a half-edge's own
  // undirected edge — and therefore its source set — is its index >> 1.
  const tagsOf = (hs) => hs.map((h) => [...edgeSources[h >> 1]]);
  return {
    vertices, edges,
    faces: faces.map((f) => (edgeSources
      ? { outer: f.outer, holes: f.holes, sources: { outer: tagsOf(f.outerHalfEdges), holes: f.holeHalfEdges.map(tagsOf) } }
      : { outer: f.outer, holes: f.holes })),
    adjacency, // adjacency[i] = array of face indices sharing an edge with face i
    // Internal detail, deliberately exposed (not a private closure) so
    // mergeFaces (below) can operate on the SAME already-built half-edge
    // structure rather than re-deriving it from raw geometry — a plain
    // JS convention this kernel already uses elsewhere for "expose the
    // module-scoped internal directly" (matching __unreasonObjectInfo's
    // own precedent in the app layer, one level down in the kernel here).
    _internal: { halfEdges, faces },
  };
}

// ----------------------------------------------------------------
// MERGE — shift-select gets all of them
// together as one area since they are contiguous... That creates a dup
// of all those border areas, joins them together." Given a set of face
// INDICES (the caller's own job to only ever pass genuinely contiguous
// ones — checked against `adjacency` before calling this, not re-
// validated here), computes their UNION's own boundary: every half-edge
// belonging to ANY included face, MINUS any edge whose two half-edges
// both belong to DIFFERENT included faces (an internal boundary between
// two merged regions — it cancels, exactly like two adjacent trim loops
// dissolving their shared edge). The remaining "external" edges are
// re-walked through the SAME Stage 4/5 machinery (buildHalfEdgesAndWalkCycles
// + assignHolesToFaces) already proven correct above — reused wholesale,
// not reimplemented, so a merge result is held to the identical
// correctness bar (including real hole handling) as a fresh arrangement.
export function mergeFaces(arrangement, faceIndices) {
  const { halfEdges, faces } = arrangement._internal;
  const included = new Set(faceIndices);
  const faceOfHalfEdge = new Map();
  faces.forEach((face, idx) => {
    for (const h of face.outerHalfEdges) faceOfHalfEdge.set(h, idx);
    for (const hs of face.holeHalfEdges) for (const h of hs) faceOfHalfEdge.set(h, idx);
  });
  const halfEdgeByOD = new Map(); // "origin_target" -> half-edge idx, built once
  halfEdges.forEach((he, h) => halfEdgeByOD.set(`${he.origin}_${he.target}`, h));
  const keptEdges = [];
  const seenUndirected = new Set();
  for (const [a, b] of arrangement.edges) {
    const hAB = halfEdgeByOD.get(`${a}_${b}`), hBA = halfEdgeByOD.get(`${b}_${a}`);
    if (hAB == null || hBA == null) continue;
    const faceA = faceOfHalfEdge.get(hAB), faceB = faceOfHalfEdge.get(hBA);
    const aIn = faceA != null && included.has(faceA);
    const bIn = faceB != null && included.has(faceB);
    if (aIn === bIn) continue; // both internal (cancels) or both external (irrelevant) — either way, not part of the merged boundary
    const key = a < b ? `${a}_${b}` : `${b}_${a}`;
    if (seenUndirected.has(key)) continue;
    seenUndirected.add(key);
    keptEdges.push([a, b]);
  }
  const prunedEdges = pruneDanglingEdgeIndices(arrangement.vertices.length, keptEdges).map((i) => keptEdges[i]);
  const { halfEdges: mergedHalfEdges, cycles } = buildHalfEdgesAndWalkCycles(arrangement.vertices, prunedEdges);
  const mergedFaces = assignHolesToFaces(cycles);
  return mergedFaces.map((f) => ({ outer: f.outer, holes: f.holes }));
}
