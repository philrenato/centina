// Trimmed-region tessellation. Keep the UV-rectangle quad grid, classify
// each cell as fully-inside / fully-outside / boundary-crossing, and clip
// only the boundary-crossing cells against the loop polygon — interior cells
// tessellate as plain quads.
//
// Holes are cut by `mergeLoopsKeyhole` (see its own header comment further
// down). Three constraints follow from it: (1) a cell's fast-path
// classification is made against the original loops, not the merged
// polygon's bbox, or a cell sitting entirely inside a hole is
// mis-tessellated (see `tessellateTrimmedSurface`'s own comment); (2) a
// self-touching clip result is triangulated as one unit, never split into
// fragments — `triangulatePolygon2D` normalizes every polygon it
// triangulates to a consistent winding, so a split-off hole fragment (real
// area, not a degenerate leftover) would have its area added back rather than
// staying excluded; (3) a hole small enough to sit entirely inside one grid
// cell, whose own bridge routes through a different cell, needs its own local
// keyhole merge (`emitCellRectMinusHoles`) — the global merged-polygon clip
// cannot represent that case.
//
// The clipping math — Sutherland-Hodgman, in the direction that is
// valid for an arbitrary (non-convex) trim loop: the loop is the
// (possibly non-convex) subject polygon, and the grid cell is the convex
// clip window (a plain axis-aligned rectangle is always convex, regardless
// of the loop's own shape). S-H only requires the clip window to be
// convex, never the subject — clipping the cell by the loop (the tempting
// other direction) would only be valid if the loop itself were convex,
// which a real trim boundary is not guaranteed to be. This also means the
// classification test (does a cell's bbox even overlap the loop's bbox)
// is a pure performance shortcut, not a correctness requirement — running
// the general clip on every cell would still be exactly correct, just
// slower for a small/local trim region.
//
// Normals are analytic, per vertex — surfacePointAndPartials gives the
// true Su/Sv at any (u,v), so every emitted vertex gets its own exact
// surface normal (su × sv) rather than relying on three.js's own
// computeVertexNormals (a mesh-averaged approximation that needs shared
// vertex topology to work correctly at all). This is what lets the output
// be a plain, non-indexed triangle soup — no dedup/sharing logic needed
// for correct shading, and no crack risk from a dedup-tolerance bug.
// Positions themselves still match exactly along a shared cell edge
// between two independently-clipped neighboring cells, since both derive
// their boundary-crossing point from the identical loop segment and the
// identical shared edge line.
//
// Manifoldness — the paragraph above is true but not sufficient, and area
// alone cannot see the gap, because every artifact involved is area-neutral.
// Agreement on the shared crossing bit-for-bit is not enough on its own: a
// cell can invent an extra collinear vertex its neighbor has no reason to
// know about (see removeCollinearSpikes), and a bridge-crossed cell can emit
// one wholly-inside-the-hole face (see dropSpuriousTriangles). Left in, an
// ordinary off-center hole on a 10x10 grid welds to Euler characteristic -4
// (an annulus must give 0), 11 edges shared by three faces, and 13
// T-junctions. Both are cleaned up, and a third pass (repairTJunctions)
// makes seam sharing explicit rather than merely likely.
//
// Known limitation. Sutherland-Hodgman is only valid for a simple subject
// polygon, and a keyholed (bridge-slit) polygon is not simple — so clipping
// the merged loop per cell is outside S-H's own domain of validity, and the
// cleanups above are repairs of its output, not a proof it was well-defined.
// Two consequences remain:
//   (1) `emitCellRectMinusHoles` — the local keyhole path, taken when a
//       hole sits entirely inside one grid cell — is lossy: area error up to
//       ~1.4% of the trimmed region (worst when the hole is near-concentric
//       with its own cell, falling to zero as it moves off-center). Its own
//       console.error fires there. That is the ear-clipper failing on a
//       symmetric keyholed polygon.
//   (2) A concentric hole pair at a coarse grid shows area error of several
//       percent; it falls to zero at 40x40-equivalent resolutions.
// The general fix for both is a per-cell polygon difference (a Weiler-
// Atherton-style boundary walk) that removes the need for a bridge at all.
// It is not attempted here.

import { surfacePointAndPartials } from './surface.mjs';
import { pointInUVPolygon, segmentsIntersect } from './trim.mjs';

function cross2(ax, ay, bx, by) { return ax * by - ay * bx; }

/* Trim holes: the keyhole/bridge merge. Per-cell polygon subtraction
   (cell ∩ outer) − holes is not used: the moment a hole sits entirely
   inside one grid cell (never touching that cell's own boundary), the
   result is an annular region, not a simple polygon at all, which would
   need a constrained triangulation.

   Instead: merge (outer loop + every hole loop) into one simple polygon
   once, up front, via the classic "keyhole" technique — for each hole,
   find a non-crossing straight bridge from a vertex of the
   currently-merged polygon to a vertex of that hole, then splice the
   hole's own boundary into the merged polygon at that point (walking the
   hole once and returning to the exact same bridge point, so the bridge
   edge is traversed twice, in opposite directions — a zero-area slit
   that changes nothing about the true shape, only its representation as
   one continuous boundary). Once this single polygon exists, every
   downstream step in this file (bbox overlap, Sutherland-Hodgman clip,
   ear-clip triangulation, pointInUVPolygon's own even-odd classification)
   runs unchanged — all of them handle an arbitrary simple (possibly
   non-convex) polygon, which is what a keyholed region is.

   Bridge selection: among every (merged-vertex, hole-vertex) pair, pick
   the shortest one whose straight segment crosses neither the current
   merged polygon's own edges nor the hole's own edges nor any as-yet-
   unmerged hole's edges (checked via the strict-inequality
   `segmentsIntersect` trim.mjs's own self-intersection check uses — its
   strict `>0`/`<0` comparisons treat the bridge's own two endpoints, which
   necessarily touch an adjacent edge of the merged polygon and the hole,
   as not a crossing, with no special-case skip needed). A bridge that
   grazes exactly through a third vertex (collinear, not a proper crossing)
   is rejected separately by `bridgeIsValid`'s distance test. */
// Perpendicular distance from point `p` to the (finite) segment a-b —
// used below to reject a bridge candidate that grazes exactly through a
// third vertex (collinear-touch, not a proper crossing — segmentsIntersect's
// own strict inequality comparisons do not catch this, see its header
// comment in trim.mjs). Two candidate bridges from concentric/symmetric
// shapes can otherwise pass exactly through an unrelated vertex.
function distPointToSegment(p, a, b) {
  const abx = b[0] - a[0], aby = b[1] - a[1];
  const lenSq = abx * abx + aby * aby;
  if (lenSq < 1e-18) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  let t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const cx = a[0] + t * abx, cy = a[1] + t * aby;
  return Math.hypot(p[0] - cx, p[1] - cy);
}
// A candidate bridge (p,q) is valid only if it (1) crosses no edge of any
// polygon in `polys` (the strict-inequality segmentsIntersect check,
// naturally excluding the bridge's own two adjacent edges — see the
// header comment above), (2) doesn't graze within `tol` of any other
// vertex of those polygons (a near-collinear touch that segmentsIntersect
// can't see), and (3) travels through the "outer minus holes" solid
// region — checked via its own midpoint against the original loops, not
// the merged polygon (whose own duplicated bridge vertices would make a
// self-consistency check meaningless): inside `outerLoop`, outside every
// loop in `allHoles`. This last check is what catches a candidate that's
// technically non-crossing (e.g. leaving from a reflex vertex of a
// non-convex outer) yet travels outside the trimmed region entirely.
function bridgeIsValid(p, q, polys, outerLoop, allHoles, tol = 1e-9) {
  for (const poly of polys) {
    const n = poly.length;
    for (let i = 0; i < n; i++) {
      const a = poly[i], b = poly[(i + 1) % n];
      if (segmentsIntersect(p, q, a, b)) return false;
      if (!samePt(a, p, tol) && !samePt(a, q, tol) && distPointToSegment(a, p, q) < tol) return false;
    }
  }
  const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
  if (pointInUVPolygon(outerLoop, mx, my) === 'outside') return false;
  for (const h of allHoles) if (pointInUVPolygon(h, mx, my) === 'inside') return false;
  return true;
}
function samePt(a, b, tol) { return Math.hypot(a[0] - b[0], a[1] - b[1]) < tol; }
// Splices `hole` into `poly` at poly[mergeAtI]/hole[holeAtJ]: walks the
// hole once, starting and ending at holeAtJ, then returns to poly[mergeAtI]
// a second time before continuing — the "slit" shape described above.
function spliceHoleAt(poly, mergeAtI, hole, holeAtJ) {
  const m = hole.length;
  const holeSeq = [];
  for (let k = 0; k <= m; k++) holeSeq.push(hole[(holeAtJ + k) % m]);
  const out = poly.slice(0, mergeAtI + 1).concat(holeSeq, [poly[mergeAtI]], poly.slice(mergeAtI + 1));
  return out;
}
// Merges `outer` + every loop in `holes` into one simple polygon. Throws
// a specific error if a hole has no valid non-crossing bridge available
// (never produces a wrong/self-crossing result).
export function mergeLoopsKeyhole(outer, holes) {
  let merged = outer.slice();
  const remaining = holes.slice();
  const allHoles = holes; // fixed for the whole merge — every candidate's midpoint must stay outside all of them, not just the ones left to process
  let guard = holes.length;
  while (remaining.length) {
    if (guard-- < 0) throw new Error('mergeLoopsKeyhole: internal loop guard tripped'); // defensive only, never reachable by construction
    const hole = remaining.shift();
    let best = null, bestDistSq = Infinity;
    for (let i = 0; i < merged.length; i++) {
      for (let j = 0; j < hole.length; j++) {
        const p = merged[i], q = hole[j];
        const dx = p[0] - q[0], dy = p[1] - q[1];
        const distSq = dx * dx + dy * dy;
        if (distSq < 1e-18) continue; // degenerate coincident bridge, skip
        if (distSq >= bestDistSq) continue;
        if (!bridgeIsValid(p, q, [merged, hole, ...remaining], outer, allHoles)) continue;
        best = { i, j }; bestDistSq = distSq;
      }
    }
    if (!best) throw new Error('mergeLoopsKeyhole: no valid non-crossing bridge found for a hole loop — the hole likely overlaps another loop');
    merged = spliceHoleAt(merged, best.i, hole, best.j);
  }
  return merged;
}

// Walks `loop` forward from index a to index b inclusive, wrapping.
function walkLoop(loop, a, b) {
  const n = loop.length;
  const out = [];
  for (let k = 0; ; k++) {
    const idx = (a + k) % n;
    out.push(loop[idx]);
    if (idx === b) break;
    if (k > n) throw new Error('walkLoop: index never reached'); // defensive only
  }
  return out;
}

/**
 * Cut an annulus (one outer loop, one hole) into two simple loops using
 * two disjoint bridges, rather than one keyhole slit.
 *
 * Why this exists alongside mergeLoopsKeyhole: that function is right for
 * tessellation — a zero-area slit leaves the region's own area untouched,
 * and an ear-clipper doesn't care that a bridge corner appears twice. It is
 * wrong for a B-rep face, because a face loop that visits the same corner
 * twice is a repeated vertex, which the welder rejects as degenerate —
 * correctly, since a face boundary traverses each of its own corners once.
 *
 * Two bridges instead of one turn the annulus into two faces sharing
 * both bridge edges. Every loop stays simple, and each bridge edge has
 * exactly two incident faces — which is what the welder needs.
 *
 * The second bridge must be independent of the first: not sharing an
 * endpoint, not crossing it, and not grazing collinearly through either of
 * its endpoints (the same near-collinear case bridgeIsValid's own tolerance
 * check exists for). Returns null when no such pair exists, so a caller can
 * refuse rather than emit a face it cannot weld.
 */
export function splitAnnulusTwoBridges(outer, hole, tol = 1e-9) {
  const candidates = [];
  for (let i = 0; i < outer.length; i++) {
    for (let j = 0; j < hole.length; j++) {
      const p = outer[i], q = hole[j];
      const dx = p[0] - q[0], dy = p[1] - q[1];
      const distSq = dx * dx + dy * dy;
      if (distSq < 1e-18) continue;
      if (!bridgeIsValid(p, q, [outer, hole], outer, [hole], tol)) continue;
      candidates.push({ i, j, distSq, p, q });
    }
  }
  if (candidates.length < 2) return null;
  candidates.sort((a, b) => a.distSq - b.distSq);

  const first = candidates[0];
  let second = null;
  for (const c of candidates) {
    if (c.i === first.i || c.j === first.j) continue; // must not share an endpoint
    if (segmentsIntersect(first.p, first.q, c.p, c.q)) continue;
    if (distPointToSegment(c.p, first.p, first.q) < tol) continue;
    if (distPointToSegment(c.q, first.p, first.q) < tol) continue;
    if (distPointToSegment(first.p, c.p, c.q) < tol) continue;
    if (distPointToSegment(first.q, c.p, c.q) < tol) continue;
    second = c;
    break;
  }
  if (!second) return null;

  return [
    [...walkLoop(outer, first.i, second.i), ...walkLoop(hole, second.j, first.j)],
    [...walkLoop(outer, second.i, first.i), ...walkLoop(hole, first.j, second.j)],
  ];
}

function loopBBox(loop) {
  let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (const [u, v] of loop) {
    if (u < uMin) uMin = u; if (u > uMax) uMax = u;
    if (v < vMin) vMin = v; if (v > vMax) vMax = v;
  }
  return { uMin, uMax, vMin, vMax };
}

// True if `inner`'s bbox sits strictly inside `outer`'s (never touching
// its edges) — used both to detect a hole entirely contained within one
// cell (the local-keyhole case below) and, defensively, to confirm
// that same cell is itself safely inside the true outer loop's own bbox.
function boxFullyInside(inner, outer, margin = 1e-9) {
  return inner.uMin >= outer.uMin + margin && inner.uMax <= outer.uMax - margin &&
         inner.vMin >= outer.vMin + margin && inner.vMax <= outer.vMax - margin;
}

// Sutherland-Hodgman: clip `subject` (array of [u,v], implicitly closed,
// any winding) against the convex rectangle [uMin,uMax]x[vMin,vMax].
// Returns a new polygon (possibly empty) — the exact intersection region,
// regardless of the subject's own convexity.
export function clipPolygonToRect(subject, uMin, uMax, vMin, vMax) {
  const planes = [
    { keep: (p) => p[0] >= uMin - 1e-9, at: (a, b) => [uMin, a[1] + (uMin - a[0]) / (b[0] - a[0]) * (b[1] - a[1])] },
    { keep: (p) => p[0] <= uMax + 1e-9, at: (a, b) => [uMax, a[1] + (uMax - a[0]) / (b[0] - a[0]) * (b[1] - a[1])] },
    { keep: (p) => p[1] >= vMin - 1e-9, at: (a, b) => [a[0] + (vMin - a[1]) / (b[1] - a[1]) * (b[0] - a[0]), vMin] },
    { keep: (p) => p[1] <= vMax + 1e-9, at: (a, b) => [a[0] + (vMax - a[1]) / (b[1] - a[1]) * (b[0] - a[0]), vMax] },
  ];
  let output = subject;
  for (const plane of planes) {
    if (output.length === 0) break;
    const input = output;
    output = [];
    for (let i = 0; i < input.length; i++) {
      const cur = input[i], prev = input[(i - 1 + input.length) % input.length];
      const curIn = plane.keep(cur), prevIn = plane.keep(prev);
      if (curIn) {
        if (!prevIn) output.push(plane.at(prev, cur));
        output.push(cur);
      } else if (prevIn) {
        output.push(plane.at(prev, cur));
      }
    }
  }
  return dedupeConsecutive(output);
}

// A loop vertex sitting exactly on a clip plane produces a well-known
// S-H degeneracy: the computed plane-crossing intersection and the kept
// vertex itself land at (numerically) the same point, emitted back-to-back
// — a zero-length edge that doesn't change the polygon's true shape but
// does confuse the ear-clipping triangulator below (a degenerate "ear"
// with a repeated vertex can silently under/over-count area). Standard
// cleanup: collapse consecutive near-duplicate points (checking the
// wraparound pair too), never changing the polygon's geometry.
function dedupeConsecutive(poly, tol = 1e-9) {
  if (poly.length < 2) return poly;
  const out = [];
  for (const p of poly) {
    const last = out[out.length - 1];
    if (!last || Math.hypot(p[0] - last[0], p[1] - last[1]) > tol) out.push(p);
  }
  while (out.length > 1 && Math.hypot(out[0][0] - out[out.length - 1][0], out[0][1] - out[out.length - 1][1]) <= tol) out.pop();
  return out;
}

// Trim holes — a bridge slit's own two coincident endpoints (poly[i-1] and
// poly[i+1] landing at the same position, with the degenerate "spike"
// vertex poly[i] between them — e.g. a bridge-junction cell where the slit
// enters and immediately leaves again) is a zero-area feature, not a bug,
// but it confuses the ear-clipper the same way a plane-crossing duplicate
// does (see dedupeConsecutive's own comment) — collapse it the same way:
// remove the spike vertex and both its neighbors' shared position collapses
// to one, iterating until stable (never changes the polygon's true
// area/shape, only removes a zero-area degenerate tip).
function removeSpikes(poly, tol = 1e-9) {
  if (poly.length < 3) return poly;
  let out = poly.slice();
  let changed = true;
  let guard = 0;
  while (changed && guard++ < out.length + 4) {
    changed = false;
    for (let i = 0; i < out.length && out.length >= 3; i++) {
      const prev = out[(i - 1 + out.length) % out.length];
      const next = out[(i + 1) % out.length];
      if (Math.hypot(prev[0] - next[0], prev[1] - next[1]) < tol) {
        out.splice(i, 1);
        changed = true;
        break;
      }
    }
  }
  return out;
}

// Manifold repair — the second, longer-range degenerate class
// Sutherland-Hodgman emits, distinct from removeSpikes' own coincident-
// neighbor case above. It is visible in welded mesh topology rather than
// in area: every artifact it produces is area-neutral.
//
// The mechanism: S-H returns one vertex list. When the true clipped region
// is bounded partly by the subject and partly by the clip window, S-H can
// splice in a window corner that does not belong to the region at all,
// reached and left along the same line — e.g. a cell whose hole-side corner
// is spliced back in as ...(16.394,40),(20,40),(10,40)... where (20,40)
// sits inside the hole. That is a zero-width needle: three collinear points
// where the walk reverses at the middle one. Its shoelace contribution is
// exactly zero, so the polygon's area stays correct — but the ear-clipper
// then triangulates it into a face whose edge spans the full cell edge
// (10,40)-(20,40), straight across the (10,40)-(16.394,40) split the
// neighboring cell correctly produced. That is what makes the welded mesh
// non-manifold: not two cells deriving a shared crossing independently and
// disagreeing (they do not — both compute 16.3940000000 bit-identically),
// but one cell inventing an extra collinear vertex the other has no reason
// to know about.
//
// A collinear reversal always traverses a zero-width needle, so removing
// the middle vertex can never change the polygon's true area or shape —
// it only deletes a degenerate tip, exactly like removeSpikes one step
// more general (removeSpikes handles the case where the needle's two ends
// coincide; this handles the case where they merely lie on one line).
function removeCollinearSpikes(poly, tol = 1e-9) {
  if (poly.length < 3) return poly;
  let out = poly.slice();
  let changed = true;
  let guard = 0;
  while (changed && guard++ < out.length + 4) {
    changed = false;
    for (let i = 0; i < out.length && out.length >= 3; i++) {
      const prev = out[(i - 1 + out.length) % out.length];
      const cur = out[i];
      const next = out[(i + 1) % out.length];
      const ax = cur[0] - prev[0], ay = cur[1] - prev[1];
      const bx = next[0] - cur[0], by = next[1] - cur[1];
      const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
      if (la < tol || lb < tol) continue; // dedupeConsecutive's own job, not this one's
      const collinear = Math.abs(ax * by - ay * bx) <= tol * Math.max(la, lb);
      const reverses = (ax * bx + ay * by) < 0;
      if (collinear && reverses) { out.splice(i, 1); changed = true; break; }
    }
  }
  return out;
}

// The three cleanups compose: collapsing a collinear needle can leave its
// two former neighbors coincident (dedupeConsecutive's case), and removing
// a coincident pair can leave a fresh collinear reversal. Running them once
// each, in any fixed order, can leave degeneracies behind (on a 10x10 grid
// with an off-center hole, a single pass leaves a spurious (20,40) that a
// second pass removes). Iterating to a fixed point closes it; the loop is
// bounded because every pass that changes anything strictly shortens the
// polygon.
function cleanupClippedPolygon(poly) {
  let out = poly;
  for (let pass = 0; pass < 8; pass++) {
    const before = out.length;
    out = dedupeConsecutive(removeCollinearSpikes(dedupeConsecutive(removeSpikes(out))));
    if (out.length === before) break;
  }
  return out;
}

function polygonSignedArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [u0, v0] = poly[i], [u1, v1] = poly[(i + 1) % poly.length];
    a += u0 * v1 - u1 * v0;
  }
  return a / 2;
}

// Standard O(n^2) ear-clipping triangulation of a simple (non-self-
// intersecting) 2D polygon — general enough for whatever shape a
// Sutherland-Hodgman clip produces (not necessarily convex), unlike a
// naive fan-from-first-vertex/centroid, which can emit triangles that dip
// outside a non-convex polygon. Returns index triples into `poly`.
export function triangulatePolygon2D(poly) {
  const n = poly.length;
  if (n < 3) return [];
  if (n === 3) return [[0, 1, 2]];
  let idx = poly.map((_, i) => i);
  if (polygonSignedArea(poly) < 0) idx.reverse(); // ear test below assumes CCW
  /* The iteration bound is taken from the original size, once. Computed
     inside the condition from `idx.length` it would shrink as ears are
     clipped, faster than the counter grows: a 64-gon clips ~58 ears and then
     meets a bound of 6*6+16, exits mid-triangulation, and — because the final
     push only fires at exactly three remaining vertices — drops the remaining
     hexagon. Ear clipping a simple n-gon takes n-2 successful clips and, with
     the scan restarting after each, O(n^2) attempts. */
  const maxIter = n * n + 16;
  const pointInTri = (p, a, b, c) => {
    const d1 = cross2(b[0] - a[0], b[1] - a[1], p[0] - a[0], p[1] - a[1]);
    const d2 = cross2(c[0] - b[0], c[1] - b[1], p[0] - b[0], p[1] - b[1]);
    const d3 = cross2(a[0] - c[0], a[1] - c[1], p[0] - c[0], p[1] - c[1]);
    const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
    const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
    return !(hasNeg && hasPos);
  };
  const triangles = [];
  let guard = 0;
  while (idx.length > 3 && guard++ < maxIter) {
    let clipped = false;
    for (let i = 0; i < idx.length; i++) {
      const iPrev = idx[(i - 1 + idx.length) % idx.length];
      const iCur = idx[i];
      const iNext = idx[(i + 1) % idx.length];
      const a = poly[iPrev], b = poly[iCur], c = poly[iNext];
      const cr = cross2(b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]);
      if (cr <= 1e-14) continue; // reflex or degenerate at this vertex — not a valid ear
      // Trim holes — a keyhole bridge's own duplicated vertex (the same
      // [u,v] position appearing at two different array indices — the bridge is
      // walked out and back) must be skipped here by coordinate, not just by
      // index: skipping only iPrev/iCur/iNext leaves the duplicate's other copy
      // in `idx`, and pointInTri is boundary-inclusive (a point sitting exactly
      // on a/b/c's own edge or vertex returns true) — so that duplicate copy
      // would register as "contained" in almost every valid ear near the slit,
      // rejecting them all and leaving an unindicated triangulation gap (see
      // emitClippedCell's own residual-area safety net for the case this still
      // slips past).
      let containsOther = false;
      for (const j of idx) {
        if (j === iPrev || j === iCur || j === iNext) continue;
        const p = poly[j];
        const sameAsA = Math.hypot(p[0] - a[0], p[1] - a[1]) < 1e-9;
        const sameAsB = Math.hypot(p[0] - b[0], p[1] - b[1]) < 1e-9;
        const sameAsC = Math.hypot(p[0] - c[0], p[1] - c[1]) < 1e-9;
        if (sameAsA || sameAsB || sameAsC) continue;
        if (pointInTri(p, a, b, c)) { containsOther = true; break; }
      }
      if (containsOther) continue;
      triangles.push([iPrev, iCur, iNext]);
      idx.splice(i, 1);
      clipped = true;
      break;
    }
    if (!clipped) break; // a degenerate input (e.g. collinear) — stop rather than loop forever
  }
  if (idx.length === 3) triangles.push([idx[0], idx[1], idx[2]]);
  return triangles;
}

/**
 * Triangle shape, not triangle area — a centroid fan for a convex loop.
 *
 * `triangulatePolygon2D` is an ear-clipper, and on a convex loop every ear it
 * can find shares one apex vertex: the result is a fan of n-2 needles rather
 * than a triangulation of any quality. On a 96-gon disc of radius 45, every
 * triangle carries a 1.875-degree angle — exactly half the 3.75 degrees one
 * edge of a 96-gon subtends, which is the arithmetic signature of a
 * single-apex fan. Area is conserved exactly (ratio 0.9993 against the true
 * disc, the whole shortfall being the polygon-vs-circle difference), so this
 * is purely a question of shape.
 *
 * Shape is invisible to a rasterizer and expensive to a path tracer. A needle
 * is a poor BVH leaf — its bounding box is mostly empty — and ray-triangle
 * intersection loses precision as a triangle degenerates, so rays leaving the
 * surface near the shared apex re-hit it. On a flat cap that reads as radial
 * streaking and a dark wedge converging on one point of the rim. The same
 * shared apex also carries every cap triangle's normal-interpolation weight.
 *
 * Adding one interior vertex at the area centroid and fanning from it gives n
 * triangles whose minimum angle is bounded by the loop's own angular sampling
 * (near 88 degrees on that same 96-gon) instead of 1.875.
 *
 * The centroid is a new point on the surface, not a new point in space. It is
 * produced in (u,v) and evaluated through the surface like every boundary
 * vertex, so on a curved patch it lands on the surface — strictly better than
 * the chord the fan's long edges already cut. On a planar patch it is exact.
 *
 * Convexity is tested, never assumed. On a non-convex loop the centroid can
 * fall outside the polygon, or outside the wedge of some edge, and a fan
 * triangle would then cover ground the polygon excludes. Returns `null` for
 * anything that is not convex within tolerance, or degenerate, so the caller
 * keeps the ear-clipper. Callers should still area-check the result: a fan's
 * signed areas sum to the polygon's area whatever its shape, so only the
 * absolute-area comparison the callers make can see a flipped triangle.
 *
 * Returns `{ points, tris }` — `points` is `poly` with the centroid appended as
 * its last entry, and `tris` indexes into `points`, wound CCW to match
 * `triangulatePolygon2D`'s own normalization.
 */
export function triangulateConvexFanFromCentroid(poly, opts = {}) {
  const n = poly.length;
  if (n < 4) return null; // a triangle is already its own best triangulation
  for (const p of poly) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) return null;
  }
  const signed = polygonSignedArea(poly);
  if (!Number.isFinite(signed)) return null;
  // Scale the degeneracy floor to the loop, so a cap in millimeters and one in
  // meters are judged the same way.
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (const [u, v] of poly) {
    if (u < minU) minU = u; if (u > maxU) maxU = u;
    if (v < minV) minV = v; if (v > maxV) maxV = v;
  }
  const span = Math.max(maxU - minU, maxV - minV);
  if (!(span > 0)) return null;
  if (Math.abs(signed) < span * span * 1e-12) return null; // collinear or slit-like — no interior to fan from
  const idx = poly.map((_, i) => i);
  if (signed < 0) idx.reverse(); // work CCW, as the ear-clipper does
  // Convex within tolerance: every turn the same way. The tolerance is relative
  // to the two edge lengths, so it reads as an angle and a densified loop's
  // floating-point wobble at a near-straight vertex does not read as reflex.
  for (let i = 0; i < n; i++) {
    const a = poly[idx[i]], b = poly[idx[(i + 1) % n]], c = poly[idx[(i + 2) % n]];
    const e1u = b[0] - a[0], e1v = b[1] - a[1];
    const e2u = c[0] - b[0], e2v = c[1] - b[1];
    const l1 = Math.hypot(e1u, e1v), l2 = Math.hypot(e2u, e2v);
    if (l1 === 0 || l2 === 0) continue; // a duplicated vertex turns nowhere
    if (cross2(e1u, e1v, e2u, e2v) < -1e-9 * l1 * l2) return null; // reflex — ear-clip it instead
  }
  // The area centroid, not the vertex average: the vertex average is pulled
  // toward whichever arc of the loop happens to be sampled more densely, and a
  // trim loop's sampling is never uniform.
  let cu = 0, cv = 0;
  for (let i = 0; i < n; i++) {
    const [u0, v0] = poly[i], [u1, v1] = poly[(i + 1) % n];
    const w = u0 * v1 - u1 * v0;
    cu += (u0 + u1) * w;
    cv += (v0 + v1) * w;
  }
  cu /= 6 * signed;
  cv /= 6 * signed;
  if (!Number.isFinite(cu) || !Number.isFinite(cv)) return null;
  const points = poly.slice();
  const c = points.push([cu, cv]) - 1;
  const tris = [];

  /* Concentric rings, not one fan. A single fan spans the whole radius in
     one triangle, so a cap sampled at 192 boundary points is 192 needles
     ~30:1, all meeting at one vertex. They tile the plane exactly and every
     normal is identical, so every geometric check passes -- and a path tracer
     still draws dark wedges radiating from the center, because rays graze
     along the sliver edges.
     Rings keep the fan's cost profile without its shape: the boundary is
     scaled toward the centroid in steps chosen so a cell is at most
     `maxCellAspect` times its own width, so the radius is crossed in several
     short triangles instead of one long one. The apex is still a fan, but
     its triangles are a ring-width across instead of a radius long.
     Safe only because the loop is convex, which is established above
     -- scaling a convex loop toward its own interior point cannot leave the
     polygon or cross itself. A reflex loop returned null long before here. */
  const maxAspect = opts.maxCellAspect == null ? 3 : opts.maxCellAspect;
  let perim = 0, radial = 0;
  for (let i = 0; i < n; i++) {
    const a = poly[idx[i]], b = poly[idx[(i + 1) % n]];
    perim += Math.hypot(b[0] - a[0], b[1] - a[1]);
    radial += Math.hypot(a[0] - cu, a[1] - cv);
  }
  const meanEdge = perim / n, meanR = radial / n;
  // A ceiling of 32 keeps a pathologically dense loop from turning one flat
  // face into a five-figure mesh; at that point the cap is not the problem.
  const rings = (maxAspect > 0 && meanEdge > 0 && Number.isFinite(meanR))
    ? Math.max(1, Math.min(32, Math.ceil(meanR / (maxAspect * meanEdge))))
    : 1;

  if (rings <= 1) {
    for (let i = 0; i < n; i++) tris.push([c, idx[i], idx[(i + 1) % n]]);
    return { points, tris };
  }

  /* The rings are decimated as they shrink. Rings of equal point count
     leave the apex as it was -- n triangles still meet at the center, just
     shorter ones -- and the traced picture keeps a small star there. A ring
     at a third of the radius has a third of the circumference and wants a
     third of the points, so each ring is resampled along the boundary and
     the innermost one carries six. The apex becomes six ordinary triangles,
     and this is cheaper than equal rings. */
  const countFor = (r) => Math.max(6, Math.min(n, Math.round(n * r / rings)));
  // A point at normalized position `s` around the boundary, interpolated in
  // index space, then pulled toward the centroid by `t`.
  const at = (s, t) => {
    const x = ((s % 1) + 1) % 1 * n;
    const i0 = Math.floor(x) % n, f = x - Math.floor(x);
    const a = poly[idx[i0]], b = poly[idx[(i0 + 1) % n]];
    const u = a[0] + (b[0] - a[0]) * f, v = a[1] + (b[1] - a[1]) * f;
    return [cu + (u - cu) * t, cv + (v - cv) * t];
  };
  const ringRows = [];
  for (let r = 1; r < rings; r++) {
    const cnt = countFor(r), t = r / rings, row = [];
    for (let i = 0; i < cnt; i++) row.push(points.push(at(i / cnt, t)) - 1);
    ringRows.push(row);
  }
  ringRows.push(idx.slice()); // the boundary is the outermost ring, as given

  // center fan to the innermost ring -- six-ish triangles, not n
  const first = ringRows[0];
  for (let i = 0; i < first.length; i++) tris.push([c, first[i], first[(i + 1) % first.length]]);

  /* Stitch two closed rings of different lengths by advancing whichever side's
     next vertex comes first in normalized position -- the standard merge, and
     the reason no T-junction appears between rings of unequal count. */
  for (let r = 0; r + 1 < ringRows.length; r++) {
    const A = ringRows[r], B = ringRows[r + 1];
    const na = A.length, nb = B.length;
    let i = 0, j = 0;
    while (i < na || j < nb) {
      const sa = (i + 1) / na, sb = (j + 1) / nb;
      if (j >= nb || (i < na && sa <= sb)) {
        tris.push([A[i % na], B[j % nb], A[(i + 1) % na]]);
        i++;
      } else {
        tris.push([A[i % na], B[j % nb], B[(j + 1) % nb]]);
        j++;
      }
    }
  }
  return { points, tris };
}

function surfaceNormalAt(srf, u, v) {
  const { su, sv } = surfacePointAndPartials(srf, u, v);
  const nx = su[1] * sv[2] - su[2] * sv[1];
  const ny = su[2] * sv[0] - su[0] * sv[2];
  const nz = su[0] * sv[1] - su[1] * sv[0];
  const len = Math.hypot(nx, ny, nz);
  if (len < 1e-12) return [0, 0, 1]; // a pole/degenerate partial — an arbitrary but finite normal, never NaN
  return [nx / len, ny / len, nz / len];
}

function makeVertex(srf, u, v) {
  const { point } = surfacePointAndPartials(srf, u, v);
  return { position: point, normal: surfaceNormalAt(srf, u, v), uv: [u, v] };
}

// Manifold repair — a triangle whose three vertices all sit strictly
// inside a hole (or all strictly outside the outer loop) cannot be
// legitimate. Every vertex of a correct trimmed tessellation lies in the
// closure of the trimmed region, so the worst a real vertex can read is
// 'boundary'; 'inside a hole' is unreachable for one. Requiring all three
// is conservative — a sliver straddling the boundary keeps at least one
// vertex on it, so this can only reject geometry that is already provably
// spurious, never a thin-but-real triangle. (This is why it is a vertex
// test and not a looser centroid test: a legitimate triangle can have a
// centroid read a hair on the wrong side of a curve.)
//
// What it catches: a bridge-crossed cell whose tangled clip result
// triangulates into one extra half-cell face lying wholly inside the hole —
// worth up to half a grid cell of spurious area, and the last obstacle
// between an ordinary off-center hole and an exactly-correct Euler
// characteristic.
function dropSpuriousTriangles(triangles, outerLoop, holes) {
  if (!outerLoop) return triangles;
  const out = [];
  for (const tri of triangles) {
    let spurious = false;
    for (const h of holes) {
      if (tri.every((v) => pointInUVPolygon(h, v.uv[0], v.uv[1]) === 'inside')) { spurious = true; break; }
    }
    if (!spurious && tri.every((v) => pointInUVPolygon(outerLoop, v.uv[0], v.uv[1]) === 'outside')) spurious = true;
    if (!spurious) out.push(tri);
  }
  return out;
}

// Manifold repair — T-junction repair, applied after triangulation, and
// that ordering is load-bearing rather than incidental.
//
// The obvious fix for a T-junction is to insert the missing crossing into
// the neighboring cell's polygon before triangulating it, so both cells
// agree on the seam. That is wrong: feeding extra collinear vertices to
// the ear-clipper corrupts its output (area error up to 280 units on a
// 10000-unit patch, and non-manifold edges return), because a
// straight-angle vertex is never a valid ear and can stall the clipper.
//
// Splitting an already-emitted triangle at a point lying exactly on one of
// its own edges is area-preserving by construction, cannot stall anything,
// and reaches the same result. Candidates are the tessellation's own
// emitted vertices, so the repair is self-consistent by definition: after
// it runs, no emitted vertex lies in the interior of an axis-aligned edge.
//
// Known limitation: only edges that are axis-aligned in UV are repaired.
// That covers every T-junction this construction can produce (they all
// live on cell boundaries, which are by definition constant-u or
// constant-v lines), but a construction that emitted a slanted shared edge
// would need this generalized.
function repairTJunctions(triangles, srf, tol = 1e-9) {
  if (!triangles.length) return triangles;
  const q = 1e-9;
  const byU = new Map(), byV = new Map();
  const add = (m, k, val) => { const a = m.get(k); if (a) a.push(val); else m.set(k, [val]); };
  for (const tri of triangles) for (const v of tri) {
    add(byU, Math.round(v.uv[0] / q), v.uv[1]);
    add(byV, Math.round(v.uv[1] / q), v.uv[0]);
  }
  for (const m of [byU, byV]) {
    for (const arr of m.values()) {
      arr.sort((x, y) => x - y);
      let w = 0;
      for (let i = 0; i < arr.length; i++) if (w === 0 || Math.abs(arr[i] - arr[w - 1]) > tol) arr[w++] = arr[i];
      arr.length = w;
    }
  }
  const out = [];
  const work = triangles.slice();
  let guard = triangles.length * 64 + 1024;
  while (work.length) {
    if (guard-- < 0) { out.push(...work); break; } // defensive only, never reachable by construction
    const tri = work.pop();
    let split = null;
    for (let e = 0; e < 3 && !split; e++) {
      const a = tri[e], b = tri[(e + 1) % 3], c = tri[(e + 2) % 3];
      let arr = null, axis = -1;
      if (Math.abs(a.uv[0] - b.uv[0]) <= tol) { arr = byU.get(Math.round(a.uv[0] / q)); axis = 1; }
      if (!arr && Math.abs(a.uv[1] - b.uv[1]) <= tol) { arr = byV.get(Math.round(a.uv[1] / q)); axis = 0; }
      if (!arr) continue;
      const lo = Math.min(a.uv[axis], b.uv[axis]), hi = Math.max(a.uv[axis], b.uv[axis]);
      // Split at whichever interior candidate is nearest the edge's own
      // midpoint, so a long edge carrying several crossings bisects rather
      // than shaving off one sliver at a time.
      const mid = (lo + hi) / 2;
      let best = null;
      for (const x of arr) if (x > lo + tol && x < hi - tol && (best === null || Math.abs(x - mid) < Math.abs(best - mid))) best = x;
      if (best === null) continue;
      const su = axis === 0 ? best : a.uv[0];
      const sv = axis === 0 ? a.uv[1] : best;
      split = [[a, makeVertex(srf, su, sv), c], [makeVertex(srf, su, sv), b, c]];
    }
    if (split) work.push(split[0], split[1]);
    else out.push(tri);
  }
  return out;
}

// The entry point. `loop` is a single outer UV trim loop (plain array
// of [u,v] pairs, implicitly closed, any winding) — pass `null`/`undefined`
// for a plain untrimmed rectangle (every cell takes the fast fully-inside
// path). uRes/vRes are segment counts, matching every other tessellation
// function in this kernel/app. `holes` (optional, default
// none) is an array of additional UV loops to cut out of `loop` — merged
// via `mergeLoopsKeyhole` into a single effective boundary once, up front
// (never per-cell), so every loop-shaped call below (bbox,
// Sutherland-Hodgman, ear-clip, pointInUVPolygon) runs unmodified against
// the merged polygon. With no `holes` there is no merge
// (`effectiveLoop === loop`). Returns a flat, non-indexed triangle list:
// [[vA, vB, vC], ...], each v = {position:[x,y,z], normal:[nx,ny,nz],
// uv:[u,v]} — not shared-vertex-indexed (see the header comment for why).
// A trim loop that strays outside the surface's own domain must not take the
// planar shortcut: the grid path clips against the domain as a side effect of
// walking it, and triangulating such a loop directly would extrapolate the
// surface far past where it exists (238x a patch's whole area in one case).
export function loopWithinDomain(loop, uMin, uMax, vMin, vMax) {
  const du = (uMax - uMin) * 1e-9, dv = (vMax - vMin) * 1e-9;
  for (const p of loop) {
    if (p[0] < uMin - du || p[0] > uMax + du) return false;
    if (p[1] < vMin - dv || p[1] > vMax + dv) return false;
  }
  return true;
}
/* Is this surface one flat piece, and if so is its parameterization affine?
   Both questions are asked of the control net, which for a NURBS surface is
   sufficient: a surface lies in the plane of its control points when they are
   coplanar, and it is affine in (u,v) when it is degree 1 both ways, unweighted,
   and its four corners form a parallelogram (the bilinear cross term vanishes).
   Conservative — a false no costs the grid path, a false yes would
   flatten a curved face, so every test below fails closed. */
const PLANAR_PATCH_TOL = 1e-7;   // relative to the patch's own extent
export function surfaceIsPlanarPatch(srf) {
  const net = srf && srf.ctrlNet;
  if (!net || !net.length || !net[0].length) return false;
  const pts = [];
  for (const row of net) for (const p of row) pts.push(p);
  if (pts.length < 3) return false;
  const o = pts[0];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  let e1 = null, n = null, scale = 0;
  for (const p of pts) {
    const d = sub(p, o); const len = Math.hypot(d[0], d[1], d[2]);
    if (len > scale) scale = len;
    if (!e1) { if (len > 1e-12) e1 = [d[0] / len, d[1] / len, d[2] / len]; continue; }
    if (n) continue;
    const c = [e1[1] * d[2] - e1[2] * d[1], e1[2] * d[0] - e1[0] * d[2], e1[0] * d[1] - e1[1] * d[0]];
    const cl = Math.hypot(c[0], c[1], c[2]);
    if (cl > 1e-12) n = [c[0] / cl, c[1] / cl, c[2] / cl];
  }
  if (!n) return true;               // degenerate or collinear net — one plane trivially
  const tol = Math.max(PLANAR_PATCH_TOL, scale * PLANAR_PATCH_TOL);
  for (const p of pts) {
    const d = sub(p, o);
    if (Math.abs(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) > tol) return false;
  }
  return true;
}
export function surfaceIsAffinePatch(srf) {
  const net = srf && srf.ctrlNet;
  if (!net || srf.degU !== 1 || srf.degV !== 1) return false;
  if (net.length !== 2 || net[0].length !== 2 || net[1].length !== 2) return false;
  const [a, b] = [net[0][0], net[0][1]], [c, d] = [net[1][0], net[1][1]];
  for (const p of [a, b, c, d]) if (Math.abs((p[3] == null ? 1 : p[3]) - 1) > 1e-12) return false;
  let m = 0;
  for (let i = 0; i < 3; i += 1) m = Math.max(m, Math.abs(d[i] - c[i] - b[i] + a[i]));
  const scale = Math.max(1e-9, Math.abs(b[0] - a[0]) + Math.abs(b[1] - a[1]) + Math.abs(b[2] - a[2])
    + Math.abs(c[0] - a[0]) + Math.abs(c[1] - a[1]) + Math.abs(c[2] - a[2]));
  return m <= scale * 1e-9;
}
/* "Near the boundary" is a question about the loop's edges, not its
   bounding box. A trimmed face whose loop spans most of its own domain —
   which is what a fillet leaves behind, having shaved only a strip — reads
   as "every cell is near the boundary" if the bbox is asked, so the interior
   fast path below is unreachable and every cell pays a full polygon clip.
   This marks the cells any loop segment passes through, conservatively: a
   segment marks its own cell-index bounding box. Over-marking costs a clip
   on a few cells; under-marking would drop real boundary detail. */
export function markBoundaryCells(loop, uMin, uMax, vMin, vMax, uRes, vRes, mark) {
  if (!loop || loop.length < 2) return;
  const du = (uMax - uMin) / uRes, dv = (vMax - vMin) / vRes;
  if (!(du > 0) || !(dv > 0)) return;
  const cu = (u) => Math.max(0, Math.min(uRes - 1, Math.floor((u - uMin) / du)));
  const cv = (v) => Math.max(0, Math.min(vRes - 1, Math.floor((v - vMin) / dv)));
  for (let i = 0; i < loop.length; i += 1) {
    const a = loop[i], b = loop[(i + 1) % loop.length];
    const i0 = Math.min(cu(a[0]), cu(b[0])), i1 = Math.max(cu(a[0]), cu(b[0]));
    const j0 = Math.min(cv(a[1]), cv(b[1])), j1 = Math.max(cv(a[1]), cv(b[1]));
    for (let ii = i0; ii <= i1; ii += 1) for (let jj = j0; jj <= j1; jj += 1) mark[ii * vRes + jj] = 1;
  }
}
// Splits any loop edge longer than one cell in u or v, so a boundary sampled
// against the grid stays sampled that finely once the grid is gone.
export function densifyUVLoop(loop, du, dv) {
  const out = [];
  for (let i = 0; i < loop.length; i += 1) {
    const a = loop[i], b = loop[(i + 1) % loop.length];
    out.push(a);
    const steps = Math.min(64, Math.max(1,
      Math.ceil(Math.max(Math.abs(b[0] - a[0]) / (du || Infinity), Math.abs(b[1] - a[1]) / (dv || Infinity)))));
    for (let s = 1; s < steps; s += 1) {
      const t = s / steps;
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  return out;
}
/* How much of a polygon a triangulation covers, against how much the
 * polygon encloses, as a signed difference in both directions. This is the
 * invariant the ear-clipper owes; a triangle count is not.
 *
 * A simple polygon of n corners triangulates into n - 2 pieces only while
 * every corner is a real one. A clip against a cell routinely leaves corners
 * that are not — collinear points along one cell edge, a coincident pair, a
 * zero-area spur where a keyhole slit was cut away — and a correct
 * triangulation consumes those without a triangle each, so a count short of
 * n - 2 is as often a well-formed answer as a hole.
 *
 * Both signs matter and for different reasons. Covering less than the polygon
 * encloses is a hole in the face. Covering more is a triangle laid over ground
 * the polygon excludes, which is what a hole's own keyhole bridge looks like
 * when it is filled in by mistake — the same failure `emitCellRectMinusHoles`
 * avoids by never splitting its merged polygon. So the caller is given the
 * two areas and compares them, which is the test the planar fast path makes
 * its own decision on.
 */
export function triangulationAreaShortfall(poly, tris) {
  const want = Math.abs(polygonSignedArea(poly));
  let got = 0;
  for (const [i, j, k] of tris) {
    const p = poly[i], q = poly[j], r = poly[k];
    got += Math.abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])) / 2;
  }
  return { want, got, err: Math.abs(got - want) };
}

export function tessellateTrimmedSurface(srf, loop, uRes, vRes, holes = []) {
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  // Trim holes — a cell's "should it be clipped at all" fast path is
  // classified against the original, unmerged loops, not the merged
  // (keyholed) polygon's overall bbox. A cell sitting entirely inside a hole
  // (far from every loop boundary) can still fall inside the merged polygon's
  // bbox (which spans from the hole all the way out to the outer boundary) —
  // routing it to the Sutherland-Hodgman clip against a subject whose own
  // concavity (the hole) surrounds the clip window with no edge crossing it
  // produces a degenerate, self-overlapping clip result, not the correct
  // empty polygon. So a cell needs the clip only if its own bbox overlaps the
  // outer loop's bbox or any hole's own bbox (i.e., a piece of some loop
  // boundary could be inside it); otherwise it's uniformly classified by one
  // center-point test against every original loop directly (inside the outer
  // and outside every hole) — a cell far from every boundary can't straddle
  // one. `holes=[]` (the default) reduces every check below to the
  // single-loop case.
  const outerLoop = loop;
  const outerBox = outerLoop ? loopBBox(outerLoop) : null;
  const holeBoxes = (holes || []).map(loopBBox);
  const mergedLoop = (outerLoop && holes && holes.length) ? mergeLoopsKeyhole(outerLoop, holes) : outerLoop;
  const triangles = [];

  const emitQuad = (u0, u1, v0, v1) => {
    const a = makeVertex(srf, u0, v0), b = makeVertex(srf, u0, v1);
    const c = makeVertex(srf, u1, v0), d = makeVertex(srf, u1, v1);
    triangles.push([a, c, b], [b, c, d]);
  };

  const emitClippedCell = (u0, u1, v0, v1) => {
    // removeSpikes right after the clip, then dedupeConsecutive's own
    // logic once more (a spike's removal can leave its two former
    // neighbors newly adjacent and near-duplicate) — see removeSpikes'
    // own header comment for why a bridge-junction cell needs this.
    const clipped = cleanupClippedPolygon(clipPolygonToRect(mergedLoop, u0, u1, v0, v1));
    if (clipped.length < 3) return; // cell is entirely outside the effective (outer-minus-holes) region
    if (Math.abs(polygonSignedArea(clipped)) < 1e-15) return; // a zero-area sliver (e.g. a lone bridge slit clipped on its own) contributes nothing
    // Trim holes — a self-touching clip result (a bridge slit crossing
    // this cell's own boundary and re-entering) is triangulated as one
    // unit directly, not split into independent fragments first — see
    // emitCellRectMinusHoles' own header comment for why splitting is wrong
    // here: `triangulatePolygon2D` normalizes every polygon it's handed to a
    // consistent winding, so a split-off fragment representing excluded
    // (hole-side) area would get its area added back rather than staying
    // excluded. The ear-clipper handles the self-touch correctly on its
    // own when given the whole clipped result in one call.
    const tris = triangulatePolygon2D(clipped);
    // Safety net — surface any residual ear-clip gap rather than silently
    // dropping area. It speaks for area lost, never for a triangle count: see
    // triangulationAreaShortfall.
    const cover = triangulationAreaShortfall(clipped, tris);
    if (cover.err > cover.want * 1e-9 + 1e-15) {
      console.error(`tessellateTrimmedSurface: ear-clip left a residual — it covered ${cover.got.toExponential(4)} of the ${cover.want.toExponential(4)} square parameter units this ${clipped.length}-vertex cell encloses, a real trim-hole edge case slipped through`);
    }
    for (const [i, j, k] of tris) {
      const [pu, pv] = clipped[i], [qu, qv] = clipped[j], [ru, rv] = clipped[k];
      triangles.push([makeVertex(srf, pu, pv), makeVertex(srf, qu, qv), makeVertex(srf, ru, rv)]);
    }
  };

  // Trim holes — the local keyhole merge for a hole entirely
  // contained within one cell (see this function's own call-site comment
  // for why the global-merge clip can't represent this case). The cell's
  // own rectangle is the relevant "outer" boundary here — no clip is needed
  // against anything, since the rectangle isn't being intersected with a
  // bigger shape, it's simply having its own hole(s) cut out directly.
  //
  // The merged polygon is not split (unlike a per-fragment approach):
  // splitting a keyhole polygon into independent fragments and triangulating
  // each one separately is only safe when every fragment is either the true
  // region or a degenerate (near-zero-area) leftover, which is what
  // emitClippedCell's own per-cell clip residuals always are. Here, one
  // "fragment" of a rect+hole merge is the hole itself, with non-trivial
  // area — `triangulatePolygon2D` normalizes every polygon it's given to a
  // consistent (CCW-positive) winding internally, so triangulating that hole
  // fragment on its own would add its area back in as if it were surface,
  // rather than leaving it excluded. The ear-clipper, handed the whole
  // (unsplit) keyholed polygon in one call, produces triangles covering only
  // the annular "rect minus hole" region.
  const emitCellRectMinusHoles = (u0, u1, v0, v1, containedHoles) => {
    const cellLoop = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]; // CCW
    const localMerged = containedHoles.length ? mergeLoopsKeyhole(cellLoop, containedHoles) : cellLoop;
    const tris = triangulatePolygon2D(localMerged);
    const localCover = triangulationAreaShortfall(localMerged, tris);
    if (localCover.err > localCover.want * 1e-9 + 1e-15) {
      console.error(`tessellateTrimmedSurface: local cell-hole ear-clip left a residual — it covered ${localCover.got.toExponential(4)} of the ${localCover.want.toExponential(4)} square parameter units this ${localMerged.length}-vertex cell encloses`);
    }
    for (const [i, j, k] of tris) {
      const [pu, pv] = localMerged[i], [qu, qv] = localMerged[j], [ru, rv] = localMerged[k];
      triangles.push([makeVertex(srf, pu, pv), makeVertex(srf, qu, qv), makeVertex(srf, ru, rv)]);
    }
  };

  /* A planar face needs no grid. The grid exists to sample curvature, and a
     flat face has none — everything between the trim boundary and the face
     interior is one plane, so cutting it into cells and clipping the trim
     polygon against every one of them buys nothing. It is the dominant cost
     of a filleted solid: once a fillet shaves a strip off a box's side, that
     side goes from one cell to thousands. Triangulating the trim loop itself
     gives the same surface, because any triangulation of a planar region is
     exact.
     Exact only if the parameterization is affine, which is why the test is
     not merely "is it planar": a bilinear patch through four coplanar but
     non-parallelogram corners maps a straight UV segment to an in-plane
     curve, and the grid's cell crossings were sampling that curve. Where that
     term is present the loop is densified to the cell size the grid would
     have used, so the boundary is sampled as finely as the grid path samples
     it and only the interior grid is dropped. */
  if (outerLoop && !(holes && holes.length)
      && loopWithinDomain(outerLoop, uMin, uMax, vMin, vMax)
      && surfaceIsPlanarPatch(srf)) {
    const poly = surfaceIsAffinePatch(srf) ? mergedLoop
      : densifyUVLoop(mergedLoop, (uMax - uMin) / uRes, (vMax - vMin) / vRes);
    /* A convex face fans from its centroid, not from a boundary vertex. This
       is the path a flat cap takes — a circle's cap is one convex loop of a few
       hundred points with no grid at all — and the ear-clipper's answer for it
       is n-2 needles sharing one rim vertex. See
       `triangulateConvexFanFromCentroid` for why a tracer cares where a
       rasterizer does not. `null` means "not convex", and the ear-clipper is
       still the general answer. */
    const fan = triangulateConvexFanFromCentroid(poly);
    /* The fast path checks itself and falls back, rather than warning. The
       ear-clipper can leave a residual on a polygon the grid path would have
       cut into easy pieces first, and a residual here is lost area on a
       visible face. Comparing the triangulated area against the polygon's own
       shoelace area is exact and costs one pass, so the choice is "provably
       the same surface" or "take the grid". It is also the centroid fan's own
       safety net: a fan's signed areas sum to the polygon's area whatever its
       shape, so a triangle that flipped because the centroid was not visible
       from some edge shows up here, in the absolute-area sum, and nowhere
       else. */
    const want = Math.abs(polygonSignedArea(poly));
    const covers = (pts, tris) => {
      let got = 0;
      for (const [ti, tj, tk] of tris) {
        const p = pts[ti], q = pts[tj], r = pts[tk];
        got += Math.abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])) / 2;
      }
      return Math.abs(got - want) <= want * 1e-9 + 1e-15;
    };
    // The fan first when it applies; the ear-clip is tried too rather than
    // conceding the whole fast path, since a fan that fails its area check says
    // nothing about whether the ear-clip would have covered the polygon.
    let pts = null, tris = null;
    if (fan && covers(fan.points, fan.tris)) { pts = fan.points; tris = fan.tris; }
    else {
      const ear = triangulatePolygon2D(poly);
      if (covers(poly, ear)) { pts = poly; tris = ear; }
    }
    if (tris) {
      for (const [ti, tj, tk] of tris) {
        const [pu, pv] = pts[ti], [qu, qv] = pts[tj], [ru, rv] = pts[tk];
        triangles.push([makeVertex(srf, pu, pv), makeVertex(srf, qu, qv), makeVertex(srf, ru, rv)]);
      }
      return repairTJunctions(dropSpuriousTriangles(triangles, outerLoop, holes || []), srf);
    }
    triangles.length = 0;
  }
  const nearEdge = outerLoop ? new Uint8Array(uRes * vRes) : null;
  if (outerLoop) {
    markBoundaryCells(outerLoop, uMin, uMax, vMin, vMax, uRes, vRes, nearEdge);
    for (const h of (holes || [])) markBoundaryCells(h, uMin, uMax, vMin, vMax, uRes, vRes, nearEdge);
  }
  for (let i = 0; i < uRes; i++) {
    const u0 = uMin + (uMax - uMin) * (i / uRes);
    const u1 = uMin + (uMax - uMin) * ((i + 1) / uRes);
    for (let j = 0; j < vRes; j++) {
      const v0 = vMin + (vMax - vMin) * (j / vRes);
      const v1 = vMin + (vMax - vMin) * ((j + 1) / vRes);
      if (!outerLoop) { emitQuad(u0, u1, v0, v1); continue; }
      const cellBox = { uMin: u0, uMax: u1, vMin: v0, vMax: v1 };
      // Trim holes — a hole small enough to sit entirely inside one cell, far
      // enough from its own keyhole bridge that the bridge gets clipped away by a
      // different cell before reaching this one, leaves this cell's own true
      // region non-simple ("rect minus hole") — clipping the global merged
      // polygon with Sutherland-Hodgman cannot represent that, since S-H (like
      // the ear-clipper it feeds) only produces a simple output from a simple
      // input, and the ear-clip safety net does not fire for it. So a second,
      // local keyhole merge is scoped to this one cell — the cell's own
      // rectangle stands in for "outer" (it is the whole relevant boundary
      // here), bridged directly to any hole entirely contained within it, then
      // triangulated with no clip at all. Guarded by a bbox-containment check
      // (the cell must itself sit inside the true outer loop's own bbox) so this
      // path is never taken for the compound case of a hole sitting in the same
      // cell the outer boundary also crosses — that case falls through to the
      // global-merge clip below.
      const containedHoles = holes.filter((h, hi) => boxFullyInside(holeBoxes[hi], cellBox));
      if (containedHoles.length && boxFullyInside(cellBox, outerBox)) {
        emitCellRectMinusHoles(u0, u1, v0, v1, containedHoles);
        continue;
      }
      if (!nearEdge[i * vRes + j]) {
        // No part of any loop's own boundary can reach this cell, so
        // it is uniformly classified against the original loops directly
        // — a single center-point test is conclusive (a cell that isn't
        // near any boundary can't straddle one).
        const cu = (u0 + u1) / 2, cv = (v0 + v1) / 2;
        const insideOuter = pointInUVPolygon(outerLoop, cu, cv) !== 'outside';
        const insideAnyHole = holes.some((h) => pointInUVPolygon(h, cu, cv) !== 'outside');
        if (insideOuter && !insideAnyHole) emitQuad(u0, u1, v0, v1);
        continue;
      }
      emitClippedCell(u0, u1, v0, v1);
    }
  }
  // Manifold repair — two whole-tessellation passes, in this
  // order: drop provably-spurious faces first (so the T-junction pass never
  // propagates a seam vertex that only a bogus face contributed), then
  // repair T-junctions. Both are no-ops for an untrimmed or hole-free
  // surface.
  return repairTJunctions(dropSpuriousTriangles(triangles, outerLoop, holes || []), srf);
}

// Total (3D) surface area of a tessellation — the standard sum of
// cross-product-derived triangle areas, for checking convergence toward a
// known analytic trim-region area.
export function tessellationArea(triangles) {
  let area = 0;
  for (const [a, b, c] of triangles) {
    const ux = b.position[0] - a.position[0], uy = b.position[1] - a.position[1], uz = b.position[2] - a.position[2];
    const vx = c.position[0] - a.position[0], vy = c.position[1] - a.position[1], vz = c.position[2] - a.position[2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    area += Math.hypot(cx, cy, cz) / 2;
  }
  return area;
}
