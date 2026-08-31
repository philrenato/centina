// FACE SPLITTING AND LOOP ASSEMBLY — Phase 6 of the boolean
// pipeline. Given ONE trimmed face (an outer trim loop plus any holes, all in
// that surface's own (u,v) domain) and the intersection PCURVES crossing it,
// produce the FRAGMENTS the face splits into, each a complete trim-loop set
// of its own, with every boundary edge tagged by where it came from.
//
// WHY THE ARRANGEMENT AND NOT A SECOND PAVING ENGINE. OCCT's vocabulary for
// this is paves and pave blocks: a PAVE is a vertex's position on a curve by
// parameter, a PAVE BLOCK is the piece of an edge between adjacent paves, and
// splitting a face is "pave every boundary and every intersection curve, then
// reassemble the blocks into loops." kernel/arrangement.mjs already does
// exactly that — its PSLG stage splits every input segment at every crossing
// (paving), its half-edges ARE the blocks between adjacent paves, and its
// leftmost-turn walk is the reassembly, with hole-to-face containment already
// handled and already tested. This module is therefore the trim-specific
// layer ON TOP of it: which fragments survive, and what each boundary edge
// means downstream. Building a parallel paving engine would be re-deriving a
// proven one for no gain.
//
// THIS MODULE IS 2D, AND THAT IS THE REPRESENTATION, NOT AN APPROXIMATION.
// A trim loop in this kernel is a (u,v) POLYLINE — kernel/trim.mjs's own T2
// decision, made deliberately — so splitting one is genuinely a planar
// polyline problem. Nothing here evaluates a surface or touches 3D.
//
// SCOPE, STATED RATHER THAN DISCOVERED: transversal splits only. A pcurve
// lying exactly ALONG a trim boundary is detected and reported (both tags
// land on the same edge) but is not a split; that is the coincident-face
// case 81 already refuses. Fragments come back in the same polyline form
// they went in, so this composes directly with makeTrimmedSurface.

import { signedArea2D, pointInUVPolygon, polylineSelfIntersects } from './trim.mjs';
import { buildPlanarArrangement } from './arrangement.mjs';

export const SRC_TRIM = 'trim';
export const SRC_INTERSECTION = 'intersection';

function closeLoop(loop) {
  if (loop.length < 2) return loop.slice();
  const a = loop[0], b = loop[loop.length - 1];
  if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-12) return loop.slice();
  return [...loop, [a[0], a[1]]];
}

function loopArea(loop) { return Math.abs(signedArea2D(loop)); }

/**
 * A point provably in a polygon's own interior, for a polygon that may be
 * non-convex and may have holes — so a centroid will not do (a centroid can
 * sit outside a non-convex polygon entirely, and inside a hole).
 *
 * DEEPEST, not merely inside. The obvious construction — step a hair inward
 * from an edge midpoint — is correct as a containment answer and wrong as a
 * PROBE: a boolean classifies this point against the other solid's own
 * TESSELLATION, and a fragment's boundary IS the intersection curve, so a
 * point a hair inside the boundary sits within the tessellation's own
 * sagitta of the other solid's surface and can classify to the wrong side.
 * That is not a tolerance to widen — a coarser mesh simply moves the wall.
 * So this searches for the point FARTHEST from every boundary edge (outer
 * and holes alike) and only falls back to the edge-offset construction for a
 * sliver too thin for the search to land in. Returns null when no candidate
 * survives, which the caller must treat as a refusal — never as "outside".
 */
export function representativeInteriorPoint(outer, holes = [], opts = {}) {
  const ring = closeLoop(outer);
  if (ring.length < 4) return null;
  const area = loopArea(ring);
  if (!(area > 0)) return null;
  const holeRings = holes.map(closeLoop);
  const deep = deepestInteriorPoint(ring, holeRings);
  if (deep) return deep;
  // Scaled to the region's own size so this behaves identically on a face
  // measured in millimeters and one measured in a unit parameter domain.
  const baseStep = opts.step ?? Math.sqrt(area) * 1e-3;
  const ccw = signedArea2D(ring) > 0;
  for (let attempt = 0; attempt < 4; attempt++) {
    const step = baseStep / Math.pow(4, attempt);
    for (let i = 0; i < ring.length - 1; i++) {
      const p = ring[i], q = ring[i + 1];
      const dx = q[0] - p[0], dy = q[1] - p[1];
      const len = Math.hypot(dx, dy);
      if (!(len > 0)) continue;
      // Inward normal: for a CCW ring the interior is to the LEFT of travel.
      const nx = ccw ? -dy / len : dy / len;
      const ny = ccw ? dx / len : -dx / len;
      const cand = [(p[0] + q[0]) / 2 + nx * step, (p[1] + q[1]) / 2 + ny * step];
      if (pointInUVPolygon(ring, cand[0], cand[1]) !== 'inside') continue;
      let inHole = false;
      for (const h of holes) {
        if (pointInUVPolygon(closeLoop(h), cand[0], cand[1]) !== 'outside') { inHole = true; break; }
      }
      if (!inHole) return cand;
    }
  }
  return null;
}

// A coarse grid sweep followed by two local refinements around the best
// sample — enough to land near the region's own widest point without
// building a real medial axis, and bounded work regardless of how many
// edges the fragment has. Distance is measured to EVERY boundary edge,
// holes included, so a point deep inside the outer loop but hugging a hole
// is correctly rejected in favor of one with real clearance all round.
function deepestInteriorPoint(ring, holeRings) {
  const edges = [];
  for (const loop of [ring, ...holeRings]) {
    for (let i = 0; i < loop.length - 1; i++) edges.push([loop[i], loop[i + 1]]);
  }
  if (!edges.length) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of ring) {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
  }
  const contains = (x, y) => {
    if (pointInUVPolygon(ring, x, y) !== 'inside') return false;
    for (const h of holeRings) if (pointInUVPolygon(h, x, y) !== 'outside') return false;
    return true;
  };
  const clearance = (x, y) => {
    let best = Infinity;
    for (const [a, b] of edges) best = Math.min(best, distPointToSegment2D(x, y, a, b));
    return best;
  };

  let bx = 0, by = 0, bd = -Infinity;
  const sweep = (x0, x1, y0, y1, n) => {
    for (let i = 0; i <= n; i++) {
      for (let j = 0; j <= n; j++) {
        const x = x0 + ((x1 - x0) * i) / n;
        const y = y0 + ((y1 - y0) * j) / n;
        if (!contains(x, y)) continue;
        const d = clearance(x, y);
        if (d > bd) { bd = d; bx = x; by = y; }
      }
    }
  };
  sweep(minX, maxX, minY, maxY, 16);
  if (!(bd > 0)) return null;
  for (let pass = 0; pass < 2; pass++) {
    const rx = (maxX - minX) / Math.pow(16, pass + 1);
    const ry = (maxY - minY) / Math.pow(16, pass + 1);
    sweep(bx - rx, bx + rx, by - ry, by + ry, 8);
  }
  return [bx, by];
}

function distPointToSegment2D(x, y, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (!(len2 > 0)) return Math.hypot(x - a[0], y - a[1]);
  let t = ((x - a[0]) * dx + (y - a[1]) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(x - (a[0] + dx * t), y - (a[1] + dy * t));
}

/**
 * Split one trimmed face by a set of intersection pcurves.
 *
 * `face`  — { outer: [[u,v], ...], holes?: [[[u,v], ...], ...] }. The outer
 *           loop bounds the face; every hole is a region NOT part of it.
 * `curves`— array of [[u,v], ...] chains, each an intersection curve already
 *           expressed in THIS surface's own parameter domain. A chain may be
 *           open (it crosses the face) or closed (an interior loop).
 *
 * Returns { ok, fragments, reason?, danglingCurves, alongBoundary }.
 *
 * Each fragment is { outer, holes, sources } where `sources.outer[i]` is the
 * tag list for the edge leaving `outer[i]` — SRC_TRIM, SRC_INTERSECTION, or
 * both when a pcurve runs exactly along the original boundary. That
 * provenance is the whole reason this returns more than polygons: a fragment
 * edge that came from the intersection is shared with the OTHER solid and
 * must be sewn, while an edge from the original trim boundary is not.
 *
 * `ok: true` with a single fragment equal to the input is a real answer — the
 * curves did not split this face — and is not a failure.
 */
export function splitFaceByCurves(face, curves, opts = {}) {
  const outer = closeLoop(face.outer || []);
  const holes = (face.holes || []).map(closeLoop);
  if (outer.length < 4) {
    return { ok: false, fragments: [], reason: 'the face has no closed outer trim loop to split' };
  }
  // A self-intersecting trim loop has no well-defined interior, so every
  // containment test below would be answering a meaningless question. The
  // area gate at the end WOULD catch it (a bowtie's signed area partly
  // cancels while its two real lobes do not), but it would report a lost
  // region rather than the actual cause — so it is named here instead.
  for (const [i, loop] of [outer, ...holes].entries()) {
    if (polylineSelfIntersects(loop)) {
      return {
        ok: false, fragments: [],
        reason: `${i === 0 ? 'the outer trim loop' : `trim hole ${i}`} intersects itself, so it has no well-defined interior to split`,
      };
    }
  }
  const faceArea = loopArea(outer) - holes.reduce((s, h) => s + loopArea(h), 0);
  if (!(faceArea > 0)) {
    return { ok: false, fragments: [], reason: 'the face has zero or negative area once its holes are subtracted, so there is nothing to split' };
  }
  // Tolerances relative to the face's own size, for the same scale-freedom
  // reason the interior-point step above is.
  const scale = Math.sqrt(faceArea);
  const weldTolerance = opts.weldTolerance ?? scale * 1e-5;

  const polylines = [outer, ...holes];
  const sources = polylines.map(() => SRC_TRIM);
  const usable = [];
  for (const c of curves || []) {
    if (!c || c.length < 2) continue;
    usable.push(c);
    polylines.push(c);
    sources.push(SRC_INTERSECTION);
  }

  const arr = buildPlanarArrangement(polylines, { weldTolerance, sources });

  // A fragment survives only if it is genuinely part of the ORIGINAL face:
  // inside the outer loop and outside every hole. The arrangement also
  // produces the interior of each hole as a bounded region — correctly, it
  // is a real region of the plane — and that is exactly what this drops.
  const fragments = [];
  let unlocatable = 0;
  for (const f of arr.faces) {
    const probe = representativeInteriorPoint(f.outer, f.holes);
    if (!probe) { unlocatable++; continue; }
    if (pointInUVPolygon(outer, probe[0], probe[1]) !== 'inside') continue;
    let inHole = false;
    for (const h of holes) {
      if (pointInUVPolygon(h, probe[0], probe[1]) !== 'outside') { inHole = true; break; }
    }
    if (inHole) continue;
    fragments.push({ outer: f.outer, holes: f.holes, sources: f.sources });
  }

  if (unlocatable) {
    return {
      ok: false, fragments: [],
      reason: `${unlocatable} region(s) of the split could not be located — a fragment with no findable interior point is degenerate, and classifying it downstream would be guessing`,
    };
  }
  if (!fragments.length) {
    return { ok: false, fragments: [], reason: 'the split produced no region inside the original face, which means the trim loops and the intersection curves do not describe a consistent face' };
  }

  // AREA CONSERVATION IS THE INVARIANT WORTH CHECKING HERE. Splitting a face
  // neither creates nor destroys area, and unlike a count check it catches a
  // fragment that was silently dropped, double-counted, or wound backwards.
  //
  // This is a BACKSTOP, and its honest status is worth stating: with the
  // self-intersection refusal above in place, no known input reaches it. It
  // is proven to fire by negative control (disable it and the bowtie case
  // returns a wrong answer instead of refusing), not by a fixture — a gate
  // that never fires on any test is a gate nobody has checked works.
  const totalArea = fragments.reduce((s, fr) => s + loopArea(fr.outer) - fr.holes.reduce((t, h) => t + loopArea(h), 0), 0);
  const areaTol = opts.areaTolerance ?? faceArea * 1e-6;
  if (Math.abs(totalArea - faceArea) > areaTol) {
    return {
      ok: false, fragments: [],
      reason: `the fragments' total area (${totalArea}) does not match the original face's (${faceArea}) — the split lost or duplicated a region rather than partitioning it`,
    };
  }

  // A curve contributing no surviving edge never split anything: it either
  // dangled inside the face or fell outside it entirely. Reported rather
  // than silently ignored, because for two genuinely closed solids an
  // intersection curve should either close or reach a boundary, so this is
  // a real signal about the input and not merely a quiet no-op.
  let sawIntersectionEdge = false;
  let alongBoundary = 0;
  for (const fr of fragments) {
    for (const tags of [...fr.sources.outer, ...fr.sources.holes.flat()]) {
      if (tags.includes(SRC_INTERSECTION)) sawIntersectionEdge = true;
      if (tags.includes(SRC_INTERSECTION) && tags.includes(SRC_TRIM)) alongBoundary++;
    }
  }

  return {
    ok: true,
    fragments,
    danglingCurves: usable.length && !sawIntersectionEdge ? usable.length : 0,
    alongBoundary,
  };
}
