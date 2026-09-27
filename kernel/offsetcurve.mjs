// OffsetCrv — a general planar curve offset (Rhino: OffsetCrv). Given a
// single NURBS curve and a plane (the construction plane every
// 2D-in-a-view command resolves via cplaneAxesForView, as Circle/Polygon/
// Squircle do for their radius point), produce a new curve offset by a signed
// distance within that plane.
//
// This is an approximate offset, as in kernel/offset.mjs's surface offset. An
// exact offset of a general NURBS curve is in general not itself a NURBS curve
// (Rhino's OffsetCrv refits for this reason). The technique: move every
// control point along the local in-plane perpendicular to the curve's tangent
// at that control point's Greville parameter (the parameter of maximum
// influence — via grevilleAbscissae/rationalCurveDerivs). The perpendicular is
// normalize(planeNormal x tangent) — a fixed 90-degree rotation within the
// given plane, so the offset stays in that plane for any input curve, planar
// or not.
//
// Exactness anchor, as in kernel/offset.mjs's flat-plane case: for a degree-1
// curve (a straight Line/Polyline segment), the tangent is the same constant
// vector at every control point, so every point moves by the same
// perpendicular vector — a rigid translation, which is exactly a parallel line
// at the requested distance. For a curved input (a circle, an interpolated
// SketchCurve), the error grows with local curvature x offset distance, the
// same limit OffsetSrf states.

import { grevilleAbscissae, rationalCurveDerivs, curvePoint, reverseCurve, assertCurve } from './curve.mjs';
import { filletCornerArc } from './primitives.mjs';
import { segmentsIntersect } from './trim.mjs';

function normalize3(v) {
  /* The absent case is checked here, not at the caller: `offsetCurve2D`'s
     refusal for a zero-length normal sits after the dereference, so
     `undefined` or `null` would throw a TypeError first. Returning null here
     lets the caller's own message cover every bad input. */
  if (!v || typeof v[0] !== 'number' || typeof v[1] !== 'number' || typeof v[2] !== 'number') return null;
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-12) return null;
  return [v[0] / len, v[1] / len, v[2] / len];
}
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

// offsetCurve2D(crv, distance, planeNormal) => a new curve object (same
// degree/knots, control points moved, weights untouched). Throws, naming the
// reason, when a control point's tangent collapses (a zero-speed point — not
// produced by this app's interpolation/construction functions, but checked
// rather than assumed).
export function offsetCurve2D(crv, distance, planeNormal) {
  assertCurve(crv, 'offsetCurve2D');
  const n = normalize3(planeNormal);
  if (!n) throw new Error('offsetCurve2D: planeNormal must be a real, nonzero vector');
  const g = grevilleAbscissae(crv);
  const newCtrlPts = crv.ctrlPts.map((p, i) => {
    const [, tanRaw] = rationalCurveDerivs(crv, g[i], 1);
    const tangent = normalize3(tanRaw);
    if (!tangent) throw new Error(`offsetCurve2D: control point ${i} has a degenerate (zero-speed) tangent, cannot offset`);
    const perp = normalize3(cross3(n, tangent));
    if (!perp) throw new Error(`offsetCurve2D: control point ${i}'s tangent is parallel to the offset plane's own normal, cannot offset`);
    return [
      p[0] + perp[0] * distance,
      p[1] + perp[1] * distance,
      p[2] + perp[2] * distance,
      p[3],
    ];
  });
  return { degree: crv.degree, knots: crv.knots.slice(), ctrlPts: newCtrlPts };
}

/* Offset corner robustness

   `offsetCurve2D` above moves one continuous curve's control points, which is
   correct for a single smooth curve. But Polygon / Join / PolyCurve results
   are piecewise (several segments meeting at corners), and offsetting each
   piece independently leaves a gap at a corner that turns away from the offset
   side and a self-crossing overlap at one that turns toward it. What follows
   handles that: corner joins (miter with a limit / round / bevel), end caps
   (flat / round), and self-intersection pruning, for both signs of the offset
   distance.

   Which side is positive, stated against the plane normal. The perpendicular
   is `normalize(cross(planeNormal, tangent))`, i.e. the direction of travel
   rotated +90 degrees about `planeNormal` by the right-hand rule. Looking down
   the plane normal (from +n toward the plane, the standard "X right, Y up"
   orientation for n = +Z), that is the left of the direction of travel.
   Consequences, all node-tested: a positive distance offsets left; for a
   closed loop whose signed area is positive when viewed from +n
   (counter-clockwise), left is the interior, so a positive distance shrinks it
   and a negative one grows it; reversing the curve flips the side exactly
   once; negating `planeNormal` flips it exactly once. This is the same formula
   `offsetCurve2D` uses.

   What is exact, and what is not, per case:
   - A polyline input (`offsetPolyline`) with miter joins is exact: every
     emitted point lies on the mitered offset outline to floating-point
     precision (the closest approach to the source is |d| to ~1e-14).
     Straight edges offset by a rigid translation; a miter point is the exact
     intersection of the two offset edge lines. A bevel join's chord is
     likewise exact as geometry (the chord between the two perpendicular
     feet), but a bevel by definition cuts the corner, so its interior points
     sit nearer than |d| to the source — as in SVG/CSS.
   - A round join and a round cap are exact circular arcs: each is a chain of
     rational quadratic spans of at most 90 degrees built by
     `filletCornerArc` (the closed-form P&T Ch.7 conic construction
     `makeArc`/Polygon Fillet/FilletCrv use).
   - Pruning removes a fold exactly at the computed crossing point, but it
     operates on a densified polyline, so any arc span that survives a
     pruning pass is left as its chord-tolerance sampling (`arcTolerance`,
     a chord-height bound, default 0.01mm) rather than an exact arc.
     Line geometry is untouched by densification, so a polyline input with
     miter/bevel joins stays exact even through a pruning pass. When no
     self-intersection exists, pruning is a bit-for-bit no-op and the
     analytic segments are returned untouched.
   - `offsetPolyCurve` (curved segments) inherits `offsetCurve2D`'s
     approximation for each segment — the corner/cap/prune machinery around
     it adds no further error, but does not remove that one. */

// Miter limit — the ratio of the miter point's distance from the corner to
// the offset distance, above which the miter is replaced by a bevel. 4
// matches the SVG/CSS `stroke-miterlimit` default (SVG 1.1 section 11.4; CSS
// Fill and Stroke Module Level 3), which defines the same quantity: the ratio
// equals 1/sin(theta/2) where theta is the interior angle between the two
// edges, so a limit of 4 falls back to a bevel below theta = 2*asin(1/4) ~=
// 28.955 degrees. It is a parameter, not a constant, so a caller can raise
// it.
export const DEFAULT_MITER_LIMIT = 4;

// Default chord-height tolerance used when a pruning pass has to densify
// an arc span. Sits between JOIN_TOLERANCE (0.001mm, "these two points are
// the same point") and the render tessellation tolerance (0.06mm) — tight
// enough that a pruned outline reads as the arc at modeling scale, loose
// enough not to explode point counts.
export const DEFAULT_ARC_TOLERANCE = 0.01;

// A pruning pass removes at least one crossing per iteration and never adds
// points, so it terminates by construction; this bound makes an unforeseen
// degeneracy throw by name instead of hanging.
const MAX_PRUNE_ITERATIONS = 4096;

function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale3(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function dist3(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }

// The offset perpendicular for a direction of travel — the one place this
// file's sign convention is defined, shared by offsetCurve2D's per-control-
// point loop above (which inlines the identical expression) and by every
// corner/cap below.
function offsetPerp(dir, n) {
  const p = normalize3(cross3(n, dir));
  if (!p) throw new Error('offset: a segment direction is parallel to the offset plane normal, so there is no in-plane perpendicular to offset along');
  return p;
}

// Signed turn angle from `dIn` to `dOut` about `n`, in (-PI, PI]. Positive =
// a left turn under the right-hand convention offsetPerp uses. This is
// filletCornerArc's internal convention, computed here so the corner
// classification below can happen before any arc is built.
function turnAngle(dIn, dOut, n) {
  return Math.atan2(dot3(cross3(dIn, dOut), n), dot3(dIn, dOut));
}

// Rodrigues rotation of a unit vector about a unit axis.
function rotateAbout(v, axis, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const k = cross3(axis, v);
  const d = dot3(axis, v) * (1 - c);
  return [v[0] * c + k[0] * s + axis[0] * d, v[1] * c + k[1] * s + axis[1] * d, v[2] * c + k[2] * s + axis[2] * d];
}

// An orthonormal in-plane 2D frame for `n`, used only by the pruning stage
// (every self-intersection test below is a 2D problem; lifting back is exact
// because the frame is orthonormal).
function planeFrame(n) {
  const alt = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = normalize3(cross3(n, alt));
  const e2 = cross3(n, e1);
  return { e1, e2 };
}
function to2D(p, frame) { return [dot3(p, frame.e1), dot3(p, frame.e2)]; }

// Corner joins

// Round join / round cap — a circular arc of radius |distance| centered on the
// original corner vertex, from the incoming edge's perpendicular foot to the
// outgoing edge's perpendicular foot, built as a chain of at-most-90-degree
// rational quadratic spans via `filletCornerArc` at the offset distance
// itself.
//
// Why filletCornerArc solves this exactly: the two offset edge lines are
// parallel to the two original edges at perpendicular distance |distance|, so
// they meet at the miter point M; filleting that corner with radius |distance|
// produces tangent points trimmed back from M by |distance|*tan(halfTurn)
// along each edge direction — and |M - (V + distance*perp)| is exactly
// |distance|*tan(halfTurn) by the same identity. So filletCornerArc's p0/p2
// land exactly on the perpendicular feet, and its apex/weight are exactly the
// circular arc's conic control point and weight. Checked numerically in
// test/offsetcurve-corners.test.mjs.
//
// The turn is split into `ceil(|turn| / 90deg)` spans for the same reason
// makeArc splits a circle into quarter spans: a single rational quadratic span
// degenerates as its half-angle approaches 90 degrees (weight -> 0, control
// apex -> infinity). Splitting keeps every span's weight away from 0 and every
// apex finite, so a near-180-degree round join (and a 180-degree round cap)
// is well-conditioned.
function roundArcSpans(vertex, dIn, dOut, distance, n, turn) {
  const spanCount = Math.max(1, Math.ceil(Math.abs(turn) / (Math.PI / 2)));
  const dirs = [];
  for (let i = 0; i <= spanCount; i++) dirs.push(normalize3(rotateAbout(dIn, n, turn * (i / spanCount))));
  const out = [];
  for (let i = 0; i < spanCount; i++) {
    const a = dirs[i], b = dirs[i + 1];
    const pa = offsetPerp(a, n), pb = offsetPerp(b, n);
    const c = dot3(pa, pb);
    const denom = 1 + c;
    const startPt = add3(vertex, scale3(pa, distance));
    const endPt = add3(vertex, scale3(pb, distance));
    if (denom < 1e-9) { out.push({ type: 'line', a: startPt, b: endPt }); continue; } // structurally unreachable at <=90deg spans; refuses to emit a non-finite apex rather than trusting that
    const apex = add3(vertex, scale3(add3(pa, pb), distance / denom));
    const f = filletCornerArc(apex, sub3(apex, a), add3(apex, b), Math.abs(distance), n);
    if (!f.ok) { out.push({ type: 'line', a: startPt, b: endPt }); continue; } // a span filletCornerArc refuses becomes its own chord rather than a wrong arc
    out.push({ type: 'arc', p0: f.p0, apex: f.apex, p2: f.p2, weight: f.weight });
  }
  return out;
}

// Build the join between two consecutive offset edges meeting at `vertex`.
// Returns `{ segments, kind }`; `segments` connects the incoming offset
// edge's end (vertex + distance*perp(dIn)) to the outgoing offset edge's start
// (vertex + distance*perp(dOut)), so the caller can concatenate directly.
//
// Outer vs inner, decided by geometry rather than an assumed winding: the
// corner opens a gap on the offset side exactly when the turn angle and the
// signed distance have opposite signs (turn * distance < 0). That single
// product covers all four combinations of left/right turn and positive/
// negative distance with one code path.
//
// An inner corner never gets a round or bevel treatment, as in every stroke-
// rendering convention (a line join is an outer-side construct): the two
// offset edges overlap there, and the correct local answer is their
// intersection — the miter point, which for an inner corner lies back inside
// the overlap. For a straight-edge input the caller can act on that directly,
// so an inner miter reports `trimTo: M` and no segments: the two neighboring
// offset edges are shortened to meet exactly at M, which removes the fold
// rather than emitting a spike for the pruning pass to clean up.
// (`offsetPolyCurve` cannot trim a curved segment back to an arbitrary point
// without knot insertion, which this module does not do, so it emits the spike
// and relies on pruning.) When even the inner miter is past the miter limit (a
// near-180-degree reversal), the join degrades to the straight connector
// between the two perpendicular feet and any resulting fold is left for
// pruning.
function cornerJoin(vertex, dIn, dOut, distance, n, opts) {
  const perpIn = offsetPerp(dIn, n), perpOut = offsetPerp(dOut, n);
  const E1 = add3(vertex, scale3(perpIn, distance));
  const S2 = add3(vertex, scale3(perpOut, distance));
  const turn = turnAngle(dIn, dOut, n);
  if (Math.abs(turn) < 1e-9) return { segments: [], kind: 'collinear', E1, S2 }; // the two offset edges already meet exactly; emitting anything here would be a zero-length artifact
  const outer = turn * distance < 0;
  const c = dot3(perpIn, perpOut);
  const denom = 1 + c;
  // ratio === |M - vertex| / |distance| === 1/cos(turn/2) === 1/sin(interiorAngle/2)
  const ratio = denom > 1e-12 ? Math.sqrt(2 / denom) : Infinity;
  const overLimit = !(ratio <= opts.miterLimit);
  const style = outer ? opts.join : 'miter';
  if (style === 'round' && outer) return { segments: roundArcSpans(vertex, dIn, dOut, distance, n, turn), kind: 'round', E1, S2 };
  if (style === 'bevel' && outer) return { segments: [{ type: 'line', a: E1, b: S2 }], kind: 'bevel', E1, S2 };
  if (overLimit) return { segments: [{ type: 'line', a: E1, b: S2 }], kind: outer ? 'bevel(miter-limit)' : 'inner(miter-limit)', E1, S2, limited: true };
  const M = add3(vertex, scale3(add3(perpIn, perpOut), distance / denom));
  if (!outer) return { segments: [{ type: 'line', a: E1, b: M }, { type: 'line', a: M, b: S2 }], kind: 'inner', E1, S2, miterPoint: M, trimTo: M };
  return { segments: [{ type: 'line', a: E1, b: M }, { type: 'line', a: M, b: S2 }], kind: 'miter', E1, S2, miterPoint: M };
}

// Caps

// A cap closes the two sides of an open curve's offset into one outline: the
// forward pass runs at +distance, the cap crosses the end, the reverse pass
// runs back at the mirrored distance, and the second cap closes the loop.
// `dirIn` is the direction of travel arriving at `pt` on the forward side.
// 'flat' is the straight chord (SVG `stroke-linecap: butt`); 'round' is the
// exact semicircular arc of radius |distance| centered on `pt` (SVG
// `stroke-linecap: round`), built through the same at-most-90-degree span
// machinery a round join uses, so a 180-degree turn stays well-conditioned.
// The sweep direction is -sign(distance)*PI, which carries the cap around the
// outside of the tip rather than doubling back through the curve.
function capSegments(pt, dirIn, distance, n, style) {
  const from = add3(pt, scale3(offsetPerp(dirIn, n), distance));
  const to = add3(pt, scale3(offsetPerp(scale3(dirIn, -1), n), distance));
  if (style === 'flat') return [{ type: 'line', a: from, b: to }];
  if (style === 'round') return roundArcSpans(pt, dirIn, scale3(dirIn, -1), distance, n, -Math.sign(distance || 1) * Math.PI);
  throw new Error(`offset: unknown cap style "${style}" (expected 'flat', 'round' or 'none')`);
}

// Densification + self-intersection pruning

// Evaluate a rational quadratic arc span (a filletCornerArc-shaped segment)
// in 3D. The 2D-only equivalent inside primitives.mjs is not exported and
// hardcodes z=0.
function arcPointAt(seg, s) {
  const b0 = (1 - s) * (1 - s), b1 = 2 * s * (1 - s) * seg.weight, b2 = s * s;
  const w = b0 + b1 + b2;
  return [
    (b0 * seg.p0[0] + b1 * seg.apex[0] + b2 * seg.p2[0]) / w,
    (b0 * seg.p0[1] + b1 * seg.apex[1] + b2 * seg.p2[1]) / w,
    (b0 * seg.p0[2] + b1 * seg.apex[2] + b2 * seg.p2[2]) / w,
  ];
}

// Sample count for one arc span at a chord-height tolerance: the span's
// own sweep is 2*acos(weight) (weight === cos(halfSweep) by construction),
// and its radius follows from the chord |p2-p0| === 2*R*sin(sweep/2). A
// chord of angular width a on radius R has sagitta R*(1-cos(a/2)), so the
// largest a meeting `tol` is 2*acos(1 - tol/R).
function arcSampleCount(seg, tol) {
  const w = Math.min(1, Math.max(-1, seg.weight));
  const sweep = 2 * Math.acos(w);
  if (!(sweep > 1e-9)) return 1;
  const chord = dist3(seg.p0, seg.p2);
  const R = chord / (2 * Math.sin(sweep / 2));
  if (!Number.isFinite(R) || R <= 0) return 8;
  const ratio = 1 - tol / R;
  if (ratio <= -1) return 2;
  const maxSpan = 2 * Math.acos(Math.min(1, Math.max(-1, ratio)));
  if (!(maxSpan > 1e-9)) return 256;
  return Math.min(256, Math.max(2, Math.ceil(sweep / maxSpan)));
}

// Flatten a segment chain into a 3D point list. Line segments contribute
// only their two endpoints, so no line geometry is approximated; arcs and
// curves are sampled.
function densifySegments(segments, arcTolerance) {
  const pts = [];
  const push = (p) => { if (!pts.length || dist3(pts[pts.length - 1], p) > 1e-12) pts.push(p); };
  for (const seg of segments) {
    if (seg.type === 'line') { push(seg.a); push(seg.b); continue; }
    if (seg.type === 'arc') {
      const n = arcSampleCount(seg, arcTolerance);
      for (let i = 0; i <= n; i++) push(arcPointAt(seg, i / n));
      continue;
    }
    if (seg.type === 'curve') {
      const crv = seg.crv;
      const u0 = crv.knots[crv.degree], u1 = crv.knots[crv.knots.length - 1 - crv.degree];
      const n = Math.max(2, seg.samples || 48);
      for (let i = 0; i <= n; i++) push(curvePoint(crv, u0 + (u1 - u0) * (i / n)));
      continue;
    }
    throw new Error(`offset: unknown segment type "${seg.type}"`);
  }
  return pts;
}

// A closed outline's point list is a ring: the first point is implicitly the
// last one's neighbor, so a repeated closing point would only contribute a
// zero-length segment to every downstream test. Dropped here once, as
// cleanPoints does on the way in.
function dropClosingDuplicate(pts, closed) {
  if (closed && pts.length > 1 && dist3(pts[0], pts[pts.length - 1]) <= 1e-9) return pts.slice(0, -1);
  return pts;
}

function signedArea2(pts2) {
  let a = 0;
  for (let i = 0; i < pts2.length; i++) {
    const p = pts2[i], q = pts2[(i + 1) % pts2.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

// Exact intersection parameter of two 2D segments already known to cross.
function segCrossParam(p0, p1, p2, p3) {
  const rx = p1[0] - p0[0], ry = p1[1] - p0[1];
  const sx = p3[0] - p2[0], sy = p3[1] - p2[1];
  const den = rx * sy - ry * sx;
  if (Math.abs(den) < 1e-15) return null;
  return ((p2[0] - p0[0]) * sy - (p2[1] - p0[1]) * sx) / den;
}

// Find the first self-crossing in an (open or closed) chain. The crossing
// predicate is trim.mjs's `segmentsIntersect` — the strict-inequality
// orientation test the trim-loop validity check and the keyhole-bridge merge
// use — so "these two edges cross" means one thing across this kernel.
function findFirstCrossing(pts2, closed) {
  const n = pts2.length;
  const segCount = closed ? n : n - 1;
  for (let i = 0; i < segCount; i++) {
    const a0 = pts2[i], a1 = pts2[(i + 1) % n];
    for (let j = i + 2; j < segCount; j++) {
      if (closed && i === 0 && j === segCount - 1) continue; // wraps around to share an endpoint with segment 0 — adjacency, not a crossing
      const b0 = pts2[j], b1 = pts2[(j + 1) % n];
      if (!segmentsIntersect(a0, a1, b0, b1)) continue;
      const t = segCrossParam(a0, a1, b0, b1);
      if (t === null) continue;
      return { i, j, t };
    }
  }
  return null;
}

// Does this offset outline still cross itself? Exported so a caller (and the
// tests) can check the pruning guarantee independently.
export function chainSelfIntersects(points3, planeNormal, closed = false) {
  const n = normalize3(planeNormal);
  if (!n) throw new Error('chainSelfIntersects: planeNormal must be a real, nonzero vector');
  const frame = planeFrame(n);
  const pts2 = points3.map((p) => to2D(p, frame));
  return findFirstCrossing(pts2, closed) !== null;
}

// Self-intersection pruning at a concave corner — remove the folded
// sub-loop rather than render it.
//
// `mergeLoopsKeyhole` does not fit here: it takes several disjoint simple
// loops and builds a self-touching (zero-area-slit) boundary out of them,
// assuming its inputs are already simple; it has no machinery for finding or
// removing a crossing. Pruning is the opposite operation: take one
// self-crossing boundary and remove the invalid sub-loops. What is shared is
// trim.mjs's `segmentsIntersect` (which mergeLoopsKeyhole's bridge validity
// check also calls) plus the shoelace signed-area orientation test trim.mjs
// uses for outer/hole winding.
//
// The rule: repeatedly find the first crossing (i, j) and delete one of the
// two sub-chains it separates, splicing in the exact crossing point. For a
// closed outline the deleted one is whichever sub-loop's signed area runs
// opposite to the outline's overall orientation — that is what a fold is, and
// it is the winding-sign reasoning of trim.mjs's outer/hole rule. For an open
// chain the only meaningful choice is the interior sub-chain, since the two
// free ends must survive.
function pruneChain(pts3, pts2, closed) {
  let a3 = pts3.slice(), a2 = pts2.slice();
  let pruned = false;
  for (let iter = 0; ; iter++) {
    if (iter > MAX_PRUNE_ITERATIONS) throw new Error('offset pruning did not converge — the offset outline still self-intersects after the iteration bound; refusing to return a silently wrong result');
    const hit = findFirstCrossing(a2, closed);
    if (!hit) break;
    pruned = true;
    const { i, j, t } = hit;
    const n = a2.length;
    const x3 = [
      a3[i][0] + (a3[(i + 1) % n][0] - a3[i][0]) * t,
      a3[i][1] + (a3[(i + 1) % n][1] - a3[i][1]) * t,
      a3[i][2] + (a3[(i + 1) % n][2] - a3[i][2]) * t,
    ];
    const x2 = [a2[i][0] + (a2[(i + 1) % n][0] - a2[i][0]) * t, a2[i][1] + (a2[(i + 1) % n][1] - a2[i][1]) * t];
    // Sub-chain A is a2[i+1..j]; removing it keeps the rest. Sub-chain B is
    // the complement; removing it keeps A.
    let keepA = false;
    if (closed) {
      const loopA = [x2].concat(a2.slice(i + 1, j + 1));
      const areaA = signedArea2(loopA);
      const areaAll = signedArea2(a2);
      keepA = Math.sign(areaA) === Math.sign(areaAll) && Math.abs(areaA) > Math.abs(areaAll) / 2;
    }
    if (keepA) {
      const next3 = [x3].concat(a3.slice(i + 1, j + 1));
      const next2 = [x2].concat(a2.slice(i + 1, j + 1));
      a3 = next3; a2 = next2;
    } else {
      const next3 = a3.slice(0, i + 1).concat([x3], a3.slice(j + 1));
      const next2 = a2.slice(0, i + 1).concat([x2], a2.slice(j + 1));
      a3 = next3; a2 = next2;
    }
    if (a2.length < (closed ? 3 : 2)) throw new Error('offset pruning collapsed the outline to nothing — the requested offset distance is larger than the shape itself can support on this side');
  }
  return { points: a3, pruned };
}

// Public entry points

function normalizeOpts(opts) {
  const join = opts.join ?? 'miter';
  if (!['miter', 'round', 'bevel'].includes(join)) throw new Error(`offset: unknown join style "${join}" (expected 'miter', 'round' or 'bevel')`);
  const miterLimit = opts.miterLimit ?? DEFAULT_MITER_LIMIT;
  if (!(miterLimit >= 1)) throw new Error('offset: miterLimit must be at least 1 (a limit below 1 could never be met by any corner)');
  const capStart = opts.capStart ?? opts.cap ?? 'none';
  const capEnd = opts.capEnd ?? opts.cap ?? 'none';
  for (const c of [capStart, capEnd]) if (!['none', 'flat', 'round'].includes(c)) throw new Error(`offset: unknown cap style "${c}" (expected 'none', 'flat' or 'round')`);
  return {
    closed: !!opts.closed,
    join, miterLimit, capStart, capEnd,
    prune: opts.prune !== false,
    arcTolerance: opts.arcTolerance ?? DEFAULT_ARC_TOLERANCE,
  };
}

// Drop consecutive duplicate points (and, for a closed loop, a repeated
// closing point) — a zero-length edge has no direction and therefore no
// perpendicular.
function cleanPoints(points, closed) {
  const out = [];
  for (const p of points) if (!out.length || dist3(out[out.length - 1], p) > 1e-9) out.push(p.slice(0, 3));
  if (closed && out.length > 1 && dist3(out[0], out[out.length - 1]) <= 1e-9) out.pop();
  return out;
}

// One side of a polyline's offset: the offset edges plus the joins between
// them, in path order. No caps, no pruning — those are the caller's job so
// that the capped-outline case can call this twice.
function offsetPolylineSide(points, distance, n, opts) {
  const N = points.length;
  const edgeCount = opts.closed ? N : N - 1;
  const dirs = [];
  for (let i = 0; i < edgeCount; i++) {
    const d = normalize3(sub3(points[(i + 1) % N], points[i]));
    if (!d) throw new Error('offset: two consecutive points coincide, so that edge has no direction');
    dirs.push(d);
  }
  // Build every offset edge first, then resolve the corners — an inner
  // miter shortens its two neighbors to meet exactly at the intersection
  // (see cornerJoin's `trimTo`), which needs the edges to exist as mutable
  // endpoints rather than as an append-only stream.
  const edges = [];
  for (let i = 0; i < edgeCount; i++) {
    const perp = offsetPerp(dirs[i], n);
    edges.push({ type: 'line', a: add3(points[i], scale3(perp, distance)), b: add3(points[(i + 1) % N], scale3(perp, distance)) });
  }
  const cornerSegs = new Array(edgeCount).fill(null); // cornerSegs[i] follows edge i
  const joinKinds = [];
  let limited = 0;
  const cornerCount = opts.closed ? edgeCount : edgeCount - 1;
  for (let i = 0; i < cornerCount; i++) {
    const k = (i + 1) % edgeCount;
    const j = cornerJoin(points[(i + 1) % N], dirs[i], dirs[k], distance, n, opts);
    joinKinds.push(j.kind);
    if (j.limited) limited++;
    if (j.trimTo) { edges[i].b = j.trimTo; edges[k].a = j.trimTo; cornerSegs[i] = []; continue; }
    cornerSegs[i] = j.segments;
  }
  const segments = [];
  for (let i = 0; i < edgeCount; i++) {
    if (dist3(edges[i].a, edges[i].b) > 1e-12) segments.push(edges[i]); // an inner miter can trim an edge to exactly zero length; emitting it would be an artifact
    if (cornerSegs[i]) for (const s of cornerSegs[i]) segments.push(s);
  }
  return { segments, joinKinds, limited, dirFirst: dirs[0], dirLast: dirs[edgeCount - 1] };
}

// offsetPolyline — the main entry point for piecewise straight-segment curves
// (Polyline, Polygon, a Join result whose segments are all straight, a
// densified PolyCurve).
//
// Returns:
//   segments   analytic {type:'line'|'arc'} chain in path order (post-prune
//              if pruning changed anything, in which case it is all lines)
//   points     the same result as a 3D polyline (arcs sampled at
//              `arcTolerance`) — always present, for rendering/verification
//   closed     whether the result is a closed outline (true for a closed
//              input, and for an open input capped at both ends)
//   pruned     whether a self-intersection was removed
//   joins      the per-corner join kinds actually used, in order
//   miterLimitFallbacks  how many corners fell back off the miter limit
export function offsetPolyline(points, distance, planeNormal, opts = {}) {
  const n = normalize3(planeNormal);
  if (!n) throw new Error('offsetPolyline: planeNormal must be a real, nonzero vector');
  if (!Number.isFinite(distance)) throw new Error('offsetPolyline: distance must be a finite number');
  const o = normalizeOpts(opts);
  const pts = cleanPoints(points, o.closed);
  if (pts.length < 2) throw new Error('offsetPolyline: need at least 2 distinct points');
  if (o.closed && pts.length < 3) throw new Error('offsetPolyline: a closed offset needs at least 3 distinct points');
  if (distance === 0) {
    // Exact identity, as offsetCurve2D does at zero distance.
    const segs = [];
    const N = pts.length;
    const edgeCount = o.closed ? N : N - 1;
    for (let i = 0; i < edgeCount; i++) segs.push({ type: 'line', a: pts[i], b: pts[(i + 1) % N] });
    return { segments: segs, points: densifySegments(segs, o.arcTolerance), closed: o.closed, pruned: false, joins: [], miterLimitFallbacks: 0 };
  }

  let segments, joins, limited, resultClosed;
  if (o.closed) {
    const side = offsetPolylineSide(pts, distance, n, o);
    segments = side.segments; joins = side.joinKinds; limited = side.limited; resultClosed = true;
  } else if (o.capStart === 'none' && o.capEnd === 'none') {
    const side = offsetPolylineSide(pts, distance, n, o);
    segments = side.segments; joins = side.joinKinds; limited = side.limited; resultClosed = false;
  } else {
    // A capped open curve becomes a closed outline: forward side, end cap,
    // reverse side (the mirrored offset of the original, reached by reversing
    // the point order rather than negating the distance — so every join is
    // computed by the same code, with a corner that was outer on one side
    // reading inner on the other), start cap.
    const fwd = offsetPolylineSide(pts, distance, n, o);
    const rev = offsetPolylineSide(pts.slice().reverse(), distance, n, o);
    segments = fwd.segments.slice();
    joins = fwd.joinKinds.concat(rev.joinKinds);
    limited = fwd.limited + rev.limited;
    if (o.capEnd !== 'none') segments = segments.concat(capSegments(pts[pts.length - 1], fwd.dirLast, distance, n, o.capEnd));
    segments = segments.concat(rev.segments);
    if (o.capStart !== 'none') segments = segments.concat(capSegments(pts[0], scale3(fwd.dirFirst, -1), distance, n, o.capStart));
    resultClosed = o.capStart !== 'none' && o.capEnd !== 'none';
  }

  const frame = planeFrame(n);
  let outPoints = dropClosingDuplicate(densifySegments(segments, o.arcTolerance), resultClosed);
  let pruned = false;
  if (o.prune) {
    const res = pruneChain(outPoints, outPoints.map((p) => to2D(p, frame)), resultClosed);
    pruned = res.pruned;
    if (pruned) {
      outPoints = res.points;
      // Once a fold has been removed the analytic chain does not describe the
      // result, so the pruned polyline itself is returned.
      segments = [];
      const last = resultClosed ? outPoints.length : outPoints.length - 1;
      for (let i = 0; i < last; i++) segments.push({ type: 'line', a: outPoints[i], b: outPoints[(i + 1) % outPoints.length] });
    }
  }
  return { segments, points: outPoints, closed: resultClosed, pruned, joins, miterLimitFallbacks: limited };
}

// offsetPolyCurve — the same machinery applied to a chain of curved segments
// (a PolyCurve / Join result), each offset by `offsetCurve2D` and stitched
// with the same corner joins, caps and pruning as above.
//
// The stitch is exact by construction, not by tolerance: a clamped B-spline
// starts and ends at its first/last control point, and the Greville abscissa
// of that control point is the domain end itself, so `offsetCurve2D` moves it
// along the perpendicular of the tangent at the same parameter this function
// reads for the join. The offset segment's endpoint is therefore identically
// `vertex + distance*perp(endTangent)` — the point `cornerJoin` connects from.
//
// The per-segment offset inherits offsetCurve2D's approximation for a curved
// segment (exact only for degree 1).
export function offsetPolyCurve(curves, distance, planeNormal, opts = {}) {
  const n = normalize3(planeNormal);
  if (!n) throw new Error('offsetPolyCurve: planeNormal must be a real, nonzero vector');
  if (!Number.isFinite(distance)) throw new Error('offsetPolyCurve: distance must be a finite number');
  if (!curves || curves.length < 1) throw new Error('offsetPolyCurve: need at least one curve segment');
  const o = normalizeOpts(opts);

  const ends = curves.map((crv) => {
    const u0 = crv.knots[crv.degree], u1 = crv.knots[crv.knots.length - 1 - crv.degree];
    const [p0, t0] = rationalCurveDerivs(crv, u0, 1);
    const [p1, t1] = rationalCurveDerivs(crv, u1, 1);
    const d0 = normalize3(t0), d1 = normalize3(t1);
    if (!d0 || !d1) throw new Error('offsetPolyCurve: a segment has a degenerate (zero-speed) tangent at one of its own ends');
    return { start: p0, end: p1, dirStart: d0, dirEnd: d1 };
  });
  for (let i = 0; i + 1 < curves.length; i++) {
    if (dist3(ends[i].end, ends[i + 1].start) > 1e-6) throw new Error(`offsetPolyCurve: segment ${i} does not meet segment ${i + 1} — the chain must be continuous before it can be offset as one outline`);
  }
  if (o.closed && dist3(ends[ends.length - 1].end, ends[0].start) > 1e-6) throw new Error('offsetPolyCurve: a closed chain must end where it starts');

  const buildSide = (list, listEnds) => {
    const segs = [];
    const kinds = [];
    let limited = 0;
    for (let i = 0; i < list.length; i++) {
      segs.push({ type: 'curve', crv: offsetCurve2D(list[i], distance, n) });
      const isLastOpen = !o.closed && i === list.length - 1;
      if (isLastOpen) continue;
      const k = (i + 1) % list.length;
      const j = cornerJoin(listEnds[i].end, listEnds[i].dirEnd, listEnds[k].dirStart, distance, n, o);
      for (const s of j.segments) segs.push(s);
      kinds.push(j.kind);
      if (j.limited) limited++;
    }
    return { segs, kinds, limited };
  };

  let segments, joins, limited, resultClosed;
  if (o.closed || (o.capStart === 'none' && o.capEnd === 'none')) {
    const side = buildSide(curves, ends);
    segments = side.segs; joins = side.kinds; limited = side.limited; resultClosed = !!o.closed;
  } else {
    const revCurves = curves.slice().reverse().map((c) => reverseCurve(c));
    const revEnds = revCurves.map((crv) => {
      const u0 = crv.knots[crv.degree], u1 = crv.knots[crv.knots.length - 1 - crv.degree];
      const [p0, t0] = rationalCurveDerivs(crv, u0, 1);
      const [p1, t1] = rationalCurveDerivs(crv, u1, 1);
      return { start: p0, end: p1, dirStart: normalize3(t0), dirEnd: normalize3(t1) };
    });
    const fwd = buildSide(curves, ends);
    const rev = buildSide(revCurves, revEnds);
    segments = fwd.segs.slice();
    joins = fwd.kinds.concat(rev.kinds);
    limited = fwd.limited + rev.limited;
    if (o.capEnd !== 'none') segments = segments.concat(capSegments(ends[ends.length - 1].end, ends[ends.length - 1].dirEnd, distance, n, o.capEnd));
    segments = segments.concat(rev.segs);
    if (o.capStart !== 'none') segments = segments.concat(capSegments(ends[0].start, scale3(ends[0].dirStart, -1), distance, n, o.capStart));
    resultClosed = o.capStart !== 'none' && o.capEnd !== 'none';
  }

  const frame = planeFrame(n);
  let outPoints = dropClosingDuplicate(densifySegments(segments, o.arcTolerance), resultClosed);
  let pruned = false;
  if (o.prune) {
    const res = pruneChain(outPoints, outPoints.map((p) => to2D(p, frame)), resultClosed);
    pruned = res.pruned;
    if (pruned) {
      outPoints = res.points;
      segments = [];
      const last = resultClosed ? outPoints.length : outPoints.length - 1;
      for (let i = 0; i < last; i++) segments.push({ type: 'line', a: outPoints[i], b: outPoints[(i + 1) % outPoints.length] });
    }
  }
  return { segments, points: outPoints, closed: resultClosed, pruned, joins, miterLimitFallbacks: limited };
}
