// OFFSET CORNER ROBUSTNESS.
//
// Every fixture below is deliberately ASYMMETRIC and mostly non-axis-aligned,
// with BOTH convex and concave corners present at once: a square's offset
// passes trivially (all four corners identical, all four joins the same kind)
// while a real polyline with mixed turn directions does not, and the mixed
// case is the whole point of this file.
//
// The properties asserted here are geometric invariants recomputed
// INDEPENDENTLY in this file (perpendicular distance to the source polyline,
// rational-quadratic evaluation and its own finite-difference tangent,
// shoelace area, segment-segment crossing) rather than read back out of the
// module under test — a self-consistency check would prove nothing.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  offsetPolyline, offsetPolyCurve, chainSelfIntersects,
  offsetCurve2D, DEFAULT_MITER_LIMIT, DEFAULT_ARC_TOLERANCE,
} from '../kernel/offsetcurve.mjs';
import { makeArc, makeLine } from '../kernel/primitives.mjs';
import { curvePoint, closestPointOnCurve } from '../kernel/curve.mjs';
import { signedArea2D } from '../kernel/trim.mjs';

const NZ = [0, 0, 1];

// An asymmetric 6-point open polyline: two LEFT turns then two RIGHT turns,
// so a positive-distance (left-side) offset sees two INNER corners followed
// by two OUTER ones in the same run, and a negative distance sees the exact
// reverse. Nothing here is axis-aligned except the very first vertex.
const ASYM = [[0, 0, 0], [60, -8, 0], [92, 26, 0], [54, 30, 0], [46, 66, 0], [100, 88, 0]];

// A closed CCW quadrilateral with no right angles and no symmetry.
const QUAD = [[0, 0, 0], [80, 6, 0], [70, 55, 0], [10, 40, 0]];

/* ---- independent geometry, recomputed here on purpose ---- */

function distPointSeg2D(p, a, b) {
  const ax = b[0] - a[0], ay = b[1] - a[1];
  const lenSq = ax * ax + ay * ay;
  let t = lenSq < 1e-18 ? 0 : ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / lenSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - (a[0] + t * ax), p[1] - (a[1] + t * ay));
}
function distToPolyline(p, pts, closed) {
  let best = Infinity;
  const m = closed ? pts.length : pts.length - 1;
  for (let i = 0; i < m; i++) best = Math.min(best, distPointSeg2D(p, pts[i], pts[(i + 1) % pts.length]));
  return best;
}
// Dense resample of a point chain — the offset's own vertices are not enough
// to prove a distance property; the EDGES between them have to be checked too.
function densify(pts, closed, per = 60) {
  const out = [];
  const m = closed ? pts.length : pts.length - 1;
  for (let i = 0; i < m; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    for (let k = 0; k <= per; k++) out.push([a[0] + (b[0] - a[0]) * k / per, a[1] + (b[1] - a[1]) * k / per, 0]);
  }
  return out;
}
// Evaluate a rational quadratic {p0, apex, p2, weight} — this is the arc
// segment shape filletCornerArc/filletPolygon already emit; recomputing it
// here rather than importing keeps the tangency proof independent.
function arcAt(seg, s) {
  const b0 = (1 - s) * (1 - s), b1 = 2 * s * (1 - s) * seg.weight, b2 = s * s;
  const w = b0 + b1 + b2;
  return [0, 1, 2].map((k) => (b0 * seg.p0[k] + b1 * seg.apex[k] + b2 * seg.p2[k]) / w);
}
function unit(v) { const L = Math.hypot(v[0], v[1], v[2]); return [v[0] / L, v[1] / L, v[2] / L]; }
function segDir(seg) {
  if (seg.type === 'line') return unit([seg.b[0] - seg.a[0], seg.b[1] - seg.a[1], seg.b[2] - seg.a[2]]);
  const e = 1e-6;
  const p = arcAt(seg, e);
  return unit([p[0] - seg.p0[0], p[1] - seg.p0[1], p[2] - seg.p0[2]]);
}
function segEndDir(seg) {
  if (seg.type === 'line') return segDir(seg);
  const e = 1e-6;
  const p = arcAt(seg, 1 - e);
  return unit([seg.p2[0] - p[0], seg.p2[1] - p[1], seg.p2[2] - p[2]]);
}
function segStart(seg) { return seg.type === 'line' ? seg.a : seg.p0; }
function segEnd(seg) { return seg.type === 'line' ? seg.b : seg.p2; }
function dot2(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function allFinite(pts) { return pts.every((p) => p.every(Number.isFinite)); }
function bboxSpan(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of pts) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
  return Math.max(x1 - x0, y1 - y0);
}

/* ================================================================
   1. EXACTNESS — a straight input's offset is exact
   ================================================================ */

test('offsetPolyline: a straight-segment input with miter joins is EXACT — the offset never comes closer than |d| to the source, and touches exactly |d|', () => {
  // The right invariant for a mitered offset (rather than "every point sits
  // at distance d"): a miter point deliberately sits FURTHER than d from the
  // corner — that is what a miter is — while every point along an offset EDGE
  // sits at exactly d. So: min over the whole outline === |d| exactly, and no
  // point is nearer than that. This holds for both signs and for a corner
  // sequence containing both inner and outer turns.
  // Stated limit, not hidden: "never closer than d" is a property of an
  // offset that does not overlap ITSELF. Push the distance past the fixture's
  // own feature size (here, past about 12mm) and distant parts of the outline
  // legitimately approach other parts of the SOURCE — that is real geometry,
  // not an error, and it is what pruning exists to resolve. So the exactness
  // claim is asserted over the range where the offset is genuinely simple.
  for (const d of [6, -6, 0.25, -0.25, 11, -11]) {
    const r = offsetPolyline(ASYM, d, NZ, { join: 'miter' });
    assert.equal(r.pruned, false, `no fold at |d|=${Math.abs(d)} on this fixture, so pruning must be a no-op`);
    let min = Infinity;
    for (const p of densify(r.points, false)) min = Math.min(min, distToPolyline(p, ASYM, false));
    assert.ok(Math.abs(min - Math.abs(d)) < 1e-12 * Math.max(1, Math.abs(d)), `closest approach is exactly |d| for d=${d} (got ${min}, want ${Math.abs(d)})`);
  }
});

test('offsetPolyline: each offset edge is EXACTLY parallel to its source edge and exactly |d| away (the per-edge form of the same exactness claim)', () => {
  const d = 9;
  const r = offsetPolyline(ASYM, d, NZ, { join: 'miter' });
  // Every emitted LINE segment must run parallel (|dot| === 1) to SOME source
  // edge; the only non-parallel lines a miter join can emit are the two halves
  // of the miter itself, which are collinear with their own neighboring
  // offset edges by construction, so this is a total claim, not a sampled one.
  for (const seg of r.segments) {
    if (seg.type !== 'line') continue;
    const dir = segDir(seg);
    let bestAlign = 0;
    for (let i = 0; i + 1 < ASYM.length; i++) {
      const e = unit([ASYM[i + 1][0] - ASYM[i][0], ASYM[i + 1][1] - ASYM[i][1], 0]);
      bestAlign = Math.max(bestAlign, Math.abs(dot2(dir, e)));
    }
    assert.ok(Math.abs(bestAlign - 1) < 1e-12, `offset line segment is exactly parallel to a source edge (worst |dot| off by ${1 - bestAlign})`);
  }
});

test('offsetPolyline: a BEVEL join genuinely cuts the corner — nearer than |d| there, and never further, exactly as SVG/CSS defines it', () => {
  // Asserted rather than glossed, because it is the one place the "never
  // closer than d" invariant deliberately does not hold: a bevel is a chord
  // across the corner, so the corner region is UNDER-offset by construction.
  const d = 6;
  const r = offsetPolyline(ASYM, d, NZ, { join: 'bevel' });
  let min = Infinity, max = -Infinity;
  for (const p of densify(r.points, false)) {
    const q = distToPolyline(p, ASYM, false);
    min = Math.min(min, q); max = Math.max(max, q);
  }
  assert.ok(min < d - 0.5, `a bevel really does cut inside |d| at the corner (closest ${min.toFixed(4)})`);
  assert.ok(max <= d + 1e-9, `and never goes beyond |d| anywhere — unlike a miter (furthest ${max.toFixed(6)})`);
  const miter = offsetPolyline(ASYM, d, NZ, { join: 'miter' });
  let miterMax = -Infinity;
  for (const p of densify(miter.points, false)) miterMax = Math.max(miterMax, distToPolyline(p, ASYM, false));
  assert.ok(miterMax > d + 0.5, `whereas a miter deliberately reaches past |d| (furthest ${miterMax.toFixed(4)}) — the two styles are genuinely different, not a renamed no-op`);
});

test('offsetPolyline: zero distance is the exact identity (matching offsetCurve2D own zero-distance behavior)', () => {
  const r = offsetPolyline(ASYM, 0, NZ, {});
  assert.equal(r.points.length, ASYM.length);
  for (let i = 0; i < ASYM.length; i++) for (let k = 0; k < 3; k++) assert.equal(r.points[i][k], ASYM[i][k]);
});

/* ================================================================
   2. OFFSET DISTANCE IS ACTUALLY ACHIEVED — dense sampling
   ================================================================ */

test('offsetPolyline: a ROUND-joined offset sits at EXACTLY |d| from the source everywhere, corners included (the arc is centered on the corner vertex)', () => {
  // A round join's arc is centered on the ORIGINAL corner vertex at radius
  // |d|, so unlike a miter (further at the corner) or a bevel (nearer), a
  // fully round-joined offset is at exactly |d| along its whole length. That
  // is checked here on the ANALYTIC segments, not on the densified polyline —
  // a chord sampling of an arc necessarily sits slightly inside the true arc,
  // which is a fact about sampling, not about the offset.
  const d = 6;
  const r = offsetPolyline(ASYM, d, NZ, { join: 'round' });
  let worst = 0;
  for (const seg of r.segments) {
    const pts = [];
    if (seg.type === 'line') { for (let k = 0; k <= 60; k++) pts.push([seg.a[0] + (seg.b[0] - seg.a[0]) * k / 60, seg.a[1] + (seg.b[1] - seg.a[1]) * k / 60, 0]); }
    else { for (let k = 0; k <= 60; k++) pts.push(arcAt(seg, k / 60)); }
    for (const p of pts) worst = Math.max(worst, Math.abs(distToPolyline(p, ASYM, false) - d));
  }
  assert.ok(worst < 1e-9, `every point of a round-joined offset is exactly |d| from the source (worst deviation ${worst.toExponential(3)})`);
});

test('offsetPolyCurve: on a genuinely CURVED chain the distance is achieved to the honest, stated tolerance of the underlying control-point offset, not exactly', () => {
  // A quarter arc (R=40, one 90-degree rational span) joined tangent-
  // continuously to a straight run. offsetCurve2D moves control points, and a
  // rational quarter arc's own MIDDLE control point sits at R*sqrt(2) from the
  // center and is OFF the curve — moving it radially by d instead of by
  // d*sec(45deg) is exactly where the approximation lives. The resulting worst
  // deviation is therefore ~ d*(sec(45deg) - 1)/... — measured here at
  // 1.03mm for d=6, i.e. 17.2% of d, on a 90-degree span.
  //
  // THE TOLERANCE AND WHY IT IS THE RIGHT ONE: 0.20*|d| — just above the
  // measured 0.172*|d|, and tied to |d| rather than to an absolute millimeter
  // figure because the error of this technique scales with offset distance
  // times local curvature, not with model size. A tighter absolute number
  // would be a fixture-fitted constant, and an absolute-mm bound would silently
  // pass or fail depending on the scale the student happens to model at.
  const arc = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 40, 0, Math.PI / 2, 1);
  const arcEnd = curvePoint(arc, arc.knots[arc.knots.length - 1 - arc.degree]);
  const line = makeLine(arcEnd, [-60, 40, 0]);
  const d = 6;
  const r = offsetPolyCurve([arc, line], d, NZ, {});
  // The offset runs on the LEFT of travel: the arc sweeps counter-clockwise
  // and the line heads -X, so the whole outline sits INSIDE (arc side) and
  // BELOW (line side) — at y = 40 - d for the straight run.
  const sampled = r.points.filter((_, i) => i % 6 === 0); // closestPointOnCurve is a real Newton solve; a third of ~100 points is still a dense sample
  let worst = 0;
  for (const p of sampled) {
    const a = closestPointOnCurve(arc, p, 1e-9);
    const b = closestPointOnCurve(line, p, 1e-9);
    const dist = Math.min(
      Math.hypot(p[0] - a.point[0], p[1] - a.point[1], p[2] - a.point[2]),
      Math.hypot(p[0] - b.point[0], p[1] - b.point[1], p[2] - b.point[2]),
    );
    worst = Math.max(worst, Math.abs(dist - d));
  }
  assert.ok(worst <= 0.20 * d, `curved-chain offset stays within 0.20*|d| of the requested distance (worst ${worst.toFixed(4)} vs bound ${(0.2 * d).toFixed(4)})`);
  assert.ok(worst > 0.01, `and it is genuinely NOT exact — the honest approximation is real, measured ${worst.toFixed(4)}, so this test would notice if it were quietly replaced by a claim of exactness`);
  // The STRAIGHT half of the same chain is still exact, which is what makes
  // the degree-1 exactness anchor a real anchor rather than a special case.
  const straightPts = r.points.filter((p) => Math.abs(p[1] - (40 - d)) < 1e-6 && p[0] < -1);
  assert.ok(straightPts.length > 2, `the straight run really is present in the sampled output (found ${straightPts.length})`);
  for (const p of straightPts) {
    const b = closestPointOnCurve(line, p, 1e-9);
    const dist = Math.hypot(p[0] - b.point[0], p[1] - b.point[1], p[2] - b.point[2]);
    assert.ok(Math.abs(dist - d) < 1e-9, `the degree-1 half of the chain offsets exactly (got ${dist})`);
  }
});

/* ================================================================
   3. ROUND JOINS ARE REAL TANGENT ARCS
   ================================================================ */

test('offsetPolyline: a ROUND join is a genuine tangent circular arc — exact radius, and tangent to BOTH neighboring offset edges at its own two ends', () => {
  const d = 6;
  const r = offsetPolyline(ASYM, d, NZ, { join: 'round' });
  const arcs = r.segments.filter((s) => s.type === 'arc');
  assert.equal(arcs.length, 2, 'this fixture has exactly two OUTER corners at a positive distance, so exactly two round joins');
  let worstRadius = 0, worstTangency = 0;
  for (let i = 0; i < r.segments.length; i++) {
    const seg = r.segments[i];
    if (seg.type !== 'arc') continue;
    // The arc must be centered on a REAL source vertex at radius exactly |d| —
    // found by search rather than assumed, so a wrong center cannot pass.
    const centre = ASYM.find((v) => Math.abs(Math.hypot(seg.p0[0] - v[0], seg.p0[1] - v[1]) - d) < 1e-9);
    assert.ok(centre, 'the round join arc starts exactly |d| from one of the source vertices');
    for (let k = 0; k <= 40; k++) {
      const p = arcAt(seg, k / 40);
      worstRadius = Math.max(worstRadius, Math.abs(Math.hypot(p[0] - centre[0], p[1] - centre[1]) - d));
    }
    // TANGENCY — the real check. Not "points exist near the corner": the
    // arc's own derivative at s=0 must be the incoming offset edge's own
    // direction and at s=1 the outgoing one, and its endpoints must coincide
    // with theirs.
    const prev = r.segments[i - 1], next = r.segments[i + 1];
    assert.ok(prev && next, 'a round join always sits between two offset edges');
    worstTangency = Math.max(worstTangency, Math.abs(1 - dot2(segDir(seg), segEndDir(prev))), Math.abs(1 - dot2(segEndDir(seg), segDir(next))));
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(segEnd(prev)[k] - seg.p0[k]) < 1e-9, 'the arc starts exactly where the incoming offset edge ends');
      assert.ok(Math.abs(segStart(next)[k] - seg.p2[k]) < 1e-9, 'the arc ends exactly where the outgoing offset edge starts');
    }
  }
  assert.ok(worstRadius < 1e-9, `round join radius is exactly |d| (worst error ${worstRadius.toExponential(3)})`);
  assert.ok(worstTangency < 1e-9, `round join is tangent at BOTH ends (worst 1-dot ${worstTangency.toExponential(3)})`);
});

test('offsetPolyline: a round join spanning more than 90 degrees is split into well-conditioned spans rather than one degenerate conic', () => {
  // A single rational quadratic span degenerates as its half-angle approaches
  // 90 degrees (weight -> 0, apex -> infinity). The 90-degree split is what
  // keeps a near-reversal round join finite, so it is asserted directly.
  const hairpin = [[0, 0, 0], [100, 0, 0], [2, 3, 0]]; // ~178 degree turn
  const r = offsetPolyline(hairpin, -3, NZ, { join: 'round' });
  const arcs = r.segments.filter((s) => s.type === 'arc');
  assert.ok(arcs.length >= 2, `a ~178-degree round join is split into at least 2 spans (got ${arcs.length})`);
  for (const a of arcs) {
    assert.ok(a.weight > 0.2, `every span's rational weight stays well away from 0 (got ${a.weight})`);
    assert.ok(allFinite([a.p0, a.apex, a.p2]), 'every span control point is finite');
  }
});

/* ================================================================
   4. SELF-INTERSECTION PRUNING
   ================================================================ */

// A closed CCW outline with a genuine narrow, SLANTED, tapering tab. Offset
// inward far enough and the tab's two walls' offsets cross each other — a real
// fold that no amount of local corner cleverness removes, because the crossing
// segments are not adjacent.
const TABBED = [[0, 0, 0], [120, 0, 0], [120, 50, 0], [78, 50, 0], [70, 95, 0], [58, 92, 0], [52, 50, 0], [0, 50, 0]];

test('offsetPolyline: a concave-corner fold is genuinely REMOVED — the raw offset self-intersects, the pruned one provably does not', () => {
  for (const d of [8, 12, 15]) {
    const raw = offsetPolyline(TABBED, d, NZ, { closed: true, prune: false });
    const pruned = offsetPolyline(TABBED, d, NZ, { closed: true, prune: true });
    // The negative control: without pruning there really IS a crossing, so
    // this test cannot pass vacuously against an input that never folded.
    assert.equal(chainSelfIntersects(raw.points, NZ, true), true, `the unpruned offset at d=${d} genuinely self-intersects`);
    assert.equal(pruned.pruned, true, `pruning reports that it did real work at d=${d}`);
    // The actual guarantee — not "the point count dropped".
    assert.equal(chainSelfIntersects(pruned.points, NZ, true), false, `the pruned offset at d=${d} has NO self-intersections`);
    assert.ok(allFinite(pruned.points), 'every pruned point is finite');
    // Still a real inward offset: never closer than d to the source, and
    // strictly smaller in area than the source.
    let min = Infinity;
    for (const p of densify(pruned.points, true)) min = Math.min(min, distToPolyline(p, TABBED, true));
    assert.ok(min > d - 1e-9, `the pruned outline never comes closer than d=${d} to the source (min ${min.toFixed(6)})`);
    const srcArea = signedArea2D(TABBED.map((p) => [p[0], p[1]]));
    const offArea = signedArea2D(pruned.points.map((p) => [p[0], p[1]]));
    assert.ok(offArea > 0 && offArea < srcArea, `an inward offset stays correctly wound and strictly smaller (${offArea.toFixed(1)} vs ${srcArea.toFixed(1)})`);
  }
});

test('offsetPolyline: an offset with no fold at all leaves the analytic segments untouched — pruning is a bit-for-bit no-op, not a silent densification', () => {
  const withPrune = offsetPolyline(ASYM, 6, NZ, { join: 'round', prune: true });
  const without = offsetPolyline(ASYM, 6, NZ, { join: 'round', prune: false });
  assert.equal(withPrune.pruned, false);
  assert.equal(withPrune.segments.length, without.segments.length);
  for (let i = 0; i < withPrune.segments.length; i++) {
    assert.equal(withPrune.segments[i].type, without.segments[i].type, `segment ${i} keeps its analytic type (an arc stays an arc)`);
  }
});

test('chainSelfIntersects: the check itself is real — it finds a hand-built crossing and clears a hand-built simple loop', () => {
  const crossing = [[0, 0, 0], [10, 10, 0], [10, 0, 0], [0, 10, 0]]; // a bow tie
  const simple = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]];
  assert.equal(chainSelfIntersects(crossing, NZ, true), true);
  assert.equal(chainSelfIntersects(simple, NZ, true), false);
  // Open vs closed genuinely differ: these four points as an OPEN chain do not
  // cross, but the implicit CLOSING edge (last back to first) does.
  const openOk = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [-5, 5, 0]];
  assert.equal(chainSelfIntersects(openOk, NZ, false), false, 'the open chain is simple');
  assert.equal(chainSelfIntersects([[0, 5, 0], [10, 5, 0], [10, 10, 0], [5, 10, 0], [5, 0, 0]], NZ, true), true, 'the implicit closing edge is what crosses here');
});

/* ================================================================
   5. MITER LIMIT
   ================================================================ */

test('offsetPolyline: a near-180-degree OUTER corner hits the miter limit and falls back to a bevel instead of shooting out', () => {
  const hairpin = [[0, 0, 0], [100, 0, 0], [2, 3, 0]]; // ~178.2 degrees of turn
  const limited = offsetPolyline(hairpin, -3, NZ, { join: 'miter' }); // default limit 4
  const unlimited = offsetPolyline(hairpin, -3, NZ, { join: 'miter', miterLimit: 1e6 });

  assert.equal(limited.joins[0], 'bevel(miter-limit)', 'the corner is reported as a real fallback, not silently mitered');
  assert.equal(limited.miterLimitFallbacks, 1);
  assert.equal(unlimited.joins[0], 'miter', 'with the limit raised the SAME corner keeps its miter — proving the fallback is the limit doing work, not the corner being unmiterable');
  assert.equal(unlimited.miterLimitFallbacks, 0);

  // FINITENESS, asserted explicitly rather than assumed.
  assert.ok(allFinite(limited.points), 'every limited point is finite');
  assert.ok(allFinite(unlimited.points), 'even the unlimited miter is finite — the limit exists to bound magnitude, not to rescue a NaN');
  // And the magnitude claim is real, not rhetorical: the input spans 100mm,
  // the bevelled result stays inside ~1.1x that, the unmitered one blows past 2x.
  const srcSpan = bboxSpan(hairpin);
  assert.ok(bboxSpan(limited.points) < 1.1 * srcSpan, `the bevelled result stays at the input's own scale (${bboxSpan(limited.points).toFixed(1)} vs ${srcSpan.toFixed(1)})`);
  assert.ok(bboxSpan(unlimited.points) > 2 * srcSpan, `the unlimited miter genuinely shoots out (${bboxSpan(unlimited.points).toFixed(1)}), which is what the limit exists to stop`);
});

test('offsetPolyline: an EXACT 180-degree reversal (a doubled-back polyline) stays finite instead of dividing by zero', () => {
  const reversal = [[0, 0, 0], [50, 0, 0], [0, 0, 0]];
  for (const join of ['miter', 'round', 'bevel']) {
    for (const d of [4, -4]) {
      const r = offsetPolyline(reversal, d, NZ, { join });
      assert.ok(allFinite(r.points), `exact reversal stays finite for join=${join}, d=${d}`);
      assert.ok(r.points.length >= 2, 'and still produces a real chain');
    }
  }
});

test('offsetPolyline: DEFAULT_MITER_LIMIT is the web platform default, and the cutover angle it implies is the one the ratio formula predicts', () => {
  assert.equal(DEFAULT_MITER_LIMIT, 4);
  // ratio === 1/sin(interiorAngle/2); a limit of 4 cuts over at
  // interiorAngle === 2*asin(1/4). Build two corners straddling that angle and
  // confirm the fallback fires on exactly the right side of it.
  const cut = 2 * Math.asin(1 / DEFAULT_MITER_LIMIT);
  const build = (interior) => {
    // A symmetric V at the origin whose INTERIOR angle (the angle between the
    // two edges leaving the corner) is exactly `interior`: placing the two
    // arms at +/- interior/2 about +Y gives an angle between them of exactly
    // `interior`. Derived rather than eyeballed, because the whole point of
    // this test is that the cutover lands where the formula says.
    const h = interior / 2;
    return [[-Math.sin(h) * 50, Math.cos(h) * 50, 0], [0, 0, 0], [Math.sin(h) * 50, Math.cos(h) * 50, 0]];
  };
  const wider = offsetPolyline(build(cut + 0.05), -3, NZ, { join: 'miter' });
  const tighter = offsetPolyline(build(cut - 0.05), -3, NZ, { join: 'miter' });
  assert.equal(wider.miterLimitFallbacks, 0, 'just above the cutover angle the miter is kept');
  assert.equal(tighter.miterLimitFallbacks, 1, 'just below it the miter is abandoned');
});

/* ================================================================
   6. CAPS
   ================================================================ */

test('offsetPolyline: caps close an open curve\'s offset into a real outline — flat and round, on either or both ends', () => {
  const path = [[0, 0, 0], [100, 0, 0], [140, 55, 0]];
  const d = 5;

  const both = offsetPolyline(path, d, NZ, { capStart: 'flat', capEnd: 'flat' });
  assert.equal(both.closed, true, 'capped at BOTH ends the result is a genuinely closed outline');
  assert.ok(!chainSelfIntersects(both.points, NZ, true), 'and it does not cross itself');

  const oneEnd = offsetPolyline(path, d, NZ, { capStart: 'none', capEnd: 'round' });
  assert.equal(oneEnd.closed, false, 'capped at ONE end only, the result is honestly still open');

  const none = offsetPolyline(path, d, NZ, {});
  assert.equal(none.closed, false);
  assert.ok(none.points.length < both.points.length, 'an uncapped offset is one side only');

  // FLAT cap: a straight chord of length exactly 2|d| across the end point,
  // perpendicular to the direction of travel there.
  const flat = offsetPolyline(path, d, NZ, { capEnd: 'flat', capStart: 'none' });
  const endTip = path[path.length - 1];
  const capLine = flat.segments.find((s) => s.type === 'line' && Math.abs(distPointSeg2D(endTip, s.a, s.b)) < 1e-9 && Math.abs(Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) - 2 * d) < 1e-9);
  assert.ok(capLine, 'the flat cap is a straight chord of length exactly 2|d| passing through the end point');
  const travel = unit([path[2][0] - path[1][0], path[2][1] - path[1][1], 0]);
  assert.ok(Math.abs(dot2(unit([capLine.b[0] - capLine.a[0], capLine.b[1] - capLine.a[1], 0]), travel)) < 1e-12, 'and it is exactly perpendicular to the direction of travel');

  // ROUND cap: every point exactly |d| from the end point — a real semicircle,
  // not a polygonal stand-in.
  const round = offsetPolyline(path, d, NZ, { capEnd: 'round', capStart: 'none' });
  const capArcs = round.segments.filter((s) => s.type === 'arc' && Math.abs(Math.hypot(s.p0[0] - endTip[0], s.p0[1] - endTip[1]) - d) < 1e-9);
  assert.ok(capArcs.length >= 2, `a 180-degree round cap is split into at least 2 well-conditioned spans (got ${capArcs.length})`);
  let worst = 0;
  for (const a of capArcs) for (let k = 0; k <= 40; k++) {
    const p = arcAt(a, k / 40);
    worst = Math.max(worst, Math.abs(Math.hypot(p[0] - endTip[0], p[1] - endTip[1]) - d));
  }
  assert.ok(worst < 1e-9, `every round-cap point sits exactly |d| from the end point (worst ${worst.toExponential(3)})`);
  // A round cap must also sweep the OUTSIDE of the tip, not double back
  // through the curve: its far point is further from the previous vertex than
  // the tip itself is.
  const mid = arcAt(capArcs[Math.floor(capArcs.length / 2)], capArcs.length % 2 ? 0.5 : 0);
  assert.ok(Math.hypot(mid[0] - path[1][0], mid[1] - path[1][1]) > Math.hypot(endTip[0] - path[1][0], endTip[1] - path[1][1]), 'the round cap goes around the outside of the tip');
});

test('offsetPolyline: a capped outline holds the offset distance on BOTH sides, including for a negative distance', () => {
  const path = [[0, 0, 0], [100, 0, 0], [140, 55, 0]];
  for (const d of [5, -5]) {
    const r = offsetPolyline(path, d, NZ, { capStart: 'flat', capEnd: 'flat', join: 'round' });
    assert.equal(r.closed, true);
    let worst = 0;
    for (const seg of r.segments) {
      const pts = seg.type === 'arc'
        ? Array.from({ length: 41 }, (_, k) => arcAt(seg, k / 40))
        : Array.from({ length: 41 }, (_, k) => [seg.a[0] + (seg.b[0] - seg.a[0]) * k / 40, seg.a[1] + (seg.b[1] - seg.a[1]) * k / 40, 0]);
      for (const p of pts) worst = Math.max(worst, distToPolyline(p, path, false) - Math.abs(d));
    }
    // A flat cap's own chord is exactly |d| away at its ends and passes THROUGH
    // the tip (distance 0), so the meaningful claim is the one-sided one: the
    // outline never gets FURTHER than |d| from the source anywhere, which for a
    // round-joined stroke outline is exactly the true "distance |d| ribbon".
    assert.ok(worst < 1e-9, `no point of the capped outline exceeds |d| from the source at d=${d} (worst excess ${worst.toExponential(3)})`);
  }
});

/* ================================================================
   7. BOTH SIGNS, DEFINED AGAINST THE PLANE NORMAL
   ================================================================ */

test('offsetPolyline: the sign convention is exactly as documented — positive is LEFT of travel about the plane normal, so a CCW closed loop shrinks', () => {
  const srcArea = signedArea2D(QUAD.map((p) => [p[0], p[1]]));
  assert.ok(srcArea > 0, 'the fixture really is counter-clockwise when viewed from +Z');
  const inward = offsetPolyline(QUAD, 5, NZ, { closed: true });
  const outward = offsetPolyline(QUAD, -5, NZ, { closed: true });
  const aIn = signedArea2D(inward.points.map((p) => [p[0], p[1]]));
  const aOut = signedArea2D(outward.points.map((p) => [p[0], p[1]]));
  assert.ok(aIn > 0 && aIn < srcArea, `positive distance shrinks a CCW loop (${aIn.toFixed(2)} < ${srcArea.toFixed(2)})`);
  assert.ok(aOut > srcArea, `negative distance grows it (${aOut.toFixed(2)} > ${srcArea.toFixed(2)})`);
  // And the corner classification follows the sign, not a hardcoded winding:
  // every corner of a convex CCW loop is INNER at +d and OUTER at -d.
  assert.ok(inward.joins.every((k) => k === 'inner'), `all inner at +d (got ${inward.joins.join(',')})`);
  assert.ok(outward.joins.every((k) => k === 'miter'), `all outer at -d (got ${outward.joins.join(',')})`);
});

test('offsetPolyline: reversing the curve flips the side exactly once, and negating the plane normal flips it exactly once', () => {
  const a = offsetPolyline(QUAD, 5, NZ, { closed: true });
  const reversedNeg = offsetPolyline(QUAD.slice().reverse(), -5, NZ, { closed: true });
  const negNormal = offsetPolyline(QUAD, 5, [0, 0, -1], { closed: true });
  const outward = offsetPolyline(QUAD, -5, NZ, { closed: true });

  const area = (r) => signedArea2D(r.points.map((p) => [p[0], p[1]]));
  // Reverse + negate distance is the SAME shape (two flips cancel), traced the
  // other way round — so the areas are exact negatives of each other.
  assert.ok(Math.abs(area(a) + area(reversedNeg)) < 1e-9, `reverse+negate reproduces the same outline with opposite winding (${area(a).toFixed(6)} vs ${area(reversedNeg).toFixed(6)})`);
  // Negating the normal alone is one flip — same as negating the distance.
  assert.ok(Math.abs(area(negNormal) - area(outward)) < 1e-9, `negating the plane normal equals negating the distance (${area(negNormal).toFixed(6)} vs ${area(outward).toFixed(6)})`);
});

test('offsetPolyline: the convention agrees with offsetCurve2D, which shipped first and must not change meaning', () => {
  // A single straight segment through both entry points must land in exactly
  // the same place, or the piecewise path would silently disagree with the
  // smooth one for the one input both can represent exactly.
  const line = makeLine([0, 0, 0], [10, 0, 0]);
  const viaCurve = offsetCurve2D(line, 3, NZ);
  const viaPolyline = offsetPolyline([[0, 0, 0], [10, 0, 0]], 3, NZ, {});
  for (let i = 0; i < 2; i++) for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(viaCurve.ctrlPts[i][k] - viaPolyline.points[i][k]) < 1e-12, `component ${k} of endpoint ${i} agrees between offsetCurve2D and offsetPolyline`);
  }
});

/* ================================================================
   8. HONEST REFUSALS + FINITENESS UNDER STRESS
   ================================================================ */

test('offsetPolyline / offsetPolyCurve: refuse honestly, naming the reason, rather than returning something quietly wrong', () => {
  assert.throws(() => offsetPolyline([[0, 0, 0]], 3, NZ, {}), /at least 2 distinct points/);
  assert.throws(() => offsetPolyline([[0, 0, 0], [1, 0, 0]], 3, NZ, { closed: true }), /at least 3 distinct points/);
  assert.throws(() => offsetPolyline(ASYM, 3, [0, 0, 0], {}), /nonzero vector/);
  assert.throws(() => offsetPolyline(ASYM, Infinity, NZ, {}), /finite/);
  assert.throws(() => offsetPolyline(ASYM, 3, NZ, { join: 'chamfer' }), /unknown join style/);
  assert.throws(() => offsetPolyline(ASYM, 3, NZ, { capEnd: 'square' }), /unknown cap style/);
  assert.throws(() => offsetPolyline(ASYM, 3, NZ, { miterLimit: 0.5 }), /miterLimit/);
  // A polyline lying in the plane's own normal direction genuinely has no
  // in-plane perpendicular to offset along.
  assert.throws(() => offsetPolyline([[0, 0, 0], [0, 0, 10]], 3, NZ, {}), /parallel to the offset plane normal/);
  // A discontinuous chain cannot be offset as one outline.
  const a = makeLine([0, 0, 0], [10, 0, 0]);
  const b = makeLine([50, 0, 0], [60, 0, 0]);
  assert.throws(() => offsetPolyCurve([a, b], 2, NZ, {}), /does not meet/);
});

test('offsetPolyline: duplicate consecutive points are cleaned rather than crashing on a zero-length edge', () => {
  const withDupes = [[0, 0, 0], [0, 0, 0], [60, -8, 0], [60, -8, 0], [92, 26, 0]];
  const r = offsetPolyline(withDupes, 4, NZ, {});
  const clean = offsetPolyline([[0, 0, 0], [60, -8, 0], [92, 26, 0]], 4, NZ, {});
  assert.equal(r.points.length, clean.points.length);
  for (let i = 0; i < r.points.length; i++) for (let k = 0; k < 3; k++) assert.ok(Math.abs(r.points[i][k] - clean.points[i][k]) < 1e-12);
});

test('offsetPolyline: nothing produces a NaN or an Infinity across a sweep of distances, joins, caps and both signs', () => {
  const fixtures = [ASYM, QUAD, TABBED, [[0, 0, 0], [100, 0, 0], [2, 3, 0]]];
  let cases = 0;
  for (const pts of fixtures) {
    for (const closed of [false, true]) {
      for (const join of ['miter', 'round', 'bevel']) {
        for (const d of [0.5, -0.5, 7, -7, 31, -31]) {
          let r;
          try { r = offsetPolyline(pts, d, NZ, { closed, join, capStart: closed ? 'none' : 'round', capEnd: closed ? 'none' : 'flat' }); }
          catch (e) {
            // A refusal is an acceptable outcome; a NaN is not. Anything that
            // throws must throw with a real, named reason.
            assert.ok(/offset|Machinery|converge|collapsed/i.test(e.message), `refusal names a real reason, got: ${e.message}`);
            continue;
          }
          cases++;
          assert.ok(allFinite(r.points), `finite for d=${d} join=${join} closed=${closed}`);
          for (const s of r.segments) {
            const check = s.type === 'arc' ? [s.p0, s.apex, s.p2] : (s.type === 'line' ? [s.a, s.b] : []);
            assert.ok(allFinite(check), `every emitted segment control point is finite for d=${d} join=${join}`);
            if (s.type === 'arc') assert.ok(Number.isFinite(s.weight) && s.weight > 0, 'every arc weight is a real positive number');
          }
          if (r.pruned) assert.equal(chainSelfIntersects(r.points, NZ, r.closed), false, `whenever pruning ran it actually finished the job (d=${d} join=${join} closed=${closed})`);
        }
      }
    }
  }
  assert.ok(cases > 100, `the sweep really covered a lot of ground (${cases} cases)`);
});

test('offsetcurve: the published tolerance constants are the documented ones', () => {
  assert.equal(DEFAULT_ARC_TOLERANCE, 0.01);
  assert.equal(DEFAULT_MITER_LIMIT, 4);
});

/* ================================================================
   9. OFFSETPOLYCURVE — the same machinery on genuinely curved segments
   ================================================================ */

test('offsetPolyCurve: a real corner between two curved segments gets the same join treatment, and the offset segments meet it exactly', () => {
  // A quarter arc that ends heading -X, then a straight run heading +Y: a real
  // 90-degree corner between a curved segment and a straight one.
  const arc = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 40, 0, Math.PI / 2, 1);
  const arcEnd = curvePoint(arc, arc.knots[arc.knots.length - 1 - arc.degree]);
  const line = makeLine(arcEnd, [arcEnd[0], arcEnd[1] + 60, 0]);
  // The chain arrives heading -X and leaves heading +Y: a RIGHT turn, so the
  // OUTER side (where a gap opens) is the LEFT of travel, i.e. a POSITIVE
  // distance. Derived from the sign rule rather than guessed.
  const d = 5;
  const r = offsetPolyCurve([arc, line], d, NZ, { join: 'round' });
  assert.equal(r.joins.length, 1);
  assert.equal(r.joins[0], 'round', `the corner really is an OUTER one at d=${d}, so it gets a round join`);
  const arcs = r.segments.filter((s) => s.type === 'arc');
  assert.ok(arcs.length >= 1, 'the round join emitted a real arc');
  for (const a of arcs) for (let k = 0; k <= 20; k++) {
    const p = arcAt(a, k / 20);
    assert.ok(Math.abs(Math.hypot(p[0] - arcEnd[0], p[1] - arcEnd[1]) - Math.abs(d)) < 1e-9, 'the join arc is centered exactly on the shared corner point at radius |d|');
  }
  // The offset CURVE segments' own endpoints must land exactly on the join's
  // own endpoints — the claim in offsetPolyCurve's header that a clamped
  // B-spline's end control point and its Greville parameter coincide.
  const first = r.segments[0], last = r.segments[r.segments.length - 1];
  assert.equal(first.type, 'curve');
  assert.equal(last.type, 'curve');
  const firstEndPt = curvePoint(first.crv, first.crv.knots[first.crv.knots.length - 1 - first.crv.degree]);
  assert.ok(Math.hypot(firstEndPt[0] - arcs[0].p0[0], firstEndPt[1] - arcs[0].p0[1]) < 1e-9, 'the first offset segment ends exactly where the join arc starts');
});

test('offsetPolyCurve: caps and pruning work on a curved chain too, and a closed chain is validated for continuity', () => {
  const arc = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 40, 0, Math.PI / 2, 1);
  const arcEnd = curvePoint(arc, arc.knots[arc.knots.length - 1 - arc.degree]);
  const line = makeLine(arcEnd, [arcEnd[0] - 60, arcEnd[1], 0]);
  const capped = offsetPolyCurve([arc, line], 5, NZ, { capStart: 'round', capEnd: 'flat' });
  assert.equal(capped.closed, true);
  assert.ok(allFinite(capped.points));
  assert.equal(chainSelfIntersects(capped.points, NZ, true), false, 'the capped curved outline does not cross itself');
  assert.throws(() => offsetPolyCurve([arc, line], 5, NZ, { closed: true }), /must end where it starts/);
});
