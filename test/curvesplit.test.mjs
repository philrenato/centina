// CURVE SPLIT AND TRIM — the gate for kernel/curvesplit.mjs.
//
// Ground truth is ANALYTIC wherever the fixture allows it. Two axis-aligned
// lines crossing at the origin split at exactly their midpoints; a circle of
// radius r cut by a line through its centre is cut at exactly (+/-r, 0); a
// line offset in z from a coplanar crossing misses by exactly that offset.
// Every one of those numbers comes from the geometry, never from the function
// under test.
//
// The cases that matter most here are the REFUSALS. A split that quietly
// produces nothing is the failure mode this whole feature exists to avoid, so
// the non-meeting, out-of-tolerance and end-crossing cases assert on the text
// of the reason as well as on the absence of pieces.

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  curveCurveEvents, splitCurveAtParams, splitCurveNetwork,
  trimCurveAtParam, sectionAtParam, crossingParamsOn, coplanarNormal,
  CURVE_SPLIT_DEFAULT_TOLERANCE,
} from '../kernel/curvesplit.mjs';
import { makeLine, makeCircle } from '../kernel/primitives.mjs';
import { curvePoint, curveDomain, curveLength } from '../kernel/curve.mjs';

const X = [1, 0, 0], Y = [0, 1, 0];

function endpoints(crv) {
  const [a, b] = curveDomain(crv);
  return [curvePoint(crv, a), curvePoint(crv, b)];
}
function near(a, b, tol = 1e-9) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tol;
}
// Total length is CONSERVED by a split — nothing is created or destroyed. The
// strongest single check available on a set of pieces, and it is independent
// of how the pieces are ordered or where the cuts landed.
function totalLength(pieces) {
  return pieces.reduce((s, p) => s + curveLength(p, ...curveDomain(p), 1e-9), 0);
}

test('two crossing lines split into four pieces, cut at the analytic crossing', () => {
  const a = makeLine([-10, 0, 0], [10, 0, 0]);
  const b = makeLine([0, -10, 0], [0, 10, 0]);
  const r = splitCurveNetwork([a, b]);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.results.length, 2, 'both lines must be cut, not just one');
  const all = r.results.flatMap((x) => x.pieces);
  assert.equal(all.length, 4, 'two crossing lines make four pieces');
  for (const res of r.results) {
    const [p0, p1] = res.pieces;
    // The shared cut point is the origin, exactly.
    assert.ok(near(endpoints(p0)[1], [0, 0, 0], 1e-9), 'first piece must end at the crossing');
    assert.ok(near(endpoints(p1)[0], [0, 0, 0], 1e-9), 'second piece must start at the crossing');
  }
  assert.ok(Math.abs(totalLength(all) - 40) < 1e-6, `length must be conserved: got ${totalLength(all)}`);
});

test('N curves split MUTUALLY, every curve against every other', () => {
  // A tic-tac-toe grid: two horizontals, two verticals. Each of the four is
  // crossed TWICE, so each becomes three pieces — 12 in total. A pairwise
  // "first against second" scheme would produce 8 and is what this rules out.
  const h1 = makeLine([-10, -3, 0], [10, -3, 0]);
  const h2 = makeLine([-10, 3, 0], [10, 3, 0]);
  const v1 = makeLine([-3, -10, 0], [-3, 10, 0]);
  const v2 = makeLine([3, -10, 0], [3, 10, 0]);
  const r = splitCurveNetwork([h1, h2, v1, v2]);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.results.length, 4);
  for (const res of r.results) assert.equal(res.pieces.length, 3, 'each line is crossed twice and must yield three pieces');
  assert.equal(r.results.flatMap((x) => x.pieces).length, 12);
  assert.ok(Math.abs(totalLength(r.results.flatMap((x) => x.pieces)) - 80) < 1e-6);
  // The two parallel horizontals never meet, and that pair is recorded as
  // producing nothing rather than being silently absent.
  const parallelPair = r.pairs.find((p) => p.i === 0 && p.j === 1);
  assert.equal(parallelPair.count, 0);
});

test('a closed curve splits into arcs, and the seam is not a cut', () => {
  const circle = makeCircle([0, 0, 0], X, Y, 5);
  const v1 = makeLine([-3, -8, 0], [-3, 8, 0]);
  const v2 = makeLine([3, -8, 0], [3, 8, 0]);
  const r = splitCurveNetwork([circle, v1, v2]);
  assert.equal(r.ok, true, r.reason);
  const circleRes = r.results.find((x) => x.index === 0);
  assert.equal(circleRes.cuts.length, 4, 'two chords cross a circle at four places');
  // FOUR cuts on a CLOSED curve make FOUR arcs, not five: the seam is an
  // ordinary point and the last arc wraps through it.
  assert.equal(circleRes.pieces.length, 4, 'a closed curve cut four times yields four arcs, not five');
  const circumference = 2 * Math.PI * 5;
  assert.ok(Math.abs(totalLength(circleRes.pieces) - circumference) < 1e-5,
    `the four arcs must add back up to the circle: got ${totalLength(circleRes.pieces)} vs ${circumference}`);
  // Each chord is cut twice by the circle, so each becomes three pieces.
  for (const res of r.results.filter((x) => x.index > 0)) assert.equal(res.pieces.length, 3);
});

test('a line through a circle centre is cut at exactly +/- the radius', () => {
  const circle = makeCircle([0, 0, 0], X, Y, 5);
  const line = makeLine([-9, 0, 0], [9, 0, 0]);
  const r = curveCurveEvents(circle, line);
  assert.equal(r.events.length, 2);
  const xs = r.events.map((e) => e.pointA[0]).sort((a, b) => a - b);
  assert.ok(Math.abs(xs[0] + 5) < 1e-7, `expected -5, got ${xs[0]}`);
  assert.ok(Math.abs(xs[1] - 5) < 1e-7, `expected +5, got ${xs[1]}`);
  for (const e of r.events) assert.equal(e.kind, 'true');
});

test('a TANGENTIAL touch is found, not refused', () => {
  // A line grazing the top of a circle. The transversal intersector in
  // curvecurve.mjs correctly refuses this (singular Jacobian); the distance
  // search is what recovers it, and the touch point is analytically (0, 5).
  const circle = makeCircle([0, 0, 0], X, Y, 5);
  const tangent = makeLine([-8, 5, 0], [8, 5, 0]);
  const r = curveCurveEvents(circle, tangent);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.events.length, 1, 'a tangent line touches a circle exactly once');
  const e = r.events[0];
  assert.equal(e.kind, 'true');
  assert.ok(near(e.pointA, [0, 5, 0], 1e-6), `touch must be at (0,5,0), got ${e.pointA}`);
  // And it really does cut the line in two.
  const s = splitCurveAtParams(tangent, [e.uB]);
  assert.equal(s.pieces.length, 2);
});

test('curves that do NOT meet refuse BY NAME, quoting the gap and the tolerance', () => {
  const a = makeLine([-10, 0, 0], [10, 0, 0]);
  const b = makeLine([-10, 4, 0], [10, 4, 0]); // parallel, 4 apart
  const r = curveCurveEvents(a, b);
  assert.equal(r.events.length, 0);
  assert.ok(/do not meet/.test(r.reason), `reason must say they do not meet: ${r.reason}`);
  assert.ok(/4\.000/.test(r.reason), `reason must quote the measured 4.000 gap: ${r.reason}`);
  assert.ok(String(r.reason).includes(String(CURVE_SPLIT_DEFAULT_TOLERANCE)),
    `reason must name the tolerance it used: ${r.reason}`);
  assert.ok(Math.abs(r.nearest - 4) < 1e-9);

  const net = splitCurveNetwork([a, b]);
  assert.equal(net.ok, false);
  assert.ok(/closest any two come is 4\.000/.test(net.reason), net.reason);
  assert.equal(net.results.length, 0, 'a refusal must produce no geometry at all');
});

// ---------------------------------------------------------------------------
// NON-COPLANAR — the rule, and both sides of it
// ---------------------------------------------------------------------------

test('non-coplanar curves WITHIN tolerance are a TRUE intersection', () => {
  const a = makeLine([-10, 0, 0], [10, 0, 0]);
  const b = makeLine([0, -10, 0.05], [0, 10, 0.05]); // misses by 0.05
  // Genuinely out of plane by the PLANARITY test's own relative measure
  // (0.05 against a ~28-unit diagonal is well outside diagonal * 1e-4), so
  // this is the skew case and not a coplanar one wearing a small number.
  assert.equal(coplanarNormal(a, b), null, 'these are genuinely not coplanar');
  const r = curveCurveEvents(a, b, { tolerance: 0.1 });
  assert.equal(r.events.length, 1);
  assert.equal(r.events[0].kind, 'true', 'inside the tolerance it is a real crossing, not an inferred one');
  assert.ok(r.events[0].gap <= 0.1, `gap ${r.events[0].gap} must be inside the tolerance`);
  assert.equal(r.apparentCount, 0);
  // Looked at straight down Z the miss is EXACTLY the 0.05 offset — the
  // analytic number. Under the auto-chosen best-fit direction it is slightly
  // less, because that direction is tilted to fit both curves rather than
  // aligned with the offset, which is the honest behaviour and the reason a
  // caller with a real viewport should pass its own direction.
  const downZ = curveCurveEvents(a, b, { tolerance: 0.1, direction: [0, 0, 1] });
  assert.ok(Math.abs(downZ.events[0].gap - 0.05) < 1e-9, `expected 0.05, got ${downZ.events[0].gap}`);
});

test('non-coplanar curves OUTSIDE tolerance are APPARENT, and inference is what admits them', () => {
  const a = makeLine([-10, 0, 0], [10, 0, 0]);
  const b = makeLine([0, -10, 2], [0, 10, 2]); // misses by 2
  // WITHOUT inference: refused, and the refusal names the real gap.
  const strict = curveCurveEvents(a, b, { tolerance: 0.001, infer: false, direction: [0, 0, 1] });
  assert.equal(strict.events.length, 0);
  assert.ok(/closest approach is 2\.000/.test(strict.reason), strict.reason);
  // WITH inference, looked at along Z: they cross when seen from above.
  const inferred = curveCurveEvents(a, b, { tolerance: 0.001, infer: true, direction: [0, 0, 1] });
  assert.equal(inferred.events.length, 1);
  assert.equal(inferred.events[0].kind, 'apparent', 'this must be reported as inferred, never as a real meeting');
  assert.equal(inferred.apparentCount, 1);
  assert.ok(Math.abs(inferred.events[0].gap - 2) < 1e-9);
  // Each curve is cut at its OWN nearest parameter to the apparent crossing.
  assert.ok(near(inferred.events[0].pointA, [0, 0, 0], 1e-7));
  assert.ok(near(inferred.events[0].pointB, [0, 0, 2], 1e-7));
});

test('an apparent intersection is VIEW-DEPENDENT, and the direction used is reported', () => {
  const a = makeLine([-10, 0, 0], [10, 0, 0]);
  const b = makeLine([0, -10, 2], [0, 10, 2]);
  // Looked at along X the two lines do NOT overlap in projection at all.
  const alongX = curveCurveEvents(a, b, { tolerance: 0.001, infer: true, direction: [1, 0, 0] });
  assert.equal(alongX.events.length, 0, 'seen edge-on down X these do not appear to cross');
  const alongZ = curveCurveEvents(a, b, { tolerance: 0.001, infer: true, direction: [0, 0, 1] });
  assert.equal(alongZ.events.length, 1);
  assert.deepEqual(alongZ.direction, [0, 0, 1]);
  assert.equal(alongZ.directionSource, 'given');
});

test('a coplanar pair reports coplanar:true and uses the plane normal', () => {
  const a = makeLine([-10, 0, 3], [10, 0, 3]);
  const b = makeLine([0, -10, 3], [0, 10, 3]);
  const r = curveCurveEvents(a, b);
  assert.equal(r.coplanar, true);
  assert.equal(r.directionSource, 'plane normal');
  assert.equal(r.events.length, 1);
  assert.equal(r.events[0].kind, 'true');
});

// ---------------------------------------------------------------------------
// TRIM
// ---------------------------------------------------------------------------

test('trimming a MIDDLE section leaves the two ends, and does NOT cut them further', () => {
  // A horizontal crossed by THREE verticals, so the remainder on the right
  // still has a crossing in it. Rhino leaves that crossing uncut — trim
  // removes the clicked section and nothing else.
  const line = makeLine([-10, 0, 0], [10, 0, 0]);
  const cutters = [
    makeLine([-5, -5, 0], [-5, 5, 0]),
    makeLine([0, -5, 0], [0, 5, 0]),
    makeLine([5, -5, 0], [5, 5, 0]),
  ];
  const { params } = crossingParamsOn(line, cutters);
  assert.equal(params.length, 3);
  // Click between the first and second cutter, at x = -2.5 -> u = 0.375.
  const r = trimCurveAtParam(line, params, 0.375);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.pieces.length, 2, 'a middle section removed leaves exactly two runs');
  const left = endpoints(r.pieces[0]), right = endpoints(r.pieces[1]);
  assert.ok(near(left[0], [-10, 0, 0], 1e-9) && near(left[1], [-5, 0, 0], 1e-9));
  // THE POINT OF THIS TEST: the right-hand run reaches all the way to +10,
  // straight through the crossing at x=5, as ONE curve.
  assert.ok(near(right[0], [0, 0, 0], 1e-9) && near(right[1], [10, 0, 0], 1e-9),
    'the remainder must stay whole through the crossings it passes, not be shredded');
  assert.ok(Math.abs(totalLength(r.pieces) - 15) < 1e-9, 'exactly the clicked 5 units are gone');
});

test('trimming an END section leaves ONE piece', () => {
  const line = makeLine([-10, 0, 0], [10, 0, 0]);
  const cutters = [makeLine([-5, -5, 0], [-5, 5, 0]), makeLine([5, -5, 0], [5, 5, 0])];
  const { params } = crossingParamsOn(line, cutters);
  const r = trimCurveAtParam(line, params, 0.05); // x = -9, the left end section
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.pieces.length, 1);
  const [p0, p1] = endpoints(r.pieces[0]);
  assert.ok(near(p0, [-5, 0, 0], 1e-9) && near(p1, [10, 0, 0], 1e-9));
  assert.ok(Math.abs(totalLength(r.pieces) - 15) < 1e-9);
});

test('trimming an arc off a closed circle leaves ONE open curve through the seam', () => {
  const circle = makeCircle([0, 0, 0], X, Y, 5);
  const cutters = [makeLine([-3, -8, 0], [-3, 8, 0]), makeLine([3, -8, 0], [3, 8, 0])];
  const { params } = crossingParamsOn(circle, cutters);
  assert.equal(params.length, 4);
  const dom = curveDomain(circle);
  // Click the arc crossing the seam (u = 0 is at angle 0, i.e. (5,0,0)).
  const r = trimCurveAtParam(circle, params, dom[0] + 1e-6);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.pieces.length, 1, 'the remainder of a circle is one curve, not three arcs');
  const kept = totalLength(r.pieces);
  // The removed arc is the one containing the seam at (5,0,0): it runs from
  // the lower to the upper crossing of the x=+3 chord, subtending 2*acos(3/5).
  const removedAngle = 2 * Math.acos(3 / 5);
  assert.ok(Math.abs(kept - (2 * Math.PI - removedAngle) * 5) < 1e-4,
    `kept arc length ${kept} does not match the analytic remainder`);
});

test('a trim with nothing crossing the curve refuses by name', () => {
  const line = makeLine([-10, 0, 0], [10, 0, 0]);
  const r = trimCurveAtParam(line, [], 0.5);
  assert.equal(r.ok, false);
  assert.ok(/nothing crosses this curve/.test(r.reason), r.reason);
  assert.equal(r.pieces.length, 0);
});

test('a crossing AT a curve end is not a cut, and is refused by name', () => {
  // A vertical standing exactly on the horizontal's right endpoint. There is
  // nothing on the far side of it, so the HORIZONTAL cannot be cut — while the
  // vertical, crossed at its own middle, genuinely can be. A mutual split has
  // to get both halves of that right.
  const line = makeLine([-10, 0, 0], [10, 0, 0]);
  const atEnd = makeLine([10, -5, 0], [10, 5, 0]);
  const r = curveCurveEvents(line, atEnd);
  assert.equal(r.events.length, 1, 'the crossing itself is real and is reported');
  const s = splitCurveAtParams(line, [r.events[0].uA]);
  assert.equal(s.ok, false);
  assert.ok(/no cut parameter falls strictly inside/.test(s.reason), s.reason);
  const net = splitCurveNetwork([line, atEnd]);
  assert.equal(net.ok, true, net.reason);
  assert.equal(net.results.length, 1, 'only the vertical is cut');
  assert.equal(net.results[0].index, 1);
  assert.equal(net.results[0].pieces.length, 2);

  // TWO lines meeting end to end — an L. Every crossing is an end on BOTH
  // curves, so nothing at all can be cut, and that refuses by name rather
  // than reporting a successful split of zero curves.
  const armA = makeLine([0, 0, 0], [10, 0, 0]);
  const armB = makeLine([0, 0, 0], [0, 10, 0]);
  const corner = splitCurveNetwork([armA, armB]);
  assert.equal(corner.ok, false);
  assert.ok(/lands on a curve end/.test(corner.reason), corner.reason);
});

test('sectionAtParam names the section a click fell in, wrap included', () => {
  const circle = makeCircle([0, 0, 0], X, Y, 5);
  const cutters = [makeLine([-3, -8, 0], [-3, 8, 0]), makeLine([3, -8, 0], [3, 8, 0])];
  const { params } = crossingParamsOn(circle, cutters);
  const dom = curveDomain(circle);
  const wrapped = sectionAtParam(circle, params, dom[0] + 1e-9);
  assert.equal(wrapped.wraps, true, 'a click just after the seam is in the wrap section');
  assert.equal(wrapped.count, 4);
  const interior = sectionAtParam(circle, params, (params[0] + params[1]) / 2);
  assert.equal(interior.wraps, false);
});

test('two curves lying ON TOP of each other refuse as an OVERLAP, not as a thousand touches', () => {
  // Every local minimum of the distance between coincident curves is zero, so
  // the tangency search would otherwise report one "intersection" per grid
  // seed and the split would shatter the curve.
  const a = makeLine([-10, 0, 0], [10, 0, 0]);
  const b = makeLine([-6, 0, 0], [6, 0, 0]);
  const r = curveCurveEvents(a, b);
  assert.equal(r.ok, false);
  assert.equal(r.overlapping, true);
  assert.equal(r.events.length, 0, 'an overlap yields no cut points at all');
  assert.ok(/lie on top of each other/.test(r.reason), r.reason);

  const net = splitCurveNetwork([a, b]);
  assert.equal(net.ok, false);
  assert.ok(/on top of each other/.test(net.reason), net.reason);
  assert.equal(net.results.length, 0);
});

test('an overlapping pair inside a larger set does not shatter the others', () => {
  // The two coincident horizontals must contribute NOTHING, while the vertical
  // still cuts and is cut normally.
  const a = makeLine([-10, 0, 0], [10, 0, 0]);
  const b = makeLine([-6, 0, 0], [6, 0, 0]);   // lies on a
  const v = makeLine([0, -10, 0], [0, 10, 0]);
  const net = splitCurveNetwork([a, b, v]);
  assert.equal(net.ok, true, net.reason);
  const overlapPair = net.pairs.find((p) => p.i === 0 && p.j === 1);
  assert.equal(overlapPair.overlapping, true);
  assert.equal(overlapPair.count, 0);
  for (const res of net.results) {
    assert.ok(res.pieces.length <= 2, `no curve may shatter: got ${res.pieces.length} pieces`);
  }
  // a and b are each crossed once by v; v is crossed by both at the same place.
  assert.equal(net.results.length, 3);
});
