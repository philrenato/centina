// INTERIOR-CORNER MITER — the long-standing backlog
// note, logged (not started) alongside the closed-rail seam fix earlier the
// "Pipe (and especially MultiPipe/T-Splines-style multipipe)
// still needs real corner awareness — smoothness/mitering/smoothing at a
// rail's own interior corners, even for an ordinary NURBS Pipe, not just the
// closed-rail seam weld already shipped... this is pretty much just a
// 'polypipe.'"
//
// ROOT CAUSE, confirmed directly against kernel/basis.mjs's own FindSpan
// (P&T A2.1) before writing any fix: for a degree<=1 rail,
// `curvePointAndTangent` evaluated at an INTERIOR control point's own
// Greville parameter is a ONE-SIDED derivative — FindSpan resolves an
// interior knot u to the span [u, next), i.e. the OUTGOING edge only. Every
// interior corner therefore got exactly one ring, oriented perpendicular to
// just ONE of its two adjacent edges — a real crease/skew, DIFFERENT from
// the closed-rail seam bug (that was two independently-oriented frames on
// the SAME repeated vertex; this is one frame, wrongly oriented, at every
// OTHER vertex).
//
// THE FIX (kernel/sweep.mjs, buildParallelTransportFrames, inside the main
// per-step loop): generalizes the seam weld's own miter technique — bisector
// tangent (normalize(tOut+tIn)) plus one more parallel-transport step — to
// every interior control-point station of a degree<=1 rail, chaining through
// `prevNormal` naturally rather than as a separate mechanism. `tIn` needs no
// extra curve evaluation: it's simply the previous step's own already-
// computed raw tangent (the segment between two adjacent control points is,
// for a degree<=1 rail, both the outgoing edge of one and the incoming edge
// of the next). Both of the seam fix's own named degenerate-corner
// fallbacks (near-180 fold-back -> outgoing tangent; near-parallel
// projection -> anyPerpendicular) carry over verbatim.
//
// HONEST LIMITATIONS, stated plainly, not silently fixed:
//  - This does NOT true-miter the CROSS-SECTION (no 1/cos(theta/2) scaling
//    to keep the outer edge from necking down) — the tube necks by
//    cos(theta/2) perpendicular to each edge at a sharp corner (~29%
//    thinner at 90 degrees), the SAME property the already-shipped seam fix
//    has. Coherent v1 scope, matching the seam fix; real per-frame section
//    SCALING (breaking sweep1Rigid's "localize once, transform rigidly"
//    structure) is v2, not attempted here.
//  - A local self-intersection near a corner whose radius exceeds the
//    shorter adjacent segment length is a real, geometrically expected
//    consequence, NOT gated against — matching this kernel's existing
//    posture elsewhere (see test below).
//  - A mixed-degree rail (Line+Arc PolyCurve degree-elevated via
//    joinCurvesC0 to one degree-2 curve) never reaches this per-control-
//    point path at all — it goes down sweep1RigidResampled's dense-sample
//    path, where a C0 corner shows up as an abrupt frame rotation between
//    dense samples. A genuinely separate, harder gap — NOT attempted here,
//    named honestly.

import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle } from '../kernel/primitives.mjs';
import { sweep1Rigid, buildParallelTransportFrames, railFrameOriginsExact } from '../kernel/sweep.mjs';
import { isFiniteNet } from '../kernel/surface.mjs';
import { isCurveClosed, curvePointAndTangent } from '../kernel/curve.mjs';

function openPolylineRail(points) {
  const ctrlPts = points.map(([x, y, z]) => [x, y, z, 1]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}
function closedPolylineRail(points) {
  const ctrlPts = points.map(([x, y, z]) => [x, y, z, 1]);
  ctrlPts.push(ctrlPts[0]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}
// A small, deliberately ASYMMETRIC (non-circular, non-square-square)
// quadrilateral profile, degree-1 clamped closed curve — Pipe itself is
// circle-only (Pipe synthesizes a circle via makeCircle), but
// sweep1Rigid is SHARED with single-profile Sweep1, which can carry an
// arbitrary profile; this proves the miter fix works through that same
// shared function for a non-circular section too.
function makeAsymmetricQuadProfile() {
  return closedPolylineRail([[3, 0, 0], [0, 2, 0], [-4, 0, 0], [0, -1, 0]]);
}
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const isOrthonormal = (f, assertMsg) => {
  assert.ok(Number.isFinite(f.xAxis[0]) && Number.isFinite(f.yAxis[0]) && Number.isFinite(f.zAxis[0]), `${assertMsg}: finite`);
  assert.ok(Math.abs(dot3(f.xAxis, f.yAxis)) < 1e-6, `${assertMsg}: xAxis ⊥ yAxis`);
  assert.ok(Math.abs(dot3(f.xAxis, f.zAxis)) < 1e-6, `${assertMsg}: xAxis ⊥ zAxis`);
  assert.ok(Math.abs(dot3(f.yAxis, f.zAxis)) < 1e-6, `${assertMsg}: yAxis ⊥ zAxis`);
  assert.ok(Math.abs(Math.hypot(...f.xAxis) - 1) < 1e-6, `${assertMsg}: xAxis unit`);
  assert.ok(Math.abs(Math.hypot(...f.zAxis) - 1) < 1e-6, `${assertMsg}: zAxis unit`);
};

test('COLLINEAR interior vertex: the miter fix is a BYTE-IDENTICAL no-op — the single most important regression guard, proving it changes nothing where there is nothing to fix', () => {
  // Three collinear points: the "corner" at index 1 doesn't actually turn.
  const rail = openPolylineRail([[0, 0, 0], [100, 0, 0], [220, 0, 0]]);
  const frames = buildParallelTransportFrames(rail);
  const cornerFrame = frames[1];
  const directTangent = curvePointAndTangent(rail, rail.knots[2]).tangent; // exactly what the OLD (un-mitered) code computed at this station
  assert.equal(cornerFrame.zAxis[0], directTangent[0], 'byte-identical tangent X (no-op, not merely close)');
  assert.equal(cornerFrame.zAxis[1], directTangent[1], 'byte-identical tangent Y (no-op, not merely close)');
  assert.equal(cornerFrame.zAxis[2], directTangent[2], 'byte-identical tangent Z (no-op, not merely close)');
});

test('a GENUINE (non-collinear) interior corner IS mitered — the ring reorients to bisect both adjacent edges, not just one', () => {
  // A 90-degree "L" rail: corner at index 1.
  const rail = openPolylineRail([[0, 0, 0], [100, 0, 0], [100, 100, 0]]);
  const frames = buildParallelTransportFrames(rail);
  const cornerFrame = frames[1];
  const oldOneSidedTangent = curvePointAndTangent(rail, rail.knots[2]).tangent; // what the OLD un-mitered code would have used (the outgoing edge only)
  // The fix changes the tangent here — it is now the bisector, not the raw one-sided outgoing tangent.
  const diff = Math.hypot(...cornerFrame.zAxis.map((v, i) => v - oldOneSidedTangent[i]));
  assert.ok(diff > 0.05, `corner frame tangent genuinely differs from the old one-sided outgoing tangent (diff ${diff.toFixed(6)}) — proves the miter actually fired`);
  // The new tangent must be the TRUE bisector of the outgoing (+X) and incoming (+Y) edges: (1,1,0)/sqrt(2).
  const expected = [Math.SQRT1_2, Math.SQRT1_2, 0];
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(cornerFrame.zAxis[i] - expected[i]) < 1e-9, `bisector tangent component ${i} matches the true 45-degree bisector exactly`);
  isOrthonormal(cornerFrame, 'mitered corner frame');
});

test('degenerate fallback: a near-180-degree fold-back interior corner stays finite/orthonormal, never NaN (a physical self-intersection there is ACCEPTED, matching real Rhino, not gated against)', () => {
  const rail = openPolylineRail([[0, 0, 0], [100, 0.001, 0], [0.002, 0.003, 0]]); // spikes almost exactly back on itself at index 1
  const frames = buildParallelTransportFrames(rail);
  isOrthonormal(frames[1], 'near-180 fold-back corner frame');
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'the swept tube stays fully finite even at this extreme fold-back corner');
});

test('radius larger than the shorter adjacent segment: NOTED, not gated — a local self-intersection near the corner is geometrically expected, matching this kernel\'s existing posture elsewhere', () => {
  const rail = openPolylineRail([[0, 0, 0], [8, 0, 0], [8, 8, 0]]); // an 8mm segment
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 20); // a 20mm-radius pipe, far larger than the 8mm segment — WILL self-intersect near the corner, and that's fine
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'still a finite, well-formed (if self-intersecting) control net — no NaN/crash, no gate refusing the input');
});

test('CLOSED pentagon rail: the existing seam weld AND 4 interior miters coexist correctly in the same tube', () => {
  const PENTAGON = [[-80, 40, 0], [0, 90, 0], [90, 30, 0], [50, -70, 0], [-70, -60, 0]]; // same fixture shape as the closed-rail seam fix's own repro
  const rail = closedPolylineRail(PENTAGON);
  assert.equal(isCurveClosed(rail), true);
  assert.equal(railFrameOriginsExact(rail), true);
  const frames = buildParallelTransportFrames(rail);
  assert.equal(frames.length, PENTAGON.length + 1);

  // The seam weld (index 0 / last) is untouched by this file's own fix —
  // still welded, still one shared frame object.
  assert.ok(frames[0] === frames[frames.length - 1], 'seam weld from the earlier round still holds: frames[0] === frames[last]');
  isOrthonormal(frames[0], 'seam frame');

  // The 4 REMAINING interior vertices (indices 1,2,3,4 of a 5-vertex closed
  // pentagon) are each a genuine corner now mitered by this file's fix —
  // each one's tangent bisects its own two adjacent edges, and each stays
  // orthonormal.
  const n = PENTAGON.length;
  for (let i = 1; i < n; i++) {
    const tOut = i < n - 1
      ? [PENTAGON[i + 1][0] - PENTAGON[i][0], PENTAGON[i + 1][1] - PENTAGON[i][1], PENTAGON[i + 1][2] - PENTAGON[i][2]]
      : [PENTAGON[0][0] - PENTAGON[i][0], PENTAGON[0][1] - PENTAGON[i][1], PENTAGON[0][2] - PENTAGON[i][2]];
    const tIn = [PENTAGON[i][0] - PENTAGON[i - 1][0], PENTAGON[i][1] - PENTAGON[i - 1][1], PENTAGON[i][2] - PENTAGON[i - 1][2]];
    const norm = (v) => { const l = Math.hypot(...v); return v.map((c) => c / l); };
    const tOutN = norm(tOut), tInN = norm(tIn);
    const sum = [tOutN[0] + tInN[0], tOutN[1] + tInN[1], tOutN[2] + tInN[2]];
    const expected = norm(sum);
    const f = frames[i];
    isOrthonormal(f, `pentagon interior corner ${i}`);
    for (let k = 0; k < 3; k++) assert.ok(Math.abs(f.zAxis[k] - expected[k]) < 1e-9, `pentagon corner ${i} tangent bisects its own two edges exactly`);
  }

  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'the whole tube — seam weld + 4 interior miters together — is a single valid, finite surface');
  const firstRing = srf.ctrlNet.map((row) => row[0]);
  const lastRing = srf.ctrlNet.map((row) => row[row.length - 1]);
  let maxRingDiff = 0;
  for (let i = 0; i < firstRing.length; i++) maxRingDiff = Math.max(maxRingDiff, Math.hypot(...firstRing[i].slice(0, 3).map((v, k) => v - lastRing[i][k])));
  assert.ok(maxRingDiff < 1e-9, 'the seam weld still produces a zero-gap closed tube with the interior miters also applied');
});

test('NON-CIRCULAR profile through a corner (sweep1Rigid is SHARED with single-profile Sweep1, which allows an arbitrary section)', () => {
  const rail = openPolylineRail([[0, 0, 0], [100, 0, 0], [100, 100, 0]]);
  const quadProfile = makeAsymmetricQuadProfile();
  const srf = sweep1Rigid(rail, quadProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'an asymmetric non-circular profile still sweeps to a valid, finite surface through the mitered corner');
  // The corner ring (V index 1, the middle rail control point) should sit in
  // the mitered frame's plane, not the old one-sided (outgoing-only) plane —
  // spot-check by confirming the ring's own points are NOT all coplanar with
  // a plane perpendicular to the pure +X outgoing edge (the OLD behavior).
  const cornerRing = srf.ctrlNet.map((row) => row[1]);
  const allXConstant = cornerRing.every((p) => Math.abs(p[0] - 100) < 1e-6);
  assert.ok(!allXConstant, 'the corner ring is no longer a plane of constant X (the old one-sided/outgoing-only orientation) — it now bisects both adjacent edges');
});

test('an ORDINARY straight rail (no corners at all) is completely unaffected — full regression guard', () => {
  const rail = openPolylineRail([[0, 0, 0], [50, 0, 0], [100, 0, 0], [150, 0, 0]]);
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  for (let i = 1; i < rail.ctrlPts.length - 1; i++) {
    const direct = curvePointAndTangent(rail, rail.knots[i + 1]).tangent;
    const frameTangent = buildParallelTransportFrames(rail)[i].zAxis;
    for (let k = 0; k < 3; k++) assert.equal(frameTangent[k], direct[k], `straight-rail station ${i} stays byte-identical to the un-mitered tangent`);
  }
});
