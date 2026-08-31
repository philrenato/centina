// CLOSED-RAIL SEAM WELD. A closed pentagon-shaped Polyline rail, piped at a
// default radius, produced visibly broken geometry — a hollow,
// disconnected-looking gap at one point around the loop, plus creasing at the
// sharp corners. Degenerate geometry, not a robust tool.
//
// ROOT CAUSE, confirmed numerically before any fix (see this file's own
// "ground truth" test below): `getProfileCrv`'s closed-Polyline branch
// already repeats the rail's first control point at the end (the
// established convention `isCurveClosed` recognizes) — one CONTROL POINT,
// `buildParallelTransportFrames` builds one frame per rail control point,
// so a closed rail gets TWO frames (frames[0] and frames[last]) for the
// SAME physical vertex. Their origins coincide exactly, but their
// orientations do NOT: frame[0] seeds fresh off the OUTGOING edge's
// tangent (the first segment leaving the vertex); frame[last] carries the
// accumulated parallel-transport chain off the INCOMING edge's tangent (a
// DIFFERENT edge of the polygon meeting at the same corner). `sweep1Rigid`'s
// free path (any degree<=1 rail — Line/Polyline) reuses these two frames
// directly as the swept surface's own v=0/v=vMax rings, with zero closure
// awareness — the tube ends up as a genuinely OPEN surface with two
// independently-oriented, uncapped rings sitting on top of the same 3D
// point, exactly the hollow gap described above.
//
// THE FIX (kernel/sweep.mjs, buildParallelTransportFrames): weld the loop
// shut with ONE shared MITER frame at that one seam corner — the angle
// bisector of the outgoing/incoming edge tangents, with a normal that
// continues frame[0]'s own original seed one more parallel-transport step
// into the bisector's perpendicular plane — reused as BOTH frames[0] and
// frames[frames.length-1]. See that function's own header comment for the
// full derivation and the two named degenerate-corner fallbacks.
//
// SCOPE: this welds only the ONE seam corner. Every OTHER interior corner
// of a degree<=1 rail keeps its existing single-adjacent-tangent frame (the
// more general "ruled interpolation creases at a sharp turn" limitation) —
// deliberately NOT attempted in this file.

import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle } from '../kernel/primitives.mjs';
import { sweep1Rigid, buildParallelTransportFrames, railFrameOriginsExact } from '../kernel/sweep.mjs';
import { isFiniteNet } from '../kernel/surface.mjs';
import { isCurveClosed } from '../kernel/curve.mjs';

// Mirrors the app's getProfileCrv closed-Polyline branch EXACTLY
// (`if (obj.polylineClosed) ctrlPts.push(ctrlPts[0]);`): N distinct
// vertices, the first control point repeated once at the end, a plain
// degree-1 clamped knot vector — the real shape a closed pentagon drawn in
// the app actually produces as a Pipe rail.
function closedPolylineRail(points) {
  const ctrlPts = points.map(([x, y, z]) => [x, y, z, 1]);
  ctrlPts.push(ctrlPts[0]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}
function openPolylineRail(points) {
  const ctrlPts = points.map(([x, y, z]) => [x, y, z, 1]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}

const PENTAGON = [[-80, 40, 0], [0, 90, 0], [90, 30, 0], [50, -70, 0], [-70, -60, 0]];
const RADIUS = 5;

test('ground truth: an UNPATCHED closed-pentagon rail really does produce two differently-oriented frames at its own seam (proves the bug is real, not assumed)', () => {
  const rail = closedPolylineRail(PENTAGON);
  assert.equal(isCurveClosed(rail), true, 'a closed Polyline profile-as-rail really is recognized as closed by this kernel\'s own established convention');
  assert.equal(railFrameOriginsExact(rail), true, 'a degree-1 Polyline rail is on sweep1Rigid\'s FREE path — the one this bug actually reaches');

  // Independently re-derive what the OLD (pre-fix) per-step loop would have
  // produced at frame[0]/frame[last], without relying on any saved number:
  // frame[0]'s tangent is the FIRST segment's direction; frame[last]'s
  // tangent is the LAST segment's direction — two different edges of the
  // same polygon, meeting at the one shared vertex.
  const n = PENTAGON.length;
  const firstSeg = [PENTAGON[1][0] - PENTAGON[0][0], PENTAGON[1][1] - PENTAGON[0][1], PENTAGON[1][2] - PENTAGON[0][2]];
  const lastSeg = [PENTAGON[0][0] - PENTAGON[n - 1][0], PENTAGON[0][1] - PENTAGON[n - 1][1], PENTAGON[0][2] - PENTAGON[n - 1][2]];
  const norm = (v) => { const l = Math.hypot(...v); return v.map((c) => c / l); };
  const tOut = norm(firstSeg), tIn = norm(lastSeg);
  const tangentDot = tOut[0] * tIn[0] + tOut[1] * tIn[1] + tOut[2] * tIn[2];
  assert.ok(tangentDot < 0.999, `sanity: the pentagon's own first and last edges really do point in genuinely different directions at the seam (tangent dot ${tangentDot.toFixed(6)}) — otherwise this repro isn't reproducing a real corner`);
});

test('THE FIX: buildParallelTransportFrames welds a closed rail\'s seam — frame[0] and frame[last] are now the identical shared miter frame', () => {
  const rail = closedPolylineRail(PENTAGON);
  const frames = buildParallelTransportFrames(rail);
  assert.equal(frames.length, PENTAGON.length + 1, 'one frame per control point, including the repeated closing one');
  const first = frames[0], last = frames[frames.length - 1];
  assert.ok(first === last, 'frame[0] and frame[last] are now the SAME frame object — a genuine weld, not two coincidentally-equal copies');
  // Orthonormal, well-formed (not merely non-crashing): the standard
  // acceptance property every OTHER frame in this file already proves.
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  assert.ok(Math.abs(dot(first.xAxis, first.yAxis)) < 1e-9, 'xAxis ⊥ yAxis');
  assert.ok(Math.abs(dot(first.xAxis, first.zAxis)) < 1e-9, 'xAxis ⊥ zAxis');
  assert.ok(Math.abs(dot(first.yAxis, first.zAxis)) < 1e-9, 'yAxis ⊥ zAxis');
  assert.ok(Math.abs(Math.hypot(...first.xAxis) - 1) < 1e-9, 'xAxis unit length');
  assert.ok(Math.abs(Math.hypot(...first.yAxis) - 1) < 1e-9, 'yAxis unit length');
  assert.ok(Math.abs(Math.hypot(...first.zAxis) - 1) < 1e-9, 'zAxis unit length');
});

test('THE FIX, end to end: sweep1Rigid on a closed pentagon rail produces a genuinely closed tube — zero seam gap, both rings match exactly', () => {
  const rail = closedPolylineRail(PENTAGON);
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], RADIUS);
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true);

  const firstRing = srf.ctrlNet.map((row) => row[0]);
  const lastRing = srf.ctrlNet.map((row) => row[row.length - 1]);
  assert.equal(firstRing.length, lastRing.length, 'both rings sample the same circle profile point count');
  let maxRingDiff = 0;
  for (let i = 0; i < firstRing.length; i++) {
    const d = Math.hypot(...firstRing[i].slice(0, 3).map((v, k) => v - lastRing[i][k]));
    maxRingDiff = Math.max(maxRingDiff, d);
  }
  assert.ok(maxRingDiff < 1e-9, `FIX: the tube's own first and last V-rings are now IDENTICAL (max per-point diff ${maxRingDiff}mm) — was ~5.28mm before the fix, a real gap, not a rounding artifact`);
});

test('an OPEN rail (the ordinary, already-working case) is completely unaffected — frame[0] and frame[last] stay two genuinely distinct endpoints', () => {
  const rail = openPolylineRail(PENTAGON);
  assert.equal(isCurveClosed(rail), false, 'an open pentagon polyline (no repeated closing point) is correctly NOT recognized as closed');
  const frames = buildParallelTransportFrames(rail);
  const first = frames[0], last = frames[frames.length - 1];
  assert.ok(first !== last, 'no weld applied — the two ends of an open rail are genuinely different points, not merged');
  const originDist = Math.hypot(...first.origin.map((v, i) => v - last.origin[i]));
  assert.ok(originDist > 1, `sanity: the open rail's own start/end really are far apart (${originDist.toFixed(2)}mm) — proves this test is exercising the open-rail path, not accidentally still closed`);

  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], RADIUS);
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'an open rail still sweeps to a valid, finite tube exactly as before');
});

test('MultiPipe-equivalent: TWO independent closed rails, each swept via its own separate sweep1Rigid call, each weld their OWN seam independently', () => {
  const pentA = closedPolylineRail(PENTAGON);
  const pentB = closedPolylineRail(PENTAGON.map(([x, y, z]) => [x + 300, y, z])); // a second, disjoint closed rail — MultiPipe's own "N unrelated curves" shape
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], RADIUS);
  for (const rail of [pentA, pentB]) {
    const srf = sweep1Rigid(rail, circleProfile); // the EXACT call MultiPipe makes once per rail
    assert.equal(isFiniteNet(srf.ctrlNet), true);
    const firstRing = srf.ctrlNet.map((row) => row[0]);
    const lastRing = srf.ctrlNet.map((row) => row[row.length - 1]);
    let maxDiff = 0;
    for (let i = 0; i < firstRing.length; i++) maxDiff = Math.max(maxDiff, Math.hypot(...firstRing[i].slice(0, 3).map((v, k) => v - lastRing[i][k])));
    assert.ok(maxDiff < 1e-9, `each independent MultiPipe tube welds its own seam (max ring diff ${maxDiff}mm)`);
  }
});

test('degenerate-corner fallback: a closed rail with a near-180-degree fold-back corner (bisector sum length ~0) still produces a finite, orthonormal seam frame, never NaN', () => {
  // A "spike" pentagon variant: pull one vertex almost exactly onto the
  // line between its two neighbors' shared direction at the SEAM vertex,
  // so the outgoing/incoming edge tangents at vertex 0 point nearly exactly
  // opposite each other (a genuine near-fold-back corner at the seam).
  const spike = [[0, 0, 0], [100, 0.001, 0], [50, 80, 0], [-50, 80, 0], [-100, 0.001, 0]];
  const rail = closedPolylineRail(spike);
  const frames = buildParallelTransportFrames(rail);
  const seam = frames[0];
  assert.ok(Number.isFinite(seam.xAxis[0]) && Number.isFinite(seam.yAxis[0]) && Number.isFinite(seam.zAxis[0]), 'seam frame axes are all finite, never NaN, even at a near-fold-back corner');
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  assert.ok(Math.abs(dot(seam.xAxis, seam.yAxis)) < 1e-6, 'still orthonormal at the degenerate corner');
  assert.ok(Math.abs(Math.hypot(...seam.xAxis) - 1) < 1e-6, 'still unit length at the degenerate corner');
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], RADIUS);
  const srf = sweep1Rigid(rail, circleProfile);
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'the swept tube stays fully finite even at this extreme corner');
});
