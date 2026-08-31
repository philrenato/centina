// VARIABLE RADIUS ALONG A SWEEP1 RAIL — added directly after the
// corner-rounding fix shipped.
// This file is WITHIN one curve only — a single Pipe's own
// rail carries several `{t, radius}` breakpoints, `t` a normalized
// rail-PARAMETER fraction (0=start, 1=end), matching this app's own
// existing `splitFrac` convention (a parameter-domain fraction, not an
// arc-length fraction — see kernel/sweep.mjs's own header comment on
// `radiusAtT`/`variableRadiusScaler` for the full derivation and the
// honest limitation this implies once a rail is corner-rounded). MultiPipe's
// own "per curve" case (each independent tube in a network getting its OWN
// radius profile) is a deliberately staged, separate follow-up round — not
// attempted here, and this file proves `sweepNProfiles`/`sweep2` are
// untouched (no radiusOpts threading exists there at all).
import test from 'node:test';
import assert from 'node:assert/strict';
import { sweep1Rigid, radiusAtT } from '../kernel/sweep.mjs';
import { makeCircle, filletOpenPolyline, filletSegmentsToCurve } from '../kernel/primitives.mjs';
import { isFiniteNet } from '../kernel/surface.mjs';

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function length(a) { return Math.hypot(a[0], a[1], a[2]); }

// A straight, OPEN, degree-1 rail along +Z with `n` evenly-spaced control
// points from z=0 to z=zMax — getProfileCrv's own real knot convention for
// a Polyline (UNIFORM per-segment parameter width [0,0,1,2,...,n-1,n-1]),
// hand-built here rather than imported (this file tests the KERNEL function
// directly, not the app's own Polyline object).
function straightRail(n, zMax) {
  const ctrlPts = [];
  for (let i = 0; i < n; i++) ctrlPts.push([0, 0, (zMax * i) / (n - 1), 1]);
  const knots = [0, 0];
  for (let i = 1; i <= n - 2; i++) knots.push(i);
  knots.push(n - 1, n - 1);
  return { degree: 1, knots, ctrlPts };
}

// A Pipe circle profile, radius 1 (its own control point at U=0 sits at
// exactly LOCAL (1,0,0,1) — makeArc's own p0 at angleStart=0 is
// center + xAxis*r*cos(0) + yAxis*r*sin(0) = (r,0,0) for center=(0,0,0),
// xAxis=(1,0,0) — so ctrlNet[0][k]'s distance from frame k's own origin is
// EXACTLY r*scaleFactor(k), a clean, direct numeric proof requiring no
// surface sampling at all.
const unitCircle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 1);

function ringRadius(srf, rowIdx, colIdx) {
  const cp = srf.ctrlNet[rowIdx][colIdx];
  const origin = srf.frames[colIdx].origin;
  return length(sub([cp[0], cp[1], cp[2]], origin));
}

// ---- radiusAtT, the pure interpolation/clamp helper ----
test('radiusAtT: single breakpoint returns that radius for any t', () => {
  const pts = [{ t: 0.4, radius: 7 }];
  assert.equal(radiusAtT(pts, 0), 7);
  assert.equal(radiusAtT(pts, 0.4), 7);
  assert.equal(radiusAtT(pts, 1), 7);
});
test('radiusAtT: two breakpoints linearly interpolate, clamp outside [minT,maxT]', () => {
  const pts = [{ t: 0.2, radius: 2 }, { t: 0.8, radius: 8 }];
  assert.equal(radiusAtT(pts, 0.2), 2);
  assert.equal(radiusAtT(pts, 0.8), 8);
  assert.ok(Math.abs(radiusAtT(pts, 0.5) - 5) < 1e-12, 'midpoint of a 2->8 span is exactly 5');
  assert.equal(radiusAtT(pts, 0), 2, 'below minT clamps to the first breakpoint, never extrapolates');
  assert.equal(radiusAtT(pts, 1), 8, 'above maxT clamps to the last breakpoint, never extrapolates');
  assert.equal(radiusAtT(pts, -5), 2);
  assert.equal(radiusAtT(pts, 50), 8);
});
test('radiusAtT: three breakpoints (bulge) — exact at the middle breakpoint, correct interpolation either side', () => {
  const pts = [{ t: 0, radius: 3 }, { t: 0.5, radius: 9 }, { t: 1, radius: 3 }];
  assert.equal(radiusAtT(pts, 0), 3);
  assert.equal(radiusAtT(pts, 0.5), 9, 'exact at the middle breakpoint');
  assert.equal(radiusAtT(pts, 1), 3);
  assert.ok(Math.abs(radiusAtT(pts, 0.25) - 6) < 1e-12, 'hand-computed: 3 + 0.5*(9-3) = 6');
  assert.ok(Math.abs(radiusAtT(pts, 0.75) - 6) < 1e-12, 'hand-computed: 9 + 0.5*(3-9) = 6');
});
test('radiusAtT: out-of-order input is sorted defensively before interpolating', () => {
  const pts = [{ t: 1, radius: 8 }, { t: 0, radius: 2 }];
  assert.ok(Math.abs(radiusAtT(pts, 0.5) - 5) < 1e-12);
});

// ---- THE single most important regression guard ----
test('sweep1Rigid: no radiusOpts at all is byte-identical to plain sweep1Rigid (free path, degree<=1 rail)', () => {
  const rail = straightRail(5, 100);
  const plain = sweep1Rigid(rail, unitCircle);
  const withNull = sweep1Rigid(rail, unitCircle, null);
  assert.deepEqual(withNull.ctrlNet, plain.ctrlNet);
  assert.deepEqual(withNull.knotsV, plain.knotsV);
  assert.equal(withNull.degV, plain.degV);
});
test('sweep1Rigid: a SINGLE breakpoint (radiusOpts present but length<=1) is byte-identical to plain sweep1Rigid too', () => {
  const rail = straightRail(5, 100);
  const plain = sweep1Rigid(rail, unitCircle);
  const oneBreakpoint = sweep1Rigid(rail, unitCircle, { radiusPoints: [{ t: 0, radius: 5 }], baseRadius: 5 });
  assert.deepEqual(oneBreakpoint.ctrlNet, plain.ctrlNet, 'a single breakpoint must fall back to the plain constant-radius tube, exactly — the app layer\'s own "radiusPoints.length<=1 means use the plain radius param" contract');
});
test('sweep1RigidResampled (degree>1 rail path): no radiusOpts is byte-identical to plain output', () => {
  // A degree-2 rail (routes through the resampled path) — reuse the corner
  // fillet machinery to build one cheaply, matching how Pipe's own
  // pipeRailForSweep composes a rounded rail in the real app.
  const res = filletOpenPolyline([[0, 0, 0], [10, 0, 0], [10, 10, 0]], 2, { closed: false });
  assert.equal(res.ok, true, res.reason);
  const rail = filletSegmentsToCurve(res.segments);
  assert.ok(rail.degree > 1, 'a filleted rail is degree-2, routes through sweep1RigidResampled');
  const plain = sweep1Rigid(rail, unitCircle);
  const withNull = sweep1Rigid(rail, unitCircle, null);
  assert.deepEqual(withNull.ctrlNet, plain.ctrlNet);
});

// ---- two breakpoints: monotonic growth, exact at both ends ----
test('sweep1Rigid: two breakpoints (small start, large end) produce a monotonically-growing tube, exact at t=0 and t=1', () => {
  const rail = straightRail(5, 100); // 5 stations at t = 0, 0.25, 0.5, 0.75, 1.0
  const radiusPoints = [{ t: 0, radius: 2 }, { t: 1, radius: 10 }];
  const srf = sweep1Rigid(rail, unitCircle, { radiusPoints, baseRadius: 1 });
  assert.ok(isFiniteNet(srf.ctrlNet));
  const measured = [0, 1, 2, 3, 4].map((col) => ringRadius(srf, 0, col));
  // Hand-computed: linear from 2 at t=0 to 10 at t=1, sampled at t=0,.25,.5,.75,1
  const expected = [2, 4, 6, 8, 10];
  measured.forEach((m, i) => assert.ok(Math.abs(m - expected[i]) < 1e-9, `station ${i}: expected ${expected[i]}, measured ${m}`));
  for (let i = 1; i < measured.length; i++) assert.ok(measured[i] > measured[i - 1] + 1e-9, `radius must strictly grow station-to-station (${measured[i - 1]} -> ${measured[i]})`);
});

// ---- three breakpoints: bulge, exact at the middle, correct in between ----
test('sweep1Rigid: three breakpoints (small-large-small bulge) — exact at the middle breakpoint, correct linear interpolation either side', () => {
  const rail = straightRail(5, 100);
  const radiusPoints = [{ t: 0, radius: 3 }, { t: 0.5, radius: 9 }, { t: 1, radius: 3 }];
  const srf = sweep1Rigid(rail, unitCircle, { radiusPoints, baseRadius: 1 });
  assert.ok(isFiniteNet(srf.ctrlNet));
  const measured = [0, 1, 2, 3, 4].map((col) => ringRadius(srf, 0, col));
  const expected = [3, 6, 9, 6, 3]; // hand-computed, see radiusAtT's own matching unit test above
  measured.forEach((m, i) => assert.ok(Math.abs(m - expected[i]) < 1e-9, `station ${i}: expected ${expected[i]}, measured ${m}`));
});

// ---- clamping outside [minT, maxT] holds the end value ----
test('sweep1Rigid: breakpoints not spanning the full [0,1] domain clamp at both ends, never extrapolate', () => {
  const rail = straightRail(5, 100); // stations at t=0, .25, .5, .75, 1
  const radiusPoints = [{ t: 0.25, radius: 4 }, { t: 0.75, radius: 8 }];
  const srf = sweep1Rigid(rail, unitCircle, { radiusPoints, baseRadius: 1 });
  const measured = [0, 1, 2, 3, 4].map((col) => ringRadius(srf, 0, col));
  const expected = [4, 4, 6, 8, 8]; // t=0 & t=.25 both clamp to 4; t=.5 interpolates to 6; t=.75 & t=1 both clamp to 8
  measured.forEach((m, i) => assert.ok(Math.abs(m - expected[i]) < 1e-9, `station ${i}: expected ${expected[i]}, measured ${m}`));
});

// ---- combined with corner-rounding: finite, exact at the true endpoints, approximately right in between ----
test('sweep1Rigid: variable radius composed with a corner-rounded rail stays finite; exact at the rail\'s true start/end; approximately right at an interior breakpoint (honest, not claimed exact)', () => {
  const res = filletOpenPolyline([[0, 0, 0], [20, 0, 0], [20, 20, 0]], 3, { closed: false });
  assert.equal(res.ok, true, res.reason);
  const rail = filletSegmentsToCurve(res.segments);
  assert.ok(rail.degree > 1, 'sanity: this rail really does route through the resampled path, the same path Pipe\'s cornerStyle:\'rounded\' uses');
  const radiusPoints = [{ t: 0, radius: 2 }, { t: 0.5, radius: 6 }, { t: 1, radius: 10 }];
  const srf = sweep1Rigid(rail, unitCircle, { radiusPoints, baseRadius: 1 });
  assert.ok(isFiniteNet(srf.ctrlNet), 'no NaN/Infinity anywhere, even combined with corner-rounding');
  // The two true rail ENDPOINTS are exact (fillet never moves them) —
  // frames[0]/frames[last] are the dense path's first/last stations.
  const startR = ringRadius(srf, 0, 0);
  const endR = ringRadius(srf, 0, srf.frames.length - 1);
  assert.ok(Math.abs(startR - 2) < 1e-6, `rail start must read exactly radiusPoints[0].radius=2, measured ${startR}`);
  assert.ok(Math.abs(endR - 10) < 1e-6, `rail end must read exactly radiusPoints[last].radius=10, measured ${endR}`);
  // An interior station near the middle of the dense frame list should land
  // APPROXIMATELY near the t=0.5 bulge (6) — not asserted exact, per this
  // file's own header comment and kernel/sweep.mjs's own honest limitation
  // (parameter-fraction, not arc-length-fraction, is more visibly
  // approximate once the rail is corner-rounded).
  const midCol = Math.round((srf.frames.length - 1) / 2);
  const midR = ringRadius(srf, 0, midCol);
  assert.ok(midR > startR && midR < endR + 1e-6 || Math.abs(midR - 6) < 3, `interior station should land roughly near the intended bulge region (measured ${midR}) — approximate, not exact, by design on a rounded rail`);
});

// ---- MultiPipe / sweepNProfiles / sweep2 are completely untouched ----
test('sweepNProfiles and sweep2 have no radiusOpts parameter — this file is single-Pipe (within-curve) scope only, per curve MultiPipe support is a separate staged follow-up', async () => {
  const { sweepNProfiles, sweep2 } = await import('../kernel/sweep.mjs');
  assert.equal(sweepNProfiles.length, 2, 'sweepNProfiles signature unchanged: (rail, profiles, uSampleCount=24, ...) — Function.length only counts params before the first one with a default');
  assert.equal(sweep2.length, 3, 'sweep2 signature unchanged: (rail1, rail2, profile, opts)');
});
