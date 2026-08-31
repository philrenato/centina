// kernel/fitcurve.mjs — fitting a NURBS curve through sampled points to a
// stated tolerance.
//
// EVERY EXPECTED ANSWER HERE IS ANALYTIC. A fitter checked against its own
// output only proves it is self-consistent; these fixtures are shapes whose
// exact form is known before the fitter runs (a circle of known radius, a
// straight line, a helix with a closed-form point at every parameter), so a
// fit that is confidently wrong has nowhere to hide.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fitCurveToPoints, maxDeviationFromCurve } from '../kernel/fitcurve.mjs';
import { curvePoint } from '../kernel/curve.mjs';

const circlePts = (R, n, cz = 0) => Array.from({ length: n }, (_, i) => {
  const t = (i / n) * Math.PI * 2;
  return [R * Math.cos(t), R * Math.sin(t), cz];
});

test('A SAMPLED CIRCLE COMES BACK AS AN EXACT CIRCLE, not a spline that looks round', () => {
  const pts = circlePts(25, 120);
  const res = fitCurveToPoints(pts, { tolerance: 1e-6, closed: true });
  assert.ok(res.ok, `refused: ${res.reason}`);
  assert.equal(res.kind, 'circle', 'the primitive path must win here — this IS a circle');
  // A rational quadratic circle is 9 control points in 4 spans, and its
  // weights alternate 1, sqrt(2)/2. That is the signature of the exact form,
  // not of a good approximation.
  assert.equal(res.curve.degree, 2);
  assert.equal(res.curve.ctrlPts.length, 9);
  const ws = res.curve.ctrlPts.map((p) => p[3]);
  assert.ok(Math.abs(ws[1] - Math.SQRT1_2) < 1e-12, `weight 1 is sqrt(2)/2 (got ${ws[1]})`);
  assert.ok(res.maxDeviation < 1e-9, `an exact circle deviates by ~0 (got ${res.maxDeviation})`);
});

test('AND ITS RADIUS IS THE ONE THAT WENT IN — measured off the curve, not off the fit report', () => {
  const pts = circlePts(25, 120);
  const { curve } = fitCurveToPoints(pts, { tolerance: 1e-6, closed: true });
  for (let i = 0; i <= 32; i++) {
    const u = curve.knots[curve.degree] + (curve.knots[curve.knots.length - 1 - curve.degree] - curve.knots[curve.degree]) * (i / 32);
    const p = curvePoint(curve, u);
    assert.ok(Math.abs(Math.hypot(p[0], p[1]) - 25) < 1e-9, `sample ${i} sits at radius ${Math.hypot(p[0], p[1])}`);
  }
});

test('A STRAIGHT RUN OF POINTS COMES BACK AS A DEGREE-1 LINE', () => {
  const pts = Array.from({ length: 40 }, (_, i) => [i * 0.5, 3 + i * 0.25, -2 + i * 0.125]);
  const res = fitCurveToPoints(pts, { tolerance: 1e-6, closed: false });
  assert.ok(res.ok, `refused: ${res.reason}`);
  assert.equal(res.kind, 'line');
  assert.equal(res.curve.degree, 1);
  assert.ok(res.maxDeviation < 1e-9);
});

test('A HELIX IS NOT A PRIMITIVE, so it takes the least-squares path and still meets its bound', () => {
  const pts = Array.from({ length: 200 }, (_, i) => {
    const t = (i / 199) * Math.PI * 4;
    return [10 * Math.cos(t), 10 * Math.sin(t), t * 1.5];
  });
  const TOL = 0.01;
  const res = fitCurveToPoints(pts, { tolerance: TOL, closed: false });
  assert.ok(res.ok, `refused: ${res.reason}`);
  assert.equal(res.kind, 'nurbs');
  assert.ok(res.maxDeviation <= TOL, `deviation ${res.maxDeviation} must clear ${TOL}`);
  // THE POINT OF FITTING AT ALL: far fewer control points than samples. A fit
  // that just interpolated all 200 would meet the tolerance and buy nothing.
  assert.ok(res.ctrlPtCount < pts.length / 3,
    `${res.ctrlPtCount} control points for 200 samples — a fit, not an interpolation`);
});

test('A TIGHTER BOUND COSTS MORE CONTROL POINTS — the loop genuinely responds to its tolerance', () => {
  const pts = Array.from({ length: 200 }, (_, i) => {
    const t = (i / 199) * Math.PI * 4;
    return [10 * Math.cos(t), 10 * Math.sin(t), t * 1.5];
  });
  const loose = fitCurveToPoints(pts, { tolerance: 0.05, closed: false });
  const tight = fitCurveToPoints(pts, { tolerance: 0.0005, closed: false });
  assert.ok(loose.ok && tight.ok);
  assert.ok(tight.ctrlPtCount > loose.ctrlPtCount,
    `tighter must cost more (loose ${loose.ctrlPtCount} vs tight ${tight.ctrlPtCount})`);
  assert.ok(tight.maxDeviation <= 0.0005 && loose.maxDeviation <= 0.05);
});

test('THE ENDPOINTS ARE INTERPOLATED EXACTLY — a shared corner cannot drift into a naked edge', () => {
  const pts = Array.from({ length: 120 }, (_, i) => {
    const t = (i / 119) * Math.PI * 2;
    return [8 * Math.cos(t) + 0.4 * Math.sin(5 * t), 5 * Math.sin(t), 0.3 * Math.cos(3 * t)];
  });
  const res = fitCurveToPoints(pts, { tolerance: 0.01, closed: false });
  assert.ok(res.ok, `refused: ${res.reason}`);
  const first = res.curve.ctrlPts[0], last = res.curve.ctrlPts[res.curve.ctrlPts.length - 1];
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(first[k] - pts[0][k]) < 1e-12, `start control point is the first input point`);
    assert.ok(Math.abs(last[k] - pts[pts.length - 1][k]) < 1e-12, `end control point is the last input point`);
  }
});

test('AN UNREACHABLE BOUND IS REFUSED, not met with a curve that misses it', () => {
  // Random-ish noise no smooth curve of any control-point count can follow to
  // a micron. The refusal is the feature: a fitter that always returns
  // something makes its own tolerance meaningless.
  const pts = Array.from({ length: 60 }, (_, i) => {
    const t = i / 59;
    const jitter = ((i * 2654435761) % 1000) / 1000 - 0.5;
    return [t * 10, jitter * 4, -jitter * 4];
  });
  const res = fitCurveToPoints(pts, { tolerance: 1e-9, closed: false });
  assert.equal(res.ok, false, 'must refuse rather than return a curve outside its bound');
  assert.match(res.reason, /no curve within/);
  // ⚠ THIS IS THE TEST THE INTERPOLATING FALLBACK BROKE, and fixing it fixed a
  // real hole rather than the test. Interpolation threads every sample, so a
  // one-sided "how far is each sample from the curve" check scored it ZERO and
  // certified a 1e-9 bound on noise. Only measuring the curve BACK to the
  // polyline sees the oscillation between the samples.
});

test('THE DEVIATION MEASURE CANNOT UNDER-REPORT — an offset curve reads at least its offset', () => {
  // Conservative by construction: distance to a dense inscribed polyline is
  // never less than distance to the curve. Checked against a case whose true
  // answer is known — points sitting exactly 2.0 off a straight line.
  // The two spans are made to match exactly (both 0..10), so BOTH directions
  // of the two-sided measure have the same true answer of 2.0 and the test is
  // about the measure rather than about mismatched endpoints.
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1]] };
  const off = Array.from({ length: 21 }, (_, i) => [i * 0.5, 2, 0]);
  const dev = maxDeviationFromCurve(off, line);
  assert.ok(dev >= 2 - 1e-9, `must be at least the true 2.0 (got ${dev})`);
  assert.ok(dev < 2 + 1e-6, `and must not wildly over-report (got ${dev})`);
});

test('A CLOSED NON-CIRCULAR LOOP FITS WITHOUT A SEAM KINK', () => {
  const pts = Array.from({ length: 160 }, (_, i) => {
    const t = (i / 160) * Math.PI * 2;
    const r = 12 + 2 * Math.cos(3 * t);
    return [r * Math.cos(t), r * Math.sin(t), 0];
  });
  const TOL = 0.02;
  const res = fitCurveToPoints(pts, { tolerance: TOL, closed: true });
  assert.ok(res.ok, `refused: ${res.reason}`);
  assert.ok(res.maxDeviation <= TOL, `deviation ${res.maxDeviation} must clear ${TOL}`);
  assert.ok(res.ctrlPtCount < pts.length, 'a fit, not an interpolation');
});

// ---------------------------------------------------------------------------
// THE REAL TARGET. Synthetic shapes prove the algorithm; this proves it on the
// thing it was written for — the marched SSI component behind a real boolean,
// from the same banked fixture test/boolean-torus-pair-sew.test.mjs drives.
// ---------------------------------------------------------------------------
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import rhino3dmFactory from 'rhino3dm';
import { surfaceFromRhino } from '../io3dm.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const rhino = await rhino3dmFactory();
const fixDoc = rhino.File3dm.fromByteArray(new Uint8Array(fs.readFileSync(
  path.join(HERE, 'fixtures', 'two_tori_seam_straddle_sew_open.3dm'))));
const fixSrfs = [];
{
  const objs = fixDoc.objects();
  for (let i = 0; i < objs.count; i++) {
    const g = objs.get(i).geometry();
    const ns = g instanceof rhino.NurbsSurface ? g : (g.toNurbsSurface ? g.toNurbsSurface() : null);
    if (ns) fixSrfs.push(surfaceFromRhino(ns));
  }
}
const realSSI = intersectSurfacesComplete(fixSrfs[0], fixSrfs[1]);

test('THE INPUT LANDS: the fixture still yields one densely marched intersection component', () => {
  assert.ok(realSSI && realSSI.ok, 'SSI refused — nothing below means anything');
  assert.equal(realSSI.components.length, 1);
  assert.ok(realSSI.components[0].samples.length > 100, 'marched densely enough to be worth fitting');
});

test("A REAL BOOLEAN'S CUT CURVE FITS AT MODEL TOLERANCE, with far fewer control points than samples", () => {
  const pts = realSSI.components[0].samples.map((s) => s.point);
  const closed = Math.hypot(...pts[0].map((v, k) => v - pts[pts.length - 1][k])) < 1e-6;
  const body = closed ? pts.slice(0, -1) : pts;
  const TOL = 0.01; // the model tolerance this fixture's own document declares
  const res = fitCurveToPoints(body, { tolerance: TOL, closed });
  assert.ok(res.ok, `refused: ${res.reason}`);
  assert.ok(res.maxDeviation <= TOL, `deviation ${res.maxDeviation} must clear ${TOL}`);
  // This IS the win: the boundary stops being a few hundred straight segments.
  assert.ok(res.ctrlPtCount < body.length / 3,
    `${res.ctrlPtCount} control points for ${body.length} samples`);
});

test('AND A BOUND BELOW THE MARCHER\'S OWN ACCURACY IS REFUSED WITH THAT SAID PLAINLY', () => {
  // Not a defect — an honest limit. The samples are only as good as the march
  // that produced them, and a fit asked to sit closer than that would be
  // reproducing noise. What matters is that it says so instead of returning a
  // curve with hundreds of control points and a tolerance it cannot support.
  const pts = realSSI.components[0].samples.map((s) => s.point);
  const closed = Math.hypot(...pts[0].map((v, k) => v - pts[pts.length - 1][k])) < 1e-6;
  const body = closed ? pts.slice(0, -1) : pts;
  const res = fitCurveToPoints(body, { tolerance: 1e-7, closed });
  assert.equal(res.ok, false);
  assert.match(res.reason, /closest was/);
  assert.match(res.reason, /below the accuracy of whatever produced these points/);
  assert.ok(res.bestDeviation > 1e-7, 'and it reports how close it actually got');
});

test('A SHORT COARSE CHAIN IS INTERPOLATED — the bounded fallback, positive control', () => {
  // The case the bound exists FOR: few points, far apart, turning hard. No
  // least-squares curve under the n < m cap can follow it, and interpolation
  // through 7 points is exactly determined and well-conditioned.
  const pts = [[0, 0, 0], [40, 10, 5], [70, 45, -5], [60, 90, 10], [15, 100, 0], [-30, 70, -8], [-20, 20, 4]];
  const res = fitCurveToPoints(pts, { tolerance: 0.01, closed: false });
  assert.ok(res.ok, `refused: ${res.reason}`);
  assert.equal(res.kind, 'interpolated', 'reported as threaded, not as a genuine reduction');
  assert.equal(res.ctrlPtCount, pts.length, 'exactly determined — one control point per sample');
});

test('AND A LONG CHAIN IS NOT — the guard is load-bearing, not decorative', () => {
  // Same shape resampled to 60 points with noise. Without the length bound
  // this would interpolate, score zero at every sample, and certify 1e-9 on
  // data that supports nothing of the kind.
  const pts = Array.from({ length: 60 }, (_, i) => {
    const t = i / 59;
    const j = ((i * 2654435761) % 1000) / 1000 - 0.5;
    return [t * 100, j * 6, -j * 6];
  });
  const res = fitCurveToPoints(pts, { tolerance: 1e-9, closed: false });
  assert.equal(res.ok, false);
  assert.match(res.reason, /too long to fall back on interpolation/);
});

/* A CLOSED, CORNERLESS CONTOUR SHORT ENOUGH TO INTERPOLATE MUST ACTUALLY
   INTERPOLATE. `closedCurveInterp` returns `{ crv, uStart, uEnd }` while
   `globalCurveInterp` returns a curve, and the shared `interp.ctrlPts` guard
   was therefore always false on the closed branch — so this path had never
   run, and a closed contour of ten points came back `ok: false`, not merely
   fitted less well. Asserting the KIND and the deviation, because a
   least-squares fallback that happens to succeed would satisfy `ok` alone. */
test('fitCurveToPoints interpolates a short closed cornerless contour', () => {
  const points = [];
  for (let i = 0; i < 10; i++) {
    const t = (i / 10) * 2 * Math.PI;
    points.push([60 * Math.cos(t), 35 * Math.sin(t) + 6 * Math.sin(3 * t), 0]);
  }
  const r = fitCurveToPoints(points, { closed: true, tolerance: 0.5 });
  assert.ok(r.ok, 'a ten-point closed contour must fit at all');
  assert.equal(r.kind, 'interpolated', 'it must take the interpolation path, not fall through to least squares');
  assert.ok(r.maxDeviation < 1e-6, `interpolation passes through its own points (got ${r.maxDeviation})`);
  /* The curve really closes — the whole point of the closed branch. */
  const cp = r.curve.ctrlPts;
  const gap = Math.hypot(cp[0][0] - cp[cp.length - 1][0], cp[0][1] - cp[cp.length - 1][1], cp[0][2] - cp[cp.length - 1][2]);
  assert.ok(gap < 1e-6, `the fitted closed contour closes (endpoint gap ${gap})`);
});

// ---------------------------------------------------------------------------
// AN EXCURSION IS INVISIBLE TO THE DEVIATION MEASURE, so it is asserted by
// LENGTH. Deviation is one-sided and measured AT the samples; a curve that
// swings clear of the data between two samples moves no sample and scores
// perfectly. Arc length is the cheapest quantity it cannot hide from — a curve
// threading a polyline is a fraction of a per cent LONGER than that polyline,
// and anything past a couple of per cent is shape nobody asked for.
// ---------------------------------------------------------------------------
const polylineLength = (pts) => {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]);
  return L;
};
const sampledArcLength = (crv, n = 4000) => {
  const p = crv.degree, U = crv.knots;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  let L = 0, prev = curvePoint(crv, t0);
  for (let i = 1; i <= n; i++) {
    const c = curvePoint(crv, t0 + ((t1 - t0) * i) / n);
    L += Math.hypot(c[0] - prev[0], c[1] - prev[1], c[2] - prev[2]);
    prev = c;
  }
  return L;
};

/* A RUN WITH ONE VERY LONG LEG IS THE CASE THAT BREAKS A CHORD-LENGTH FIT, and
   four points is not few enough to be safe from it. These are real coordinates
   off a traced letter "j": three points a fraction of a unit apart at one end
   and a seventeen-unit straight leg to the other. Chord-length parametrisation
   spends 96% of the domain on that leg, so all the shape change has to happen
   in the remaining 4% — which the curve can only do by throwing a control point
   thirty units clear of data that never leaves x = 1.4 to 1.7.

   The deviation at the four samples is 4e-11 either way. Only the length
   separates them. */
test('A RUN WITH ONE LONG LEG IS NOT THREADED INTO A LOOP — four points is not too few to oscillate', () => {
  const pts = [[1.70, 14.58, 0], [1.66, -2.28, 0], [1.56, -2.92, 0], [1.39, -3.20, 0]];
  const res = fitCurveToPoints(pts, { tolerance: 0.114, closed: false, exactEndpoints: true });
  assert.ok(res.ok, `refused: ${res.reason}`);
  const ratio = sampledArcLength(res.curve) / polylineLength(pts);
  assert.ok(ratio <= 1.02, `the fit is ${ratio.toFixed(4)}x the length of its own data`);
  // And it stays inside the data's own footprint — this run is very nearly a
  // straight line and the curve has no licence to leave it.
  const p = res.curve.degree, U = res.curve.knots;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  for (let i = 0; i <= 200; i++) {
    const c = curvePoint(res.curve, t0 + ((t1 - t0) * i) / 200);
    assert.ok(c[0] > 1.0 && c[0] < 2.1, `sample ${i} sits at x = ${c[0].toFixed(3)}, outside the data's own 1.39..1.70`);
  }
});

/* THE SAME FAILURE ON THE LEAST-SQUARES BRANCH, where the count is chosen
   rather than forced. Six points off a traced letter "t": four of them clustered
   at one end and a nine-unit leg to the last. */
test('A LEAST-SQUARES FIT DOES NOT BULGE OFF A LONG UNSAMPLED LEG', () => {
  const pts = [[6.97, 2.09, 0], [5.58, 2.09, 0], [5.20, 2.23, 0], [4.90, 2.53, 0], [4.77, 2.92, 0], [4.75, 12.55, 0]];
  const res = fitCurveToPoints(pts, { tolerance: 0.114, closed: false, exactEndpoints: true });
  assert.ok(res.ok, `refused: ${res.reason}`);
  const ratio = sampledArcLength(res.curve) / polylineLength(pts);
  assert.ok(ratio <= 1.02, `the fit is ${ratio.toFixed(4)}x the length of its own data`);
  assert.ok(res.maxDeviation <= 0.114, `and still holds its bound (${res.maxDeviation})`);
});

/* EVERY CANDIDATE CARRIES ITS OWN EXCURSION, so a caller reading `triedCounts`
   can see WHY a count was passed over rather than only that it was. A candidate
   inside the corridor scores zero or less. */
test('EVERY TRIED COUNT REPORTS HOW FAR IT LEFT THE CORRIDOR', () => {
  // A superellipse, so the circle/ellipse recogniser cannot answer first and
  // the least-squares search actually runs.
  const pts = Array.from({ length: 24 }, (_, i) => {
    const t = (i / 24) * Math.PI * 2, c = Math.cos(t), s = Math.sin(t);
    return [25 * Math.sign(c) * Math.abs(c) ** (2 / 2.6), 25 * Math.sign(s) * Math.abs(s) ** (2 / 2.6), 0];
  });
  const res = fitCurveToPoints(pts, { tolerance: 0.5, closed: true });
  assert.ok(res.ok, `refused: ${res.reason}`);
  assert.ok(res.triedCounts.length > 0, 'the search reports what it tried');
  for (const t of res.triedCounts) {
    assert.equal(typeof t.corridorExcess, 'number', `candidate at ${t.ctrlPts} control points reports no corridor excess`);
    assert.ok(Number.isFinite(t.corridorExcess), 'and it is a real number');
  }
  // The search really discriminated: it saw at least one candidate outside the
  // corridor and at least one inside. A corridor that were always satisfied, or
  // never, would be a number nobody reads.
  assert.ok(res.triedCounts.some((t) => t.corridorExcess > 0), 'some candidate must have left the corridor');
  assert.ok(res.triedCounts.some((t) => t.corridorExcess <= 0), 'and some candidate must have stayed inside it');
});
