import test from 'node:test';
import assert from 'node:assert/strict';
import { makeEllipse, makeCircle } from '../kernel/primitives.mjs';
import { curvePoint } from '../kernel/curve.mjs';
import { fitLine, fitPlane, fitCircle, fitEllipse, fitAll, FIT_REFUSAL } from '../kernel/refit.mjs';

// ---------------------------------------------------------------------
// FIXTURES — deliberately ROTATED, OFF-ORIGIN, and NON-AXIS-ALIGNED, per
// this project's own standing rule against symmetric test geometry. An
// ellipse centered at the origin with its axes on world X/Y would let an
// axis-assignment or sign bug pass unnoticed: every candidate frame would
// give the same answer. Every fixture here therefore lives on an oblique
// plane (normal along [1,2,3]), off the origin, with its own major axis
// rotated a further 31 degrees WITHIN that plane so it coincides with
// neither world axes nor the module's own canonical plane basis.
// ---------------------------------------------------------------------
function norm3(v) { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; }
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function dist3(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }

const OBLIQUE_NORMAL = norm3([1, 2, 3]);
const CENTER = [37, -12, 55];
const RX = 41, RY = 17;

function obliqueFrame(rotDeg) {
  const n = OBLIQUE_NORMAL;
  const u = norm3(cross3(n, [0, 0, 1]));
  const v = cross3(n, u);
  const t = (rotDeg * Math.PI) / 180;
  const ex = norm3([
    u[0] * Math.cos(t) + v[0] * Math.sin(t),
    u[1] * Math.cos(t) + v[1] * Math.sin(t),
    u[2] * Math.cos(t) + v[2] * Math.sin(t),
  ]);
  return { normal: n, ex, ey: cross3(n, ex) };
}

// Sample a curve at `n` evenly spaced PARAMETER values across its own
// knot domain. Sampled points, never raw control points — a rational
// curve's control points are not on the curve (see refit.mjs's own input
// convention note), so fitting them would be testing the wrong thing.
function sampleCurve(crv, n) {
  const uMin = crv.knots[crv.degree];
  const uMax = crv.knots[crv.knots.length - 1 - crv.degree];
  const out = [];
  for (let i = 0; i < n; i++) out.push(curvePoint(crv, uMin + ((uMax - uMin) * i) / n));
  return out;
}

function ellipseFixture(rotDeg = 31, segments = 4, count = 60) {
  const { normal, ex, ey } = obliqueFrame(rotDeg);
  const crv = makeEllipse(CENTER, ex, ey, RX, RY, segments);
  return { normal, ex, ey, crv, points: sampleCurve(crv, count) };
}

// ---------------------------------------------------------------------
// ROUND-TRIP AGAINST THE REAL KERNEL — the strongest check available:
// build with known params, sample, fit, recover the params.
// ---------------------------------------------------------------------
test('fitEllipse round-trips a real makeEllipse: every recovered param matches the built one to ~1e-13', () => {
  const { normal, ex, points } = ellipseFixture();
  const f = fitEllipse(points);
  assert.equal(f.ok, true, f.detail);
  assert.equal(f.shape, 'ellipse');

  // Center, radii: absolute model units on a ~41mm shape at ~65mm from
  // the origin — so these tolerances are genuinely tight, not generous.
  assert.ok(dist3(f.center, CENTER) < 1e-11, `center off by ${dist3(f.center, CENTER)}`);
  assert.ok(Math.abs(f.radiusX - RX) < 1e-11, `radiusX ${f.radiusX} vs ${RX}`);
  assert.ok(Math.abs(f.radiusY - RY) < 1e-11, `radiusY ${f.radiusY} vs ${RY}`);

  // Axis directions: |dot| === 1 (a direction is recovered up to sign by
  // construction — see refit.mjs's canonicalization note — so the SIGN is
  // deliberately not asserted here; determinism of that sign has its own
  // test below).
  assert.ok(Math.abs(Math.abs(dot3(f.xAxis, ex)) - 1) < 1e-12, `xAxis dot ${dot3(f.xAxis, ex)}`);
  assert.ok(Math.abs(Math.abs(dot3(f.normal, normal)) - 1) < 1e-12, `normal dot ${dot3(f.normal, normal)}`);

  // radiusX is the semi-MAJOR axis, always (the stated determinism rule).
  assert.ok(f.radiusX >= f.radiusY);
  assert.equal(f.circular, false);

  // The measured deviation on an EXACT ellipse is float noise, not a
  // tolerance that had to be relaxed to pass.
  assert.ok(f.worst < 1e-10, `worst ${f.worst}`);
  assert.ok(f.rms < 1e-10, `rms ${f.rms}`);
  assert.ok(f.planeWorst < 1e-10, `planeWorst ${f.planeWorst}`);
  assert.equal(f.count, points.length);
});

test('fitEllipse recovered params REBUILD the same curve: makeEllipse(fit) reproduces the original geometry', () => {
  // The strongest form of the round-trip — not "the numbers look close"
  // but "feeding the recovered params straight back into the kernel's own
  // constructor produces the same curve," which is exactly what a
  // strategy-switch / conversion caller actually does.
  const { crv, points } = ellipseFixture();
  const f = fitEllipse(points);
  assert.equal(f.ok, true);
  const rebuilt = makeEllipse(f.center, f.xAxis, f.yAxis, f.radiusX, f.radiusY, 4);
  const rebuiltPts = sampleCurve(rebuilt, 200);
  const origPts = sampleCurve(crv, 200);
  // Both curves trace the same ellipse, but a rebuilt curve's own start
  // angle / sweep direction follow the RECOVERED frame, which may differ
  // from the original's by a rotation or a reflection. So compare SHAPES:
  // every rebuilt sample must lie on the ORIGINAL fitted ellipse.
  const back = fitEllipse(rebuiltPts);
  assert.equal(back.ok, true);
  assert.ok(dist3(back.center, CENTER) < 1e-10);
  assert.ok(Math.abs(back.radiusX - RX) < 1e-10);
  assert.ok(Math.abs(back.radiusY - RY) < 1e-10);
  // And, directly: the original points are on the REBUILT ellipse too.
  const cross = fitEllipse(origPts);
  assert.ok(cross.worst < 1e-10);
  assert.ok(rebuiltPts.every((p) => p.every(Number.isFinite)));
});

test('fitEllipse works on a HALF arc, not just a closed loop', () => {
  // A conic fit degrades on short arcs (a documented property of the
  // algebraic method, named honestly in refit.mjs's header). Half an
  // ellipse is the realistic worst case a recipe re-derivation sees, and
  // it still recovers exactly here.
  const { points } = ellipseFixture(31, 4, 240);
  const half = points.slice(0, 120);
  const f = fitEllipse(half);
  assert.equal(f.ok, true, f.detail);
  assert.ok(Math.abs(f.radiusX - RX) < 1e-8, `radiusX ${f.radiusX}`);
  assert.ok(Math.abs(f.radiusY - RY) < 1e-8, `radiusY ${f.radiusY}`);
  assert.ok(f.worst < 1e-8, `worst ${f.worst}`);
});

test('fitEllipse recovers the major axis of an AXIS-ALIGNED ellipse (semi-major eigenvector from the best-conditioned row)', () => {
  // A DELIBERATE EXCEPTION to this file's own oblique/rotated fixture rule,
  // and the reason the rule let a real bug through: every other fixture here
  // is rotated 31 degrees inside an oblique plane specifically so no axis
  // assignment can pass by coincidence — which also meant nothing ever fed
  // the fitter the ONE case an app produces constantly, an ellipse whose own
  // axes line up with its fitted plane's own basis. There B (the off-diagonal
  // conic term) is float dust rather than zero, so the old fixed-row
  // eigenvector read [dust, dust] and returned a direction that was pure
  // noise: the radii came back exact (40 and 20) while the major axis was
  // reported ~43 degrees off, and the deviation for an EXACT ellipse read
  // 14+ mm. Both world orientations are checked, since only ONE of the two
  // lands the major axis on the fitted plane's second basis vector (the
  // failing side), and which one that is depends on the plane fit's own
  // deterministic basis, not on anything the caller controls.
  for (const [label, rx, ry] of [['major along X', 40, 20], ['major along Y', 20, 40]]) {
    const crv = makeEllipse([0, 0, 0], [1, 0, 0], [0, 1, 0], rx, ry, 4);
    const points = sampleCurve(crv, 64);
    const f = fitEllipse(points);
    assert.equal(f.ok, true, `${label}: ${f.detail}`);
    // Radii were ALWAYS right, even when the axis was garbage — asserting
    // them alone would have kept passing through the bug.
    assert.ok(Math.abs(f.radiusX - Math.max(rx, ry)) < 1e-9, `${label}: radiusX ${f.radiusX}`);
    assert.ok(Math.abs(f.radiusY - Math.min(rx, ry)) < 1e-9, `${label}: radiusY ${f.radiusY}`);
    // The real assertion: the recovered major axis is the world axis the
    // longer radius was actually built along (up to sign), and the measured
    // deviation is float noise rather than a large, confident, wrong number.
    const expected = rx >= ry ? [1, 0, 0] : [0, 1, 0];
    assert.ok(
      Math.abs(Math.abs(dot3(f.xAxis, expected)) - 1) < 1e-9,
      `${label}: xAxis ${JSON.stringify(f.xAxis)} should be +/- ${JSON.stringify(expected)}`,
    );
    assert.ok(f.worst < 1e-9, `${label}: worst ${f.worst} — a wrong axis shows up here as a large deviation`);
  }
});

test('fitCircle round-trips a real makeCircle, and its deviation is the true 3D distance to the circle', () => {
  const { normal, ex, ey } = obliqueFrame(17);
  const R = 23;
  const crv = makeCircle(CENTER, ex, ey, R, 4);
  const points = sampleCurve(crv, 48);
  const f = fitCircle(points);
  assert.equal(f.ok, true, f.detail);
  assert.equal(f.shape, 'circle');
  assert.ok(dist3(f.center, CENTER) < 1e-11, `center off by ${dist3(f.center, CENTER)}`);
  assert.ok(Math.abs(f.radius - R) < 1e-11, `radius ${f.radius}`);
  assert.ok(Math.abs(Math.abs(dot3(f.normal, normal)) - 1) < 1e-12);
  assert.ok(f.worst < 1e-10, `worst ${f.worst}`);

  // Recovered params feed makeCircle directly (the Circle operator's
  // param shape), and reproduce the same circle.
  const rebuilt = makeCircle(f.center, f.xAxis, f.yAxis, f.radius, 4);
  for (const p of sampleCurve(rebuilt, 60)) {
    assert.ok(Math.abs(dist3(p, CENTER) - R) < 1e-10);
    assert.ok(Math.abs(dot3([p[0] - CENTER[0], p[1] - CENTER[1], p[2] - CENTER[2]], normal)) < 1e-10);
  }

  // A point pushed OUT OF PLANE must raise the deviation by that full
  // out-of-plane amount (the metric is the real 3D distance to the circle
  // curve, not the in-plane radial miss alone).
  const off = points.slice();
  off[3] = [off[3][0] + normal[0] * 0.4, off[3][1] + normal[1] * 0.4, off[3][2] + normal[2] * 0.4];
  const g = fitCircle(off, { planarTol: 5 });
  assert.equal(g.ok, true, g.detail);
  assert.ok(g.worst > 0.3, `worst ${g.worst} should reflect the 0.4 out-of-plane push`);
});

test('fitLine round-trips a known line and reports its own start/end and length', () => {
  const dir = norm3([2, 1.5, -0.25]);
  const p0 = [3, -5, 11];
  const points = [];
  for (let i = 0; i < 9; i++) points.push([p0[0] + dir[0] * i * 2.5, p0[1] + dir[1] * i * 2.5, p0[2] + dir[2] * i * 2.5]);
  const f = fitLine(points);
  assert.equal(f.ok, true);
  assert.equal(f.shape, 'line');
  assert.ok(f.worst < 1e-12, `worst ${f.worst}`);
  assert.ok(Math.abs(Math.abs(dot3(f.direction, dir)) - 1) < 1e-12);
  assert.ok(Math.abs(f.length - 8 * 2.5) < 1e-10, `length ${f.length}`);
  // start/end are the real extreme points of the input set, in some order.
  const last = points[8];
  const nearest = Math.min(dist3(f.start, p0) + dist3(f.end, last), dist3(f.start, last) + dist3(f.end, p0));
  assert.ok(nearest < 1e-10, `endpoints off by ${nearest}`);
  assert.ok([...f.point, ...f.direction, ...f.start, ...f.end, f.length, f.worst, f.rms].every(Number.isFinite));
});

test('fitPlane recovers an oblique plane exactly, with a right-handed deterministic basis', () => {
  const { normal, ex } = obliqueFrame(31);
  const { points } = ellipseFixture(31);
  const f = fitPlane(points);
  assert.equal(f.ok, true);
  assert.ok(Math.abs(Math.abs(dot3(f.normal, normal)) - 1) < 1e-12);
  assert.ok(f.worst < 1e-10, `worst ${f.worst}`);
  // Orthonormal, right-handed: xAxis x yAxis === normal.
  assert.ok(Math.abs(dot3(f.xAxis, f.yAxis)) < 1e-14);
  assert.ok(Math.abs(dot3(f.xAxis, f.normal)) < 1e-14);
  const xy = cross3(f.xAxis, f.yAxis);
  assert.ok(dist3(xy, f.normal) < 1e-12, `xAxis x yAxis !== normal`);
  // The plane's basis is derived from the NORMAL, never from the data's
  // in-plane spread — so it does NOT coincide with the ellipse's own
  // major axis, which is exactly the property that keeps it stable for a
  // near-circular point set.
  assert.ok(Math.abs(Math.abs(dot3(f.xAxis, ex)) - 1) > 1e-6);
});

// ---------------------------------------------------------------------
// GRACEFUL DEGRADATION — the deviation metric has to MEAN something.
// ---------------------------------------------------------------------
test('the reported deviation tracks an injected perturbation, and scales linearly with it', () => {
  // Perturb each sample RADIALLY (alternating in/out) by a known amount.
  // Radial is deliberately not the same direction as the ellipse's own
  // surface normal, so the measured perpendicular distance is a known
  // FRACTION of the injected offset rather than equal to it — the honest
  // check is therefore (a) the same order of magnitude, and (b) exactly
  // linear in the injected amount, which is what proves the number is
  // measured rather than decorative.
  const { points } = ellipseFixture();
  const results = [];
  for (const d of [0.01, 0.1, 1.0]) {
    const perturbed = points.map((p, i) => {
      const rel = [p[0] - CENTER[0], p[1] - CENTER[1], p[2] - CENTER[2]];
      const u = norm3(rel);
      const s = (i % 2 === 0 ? 1 : -1) * d;
      return [p[0] + u[0] * s, p[1] + u[1] * s, p[2] + u[2] * s];
    });
    const f = fitEllipse(perturbed);
    assert.equal(f.ok, true, f.detail);
    assert.ok(Number.isFinite(f.worst) && Number.isFinite(f.rms));
    results.push({ d, worst: f.worst, rms: f.rms });
  }
  // Same order of magnitude as the injected amount, both ends.
  for (const r of results) {
    assert.ok(r.rms > 0.5 * r.d, `rms ${r.rms} too small for injected ${r.d}`);
    assert.ok(r.worst < 1.5 * r.d, `worst ${r.worst} too large for injected ${r.d}`);
    assert.ok(r.worst >= r.rms, 'worst must be at least RMS');
  }
  // Linear: 10x the perturbation, 10x the reported deviation (within 5%).
  for (let i = 1; i < results.length; i++) {
    const ratio = results[i].rms / results[i - 1].rms;
    assert.ok(Math.abs(ratio - 10) < 0.5, `rms did not scale linearly: ratio ${ratio}`);
  }
  // An UNperturbed fit of the same fixture reports essentially zero, so
  // the numbers above are genuinely responding to the injected error.
  assert.ok(fitEllipse(points).rms < 1e-10);
});

test('a parabola handed to the ellipse fit returns an ELLIPSE (never a hyperbola) — with an honest, large deviation', () => {
  // This is the case an unconstrained algebraic conic fit gets wrong: it
  // would happily return a hyperbola or parabola. The Fitzgibbon
  // ellipse-specific constraint makes that unreachable, so the answer is
  // structurally an ellipse — and the MEASURED deviation is what tells the
  // caller the answer is a bad description of the data. Both halves
  // matter; either alone would be misleading.
  const pts = [];
  for (let x = -2; x <= 3; x += 1) pts.push([x, x * x, 0]);
  const f = fitEllipse(pts);
  assert.equal(f.ok, true, f.detail);
  assert.ok(f.radiusX > 0 && f.radiusY > 0, 'a real, positive-radius ellipse');
  assert.ok([...f.center, ...f.xAxis, ...f.yAxis, f.radiusX, f.radiusY, f.worst, f.rms].every(Number.isFinite));
  // The extent of the data is ~9 units; a worst deviation on that order
  // is the honest "this is not an ellipse" signal.
  assert.ok(f.worst > 0.05, `worst ${f.worst} should be a real, visible miss`);

  // A PERFECTLY SYMMETRIC parabola sample is the harder case, and worth
  // recording rather than avoiding: its scatter matrix is genuinely
  // singular, so the fit refuses DEGENERATE_CONIC instead of returning an
  // ellipse at all. Both outcomes are honest — what never happens is a
  // hyperbola or a parabola coming back wearing `ok: true`.
  const sym = [];
  for (let x = -3; x <= 3; x += 0.5) sym.push([x, x * x, 0]);
  const s = fitEllipse(sym);
  assert.equal(s.ok, false);
  assert.equal(s.reason, FIT_REFUSAL.DEGENERATE_CONIC);
});

// ---------------------------------------------------------------------
// REFUSALS — every path, each with its own case.
// ---------------------------------------------------------------------
test('refusal: TOO_FEW_POINTS, per fit, at each fit own structural minimum', () => {
  const { points } = ellipseFixture();
  assert.equal(fitLine(points.slice(0, 1)).reason, FIT_REFUSAL.TOO_FEW_POINTS);
  assert.equal(fitPlane(points.slice(0, 2)).reason, FIT_REFUSAL.TOO_FEW_POINTS);
  assert.equal(fitCircle(points.slice(0, 2)).reason, FIT_REFUSAL.TOO_FEW_POINTS);
  assert.equal(fitEllipse(points.slice(0, 4)).reason, FIT_REFUSAL.TOO_FEW_POINTS);
  // And one MORE point than the minimum is accepted, so the boundary is
  // a real minimum and not an off-by-one refusing valid input.
  assert.equal(fitLine(points.slice(0, 2)).ok, true);
  assert.equal(fitPlane(points.slice(0, 3)).ok, true);
  assert.equal(fitEllipse(points.slice(0, 5)).ok, true);
  // Every refusal names itself in plain language, not just a code.
  const r = fitEllipse(points.slice(0, 4));
  assert.equal(r.ok, false);
  assert.ok(typeof r.detail === 'string' && r.detail.length > 0);
});

test('refusal: COINCIDENT_POINTS — enough points, too few DISTINCT ones', () => {
  const p = [1.5, -2.5, 7];
  const dup = [p, p.slice(), p.slice(), p.slice(), p.slice(), p.slice()];
  assert.equal(fitEllipse(dup).reason, FIT_REFUSAL.COINCIDENT_POINTS);
  assert.equal(fitCircle(dup).reason, FIT_REFUSAL.COINCIDENT_POINTS);
  assert.equal(fitPlane(dup).reason, FIT_REFUSAL.COINCIDENT_POINTS);
  assert.equal(fitLine(dup).reason, FIT_REFUSAL.COINCIDENT_POINTS);
  // Two genuinely distinct points among six duplicates: a line is
  // determined, a plane and a conic are not.
  const two = [p, p.slice(), p.slice(), [p[0] + 9, p[1] + 3, p[2] - 1], p.slice(), p.slice()];
  assert.equal(fitLine(two).ok, true);
  assert.equal(fitPlane(two).reason, FIT_REFUSAL.COINCIDENT_POINTS);
});

test('refusal: COLLINEAR — a line determines neither a plane nor a conic', () => {
  const col = [];
  for (let i = 0; i < 8; i++) col.push([2 * i - 3, 3 * i + 1, -i + 4]);
  assert.equal(fitPlane(col).reason, FIT_REFUSAL.COLLINEAR);
  assert.equal(fitCircle(col).reason, FIT_REFUSAL.COLLINEAR);
  assert.equal(fitEllipse(col).reason, FIT_REFUSAL.COLLINEAR);
  // fitLine is the one fit for which collinear input is the GOOD case.
  const lf = fitLine(col);
  assert.equal(lf.ok, true);
  assert.ok(lf.worst < 1e-12);
  // The refusal carries the measured off-line RMS and the tolerance it
  // was compared against, so a caller can loosen with real numbers.
  const r = fitPlane(col);
  assert.ok(Number.isFinite(r.lineRms) && Number.isFinite(r.collinearTol));
});

test('refusal: NOT_PLANAR — and the measured planar deviation comes back with it', () => {
  const { normal, points } = ellipseFixture();
  const off = points.map((p, i) => {
    const s = (i % 2 ? 0.5 : -0.5);
    return [p[0] + normal[0] * s, p[1] + normal[1] * s, p[2] + normal[2] * s];
  });
  const e = fitEllipse(off);
  assert.equal(e.reason, FIT_REFUSAL.NOT_PLANAR);
  assert.ok(Math.abs(e.planeWorst - 0.5) < 1e-9, `planeWorst ${e.planeWorst}`);
  assert.ok(Number.isFinite(e.planarTol));
  assert.equal(fitCircle(off).reason, FIT_REFUSAL.NOT_PLANAR);
  // A caller that supplies its own, looser tolerance gets a real fit —
  // the refusal is the caller's policy, not the module's opinion.
  const loose = fitEllipse(off, { planarTol: 2 });
  assert.equal(loose.ok, true, loose.detail);
  assert.ok(Math.abs(loose.planeWorst - 0.5) < 1e-9);
  assert.ok(loose.worst >= 0.5 - 1e-6, 'the out-of-plane miss is counted in the deviation');
});

test('refusal: DEGENERATE_CONIC — the last-resort guard genuinely fires', () => {
  // Reaching this needs the earlier COLLINEAR guard deliberately disabled
  // (collinearTol: 0), which is exactly the point: it is defense in
  // depth. A near-collinear set that slips past the geometric check still
  // refuses at the numerical one rather than returning whatever the
  // singular system produced.
  const near = [[0, 0, 0], [1, 1e-12, 0], [2, 0, 0], [3, -1e-12, 0], [4, 0, 0], [5, 1e-12, 0]];
  const r = fitEllipse(near, { collinearTol: 0, planarTol: 1e9 });
  assert.equal(r.ok, false);
  assert.equal(r.reason, FIT_REFUSAL.DEGENERATE_CONIC);
  assert.ok(typeof r.detail === 'string' && r.detail.length > 0);
});

test('refusal: NOT_FINITE — a NaN anywhere in the input refuses, it never leaks into a result', () => {
  const { points } = ellipseFixture();
  const bad = points.slice();
  bad[7] = [NaN, 1, 0];
  assert.equal(fitEllipse(bad).reason, FIT_REFUSAL.NOT_FINITE);
  assert.equal(fitCircle(bad).reason, FIT_REFUSAL.NOT_FINITE);
  assert.equal(fitPlane(bad).reason, FIT_REFUSAL.NOT_FINITE);
  assert.equal(fitLine(bad).reason, FIT_REFUSAL.NOT_FINITE);
  const inf = points.slice();
  inf[2] = [0, Infinity, 0];
  assert.equal(fitEllipse(inf).reason, FIT_REFUSAL.NOT_FINITE);
  // A malformed entry (not a point at all) refuses the same way.
  assert.equal(fitEllipse([...points.slice(0, 4), null]).reason, FIT_REFUSAL.NOT_FINITE);
  // No array at all.
  assert.equal(fitEllipse(undefined).reason, FIT_REFUSAL.TOO_FEW_POINTS);
});

// ---------------------------------------------------------------------
// DETERMINISM + THE PERFECT-CIRCLE CASE
// ---------------------------------------------------------------------
test('determinism: a near-degenerate input fitted twice returns byte-identical results', () => {
  const { normal, ex, ey } = obliqueFrame(31);
  // Radii that differ by one part in ~2e8 — the axis directions here are
  // numerically arbitrary, which is exactly where an unstable
  // implementation flickers.
  const crv = makeEllipse(CENTER, ex, ey, 20, 20.0000001, 4);
  const pts = sampleCurve(crv, 32);
  const a = fitEllipse(pts);
  const b = fitEllipse(pts);
  assert.equal(a.ok, true);
  assert.deepEqual(a, b);
  assert.equal(JSON.stringify(a), JSON.stringify(b));
  // Same for every other fit, on the same near-degenerate set.
  assert.deepEqual(fitCircle(pts), fitCircle(pts));
  assert.deepEqual(fitPlane(pts), fitPlane(pts));
  assert.deepEqual(fitLine(pts), fitLine(pts));
  // And the near-circular case is handled by the stated rule rather than
  // by eigen-noise: it reads as circular and the two radii agree exactly.
  assert.equal(a.circular, true);
  assert.equal(a.radiusX, a.radiusY);
  assert.ok(Math.abs(Math.abs(dot3(a.normal, normal)) - 1) < 1e-10);
});

test('a PERFECT circle handed to fitEllipse: circular flag, equal radii, canonical (not arbitrary) axes', () => {
  const { normal, ex, ey } = obliqueFrame(53);
  const R = 23;
  const pts = sampleCurve(makeCircle(CENTER, ex, ey, R, 4), 40);
  const f = fitEllipse(pts);
  assert.equal(f.ok, true, f.detail);
  assert.equal(f.circular, true);
  assert.equal(f.radiusX, f.radiusY, 'a circle must not report two different radii');
  assert.ok(Math.abs(f.radiusX - R) < 1e-10, `radius ${f.radiusX}`);
  assert.ok(dist3(f.center, CENTER) < 1e-10);
  assert.ok(f.worst < 1e-10, `worst ${f.worst}`);

  // The returned axes are the PLANE's own canonical basis (derived from
  // the normal alone), not whatever the conic eigen-solve happened to
  // produce — a circle's axes are genuinely arbitrary, so the module
  // returns a stable frame rather than an unstable measurement.
  const plane = fitPlane(pts);
  assert.deepEqual(f.xAxis, plane.xAxis);
  assert.ok(dist3(f.yAxis, cross3(f.normal, f.xAxis)) < 1e-14, 'yAxis must be cross(normal, xAxis)');

  // Rotating the SAME circle's construction frame in-plane must not
  // change the reported axes at all — the actual anti-flicker property.
  const other = obliqueFrame(11);
  const pts2 = sampleCurve(makeCircle(CENTER, other.ex, other.ey, R, 4), 40);
  const g = fitEllipse(pts2);
  assert.equal(g.circular, true);
  assert.ok(dist3(g.xAxis, f.xAxis) < 1e-12, 'a circle rebuilt in a rotated frame reported different axes');
  assert.ok(dist3(g.yAxis, f.yAxis) < 1e-12);
});

test('axis assignment is stable: radiusX is always the major axis, and the frame stays right-handed', () => {
  // Build the SAME ellipse twice, once with the two radii passed in the
  // opposite order (and the frame rotated 90 degrees to match), so the
  // shape is identical but the CONSTRUCTION labeled its axes the other
  // way round. The fit must report the same major axis either way.
  const { normal, ex, ey } = obliqueFrame(31);
  const a = fitEllipse(sampleCurve(makeEllipse(CENTER, ex, ey, RX, RY, 4), 60));
  const b = fitEllipse(sampleCurve(makeEllipse(CENTER, ey, [-ex[0], -ex[1], -ex[2]], RY, RX, 4), 60));
  assert.equal(a.ok, true); assert.equal(b.ok, true);
  assert.ok(Math.abs(a.radiusX - b.radiusX) < 1e-9, `${a.radiusX} vs ${b.radiusX}`);
  assert.ok(Math.abs(a.radiusY - b.radiusY) < 1e-9);
  assert.ok(a.radiusX > a.radiusY && b.radiusX > b.radiusY);
  assert.ok(dist3(a.xAxis, b.xAxis) < 1e-9, 'the same shape reported two different major axes');
  for (const f of [a, b]) {
    assert.ok(Math.abs(dot3(f.xAxis, f.yAxis)) < 1e-12);
    assert.ok(Math.abs(dot3(f.xAxis, f.normal)) < 1e-12);
    assert.ok(dist3(cross3(f.xAxis, f.yAxis), f.normal) < 1e-12, 'frame must be right-handed');
    assert.ok(Math.abs(Math.abs(dot3(f.normal, normal)) - 1) < 1e-10);
  }
});

// ---------------------------------------------------------------------
// FINITENESS + SHAPE OF EVERY RETURN
// ---------------------------------------------------------------------
test('every returned number is finite, on ordinary and on adversarial input alike', () => {
  const { points } = ellipseFixture();
  const cases = [
    points,
    points.slice(0, 5),
    points.map((p, i) => [p[0] + (i % 3) * 0.7, p[1] - (i % 5) * 0.4, p[2] + (i % 7) * 0.2]),
    [[0, 0, 0], [1, 0.0001, 0], [2, 0, 0], [3, -0.0001, 0], [4, 0, 0], [5, 0.0002, 0]],
    [[1e6, -1e6, 1e6], [1e6 + 1, -1e6, 1e6], [1e6, -1e6 + 1, 1e6], [1e6 + 1, -1e6 + 1, 1e6], [1e6 + 0.5, -1e6 + 2, 1e6]],
  ];
  for (const pts of cases) {
    const all = fitAll(pts, { planarTol: 1e9 });
    for (const [name, r] of Object.entries(all)) {
      assert.equal(typeof r.ok, 'boolean', name);
      if (!r.ok) {
        assert.ok(Object.values(FIT_REFUSAL).includes(r.reason), `${name}: unknown reason ${r.reason}`);
        assert.ok(typeof r.detail === 'string' && r.detail.length > 0, `${name}: no detail`);
        continue;
      }
      const nums = [];
      for (const v of Object.values(r)) {
        if (typeof v === 'number') nums.push(v);
        else if (Array.isArray(v)) nums.push(...v);
      }
      assert.ok(nums.every(Number.isFinite), `${name}: a non-finite number escaped: ${JSON.stringify(r)}`);
      assert.ok(r.worst >= 0 && r.rms >= 0, `${name}: negative deviation`);
      assert.ok(r.worst >= r.rms - 1e-12, `${name}: worst < rms`);
    }
  }
});

test('params come back as PLAIN DATA in exactly the operator param shape (never a class instance)', () => {
  // A hard constraint, and the one the param editor depends
  // on: a recovered param bag must be plain arrays/numbers, serializable
  // as-is, with no revival step.
  const { points } = ellipseFixture();
  const f = fitEllipse(points);
  assert.equal(f.ok, true);
  for (const key of ['center', 'normal', 'xAxis', 'yAxis']) {
    assert.ok(Array.isArray(f[key]), `${key} must be a plain array`);
    assert.equal(f[key].length, 3);
    assert.ok(f[key].every((v) => typeof v === 'number'));
  }
  for (const key of ['radiusX', 'radiusY', 'worst', 'rms']) assert.equal(typeof f[key], 'number');
  // Round-trips through JSON unchanged — the real test of "plain data".
  assert.deepEqual(JSON.parse(JSON.stringify(f)), f);
  // And it feeds the kernel constructor directly, no translation.
  const rebuilt = makeEllipse(f.center, f.xAxis, f.yAxis, f.radiusX, f.radiusY, 4);
  assert.ok(rebuilt.ctrlPts.flat().every(Number.isFinite));
  // No `segments` is invented — the fit describes the SHAPE, not the
  // representation the app happened to build it with.
  assert.equal('segments' in f, false);
});

test('fitAll runs every candidate and ranks none of them', () => {
  const { points } = ellipseFixture();
  const all = fitAll(points);
  assert.deepEqual(Object.keys(all).sort(), ['circle', 'ellipse', 'line', 'plane']);
  assert.equal(all.ellipse.ok, true);
  assert.equal(all.plane.ok, true);
  // A genuine ellipse is NOT a circle: the circle fit succeeds (the
  // points are planar and non-collinear) but its measured deviation is
  // large, which is exactly the evidence a caller needs to decide which
  // recipe to name.
  assert.equal(all.circle.ok, true);
  assert.ok(all.circle.worst > 5, `circle worst ${all.circle.worst} should be a real miss on a 41x17 ellipse`);
  assert.ok(all.ellipse.worst < 1e-10);
  // The line fit also "succeeds" and is also honestly terrible.
  assert.equal(all.line.ok, true);
  assert.ok(all.line.worst > 5);
  // Nothing in the result claims a winner.
  assert.equal('best' in all, false);
});
