// Text outlines, measured as geometry — kernel/text.mjs end to end, over
// traced glyphs.
//
// Kept apart from text.test.mjs, whose assertions about Text read a record:
// a curve count, a corner count, a `closed` flag a builder wrote as a
// literal. None of them looks at the shape. A closed least-squares fit that
// ran a sixth of a lap past its own start and retraced the bottom of every
// round letter would pass all of them, because the only thing that can catch
// it is the length of the curve against the length of the contour it fits.
//
// So the assertions here are all about the curve as a curve: is it as long as
// the thing it fits, does it cross itself, do its two ends actually meet. Length
// is the one that carries the most, because it is the only cheap quantity an
// excursion cannot hide from: deviation is measured at the samples, and a curve
// that swings clear of the data between two samples moves no sample at all.
//
// The glyphs are banked, deliberately. There is no font file in this project;
// outlines come out of a browser rasterizer, which a node test does not have
// and which would in any case give a different typeface on a different
// machine. `test/fixtures/glyph_contours_sans_220.json` holds traced
// outlines for a pangram's alphabet at the shipped default of 220 px/em, so
// these are measurements of a real face and the same ones on every
// machine. Regenerating it is a deliberate act: the numbers below are stated
// against these outlines.
//
// An alphabet of cornered letters cannot see half of this. `contourToCurve`
// sends a contour with detected corners down the corner-split chain of open
// fits and a contour with none down a single closed fit, and the two branches
// fail differently: the closed one by retracing past its own seam, the open one
// by a span swinging clear of a long straight leg that carries no sample to
// hold it down. "E" has no cornerless contour at all. The counts asserted
// below pin how many contours here reach each branch, so a change that stops
// exercising one fails instead of passing unnoticed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { buildTextCurves, contourToCurve, simplifyClosedContour, glyphCoverageToContours } from '../kernel/text.mjs';
import { curvePoint, isCurveClosed } from '../kernel/curve.mjs';
import { curveSelfIntersects } from '../kernel/selfintersect.mjs';

const ROOT = path.resolve(import.meta.dirname, '..');
const BANK = JSON.parse(readFileSync(path.join(ROOT, 'test/fixtures/glyph_contours_sans_220.json'), 'utf8'));
const PANGRAM = 'The quick brown fox jumps over the lazy dog';
const SIZE = 20;

const glyphs = Object.fromEntries(Object.entries(BANK.glyphs).map(([ch, g]) => [ch, {
  advance: g.advance,
  contours: g.contours.map((c) => {
    const pts = [];
    for (let i = 0; i < c.xy.length; i += 2) pts.push([c.xy[i], c.xy[i + 1]]);
    return { outer: c.outer, pts };
  }),
}]));

function build(text, opts = {}) {
  return buildTextCurves(glyphs, { text, size: SIZE, capHeight: BANK.capHeight, pixelSize: 1 / BANK.em, ...opts });
}

// The polygon buildTextCurves itself fitted — the same scale and the same
// simplification, so the length comparison below is against the curve's own
// input rather than against a differently prepared copy of it.
function sourcePolygon(ch, contourIndex, opts = {}) {
  const scale = SIZE / BANK.capHeight;
  const eps = (1 / BANK.em) * scale * (opts.simplifyPixels ?? 0.55);
  const flat = glyphs[ch].contours[contourIndex].pts.map(([x, y]) => [x * scale, y * scale]);
  return simplifyClosedContour(flat, eps);
}

function polygonPerimeter(pts) {
  let L = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    L += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return L;
}

function polygonDiagonal(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  return Math.hypot(x1 - x0, y1 - y0);
}

// Sampled arc length. 4000 segments over a letter 20mm tall is a chord error
// far below anything asserted here, and it under-reports rather than over-,
// so a curve that fails the length bound is at least that long.
function arcLength(crv, n = 4000) {
  const p = crv.degree, U = crv.knots;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  let L = 0, prev = curvePoint(crv, t0);
  for (let i = 1; i <= n; i++) {
    const c = curvePoint(crv, t0 + ((t1 - t0) * i) / n);
    L += Math.hypot(c[0] - prev[0], c[1] - prev[1], c[2] - prev[2]);
    prev = c;
  }
  return L;
}

function ctrlGap(crv) {
  const a = crv.ctrlPts[0], b = crv.ctrlPts[crv.ctrlPts.length - 1];
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// Every measurement one loop can be judged on, in one place, so the tests
// below differ only in which population they run it over.
function measure(built) {
  const out = [];
  for (const c of built.curves) {
    const poly = sourcePolygon(c.char, c.contour);
    const si = curveSelfIntersects(c.crv, {});
    out.push({
      label: `${c.char}#${c.contour}`,
      kind: c.kind,
      lengthRatio: arcLength(c.crv) / polygonPerimeter(poly),
      gapRatio: ctrlGap(c.crv) / polygonDiagonal(poly),
      selfIntersects: si.selfIntersects,
      tested: si.tested,
      reason: si.reason,
    });
  }
  return out;
}

// A loop may not be longer than the thing it fits. This is the direct
// statement that a fit neither retraces nor wanders: a curve threading a
// polygon is slightly longer than that polygon (the polygon is inscribed in the
// shape both are measurements of), by a fraction of a percent at this sampling
// — measured at 1.0013 to 1.0041 across this bank, over both branches. 1.02
// leaves an order of magnitude of headroom over the measured excess and still
// catches a closed retrace (which starts at about 1.15) and an open span
// swinging off a long straight leg (which reached 1.51 on this bank).
const LENGTH_BOUND = 1.02;
// The two ends of a closed loop must meet, not merely come close. Relative to
// the glyph's own diagonal, so it means the same thing at any cap height.
const CLOSURE_BOUND = 1e-9;

test('both fit branches are reached by the fixture', () => {
  const rows = measure(build(PANGRAM));
  const nurbs = rows.filter((r) => r.kind === 'nurbs');
  const spans = rows.filter((r) => r.kind !== 'nurbs');
  // Every assertion below is quantified over both populations, so a fixture
  // that stopped entering one of them would leave half of this file true and
  // meaningless.
  assert.equal(nurbs.length, 13, `${PANGRAM.length} characters give 13 cornerless contours (got ${nurbs.length})`);
  assert.equal(spans.length, 37, `and 37 corner-split ones (got ${spans.length})`);
  for (const ch of ['O', 'o', 'g']) {
    const one = measure(build(ch)).filter((r) => r.kind === 'nurbs');
    assert.ok(one.length > 0, `"${ch}" must have at least one cornerless contour to be a useful fixture`);
  }
  for (const ch of ['m', 'j', 't']) {
    const one = measure(build(ch)).filter((r) => r.kind !== 'nurbs');
    assert.ok(one.length > 0, `"${ch}" must have at least one corner-split contour to be a useful fixture`);
  }
});

test('no loop is longer than its own contour — neither branch, the retrace and excursion assertion', () => {
  const bad = [];
  for (const text of ['O', 'o', 'g', 'e', 'Type', PANGRAM]) {
    for (const r of measure(build(text))) {
      if (r.lengthRatio > LENGTH_BOUND) bad.push(`${text}:${r.label} kind=${r.kind} arc/perimeter=${r.lengthRatio.toFixed(4)}`);
    }
  }
  assert.deepEqual(bad, [], `loops longer than the contour they fit:\n  ${bad.join('\n  ')}`);
});

test('no loop crosses itself, on either branch, and "not tested" is not a pass', () => {
  const bad = [], untested = [];
  for (const text of ['O', 'o', 'g', 'e', 'Type', PANGRAM]) {
    for (const r of measure(build(text))) {
      if (!r.tested) untested.push(`${text}:${r.label} (${r.reason})`);
      else if (r.selfIntersects) bad.push(`${text}:${r.label} kind=${r.kind}`);
    }
  }
  assert.deepEqual(untested, [], `self-intersection could not be judged for:\n  ${untested.join('\n  ')}`);
  assert.deepEqual(bad, [], `loops that cross themselves:\n  ${bad.join('\n  ')}`);
});

test('a closed fit closes — first and last control point, not a flag', () => {
  const bad = [];
  for (const text of ['O', 'o', 'g', 'e', 'Type', PANGRAM]) {
    for (const r of measure(build(text))) {
      if (r.gapRatio > CLOSURE_BOUND) bad.push(`${text}:${r.label} kind=${r.kind} gap/diagonal=${r.gapRatio.toExponential(2)}`);
    }
  }
  assert.deepEqual(bad, [], `loops whose two ends do not meet:\n  ${bad.join('\n  ')}`);
});

// The three hardest loops in this bank, named, so the general assertions above
// cannot pass by accident on an easier population. All three come from the
// corner-split open fit and each is a different way for a span to leave its own
// data — which is the failure a deviation measure structurally cannot see,
// because deviation is measured at the samples and an excursion happens between
// them:
//   - "t" span 4->9: six points, five control points, and the last leg nine
//     units long with nothing on it. Under chord-length parametrization the fit
//     comes back 1.57x its own polyline.
//   - "j" span 9->12: four points, three of them a fraction of a unit apart and
//     the fourth seventeen units away, through the interpolation last resort —
//     2.10x its own polyline, fourteen units clear of a straight run.
//   - "m" span 14->25: an arch whose two long straight legs carry no interior
//     sample at all, so the fit overshoots its own end corner and crosses the
//     baseline segment that starts there.
// Asserted as shape, not as a count: each of them must be clean, and each must
// stay inside the same length bound every other loop is held to.
test('the three hardest loops in this bank are clean — named, because a population average would hide them', () => {
  const rows = measure(build(PANGRAM));
  const wanted = ['j#0', 'm#0', 't#0'];
  const found = wanted.map((label) => {
    const r = rows.find((row) => row.label === label);
    assert.ok(r, `${label} must be in the pangram's own loops to be a fixture`);
    assert.notEqual(r.kind, 'nurbs', `${label} must reach the corner-split branch (kind ${r.kind})`);
    return r;
  });
  const crossing = found.filter((r) => !r.tested || r.selfIntersects).map((r) => r.label);
  assert.deepEqual(crossing, [], `these must not cross themselves:\n  ${crossing.join('\n  ')}`);
  const long = found.filter((r) => r.lengthRatio > LENGTH_BOUND)
    .map((r) => `${r.label} arc/perimeter=${r.lengthRatio.toFixed(4)}`);
  assert.deepEqual(long, [], `and must not wander off their own contour:\n  ${long.join('\n  ')}`);
});

test('one detected corner is one span all the way round, not a degree-1 polyline', () => {
  const ring = Array.from({ length: 20 }, (_, i) => {
    const t = (i / 20) * Math.PI * 2;
    return [10 * Math.cos(t), 10 * Math.sin(t), 0];
  });
  const one = contourToCurve(ring, [5], { tolerance: 0.5 });
  const two = contourToCurve(ring, [5, 12], { tolerance: 0.5 });
  assert.equal(one.kind, 'spans', `one corner must still take the corner-split branch (got ${one.kind})`);
  assert.equal(one.crv.degree, two.crv.degree, 'and produce the same degree of curve as two corners do');
  assert.equal(one.spanCount, 1, 'as a single span running the whole loop');
  assert.ok(isCurveClosed(one.crv, 1e-9), 'whose two ends meet at the corner it was split on');
  assert.ok(!curveSelfIntersects(one.crv, {}).selfIntersects, 'and which does not cross itself');
  // The polyline fallback is the floor under a fit that refuses, not somewhere
  // a working contour lands by accident.
  const built = build(PANGRAM);
  assert.ok(built.report.polylineFallbacks <= 1,
    `at most one contour of the pangram may fall back to a polyline (got ${built.report.polylineFallbacks})`);
});

test('a traced contour is a ring, without its start repeated', () => {
  // A filled disc on a coverage grid — the smallest thing that produces a
  // closed traced contour.
  const W = 40, H = 40, cov = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) cov[y * W + x] = Math.hypot(x - 19.5, y - 19.5) < 14 ? 1 : 0;
  }
  const contours = glyphCoverageToContours(cov, W, H, { threshold: 0.5 });
  assert.equal(contours.length, 1, 'a disc traces as one contour');
  const pts = contours[0].pts;
  const first = pts[0], last = pts[pts.length - 1];
  assert.ok(Math.hypot(first[0] - last[0], first[1] - last[1]) > 1e-12,
    `the ring must not repeat its own start (first ${first}, last ${last})`);
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    assert.ok(Math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-12, `no zero-length edge (index ${i})`);
  }
  // And the same, over every glyph in the bank, not only the synthetic case.
  for (const [ch, g] of Object.entries(BANK.glyphs)) {
    for (let i = 0; i < g.contours.length; i++) {
      const xy = g.contours[i].xy;
      assert.ok(Math.hypot(xy[0] - xy[xy.length - 2], xy[1] - xy[xy.length - 1]) > 1e-12,
        `${ch}#${i} repeats its own start`);
    }
  }
});

test('the winding survives all of it — outers counter-clockwise, counters clockwise', () => {
  const built = build(PANGRAM);
  const signedArea = (crv) => {
    const p = crv.degree, U = crv.knots;
    const t0 = U[p], t1 = U[U.length - 1 - p];
    let a = 0, prev = curvePoint(crv, t0);
    for (let i = 1; i <= 720; i++) {
      const c = curvePoint(crv, t0 + ((t1 - t0) * i) / 720);
      a += prev[0] * c[1] - c[0] * prev[1];
      prev = c;
    }
    return a / 2;
  };
  for (const c of built.curves) {
    const a = signedArea(c.crv);
    if (c.outer) assert.ok(a > 0, `${c.char}#${c.contour} is an outer loop and must run counter-clockwise (area ${a.toFixed(3)})`);
    else assert.ok(a < 0, `${c.char}#${c.contour} is a counter and must run clockwise (area ${a.toFixed(3)})`);
  }
});
