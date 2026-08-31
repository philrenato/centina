// kernel/text.mjs — glyph coverage raster -> closed NURBS outlines.
//
// The fixtures are SYNTHETIC coverage rasters rather than real font glyphs,
// deliberately: a node test cannot rasterise a typeface, and a shape whose
// exact answer is known by hand (a square annulus, an antialiased disc) is a
// stronger oracle than a letter whose "right" outline nobody can write down.
import assert from 'node:assert/strict';
import {
  glyphCoverageToContours, contourSignedArea, pointInContour,
  simplifyClosedContour, contourCornerIndices, layoutTextGlyphs,
  polylineCurve, buildTextCurves,
} from '../kernel/text.mjs';

let passed = 0;
function test(name, fn) { fn(); passed++; console.log('  ok  ' + name); }

const W = 140, H = 140;
function raster(fn) {
  const a = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) a[y * W + x] = fn(x, y);
  return a;
}
// 100x100 square with a 40x40 square hole — the shape of an "O" reduced to
// something whose area, corner count and winding are all known exactly.
const annulus = raster((x, y) => ((x >= 20 && x < 120 && y >= 20 && y < 120) && !(x >= 50 && x < 90 && y >= 50 && y < 90)) ? 1 : 0);
const EM = { transform: { sx: 1 / 100, sy: -1 / 100, tx: -0.2, ty: 1.2 }, minArea: 1e-5 };

test('an annulus traces as exactly two contours', () => {
  const cs = glyphCoverageToContours(annulus, W, H, EM);
  assert.equal(cs.length, 2);
});

test('the outer contour is CCW and the hole is CW', () => {
  const cs = glyphCoverageToContours(annulus, W, H, EM);
  const outer = cs.find((c) => c.outer), hole = cs.find((c) => !c.outer);
  assert.ok(outer && hole);
  assert.ok(contourSignedArea(outer.pts) > 0, 'outer counter-clockwise');
  assert.ok(contourSignedArea(hole.pts) < 0, 'hole clockwise');
});

test('the traced areas match the drawn areas', () => {
  const cs = glyphCoverageToContours(annulus, W, H, EM);
  const outer = cs.find((c) => c.outer), hole = cs.find((c) => !c.outer);
  // Not exactly 1.00 and 0.16: marching squares chamfers a right-angled corner
  // by half a cell diagonally, which costs 0.125 px^2 per corner — 0.00005 em^2
  // over four corners, and that is the number these come back short by. The
  // simplification pass then straightens the chamfer away, which is why the
  // fitted curve below still has exactly four corners rather than eight.
  assert.ok(Math.abs(Math.abs(outer.area) - 1.0) < 1e-4, `outer ${outer.area}`);
  assert.ok(Math.abs(Math.abs(hole.area) - 0.16) < 1e-4, `hole ${hole.area}`);
});

test('nesting is decided by containment, not by area sign', () => {
  const cs = glyphCoverageToContours(annulus, W, H, EM);
  const outer = cs.find((c) => c.outer), hole = cs.find((c) => !c.outer);
  assert.ok(pointInContour(hole.pts[0], outer.pts));
  assert.ok(!pointInContour(outer.pts[0], hole.pts));
});

test('a square simplifies to exactly four vertices and four corners', () => {
  const cs = glyphCoverageToContours(annulus, W, H, EM);
  for (const c of cs) {
    const s = simplifyClosedContour(c.pts, 0.0055);
    assert.equal(s.length, 4, 'four vertices');
    assert.equal(contourCornerIndices(s, 32).length, 4, 'four corners');
  }
});

test('a square annulus builds two exact degree-1 loops', () => {
  const cs = glyphCoverageToContours(annulus, W, H, EM);
  const built = buildTextCurves({ O: { contours: cs, advance: 1.1 } }, { text: 'O', size: 20, capHeight: 1, pixelSize: 1 / 100 });
  assert.equal(built.curves.length, 2);
  for (const c of built.curves) {
    assert.equal(c.crv.degree, 1, 'a straight-sided glyph needs no splines');
    assert.equal(c.crv.ctrlPts.length, 5, 'four corners plus the closing point');
    assert.equal(c.crv.knots.length, c.crv.ctrlPts.length + c.crv.degree + 1, 'well-formed knot vector');
    assert.ok(c.crv.ctrlPts.every((p) => p[3] === 1), 'non-rational');
  }
  assert.equal(built.report.polylineFallbacks, 0);
});

test('an antialiased disc is recognized as an exact rational conic', () => {
  const disc = raster((x, y) => Math.min(1, Math.max(0, 0.5 + (50 - Math.hypot(x - 70 + 0.5, y - 70 + 0.5)))));
  const cs = glyphCoverageToContours(disc, W, H, EM);
  assert.equal(cs.length, 1);
  // pi/4 for a radius-0.5-em disc; sub-pixel accurate because the coverage is
  // a real antialiased ramp, not a binary mask.
  assert.ok(Math.abs(Math.abs(cs[0].area) - Math.PI / 4) < 2e-4, `area ${cs[0].area}`);
  const built = buildTextCurves({ O: { contours: cs, advance: 1.1 } }, { text: 'O', size: 20, capHeight: 1, pixelSize: 1 / 100 });
  assert.equal(built.curves.length, 1);
  assert.equal(built.curves[0].crv.degree, 2);
  assert.ok(built.curves[0].crv.ctrlPts.some((p) => Math.abs(p[3] - Math.SQRT1_2) < 1e-6), 'carries the conic weights');
});

test('a contour running off the raster edge is dropped, never chorded shut', () => {
  const bleeding = raster((x, y) => (x >= 60 ? 1 : 0));
  const cs = glyphCoverageToContours(bleeding, W, H, EM);
  assert.equal(cs.length, 0);
});

test('layout advances by the font advance plus tracking, and applies kerning', () => {
  const metrics = { advance: { A: 0.6, V: 0.6 }, kern: { AV: -0.08 } };
  const plain = layoutTextGlyphs('AV', metrics, { tracking: 0 });
  assert.equal(plain.placements[0].x, 0);
  assert.ok(Math.abs(plain.placements[1].x - 0.52) < 1e-12, 'kerned pair pulls in');
  const tracked = layoutTextGlyphs('AV', metrics, { tracking: 0.1 });
  assert.ok(Math.abs(tracked.placements[1].x - 0.62) < 1e-12, 'tracking adds on top');
});

test('an unmeasurable pair falls back to zero kerning rather than a guess', () => {
  const l = layoutTextGlyphs('AV', { advance: { A: 0.6, V: 0.6 }, kern: {} }, {});
  assert.ok(Math.abs(l.placements[1].x - 0.6) < 1e-12);
});

test('a backslash-n escape is a line break and lines run downward', () => {
  const l = layoutTextGlyphs('A\\nA', { advance: { A: 0.6 }, kern: {} }, { lineSpacing: 1.5 });
  assert.equal(l.lineCount, 2);
  assert.equal(l.placements[0].y, 0);
  assert.equal(l.placements[1].y, -1.5);
});

test('center and right alignment shift the shorter line', () => {
  const m = { advance: { A: 0.6 }, kern: {} };
  const c = layoutTextGlyphs('AA\\nA', m, { align: 'center' });
  assert.ok(Math.abs(c.placements[2].x - 0.3) < 1e-12);
  const r = layoutTextGlyphs('AA\\nA', m, { align: 'right' });
  assert.ok(Math.abs(r.placements[2].x - 0.6) < 1e-12);
});

test('a character with no glyph is reported, not silently swallowed', () => {
  const l = layoutTextGlyphs('AxA', { advance: { A: 0.6 }, kern: {} }, {});
  assert.deepEqual(l.missing, ['x']);
  assert.equal(l.placements.length, 2);
});

test('the polyline fallback is a well-formed closed degree-1 curve', () => {
  const crv = polylineCurve([[0, 0, 0], [1, 0, 0], [1, 1, 0]], true);
  assert.equal(crv.degree, 1);
  assert.equal(crv.ctrlPts.length, 4);
  assert.equal(crv.knots.length, crv.ctrlPts.length + 2);
  assert.deepEqual(crv.ctrlPts[0], crv.ctrlPts[3]);
});

console.log(`\n${passed}/${passed} kernel/text.mjs tests passed.`);
