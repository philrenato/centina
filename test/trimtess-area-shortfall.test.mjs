// The triangulation safety net measures area, and a triangle count is not area.
//
// `tessellateTrimmedSurface` clips its trim polygon against every grid cell and
// ear-clips what is left. When the clipper leaves part of a cell uncovered the
// face has a hole in it, so the net that catches that has to be right in both
// directions: it must fire on lost area, and it must stay quiet on a
// polygon that triangulates correctly into fewer than n - 2 triangles.
//
// The second half is the one a triangle count gets wrong. A clip routinely
// leaves corners that are not true corners — three points along one cell edge, or
// a coincident pair — and a correct triangulation consumes those without a
// triangle each, so the count comes in short on a perfectly covered polygon.
//
// The measure is two-sided, and the second direction is not symmetry for its
// own sake: covering more than the polygon encloses is a triangle laid over
// ground the polygon excludes, which is what a keyhole bridge filled in by
// mistake looks like.
import test from 'node:test';
import assert from 'node:assert';
import { triangulatePolygon2D, triangulationAreaShortfall } from '../kernel/trimtess.mjs';

const area = (poly) => {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const q = poly[(i + 1) % poly.length];
    a += poly[i][0] * q[1] - q[0] * poly[i][1];
  }
  return Math.abs(a) / 2;
};

test('a clean triangulation reports no shortfall', () => {
  const square = [[0, 0], [4, 0], [4, 3], [0, 3]];
  const tris = triangulatePolygon2D(square);
  const cover = triangulationAreaShortfall(square, tris);
  assert.equal(cover.want, 12);
  assert.ok(cover.err < 1e-12, `covered fully, error ${cover.err}`);
});

test('collinear and coincident corners triangulate into fewer than n-2 — a count calls that a residual and the area does not', () => {
  // A corner planted mid-edge and a coincident pair beside it, on both long
  // sides: the shape a clip against a cell boundary leaves as a matter of
  // course, and one this ear-clipper covers exactly.
  const poly = [[0, 0], [2, 0], [2, 0], [4, 0], [4, 3], [2, 3], [2, 3], [0, 3]];
  const tris = triangulatePolygon2D(poly);
  assert.ok(tris.length < poly.length - 2,
    `the fixture does trip a triangle count — ${tris.length} triangles where a count expects ${poly.length - 2}`);
  const cover = triangulationAreaShortfall(poly, tris);
  assert.ok(cover.err <= cover.want * 1e-9 + 1e-15,
    `while the area it covers matches the area it encloses — ${cover.got} against ${cover.want}`);
});

test('and it still fires on lost area', () => {
  const square = [[0, 0], [4, 0], [4, 3], [0, 3]];
  const tris = triangulatePolygon2D(square);
  assert.ok(tris.length >= 2, 'the fixture has something to drop');
  const dropped = tris.slice(0, tris.length - 1);
  const cover = triangulationAreaShortfall(square, dropped);
  assert.ok(cover.err > 0, `a dropped triangle is reported as an area error (${cover.err})`);
  const [i, j, k] = tris[tris.length - 1];
  const lost = area([square[i], square[j], square[k]]);
  assert.ok(Math.abs(cover.err - lost) < 1e-9,
    `and the amount reported is exactly that triangle's own area — ${cover.err} against ${lost}`);
});
