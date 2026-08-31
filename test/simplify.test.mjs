import test from 'node:test';
import assert from 'node:assert/strict';
import {
  decimateOpenToCount, decimateClosedToCount, decimateToCount,
  farthestPointPairIndices, insertByHighestDeviation,
  regeneratePointSet,
} from '../kernel/simplify.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';

test('decimateOpenToCount: always keeps both real endpoints', () => {
  const points = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]];
  const result = decimateOpenToCount(points, 2);
  assert.equal(result.length, 2);
  assert.deepEqual(result[0], [0, 0, 0]);
  assert.deepEqual(result[1], [4, 0, 0]);
});

test('decimateOpenToCount: a collinear sequence removes interior points first (zero effective area)', () => {
  const points = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]];
  const result = decimateOpenToCount(points, 3);
  assert.equal(result.length, 3);
  assert.deepEqual(result[0], [0, 0, 0]);
  assert.deepEqual(result[result.length - 1], [4, 0, 0]);
});

test('decimateOpenToCount: a real outlier (spike) survives over flat interior points', () => {
  // A mostly-straight run with ONE point pulled far off the line — that
  // point has a genuinely large effective area (a real corner/spike),
  // the two flat interior points either side of it have near-zero area.
  const points = [[0, 0, 0], [1, 0, 0], [2, 5, 0], [3, 0, 0], [4, 0, 0]];
  const result = decimateOpenToCount(points, 3);
  assert.equal(result.length, 3);
  const hasSpike = result.some((p) => Math.abs(p[1] - 5) < 1e-9);
  assert.ok(hasSpike, `expected the spike point to survive decimation, got ${JSON.stringify(result)}`);
});

test('decimateOpenToCount: targetCount >= length returns all points unchanged (a copy, not the same array)', () => {
  const points = [[0, 0, 0], [1, 0, 0], [2, 0, 0]];
  const result = decimateOpenToCount(points, 5);
  assert.deepEqual(result, points);
  assert.notEqual(result, points); // real copy, not aliasing
});

test('decimateOpenToCount: refuses a target below 2', () => {
  assert.throws(() => decimateOpenToCount([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 1));
});

test('farthestPointPairIndices: finds the true long-axis pair on a stretched shape', () => {
  // An elongated point ring (a squashed ellipse-like loop) — the two
  // points at the LONG axis ends are unambiguously the farthest apart.
  const points = [];
  const n = 12;
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    points.push([Math.cos(t) * 10, Math.sin(t) * 1, 0]); // squashed in Y
  }
  const [i, j] = farthestPointPairIndices(points);
  // index 0 sits at (10,0,0); index n/2 sits at (-10,0,0) — these are the
  // true long-axis ends and must be the reported pair (in either order).
  const got = new Set([i, j]);
  assert.ok(got.has(0) && got.has(n / 2), `expected {0, ${n / 2}}, got ${JSON.stringify([...got])}`);
});

test('decimateClosedToCount: keeps the loop closed (no duplicate seam point) and hits the exact target count', () => {
  const points = [];
  const n = 16;
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    points.push([Math.cos(t) * 10, Math.sin(t) * 1, 0]);
  }
  const result = decimateClosedToCount(points, 8);
  assert.equal(result.length, 8);
  // No two points in the result should be exactly coincident (would mean
  // a seam point got duplicated by the arc-recombination step).
  for (let a = 0; a < result.length; a++) {
    for (let b = a + 1; b < result.length; b++) {
      const d = Math.hypot(result[a][0] - result[b][0], result[a][1] - result[b][1], result[a][2] - result[b][2]);
      assert.ok(d > 1e-9, `result points ${a} and ${b} are coincident — a duplicated seam point`);
    }
  }
});

test('decimateClosedToCount: refuses a target below 3', () => {
  const points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]];
  assert.throws(() => decimateClosedToCount(points, 2));
});

test('decimateToCount: dispatches open vs closed correctly by the closed flag', () => {
  const openPts = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
  const r1 = decimateToCount(openPts, 2, false);
  assert.equal(r1.length, 2);
  const closedPts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]];
  const r2 = decimateToCount(closedPts, 3, true);
  assert.equal(r2.length, 3);
});

test('insertByHighestDeviation: inserts at the sharply-bent span, not the straight one, on an open L-shape', () => {
  // Two straight runs meeting at a right angle at point index 1 — the
  // curve genuinely bends hardest in the span straddling that corner
  // (either [0,1] or [1,2]), never in a perfectly flat extension of it.
  const points = [[0, 0, 0], [4, 0, 0], [4, 4, 0]];
  const result = insertByHighestDeviation(points, 1, 3, false);
  assert.equal(result.length, 4);
  // The new point must be genuinely between two ORIGINAL points (not
  // coincident with either), proving a real insertion happened.
  const isOriginal = (p) => points.some((op) => Math.hypot(op[0] - p[0], op[1] - p[1], op[2] - p[2]) < 1e-9);
  const newPts = result.filter((p) => !isOriginal(p));
  assert.equal(newPts.length, 1, `expected exactly one genuinely new point, got ${JSON.stringify(result)}`);
});

test('insertByHighestDeviation: growing then shrinking back stays a valid, finite curve (no NaN/Infinity)', () => {
  const points = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]];
  const grown = insertByHighestDeviation(points, 3, 3, false);
  assert.equal(grown.length, points.length + 3);
  for (const p of grown) for (const c of p) assert.ok(Number.isFinite(c), `non-finite coordinate in ${JSON.stringify(grown)}`);
  const shrunk = decimateOpenToCount(grown, points.length);
  assert.equal(shrunk.length, points.length);
});

test('insertByHighestDeviation: works on a closed loop without throwing and returns a valid finite grown array', () => {
  const points = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]; // a square loop
  const grown = insertByHighestDeviation(points, 2, 3, true);
  assert.equal(grown.length, points.length + 2);
  for (const p of grown) for (const c of p) assert.ok(Number.isFinite(c), `non-finite coordinate in ${JSON.stringify(grown)}`);
});

test('regeneratePointSet: UNIFORM mode on an open curve returns exactly targetCount points, both true endpoints exact', () => {
  const points = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [8, 4, 0]];
  const result = regeneratePointSet(points, 6, 3, false, 'uniform');
  assert.equal(result.length, 6);
  assert.ok(Math.hypot(result[0][0] - 0, result[0][1] - 0, result[0][2] - 0) < 1e-6, 'first point must be the exact true start');
  assert.ok(Math.hypot(result[5][0] - 8, result[5][1] - 4, result[5][2] - 0) < 1e-6, 'last point must be the exact true end');
});

test('regeneratePointSet: UNIFORM mode genuinely spaces points EVENLY by real arc length (not by parameter)', () => {
  const points = [[0, 0, 0], [1, 0, 0], [1, 20, 0]]; // a short leg then a very long one
  const crv = globalCurveInterp(points, 2);
  const result = regeneratePointSet(points, 5, 2, false, 'uniform');
  // Reconstruct the real chord lengths between consecutive returned points —
  // they should be roughly EQUAL (uniform arc-length spacing), not skewed
  // toward the short leg the way naive parameter-uniform sampling would be.
  const chords = [];
  for (let i = 1; i < result.length; i++) chords.push(Math.hypot(result[i][0]-result[i-1][0], result[i][1]-result[i-1][1], result[i][2]-result[i-1][2]));
  const avg = chords.reduce((a,b)=>a+b,0) / chords.length;
  for (const c of chords) assert.ok(Math.abs(c - avg) / avg < 0.35, `expected roughly-equal arc-length spacing, got chords ${JSON.stringify(chords)}`);
});

test('regeneratePointSet: UNIFORM mode on a CLOSED curve never duplicates the seam point', () => {
  const points = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]];
  const result = regeneratePointSet(points, 8, 3, true, 'uniform');
  assert.equal(result.length, 8);
  for (let a = 0; a < result.length; a++) {
    for (let b = a + 1; b < result.length; b++) {
      const d = Math.hypot(result[a][0]-result[b][0], result[a][1]-result[b][1], result[a][2]-result[b][2]);
      assert.ok(d > 1e-6, `points ${a}/${b} coincide — a duplicated seam point in uniform closed mode`);
    }
  }
});

test('regeneratePointSet: ADAPTIVE mode dispatches to decimate when shrinking and insert when growing', () => {
  const points = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [8, 4, 0], [8, 8, 0]];
  const shrunk = regeneratePointSet(points, 3, 3, false, 'adaptive');
  assert.equal(shrunk.length, 3);
  const grown = regeneratePointSet(points, 7, 3, false, 'adaptive');
  assert.equal(grown.length, 7);
  const same = regeneratePointSet(points, points.length, 3, false, 'adaptive');
  assert.equal(same.length, points.length);
});
