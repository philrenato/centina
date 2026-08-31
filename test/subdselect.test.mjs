import test from 'node:test';
import assert from 'node:assert/strict';
import { edgeKey } from '../kernel/subd.mjs';
import { edgeLoopFromSeed, edgeRingFromSeed, oppositeEdgeInFace, faceLoopFromSeed, faceLoopDirections, nearestFaceEdgeToPoint } from '../kernel/subdselect.mjs';

// A plain, non-closed UxV quad grid (a flat "plane cage" — deliberately
// built with DIFFERENT column/row counts, cols=5/rows=3, so a loop and a
// ring from the same interior seed edge have genuinely different lengths,
// not just different edges by coincidence of a square grid). Mirrors
// kernel/subdprimitives.mjs's own superbPlaneCage topology exactly (a
// plain (cols+1)x(rows+1) vertex grid, quad faces in row-major order) but
// built directly here since that function forces cols===rows.
function buildGridCage(cols, rows) {
  const idx = (i, j) => j * (cols + 1) + i;
  const vertices = [];
  for (let j = 0; j <= rows; j++) for (let i = 0; i <= cols; i++) vertices.push([i, j, 0]);
  const faces = [];
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      faces.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
    }
  }
  return { vertices, faces, creases: {} };
}

// An OPEN tube (n sides, m height row-bands, m+1 rings of n vertices each,
// no top/bottom caps) — genuinely CLOSED circumferentially (every
// interior ring is a real cycle) but OPEN lengthwise (no cap faces), so a
// loop seeded on an interior ring's own circumferential edge can close
// into a real cycle while a ring seeded on the same edge stays open
// (terminates at the two real top/bottom boundaries) — the two mechanisms
// proven on the SAME fixture, genuinely exercising both a closed-loop and
// an open-ring result.
function buildOpenTubeCage(n, m, radius = 10, height = 20) {
  const idx = (r, s) => r * n + s;
  const vertices = [];
  for (let r = 0; r <= m; r++) {
    for (let s = 0; s < n; s++) {
      const a = (s / n) * Math.PI * 2;
      vertices.push([radius * Math.cos(a), radius * Math.sin(a), (r / m) * height]);
    }
  }
  const faces = [];
  for (let r = 0; r < m; r++) {
    for (let s = 0; s < n; s++) {
      const sNext = (s + 1) % n;
      faces.push([idx(r, s), idx(r, sNext), idx(r + 1, sNext), idx(r + 1, s)]);
    }
  }
  return { vertices, faces, creases: {} };
}

test('edgeLoopFromSeed: an interior circumferential edge on a plain grid loops across the WHOLE row (vertex-chase), terminating at both boundary columns', () => {
  const cage = buildGridCage(5, 3); // 6x4 vertex grid, faces 5x3
  const idx = (i, j) => j * 6 + i;
  const seed = edgeKey(idx(2, 1), idx(3, 1));
  const loop = edgeLoopFromSeed(cage, seed);
  assert.equal(loop.length, 5); // the whole row j=1 has exactly 5 horizontal edges (i=0..5)
  const expected = new Set([0, 1, 2, 3, 4].map((i) => edgeKey(idx(i, 1), idx(i + 1, 1))));
  assert.deepEqual(new Set(loop), expected);
});

test('edgeRingFromSeed: the SAME seed edge rings across faces instead (opposite-edge-in-quad chase), a genuinely DIFFERENT, disjoint-except-seed set', () => {
  const cage = buildGridCage(5, 3);
  const idx = (i, j) => j * 6 + i;
  const seed = edgeKey(idx(2, 1), idx(3, 1));
  const ring = edgeRingFromSeed(cage, seed);
  assert.equal(ring.length, 4); // 4 row-boundaries (j=0..3) at the same i=2..3 column
  const expected = new Set([0, 1, 2, 3].map((j) => edgeKey(idx(2, j), idx(3, j))));
  assert.deepEqual(new Set(ring), expected);
});

test('loop and ring from the same seed are genuinely different edge sets, sharing only the seed itself', () => {
  const cage = buildGridCage(5, 3);
  const idx = (i, j) => j * 6 + i;
  const seed = edgeKey(idx(2, 1), idx(3, 1));
  const loop = new Set(edgeLoopFromSeed(cage, seed));
  const ring = new Set(edgeRingFromSeed(cage, seed));
  assert.notEqual(loop.size, 0);
  assert.notEqual(ring.size, 0);
  assert.notDeepEqual(loop, ring); // genuinely different sets
  const intersection = [...loop].filter((k) => ring.has(k));
  assert.deepEqual(intersection, [seed]); // only the seed itself is shared
});

test('edgeLoopFromSeed CLOSES into a real cycle around a circumferential edge on an open tube (genuinely closed loop, not truncated at a fake boundary)', () => {
  const n = 6, m = 3;
  const cage = buildOpenTubeCage(n, m);
  const idx = (r, s) => r * n + s;
  const seed = edgeKey(idx(1, 0), idx(1, 1)); // ring r=1 is a real interior ring (touches bands 0 and 1, valence 4 everywhere)
  const loop = edgeLoopFromSeed(cage, seed);
  assert.equal(loop.length, n); // closes after exactly n edges, not n+1 (never re-adds the seed)
  const expected = new Set(Array.from({ length: n }, (_, s) => edgeKey(idx(1, s), idx(1, (s + 1) % n))));
  assert.deepEqual(new Set(loop), expected);
});

test('edgeRingFromSeed on the SAME tube edge stays OPEN (terminates at both real top/bottom boundaries), a genuinely different result from the closed loop', () => {
  const n = 6, m = 3;
  const cage = buildOpenTubeCage(n, m);
  const idx = (r, s) => r * n + s;
  const seed = edgeKey(idx(1, 0), idx(1, 1));
  const ring = edgeRingFromSeed(cage, seed);
  assert.equal(ring.length, m + 1); // one edge per ring (r=0..m), all at the same side-column s=0..1
  const expected = new Set(Array.from({ length: m + 1 }, (_, r) => edgeKey(idx(r, 0), idx(r, 1))));
  assert.deepEqual(new Set(ring), expected);
});

test('a circumferential edge sitting ON the tube\'s own OPEN BOUNDARY selects the whole boundary ring — the naked-edge continuation rule, so double-clicking an open edge behaves like double-clicking an interior one instead of selecting a lone edge', () => {
  const n = 6, m = 3;
  const cage = buildOpenTubeCage(n, m);
  const idx = (r, s) => r * n + s;
  const seed = edgeKey(idx(0, 0), idx(0, 1)); // ring r=0 IS the open boundary
  const loop = edgeLoopFromSeed(cage, seed);
  const expected = new Set(Array.from({ length: n }, (_, s) => edgeKey(idx(0, s), idx(0, (s + 1) % n))));
  assert.deepEqual(new Set(loop), expected);
  // ...and it stays ON the boundary: the tube's own OTHER open ring, and
  // every vertical edge running between the two, are genuinely excluded.
  const other = new Set(Array.from({ length: n }, (_, s) => edgeKey(idx(m, s), idx(m, (s + 1) % n))));
  for (const k of loop) assert.equal(other.has(k), false);
});

test('a boundary walk stops honestly at a PINCH POINT (two quads meeting at a single shared vertex — four naked edges there, no unique continuation) rather than guessing which way to turn', () => {
  // Two unit quads touching only at vertex 2. Every edge is naked; vertex 2
  // carries four of them.
  const cage = {
    vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 1, 0], [3, 1, 0], [3, 2, 0], [2, 2, 0]],
    faces: [[0, 1, 2, 3], [2, 4, 5, 6]],
    creases: {},
  };
  const loop = edgeLoopFromSeed(cage, edgeKey(0, 1));
  // Walks its own quad's border up to the pinch and stops there, both ways —
  // never crossing into the second quad.
  assert.equal(loop.includes(edgeKey(2, 4)), false);
  assert.equal(loop.includes(edgeKey(4, 5)), false);
});

test('oppositeEdgeInFace returns null for a non-quad face (ring has no defined continuation through an n-gon)', () => {
  const triCage = { vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1, 2]], creases: {} };
  assert.equal(oppositeEdgeInFace(triCage, 0, edgeKey(0, 1)), null);
});

test('edgeLoopFromSeed/edgeRingFromSeed both throw honestly for a seed key that is not a real edge of the cage', () => {
  const cage = buildGridCage(2, 2);
  assert.throws(() => edgeLoopFromSeed(cage, '999_1000'), /not a real edge/);
  assert.throws(() => edgeRingFromSeed(cage, '999_1000'), /not a real edge/);
});

test('a single-quad cage: the loop is that quad\'s own whole border (every edge is naked, so the boundary rule carries it all the way around); ring still finds the ONE opposite edge within that same quad, then stops (no further face to cross into)', () => {
  const cage = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], faces: [[0, 1, 2, 3]], creases: {} };
  const seed = edgeKey(0, 1);
  assert.deepEqual(new Set(edgeLoopFromSeed(cage, seed)), new Set([edgeKey(0, 1), edgeKey(1, 2), edgeKey(2, 3), edgeKey(3, 0)]));
  assert.deepEqual(new Set(edgeRingFromSeed(cage, seed)), new Set([seed, edgeKey(2, 3)]));
});

// ---------------------------------------------------------------
// FACE LOOP — the face-based sibling of the edge ring. Same walk,
// different thing collected, so these tests deliberately prove the two
// DIRECTIONS are genuinely different results (which is what makes the
// direction a real choice a caller has to make, not a detail).
// ---------------------------------------------------------------

test('faceLoopFromSeed: the two directions through one quad give genuinely DIFFERENT strips (a column vs a row), and the seed face is in both', () => {
  const cage = buildGridCage(5, 3); // faces row-major, face(i,j) = j*5 + i
  const seedFace = 1 * 5 + 2; // interior cell (i=2, j=1)
  const [dirA, dirB] = faceLoopDirections(cage, seedFace);
  assert.ok(dirA && dirB, 'a quad has two real loop directions');

  const colStrip = faceLoopFromSeed(cage, seedFace, dirA);
  const rowStrip = faceLoopFromSeed(cage, seedFace, dirB);

  // The grid is deliberately 5 wide x 3 tall, so a row and a column can
  // never coincide by accident of a square fixture.
  assert.deepEqual(new Set(colStrip), new Set([2, 7, 12]));       // whole column i=2
  assert.deepEqual(new Set(rowStrip), new Set([5, 6, 7, 8, 9]));  // whole row j=1
  assert.ok(colStrip.includes(seedFace) && rowStrip.includes(seedFace));
  // The ambiguity is REAL — this is exactly why a face-only caller has to
  // guess, and why guessing wrong has to be recoverable.
  assert.notDeepEqual(new Set(colStrip), new Set(rowStrip));
});

test('faceLoopFromSeed: a strip running around a genuinely CLOSED band closes at exactly n faces instead of circling forever', () => {
  const n = 8, m = 3;
  const cage = buildOpenTubeCage(n, m);
  const seedFace = 1 * n + 0; // interior ring r=1
  const [lengthwise, circumferential] = faceLoopDirections(cage, seedFace);

  const around = faceLoopFromSeed(cage, seedFace, circumferential);
  assert.equal(around.length, n, 'closes after exactly one lap, no repeats');
  assert.equal(new Set(around).size, n, 'no face collected twice');
  assert.deepEqual(new Set(around), new Set(Array.from({ length: n }, (_, s) => n + s)));

  // The SAME seed face, the other way, is open — it terminates at the
  // tube's two real boundaries rather than closing.
  const along = faceLoopFromSeed(cage, seedFace, lengthwise);
  assert.equal(along.length, m);
  assert.deepEqual(new Set(along), new Set([0, 8, 16]));
});

test('faceLoopFromSeed: terminates honestly at a boundary, and refuses malformed input by name rather than guessing', () => {
  const cage = buildGridCage(3, 1); // a single row of 3 faces, every face on the boundary
  const [dirA, dirB] = faceLoopDirections(cage, 0);
  // Across the row: the whole strip, ending at both open ends.
  assert.deepEqual(new Set(faceLoopFromSeed(cage, 0, dirB)), new Set([0, 1, 2]));
  // The other way there is nothing at all beyond the seed — both sides are boundary.
  assert.deepEqual(faceLoopFromSeed(cage, 0, dirA), [0]);

  assert.throws(() => faceLoopFromSeed(cage, 99, dirA), /not a real face/);
  assert.throws(() => faceLoopFromSeed(cage, 0, 'nope'), /not a real edge/);
  // A real edge of the cage, but not of THIS face — refused rather than
  // silently walked from somewhere the caller didn't ask about.
  const farEdge = faceLoopDirections(cage, 2)[1];
  assert.throws(() => faceLoopFromSeed(cage, 0, farEdge), /not an edge of face/);
});

test('faceLoopDirections: an n-gon has no well-defined pairing and says so, rather than inventing one', () => {
  const cage = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1.5, 0], [0, 1, 0]], faces: [[0, 1, 2, 3, 4]], creases: {} };
  assert.deepEqual(faceLoopDirections(cage, 0), []);
});

test('nearestFaceEdgeToPoint: the click position picks the direction — pressing near an edge selects the strip that CROSSES that edge', () => {
  const cage = buildGridCage(5, 3); // face(i,j) = j*5 + i, cells are unit squares
  const seedFace = 1 * 5 + 2; // cell spanning x 2..3, y 1..2

  // Press near the cell's RIGHT edge -> the strip runs left-right (a row).
  const rightish = nearestFaceEdgeToPoint(cage, seedFace, [2.95, 1.5, 0]);
  assert.deepEqual(new Set(faceLoopFromSeed(cage, seedFace, rightish)), new Set([5, 6, 7, 8, 9]));

  // Press near the LEFT edge -> the SAME row. Opposite edges are one
  // direction, so both presses agree, which is the behaviour that makes
  // this feel predictable rather than fiddly.
  const leftish = nearestFaceEdgeToPoint(cage, seedFace, [2.05, 1.5, 0]);
  assert.deepEqual(new Set(faceLoopFromSeed(cage, seedFace, leftish)), new Set([5, 6, 7, 8, 9]));

  // Press near the TOP edge -> the strip runs up-down (a column) instead.
  const topish = nearestFaceEdgeToPoint(cage, seedFace, [2.5, 1.95, 0]);
  assert.deepEqual(new Set(faceLoopFromSeed(cage, seedFace, topish)), new Set([2, 7, 12]));

  // The two presses genuinely disagree — the click is really steering.
  assert.notEqual(rightish, topish);
});

test('nearestFaceEdgeToPoint: a dead-centre press is a real tie, and resolves the SAME way every time rather than arbitrarily', () => {
  const cage = buildGridCage(3, 3);
  const centre = [1.5, 1.5, 0]; // exact centre of face (1,1)
  const first = nearestFaceEdgeToPoint(cage, 1 * 3 + 1, centre);
  for (let k = 0; k < 5; k++) {
    assert.equal(nearestFaceEdgeToPoint(cage, 1 * 3 + 1, centre), first, 'stable across repeated reads');
  }
  // Still a usable direction, not null — an ambiguous press yields a real
  // strip, and the explicit re-read control is what offers the other one.
  assert.equal(faceLoopFromSeed(cage, 1 * 3 + 1, first).length, 3);
});

test('nearestFaceEdgeToPoint: answers for an n-gon too (nearest of n), and refuses a face that does not exist', () => {
  const cage = { vertices: [[0, 0, 0], [2, 0, 0], [2, 2, 0], [1, 3, 0], [0, 2, 0]], faces: [[0, 1, 2, 3, 4]], creases: {} };
  assert.equal(nearestFaceEdgeToPoint(cage, 0, [1, -0.2, 0]), edgeKey(0, 1)); // just below the bottom edge
  assert.equal(nearestFaceEdgeToPoint(cage, 0, [2.2, 1, 0]), edgeKey(1, 2)); // just right of the right edge
  assert.throws(() => nearestFaceEdgeToPoint(cage, 9, [0, 0, 0]), /not a real face/);
});
