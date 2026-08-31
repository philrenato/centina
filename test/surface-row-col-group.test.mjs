// Structured U/V control-point row/column
// selection. surfaceRowGroup(srf, i0)/surfaceColGroup(srf, j0) must return
// every (row, col) index belonging to the requested row/column, expanded
// through surfaceStructuralGroup so a CLOSED direction's seam (and any
// pole) is always seam-complete, never a torn half-selection. Proven
// against real revolve()/extrude() surfaces, the same fixtures
// seam-pole-structural-group.test.mjs already uses.
import test from 'node:test';
import assert from 'node:assert/strict';
import { surfaceClosure, surfaceRowGroup, surfaceColGroup, surfaceCellCounts, surfaceCellStrip, nearestCellDirection } from '../kernel/surface.mjs';
import { makeCircle, revolve, extrude } from '../kernel/primitives.mjs';

function asSet(list) { return new Set(list.map((p) => `${p.i}|${p.j}`)); }

test('surfaceRowGroup: a cylinder (extrude of a closed circle, closed in U, open in V) — row 0 pulls in row (nu-1) too, an interior row stays alone', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2);
  const srf = extrude(profile, [0, 0, 1], 5);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true);
  assert.equal(closedV, false);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;

  const row0 = surfaceRowGroup(srf, 0);
  const expect0 = asSet([...Array(nv)].flatMap((_, j) => [{ i: 0, j }, { i: nu - 1, j }]));
  assert.equal(asSet(row0).size, expect0.size, `row 0 seam-complete: got ${row0.length} entries`);
  for (const k of expect0) assert.ok(asSet(row0).has(k), `row0 group missing ${k}`);

  const rowLast = surfaceRowGroup(srf, nu - 1);
  assert.deepEqual(asSet(rowLast), expect0, 'row (nu-1) resolves to the IDENTICAL seam-complete set as row 0');

  const mid = Math.floor(nu / 2);
  assert.notEqual(mid, 0); assert.notEqual(mid, nu - 1);
  const rowMid = surfaceRowGroup(srf, mid);
  const expectMid = asSet([...Array(nv)].map((_, j) => ({ i: mid, j })));
  assert.deepEqual(asSet(rowMid), expectMid, 'an interior row is just its own row, no seam pulled in');
});

test('surfaceColGroup: the SAME cylinder — every column is open (V is not closed), so a column group is always just that one column', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2);
  const srf = extrude(profile, [0, 0, 1], 5);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  for (let j = 0; j < nv; j++) {
    const col = surfaceColGroup(srf, j);
    const expect = asSet([...Array(nu)].map((_, i) => ({ i, j })));
    assert.deepEqual(asSet(col), expect, `column ${j} should be exactly its own ${nu} rows, none from a nonexistent seam`);
  }
});

test('surfaceRowGroup/surfaceColGroup: a torus (revolve of a closed profile, closed in BOTH directions) — a corner row AND corner column both resolve to the full 4-way seam identification', () => {
  const minorProfile = makeCircle([5, 0, 0], [1, 0, 0], [0, 0, 1], 1);
  const srf = revolve(minorProfile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true);
  assert.equal(closedV, true);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;

  // the corner (0,0) is identified with (nu-1,0), (0,nv-1), (nu-1,nv-1) —
  // so row 0 must ALSO drag in row (nu-1), fully, because column 0 and
  // column (nv-1) of row (nu-1) are themselves seam-identified with row 0.
  const row0 = surfaceRowGroup(srf, 0);
  const rowSet = asSet(row0);
  for (let j = 0; j < nv; j++) {
    assert.ok(rowSet.has(`0|${j}`), `row0 group missing (0,${j})`);
    assert.ok(rowSet.has(`${nu - 1}|${j}`), `row0 group missing seam-paired (${nu - 1},${j})`);
  }
  assert.equal(rowSet.size, 2 * nv, 'row0 group is exactly the union of both physical rows, nothing extra');

  const col0 = surfaceColGroup(srf, 0);
  const colSet = asSet(col0);
  for (let i = 0; i < nu; i++) {
    assert.ok(colSet.has(`${i}|0`), `col0 group missing (${i},0)`);
    assert.ok(colSet.has(`${i}|${nv - 1}`), `col0 group missing seam-paired (${i},${nv - 1})`);
  }
  assert.equal(colSet.size, 2 * nu, 'col0 group is exactly the union of both physical columns, nothing extra');

  // an interior row/column has no seam partner at all.
  const midI = Math.floor(nu / 2), midJ = Math.floor(nv / 2);
  assert.notEqual(midI, 0); assert.notEqual(midI, nu - 1);
  assert.notEqual(midJ, 0); assert.notEqual(midJ, nv - 1);
  assert.equal(surfaceRowGroup(srf, midI).length, nv, 'an interior row stays exactly its own row');
  assert.equal(surfaceColGroup(srf, midJ).length, nu, 'an interior column stays exactly its own column');
});

test('surfaceRowGroup: a full revolve of an open profile touching the axis at BOTH ends (a sphere-like pole-to-pole case) — the pole row group is exactly its own row (already seam-complete by being one physical point), never bleeds into the other pole', () => {
  const profile = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[0, 0, 10, 1], [10, 0, 0, Math.SQRT1_2], [0, 0, -10, 1]] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  const topGroup = surfaceRowGroup(srf, 0);
  const topSet = asSet(topGroup);
  assert.equal(topSet.size, nv, 'the pole row group is exactly its own row');
  for (let j = 0; j < nv; j++) assert.ok(topSet.has(`0|${j}`));
  for (let j = 0; j < nv; j++) assert.ok(!topSet.has(`${nu - 1}|${j}`), 'the two distinct poles must never be welded together via row-select');
});

test('surfaceRowGroup/surfaceColGroup: a partial (non-closed) revolve — every row/column group is exactly its own row/column, no seam pairing anywhere', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[3, 0, 0, 1], [3, 0, 10, 1]] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 1.5 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false);
  assert.equal(closedV, false);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  for (let i = 0; i < nu; i++) assert.equal(surfaceRowGroup(srf, i).length, nv, `row ${i} should be exactly its own ${nv} columns`);
  for (let j = 0; j < nv; j++) assert.equal(surfaceColGroup(srf, j).length, nu, `col ${j} should be exactly its own ${nu} rows`);
});

// ---------------------------------------------------------------
// FACE (ISOCURVE-CELL) CHAINS — the ordinary-surface counterpart of a
// SubD face loop. Same fixtures, so the closed-direction behaviour is
// proven against a surface genuinely known to be closed rather than a
// hand-built grid that only looks like one.
// ---------------------------------------------------------------

test('surfaceCellStrip: a cell strip runs the full width or height of the cell grid, and the two directions genuinely differ', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2);
  const srf = extrude(profile, [0, 0, 1], 5);
  const [uCells, vCells] = surfaceCellCounts(srf);
  assert.ok(uCells > 1 && vCells >= 1, 'fixture has a real cell grid to walk');

  const alongU = surfaceCellStrip(srf, 1, 0, 'u');
  const alongV = surfaceCellStrip(srf, 1, 0, 'v');
  assert.equal(alongU.length, uCells);
  assert.equal(alongV.length, vCells);
  // Every cell in a 'u' strip shares the seed's j; every cell in a 'v'
  // strip shares its i. That is the whole claim of a strip.
  assert.ok(alongU.every((c) => c.j === 0));
  assert.ok(alongV.every((c) => c.i === 1));
  assert.ok(alongU.some((c) => c.i === 1) && alongV.some((c) => c.j === 0), 'the seed cell is in both');
});

test('surfaceCellStrip: a CLOSED direction needs no seam pairing — cells sit between Grevilles, so there is no duplicated cell to pair', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2);
  const srf = extrude(profile, [0, 0, 1], 5); // closed in U
  assert.equal(surfaceClosure(srf).closedU, true);
  const [uCells] = surfaceCellCounts(srf);
  const strip = surfaceCellStrip(srf, 0, 0, 'u');
  // Exactly one cell per column, no coincident duplicate — the contrast
  // with surfaceRowGroup on this SAME closed fixture, which must pull in
  // the seam's second copy.
  assert.equal(strip.length, uCells);
  assert.equal(new Set(strip.map((c) => c.i)).size, uCells);
  assert.ok(surfaceRowGroup(srf, 0).length > srf.ctrlNet[0].length, 'control-point rows DO pair at the seam, cells do not');
});

test('surfaceCellStrip: refuses an out-of-range cell or an unknown direction by name', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2);
  const srf = extrude(profile, [0, 0, 1], 5);
  const [uCells, vCells] = surfaceCellCounts(srf);
  assert.throws(() => surfaceCellStrip(srf, uCells, 0, 'u'), /outside this surface/);
  assert.throws(() => surfaceCellStrip(srf, 0, vCells, 'u'), /outside this surface/);
  assert.throws(() => surfaceCellStrip(srf, 0, 0, 'w'), /must be 'u' or 'v'/);
});

test('nearestCellDirection: the press position picks the direction, compared in CELL FRACTIONS so an unequal domain scale cannot decide it', () => {
  // Near a constant-u edge (left or right) -> the strip that crosses it.
  assert.equal(nearestCellDirection(0.02, 0.5), 'u');
  assert.equal(nearestCellDirection(0.98, 0.5), 'u');
  // Near a constant-v edge (bottom or top) -> the other strip.
  assert.equal(nearestCellDirection(0.5, 0.02), 'v');
  assert.equal(nearestCellDirection(0.5, 0.98), 'v');
  // Opposite edges agree, so there is no thin band where a few pixels
  // flip the answer — it only changes across the cell's own diagonal.
  assert.equal(nearestCellDirection(0.1, 0.5), nearestCellDirection(0.9, 0.5));
  // A dead-centre press is a real tie and resolves the SAME way every
  // time, so re-reading one press cannot appear to do nothing.
  const centre = nearestCellDirection(0.5, 0.5);
  for (let k = 0; k < 5; k++) assert.equal(nearestCellDirection(0.5, 0.5), centre);
});
