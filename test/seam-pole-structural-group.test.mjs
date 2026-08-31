// Editing a control point on a seam tears
// the surface. surfaceStructuralGroup(srf, i, j) must find every OTHER
// (row, col) index that is the SAME physical point (a closed-direction
// seam pair, or a revolve pole's whole collapsed row), so a control-point
// write path can move all of them together and never open a gap that
// wasn't really there. Proven against real revolve()/extrude() surfaces
// (never a hand-faked control net), the same fixtures primitives.test.mjs
// already uses to prove surfaceClosure/nakedEdgeCount.
import test from 'node:test';
import assert from 'node:assert/strict';
import { surfaceClosure, nakedEdgeCount, surfaceStructuralGroup } from '../kernel/surface.mjs';
import { makeCircle, revolve, extrude } from '../kernel/primitives.mjs';

test('surfaceStructuralGroup: a cylinder (extrude of a closed circle) — closed in U (the profile), open in V — pairs row 0 with row (nu-1) at every column, and reports nothing for an interior row', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2); // radius 2 — a full circle is always >= 4 arc spans (MAX_ARC_SPAN), 9 control points
  const srf = extrude(profile, [0, 0, 1], 5);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true);
  assert.equal(closedV, false);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  assert.equal(nv, 2, 'a straight extrude has exactly 2 V-columns (bottom, top)');
  for (let j = 0; j < nv; j++) {
    const g0 = surfaceStructuralGroup(srf, 0, j);
    assert.equal(g0.length, 1, `row 0 col ${j} should have exactly one seam partner`);
    assert.deepEqual(g0[0], { i: nu - 1, j });
    const gLast = surfaceStructuralGroup(srf, nu - 1, j);
    assert.equal(gLast.length, 1);
    assert.deepEqual(gLast[0], { i: 0, j });
  }
  // a genuinely non-U-boundary row: seg=2 gives 9 rows (0..8), row 4 is interior
  assert.ok(nu > 2, 'fixture must have an interior row to test');
  const mid = Math.floor(nu / 2);
  assert.notEqual(mid, 0);
  assert.notEqual(mid, nu - 1);
  for (let j = 0; j < nv; j++) {
    assert.deepEqual(surfaceStructuralGroup(srf, mid, j), [], 'an interior row has no seam partner and is not a pole (only 2 columns, not coincident)');
  }
});

test('surfaceStructuralGroup: a full revolve of a closed profile (a torus-shaped case) — closed in BOTH directions — a corner pairs with all 3 of its own transitive identifications', () => {
  const minorProfile = makeCircle([5, 0, 0], [1, 0, 0], [0, 0, 1], 1); // closed minor circle in the X/Z plane, offset from the revolve axis
  const srf = revolve(minorProfile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true, 'the minor profile itself is closed');
  assert.equal(closedV, true, 'a full 2*PI sweep is closed too — genuinely closed in both directions');
  assert.equal(nakedEdgeCount(srf), 0);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  const corner = surfaceStructuralGroup(srf, 0, 0);
  // (0,0) identifies with (nu-1,0) via closedU, (0,nv-1) via closedV, and
  // (nu-1,nv-1) transitively via both — exactly 3 other index pairs, the
  // real "rectangle with opposite edges identified" torus topology, not
  // just a single one-hop pairing.
  const asSet = new Set(corner.map((p) => `${p.i}|${p.j}`));
  assert.equal(asSet.size, 3, `expected exactly 3 transitive partners, got ${JSON.stringify(corner)}`);
  assert.ok(asSet.has(`${nu - 1}|0`));
  assert.ok(asSet.has(`0|${nv - 1}`));
  assert.ok(asSet.has(`${nu - 1}|${nv - 1}`));
  // a genuinely interior, non-seam, non-pole point has no partner at all
  const midI = Math.floor(nu / 2), midJ = Math.floor(nv / 2);
  assert.notEqual(midI, 0); assert.notEqual(midI, nu - 1);
  assert.notEqual(midJ, 0); assert.notEqual(midJ, nv - 1);
  assert.deepEqual(surfaceStructuralGroup(srf, midI, midJ), []);
});

test('surfaceStructuralGroup: a full revolve of an OPEN profile touching the axis at BOTH ends (a sphere-like pole-to-pole case) — each pole row collapses ALL its own columns into one group, and undoing that never welds the two DIFFERENT poles together', () => {
  // A profile from one pole (on-axis) to the other (on-axis) via one real
  // off-axis point in between — revolve()'s own 1e-9 coincidence check
  // (r < 1e-9) fires on both ends, matching an hourglass/sphere silhouette.
  const profile = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[0, 0, 10, 1], [10, 0, 0, Math.SQRT1_2], [0, 0, -10, 1]] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  assert.equal(nu, 3, 'degree-2, 3-control-point profile');
  // row 0 (the top pole) collapses into one group spanning every OTHER
  // column (the function excludes the queried point itself, matching
  // polyCurveJointSiblings/findCageMirrorPartners' own "every OTHER
  // index" contract) — plus its own closedV seam partner at column
  // 0/last, but since the whole row is already one point, that seam
  // pairing is subsumed into the same set.
  const topPoleGroup = surfaceStructuralGroup(srf, 0, 0);
  const topSet = new Set([...topPoleGroup.map((p) => `${p.i}|${p.j}`), '0|0']);
  for (let j = 0; j < nv; j++) assert.ok(topSet.has(`0|${j}`), `top pole group missing column ${j}`);
  assert.equal(topSet.size, nv, 'top pole group is exactly its own row, no more');
  // the BOTTOM pole (row nu-1) is a physically DIFFERENT point — must never
  // appear in the top pole's own group.
  for (let j = 0; j < nv; j++) assert.ok(!topSet.has(`${nu - 1}|${j}`), 'the two distinct poles must never be welded together');
  // symmetric check from the bottom pole's own perspective
  const bottomPoleGroup = surfaceStructuralGroup(srf, nu - 1, 0);
  const bottomSet = new Set([...bottomPoleGroup.map((p) => `${p.i}|${p.j}`), `${nu - 1}|0`]);
  assert.equal(bottomSet.size, nv);
  for (let j = 0; j < nv; j++) assert.ok(bottomSet.has(`${nu - 1}|${j}`));
  // the real off-axis equator row (row 1) is closed in V (a full sweep)
  // but NOT a pole (its own columns are genuinely distinct points).
  const equatorGroup = surfaceStructuralGroup(srf, 1, 0);
  assert.equal(equatorGroup.length, 1, 'the equator row is closedV-paired at its own seam only, not a pole');
  assert.deepEqual(equatorGroup[0], { i: 1, j: nv - 1 });
});

test('surfaceStructuralGroup: an interior pole (an hourglass profile touching the axis at a MIDDLE control point, not an endpoint) is caught too, not just a boundary row', () => {
  // 5-point profile: off-axis, off-axis, ON-AXIS (interior), off-axis, off-axis —
  // the exact "hourglass/goblet" case this codebase's own subdconvert history
  // names as a real reachable pole that isn't at row 0 or row (nu-1).
  const profile = {
    degree: 2, knots: [0, 0, 0, 0.5, 1, 1, 1],
    ctrlPts: [[6, 0, 0, 1], [9, 0, 5, 1], [0, 0, 10, 1], [9, 0, 15, 1], [6, 0, 20, 1]],
  };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  const mid = 2; // the on-axis control point, dead center of a 5-point profile
  assert.ok(mid > 0 && mid < nu - 1, 'the pole must genuinely be an interior row for this test to mean anything');
  const g = surfaceStructuralGroup(srf, mid, 0);
  const set = new Set(g.map((p) => `${p.i}|${p.j}`));
  assert.equal(set.size, nv - 1, 'every OTHER column of the interior pole row, none of any other row');
  for (let j = 0; j < nv; j++) if (j !== 0) assert.ok(set.has(`${mid}|${j}`));
  for (const p of g) assert.equal(p.i, mid, 'must never reach into a neighboring, non-pole row');
});

test('surfaceStructuralGroup: a partial (non-closed) revolve has no seam pairing anywhere', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[3, 0, 0, 1], [3, 0, 10, 1]] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 1.5 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false);
  assert.equal(closedV, false);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    assert.deepEqual(surfaceStructuralGroup(srf, i, j), [], `(${i},${j}) should have no structural partner on an open surface`);
  }
});
