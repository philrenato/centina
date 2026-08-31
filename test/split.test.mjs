import test from 'node:test';
import assert from 'node:assert/strict';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { surfacePoint, isFiniteNet, surfaceClosure } from '../kernel/surface.mjs';
import { splitSurface, c0KnotParams, splitSurfaceAtC0Lines, surfaceCreaseParams } from '../kernel/split.mjs';

// A genuinely non-trivial fixture, matching this project's own "avoid
// too-simple test geometry" lesson: a real curved profile (not a
// straight line), revolved through a PARTIAL sweep (270 deg, not a full
// 360) so BOTH the U (profile) and V (rotation) directions are OPEN —
// splittable in either direction on the same fixture, and genuinely
// rational/curved in both.
function partialRevolve() {
  const profile = globalCurveInterp([[10, 0, 0], [16, 0, 20], [12, 0, 45], [18, 0, 70]], 3);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, (3 * Math.PI) / 2);
}

function fullRevolve() {
  const profile = globalCurveInterp([[10, 0, 0], [16, 0, 20], [12, 0, 45]], 2);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}

test('splitSurface: splitting along U reproduces the ORIGINAL surface exactly at every one of its own real stations', () => {
  const srf = partialRevolve();
  const uMid = (srf.knotsU[0] + srf.knotsU[srf.knotsU.length - 1]) / 2;
  const { first, second } = splitSurface(srf, 'u', uMid);
  assert.equal(isFiniteNet(first.ctrlNet), true);
  assert.equal(isFiniteNet(second.ctrlNet), true);

  for (const half of [first, second]) {
    for (const u of half.uStations) {
      for (const v of half.vStations) {
        const expected = surfacePoint(srf, u, v);
        const actual = surfacePoint(half, u, v);
        const err = Math.hypot(actual[0] - expected[0], actual[1] - expected[1], actual[2] - expected[2]);
        assert.ok(err < 1e-6, `split half should reproduce the original surface EXACTLY at its own real station (u=${u},v=${v}), got error ${err}`);
      }
    }
  }
});

test('splitSurface: between-station points are a close, honest approximation (not exact, stated as such elsewhere in this kernel)', () => {
  const srf = partialRevolve();
  const uMid = (srf.knotsU[0] + srf.knotsU[srf.knotsU.length - 1]) / 2;
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const { first, second } = splitSurface(srf, 'u', uMid);
  for (const [half, lo, hi] of [[first, uMin, uMid], [second, uMid, uMax]]) {
    for (let i = 1; i < 4; i++) {
      const u = lo + (hi - lo) * (i / 4); // deliberately off-station
      const v = half.vStations[2];
      const expected = surfacePoint(srf, u, v);
      const actual = surfacePoint(half, u, v);
      const err = Math.hypot(actual[0] - expected[0], actual[1] - expected[1], actual[2] - expected[2]);
      assert.ok(err < 0.01, `between-station approximation should still be visually close (u=${u},v=${v}), got error ${err}`);
    }
  }
});

test('splitSurface: the two halves share a byte-identical boundary control-point row (no gap)', () => {
  const srf = partialRevolve();
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const uSplit = uMin + (uMax - uMin) * 0.37; // an off-center split, not a coincidental midpoint
  const { first, second } = splitSurface(srf, 'u', uSplit);
  const firstBoundaryRow = first.ctrlNet[first.ctrlNet.length - 1];
  const secondBoundaryRow = second.ctrlNet[0];
  assert.equal(firstBoundaryRow.length, secondBoundaryRow.length);
  for (let i = 0; i < firstBoundaryRow.length; i++) {
    for (let c = 0; c < 4; c++) {
      assert.equal(firstBoundaryRow[i][c], secondBoundaryRow[i][c],
        `boundary control point ${i}, component ${c} must be BYTE-IDENTICAL between halves`);
    }
  }
});

test('splitSurface: splitting along V (the other direction) also reproduces the original exactly at its own real stations', () => {
  const srf = partialRevolve();
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const vSplit = vMin + (vMax - vMin) * 0.6;
  const { first, second } = splitSurface(srf, 'v', vSplit);
  assert.equal(isFiniteNet(first.ctrlNet), true);
  assert.equal(isFiniteNet(second.ctrlNet), true);
  for (const half of [first, second]) {
    for (const u of half.uStations) {
      for (const v of half.vStations) {
        const expected = surfacePoint(srf, u, v);
        const actual = surfacePoint(half, u, v);
        const err = Math.hypot(actual[0] - expected[0], actual[1] - expected[1], actual[2] - expected[2]);
        assert.ok(err < 1e-6, `V-split half should reproduce the original EXACTLY at its own real station (u=${u},v=${v}), got error ${err}`);
      }
    }
  }
});

test('splitSurface: refuses a split parameter at or beyond either domain end', () => {
  const srf = partialRevolve();
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  assert.throws(() => splitSurface(srf, 'u', uMin), /strictly interior/);
  assert.throws(() => splitSurface(srf, 'u', uMax), /strictly interior/);
  assert.throws(() => splitSurface(srf, 'u', uMin - 1), /strictly interior/);
});

test('splitSurface: refuses splitting a CLOSED direction honestly', () => {
  const srf = fullRevolve();
  const { closedV } = surfaceClosure(srf);
  assert.equal(closedV, true, 'fixture sanity check: a full 360 revolve must be closed in V');
  const vMid = (srf.knotsV[0] + srf.knotsV[srf.knotsV.length - 1]) / 2;
  assert.throws(() => splitSurface(srf, 'v', vMid), /CLOSED/);
});

test('splitSurface: refuses an invalid direction string', () => {
  const srf = partialRevolve();
  assert.throws(() => splitSurface(srf, 'w', 0.5), /must be 'u' or 'v'/);
});

// ── SPLITTING AT A CREASE THE SURFACE ALREADY HAS ───────────────────────────
// Distinct from splitSurface above in both method and scope: this cuts only at
// knots whose multiplicity reaches the degree, which the surface is already
// only C0 across, so the cut is EXACT (knot insertion, no sampling, no fitting)
// and it works on a direction that is CLOSED, which splitSurface refuses.
test('a crease split is EXACT — each piece is the original surface, not a re-fit of it', () => {
  // A closed square profile extruded: the "one side surface" an extrude really
  // produces, whose four corners are creases INSIDE a single face.
  const P = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const srf = {
    degU: 1, knotsU: [0, 0, 1, 2, 3, 4, 4], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: P.map((p) => [[p[0], p[1], 0, 1], [p[0], p[1], 20, 1]]),
  };
  assert.deepEqual(c0KnotParams(srf.knotsU, srf.degU), [1, 2, 3]);
  const pieces = splitSurfaceAtC0Lines(srf, 'u');
  assert.equal(pieces.length, 4, 'four creases-worth of profile gives four faces');
  let worst = 0;
  for (const pc of pieces) {
    const lo = pc.knotsU[0], hi = pc.knotsU[pc.knotsU.length - 1];
    for (let t = 0; t <= 12; t++) {
      const u = lo + (hi - lo) * (t / 12);
      for (const v of [0, 0.37, 1]) {
        const a = surfacePoint(pc, u, v), b = surfacePoint(srf, u, v);
        worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
      }
    }
  }
  // Not "within tolerance" — identical. A sampled re-fit could not make this
  // claim, which is the whole reason this path exists beside splitSurface.
  assert.equal(worst, 0, `a crease split must be exact, worst deviation ${worst}`);
});

test('a SMOOTH join is not a crease, and splitting there would invent an edge', () => {
  // Interior knot at multiplicity 1 on a cubic: C2, not a crease.
  const srf = {
    degU: 3, knotsU: [0, 0, 0, 0, 0.5, 1, 1, 1, 1], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [[0, 0], [1, 0], [2, 1], [3, 0], [4, 0]].map((p) => [[p[0], p[1], 0, 1], [p[0], p[1], 1, 1]]),
  };
  assert.deepEqual(c0KnotParams(srf.knotsU, srf.degU), []);
  assert.equal(splitSurfaceAtC0Lines(srf, 'u').length, 1, 'returns the surface untouched so a caller can apply it unconditionally');
  assert.equal(splitSurfaceAtC0Lines(srf, 'u')[0], srf, 'and returns it by identity, not a copy');
});

test('the crease split preserves WEIGHTS — a rational surface stays rational', () => {
  // Two quarter-circle-ish rational spans meeting at a full-multiplicity knot.
  const w = Math.SQRT1_2;
  const srf = {
    degU: 2, knotsU: [0, 0, 0, 1, 1, 2, 2, 2], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [
      [[1, 0, 0, 1], [1, 0, 5, 1]], [[1, 1, 0, w], [1, 1, 5, w]], [[0, 1, 0, 1], [0, 1, 5, 1]],
      [[-1, 1, 0, w], [-1, 1, 5, w]], [[-1, 0, 0, 1], [-1, 0, 5, 1]],
    ],
  };
  assert.deepEqual(c0KnotParams(srf.knotsU, srf.degU), [1]);
  // This fixture is SMOOTH across that knot (see the circle test below), so the
  // combinatorial split has to be asked for by name.
  const pieces = splitSurfaceAtC0Lines(srf, 'u', { allCandidates: true });
  assert.equal(pieces.length, 2);
  assert.ok(pieces.some((p) => p.ctrlNet.some((row) => row.some((pt) => Math.abs(pt[3] - w) < 1e-12))),
    'the non-unit weight survives the split — dropping it would silently straighten the arc');
  let worst = 0;
  for (const pc of pieces) {
    const lo = pc.knotsU[0], hi = pc.knotsU[pc.knotsU.length - 1];
    for (let t = 0; t <= 12; t++) {
      const u = lo + (hi - lo) * (t / 12);
      const a = surfacePoint(pc, u, 0.5), b = surfacePoint(srf, u, 0.5);
      worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    }
  }
  assert.ok(worst < 1e-12, `rational split must stay exact, got ${worst}`);
});

test('a NURBS circle is NOT a crease — multiplicity == degree is a candidate, not a verdict', () => {
  // The counterexample that matters. A standard circle is degree 2 with every
  // quarter-point knot at multiplicity 2, and perfectly smooth across all of
  // them because the control legs meeting there are collinear. Trusting the
  // knot vector alone shatters every cylinder into four faces and invents eight
  // edges that do not exist.
  const w = Math.SQRT1_2, R = 10;
  const cp = [[R, 0], [R, R], [0, R], [-R, R], [-R, 0], [-R, -R], [0, -R], [R, -R], [R, 0]];
  const ws = [1, w, 1, w, 1, w, 1, w, 1];
  const cyl = {
    degU: 2, knotsU: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: cp.map((p, i) => [[p[0], p[1], 0, ws[i]], [p[0], p[1], 25, ws[i]]]),
  };
  assert.equal(c0KnotParams(cyl.knotsU, cyl.degU).length, 3, 'three interior knots DO reach full multiplicity');
  assert.deepEqual(surfaceCreaseParams(cyl, 'u'), [], 'and not one of them is a real tangent break');
  assert.equal(splitSurfaceAtC0Lines(cyl, 'u').length, 1, 'so the cylinder stays one face');
});

test('a crease is found even where it is sharp at only one end of the edge', () => {
  // A prism whose profile kink CLOSES toward the top: sharp at v=0, straight at
  // v=1. Sampling only the middle station would call this smooth for half the
  // surfaces it is wrong about.
  const srf = {
    degU: 1, knotsU: [0, 0, 1, 2, 2], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [
      [[0, 0, 0, 1], [0, 0, 10, 1]],
      [[10, 0, 0, 1], [10, 0, 10, 1]],
      [[20, 10, 0, 1], [20, 0, 10, 1]],
    ],
  };
  assert.deepEqual(surfaceCreaseParams(srf, 'u'), [1], 'the kink at one end is enough to make it an edge');
});

test('a POLE is not a crease, and the answer cannot depend on where the object was built', () => {
  // A flat circular cap: one whole control row collapsed to the centre. The
  // one-sided differences there are not a direction at all — at the origin they
  // come out exactly zero, and far from it they come out as float noise at that
  // magnitude, which has a uniformly random direction. The same disc must
  // answer identically wherever it sits.
  const w = Math.SQRT1_2, R = 20;
  const rim = [[R, 0], [R, R], [0, R], [-R, R], [-R, 0], [-R, -R], [0, -R], [R, -R], [R, 0]];
  const ws = [1, w, 1, w, 1, w, 1, w, 1];
  const discAt = (ox, oy) => ({
    degU: 1, knotsU: [0, 0, 1, 1], degV: 2, knotsV: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4],
    // U row 0 is the collapsed centre; U row 1 is the rim.
    ctrlNet: [
      rim.map((_, i) => [ox, oy, 0, ws[i]]),
      rim.map((p, i) => [ox + p[0] * ws[i], oy + p[1] * ws[i], 0, ws[i]].map((c, d) => (d < 2 ? (d === 0 ? ox + p[0] : oy + p[1]) : c))),
    ],
  });
  for (const [ox, oy] of [[0, 0], [400, 40], [-1e5, 7e4]]) {
    const d = discAt(ox, oy);
    assert.equal(c0KnotParams(d.knotsV, d.degV).length, 3, 'the circle direction does have full-multiplicity knots');
    assert.deepEqual(surfaceCreaseParams(d, 'v'), [], `a disc at (${ox}, ${oy}) has no crease — the pole must not read as one`);
  }
});
