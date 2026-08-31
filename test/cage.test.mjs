import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bernstein, makeRestLattice, latticeLocalCoords, evaluateLattice,
  deformPoint, pointInsideLatticeBounds, deformWithBands,
  bandInfluenceWeight, CAGE_BOUNDARY_FALLOFF_MARGIN, smoothstep,
  ringIndicesAlongAxis, applyRingTransform, rotatePointAboutAxis,
  scaleRadially, twistRingTransform, taperRingTransform, shearRingTransform,
} from '../kernel/cage.mjs';
import { add, scale } from '../kernel/vec3.mjs';

function lerp(a, b, t) { return a + (b - a) * t; }

test('bernstein basis: values at s=0 and s=1 match the defining endpoint property (all weight on i=0 / i=n)', () => {
  for (const n of [1, 2, 3, 4]) {
    for (let i = 0; i <= n; i++) {
      assert.ok(Math.abs(bernstein(n, i, 0) - (i === 0 ? 1 : 0)) < 1e-12, `n=${n} i=${i} s=0`);
      assert.ok(Math.abs(bernstein(n, i, 1) - (i === n ? 1 : 0)) < 1e-12, `n=${n} i=${i} s=1`);
    }
  }
});

test('bernstein basis partitions unity at an arbitrary interior s (real, not endpoint-only)', () => {
  for (const n of [1, 2, 3, 5]) {
    for (const s of [0.1, 0.37, 0.5, 0.83]) {
      let sum = 0;
      for (let i = 0; i <= n; i++) sum += bernstein(n, i, s);
      assert.ok(Math.abs(sum - 1) < 1e-10, `n=${n} s=${s} sum=${sum}`);
    }
  }
});

test('latticeLocalCoords: the 8 corners of an axis-aligned box map to exactly (0/1,0/1,0/1)', () => {
  const lattice = makeRestLattice([0, 0, 0], [10, 20, 30], 3, 4, 2);
  const corners = [
    [0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30],
    [10, 20, 0], [10, 0, 30], [0, 20, 30], [10, 20, 30],
  ];
  const expected = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ];
  for (let i = 0; i < corners.length; i++) {
    const [s, t, u] = latticeLocalCoords(lattice, corners[i]);
    assert.ok(Math.abs(s - expected[i][0]) < 1e-9 && Math.abs(t - expected[i][1]) < 1e-9 && Math.abs(u - expected[i][2]) < 1e-9,
      `corner ${JSON.stringify(corners[i])} -> got (${s},${t},${u}), expected ${JSON.stringify(expected[i])}`);
  }
});

// THE LOAD-BEARING IDENTITY this whole module leans on: a Bernstein/Bezier
// interpolation of a perfectly linear (unmoved) control net reproduces the
// exact linear map, at ANY degree/density combination, not just a
// trilinear (density-2) special case. Proven directly here, not assumed —
// deformPoint on an untouched rest lattice must return the query point
// exactly, for points well inside AND at the box's own boundary, across
// several different densities.
test('an UNDEFORMED lattice (rest control points, untouched) maps every point to itself exactly — real degree/density generalization, not just trilinear', () => {
  for (const [du, dv, dw] of [[2, 2, 2], [3, 2, 2], [3, 4, 5], [6, 6, 6]]) {
    const lattice = makeRestLattice([-5, -5, -5], [5, 5, 5], du, dv, dw);
    const testPoints = [
      [0, 0, 0], [-5, -5, -5], [5, 5, 5], [2.3, -1.7, 4.1], [-4.9, 4.9, -0.1], [1, 1, 1],
    ];
    for (const p of testPoints) {
      const out = deformPoint(lattice, p);
      assert.ok(Math.hypot(out[0] - p[0], out[1] - p[1], out[2] - p[2]) < 1e-8,
        `density ${du}x${dv}x${dw}, point ${JSON.stringify(p)} -> got ${JSON.stringify(out)} (expected identity)`);
    }
  }
});

test('moving ONE corner control point produces a real, smooth, localized bulge — exact at that corner, decaying toward the opposite corner, not a rigid whole-lattice shift', () => {
  const lattice = makeRestLattice([0, 0, 0], [10, 10, 10], 2, 2, 2); // trilinear, the coarsest real case
  // Move the (0,0,0)-corner control point (index [0][0][0]) by a real, known offset.
  const offset = [3, -2, 1];
  lattice.ctrlPts[0][0][0] = [lattice.ctrlPts[0][0][0][0] + offset[0], lattice.ctrlPts[0][0][0][1] + offset[1], lattice.ctrlPts[0][0][0][2] + offset[2]];

  // Exactness AT that corner (s=t=u=0 -> Bernstein degree-1 basis puts ALL
  // weight on the moved control point, per the endpoint property already
  // proven above).
  const atCorner = deformPoint(lattice, [0, 0, 0]);
  assert.ok(Math.hypot(atCorner[0] - (0 + offset[0]), atCorner[1] - (0 + offset[1]), atCorner[2] - (0 + offset[2])) < 1e-9,
    `at the moved corner itself, deform must be exact — got ${JSON.stringify(atCorner)}`);

  // The OPPOSITE corner (10,10,10) is entirely unaffected (Bernstein basis
  // there puts zero weight on the (0,0,0) control point).
  const atOpposite = deformPoint(lattice, [10, 10, 10]);
  assert.ok(Math.hypot(atOpposite[0] - 10, atOpposite[1] - 10, atOpposite[2] - 10) < 1e-9,
    `the opposite corner must stay exactly at its own rest position — got ${JSON.stringify(atOpposite)}`);

  // A point near the moved corner is displaced MORE than a point near the
  // center, which is displaced more than a point near the opposite corner
  // — a real, monotonic falloff, not a uniform or reversed one.
  const dispAt = (p) => { const d = deformPoint(lattice, p); return Math.hypot(d[0] - p[0], d[1] - p[1], d[2] - p[2]); };
  const near = dispAt([1, 1, 1]);
  const center = dispAt([5, 5, 5]);
  const far = dispAt([9, 9, 9]);
  assert.ok(near > center && center > far, `expected a real monotonic falloff near(${near}) > center(${center}) > far(${far})`);
  // NOT exactly zero — a real, found-while-testing correction to this
  // test's own first assumption, not a kernel bug: tensor Bernstein/Bezier
  // basis functions are GLOBAL, not compactly supported, so a point near
  // (but not exactly at) the opposite corner still carries some nonzero
  // residual weight on the moved control point — exactly
  // (1-s)(1-t)(1-u) * |offset| at s=t=u=0.9, which computes to
  // 0.1^3 * sqrt(14) ~= 0.00374, matching what's observed here. The
  // EXACT-zero claim only holds at s/t/u exactly 0 or 1 (the opposite
  // corner ITSELF, already proven exact above via atOpposite) — "near"
  // the opposite corner is a genuinely different, weaker claim.
  assert.ok(far < 0.01, `the far corner's own residual influence should be small (real global-basis decay, not exactly zero) — got ${far}`);
});

test('pointInsideLatticeBounds correctly distinguishes inside/outside/boundary', () => {
  const lattice = makeRestLattice([0, 0, 0], [10, 10, 10], 3, 3, 3);
  assert.equal(pointInsideLatticeBounds(lattice, [5, 5, 5]), true, 'center is inside');
  assert.equal(pointInsideLatticeBounds(lattice, [0, 0, 0]), true, 'exact corner counts as inside (boundary-inclusive)');
  assert.equal(pointInsideLatticeBounds(lattice, [10, 10, 10]), true, 'opposite exact corner counts as inside too');
  assert.equal(pointInsideLatticeBounds(lattice, [-1, 5, 5]), false, 'outside on one axis is excluded');
  assert.equal(pointInsideLatticeBounds(lattice, [5, 5, 11]), false, 'outside past the far face is excluded');
});

test('deformWithBands: a point outside every band is left completely untouched (zero total displacement)', () => {
  const bandA = makeRestLattice([0, 0, 0], [10, 10, 10], 2, 2, 2);
  bandA.ctrlPts[0][0][0] = [-5, -5, -5]; // a real, large displacement, but only for points INSIDE band A's own box
  const farPoint = [100, 100, 100];
  const out = deformWithBands([bandA], farPoint);
  assert.ok(Math.hypot(out[0] - farPoint[0], out[1] - farPoint[1], out[2] - farPoint[2]) < 1e-9,
    `a point outside every band's own box must be untouched — got ${JSON.stringify(out)}`);
});

test('deformWithBands: two independent bands with DIFFERENT rest boxes contribute additively at a point inside both — real superposition, not one band overriding the other', () => {
  // Band A: a coarse lattice over [0,10]^3, corner (0,0,0) moved by (1,0,0).
  const bandA = makeRestLattice([0, 0, 0], [10, 10, 10], 2, 2, 2);
  bandA.ctrlPts[0][0][0] = add2(bandA.ctrlPts[0][0][0], [1, 0, 0]);
  // Band B: a DIFFERENT, overlapping lattice over [-5,5]^3, its own corner
  // (0,0,0)-ish (index [0][0][0], at local rest position (-5,-5,-5)) moved
  // by (0,1,0) instead — independent rest frame, independent edit.
  const bandB = makeRestLattice([-5, -5, -5], [5, 5, 5], 2, 2, 2);
  bandB.ctrlPts[1][1][1] = add2(bandB.ctrlPts[1][1][1], [0, 0, 1]); // the (5,5,5) corner of band B

  function add2(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }

  // The point (0,0,0) is band A's own moved corner AND band B's own far
  // corner from ITS moved one (5,5,5 corner) — so at (0,0,0), band A
  // contributes its FULL (1,0,0) displacement (exact-at-corner property),
  // while band B contributes only a PARTIAL (decayed) amount of its own
  // (0,0,1) displacement (since (0,0,0) is band B's OPPOSITE corner from
  // (5,5,5), band B's own contribution there should be tiny, not zero,
  // since band B is a different lattice with its own corner at (-5,-5,-5)
  // unmoved and (5,5,5) moved — (0,0,0) is the CENTER of band B's box).
  const combined = deformWithBands([bandA, bandB], [0, 0, 0]);
  const soloA = deformWithBands([bandA], [0, 0, 0]);
  const soloB = deformWithBands([bandB], [0, 0, 0]);
  const soloADisp = [soloA[0] - 0, soloA[1] - 0, soloA[2] - 0];
  const soloBDisp = [soloB[0] - 0, soloB[1] - 0, soloB[2] - 0];
  const expectedCombined = [soloADisp[0] + soloBDisp[0], soloADisp[1] + soloBDisp[1], soloADisp[2] + soloBDisp[2]];
  assert.ok(
    Math.hypot(combined[0] - expectedCombined[0], combined[1] - expectedCombined[1], combined[2] - expectedCombined[2]) < 1e-9,
    `combined must equal the SUM of each band's own solo displacement — got combined=${JSON.stringify(combined)}, expected=${JSON.stringify(expectedCombined)}`,
  );
  // And band A's own solo contribution at its OWN moved corner must be exact.
  assert.ok(Math.abs(soloADisp[0] - 1) < 1e-9 && Math.abs(soloADisp[1]) < 1e-9 && Math.abs(soloADisp[2]) < 1e-9,
    `band A alone at (0,0,0) (its own moved corner) should be exactly (1,0,0) displacement, got ${JSON.stringify(soloADisp)}`);
});

test('a finer lattice (higher density) still reproduces the identity map and a real corner-move bulge, generalizing beyond the coarse trilinear case', () => {
  const lattice = makeRestLattice([0, 0, 0], [8, 8, 8], 4, 4, 4);
  // Move an INTERIOR (not corner) control point — a real case a coarse
  // 2x2x2 lattice couldn't even express (it has no interior points at all).
  const before = lattice.ctrlPts[1][1][1];
  lattice.ctrlPts[1][1][1] = [before[0], before[1] + 2, before[2]];
  const nearby = deformPoint(lattice, [8 / 3, 8 / 3, 8 / 3]); // exactly the interior control point's own rest position
  const far = deformPoint(lattice, [8, 0, 0]); // a distant corner, should stay ~unmoved
  const nearbyDisp = nearby[1] - 8 / 3;
  const farDisp = Math.abs(far[1] - 0);
  // A real, found-while-testing correction to this test's own first
  // assumption, not a kernel bug: an INTERIOR control point of a degree-3
  // Bezier volume does NOT interpolate exactly at its own "home" parameter
  // the way a CORNER control point does (corners have Bernstein weight
  // exactly 1.0 there; interior index i=1 of degree n=3 peaks at
  // B_1^3(1/3) = C(3,1)*(1/3)*(2/3)^2 = 4/9 per axis, cubed across 3 axes
  // = (4/9)^3 ~= 0.0878 of the full 2.0 offset ~= 0.1756 — exactly what's
  // observed here). This is the well-known, correct non-interpolating
  // property of interior Bezier/B-spline control points, not a defect —
  // the ORIGINAL claim ("near 0.5") assumed corner-like exact
  // interpolation, which only applies at the lattice's own 8 true corners.
  assert.ok(nearbyDisp > 0.15, `moving an interior CV should still produce a real, substantial LOCAL displacement near it (not full exact-interpolation strength, but not negligible either) — got ${nearbyDisp}`);
  assert.ok(farDisp < nearbyDisp, `a distant corner should be displaced far less than the region right at the moved interior CV — near=${nearbyDisp} far=${farDisp}`);
});

test('makeRestLattice refuses a degenerate density (<2 control points on any axis) rather than silently building a broken lattice', () => {
  assert.throws(() => makeRestLattice([0, 0, 0], [1, 1, 1], 1, 2, 2));
  assert.throws(() => makeRestLattice([0, 0, 0], [1, 1, 1], 2, 0, 2));
});

// A hard boolean inside/outside gate at a band's own face produces a C0
// CLIFF: dragging a corner of a 10x10x10 box lattice by a real, large
// offset, two points 0.002mm apart straddling the face would jump by the
// full corner-drag magnitude across that sub-millimeter step. Proven
// fixed here, not just asserted: the same two points now differ by a
// tiny, SMOOTH amount, not the full displacement.
test('deformWithBands: crossing a band boundary is now a SMOOTH transition, not the audit-confirmed C0 cliff', () => {
  const lattice = makeRestLattice([0, 0, 0], [10, 10, 10], 2, 2, 2);
  lattice.ctrlPts[1][1][1] = [15, 10, 10]; // drag the (10,10,10) corner outward by 5 on X — the audit's own exact repro
  const pExact = [10, 9.999, 9.999];
  const pOut = [10.001, 9.999, 9.999]; // 0.001mm past the face — the audit's own exact probe distance
  const dExact = deformWithBands([lattice], pExact);
  const dOut = deformWithBands([lattice], pOut);
  const jump = Math.hypot(dExact[0] - dOut[0], dExact[1] - dOut[1], dExact[2] - dOut[2]);
  // The OLD behavior (verified during the audit itself, not assumed):
  // dOut had ZERO displacement (dOut === pOut exactly), a jump of ~4.998mm
  // between dExact and dOut. The fix must make this small — a real,
  // smooth transition — not the audit's own confirmed ~5mm cliff.
  assert.ok(jump < 0.05, `expected a small, smooth jump crossing the boundary at this probe distance, got ${jump}mm (the audit's own confirmed pre-fix cliff was ~4.998mm)`);
  // This is a genuine RAMP across the whole margin, not just a widened
  // hard edge (which would show this same tiny jump at 0.001mm past the
  // face, then an equally abrupt cliff further out): a point at the
  // margin's own outer edge (1mm past the face, margin = 10% of this
  // 10-unit box) must show a MUCH larger drop from dExact than the point
  // only 0.001mm past it did — proving the transition is genuinely
  // spread across the whole margin, not concentrated at one edge of it.
  const pAtMarginEdge = [10 + 1.0, 9.999, 9.999]; // margin = 0.1 * 10 units = 1 unit past the face
  const dAtMarginEdge = deformWithBands([lattice], pAtMarginEdge);
  const jumpAtMarginEdge = Math.hypot(dExact[0] - dAtMarginEdge[0], dExact[1] - dAtMarginEdge[1], dExact[2] - dAtMarginEdge[2]);
  assert.ok(jumpAtMarginEdge > jump * 10, `expected the drop at the margin's own outer edge to be substantially larger than 0.001mm past the face (a real ramp, not a step) — got ${jump}mm at 0.001mm vs ${jumpAtMarginEdge}mm at the margin edge`);
});

test('bandInfluenceWeight: strictly inside (incl. every corner) is always EXACTLY 1 — the fix never weakens real FFD exactness, only smooths what happens crossing OUT', () => {
  const lattice = makeRestLattice([0, 0, 0], [10, 10, 10], 3, 3, 3);
  for (const p of [[0, 0, 0], [10, 10, 10], [5, 5, 5], [0, 10, 3], [10, 0, 10], [2.3, 7.8, 0]]) {
    assert.equal(bandInfluenceWeight(lattice, p), 1, `expected exactly 1 for a strictly-inside/on-face point ${JSON.stringify(p)}`);
  }
});

test('bandInfluenceWeight: monotonically decays from 1 (at the face) to 0 (at the margin\'s outer edge), then stays exactly 0 beyond it', () => {
  const lattice = makeRestLattice([0, 0, 0], [10, 10, 10], 2, 2, 2);
  const margin = CAGE_BOUNDARY_FALLOFF_MARGIN * 10; // margin is a FRACTION of local [0,1] extent; this box is 10 units/axis
  let prev = 1;
  for (const frac of [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) {
    const w = bandInfluenceWeight(lattice, [10 + frac * margin, 5, 5]);
    assert.ok(w <= prev + 1e-12, `expected a monotonically NON-INCREASING weight as distance past the face grows, got ${w} after ${prev} at frac=${frac}`);
    prev = w;
  }
  assert.ok(Math.abs(prev) < 1e-9, `expected exactly 0 right at the margin's own outer edge, got ${prev}`);
  assert.equal(bandInfluenceWeight(lattice, [10 + margin * 2, 5, 5]), 0, 'expected exactly 0 well beyond the margin, not a residual');
});

test('deformWithBands: a point far outside every band (well beyond the falloff margin) is STILL left completely untouched, matching the pre-fix guarantee exactly', () => {
  const bandA = makeRestLattice([0, 0, 0], [10, 10, 10], 2, 2, 2);
  bandA.ctrlPts[0][0][0] = [-5, -5, -5];
  const farPoint = [100, 100, 100];
  const out = deformWithBands([bandA], farPoint);
  assert.ok(Math.hypot(out[0] - farPoint[0], out[1] - farPoint[1], out[2] - farPoint[2]) < 1e-9,
    `a point far outside every band's own box (well beyond its falloff margin) must still be untouched — got ${JSON.stringify(out)}`);
});

// ================================================================
// RING-BASED CAGE ARRANGEMENT — Twist / Taper / Shear
// ================================================================

test('ringIndicesAlongAxis: partitions every control point into exactly one ring, for all three axis choices, on an ASYMMETRIC (off-origin, unequal-density) fixture', () => {
  // Deliberately off-origin and unequal per-axis density — a symmetric
  // cube fixture cannot distinguish a correctly-indexed ring from one
  // whose axes got silently transposed; this one can.
  const lattice = makeRestLattice([-7, 2, 100], [13, 9, 140], 4, 2, 3);
  for (const [axis, expectedCount] of [['U', 4], ['V', 2], ['W', 3]]) {
    const rings = ringIndicesAlongAxis(lattice, axis);
    assert.equal(rings.length, expectedCount, `axis ${axis}: expected ${expectedCount} rings, got ${rings.length}`);
    const seen = new Set();
    let total = 0;
    for (const ring of rings) {
      for (const [i, j, k] of ring) {
        const key = `${i},${j},${k}`;
        assert.ok(!seen.has(key), `axis ${axis}: index triple ${key} appeared in more than one ring`);
        seen.add(key);
        total++;
      }
    }
    assert.equal(total, 4 * 2 * 3, `axis ${axis}: expected every one of the 24 control points covered exactly once, got ${total}`);
  }
  assert.throws(() => ringIndicesAlongAxis(lattice, 'X'), /axis must be/, 'an invalid axis name must refuse honestly, not silently default to one');
});

test('applyRingTransform: an identity ring transform reproduces the exact rest lattice, on an off-origin/unequal-density fixture, for all three axes — never mutates the input', () => {
  const lattice = makeRestLattice([-7, 2, 100], [13, 9, 140], 4, 3, 2);
  const before = JSON.stringify(lattice.ctrlPts);
  for (const axis of ['U', 'V', 'W']) {
    const out = applyRingTransform(lattice, axis, () => (p) => p);
    assert.deepEqual(out.min, lattice.min, `axis ${axis}: min must be preserved verbatim`);
    assert.deepEqual(out.max, lattice.max, `axis ${axis}: max must be preserved verbatim`);
    for (let i = 0; i < lattice.densityU; i++) for (let j = 0; j < lattice.densityV; j++) for (let k = 0; k < lattice.densityW; k++) {
      const a = lattice.ctrlPts[i][j][k], b = out.ctrlPts[i][j][k];
      assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-12, `axis ${axis}: point [${i},${j},${k}] must be unchanged under an identity ring transform`);
    }
  }
  assert.equal(JSON.stringify(lattice.ctrlPts), before, 'applyRingTransform must never mutate its input lattice');
});

test('rotatePointAboutAxis / scaleRadially: the axis-parallel component is invariant, the perpendicular component is what actually moves', () => {
  const center = [1, 2, 3], axisDir = [1, 0, 0];
  const p = [11, 2 + 5, 3]; // 10 units along axis, 5 units perpendicular (along +Y)
  const rotated = rotatePointAboutAxis(p, center, axisDir, Math.PI / 2);
  assert.ok(Math.abs(rotated[0] - 11) < 1e-9, `expected the axis-parallel (X) component unchanged by rotation about that same axis, got ${rotated[0]}`);
  assert.ok(Math.abs((rotated[1] - center[1]) ** 2 + (rotated[2] - center[2]) ** 2 - 25) < 1e-9, 'expected the perpendicular radius (5) preserved by a pure rotation');
  const scaled = scaleRadially(p, center, axisDir, 2);
  assert.ok(Math.abs(scaled[0] - 11) < 1e-9, `expected the axis-parallel component unchanged by a radial scale, got ${scaled[0]}`);
  assert.ok(Math.abs(scaled[1] - (center[1] + 10)) < 1e-9 && Math.abs(scaled[2] - center[2]) < 1e-9, `expected the perpendicular offset doubled (5 -> 10), got [${scaled[1]},${scaled[2]}]`);
});

test('twistRingTransform / taperRingTransform / shearRingTransform: a UNIFORM (start===end) ring arrangement reproduces the exact single-point affine formula at ANY query point, not just at the control points — the real regression proof against today\'s existing point-formula math', () => {
  const center = [0, 0, 0], axisDir = [1, 0, 0];
  const lattice = makeRestLattice([-10, -5, -5], [10, 5, 5], 3, 3, 3);
  const testPoints = [[3, 1.4, -2.1], [-8, 4.9, 0], [0, 0, 0], [9.9, -4.9, 4.9]];

  const twistAngle = 0.6; // radians, arbitrary, non-trivial
  const twisted = applyRingTransform(lattice, 'U', twistRingTransform({ center, axisDir, startAngle: twistAngle, endAngle: twistAngle }));
  for (const p of testPoints) {
    const viaCage = deformPoint(twisted, p);
    const viaFormula = rotatePointAboutAxis(p, center, axisDir, twistAngle);
    assert.ok(Math.hypot(viaCage[0] - viaFormula[0], viaCage[1] - viaFormula[1], viaCage[2] - viaFormula[2]) < 1e-8,
      `uniform twist: point ${JSON.stringify(p)} via cage ${JSON.stringify(viaCage)} != direct rotation ${JSON.stringify(viaFormula)}`);
  }

  const taperFactor = 1.7;
  const tapered = applyRingTransform(lattice, 'U', taperRingTransform({ center, axisDir, startFactor: taperFactor, endFactor: taperFactor }));
  for (const p of testPoints) {
    const viaCage = deformPoint(tapered, p);
    const viaFormula = scaleRadially(p, center, axisDir, taperFactor);
    assert.ok(Math.hypot(viaCage[0] - viaFormula[0], viaCage[1] - viaFormula[1], viaCage[2] - viaFormula[2]) < 1e-8,
      `uniform taper: point ${JSON.stringify(p)} via cage ${JSON.stringify(viaCage)} != direct scale ${JSON.stringify(viaFormula)}`);
  }

  const shearDir = [0, 1, 0], shearAmount = 4.2;
  const sheared = applyRingTransform(lattice, 'U', shearRingTransform({ shearDir, startAmount: shearAmount, endAmount: shearAmount }));
  for (const p of testPoints) {
    const viaCage = deformPoint(sheared, p);
    const viaFormula = add(p, scale(shearDir, shearAmount));
    assert.ok(Math.hypot(viaCage[0] - viaFormula[0], viaCage[1] - viaFormula[1], viaCage[2] - viaFormula[2]) < 1e-8,
      `uniform shear: point ${JSON.stringify(p)} via cage ${JSON.stringify(viaCage)} != direct translate ${JSON.stringify(viaFormula)}`);
  }
});

// THE LOAD-BEARING FIDELITY PROOF — a genuinely per-ring-VARYING twist,
// measured through deformPoint (the real evaluated geometry), never the
// raw control points (which ARE rotated by their own exact assigned
// angle, by construction — measuring those would prove nothing about the
// blended result an actual student sees). Ground truth derived fresh
// in-test via the same "blend points on a circle as 2D vectors" model,
// not copied from any prior number.
test('TWIST interior-blending artifact: a real, measured angle lag AND radius contraction at an off-ring-boundary query station — proven, not just asserted, and checked at BOTH a symmetric and an asymmetric station', () => {
  const R = 10;
  const center = [0, 0, 0], axisDir = [1, 0, 0];
  // V/W density 2, positioned so control point (i,0,0) sits at exactly
  // (x_i, R, 0) for every ring i — i.e. angle0=0 in the Y-Z plane,
  // constant across rings, isolating a clean "radius R at angle 0" model.
  const lattice = makeRestLattice([-10, R, 0], [10, R + 3, 3], 3, 2, 2);
  const startAngle = 0, endAngle = Math.PI / 2; // rings at t=0,0.5,1 -> 0/45/90 degrees
  const twisted = applyRingTransform(lattice, 'U', twistRingTransform({ center, axisDir, startAngle, endAngle }));

  function measureAt(s) {
    const worldX = -10 + s * 20; // v=0, w=0 exactly -> isolates the U-direction Bernstein blend alone
    const p = deformPoint(twisted, [worldX, R, 0]);
    const yRel = p[1] - center[1], zRel = p[2] - center[2];
    return { radius: Math.hypot(yRel, zRel), angle: Math.atan2(zRel, yRel) };
  }

  // Independently derive the expected blend via the SAME Bernstein-basis-
  // of-rotated-2D-vectors model, computed fresh here (not read off the
  // implementation under test).
  function expectedAt(s) {
    const angles = [0, 1, 2].map((i) => lerp(startAngle, endAngle, i / 2));
    const weights = [0, 1, 2].map((i) => bernstein(2, i, s));
    let y = 0, z = 0;
    for (let i = 0; i < 3; i++) { y += weights[i] * R * Math.cos(angles[i]); z += weights[i] * R * Math.sin(angles[i]); }
    return { radius: Math.hypot(y, z), angle: Math.atan2(z, y) };
  }

  // s=0.5: EXACT by symmetry (both angle and the well-known 0.8536R
  // contraction) — a real, checkable closed-form value, not just
  // "whatever the model predicts."
  const mid = measureAt(0.5);
  assert.ok(Math.abs(mid.angle - Math.PI / 4) < 1e-9, `expected the symmetric mid-station angle to be EXACTLY 45deg (a real property of this symmetric case), got ${(mid.angle * 180 / Math.PI).toFixed(4)}deg`);
  assert.ok(Math.abs(mid.radius - R * 0.853553) < 1e-4, `expected the well-known symmetric-blend radius contraction to R*0.8536, got ${mid.radius.toFixed(4)} (R=${R})`);

  // s=0.25: the ASYMMETRIC station where a real angle LAG is actually
  // measurable (the symmetric station above cannot show one) — cross-
  // checked against the independently-derived model above, and also
  // proven to genuinely differ from the naive "just interpolate the
  // angle linearly" expectation (22.5deg), confirming this is a real
  // deviation from the ring's own assigned parametric angle, not zero.
  const asym = measureAt(0.25);
  const model = expectedAt(0.25);
  assert.ok(Math.abs(asym.angle - model.angle) < 1e-9, `measured angle ${(asym.angle * 180 / Math.PI).toFixed(3)}deg must match the independently-derived Bernstein-blend model ${(model.angle * 180 / Math.PI).toFixed(3)}deg`);
  const naiveExpectedDeg = 22.5; // what a linear interpolation of the ANGLE itself would give at s=0.25
  const measuredDeg = asym.angle * 180 / Math.PI;
  assert.ok(Math.abs(measuredDeg - naiveExpectedDeg) > 0.5, `expected a REAL, measurable lag from the naively-assigned 22.5deg (the ring's own requested angle at t=0.25) — got ${measuredDeg.toFixed(3)}deg, lag ${(naiveExpectedDeg - measuredDeg).toFixed(3)}deg`);
  assert.ok(asym.radius < R - 0.5, `expected a real, measurable radius contraction at the asymmetric station too, got ${asym.radius.toFixed(4)} (R=${R})`);
});

test('TAPER: EXACT at any query point under the default LINEAR falloff (Bernstein linear-precision proof) — the artifact only appears under a genuinely NONLINEAR falloff', () => {
  const center = [0, 0, 0], axisDir = [1, 0, 0];
  const lattice = makeRestLattice([-10, -6, -6], [10, 6, 6], 4, 2, 2); // N=4 rings, degree-3 Bernstein in U
  const startFactor = 1, endFactor = 3;

  // LINEAR falloff (the default): exact everywhere, not just at rings.
  const linear = applyRingTransform(lattice, 'U', taperRingTransform({ center, axisDir, startFactor, endFactor }));
  for (const s of [0.1, 0.33, 0.5, 0.72, 0.9]) {
    const worldX = -10 + s * 20;
    const p = deformPoint(linear, [worldX, 6, 0]);
    const expectedFactor = lerp(startFactor, endFactor, s);
    const measuredFactor = Math.hypot(p[1] - center[1], p[2] - center[2]) / 6; // 6 = the original perpendicular offset (V=max, W=min)
    assert.ok(Math.abs(measuredFactor - expectedFactor) < 1e-9, `linear-falloff taper at s=${s}: expected the EXACT continuously-interpolated factor ${expectedFactor}, got ${measuredFactor}`);
  }

  // NONLINEAR (smoothstep) falloff: a real, measurable deviation from
  // continuously evaluating the same falloff at the query station,
  // proving the artifact genuinely depends on falloff nonlinearity, not
  // on taper itself.
  const nonlinear = applyRingTransform(lattice, 'U', taperRingTransform({ center, axisDir, startFactor, endFactor, falloff: smoothstep }));
  const s = 0.25, worldX = -10 + s * 20;
  const p = deformPoint(nonlinear, [worldX, 6, 0]);
  const measuredFactor = Math.hypot(p[1] - center[1], p[2] - center[2]) / 6;
  const naiveContinuousFactor = lerp(startFactor, endFactor, smoothstep(s)); // what evaluating the falloff CONTINUOUSLY at s would give
  assert.ok(Math.abs(measuredFactor - naiveContinuousFactor) > 1e-4, `expected a real, nonzero deviation between the ring-blended factor (${measuredFactor}) and the continuously-evaluated falloff (${naiveContinuousFactor}) under a nonlinear falloff`);
});

test('SHEAR: EXACT at any query point under the default LINEAR falloff, matching the doc\'s own stated exemption from the interior-blending problem', () => {
  const shearDir = [0, 1, 0];
  const lattice = makeRestLattice([-10, -6, -6], [10, 6, 6], 4, 2, 2);
  const startAmount = 0, endAmount = 5;
  const sheared = applyRingTransform(lattice, 'U', shearRingTransform({ shearDir, startAmount, endAmount }));
  for (const s of [0.05, 0.4, 0.5, 0.61, 0.95]) {
    const worldX = -10 + s * 20;
    const before = [worldX, -6, -6];
    const p = deformPoint(sheared, before);
    const expectedAmount = lerp(startAmount, endAmount, s);
    assert.ok(Math.abs((p[1] - before[1]) - expectedAmount) < 1e-9, `expected the EXACT continuously-interpolated shear amount ${expectedAmount} at s=${s}, got ${p[1] - before[1]}`);
    assert.ok(Math.abs(p[0] - before[0]) < 1e-9 && Math.abs(p[2] - before[2]) < 1e-9, 'shear must leave the non-displaced axes untouched');
  }
});

test('applyRingTransform along a NON-U axis (V) genuinely keys on that axis, not a scrambled one — a real per-ring-varying transform on V leaves points differing only in U or W at the SAME V ring identically transformed', () => {
  const center = [0, 0, 0], axisDir = [0, 1, 0];
  const lattice = makeRestLattice([-3, -8, 100], [3, 8, 106], 2, 3, 2); // off-origin, unequal density, twist axis is V here
  const twisted = applyRingTransform(lattice, 'V', twistRingTransform({ center, axisDir, startAngle: 0, endAngle: Math.PI / 3 }));
  // two query points sitting at the SAME V (ring-boundary, so no blending
  // ambiguity) but different U/W — must receive the identical rotation.
  const v = -8; // ring 0 -> t=0 -> angle 0, a real no-op check
  const a = deformPoint(twisted, [-3, v, 100]);
  const b = deformPoint(twisted, [3, v, 106]);
  assert.ok(Math.abs(a[0] - (-3)) < 1e-9 && Math.abs(a[2] - 100) < 1e-9, `expected ring 0 (t=0, angle=0) to be a real no-op, got ${JSON.stringify(a)}`);
  assert.ok(Math.abs(b[0] - 3) < 1e-9 && Math.abs(b[2] - 106) < 1e-9, `expected ring 0 (t=0, angle=0) to be a real no-op for a second, differently-U/W point too, got ${JSON.stringify(b)}`);
  // ring at the OTHER end (t=1, angle=60deg) must genuinely differ from
  // ring 0's own untouched result, at the exact same U/W query point.
  const atStart = deformPoint(twisted, [-3, -8, 100]);
  const atEnd = deformPoint(twisted, [-3, 8, 100]);
  const moved = Math.hypot(atStart[0] - atEnd[0], atStart[1] - atEnd[1], atStart[2] - atEnd[2]);
  assert.ok(moved > 1, `expected ring 0 vs the far ring (60deg rotation about the V axis) to genuinely differ at the same U/W query, moved only ${moved}`);
});
