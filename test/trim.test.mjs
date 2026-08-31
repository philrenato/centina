import test from 'node:test';
import assert from 'node:assert/strict';
import {
  projectCurveToSurfaceUV, refineClosestPointOnSurface, signedArea2D,
  pointInUVPolygon, polylineSelfIntersects, trimLoopsValid, makeTrimmedSurface,
  trivialTrimLoop, trimmedNakedEdgeCount, closestPointOnTriangleMesh, projectCurveToMesh,
} from '../kernel/trim.mjs';
import { surfacePoint, closestPointOnSurface, nakedEdgeCount } from '../kernel/surface.mjs';
import { extractIsocurveU, extractIsocurveV } from '../kernel/isocurve.mjs';
import { makeCircle, makeArc, makeLine, extrude, revolve } from '../kernel/primitives.mjs';

// A multi-span (non-[0,1]-domain) revolve — real "stress" in the sense of
// exercising more than one knot span — but deliberately kept to a profile
// that stays within ONE quadrant (x always positive across the whole
// profile). This is a DIFFERENT choice from surface-point-inversion.test.mjs's
// own 270-degree stress fixture, made deliberately after directly hitting a
// real, separate, pre-existing kernel property while building this file's
// projectCurveToSurfaceUV tests: a revolve profile that crosses the
// revolve-axis's own perpendicular (x changes sign across the profile) has
// each row's own LOCAL xHat/yHat reference frame flip between rows —
// closestPointOnSurface's coarse grid search can then genuinely converge to
// a WORSE local minimum for a target on the "other side," a real,
// independent finding worth naming (not a bug in this file's own new
// code — confirmed directly against closestPointOnSurface alone, unrelated
// to projectCurveToSurfaceUV). Named as a real, separate,
// out-of-scope-this-round finding; this
// fixture just avoids the ambiguous region entirely, the same discipline
// surface-point-inversion.test.mjs's own header comment already uses
// ("Kept inside the FIRST arc span... deliberately").
function makeStressSurface() {
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 5, 0.2, 1.2); // stays in one quadrant, x always > 0
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2.5); // ~143 deg, 2 knot spans in V — real multi-span domain
}

test('refineClosestPointOnSurface, warm-started from the TRUE (u,v), stays there for a point already on the surface', () => {
  const srf = makeStressSurface();
  const target = surfacePoint(srf, 0.7, 1.4);
  const r = refineClosestPointOnSurface(srf, target, 0.7, 1.4);
  assert.ok(r.distance < 1e-9, `expected ~0 distance, got ${r.distance}`);
  assert.ok(Math.abs(r.u - 0.7) < 1e-6 && Math.abs(r.v - 1.4) < 1e-6);
});

test('refineClosestPointOnSurface, warm-started from a NEARBY (not exact) seed, still converges to the true point', () => {
  const srf = makeStressSurface();
  const target = surfacePoint(srf, 0.7, 1.4);
  const r = refineClosestPointOnSurface(srf, target, 0.75, 1.35); // seed offset from the truth
  assert.ok(r.distance < 1e-6, `expected convergence despite an offset seed, got ${r.distance}`);
});

// PROJECT AN ISOCURVE BACK ONTO ITS OWN SURFACE — an isocurve extracted via
// extractIsocurveU/V (kernel/isocurve.mjs) lies EXACTLY on the surface by
// construction (its own real exactness proof, test/isocurve.test.mjs), so
// projecting it back must reproduce a UV polyline with one coordinate
// essentially CONSTANT (the fixed u or v it was extracted at) and the other
// coordinate tracking the curve's own real parameter — a real, independent
// ground truth, not a tautology.
test('projectCurveToSurfaceUV on a real isocurve reproduces the correct constant-U line in UV space', () => {
  const srf = makeStressSurface();
  const uFixed = 0.6;
  const isoV = extractIsocurveU(srf, uFixed); // varies along V, u pinned at uFixed
  const result = projectCurveToSurfaceUV(isoV, srf, { samples: 40, tolerance: 1e-4 });
  assert.equal(result.ok, true, `expected a real projection, got refusal: ${result.reason}`);
  assert.equal(result.uv.length, 41);
  for (const [u] of result.uv) assert.ok(Math.abs(u - uFixed) < 1e-3, `expected u~=${uFixed}, got ${u}`);
  // and V must range across the surface's own real V domain, not collapse to one point
  const vs = result.uv.map(([, v]) => v);
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  assert.ok(Math.abs(Math.min(...vs) - vMin) < 1e-2, `expected V range to start near ${vMin}`);
  assert.ok(Math.abs(Math.max(...vs) - vMax) < 1e-2, `expected V range to end near ${vMax}`);
});

test('projectCurveToSurfaceUV on a real isocurve reproduces the correct constant-V line in UV space', () => {
  const srf = makeStressSurface();
  const vFixed = 0.8;
  const isoU = extractIsocurveV(srf, vFixed); // varies along U, v pinned at vFixed
  const result = projectCurveToSurfaceUV(isoU, srf, { samples: 40, tolerance: 1e-4 });
  assert.equal(result.ok, true, `expected a real projection, got refusal: ${result.reason}`);
  for (const [, v] of result.uv) assert.ok(Math.abs(v - vFixed) < 1e-3, `expected v~=${vFixed}, got ${v}`);
});

// The re-projected UV polyline, evaluated back through surfacePoint, must
// land back at the SAME real 3D positions the original isocurve visits —
// the actual end-to-end round-trip proof, independent of the u/v-constant
// check above.
test('projectCurveToSurfaceUV round-trips to the correct real 3D positions', () => {
  const srf = makeStressSurface();
  const uFixed = 0.4;
  const isoV = extractIsocurveU(srf, uFixed); // u=0.4 is safely within the surface's own [0,1] U-span
  const result = projectCurveToSurfaceUV(isoV, srf, { samples: 30, tolerance: 1e-4 });
  assert.equal(result.ok, true);
  // Every returned (u,v) must re-evaluate, via surfacePoint, back onto the
  // real surface — the actual end-to-end round-trip proof.
  for (const [u, v] of result.uv) {
    const p3 = surfacePoint(srf, u, v);
    const cp = closestPointOnSurface(srf, p3);
    assert.ok(cp.distance < 1e-6, 'reprojected point should sit exactly back on the surface');
  }
});

test('projectCurveToSurfaceUV honestly refuses a curve that does NOT lie on the surface', () => {
  const srf = makeStressSurface();
  const isoV = extractIsocurveU(srf, 1.0);
  const displaced = { ...isoV, ctrlPts: isoV.ctrlPts.map(([x, y, z, w]) => [x + 5, y + 5, z + 5, w]) };
  const result = projectCurveToSurfaceUV(displaced, srf, { tolerance: 1e-3 });
  assert.equal(result.ok, false);
  assert.ok(/not on the surface/i.test(result.reason));
});

test('signedArea2D: a CCW unit square has positive area 1, CW has negative area -1', () => {
  const ccw = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const cw = [[0, 0], [0, 1], [1, 1], [1, 0]];
  assert.ok(Math.abs(signedArea2D(ccw) - 1) < 1e-12);
  assert.ok(Math.abs(signedArea2D(cw) + 1) < 1e-12);
});

test('pointInUVPolygon: correctly classifies inside, outside, and boundary for a known square', () => {
  const square = [[0, 0], [2, 0], [2, 2], [0, 2]];
  assert.equal(pointInUVPolygon(square, 1, 1), 'inside');
  assert.equal(pointInUVPolygon(square, 5, 5), 'outside');
  assert.equal(pointInUVPolygon(square, -1, 1), 'outside');
  assert.equal(pointInUVPolygon(square, 0, 1), 'boundary');
  assert.equal(pointInUVPolygon(square, 1, 0), 'boundary');
});

test('polylineSelfIntersects: a convex loop does not self-intersect, a bowtie does', () => {
  const triangle = [[0, 0], [4, 0], [2, 3]];
  assert.equal(polylineSelfIntersects(triangle), false);
  const square = [[0, 0], [2, 0], [2, 2], [0, 2]];
  assert.equal(polylineSelfIntersects(square), false);
  // Bowtie: (0,0)->(2,2)->(2,0)->(0,2)->back to (0,0) — the first and third
  // segments genuinely cross in the middle.
  const bowtie = [[0, 0], [2, 2], [2, 0], [0, 2]];
  assert.equal(polylineSelfIntersects(bowtie), true);
});

test('trimLoopsValid: a single simple loop is valid; a self-intersecting loop is refused', () => {
  const square = [[0, 0], [2, 0], [2, 2], [0, 2]];
  assert.equal(trimLoopsValid([square]).ok, true);
  const bowtie = [[0, 0], [2, 2], [2, 0], [0, 2]];
  const r = trimLoopsValid([bowtie]);
  assert.equal(r.ok, false);
  assert.ok(/self-intersect/i.test(r.reason));
});

test('trimLoopsValid: outer + hole with OPPOSITE winding is valid; SAME winding is refused', () => {
  const outerCCW = [[0, 0], [4, 0], [4, 4], [0, 4]];
  const holeCW = [[1, 1], [1, 2], [2, 2], [2, 1]]; // opposite winding to outerCCW
  const holeCCWBad = [[1, 1], [2, 1], [2, 2], [1, 2]]; // SAME winding as outerCCW
  assert.equal(trimLoopsValid([outerCCW, holeCW]).ok, true);
  const bad = trimLoopsValid([outerCCW, holeCCWBad]);
  assert.equal(bad.ok, false);
  assert.ok(/same direction/i.test(bad.reason));
});

test('trimLoopsValid rejects an empty loop list and a degenerate <3-point loop', () => {
  assert.equal(trimLoopsValid([]).ok, false);
  assert.equal(trimLoopsValid([[[0, 0], [1, 1]]]).ok, false);
});

test('trivialTrimLoop reproduces the surface\'s own exact parametric rectangle corners, CCW', () => {
  const srf = extrude(makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 4), [0, 0, 1], 10);
  const loop = trivialTrimLoop(srf);
  assert.equal(loop.length, 4);
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  assert.deepEqual(loop[0], [uMin, vMin]);
  assert.deepEqual(loop[2], [uMax, vMax]);
  assert.ok(signedArea2D(loop) > 0, 'trivial trim loop should be CCW by construction');
});

test('makeTrimmedSurface on the trivial untrimmed case is valid, matching the "no migration needed" claim', () => {
  const srf = extrude(makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 4), [0, 0, 1], 10);
  const trimmed = makeTrimmedSurface(srf, [trivialTrimLoop(srf)]);
  assert.equal(trimmed.valid, true);
  assert.equal(trimmed.loops.length, 1);
  assert.equal(trimmed.loops[0].edge3d, null);
  assert.equal(trimmed.loops[0].tolerance, null);
});

test('makeTrimmedSurface with a real hole (outer + opposite-wound inner loop) is valid and reports the correct outer index', () => {
  const srf = extrude(makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 4), [0, 0, 1], 10);
  const outer = trivialTrimLoop(srf); // CCW, full rectangle
  const uMid = (srf.knotsU[0] + srf.knotsU[srf.knotsU.length - 1]) / 2;
  const vMid = (srf.knotsV[0] + srf.knotsV[srf.knotsV.length - 1]) / 2;
  const hole = [[uMid - 0.2, vMid - 0.2], [uMid - 0.2, vMid + 0.2], [uMid + 0.2, vMid + 0.2], [uMid + 0.2, vMid - 0.2]]; // CW
  const trimmed = makeTrimmedSurface(srf, [outer, hole]);
  assert.equal(trimmed.valid, true, trimmed.invalidReason);
  assert.equal(trimmed.outerIdx, 0);
});

// TRIM-AWARE NAKED-EDGE COUNT — the long-named
// "trim-aware naked-edge logic isn't built yet" gap (named honestly right
// in the Properties panel since the day Trim shipped), closed this file.
// A tube fixture (extrude of a closed circle profile) is the SAME
// known-good ground truth test/primitives.test.mjs's own surfaceClosure/
// nakedEdgeCount tests already establish: closed in U (the circle), open
// in V (the extrude direction) — exactly 2 plain naked edges.
function makeTubeSurface() {
  return extrude(makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 4), [0, 0, 1], 10);
}
// An open-in-both-directions fixture (extrude of an OPEN arc profile) —
// the SAME "4 naked edges" ground truth test/primitives.test.mjs's own
// extrude-of-open-profile case establishes, used here so the
// "exterior piece + hole" case is proven against a DIFFERENT base count
// than the tube (2), not just re-proving the same number twice.
function makeOpenBothSurface() {
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 5, 0.2, 1.2);
  return extrude(profile, [0, 0, 1], 10);
}
function midUVHole(srf, halfWidth = 0.2) {
  const uMid = (srf.knotsU[0] + srf.knotsU[srf.knotsU.length - 1]) / 2;
  const vMid = (srf.knotsV[0] + srf.knotsV[srf.knotsV.length - 1]) / 2;
  return [[uMid - halfWidth, vMid - halfWidth], [uMid - halfWidth, vMid + halfWidth], [uMid + halfWidth, vMid + halfWidth], [uMid + halfWidth, vMid - halfWidth]];
}
function midUVOuterLoop(srf, halfWidth = 0.3) {
  // An ordinary INTERIOR trim loop — a real Trim command's own shape
  // (strictly inside the domain, nothing like the full rectangle) — CCW,
  // deliberately larger than midUVHole's own hole so the two can coexist
  // (a donut: outer interior loop with a smaller hole cut into it).
  const uMid = (srf.knotsU[0] + srf.knotsU[srf.knotsU.length - 1]) / 2;
  const vMid = (srf.knotsV[0] + srf.knotsV[srf.knotsV.length - 1]) / 2;
  return [[uMid - halfWidth, vMid - halfWidth], [uMid + halfWidth, vMid - halfWidth], [uMid + halfWidth, vMid + halfWidth], [uMid - halfWidth, vMid + halfWidth]];
}

test('trimmedNakedEdgeCount with trimLoop=null is byte-identical to plain nakedEdgeCount — a strict superset, not a reinterpretation', () => {
  const tube = makeTubeSurface();
  assert.equal(trimmedNakedEdgeCount(tube, null), nakedEdgeCount(tube));
  assert.equal(trimmedNakedEdgeCount(tube, null), 2);
  const openBoth = makeOpenBothSurface();
  assert.equal(trimmedNakedEdgeCount(openBoth, null), nakedEdgeCount(openBoth));
  assert.equal(trimmedNakedEdgeCount(openBoth, null), 4);
});

test('trimmedNakedEdgeCount: an ordinary Trim (a genuine INTERIOR loop, no holes) has exactly 1 naked edge — the loop itself — regardless of the underlying surface\'s own U/V closure', () => {
  const tube = makeTubeSurface(); // plain nakedEdgeCount would say 2 here — WRONG for a real interior trim
  const loop = midUVOuterLoop(tube);
  assert.equal(trimmedNakedEdgeCount(tube, loop, []), 1);
  const openBoth = makeOpenBothSurface(); // plain nakedEdgeCount would say 4 here — WRONG for a real interior trim
  const loop2 = midUVOuterLoop(openBoth);
  assert.equal(trimmedNakedEdgeCount(openBoth, loop2, []), 1);
});

test('trimmedNakedEdgeCount: a Trim with N holes (a donut — interior outer loop plus interior holes) counts the outer loop plus every hole', () => {
  const tube = makeTubeSurface();
  const outer = midUVOuterLoop(tube, 0.35);
  const hole = midUVHole(tube, 0.1);
  assert.equal(trimmedNakedEdgeCount(tube, outer, [hole]), 2); // 1 outer + 1 hole
  assert.equal(trimmedNakedEdgeCount(tube, outer, [hole, hole]), 3); // 1 outer + 2 holes (synthetic, geometry aside — proving the count arithmetic, not re-validating winding)
});

test('trimmedNakedEdgeCount: the Split-family EXTERIOR piece (trimLoop IS the full untrimmed rectangle, one hole cut where the other piece was) keeps the surface\'s own real closure-based count PLUS one per hole', () => {
  const tube = makeTubeSurface();
  const rect = trivialTrimLoop(tube);
  const hole = midUVHole(tube);
  assert.equal(trimmedNakedEdgeCount(tube, rect, [hole]), nakedEdgeCount(tube) + 1);
  assert.equal(trimmedNakedEdgeCount(tube, rect, [hole]), 3);
  const openBoth = makeOpenBothSurface();
  const rect2 = trivialTrimLoop(openBoth);
  const hole2 = midUVHole(openBoth);
  assert.equal(trimmedNakedEdgeCount(openBoth, rect2, [hole2]), nakedEdgeCount(openBoth) + 1);
  assert.equal(trimmedNakedEdgeCount(openBoth, rect2, [hole2]), 5);
});

test('trimmedNakedEdgeCount: the full-rectangle detection is robust to a CLONED trimLoop array (Copy/Paste, undo/redo, and the debug-export field all clone via .map, never keep the same reference)', () => {
  const tube = makeTubeSurface();
  const clonedRect = trivialTrimLoop(tube).map((p) => [...p]); // a fresh array, same values, no shared reference with a fresh trivialTrimLoop(tube) call
  assert.notEqual(clonedRect, trivialTrimLoop(tube)); // genuinely a different array reference
  assert.equal(trimmedNakedEdgeCount(tube, clonedRect, []), nakedEdgeCount(tube));
});

// ============================================================
// PROJECT onto a NURBS TORUS — a real reported bug: a Line projected onto
// a NURBS torus produced garbage geometry. Root
// cause traced to refineClosestPointOnSurface's old hard-clamp behavior:
// warm-started sample-to-sample along a curve, a step wanting to cross a
// closed direction's own seam used to pin dead against the domain edge and
// get permanently stuck there, producing long runs of byte-identical mapped
// points (which then made globalCurveInterp's own coefficient matrix
// singular). The exact same torus construction test/isocurve.test.mjs's
// own "zero naked edges" test already confirms closed in BOTH directions.
// ============================================================
function makeTorusSrf() {
  const profile = makeCircle([5, 0, 0], [0, 0, 1], [1, 0, 0], 1, 4);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}

test('refineClosestPointOnSurface crosses a closed direction\'s own seam instead of pinning against it', () => {
  const srf = makeTorusSrf();
  const uMax = srf.knotsU[srf.knotsU.length - 1], vMax = srf.knotsV[srf.knotsV.length - 1];
  // Seed near the U-domain's far edge; the true target sits well past it,
  // reachable only by continuing PAST u=uMax and wrapping back near u=uMin
  // (the old clamp-and-break behavior gets stuck at u=uMax exactly and can
  // never reach anywhere near here).
  const seedU = uMax * 0.98, v0 = vMax * 0.5;
  const targetU = uMax * 0.15;
  const target = surfacePoint(srf, targetU, v0);
  const r = refineClosestPointOnSurface(srf, target, seedU, v0);
  assert.ok(r.distance < 1e-4, `expected the search to cross the seam and converge near-exactly, got distance=${r.distance} (a stuck-at-boundary search would report a distance comparable to the tube diameter, ~2)`);
});

test('projectCurveToSurfaceUV: a real Line dipping through a torus tube succeeds and every mapped point is genuinely near the source line (not just coincidentally on the torus)', () => {
  const srf = makeTorusSrf(); // major radius 5, minor radius 1, tube axis = Z
  // A line running through the tube region at an angle, guaranteed (by the
  // sheer number of samples at this scale) to require the search to cross
  // at least one of the torus's two closed directions along its own walk.
  const line = makeLine([3.5, -2, 0.5], [6.5, 2, -0.5]);
  const tolerance = 1.5;
  const result = projectCurveToSurfaceUV(line, srf, { samples: 64, tolerance });
  assert.equal(result.ok, true, `expected a real projection, got refusal: ${result.reason}`);
  assert.equal(result.uv.length, 65);
  // Evaluating any (u,v) on this srf trivially satisfies the torus's own
  // implicit equation regardless of whether the SEARCH converged well (the
  // surface itself is always an exact torus by construction) — that alone
  // would be a vacuous check. The real correctness proof is that each
  // mapped point is genuinely CLOSE to the ORIGINAL line sample it was
  // pulled from, re-derived independently here rather than trusted from
  // the function's own internal tolerance check.
  let worstDist = 0;
  for (let i = 0; i <= 64; i++) {
    const t = i / 64;
    const [lx, ly, lz] = [3.5 + t * 3, -2 + t * 4, 0.5 + t * -1]; // same param as makeLine(p0,p1) traces
    const [u, v] = result.uv[i];
    const [x, y, z] = surfacePoint(srf, u, v);
    const d = Math.hypot(x - lx, y - ly, z - lz);
    if (d > worstDist) worstDist = d;
    // Independent sanity: this point really is on the true analytic torus.
    const eq = Math.abs((Math.hypot(x, y) - 5) ** 2 + z * z - 1);
    assert.ok(eq < 1e-6, `mapped point ${i} does not satisfy the torus's own true implicit equation (residual ${eq})`);
  }
  assert.ok(worstDist <= tolerance, `every mapped point must land within the requested tolerance of the real source line; worst was ${worstDist}`);
});

// ============================================================
// PROJECT onto a MESH-ONLY target — a second reported gap: Project refused
// outright on a SuperB (SubD) object, which
// carries no explicit NURBS `srf` at all, only a tessellated limit-surface
// mesh. This is the sibling path for exactly that case (and, generalized
// in the app's own wiring, every other panel/mesh-based container —
// RuledLoft/PolySurface/MultiPipe/Split): a brute-force point-to-triangle
// closest point, no iteration, so it structurally cannot diverge or get
// stuck the way the NURBS Gauss-Newton path can.
// ============================================================
// A flat quad, two triangles, in the z=0 plane, x,y in [0,10] — a simple,
// independently-checkable target.
const quadTriangles = [
  [[0, 0, 0], [10, 0, 0], [10, 10, 0]],
  [[0, 0, 0], [10, 10, 0], [0, 10, 0]],
];

test('closestPointOnTriangleMesh: a point directly above the interior projects straight down', () => {
  const { point, distance } = closestPointOnTriangleMesh(quadTriangles, [5, 5, 3]);
  assert.ok(Math.abs(point[0] - 5) < 1e-9 && Math.abs(point[1] - 5) < 1e-9 && Math.abs(point[2]) < 1e-9);
  assert.ok(Math.abs(distance - 3) < 1e-9, `expected distance 3, got ${distance}`);
});

test('closestPointOnTriangleMesh: a point beyond a corner snaps to that vertex', () => {
  const { point, distance } = closestPointOnTriangleMesh(quadTriangles, [-2, -2, 0]);
  assert.deepEqual(point, [0, 0, 0]);
  assert.ok(Math.abs(distance - Math.hypot(2, 2)) < 1e-9);
});

test('closestPointOnTriangleMesh: a point beyond an edge (not a corner) snaps to the nearest edge point', () => {
  const { point, distance } = closestPointOnTriangleMesh(quadTriangles, [-2, 5, 0]);
  assert.ok(Math.abs(point[0]) < 1e-9 && Math.abs(point[1] - 5) < 1e-9 && Math.abs(point[2]) < 1e-9);
  assert.ok(Math.abs(distance - 2) < 1e-9, `expected distance 2, got ${distance}`);
});

test('projectCurveToMesh: a line hovering directly above a flat mesh projects straight down onto it, within tolerance', () => {
  const line = makeLine([2, 2, 2], [8, 8, 2]);
  const tolerance = 5;
  const result = projectCurveToMesh(line, quadTriangles, { samples: 10, tolerance });
  assert.equal(result.ok, true, `expected a real projection, got refusal: ${result.reason}`);
  assert.equal(result.points.length, 11);
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    const expectedX = 2 + t * 6, expectedY = 2 + t * 6;
    const [x, y, z] = result.points[i];
    assert.ok(Math.abs(x - expectedX) < 1e-6, `point ${i} x mismatch: ${x} vs ${expectedX}`);
    assert.ok(Math.abs(y - expectedY) < 1e-6, `point ${i} y mismatch: ${y} vs ${expectedY}`);
    assert.ok(Math.abs(z) < 1e-6, `point ${i} should land exactly on the flat mesh (z=0), got z=${z}`);
  }
});

test('projectCurveToMesh: an honest refusal when every sample is genuinely too far from the mesh', () => {
  const line = makeLine([2, 2, 2], [8, 8, 2]); // 2mm off the z=0 mesh
  const result = projectCurveToMesh(line, quadTriangles, { samples: 10, tolerance: 1 });
  assert.equal(result.ok, false);
  assert.match(result.reason, /away, above tolerance/);
});

test('projectCurveToMesh: an empty triangle list refuses honestly instead of crashing', () => {
  const line = makeLine([0, 0, 0], [1, 1, 1]);
  const result = projectCurveToMesh(line, [], { samples: 10, tolerance: 1 });
  assert.equal(result.ok, false);
  assert.match(result.reason, /no real surface geometry/);
});
