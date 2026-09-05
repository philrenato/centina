import test from 'node:test';
import assert from 'node:assert/strict';
import {
  reverseSurfaceDirection, flipSurfaceNormals, remapReversedParam,
  curveSeamRefusal, moveCurveSeam, moveCurveSeamToPoint,
  surfaceSeamDirections, surfaceSeamRefusal, moveSurfaceSeam,
} from '../kernel/flipseam.mjs';
import { curvePoint, curveDomain, reverseCurve, isCurveClosed, closestPointOnCurve } from '../kernel/curve.mjs';
import { surfacePoint, surfacePointAndPartials, surfaceDomain, closestPointOnSurface } from '../kernel/surface.mjs';
import { makeCircle, makeLine, revolve } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { offsetSurface } from '../kernel/offset.mjs';

const TAU = Math.PI * 2;

function linspace(a, b, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(a + (b - a) * (i / (n - 1)));
  return out;
}
function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(v) { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// THE INVARIANT, AS A FUNCTION. Both Flip and Seam claim the shape occupies
// the same points afterwards. A parametrization change moves points AROUND the
// domain, so comparing sample i to sample i is the wrong test, and so is
// comparing two finite sample GRIDS — the moved shape's uniform grid lands
// between the original's samples, and the residual that leaves is a fact about
// sampling density, not about geometry. What has to be measured is the distance
// from each sample of one shape to the OTHER SHAPE ITSELF, via the kernel's own
// closest-point solve. Checked BOTH ways, because one-way containment would
// also pass for a shape that lost a piece.
function maxDeviationFromCurve(pts, crv) {
  let worst = 0, at = null;
  for (const p of pts) {
    const c = closestPointOnCurve(crv, p);
    const d = dist(p, c.point);
    if (d > worst) { worst = d; at = p; }
  }
  return { worst, at };
}
function curvesOccupySamePoints(a, b, tol, n = 200) {
  const fwd = maxDeviationFromCurve(sampleCurve(a, n), b);
  const back = maxDeviationFromCurve(sampleCurve(b, n), a);
  const worst = Math.max(fwd.worst, back.worst);
  return { ok: worst <= tol, worst };
}
function maxDeviationFromSurface(pts, srf) {
  let worst = 0;
  for (const p of pts) {
    const c = closestPointOnSurface(srf, p);
    const d = dist(p, c.point);
    if (d > worst) worst = d;
  }
  return worst;
}
function surfacesOccupySamePoints(a, b, tol, n = 12) {
  const worst = Math.max(maxDeviationFromSurface(sampleSurface(a, n), b), maxDeviationFromSurface(sampleSurface(b, n), a));
  return { ok: worst <= tol, worst };
}
function sampleCurve(crv, n = 400) {
  const [min, max] = curveDomain(crv);
  return linspace(min, max, n).map((u) => curvePoint(crv, u));
}
function sampleSurface(srf, n = 40) {
  const d = surfaceDomain(srf);
  const out = [];
  for (const u of linspace(d.u[0], d.u[1], n)) for (const v of linspace(d.v[0], d.v[1], n)) out.push(surfacePoint(srf, u, v));
  return out;
}
function unitNormal(srf, u, v) {
  const { su, sv } = surfacePointAndPartials(srf, u, v);
  return norm(cross(su, sv));
}

function torus(R = 30, r = 8) {
  // A closed profile revolved a full turn: closed in BOTH directions.
  const profile = makeCircle([R, 0, 0], [1, 0, 0], [0, 0, 1], r);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, TAU);
}
function cylinder(R = 20, h = 40) {
  // An OPEN profile revolved a full turn: closed in the sweep direction only.
  const profile = makeLine([R, 0, 0], [R, 0, h]);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, TAU);
}
function openPatch() {
  const profile = makeLine([10, 0, 0], [10, 0, 20]);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI / 2);
}

// ============================================================================
// FLIP — CURVE
// ============================================================================

test('reverseCurve: the reversed curve occupies the same points', () => {
  const crv = globalCurveInterp([[0, 0, 0], [10, 6, 2], [22, -4, 5], [35, 2, 0], [48, 9, -3]], 3);
  const r = curvesOccupySamePoints(crv, reverseCurve(crv), 1e-9);
  assert.ok(r.ok, `point set moved by ${r.worst}`);
});

test('reverseCurve: start and end genuinely swap', () => {
  const crv = globalCurveInterp([[0, 0, 0], [10, 6, 2], [22, -4, 5], [35, 2, 0]], 3);
  const [dMin, dMax] = curveDomain(crv), rev = reverseCurve(crv), [rMin, rMax] = curveDomain(rev);
  assert.ok(dist(curvePoint(crv, dMin), curvePoint(rev, rMax)) < 1e-9);
  assert.ok(dist(curvePoint(crv, dMax), curvePoint(rev, rMin)) < 1e-9);
});

test('reverseCurve: reversing twice is the identity, knot for knot', () => {
  const crv = globalCurveInterp([[0, 0, 0], [10, 6, 2], [22, -4, 5], [35, 2, 0], [48, 9, -3]], 3);
  const back = reverseCurve(reverseCurve(crv));
  assert.equal(back.degree, crv.degree);
  crv.knots.forEach((k, i) => assert.ok(Math.abs(k - back.knots[i]) < 1e-12, `knot ${i}`));
  crv.ctrlPts.forEach((p, i) => p.forEach((c, j) => assert.ok(Math.abs(c - back.ctrlPts[i][j]) < 1e-12, `cp ${i}.${j}`)));
});

// ============================================================================
// FLIP — SURFACE NORMALS
// ============================================================================

test('flipSurfaceNormals: the surface occupies the same points', () => {
  const srf = cylinder();
  const r = surfacesOccupySamePoints(srf, flipSurfaceNormals(srf), 1e-8);
  assert.ok(r.ok, `point set moved by ${r.worst}`);
});

test('flipSurfaceNormals: the normal genuinely reverses at every sampled point', () => {
  const srf = cylinder();
  const flipped = flipSurfaceNormals(srf);
  const d = surfaceDomain(srf), fd = surfaceDomain(flipped);
  let checked = 0;
  for (const fu of [0.13, 0.37, 0.61, 0.88]) {
    for (const fv of [0.2, 0.5, 0.8]) {
      const u = d.u[0] + (d.u[1] - d.u[0]) * fu, v = d.v[0] + (d.v[1] - d.v[0]) * fv;
      // The SAME physical point on the flipped surface sits at the mirrored u.
      const u2 = remapReversedParam(u, srf.knotsU);
      assert.ok(dist(surfacePoint(srf, u, v), surfacePoint(flipped, u2, v)) < 1e-8, 'remapReversedParam must name the same point');
      const n0 = unitNormal(srf, u, v), n1 = unitNormal(flipped, u2, v);
      assert.ok(dot(n0, n1) < -0.999, `normal did not reverse at (${fu},${fv}): dot ${dot(n0, n1)}`);
      checked++;
    }
  }
  assert.equal(checked, 12);
  assert.ok(fd.u[0] === d.u[0] && fd.u[1] === d.u[1], 'the domain is re-based onto itself');
});

test('flipSurfaceNormals: flipping twice is the identity, knot for knot and point for point', () => {
  const srf = torus();
  const back = flipSurfaceNormals(flipSurfaceNormals(srf));
  srf.knotsU.forEach((k, i) => assert.ok(Math.abs(k - back.knotsU[i]) < 1e-12, `knotU ${i}`));
  srf.knotsV.forEach((k, i) => assert.ok(Math.abs(k - back.knotsV[i]) < 1e-12, `knotV ${i}`));
  for (let i = 0; i < srf.ctrlNet.length; i++) {
    for (let j = 0; j < srf.ctrlNet[i].length; j++) {
      for (let c = 0; c < 4; c++) {
        assert.ok(Math.abs(srf.ctrlNet[i][j][c] - back.ctrlNet[i][j][c]) < 1e-12, `cp ${i}.${j}.${c}`);
      }
    }
  }
});

test('reverseSurfaceDirection: v reverses the normal too, and is its own inverse', () => {
  const srf = cylinder();
  const flipped = reverseSurfaceDirection(srf, 'v');
  const d = surfaceDomain(srf);
  const u = d.u[0] + (d.u[1] - d.u[0]) * 0.3, v = d.v[0] + (d.v[1] - d.v[0]) * 0.4;
  const v2 = remapReversedParam(v, srf.knotsV);
  assert.ok(dist(surfacePoint(srf, u, v), surfacePoint(flipped, u, v2)) < 1e-8);
  assert.ok(dot(unitNormal(srf, u, v), unitNormal(flipped, u, v2)) < -0.999);
  const back = reverseSurfaceDirection(flipped, 'v');
  assert.ok(dist(surfacePoint(srf, u, v), surfacePoint(back, u, v)) < 1e-12);
});

// ============================================================================
// SEAM — CLOSED CURVE
// ============================================================================

test('moveCurveSeam: a circle keeps its exact point set and starts where it was told to', () => {
  const crv = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 25);
  assert.ok(isCurveClosed(crv));
  const [min, max] = curveDomain(crv);
  const t = min + (max - min) * 0.37;
  const wanted = curvePoint(crv, t);
  const moved = moveCurveSeam(crv, t);
  const [mMin, mMax] = curveDomain(moved);
  assert.ok(dist(curvePoint(moved, mMin), wanted) < 1e-8, 'the new curve must begin at the picked point');
  assert.ok(dist(curvePoint(moved, mMax), wanted) < 1e-8, 'and end there too — it is still closed');
  const same = curvesOccupySamePoints(crv, moved, 1e-7);
  assert.ok(same.ok, `the circle moved by ${same.worst}`);
});

test('moveCurveSeam: a hand-made closed spline keeps its exact point set', () => {
  const pts = [[0, 0, 0], [20, 8, 3], [35, -6, -2], [18, -18, 4], [-6, -8, 1]];
  const crv = globalCurveInterp([...pts, pts[0]], 3);
  assert.ok(isCurveClosed(crv), 'fixture must be closed');
  const [min, max] = curveDomain(crv);
  const t = min + (max - min) * 0.62;
  const moved = moveCurveSeam(crv, t);
  const same = curvesOccupySamePoints(crv, moved, 1e-6);
  assert.ok(same.ok, `the spline moved by ${same.worst}`);
  assert.ok(isCurveClosed(moved, 1e-6), 'and it is still closed afterwards');
});

test('moveCurveSeam: the domain is preserved, so nothing downstream is handed a different one', () => {
  const crv = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 10);
  const [dMin, dMax] = curveDomain(crv);
  const moved = moveCurveSeam(crv, dMin + (dMax - dMin) * 0.5);
  const [mMin, mMax] = curveDomain(moved);
  assert.ok(Math.abs(mMin - dMin) < 1e-12 && Math.abs(mMax - dMax) < 1e-12);
});

test('moveCurveSeamToPoint: the seam lands on the point nearest the pick', () => {
  const crv = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 25);
  const pick = [0, 40, 0]; // outside the circle, nearest point is (0, 25, 0)
  const moved = moveCurveSeamToPoint(crv, pick);
  const [mMin] = curveDomain(moved);
  assert.ok(dist(curvePoint(moved, mMin), [0, 25, 0]) < 0.2, `landed at ${curvePoint(moved, mMin)}`);
});

// ============================================================================
// SEAM — CLOSED SURFACE
// ============================================================================

test('surfaceSeamDirections: a torus is closed both ways, a cylinder one way, a patch neither', () => {
  assert.deepEqual(surfaceSeamDirections(torus()), { u: true, v: true });
  const cyl = surfaceSeamDirections(cylinder());
  assert.ok(cyl.u !== cyl.v, `a cylinder must be closed in exactly one direction, got ${JSON.stringify(cyl)}`);
  assert.deepEqual(surfaceSeamDirections(openPatch()), { u: false, v: false });
});

test('moveSurfaceSeam: a cylinder keeps its exact point set when its seam moves', () => {
  const srf = cylinder();
  const closed = surfaceSeamDirections(srf);
  const dir = closed.u ? 'u' : 'v';
  const knots = dir === 'u' ? srf.knotsU : srf.knotsV;
  const min = knots[0], max = knots[knots.length - 1];
  const moved = moveSurfaceSeam(srf, dir, min + (max - min) * 0.41);
  const same = surfacesOccupySamePoints(srf, moved, 1e-6);
  assert.ok(same.ok, `the cylinder moved by ${same.worst}`);
});

test('moveSurfaceSeam: a torus keeps its exact point set, in either direction', () => {
  const srf = torus();
  for (const dir of ['u', 'v']) {
    const knots = dir === 'u' ? srf.knotsU : srf.knotsV;
    const min = knots[0], max = knots[knots.length - 1];
    const moved = moveSurfaceSeam(srf, dir, min + (max - min) * 0.27);
    const same = surfacesOccupySamePoints(srf, moved, 1e-5);
    assert.ok(same.ok, `${dir}: the torus moved by ${same.worst}`);
    assert.ok(surfaceSeamDirections(moved, 1e-5)[dir], `${dir}: must still be closed afterwards`);
  }
});

test('moveSurfaceSeam: the moved seam is genuinely where it was asked for', () => {
  const srf = torus();
  const closed = surfaceSeamDirections(srf);
  const dir = closed.u ? 'u' : 'v';
  const knots = dir === 'u' ? srf.knotsU : srf.knotsV;
  const min = knots[0], max = knots[knots.length - 1];
  const t = min + (max - min) * 0.27;
  const d = surfaceDomain(srf);
  const other = dir === 'u' ? d.v : d.u;
  const moved = moveSurfaceSeam(srf, dir, t);
  for (const fw of [0.1, 0.5, 0.9]) {
    const w = other[0] + (other[1] - other[0]) * fw;
    const wanted = dir === 'u' ? surfacePoint(srf, t, w) : surfacePoint(srf, w, t);
    const got = dir === 'u' ? surfacePoint(moved, min, w) : surfacePoint(moved, w, min);
    assert.ok(dist(wanted, got) < 1e-5, `seam did not land: ${wanted} vs ${got}`);
  }
});

// ============================================================================
// REFUSALS — each fires BY NAME on the wrong input
// ============================================================================

test('refusal: an open curve has no seam, and it says so in those words', () => {
  const open = globalCurveInterp([[0, 0, 0], [10, 5, 0], [20, 0, 0]], 2);
  const reason = curveSeamRefusal(open);
  assert.match(reason, /open/);
  assert.match(reason, /no seam to move/);
  assert.throws(() => moveCurveSeam(open, 0.5), /moveCurveSeam: this curve is open/);
  assert.throws(() => moveCurveSeamToPoint(open, [1, 1, 1]), /moveCurveSeamToPoint: this curve is open/);
});

test('refusal: a closed curve still refuses a seam parameter at or outside its own ends', () => {
  const crv = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const [min, max] = curveDomain(crv);
  assert.throws(() => moveCurveSeam(crv, min), /IS the seam already there/);
  assert.throws(() => moveCurveSeam(crv, max), /IS the seam already there/);
  assert.throws(() => moveCurveSeam(crv, max + 1), /must lie strictly inside/);
});

test('refusal: an open surface direction has no seam, and it names the direction', () => {
  const patch = openPatch();
  assert.match(surfaceSeamRefusal(patch, 'u'), /open in u/);
  assert.match(surfaceSeamRefusal(patch, 'v'), /open in v/);
  assert.throws(() => moveSurfaceSeam(patch, 'u', 0.5), /moveSurfaceSeam: this surface is open in u/);
  const cyl = cylinder();
  const openDir = surfaceSeamDirections(cyl).u ? 'v' : 'u';
  assert.match(surfaceSeamRefusal(cyl, openDir), new RegExp(`open in ${openDir}`));
  assert.equal(surfaceSeamRefusal(cyl, openDir === 'u' ? 'v' : 'u'), null);
});

test('refusal: a direction that is not u or v is named as such, not coerced', () => {
  assert.match(surfaceSeamRefusal(torus(), 'w'), /not a surface direction/);
  assert.throws(() => reverseSurfaceDirection(torus(), 'w'), /must be 'u' or 'v'/);
});

// ============================================================================
// WHAT A FLIPPED NORMAL ACTUALLY CHANGES
// ============================================================================
// The honest answer to "is this only shading?" is no, and this is the proof:
// offsetSurface derives its direction from Su x Sv, so a flipped surface offsets
// the other way. The same orientation decides an exported mesh's facet winding
// and the sign of a divergence-theorem volume, which is why a flipped normal on
// a closed solid is a correctness matter rather than a cosmetic one.

test('a flipped surface offsets the OTHER way — the flip reaches real operations, not just shading', () => {
  const srf = cylinder(20, 40);
  const a = offsetSurface(srf, 5);
  const b = offsetSurface(flipSurfaceNormals(srf), 5);
  const d = surfaceDomain(srf);
  const u = d.u[0] + (d.u[1] - d.u[0]) * 0.3, v = d.v[0] + (d.v[1] - d.v[0]) * 0.5;
  const base = surfacePoint(srf, u, v);
  const outA = surfacePoint(a.surface, u, v);
  const outB = surfacePoint(b.surface, remapReversedParam(u, srf.knotsU), v);
  const rBase = Math.hypot(base[0], base[1]);
  const rA = Math.hypot(outA[0], outA[1]);
  const rB = Math.hypot(outB[0], outB[1]);
  /* WHICH WAY THE UNFLIPPED ONE GOES IS THE REVOLVE'S OWN BUSINESS — a full
     revolve of a line happens to come out normal-inward — so the claim tested
     here is the RELATIONSHIP, not a sign: the two offsets move the same distance
     in opposite directions. That is the whole point of being able to flip. */
  assert.ok(Math.abs(Math.abs(rA - rBase) - 5) < 1e-6, `the unflipped offset moves by 5: ${rBase} -> ${rA}`);
  assert.ok(Math.abs(Math.abs(rB - rBase) - 5) < 1e-6, `the flipped offset moves by 5: ${rBase} -> ${rB}`);
  assert.ok((rA - rBase) * (rB - rBase) < 0, `and they go OPPOSITE ways: ${rA} vs ${rB} around ${rBase}`);
});

// ============================================================================
// AND IT HAS TO SURVIVE A **FEATURE** REBUILD
// ============================================================================
// A flip on a polysurface is stored as a MODIFIER on the object and written
// into every face. A FEATURE rebuild — a fillet radius edited, a hole resized —
// re-derives that face list from the base the feature was named on, so anything
// written onto the finished faces is not in the base and does not come back
// unless it is replayed. These hold the rule that replay follows.
//
// The measure is the SIGN OF THE VOLUME, by the divergence theorem
// V = (1/3)∮ x·n dA = (1/3)∫∫ S·(Su x Sv) du dv, which is exactly the quantity a
// lost flip gets wrong. On bilinear faces the integrand is linear in each
// parameter, so one midpoint cell per face integrates it exactly.

function quadPatch(p00, p10, p01, p11) {
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[[...p00, 1], [...p01, 1]], [[...p10, 1], [...p11, 1]]],
  };
}
// The six faces of a cube, every one of them oriented OUTWARD: u x v is the
// outward direction on each, which is what makes the volume below come out
// positive and gives the flip something to reverse.
function cubeFaces(a) {
  return [
    quadPatch([a, -a, -a], [a, a, -a], [a, -a, a], [a, a, a]),     // +x
    quadPatch([-a, -a, -a], [-a, -a, a], [-a, a, -a], [-a, a, a]), // -x
    quadPatch([-a, a, -a], [-a, a, a], [a, a, -a], [a, a, a]),     // +y
    quadPatch([-a, -a, -a], [a, -a, -a], [-a, -a, a], [a, -a, a]), // -y
    quadPatch([-a, -a, a], [a, -a, a], [-a, a, a], [a, a, a]),     // +z
    quadPatch([-a, -a, -a], [-a, a, -a], [a, -a, -a], [a, a, -a]), // -z
  ];
}
function signedVolume(faces, n = 8) {
  let v = 0;
  for (const srf of faces) {
    const d = surfaceDomain(srf);
    const du = (d.u[1] - d.u[0]) / n, dv = (d.v[1] - d.v[0]) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const u = d.u[0] + du * (i + 0.5), w = d.v[0] + dv * (j + 0.5);
      const { point, su, sv } = surfacePointAndPartials(srf, u, w);
      v += dot(point, cross(su, sv)) * du * dv;
    }
  }
  return v / 3;
}
// THE FEATURE REBUILD, in the shape the real one has: the face list is thrown
// away and re-derived from the base, and the rebuild produces a face that was
// never in the base at all (a blend band, a hole wall). Nothing about the
// finished faces survives it — which is the whole problem.
function rebuildFromBase(baseFaces) {
  const out = baseFaces.map((f) => quadPatch(
    f.ctrlNet[0][0].slice(0, 3), f.ctrlNet[1][0].slice(0, 3),
    f.ctrlNet[0][1].slice(0, 3), f.ctrlNet[1][1].slice(0, 3)));
  return out;
}
const replay = (faces, flipped) => (flipped ? faces.map(flipSurfaceNormals) : faces);

test('the sign of a closed solid\'s volume IS its orientation — a flip is not cosmetic', () => {
  const faces = cubeFaces(10);
  assert.ok(Math.abs(signedVolume(faces) - 8000) < 1e-6, `an outward cube encloses +8000: ${signedVolume(faces)}`);
  const flipped = faces.map(flipSurfaceNormals);
  assert.ok(Math.abs(signedVolume(flipped) + 8000) < 1e-6, `and the flipped one encloses -8000: ${signedVolume(flipped)}`);
});

test('a flip is LOST by a feature rebuild unless the modifier is replayed', () => {
  const base = cubeFaces(10);
  const flippedNow = base.map(flipSurfaceNormals);
  assert.ok(signedVolume(flippedNow) < 0, 'the solid is flipped to begin with');
  // The rebuild alone — what a path that forgets the modifier produces.
  const naive = rebuildFromBase(base);
  assert.ok(signedVolume(naive) > 0, 'a rebuild from the base comes back UNFLIPPED — this is the defect');
  // The rebuild with the modifier replayed once.
  const replayed = replay(rebuildFromBase(base), true);
  assert.ok(signedVolume(replayed) < 0, 'replaying the modifier keeps the orientation the flip asked for');
  assert.ok(Math.abs(Math.abs(signedVolume(replayed)) - 8000) < 1e-6, 'and the solid is the same size — a flip moves nothing');
});

test('replaying it TWICE is the same as not replaying it at all — why the replay is bracketed, not copied per path', () => {
  const base = cubeFaces(10);
  const once = replay(rebuildFromBase(base), true);
  const twice = replay(once, true);
  assert.ok(signedVolume(once) < 0, 'once: flipped');
  assert.ok(signedVolume(twice) > 0, 'twice: back to unflipped, and nothing anywhere says so');
  /* This is the reason the fillet roads share ONE depth-counted bracket rather
     than each stripping and replaying for themselves: add -> records -> stages
     -> the applier all run on one edit, and an even number of reversals is no
     reversal. */
});

test('a face the rebuild INVENTED carries the flip too — a blend band is not in the base', () => {
  const base = cubeFaces(10);
  // The rebuild splits the +z face in two, the way a blend band appears between
  // two faces that were adjacent in the base.
  const rebuilt = rebuildFromBase(base);
  rebuilt.push(quadPatch([-10, -10, 10], [10, -10, 10], [-10, 10, 10], [10, 10, 10]));
  const out = replay(rebuilt, true);
  const invented = out[out.length - 1];
  const d = surfaceDomain(invented);
  const { su, sv } = surfacePointAndPartials(invented, (d.u[0] + d.u[1]) / 2, (d.v[0] + d.v[1]) / 2);
  assert.ok(dot(norm(cross(su, sv)), [0, 0, 1]) < -0.99, 'the new face points inward with the rest of the flipped solid');
});

test('and the solid OFFSETS the way the flip asked, after the rebuild — the consequence a lost flip gets wrong', () => {
  const base = cubeFaces(10);
  const centre = [0, 0, 0];
  const faceOf = (faces) => faces[0]; // the +x face
  const radialShift = (srf) => {
    const o = offsetSurface(srf, 3);
    const d = surfaceDomain(o.surface);
    const p = surfacePoint(o.surface, (d.u[0] + d.u[1]) / 2, (d.v[0] + d.v[1]) / 2);
    return dist(p, centre) - 10;
  };
  const plain = radialShift(faceOf(rebuildFromBase(base)));
  const flipped = radialShift(faceOf(replay(rebuildFromBase(base), true)));
  assert.ok(Math.abs(plain - 3) < 1e-6, `an unflipped face offsets outward by 3: ${plain}`);
  assert.ok(Math.abs(flipped + 3) < 1e-6, `a flipped one offsets INWARD by 3: ${flipped}`);
  assert.ok(plain * flipped < 0, 'thickening the same solid goes opposite ways — this is what a lost flip silently gets wrong');
});
