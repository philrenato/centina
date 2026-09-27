import test from 'node:test';
import assert from 'node:assert/strict';
import { intersectSurfaces, intersectSurfacesComplete, seedSurfaceIntersection, solveSquareSystem } from '../kernel/ssi.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { makeCircle, extrude, revolve } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';

test('solveSquareSystem solves a known small linear system exactly', () => {
  // 2x2: [[2,1],[1,3]] x = [5,10] -> x=[1,3]
  const x = solveSquareSystem([[2, 1], [1, 3]], [5, 10]);
  assert.ok(Math.abs(x[0] - 1) < 1e-9);
  assert.ok(Math.abs(x[1] - 3) < 1e-9);
});

test('solveSquareSystem returns null for a singular system rather than a wrong answer', () => {
  const x = solveSquareSystem([[1, 1], [2, 2]], [1, 2]);
  assert.equal(x, null);
});

// Two cylinders, built from this kernel's own extrude() (Ch.8 ruled
// surface) — an Extrude surface this app produces, closed in U (the
// circular cross-section), open in V (the finite length). Cylinders have no
// pole and no per-row arc weight structure, so the fixture does not depend
// on revolve()'s pole handling, while still being a "solid of revolution"
// cross-section and an Extrude surface, matching the case "two Revolve or
// Extrude surfaces intersecting".
//
// Deliberately unequal radii and an axis offset (not through a common
// point). With equal radii and both axes crossing at the origin (the
// textbook "Steinmetz solid"), the two elliptical intersection branches
// cross each other at (0,+-R,0) — algebraic singular points of the full
// intersection locus — and a coarse seed search landing near one of them
// correctly triggers this module's own near-tangent refusal (the direction
// is ill-defined exactly at a self-crossing point). Unequal radii + an
// offset axis is the representative case (two differently-sized pipes
// welded at an angle, axes not meeting) and produces a single smooth
// transversal loop with no such singularity.
function makeCylinderZ(R, halfHeight) {
  const profile = makeCircle([0, 0, -halfHeight], [1, 0, 0], [0, 1, 0], R, 4);
  return extrude(profile, [0, 0, 1], 2 * halfHeight); // axis Z through the origin
}
function makeCylinderXOffset(R, halfLength, zOffset) {
  const profile = makeCircle([-halfLength, 0, zOffset], [0, 1, 0], [0, 0, 1], R, 4);
  return extrude(profile, [1, 0, 0], 2 * halfLength); // axis X through (*, 0, zOffset)
}

test('cylinder fixtures: every evaluated point is at the correct radial distance from its own axis', () => {
  const R1 = 5, R2 = 3, z0 = 2;
  const cz = makeCylinderZ(R1, 9);
  const cx = makeCylinderXOffset(R2, 9, z0);
  const uMax = cz.knotsU[cz.knotsU.length - 1];
  for (let i = 0; i <= 8; i++) {
    const u = uMax * i / 8;
    const pz = surfacePoint(cz, u, 0.37);
    assert.ok(Math.abs(Math.hypot(pz[0], pz[1]) - R1) < 1e-9, `cylinderZ off-radius at u=${u}: ${pz}`);
    const px = surfacePoint(cx, u, 0.61);
    assert.ok(Math.abs(Math.hypot(px[1], px[2] - z0) - R2) < 1e-9, `cylinderX off-radius at u=${u}: ${px}`);
  }
});

// Two cylinders of different radii, perpendicular, offset axes — "two
// Revolve or Extrude surfaces intersecting" in its generic
// (non-degenerate) form. Known closed-form
// intersection (independent of this kernel, elementary algebra): cylinder1
// (axis Z through the origin, radius R1) is x^2+y^2=R1^2; cylinder2 (axis X
// through (*,0,z0), radius R2) is y^2+(z-z0)^2=R2^2. Substituting
// x=R1*cos(theta), y=R1*sin(theta) (an exact parametrization of cylinder1's
// own circle) gives (z-z0)^2 = R2^2 - R1^2*sin^2(theta) — real only where
// R1^2*sin^2(theta) <= R2^2, i.e. two disjoint theta arcs (R2<R1: the
// smaller pipe only pierces the larger one over a bounded front/back
// range), each combining the +/- sqrt branches into one smooth closed loop
// meeting where the sqrt hits zero: a pipe-through-pipe cut.
test('intersectSurfaces on two differently-sized, offset-axis perpendicular cylinders finds a closed loop matching the known analytic implicit equations exactly', () => {
  const R1 = 5, R2 = 3, z0 = 2;
  const cz = makeCylinderZ(R1, 9); // generous half-height, so the loop fits without being clipped
  const cx = makeCylinderXOffset(R2, 9, z0);
  // A smaller stepLen than the open-curve test below — this loop pinches
  // tightly near where its +/- z branches meet (the smaller cylinder just
  // grazing the edge of its own valid theta range), a higher-curvature
  // region where a larger predictor step degrades the corrector's own
  // convergence (stepLen=0.2 leaves ~1e-4 residual error right at the
  // pinch, stepLen=0.08 tightens it to ~8e-7).
  const result = intersectSurfaces(cz, cx, { stepLen: 0.08, maxSteps: 800 });
  assert.equal(result.ok, true, `expected an intersection, got refusal: ${result.reason}`);
  assert.equal(result.closed, true, 'a smaller cylinder piercing a larger one, front region only, must produce a closed loop');
  assert.ok(result.samples.length > 10, `expected a full march, got only ${result.samples.length} samples`);

  for (const s of result.samples) {
    const [x, y, z] = s.point;
    const r1 = Math.hypot(x, y); // must lie on cylinder1 (axis Z through origin)
    const r2 = Math.hypot(y, z - z0); // must lie on cylinder2 (axis X through z=z0)
    assert.ok(Math.abs(r1 - R1) < 1e-4, `sample off cylinder1: r1=${r1}`);
    assert.ok(Math.abs(r2 - R2) < 1e-4, `sample off cylinder2: r2=${r2}`);
  }

  const first = result.samples[0].point, last = result.samples[result.samples.length - 1].point;
  const closeDist = Math.hypot(first[0] - last[0], first[1] - last[1], first[2] - last[2]);
  assert.ok(closeDist < 1e-6, `loop did not close bit-exactly: ${closeDist}`);

  // The loop must stay on the "front" side (x>0, near cylinder1's own
  // u1=0) — a bounded loop, not something that wrapped implausibly.
  for (const s of result.samples) assert.ok(s.point[0] > 0, `expected the front-side loop to stay at x>0, got x=${s.point[0]}`);
});

test('intersectSurfaces on two cylinders far apart refuses (no intersection exists)', () => {
  const R1 = 5, R2 = 3;
  const cz = makeCylinderZ(R1, 9);
  const cxFar = makeCylinderXOffset(R2, 9, 100); // shifted far away in Z
  const result = intersectSurfaces(cz, cxFar);
  assert.equal(result.ok, false);
  assert.ok(/no intersection/i.test(result.reason), `expected a "no intersection" refusal, got: ${result.reason}`);
});

test('seedSurfaceIntersection on two overlapping cylinders finds a point on both surfaces (residual ~0)', () => {
  const R1 = 5, R2 = 3, z0 = 2;
  const cz = makeCylinderZ(R1, 9);
  const cx = makeCylinderXOffset(R2, 9, z0);
  const seed = seedSurfaceIntersection(cz, cx);
  assert.ok(seed.distance < 1e-6, `seed residual too large: ${seed.distance}`);
});

// Open curve case: the same two cylinders, but cylinder1's own half-height
// is shorter than the loop's own z-extent — the closed loop found above
// runs off cylinder1's own top/bottom parametric boundary (v1=1 / v1=0)
// partway through, so the curve is an open arc, terminating exactly where
// the loop would have crossed z=+-halfHeight1.
// Checks the boundary-exit path (the second SSI case: "an open
// curve... running off to a surface's own boundary"), not just the
// closed-loop path above.
test('intersectSurfaces on a short cylinder clipping the loop finds an open curve terminating exactly at the short cylinder boundary', () => {
  const R1 = 5, R2 = 3, z0 = 2;
  const halfHeight1 = 2.4; // shorter than the closed loop's own z-extent around z0 -> clips it
  const cz = makeCylinderZ(R1, halfHeight1);
  const cx = makeCylinderXOffset(R2, 9, z0);
  const result = intersectSurfaces(cz, cx, { stepLen: 0.15, maxSteps: 400 });
  assert.equal(result.ok, true, `expected an intersection, got refusal: ${result.reason}`);
  assert.equal(result.closed, false, 'the short cylinder clips the loop — this must be an open curve, not closed');
  assert.ok(result.samples.length > 5, `expected a full march, got only ${result.samples.length} samples`);

  for (const s of result.samples) {
    const [x, y, z] = s.point;
    const r1 = Math.hypot(x, y);
    const r2 = Math.hypot(y, z - z0);
    assert.ok(Math.abs(r1 - R1) < 1e-4, `sample off cylinder1: r1=${r1}`);
    assert.ok(Math.abs(r2 - R2) < 1e-4, `sample off cylinder2: r2=${r2}`);
    assert.ok(Math.abs(z) <= halfHeight1 + 1e-4, `sample z=${z} must stay within cylinder1's own +-${halfHeight1} bound`);
  }

  const zFirst = result.samples[0].point[2];
  const zLast = result.samples[result.samples.length - 1].point[2];
  assert.ok(Math.abs(Math.abs(zFirst) - halfHeight1) < 1e-4, `start z=${zFirst} should be at the cylinder1 boundary +-${halfHeight1}`);
  assert.ok(Math.abs(Math.abs(zLast) - halfHeight1) < 1e-4, `end z=${zLast} should be at the cylinder1 boundary +-${halfHeight1}`);
});

// Closure must not overshoot into a second lap.
//
// Testing closure against the current sample's distance to the seed, with
// a tolerance of half a step, is marginal by construction: with samples one
// step apart, the nearest one to the seed can sit a full half-step away, so
// a seed landing midway between two samples is a dead tie decided by float
// noise. On this fixture the two samples straddling the seed read 0.6140
// and 0.6145 against a tolerance of 0.6141 and 0.6142; a missed closure
// keeps marching and closes a full lap later. Closure is therefore tested
// against the distance to the marched segment.
//
// The failure is silent and lands three stages downstream: a doubly-traced
// loop self-overlaps in UV, the face arrangement cuts hundreds of fragments
// out of it, and the sew reports non-manifold -- which reads like a topology
// bug, not a marching one. Hence an explicit lap count here: sample count
// alone would not name what went wrong.
//
// Against a sample-distance test this fixture marches 403 samples over
// 2.0000 laps; against the segment-distance test, 204 samples over 1.0000
// laps. Segment distance <= endpoint distance always, so the segment test
// can only close earlier, never later.
test('a marched closed loop closes on its first lap, not its second', () => {
  const R = 40;
  const centres = { B: [55, 0, 0], C: [27, 46, 0] };
  // Two R=40 spheres. Deliberately not exact spheres -- these are revolves of
  // a 7-point interpolated profile, i.e. what this app builds, whose
  // intersection curve is a wavy non-planar loop. The aliasing this guards
  // against depends on the marched step spacing, which an idealized exact
  // circle would not reproduce.
  const profile = (() => {
    const pts = [];
    for (let i = 0; i < 7; i++) {
      const th = -Math.PI / 2 + (Math.PI * i) / 6;
      pts.push([R * Math.cos(th), 0, R * Math.sin(th)]);
    }
    return globalCurveInterp(pts, 3);
  })();
  const sphereAt = (c) => {
    const s = revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
    return { ...s, ctrlNet: s.ctrlNet.map((row) => row.map(([x, y, z, w]) => [x + c[0] * w, y + c[1] * w, z + c[2] * w, w])) };
  };
  const B = sphereAt(centres.B), C = sphereAt(centres.C);

  const res = intersectSurfacesComplete(B, C);
  assert.ok(res.ok, `SSI should find the B/C intersection: ${res.reason ?? ''}`);
  assert.equal(res.components.length, 1, 'two overlapping spheres share exactly one intersection loop');
  const comp = res.components[0];
  assert.ok(comp.closed, 'and that loop is closed');

  // Total turn about the exact circle axis. For two equal-radius spheres the
  // intersection plane is perpendicular to the center line, so the axis is
  // known in closed form -- no centroid fitting, nothing derived from the
  // marched samples themselves, so this cannot agree with the code under test
  // by construction.
  const d = centres.C.map((x, i) => x - centres.B[i]);
  const dLen = Math.hypot(...d);
  const axis = d.map((x) => x / dLen);
  const mid = centres.B.map((x, i) => x + axis[i] * (dLen / 2));
  const tmp = Math.abs(axis[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const e1raw = cross(axis, tmp), e1L = Math.hypot(...e1raw);
  const e1 = e1raw.map((x) => x / e1L), e2 = cross(axis, e1);

  let turn = 0, prev = null;
  for (const s of comp.samples) {
    const rel = s.point.map((x, i) => x - mid[i]);
    const ang = Math.atan2(dot(rel, e2), dot(rel, e1));
    if (prev !== null) {
      let dA = ang - prev;
      while (dA > Math.PI) dA -= 2 * Math.PI;
      while (dA < -Math.PI) dA += 2 * Math.PI;
      turn += dA;
    }
    prev = ang;
  }
  const laps = Math.abs(turn) / (2 * Math.PI);
  assert.ok(Math.abs(laps - 1) < 0.02, `the loop should be traced exactly once, got ${laps.toFixed(4)} laps`);
});
