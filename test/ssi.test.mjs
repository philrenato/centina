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

// Two real cylinders, built from this kernel's own extrude() (Ch.8 ruled
// surface) — a real Extrude surface this app already produces, closed in U
// (the circular cross-section), open in V (the finite length). Chosen over
// a Revolve-of-an-arc sphere/barrel specifically because probing this file
// surfaced a genuine, pre-existing exactness gap in revolve()'s own pole
// handling (named separately, and fixed in its own round) —
// cylinders sidestep that unrelated issue entirely (no pole, no per-row arc
// weight structure at all) while still being a real "solid of revolution"
// cross-section and a real Extrude surface, matching the named
// example ("two Revolve or Extrude surfaces intersecting").
//
// DELIBERATELY UNEQUAL radii and an axis OFFSET (not through a common
// point) — a first attempt at this test used EQUAL radii with both axes
// crossing at the origin (the textbook "Steinmetz solid"), which turned out
// to be a genuinely bad canonical test case: that configuration's two
// elliptical intersection branches actually CROSS EACH OTHER at (0,+-R,0) —
// real algebraic singular points of the full intersection locus, not a
// fixture mistake — and the coarse seed search kept landing near one of
// them, correctly triggering this module's own near-tangent refusal (the
// direction is genuinely ill-defined exactly at a self-crossing point).
// Unequal radii + an offset axis is the actually-representative case (two
// differently-sized pipes welded at an angle, axes not meeting) and
// produces a single smooth transversal loop with no such singularity —
// confirmed directly, not assumed, before trusting this as the test fixture.
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

// TWO REAL CYLINDERS OF DIFFERENT RADII, PERPENDICULAR, OFFSET AXES — the
// task's own named "two Revolve or Extrude surfaces intersecting" student
// scenario, in its generic (non-degenerate) form. KNOWN closed-form
// intersection (independent of this kernel, elementary algebra): cylinder1
// (axis Z through the origin, radius R1) is x^2+y^2=R1^2; cylinder2 (axis X
// through (*,0,z0), radius R2) is y^2+(z-z0)^2=R2^2. Substituting
// x=R1*cos(theta), y=R1*sin(theta) (an exact parametrization of cylinder1's
// own circle) gives (z-z0)^2 = R2^2 - R1^2*sin^2(theta) — real only where
// R1^2*sin^2(theta) <= R2^2, i.e. two disjoint theta arcs (R2<R1: the
// smaller pipe only pierces the larger one over a bounded front/back
// range), each combining the +/- sqrt branches into ONE smooth closed loop
// meeting where the sqrt hits zero. A real pipe-through-pipe cut, not a
// synthetic edge case.
test('intersectSurfaces on two real, differently-sized, offset-axis perpendicular cylinders finds a CLOSED loop matching the known analytic implicit equations exactly', () => {
  const R1 = 5, R2 = 3, z0 = 2;
  const cz = makeCylinderZ(R1, 9); // generous half-height, so the loop fits without being clipped
  const cx = makeCylinderXOffset(R2, 9, z0);
  // A smaller stepLen than the open-curve test below — this loop pinches
  // tightly near where its +/- z branches meet (the smaller cylinder just
  // grazing the edge of its own valid theta range), a real higher-curvature
  // region where a larger predictor step measurably degrades the
  // corrector's own convergence (found directly: stepLen=0.2 left ~1e-4
  // residual error right at the pinch, stepLen=0.08 tightens it to ~8e-7).
  const result = intersectSurfaces(cz, cx, { stepLen: 0.08, maxSteps: 800 });
  assert.equal(result.ok, true, `expected a real intersection, got refusal: ${result.reason}`);
  assert.equal(result.closed, true, 'a smaller cylinder piercing a larger one, front region only, must produce a CLOSED loop');
  assert.ok(result.samples.length > 10, `expected a real march, got only ${result.samples.length} samples`);

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

  // The loop must genuinely stay on the "front" side (x>0, near cylinder1's
  // own u1=0) — a real, bounded loop, not something that wrapped implausibly.
  for (const s of result.samples) assert.ok(s.point[0] > 0, `expected the front-side loop to stay at x>0, got x=${s.point[0]}`);
});

test('intersectSurfaces on two cylinders far apart honestly refuses (no intersection exists)', () => {
  const R1 = 5, R2 = 3;
  const cz = makeCylinderZ(R1, 9);
  const cxFar = makeCylinderXOffset(R2, 9, 100); // shifted far away in Z
  const result = intersectSurfaces(cz, cxFar);
  assert.equal(result.ok, false);
  assert.ok(/no intersection/i.test(result.reason), `expected an honest "no intersection" refusal, got: ${result.reason}`);
});

test('seedSurfaceIntersection on two overlapping cylinders finds a point genuinely on BOTH surfaces (residual ~0)', () => {
  const R1 = 5, R2 = 3, z0 = 2;
  const cz = makeCylinderZ(R1, 9);
  const cx = makeCylinderXOffset(R2, 9, z0);
  const seed = seedSurfaceIntersection(cz, cx);
  assert.ok(seed.distance < 1e-6, `seed residual too large: ${seed.distance}`);
});

// OPEN CURVE case: the SAME two cylinders, but cylinder1's own half-height
// is now shorter than the loop's own real z-extent — the true closed loop
// found above genuinely runs off cylinder1's own top/bottom parametric
// boundary (v1=1 / v1=0) partway through, so the real curve is an OPEN ARC,
// terminating exactly where the loop would have crossed z=+-halfHeight1.
// Proves the boundary-exit path (the second SSI case: "an open
// curve... running off to a surface's own boundary"), not just the
// closed-loop path above.
test('intersectSurfaces on a SHORT cylinder clipping the loop finds an OPEN curve terminating exactly at the short cylinder boundary', () => {
  const R1 = 5, R2 = 3, z0 = 2;
  const halfHeight1 = 2.4; // shorter than the closed loop's own z-extent around z0 -> clips it
  const cz = makeCylinderZ(R1, halfHeight1);
  const cx = makeCylinderXOffset(R2, 9, z0);
  const result = intersectSurfaces(cz, cx, { stepLen: 0.15, maxSteps: 400 });
  assert.equal(result.ok, true, `expected a real intersection, got refusal: ${result.reason}`);
  assert.equal(result.closed, false, 'the short cylinder clips the loop — this must be an OPEN curve, not closed');
  assert.ok(result.samples.length > 5, `expected a real march, got only ${result.samples.length} samples`);

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

// CLOSURE MUST NOT OVERSHOOT INTO A SECOND LAP.
//
// The marcher used to test closure against the current SAMPLE's distance to
// the seed, with a tolerance of half a step. That is marginal BY
// CONSTRUCTION: with samples one step apart, the nearest one to the seed can
// sit a full half-step away, so a seed landing midway between two samples is
// a dead tie decided by float noise. Measured on this exact fixture before
// the fix: the two samples straddling the seed read 0.6140 and 0.6145
// against a tolerance of 0.6141 and 0.6142. The march missed, kept going,
// and closed a full lap later.
//
// The failure is silent and lands three stages downstream: a doubly-traced
// loop self-overlaps in UV, the face arrangement cuts hundreds of fragments
// out of it, and the sew reports NON-MANIFOLD -- which reads like a topology
// bug, not a marching one. It is worth an explicit lap count here for that
// reason: sample count alone would not name what went wrong.
//
// NEGATIVE CONTROL, run by hand when this landed: against the old
// sample-distance test this fixture marched 403 samples over 2.0000 laps;
// against the segment-distance test, 204 samples over 1.0000 laps. The other
// two pairs of the same three-sphere fixture were byte-identical either way
// (230 and 244 samples, 1.0000 laps) -- segment distance <= endpoint
// distance always, so the new test can only ever close EARLIER, never later.
test('a marched closed loop closes on its FIRST lap, not its second', () => {
  const R = 40;
  const centres = { B: [55, 0, 0], C: [27, 46, 0] };
  // Two R=40 spheres. Deliberately NOT exact spheres -- these are revolves of
  // a 7-point interpolated profile, i.e. what this app actually builds, whose
  // intersection curve is a genuinely wavy non-planar loop. That is the point:
  // the aliasing this guards against depends on real step spacing, which an
  // idealised exact circle would not reproduce.
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

  // Total turn about the EXACT circle axis. For two equal-radius spheres the
  // intersection plane is perpendicular to the centre line, so the axis is
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
