// Root frames and fibers (kernel/fiber.mjs) — the core of Phase 1.
//
// Fixtures are real revolved surfaces, not hand-written control nets: a
// sphere (closed in V, degenerate at both poles) and a cylinder (closed in V,
// no poles). Between them they carry every case the emitter has to refuse or
// wrap, which a flat test patch would not.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeArc, makeLine, makeCircle, revolve } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { reverseCurve } from '../kernel/curve.mjs';
import { dot, cross, length, sub, normalize } from '../kernel/vec3.mjs';
import {
  surfaceRootFrames, surfaceFrameAt, aimDirection,
  clampToHemisphere, fiberPoints, knotDomain,
  curveRootFrames, curvePlane, planarLoopSignedArea,
} from '../kernel/fiber.mjs';

const R = 20;
// A half-meridian from south pole to north pole, revolved a full turn: a real
// sphere, with genuinely degenerate poles at v=0 and v=1.
const meridian = makeArc([0, 0, 0], [0, 0, 1], [1, 0, 0], R, 0, Math.PI); // north pole -> equator -> south pole, in XZ
const sphere = revolve(meridian, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
// A cylinder wall: closed in u, no poles anywhere.
const cylinder = revolve(makeLine([R, 0, -30], [R, 0, 30]), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);

test('frames on a sphere are orthonormal', () => {
  const { frames } = surfaceRootFrames(sphere, 8, 6);
  assert.ok(frames.length >= 20, `enough frames to be a real sample (${frames.length})`);
  for (const f of frames) {
    assert.ok(Math.abs(length(f.normal) - 1) < 1e-9, 'normal is unit');
    assert.ok(Math.abs(length(f.tangentU) - 1) < 1e-9, 'tangentU is unit');
    assert.ok(Math.abs(length(f.tangentV) - 1) < 1e-9, 'tangentV is unit');
    assert.ok(Math.abs(dot(f.normal, f.tangentU)) < 1e-9, 'normal perpendicular to tangentU');
    assert.ok(Math.abs(dot(f.normal, f.tangentV)) < 1e-9, 'normal perpendicular to tangentV');
    assert.ok(Math.abs(dot(f.tangentU, f.tangentV)) < 1e-9, 'tangentU perpendicular to tangentV');
    // Right-handed: tangentU x tangentV must be the normal, not its opposite.
    const handed = cross(f.tangentU, f.tangentV);
    assert.ok(dot(handed, f.normal) > 0.999, 'basis is right-handed');
  }
});

test('every sphere frame is radial — the normal really points out of the surface', () => {
  const { frames } = surfaceRootFrames(sphere, 10, 7);
  for (const f of frames) {
    const radial = normalize(f.position);
    const alignment = Math.abs(dot(f.normal, radial));
    // On a sphere the surface normal IS the radial direction, so this is an
    // analytic check against the geometry rather than against the emitter's
    // own output. Sign may be inward or outward depending on the revolve's
    // own winding; what must hold is that it is radial at all.
    assert.ok(alignment > 0.999, `normal is radial (got ${alignment.toFixed(6)})`);
    assert.ok(Math.abs(length(f.position) - R) < 1e-6, 'root really lies on the sphere');
  }
});

test('a degenerate pole is refused, not emitted as a NaN fiber', () => {
  // Sampling the poles directly: one of v=0 / v=1 is the collapsed row.
  // The poles are the PROFILE's own ends, so they live at the extremes of u —
  // and u runs [0,2] here, not [0,1]. Reading the domain off the surface is the
  // whole point; hard-coding 0..1 is what hid this the first time.
  const [u0, u1] = knotDomain(sphere.knotsU, sphere.degU);
  const atPole = [surfaceFrameAt(sphere, u0, 0.5), surfaceFrameAt(sphere, u1, 0.5)];
  assert.ok(atPole.some((f) => f === null), 'at least one pole row refuses');
  for (const f of atPole) {
    if (!f) continue;
    for (const v of [...f.normal, ...f.tangentU, ...f.tangentV]) assert.ok(Number.isFinite(v), 'no NaN escapes a pole');
  }
  // And a grid spanning the poles reports the refusals rather than swallowing
  // them — a count of zero and a silent drop look identical from outside.
  const { frames, skippedPoles } = surfaceRootFrames(sphere, 6, 5);
  assert.ok(skippedPoles > 0, `poles were counted (${skippedPoles})`);
  for (const f of frames) for (const v of [...f.position, ...f.normal]) assert.ok(Number.isFinite(v), 'no NaN in any kept frame');
});

test('a closed direction does not lay a double row down its seam', () => {
  const { frames } = surfaceRootFrames(cylinder, 8, 3);
  // V is the closed (revolved) direction here. If the emitter sampled both
  // ends of it, it would place two coincident columns of roots on the seam.
  const seen = [];
  for (const f of frames) {
    for (const g of seen) {
      assert.ok(length(sub(f.position, g.position)) > 1e-6,
        `no two roots are coincident (seam duplicated at ${JSON.stringify(f.position)})`);
    }
    seen.push(f);
  }
});

test('aim: lift 90 is the normal, lift 0 sweep 0 is tangentU', () => {
  const f = surfaceFrameAt(cylinder, 0.5, 1.0);
  assert.ok(f, 'a real frame on the cylinder wall');
  const out = aimDirection(f, 90, 0);
  assert.ok(dot(out, f.normal) > 0.999999, 'lift 90 aims along the normal');
  const flat = aimDirection(f, 0, 0);
  assert.ok(dot(flat, f.tangentU) > 0.999999, 'lift 0 / sweep 0 aims along tangentU');
  const across = aimDirection(f, 0, 90);
  assert.ok(dot(across, f.tangentV) > 0.999999, 'sweep 90 aims along tangentV');
  // The whole point of two angles: sweeping lift is a continuous rise out of
  // the tangent plane, never a mode switch.
  let prev = -1;
  for (let d = 0; d <= 90; d += 10) {
    const c = dot(aimDirection(f, d, 0), f.normal);
    assert.ok(c > prev, `lift ${d} rises monotonically out of the plane`);
    prev = c;
  }
});

test('T1 refuses to launch a fiber INTO its host', () => {
  const f = surfaceFrameAt(cylinder, 0.5, 1.0);
  const inward = normalize([-f.normal[0], -f.normal[1], -f.normal[2]]);
  const clamped = clampToHemisphere(inward, f.normal, 5);
  assert.ok(dot(clamped, f.normal) > 0, 'a fiber aimed straight into the host is turned outward');
  assert.ok(Math.abs(length(clamped) - 1) < 1e-9, 'and stays a unit direction');
  // A direction already comfortably outward is left alone, or the clamp would
  // be quietly re-aiming every fiber rather than guarding the bad ones.
  const fine = aimDirection(f, 60, 0);
  const same = clampToHemisphere(fine, f.normal, 5);
  assert.ok(length(sub(fine, same)) < 1e-12, 'an already-valid direction is untouched');
});

test('T2 keeps droop out of the host — and is shown to be doing something', () => {
  const f = surfaceFrameAt(cylinder, 0.5, 1.0);
  const opts = { cvs: 12, lengthValue: 30, liftDeg: 90, droopAmount: 1.2, recoilAmount: 0, intensity: 1 };
  // Droop pointed straight back into the wall: the concave-host case, forced.
  const into = [-f.normal[0], -f.normal[1], -f.normal[2]];

  const guarded = fiberPoints(f, { ...opts, droopDir: into, tangentPlaneGuard: true });
  for (const p of guarded) {
    assert.ok(dot(sub(p, f.position), f.normal) > -1e-9, 'every guarded sample stays on the outward side of its root tangent plane');
  }

  // THE CONTROL. A guard that cannot be shown to change the outcome is not
  // tested — with it off, this same fixture must genuinely violate.
  const bare = fiberPoints(f, { ...opts, droopDir: into, tangentPlaneGuard: false });
  const violations = bare.filter((p) => dot(sub(p, f.position), f.normal) < -1e-9);
  assert.ok(violations.length > 0, 'with the guard off the same fixture really does penetrate');
});

test('a fiber starts exactly at its root and has the requested point count', () => {
  const f = surfaceFrameAt(sphere, 1.0, 1.0);
  const pts = fiberPoints(f, { cvs: 6, lengthValue: 12 });
  assert.equal(pts.length, 6);
  assert.ok(length(sub(pts[0], f.position)) < 1e-9, 'the first sample IS the root');
  for (const p of pts) for (const v of p) assert.ok(Number.isFinite(v), 'no NaN anywhere in a fiber');
  // It must actually go somewhere — a fiber collapsed to its root would pass
  // every check above.
  assert.ok(length(sub(pts[pts.length - 1], f.position)) > 1, 'the fiber has real extent');
});

// ---------------------------------------------------------------------------
// CURVE HOST
//
// Fixtures are the two the spec itself names, plus a line. A CIRCLE checks that
// outward means outward at all; a 5-POINT STAR checks it at reflex vertices,
// where a locally-derived side would flip; a HELIX checks the frame does not
// twist, which is the silent-wrong-result case the spec calls out by name.
// ---------------------------------------------------------------------------

// A closed degree-1 loop through `pts`. Degree 1 with n control points needs
// n + 2 knots, clamped at both ends.
function closedPolylineCrv(pts) {
  const ctrlPts = [...pts.map((p) => [p[0], p[1], p[2], 1]), [pts[0][0], pts[0][1], pts[0][2], 1]];
  const n = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= n - 2; i++) knots.push(i);
  knots.push(n - 1, n - 1);
  return { degree: 1, knots, ctrlPts };
}

function starPoints(outer, inner, arms, z = 0) {
  const pts = [];
  for (let i = 0; i < arms * 2; i++) {
    const a = (i / (arms * 2)) * Math.PI * 2;
    const r = i % 2 === 0 ? outer : inner;
    pts.push([Math.cos(a) * r, Math.sin(a) * r, z]);
  }
  return pts;
}

// Is a 2D point inside this polygon? Ray casting, used to prove "outward"
// geometrically rather than by comparing against a centroid — on a star the
// centroid test passes for the wrong reason at an arm tip and fails at a notch.
function insidePolygonXY(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// NET rotation of the frame about its own tangent along the chain — SIGNED, and
// that matters. Summing absolute per-step angles measures the difference between
// two RMF approximations (this reference uses projection, the emitter uses
// double reflection) and accumulates ~0.4 rad of pure discretization noise over
// 60 stations while the frame is not twisting at all. Twist is a NET quantity:
// it is what fails to cancel. A carried frame nets ~0; a per-point arbitrary
// perpendicular nets the whole rotation of its reference vector.
function netTwist(frames) {
  let sum = 0;
  for (let i = 1; i < frames.length; i++) {
    const t = frames[i].tangentU;
    const prev = frames[i - 1].normal;
    // Previous normal, projected into the plane perpendicular to THIS tangent,
    // is where a non-twisting frame would have put this normal.
    const proj = sub(prev, scaleV(t, dot(prev, t)));
    if (length(proj) < 1e-12) continue;
    const expected = normalize(proj);
    const c = Math.max(-1, Math.min(1, dot(expected, frames[i].normal)));
    const ang = Math.acos(c);
    // Which way round the tangent it turned.
    sum += dot(cross(expected, frames[i].normal), t) < 0 ? -ang : ang;
  }
  return sum;
}
function scaleV(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }

const circle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 25, 8);

test('a circle is detected as planar, a helix is not, and a line has no plane', () => {
  const cp = curvePlane(circle);
  assert.ok(cp, 'the circle is planar');
  assert.ok(Math.abs(Math.abs(dot(cp.normal, [0, 0, 1])) - 1) < 1e-9, 'and its plane is the XY plane');

  const helixPts = [];
  for (let i = 0; i <= 40; i++) {
    const a = (i / 40) * Math.PI * 4;
    helixPts.push([Math.cos(a) * 20, Math.sin(a) * 20, (i / 40) * 60]);
  }
  assert.equal(curvePlane(globalCurveInterp(helixPts, 3)), null, 'a helix is not planar');

  // A straight line lies in infinitely many planes, so there is no outward to
  // pick and the emitter must fall back rather than invent a side.
  assert.equal(curvePlane(makeLine([0, 0, 0], [10, 10, 10])), null, 'a line reports no plane');
});

test('roots on a curve are spaced by real arc length, not by parameter', () => {
  const { frames } = curveRootFrames(circle, 12);
  assert.equal(frames.length, 12, 'a closed curve gets exactly count roots, no seam duplicate');
  const gaps = [];
  for (let i = 0; i < frames.length; i++) {
    gaps.push(length(sub(frames[(i + 1) % frames.length].position, frames[i].position)));
  }
  const spread = Math.max(...gaps) - Math.min(...gaps);
  assert.ok(spread < Math.max(...gaps) * 0.02, `equal arc spacing on a circle (spread ${spread.toFixed(6)})`);
});

test('frames on a planar curve are orthonormal, outward, and lie in the plane', () => {
  const { frames, planar } = curveRootFrames(circle, 16);
  assert.ok(planar, 'reported planar');
  for (const f of frames) {
    assert.ok(Math.abs(length(f.normal) - 1) < 1e-9, 'normal is unit');
    assert.ok(Math.abs(dot(f.normal, f.tangentU)) < 1e-9, 'normal perpendicular to tangent');
    assert.ok(Math.abs(dot(f.normal, f.tangentV)) < 1e-9, 'normal perpendicular to tangentV');
    // tangentV is the PLANE normal on a planar curve, which is what makes
    // "flat in the plane, then lift out of it" mean anything.
    assert.ok(Math.abs(Math.abs(dot(f.tangentV, [0, 0, 1])) - 1) < 1e-9, 'tangentV is the plane normal');
    // Outward: away from the center, for a circle.
    assert.ok(dot(f.normal, normalize(f.position)) > 0.999, 'normal points away from the center');
  }
});

test('outward stays outward when the winding reverses', () => {
  // Same circle traversed the other way. A per-point perpendicular with no
  // winding term points INWARD for this one, which is the bug the signed area
  // exists to prevent.
  const { frames } = curveRootFrames(reverseCurve(circle), 16);
  for (const f of frames) {
    assert.ok(dot(f.normal, normalize(f.position)) > 0.999, 'still points away from the center');
  }
});

test('a 5-point star points outward at its reflex vertices too', () => {
  const pts = starPoints(30, 12, 5);
  const star = closedPolylineCrv(pts);
  const { frames, planar } = curveRootFrames(star, 40);
  assert.ok(planar, 'the star is planar');
  assert.ok(frames.length >= 30, `a real sample of roots (${frames.length})`);
  let outside = 0;
  for (const f of frames) {
    // Step a short way along the normal. If the normal is genuinely outward,
    // that lands OUTSIDE the polygon — at an arm tip and in a notch alike.
    const probe = [f.position[0] + f.normal[0] * 2, f.position[1] + f.normal[1] * 2, 0];
    if (!insidePolygonXY(probe, pts)) outside++;
  }
  assert.equal(outside, frames.length, `every root's normal leaves the star (${outside}/${frames.length})`);
});

test('a non-planar curve gets a rotation-minimizing frame, not a per-point perpendicular', () => {
  const helixPts = [];
  for (let i = 0; i <= 60; i++) {
    const a = (i / 60) * Math.PI * 4;
    helixPts.push([Math.cos(a) * 20, Math.sin(a) * 20, (i / 60) * 60]);
  }
  const helix = globalCurveInterp(helixPts, 3);
  const { frames, planar } = curveRootFrames(helix, 60);
  assert.equal(planar, false, 'the helix is not planar');
  for (const f of frames) {
    assert.ok(Math.abs(length(f.normal) - 1) < 1e-9, 'normal is unit');
    assert.ok(Math.abs(dot(f.normal, f.tangentU)) < 1e-9, 'normal perpendicular to tangent');
  }

  // The differential the spec asks for: the same stations, framed the naive way
  // (a fresh arbitrary perpendicular per point) twist substantially over two
  // turns; the carried frame must not.
  const naive = frames.map((f) => {
    const t = f.tangentU;
    const ref = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    return { tangentU: t, normal: normalize(cross(t, ref)) };
  });
  const carried = netTwist(frames);
  const arbitrary = netTwist(naive);
  assert.ok(Math.abs(carried) < 0.05, `the carried frame nets no twist (${carried.toFixed(6)} rad total)`);
  assert.ok(Math.abs(arbitrary) > Math.abs(carried) * 20, `and a per-point perpendicular twists far more (${arbitrary.toFixed(3)} vs ${carried.toFixed(6)} rad net)`);
});
