import test from 'node:test';
import assert from 'node:assert/strict';
import { makeArc, makeCircle } from '../kernel/primitives.mjs';
import { buildParallelTransportFrames, sweep1Rigid, localizeSectionToFrame } from '../kernel/sweep.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';
import { curvePoint, buildArcLengthTable, paramAtArcLength } from '../kernel/curve.mjs';
import { dot, length, sub } from '../kernel/vec3.mjs';
import { projectPointToPlane, projectCurveToPlane } from '../kernel/project.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';

function assertOrthonormal(f, tol = 1e-8) {
  assert.ok(Math.abs(length(f.xAxis) - 1) < tol, `xAxis length ${length(f.xAxis)}`);
  assert.ok(Math.abs(length(f.yAxis) - 1) < tol, `yAxis length ${length(f.yAxis)}`);
  assert.ok(Math.abs(length(f.zAxis) - 1) < tol, `zAxis length ${length(f.zAxis)}`);
  assert.ok(Math.abs(dot(f.xAxis, f.yAxis)) < tol, `x.y ${dot(f.xAxis, f.yAxis)}`);
  assert.ok(Math.abs(dot(f.xAxis, f.zAxis)) < tol, `x.z ${dot(f.xAxis, f.zAxis)}`);
  assert.ok(Math.abs(dot(f.yAxis, f.zAxis)) < tol, `y.z ${dot(f.yAxis, f.zAxis)}`);
}

test('parallel-transport frames along a curved rail are all orthonormal', () => {
  // A quarter-circle rail, a plausible spout/handle path.
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 5, 0, Math.PI / 2);
  const frames = buildParallelTransportFrames(rail);
  assert.ok(frames.length > 0);
  for (const f of frames) assertOrthonormal(f);
});

test('parallel-transport frames stay orthonormal along a STRAIGHT rail too (no Frenet flip)', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 0, 10, 1]] };
  const frames = buildParallelTransportFrames(rail);
  for (const f of frames) assertOrthonormal(f);
});

test('sweep1Rigid of a small circle section along a curved rail is finite and orthonormal', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 8, 0, Math.PI / 3);
  const section = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.5);
  const srf = sweep1Rigid(rail, section);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  for (const f of srf.frames) assertOrthonormal(f);
  // spot-check the surface is actually tube-like: every U ring at a fixed V
  // stays roughly `radius` from that frame's origin.
  const f0 = srf.frames[0];
  for (let u = 0; u <= 4; u += 0.3) {
    const p = surfacePoint(srf, u, srf.knotsV[srf.degV]);
    const d = length(sub(p, f0.origin));
    assert.ok(Math.abs(d - 0.5) < 1e-6, `u=${u} d=${d}`);
  }
});

// localizeSectionToFrame is the piece a real app command needs that the
// teapot's own sweep1Rigid calls never exercised — every teapot section is
// ALREADY a circle centered at the world origin (sweep1Rigid's own
// "local offsets" convention, trivially satisfied). A real Sweep1 command
// picks a profile curve ANYWHERE in the document, in world coordinates,
// and must re-project it first.
test('localizeSectionToFrame + sweep1Rigid: a profile drawn OFF-ORIGIN, perpendicular to a straight rail, sweeps starting from its OWN original position exactly', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 0, 20, 1]] };
  // A circle drawn at (5,3,0) — well off both the rail's own line AND the
  // world origin — but still IN the plane perpendicular to the rail's
  // tangent (XY, since the rail runs along Z), so nothing is lost when
  // localizeSectionToFrame drops the frame's own tangent-direction
  // component.
  const profile = makeCircle([5, 3, 0], [1, 0, 0], [0, 1, 0], 2);
  const frames = buildParallelTransportFrames(rail);
  const local = localizeSectionToFrame(profile, frames[0]);
  const srf = sweep1Rigid(rail, local);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  // Decompose-then-reconstruct through the SAME frame is an exact
  // orthonormal-basis round-trip — the swept surface's own v=0 boundary
  // should reproduce the ORIGINAL profile curve exactly, not an
  // approximation, regardless of which direction the frame's own X/Y
  // axes actually point in (parallel transport's first frame is built
  // from anyPerpendicular, not necessarily world X/Y).
  const vMin = srf.knotsV[0];
  for (let u = 0; u <= 4; u += 0.11) {
    const expected = curvePoint(profile, u);
    const actual = surfacePoint(srf, u, vMin);
    const d = length(sub(actual, expected));
    assert.ok(d < 1e-9, `u=${u}: expected ${expected}, got ${actual}, d=${d}`);
  }
  // And the FAR end of the rail (v=vMax) has genuinely moved away from
  // the original profile's own position — a real sweep, not a no-op.
  const vMax = srf.knotsV[srf.knotsV.length - 1];
  const nearEnd = surfacePoint(srf, 0, vMin);
  const farEnd = surfacePoint(srf, 0, vMax);
  assert.ok(length(sub(farEnd, nearEnd)) > 15, 'the swept surface actually spans the ~20mm rail length, not degenerate');
});

// Real bug, found from a swept surface that visually didn't hug its own
// profile curve's shape: localizeSectionToFrame/sweep1Rigid used to keep
// only the LOCAL X/Y of a picked profile, silently flattening any depth
// along the frame's own tangent direction into a single plane. A profile
// with real 3D shape (not confined to a plane perpendicular to the rail)
// must sweep with that shape intact, not a flattened projection of it.
test('localizeSectionToFrame + sweep1Rigid: a NON-PLANAR profile (real depth along the rail\'s own tangent direction) sweeps with its FULL 3D shape intact, not flattened', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 20, 0, Math.PI / 2);
  // A profile that rises in Z as it goes — NOT confined to the plane
  // perpendicular to the rail's own start tangent.
  const profile = { degree: 3, knots: [0, 0, 0, 0, 1, 1, 1, 1], ctrlPts: [[5, 3, 0, 1], [5, 3, 4, 1], [8, 3, 4, 1], [8, 3, 8, 1]] };
  const frames = buildParallelTransportFrames(rail);
  const local = localizeSectionToFrame(profile, frames[0]);
  const srf = sweep1Rigid(rail, local);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  const vMin = srf.knotsV[0];
  for (let u = 0; u <= 1; u += 0.13) {
    const expected = curvePoint(profile, u);
    const actual = surfacePoint(srf, u, vMin);
    const d = length(sub(actual, expected));
    assert.ok(d < 1e-9, `u=${u}: expected ${expected}, got ${actual}, d=${d} — the profile's own Z-depth must survive the round-trip, not get flattened to 0`);
  }
});

// Real bug, found from live testing with a Rhino comparison and a direct
// report: "they are controlling 2 edges of a 4 edge NURBS patch... the
// surface should always have two edges that match the curvature and
// position of the curves [when they touch]." Confirmed true for a
// STRAIGHT rail (2 control points — curve value at a clamped endpoint
// always equals that endpoint's own control point, so frame origins were
// already exact there) but FALSE for a genuinely curved rail with
// interior control points: buildParallelTransportFrames used to place
// frame origins at curve VALUES sampled at the Greville parameters, then
// reused those as NEW control points (same knots/degree) for the swept
// surface's V-direction — a curve's value at a parameter is generally
// NOT equal to any control point for degree>1, so treating samples AS
// control points does not reproduce the original curve, and the mismatch
// grows with the rail's own curvature (invisible on a gentle curve,
// visibly twisted/ballooned on a sharp S-curve — exactly the reported
// bug). ORIGINALLY fixed by using the rail's own control points directly
// as frame origins.
//
// SUPERSEDED by the sparse/curved-rail fix (see
// kernel/sweep.mjs's own `railFrameOriginsExact` comment and
// test/sweep-sparse-rail.test.mjs): that "use the rail's raw control
// points" fix only ever worked BECAUSE it made this test's row-0 a literal
// copy of the rail's own control net (same array values, same knots,
// degree — a trivial identity, not a proof those control points are
// actually near the true curve). For THIS SAME 4-point wavy rail, the raw
// control points are themselves badly off-curve (reproduced directly
// below — the standard "sparse/high-curvature interpolation" problem, the
// same mechanism as the sparse-rail bug, just discovered earlier and
// misdiagnosed as fixed). The current, CORRECT guarantee is no longer
// "the edge is a byte-identical copy of the rail's own (possibly
// off-curve) control polygon" — it's "the edge tracks the rail's TRUE
// curve closely," proven via an arc-length-mapped comparison (paramAtArcLength,
// the same technique test/sweep-rational-rail.test.mjs already established)
// so the comparison isn't confounded by the surface's own arc-length
// V-parametrization vs. the rail's raw knot parametrization.
test('sweep1Rigid: a CURVED rail\'s own edge tracks the rail\'s TRUE path closely (not a byte-identical, possibly off-curve, control-net copy)', () => {
  // A genuinely wavy, multi-control-point, non-rational rail — the same
  // shape (open interpolation through several points) SketchCurve itself
  // builds, and the same shape that exposed the ORIGINAL bug live.
  const rail = globalCurveInterp([
    [0, 0, 0], [20, 10, 0], [40, -5, 0], [60, 15, 0],
  ], 3);
  assert.ok(rail.ctrlPts.length > 2, 'the rail has real interior control points, not just 2 endpoints — the case that was broken');
  // Ground truth, checked directly (not assumed): this rail's own interior
  // control points really are badly off the true curve — the same
  // mechanism the sparse-rail fix addresses, just on a different rail.
  const [mx, my, mz] = rail.ctrlPts[1];
  const trueNearMid = curvePoint(rail, 0.25);
  const rawDeviation = length(sub([mx, my, mz], trueNearMid));
  assert.ok(rawDeviation > 5, `sanity: this rail's own interior control point really is off-curve (deviation ${rawDeviation.toFixed(2)}) — otherwise this isn't testing the same problem class`);

  // A simple straight profile whose FIRST control point sits exactly at
  // the rail's own start (0,0,0) — "these were touching."
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 0, 15, 1]] };
  const frames = buildParallelTransportFrames(rail);
  const local = localizeSectionToFrame(profile, frames[0]);
  const srf = sweep1Rigid(rail, local);
  assert.equal(isFiniteNet(srf.ctrlNet), true);

  // The U=0 isocurve (profile's own first control point, zero local
  // offset from frames[0].origin) must track the rail's TRUE curve
  // closely at many V values, mapped from the surface's own arc-length V
  // parametrization back to the rail's raw parameter via paramAtArcLength
  // — not a same-array-identity tautology, a real independent ground-truth
  // check against curvePoint.
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const table = buildArcLengthTable(rail, uMin, uMax);
  const total = table.total;
  const TOLERANCE = total * 0.005; // 0.5% of the rail's own true length — tight, matching the sparse-rail fix's own established margin
  let maxErr = 0;
  for (let i = 0; i < 41; i++) {
    const v = (i + 0.5) / 41; // deliberately off the surface's own internal dense-sample grid
    const uTrue = paramAtArcLength(table, v * total);
    const onSurface = surfacePoint(srf, 0, v);
    const onRail = curvePoint(rail, uTrue);
    maxErr = Math.max(maxErr, length(sub(onSurface, onRail)));
  }
  assert.ok(maxErr < TOLERANCE, `max tracking error ${maxErr.toFixed(4)} exceeds the tight ${TOLERANCE.toFixed(4)}-unit tolerance (rail true length ${total.toFixed(2)})`);
});

test('projectPointToPlane: a point directly above a plane origin projects to (0, height)', () => {
  const plane = { origin: [0, 0, 0], uAxis: [1, 0, 0], vAxis: [0, 0, 1], scale: 2 };
  const p = projectPointToPlane([0, 5, 3], plane); // 5 units off-plane (ignored, no normal-axis term), 3 up
  assert.ok(Math.abs(p[0] - 0) < 1e-10);
  assert.ok(Math.abs(p[1] - 6) < 1e-10); // 3 * scale(2)
});

test('projectCurveToPlane on a circle in the XY plane, viewed from the XZ plane, flattens to a line', () => {
  const circle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 4); // lies in world XY (z=0)
  const plane = { origin: [0, 0, 0], uAxis: [1, 0, 0], vAxis: [0, 0, 1], scale: 1 }; // views XZ
  const pts = projectCurveToPlane(circle, plane, 32);
  for (const [, v] of pts) assert.ok(Math.abs(v) < 1e-9); // circle's z is always 0
});
