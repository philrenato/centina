// The P0 acceptance gate: body revolve, spout
// sweep, handle sweep, lid revolve — all finite, all frames orthonormal.
// This isn't a visual/aesthetic teapot (that's for the viewport in P1) — it
// proves the four constructions the whole v1 toolset leans on (revolve,
// sweep, and the pole-point degenerate case both need) compose correctly
// end to end, including two on-axis pole points (body's neck and foot) that
// would divide-by-zero-radius if the degenerate branch were wrong.

import test from 'node:test';
import assert from 'node:assert/strict';
import { revolve, makeCircle, makeArc } from '../kernel/primitives.mjs';
import { sweep1Rigid } from '../kernel/sweep.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';
import { dot, length } from '../kernel/vec3.mjs';

function assertOrthonormal(f, tol = 1e-8) {
  assert.ok(Math.abs(length(f.xAxis) - 1) < tol);
  assert.ok(Math.abs(length(f.yAxis) - 1) < tol);
  assert.ok(Math.abs(length(f.zAxis) - 1) < tol);
  assert.ok(Math.abs(dot(f.xAxis, f.yAxis)) < tol);
  assert.ok(Math.abs(dot(f.xAxis, f.zAxis)) < tol);
  assert.ok(Math.abs(dot(f.yAxis, f.zAxis)) < tol);
}

function assertSurfaceFinite(srf, uSamples = 9, vSamples = 9) {
  assert.equal(isFiniteNet(srf.ctrlNet), true, 'control net has a non-finite or non-positive-weight point');
  const uMax = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const uMin = srf.knotsU[srf.degU];
  const vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const vMin = srf.knotsV[srf.degV];
  for (let i = 0; i <= uSamples; i++) {
    const u = uMin + (i / uSamples) * (uMax - uMin);
    for (let j = 0; j <= vSamples; j++) {
      const v = vMin + (j / vSamples) * (vMax - vMin);
      const p = surfacePoint(srf, u, v);
      assert.ok(p.every(Number.isFinite), `non-finite point at u=${u} v=${v}: ${p}`);
    }
  }
}

test('teapot BODY: revolve of a profile with two on-axis pole points (foot + neck)', () => {
  // (r, z) profile: pole at the foot, belly out, pole again at the neck.
  const profile = {
    degree: 2,
    knots: [0, 0, 0, 1, 2, 3, 3, 3],
    ctrlPts: [
      [0, 0, 0, 1],   // pole: foot
      [6, 0, 1, 1],
      [7, 0, 4, 1],
      [5, 0, 7, 1],
      [0, 0, 8, 1],   // pole: neck
    ],
  };
  const body = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  assertSurfaceFinite(body);
  // both poles collapse to the axis regardless of v
  for (let v = 0; v <= 4; v += 0.37) {
    const foot = surfacePoint(body, 0, v);
    const neck = surfacePoint(body, 3, v);
    assert.ok(Math.abs(foot[0]) < 1e-9 && Math.abs(foot[1]) < 1e-9 && Math.abs(foot[2]) < 1e-9);
    assert.ok(Math.abs(neck[0]) < 1e-9 && Math.abs(neck[1]) < 1e-9 && Math.abs(neck[2] - 8) < 1e-9);
  }
});

test('teapot LID: revolve of a dome profile (rim to apex pole)', () => {
  const profile = {
    degree: 2,
    knots: [0, 0, 0, 1, 2, 2, 2],
    ctrlPts: [
      [4, 0, 0, 1],   // rim
      [4, 0, 0.5, 1],
      [2, 0, 1.5, 1],
      [0, 0, 2, 1],   // pole: apex
    ],
  };
  const lid = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  assertSurfaceFinite(lid);
  const apex = surfacePoint(lid, 2, 1.5);
  assert.ok(Math.abs(apex[0]) < 1e-9 && Math.abs(apex[1]) < 1e-9 && Math.abs(apex[2] - 2) < 1e-9);
});

test('teapot SPOUT: rigid sweep of a small circle along a rising, curving rail', () => {
  const rail = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[6, 0, 3, 1], [10, 0, 5, 1], [13, 0, 6, 1]] };
  const section = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.8);
  const spout = sweep1Rigid(rail, section);
  assertSurfaceFinite(spout);
  assert.ok(spout.frames.length >= 2);
  for (const f of spout.frames) assertOrthonormal(f);
});

test('teapot HANDLE: rigid sweep of a small circle along a C-shaped arc rail', () => {
  const rail = makeArc([-7, 0, 4], [0, 0, 1], [1, 0, 0], 3, -Math.PI / 2, Math.PI); // a half-loop
  const section = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.6);
  const handle = sweep1Rigid(rail, section);
  assertSurfaceFinite(handle);
  assert.ok(handle.frames.length >= 2);
  for (const f of handle.frames) assertOrthonormal(f);
});

test('teapot: all four parts assembled have zero NaN/Infinity anywhere (the acceptance line)', () => {
  const bodyProfile = { degree: 2, knots: [0, 0, 0, 1, 2, 3, 3, 3], ctrlPts: [[0,0,0,1],[6,0,1,1],[7,0,4,1],[5,0,7,1],[0,0,8,1]] };
  const lidProfile = { degree: 2, knots: [0, 0, 0, 1, 2, 2, 2], ctrlPts: [[4,0,0,1],[4,0,0.5,1],[2,0,1.5,1],[0,0,2,1]] };
  const spoutRail = { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[6,0,3,1],[10,0,5,1],[13,0,6,1]] };
  const handleRail = makeArc([-7, 0, 4], [0, 0, 1], [1, 0, 0], 3, -Math.PI / 2, Math.PI);

  const parts = [
    revolve(bodyProfile, [0,0,0], [0,0,1], 0, 2*Math.PI),
    revolve(lidProfile, [0,0,0], [0,0,1], 0, 2*Math.PI),
    sweep1Rigid(spoutRail, makeCircle([0,0,0],[1,0,0],[0,1,0],0.8)),
    sweep1Rigid(handleRail, makeCircle([0,0,0],[1,0,0],[0,1,0],0.6)),
  ];
  for (const part of parts) {
    assert.equal(isFiniteNet(part.ctrlNet), true);
    if (part.frames) for (const f of part.frames) assertOrthonormal(f);
  }
});
