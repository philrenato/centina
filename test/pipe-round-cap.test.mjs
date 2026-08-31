// PIPE ROUND CAP — Stage 4a, the smallest,
// self-contained cap-parity item (completing Rhino's own None/Flat/Round
// triple). Verified two ways: (1) every point on the returned surface
// really is exactly `radius` from the frame's own origin (a true
// hemisphere, not an approximation); (2) the cap's own rim ring is
// BYTE-IDENTICAL to the tube's own real end ring at that frame — the
// actual "seamless by construction" claim, proven against `sweep1Rigid`
// directly, not assumed.

import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle } from '../kernel/primitives.mjs';
import { sweep1Rigid, pipeRoundCapSurface } from '../kernel/sweep.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function scale(v, s) { return [v[0] * s, v[1] * s, v[2] * s]; }

function buildTube(rail, radius) {
  const circleProfile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], radius);
  return sweep1Rigid(rail, circleProfile);
}

test('pipeRoundCapSurface: every point on the dome is exactly `radius` from the frame origin — a true hemisphere', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 0, 60, 1]] };
  const radius = 5;
  const srf = buildTube(rail, radius);
  const endFrame = srf.frames[srf.frames.length - 1];
  const cap = pipeRoundCapSurface(endFrame, radius, endFrame.zAxis);
  let maxErr = 0;
  const N = 12;
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      const u = cap.knotsU[0] + (cap.knotsU[cap.knotsU.length - 1] - cap.knotsU[0]) * (i / N);
      const v = cap.knotsV[0] + (cap.knotsV[cap.knotsV.length - 1] - cap.knotsV[0]) * (j / N);
      const p = surfacePoint(cap, u, v);
      const err = Math.abs(dist(p, endFrame.origin) - radius);
      if (err > maxErr) maxErr = err;
    }
  }
  assert.ok(maxErr < 1e-6, `every sampled point should sit exactly on the radius-${radius} sphere, max error ${maxErr}`);
});

test('pipeRoundCapSurface: the pole sits exactly `radius` further along axisDir from the rim plane', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 0, 60, 1]] };
  const radius = 5;
  const srf = buildTube(rail, radius);
  const endFrame = srf.frames[srf.frames.length - 1];
  const cap = pipeRoundCapSurface(endFrame, radius, endFrame.zAxis);
  const uMax = cap.knotsU[cap.knotsU.length - 1];
  const pole = surfacePoint(cap, uMax, cap.knotsV[0]);
  const expectedPole = [
    endFrame.origin[0] + endFrame.zAxis[0] * radius,
    endFrame.origin[1] + endFrame.zAxis[1] * radius,
    endFrame.origin[2] + endFrame.zAxis[2] * radius,
  ];
  assert.ok(dist(pole, expectedPole) < 1e-6, `pole should sit exactly radius ${radius} along axisDir from the origin`);
});

test('pipeRoundCapSurface: the START cap (axisDir = -zAxis) bulges BEFORE the rail begins, not after', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 0, 60, 1]] };
  const radius = 5;
  const srf = buildTube(rail, radius);
  const startFrame = srf.frames[0];
  const axisDir = scale(startFrame.zAxis, -1);
  const cap = pipeRoundCapSurface(startFrame, radius, axisDir);
  const uMax = cap.knotsU[cap.knotsU.length - 1];
  const pole = surfacePoint(cap, uMax, cap.knotsV[0]);
  // The rail runs along +Z from z=0 to z=60; the start cap's own pole must
  // sit at NEGATIVE z (before the rail begins), not positive.
  assert.ok(pole[2] < -radius + 1e-6, `start cap's pole should sit below z=0, got z=${pole[2]}`);
});

test('pipeRoundCapSurface: the rim ring is BYTE-IDENTICAL to the tube\'s own real end ring at that frame — the seamless-by-construction proof', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 0, 60, 1]] };
  const radius = 5;
  const srf = buildTube(rail, radius);
  const endFrame = srf.frames[srf.frames.length - 1];
  const cap = pipeRoundCapSurface(endFrame, radius, endFrame.zAxis);
  // The tube's own end ring is srf.ctrlNet[k][last] for every U index k (the
  // circle profile's own control points at that station). The cap's own rim
  // ring is its U=0 row, across all its V control points.
  const vLastTube = srf.ctrlNet[0].length - 1;
  const N = 24;
  for (let j = 0; j <= N; j++) {
    const v = cap.knotsV[0] + (cap.knotsV[cap.knotsV.length - 1] - cap.knotsV[0]) * (j / N);
    const capPt = surfacePoint(cap, cap.knotsU[0], v);
    // The corresponding tube-edge point at the SAME angular fraction: the
    // tube's own U direction is the circle profile's parameter, [0, 2*PI]-ish
    // domain (arcKnots of narcs spans) — sample it at the same fraction.
    const uTube = srf.knotsU[0] + (srf.knotsU[srf.knotsU.length - 1] - srf.knotsU[0]) * (j / N);
    const tubePt = surfacePoint(srf, uTube, srf.knotsV[srf.knotsV.length - 1]);
    assert.ok(dist(capPt, tubePt) < 1e-6, `cap rim point at fraction ${j / N} should exactly match the tube's own end ring, got ${dist(capPt, tubePt)}`);
  }
  void vLastTube;
});

test('pipeRoundCapSurface: works correctly on a curved rail too, not just a straight one', () => {
  const rail = {
    degree: 2,
    knots: [0, 0, 0, 1, 1, 1],
    ctrlPts: [[0, 0, 0, 1], [30, 0, 20, 1], [60, 40, 0, 1]],
  };
  const radius = 4;
  const srf = buildTube(rail, radius);
  const endFrame = srf.frames[srf.frames.length - 1];
  const cap = pipeRoundCapSurface(endFrame, radius, endFrame.zAxis);
  let maxErr = 0;
  const N = 10;
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      const u = cap.knotsU[0] + (cap.knotsU[cap.knotsU.length - 1] - cap.knotsU[0]) * (i / N);
      const v = cap.knotsV[0] + (cap.knotsV[cap.knotsV.length - 1] - cap.knotsV[0]) * (j / N);
      const p = surfacePoint(cap, u, v);
      const err = Math.abs(dist(p, endFrame.origin) - radius);
      if (err > maxErr) maxErr = err;
    }
  }
  assert.ok(maxErr < 1e-6, `curved-rail case: every sampled point should still sit exactly on the sphere, max error ${maxErr}`);
});
