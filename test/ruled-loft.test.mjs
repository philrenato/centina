import test from 'node:test';
import assert from 'node:assert/strict';
import { ruledLoftPanels } from '../kernel/loft.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';

// CLOSED case (a triangle-like 3-vertex profile, small enough to hand-check
// every panel's own corners) — 3 panels, one per edge, each an EXACT
// bilinear patch reproducing its own 2 input points on each side.
test('ruledLoftPanels (closed): N panels for N vertices, each panel\'s 4 corners exactly match the real input points', () => {
  const a = [[0, 0, 0], [10, 0, 0], [5, 10, 0]];
  const b = [[0, 0, 10], [10, 0, 10], [5, 10, 10]];
  const panels = ruledLoftPanels(a, b, true);
  assert.equal(panels.length, 3, 'closed 3-vertex profile: 3 wraparound edges/panels');
  for (let i = 0; i < 3; i++) {
    const j = (i + 1) % 3;
    const srf = panels[i];
    assert.equal(isFiniteNet(srf.ctrlNet), true, `panel ${i}: finite control net`);
    assert.equal(srf.degU, 1); assert.equal(srf.degV, 1);
    const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
    const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
    const corners = {
      '(uMin,vMin)=A[i]': [surfacePoint(srf, uMin, vMin), a[i]],
      '(uMin,vMax)=A[i+1]': [surfacePoint(srf, uMin, vMax), a[j]],
      '(uMax,vMin)=B[i]': [surfacePoint(srf, uMax, vMin), b[i]],
      '(uMax,vMax)=B[i+1]': [surfacePoint(srf, uMax, vMax), b[j]],
    };
    for (const [name, [p, expected]] of Object.entries(corners)) {
      assert.ok(Math.hypot(p[0] - expected[0], p[1] - expected[1], p[2] - expected[2]) < 1e-9, `panel ${i} ${name}: got ${p}, expected ${expected}`);
    }
  }
});

// OPEN case — N-1 panels (no wraparound edge), same exactness check.
test('ruledLoftPanels (open): N-1 panels for N vertices, no wraparound edge, corners still exact', () => {
  const a = [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]];
  const b = [[0, 0, 10], [10, 0, 10], [20, 0, 10], [30, 0, 10]];
  const panels = ruledLoftPanels(a, b, false);
  assert.equal(panels.length, 3, 'open 4-vertex profile: 3 edges (no wraparound), not 4');
  for (let i = 0; i < 3; i++) {
    const srf = panels[i];
    const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
    const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
    assert.deepEqual(surfacePoint(srf, uMin, vMin), a[i]);
    assert.deepEqual(surfacePoint(srf, uMin, vMax), a[i + 1]);
    assert.deepEqual(surfacePoint(srf, uMax, vMin), b[i]);
    assert.deepEqual(surfacePoint(srf, uMax, vMax), b[i + 1]);
  }
});

// Every panel is genuinely FLAT (a real ruled/bilinear patch, not an
// approximation) — the midpoint of a panel must sit exactly at the average
// of its 4 corners, the defining property of a bilinear surface.
test('ruledLoftPanels: each panel is exactly flat/bilinear (midpoint = average of the 4 corners)', () => {
  const a = [[0, 0, 0], [10, 2, 0], [20, -3, 0]];
  const b = [[1, 0, 10], [9, 5, 10], [22, -1, 10]];
  const panels = ruledLoftPanels(a, b, false);
  for (const srf of panels) {
    const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
    const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
    const c00 = surfacePoint(srf, uMin, vMin), c01 = surfacePoint(srf, uMin, vMax);
    const c10 = surfacePoint(srf, uMax, vMin), c11 = surfacePoint(srf, uMax, vMax);
    const avg = [0, 1, 2].map((k) => (c00[k] + c01[k] + c10[k] + c11[k]) / 4);
    const mid = surfacePoint(srf, (uMin + uMax) / 2, (vMin + vMax) / 2);
    assert.ok(Math.hypot(mid[0] - avg[0], mid[1] - avg[1], mid[2] - avg[2]) < 1e-9, `midpoint ${mid} != corner average ${avg}`);
  }
});

// Defensive refusals — never actually reachable from the app layer's own
// eligibility gate (loftRuledEligible), but the kernel function itself
// should refuse honestly on its own, not just trust its caller.
test('ruledLoftPanels refuses a mismatched vertex count with a clear, named error', () => {
  assert.throws(() => ruledLoftPanels([[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 1], [2, 0, 1]], false), /same vertex count/);
});
test('ruledLoftPanels refuses fewer than 2 vertices per profile', () => {
  assert.throws(() => ruledLoftPanels([[0, 0, 0]], [[0, 0, 1]], false), /at least 2 vertices/);
});
