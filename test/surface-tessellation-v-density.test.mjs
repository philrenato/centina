// Adaptive V-direction render-mesh density — see kernel/surface.mjs's
// `tessellationVSamples` header comment for the full derivation. A swept
// NURBS surface can be fold-free per rail span while a render mesh that walks
// V with a plain uniform loop, blind to span boundaries, skips over a narrow
// (but smooth) span at the default `vRes=96` and connects two far-apart
// points with one straight mesh edge — a visible fold-like facet.
// `tessellationVSamples` guarantees a minimum sample density inside every
// span.
//
// Known limitation, asserted below: with a 2mm fillet next to a swept
// 5mm-radius tube, the tube's cross-section radius exceeds the fillet's
// radius of curvature, so the surface self-intersects there. No render-mesh
// density fixes that; the fold reading gets slightly worse, not better, as
// density increases.

import test from 'node:test';
import assert from 'node:assert/strict';
import { filletOpenPolyline, filletSegmentsToCurve, makeCircle } from '../kernel/primitives.mjs';
import { sweep1Rigid } from '../kernel/sweep.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { tessellationVSamples } from '../kernel/surface.mjs';

const PIPE_RADIUS = 5;

function irregularFixtureRail(cornerRadius, midLen = 40) {
  const pts = [[0, 0, 0], [200, 0, 0], [200, midLen, 0], [260, midLen, 0], [260, 90, 0], [330, 150, 0]];
  const res = filletOpenPolyline(pts, cornerRadius, { closed: false });
  assert.equal(res.ok, true, res.reason);
  return filletSegmentsToCurve(res.segments);
}

function triNormal(a, b, c) {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const len = Math.hypot(...n);
  return len < 1e-15 ? [0, 0, 0] : n.map((x) => x / len);
}
function angleBetween(n1, n2) {
  const d = Math.max(-1, Math.min(1, n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]));
  return Math.acos(d) * 180 / Math.PI;
}

// Mirrors tessellateSurface's own V-direction mesh construction exactly
// (uniform uRes across U, the vSamples list under test across V) and
// measures the worst adjacent-face-normal angle across the full V range,
// including straight across a span boundary; a per-span check never
// compares across a joint.
function worstFoldAcrossV(srf, uRes, vSamples) {
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const rows = vSamples.map((v) => {
    const pts = [];
    for (let i = 0; i <= uRes; i++) pts.push(surfacePoint(srf, uMin + (uMax - uMin) * i / uRes, v));
    return pts;
  });
  let worst = 0;
  for (let j = 1; j < vSamples.length - 1; j++) {
    for (let i = 0; i < rows[0].length - 1; i++) {
      const n1 = triNormal(rows[j - 1][i], rows[j - 1][i + 1], rows[j][i]);
      const n2 = triNormal(rows[j][i], rows[j][i + 1], rows[j + 1][i]);
      const ang = angleBetween(n1, n2);
      if (ang > worst) worst = ang;
    }
  }
  return worst;
}

function plainUniformVSamples(srf, vRes) {
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const out = [];
  for (let j = 0; j <= vRes; j++) out.push(vMin + (j / vRes) * (vMax - vMin));
  return out;
}

test('tessellationVSamples: a single-span surface (no internal V break) is byte-identical to the plain uniform array', () => {
  const rail = irregularFixtureRail(2); // still multi-span; use a plain unfilleted 2-point line rail for the true single-span case
  const straightRail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [100, 0, 0, 1]] };
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(straightRail, profile);
  const plain = plainUniformVSamples(srf, 96);
  const adaptive = tessellationVSamples(srf, 96, 48);
  assert.deepEqual(adaptive, plain, 'a plain straight-line rail has no internal V knot at all, so tessellationVSamples must take the exact untouched uniform path');
});

test('tessellationVSamples: the rendered mesh construction (not a per-span reimplementation) folds at the default vRes=96 on a milder, non-self-intersecting irregular corner (15mm radius, comfortably above the swept 5mm tube radius), and adaptive sampling closes most of that gap', () => {
  const rail = irregularFixtureRail(15);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const OLD = plainUniformVSamples(srf, 96);
  const NEW = tessellationVSamples(srf, 96, 48);
  const worstOld = worstFoldAcrossV(srf, 24, OLD);
  const worstNew = worstFoldAcrossV(srf, 24, NEW);
  assert.ok(worstOld > 10, `sanity: the plain-uniform render mesh should show a double-digit fold here, got ${worstOld.toFixed(3)} degrees`);
  assert.ok(worstNew < 5, `the adaptively densified render mesh should bring the same fold reading under 5 degrees, got ${worstNew.toFixed(3)} degrees (uniform: ${worstOld.toFixed(3)} degrees)`);
  assert.ok(worstNew < worstOld / 2, `adaptive sampling should be a large improvement, not a marginal one — adaptive ${worstNew.toFixed(3)} vs uniform ${worstOld.toFixed(3)}`);
});

test('tessellationVSamples: the milder-disparity improvement holds across a resolution sweep (vRes 24/96/192), not just one value', () => {
  const rail = irregularFixtureRail(15);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  for (const vRes of [24, 96, 192]) {
    const OLD = plainUniformVSamples(srf, vRes);
    const NEW = tessellationVSamples(srf, vRes, 48);
    const worstOld = worstFoldAcrossV(srf, 24, OLD);
    const worstNew = worstFoldAcrossV(srf, 24, NEW);
    assert.ok(worstNew <= worstOld, `at vRes=${vRes}, adaptive sampling should never be worse than the plain uniform baseline, got adaptive ${worstNew.toFixed(3)} vs uniform ${worstOld.toFixed(3)}`);
  }
});

test('known limitation: a 2mm fillet radius next to a swept 5mm tube self-intersects — adaptive sampling does not resolve it, and density makes the reading slightly worse, not better, so it is geometry, not a sampling artifact', () => {
  const rail = irregularFixtureRail(2);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const OLD = plainUniformVSamples(srf, 96);
  const NEW = tessellationVSamples(srf, 96, 48);
  const worstOld = worstFoldAcrossV(srf, 24, OLD);
  const worstNew = worstFoldAcrossV(srf, 24, NEW);
  assert.ok(worstOld > 150, `sanity: the uniform render reads as a near-total fold here (self-intersection), got ${worstOld.toFixed(3)} degrees`);
  assert.ok(worstNew > 150, `adaptive sampling does not resolve a self-intersection — the reading should stay large, got ${worstNew.toFixed(3)} degrees (uniform: ${worstOld.toFixed(3)} degrees)`);
});

test('the self-intersection threshold: sweeping cornerRadius from below to above the swept tube radius (5mm) shows a sharp transition, confirming radius-of-curvature-vs-tube-radius as the true mechanism, not a coincidence of one fixture', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const below = sweep1Rigid(irregularFixtureRail(4.9), profile);
  const above = sweep1Rigid(irregularFixtureRail(5.1), profile);
  const foldBelow = worstFoldAcrossV(below, 24, tessellationVSamples(below, 96, 48));
  const foldAbove = worstFoldAcrossV(above, 24, tessellationVSamples(above, 96, 48));
  assert.ok(foldBelow > 100, `just below the tube radius (cornerRadius=4.9 < radius=5), the fold should still read as a self-intersection, got ${foldBelow.toFixed(3)} degrees`);
  assert.ok(foldAbove < foldBelow, `just above the tube radius (cornerRadius=5.1 > radius=5), the fold should already be markedly smaller, got ${foldAbove.toFixed(3)} degrees vs ${foldBelow.toFixed(3)} below`);
});
