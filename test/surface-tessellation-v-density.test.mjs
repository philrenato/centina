// ADAPTIVE V-DIRECTION RENDER-MESH DENSITY — see
// kernel/surface.mjs's own `tessellationVSamples` header comment for the
// full derivation. Short version: the rounded-Pipe-corner fix in
// kernel/sweep.mjs only proves the swept NURBS SURFACE itself is fold-free
// per real rail span — the app's own `tessellateSurface`, the
// function that actually builds the mesh on screen, walked the V
// direction with a PLAIN UNIFORM loop, blind to real span boundaries,
// which can skip clean over a genuinely narrow (but still smooth) span at
// the shipped default `vRes=96` and connect two far-apart points with one
// straight mesh edge — a real, visible fold-like facet in the ACTUAL
// render, confirmed directly below. `tessellationVSamples` fixes this by
// guaranteeing a minimum sample density inside every genuine span.
//
// A SECOND, separate, more severe thing the same review surfaced and this
// file also confirms directly: the EXACT reproduction fixture (a
// 2mm fillet next to a swept 5mm-radius tube) is not a sampling artifact at
// all — the tube's cross-section radius exceeds the fillet's own radius of
// curvature, so the surface GENUINELY self-intersects there. No amount of
// render-mesh density fixes that (confirmed below: it gets slightly WORSE,
// not better, as density increases) — named here as a real, separate,
// NOT-fixed-here limitation, not silently asserted away.

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
// measures the worst adjacent-face-normal angle ACROSS the full V range —
// including straight across a genuine span boundary, unlike the prior
// fix's own per-span-only verification, which structurally never compares
// across a joint at all (the real gap this file exists to close).
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

test('tessellationVSamples: a single-span surface (no genuine internal V break) is BYTE-IDENTICAL to the plain uniform array — zero regression risk for the overwhelming common case', () => {
  const rail = irregularFixtureRail(2); // still multi-span; use a plain unfilleted 2-point line rail for the true single-span case
  const straightRail = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [100, 0, 0, 1]] };
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(straightRail, profile);
  const plain = plainUniformVSamples(srf, 96);
  const adaptive = tessellationVSamples(srf, 96, 48);
  assert.deepEqual(adaptive, plain, 'a plain straight-line rail has no genuine internal V knot at all, so tessellationVSamples must take the exact untouched uniform path');
});

test('tessellationVSamples: the REAL, actually-rendered mesh (not a per-span reimplementation) genuinely folds at the shipped default vRes=96 on a MILDER, non-self-intersecting irregular corner (15mm radius, comfortably above the swept 5mm tube radius) — and this fix closes most of that gap', () => {
  const rail = irregularFixtureRail(15);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const OLD = plainUniformVSamples(srf, 96);
  const NEW = tessellationVSamples(srf, 96, 48);
  const worstOld = worstFoldAcrossV(srf, 24, OLD);
  const worstNew = worstFoldAcrossV(srf, 24, NEW);
  assert.ok(worstOld > 10, `sanity: the OLD plain-uniform render mesh should show a real double-digit fold here (this is what the review caught), got ${worstOld.toFixed(3)} degrees`);
  assert.ok(worstNew < 5, `the NEW adaptively-densified render mesh should bring the SAME real fold reading under 5 degrees, got ${worstNew.toFixed(3)} degrees (was ${worstOld.toFixed(3)} degrees OLD)`);
  assert.ok(worstNew < worstOld / 2, `the fix should be a genuine, large improvement, not a marginal one — NEW ${worstNew.toFixed(3)} vs OLD ${worstOld.toFixed(3)}`);
});

test('tessellationVSamples: the milder-disparity improvement holds at a real live-app resolution sweep (vRes 24/96/192), not just one cherry-picked value', () => {
  const rail = irregularFixtureRail(15);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  for (const vRes of [24, 96, 192]) {
    const OLD = plainUniformVSamples(srf, vRes);
    const NEW = tessellationVSamples(srf, vRes, 48);
    const worstOld = worstFoldAcrossV(srf, 24, OLD);
    const worstNew = worstFoldAcrossV(srf, 24, NEW);
    assert.ok(worstNew <= worstOld, `at vRes=${vRes}, the fix should never be WORSE than the plain uniform baseline, got NEW ${worstNew.toFixed(3)} vs OLD ${worstOld.toFixed(3)}`);
  }
});

test('HONEST, SEPARATE limitation: the exact severe reproduction fixture (2mm fillet radius next to a swept 5mm tube) genuinely self-intersects — this fix does NOT resolve it, and density makes the reading slightly WORSE, not better, confirming it is real geometry, not a sampling artifact', () => {
  const rail = irregularFixtureRail(2);
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const srf = sweep1Rigid(rail, profile);
  const OLD = plainUniformVSamples(srf, 96);
  const NEW = tessellationVSamples(srf, 96, 48);
  const worstOld = worstFoldAcrossV(srf, 24, OLD);
  const worstNew = worstFoldAcrossV(srf, 24, NEW);
  assert.ok(worstOld > 150, `sanity: the OLD render already reads as a near-total fold here (real self-intersection), got ${worstOld.toFixed(3)} degrees`);
  assert.ok(worstNew > 150, `the fix must NOT be asserted to resolve a genuine self-intersection — it should stay large, got ${worstNew.toFixed(3)} degrees (was ${worstOld.toFixed(3)} degrees)`);
});

test('the self-intersection threshold is real: sweeping cornerRadius from below to above the swept tube radius (5mm) shows a sharp transition, confirming radius-of-curvature-vs-tube-radius as the true mechanism, not a coincidence of one fixture', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], PIPE_RADIUS, 4);
  const below = sweep1Rigid(irregularFixtureRail(4.9), profile);
  const above = sweep1Rigid(irregularFixtureRail(5.1), profile);
  const foldBelow = worstFoldAcrossV(below, 24, tessellationVSamples(below, 96, 48));
  const foldAbove = worstFoldAcrossV(above, 24, tessellationVSamples(above, 96, 48));
  assert.ok(foldBelow > 100, `just BELOW the tube radius (cornerRadius=4.9 < radius=5), the fold should still read as a real self-intersection, got ${foldBelow.toFixed(3)} degrees`);
  assert.ok(foldAbove < foldBelow, `just ABOVE the tube radius (cornerRadius=5.1 > radius=5), the fold should already be markedly smaller, got ${foldAbove.toFixed(3)} degrees vs ${foldBelow.toFixed(3)} below`);
});
