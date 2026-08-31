import test from 'node:test';
import assert from 'node:assert/strict';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { curvePoint, closestPointOnCurve } from '../kernel/curve.mjs';
import { extractIsocurveU, extractIsocurveV } from '../kernel/isocurve.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { boundSurfaceFromLoop } from '../kernel/loft.mjs';

// A genuinely non-trivial closed 4-edge loop — every edge actually curves
// (not a straight-sided rectangle, which could hide a real bug the same
// way a too-simple fixture has before elsewhere in this project). Built
// so adjacent edges share their corner point EXACTLY (globalCurveInterp
// passes through its given points exactly), matching the real, common
// "draw 4 boundary curves meeting at their corners" student gesture.
function c0() { return globalCurveInterp([[0, 0, 0], [50, -8, 0], [100, 0, 0]], 2); } // bottom, bows down
function c1() { return globalCurveInterp([[100, 0, 0], [112, 40, 0], [100, 80, 0]], 2); } // right, bows out
function c2() { return globalCurveInterp([[100, 80, 0], [50, 92, 0], [0, 80, 0]], 2); } // top, bows up
function c3() { return globalCurveInterp([[0, 80, 0], [-10, 40, 0], [0, 0, 0]], 2); } // left, bows in

test('boundSurfaceFromLoop: the resulting surface reproduces all 4 boundary curves’ own real SHAPE exactly, not just at corners', () => {
  const srf = boundSurfaceFromLoop(c0(), c1(), c2(), c3());
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];

  // Extract the surface's own 4 real boundary edges via the SAME
  // isoparametric-curve machinery ExtractBorder already relies on — a
  // real, independent curve object, not a resample.
  const edgeV0 = extractIsocurveV(srf, vMin); // should trace c0 (rail 0)
  const edgeV1 = extractIsocurveV(srf, vMax); // should trace c2 (rail 1, un-reversed)
  const edgeU0 = extractIsocurveU(srf, uMin); // should trace c3 (un-reversed)
  const edgeU1 = extractIsocurveU(srf, uMax); // should trace c1

  // SHAPE match, not parametrization match — a real, found-via-testing
  // distinction (see boundSurfaceFromLoop's own header comment): the
  // returned edge is a FRESH re-interpolation, so its own parameter no
  // longer runs at the original curve's exact speed, even though it
  // traces the identical 3D path. closestPointOnCurve is the honest way
  // to check "is this point genuinely ON the original curve," regardless
  // of which parameter value gets it there. A first draft of this test
  // compared "same raw parameter fraction" instead and found a spurious
  // ~0.5mm gap that looked like a real math bug until re-diagnosed this
  // way — logged in boundSurfaceFromLoop's own comment too, not just here.
  const N = 12;
  function assertEdgeShapeMatches(edgeCrv, originalCrv, label) {
    const e0 = edgeCrv.knots[0], e1 = edgeCrv.knots[edgeCrv.knots.length - 1];
    let maxErr = 0;
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      const pEdge = curvePoint(edgeCrv, e0 + t * (e1 - e0));
      const hit = closestPointOnCurve(originalCrv, pEdge);
      const pOrig = curvePoint(originalCrv, hit.u);
      const err = Math.hypot(pEdge[0] - pOrig[0], pEdge[1] - pOrig[1], pEdge[2] - pOrig[2]);
      maxErr = Math.max(maxErr, err);
    }
    assert.ok(maxErr < 5e-3, `${label}: every sampled edge point should land essentially ON the true original curve, max deviation ${maxErr}`);
  }
  assertEdgeShapeMatches(edgeV0, c0(), 'bottom edge (v=vMin) traces c0');
  assertEdgeShapeMatches(edgeV1, c2(), 'top edge (v=vMax) traces c2');
  assertEdgeShapeMatches(edgeU0, c3(), 'left edge (u=uMin) traces c3');
  assertEdgeShapeMatches(edgeU1, c1(), 'right edge (u=uMax) traces c1');

  // The 4 CORNERS are a stronger, unambiguous check (no reparametrization
  // question at all — a domain boundary is a domain boundary): the
  // surface's own 4 corners must exactly match the loop's own 4 real
  // corner points.
  const corners = [
    [surfacePoint(srf, uMin, vMin), curvePoint(c0(), c0().knots[0])],
    [surfacePoint(srf, uMax, vMin), curvePoint(c0(), c0().knots[c0().knots.length - 1])],
    [surfacePoint(srf, uMax, vMax), curvePoint(c2(), c2().knots[0])],
    [surfacePoint(srf, uMin, vMax), curvePoint(c2(), c2().knots[c2().knots.length - 1])],
  ];
  for (const [p, q] of corners) {
    const err = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    assert.ok(err < 1e-6, `a surface corner should exactly match its real input-curve corner, got error ${err}`);
  }
});

test('boundSurfaceFromLoop: the interior genuinely bulges with the boundary curves, not a flat bilinear blend', () => {
  const srf = boundSurfaceFromLoop(c0(), c1(), c2(), c3());
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const center = surfacePoint(srf, (uMin + uMax) / 2, (vMin + vMax) / 2);
  // The flat/bilinear-blend center would sit near Y=40 (the straight
  // average of the 4 corners, all at Y=0 or Y=80) — the real surface,
  // genuinely following its bowed boundary curves, should read noticeably
  // different (the top/bottom edges bow away from the corners).
  assert.ok(Math.abs(center[1] - 40) > 1, `center Y should deviate meaningfully from a flat 40 blend (genuine curvature), got ${center[1]}`);
});

test('boundSurfaceFromLoop: refuses honestly when the 4 curves do not actually form a closed loop', () => {
  const badC3 = globalCurveInterp([[0, 80, 0], [-10, 40, 0], [5, 5, 0]], 2); // doesn't reach back to c0's start
  assert.throws(() => boundSurfaceFromLoop(c0(), c1(), c2(), badC3), /don't share an endpoint/);
});

test('boundSurfaceFromLoop: a real straight-sided fixture stays a genuine ruled/flat check — sanity cross-check', () => {
  const s0 = globalCurveInterp([[0, 0, 0], [50, 0, 0], [100, 0, 0]], 1);
  const s1 = globalCurveInterp([[100, 0, 0], [100, 40, 0], [100, 80, 0]], 1);
  const s2 = globalCurveInterp([[100, 80, 0], [50, 80, 0], [0, 80, 0]], 1);
  const s3 = globalCurveInterp([[0, 80, 0], [0, 40, 0], [0, 0, 0]], 1);
  const srf = boundSurfaceFromLoop(s0, s1, s2, s3);
  // A flat rectangle bound surface should be, well, flat: Z stays exactly 0
  // everywhere, and the interior point should be genuinely interior.
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const p = surfacePoint(srf, (uMin + uMax) / 2, (vMin + vMax) / 2);
  assert.ok(Math.abs(p[2]) < 1e-6, `flat rectangle's own interior should stay exactly Z=0, got ${p[2]}`);
  assert.ok(p[0] > 10 && p[0] < 90 && p[1] > 10 && p[1] < 70, `interior point should be genuinely interior, got ${JSON.stringify(p)}`);
});
