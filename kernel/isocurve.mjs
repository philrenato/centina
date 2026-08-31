// ExtractIsocurve — Piegl & Tiller's own defining property of a NURBS
// surface's isoparametric curves: holding ONE parameter fixed collapses the
// surface's tensor-product sum into an ordinary NURBS CURVE running along
// the OTHER parameter, of that direction's own degree/knots exactly — not
// a resampled polyline approximation of one.
//
//   S(u0, v) = sum_j  [ sum_i N_i,p(u0) * Pw[i][j] ]  * N_j,q(v)
//                       \_________________________/
//                       the isocurve's own j-th homogeneous control point
//
// This is the SAME basis-function machinery surfacePoint (surface.mjs)
// already runs for a single point, and the SAME machinery the Wireframe
// isocurve overlay already uses to SAMPLE points along an isocurve for
// display — the one genuinely new piece here is contracting the FULL
// control net along one direction (every row/column, not just the p+1 or
// q+1 nonzero terms near one sample) to get a real, independent curve
// OBJECT (degree + knots + control points) rather than a stream of sampled
// points with no curve identity of their own.
//
// A NurbsCrv here, matching curve.mjs/interpolate.mjs's own convention, is
// { degree, knots, ctrlPts } with ctrlPts = [x, y, z, w] (real point + its
// weight, not premultiplied) — this app's SketchCurve representation, so
// an extracted isocurve slots directly into the document as an ordinary,
// fully-editable/joinable/extrudable curve object.

import { findSpan, basisFuns } from './basis.mjs';
import { surfaceClosure } from './surface.mjs';
import { grevilleAbscissae } from './curve.mjs';

// Fix U at `u`, return the real NurbsCrv running along the surface's own V
// direction (degree degV, knots knotsV).
export function extractIsocurveU(srf, u) {
  const { degU: p, knotsU: U, degV: q, knotsV: V, ctrlNet } = srf;
  const n = ctrlNet.length - 1;
  const m = ctrlNet[0].length - 1;
  const uspan = findSpan(n, p, u, U);
  const Nu = basisFuns(uspan, u, p, U);
  const ctrlPts = [];
  for (let j = 0; j <= m; j++) {
    const Qw = [0, 0, 0, 0];
    for (let k = 0; k <= p; k++) {
      const [x, y, z, w] = ctrlNet[uspan - p + k][j];
      Qw[0] += Nu[k] * x * w;
      Qw[1] += Nu[k] * y * w;
      Qw[2] += Nu[k] * z * w;
      Qw[3] += Nu[k] * w;
    }
    ctrlPts.push([Qw[0] / Qw[3], Qw[1] / Qw[3], Qw[2] / Qw[3], Qw[3]]);
  }
  return { degree: q, knots: V.slice(), ctrlPts };
}

// Fix V at `v`, return the real NurbsCrv running along the surface's own U
// direction (degree degU, knots knotsU) — the symmetric twin of the above.
export function extractIsocurveV(srf, v) {
  const { degU: p, knotsU: U, degV: q, knotsV: V, ctrlNet } = srf;
  const n = ctrlNet.length - 1;
  const m = ctrlNet[0].length - 1;
  const vspan = findSpan(m, q, v, V);
  const Nv = basisFuns(vspan, v, q, V);
  const ctrlPts = [];
  for (let i = 0; i <= n; i++) {
    const Qw = [0, 0, 0, 0];
    for (let l = 0; l <= q; l++) {
      const [x, y, z, w] = ctrlNet[i][vspan - q + l];
      Qw[0] += Nv[l] * x * w;
      Qw[1] += Nv[l] * y * w;
      Qw[2] += Nv[l] * z * w;
      Qw[3] += Nv[l] * w;
    }
    ctrlPts.push([Qw[0] / Qw[3], Qw[1] / Qw[3], Qw[2] / Qw[3], Qw[3]]);
  }
  return { degree: p, knots: U.slice(), ctrlPts };
}

// ExtractBorder — a surface's own 4 parametric-domain edges (u=uMin,
// u=uMax, v=vMin, v=vMax), each just extractIsocurveU/V called AT a
// domain boundary rather than an interior parameter — genuinely zero new
// evaluation math beyond the two functions just above. The one real
// design decision this needed: which of the 4 are actually NAKED
// (free) edges, since a surface that WRAPS in a direction (surfaceClosure,
// already built for the INSPECT/WHAT "Open Edges" report) has its own
// u=uMin/u=uMax boundary curves COINCIDE exactly (same control ROW,
// literally the same curve) — extracting both would silently return a
// duplicate. Real Rhino's own DupBorder/ExtractWireframe border case
// has the identical convention: a closed surface's own seam is internal,
// not a border. Filtering BY CONSTRUCTION here (never emitting the
// closed direction's pair at all) means there is no seam-duplicate case
// to dedupe after the fact — the same class of cosmetic gap Divide/
// DivideSrf both left open for their own closed-curve/closed-surface
// case doesn't arise for ExtractBorder at all, by design, not by luck.
// A surface closed in BOTH directions (e.g. a full torus, buildable via
// Revolve of a closed profile through a full sweep) correctly returns
// an EMPTY array — a genuinely valid answer (nakedEdgeCount(srf) === 0
// for the identical reason), not a bug to special-case away.
export function extractBorderCurves(srf, tol = 1e-6) {
  const { closedU, closedV } = surfaceClosure(srf, tol);
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const borders = [];
  if (!closedU) {
    borders.push({ edge: 'uMin', crv: extractIsocurveU(srf, uMin) });
    borders.push({ edge: 'uMax', crv: extractIsocurveU(srf, uMax) });
  }
  if (!closedV) {
    borders.push({ edge: 'vMin', crv: extractIsocurveV(srf, vMin) });
    borders.push({ edge: 'vMax', crv: extractIsocurveV(srf, vMax) });
  }
  return borders;
}

// ExtractWireframe — the spec: "All isocurves at the
// object's current ISOCURVEDENSITY setting become real, independent curve
// objects at once (batch EXTRACTISOCURVE)." There is no real, adjustable
// per-object ISOCURVEDENSITY setting anywhere in this app yet (grep-
// confirmed before writing this — only ISOCURVE_SAMPLE_COUNT exists, the
// unrelated RENDER-tessellation density of a single already-extracted
// curve's own display polyline, not a count of HOW MANY isocurves to
// extract). So v1 reuses the ONE density convention this app already has
// and already ships live: the WIREFRAME status-row overlay's own Greville-
// abscissae placement (the app's `isocurveParams`/
// `buildIsocurveOverlay` — one isocurve per control-point ROW in each
// direction, by construction always including both domain edges for a
// clamped knot vector, since a curve's first/last Greville values land
// exactly on its own boundary). Reused verbatim rather than inventing a
// second, different density rule for what is otherwise the identical
// visual promise ("show me the surface's own isocurve structure").
//
// Genuinely ZERO new evaluation math: grevilleAbscissae (curve.mjs,
// already proven for Sweep1's own frame placement AND the Wireframe
// overlay) picks the fixed parameter values; extractIsocurveU/V (this
// file, already proven for ExtractIsocurve/ExtractBorder) turn each one
// into a REAL, exact curve — a fixed-U row (running along V) always uses
// extractIsocurveU; a fixed-V row (running along U) always uses
// extractIsocurveV, the identical pairing buildIsocurveOverlay itself uses
// for its own uVals/vVals loops.
//
// SEAM DUPLICATION on a surface closed in one direction is DELIBERATELY
// NOT filtered out here — unlike extractBorderCurves' own closed-direction
// filter just above, this function matches the WIREFRAME OVERLAY's own
// exact convention (which never filters by closure either): a closed
// cylinder's own seam row appears twice (once as the first Greville value,
// once as the last). This is the SAME already-accepted cosmetic gap
// Divide/DivideSrf's own closed-curve/closed-surface case already has (see
// extractBorderCurves' own comment for that established precedent) — a
// real, named parallel to an existing accepted quirk, not a new bug, and
// not silently different from the very overlay this command claims to
// batch-extract.
export function extractWireframeCurves(srf) {
  const uVals = grevilleAbscissae({ degree: srf.degU, knots: srf.knotsU, ctrlPts: new Array(srf.ctrlNet.length) });
  const vVals = grevilleAbscissae({ degree: srf.degV, knots: srf.knotsV, ctrlPts: new Array(srf.ctrlNet[0].length) });
  const wires = [];
  for (const u of uVals) wires.push({ dir: 'U', param: u, crv: extractIsocurveU(srf, u) });
  for (const v of vVals) wires.push({ dir: 'V', param: v, crv: extractIsocurveV(srf, v) });
  return wires;
}
