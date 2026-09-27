// ExtractIsocurve — Piegl & Tiller's defining property of a NURBS surface's
// isoparametric curves: holding one parameter fixed collapses the surface's
// tensor-product sum into an ordinary NURBS curve running along the other
// parameter, of that direction's degree/knots exactly — not a resampled
// polyline approximation of one.
//
//   S(u0, v) = sum_j  [ sum_i N_i,p(u0) * Pw[i][j] ]  * N_j,q(v)
//                       \_________________________/
//                       the isocurve's own j-th homogeneous control point
//
// This is the basis-function machinery surfacePoint (surface.mjs) runs for a
// single point, and that the Wireframe isocurve overlay uses to sample points
// along an isocurve for display. What is added here is contracting the full
// control net along one direction (every row/column, not just the p+1 or q+1
// nonzero terms near one sample) to get an independent curve object (degree +
// knots + control points) rather than a stream of sampled points.
//
// A NurbsCrv here, as in curve.mjs/interpolate.mjs, is
// { degree, knots, ctrlPts } with ctrlPts = [x, y, z, w] (point + its weight,
// not premultiplied) — the SketchCurve representation, so an extracted
// isocurve is an ordinary editable/joinable/extrudable curve object.

import { findSpan, basisFuns } from './basis.mjs';
import { surfaceClosure } from './surface.mjs';
import { grevilleAbscissae } from './curve.mjs';

// Fix U at `u`, return the NurbsCrv running along the surface's V direction
// (degree degV, knots knotsV).
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

// Fix V at `v`, return the NurbsCrv running along the surface's U direction
// (degree degU, knots knotsU) — the symmetric twin of the above.
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

// ExtractBorder — a surface's 4 parametric-domain edges (u=uMin, u=uMax,
// v=vMin, v=vMax), each extractIsocurveU/V called at a domain boundary. The
// design decision is which of the 4 are naked (free) edges: a surface that
// wraps in a direction (surfaceClosure, as used by the "Open Edges" report)
// has its u=uMin/u=uMax boundary curves coincide exactly (the same control
// row), so extracting both would return a duplicate. Rhino's
// DupBorder/ExtractWireframe border case has the same convention: a closed
// surface's seam is internal, not a border. The closed direction's pair is
// never emitted, so there is no seam duplicate to remove afterwards.
// A surface closed in both directions (e.g. a full torus, from Revolve of a
// closed profile through a full sweep) returns an empty array — a valid
// answer (nakedEdgeCount(srf) === 0 for the same reason).
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

// ExtractWireframe — every isocurve at the object's isocurve density becomes
// an independent curve object at once (a batch ExtractIsocurve). There is no
// adjustable per-object isocurve-density setting in this app
// (ISOCURVE_SAMPLE_COUNT is the render-tessellation density of one curve's
// display polyline, not a count of isocurves), so this uses the density
// convention of the Wireframe overlay: Greville-abscissae placement (the
// app's `isocurveParams`/`buildIsocurveOverlay` — one isocurve per
// control-point row in each direction, always including both domain edges for
// a clamped knot vector, since a curve's first/last Greville values land
// exactly on its boundary).
//
// No new evaluation math: grevilleAbscissae (curve.mjs) picks the fixed
// parameter values and extractIsocurveU/V turn each one into an exact curve —
// a fixed-U row (running along V) uses extractIsocurveU; a fixed-V row
// (running along U) uses extractIsocurveV, the same pairing
// buildIsocurveOverlay uses for its uVals/vVals loops.
//
// Seam duplication on a surface closed in one direction is not filtered here,
// unlike extractBorderCurves: this function matches the Wireframe overlay,
// which does not filter by closure either, so a closed cylinder's seam row
// appears twice (once as the first Greville value, once as the last). This is
// the same known limitation Divide/DivideSrf have for closed curves and
// surfaces.
export function extractWireframeCurves(srf) {
  const uVals = grevilleAbscissae({ degree: srf.degU, knots: srf.knotsU, ctrlPts: new Array(srf.ctrlNet.length) });
  const vVals = grevilleAbscissae({ degree: srf.degV, knots: srf.knotsV, ctrlPts: new Array(srf.ctrlNet[0].length) });
  const wires = [];
  for (const u of uVals) wires.push({ dir: 'U', param: u, crv: extractIsocurveU(srf, u) });
  for (const v of vVals) wires.push({ dir: 'V', param: v, crv: extractIsocurveV(srf, v) });
  return wires;
}
