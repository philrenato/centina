// Coons patch — the bilinearly blended Coons surface over four NURBS curves,
// as one NURBS whose boundary is each curve: S = R_u + R_v - B, where R_u
// rules between the two u-curves, R_v between the two v-curves and B is the
// bilinear sheet through the four corners. Each term is exact in the tensor
// basis once the opposite curves share a degree and a knot vector: a linear
// function of the parameter has B-spline coefficients equal to its values at
// the Greville abscissae, so the ruling and the bilinear term are written
// straight into the control net, no sampling and no refit. The loop is
// c0 (u at v=0), c1 (v at u=1), c2 (u at v=1, running back), c3 (v at u=0,
// running back), head to tail, as boundSurfaceFromLoop takes it.
//
// Weights ride along in homogeneous coordinates: the boundaries are exact
// whenever the curves meeting at a corner carry the same weight there (every
// caller here passes weight-1 curves).
import { insertKnot, degreeElevateCurve, rescaleCurveDomain } from './knots.mjs';
import { reverseCurve } from './curve.mjs';

// Two curves brought to one degree and one knot vector on [0, 1].
export function coonsCompatible(a, b) {
  const p = Math.max(a.degree, b.degree);
  let A = rescaleCurveDomain(degreeElevateCurve(a, p), 0, 1);
  let B = rescaleCurveDomain(degreeElevateCurve(b, p), 0, 1);
  const mult = (knots, u) => knots.filter((k) => Math.abs(k - u) < 1e-9).length;
  const interior = (c) => c.knots.slice(c.degree + 1, c.knots.length - c.degree - 1);
  const values = [];
  for (const u of [...interior(A), ...interior(B)]) if (!values.some((v) => Math.abs(v - u) < 1e-9)) values.push(u);
  for (const u of values) {
    const m = Math.max(mult(A.knots, u), mult(B.knots, u));
    if (mult(A.knots, u) < m) A = insertKnot(A, u, m - mult(A.knots, u));
    if (mult(B.knots, u) < m) B = insertKnot(B, u, m - mult(B.knots, u));
  }
  return [A, B];
}

const greville = (knots, p) => Array.from({ length: knots.length - p - 1 }, (_, i) => { let s = 0; for (let k = 1; k <= p; k++) s += knots[i + k]; return s / p; });

export function coonsPatch(c0, c1, c2, c3) {
  const [U0, U1] = coonsCompatible(c0, reverseCurve(c2));
  const [V0, V1] = coonsCompatible(reverseCurve(c3), c1);
  const p = U0.degree, q = V0.degree;
  const xu = greville(U0.knots, p), xv = greville(V0.knots, q);
  const P00 = U0.ctrlPts[0], P10 = U0.ctrlPts[U0.ctrlPts.length - 1], P01 = U1.ctrlPts[0], P11 = U1.ctrlPts[U1.ctrlPts.length - 1];
  const ctrlNet = [];
  for (let i = 0; i < xu.length; i++) {
    const row = [];
    const s = xu[i];
    for (let j = 0; j < xv.length; j++) {
      const t = xv[j];
      const pt = [0, 0, 0, 0];
      for (let c = 0; c < 4; c++) {
        pt[c] = (1 - t) * U0.ctrlPts[i][c] + t * U1.ctrlPts[i][c]
          + (1 - s) * V0.ctrlPts[j][c] + s * V1.ctrlPts[j][c]
          - ((1 - s) * (1 - t) * P00[c] + s * (1 - t) * P10[c] + (1 - s) * t * P01[c] + s * t * P11[c]);
      }
      row.push(pt);
    }
    ctrlNet.push(row);
  }
  return { degU: p, knotsU: U0.knots.slice(), degV: q, knotsV: V0.knots.slice(), ctrlNet };
}
