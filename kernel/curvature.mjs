// SECOND PARTIALS, AND THE CURVATURES THEY GIVE.
//
// This is the quantity the app has been missing at both ends: it is the DISPLAY
// half of curvature analysis, and it is the
// prerequisite for G2 continuity, which constrains a third control row through a
// relation between second derivatives. One piece of maths, two features.
//
// ⚠ RATIONAL SURFACES NEED THE QUOTIENT RULE, and skipping it is the classic
// way to get curvature that looks plausible and is wrong. A NURBS surface is
// A(u,v) / w(u,v), and the second derivative of a quotient is not the quotient
// of second derivatives. Every primitive sphere and cylinder in this app is
// rational, so a version that ignored weights would be wrong on exactly the
// surfaces whose curvature is best known — which is why the test asserts
// against the closed forms rather than against itself.
//
// The rational derivative recurrence is Piegl & Tiller's (The NURBS Book, A4.4),
// specialised to order 2 rather than written generally — the general form needs
// binomial tables and there are only six terms.
import { findSpan, dersBasisFuns } from './basis.mjs';

// ⚠ THE CONVENTION IS THE KERNEL'S, NOT THE TEXTBOOK'S. `ctrlNet` stores PLAIN
// Euclidean positions with the weight alongside — `toHomogeneousNet` in
// surface.mjs is what multiplies through. Assuming the positions were already
// pre-multiplied gives curvature that varies smoothly, plots convincingly, and
// is wrong on every rational surface, which is every primitive sphere and
// cylinder in this app.
function toHomogeneous(cp) {
  const w = cp[3] === undefined ? 1 : cp[3];
  return [cp[0] * w, cp[1] * w, cp[2] * w, w];
}

// All homogeneous partials up to total order 2: A, Au, Av, Auu, Auv, Avv.
function homogeneousDerivs2(srf, u, v) {
  const { degU: p, degV: q, knotsU: U, knotsV: V } = srf;
  const n = srf.ctrlNet.length - 1;
  const m = srf.ctrlNet[0].length - 1;
  const uspan = findSpan(n, p, u, U);
  const vspan = findSpan(m, q, v, V);
  const du = Math.min(2, p), dv = Math.min(2, q);
  const dersU = dersBasisFuns(uspan, u, p, du, U);
  const dersV = dersBasisFuns(vspan, v, q, dv, V);
  const at = (ders, order, k) => (order <= (ders.length - 1) ? ders[order][k] : 0);
  const acc = { A: [0, 0, 0, 0], Au: [0, 0, 0, 0], Av: [0, 0, 0, 0], Auu: [0, 0, 0, 0], Auv: [0, 0, 0, 0], Avv: [0, 0, 0, 0] };
  for (let k = 0; k <= p; k++) {
    for (let l = 0; l <= q; l++) {
      const cpRaw = srf.ctrlNet[uspan - p + k][vspan - q + l];
      const w = cpRaw[3] === undefined ? 1 : cpRaw[3];
      const cp = toHomogeneous(cpRaw);
      const nu0 = at(dersU, 0, k), nu1 = at(dersU, 1, k), nu2 = at(dersU, 2, k);
      const nv0 = at(dersV, 0, l), nv1 = at(dersV, 1, l), nv2 = at(dersV, 2, l);
      for (let c = 0; c < 4; c++) {
        const val = c === 3 ? w : cp[c];
        acc.A[c] += nu0 * nv0 * val;
        acc.Au[c] += nu1 * nv0 * val;
        acc.Av[c] += nu0 * nv1 * val;
        acc.Auu[c] += nu2 * nv0 * val;
        acc.Auv[c] += nu1 * nv1 * val;
        acc.Avv[c] += nu0 * nv2 * val;
      }
    }
  }
  return acc;
}

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len3 = (a) => Math.hypot(a[0], a[1], a[2]);

// S, Su, Sv, Suu, Suv, Svv — the real, de-homogenised derivatives.
export function surfaceDerivs2(srf, u, v) {
  const H = homogeneousDerivs2(srf, u, v);
  const w = H.A[3], wu = H.Au[3], wv = H.Av[3], wuu = H.Auu[3], wuv = H.Auv[3], wvv = H.Avv[3];
  const A = [H.A[0], H.A[1], H.A[2]];
  const Au = [H.Au[0], H.Au[1], H.Au[2]];
  const Av = [H.Av[0], H.Av[1], H.Av[2]];
  const Auu = [H.Auu[0], H.Auu[1], H.Auu[2]];
  const Auv = [H.Auv[0], H.Auv[1], H.Auv[2]];
  const Avv = [H.Avv[0], H.Avv[1], H.Avv[2]];
  const S = scale3(A, 1 / w);
  const Su = scale3(sub3(Au, scale3(S, wu)), 1 / w);
  const Sv = scale3(sub3(Av, scale3(S, wv)), 1 / w);
  // Second order: subtract every lower-order term the weight contributes.
  const Suu = scale3(sub3(sub3(Auu, scale3(Su, 2 * wu)), scale3(S, wuu)), 1 / w);
  const Svv = scale3(sub3(sub3(Avv, scale3(Sv, 2 * wv)), scale3(S, wvv)), 1 / w);
  const Suv = scale3(sub3(sub3(sub3(Auv, scale3(Su, wv)), scale3(Sv, wu)), scale3(S, wuv)), 1 / w);
  return { S, Su, Sv, Suu, Suv, Svv };
}

// GAUSSIAN AND MEAN CURVATURE, from the two fundamental forms.
//
// Sign convention, stated because it is otherwise a coin toss: the normal is
// Su x Sv normalised, so a sphere built by revolving outward reads POSITIVE
// mean curvature. Gaussian curvature has no such ambiguity — it is positive on
// a dome or a bowl alike, zero on anything developable, negative on a saddle,
// which is exactly what makes it the honest one to colour by.
export function surfaceCurvature(srf, u, v) {
  const d = surfaceDerivs2(srf, u, v);
  const n = cross3(d.Su, d.Sv);
  const nl = len3(n);
  // A POLE HAS NO TANGENT PLANE. Su x Sv vanishes where the surface degenerates
  // to a point (the top of a sphere, the apex of a cone), and every quantity
  // below divides by it. Reported as null rather than as Infinity or a large
  // number that would dominate any colour scale built from it.
  if (nl < 1e-12) return { ok: false, reason: 'degenerate point (pole) — no tangent plane, so no curvature', K: null, H: null, k1: null, k2: null, normal: null };
  const N = scale3(n, 1 / nl);
  const E = dot3(d.Su, d.Su), F = dot3(d.Su, d.Sv), G = dot3(d.Sv, d.Sv);
  const L = dot3(d.Suu, N), M = dot3(d.Suv, N), Nn = dot3(d.Svv, N);
  const den = E * G - F * F;
  if (Math.abs(den) < 1e-18) return { ok: false, reason: 'degenerate first fundamental form', K: null, H: null, k1: null, k2: null, normal: N };
  const K = (L * Nn - M * M) / den;
  const Hm = (E * Nn - 2 * F * M + G * L) / (2 * den);
  // Principal curvatures are the roots of k^2 - 2Hk + K = 0. The discriminant
  // is non-negative in exact arithmetic and can go slightly negative in
  // floating point on an umbilic, where the two roots are equal.
  const disc = Math.max(0, Hm * Hm - K);
  const root = Math.sqrt(disc);
  return { ok: true, K, H: Hm, k1: Hm + root, k2: Hm - root, normal: N };
}

// The tightest bend anywhere on the surface, and where. This is the "minimum
// radius of curvature" check — the number that says whether a
// shape can be cut with a given tool, or moulded at all.
export function minimumRadiusOfCurvature(srf, samplesU = 24, samplesV = 24) {
  const dom = (knots, deg) => [knots[deg], knots[knots.length - 1 - deg]];
  const [uMin, uMax] = dom(srf.knotsU, srf.degU);
  const [vMin, vMax] = dom(srf.knotsV, srf.degV);
  let worst = null;
  for (let i = 0; i < samplesU; i++) {
    for (let j = 0; j < samplesV; j++) {
      const u = uMin + ((uMax - uMin) * i) / (samplesU - 1);
      const v = vMin + ((vMax - vMin) * j) / (samplesV - 1);
      const c = surfaceCurvature(srf, u, v);
      if (!c.ok) continue;
      const kMax = Math.max(Math.abs(c.k1), Math.abs(c.k2));
      if (kMax < 1e-12) continue; // flat here — an infinite radius is not a constraint
      const r = 1 / kMax;
      if (!worst || r < worst.radius) worst = { radius: r, u, v };
    }
  }
  return worst; // null when the surface is flat everywhere, which is the honest answer
}
