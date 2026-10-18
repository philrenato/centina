// Exact measures of NURBS geometry, each with a convergence verdict.
//
//   planarCurveArea(crv)                         area inside a closed planar curve
//   closedSurfaceVolume(srf | srf[])             volume inside closed surfaces
//   revolveRegionVolume(meridian, axis, clip)    volume of a revolved region (Pappus)
//   fractionToHeight(meridian, axis, fraction)   the height that holds a fraction of it
//
// Every function returns { value, exact, gap, reason }:
//   value   the measure, or null when the geometry has no answer
//   exact   true when the quadrature at n cells and at 2n cells agree within
//           1e-9 relative; value is the 2n figure
//   gap     |value(2n) - value(n)|, in the measure's own units
//   reason  null with a value; otherwise names why there is none
//           ("not closed", "not planar", "crosses itself", ...)
//
// The integrals are taken over the exact NURBS, cell by cell, with cells aligned
// to the knots: the derivative of a NURBS is discontinuous across a
// full-multiplicity interior knot, and a cell straddling one integrates a kink
// (kernel/flatten.mjs nurbsSurfaceArea measures the same loss on a revolve).
// A polynomial span's integrand is a polynomial, which Gauss-Legendre of the
// chosen order integrates exactly, so n and 2n agree at once; a rational span
// converges, and the cell count doubles until it does or the budget runs out.
//
// Sign comes from closure, never from winding: a closed boundary bounds one
// region, whose measure is the magnitude of the signed integral. A flipped
// surface or a clockwise curve gives the same value.
//
// Coordinates are taken relative to the control points' centroid before
// integrating, so a shape 1e5 mm from the origin loses no digits to the
// cancellation an origin-fanned integral would carry.
//
// Self-contained (its own span search and basis derivatives) so the app's pasted
// copy needs no bindings.

const REL_AGREE = 1e-9;
const GAUSS = new Map();

// Gauss-Legendre nodes and weights on [-1, 1], by Newton on P_n.
function gaussLegendre(n) {
  if (GAUSS.has(n)) return GAUSS.get(n);
  const out = [];
  for (let i = 0; i < n; i += 1) {
    let x = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));
    let dp = 1;
    for (let it = 0; it < 100; it += 1) {
      let p0 = 1, p1 = x;
      for (let k = 2; k <= n; k += 1) { const p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
      dp = n * (x * p1 - p0) / (x * x - 1);
      const dx = p1 / dp;
      x -= dx;
      if (Math.abs(dx) < 1e-16) break;
    }
    let p0 = 1, p1 = x;
    for (let k = 2; k <= n; k += 1) { const p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
    dp = n * (x * p1 - p0) / (x * x - 1);
    out.push([x, 2 / ((1 - x * x) * dp * dp)]);
  }
  GAUSS.set(n, out);
  return out;
}

function spanIndex(n, p, u, U) {
  if (u >= U[n + 1]) return n;
  if (u <= U[p]) return p;
  let lo = p, hi = n + 1, mid = (lo + hi) >> 1;
  while (u < U[mid] || u >= U[mid + 1]) {
    if (u < U[mid]) hi = mid; else lo = mid;
    mid = (lo + hi) >> 1;
  }
  return mid;
}

// Basis values and first derivatives at u on `span` (Piegl & Tiller A2.3, n = 1).
function basisAndDeriv(span, u, p, U) {
  const ndu = Array.from({ length: p + 1 }, () => new Array(p + 1).fill(0));
  const left = new Array(p + 1).fill(0), right = new Array(p + 1).fill(0);
  ndu[0][0] = 1;
  for (let j = 1; j <= p; j += 1) {
    left[j] = u - U[span + 1 - j];
    right[j] = U[span + j] - u;
    let saved = 0;
    for (let r = 0; r < j; r += 1) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const t = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * t;
      saved = left[j - r] * t;
    }
    ndu[j][j] = saved;
  }
  const N = new Array(p + 1), D = new Array(p + 1).fill(0);
  for (let j = 0; j <= p; j += 1) N[j] = ndu[j][p];
  if (p === 0) return [N, D];
  for (let r = 0; r <= p; r += 1) {
    let d = 0;
    const rk = r - 1, pk = p - 1;
    if (r >= 1) d += (1 / ndu[pk + 1][rk]) * ndu[rk][pk];
    if (r <= pk) d -= (1 / ndu[pk + 1][r]) * ndu[r][pk];
    D[r] = d * p;
  }
  return [N, D];
}

function curveDomainOf(crv) {
  const p = crv.degree, U = crv.knots;
  return [U[p], U[U.length - 1 - p]];
}

// Euclidean point and first derivative of a curve at u.
function curvePD(crv, u) {
  const p = crv.degree, U = crv.knots, P = crv.ctrlPts;
  const span = spanIndex(P.length - 1, p, u, U);
  const [N, D] = basisAndDeriv(span, u, p, U);
  const S = [0, 0, 0, 0], Su = [0, 0, 0, 0];
  for (let k = 0; k <= p; k += 1) {
    const c = P[span - p + k], w = c[3] ?? 1;
    const h = [c[0] * w, c[1] * w, c[2] * w, w];
    for (let t = 0; t < 4; t += 1) { S[t] += N[k] * h[t]; Su[t] += D[k] * h[t]; }
  }
  const w = S[3];
  const pt = [S[0] / w, S[1] / w, S[2] / w];
  return { point: pt, d: [(Su[0] - Su[3] * pt[0]) / w, (Su[1] - Su[3] * pt[1]) / w, (Su[2] - Su[3] * pt[2]) / w] };
}

// Point and both first partials of a surface at (u, v).
function surfacePD(srf, u, v) {
  const p = srf.degU, q = srf.degV, net = srf.ctrlNet;
  const us = spanIndex(net.length - 1, p, u, srf.knotsU);
  const vs = spanIndex(net[0].length - 1, q, v, srf.knotsV);
  const [Nu, Du] = basisAndDeriv(us, u, p, srf.knotsU);
  const [Nv, Dv] = basisAndDeriv(vs, v, q, srf.knotsV);
  const S = [0, 0, 0, 0], Su = [0, 0, 0, 0], Sv = [0, 0, 0, 0];
  for (let k = 0; k <= p; k += 1) {
    for (let l = 0; l <= q; l += 1) {
      const c = net[us - p + k][vs - q + l], w = c[3] ?? 1;
      const a = Nu[k] * Nv[l], b = Du[k] * Nv[l], e = Nu[k] * Dv[l];
      const h0 = c[0] * w, h1 = c[1] * w, h2 = c[2] * w;
      S[0] += a * h0; S[1] += a * h1; S[2] += a * h2; S[3] += a * w;
      Su[0] += b * h0; Su[1] += b * h1; Su[2] += b * h2; Su[3] += b * w;
      Sv[0] += e * h0; Sv[1] += e * h1; Sv[2] += e * h2; Sv[3] += e * w;
    }
  }
  const w = S[3];
  const pt = [S[0] / w, S[1] / w, S[2] / w];
  return {
    point: pt,
    su: [(Su[0] - Su[3] * pt[0]) / w, (Su[1] - Su[3] * pt[1]) / w, (Su[2] - Su[3] * pt[2]) / w],
    sv: [(Sv[0] - Sv[3] * pt[0]) / w, (Sv[1] - Sv[3] * pt[1]) / w, (Sv[2] - Sv[3] * pt[2]) / w],
  };
}

// Distinct knot values inside [lo, hi]: the span boundaries.
function spanBreaks(U, lo, hi) {
  const out = [];
  for (const k of U) {
    if (k < lo || k > hi) continue;
    if (!out.length || k - out[out.length - 1] > 1e-12 * Math.max(1, Math.abs(hi - lo))) out.push(k);
  }
  return out;
}

function isRationalPts(pts) {
  const w0 = pts[0][3] ?? 1;
  for (const c of pts) if (Math.abs((c[3] ?? 1) - w0) > 1e-14 * Math.abs(w0)) return true;
  return false;
}

function bboxDiag(pts) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const c of pts) for (let k = 0; k < 3; k += 1) { if (c[k] < lo[k]) lo[k] = c[k]; if (c[k] > hi[k]) hi[k] = c[k]; }
  return Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
}

function centroidOf(pts) {
  const c = [0, 0, 0];
  for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  return [c[0] / pts.length, c[1] / pts.length, c[2] / pts.length];
}

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);

const noAnswer = (reason) => ({ value: null, exact: false, gap: Infinity, reason });

/* Doubling until n and 2n agree. `integrate(c)` returns the signed integral with
   c cells per interval and the number of evaluations it spent. Stops at agreement,
   at maxCells, or when the next doubling would exceed the evaluation budget. */
function converge(integrate, cells0, maxCells, budget) {
  let c = Math.max(1, cells0 | 0);
  let a = integrate(c);
  let spent = a.evals;
  for (;;) {
    const b = integrate(2 * c);
    spent += b.evals;
    const gap = Math.abs(b.value - a.value);
    const exact = gap <= REL_AGREE * Math.max(Math.abs(b.value), 1e-300);
    if (exact || 2 * c >= maxCells || spent + 2 * b.evals > budget) return { value: b.value, exact, gap };
    a = b; c *= 2;
  }
}

function validCurve(crv) {
  return !!(crv && Array.isArray(crv.knots) && Array.isArray(crv.ctrlPts) && Number.isFinite(crv.degree)
    && crv.degree >= 1 && crv.ctrlPts.length >= crv.degree + 1 && crv.knots.length === crv.ctrlPts.length + crv.degree + 1
    && crv.ctrlPts.every((c) => Array.isArray(c) && c.length >= 3 && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2])));
}

/* A plane through the control points: the point farthest from the centroid, then
   the point that makes the largest cross product with it. Newell's normal is not
   used because it vanishes on a figure-eight, whose signed area is zero. */
function planeOfPoints(pts, c) {
  let a = null, best = -1;
  for (const p of pts) { const d = norm3(sub3(p, c)); if (d > best) { best = d; a = sub3(p, c); } }
  if (!(best > 0)) return null;
  let n = null; best = -1;
  for (const p of pts) { const x = cross3(a, sub3(p, c)); const m = norm3(x); if (m > best) { best = m; n = x; } }
  if (!(best > 1e-12 * norm3(a) * norm3(a))) return null;
  const nn = norm3(n);
  n = [n[0] / nn, n[1] / nn, n[2] / nn];
  const al = norm3(a);
  const e1 = [a[0] / al, a[1] / al, a[2] / al];
  return { n, e1, e2: cross3(n, e1) };
}

function segmentsCross(a, b, c, d) {
  const o = (p, q, r) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const s = Math.max(Math.abs(b[0] - a[0]) + Math.abs(b[1] - a[1]), Math.abs(d[0] - c[0]) + Math.abs(d[1] - c[1]));
  const eps = 1e-12 * s * s;
  const d1 = o(a, b, c), d2 = o(a, b, d), d3 = o(c, d, a), d4 = o(c, d, b);
  return ((d1 > eps && d2 < -eps) || (d1 < -eps && d2 > eps)) && ((d3 > eps && d4 < -eps) || (d3 < -eps && d4 > eps));
}

// Whether a closed polyline in 2D crosses itself (non-adjacent segments only).
function polylineCrossesItself(P) {
  const m = P.length - 1;   // P[m] === P[0]
  if (m < 4) return false;
  const lo = [], hi = [];
  for (let i = 0; i < m; i += 1) {
    lo.push([Math.min(P[i][0], P[i + 1][0]), Math.min(P[i][1], P[i + 1][1])]);
    hi.push([Math.max(P[i][0], P[i + 1][0]), Math.max(P[i][1], P[i + 1][1])]);
  }
  for (let i = 0; i < m; i += 1) {
    for (let j = i + 2; j < m; j += 1) {
      if (i === 0 && j === m - 1) continue;
      if (lo[i][0] > hi[j][0] || lo[j][0] > hi[i][0] || lo[i][1] > hi[j][1] || lo[j][1] > hi[i][1]) continue;
      if (segmentsCross(P[i], P[i + 1], P[j], P[j + 1])) return true;
    }
  }
  return false;
}

/**
 * Area enclosed by a closed planar NURBS curve: Green's theorem in the curve's
 * own plane, A = 1/2 |∮ (x y' - y x') du|, Gauss-Legendre per knot span.
 * opts: { tol (mm, closure and planarity), cellsPerSpan, maxCellsPerSpan, order, budget }
 */
export function planarCurveArea(crv, opts = {}) {
  if (!validCurve(crv)) return noAnswer('not a curve');
  const P = crv.ctrlPts;
  const diag = bboxDiag(P);
  if (!(diag > 0)) return noAnswer('encloses nothing');
  const tol = opts.tol ?? 1e-6 * diag;
  const [u0, u1] = curveDomainOf(crv);
  const a = curvePD(crv, u0).point, b = curvePD(crv, u1).point;
  if (norm3(sub3(a, b)) > tol) return noAnswer('not closed');
  const c = centroidOf(P);
  const plane = planeOfPoints(P, c);
  if (!plane) return noAnswer('encloses nothing');
  for (const p of P) if (Math.abs(dot3(sub3(p, c), plane.n)) > tol) return noAnswer('not planar');
  const to2 = (p) => { const r = sub3(p, c); return [dot3(r, plane.e1), dot3(r, plane.e2)]; };
  const breaks = spanBreaks(crv.knots, u0, u1);
  const spans = breaks.length - 1;
  const per = crv.degree === 1 ? 1 : (spans > 100 ? 8 : 24);
  const poly = [];
  for (let s = 0; s < spans; s += 1) {
    for (let k = 0; k < per; k += 1) poly.push(to2(curvePD(crv, breaks[s] + (breaks[s + 1] - breaks[s]) * (k / per)).point));
  }
  poly.push(poly[0]);
  if (polylineCrossesItself(poly)) return noAnswer('crosses itself');
  const rational = isRationalPts(P);
  const G = gaussLegendre(opts.order ?? (rational ? Math.max(crv.degree + 1, 8) : crv.degree + 1));
  const integrate = (cells) => {
    let sum = 0, evals = 0;
    for (let s = 0; s < spans; s += 1) {
      const sa = breaks[s], sb = breaks[s + 1];
      for (let k = 0; k < cells; k += 1) {
        const ca = sa + (sb - sa) * (k / cells), cb = sa + (sb - sa) * ((k + 1) / cells);
        const h = (cb - ca) / 2, m = (cb + ca) / 2;
        for (const [x, w] of G) {
          const e = curvePD(crv, m + h * x);
          const r = sub3(e.point, c);
          const px = dot3(r, plane.e1), py = dot3(r, plane.e2);
          const dx = dot3(e.d, plane.e1), dy = dot3(e.d, plane.e2);
          sum += 0.5 * (px * dy - py * dx) * w * h;
          evals += 1;
        }
      }
    }
    return { value: sum, evals };
  };
  const r = converge(integrate, opts.cellsPerSpan ?? 1, opts.maxCellsPerSpan ?? 64, opts.budget ?? 2e5);
  return { value: Math.abs(r.value), exact: r.exact, gap: r.gap, reason: null };
}

function validSurface(srf) {
  if (!srf || !Array.isArray(srf.ctrlNet) || !Array.isArray(srf.ctrlNet[0]) || !Array.isArray(srf.knotsU) || !Array.isArray(srf.knotsV)) return false;
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  if (!(srf.degU >= 1 && srf.degV >= 1)) return false;
  if (srf.knotsU.length !== nu + srf.degU + 1 || srf.knotsV.length !== nv + srf.degV + 1) return false;
  return srf.ctrlNet.every((row) => row.length === nv && row.every((c) => Array.isArray(c) && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2])));
}

/* Closed: every parametric side is either collapsed to a point (a pole) or lies
   on the opposite side, point for point (a seam). Read from the control net,
   whose boundary rows ARE the boundary curves of a clamped surface. */
function surfaceIsClosed(srf, tol) {
  const net = srf.ctrlNet, nu = net.length, nv = net[0].length;
  const same = (p, q) => norm3(sub3(p, q)) <= tol;
  const collapsed = (pts) => pts.every((p) => same(p, pts[0]));
  const rowU = (i) => net[i];
  const colV = (j) => net.map((row) => row[j]);
  const sideOk = (A, B) => (collapsed(A) || A.every((p, k) => same(p, B[k]))) && (collapsed(B) || B.every((p, k) => same(p, A[k])));
  return sideOk(rowU(0), rowU(nu - 1)) && sideOk(colV(0), colV(nv - 1));
}

function oneShellVolume(srf, opts) {
  if (!validSurface(srf)) return noAnswer('not a surface');
  const flat = srf.ctrlNet.flat();
  const diag = bboxDiag(flat);
  if (!(diag > 0)) return noAnswer('encloses nothing');
  if (!surfaceIsClosed(srf, opts.tol ?? 1e-6 * diag)) return noAnswer('not closed');
  const c = centroidOf(flat);
  const p = srf.degU, q = srf.degV;
  const [ua, ub] = [srf.knotsU[p], srf.knotsU[srf.knotsU.length - 1 - p]];
  const [va, vb] = [srf.knotsV[q], srf.knotsV[srf.knotsV.length - 1 - q]];
  const bu = spanBreaks(srf.knotsU, ua, ub), bv = spanBreaks(srf.knotsV, va, vb);
  const rational = isRationalPts(flat);
  // x · (Su × Sv) on a polynomial span is degree 3p-1 in u; ceil(3p/2) nodes integrate it.
  const ou = Math.max(Math.ceil(3 * p / 2), rational ? 6 : 1) + (rational ? 0 : 1);
  const ov = Math.max(Math.ceil(3 * q / 2), rational ? 6 : 1) + (rational ? 0 : 1);
  const GU = gaussLegendre(opts.order ?? ou), GV = gaussLegendre(opts.order ?? ov);
  const integrate = (cells) => {
    let sum = 0, evals = 0;
    for (let s = 0; s + 1 < bu.length; s += 1) {
      for (let a = 0; a < cells; a += 1) {
        const u0 = bu[s] + (bu[s + 1] - bu[s]) * (a / cells), u1 = bu[s] + (bu[s + 1] - bu[s]) * ((a + 1) / cells);
        const hu = (u1 - u0) / 2, mu = (u1 + u0) / 2;
        for (let t = 0; t + 1 < bv.length; t += 1) {
          for (let b = 0; b < cells; b += 1) {
            const v0 = bv[t] + (bv[t + 1] - bv[t]) * (b / cells), v1 = bv[t] + (bv[t + 1] - bv[t]) * ((b + 1) / cells);
            const hv = (v1 - v0) / 2, mv = (v1 + v0) / 2;
            for (const [x, wx] of GU) {
              for (const [y, wy] of GV) {
                const e = surfacePD(srf, mu + hu * x, mv + hv * y);
                sum += dot3(sub3(e.point, c), cross3(e.su, e.sv)) * wx * wy * hu * hv / 3;
                evals += 1;
              }
            }
          }
        }
      }
    }
    return { value: sum, evals };
  };
  const r = converge(integrate, opts.cellsPerSpan ?? 1, opts.maxCellsPerSpan ?? 32, opts.budget ?? 4e5);
  return { value: Math.abs(r.value), exact: r.exact, gap: r.gap, reason: null };
}

/**
 * Volume enclosed by one closed untrimmed NURBS surface, or the sum over an
 * array of disjoint closed surfaces: the divergence theorem,
 * V = 1/3 |∬ (S - c) · (Su × Sv) du dv|, over knot-aligned cells.
 * opts: { tol, cellsPerSpan, maxCellsPerSpan, order, budget }
 */
export function closedSurfaceVolume(srf, opts = {}) {
  const shells = Array.isArray(srf) ? srf : [srf];
  if (!shells.length) return noAnswer('not a surface');
  let value = 0, gap = 0, exact = true;
  for (const s of shells) {
    const r = oneShellVolume(s, opts);
    if (r.value == null) return r;
    value += r.value; gap += r.gap; exact = exact && r.exact;
  }
  return { value, exact, gap, reason: null };
}

function reverseCrv(crv) {
  const U = crv.knots, a = U[0], b = U[U.length - 1];
  return { degree: crv.degree, knots: U.map((k) => a + b - k).reverse(), ctrlPts: crv.ctrlPts.slice().reverse() };
}

/* The meridian's frame about the axis: h along the axis, r the signed distance
   in the half-plane that holds the curve. Null with a reason when the curve
   leaves that plane or straddles the axis. */
function axisFrameOf(crv, axis, tol) {
  const o = (axis && axis.origin) || [0, 0, 0];
  const d0 = (axis && axis.direction) || [0, 0, 1];
  const dl = norm3(d0);
  if (!(dl > 0)) return { reason: 'no axis' };
  const d = [d0[0] / dl, d0[1] / dl, d0[2] / dl];
  let far = null, best = 0;
  for (const p of crv.ctrlPts) {
    const v = sub3(p, o), h = dot3(v, d);
    const perp = [v[0] - h * d[0], v[1] - h * d[1], v[2] - h * d[2]];
    const m = norm3(perp);
    if (m > best) { best = m; far = perp; }
  }
  if (!(best > tol)) return { reason: 'encloses nothing' };
  const e = [far[0] / best, far[1] / best, far[2] / best];
  const n = cross3(d, e);
  for (const p of crv.ctrlPts) {
    const v = sub3(p, o);
    if (Math.abs(dot3(v, n)) > tol) return { reason: 'not planar' };
    if (dot3(v, e) < -tol) return { reason: 'crosses the axis' };
  }
  return { o, d, e, hOf: (p) => dot3(sub3(p, o), d), rOf: (p) => dot3(sub3(p, o), e) };
}

/* The region a revolve volume integrates over: the parameter range of the
   meridian, the cap height, and the lowest height the region reaches.
   Closed meridian: the whole loop. Open meridian: from the end on the axis to
   the first height maximum along the curve (the rim), after any initial descent
   (a punt) or flat (a floor); the cap is the horizontal from the rim to the axis. */
function revolveRegion(meridian, axis, opts) {
  if (!validCurve(meridian)) return { reason: 'not a curve' };
  const diag = bboxDiag(meridian.ctrlPts);
  if (!(diag > 0)) return { reason: 'encloses nothing' };
  const tol = opts.tol ?? 1e-6 * diag;
  let crv = meridian;
  const F = axisFrameOf(crv, axis, tol);
  if (F.reason) return F;
  let [u0, u1] = curveDomainOf(crv);
  const pa = curvePD(crv, u0).point, pb = curvePD(crv, u1).point;
  const closed = norm3(sub3(pa, pb)) <= tol;
  let uEnd = u1;
  if (!closed) {
    const onA = Math.abs(F.rOf(pa)) <= tol, onB = Math.abs(F.rOf(pb)) <= tol;
    if (!onA && !onB) return { reason: 'not on the axis' };
    if (!onA) { crv = reverseCrv(crv); [u0, u1] = curveDomainOf(crv); }
    const breaks = spanBreaks(crv.knots, u0, u1);
    const us = [];
    for (let s = 0; s + 1 < breaks.length; s += 1) for (let k = 0; k < 32; k += 1) us.push(breaks[s] + (breaks[s + 1] - breaks[s]) * (k / 32));
    us.push(u1);
    const hs = us.map((u) => F.hOf(curvePD(crv, u).point));
    const eps = 1e-12 * diag;
    let i = 0;
    while (i + 1 < hs.length && hs[i + 1] <= hs[i] + eps) i += 1;
    while (i + 1 < hs.length && hs[i + 1] >= hs[i] - eps) i += 1;
    uEnd = us[i];
    if (i + 1 < hs.length && i > 0) {
      let a = us[i - 1], b = us[i + 1];
      const g = (Math.sqrt(5) - 1) / 2;
      for (let it = 0; it < 200 && b - a > 1e-15 * Math.max(1, Math.abs(u1 - u0)); it += 1) {
        const x1 = b - g * (b - a), x2 = a + g * (b - a);
        if (F.hOf(curvePD(crv, x1).point) < F.hOf(curvePD(crv, x2).point)) a = x1; else b = x2;
      }
      uEnd = (a + b) / 2;
    }
  }
  const breaks = spanBreaks(crv.knots, u0, uEnd);
  if (breaks[breaks.length - 1] < uEnd) breaks.push(uEnd);
  let hLo = Infinity, hHi = -Infinity;
  for (let s = 0; s + 1 < breaks.length; s += 1) {
    for (let k = 0; k <= 32; k += 1) {
      const h = F.hOf(curvePD(crv, breaks[s] + (breaks[s + 1] - breaks[s]) * (k / 32)).point);
      if (h < hLo) hLo = h; if (h > hHi) hHi = h;
    }
  }
  return { crv, F, breaks, hLo, hHi, rational: isRationalPts(crv.ctrlPts) };
}

/* pi ∫ r² h' du over the meridian where h ≤ H: the clipped region's boundary is
   those curve pieces plus horizontal cuts and the axis, and both of those carry
   dh = 0 or r = 0, so they contribute nothing. Cells are split where h crosses H
   so no cell integrates the mask's step. */
function revolveIntegral(R, H, opts) {
  const { crv, F } = R;
  const hOf = (u) => F.hOf(curvePD(crv, u).point);
  const pieces = [];
  for (let s = 0; s + 1 < R.breaks.length; s += 1) {
    const a = R.breaks[s], b = R.breaks[s + 1];
    if (H == null) { pieces.push([a, b]); continue; }
    const cuts = [a];
    let ua = a, ga = hOf(a) - H;
    for (let k = 1; k <= 32; k += 1) {
      const ub = a + (b - a) * (k / 32), gb = hOf(ub) - H;
      if ((ga < 0 && gb > 0) || (ga > 0 && gb < 0)) {
        let lo = ua, hi = ub, glo = ga;
        for (let it = 0; it < 200 && hi - lo > 1e-16 * Math.max(1, Math.abs(b)); it += 1) {
          const m = (lo + hi) / 2, gm = hOf(m) - H;
          if ((gm < 0) === (glo < 0)) { lo = m; glo = gm; } else hi = m;
        }
        cuts.push((lo + hi) / 2);
      } else if (gb === 0 && k < 32) cuts.push(ub);
      ua = ub; ga = gb;
    }
    cuts.push(b);
    for (let k = 0; k + 1 < cuts.length; k += 1) {
      const m = (cuts[k] + cuts[k + 1]) / 2;
      if (cuts[k + 1] > cuts[k] && hOf(m) <= H) pieces.push([cuts[k], cuts[k + 1]]);
    }
  }
  // r² h' on a polynomial span is degree 3p-1; ceil(3p/2) nodes integrate it.
  const G = gaussLegendre(opts.order ?? (R.rational ? Math.max(crv.degree * 3, 8) : Math.ceil(3 * crv.degree / 2) + 1));
  const integrate = (cells) => {
    let sum = 0, evals = 0;
    for (const [a, b] of pieces) {
      for (let k = 0; k < cells; k += 1) {
        const ca = a + (b - a) * (k / cells), cb = a + (b - a) * ((k + 1) / cells);
        const h = (cb - ca) / 2, m = (cb + ca) / 2;
        for (const [x, w] of G) {
          const e = curvePD(crv, m + h * x);
          const r = F.rOf(e.point);
          sum += Math.PI * r * r * dot3(e.d, F.d) * w * h;
          evals += 1;
        }
      }
    }
    return { value: sum, evals };
  };
  const r = converge(integrate, opts.cellsPerSpan ?? 1, opts.maxCellsPerSpan ?? 64, opts.budget ?? 2e5);
  return { value: Math.abs(r.value), exact: r.exact, gap: r.gap, reason: null };
}

/**
 * Volume of the region a meridian sweeps about an axis (Pappus: 2π ∫ r dA,
 * taken by Green's theorem as π ∮ r² dh). A closed meridian bounds its own
 * region; an open one must start (or end) on the axis and bounds a cavity up to
 * its first height maximum, capped there. clipHeight (along the axis from
 * axis.origin) clips the region to h ≤ clipHeight; null takes all of it.
 * axis: { origin, direction }, default the world Z axis.
 */
export function revolveRegionVolume(meridian, axis, clipHeight = null, opts = {}) {
  const R = revolveRegion(meridian, axis, opts);
  if (R.reason) return noAnswer(R.reason);
  const H = clipHeight == null ? null : Math.min(clipHeight, R.hHi);
  if (H != null && H <= R.hLo) return { value: 0, exact: true, gap: 0, reason: null };
  return revolveIntegral(R, H, opts);
}

/**
 * The height (along the axis from axis.origin) at which the revolved region
 * holds `fraction` (0..1) of its volume. Bisection: the clipped volume grows
 * monotonically with the clip height. gap is the final bracket width.
 */
export function fractionToHeight(meridian, axis, fraction, opts = {}) {
  if (!(fraction >= 0 && fraction <= 1)) return noAnswer('out of range');
  const R = revolveRegion(meridian, axis, opts);
  if (R.reason) return noAnswer(R.reason);
  const full = revolveIntegral(R, null, opts);
  if (fraction === 0) return { value: R.hLo, exact: full.exact, gap: 0, reason: null };
  if (fraction === 1) return { value: R.hHi, exact: full.exact, gap: 0, reason: null };
  const want = fraction * full.value;
  let lo = R.hLo, hi = R.hHi, exact = full.exact;
  const stop = 1e-12 * Math.max(1, Math.abs(R.hHi - R.hLo));
  for (let it = 0; it < 200 && hi - lo > stop; it += 1) {
    const m = (lo + hi) / 2;
    const v = revolveIntegral(R, m, opts);
    exact = exact && v.exact;
    if (v.value < want) lo = m; else hi = m;
  }
  return { value: (lo + hi) / 2, exact, gap: hi - lo, reason: null };
}
