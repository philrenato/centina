// Revolve along rails. The profile turns about the axis as Revolve turns it,
// and at every angle each rail sets how far out (and how high) the profile sits
// there. Rhino's RailRevolve takes one rail; this takes any number.
//
// Cylindrical coordinates about the axis: the start half-plane (angle 0) holds
// the profile's control-point centroid. A rail is read as r(θ), z(θ) by sampling
// it and unwrapping its angle; where the rail does not reach an angle its
// nearest end holds. Each rail grips the profile at the profile point nearest
// its own crossing of the start half-plane, and at every angle that point is
// put on the rail: the profile scales out from the axis by r_rail(θ)/r_grip and
// rises by z_rail(θ) - z_grip. A rail that does not touch the profile pulls it
// on from the start. Between grips the ratio and rise blend linearly along the
// profile's parameter, and past the outermost grips they hold.
//
// The surface is skinned through sections at even angles: each profile control
// point traces a cubic interpolated at those angles, with one knot vector for
// all of them, so the result is one tensor-product surface. Profile weights ride
// along unchanged, so every section is the profile's own rational shape. A full
// turn closes with matching end derivatives (a central difference across the
// seam), so the seam is not a crease.
//
// The surface's U is the profile and its V the turn, as Revolve's.

import { curvePoint } from './curve.mjs';
import { solveLinearSystem, averagingKnotVector, interpAtParams } from './interpolate.mjs';
import { findSpan, basisFuns, dersBasisFuns } from './basis.mjs';
import { anyPerpendicular } from './vec3.mjs';

const TWO_PI = Math.PI * 2;
// One section per this many radians, between the bounds below.
export const RAIL_REVOLVE_SECTION_STEP = Math.PI / 36;
export const RAIL_REVOLVE_MIN_SECTIONS = 4;
export const RAIL_REVOLVE_MAX_SECTIONS = 180;
const RAIL_SAMPLES_MIN = 96;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : null; };

// The frame: axis D, e0 toward the profile's centroid, e1 = D x e0.
export function railRevolveFrame(profile, axisPoint, axisDir) {
  const D = norm(axisDir);
  if (!D) return null;
  const pts = profile.ctrlPts;
  const c = [0, 0, 0];
  for (const p of pts) { c[0] += p[0] / pts.length; c[1] += p[1] / pts.length; c[2] += p[2] / pts.length; }
  const rel = sub(c, axisPoint);
  const X = sub(rel, [D[0] * dot(rel, D), D[1] * dot(rel, D), D[2] * dot(rel, D)]);
  const e0 = norm(X) || anyPerpendicular(D);
  return { A: axisPoint, D, e0, e1: cross(D, e0) };
}

function cylOf(frame, p) {
  const rel = sub(p, frame.A);
  const x = dot(rel, frame.e0), y = dot(rel, frame.e1);
  return { theta: Math.atan2(y, x), r: Math.hypot(x, y), z: dot(rel, frame.D) };
}

// A rail as r(θ), z(θ): samples with unwrapped angle, sorted by angle, shifted
// by whole turns so angle 0 falls inside the rail's range where it can.
export function railTable(rail, frame) {
  const k = rail.knots, u0 = k[0], u1 = k[k.length - 1];
  const spans = Math.max(1, new Set(k).size - 1);
  const n = Math.max(RAIL_SAMPLES_MIN, spans * 16);
  const rows = [];
  let prev = null, turn = 0;
  for (let i = 0; i <= n; i += 1) {
    const c = cylOf(frame, curvePoint(rail, u0 + ((u1 - u0) * i) / n));
    if (prev != null) {
      let d = c.theta + turn - prev;
      while (d > Math.PI) { turn -= TWO_PI; d -= TWO_PI; }
      while (d < -Math.PI) { turn += TWO_PI; d += TWO_PI; }
    }
    const t = c.theta + turn;
    rows.push({ t, r: c.r, z: c.z });
    prev = t;
  }
  rows.sort((a, b) => a.t - b.t);
  const lo = rows[0].t, hi = rows[rows.length - 1].t;
  const shift = lo > 0 ? -Math.ceil((lo - 1e-9) / TWO_PI) * TWO_PI : hi < 0 ? Math.ceil(-hi / TWO_PI - 1e-9) * TWO_PI : 0;
  if (shift) for (const row of rows) row.t += shift;
  return { rows, lo: rows[0].t, hi: rows[rows.length - 1].t, full: rows[rows.length - 1].t - rows[0].t >= TWO_PI - 1e-6 };
}

// r and z of a rail at angle θ; a rail making a full turn repeats.
export function railAt(table, theta) {
  const { rows, lo, hi, full } = table;
  let t = theta;
  if (full) t = lo + ((((t - lo) % TWO_PI) + TWO_PI) % TWO_PI);
  if (t <= lo) return { r: rows[0].r, z: rows[0].z };
  if (t >= hi) return { r: rows[rows.length - 1].r, z: rows[rows.length - 1].z };
  let a = 0, b = rows.length - 1;
  while (b - a > 1) { const m = (a + b) >> 1; if (rows[m].t <= t) a = m; else b = m; }
  const s = rows[b].t > rows[a].t ? (t - rows[a].t) / (rows[b].t - rows[a].t) : 0;
  return { r: rows[a].r + (rows[b].r - rows[a].r) * s, z: rows[a].z + (rows[b].z - rows[a].z) * s };
}

// Where a rail grips the profile: the profile parameter nearest, in (r, z), to
// the rail's crossing of the start half-plane.
function gripParam(profile, frame, at0) {
  const k = profile.knots, u0 = k[0], u1 = k[k.length - 1];
  const n = 256;
  let best = u0, bestD = Infinity;
  for (let i = 0; i <= n; i += 1) {
    const u = u0 + ((u1 - u0) * i) / n;
    const c = cylOf(frame, curvePoint(profile, u));
    const d = Math.hypot(c.r - at0.r, c.z - at0.z);
    if (d < bestD) { bestD = d; best = u; }
  }
  return best;
}

function greville(crv) {
  const p = crv.degree, k = crv.knots;
  return crv.ctrlPts.map((_, i) => { let s = 0; for (let j = 1; j <= p; j += 1) s += k[i + j]; return p > 0 ? s / p : k[i]; });
}

// Ratio and rise at each control point for one angle, blended between grips.
function blendAt(grips, g) {
  if (g <= grips[0].u) return grips[0];
  const last = grips[grips.length - 1];
  if (g >= last.u) return last;
  for (let i = 0; i + 1 < grips.length; i += 1) {
    const a = grips[i], b = grips[i + 1];
    if (g <= b.u) { const s = b.u > a.u ? (g - a.u) / (b.u - a.u) : 0; return { s: a.s + (b.s - a.s) * s, dz: a.dz + (b.dz - a.dz) * s }; }
  }
  return last;
}

// One knot vector for every row: cubic through rows sampled at `ubar`, with
// end derivatives (P&T 9.2.2, interior knots at the interior parameters).
function interpRowsWithEndDerivs(rows, ubar, d0s, dns) {
  const n = ubar.length - 1, p = 3, N = n + 3;
  const U = [];
  for (let i = 0; i <= p; i += 1) U.push(ubar[0]);
  for (let i = 1; i <= n - 1; i += 1) U.push(ubar[i]);
  for (let i = 0; i <= p; i += 1) U.push(ubar[n]);
  const A = Array.from({ length: N }, () => new Array(N).fill(0));
  const put = (row, u, der) => {
    const span = findSpan(N - 1, p, u, U);
    const vals = der ? dersBasisFuns(span, u, p, 1, U)[1] : basisFuns(span, u, p, U);
    for (let i = 0; i <= p; i += 1) A[row][span - p + i] = vals[i];
  };
  put(0, ubar[0], true);
  for (let k = 0; k <= n; k += 1) put(k + 1, ubar[k], false);
  put(n + 2, ubar[n], true);
  const cols = [];
  rows.forEach((pts, i) => {
    for (let c = 0; c < 3; c += 1) cols.push([d0s[i][c], ...pts.map((q) => q[c]), dns[i][c]]);
  });
  const sol = solveLinearSystem(A, cols);
  return { knots: U, rows: rows.map((_, i) => sol[3 * i].map((x, j) => [x, sol[3 * i + 1][j], sol[3 * i + 2][j]])) };
}

/**
 * railRevolve(profile, rails, axisPoint, axisDir, sweep, opts)
 * profile, rails: NURBS curves { degree, knots, ctrlPts:[x,y,z,w] }.
 * sweep: radians, signed. opts.sections overrides the section count.
 * Returns { ok, srf, sections, grips } or { ok: false, reason }.
 */
export function railRevolve(profile, rails, axisPoint, axisDir, sweep, opts = {}) {
  const frame = railRevolveFrame(profile, axisPoint, axisDir);
  if (!frame) return { ok: false, reason: 'the axis has no direction' };
  if (!(Math.abs(sweep) > 1e-9)) return { ok: false, reason: 'the angle is zero' };
  const closed = Math.abs(sweep) >= TWO_PI - 1e-9;
  const M = opts.sections ?? Math.min(RAIL_REVOLVE_MAX_SECTIONS, Math.max(RAIL_REVOLVE_MIN_SECTIONS, Math.ceil(Math.abs(sweep) / RAIL_REVOLVE_SECTION_STEP)));
  const tables = (rails || []).map((r) => railTable(r, frame));
  const grips0 = tables.map((tb) => {
    const u = gripParam(profile, frame, railAt(tb, 0));
    return { tb, u, at0: cylOf(frame, curvePoint(profile, u)) };
  }).sort((a, b) => a.u - b.u);
  const local = profile.ctrlPts.map((p) => { const c = cylOf(frame, [p[0], p[1], p[2]]); return { r: c.r, phi: c.theta, z: c.z, w: p[3] ?? 1 }; });
  const g = greville(profile);
  const count = closed ? M : M + 1;
  const thetas = Array.from({ length: count }, (_, j) => (sweep * j) / M);
  const sections = thetas.map((th) => {
    const grips = grips0.map((q) => {
      const at = railAt(q.tb, th);
      return { u: q.u, s: q.at0.r > 1e-9 ? at.r / q.at0.r : 1, dz: at.z - q.at0.z };
    });
    return local.map((c, i) => {
      const f = grips.length ? blendAt(grips, g[i]) : { s: 1, dz: 0 };
      const r = c.r * f.s, z = c.z + f.dz, a = c.phi + th;
      const ca = Math.cos(a) * r, sa = Math.sin(a) * r;
      return [frame.A[0] + frame.D[0] * z + frame.e0[0] * ca + frame.e1[0] * sa,
        frame.A[1] + frame.D[1] * z + frame.e0[1] * ca + frame.e1[1] * sa,
        frame.A[2] + frame.D[2] * z + frame.e0[2] * ca + frame.e1[2] * sa];
    });
  });
  // rows[i]: control point i's path across the sections.
  const rows = local.map((_, i) => sections.map((s) => s[i]));
  let knotsV, netRows;
  if (closed) {
    const ubar = Array.from({ length: M + 1 }, (_, j) => j / M);
    const closedRows = rows.map((r) => [...r, r[0]]);
    const ders = rows.map((r) => { const a = r[1], b = r[M - 1]; return [(a[0] - b[0]) * M / 2, (a[1] - b[1]) * M / 2, (a[2] - b[2]) * M / 2]; });
    const sol = interpRowsWithEndDerivs(closedRows, ubar, ders, ders);
    knotsV = sol.knots; netRows = sol.rows;
  } else {
    const ubar = Array.from({ length: M + 1 }, (_, j) => j / M);
    const p = Math.min(3, M);
    knotsV = averagingKnotVector(ubar, p);
    netRows = rows.map((r) => interpAtParams(r, p, ubar, knotsV).map((q) => [q[0], q[1], q[2]]));
  }
  const ctrlNet = netRows.map((row, i) => row.map((q) => [q[0], q[1], q[2], local[i].w]));
  const srf = { degU: profile.degree, knotsU: profile.knots.slice(), degV: closed ? 3 : Math.min(3, M), knotsV, ctrlNet };
  return { ok: true, srf, sections: thetas.length, closed, grips: grips0.map((q) => q.u) };
}
