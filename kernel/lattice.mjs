// Lattice: a trivariate B-spline free-form deformation whose point count in
// each direction can change without moving the shape it holds.
//
// A lattice is a box frame (origin, three orthonormal axes, a size along each)
// and a tensor grid of control points with, per direction, a count n >= 2, a
// degree p = min(3, n - 1) and a clamped knot vector over [0, 1]. A point is
// deformed by taking its box coordinates (s, t, u) in [0, 1]^3 and evaluating
// the volume there.
//
// Identity: a rest lattice puts each control point at the Greville abscissae
// of its knots, and a clamped B-spline reproduces a linear function from its
// Greville points, so an unmoved lattice maps every point inside the box to
// itself at every count and degree.
//
// Raising a count is exact: while the new count is at most 4 the direction is
// a Bezier and is degree-elevated; past that it stays cubic and one knot is
// inserted at the middle of the widest span. Both leave the volume unchanged.
// Lowering a count is a least-squares refit, end points held, over each
// candidate space one step coarser (each interior knot removed in turn, or one
// Bezier degree down); the candidate that moves the volume least is kept and
// its drift reported.
//
// Points outside the box take the displacement at the nearest box point,
// faded to zero across a margin of 10% of the box per axis. A lattice marked
// `outside: 'extrapolate'` instead extends its end polynomial pieces past the
// box before the same fade, which is what the earlier per-surface Bernstein
// bands did; latticeFromLegacyBand converts one of those exactly.
//
// Pure: plain arrays in and out, no imports.

export const LATTICE_MIN_COUNT = 2;
export const LATTICE_MAX_COUNT = 12;
const OUTSIDE_MARGIN = 0.1;
const DRIFT_GRID = 9;

function v3add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function v3sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function v3scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function v3dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function v3len(a) { return Math.sqrt(v3dot(a, a)); }
function v3cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function v3norm(a) { const l = v3len(a); return l > 1e-300 ? v3scale(a, 1 / l) : [0, 0, 0]; }

export function latticeDegreeFor(n) { return Math.min(3, n - 1); }

// Clamped knots with the interior spaced evenly.
export function uniformClampedKnots(n, p) {
  const knots = [];
  for (let i = 0; i <= p; i++) knots.push(0);
  const interior = n - p - 1;
  for (let i = 1; i <= interior; i++) knots.push(i / (interior + 1));
  for (let i = 0; i <= p; i++) knots.push(1);
  return knots;
}

export function grevilleAbscissae(knots, p, n) {
  const g = [];
  for (let i = 0; i < n; i++) {
    if (p === 0) { g.push(knots[i]); continue; }
    let s = 0;
    for (let k = 1; k <= p; k++) s += knots[i + k];
    g.push(s / p);
  }
  return g;
}

// All n basis values at x (Cox-de Boor on the clamped knots). x is clamped to
// [0, 1] unless `extend`, which evaluates the end span's polynomial past it.
export function bsplineBasis(knots, p, n, x, extend = false) {
  const clamped = Math.min(1, Math.max(0, x));
  const xs = extend ? x : clamped;
  let span = p;
  if (clamped >= knots[n]) span = n - 1;
  else while (span < n - 1 && clamped >= knots[span + 1]) span++;
  const N = new Array(p + 1).fill(0);
  N[0] = 1;
  const left = new Array(p + 1).fill(0), right = new Array(p + 1).fill(0);
  for (let j = 1; j <= p; j++) {
    left[j] = xs - knots[span + 1 - j];
    right[j] = knots[span + j] - xs;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const den = right[r + 1] + left[j - r];
      const temp = den !== 0 ? N[r] / den : 0;
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    N[j] = saved;
  }
  const out = new Array(n).fill(0);
  for (let r = 0; r <= p; r++) out[span - p + r] = N[r];
  return out;
}

// Symmetric 3x3 eigenvectors by Jacobi rotation, largest eigenvalue first.
function symmetricEigen3(m) {
  const a = m.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-15) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const pairs = [0, 1, 2].map((i) => ({ val: a[i][i], vec: [v[0][i], v[1][i], v[2][i]] }));
  pairs.sort((x, y) => y.val - x.val);
  return pairs;
}

// The box that holds `points`, padded by `pad` of its size per axis (a flat
// axis gets an absolute floor). 'world' keeps the world axes; 'part' takes the
// principal axes of the points, right-handed, longest first.
export function latticeFrame(points, mode = 'world', pad = 0.12) {
  if (!points.length) throw new Error('latticeFrame: no points');
  let axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  if (mode === 'part' && points.length >= 3) {
    const c = [0, 0, 0];
    for (const p of points) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
    c[0] /= points.length; c[1] /= points.length; c[2] /= points.length;
    const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const p of points) {
      const d = v3sub(p, c);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += d[i] * d[j];
    }
    const eig = symmetricEigen3(cov);
    const a0 = v3norm(eig[0].vec);
    let a1 = v3norm(v3sub(eig[1].vec, v3scale(a0, v3dot(eig[1].vec, a0))));
    if (v3len(a1) < 0.5) a1 = v3norm(v3cross(a0, Math.abs(a0[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
    const a2 = v3norm(v3cross(a0, a1));
    if (v3len(a2) > 0.5) axes = [a0, a1, a2];
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let d = 0; d < 3; d++) {
      const x = v3dot(p, axes[d]);
      if (x < lo[d]) lo[d] = x;
      if (x > hi[d]) hi[d] = x;
    }
  }
  const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const big = Math.max(ext[0], ext[1], ext[2], 1e-6);
  const size = [], start = [];
  for (let d = 0; d < 3; d++) {
    const padD = Math.max(ext[d] * pad, big * 0.02, 1e-3);
    start.push(lo[d] - padD);
    size.push(ext[d] + 2 * padD);
  }
  let origin = [0, 0, 0];
  for (let d = 0; d < 3; d++) origin = v3add(origin, v3scale(axes[d], start[d]));
  return { origin, axes, size };
}

export function latticeIndex(lattice, i, j, k) {
  const [, nv, nw] = lattice.counts;
  return (i * nv + j) * nw + k;
}

// Where the frame puts box coordinates (s, t, u).
export function latticeFramePoint(lattice, s, t, u) {
  const { origin, axes, size } = lattice;
  let p = origin.slice();
  p = v3add(p, v3scale(axes[0], s * size[0]));
  p = v3add(p, v3scale(axes[1], t * size[1]));
  p = v3add(p, v3scale(axes[2], u * size[2]));
  return p;
}

// A rest lattice over `frame` with `counts` points per direction.
export function makeLattice(frame, counts) {
  const c = counts.map((n) => Math.max(LATTICE_MIN_COUNT, Math.min(LATTICE_MAX_COUNT, Math.round(n))));
  const degrees = c.map(latticeDegreeFor);
  const knots = c.map((n, d) => uniformClampedKnots(n, degrees[d]));
  const lattice = {
    origin: frame.origin.slice(), axes: frame.axes.map((a) => a.slice()), size: frame.size.slice(),
    counts: c, degrees, knots, pts: [],
  };
  lattice.pts = restPoints(lattice);
  return lattice;
}

function restPoints(lattice) {
  const [nu, nv, nw] = lattice.counts;
  const g = lattice.counts.map((n, d) => grevilleAbscissae(lattice.knots[d], lattice.degrees[d], n));
  const pts = [];
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) for (let k = 0; k < nw; k++) {
    pts.push(latticeFramePoint(lattice, g[0][i], g[1][j], g[2][k]));
  }
  return pts;
}

export function latticeRestPoints(lattice) { return restPoints(lattice); }

export function cloneLattice(lattice) {
  return {
    origin: lattice.origin.slice(), axes: lattice.axes.map((a) => a.slice()), size: lattice.size.slice(),
    counts: lattice.counts.slice(), degrees: lattice.degrees.slice(), knots: lattice.knots.map((k) => k.slice()),
    pts: lattice.pts.map((p) => p.slice()),
    ...(lattice.outside ? { outside: lattice.outside } : {}),
  };
}

// Every control point still at its rest place (exact comparison).
export function latticeIsRest(lattice) {
  const rest = restPoints(lattice);
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i], b = lattice.pts[i];
    if (a[0] !== b[0] || a[1] !== b[1] || a[2] !== b[2]) return false;
  }
  return true;
}

export function latticeLocal(lattice, p) {
  const d = v3sub(p, lattice.origin);
  return [0, 1, 2].map((a) => v3dot(d, lattice.axes[a]) / (lattice.size[a] > 1e-12 ? lattice.size[a] : 1e-12));
}

function evalWithBases(lattice, bu, bv, bw) {
  const [nu, nv, nw] = lattice.counts;
  const pts = lattice.pts;
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < nu; i++) {
    const a = bu[i];
    if (a === 0) continue;
    for (let j = 0; j < nv; j++) {
      const ab = a * bv[j];
      if (ab === 0) continue;
      const base = (i * nv + j) * nw;
      for (let k = 0; k < nw; k++) {
        const w = ab * bw[k];
        if (w === 0) continue;
        const q = pts[base + k];
        x += w * q[0]; y += w * q[1]; z += w * q[2];
      }
    }
  }
  return [x, y, z];
}

export function latticeEval(lattice, s, t, u, extend = false) {
  const [nu, nv, nw] = lattice.counts;
  return evalWithBases(lattice,
    bsplineBasis(lattice.knots[0], lattice.degrees[0], nu, s, extend),
    bsplineBasis(lattice.knots[1], lattice.degrees[1], nv, t, extend),
    bsplineBasis(lattice.knots[2], lattice.degrees[2], nw, u, extend));
}

function smoothstep(t) { const c = Math.max(0, Math.min(1, t)); return c * c * (3 - 2 * c); }
function outsideWeight(c) {
  if (c >= 0 && c <= 1) return 1;
  const d = c < 0 ? -c : c - 1;
  return d >= OUTSIDE_MARGIN ? 0 : 1 - smoothstep(d / OUTSIDE_MARGIN);
}

export function latticeDeformPoint(lattice, p) {
  const st = latticeLocal(lattice, p);
  const inside = st.every((c) => c >= 0 && c <= 1);
  if (inside) return latticeEval(lattice, st[0], st[1], st[2]);
  const w = outsideWeight(st[0]) * outsideWeight(st[1]) * outsideWeight(st[2]);
  if (w <= 0) return p.slice();
  if (lattice.outside === 'extrapolate') {
    // The rest lattice is the identity polynomial, so its extension at p is p.
    return v3add(p, v3scale(v3sub(latticeEval(lattice, st[0], st[1], st[2], true), p), w));
  }
  const cl = st.map((c) => Math.min(1, Math.max(0, c)));
  const moved = latticeEval(lattice, cl[0], cl[1], cl[2]);
  const rest = latticeFramePoint(lattice, cl[0], cl[1], cl[2]);
  return v3add(p, v3scale(v3sub(moved, rest), w));
}

export function latticeDeformPoints(lattice, points) {
  return points.map((p) => latticeDeformPoint(lattice, p));
}

// Several lattices at once: their displacements add.
export function latticeDeformPointsMany(lattices, points) {
  if (lattices.length === 1) return latticeDeformPoints(lattices[0], points);
  return points.map((p) => {
    let out = p.slice();
    for (const L of lattices) out = v3add(out, v3sub(latticeDeformPoint(L, p), p));
    return out;
  });
}

// One earlier per-surface band ({min, max, densityU/V/W, ctrlPts[i][j][k]}, a
// Bernstein lattice of degree density - 1 on an axis-aligned box) as a lattice
// that deforms every point the same way.
export function latticeFromLegacyBand(band) {
  const counts = [band.densityU, band.densityV, band.densityW];
  const degrees = counts.map((n) => n - 1);
  const pts = [];
  for (let i = 0; i < counts[0]; i++) for (let j = 0; j < counts[1]; j++) for (let k = 0; k < counts[2]; k++) {
    pts.push(band.ctrlPts[i][j][k].slice(0, 3));
  }
  return {
    origin: band.min.slice(), axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    size: [band.max[0] - band.min[0], band.max[1] - band.min[1], band.max[2] - band.min[2]],
    counts, degrees, knots: counts.map((n, d) => uniformClampedKnots(n, degrees[d])),
    pts, outside: 'extrapolate',
  };
}

// The lattice carried through an affine map given as a column-major 4x4 (a
// rotation and a scale along the box's own axes keep it a box).
export function latticeTransformed(lattice, e) {
  const pt = (q) => [e[0] * q[0] + e[4] * q[1] + e[8] * q[2] + e[12], e[1] * q[0] + e[5] * q[1] + e[9] * q[2] + e[13], e[2] * q[0] + e[6] * q[1] + e[10] * q[2] + e[14]];
  const dir = (a) => [e[0] * a[0] + e[4] * a[1] + e[8] * a[2], e[1] * a[0] + e[5] * a[1] + e[9] * a[2], e[2] * a[0] + e[6] * a[1] + e[10] * a[2]];
  const out = cloneLattice(lattice);
  out.origin = pt(lattice.origin);
  out.axes = lattice.axes.map((a) => v3norm(dir(a)));
  out.size = lattice.size.map((s, d) => s * v3len(dir(lattice.axes[d])));
  out.pts = lattice.pts.map(pt);
  return out;
}

// One row of control points along direction `axis`, at fixed other indices.
function rowsAlong(lattice, axis) {
  const [nu, nv, nw] = lattice.counts;
  const n = lattice.counts[axis];
  const rows = [];
  const others = [[nv, nw], [nu, nw], [nu, nv]][axis];
  for (let a = 0; a < others[0]; a++) for (let b = 0; b < others[1]; b++) {
    const row = [];
    for (let r = 0; r < n; r++) {
      const ijk = axis === 0 ? [r, a, b] : axis === 1 ? [a, r, b] : [a, b, r];
      row.push(lattice.pts[latticeIndex(lattice, ijk[0], ijk[1], ijk[2])]);
    }
    rows.push({ a, b, row });
  }
  return rows;
}

function withRows(lattice, axis, n, degree, knots, rowMap) {
  const out = cloneLattice(lattice);
  out.counts[axis] = n;
  out.degrees[axis] = degree;
  out.knots[axis] = knots.slice();
  const total = out.counts[0] * out.counts[1] * out.counts[2];
  out.pts = new Array(total);
  for (const { a, b, row } of rowsAlong(lattice, axis)) {
    const next = rowMap(row);
    for (let r = 0; r < n; r++) {
      const ijk = axis === 0 ? [r, a, b] : axis === 1 ? [a, r, b] : [a, b, r];
      out.pts[latticeIndex(out, ijk[0], ijk[1], ijk[2])] = next[r];
    }
  }
  return out;
}

// Bezier degree p -> p + 1 on one row.
function elevateRow(row) {
  const p = row.length - 1;
  const out = [row[0].slice()];
  for (let i = 1; i <= p; i++) {
    const a = i / (p + 1);
    out.push(v3add(v3scale(row[i - 1], a), v3scale(row[i], 1 - a)));
  }
  out.push(row[p].slice());
  return out;
}

// Boehm insertion of x into a row of degree p over `knots`.
function insertKnotRow(row, knots, p, x) {
  let k = p;
  while (k < knots.length - p - 2 && x >= knots[k + 1]) k++;
  const out = [];
  for (let i = 0; i <= row.length; i++) {
    if (i <= k - p) out.push(row[i].slice());
    else if (i >= k + 1) out.push(row[i - 1].slice());
    else {
      const alpha = (x - knots[i]) / (knots[i + p] - knots[i]);
      out.push(v3add(v3scale(row[i], alpha), v3scale(row[i - 1], 1 - alpha)));
    }
  }
  return out;
}

function widestSpanMid(knots) {
  let best = 0, at = 0.5;
  for (let i = 0; i < knots.length - 1; i++) {
    const w = knots[i + 1] - knots[i];
    if (w > best + 1e-15) { best = w; at = (knots[i] + knots[i + 1]) / 2; }
  }
  return at;
}

function raiseOne(lattice, axis) {
  const n = lattice.counts[axis], p = lattice.degrees[axis];
  if (n + 1 <= 4 && p === n - 1) {
    return withRows(lattice, axis, n + 1, p + 1, uniformClampedKnots(n + 1, p + 1), elevateRow);
  }
  const knots = lattice.knots[axis];
  const x = widestSpanMid(knots);
  const next = [...knots, x].sort((a, b) => a - b);
  return withRows(lattice, axis, n + 1, p, next, (row) => insertKnotRow(row, knots, p, x));
}

// Solves the small dense system A x = b in place (partial pivoting).
function solveDense(A, b) {
  const n = A.length;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
    const d = A[c][c];
    if (Math.abs(d) < 1e-300) return null;
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / d;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] = v3sub(b[r], v3scale(b[c], f));
    }
  }
  const x = new Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let acc = b[r].slice();
    for (let k = r + 1; k < n; k++) acc = v3sub(acc, v3scale(x[k], A[r][k]));
    x[r] = v3scale(acc, 1 / A[r][r]);
  }
  return x;
}

// Least-squares fit of every row onto (n, p, knots), end points held.
function refitAxis(lattice, axis, n, p, knots) {
  const oldN = lattice.counts[axis], oldP = lattice.degrees[axis], oldK = lattice.knots[axis];
  const M = Math.max(8 * Math.max(n, oldN), 48);
  const samples = [];
  for (let m = 0; m < M; m++) samples.push(m / (M - 1));
  const Bold = samples.map((x) => bsplineBasis(oldK, oldP, oldN, x));
  const Bnew = samples.map((x) => bsplineBasis(knots, p, n, x));
  const inner = n - 2;
  return withRows(lattice, axis, n, p, knots, (row) => {
    const target = Bold.map((bw) => {
      let acc = [0, 0, 0];
      for (let i = 0; i < oldN; i++) if (bw[i]) acc = v3add(acc, v3scale(row[i], bw[i]));
      return acc;
    });
    const first = row[0].slice(), last = row[oldN - 1].slice();
    if (inner <= 0) return [first, last];
    const A = Array.from({ length: inner }, () => new Array(inner).fill(0));
    const rhs = Array.from({ length: inner }, () => [0, 0, 0]);
    for (let m = 0; m < M; m++) {
      const bw = Bnew[m];
      const resid = v3sub(v3sub(target[m], v3scale(first, bw[0])), v3scale(last, bw[n - 1]));
      for (let r = 0; r < inner; r++) {
        const br = bw[r + 1];
        if (!br) continue;
        rhs[r] = v3add(rhs[r], v3scale(resid, br));
        for (let c = 0; c < inner; c++) A[r][c] += br * bw[c + 1];
      }
    }
    const x = solveDense(A, rhs);
    if (!x) return null;
    return [first, ...x, last];
  });
}

// The largest distance between two lattices' volumes on a sample grid.
export function latticeDrift(a, b) {
  let worst = 0;
  for (let i = 0; i < DRIFT_GRID; i++) for (let j = 0; j < DRIFT_GRID; j++) for (let k = 0; k < DRIFT_GRID; k++) {
    const s = i / (DRIFT_GRID - 1), t = j / (DRIFT_GRID - 1), u = k / (DRIFT_GRID - 1);
    const d = v3len(v3sub(latticeEval(a, s, t, u), latticeEval(b, s, t, u)));
    if (d > worst) worst = d;
  }
  return worst;
}

function lowerOne(lattice, axis) {
  const n = lattice.counts[axis], p = lattice.degrees[axis];
  const target = n - 1;
  const candidates = [];
  if (target <= 4) {
    const q = latticeDegreeFor(target);
    candidates.push({ p: q, knots: uniformClampedKnots(target, q) });
  } else {
    const knots = lattice.knots[axis];
    for (let i = p + 1; i < knots.length - p - 1; i++) {
      candidates.push({ p, knots: knots.filter((_, k) => k !== i) });
    }
    // A Bezier of high degree (an earlier band) has no knot to remove: one degree down.
    if (!candidates.length) candidates.push({ p: target - 1, knots: uniformClampedKnots(target, target - 1) });
  }
  let best = null;
  for (const c of candidates) {
    let fit;
    try { fit = refitAxis(lattice, axis, target, c.p, c.knots); } catch (e) { fit = null; }
    if (!fit || fit.pts.some((q) => !q || !q.every(Number.isFinite))) continue;
    const drift = latticeDrift(lattice, fit);
    if (!best || drift < best.drift) best = { lattice: fit, drift };
  }
  return best;
}

// Sets the count along `axis` (0, 1, 2) to `n`. Up is exact; down returns the
// refit and how far it moved the volume (`drift`, in the lattice's units).
export function latticeSetCount(lattice, axis, n) {
  const want = Math.max(LATTICE_MIN_COUNT, Math.min(LATTICE_MAX_COUNT, Math.round(n)));
  let cur = cloneLattice(lattice);
  if (want === cur.counts[axis]) return { lattice: cur, exact: true, drift: 0 };
  if (want > cur.counts[axis]) {
    while (cur.counts[axis] < want) cur = raiseOne(cur, axis);
    return { lattice: cur, exact: true, drift: 0 };
  }
  while (cur.counts[axis] > want) {
    const step = lowerOne(cur, axis);
    if (!step) return { lattice: null, exact: false, drift: null };
    cur = step.lattice;
  }
  return { lattice: cur, exact: false, drift: latticeDrift(lattice, cur) };
}

// A lattice whose points are an affine image of `lattice`'s: the frame is kept,
// so every deformed point moves by the same affine map.
export function latticeMapPoints(lattice, fn) {
  const out = cloneLattice(lattice);
  out.pts = out.pts.map((p) => fn(p.slice()));
  return out;
}
