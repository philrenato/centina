// Refit — recover a canonical shape's parameters from a point set, and
// report how well that shape fits.
//
// Switching an Ellipse from center-radius to two-point, and an edited
// Ellipse becoming something else, are the same operation: both re-derive
// a recipe's params from the object's current geometry, then swap which
// recipe drives it. This module is that re-derivation, plus the measured
// evidence a caller needs to decide whether the swap is valid.
//
// What this returns. Every fit returns the recovered params as plain data
// in the shape the app's operator entries use (`center`/`xAxis`/`yAxis`/`radiusX`/`radiusY`
// for Ellipse; `center`/`xAxis`/`yAxis`/`radius` for Circle) — plain
// number arrays, never a THREE.Vector3, so a recovered param bag can be
// handed straight to `evaluate()` or to a param editor with no
// translation step. It also returns a measured deviation in model
// units (worst and RMS), not a normalized 0-1 "confidence": a caller
// deciding what to call an object needs a distance it can display and
// compare against its own modeling tolerance. `segments` is never
// returned by any fit — that is a representation choice (how many exact
// rational arc spans the app chose to build the curve from), not a
// property of the shape.
//
// Refusals are return values, not throws — unlike this kernel's usual
// `throw`-on-refusal convention (offsetCurve2D, mergeLoopsKeyhole). Those
// functions are asked to build something and a refusal is exceptional. A
// fitter is asked a question ("is this still an
// ellipse?") whose answer is routinely no; making the ordinary negative
// answer an exception would push every caller into a try/catch around a
// non-exceptional case. Every result therefore carries `ok`, and a
// refusal carries a named `reason` (see FIT_REFUSAL) plus a plain-language
// `detail`.
//
// Input convention — points are Cartesian [x, y, z] (or {x, y, z}), never
// a NURBS control point's own homogeneous [x, y, z, w]. A rational curve's
// raw control points are not on the curve at all (a circle's own
// tangent-corner control points sit radius*sqrt(2) out), so a caller
// fitting a curve should pass sampled points (curvePoint / the app's own
// cached sample chains), or de-homogenized control points if it
// wants the control polygon. A 4th array element is ignored, not divided
// out — passing homogeneous points is a caller error this module cannot
// detect, and this comment is the warning.
//
// Cited methods
// - Ellipse: the direct least-squares ellipse-specific fit of Fitzgibbon,
//   Pilu & Fisher, "Direct Least Square Fitting of Ellipses," IEEE TPAMI
//   21(5), 1999, 476-480 — implemented in the numerically stable
//   reformulation of Halir & Flusser, "Numerically Stable Direct Least
//   Squares Fitting of Ellipses," Proc. WSCG'98, 125-132. Halir & Flusser
//   is what this module implements: Fitzgibbon's original solves a 6x6 generalized eigenproblem
//   whose scatter matrix is close to singular for real data, while the
//   Halir-Flusser block decomposition reduces it to a 3x3 problem on a
//   better-conditioned system. An ellipse-specific method is needed because
//   a plain algebraic conic fit minimizing the same residual can
//   return a hyperbola or parabola for points that are nearly-but-not-
//   quite an ellipse — the near-miss input this module exists to
//   judge. Fitzgibbon's constraint (4ac - b^2 = 1) makes an ellipse the
//   only reachable answer, by construction.
// - Circle: Kasa's algebraic circle fit (I. Kasa, "A circle fitting
//   procedure and its error analysis," IEEE Trans. Instrum. Meas. 25,
//   1976) as an initial guess, refined by Gauss-Newton on the true
//   orthogonal distance residual (r_i - R) so the returned answer is a
//   geometric least-squares circle, not the algebraically-biased
//   Kasa estimate (which is known to pull the radius short on partial
//   arcs).
// - Plane / line: principal component analysis of the point set's own
//   covariance matrix, eigen-decomposed by the classical cyclic Jacobi
//   rotation method for symmetric matrices. The best-fit plane's normal
//   is the eigenvector of the smallest eigenvalue; the best-fit line's
//   direction is the eigenvector of the largest. Both are the exact
//   orthogonal-distance least-squares answers, not approximations.
// - Point-to-ellipse distance: D. Eberly, "Distance from a Point to an
//   Ellipse, an Ellipsoid, or a Hyperellipsoid" (Geometric Tools) — a
//   bracketed bisection on a scalar whose bracketing function is provably
//   monotone, so it converges unconditionally with no derivative and no
//   starting guess. There is no closed form for this distance; the
//   reported ellipse deviation is exact to the stated bisection
//   tolerance, not in closed form.
//
// Limit of the ellipse fit. Fitzgibbon/Halir-
// Flusser minimizes an algebraic residual, not the geometric one — the
// returned ellipse is not guaranteed to be the orthogonal-distance
// optimum, and on a short arc the algebraic bias is well documented in
// the literature. What this module guarantees instead is that the
// reported deviation is the true geometric distance to the ellipse it
// returned. (The circle fit does not share this limit — its Gauss-Newton
// refinement converges on the geometric optimum directly, which is cheap
// for 3 unknowns; the equivalent for an ellipse is 5 coupled unknowns
// whose residual has no closed-form derivative, and is not attempted
// here.)

// Refusal vocabulary
export const FIT_REFUSAL = Object.freeze({
  // Fewer points than the fit structurally needs (line 2, plane 3,
  // circle 3, ellipse 5 — a conic has 5 degrees of freedom).
  TOO_FEW_POINTS: 'TOO_FEW_POINTS',
  // Enough points were passed, but too few distinct ones survive a
  // coincidence check — three copies of the same point do not determine
  // a plane no matter how they are counted.
  COINCIDENT_POINTS: 'COINCIDENT_POINTS',
  // A NaN/Infinity reached the input.
  NOT_FINITE: 'NOT_FINITE',
  // The points lie (within tolerance) on a single straight line. A plane
  // through a line is ambiguous; a circle or ellipse through one is not
  // determined at all.
  COLLINEAR: 'COLLINEAR',
  // The point set is non-planar beyond the caller's tolerance,
  // so no planar shape describes it. The measured planar deviation is
  // reported alongside, so a caller can loosen its own tolerance with a
  // number in hand.
  NOT_PLANAR: 'NOT_PLANAR',
  // The conic system came back without a usable ellipse solution (a
  // singular scatter matrix, or recovered semi-axes that are not real and
  // positive). Structurally rare, because the ellipse-specific constraint
  // is what rules out the hyperbola/parabola branches in the first place
  // — this is the last-resort guard for a numerically singular system.
  DEGENERATE_CONIC: 'DEGENERATE_CONIC',
});

// Small vector / matrix helpers (local, so this module has no imports;
// they operate only on the plain [x,y,z] arrays the fitters normalize
// their input into).
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function len3(a) { return Math.hypot(a[0], a[1], a[2]); }
function normalize3(a) {
  const l = len3(a);
  if (!(l > 1e-300) || !Number.isFinite(l)) return null;
  return [a[0] / l, a[1] / l, a[2] / l];
}
function addScaled3(o, d, s) { return [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s]; }

// Solve a 3x3 linear system by Gaussian elimination with partial
// pivoting. Returns null (never a garbage answer) when the matrix is
// singular to working precision.
function solve3(Ain, bin) {
  const A = [Ain[0].slice(), Ain[1].slice(), Ain[2].slice()];
  const b = bin.slice();
  for (let c = 0; c < 3; c++) {
    let piv = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (!(Math.abs(A[piv][c]) > 1e-300)) return null;
    if (piv !== c) { const t = A[piv]; A[piv] = A[c]; A[c] = t; const tb = b[piv]; b[piv] = b[c]; b[c] = tb; }
    for (let r = c + 1; r < 3; r++) {
      const f = A[r][c] / A[c][c];
      if (f === 0) continue;
      for (let k = c; k < 3; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const x = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < 3; k++) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
  return x.every(Number.isFinite) ? x : null;
}

function invert3(M) {
  const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const cols = [];
  for (let c = 0; c < 3; c++) {
    const x = solve3(M, [I[0][c], I[1][c], I[2][c]]);
    if (!x) return null;
    cols.push(x);
  }
  // cols[c] is the c-th column of the inverse.
  return [
    [cols[0][0], cols[1][0], cols[2][0]],
    [cols[0][1], cols[1][1], cols[2][1]],
    [cols[0][2], cols[1][2], cols[2][2]],
  ];
}

function matmul3(A, B) {
  const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += A[i][k] * B[k][j];
    R[i][j] = s;
  }
  return R;
}
function matvec3(A, v) {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
  ];
}

// Classical cyclic Jacobi eigen-decomposition of a symmetric 3x3 matrix.
// Returns eigenvalues ascending with their matching (unit) eigenvectors.
// Chosen over a closed-form cubic because Jacobi stays
// accurate for a nearly-degenerate spectrum, which is the case a
// near-planar or near-collinear point set produces.
// Exported because it is the only symmetric eigensolver in the kernel and the
// principal-axis frames built elsewhere need it.
export function jacobiEigenSym3(Ain) {
  const A = [Ain[0].slice(), Ain[1].slice(), Ain[2].slice()];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const pairs = [[0, 1], [0, 2], [1, 2]];
  for (let sweep = 0; sweep < 64; sweep++) {
    const off = Math.abs(A[0][1]) + Math.abs(A[0][2]) + Math.abs(A[1][2]);
    const scale = Math.abs(A[0][0]) + Math.abs(A[1][1]) + Math.abs(A[2][2]);
    if (off <= 1e-18 * scale || off === 0) break;
    for (const [p, q] of pairs) {
      const apq = A[p][q];
      if (Math.abs(apq) <= 1e-300) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * apq);
      const sgn = theta >= 0 ? 1 : -1;
      const t = sgn / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      const app = A[p][p], aqq = A[q][q];
      A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      A[p][q] = A[q][p] = 0;
      for (let r = 0; r < 3; r++) {
        if (r === p || r === q) continue;
        const arp = A[r][p], arq = A[r][q];
        A[r][p] = A[p][r] = c * arp - s * arq;
        A[r][q] = A[q][r] = s * arp + c * arq;
      }
      for (let r = 0; r < 3; r++) {
        const vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c * vrp - s * vrq;
        V[r][q] = s * vrp + c * vrq;
      }
    }
  }
  const out = [0, 1, 2].map((i) => ({
    value: A[i][i],
    vector: [V[0][i], V[1][i], V[2][i]],
  }));
  // Ascending by eigenvalue, with the original column index as a
  // deterministic tie-break so an exactly-degenerate spectrum (a sphere-
  // like covariance) cannot reorder between calls.
  out.forEach((e, i) => { e._i = i; });
  out.sort((a, b) => (a.value - b.value) || (a._i - b._i));
  return out.map((e) => ({ value: e.value, vector: e.vector }));
}

// Deterministic orientation
//
// An eigenvector's sign is mathematically arbitrary, and a plane's own
// in-plane basis is arbitrary too. A caller is going to put these numbers
// in a Properties panel, where an arbitrary flip between two calls reads
// as a bug. Both are therefore canonicalized here rather than left to
// whatever the solver happened to produce:
//
//   * a direction's sign is fixed so its largest-magnitude component is
//     positive, with the lowest index winning an exact tie;
//   * a plane's in-plane X axis is derived from the plane normal alone
//     (cross the normal with whichever world axis it is least aligned
//     with), never from the data's own in-plane spread — a data-derived
//     in-plane axis flips arbitrarily on a near-circular point set.
//
// Both rules still have a measure-zero flip surface (a direction whose
// largest component crosses zero); that is unavoidable for any sign
// convention.
function canonicalizeDirection(v) {
  const n = normalize3(v);
  if (!n) return null;
  let idx = 0, mx = Math.abs(n[0]);
  for (let i = 1; i < 3; i++) {
    const a = Math.abs(n[i]);
    if (a > mx + 1e-15) { mx = a; idx = i; }
  }
  return n[idx] < 0 ? [-n[0], -n[1], -n[2]] : n;
}

function deterministicPlaneBasis(normalRaw) {
  const normal = canonicalizeDirection(normalRaw);
  if (!normal) return null;
  let best = 0, bestAbs = Math.abs(normal[0]);
  for (let i = 1; i < 3; i++) {
    const a = Math.abs(normal[i]);
    if (a < bestAbs - 1e-15) { bestAbs = a; best = i; }
  }
  const e = [0, 0, 0];
  e[best] = 1;
  const xAxis = canonicalizeDirection(cross3(e, normal));
  if (!xAxis) return null;
  const yAxis = cross3(normal, xAxis);
  return { normal, xAxis, yAxis };
}

// Input normalization and shared stats
function toPoint(p) {
  if (Array.isArray(p)) {
    const x = p[0], y = p[1], z = p[2] ?? 0;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    return [x, y, z];
  }
  if (p && typeof p === 'object') {
    const x = p.x, y = p.y, z = p.z ?? 0;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    return [x, y, z];
  }
  return null;
}

function refuse(reason, detail, extra) {
  return Object.assign({ ok: false, reason, detail }, extra || {});
}

// Normalize the input point list once, and compute the scale-setting
// statistics every fit's tolerances key off. `extent` is the bounding-box
// diagonal — a single "how big is this thing" number that a
// relative tolerance can multiply without caring which axis the shape
// happens to lie along.
function prepare(pointsIn, minCount, opts) {
  if (!Array.isArray(pointsIn)) {
    return { bad: refuse(FIT_REFUSAL.TOO_FEW_POINTS, 'no point array was supplied') };
  }
  const pts = [];
  for (const raw of pointsIn) {
    const p = toPoint(raw);
    if (!p) {
      return { bad: refuse(FIT_REFUSAL.NOT_FINITE, 'a point was missing, malformed, or contained NaN/Infinity') };
    }
    pts.push(p);
  }
  if (pts.length < minCount) {
    return { bad: refuse(FIT_REFUSAL.TOO_FEW_POINTS, `${pts.length} point(s) supplied, ${minCount} needed`) };
  }
  let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  const centroid = [0, 0, 0];
  for (const p of pts) {
    for (let i = 0; i < 3; i++) {
      if (p[i] < min[i]) min[i] = p[i];
      if (p[i] > max[i]) max[i] = p[i];
      centroid[i] += p[i];
    }
  }
  for (let i = 0; i < 3; i++) centroid[i] /= pts.length;
  const extent = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);

  // Distinct-point count, so "5 points" that are really 2 points and
  // three duplicates refuses instead of fitting noise. O(n^2),
  // which is fine at the scale a recipe re-derivation ever runs at (a
  // curve's control points or a few hundred samples).
  const coincidentTol = Number.isFinite(opts.coincidentTol)
    ? opts.coincidentTol
    : Math.max(extent * 1e-9, 1e-12);
  const distinct = [];
  for (const p of pts) {
    let dup = false;
    for (const q of distinct) {
      if (Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) <= coincidentTol) { dup = true; break; }
    }
    if (!dup) distinct.push(p);
  }
  if (distinct.length < minCount) {
    return {
      bad: refuse(
        FIT_REFUSAL.COINCIDENT_POINTS,
        `only ${distinct.length} distinct point(s) among ${pts.length} supplied, ${minCount} needed`,
      ),
    };
  }
  return { pts, centroid, extent, distinctCount: distinct.length, coincidentTol };
}

function covariance3(pts, c) {
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (const p of pts) {
    const dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    xx += dx * dx; xy += dx * dy; xz += dx * dz;
    yy += dy * dy; yz += dy * dz; zz += dz * dz;
  }
  return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
}

function worstAndRms(devs) {
  let worst = 0, sum = 0;
  for (const d of devs) {
    if (d > worst) worst = d;
    sum += d * d;
  }
  return { worst, rms: Math.sqrt(sum / devs.length) };
}

// Line
// fitLine(points, opts) -> {
//   ok, point, direction, start, end, length, worst, rms, count
// }
// `point` is the centroid (a point on the fitted line);
// `start`/`end` are the two extreme projections along the direction, so a
// caller re-deriving a Line recipe (which is start+end, not
// point+direction) has them without re-projecting.
export function fitLine(pointsIn, opts = {}) {
  const prep = prepare(pointsIn, 2, opts);
  if (prep.bad) return prep.bad;
  const { pts, centroid, extent } = prep;

  const eig = jacobiEigenSym3(covariance3(pts, centroid));
  const direction = canonicalizeDirection(eig[2].vector);
  if (!direction) {
    return refuse(FIT_REFUSAL.COINCIDENT_POINTS, 'the points have no measurable spread in any direction');
  }
  const devs = [];
  let tMin = Infinity, tMax = -Infinity;
  for (const p of pts) {
    const rel = sub3(p, centroid);
    const t = dot3(rel, direction);
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
    const perp = [rel[0] - direction[0] * t, rel[1] - direction[1] * t, rel[2] - direction[2] * t];
    devs.push(len3(perp));
  }
  const { worst, rms } = worstAndRms(devs);
  return {
    ok: true,
    shape: 'line',
    point: centroid.slice(),
    direction,
    start: addScaled3(centroid, direction, tMin),
    end: addScaled3(centroid, direction, tMax),
    length: tMax - tMin,
    worst,
    rms,
    count: pts.length,
    extent,
  };
}

// Plane
// fitPlane(points, opts) -> {
//   ok, origin, normal, xAxis, yAxis, worst, rms, count
// }
// Refuses COLLINEAR: infinitely many planes contain a straight line, so
// a collinear point set has no best-fit plane, and returning whichever one
// the solver landed on would be a confident wrong answer.
export function fitPlane(pointsIn, opts = {}) {
  const prep = prepare(pointsIn, 3, opts);
  if (prep.bad) return prep.bad;
  const { pts, centroid, extent } = prep;

  const eig = jacobiEigenSym3(covariance3(pts, centroid));
  const n = pts.length;
  // RMS distance of the points from their own best-fit line. If that is
  // ~0 the set is a line, not a plane.
  const lineRms = Math.sqrt(Math.max(0, eig[0].value + eig[1].value) / n);
  const collinearTol = Number.isFinite(opts.collinearTol)
    ? opts.collinearTol
    : Math.max(extent * 1e-6, 1e-12);
  if (lineRms <= collinearTol) {
    return refuse(
      FIT_REFUSAL.COLLINEAR,
      `the points lie on a straight line (RMS off-line ${lineRms.toExponential(3)}, tolerance ${collinearTol.toExponential(3)}); a plane through a line is not determined`,
      { lineRms, collinearTol },
    );
  }
  const basis = deterministicPlaneBasis(eig[0].vector);
  if (!basis) {
    return refuse(FIT_REFUSAL.DEGENERATE_CONIC, 'the best-fit plane normal collapsed to zero length');
  }
  const devs = pts.map((p) => Math.abs(dot3(sub3(p, centroid), basis.normal)));
  const { worst, rms } = worstAndRms(devs);
  return {
    ok: true,
    shape: 'plane',
    origin: centroid.slice(),
    normal: basis.normal,
    xAxis: basis.xAxis,
    yAxis: basis.yAxis,
    worst,
    rms,
    count: pts.length,
    extent,
  };
}

// Shared front half of every planar fit: fit the plane, check the caller's
// planarity tolerance, and project into the plane's own deterministic 2D
// frame. Returns either a refusal or {plane, uv}.
function planarFrame(pointsIn, minCount, opts) {
  const prep = prepare(pointsIn, minCount, opts);
  if (prep.bad) return { bad: prep.bad };
  const plane = fitPlane(prep.pts, opts);
  if (!plane.ok) return { bad: plane };

  const planarTol = Number.isFinite(opts.planarTol)
    ? opts.planarTol
    : Math.max(prep.extent * 1e-6, 1e-12);
  if (plane.worst > planarTol) {
    return {
      bad: refuse(
        FIT_REFUSAL.NOT_PLANAR,
        `the points are not planar: worst deviation from the best-fit plane is ${plane.worst.toExponential(3)}, tolerance ${planarTol.toExponential(3)}`,
        { planeWorst: plane.worst, planeRms: plane.rms, planarTol },
      ),
    };
  }
  const uv = prep.pts.map((p) => {
    const rel = sub3(p, plane.origin);
    return [dot3(rel, plane.xAxis), dot3(rel, plane.yAxis), dot3(rel, plane.normal)];
  });
  return { prep, plane, uv, planarTol };
}

// Circle
// fitCircle(points, opts) -> {
//   ok, center, normal, xAxis, yAxis, radius,
//   worst, rms, planeWorst, planeRms, count
// }
// `center`/`xAxis`/`yAxis`/`radius` are exactly the Circle operator's own
// param shape, so the result feeds `makeCircle(center, xAxis, yAxis,
// radius)` directly. Deviation is the true 3D distance from each point to
// the circle curve — hypot(in-plane radial miss, out-of-plane miss) — not
// the in-plane radial miss alone, so a point floating above the plane is
// counted.
export function fitCircle(pointsIn, opts = {}) {
  const framed = planarFrame(pointsIn, 3, opts);
  if (framed.bad) return framed.bad;
  const { prep, plane, uv } = framed;

  // Kasa algebraic fit: x^2 + y^2 + D x + E y + F = 0, linear in (D,E,F).
  let Sx = 0, Sy = 0, Sxx = 0, Syy = 0, Sxy = 0, Sz = 0, Sxz = 0, Syz = 0;
  const n = uv.length;
  for (const [x, y] of uv) {
    const z = x * x + y * y;
    Sx += x; Sy += y; Sxx += x * x; Syy += y * y; Sxy += x * y;
    Sz += z; Sxz += x * z; Syz += y * z;
  }
  const A = [[Sxx, Sxy, Sx], [Sxy, Syy, Sy], [Sx, Sy, n]];
  const rhs = [-Sxz, -Syz, -Sz];
  const sol = solve3(A, rhs);
  if (!sol) {
    return refuse(FIT_REFUSAL.COLLINEAR, 'the circle system is singular — the points do not determine a circle');
  }
  let cx = -sol[0] / 2, cy = -sol[1] / 2;
  let r2 = cx * cx + cy * cy - sol[2];
  if (!(r2 > 0) || !Number.isFinite(r2)) {
    return refuse(FIT_REFUSAL.COLLINEAR, 'the algebraic circle fit produced a non-positive squared radius');
  }
  let R = Math.sqrt(r2);

  // Gauss-Newton on the true orthogonal residual f_i = r_i - R. Cheap for
  // 3 unknowns, and it makes the returned answer a
  // geometric least-squares circle rather than Kasa's biased estimate.
  for (let iter = 0; iter < 30; iter++) {
    let JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    let Jtf = [0, 0, 0];
    let anyDegenerate = false;
    for (const [x, y] of uv) {
      const dx = x - cx, dy = y - cy;
      const r = Math.hypot(dx, dy);
      if (!(r > 1e-300)) { anyDegenerate = true; break; }
      const f = r - R;
      const j = [-dx / r, -dy / r, -1];
      for (let a = 0; a < 3; a++) {
        Jtf[a] += j[a] * f;
        for (let b = 0; b < 3; b++) JtJ[a][b] += j[a] * j[b];
      }
    }
    if (anyDegenerate) break;
    const step = solve3(JtJ, [-Jtf[0], -Jtf[1], -Jtf[2]]);
    if (!step) break;
    cx += step[0]; cy += step[1]; R += step[2];
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(R) || R <= 0) {
      return refuse(FIT_REFUSAL.COLLINEAR, 'the geometric circle refinement diverged — the points do not determine a circle');
    }
    if (Math.hypot(step[0], step[1], step[2]) <= 1e-14 * Math.max(1, R)) break;
  }

  const center = addScaled3(addScaled3(plane.origin, plane.xAxis, cx), plane.yAxis, cy);
  const devs = uv.map(([x, y, w]) => {
    const radial = Math.hypot(x - cx, y - cy) - R;
    return Math.hypot(radial, w);
  });
  const { worst, rms } = worstAndRms(devs);
  return {
    ok: true,
    shape: 'circle',
    center,
    normal: plane.normal,
    xAxis: plane.xAxis,
    yAxis: plane.yAxis,
    radius: R,
    worst,
    rms,
    planeWorst: plane.worst,
    planeRms: plane.rms,
    count: prep.pts.length,
    extent: prep.extent,
  };
}

// Ellipse — Halir & Flusser (1998) / Fitzgibbon, Pilu & Fisher (1999)

// Real eigenvalues of a general (non-symmetric) 3x3 via its characteristic
// cubic, solved with the trigonometric form for three real roots and
// Cardano for the single-real-root case. Small and self-contained; the
// matrix Halir-Flusser hands us is only 3x3 by construction.
function realEigenvalues3(M) {
  const tr = M[0][0] + M[1][1] + M[2][2];
  const m2 =
    (M[0][0] * M[1][1] - M[0][1] * M[1][0]) +
    (M[0][0] * M[2][2] - M[0][2] * M[2][0]) +
    (M[1][1] * M[2][2] - M[1][2] * M[2][1]);
  const det =
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  // lambda^3 + a lambda^2 + b lambda + c = 0
  const a = -tr, b = m2, c = -det;
  const p = b - (a * a) / 3;
  const q = (2 * a * a * a) / 27 - (a * b) / 3 + c;
  const shift = -a / 3;
  const roots = [];
  if (Math.abs(p) < 1e-300) {
    const t = Math.cbrt(-q);
    roots.push(t + shift);
  } else {
    const disc = (q / 2) * (q / 2) + (p / 3) * (p / 3) * (p / 3);
    if (disc > 0) {
      const s = Math.sqrt(disc);
      roots.push(Math.cbrt(-q / 2 + s) + Math.cbrt(-q / 2 - s) + shift);
    } else {
      const m = 2 * Math.sqrt(-p / 3);
      let arg = (3 * q) / (p * m);
      arg = Math.max(-1, Math.min(1, arg));
      const phi = Math.acos(arg) / 3;
      for (let k = 0; k < 3; k++) {
        roots.push(m * Math.cos(phi - (2 * Math.PI * k) / 3) + shift);
      }
    }
  }
  return roots.filter(Number.isFinite);
}

// A unit null-vector of (M - lambda I) for a real eigenvalue lambda, taken
// as the largest cross product among row pairs (a rank-2 matrix's null
// space is exactly that cross product's direction).
function eigenvectorFor3(M, lambda) {
  const B = [
    [M[0][0] - lambda, M[0][1], M[0][2]],
    [M[1][0], M[1][1] - lambda, M[1][2]],
    [M[2][0], M[2][1], M[2][2] - lambda],
  ];
  let best = null, bestLen = 0;
  for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) {
    const v = cross3(B[i], B[j]);
    const l = len3(v);
    if (l > bestLen) { bestLen = l; best = v; }
  }
  if (!best || !(bestLen > 1e-300)) return null;
  return [best[0] / bestLen, best[1] / bestLen, best[2] / bestLen];
}

// Eberly's point-to-ellipse distance in the ellipse's own axis frame,
// semi-axes a >= b > 0. Bracketed bisection on a provably monotone
// function — no derivative, no starting guess, unconditional convergence.
function distancePointToEllipse2D(a, b, px, py) {
  const x0 = Math.abs(px), y0 = Math.abs(py);
  const a2 = a * a, b2 = b * b;
  let cx, cy;
  if (y0 > 0) {
    if (x0 > 0) {
      let t0 = -b2 + b * y0;
      let t1 = -b2 + Math.hypot(a * x0, b * y0);
      const F = (t) => {
        const u = (a * x0) / (t + a2);
        const v = (b * y0) / (t + b2);
        return u * u + v * v - 1;
      };
      let t = 0.5 * (t0 + t1);
      for (let i = 0; i < 120; i++) {
        t = 0.5 * (t0 + t1);
        const f = F(t);
        if (f > 0) t0 = t;
        else if (f < 0) t1 = t;
        else break;
        if (t1 - t0 <= 1e-16 * Math.max(1, Math.abs(t1))) break;
      }
      cx = (a2 * x0) / (t + a2);
      cy = (b2 * y0) / (t + b2);
    } else {
      cx = 0; cy = b;
    }
  } else {
    const num = a * x0;
    const den = a2 - b2;
    if (den > 0 && num < den) {
      const xde = num / den;
      cx = a * xde;
      cy = b * Math.sqrt(Math.max(0, 1 - xde * xde));
    } else {
      cx = a; cy = 0;
    }
  }
  return Math.hypot(cx - x0, cy - y0);
}

// fitEllipse(points, opts) -> {
//   ok, center, normal, xAxis, yAxis, radiusX, radiusY,
//   circular, worst, rms, planeWorst, planeRms, count
// }
// `center`/`xAxis`/`yAxis`/`radiusX`/`radiusY` are exactly
// the Ellipse operator's own param shape, so the result feeds
// `makeEllipse(center, xAxis, yAxis, radiusX, radiusY)` directly.
//
// Axis assignment is deterministic by rule, not by solver accident:
//   * radiusX is always the semi-major (larger) axis and radiusY the
//     semi-minor, so a caller never sees the two swap places between two
//     fits of nearly-identical input;
//   * xAxis's sign is canonicalized (largest component positive), and
//     yAxis is then derived as cross(normal, xAxis) rather than fitted
//     independently, so the returned frame is always right-handed and
//     yAxis can never disagree with xAxis about orientation.
//
// The perfect-circle case: when the
// two recovered radii agree to within `circularTol` (relative, default
// 1e-6) the ellipse's axis directions are arbitrary — every
// direction is a principal axis of a circle — and the conic eigen-solve
// returns whatever floating-point noise breaks the tie, which would
// flicker in a Properties panel. In
// that case this returns `circular: true`, both radii set to their mean,
// and the axes snapped to the plane's own canonical basis. The recovered
// shape is unchanged (a circle is a circle in any frame).
export function fitEllipse(pointsIn, opts = {}) {
  const framed = planarFrame(pointsIn, 5, opts);
  if (framed.bad) return framed.bad;
  const { prep, plane, uv } = framed;

  // Collinearity is fatal for a conic and is checked in the plane's own
  // 2D frame (fitPlane's own COLLINEAR check covers the fully-3D
  // case; this catches a set that is planar but degenerate within the
  // plane). `collinearTol: 0` disables it, which is how the
  // deeper DEGENERATE_CONIC guard below can be exercised directly.
  const collinearTol = Number.isFinite(opts.collinearTol)
    ? opts.collinearTol
    : Math.max(prep.extent * 1e-6, 1e-12);
  if (collinearTol > 0) {
    const c2 = [0, 0];
    for (const [x, y] of uv) { c2[0] += x; c2[1] += y; }
    c2[0] /= uv.length; c2[1] /= uv.length;
    let sxx = 0, sxy = 0, syy = 0;
    for (const [x, y] of uv) {
      const dx = x - c2[0], dy = y - c2[1];
      sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
    }
    const half = (sxx + syy) / 2;
    const gap = Math.hypot((sxx - syy) / 2, sxy);
    const minEig = Math.max(0, half - gap);
    const spread = Math.sqrt(minEig / uv.length);
    if (spread <= collinearTol) {
      return refuse(
        FIT_REFUSAL.COLLINEAR,
        `the points lie on a straight line within the fitted plane (RMS off-line ${spread.toExponential(3)}, tolerance ${collinearTol.toExponential(3)}); no ellipse is determined`,
        { lineRms: spread, collinearTol },
      );
    }
  }

  // Normalize before the conic solve — Halir & Flusser's stated
  // motivation is conditioning, and the largest
  // conditioning gain available is not solving a quartic-scaled system in
  // raw model coordinates. Translation + a uniform scale, so directions
  // survive untouched and only the center and radii need mapping back.
  let mx = 0, my = 0;
  for (const [x, y] of uv) { mx += x; my += y; }
  mx /= uv.length; my /= uv.length;
  let s = 0;
  for (const [x, y] of uv) s = Math.max(s, Math.hypot(x - mx, y - my));
  if (!(s > 0) || !Number.isFinite(s)) {
    return refuse(FIT_REFUSAL.COINCIDENT_POINTS, 'the in-plane points have no measurable spread');
  }

  // Halir-Flusser block decomposition.
  const S1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const S2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const S3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const [xr, yr] of uv) {
    const x = (xr - mx) / s, y = (yr - my) / s;
    const d1 = [x * x, x * y, y * y];
    const d2 = [x, y, 1];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        S1[i][j] += d1[i] * d1[j];
        S2[i][j] += d1[i] * d2[j];
        S3[i][j] += d2[i] * d2[j];
      }
    }
  }
  const S3inv = invert3(S3);
  if (!S3inv) {
    return refuse(FIT_REFUSAL.DEGENERATE_CONIC, 'the conic scatter matrix is singular — no ellipse is determined by these points');
  }
  const S2T = [[S2[0][0], S2[1][0], S2[2][0]], [S2[0][1], S2[1][1], S2[2][1]], [S2[0][2], S2[1][2], S2[2][2]]];
  const Tm = matmul3(S3inv, S2T).map((row) => row.map((v) => -v));
  const Mfull = matmul3(S2, Tm).map((row, i) => row.map((v, j) => S1[i][j] + v));
  // Premultiply by C1^-1 for the ellipse constraint 4ac - b^2 = 1, i.e.
  // C1 = [[0,0,2],[0,-1,0],[2,0,0]] -> C1^-1 rows are (row2/2, -row1, row0/2).
  const M = [
    [Mfull[2][0] / 2, Mfull[2][1] / 2, Mfull[2][2] / 2],
    [-Mfull[1][0], -Mfull[1][1], -Mfull[1][2]],
    [Mfull[0][0] / 2, Mfull[0][1] / 2, Mfull[0][2] / 2],
  ];

  let a1 = null;
  let bestCond = 0;
  for (const lambda of realEigenvalues3(M)) {
    const v = eigenvectorFor3(M, lambda);
    if (!v) continue;
    const cond = 4 * v[0] * v[2] - v[1] * v[1];
    if (cond > bestCond) { bestCond = cond; a1 = v; }
  }
  if (!a1) {
    return refuse(
      FIT_REFUSAL.DEGENERATE_CONIC,
      'no eigenvector of the constrained conic system satisfies 4ac - b^2 > 0 — these points do not determine an ellipse',
    );
  }
  const a2 = matvec3(Tm, a1);
  const [A, B, C] = a1;
  const [D, E, F] = a2;

  // Geometric params from the conic, via the 2x2 quadratic form rather
  // than the closed-form semi-axis expressions: the eigen-decomposition of
  // [[A, B/2], [B/2, C]] gives the axis directions directly and stays
  // well-behaved as the ellipse approaches a circle, where the closed-form
  // rotation-angle expressions become a 0/0.
  const Mq = [[A, B / 2], [B / 2, C]];
  const cen = solve3(
    [[2 * Mq[0][0], 2 * Mq[0][1], 0], [2 * Mq[1][0], 2 * Mq[1][1], 0], [0, 0, 1]],
    [-D, -E, 0],
  );
  if (!cen) {
    return refuse(FIT_REFUSAL.DEGENERATE_CONIC, 'the conic has no finite center — these points do not determine an ellipse');
  }
  const [ux, uy] = cen;
  const fPrime = F + (D * ux + E * uy) / 2;
  // Centered form: [u v] Mq [u; v] = -fPrime.
  const tr2 = Mq[0][0] + Mq[1][1];
  const gap2 = Math.hypot((Mq[0][0] - Mq[1][1]) / 2, Mq[0][1]);
  const lam1 = tr2 / 2 + gap2; // larger  -> shorter semi-axis
  const lam2 = tr2 / 2 - gap2; // smaller -> longer  semi-axis
  const k = -fPrime;
  if (!(k / lam1 > 0) || !(k / lam2 > 0)) {
    return refuse(
      FIT_REFUSAL.DEGENERATE_CONIC,
      'the recovered conic has no pair of real, positive semi-axes — these points do not determine an ellipse',
    );
  }
  const rMinorN = Math.sqrt(k / lam1);
  const rMajorN = Math.sqrt(k / lam2);
  // Eigenvector for lam2 (the semi-major direction), taken from whichever
  // row of (Mq - lam2*I) is larger in magnitude — the same "use the best-
  // conditioned row" technique eigenvectorFor3 above uses for the 3x3
  // case. A fixed row is wrong for an ordinary input: an ellipse that is
  // axis-aligned within its own fitted plane. There B is not exactly zero
  // but float dust, and lam2 equals the matching diagonal entry to
  // rounding, so both components of that one row are dust of comparable
  // size and the normalized direction is noise (on a 40x20 ellipse sampled
  // on-curve, a fixed row gives a major axis 43 degrees off and a 14-19 mm
  // deviation for an exact ellipse). For a symmetric Mq the larger-row
  // choice agrees with either row wherever that row is well conditioned,
  // and B exactly 0 needs no special case: the non-degenerate row is then
  // the larger one.
  const rowA = [Mq[0][0] - lam2, Mq[0][1]];
  const rowB = [Mq[1][0], Mq[1][1] - lam2];
  let ax2 = Math.hypot(rowA[0], rowA[1]) >= Math.hypot(rowB[0], rowB[1])
    ? [-rowA[1], rowA[0]]
    : [-rowB[1], rowB[0]];
  const axLen = Math.hypot(ax2[0], ax2[1]);
  if (!(axLen > 1e-300)) {
    return refuse(FIT_REFUSAL.DEGENERATE_CONIC, 'the conic axis direction collapsed to zero length');
  }
  ax2 = [ax2[0] / axLen, ax2[1] / axLen];

  // Un-normalize (uniform scale + translation: directions untouched).
  const cxLocal = mx + s * ux;
  const cyLocal = my + s * uy;
  let radiusX = s * rMajorN;
  let radiusY = s * rMinorN;
  const center = addScaled3(addScaled3(plane.origin, plane.xAxis, cxLocal), plane.yAxis, cyLocal);

  let xAxis = canonicalizeDirection([
    plane.xAxis[0] * ax2[0] + plane.yAxis[0] * ax2[1],
    plane.xAxis[1] * ax2[0] + plane.yAxis[1] * ax2[1],
    plane.xAxis[2] * ax2[0] + plane.yAxis[2] * ax2[1],
  ]);
  if (!xAxis) {
    return refuse(FIT_REFUSAL.DEGENERATE_CONIC, 'the recovered major-axis direction collapsed to zero length');
  }

  const circularTol = Number.isFinite(opts.circularTol) ? opts.circularTol : 1e-6;
  const circular = Math.abs(radiusX - radiusY) <= circularTol * Math.max(radiusX, radiusY);
  if (circular) {
    const mean = (radiusX + radiusY) / 2;
    radiusX = mean;
    radiusY = mean;
    xAxis = plane.xAxis;
  }
  const yAxis = cross3(plane.normal, xAxis);

  // Deviations: the true 3D distance to the ellipse curve — the exact
  // in-plane point-to-ellipse distance (Eberly) combined with the
  // out-of-plane miss.
  const devs = uv.map(([x, y, w]) => {
    const rel = [x - cxLocal, y - cyLocal];
    const localX = rel[0] * dot3(xAxis, plane.xAxis) + rel[1] * dot3(xAxis, plane.yAxis);
    const localY = rel[0] * dot3(yAxis, plane.xAxis) + rel[1] * dot3(yAxis, plane.yAxis);
    const inPlane = distancePointToEllipse2D(radiusX, radiusY, localX, localY);
    return Math.hypot(inPlane, w);
  });
  const { worst, rms } = worstAndRms(devs);

  const out = {
    ok: true,
    shape: 'ellipse',
    center,
    normal: plane.normal,
    xAxis,
    yAxis,
    radiusX,
    radiusY,
    circular,
    worst,
    rms,
    planeWorst: plane.worst,
    planeRms: plane.rms,
    count: prep.pts.length,
    extent: prep.extent,
  };
  // Last-resort finiteness check, like the kernel's isFiniteNet: a NaN
  // must never leave this module with `ok: true`.
  const allFinite = [
    ...out.center, ...out.normal, ...out.xAxis, ...out.yAxis,
    out.radiusX, out.radiusY, out.worst, out.rms,
  ].every(Number.isFinite);
  if (!allFinite || !(out.radiusX > 0) || !(out.radiusY > 0)) {
    return refuse(FIT_REFUSAL.DEGENERATE_CONIC, 'the recovered ellipse contains a non-finite or non-positive value');
  }
  return out;
}

// fitAll — every candidate at once, for a caller comparing recipes.
//
// It runs the four fits and hands back all four results, refusals
// included, without ranking them. Ranking needs the caller's own modeling
// tolerance and its own preference for which recipe to name.
export function fitAll(pointsIn, opts = {}) {
  return {
    line: fitLine(pointsIn, opts),
    plane: fitPlane(pointsIn, opts),
    circle: fitCircle(pointsIn, opts),
    ellipse: fitEllipse(pointsIn, opts),
  };
}
