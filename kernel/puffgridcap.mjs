// GRIDCAP — a POLE-FREE quad cage for Puff: rings to a hole, then a grid patch.
//
// WHY. The shipped cage converges M rings of constant N onto a single apex, and both measured
// defects are structural consequences of that: the apex fan puts near-coplanar cap faces back to
// back (worst 178.3 degrees on a C, 58.3 on a hand-drawn loop, measured beside this file's own
// output), and constant-N rings converging on a point MUST
// produce vanishing edges (min edge 0.0050 against a rim edge of 0.0654 — edge-length CV 0.957 on
// a disc). No relaxation fixes either, because the topology is the cause.
//
// WHAT. Rings are kept only to a HOLE at plan fraction `capFrac` of each spoke, ring depths spaced
// by equal 3D ARC LENGTH along the profile (the same surface, sampled evenly instead of evenly-in-
// plan). The hole — an N-edge loop — is closed by an (a x b) grid patch whose boundary IS the hole
// loop: 2(a-1) + 2(b-1) = N. For a disc the patch is square and the cage is a quad-sphere; for a
// spined shape the patch is long and thin — the RIDGE — and at b = 2 it degenerates to exactly the
// slit ladder ("a slit traversed out and back"). Interior vertices take z from the SAME distance-
// field law as every ring, so the surface the tool produces is unchanged; only the lattice
// sampling of it changes. Grid interior vertices are valence 4, the four patch corners valence 3 —
// the cube-sphere's corner, not a pole. Euler stays 2 for every (a, b); the guard asserts it.
//
// HOW THE PATCH INTERIOR IS PLACED, and the two dead ends that preceded it, both measured:
//   - Coons transfinite blend alone FOLDS on any bent or non-convex hole: 19 negative-plan-area
//     patch quads on a hand-drawn loop, 32 on a comma.
//   - A plain Laplacian (harmonic) relax untangles the convex cases and leaves folds hugging every
//     concave boundary run (10 on the C, 33 on the comma) — the textbook failure of harmonic maps
//     into a non-convex domain.
//   So the interior is relaxed by WINSLOW/TTM elliptic smoothing — the structured-grid generator
//   built for bent C-shaped channels — and the fold count is the stop condition, not a pass count.
//
// HOW (a, b) AND THE CORNERS ARE CHOSEN: BY MEASUREMENT, NOT BY FORMULA. A tensor patch forces
// opposite arcs equal, and a comma's two tips sit 33/63 edges apart around its hole loop — no
// corner placement can center both. An aspect-ratio formula picked b = 5 there and the columns
// crossed near the wrapped tip (worst dihedral 136 degrees, 5 residual folds Winslow could not
// remove). So the builder enumerates a small candidate set of (rows, corner) placements, builds
// each cheap patch, and scores it on the thing that actually matters — fold count, then worst 3D
// dihedral inside the patch — keeping the winner.
//
// ⚠ THE SILHOUETTE IS SACRED AND THE PATCH CANNOT REACH IT. Ring 0 is byte-identical to the
// prepared outline, the patch boundary is ring M0 (deep inside), and every relaxed ring vertex is
// rejected rather than moved if the move would leave the polygon. Containment is MEASURED in the
// guard (no vertex outside the outline beyond rounding), not argued here.
//
// ⚠ NOTHING HERE IS A TUNED WORLD-UNIT CONSTANT. capFrac is a fraction of the spoke, M0 comes
// from the rim edge length, rows from a measured search — every bound scales with the shape.

import { distanceToBoundary, pointInPolygon } from './puffoutline.mjs';

/** Arc-length coordinate of each target along the spine polyline — the tips of the cap are its
 *  extremes. Projection onto segments, not nearest sample: a coarse spine must not quantize. */
export function sigmaAlong(targets, spinePts) {
  const n = spinePts.length;
  const cum = new Float64Array(n);
  for (let j = 1; j < n; j++) {
    cum[j] = cum[j - 1] + Math.hypot(spinePts[j][0] - spinePts[j - 1][0], spinePts[j][1] - spinePts[j - 1][1]);
  }
  return targets.map((t) => {
    let best = Infinity, at = 0;
    for (let j = 0; j + 1 < n; j++) {
      const ax = spinePts[j][0], ay = spinePts[j][1];
      const ex = spinePts[j + 1][0] - ax, ey = spinePts[j + 1][1] - ay;
      const L2 = ex * ex + ey * ey;
      let u = L2 > 0 ? ((t[0] - ax) * ex + (t[1] - ay) * ey) / L2 : 0;
      u = u < 0 ? 0 : u > 1 ? 1 : u;
      const dd = Math.hypot(ax + u * ex - t[0], ay + u * ey - t[1]);
      if (dd < best) { best = dd; at = cum[j] + u * (cum[j + 1] - cum[j]); }
    }
    return at;
  });
}

// One patch candidate: Coons init + Winslow untangle over the hole loop, corners at i0.
// Returns interior xy, residual fold count, and iterations spent — cheap enough to enumerate.
function buildPatchGrid(hole, i0, am, bm) {
  const N = hole.length;
  const hi = (t) => ((t % N) + N) % N;
  const i1 = hi(i0 + am), i3 = hi(i0 - bm);
  const bnd = (r, c) => {
    if (r === 0) return hi(i0 + c);
    if (r === bm) return hi(i3 - c);
    if (c === 0) return hi(i0 - r);
    return hi(i1 + r);                                       // c === am
  };
  const interiorXY = [];
  for (let r = 1; r < bm; r++) for (let c = 1; c < am; c++) {
    const u = c / am, v = r / bm;
    const S0 = hole[bnd(0, c)], S1 = hole[bnd(bm, c)];
    const E0 = hole[bnd(r, 0)], E1 = hole[bnd(r, am)];
    const C00 = hole[bnd(0, 0)], C10 = hole[bnd(0, am)], C01 = hole[bnd(bm, 0)], C11 = hole[bnd(bm, am)];
    const x = (1 - v) * S0[0] + v * S1[0] + (1 - u) * E0[0] + u * E1[0]
      - ((1 - u) * (1 - v) * C00[0] + u * (1 - v) * C10[0] + (1 - u) * v * C01[0] + u * v * C11[0]);
    const y = (1 - v) * S0[1] + v * S1[1] + (1 - u) * E0[1] + u * E1[1]
      - ((1 - u) * (1 - v) * C00[1] + u * (1 - v) * C10[1] + (1 - u) * v * C01[1] + u * v * C11[1]);
    interiorXY.push([x, y]);
  }
  const gxy = (r, c) => (r === 0 || r === bm || c === 0 || c === am)
    ? hole[bnd(r, c)] : interiorXY[(r - 1) * (am - 1) + (c - 1)];
  const foldCount = () => {
    let neg = 0;
    for (let r = 0; r < bm; r++) for (let c = 0; c < am; c++) {
      const p00 = gxy(r, c), p10 = gxy(r, c + 1), p11 = gxy(r + 1, c + 1), p01 = gxy(r + 1, c);
      const a2 = (p10[0] - p00[0]) * (p11[1] - p00[1]) - (p11[0] - p00[0]) * (p10[1] - p00[1])
               + (p11[0] - p00[0]) * (p01[1] - p00[1]) - (p01[0] - p00[0]) * (p11[1] - p00[1]);
      if (a2 < 0) neg++;
    }
    return neg;
  };
  let folds = foldCount();
  if (bm >= 2 && am >= 2) {
    let span = 0;
    for (const p of hole) span = Math.max(span, Math.abs(p[0]), Math.abs(p[1]));
    const conv = Math.pow(1e-6 * Math.max(span, 1e-12), 2);
    let tail = 40;
    for (let it = 0; it < 600 && tail > 0; it++) {
      let moved = 0;
      for (let r = 1; r < bm; r++) for (let c = 1; c < am; c++) {
        const t = interiorXY[(r - 1) * (am - 1) + (c - 1)];
        const xc1 = gxy(r, c - 1), xc2 = gxy(r, c + 1);      // xi = along the row (c)
        const xe1 = gxy(r - 1, c), xe2 = gxy(r + 1, c);      // eta = across rows (r)
        const d11 = gxy(r + 1, c + 1), d00 = gxy(r - 1, c - 1), d10 = gxy(r - 1, c + 1), d01 = gxy(r + 1, c - 1);
        const xxi = (xc2[0] - xc1[0]) / 2, yxi = (xc2[1] - xc1[1]) / 2;
        const xet = (xe2[0] - xe1[0]) / 2, yet = (xe2[1] - xe1[1]) / 2;
        const al = xet * xet + yet * yet;
        const be = xxi * xet + yxi * yet;
        const ga = xxi * xxi + yxi * yxi;
        const den = 2 * (al + ga);
        if (!(den > 1e-30)) continue;
        const nx = (al * (xc1[0] + xc2[0]) + ga * (xe1[0] + xe2[0])
          - 0.5 * be * (d11[0] - d10[0] - d01[0] + d00[0])) / den;
        const ny = (al * (xc1[1] + xc2[1]) + ga * (xe1[1] + xe2[1])
          - 0.5 * be * (d11[1] - d10[1] - d01[1] + d00[1])) / den;
        const dx = 0.8 * (nx - t[0]), dy = 0.8 * (ny - t[1]);
        t[0] += dx; t[1] += dy;
        const m2 = dx * dx + dy * dy; if (m2 > moved) moved = m2;
      }
      // The fold scan every pass was 60% of the build (266 ms a comma under the profiler); every
      // 8th pass converges to the same grids — the folds question only changes on that timescale —
      // and a converged pass (max move under 1e-6 of the span) ends the tail early.
      if (it % 8 === 7 || moved < conv) {
        folds = foldCount();
        if (folds === 0) tail = Math.min(tail, moved < conv ? 0 : tail - 8);
      }
    }
    folds = foldCount();
  }
  return { interiorXY, folds, bnd, gxy };
}

// Worst dihedral across patch-internal edges AND the seam to the last band, with z sampled by
// `zAt`. The candidate score. ⚠ THE SEAM IS IN THE SCORE BECAUSE ITS ABSENCE WAS EXPLOITED: with
// patch-internal edges alone, a 1-row slit across a whole DISC scored best — every slit cell lies
// in the flat top, so their normals agree with each other perfectly while disagreeing with every
// band face they meet (max cage edge 0.9995 on a unit disc, CV 1.21). The score must see the join.
function patchWorstDihedral(gxy, zAt, am, bm, N, bnd, hole, zHole, ringIn, zIn) {
  const ang = (a, b) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180 / Math.PI;
  const norm = new Array(bm);
  for (let r = 0; r < bm; r++) {
    norm[r] = new Array(am);
    for (let c = 0; c < am; c++) {
      const p00 = gxy(r, c), p10 = gxy(r, c + 1), p11 = gxy(r + 1, c + 1), p01 = gxy(r + 1, c);
      const z00 = zAt(p00), z10 = zAt(p10), z11 = zAt(p11), z01 = zAt(p01);
      const ux = p11[0] - p00[0], uy = p11[1] - p00[1], uz = z11 - z00;
      const vx = p01[0] - p10[0], vy = p01[1] - p10[1], vz = z01 - z10;
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const L = Math.hypot(nx, ny, nz) || 1;
      norm[r][c] = [nx / L, ny / L, nz / L];
    }
  }
  let worst = 0;
  for (let r = 0; r < bm; r++) for (let c = 0; c < am; c++) {
    if (c + 1 < am) worst = Math.max(worst, ang(norm[r][c], norm[r][c + 1]));
    if (r + 1 < bm) worst = Math.max(worst, ang(norm[r][c], norm[r + 1][c]));
  }
  // The seam: each hole edge is shared by one boundary patch cell and one band face — the face
  // (ringIn[i], ringIn[j], hole[j], hole[i]), normal from its diagonals.
  const bandNorm = (i) => {
    const j = (i + 1) % N;
    const ux = hole[j][0] - ringIn[i][0], uy = hole[j][1] - ringIn[i][1], uz = zHole[j] - zIn[i];
    const vx = hole[i][0] - ringIn[j][0], vy = hole[i][1] - ringIn[j][1], vz = zHole[i] - zIn[j];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const L = Math.hypot(nx, ny, nz) || 1;
    return [nx / L, ny / L, nz / L];
  };
  // Which patch cell borders hole edge (min(i,j) -> ...)? Walk the grid boundary.
  for (let c = 0; c < am; c++) {                                 // row 0 side: edge bnd(0,c)->bnd(0,c+1)
    worst = Math.max(worst, ang(norm[0][c], bandNorm(bnd(0, c))));
  }
  for (let c = 0; c < am; c++) {                                 // row bm side: edge bnd(bm,c+1)->bnd(bm,c)
    worst = Math.max(worst, ang(norm[bm - 1][c], bandNorm(bnd(bm, c + 1))));
  }
  for (let r = 0; r < bm; r++) {                                 // col 0 side: edge bnd(r+1,0)->bnd(r,0)
    worst = Math.max(worst, ang(norm[r][0], bandNorm(bnd(r + 1, 0))));
  }
  for (let r = 0; r < bm; r++) {                                 // col am side: edge bnd(r,am)->bnd(r+1,am)
    worst = Math.max(worst, ang(norm[r][am - 1], bandNorm(bnd(r, am))));
  }
  return worst;
}

/**
 * Build the whole puff mesh with the grid cap.
 * @param pts      prepared outline, N points, CCW — ring 0, byte-identical
 * @param targets  per-point interior target ([x,y] per boundary point; the single center repeated
 *                 on the star-shaped path, the smoothed spine attachment on the spined path)
 * @param sigma    per-point coordinate along the spine (null on the star path — PCA stands in)
 * @param o        { dmax, effH, prof, smoothing, bottomScale, capFrac, capBands, capRows }
 */
export function buildGridCage(pts, targets, sigma, o) {
  const N = pts.length;
  const { dmax, effH, prof, smoothing = 2, bottomScale = 1, capFrac = 0.5 } = o;
  const zOf = (x, y) => effH * prof(Math.min(Math.max(distanceToBoundary(pts, x, y) / dmax, 0), 1));
  const zAtP = (p) => zOf(p[0], p[1]);
  const perim = pts.reduce((s, p, i) => {
    const q = pts[(i + 1) % N]; return s + Math.hypot(q[0] - p[0], q[1] - p[1]);
  }, 0);
  const e0 = perim / N;                          // rim edge — the length everything is sized to

  // ---- rings, arc-length spaced along each spoke down to the hole ------------------------------
  // The spoke from pts[i] to targets[i] carries the surface curve (xy(f), z(xy(f))). Rings sit at
  // equal 3D arc steps of that curve, not equal plan steps — a hemisphere's rim band is near
  // vertical, and equal-plan spacing is what stretched it 4:1 in the shipped cage.
  const S = 24;
  const arc = new Array(N);
  let meanArc = 0;
  for (let i = 0; i < N; i++) {
    const a = new Float64Array(S + 1);
    let px = pts[i][0], py = pts[i][1], pz = 0;
    for (let j = 1; j <= S; j++) {
      const f = capFrac * j / S;
      const x = pts[i][0] + (targets[i][0] - pts[i][0]) * f;
      const y = pts[i][1] + (targets[i][1] - pts[i][1]) * f;
      const z = zOf(x, y);
      a[j] = a[j - 1] + Math.hypot(x - px, y - py, z - pz);
      px = x; py = y; pz = z;
    }
    arc[i] = a; meanArc += a[S] / N;
  }
  const M0 = o.capBands != null ? Math.max(1, o.capBands | 0)
    : Math.max(2, Math.min(48, Math.round(meanArc / e0)));

  const ringXY = [];
  for (let k = 0; k <= M0; k++) {
    const ring = new Array(N);
    for (let i = 0; i < N; i++) {
      if (k === 0) { ring[i] = [pts[i][0], pts[i][1]]; continue; }
      const want = arc[i][S] * k / M0;
      let j = 1; while (j < S && arc[i][j] < want) j++;
      const a0 = arc[i][j - 1], a1 = arc[i][j];
      const fj = capFrac * (j - 1 + (a1 > a0 ? (want - a0) / (a1 - a0) : 0)) / S;
      ring[i] = [pts[i][0] + (targets[i][0] - pts[i][0]) * fj,
                 pts[i][1] + (targets[i][1] - pts[i][1]) * fj];
    }
    ringXY.push(ring);
  }

  // ---- per-ring loop smoothing, depth-weighted, guarded ----------------------------------------
  // At a concave dent the spokes converge and the ring polylines kink — measured 64 degrees across
  // a spoke between bands 3 and 4 of a hand-drawn loop with no smoothing at all. Each ring LOOP is
  // smoothed independently (a damped 1D Laplacian around the loop), weighted by depth: ring 1
  // barely moves and stays true to the drawn shape, the hole rounds the most — which is also what
  // the patch wants for a boundary. Smoothing each ring separately, rather than the 2D lattice, is
  // deliberate and paid for: a full lattice Laplacian was tried first and REDISTRIBUTED the rings
  // toward uniform plan spacing, undoing the arc-length placement (disc subdivided worst 4.2 -> 16.5
  // degrees). A move that would leave the polygon is REJECTED, not clamped: the silhouette bound is
  // not negotiable and a clamp is a second algorithm to verify.
  const relaxPasses = o.capRelax != null ? Math.max(0, o.capRelax | 0) : 24;
  for (let it = 0; it < relaxPasses; it++) {
    for (let k = 1; k <= M0; k++) {
      const lam = 0.5 * k / M0;
      const ring = ringXY[k], nxt = ring.map(p => p.slice());
      for (let i = 0; i < N; i++) {
        const im = (i - 1 + N) % N, ip = (i + 1) % N;
        const nx = ring[i][0] + lam * ((ring[im][0] + ring[ip][0]) / 2 - ring[i][0]);
        const ny = ring[i][1] + lam * ((ring[im][1] + ring[ip][1]) / 2 - ring[i][1]);
        if (pointInPolygon(pts, nx, ny)) { nxt[i][0] = nx; nxt[i][1] = ny; }
      }
      ringXY[k] = nxt;
    }
  }

  // ---- the field on the ring lattice, smoothed exactly as the shipped S6 does ------------------
  const d = [];
  for (let k = 0; k <= M0; k++) {
    const row = new Float64Array(N);
    for (let i = 0; i < N; i++) row[i] = k === 0 ? 0 : distanceToBoundary(pts, ringXY[k][i][0], ringXY[k][i][1]);
    d.push(row);
  }
  // ⚠ THE HOLE RING (k = M0) IS LOCKED, LIKE RING 0, AND THE PICTURE FOUND IT. The shipped S6
  // stencil clamps its inward neighbor at the last ring, so the one-sided average drags the field
  // down exactly there — on a disc the hole ring sagged to 0.9938 of the sphere radius, a visible
  // circular dimple at the patch seam in the render. The hole's raw distance is exact, and the
  // patch interior samples the raw field too, so locking it removes both the sag and the seam
  // mismatch in one move.
  const passes = Math.max(0, smoothing | 0);
  for (let it = 0; it < passes; it++) {
    const next = d.map(r => Float64Array.from(r));
    for (let k = 1; k < M0; k++) {
      for (let i = 0; i < N; i++) {
        const im = (i - 1 + N) % N, ip = (i + 1) % N;
        const avg = (d[k][im] + d[k][ip] + d[k - 1][i] + d[k + 1][i]) / 4;
        next[k][i] = d[k][i] + 0.5 * (avg - d[k][i]);
      }
    }
    for (let k = 0; k <= M0; k++) d[k].set(next[k]);
  }
  const zRing = [];
  for (let k = 0; k <= M0; k++) {
    const row = new Float64Array(N);
    for (let i = 0; i < N; i++) row[i] = effH * prof(Math.min(Math.max(d[k][i] / dmax, 0), 1));
    zRing.push(row);
  }

  // ---- the patch: candidates measured, winner kept ---------------------------------------------
  const hole = ringXY[M0];
  const half = N / 2;
  // sigma over the HOLE: the spine coordinate when there is one, the long-PCA projection when not.
  let sg = sigma, L, W;
  if (sg) {
    let mn = Infinity, mx = -Infinity;
    for (const s of sg) { if (s < mn) mn = s; if (s > mx) mx = s; }
    L = mx - mn;
    let w = 0; for (let i = 0; i < N; i++) w += Math.hypot(hole[i][0] - targets[i][0], hole[i][1] - targets[i][1]);
    W = 2 * w / N;
  } else {
    let cx = 0, cy = 0; for (const p of hole) { cx += p[0] / N; cy += p[1] / N; }
    let sxx = 0, sxy = 0, syy = 0;
    for (const p of hole) { const x = p[0] - cx, y = p[1] - cy; sxx += x * x; sxy += x * y; syy += y * y; }
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy), c = Math.cos(th), s = Math.sin(th);
    let mn1 = Infinity, mx1 = -Infinity, mn2 = Infinity, mx2 = -Infinity;
    sg = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const x = hole[i][0] - cx, y = hole[i][1] - cy;
      const u = x * c + y * s, v = -x * s + y * c;
      sg[i] = u;
      if (u < mn1) mn1 = u; if (u > mx1) mx1 = u;
      if (v < mn2) mn2 = v; if (v > mx2) mx2 = v;
    }
    L = Math.max(mx1 - mn1, mx2 - mn2); W = Math.min(mx1 - mn1, mx2 - mn2);
    if (mx1 - mn1 < mx2 - mn2) for (let i = 0; i < N; i++) {
      const x = hole[i][0] - cx, y = hole[i][1] - cy; sg[i] = -x * s + y * c;
    }
  }
  let imn = 0, imx = 0;
  for (let i = 0; i < N; i++) { if (sg[i] < sg[imn]) imn = i; if (sg[i] > sg[imx]) imx = i; }
  const cd = (x, y2) => { const t = Math.abs(((x - y2) % N + N) % N); return Math.min(t, N - t); };

  // Candidate rows: the aspect formula, the two splits that make one side match the tips' natural
  // index distance exactly, and the slit — deduplicated and clamped. Candidate corners per rows
  // value: the best tip-centering index and a few offsets around it. Winner by (folds, dihedral).
  const tipGap = cd(imx, imn);                               // natural side lengths: tipGap, N - tipGap
  const rowCand = new Set();
  const clampRows = (x) => Math.max(1, Math.min(half - 1, x));
  const arRows = clampRows(Math.round(half * W / (L + W)));
  rowCand.add(arRows);
  rowCand.add(clampRows(half - tipGap));
  rowCand.add(clampRows(half - (N - tipGap)));
  rowCand.add(clampRows(arRows * 2));
  rowCand.add(clampRows(half >> 1));                         // the square split — a disc's answer
  rowCand.add(clampRows(Math.round((arRows + (half >> 1)) / 2)));
  if (o.capRows != null) { rowCand.clear(); rowCand.add(clampRows(o.capRows | 0)); }
  // Mean 3D band edge — the length every cap edge is asked to match. Uniformity is IN the score
  // because its absence was exploited too: scored on dihedrals alone, a 47x1 slit won on a DISC
  // whose profile has a conical peak (the slit's cells agree with each other about the cone), and
  // its cross edges were 20x the band edge (cage CV 1.43). U is the RMS of log(len/eBand) over
  // patch and seam edges; the weight makes a 10% deviation cost 3 degrees.
  let eBand = 0, nBand = 0;
  for (let k = 0; k < M0; k++) for (let i = 0; i < N; i++) {
    const j = (i + 1) % N;
    eBand += Math.hypot(ringXY[k][i][0] - ringXY[k][j][0], ringXY[k][i][1] - ringXY[k][j][1], zRing[k][i] - zRing[k][j]);
    eBand += Math.hypot(ringXY[k][i][0] - ringXY[k + 1][i][0], ringXY[k][i][1] - ringXY[k + 1][i][1], zRing[k][i] - zRing[k + 1][i]);
    nBand += 2;
  }
  eBand /= nBand;
  let best = null;
  for (const bm of rowCand) {
    const am = half - bm;
    let ci0 = 0, cbest = Infinity;
    for (let c = 0; c < N; c++) {
      const cost = cd(c + am + bm / 2, imx) + cd(c - bm / 2, imn);
      if (cost < cbest) { cbest = cost; ci0 = c; }
    }
    for (const off of [0, -4, 4, -8, 8]) {
      const i0 = ((ci0 + off) % N + N) % N;
      const g = buildPatchGrid(hole, i0, am, bm);
      // Exact distance is O(N) per query and the score asks for each grid point several times —
      // cached per candidate, keyed by the point ARRAY, which is stable within one.
      const zC = new Map();
      const zAt = (p) => { let v = zC.get(p); if (v === undefined) { v = zAtP(p); zC.set(p, v); } return v; };
      const worst = patchWorstDihedral(g.gxy, zAt, am, bm, N, g.bnd,
        hole, zRing[M0], ringXY[M0 - 1], zRing[M0 - 1]);
      let u = 0, ne = 0;
      const el = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], zAt(p) - zAt(q));
      for (let r = 0; r <= bm; r++) for (let c = 0; c < am; c++) {
        const l = el(g.gxy(r, c), g.gxy(r, c + 1)); u += Math.pow(Math.log(l / eBand), 2); ne++;
      }
      for (let r = 0; r < bm; r++) for (let c = 0; c <= am; c++) {
        const l = el(g.gxy(r, c), g.gxy(r + 1, c)); u += Math.pow(Math.log(l / eBand), 2); ne++;
      }
      const U = Math.sqrt(u / ne);
      const score = g.folds * 1e6 + worst + 30 * U;
      if (!best || score < best.score) best = { score, i0, am, bm, ...g };
      if (o.capRows != null && off === 0) break;             // explicit rows: honour it exactly
    }
  }
  const { i0, am, bm, interiorXY, bnd } = best;
  const nI = interiorXY.length;                              // (bm-1)*(am-1) interior vertices

  // ---- assembly: one shared equator, two independent halves — the shipped structure ------------
  const V = N * (2 * M0 + 1) + 2 * nI;
  const positions = new Float32Array(V * 3);
  const put = (idx, x, y, z) => { positions[idx * 3] = x; positions[idx * 3 + 1] = y; positions[idx * 3 + 2] = z; };
  const front = (k, i) => (k === 0 ? i : N + (k - 1) * N + i);
  const FI = N + M0 * N;                                     // front patch interior base
  const back = (k, i) => (k === 0 ? i : FI + nI + (k - 1) * N + i);
  const BI = FI + nI + M0 * N;                               // back patch interior base
  const fg = (r, c) => (r === 0 || r === bm || c === 0 || c === am)
    ? front(M0, bnd(r, c)) : FI + (r - 1) * (am - 1) + (c - 1);
  const bg = (r, c) => (r === 0 || r === bm || c === 0 || c === am)
    ? back(M0, bnd(r, c)) : BI + (r - 1) * (am - 1) + (c - 1);

  for (let i = 0; i < N; i++) put(i, pts[i][0], pts[i][1], 0);
  for (let k = 1; k <= M0; k++) for (let i = 0; i < N; i++) {
    put(front(k, i), ringXY[k][i][0], ringXY[k][i][1], zRing[k][i]);
    put(back(k, i), ringXY[k][i][0], ringXY[k][i][1], -zRing[k][i] * bottomScale);
  }
  for (let t = 0; t < nI; t++) {
    const [x, y] = interiorXY[t]; const z = zOf(x, y);
    put(FI + t, x, y, z);
    put(BI + t, x, y, -z * bottomScale);
  }

  const F = 2 * (M0 * N + am * bm);
  const quads = new Uint32Array(F * 4);
  let q = 0;
  for (let k = 0; k < M0; k++) for (let i = 0; i < N; i++) {
    const j = (i + 1) % N;
    quads[q++] = front(k, i); quads[q++] = front(k, j); quads[q++] = front(k + 1, j); quads[q++] = front(k + 1, i);
  }
  for (let r = 0; r < bm; r++) for (let c = 0; c < am; c++) {
    quads[q++] = fg(r, c); quads[q++] = fg(r, c + 1); quads[q++] = fg(r + 1, c + 1); quads[q++] = fg(r + 1, c);
  }
  for (let k = 0; k < M0; k++) for (let i = 0; i < N; i++) {
    const j = (i + 1) % N;
    quads[q++] = back(k, i); quads[q++] = back(k + 1, i); quads[q++] = back(k + 1, j); quads[q++] = back(k, j);
  }
  for (let r = 0; r < bm; r++) for (let c = 0; c < am; c++) {
    quads[q++] = bg(r, c); quads[q++] = bg(r + 1, c); quads[q++] = bg(r + 1, c + 1); quads[q++] = bg(r, c + 1);
  }

  return {
    positions, quads, nv: V, M: M0, field: d,
    cap: { kind: 'grid', a: am + 1, b: bm + 1, i0, rows: bm, cols: am, interior: nI, capFrac, folds: best.folds },
  };
}
