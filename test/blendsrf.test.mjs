// A blend surface meets both surfaces: at every station it leaves A along A's
// own cross-boundary derivative and arrives at B along B's (G1), with the
// curvature vectors matching too at G2; the boundary curves are the edges;
// bulge changes the interior and not the boundaries; B's direction is chosen
// so the ends pair.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { blendSurfaceBetweenEdges } from '../kernel/blendsrf.mjs';
import { surfacePoint, surfacePointAndPartials } from '../kernel/surface.mjs';
import { extractIsocurveU, extractIsocurveV } from '../kernel/isocurve.mjs';
import { rationalCurveDerivs } from '../kernel/curve.mjs';

// A bicubic Bezier sheet from a height function over [x0,x1]×[y0,y1].
function sheet(x0, x1, y0, y1, z) {
  const n = 4, ctrlNet = [];
  for (let i = 0; i < n; i++) { const row = []; for (let j = 0; j < n; j++) { const x = x0 + (x1 - x0) * i / (n - 1), y = y0 + (y1 - y0) * j / (n - 1); row.push([x, y, z(x, y), 1]); } ctrlNet.push(row); }
  return { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 1, 1, 1, 1], ctrlNet };
}
const norm = (a) => { const L = Math.hypot(...a); return a.map((x) => x / L); };
const angle = (a, b) => Math.acos(Math.max(-1, Math.min(1, norm(a)[0] * norm(b)[0] + norm(a)[1] * norm(b)[1] + norm(a)[2] * norm(b)[2]))) * 180 / Math.PI;
const crossN = (a, b) => norm([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);
// The surface normal at (u, v) from the partials.
function normalAt(srf, u, v) { const r = surfacePointAndPartials(srf, u, v); return crossN(r.su, r.sv); }

test('blendSurfaceBetweenEdges: two flat sheets in one plane — the blend is the flat strip between them, G2, its boundaries the edges', () => {
  const A = sheet(-60, -20, 0, 40, () => 0), B = sheet(20, 60, 0, 40, () => 0);
  const r = blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }, { continuity: 2, stations: 12 });
  assert.ok(r.ok, r.reason);
  assert.equal(r.srf.degU, 5);
  assert.equal(r.srf.ctrlNet.length, 6);
  for (let i = 0; i <= 8; i++) {
    const v = i / 8;
    const p0 = surfacePoint(r.srf, 0, v), p1 = surfacePoint(r.srf, 1, v), pm = surfacePoint(r.srf, 0.5, v);
    assert.ok(Math.abs(p0[0] + 20) < 1e-6 && Math.abs(p1[0] - 20) < 1e-6, `boundaries on the edges (${p0[0]}, ${p1[0]})`);
    assert.ok(Math.abs(pm[2]) < 1e-9 && pm[0] > -20 && pm[0] < 20 && Math.abs(pm[1] - p0[1]) < 1e-6, `flat and straight across (${pm})`);
  }
});

test('blendSurfaceBetweenEdges: the blend leaves A into the gap and arrives into B — it never overshoots past either edge', () => {
  const A = sheet(-80, -30, 0, 40, () => 0), B = sheet(30, 80, 0, 40, () => 20);
  const r = blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }, { stations: 8 });
  assert.ok(r.ok);
  for (let i = 0; i <= 20; i++) { const p = surfacePoint(r.srf, i / 20, 0.5); assert.ok(p[0] >= -30 - 1e-6 && p[0] <= 30 + 1e-6, `x ${p[0].toFixed(2)} stays between the edges`); assert.ok(p[2] >= -1e-6 && p[2] <= 20 + 1e-6, `z ${p[2].toFixed(2)} stays between the sheets`); }
  // Monotone in x: an S between two flat sheets, not a hook.
  let prev = -Infinity; for (let i = 0; i <= 20; i++) { const x = surfacePoint(r.srf, i / 20, 0.5)[0]; assert.ok(x >= prev - 1e-9, 'x never turns back'); prev = x; }
});

test('blendSurfaceBetweenEdges: a flat sheet and a tilted one — G1 at both edges, and G2 matches curvature (zero on planes)', () => {
  const A = sheet(-60, -20, 0, 40, () => 0);
  const B = sheet(20, 60, 0, 40, (x) => (x - 20) * 0.8); // rising plane
  for (const continuity of [1, 2]) {
    const r = blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }, { continuity, stations: 16 });
    assert.ok(r.ok, r.reason);
    for (let i = 0; i <= 6; i++) {
      const v = i / 6;
      // Normals across the seam: the blend at u=0 vs A at u=1; the blend at u=1 vs B at u=0.
      const nA = normalAt(A, 1, v), n0 = normalAt(r.srf, 0, v);
      const nB = normalAt(B, 0, v), n1 = normalAt(r.srf, 1, v);
      assert.ok(angle(nA, n0) < 0.05 || angle(nA, n0) > 179.95, `G1 at A (${angle(nA, n0).toFixed(3)}°) at v ${v.toFixed(2)}`);
      assert.ok(angle(nB, n1) < 0.05 || angle(nB, n1) > 179.95, `G1 at B (${angle(nB, n1).toFixed(3)}°) at v ${v.toFixed(2)}`);
    }
    if (continuity === 2) {
      // Curvature of the blend's cross-section at its ends is zero (both neighbors are planes).
      for (const v of [0.25, 0.75]) {
        const iso = extractIsocurveV(r.srf, v);
        for (const u of [0, 1]) {
          const [, C1, C2] = rationalCurveDerivs(iso, u, 2);
          const s = Math.hypot(...C1);
          const k = Math.hypot(...[C1[1] * C2[2] - C1[2] * C2[1], C1[2] * C2[0] - C1[0] * C2[2], C1[0] * C2[1] - C1[1] * C2[0]]) / (s * s * s);
          assert.ok(k < 1e-6, `curvature at u ${u} is the planes' zero (${k.toExponential(2)})`);
        }
      }
    }
  }
});

test('blendSurfaceBetweenEdges: bulge moves the interior and leaves the boundaries; B is walked the way that pairs the ends', () => {
  const A = sheet(-60, -20, 0, 40, () => 0), B = sheet(20, 60, 0, 40, (x) => (x - 20) * 0.8);
  const a = blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }, { stations: 12 });
  const b = blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }, { stations: 12, bulgeA: 2, bulgeB: 0.5 });
  assert.ok(a.ok && b.ok);
  const mid = (r) => surfacePoint(r.srf, 0.5, 0.5);
  assert.ok(Math.hypot(...mid(a).map((x, i) => x - mid(b)[i])) > 0.5, 'bulge changes the middle');
  for (const v of [0, 0.5, 1]) for (const u of [0, 1]) assert.ok(Math.hypot(...surfacePoint(a.srf, u, v).map((x, i) => x - surfacePoint(b.srf, u, v)[i])) < 1e-6, 'and not the boundaries');
  // B reversed: a sheet whose V runs the other way still pairs ends.
  const Brev = { ...B, ctrlNet: B.ctrlNet.map((row) => [...row].reverse()) };
  const c = blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: Brev, edge: 'u0' }, { stations: 12 });
  assert.ok(c.ok && c.reverseB === true, 'B walked reversed');
  const p = surfacePoint(c.srf, 1, 0);
  assert.ok(Math.abs(p[1] - surfacePoint(c.srf, 0, 0)[1]) < 1e-6, `the ends pair (y ${p[1]})`);
});

test('blendSurfaceBetweenEdges: refusals — touching edges, a bad edge name, a non-positive bulge', () => {
  const A = sheet(-60, -20, 0, 40, () => 0), B = sheet(-20, 20, 0, 40, () => 0);
  assert.equal(blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }).ok, false);
  assert.match(blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }).reason, /touch/);
  assert.match(blendSurfaceBetweenEdges({ srf: A, edge: 'x9' }, { srf: B, edge: 'u0' }).reason, /edges/);
  assert.match(blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }, { bulgeA: 0 }).reason, /bulge/);
});

test('blendSurfaceBetweenEdges: adaptive skinning measures between stations and refines to tolerance', () => {
  const A = sheet(-60, -20, 0, 80, (_, y) => 18 * Math.sin(y * Math.PI / 40));
  const B = sheet(20, 60, 0, 80, (x, y) => 12 * Math.cos(y * Math.PI / 40) + 0.35 * (x - 20));
  const r = blendSurfaceBetweenEdges({ srf: A, edge: 'u1' }, { srf: B, edge: 'u0' }, {
    continuity: 2, stations: 4, adaptive: true, tolerance: 0.01, maxStations: 97,
  });
  assert.ok(r.ok, r.reason);
  assert.equal(r.refined, true, 'the deliberately coarse four-section skin is not accepted on faith');
  assert.ok(r.stations > 4, `station count grew (${r.stations})`);
  assert.ok(r.toleranceMet && r.deviation <= 0.01, `measured ${r.deviation}mm <= 0.01mm`);
});

// Junction
import { blendSurfaceJunction } from '../kernel/blendsrf.mjs';
function sheetAt(cx, cy, dirDeg, tiltDeg = 0, w = 40, len = 50) {
  const th = dirDeg * Math.PI / 180, tl = tiltDeg * Math.PI / 180;
  const ax = [Math.cos(th) * Math.cos(tl), Math.sin(th) * Math.cos(tl), Math.sin(tl)], ay = [-Math.sin(th), Math.cos(th), 0];
  const n = 4, ctrlNet = [];
  for (let i = 0; i < n; i++) { const row = []; for (let j = 0; j < n; j++) { const a = (i / (n - 1)) * len, b = (j / (n - 1) - 0.5) * w; row.push([cx + ax[0] * a + ay[0] * b, cy + ax[1] * a + ay[1] * b, ax[2] * a, 1]); } ctrlNet.push(row); }
  return { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 1, 1, 1, 1], ctrlNet };
}
const star = (N, R = 40, tilt = 0) => Array.from({ length: N }, (_, i) => { const th = i * 360 / N; return { srf: sheetAt(R * Math.cos(th * Math.PI / 180), R * Math.sin(th * Math.PI / 180), th, i === 0 ? tilt : 0), edge: 'u0' }; });

import { surfaceDerivs2 } from '../kernel/curvature.mjs';
const gap3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const bnd = (S, side, t) => { const u0 = S.knotsU[0], u1 = S.knotsU[S.knotsU.length - 1], v0 = S.knotsV[0], v1 = S.knotsV[S.knotsV.length - 1]; return side === 0 ? surfacePoint(S, u0 + (u1 - u0) * t, v0) : side === 1 ? surfacePoint(S, u1, v0 + (v1 - v0) * t) : side === 2 ? surfacePoint(S, u0 + (u1 - u0) * t, v1) : surfacePoint(S, u0, v0 + (v1 - v0) * t); };
const unitCross = (a, b) => { const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; const L = Math.hypot(...c); return c.map((x) => x / L); };
// Every quad of a guided cap: its v = 0 side lies on its sheet's blend edge, the
// normals agree there, the normal curvature across agrees (G2), and the seams
// between quads — the sheet spokes and the free-edge spokes — are one curve.
function assertGuidedCap(parts, r, N, k = 2) {
  assert.ok(r.ok, r.reason);
  assert.equal(r.faceUp, true, 'sheets in one plane take the cap');
  assert.equal(r.arms.length, 0, 'the cap has no arms');
  assert.equal(r.centre.length, 2 * N, 'two ribbon quads per sheet');
  assert.ok(r.centre.every((f) => f.degU <= 11 && f.degV <= 11), `every quad is degree 11 or less, as a .3dm holds (${r.centre.map((f) => `${f.degU}/${f.degV}`).join(' ')})`);
  let worstOn = 0, worstAng = 0, worstK = 0, worstSeam = 0;
  for (let i = 0; i < N; i++) {
    const k = r.order[i], srf = parts[k].srf;
    for (const q of [r.centre[2 * i], r.centre[2 * i + 1]]) for (let t = 0; t <= 10; t++) {
      const D = surfaceDerivs2(q, t / 10, 0);
      let best = 0, bd = Infinity;
      for (let m = 0; m <= 400; m++) { const d = gap3(D.S, surfacePoint(srf, 0, m / 400)); if (d < bd) { bd = d; best = m / 400; } }
      worstOn = Math.max(worstOn, bd);
      const E = surfaceDerivs2(srf, 0, best);
      const nQ = unitCross(D.Su, D.Sv), nS = unitCross(E.Su, E.Sv);
      worstAng = Math.max(worstAng, Math.acos(Math.min(1, Math.abs(nQ[0] * nS[0] + nQ[1] * nS[1] + nQ[2] * nS[2]))) * 180 / Math.PI);
      const kq = (D.Svv[0] * nS[0] + D.Svv[1] * nS[1] + D.Svv[2] * nS[2]) / (Math.hypot(...D.Sv) ** 2);
      const ks = (E.Suu[0] * nS[0] + E.Suu[1] * nS[1] + E.Suu[2] * nS[2]) / (Math.hypot(...E.Su) ** 2);
      worstK = Math.max(worstK, Math.abs(kq - ks));
    }
    const nx = 2 * ((i + 1) % N);
    for (let t = 0; t <= 20; t++) {
      worstSeam = Math.max(worstSeam, gap3(bnd(r.centre[2 * i], 1, t / 20), bnd(r.centre[2 * i + 1], 3, t / 20)));
      worstSeam = Math.max(worstSeam, gap3(bnd(r.centre[2 * i + 1], 2, 1 - t / 20), bnd(r.centre[nx], 2, t / 20)));
    }
  }
  // G1 across every spoke, and one tangent plane at the center.
  const nrmAt = (S, u, v) => { const D = surfaceDerivs2(S, u, v); return unitCross(D.Su, D.Sv); };
  const ang = (a, b) => Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180 / Math.PI;
  const transverseK = (S, u, v, seamAlongV) => {
    const D = surfaceDerivs2(S, u, v), T = seamAlongV ? D.Sv : D.Su, X = seamAlongV ? D.Su : D.Sv;
    const XX = seamAlongV ? D.Suu : D.Svv, XT = D.Suv, TT = seamAlongV ? D.Svv : D.Suu;
    const q = -(X[0] * T[0] + X[1] * T[1] + X[2] * T[2]) / (T[0] ** 2 + T[1] ** 2 + T[2] ** 2);
    const V = X.map((x, j) => x + q * T[j]), A = XX.map((x, j) => x + 2 * q * XT[j] + q * q * TT[j]);
    const n = unitCross(D.Su, D.Sv);
    return { k: (A[0] * n[0] + A[1] * n[1] + A[2] * n[2]) / (V[0] ** 2 + V[1] ** 2 + V[2] ** 2), n };
  };
  let worstSpoke = 0, worstSpokeK = 0;
  for (let i = 0; i < N; i++) {
    const start = r.centre[2 * i], end = r.centre[2 * i + 1], nextStart = r.centre[2 * ((i + 1) % N)];
    for (let t = 0; t <= 20; t++) {
      const v = t / 20;
      worstSpoke = Math.max(worstSpoke, ang(nrmAt(start, 1, v), nrmAt(end, 0, v)), ang(nrmAt(end, 1 - v, 1), nrmAt(nextStart, v, 1)));
      if (t >= 3 && t <= 17) {
        const ea = transverseK(end, 0, v, true), eb = transverseK(start, 1, v, true);
        const ma = transverseK(end, 1 - v, 1, false), mb = transverseK(nextStart, v, 1, false);
        worstSpokeK = Math.max(worstSpokeK,
          Math.abs(ea.k - (ea.n[0] * eb.n[0] + ea.n[1] * eb.n[1] + ea.n[2] * eb.n[2] < 0 ? -eb.k : eb.k)),
          Math.abs(ma.k - (ma.n[0] * mb.n[0] + ma.n[1] * mb.n[1] + ma.n[2] * mb.n[2] < 0 ? -mb.k : mb.k)));
      }
    }
  }
  assert.ok(worstSpoke < 1e-4, `G1 across every spoke (${worstSpoke.toExponential(2)}°)`);
  if (k >= 2) assert.ok(worstSpokeK < 1e-4, `G2 along the interior of every spoke, outside the extraordinary vertices (normal curvature jump ${worstSpokeK.toExponential(2)})`);
  assert.ok(worstOn < 1e-9, `the quads sit on the sheet edges (${worstOn.toExponential(2)})`);
  assert.ok(worstAng < 1e-4, `G1 along every sheet edge (${worstAng.toExponential(2)}°)`);
  if (k >= 2) assert.ok(worstK < 1e-9, `G2 along every sheet edge (normal curvature off by ${worstK.toExponential(2)})`);
  assert.ok(worstSeam < 1e-9, `the spokes are shared exactly (${worstSeam.toExponential(2)})`);
}

test('blendSurfaceJunction: a Y and an X of flat sheets — a guided cap of 2N ribbon quads, G2 to every sheet, seams exact', () => {
  for (const N of [3, 4]) {
    const parts = star(N);
    assertGuidedCap(parts, blendSurfaceJunction(parts, { continuity: 2 }), N);
  }
});

test('blendSurfaceJunction: unequal sheet setbacks meet at the directions’ center, not the corners’ biased mean', () => {
  const radii = [20, 45, 75];
  const parts = radii.map((R, i) => { const th = i * 120, a = th * Math.PI / 180; return {
    srf: sheetAt(R * Math.cos(a), R * Math.sin(a), th), edge: 'u0',
  }; });
  const r = blendSurfaceJunction(parts, { continuity: 2 });
  assertGuidedCap(parts, r, 3);
  assert.ok(Math.hypot(...r.spine[0]) < 1e-7, `the inward sheet directions meet at the origin (${r.spine[0]})`);
});

const bentAt = (cx, cy, dirDeg, w = 40, len = 50, bend = 0.01) => {
  const th = dirDeg * Math.PI / 180, ax = [Math.cos(th), Math.sin(th)], ay = [-Math.sin(th), Math.cos(th)];
  const n = 4, ctrlNet = [];
  for (let i = 0; i < n; i++) { const row = []; for (let j = 0; j < n; j++) { const a = (i / (n - 1)) * len, b = (j / (n - 1) - 0.5) * w; row.push([cx + ax[0] * a + ay[0] * b, cy + ax[1] * a + ay[1] * b, bend * a * a, 1]); } ctrlNet.push(row); }
  return { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 1, 1, 1, 1], ctrlNet };
};
const bentStar = () => [0, 120, 240].map((th) => ({ srf: bentAt(40 * Math.cos(th * Math.PI / 180), 40 * Math.sin(th * Math.PI / 180), th), edge: 'u0' }));

test('blendSurfaceJunction: G3 — the third cross derivative at the sheet is the sheet\'s (zero on a parabolic sheet); G2 leaves one', () => {
  const parts = bentStar();
  const third = (k) => {
    const r = blendSurfaceJunction(parts, { continuity: k });
    assert.ok(r.ok, r.reason);
    let worst = 0;
    for (const q of r.centre) for (let t = 1; t < 10; t++) {
      const u = t / 10, h = 1e-4;
      const a = surfaceDerivs2(q, u, 0), b = surfaceDerivs2(q, u, h);
      const d3 = Math.hypot((b.Svv[0] - a.Svv[0]) / h, (b.Svv[1] - a.Svv[1]) / h, (b.Svv[2] - a.Svv[2]) / h);
      worst = Math.max(worst, d3 / Math.pow(Math.hypot(...a.Sv), 3));
    }
    return worst;
  };
  const g2 = third(2), g3 = third(3);
  assert.ok(g3 < 1e-4, `G3: no third derivative of its own at the sheet (${g3.toExponential(2)})`);
  assert.ok(g2 > 1e-3, `G2 leaves one, so the G3 check can fail (${g2.toExponential(2)})`);
});

test('blendSurfaceJunction: bent sheets — the cap matches their normal curvature across the edge, not just the plane', () => {
  const parts = bentStar();
  for (const k of [1, 2, 3]) assertGuidedCap(parts, blendSurfaceJunction(parts, { continuity: k }), 3, k);
  const r = blendSurfaceJunction(parts, { continuity: 2 });
  // The free edges leave each corner along the sheet's side edge: their start
  // tangent is the sheet's cross direction there.
  for (let i = 0; i < 3; i++) {
    const F = r.rails[i];
    const [, T] = rationalCurveDerivs(F, 0, 1);
    const k = r.order[i], srf = parts[k].srf, v = r.reversed[k] ? 0 : 1;
    const E = surfaceDerivs2(srf, 0, v);
    const cosang = Math.abs((T[0] * E.Su[0] + T[1] * E.Su[1] + T[2] * E.Su[2]) / (Math.hypot(...T) * Math.hypot(...E.Su)));
    assert.ok(cosang > 1 - 1e-9, `free edge ${i} leaves along the side edge (cos ${cosang})`);
  }
});

test('blendSurfaceJunction: three tall sheets meet at a spine the full height — the ports keep the edges\' run', () => {
  const tall = (deg) => { const th = deg * Math.PI / 180, ax = [Math.cos(th), Math.sin(th)]; const ctrlNet = []; for (let i = 0; i < 2; i++) { const row = []; for (let j = 0; j < 2; j++) row.push([ax[0] * (20 + 10 * i), ax[1] * (20 + 10 * i), 200 * j, 1]); ctrlNet.push(row); } return { srf: { degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1], ctrlNet }, edge: 'u0' }; };
  const r = blendSurfaceJunction([tall(0), tall(120), tall(240)], { stations: 8 });
  assert.ok(r.ok, r.reason);
  assert.equal(r.faceUp, false, 'sheets edge-on to the junction take ribbons');
  assert.equal(r.centre.length, 3, 'one ribbon per port');
  assert.ok(Math.abs(r.spine[0][2]) < 1e-6 && Math.abs(r.spine[1][2] - 200) < 1e-6, `the spine runs the full height (${r.spine.map((p) => p[2].toFixed(1))})`);
  for (const arm of r.arms) { const top = surfacePoint(arm, 1, 1), bottom = surfacePoint(arm, 1, 0); assert.ok(Math.abs(Math.max(top[2], bottom[2]) - 200) < 1e-6 && Math.abs(Math.min(top[2], bottom[2])) < 1e-6, 'each port spans z 0..200'); }
});

test('blendSurfaceJunction: one sheet tilted 30° still builds as a cap; refusals name two edges and a line', () => {
  const parts = star(3, 40, 30);
  const r = blendSurfaceJunction(parts, { continuity: 2 });
  assertGuidedCap(parts, r, 3);
  assert.match(blendSurfaceJunction(parts.slice(0, 2)).reason, /three or more/);
  const line = [{ srf: sheetAt(-60, 0, 180), edge: 'u0' }, { srf: sheetAt(0, 0, 0), edge: 'u0' }, { srf: sheetAt(60, 0, 0), edge: 'u0' }];
  assert.ok(!blendSurfaceJunction(line).ok, 'three edges on one line refuse');
});

test('blendSurfaceJunction: sheets between lying flat and standing across still build one junction', () => {
  const tilted = [0, 120, 240].map((th) => { const a = th * Math.PI / 180; return {
    srf: sheetAt(40 * Math.cos(a), 40 * Math.sin(a), th, 45), edge: 'u0',
  }; });
  const r = blendSurfaceJunction(tilted);
  assert.equal(r.ok, true, r.reason);
  const mixed = blendSurfaceJunction(star(3, 40, 60));
  assert.equal(mixed.ok, true, mixed.reason);
});

test('blendSurfaceJunction: sheets sloping away from a cap lift its center — each spoke rises to it, no dimple', () => {
  const slopeSheet = (dirDeg, slopeDeg, w = 40, len = 50) => {
    const th = dirDeg * Math.PI / 180, sl = -slopeDeg * Math.PI / 180, cx = 40 * Math.cos(th), cy = 40 * Math.sin(th);
    const ax = [Math.cos(th) * Math.cos(sl), Math.sin(th) * Math.cos(sl), Math.sin(sl)], ay = [-Math.sin(th), Math.cos(th), 0];
    const ctrlNet = [];
    for (let i = 0; i < 4; i++) { const row = []; for (let j = 0; j < 4; j++) { const a = (i / 3) * len, b = (j / 3 - 0.5) * w; row.push([cx + ax[0] * a + ay[0] * b, cy + ax[1] * a + ay[1] * b, ax[2] * a, 1]); } ctrlNet.push(row); }
    return { srf: { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 1, 1, 1, 1], ctrlNet }, edge: 'u0' };
  };
  for (const slope of [19, 35]) {
    const r = blendSurfaceJunction([0, 120, 240].map((th) => slopeSheet(th, slope)), { continuity: 2 });
    assert.equal(r.ok, true, r.reason);
    for (let q = 0; q < r.centre.length; q += 2) {
      const f = r.centre[q], u1 = f.knotsU[f.knotsU.length - 1 - f.degU], v0 = f.knotsV[f.degV], v1 = f.knotsV[f.knotsV.length - 1 - f.degV];
      const z = Array.from({ length: 11 }, (_, i) => surfacePoint(f, u1, v0 + (v1 - v0) * i / 10)[2]);
      for (let i = 1; i < z.length; i++) assert.ok(z[i] >= z[i - 1] - 1e-9, `slope ${slope}: the spoke climbs all the way to the center (${z.map((x) => x.toFixed(2)).join(' ')})`);
      assert.ok(z[10] > 0.1 * slope, `slope ${slope}: the center sits above the edges (${z[10].toFixed(2)})`);
    }
  }
});

test('blendSurfaceJunction: the free edges of a cap join the side edges like a fillet — no hook, the same on every side', () => {
  const wide = (R, W) => [0, 120, 240].map((th) => { const a = th * Math.PI / 180, ax = [Math.cos(a), Math.sin(a), 0], ay = [-Math.sin(a), Math.cos(a), 0], cx = R * ax[0], cy = R * ax[1], ctrlNet = [];
    for (let i = 0; i < 4; i++) { const row = []; for (let j = 0; j < 4; j++) { const u = (i / 3) * 50, v = (j / 3 - 0.5) * W; row.push([cx + ax[0] * u + ay[0] * v, cy + ax[1] * u + ay[1] * v, 0, 1]); } ctrlNet.push(row); }
    return { srf: { degU: 3, degV: 3, knotsU: [0, 0, 0, 0, 1, 1, 1, 1], knotsV: [0, 0, 0, 0, 1, 1, 1, 1], ctrlNet }, edge: 'u0' }; });
  for (const [R, W] of [[40, 40], [25, 40], [40, 60]]) for (const bulge of [1, 1.4]) {
    const r = blendSurfaceJunction(wide(R, W), { continuity: 2, bulge });
    assert.equal(r.ok, true, r.reason);
    const peaks = r.rails.slice(0, 3).map((c) => {
      const lo = c.knots[c.degree], hi = c.knots[c.knots.length - 1 - c.degree];
      let flips = 0, prev = 0, peak = 0;
      for (let i = 1; i < 200; i++) {
        const [, d1, d2] = rationalCurveDerivs(c, lo + (hi - lo) * i / 200, 2);
        const k = (d1[0] * d2[1] - d1[1] * d2[0]) / Math.pow(Math.hypot(...d1), 3);
        if (prev && Math.sign(k) !== Math.sign(prev) && Math.abs(k) > 1e-6) flips++;
        if (Math.abs(k) > 1e-9) prev = k;
        peak = Math.max(peak, Math.abs(k));
      }
      assert.equal(flips, 0, `R${R} W${W} bulge ${bulge}: the free edge turns one way only`);
      return peak;
    });
    assert.ok(Math.max(...peaks) - Math.min(...peaks) < 1e-6, `R${R} W${W}: a symmetric Y has three identical free edges (${peaks.map((p) => p.toFixed(4))})`);
  }
});
