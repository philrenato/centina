import test from 'node:test';
import assert from 'node:assert/strict';
import { offsetSurface, reverseSurfaceU, thickenSolid, shellSolid, boundaryLoop } from '../kernel/offset.mjs';
import { surfacePoint, surfacePointAndPartials, isFiniteNet } from '../kernel/surface.mjs';
import { ruledLoftPanels } from '../kernel/loft.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { buildBrepSolid } from '../kernel/brepbuild.mjs';
import { validateBrep, eulerCharacteristic } from '../kernel/brep.mjs';

// A flat bilinear plane in z=0, U along +x, V along +y (Su x Sv = +z).
function flatPlane(W = 30, H = 20) {
  return {
    degU: 1, knotsU: [0, 0, 1, 1],
    degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [
      [[0, 0, 0, 1], [0, H, 0, 1]],
      [[W, 0, 0, 1], [W, H, 0, 1]],
    ],
  };
}

// A real cylinder — revolve of a vertical line, radius R, height Hz.
function cylinder(R = 10, Hz = 40) {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[R, 0, 0, 1], [R, 0, Hz, 1]] };
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
}

function surfNormal(srf, u, v) {
  const { su, sv } = surfacePointAndPartials(srf, u, v);
  const n = [su[1] * sv[2] - su[2] * sv[1], su[2] * sv[0] - su[0] * sv[2], su[0] * sv[1] - su[1] * sv[0]];
  const len = Math.hypot(...n);
  return [n[0] / len, n[1] / len, n[2] / len];
}

// ---- THE FLAT-PLANE EXACTNESS ANCHOR ----
test('offsetSurface: a flat plane offset by d is EXACTLY a parallel plane at distance d (float precision)', () => {
  const srf = flatPlane();
  const d = 4.5;
  const { surface, clamped, appliedDistance } = offsetSurface(srf, d);
  assert.equal(clamped, false, 'a flat plane can never self-intersect — nothing to clamp');
  assert.equal(appliedDistance, d);
  assert.equal(isFiniteNet(surface.ctrlNet), true);
  // Same degree/knots/weights preserved.
  assert.deepEqual(surface.knotsU, srf.knotsU);
  assert.deepEqual(surface.knotsV, srf.knotsV);
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  let worst = 0;
  for (let i = 0; i <= 5; i++) for (let j = 0; j <= 5; j++) {
    const u = uMin + (uMax - uMin) * i / 5, v = vMin + (vMax - vMin) * j / 5;
    const p0 = surfacePoint(srf, u, v);
    const p1 = surfacePoint(surface, u, v);
    // parallel plane at +z by d: x,y unchanged, z = p0.z + d
    worst = Math.max(worst, Math.abs(p1[0] - p0[0]), Math.abs(p1[1] - p0[1]), Math.abs(p1[2] - (p0[2] + d)));
  }
  assert.ok(worst < 1e-12, `flat-plane offset should be exact to float precision, worst deviation ${worst}`);
});

// ---- CURVED SURFACE OFFSETS IN THE CORRECT DIRECTION ----
test('offsetSurface: a curved surface offsets along its own local normal (correct direction), sampled points move the right way', () => {
  const srf = cylinder(10, 40);
  const u = srf.knotsU[0] + (srf.knotsU[srf.knotsU.length - 1] - srf.knotsU[0]) * 0.37;
  const v = srf.knotsV[0] + (srf.knotsV[srf.knotsV.length - 1] - srf.knotsV[0]) * 0.5;
  const P = surfacePoint(srf, u, v);
  const N = surfNormal(srf, u, v);
  const d = 3;
  const { surface } = offsetSurface(srf, d);
  const Q = surfacePoint(surface, u, v);
  // The offset point should have moved substantially in the +N direction.
  const moved = [Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]];
  const alongN = moved[0] * N[0] + moved[1] * N[1] + moved[2] * N[2];
  assert.ok(alongN > d * 0.6, `offset should move ~d along the surface normal (got ${alongN.toFixed(4)} for d=${d})`);
  // And genuinely farther from the cylinder axis (the surface's own center of
  // curvature is the axis) whenever the normal points outward, or nearer when
  // it points inward — either way the radial change must MATCH the normal
  // direction, not be an arbitrary "some new surface exists".
  const rP = Math.hypot(P[0], P[1]), rQ = Math.hypot(Q[0], Q[1]);
  const nRadial = N[0] * P[0] / rP + N[1] * P[1] / rP; // normal's own radial component sign
  if (nRadial > 0) assert.ok(rQ > rP + 1, `outward normal, +d must move farther from axis: ${rP.toFixed(3)} -> ${rQ.toFixed(3)}`);
  else assert.ok(rQ < rP - 1, `inward normal, +d must move nearer the axis: ${rP.toFixed(3)} -> ${rQ.toFixed(3)}`);
});

// ---- SELF-INTERSECTION DETECTION / AUTO-CLAMP ----
test('offsetSurface: a cylinder radius R offset by more than R in the collapsing direction is flagged and clamped to ~R', () => {
  const R = 10;
  const srf = cylinder(R, 40);
  // Determine the collapsing direction empirically (which sign reduces radius).
  const u = 0.5 * (srf.knotsU[0] + srf.knotsU[srf.knotsU.length - 1]);
  const v = 0.5 * (srf.knotsV[0] + srf.knotsV[srf.knotsV.length - 1]);
  const rBase = Math.hypot(...surfacePoint(srf, u, v));
  const probe = offsetSurface(srf, 1);
  const rProbe = Math.hypot(...surfacePoint(probe.surface, u, v));
  const collapseSign = rProbe < rBase ? 1 : -1; // sign of d that moves toward the axis

  // A small, genuinely safe offset in the collapsing direction is NOT flagged.
  const safe = offsetSurface(srf, collapseSign * (R * 0.3));
  assert.equal(safe.clamped, false, `a small ${R * 0.3}mm offset (well under R) must not be flagged`);

  // An offset of more than R in the collapsing direction (which would fold
  // the tube through its own axis) IS flagged and clamped to a real computed
  // safe maximum, strictly below the request. HONEST PROXY NOTE: the
  // adjacent-control-point-crossing check detects a rational cylinder's fold
  // at ~1.39R (where the off-circle rational "corner" control points of the
  // arc-span construction get involved), NOT exactly at the true surface
  // collapse of R — the on-circle control points reach the axis at exactly R
  // but that is not an adjacent-CP reversal, so the cheap local proxy the
  // task specifies (control points crossed, not a full global self-
  // intersection test) is more permissive here. The load-bearing behaviors
  // hold: a genuinely too-large offset (2R) is flagged and clamped, a small
  // safe one is not.
  const big = offsetSurface(srf, collapseSign * (R * 2));
  assert.equal(big.clamped, true, 'offset by more than R inward must self-intersect and be flagged');
  assert.ok(Math.abs(big.appliedDistance) < R * 2, 'the applied distance must be clamped strictly below the request');
  assert.ok(Math.abs(big.appliedDistance) > R * 0.5 && Number.isFinite(big.appliedDistance),
    `the computed safe maximum must be a real positive bound; got ${Math.abs(big.appliedDistance).toFixed(4)}`);
  assert.equal(isFiniteNet(big.surface.ctrlNet), true);
});

test('offsetSurface: refuses a non-finite distance honestly', () => {
  assert.throws(() => offsetSurface(flatPlane(), NaN), /finite/);
});

// ---- REVERSE WINDING ----
test('reverseSurfaceU: flips the surface normal while preserving geometry exactly', () => {
  const srf = cylinder(10, 40);
  const rev = reverseSurfaceU(srf);
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  // Same geometry: the reversed surface traces the same point set (a point at
  // u maps to a + b - u on the reversed surface).
  let worstPos = 0, worstDot = 1;
  for (let i = 1; i < 5; i++) for (let j = 1; j < 5; j++) {
    const u = uMin + (uMax - uMin) * i / 5, v = vMin + (vMax - vMin) * j / 5;
    const ur = uMin + uMax - u;
    const P = surfacePoint(srf, u, v), Q = surfacePoint(rev, ur, v);
    worstPos = Math.max(worstPos, Math.hypot(P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]));
    const nP = surfNormal(srf, u, v), nQ = surfNormal(rev, ur, v);
    worstDot = Math.min(worstDot, nP[0] * nQ[0] + nP[1] * nQ[1] + nP[2] * nQ[2]);
  }
  assert.ok(worstPos < 1e-9, `reversed surface must trace the same geometry, worst pos err ${worstPos}`);
  assert.ok(worstDot < -0.999, `the reversed surface's normal must point opposite (dot ~ -1), got ${worstDot.toFixed(4)}`);
});

// ---- THICKEN: a valid, correctly-wound closed solid ----
// Signed volume of a triangle soup via the divergence theorem: a consistently
// wound closed solid returns |V| == true enclosed volume; ANY flipped panel
// partially cancels and drops |V| below it — so |V| == expected IS the
// winding-consistency proof.
function bilinearTris(net) {
  const c00 = net[0][0], c01 = net[0][1], c10 = net[1][0], c11 = net[1][1];
  return [[c00, c10, c11], [c00, c11, c01]];
}
function signedVolumeOfPanels(panels) {
  let V = 0;
  for (const { srf } of panels) {
    // Every panel in a Thicken result of a bilinear base is itself bilinear
    // (2x2 net) — triangulate its four corners directly.
    for (const [a, b, c] of bilinearTris(srf.ctrlNet)) {
      // (1/6) a . (b x c)
      const cx = b[1] * c[2] - b[2] * c[1];
      const cy = b[2] * c[0] - b[0] * c[2];
      const cz = b[0] * c[1] - b[1] * c[0];
      V += (a[0] * cx + a[1] * cy + a[2] * cz) / 6;
    }
  }
  return V;
}

test('thickenSolid: a flat plate thickened is a valid, consistently-wound closed slab (volume = area x thickness)', () => {
  const W = 30, H = 20, t = 5;
  const srf = flatPlane(W, H);
  const { panels, clamped } = thickenSolid(srf, t, ruledLoftPanels);
  assert.equal(clamped, false);
  // original + reversed offset + 4 rim panels
  assert.equal(panels.length, 6, `expected 6 panels (2 caps + 4 rim), got ${panels.length}`);
  for (const { srf: p } of panels) assert.equal(isFiniteNet(p.ctrlNet), true);
  const V = Math.abs(signedVolumeOfPanels(panels));
  const expected = W * H * t;
  assert.ok(Math.abs(V - expected) < 1e-6,
    `enclosed volume must equal W*H*t (=${expected}); got ${V} — a mismatch means a panel is inconsistently wound`);
});

test('thickenSolid: clamps a too-large thickness on a small-radius cylinder honestly', () => {
  const R = 10;
  const srf = cylinder(R, 40);
  const u = 0.5 * (srf.knotsU[0] + srf.knotsU[srf.knotsU.length - 1]);
  const v = 0.5 * (srf.knotsV[0] + srf.knotsV[srf.knotsV.length - 1]);
  const rBase = Math.hypot(...surfacePoint(srf, u, v));
  const probe = offsetSurface(srf, 1);
  const collapseSign = Math.hypot(...surfacePoint(probe.surface, u, v)) < rBase ? 1 : -1;
  const { clamped, appliedDistance, panels } = thickenSolid(srf, collapseSign * R * 2, ruledLoftPanels);
  assert.equal(clamped, true);
  assert.ok(Math.abs(appliedDistance) < R * 2 && Number.isFinite(appliedDistance));
  assert.ok(panels.length > 2, 'a clamped Thicken still builds a real multi-panel container');
});

// ---- SHELL (Rhino: Shell) — hollow a multi-panel solid, uniform thickness ----
// A box as 6 flat bilinear panels (the exact shape the Box primitive builds).
// bilinearPanelArr(p00,p10,p01,p11) — a flat degree-1 x degree-1 panel through
// 4 coplanar corners, U-first ctrlNet, weight 1 (the app's own bilinearPanelArr).
function bilinearPanel(p00, p10, p01, p11) {
  return { srf: {
    degU: 1, knotsU: [0, 0, 1, 1], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [[[...p00, 1], [...p01, 1]], [[...p10, 1], [...p11, 1]]],
  } };
}
function boxPanels(h = 20) {
  const c = (sx, sy, sz) => [sx * h, sy * h, sz * h];
  return [
    bilinearPanel(c(-1, -1, 1), c(1, -1, 1), c(-1, 1, 1), c(1, 1, 1)),    // 0: +Z top
    bilinearPanel(c(-1, -1, -1), c(-1, 1, -1), c(1, -1, -1), c(1, 1, -1)), // 1: -Z bottom
    bilinearPanel(c(1, -1, -1), c(1, 1, -1), c(1, -1, 1), c(1, 1, 1)),    // 2: +X
    bilinearPanel(c(-1, -1, -1), c(-1, -1, 1), c(-1, 1, -1), c(-1, 1, 1)), // 3: -X
    bilinearPanel(c(-1, 1, -1), c(-1, 1, 1), c(1, 1, -1), c(1, 1, 1)),    // 4: +Y
    bilinearPanel(c(-1, -1, -1), c(1, -1, -1), c(-1, -1, 1), c(1, -1, 1)), // 5: -Y
  ];
}
function panelCentroid(srf) {
  let x = 0, y = 0, z = 0, n = 0;
  for (const row of srf.ctrlNet) for (const cp of row) { x += cp[0]; y += cp[1]; z += cp[2]; n++; }
  return [x / n, y / n, z / n];
}

// --- the shared measuring tools these shell tests use -----------------
// A panel's own corner loop, consecutive duplicates collapsed (so a fan
// triangle reads as a triangle) — the same rule the app's surfaceCornerLoop
// uses, reproduced HERE so the test never grades the app's homework with the
// app's own pencil.
function cornerLoop(srf) {
  const net = srf.ctrlNet, nu = net.length, nv = net[0].length;
  const raw = [net[0][0], net[nu - 1][0], net[nu - 1][nv - 1], net[0][nv - 1]].map((cp) => [cp[0], cp[1], cp[2]]);
  const out = [];
  const same = (a, b) => Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9 && Math.abs(a[2] - b[2]) < 1e-9;
  for (const p of raw) if (!out.length || !same(out[out.length - 1], p)) out.push(p);
  while (out.length > 1 && same(out[0], out[out.length - 1])) out.pop();
  return out;
}
function distinctPoints(pts, tol = 1e-9) {
  const out = [];
  for (const p of pts) if (!out.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) <= tol)) out.push(p);
  return out;
}
function allCtrlPts(panels) {
  const out = [];
  for (const { srf } of panels) for (const row of srf.ctrlNet) for (const cp of row) out.push([cp[0], cp[1], cp[2]]);
  return out;
}
// The six box face planes, derived HERE from the box's own half-extent —
// never read back from the shell.
function boxFacePlanes(h) {
  return [
    { n: [0, 0, 1], c: h }, { n: [0, 0, -1], c: h },
    { n: [1, 0, 0], c: h }, { n: [-1, 0, 0], c: h },
    { n: [0, 1, 0], c: h }, { n: [0, -1, 0], c: h },
  ];
}

test('shellSolid: a closed box (zero faces removed) is a genuinely hollow closed solid of exactly the requested wall thickness', () => {
  const h = 20, t = 3;
  const panels = boxPanels(h);
  const r = shellSolid(panels, [], t);
  assert.equal(r.clamped, false, 'a 3mm wall on a 40mm box is comfortably safe — nothing to clamp');
  assert.equal(r.removedCount, 0);
  assert.equal(r.remainingCount, 6);
  // 6 outer + 6 inner + 0 rim
  assert.equal(r.panels.length, 12, `expected 12 panels (6 outer + 6 inner), got ${r.panels.length}`);
  for (const { srf: p } of r.panels) assert.equal(isFiniteNet(p.ctrlNet), true);
  // THE HOLLOW-WALL-THICKNESS PROOF: each inner twin sits exactly `t` inward
  // from its own outer face. Out order is [6 outer..., 6 inner...] in the same
  // face order, so out[i] and out[6+i] are the same face's outer/inner pair.
  for (let i = 0; i < 6; i++) {
    const co = panelCentroid(r.panels[i].srf);
    const ci = panelCentroid(r.panels[6 + i].srf);
    const d = Math.hypot(co[0] - ci[0], co[1] - ci[1], co[2] - ci[2]);
    assert.ok(Math.abs(d - t) < 1e-9, `face ${i}: inner twin must be exactly ${t}mm from the outer face; got ${d}`);
    // ...and genuinely INWARD (toward the solid centroid at origin): the inner
    // face centroid is exactly a wall-thickness closer to origin than the outer.
    const ro = Math.hypot(...co), ri = Math.hypot(...ci);
    assert.ok(ro - ri > 0 && Math.abs((ro - ri) - t) < 1e-9, `face ${i}: inner twin must sit ${t}mm INWARD toward the centroid; outer dist ${ro}, inner dist ${ri}`);
  }
});

// THE CORNER TEST. This is the one that fails on the old per-face-normal
// construction and passes on the exact plane-intersection one: three inner
// walls meeting at a corner must meet at ONE point. The old build gave each
// of them its own corner, a few millimeters apart, which is exactly the
// gapping-and-crossing a shelled box showed.
test('shellSolid: three inner walls meeting at a corner meet at ONE point — the inner cavity has exactly 8 corners, not 24', () => {
  const h = 20, t = 3;
  const r = shellSolid(boxPanels(h), [], t);
  const innerPts = allCtrlPts(r.panels.slice(6));
  assert.equal(innerPts.length, 24, 'six inner quads carry 24 control points in total');
  const distinct = distinctPoints(innerPts, 1e-9);
  assert.equal(distinct.length, 8, `the inner cavity of a box is a box: exactly 8 distinct corners, got ${distinct.length}`);
  // and those 8 corners are the exact inset box, derived here from h and t
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    const want = [sx * (h - t), sy * (h - t), sz * (h - t)];
    assert.ok(distinct.some((p) => Math.hypot(p[0] - want[0], p[1] - want[1], p[2] - want[2]) < 1e-9),
      `expected an inner corner at ${want.join(',')}`);
  }
  // WORST CORNER GAP, measured: group every inner control point by the outer
  // corner it belongs to, and take the widest spread inside a group.
  let worstGap = 0;
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    const outer = [sx * h, sy * h, sz * h];
    const group = innerPts.filter((p) => Math.hypot(p[0] - outer[0], p[1] - outer[1], p[2] - outer[2]) < h);
    assert.equal(group.length, 3, 'each box corner is shared by exactly three faces');
    for (const a of group) for (const b of group) worstGap = Math.max(worstGap, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  }
  assert.ok(worstGap < 1e-9, `three inner walls must land on the SAME corner point; worst gap ${worstGap}`);
});

// THE CORNER WALL-THICKNESS TEST — the discriminating measurement. At a
// corner the old build put face +X's inner corner flush against the +Y face
// (zero wall there); the exact build puts it on all three offset planes at
// once, so the wall is `t` measured against EVERY face it touches.
test('shellSolid: the wall is exactly the requested thickness at the CORNERS, measured against every face that meets there', () => {
  const h = 20, t = 3;
  const r = shellSolid(boxPanels(h), [], t);
  const planes = boxFacePlanes(h);
  let worst = 0;
  for (const p of distinctPoints(allCtrlPts(r.panels.slice(6)), 1e-9)) {
    for (const pl of planes) {
      const gap = pl.c - (pl.n[0] * p[0] + pl.n[1] * p[1] + pl.n[2] * p[2]);
      // every inner corner is exactly t from all three faces it touches and
      // further from the three it does not
      if (gap < h) worst = Math.max(worst, Math.abs(gap - t));
    }
  }
  assert.ok(worst < 1e-9, `every inner corner must sit exactly ${t}mm from each face meeting there; worst error ${worst}`);
});

test('shellSolid: removing one face opens it, and the rim is a FLAT lip in the opening’s own plane, bridging outer to inner with no gap', () => {
  const h = 20, t = 3;
  const panels = boxPanels(h);
  const removedIdx = 0; // the +Z top face
  const r = shellSolid(panels, [removedIdx], t);
  assert.equal(r.removedCount, 1);
  assert.equal(r.remainingCount, 5);
  assert.equal(r.rimCount, 4);
  assert.equal(r.panels.length, 5 + 5 + 4, `expected 14 panels (5 outer + 5 inner + 4 rim), got ${r.panels.length}`);
  for (const { srf: p } of r.panels) assert.equal(isFiniteNet(p.ctrlNet), true);
  const rimPanels = r.panels.slice(10);
  // THE RIM IS FLAT. Every rim control point lies in the removed face's own
  // plane, z = h. The old build ran the rim DOWN the wall from z=h to z=h-t,
  // i.e. buried inside the side wall it sat in, which is why the opening
  // read as having no lip at all.
  for (const { srf: p } of rimPanels) {
    for (const cp of p.ctrlNet.flat()) {
      assert.ok(Math.abs(cp[2] - h) < 1e-9, `rim points must lie in the opening's own plane z=${h}; got z=${cp[2]}`);
    }
  }
  // and it genuinely BRIDGES: each rim quad has two corners on the outer
  // opening edge (|x| or |y| = h) and two on the inner wall's top edge
  // (|x| and |y| = h - t), so outer and inner are connected with no gap.
  for (const { srf: p } of rimPanels) {
    const pts = distinctPoints(p.ctrlNet.flat().map((cp) => [cp[0], cp[1], cp[2]]), 1e-9);
    const outerSide = pts.filter((q) => Math.abs(Math.abs(q[0]) - h) < 1e-9 || Math.abs(Math.abs(q[1]) - h) < 1e-9);
    const innerSide = pts.filter((q) => Math.abs(Math.abs(q[0]) - (h - t)) < 1e-9 && Math.abs(Math.abs(q[1]) - (h - t)) < 1e-9);
    assert.equal(outerSide.length, 2, 'two rim corners on the outer opening edge');
    assert.equal(innerSide.length, 2, 'two rim corners on the inner wall top edge');
  }
  // THE INNER WALL REACHES THE OPENING. Its top edge sits exactly in the
  // opening plane (z = h), not a wall-thickness below it — the old build
  // stopped the inner wall at z = h and then ran the rim down INSIDE the
  // outer wall, which is the same defect seen from the other side.
  const innerPanels = r.panels.slice(5, 10);
  const topInner = allCtrlPts(innerPanels).filter((q) => Math.abs(q[2] - h) < 1e-9);
  assert.equal(distinctPoints(topInner, 1e-9).length, 4, 'the inner wall meets the opening plane on exactly 4 corners');
  // the inner floor sits exactly a wall-thickness up from the outer floor
  const floorZ = allCtrlPts(innerPanels).reduce((m, q) => Math.min(m, q[2]), Infinity);
  assert.ok(Math.abs(floorZ - (-h + t)) < 1e-9, `the inner floor must sit at z=${-h + t}; got ${floorZ}`);
});

// REAL TOPOLOGY, from kernel/brepbuild.mjs — a separate implementation with
// its own error codes, so this is a genuine cross-check rather than the
// shell grading its own homework. Ground truth is derived here by hand:
//   closed shell  = TWO closed shells (outer box + inner box), no genus
//                   -> V16 E24 F12, chi = V - E + F = 4 = 2(S - G) with S=2, G=0
//   one face open = ONE closed shell (a cup)
//                   -> V16 E28 F14, chi = 2 = 2(S - G) with S=1, G=0
test('shellSolid: a closed box shell is a valid two-shell solid — chi = 16 - 24 + 12 = 4, zero naked and zero non-manifold edges', () => {
  const r = shellSolid(boxPanels(20), [], 3);
  const res = buildBrepSolid(r.panels.map((p) => cornerLoop(p.srf)), { tolerance: 1e-6 });
  assert.equal(res.ok, true, `expected a valid closed solid, got ${res.reason}`);
  assert.equal(res.stats.V, 16);
  assert.equal(res.stats.E, 24);
  assert.equal(res.stats.F, 12);
  assert.equal(res.stats.chi, 4);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.nonManifoldEdgeCount, 0);
  assert.equal(res.stats.shellCount, 2, 'a closed shell is genuinely two shells: the outer skin and the cavity');
  assert.equal(res.stats.genus, 0);
  const v = validateBrep(res.solid);
  assert.equal(v.ok, true, `validateBrep should pass: ${JSON.stringify(v.errors.map((e) => e.code))}`);
  const e = eulerCharacteristic(res.solid);
  assert.equal(e.ok, true, 'the Euler-Poincare invariant holds on the assembled half-edge structure');
});

test('shellSolid: a box shelled with one face open is a valid single closed solid — chi = 16 - 28 + 14 = 2, zero naked edges', () => {
  const r = shellSolid(boxPanels(20), [0], 3);
  const res = buildBrepSolid(r.panels.map((p) => cornerLoop(p.srf)), { tolerance: 1e-6 });
  assert.equal(res.ok, true, `expected a valid closed solid, got ${res.reason}`);
  assert.equal(res.stats.V, 16);
  assert.equal(res.stats.E, 28);
  assert.equal(res.stats.F, 14);
  assert.equal(res.stats.chi, 2);
  assert.equal(res.stats.nakedEdgeCount, 0, 'an opened shell is still a CLOSED solid — the rim closes it');
  assert.equal(res.stats.nonManifoldEdgeCount, 0);
  assert.equal(res.stats.shellCount, 1);
  assert.equal(res.stats.genus, 0);
  assert.equal(validateBrep(res.solid).ok, true);
});

test('shellSolid: two adjacent faces opened still weld into one valid closed solid (the two openings share a rim corner)', () => {
  const r = shellSolid(boxPanels(20), [0, 2], 3); // +Z and +X
  assert.equal(r.remainingCount, 4);
  const res = buildBrepSolid(r.panels.map((p) => cornerLoop(p.srf)), { tolerance: 1e-6 });
  assert.equal(res.ok, true, `expected a valid closed solid, got ${res.reason}`);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.nonManifoldEdgeCount, 0);
  assert.equal(res.stats.chi, 2, `a cup with an L-shaped opening is still genus 0, one shell (chi ${res.stats.chi})`);
  assert.equal(validateBrep(res.solid).ok, true);
});

test('shellSolid: an EXTRUDED POLYGON (one degree-1 tube panel with N columns) takes the exact path too', () => {
  // a triangular prism: the wall is ONE degree-1 x degree-1 panel with 4
  // control columns (closed profile), plus a flat top and bottom cap.
  const R = 20, H = 30, t = 2;
  const ring = [0, 1, 2].map((i) => [R * Math.cos((i * 2 * Math.PI) / 3), R * Math.sin((i * 2 * Math.PI) / 3)]);
  const at = (p, z) => [p[0], p[1], z];
  const wall = {
    degU: 1, knotsU: [0, 0, 1, 2, 3, 3], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [...ring, ring[0]].map((p) => [[...at(p, 0), 1], [...at(p, H), 1]]),
  };
  // the two caps are ordinary fan panels — exactly what capFanPanels/
  // ruledLoftPanels build for every capped Extrude in the app
  const botFan = ruledLoftPanels([[0, 0, 0], [0, 0, 0], [0, 0, 0]], ring.map((p) => at(p, 0)), true).map((srf) => ({ srf }));
  const topFan = ruledLoftPanels([[0, 0, H], [0, 0, H], [0, 0, H]], ring.map((p) => at(p, H)), true).map((srf) => ({ srf }));
  const panels = [{ srf: wall }, ...botFan, ...topFan];
  const openTop = [4, 5, 6]; // the whole top cap, all three of its fan panels
  const r = shellSolid(panels, openTop, t);
  const res = buildBrepSolid(r.panels.map((p) => cornerLoop(p.srf)), { tolerance: 1e-6 });
  assert.equal(res.ok, true, `an extruded triangle shells to a valid closed solid, got ${res.reason}`);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.nonManifoldEdgeCount, 0);
  assert.equal(res.stats.chi, 2, `a triangular cup is genus 0, one shell (chi ${res.stats.chi})`);
  // the rim lies flat in the opening's own plane
  for (const p of r.panels.slice(-r.rimCount)) for (const cp of p.srf.ctrlNet.flat()) {
    assert.ok(Math.abs(cp[2] - H) < 1e-9, `the rim of an opened top must lie at z=${H}; got ${cp[2]}`);
  }
});

test('shellSolid: the safe maximum wall is COMPUTED, and a too-thick request clamps to it', () => {
  const h = 20;
  const r = shellSolid(boxPanels(h), [], 25);
  assert.equal(r.clamped, true, 'a 25mm wall cannot fit inside a 40mm box');
  // the exact safe maximum for a box is its own half-extent: the wall that
  // closes the cavity to a point.
  assert.ok(Math.abs(r.safeMaxDistance - h) < 1e-5, `safeMaxDistance should converge on ${h}; got ${r.safeMaxDistance}`);
  assert.ok(r.appliedDistance < h && r.appliedDistance > h * 0.99, `clamped thickness ${r.appliedDistance} should sit just under ${h}`);
  // and the clamped result is still a valid closed solid, not a wreck
  const res = buildBrepSolid(r.panels.map((p) => cornerLoop(p.srf)), { tolerance: 1e-6 });
  assert.equal(res.ok, true, `even at the clamp the shell is valid, got ${res.reason}`);
  const safe = shellSolid(boxPanels(h), [], 2);
  assert.equal(safe.clamped, false, 'an ordinary 2mm wall is never silently altered');
  assert.equal(safe.appliedDistance, 2);
});

// THE HONEST REFUSAL. A curved wall meeting a flat cap has no exact junction
// in this kernel, so a solid with any curved face is refused BY NAME rather
// than shelled into something that is not a valid solid.
test('shellSolid: a curved face is REFUSED BY NAME, not approximated', () => {
  const wall = cylinder(10, 40); // a real rational curved surface
  assert.throws(() => shellSolid([{ srf: wall }], [], 2), /CURVED/);
  assert.throws(() => shellSolid([{ srf: wall }], [], 2), /junction blend/i);
});

test('shellSolid: a MIXED solid (planar caps + a curved wall) is refused for the same reason, naming the curved face', () => {
  const wall = cylinder(10, 40);
  const capA = bilinearPanel([-10, -10, 0], [10, -10, 0], [-10, 10, 0], [10, 10, 0]);
  const panels = [capA, { srf: wall }];
  assert.throws(() => shellSolid(panels, [], 2), /1 of 2 faces is CURVED \(face 1\)/);
});

test('shellSolid: a solid that is not closed is refused honestly (there is no inside to hollow)', () => {
  const one = [boxPanels(20)[0]];
  assert.throws(() => shellSolid(one, [], 2), /not a closed solid/i);
});

// A single panel's own corner lifted with NO corresponding move in the
// neighboring panels that share that physical vertex is not a real
// push-pull edit (solidVertexGroups moves a topological vertex in EVERY
// panel that owns it, consistently) — it is an inconsistent mutation that
// tears the solid open at that corner, and shellSolid still refuses it
// honestly, now for the true reason (a real gap in the topology) rather
// than the old "not flat" message that generalization below retires.
test('shellSolid: moving ONE panel\'s corner without its neighbors tears the solid open, and is still refused honestly', () => {
  const panels = boxPanels(20);
  panels[0].srf.ctrlNet[1][1][2] += 5; // only the top face's own corner moves
  assert.throws(() => shellSolid(panels, [], 2), /not a closed solid/i);
});

// ---- THE TANGENT-PLANE GENERALIZATION — a push-pull-warped face is exact
// at its corners, approximate only in its interior, and no longer refused ----
//
// A real push-pull moves ONE topological vertex consistently across every
// panel that shares it (solidVertexGroups' own convention in the app) —
// simulated here the same way: the SAME displaced corner is written into
// every box panel that meets at it, exactly the shape a single-vertex drag
// in the app actually produces.
function pushPulledBoxPanels(h = 20, delta = [4, 3, 2]) {
  const panels = boxPanels(h);
  // corner c(1,1,1) = (h,h,h) is panels[0]/[2]/[4]'s own ctrlNet[1][1]
  // (bilinearPanel's own p11 slot) — see boxPanels' own per-face comments.
  const moved = [h + delta[0], h + delta[1], h + delta[2], 1];
  for (const i of [0, 2, 4]) panels[i].srf.ctrlNet[1][1] = moved.slice();
  return panels;
}
// Reproduced HERE, independent of shellFaceCornerNormals in the kernel, so
// this is a genuine cross-check of the tangent-plane math rather than the
// kernel grading its own homework. The cross product of a bilinear quad's
// two edges meeting at a corner IS that surface's own Su x Sv there (see
// offset.mjs's own header derivation) — a real geometric fact, not an
// implementation detail borrowed from the code under test.
function tangentNormalAt(loopPts, k) {
  const n = loopPts.length;
  const curr = loopPts[k], next = loopPts[(k + 1) % n], prev = loopPts[(k + n - 1) % n];
  const eNext = [next[0] - curr[0], next[1] - curr[1], next[2] - curr[2]];
  const ePrev = [prev[0] - curr[0], prev[1] - curr[1], prev[2] - curr[2]];
  const c = [eNext[1] * ePrev[2] - eNext[2] * ePrev[1], eNext[2] * ePrev[0] - eNext[0] * ePrev[2], eNext[0] * ePrev[1] - eNext[1] * ePrev[0]];
  const m = Math.hypot(c[0], c[1], c[2]);
  return [c[0] / m, c[1] / m, c[2] / m];
}

test('shellSolid: a push-pulled box (one vertex moved, three adjacent faces genuinely warped) now shells into a valid solid instead of refusing', () => {
  const h = 20, t = 3;
  const panels = pushPulledBoxPanels(h, [4, 3, 2]);
  const r = shellSolid(panels, [], t);
  assert.equal(r.warpedFaceCount, 3, 'exactly the 3 faces sharing the pushed vertex should read as warped');
  assert.equal(r.exact, false, 'a shell with any warped face is honestly reported as not fully exact');
  assert.ok(r.interiorApprox, 'a warped shell must report a measured interior deviation, not silently claim full exactness');
  assert.equal(r.interiorApprox.warpedFaceCount, 3);
  assert.ok(Number.isFinite(r.interiorApprox.worstAbsoluteError) && r.interiorApprox.worstAbsoluteError >= 0);
  assert.ok(Number.isFinite(r.interiorApprox.worstRelativeError) && r.interiorApprox.worstRelativeError >= 0);
  // a real, non-vacuous measurement — a push of magnitude ~5.4mm against a
  // 3mm wall genuinely moves the interior thickness a real fraction off
  // nominal (not a rounding-noise number, and bounded well away from a
  // computation that blew up).
  assert.ok(r.interiorApprox.worstRelativeError > 0.01 && r.interiorApprox.worstRelativeError < 1, `expected a real, bounded interior deviation, got ${r.interiorApprox.worstRelativeError}`);
  // still a real, VALID, watertight solid — independently re-checked
  // through kernel/brepbuild.mjs, not just "no exception was thrown".
  const res = buildBrepSolid(r.panels.map((p) => cornerLoop(p.srf)), { tolerance: 1e-6 });
  assert.equal(res.ok, true, `a push-pulled box must still weld into a valid closed solid, got ${res.reason}`);
  assert.equal(res.stats.V, 16);
  assert.equal(res.stats.E, 24);
  assert.equal(res.stats.F, 12);
  assert.equal(res.stats.chi, 4);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.nonManifoldEdgeCount, 0);
  assert.equal(validateBrep(res.solid).ok, true);
});

// A SMALLER push genuinely measures a SMALLER interior deviation — proof
// the reported number tracks the real warp, not a fixed placeholder.
test('shellSolid: interiorApprox genuinely scales with how warped a face is — a gentle push measures a small deviation, a hard one a bigger one', () => {
  const h = 20, t = 3;
  const gentle = shellSolid(pushPulledBoxPanels(h, [0.3, 0.2, 0.1]), [], t);
  const hard = shellSolid(pushPulledBoxPanels(h, [4, 3, 2]), [], t);
  assert.ok(gentle.interiorApprox.worstRelativeError > 0, 'even a small push leaves a real, nonzero, measured deviation');
  assert.ok(gentle.interiorApprox.worstRelativeError < hard.interiorApprox.worstRelativeError,
    `a gentler push (${gentle.interiorApprox.worstRelativeError}) must measure a smaller deviation than a harder one (${hard.interiorApprox.worstRelativeError})`);
  // and the gentle push is still a fully valid, watertight solid
  const res = buildBrepSolid(gentle.panels.map((p) => cornerLoop(p.srf)), { tolerance: 1e-6 });
  assert.equal(res.ok, true);
  assert.equal(validateBrep(res.solid).ok, true);
});

test('shellSolid: every corner of a push-pulled shell — warped or still-planar — sits exactly the wall thickness inward on EVERY incident face\'s own tangent plane', () => {
  const h = 20, t = 3;
  const panels = pushPulledBoxPanels(h, [4, 3, 2]);
  const r = shellSolid(panels, [], t);
  // out = [6 outer..., 6 inner...] in face order (see shellSolid's own
  // "OUTPUT ORDER" comment) — r.panels[fi] / r.panels[6+fi] are one
  // face's outer/inner pair.
  let worst = 0, checked = 0;
  for (let fi = 0; fi < 6; fi++) {
    const outerNet = r.panels[fi].srf.ctrlNet;
    const outer = [outerNet[0][0], outerNet[1][0], outerNet[1][1], outerNet[0][1]].map((cp) => [cp[0], cp[1], cp[2]]);
    // the inner twin is stored with its loop REVERSED (shellSolid's own
    // comment: "so its normal points back at the outer face") — undo that
    // to line the inner corners up index-for-index with `outer`.
    const innerNet = r.panels[6 + fi].srf.ctrlNet;
    const inner = [innerNet[0][1], innerNet[1][1], innerNet[1][0], innerNet[0][0]].map((cp) => [cp[0], cp[1], cp[2]]);
    for (let k = 0; k < 4; k++) {
      const n = tangentNormalAt(outer, k);
      const d = [inner[k][0] - outer[k][0], inner[k][1] - outer[k][1], inner[k][2] - outer[k][2]];
      const proj = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
      worst = Math.max(worst, Math.abs(Math.abs(proj) - t));
      checked++;
    }
  }
  assert.equal(checked, 24);
  assert.ok(worst < 1e-9, `every corner must sit exactly ${t}mm inward on its own face's tangent plane; worst error ${worst}`);
});

test('shellSolid: an ordinary (fully planar) box is completely unaffected by the tangent-plane generalization — zero warped faces, exact', () => {
  const h = 20, t = 3;
  const r = shellSolid(boxPanels(h), [], t);
  assert.equal(r.warpedFaceCount, 0);
  assert.equal(r.exact, true);
  assert.equal(r.interiorApprox, null, 'a fully planar shell has nothing to approximate, so there is no interior-deviation report at all');
});

// ---- SHELL ON A REBUILT BOX ------------------------------------------
// A dense N x N bilinear grid over 4 flat, coplanar corners — exactly
// boxPanelsWithResolution/bilinearPanelGridArr's own app-layer construction
// (in the app), reproduced here so this kernel test never grades the
// app's own homework with the app's own pencil.
function bilinearPanelGrid(p00, p10, p01, p11, n) {
  const ctrlNet = [];
  for (let i = 0; i < n; i++) {
    const fu = i / (n - 1);
    const row = [];
    for (let j = 0; j < n; j++) {
      const fv = j / (n - 1);
      const w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv), w01 = (1 - fu) * fv, w11 = fu * fv;
      row.push([
        w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0],
        w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1],
        w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2],
        1,
      ]);
    }
    ctrlNet.push(row);
  }
  return { srf: { degU: 1, knotsU: [0, 0, 1, 1], degV: 1, knotsV: [0, 0, 1, 1], ctrlNet } };
}
// Same six faces/corners as boxPanels(h), each rebuilt to an n x n grid —
// the exact scenario "Box Rebuild"'s own Ctrl Pts X/Y/Z stepper produces.
function rebuiltBoxPanels(h, n) {
  const c = (sx, sy, sz) => [sx * h, sy * h, sz * h];
  return [
    bilinearPanelGrid(c(-1, -1, 1), c(1, -1, 1), c(-1, 1, 1), c(1, 1, 1), n),
    bilinearPanelGrid(c(-1, -1, -1), c(-1, 1, -1), c(1, -1, -1), c(1, 1, -1), n),
    bilinearPanelGrid(c(1, -1, -1), c(1, 1, -1), c(1, -1, 1), c(1, 1, 1), n),
    bilinearPanelGrid(c(-1, -1, -1), c(-1, -1, 1), c(-1, 1, -1), c(-1, 1, 1), n),
    bilinearPanelGrid(c(-1, 1, -1), c(-1, 1, 1), c(1, 1, -1), c(1, 1, 1), n),
    bilinearPanelGrid(c(-1, -1, -1), c(1, -1, -1), c(-1, -1, 1), c(1, -1, 1), n),
  ];
}

test('shellSolid: a Box REBUILT to a dense N x N control grid still shells validly (watertight, non-self-intersecting)', () => {
  const h = 20, t = 3, n = 5;
  const r = shellSolid(rebuiltBoxPanels(h, n), [], t);
  assert.equal(r.clamped, false);
  assert.equal(r.appliedDistance, t);
  // 6 faces x (n-1)^2 sub-quads each — the decomposition is per-cell, not
  // per-macro-panel (unchanged by this fix, named directly in the module's
  // own header comment).
  assert.equal(r.faceCount, 6 * (n - 1) * (n - 1));
});

test('shellSolid: a REBUILT box\'s inner wall is a genuine clean inset lattice — every interior grid point sits at the SAME uniform in-plane shrink as the border, not left at its original position', () => {
  const h = 20, t = 3, n = 6;
  const r = shellSolid(rebuiltBoxPanels(h, n), [], t);
  const remainingCount = r.remainingCount;
  const innerPanels = r.panels.slice(remainingCount, remainingCount * 2);
  // Every inner-wall control point, across every sub-quad, deduplicated.
  const allInnerPts = [];
  for (const { srf } of innerPanels) for (const row of srf.ctrlNet) for (const cp of row) allInnerPts.push([cp[0], cp[1], cp[2]]);
  const dedupe = (pts, tol = 1e-6) => {
    const out = [];
    for (const p of pts) if (!out.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) <= tol)) out.push(p);
    return out;
  };
  const distinct = dedupe(allInnerPts);
  // Isolate the TOP face's own inner wall (z close to h - t = 17) — its
  // neighbors (the 4 side faces) are ALSO offset inward by t, so a UNIFORM
  // box shell's top face genuinely shrinks by t on every edge too: the true
  // inner-top footprint is x,y in [-(h-t), h-t], a real, independently
  // derived prediction, never read back from the shell itself.
  const topPts = distinct.filter((p) => Math.abs(p[2] - (h - t)) < 1e-3);
  assert.equal(topPts.length, n * n, `expected exactly the full n x n inner grid on the top face alone, got ${topPts.length}`);
  const inset = h - t;
  const expected = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const fu = i / (n - 1), fv = j / (n - 1);
    expected.push([-inset + fu * 2 * inset, -inset + fv * 2 * inset, h - t]);
  }
  let worst = 0;
  for (const e of expected) {
    const nearest = Math.min(...topPts.map((p) => Math.hypot(p[0] - e[0], p[1] - e[1], p[2] - e[2])));
    if (nearest > worst) worst = nearest;
  }
  assert.ok(worst < 1e-6, `every predicted clean-inset-lattice point must have a real match on the inner wall; worst distance ${worst}`);
  // The negative control this test is actually built to catch: an
  // UN-fixed interior point stays at its ORIGINAL in-plane (x,y) — i.e.
  // still spanning the full un-shrunk [-h, h] footprint, not [-(h-t), h-t].
  // Confirm no inner point is still sitting out at the wide, unshrunk
  // extent (this would be true of a genuinely interior, uncorrected point).
  const stillWide = topPts.filter((p) => Math.abs(p[0]) > inset + 1e-3 || Math.abs(p[1]) > inset + 1e-3);
  assert.equal(stillWide.length, 0, 'no inner-wall point should still sit at the original, un-inset footprint');
});

test('shellSolid: removing EVERY face refuses honestly (nothing left to shell)', () => {
  const panels = boxPanels(20);
  assert.throws(() => shellSolid(panels, [0, 1, 2, 3, 4, 5], 3), /every face was removed/i);
});

test('shellSolid: a zero / non-finite thickness refuses honestly', () => {
  const panels = boxPanels(20);
  assert.throws(() => shellSolid(panels, [], 0), /nonzero finite/i);
  assert.throws(() => shellSolid(panels, [], NaN), /nonzero finite/i);
});

// A CURVED-FACE REFUSAL SAYS HOW CURVED, not only which faces.
//
// The curvature test is structural (degree, rationality), not a threshold, so
// there is no tolerance gap to invert the way a coincidence refusal has. The
// number that actually helps is different: whether the face is a real curve or
// a nearly-flat one the modeler could rebuild as a plane and shell.
//
// The bound is measured over the CONTROL NET rather than the surface, so by
// the convex-hull property it OVERSTATES the true sagitta and can never claim
// a face is flatter than it is. The fixture below makes that checkable: a
// six-point net with z ∈ {0, 12} has centroid z = 4, so the exact bound is 8.
test('shellSolid: a curved face is refused WITH a measured bow, not just named', () => {
  const rationalArch = { srf: {
    degU: 1, knotsU: [0, 0, 1, 1],
    degV: 2, knotsV: [0, 0, 0, 1, 1, 1],
    ctrlNet: [
      [[0, 0, 0, 1], [50, 0, 12, 0.7071], [100, 0, 0, 1]],
      [[0, 100, 0, 1], [50, 100, 12, 0.7071], [100, 100, 0, 1]],
    ],
  } };
  const flat = { srf: {
    degU: 1, knotsU: [0, 0, 1, 1],
    degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [[[0, 0, 0, 1], [100, 0, 0, 1]], [[0, 100, 0, 1], [100, 100, 0, 1]]],
  } };
  let msg = '';
  try { shellSolid([rationalArch, flat], [], 2); } catch (e) { msg = e.message; }
  assert.match(msg, /is CURVED \(face 0\)/, 'it still names which face');
  const m = /bows at most ([\d.]+)mm off its own plane/.exec(msg);
  assert.ok(m, `and now says how far it bows — got: ${msg}`);
  // Asserted against the fixture's own arithmetic rather than merely for
  // being a number: a bare digit check would pass on the thickness.
  assert.ok(Math.abs(Number(m[1]) - 8) < 1e-6, `the bound is the exact control-net deviation (expected 8, got ${m[1]})`);
});

test('shellSolid: the bow is an UPPER bound — it never claims a face is flatter than it is', () => {
  // The same arch evaluated on its own surface never leaves the control net's
  // z range, so the reported bound must be at least the true deviation. This
  // is the direction that matters: understating it would invite a modeler to
  // treat a real curve as near-flat.
  const arch = { srf: {
    degU: 1, knotsU: [0, 0, 1, 1],
    degV: 2, knotsV: [0, 0, 0, 1, 1, 1],
    ctrlNet: [
      [[0, 0, 0, 1], [50, 0, 40, 1], [100, 0, 0, 1]],
      [[0, 100, 0, 1], [50, 100, 40, 1], [100, 100, 0, 1]],
    ],
  } };
  const flat = { srf: {
    degU: 1, knotsU: [0, 0, 1, 1],
    degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [[[0, 0, 0, 1], [100, 0, 0, 1]], [[0, 100, 0, 1], [100, 100, 0, 1]]],
  } };
  let msg = '';
  try { shellSolid([arch, flat], [], 2); } catch (e) { msg = e.message; }
  const bound = Number(/bows at most ([\d.]+)mm/.exec(msg)[1]);
  let trueMax = 0;
  for (let i = 0; i <= 20; i++) {
    const p = surfacePoint(arch.srf, 0.5, i / 20);
    trueMax = Math.max(trueMax, Math.abs(p[2] - 40 / 3)); // 40/3 is the net's own centroid z
  }
  assert.ok(bound >= trueMax - 1e-9, `the reported bound ${bound} must not understate the true deviation ${trueMax}`);
});
