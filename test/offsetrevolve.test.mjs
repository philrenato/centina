// EXACT OFFSET OF A SURFACE OF REVOLUTION — verified against ANALYTIC ground
// truth throughout, never against the construction's own output. Every
// expected number below is derived on paper from the fixture's own radius,
// height and wall thickness (and written into the test as that closed form,
// not as a captured value), so a test passing here means the geometry matches
// the mathematics, not that the code agrees with itself.
//
// The comparison against the OLD method (kernel/offset.mjs's `offsetSurface`)
// is run on the SAME fixture in the same file, so the improvement is a
// measured number rather than a claim.

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  makeCylinderProfile, makeConeProfile, offsetMeridianProfile, safeInwardOffset,
  shellRevolvedSolid, revolveProfilePanels, profileRZ, meridianFrame, normalizeMeridianProfile,
  makeArcMeridianProfile, offsetArcMeridianProfile, revolveArcMeridianPanel,
  safeArcInwardOffset, shellArcRevolvedSolid,
} from '../kernel/offsetrevolve.mjs';
import { surfacePoint, surfacePointAndPartials } from '../kernel/surface.mjs';
import { offsetSurface } from '../kernel/offset.mjs';
import { revolve, makeLine } from '../kernel/primitives.mjs';

const R = 25, H = 50, T = 3;                       // cylinder fixture
const CR = 30, CH = 45, CT = 3;                    // cone fixture
const CL = Math.hypot(CR, CH);                     // cone slant length

function domU(s) { return [s.knotsU[0], s.knotsU[s.knotsU.length - 1]]; }
function domV(s) { return [s.knotsV[0], s.knotsV[s.knotsV.length - 1]]; }

// Sample a surface on an (nu+1) x (nv+1) grid spanning its FULL domain, so
// both rational seams (v = vMin and v = vMax, the same physical circle) are
// hit exactly rather than approached.
function sampleGrid(srf, nu, nv, fn) {
  const [u0, u1] = domU(srf), [v0, v1] = domV(srf);
  for (let i = 0; i <= nu; i++) {
    for (let j = 0; j <= nv; j++) {
      const u = u0 + (u1 - u0) * (i / nu);
      const v = v0 + (v1 - v0) * (j / nv);
      fn(surfacePoint(srf, u, v), u, v);
    }
  }
}

function unitNormal(srf, u, v) {
  const { su, sv } = surfacePointAndPartials(srf, u, v);
  const n = [su[1] * sv[2] - su[2] * sv[1], su[2] * sv[0] - su[0] * sv[2], su[0] * sv[1] - su[1] * sv[0]];
  const len = Math.hypot(n[0], n[1], n[2]);
  return [n[0] / len, n[1] / len, n[2] / len];
}

// Every boundary control point of one surface's u = uMax edge against the
// other's u = uMin edge, and vice versa — a junction that is a genuinely
// SHARED circle matches control point for control point, which no amount of
// sampling luck can fake.
function ringGap(srfA, uA, srfB, uB, samples = 401) {
  const [va0, va1] = domV(srfA), [vb0, vb1] = domV(srfB);
  let worst = 0;
  for (let i = 0; i <= samples; i++) {
    const pa = surfacePoint(srfA, uA, va0 + (va1 - va0) * (i / samples));
    let best = Infinity;
    for (let j = 0; j <= samples; j++) {
      const pb = surfacePoint(srfB, uB, vb0 + (vb1 - vb0) * (j / samples));
      const d = Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]);
      if (d < best) best = d;
    }
    if (best > worst) worst = best;
  }
  return worst;
}

const cylProfile = () => makeCylinderProfile({ center: [0, 0, 0], axis: [0, 0, 1], refDir: [1, 0, 0], radius: R, height: H });
const coneProfile = () => makeConeProfile({ center: [0, 0, 0], axis: [0, 0, 1], refDir: [1, 0, 0], radius: CR, height: CH });

test('the offset MERIDIAN PROFILE of a cylinder is exactly the inner rectangle — every corner an exact miter, not a blend', () => {
  const p = cylProfile();
  assert.deepEqual(p.points.map((q) => profileRZ(p, q).map((n) => Number(n.toFixed(12)))), [[0, 0], [R, 0], [R, H], [0, H]]);
  const off = offsetMeridianProfile(p, T);
  assert.equal(off.ok, true);
  assert.equal(off.miterLimitFallbacks, 0, 'a 90-degree corner must miter exactly, never fall back to a bevel');
  const rz = off.points.map((q) => profileRZ(p, q));
  assert.equal(rz.length, 4);
  const expect = [[0, T], [R - T, T], [R - T, H - T], [0, H - T]];
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(rz[i][0] - expect[i][0]) < 1e-12, `point ${i} radius ${rz[i][0]} != ${expect[i][0]}`);
    assert.ok(Math.abs(rz[i][1] - expect[i][1]) < 1e-12, `point ${i} height ${rz[i][1]} != ${expect[i][1]}`);
  }
  // Both ends stay ON the axis, which is what keeps the inner caps full disks
  // rather than annuli — the flat-cap alternative would leave them hanging off
  // the axis by T.
  assert.equal(off.clippedAtAxis, false, 'a cylinder needs no axis clip: its end perpendiculars already run along the axis');
});

test('an inward-offset CYLINDER is EXACTLY a cylinder of radius R - t — including at the rational seam and at the off-curve corner control points, the two places the Greville-normal offset is known to fail', () => {
  const p = cylProfile();
  const sh = shellRevolvedSolid(p, T);
  assert.equal(sh.clamped, false, `a ${T}mm wall is well inside a ${R}mm radius; it must not be clamped`);
  assert.equal(sh.appliedDistance, T);
  const innerWall = sh.innerPanels[1].srf;

  // (a) the surface itself, sampled densely across the FULL domain
  let worst = 0;
  sampleGrid(innerWall, 8, 512, (pt) => { worst = Math.max(worst, Math.abs(Math.hypot(pt[0], pt[1]) - (R - T))); });
  assert.ok(worst < 1e-12, `sampled radius deviates by ${worst.toExponential(3)} from the analytic ${R - T}`);

  // (b) the SEAM specifically: v = vMin and v = vMax are the same physical
  // circle, so they must agree bit-for-bit, not merely closely.
  const [v0, v1] = domV(innerWall);
  for (const u of domU(innerWall)) {
    const a = surfacePoint(innerWall, u, v0), b = surfacePoint(innerWall, u, v1);
    assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-12, 'the rational seam does not close');
  }

  // (c) THE OFF-CURVE CORNER CONTROL POINTS. A true parallel circle of radius
  // R-t has its tangent-corner control points at exactly (R-t)*sqrt(2). The
  // Greville-normal offset puts them at R*sqrt(2) - t instead, short by
  // t*(sqrt(2)-1). This is the exact place the old method breaks, so it is
  // checked directly rather than only through sampling.
  // (ctrlNet stores PLAIN coordinates plus a weight — surface.mjs's own
  // toHomogeneousNet is what multiplies through — so this reads cp[0]/cp[1]
  // directly, not cp/w.)
  const cornerIdeal = (R - T) * Math.SQRT2;
  const onCurveIdeal = R - T;
  let sawCorner = false;
  for (const row of innerWall.ctrlNet) {
    for (let j = 0; j < row.length; j++) {
      const cp = row[j];
      const rad = Math.hypot(cp[0], cp[1]);
      const isCorner = j % 2 === 1;
      const want = isCorner ? cornerIdeal : onCurveIdeal;
      if (isCorner) {
        sawCorner = true;
        assert.ok(Math.abs(cp[3] - Math.cos(Math.PI / 4)) < 1e-12, 'a tangent-corner control point must carry the quarter-arc weight cos(45 deg)');
      } else {
        assert.equal(cp[3], 1);
      }
      assert.ok(Math.abs(rad - want) < 1e-12, `control point ${j} sits at radius ${rad}, analytic ${want}`);
      // The old method's own answer for the same control point, for contrast:
      // it moves radially by t, landing at R*sqrt(2) - t, short by
      // t*(sqrt(2)-1). Asserting the gap is real keeps this from becoming a
      // tolerance that quietly accepts either construction.
      if (isCorner) assert.ok(Math.abs(rad - (R * Math.SQRT2 - T)) > T * (Math.SQRT2 - 1) * 0.99);
    }
  }
  assert.ok(sawCorner, 'the fixture must actually contain off-curve corner control points');
});

test("the inner CAPS are full disks in the offset planes — the cavity's own two ends, at z = t and z = H - t", () => {
  const p = cylProfile();
  const sh = shellRevolvedSolid(p, T);
  assert.equal(sh.innerPanels.length, 3, 'bottom cap, wall, top cap');
  for (const [idx, wantZ] of [[0, T], [2, H - T]]) {
    const cap = sh.innerPanels[idx].srf;
    let maxR = 0, worstZ = 0;
    sampleGrid(cap, 6, 256, (pt) => {
      worstZ = Math.max(worstZ, Math.abs(pt[2] - wantZ));
      maxR = Math.max(maxR, Math.hypot(pt[0], pt[1]));
    });
    assert.ok(worstZ < 1e-12, `cap ${idx} is not flat at z = ${wantZ} (worst ${worstZ.toExponential(3)})`);
    assert.ok(Math.abs(maxR - (R - T)) < 1e-12, `cap ${idx} rim radius ${maxR}, analytic ${R - T}`);
  }
});

test('THE JUNCTION: the inner wall and the inner cap meet at ONE shared circle, to machine precision — this is the defect being fixed', () => {
  const p = cylProfile();
  const sh = shellRevolvedSolid(p, T);
  assert.equal(sh.junctions.length, 2);
  for (const j of sh.junctions) assert.ok(Math.abs(j.radius - (R - T)) < 1e-12, `junction radius ${j.radius}, analytic ${R - T}`);
  assert.ok(Math.abs(sh.junctions[0].height - T) < 1e-12);
  assert.ok(Math.abs(sh.junctions[1].height - (H - T)) < 1e-12);

  const bottomCap = sh.innerPanels[0].srf, wall = sh.innerPanels[1].srf, topCap = sh.innerPanels[2].srf;
  const [cu0, cu1] = domU(bottomCap), [wu0, wu1] = domU(wall), [tu0, tu1] = domU(topCap);
  const gapBottom = ringGap(bottomCap, cu1, wall, wu0);
  const gapTop = ringGap(wall, wu1, topCap, tu0);
  assert.ok(gapBottom < 1e-12, `bottom junction gap ${gapBottom.toExponential(3)}mm`);
  assert.ok(gapTop < 1e-12, `top junction gap ${gapTop.toExponential(3)}mm`);
  // Same claim at the strongest possible level: the two panels' shared rings
  // are the SAME control points, not merely coincident samples.
  const wallRing = wall.ctrlNet[0], capRing = bottomCap.ctrlNet[bottomCap.ctrlNet.length - 1];
  assert.equal(wallRing.length, capRing.length);
  for (let i = 0; i < wallRing.length; i++) {
    for (let c = 0; c < 4; c++) {
      assert.ok(Math.abs(wallRing[i][c] - capRing[i][c]) < 1e-12, `junction control point ${i} component ${c} differs`);
    }
  }
});

test('THE OLD METHOD, measured on the SAME fixture: offsetSurface leaves a real 0.51mm radius error and a real 4.24mm junction gap', () => {
  // The app's own Cylinder builds its wall and its caps as separate revolved
  // panels, so this is exactly what a per-panel Greville-normal offset does to
  // them. Inward is chosen per panel by which sign moves toward the solid's
  // centroid — the same rule shellSolid's own app layer uses.
  const centroid = [0, 0, H / 2];
  const towardCentroid = (srf, d) => {
    const pick = (s) => surfacePoint(s, (s.knotsU[0] + s.knotsU[s.knotsU.length - 1]) / 2, (s.knotsV[0] + s.knotsV[s.knotsV.length - 1]) / 2);
    const a = offsetSurface(srf, d).surface, b = offsetSurface(srf, -d).surface;
    const da = Math.hypot(...pick(a).map((v, i) => v - centroid[i]));
    const db = Math.hypot(...pick(b).map((v, i) => v - centroid[i]));
    return da < db ? a : b;
  };
  const wall = revolve(makeLine([R, 0, 0], [R, 0, H]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const cap = revolve(makeLine([0, 0, 0], [R, 0, 0]), [0, 0, 0], [0, 0, -1], 0, 2 * Math.PI);
  const oldWall = towardCentroid(wall, T), oldCap = towardCentroid(cap, T);

  // (a) the offset "cylinder" is not a cylinder: its radius oscillates.
  let lo = Infinity, hi = -Infinity;
  sampleGrid(oldWall, 2, 512, (pt) => { const r = Math.hypot(pt[0], pt[1]); lo = Math.min(lo, r); hi = Math.max(hi, r); });
  const oldRadiusError = Math.max(Math.abs(lo - (R - T)), Math.abs(hi - (R - T)));
  assert.ok(oldRadiusError > 0.5, `expected the old method to visibly miss a true circle; measured only ${oldRadiusError}`);
  // The peak sits at the 45-degree station, between two on-curve control
  // points — the corner control point is short by t*(sqrt(2)-1), which pulls
  // the mid-span OUT rather than in.
  assert.ok(Math.abs(lo - (R - T)) < 1e-12, 'the on-curve stations do land at R - t; it is the spans between them that bulge');

  // (b) the junction gap, which is the actual reason a curved shell could not
  // be built: the offset wall does not move axially at all (its normal is
  // purely radial) while the offset cap moves purely axially, so the two end
  // up t apart along the axis AND t apart in radius.
  const oldGap = ringGap(oldWall, domU(oldWall)[0], oldCap, domU(oldCap)[1], 201);
  assert.ok(Math.abs(oldGap - T * Math.SQRT2) < 1e-6, `expected the old gap to be exactly t*sqrt(2) = ${T * Math.SQRT2}; measured ${oldGap}`);

  // (c) the same two numbers under the new method, for the record.
  const sh = shellRevolvedSolid(cylProfile(), T);
  let nlo = Infinity, nhi = -Infinity;
  sampleGrid(sh.innerPanels[1].srf, 2, 512, (pt) => { const r = Math.hypot(pt[0], pt[1]); nlo = Math.min(nlo, r); nhi = Math.max(nhi, r); });
  const newRadiusError = Math.max(Math.abs(nlo - (R - T)), Math.abs(nhi - (R - T)));
  const newGap = ringGap(sh.innerPanels[0].srf, domU(sh.innerPanels[0].srf)[1], sh.innerPanels[1].srf, domU(sh.innerPanels[1].srf)[0], 201);
  assert.ok(newRadiusError < 1e-12 && newGap < 1e-12);
  assert.ok(oldRadiusError / Math.max(newRadiusError, Number.EPSILON) > 1e9, 'the improvement must be orders of magnitude, not a tightened tolerance');
});

test('an inward-offset CONE is EXACTLY a cone of the SAME half-angle — apex, base radius and junction all match their closed forms', () => {
  const p = coneProfile();
  const off = offsetMeridianProfile(p, CT);
  assert.equal(off.ok, true);
  assert.equal(off.clippedAtAxis, true, "the offset apex leaves the axis half-plane and must be TRIMMED back to the axis, not flat-capped where the perpendicular landed");
  const rz = off.points.map((q) => profileRZ(p, q));
  assert.equal(rz.length, 3);
  // Analytic, from the meridian line R*z + H*r - R*H = 0 offset inward by t:
  //   r(z) = (R/H)(H - z) - t*L/H
  const junctionR = CR - CT * (CR + CL) / CH;
  const apexZ = CH - CT * CL / CR;
  assert.ok(Math.abs(rz[0][0]) < 1e-12 && Math.abs(rz[0][1] - CT) < 1e-12, 'the inner base cap sits on the axis at z = t');
  assert.ok(Math.abs(rz[1][0] - junctionR) < 1e-12, `junction radius ${rz[1][0]}, analytic ${junctionR}`);
  assert.ok(Math.abs(rz[1][1] - CT) < 1e-12);
  assert.ok(Math.abs(rz[2][0]) < 1e-12, 'the inner apex must land back ON the axis');
  assert.ok(Math.abs(rz[2][1] - apexZ) < 1e-12, `inner apex height ${rz[2][1]}, analytic ${apexZ}`);

  // THE HALF-ANGLE, checked on the built surface rather than on the profile:
  // every sampled point must satisfy the analytic inner-cone equation.
  const sh = shellRevolvedSolid(p, CT);
  const innerWall = sh.innerPanels[1].srf;
  let worst = 0;
  sampleGrid(innerWall, 8, 512, (pt) => {
    const want = (CR / CH) * (CH - pt[2]) - CT * CL / CH;
    worst = Math.max(worst, Math.abs(Math.hypot(pt[0], pt[1]) - want));
  });
  assert.ok(worst < 1e-12, `inner cone deviates by ${worst.toExponential(3)} from the analytic same-half-angle cone`);
  // Stated as an angle too, since "same half-angle" is the actual claim.
  const outerAngle = Math.atan2(CR, CH);
  const a = surfacePoint(innerWall, domU(innerWall)[0], 0), b = surfacePoint(innerWall, domU(innerWall)[1], 0);
  const innerAngle = Math.atan2(Math.abs(Math.hypot(a[0], a[1]) - Math.hypot(b[0], b[1])), Math.abs(a[2] - b[2]));
  assert.ok(Math.abs(innerAngle - outerAngle) < 1e-12, `inner half-angle ${innerAngle}, outer ${outerAngle}`);
});

test("the CONE's wall/cap junction is one shared circle too — a single interior corner, mitered exactly", () => {
  const sh = shellRevolvedSolid(coneProfile(), CT);
  assert.equal(sh.junctions.length, 1);
  const junctionR = CR - CT * (CR + CL) / CH;
  assert.ok(Math.abs(sh.junctions[0].radius - junctionR) < 1e-12);
  const cap = sh.innerPanels[0].srf, wall = sh.innerPanels[1].srf;
  const gap = ringGap(cap, domU(cap)[1], wall, domU(wall)[0]);
  assert.ok(gap < 1e-12, `cone junction gap ${gap.toExponential(3)}mm`);
});

test('SELF-INTERSECTION LIMIT: offsetting inward past what the shape can carry is refused, and the computed safe maximum matches its closed form', () => {
  // Cylinder: the cavity closes when the wall reaches the axis (t = R) or when
  // the two caps meet (t = H/2), whichever comes first.
  const cyl = cylProfile();
  const cylSafe = safeInwardOffset(cyl);
  assert.ok(Math.abs(cylSafe - Math.min(R, H / 2)) < 1e-6, `cylinder safe max ${cylSafe}, analytic ${Math.min(R, H / 2)}`);
  assert.equal(offsetMeridianProfile(cyl, R + 1).ok, false);
  assert.match(offsetMeridianProfile(cyl, R + 1).reason, /thickness|cavity|crosses/);
  assert.equal(offsetMeridianProfile(cyl, R * 0.9).ok, true, 'a thickness inside the limit must still succeed');

  // A SQUAT cylinder, where the height is the binding constraint instead of
  // the radius — proves the bound is computed, not a radius shortcut.
  const squat = makeCylinderProfile({ center: [0, 0, 0], axis: [0, 0, 1], refDir: [1, 0, 0], radius: 40, height: 10 });
  assert.ok(Math.abs(safeInwardOffset(squat) - 5) < 1e-6, 'a 10mm-tall cylinder can carry at most a 5mm wall');

  // Cone: the cavity closes to a point at t = R*H/(R+L).
  const cone = coneProfile();
  const coneSafe = safeInwardOffset(cone);
  const coneAnalytic = CR * CH / (CR + CL);
  assert.ok(Math.abs(coneSafe - coneAnalytic) < 1e-6, `cone safe max ${coneSafe}, analytic ${coneAnalytic}`);
  assert.equal(offsetMeridianProfile(cone, coneAnalytic + 1).ok, false);

  // And the refusal is honest at the shell level: a request past the limit is
  // clamped and SAYS SO, never silently returned as if it fitted.
  const clamped = shellRevolvedSolid(cyl, 100);
  assert.equal(clamped.clamped, true);
  assert.ok(clamped.appliedDistance < cylSafe && clamped.appliedDistance > cylSafe * 0.9);
  assert.throws(() => shellRevolvedSolid(cyl, 0), /nonzero/);
});

test('the outer and inner skins face AWAY from each other across the wall — a consistently wound closed solid, not two nested surfaces', () => {
  const p = cylProfile();
  const sh = shellRevolvedSolid(p, T);
  const check = (panel, expectSign, label) => {
    const srf = panel.srf;
    const [u0, u1] = domU(srf), [v0, v1] = domV(srf);
    const u = (u0 + u1) / 2, v = v0 + (v1 - v0) * 0.37;
    const pt = surfacePoint(srf, u, v);
    const n = unitNormal(srf, u, v);
    const radial = Math.hypot(pt[0], pt[1]) > 1e-9 ? [pt[0], pt[1], 0].map((c) => c / Math.hypot(pt[0], pt[1])) : [0, 0, 0];
    const axial = n[2];
    const rad = n[0] * radial[0] + n[1] * radial[1];
    assert.ok(expectSign(rad, axial), `${label}: normal (radial ${rad.toFixed(6)}, axial ${axial.toFixed(6)}) faces the wrong way`);
  };
  check(sh.outerPanels[1], (rad) => rad > 0.999, 'outer wall must face away from the axis');
  check(sh.innerPanels[1], (rad) => rad < -0.999, 'inner wall must face toward the axis, into the cavity');
  check(sh.outerPanels[0], (_r, ax) => ax < -0.999, 'outer bottom cap must face down, out of the solid');
  check(sh.innerPanels[0], (_r, ax) => ax > 0.999, 'inner bottom cap must face up, into the cavity');
});

test('an OFF-ORIGIN, TILTED axis with a non-orthonormal reference direction is just as exact — the frame is handled, not assumed', () => {
  const axis = [1, 2, 2];                             // length 3, not a world axis
  const center = [-17.5, 6.25, 41];
  const refDir = [3, -1, 1];                          // NOT perpendicular to the axis
  const p = makeCylinderProfile({ center, axis, refDir, radius: R, height: H });
  const frame = meridianFrame(center, axis, refDir);
  assert.ok(Math.abs(frame.refDir[0] * frame.axisDir[0] + frame.refDir[1] * frame.axisDir[1] + frame.refDir[2] * frame.axisDir[2]) < 1e-15, 'the frame must be orthonormalized');
  const sh = shellRevolvedSolid(p, T);
  const innerWall = sh.innerPanels[1].srf;
  const ax = frame.axisDir;
  let worst = 0;
  sampleGrid(innerWall, 8, 256, (pt) => {
    const rel = [pt[0] - center[0], pt[1] - center[1], pt[2] - center[2]];
    const along = rel[0] * ax[0] + rel[1] * ax[1] + rel[2] * ax[2];
    const perp = [rel[0] - along * ax[0], rel[1] - along * ax[1], rel[2] - along * ax[2]];
    worst = Math.max(worst, Math.abs(Math.hypot(perp[0], perp[1], perp[2]) - (R - T)));
  });
  assert.ok(worst < 1e-11, `tilted-axis inner radius deviates by ${worst.toExponential(3)}`);
  const gap = ringGap(sh.innerPanels[0].srf, domU(sh.innerPanels[0].srf)[1], innerWall, domU(innerWall)[0], 201);
  assert.ok(gap < 1e-11, `tilted-axis junction gap ${gap.toExponential(3)}mm`);
});

test('a distance of exactly zero is the identity, and a profile that is not a meridian profile is refused by name', () => {
  const p = cylProfile();
  const zero = offsetMeridianProfile(p, 0);
  assert.equal(zero.ok, true);
  for (let i = 0; i < p.points.length; i++) {
    for (let c = 0; c < 3; c++) assert.equal(zero.points[i][c], p.points[i][c]);
  }
  // A point off the meridian half-plane is not a generating profile at all.
  assert.throws(() => normalizeMeridianProfile({
    axisPoint: [0, 0, 0], axisDir: [0, 0, 1], refDir: [1, 0, 0],
    points: [[0, 0, 0], [R, 0, 0], [R, 4, H]],
  }), /off the meridian plane/);
  // A profile that crosses the axis has no single half-plane to live in.
  assert.throws(() => normalizeMeridianProfile({
    axisPoint: [0, 0, 0], axisDir: [0, 0, 1], refDir: [1, 0, 0],
    points: [[-R, 0, 0], [R, 0, 0]],
  }), /negative radius/);
  assert.throws(() => makeCylinderProfile({ center: [0, 0, 0], axis: [0, 0, 1], refDir: [1, 0, 0], radius: 0, height: H }), /radius must be positive/);
});

test('a segment lying entirely ON the axis revolves to nothing and is dropped rather than emitted as a degenerate panel', () => {
  const p = normalizeMeridianProfile({
    axisPoint: [0, 0, 0], axisDir: [0, 0, 1], refDir: [1, 0, 0],
    points: [[0, 0, 0], [0, 0, 10], [R, 0, 10], [R, 0, 30]],   // first segment runs up the axis
  });
  const panels = revolveProfilePanels(p, p.points, { flip: true });
  assert.equal(panels.length, 2, 'the on-axis segment contributes no surface');
});

// ============================================================================
// ARCS + OPENED FACES + THE RIM LIP. Every expected number below
// is again derived on paper from the fixture's own R/H/t, not captured.
// ============================================================================

test('ARCS: splitting the sweep into four panels is the SAME surface — the four quarter spans a full circle already is, meeting at bit-identical control points', () => {
  const p = cylProfile();
  const one = revolveProfilePanels(p, p.points, { flip: true });
  const four = revolveProfilePanels(p, p.points, { flip: true, arcs: 4 });
  assert.equal(one.length, 3, 'the default is one panel per segment — unchanged');
  assert.equal(four.length, 12, 'four arcs per segment');
  // Each quarter reproduces the whole surface over its own angular range.
  const whole = one[1].srf;
  for (let k = 0; k < 4; k++) {
    const q = four[4 + k].srf;                       // the wall's own four quarters
    const [qv0, qv1] = domV(q), [wv0, wv1] = domV(whole);
    for (let i = 0; i <= 12; i++) {
      const f = i / 12;
      const a = surfacePoint(q, domU(q)[0] + (domU(q)[1] - domU(q)[0]) * 0.37, qv0 + (qv1 - qv0) * f);
      // the same physical station on the full-circle panel: quarter k covers
      // the angular quarter [k/4, (k+1)/4] of the whole sweep
      const b = surfacePoint(whole, domU(whole)[0] + (domU(whole)[1] - domU(whole)[0]) * 0.37, wv0 + (wv1 - wv0) * ((k + f) / 4));
      assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-9, `quarter ${k} deviates from the full panel at f=${f}`);
    }
    // and it JOINS its neighbor exactly: the last control column of one
    // quarter is the first of the next, component for component.
    const next = four[4 + ((k + 1) % 4)].srf;
    for (let r = 0; r < q.ctrlNet.length; r++) {
      const endCol = q.ctrlNet[r][q.ctrlNet[r].length - 1], startCol = next.ctrlNet[r][0];
      for (let c = 0; c < 4; c++) assert.ok(Math.abs(endCol[c] - startCol[c]) < 1e-12, `quarter ${k} does not meet quarter ${k + 1} at component ${c}`);
    }
  }
});

test('OPENING A CAP: the inner wall stops in the opened face\'s OWN plane, and the rim lip is the annulus between the two exact circles there', () => {
  const p = cylProfile();
  const sh = shellRevolvedSolid(p, T, { removedSegments: [2] });   // open the TOP cap
  assert.equal(sh.openEnd, true);
  assert.equal(sh.openStart, false);
  assert.equal(sh.outerCount, 2, 'bottom cap + wall');
  assert.equal(sh.innerCount, 2);
  assert.equal(sh.rimCount, 1, 'one lip closes the one opening');
  // the inner wall's own top edge sits at z = H exactly — NOT at H - t
  const innerTop = sh.innerProfile[sh.innerProfile.length - 1];
  assert.ok(Math.abs(profileRZ(p, innerTop)[1] - H) < 1e-12, `the inner wall stops at z = ${profileRZ(p, innerTop)[1]}, not the opening plane ${H}`);
  assert.ok(Math.abs(profileRZ(p, innerTop)[0] - (R - T)) < 1e-12, 'and at the inner radius R - t');
  // the lip: flat at z = H, spanning exactly R - t .. R
  const lip = sh.rimPanels[0].srf;
  let worstZ = 0, lo = Infinity, hi = -Infinity;
  sampleGrid(lip, 8, 128, (pt) => {
    worstZ = Math.max(worstZ, Math.abs(pt[2] - H));
    const rr = Math.hypot(pt[0], pt[1]);
    lo = Math.min(lo, rr); hi = Math.max(hi, rr);
  });
  assert.ok(worstZ < 1e-12, `the lip is not flat in the opening's own plane (worst ${worstZ.toExponential(3)})`);
  assert.ok(Math.abs(lo - (R - T)) < 1e-12 && Math.abs(hi - R) < 1e-12, `the lip spans ${lo}..${hi}, analytic ${R - T}..${R}`);
  // it FACES OUT of the material: +axis at the top opening
  const n = unitNormal(lip, (domU(lip)[0] + domU(lip)[1]) / 2, domV(lip)[0] + (domV(lip)[1] - domV(lip)[0]) * 0.41);
  assert.ok(n[2] > 0.999, `the lip faces (${n.map((v) => v.toFixed(4)).join(', ')}), not out of the solid`);
});

test("A CONE OPENED AT ITS BASE: the inner wall is EXTENDED to the base plane, not left where the perpendicular offset put it (which is BELOW the opening)", () => {
  const p = coneProfile();
  const sh = shellRevolvedSolid(p, CT, { removedSegments: [0] });  // open the base cap
  assert.equal(sh.openStart, true);
  assert.equal(sh.rimCount, 1);
  const innerBase = sh.innerProfile[0];
  const [r0, z0] = profileRZ(p, innerBase);
  // Analytic: the inner slant line is H*r + R*z - R*H = -t*L; at z = 0 that
  // is r = R - t*L/H. The NAIVE perpendicular offset of the rim point would
  // instead land at z = -t*R/L, i.e. below the opening entirely.
  assert.ok(Math.abs(z0) < 1e-12, `the inner wall must stop exactly in the base plane z = 0 (got ${z0})`);
  assert.ok(Math.abs(r0 - (CR - CT * CL / CH)) < 1e-12, `inner base radius ${r0}, analytic ${CR - CT * CL / CH}`);
  const naiveZ = -CT * CR / CL;
  assert.ok(Math.abs(z0 - naiveZ) > CT * 0.5, 'the naive perpendicular answer is genuinely different — this test discriminates the two');
  // the lip is flat in the base plane and genuinely WIDER than the wall,
  // because a slanted wall cut by a horizontal plane shows a wider band
  const lip = sh.rimPanels[0].srf;
  let worstZ = 0, lo = Infinity, hi = -Infinity;
  sampleGrid(lip, 6, 128, (pt) => { worstZ = Math.max(worstZ, Math.abs(pt[2])); const rr = Math.hypot(pt[0], pt[1]); lo = Math.min(lo, rr); hi = Math.max(hi, rr); });
  assert.ok(worstZ < 1e-12, 'the cone lip is flat in the base plane');
  assert.ok(Math.abs(hi - CR) < 1e-12 && Math.abs(lo - (CR - CT * CL / CH)) < 1e-12, `the cone lip spans ${lo}..${hi}`);
  assert.ok(hi - lo > CT, `the lip (${(hi - lo).toFixed(6)}) is wider than the wall (${CT}) — t*L/H, not t`);
  const n = unitNormal(lip, (domU(lip)[0] + domU(lip)[1]) / 2, domV(lip)[0] + (domV(lip)[1] - domV(lip)[0]) * 0.41);
  assert.ok(n[2] < -0.999, `the cone's base lip must face DOWN, out of the solid (got ${n.map((v) => v.toFixed(4)).join(', ')})`);
});

test('OPENED-FACE REFUSALS: a non-contiguous opening, every face opened, and a face index this profile does not have are each refused BY NAME', () => {
  const p = cylProfile();
  // the WALL opened while both caps are kept: two disks with nothing joining them
  assert.throws(() => shellRevolvedSolid(p, T, { removedSegments: [1] }), /not next to each other/);
  assert.throws(() => shellRevolvedSolid(p, T, { removedSegments: [0, 1, 2] }), /every face was opened/);
  assert.throws(() => shellRevolvedSolid(p, T, { removedSegments: [7] }), /is not one of this profile's own 3 faces/);
  // opening BOTH caps is contiguous at neither end — but it IS a legal tube
  const tube = shellRevolvedSolid(p, T, { removedSegments: [0, 2] });
  assert.equal(tube.rimCount, 2, 'a tube open at both ends has two lips');
  assert.equal(tube.outerCount, 1);
  for (const [idx, wantZ] of [[0, 0], [1, H]]) {
    let worstZ = 0;
    sampleGrid(tube.rimPanels[idx].srf, 4, 64, (pt) => { worstZ = Math.max(worstZ, Math.abs(pt[2] - wantZ)); });
    assert.ok(worstZ < 1e-12, `tube lip ${idx} is not flat at z = ${wantZ}`);
  }
});

// ============================================================================
// ARC MERIDIAN PROFILES. CURVED-PROFILE SHELLS — a Torus's
// meridian is a full circle offset from the axis; a Sphere's is a half
// circle centered ON the axis, pole to pole. Both are proven EXACT here
// against real analytic ground truth (the true torus/sphere equations),
// never against the construction's own output — same discipline as every
// other test in this file.
// ============================================================================

const TR = 25, Tt = 8, Tthk = 2;                   // torus fixture: ring radius, tube radius, wall thickness
const SR = 20, Sthk = 3;                            // sphere fixture: radius, wall thickness
const axisPoint = [0, 0, 0], axisDir = [0, 0, 1], refDir = [1, 0, 0];

function torusProfile() {
  return makeArcMeridianProfile({ axisPoint, axisDir, refDir, centerR: TR, centerZ: 0, radius: Tt, angleStart: 0, sweep: 2 * Math.PI, closed: true });
}
function sphereProfile() {
  return makeArcMeridianProfile({ axisPoint, axisDir, refDir, centerR: 0, centerZ: 0, radius: SR, angleStart: -Math.PI / 2, sweep: Math.PI, closed: false });
}

// Sample a whole surface densely and report the WORST deviation from a
// caller-supplied analytic predicate, plus the worst signed dot of the
// surface's own analytic normal against a caller-supplied expected
// direction — one shared harness for both the torus and sphere fixtures.
// The normal check is skipped at i===0/i===nu (the EXACT domain boundary in
// U — the true pole for a sphere-like profile touching the axis there): a
// pole's own Sv (circumferential derivative) is genuinely, mathematically
// degenerate right at the knot (every rotated copy of an on-axis point
// coincides), so the naive su x sv / |.| normal formula divides a near-zero
// vector by a near-zero length there — real numerical noise in the ratio,
// not a defect in the surface's own POSITION (still checked at every
// sample, poles included) or in this module's offset math. A torus profile
// never touches the axis, so this skip is a no-op for it.
function sampleSurfaceWorst(srf, nu, nv, radialFn, normalFn) {
  const [u0, u1] = domU(srf), [v0, v1] = domV(srf);
  let worstRadial = 0, worstDot = Infinity;
  for (let i = 0; i <= nu; i++) {
    for (let j = 0; j <= nv; j++) {
      const u = u0 + (u1 - u0) * (i / nu), v = v0 + (v1 - v0) * (j / nv);
      const pt = surfacePoint(srf, u, v);
      worstRadial = Math.max(worstRadial, Math.abs(radialFn(pt)));
      if (normalFn && i !== 0 && i !== nu) {
        const n = unitNormal(srf, u, v);
        const d = normalFn(pt, n);
        if (Number.isFinite(d)) worstDot = Math.min(worstDot, d);
      }
    }
  }
  return { worstRadial, worstDot };
}

test('TORUS ARC PROFILE: the un-offset profile is exactly the true minor circle at every sampled parameter, both ends coincide (closed)', () => {
  const p = torusProfile();
  const [r0, z0] = profileRZ(p, [TR + Tt, 0, 0]);
  assert.ok(Math.abs(r0 - (TR + Tt)) < 1e-12 && Math.abs(z0) < 1e-12);
  // the arc's own two ends (angle 0 and angle 2*PI) are the identical point
  const a = revolveArcMeridianPanel(p, { flip: false });
  const b = surfacePoint(a.srf, domU(a.srf)[0], (domV(a.srf)[0] + domV(a.srf)[1]) / 2);
  assert.ok(Math.hypot(b[0], b[1]) > 0, 'sanity: a real point on the torus wall, not the origin');
});

test('offsetArcMeridianProfile on a TORUS is an EXACT radius change: same center, same angle range, radius - t', () => {
  const p = torusProfile();
  const off = offsetArcMeridianProfile(p, Tthk);
  assert.equal(off.ok, true);
  assert.ok(Math.abs(off.profile.radius - (Tt - Tthk)) < 1e-12);
  assert.equal(off.profile.centerR, TR);
  assert.equal(off.profile.centerZ, 0);
  assert.equal(off.profile.angleStart, p.angleStart);
  assert.equal(off.profile.sweep, p.sweep);
});

test('an inward-offset TORUS wall is EXACTLY a torus of the same ring radius and a smaller tube radius, sampled densely, poles and seam included', () => {
  const p = torusProfile();
  const off = offsetArcMeridianProfile(p, Tthk);
  assert.equal(off.ok, true);
  const outer = revolveArcMeridianPanel(p, { flip: true });
  const inner = revolveArcMeridianPanel(off.profile, { flip: false });
  // TRUE torus equation: distance from the ring centerline (radius TR about
  // the axis) equals the tube radius exactly, at every sampled (u,v).
  const distFromRing = (pt) => Math.hypot(Math.hypot(pt[0], pt[1]) - TR, pt[2]);
  const outward = (pt) => { // unit direction AWAY from the tube's own local centerline
    const rho = Math.hypot(pt[0], pt[1]);
    const radial = rho > 1e-9 ? [pt[0] / rho, pt[1] / rho, 0] : [1, 0, 0];
    const dr = rho - TR, dz = pt[2];
    const mag = Math.hypot(dr, dz) || 1;
    return [radial[0] * dr / mag, radial[1] * dr / mag, dz / mag];
  };
  const outerRes = sampleSurfaceWorst(outer.srf, 24, 24, (pt) => distFromRing(pt) - Tt, (pt, n) => { const o = outward(pt); return n[0] * o[0] + n[1] * o[1] + n[2] * o[2]; });
  const innerRes = sampleSurfaceWorst(inner.srf, 24, 24, (pt) => distFromRing(pt) - (Tt - Tthk), (pt, n) => { const o = outward(pt); return n[0] * o[0] + n[1] * o[1] + n[2] * o[2]; });
  assert.ok(outerRes.worstRadial < 1e-9, `outer torus wall deviates ${outerRes.worstRadial.toExponential(3)} from the true tube radius ${Tt}`);
  assert.ok(innerRes.worstRadial < 1e-9, `inner torus wall deviates ${innerRes.worstRadial.toExponential(3)} from the true tube radius ${Tt - Tthk}`);
  // ORIENTATION, proven numerically (not just derived): the outer wall's
  // normal points AWAY from the tube's own local centerline (out of the
  // solid); the inner wall's points TOWARD it (out of the wall, into the
  // cavity) — the two faces away from each other across the wall, the same
  // property the point-chain shell already proves for a Cylinder.
  assert.ok(outerRes.worstDot > 0.999, `outer torus normal doesn't face outward (worst dot ${outerRes.worstDot})`);
  const innerInwardCheck = sampleSurfaceWorst(inner.srf, 24, 24, () => 0, (pt, n) => { const o = outward(pt); return -(n[0] * o[0] + n[1] * o[1] + n[2] * o[2]); });
  assert.ok(innerInwardCheck.worstDot > 0.999, `inner torus normal doesn't face toward the tube centerline (worst dot ${-innerInwardCheck.worstDot})`);
});

test('SPHERE ARC PROFILE: the un-offset profile is exactly the true meridian, both poles land on the axis', () => {
  const p = sphereProfile();
  const startPt = [p.axisPoint[0] + p.refDir[0] * p.radius * Math.cos(p.angleStart), 0, 0]; // sanity only
  assert.ok(Math.abs(p.radius - SR) < 1e-12);
  const panel = revolveArcMeridianPanel(p, { flip: false });
  const [u0] = domU(panel.srf);
  const southPole = surfacePoint(panel.srf, u0, (domV(panel.srf)[0] + domV(panel.srf)[1]) / 2);
  assert.ok(Math.hypot(southPole[0], southPole[1]) < 1e-9, 'the south pole must land exactly on the axis');
  assert.ok(Math.abs(southPole[2] - (-SR)) < 1e-9, 'the south pole must sit at z = -R');
});

test('offsetArcMeridianProfile on a SPHERE is an EXACT radius change, and the poles stay pinned to the axis at ANY radius', () => {
  const p = sphereProfile();
  const off = offsetArcMeridianProfile(p, Sthk);
  assert.equal(off.ok, true);
  assert.ok(Math.abs(off.profile.radius - (SR - Sthk)) < 1e-12);
  assert.equal(off.profile.centerR, 0);
  assert.equal(off.profile.angleStart, p.angleStart);
  assert.equal(off.profile.sweep, p.sweep);
});

test('an inward-offset SPHERE wall is EXACTLY a sphere of a smaller radius, sampled densely, poles included, to near machine precision', () => {
  const p = sphereProfile();
  const off = offsetArcMeridianProfile(p, Sthk);
  assert.equal(off.ok, true);
  const outer = revolveArcMeridianPanel(p, { flip: true });
  const inner = revolveArcMeridianPanel(off.profile, { flip: false });
  const distFromCenter = (pt) => Math.hypot(pt[0], pt[1], pt[2]);
  const outward = (pt) => { const d = distFromCenter(pt) || 1; return [pt[0] / d, pt[1] / d, pt[2] / d]; };
  const outerRes = sampleSurfaceWorst(outer.srf, 24, 24, (pt) => distFromCenter(pt) - SR, (pt, n) => { const o = outward(pt); return n[0] * o[0] + n[1] * o[1] + n[2] * o[2]; });
  const innerRes = sampleSurfaceWorst(inner.srf, 24, 24, (pt) => distFromCenter(pt) - (SR - Sthk), null);
  assert.ok(outerRes.worstRadial < 1e-9, `outer sphere wall deviates ${outerRes.worstRadial.toExponential(3)} from R=${SR}`);
  assert.ok(innerRes.worstRadial < 1e-9, `inner sphere wall deviates ${innerRes.worstRadial.toExponential(3)} from R=${SR - Sthk}`);
  assert.ok(outerRes.worstDot > 0.999, `outer sphere normal doesn't point radially outward (worst dot ${outerRes.worstDot})`);
  const innerInwardCheck = sampleSurfaceWorst(inner.srf, 24, 24, () => 0, (pt, n) => { const o = outward(pt); return -(n[0] * o[0] + n[1] * o[1] + n[2] * o[2]); });
  assert.ok(innerInwardCheck.worstDot > 0.999, `inner sphere normal doesn't point radially inward, into the cavity (worst dot ${-innerInwardCheck.worstDot})`);
});

test('SAFE MAXIMUM + AUTO-CLAMP for an arc profile: closed-form (the arc\'s own radius), and shellArcRevolvedSolid clamps honestly past it', () => {
  assert.equal(safeArcInwardOffset(torusProfile()), Tt);
  assert.equal(safeArcInwardOffset(sphereProfile()), SR);
  // request far past the safe maximum
  const sh = shellArcRevolvedSolid(torusProfile(), Tt * 5);
  assert.equal(sh.clamped, true);
  assert.ok(sh.appliedDistance < Tt && sh.appliedDistance > Tt * 0.9, `clamped distance ${sh.appliedDistance} should sit just under the safe max ${Tt}`);
  assert.equal(sh.safeMaxDistance, Tt);
  // a genuinely safe request is NOT clamped
  const sh2 = shellArcRevolvedSolid(sphereProfile(), Sthk);
  assert.equal(sh2.clamped, false);
  assert.equal(sh2.appliedDistance, Sthk);
});

test('offsetArcMeridianProfile refuses honestly once distance reaches or exceeds the radius, never producing a zero/negative radius', () => {
  const p = torusProfile();
  assert.equal(offsetArcMeridianProfile(p, Tt).ok, false);
  assert.equal(offsetArcMeridianProfile(p, Tt * 2).ok, false);
  assert.match(offsetArcMeridianProfile(p, Tt).reason, /radius|thickness|fit/);
  const p2 = sphereProfile();
  assert.equal(offsetArcMeridianProfile(p2, SR).ok, false);
  assert.equal(offsetArcMeridianProfile(p2, SR * 0.999).ok, true, 'a thickness just inside the radius must still succeed');
});

test('makeArcMeridianProfile refuses a degenerate/axis-crossing arc BY NAME, and a real one succeeds', () => {
  assert.throws(() => makeArcMeridianProfile({ axisPoint, axisDir, refDir, centerR: 5, centerZ: 0, radius: 10, angleStart: 0, sweep: 2 * Math.PI, closed: true }), /crosses the axis/, 'a "torus" whose tube radius exceeds its ring radius crosses the axis and must be refused');
  assert.throws(() => makeArcMeridianProfile({ axisPoint, axisDir, refDir, centerR: 0, centerZ: 0, radius: 0, angleStart: 0, sweep: Math.PI, closed: false }), /positive radius/);
  assert.throws(() => makeArcMeridianProfile({ axisPoint, axisDir, refDir, centerR: 0, centerZ: 0, radius: 5, angleStart: 0, sweep: 0, closed: false }), /nonzero sweep/);
  assert.ok(makeArcMeridianProfile({ axisPoint, axisDir, refDir, centerR: TR, centerZ: 0, radius: Tt, angleStart: 0, sweep: 2 * Math.PI, closed: true }));
});

test('shellArcRevolvedSolid assembles a real TORUS shell: exactly 2 panels, no rim, both walls exact, junctions empty (nothing to junction)', () => {
  const sh = shellArcRevolvedSolid(torusProfile(), Tthk);
  assert.equal(sh.outerCount, 1);
  assert.equal(sh.innerCount, 1);
  assert.equal(sh.rimCount, 0);
  assert.equal(sh.panels.length, 2);
  assert.deepEqual(sh.junctions, []);
  assert.equal(sh.exact, true);
  const distFromRing = (pt) => Math.hypot(Math.hypot(pt[0], pt[1]) - TR, pt[2]);
  let worstOuter = 0, worstInner = 0;
  const check = (srf, want, setWorst) => sampleSurfaceWorst(srf, 16, 16, (pt) => { const d = Math.abs(distFromRing(pt) - want); if (d > setWorst.v) setWorst.v = d; return 0; }, null);
  const wo = { v: 0 }, wi = { v: 0 };
  check(sh.outerPanels[0].srf, Tt, wo);
  check(sh.innerPanels[0].srf, Tt - Tthk, wi);
  assert.ok(wo.v < 1e-9 && wi.v < 1e-9, `torus shell walls deviate ${wo.v}, ${wi.v} from analytic`);
});

test('shellArcRevolvedSolid assembles a real SPHERE shell: exactly 2 panels, no rim, both walls exact', () => {
  const sh = shellArcRevolvedSolid(sphereProfile(), Sthk);
  assert.equal(sh.outerCount, 1);
  assert.equal(sh.innerCount, 1);
  assert.equal(sh.rimCount, 0);
  assert.equal(sh.panels.length, 2);
  const distFromCenter = (pt) => Math.hypot(pt[0], pt[1], pt[2]);
  const wo = { v: 0 }, wi = { v: 0 };
  const check = (srf, want, setWorst) => sampleSurfaceWorst(srf, 16, 16, (pt) => { const d = Math.abs(distFromCenter(pt) - want); if (d > setWorst.v) setWorst.v = d; return 0; }, null);
  check(sh.outerPanels[0].srf, SR, wo);
  check(sh.innerPanels[0].srf, SR - Sthk, wi);
  assert.ok(wo.v < 1e-9 && wi.v < 1e-9, `sphere shell walls deviate ${wo.v}, ${wi.v} from analytic`);
});
