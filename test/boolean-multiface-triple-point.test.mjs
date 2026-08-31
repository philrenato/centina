// A MULTI-FACE OPERAND MEETS THE OTHER SOLID AT TRIPLE POINTS, AND THEY HAVE
// TO BE SHARED.
//
// One surface against one surface has no triple point: the two faces share one
// curve and take their boundary points from the same sample array. Give either
// operand a second face and the picture changes — a prism's side face and its
// cap share an edge, so the other solid's surface crosses all three at once,
// and each of those faces finds that corner its own way. The side face's march
// ENDS there, on all three surfaces to machine precision; the cap's curve has
// no sample there at all and its face falls back to a chord crossing, a
// thousand times further out than the weld tolerance. The shell then comes back
// with one sliver triangle per triple point.
//
// EVERY OPERAND HERE IS MEASURED BEFORE ANYTHING IS ASSERTED ABOUT IT. A
// fixture nobody measured is a free variable in every result resting on it, and
// this file's operands are built by control-point arithmetic, which is exactly
// where a fixture has silently drifted before.
import test from 'node:test';
import assert from 'node:assert/strict';
import { booleanSolids, shareTriplePoints } from '../kernel/boolean.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { surfacePoint, closestPointOnSurface } from '../kernel/surface.mjs';

const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function quadSurface(p00, p10, p11, p01) {
  const w = (p) => [p[0], p[1], p[2], 1];
  return { degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]] };
}
function starPoints(points, outerR, innerR, cx = 0, cy = 0) {
  const out = [];
  for (let i = 0; i < points * 2; i++) {
    const a = (i / (points * 2)) * Math.PI * 2 + Math.PI / 2;
    const r = i % 2 ? innerR : outerR;
    out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return out;
}
const CAP_HALF = 30;
const capPlane = (z) => quadSurface([-CAP_HALF, -CAP_HALF, z], [CAP_HALF, -CAP_HALF, z],
  [CAP_HALF, CAP_HALF, z], [-CAP_HALF, CAP_HALF, z]);
const capTrimLoop = (pts) => pts.map(([x, y]) => [(x + CAP_HALF) / (2 * CAP_HALF), (y + CAP_HALF) / (2 * CAP_HALF)]);
function tess(faces, res) {
  const tris = [];
  for (const f of faces) {
    for (const t of tessellateTrimmedSurface(f.srf, f.trimLoop ?? trivialTrimLoop(f.srf), res, res, f.trimHoles ?? [])) {
      tris.push(t.map((v) => v.position));
    }
  }
  return tris;
}
// SIDE FACES FIRST, THEN THE TWO CAPS, so a face index reads directly: 0-9 are
// the star's own walls and 10/11 are its ends. The tests below name faces by
// index and the ordering is what makes that legible.
function starPrism(outerR, innerR, z0, z1, cx = 0, cy = 0) {
  const pts = starPoints(5, outerR, innerR, cx, cy);
  const faces = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    faces.push({ srf: quadSurface([a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]) });
  }
  faces.push({ srf: capPlane(z0), trimLoop: capTrimLoop(pts) });
  faces.push({ srf: capPlane(z1), trimLoop: capTrimLoop(pts) });
  return { faces, triangles: tess(faces, 8), corners: pts, z0, z1 };
}
// A REVOLVED PROFILE THAT GOES IN AND OUT AGAIN, not a ball. The radius has to
// rise, fall and rise for the operand to be genuinely curved rather than a
// primitive wearing a NURBS coat, and the test below measures that it does.
const BLOB_PROFILE = [[0, 0, 0], [12, 0, 6], [8, 0, 14], [15, 0, 24], [6, 0, 34], [0, 0, 40]];
function blob() {
  const srf = revolve(globalCurveInterp(BLOB_PROFILE, 3), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const faces = [{ srf }];
  return { faces, triangles: tess(faces, 40), srf };
}
function sweepCurves(A, B) {
  const curves = [];
  for (let i = 0; i < A.faces.length; i++) {
    for (let j = 0; j < B.faces.length; j++) {
      let r;
      try { r = intersectSurfacesComplete(A.faces[i].srf, B.faces[j].srf); } catch { continue; }
      if (!r || !r.ok) continue;
      for (const c of r.components) {
        if (c.samples.length >= 2) curves.push({ samples: c.samples.map((s) => s.point), faceA: i, faceB: j });
      }
    }
  }
  return curves;
}
// Every point of a (u,v) grid on the surface, as { r, z }. Read off the SURFACE
// rather than off the profile's control points: a rational control net's stored
// xyz is not the curve, and reading one as if it were lets a "radius 20 ball"
// span 17.73 to 22.42 while every assertion resting on it still passes.
//
// BOTH PARAMETER DIRECTIONS ARE SWEPT, deliberately. Which of u and v carries
// the profile and which the revolve angle is a fact about `revolve`'s own
// output ordering; holding one axis fixed on the assumption that it is the
// angular one samples a single meridian at best and the axis itself at worst,
// and either reads as a radius that says nothing about the body.
function surfaceSamples(srf, nu = 48, nv = 200) {
  const uK = srf.knotsU, vK = srf.knotsV;
  const out = [];
  for (let i = 0; i <= nu; i++) {
    for (let j = 0; j <= nv; j++) {
      const p = surfacePoint(srf, uK[0] + (uK[uK.length - 1] - uK[0]) * (i / nu), vK[0] + (vK[vK.length - 1] - vK[0]) * (j / nv));
      out.push({ r: Math.hypot(p[0], p[1]), z: p[2] });
    }
  }
  return out;
}
// The widest the body gets in a thin slice about `z`. Widest rather than
// nearest, because a slice holds a whole ring of samples and any one of them
// would be an arbitrary pick.
function radiusAtZ(samples, z, band = 0.4) {
  let r = 0;
  for (const s of samples) if (Math.abs(s.z - z) <= band) r = Math.max(r, s.r);
  return r;
}

test('the operands are what this file claims: a genuinely wavy revolve and a 12-face star prism that really interpenetrate', () => {
  const A = blob();
  const B = starPrism(18, 7, 10, 28);

  // The blob, measured on its own surface over a dense grid.
  const samples = surfaceSamples(A.srf);
  const rMax = Math.max(...samples.map((s) => s.r));
  const zMin = Math.min(...samples.map((s) => s.z)), zMax = Math.max(...samples.map((s) => s.z));
  assert.ok(zMax - zMin > 39 && zMax - zMin < 41, `blob height ${(zMax - zMin).toFixed(3)}`);
  assert.ok(rMax > 14 && rMax < 16, `blob max radius ${rMax.toFixed(3)}`);
  // NOT MONOTONE — the waist between the two bulges is what makes this organic
  // rather than a ball, and a ball would pass every other check here.
  const rWaist = radiusAtZ(samples, 14);
  const rBelly = radiusAtZ(samples, 24);
  const rLower = radiusAtZ(samples, 6);
  assert.ok(rWaist < rLower - 1, `waist ${rWaist.toFixed(3)} is not narrower than the lower bulge ${rLower.toFixed(3)}`);
  assert.ok(rBelly > rWaist + 4, `belly ${rBelly.toFixed(3)} is not wider than the waist ${rWaist.toFixed(3)}`);

  // The prism, measured from its own faces rather than its constructor's input.
  assert.equal(B.faces.length, 12);
  const cornerRadii = B.corners.map(([x, y]) => Math.hypot(x, y)).map((r) => +r.toFixed(6));
  assert.deepEqual([...new Set(cornerRadii)].sort((a, b) => a - b), [7, 18]);
  for (const f of B.faces.slice(0, 10)) {
    const zs = f.srf.ctrlNet.flat().map((c) => c[2] / c[3]);
    assert.equal(Math.min(...zs), 10);
    assert.equal(Math.max(...zs), 28);
  }

  // AND THEY REALLY CROSS. At each cap height the blob's radius must fall
  // strictly between the star's inner and outer radius, or the cap's own trim
  // loop is never cut and there is no triple point to share.
  for (const z of [10, 28]) {
    const r = radiusAtZ(samples, z);
    assert.ok(r > 7 && r < 18, `blob radius ${r.toFixed(3)} at z=${z} does not straddle the star's 7..18`);
  }
});

test('the triple point is already computed exactly — it is only missing from the other curve', () => {
  const A = blob();
  const B = starPrism(18, 7, 10, 28);
  const curves = sweepCurves(A, B);
  const capCurve = curves.find((c) => c.faceB === 10);
  const sideCurve = curves.find((c) => c.faceB === 9);
  assert.ok(capCurve && sideCurve, 'the blob must cut both the bottom cap and side face 9');

  // The side face's march runs out of its own domain ON the cap plane, and that
  // endpoint is the triple point: it lies on the blob, on the side face and on
  // the cap plane at once.
  const s = sideCurve.samples;
  const end = Math.abs(s[0][2] - 10) < 1e-9 ? s[0] : s[s.length - 1];
  assert.ok(Math.abs(end[2] - 10) < 1e-9, 'side curve does not reach the bottom cap plane');
  assert.ok(closestPointOnSurface(A.srf, end).distance < 1e-12, 'triple point is not on the blob');
  assert.ok(closestPointOnSurface(B.faces[9].srf, end).distance < 1e-12, 'triple point is not on its own side face');
  assert.ok(closestPointOnSurface(B.faces[10].srf, end).distance < 1e-12, 'triple point is not on the cap plane');

  // And the cap curve — which is the ONLY curve the cap face is split by — does
  // not carry it. This gap is the whole defect: the cap face has to invent the
  // corner from a chord, and lands far outside the sew's 1e-4 weld tolerance.
  const nearest = Math.min(...capCurve.samples.map((q) => d3(q, end)));
  assert.ok(nearest > 1e-2, `the cap curve already has a sample ${nearest.toExponential(2)} away, so the premise does not hold`);

  // Sharing puts it there, and nowhere else: the enriched curve gains exactly
  // the reported number of samples, and the endpoint is now on it exactly.
  const shared = shareTriplePoints(curves, A, B, {});
  assert.ok(shared.inserted > 0, 'no triple point was shared at all');
  const before = curves.reduce((n, c) => n + c.samples.length, 0);
  const after = shared.curves.reduce((n, c) => n + c.samples.length, 0);
  assert.equal(after - before, shared.inserted);
  const enrichedCap = shared.curves.find((c) => c.faceB === 10);
  assert.ok(Math.min(...enrichedCap.samples.map((q) => d3(q, end))) < 1e-12,
    'the shared curve still does not pass through the triple point');

  // Every point put into a curve lies on BOTH of that curve's surfaces, which
  // is the property that makes it a point OF that curve rather than a point
  // that merely passed nearby.
  for (let i = 0; i < curves.length; i++) {
    const raw = new Set(curves[i].samples);
    const srfA = A.faces[curves[i].faceA].srf, srfB = B.faces[curves[i].faceB].srf;
    for (const p of shared.curves[i].samples) {
      if (raw.has(p)) continue;
      assert.ok(closestPointOnSurface(srfA, p).distance < 1e-12, 'an inserted point is not on the curve\'s first surface');
      assert.ok(closestPointOnSurface(srfB, p).distance < 1e-12, 'an inserted point is not on the curve\'s second surface');
    }
  }
});

test('sharing is refused between curves with no face in common, and between two COMPONENTS of one face pair', () => {
  const A = blob();
  const B = starPrism(18, 7, 10, 28);
  const curves = sweepCurves(A, B);
  const cap = curves.find((c) => c.faceB === 10);
  const side = curves.find((c) => c.faceB === 9);

  // No face in common: nothing sees both, so a shared vertex between them would
  // be a corner nothing asked for.
  const disjoint = [
    { ...cap, faceA: 0, faceB: 10 },
    { ...side, faceA: 1, faceB: 9 },
  ];
  assert.equal(shareTriplePoints(disjoint, { faces: [...A.faces, ...A.faces] }, B, {}).inserted, 0);

  // BOTH faces in common: these are separate components of one surface-surface
  // intersection and are disjoint by construction, so no endpoint of either is
  // a point of the other. The on-surface test cannot see that — every sample of
  // either lies on both surfaces — so the exclusion has to be structural.
  const sameFacePair = [
    { ...cap, faceA: 0, faceB: 9 },
    { ...side, faceA: 0, faceB: 9 },
  ];
  assert.equal(shareTriplePoints(sameFacePair, A, B, {}).inserted, 0);
});

test('a 12-face star prism against a wavy revolve closes under all three operators, concentric and off-axis', () => {
  for (const [name, B] of [
    ['concentric', starPrism(18, 7, 10, 28)],
    ['off-axis', starPrism(16, 6, 8, 30, 6, 3)],
  ]) {
    const A = blob();
    const curves = sweepCurves(A, B);
    assert.ok(curves.length >= 12, `${name}: only ${curves.length} cut curves`);
    for (const op of ['union', 'intersect', 'difference']) {
      const res = booleanSolids(A, B, curves, op);
      assert.ok(res.ok, `${name} ${op}: ${res.reason}`);
      assert.equal(res.stats.nakedEdgeCount, 0, `${name} ${op} naked edges`);
      assert.equal(res.stats.nonManifoldEdgeCount, 0, `${name} ${op} non-manifold edges`);
      assert.ok(res.triplePoints > 0, `${name} ${op}: nothing was shared, so this closed for some other reason`);
      // The residual actually welded across the shared curves, reported so a
      // pass here is visibly comfortable rather than scraping the tolerance.
      assert.ok(res.worstSharedGap < 1e-6, `${name} ${op}: worst weld gap ${res.worstSharedGap.toExponential(2)}`);
    }
  }
});

test('one face against one face has no triple point, so the sharing pass is a no-op there', () => {
  // With a single surface on each side there is no third face anywhere and
  // nothing to share. This is the control that says the pass changed the
  // multi-face result and left the one-face case exactly as it was.
  //
  // Two OVERLAPPING organic bodies, not two that miss each other: a pair with
  // no intersection curve at all would make every assertion below vacuous, and
  // the count is checked rather than assumed.
  const A = blob();
  const other = blob();
  const B = {
    // Euclidean xyz in the net, so the offset carries no weight factor.
    faces: other.faces.map((f) => ({ srf: { ...f.srf, ctrlNet: f.srf.ctrlNet.map((row) => row.map(([x, y, z, w]) => [x + 9, y + 3, z + 5, w])) } })),
  };
  B.triangles = tess(B.faces, 40);
  const curves = sweepCurves(A, B);
  assert.ok(curves.length > 0, 'the two blobs must actually intersect for this control to say anything');
  assert.ok(curves.some((c) => c.samples.length > 10), 'the intersection must be a real curve, not a graze');
  assert.equal(shareTriplePoints(curves, A, B, {}).inserted, 0);
});

test('a one-surface pair that DOES close still closes, and closes without sharing anything', () => {
  // The revolved-sphere union is the standing single-pair case. It has to stay
  // at zero naked edges AND report that nothing was shared, so a pass here
  // cannot be the new machinery quietly rescuing it.
  const ball = (cx) => {
    const pts = [];
    for (let i = 0; i < 7; i++) {
      const th = -Math.PI / 2 + (Math.PI * i) / 6;
      pts.push([40 * Math.cos(th), 0, 40 * Math.sin(th)]);
    }
    const s = revolve(globalCurveInterp(pts, 3), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
    // TRANSLATE WITHOUT THE WEIGHT. This control net stores EUCLIDEAN xyz
    // alongside the weight, and a revolve's rational circle carries
    // w = sqrt(2)/2 on half its columns — scaling the offset by w moves those
    // columns only 0.707 as far, and the "radius 40 sphere" then measures
    // 35.018 to 45.329. The check below is what catches it.
    const srf = { ...s, ctrlNet: s.ctrlNet.map((row) => row.map(([x, y, z, w]) => [x + cx, y, z, w])) };
    return { faces: [{ srf }], triangles: tess([{ srf }], 40), srf };
  };
  const A = ball(0), B = ball(55);
  // MEASURED, not assumed: both bodies really are radius 40 about their own
  // stated centre, and 55 apart, so they genuinely overlap. Distance from the
  // centre, not from the axis — a body translated in x has no radial symmetry
  // about the z axis to read.
  for (const [cx, X] of [[0, A], [55, B]]) {
    let lo = Infinity, hi = -Infinity;
    const uK = X.srf.knotsU, vK = X.srf.knotsV;
    for (let i = 0; i <= 24; i++) {
      for (let j = 0; j <= 60; j++) {
        const p = surfacePoint(X.srf, uK[0] + (uK[uK.length - 1] - uK[0]) * (i / 24), vK[0] + (vK[vK.length - 1] - vK[0]) * (j / 60));
        const d = Math.hypot(p[0] - cx, p[1], p[2]);
        lo = Math.min(lo, d); hi = Math.max(hi, d);
      }
    }
    assert.ok(lo > 39.8 && hi < 40.2, `sphere at x=${cx} spans radius ${lo.toFixed(3)}..${hi.toFixed(3)}`);
  }
  const curves = sweepCurves(A, B);
  assert.ok(curves.length === 1 && curves[0].samples.length > 50, 'expected one well-sampled intersection circle');
  const res = booleanSolids(A, B, curves, 'union');
  assert.ok(res.ok, `union: ${res.reason}`);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.chi, 2);
  assert.equal(res.triplePoints, 0);
});
