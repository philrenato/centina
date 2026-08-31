// A CUT THAT STRADDLES A SEAM IS HANDLED FOR REAL — IT NO LONGER REFUSES.
//
// THIS FILE ONCE ASSERTED THE OPPOSITE, DELIBERATELY, AND THAT IS WORTH
// SAYING OUT LOUD. Its own header used to end: "The refusal is a placeholder
// for a capability, not a resting place. Handling an even-crossing loop for
// real means splitting it at its crossings and closing each subchain through
// the domain boundary." That capability is now built (seamStraddleChains),
// so the refusal this file was written to pin is genuinely gone. The test
// below is rewritten to the invariant the refusal was standing in for, not
// deleted — a straddling cut must produce the RIGHT answer, and "refuses by
// name" was only ever the honest interim version of that.
//
// A loop crossing a closed direction's seam an EVEN number of times is not a
// wrap — it goes over and comes back. It has no once-around spine, and
// seamCrossingSpine still says so by name (TEST 1, unchanged). What changed
// is what the boolean does NEXT: rather than refuse, it splits the loop at
// its crossings and closes each subchain along the domain edge it ends on,
// yielding one UV piece per side of the seam. The pieces weld in the ordinary
// sew because a seam's two UV copies, (aMin,o) and (aMax,o), are the same 3D
// point.
//
// WHY THIS IS GENERIC AND NOT AN EXOTIC FIXTURE. For two solids of revolution
// each operand's seam meridian generally points straight through the other
// body, so the cut lands across it. Whether any given pair "worked" used to be
// decided by which way its seams happened to face. THAT is the real invariant
// this file now pins: the SAME pair must produce the SAME solid regardless of
// where its seams sit. TEST 2 asserts the straddling pair closes; TEST 2b
// asserts it closes to the same measured solid as the seams-rotated-off
// control; TEST 3 keeps that control as an independent check.
//
// NOT ASSERTED HERE, NAMED HONESTLY: absolute volume against the analytic
// two-sphere lens. Measuring it means tessellating each kept fragment
// independently, which is not watertight at the shared edges (and trips
// trimtess's own known ear-clip residual on this fixture), so the numbers it
// produces are not an oracle. Both the straddle and the control read ~2.7%
// under analytic THE SAME WAY — a pre-existing property of that measurement
// on this fixture, not something the straddle handling introduces, which is
// exactly what TEST 2b's straddle-equals-control comparison establishes
// without needing to trust either number on its own.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { booleanSolids } from '../kernel/boolean.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { trivialTrimLoop, seamCrossingSpine, projectPointsToSurfaceUV } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { massProperties } from '../kernel/massprops.mjs';

function tess(faces, res) {
  const tris = [];
  for (const f of faces) {
    for (const t of tessellateTrimmedSurface(f.srf, f.trimLoop ?? trivialTrimLoop(f.srf), res, res, f.trimHoles ?? [])) {
      tris.push(t.map((v) => v.position));
    }
  }
  return tris;
}

// A sphere built the way the organic probe builds its blobs: an interpolated
// pole-to-pole half arc, revolved. `seamSign` puts the profile meridian (and
// therefore the seam) at +x or -x, which is the only thing that differs
// between the failing fixture and the passing control.
const PROFILE_POINTS = 7;
function sphere(cx, seamSign) {
  const pts = [];
  for (let i = 0; i < PROFILE_POINTS; i++) {
    const th = -Math.PI / 2 + (Math.PI * i) / (PROFILE_POINTS - 1);
    pts.push([seamSign * 20 * Math.cos(th), 0, 20 * Math.sin(th)]);
  }
  const srf = revolve(globalCurveInterp(pts, 3), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  // TRANSLATE IN EUCLIDEAN COORDINATES, NOT HOMOGENEOUS ONES. This net stores
  // the dehomogenized position plus a separate weight, so the shift is `x + cx`.
  // Writing `x + cx * w` is the homogeneous form, and a revolve's own rational
  // circle carries w = sqrt(2)/2 on its off-axis columns — so those points moved
  // only 0.707 as far as the rest and the "radius 20 ball" came out spanning
  // 17.29 to 22.91, a 28% deformation. Every assertion below still passed on it,
  // which is the point: a fixture nobody measures can be any shape at all.
  const shifted = cx === 0
    ? srf
    : { ...srf, ctrlNet: srf.ctrlNet.map((row) => row.map(([x, y, z, w]) => [x + cx, y, z, w])) };
  const faces = [{ srf: shifted }];
  return { faces, triangles: tess(faces, 40) };
}

function cutCurves(A, B) {
  const out = [];
  for (let i = 0; i < A.faces.length; i++) {
    for (let j = 0; j < B.faces.length; j++) {
      const r = intersectSurfacesComplete(A.faces[i].srf, B.faces[j].srf);
      if (!r || !r.ok) continue;
      // One physical circle; the marcher can seed it more than once. Keep the
      // richest component so the fixture is one cut, as the geometry is.
      const best = r.components.reduce((m, c) => (c.samples.length > (m?.samples.length ?? 0) ? c : m), null);
      // An SSI sample is a record; the boolean wants the bare 3D position.
      if (best) out.push({ samples: best.samples.map((s) => s.point), faceA: i, faceB: j });
    }
  }
  return out;
}

// MEASURE THE FIXTURE BEFORE REASONING ABOUT IT. Every test below is a claim
// about two radius-20 balls one radius apart; none of them notices if the balls
// are some other shape. They did not notice for a long time — a homogeneous
// translate on a euclidean control net left one of them spanning 17.29 to 22.91
// while every assertion still passed. A fixture that is never measured is a
// free variable in every result that rests on it.
test('SEAM STRADDLE fixture: both operands really are radius-20 spheres at the centers claimed', () => {
  // Exactly the three operands the tests below actually build — measuring some
  // other sphere would leave the ones under test unmeasured, which is the very
  // gap this closes.
  for (const [label, cx, seamSign] of [['A +seam', 0, 1], ['A -seam (control)', 0, -1], ['B', 25, 1]]) {
    const s = sphere(cx, seamSign).faces[0].srf;
    const uK = s.knotsU, vK = s.knotsV;
    let lo = Infinity, hi = -Infinity;
    for (let a = 0; a <= 40; a++) {
      for (let b = 0; b <= 40; b++) {
        const u = uK[0] + ((uK[uK.length - 1] - uK[0]) * a) / 40;
        const v = vK[0] + ((vK[vK.length - 1] - vK[0]) * b) / 40;
        const p = surfacePoint(s, u, v);
        const r = Math.hypot(p[0] - cx, p[1], p[2]);
        lo = Math.min(lo, r); hi = Math.max(hi, r);
      }
    }
    // Tolerance is for the degree-3 interpolation of a half arc through 7
    // points, which is a real and expected approximation; 28% is not.
    assert.ok(hi - lo < 0.2 && Math.abs((hi + lo) / 2 - 20) < 0.2,
      `operand ${label} (cx=${cx}) spans radius ${lo.toFixed(2)}..${hi.toFixed(2)} about (${cx},0,0), wanted ~20`);
  }
});

test('SEAM STRADDLE: the fixture genuinely straddles — an even number of seam crossings, which is NOT a wrap', () => {
  const A = sphere(0, 1), B = sphere(25, 1);
  const curves = cutCurves(A, B);
  assert.equal(curves.length, 1, 'one intersection circle');

  const proj = projectPointsToSurfaceUV(curves[0].samples, A.faces[0].srf);
  assert.ok(proj.ok, 'the cut projects onto A');
  const spine = seamCrossingSpine(proj.uv, A.faces[0].srf);

  // The load-bearing distinction: it is refused, and refused for STRADDLING
  // specifically — not for the benign "never touches a seam" reason, which is
  // what the control below gets and what must still pass through untouched.
  assert.equal(spine.ok, false, 'a straddling loop has no once-around spine');
  assert.equal(spine.code, 'seam-straddle', `refused as a straddle, got ${spine.code}`);
  assert.match(spine.reason, /crosses the seam 2 times and returns/);

  // And B, the other operand, is NOT entangled with its own seam — proving
  // the failure belongs to one face rather than to the pair being spheres.
  const projB = projectPointsToSurfaceUV(curves[0].samples, B.faces[0].srf);
  assert.equal(seamCrossingSpine(projB.uv, B.faces[0].srf).code, 'no-seam-crossing');
});

test('SEAM STRADDLE: all three operators now CLOSE — the straddle is handled, not refused', () => {
  const A = sphere(0, 1), B = sphere(25, 1);
  const curves = cutCurves(A, B);

  for (const op of ['union', 'intersect', 'difference']) {
    const r = booleanSolids(A, B, curves, op);
    assert.equal(r.ok, true, `${op} closes: ${r.reason ?? ''}`);
    assert.equal(r.stats.nakedEdgeCount, 0, `${op} has no naked edges`);
    assert.equal(r.stats.nonManifoldEdgeCount, 0, `${op} has no non-manifold edges`);
    assert.equal(r.stats.chi, 2, `${op} is one closed shell`);
    assert.equal(r.stats.shellCount, 1);
    assert.equal(r.stats.genus, 0, `${op} of two overlapping balls is genus 0`);

    // ...and the seam is genuinely gone as a REPORTED CAUSE, not merely
    // survived: no face may still be coming back with the straddle refusal.
    assert.equal(r.faceReports.some((f) => f.code === 'seam-straddle'), false,
      `${op} no longer reports a seam straddle`);
  }
});

test('SEAM STRADDLE: the straddling pair produces the SAME solid as the seams-rotated-off control', () => {
  // THE LOAD-BEARING CLAIM OF THE WHOLE FEATURE. These two pairs are the same
  // geometry — same radii, same offset, same poles — differing ONLY in where
  // each profile meridian (and therefore the seam) sits. A boolean whose
  // answer depends on that is wrong even when it closes, so agreeing with the
  // control is a stronger statement than closing.
  //
  // Measured by tessellating each kept fragment and taking area + |volume|.
  // Neither number is trusted as an absolute (see the header) — they are
  // compared against EACH OTHER, where the shared measurement bias cancels.
  const straddle = { A: sphere(0, 1), B: sphere(25, 1) };
  const control = { A: sphere(0, -1), B: sphere(25, 1) };

  const measure = ({ A, B }, op) => {
    const r = booleanSolids(A, B, cutCurves(A, B), op);
    assert.equal(r.ok, true, `${op} closes: ${r.reason ?? ''}`);
    const tris = [];
    for (const f of r.fragments) {
      for (const t of tessellateTrimmedSurface(f.srf, f.outer, 60, 60, f.holes ?? [])) {
        tris.push(t.map((v) => v.position));
      }
    }
    const m = massProperties(tris);
    return { area: m.area, volume: Math.abs(m.volume) };
  };

  for (const op of ['union', 'intersect', 'difference']) {
    const s = measure(straddle, op);
    const c = measure(control, op);
    // Same solid to a relative 1e-4 — far tighter than the ~2.7% the shared
    // measurement bias sits at, so this cannot pass by both being equally
    // vague. A fragment kept on the wrong side of the cut, or a straddle
    // piece over- or under-covering, moves these by percent, not by 1e-4.
    assert.ok(Math.abs(s.volume - c.volume) < c.volume * 1e-4,
      `${op} volume matches the control (${s.volume} vs ${c.volume})`);
    assert.ok(Math.abs(s.area - c.area) < c.area * 1e-4,
      `${op} area matches the control (${s.area} vs ${c.area})`);
  }

  // The one thing that legitimately DIFFERS: intersect keeps A's cap, and on
  // the straddling pair that cap is cut in two by the domain rectangle, so it
  // arrives as two UV pieces where the control's is one. Same surface, one
  // more seam — asserted so a future change that quietly stopped splitting
  // (or started splitting the control) is caught rather than absorbed.
  const sFrags = booleanSolids(straddle.A, straddle.B, cutCurves(straddle.A, straddle.B), 'intersect').fragments.length;
  const cFrags = booleanSolids(control.A, control.B, cutCurves(control.A, control.B), 'intersect').fragments.length;
  assert.equal(cFrags, 2, 'the control keeps one piece per operand');
  assert.equal(sFrags, 3, 'the straddling pair splits A’s cap into two UV pieces');
});

test('SEAM STRADDLE CONTROL: the SAME pair with its seams rotated off the cut still closes on all three operators', () => {
  // Identical construction, identical poles, identical wrapped single-face
  // solids. The ONLY difference is where each profile meridian sits. If the
  // fix had over-refused — treating any seam-adjacent loop as unusable — this
  // would refuse too, and the refusal above would prove nothing.
  const A = sphere(0, -1), B = sphere(25, 1);
  const curves = cutCurves(A, B);
  assert.equal(curves.length, 1);

  const projA = projectPointsToSurfaceUV(curves[0].samples, A.faces[0].srf);
  assert.equal(seamCrossingSpine(projA.uv, A.faces[0].srf).code, 'no-seam-crossing',
    'the control genuinely avoids the seam — otherwise it is not a control');

  for (const op of ['union', 'intersect', 'difference']) {
    const r = booleanSolids(A, B, curves, op);
    assert.equal(r.ok, true, `${op} closes: ${r.reason ?? ''}`);
    assert.equal(r.stats.nakedEdgeCount, 0, `${op} has no naked edges`);
    assert.equal(r.stats.chi, 2, `${op} is one closed shell`);
    assert.equal(r.stats.shellCount, 1);
  }
});

test('SEAM STRADDLE: a benign refusal still passes the raw curve through — the ordinary interior cut is untouched', () => {
  // unwrapSeamCut sees a refusal on EVERY ordinary boolean, because a cut that
  // never approaches a seam refuses with 'no-seam-crossing'. Treating a refusal
  // as failure without reading the code would refuse every boolean in the app.
  // This pins the two codes that must remain benign.
  const A = sphere(0, -1), B = sphere(25, 1);
  const curves = cutCurves(A, B);
  const proj = projectPointsToSurfaceUV(curves[0].samples, A.faces[0].srf);

  const benign = seamCrossingSpine(proj.uv, A.faces[0].srf);
  assert.equal(benign.ok, false);
  assert.equal(benign.code, 'no-seam-crossing');

  const tooFew = seamCrossingSpine([[0.1, 0.1], [0.2, 0.2]], A.faces[0].srf);
  assert.equal(tooFew.code, 'too-few-points');

  // ...and the boolean built on top of that benign refusal genuinely closes,
  // which is the property the passthrough exists to preserve.
  assert.equal(booleanSolids(A, B, curves, 'union').ok, true);
});
